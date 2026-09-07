"""Model components for the independent leakage-free C-PAC pipeline.

This module has no filesystem access and no global model/optimizer instances.
Every training call in ``cpac_leakage_free_nested_cv.py`` constructs a fresh
``BrainInnovationSystem`` and optimizer.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SE_ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        hidden = max(dim // 4, 1)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.se = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.net(x)
        return x + residual * self.se(residual)


class VB_OCREAD(nn.Module):
    def __init__(
        self,
        num_nodes: int = 200,
        in_dim: int = 15,
        hidden_dim: int = 64,
        num_clusters: int = 7,
        prior_anchors: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.eval_ocread_mode = "mu"
        self.register_buffer("initial_anchors", None)
        self.node_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.dynamic_gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        if prior_anchors is None:
            anchors = torch.randn(num_clusters, hidden_dim)
            nn.init.orthogonal_(anchors)
        else:
            anchors = self._gram_schmidt(prior_anchors)
        self.mu = nn.Parameter(anchors)
        self.initial_anchors = anchors.clone().detach()
        self.logvar = nn.Parameter(torch.full((num_clusters, hidden_dim), -7.0))

    def set_eval_ocread_mode(self, mode: str) -> None:
        if mode not in {"mu", "sample"}:
            raise ValueError(f"Unsupported OCREAD evaluation mode: {mode}")
        self.eval_ocread_mode = mode

    @staticmethod
    def _gram_schmidt(values: torch.Tensor) -> torch.Tensor:
        basis = torch.zeros_like(values)
        for index in range(values.size(0)):
            vector = values[index]
            for previous in range(index):
                denominator = torch.dot(basis[previous], basis[previous]) + 1e-8
                vector = vector - (
                    torch.dot(basis[previous], values[index]) / denominator
                ) * basis[previous]
            basis[index] = vector / (torch.norm(vector) + 1e-8)
        return basis

    def forward(
        self, x_frequency_flat: torch.Tensor, tau: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x_frequency_flat.size(0)
        nodes = x_frequency_flat.view(batch_size, self.num_nodes, -1)
        encoded = self.node_mlp(nodes)
        if self.training or self.eval_ocread_mode == "sample":
            standard_deviation = torch.exp(0.5 * self.logvar)
            anchors = self.mu + torch.randn_like(standard_deviation) * standard_deviation
        else:
            anchors = self.mu
        gate = self.dynamic_gate(encoded)
        logits = torch.matmul(encoded, anchors.t()) / (self.hidden_dim**0.5)
        assignments = torch.softmax((logits * gate) / tau, dim=-1)
        pooled = torch.bmm(assignments.transpose(1, 2), encoded)
        return pooled.reshape(batch_size, -1), self.mu, self.logvar


class BrainInnovationSystem(nn.Module):
    """Three-branch FC/frequency/demographic fusion model."""

    def __init__(
        self,
        space_dim: int,
        prior_anchors: torch.Tensor,
        hidden_dim: int = 80,
        align_dim: int = 160,
        dropout_sc: float = 0.25,
        dropout_cls: float = 0.25,
        num_res_blocks: int = 4,
    ):
        super().__init__()
        self.sc_mixer = nn.Sequential(
            nn.Linear(space_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_sc),
            nn.Linear(256, align_dim),
            nn.LayerNorm(align_dim),
            nn.GELU(),
        )
        self.ocread = VB_OCREAD(
            num_nodes=200,
            in_dim=15,
            hidden_dim=hidden_dim,
            num_clusters=7,
            prior_anchors=prior_anchors,
        )
        self.freq_align = nn.Sequential(
            nn.Linear(7 * hidden_dim, align_dim),
            nn.LayerNorm(align_dim),
            nn.GELU(),
        )
        self.demo_align = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, align_dim),
            nn.LayerNorm(align_dim),
            nn.GELU(),
        )
        self.attention_gate = nn.Sequential(
            nn.Linear(align_dim * 3, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        self.res_blocks = nn.Sequential(
            *[
                SE_ResidualBlock(align_dim, dropout=dropout_cls)
                for _ in range(num_res_blocks)
            ]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(align_dim),
            nn.GELU(),
            nn.Dropout(dropout_cls),
            nn.Linear(align_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout_cls * 0.5),
            nn.Linear(64, 1),
        )

    def set_eval_ocread_mode(self, mode: str) -> None:
        self.ocread.set_eval_ocread_mode(mode)

    def forward(
        self,
        x_space: torch.Tensor,
        x_frequency: torch.Tensor,
        x_demographic: torch.Tensor,
        tau: float = 1.0,
        training: bool = True,
        noise_std: float = 0.01,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if training:
            x_space = x_space + torch.randn_like(x_space) * noise_std
            x_frequency = (
                x_frequency + torch.randn_like(x_frequency) * noise_std * 0.3
            )
        space_feature = self.sc_mixer(x_space)
        frequency_pooled, mu, logvar = self.ocread(x_frequency, tau=tau)
        frequency_feature = self.freq_align(frequency_pooled)
        demographic_feature = self.demo_align(x_demographic)
        attention = torch.softmax(
            self.attention_gate(
                torch.cat(
                    [space_feature, frequency_feature, demographic_feature], dim=1
                )
            ),
            dim=-1,
        )
        fused = (
            attention[:, 0:1] * space_feature
            + attention[:, 1:2] * frequency_feature
            + attention[:, 2:3] * demographic_feature
        )
        fused = self.res_blocks(fused)
        return self.classifier(fused).squeeze(-1), fused, mu, logvar

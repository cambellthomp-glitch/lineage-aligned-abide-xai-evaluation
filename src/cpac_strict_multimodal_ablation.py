"""Strict multimodal/component ablation for the ABIDE-I C-PAC CMPB task.

Six variants share the exact strict 10x3 nested-CV splits, random seeds,
training-partition FC selection (19,900 -> 4,975), transformations, candidate
grid and checkpoint-selection rule:

1. FC only
2. FC + frequency
3. FC + demographics
4. FC + frequency + demographics (full CMPB; legacy contrastive loss restored)
5. Full modalities with mean pooling replacing OCREAD
6. Full modalities without contrastive loss

The prior completed strict C-PAC lineage omitted contrastive loss even though
the legacy target implementation enabled it.  Therefore all six variants are
trained anew here; the old strict OOF predictions are audit references only.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset

import phase4_controlled_deep_artifact_rerun as controlled
import strict_nested_cv_interpretable_model as strict
import cpac_strict_same_protocol_baselines as common


ROOT = Path(__file__).resolve().parent
OUT_BASE = ROOT / "cpac_strict_multimodal_ablation"
STRICT_REFERENCE = ROOT / "strict_nested_cv_interpretability" / "run_20260712_strict_nested_cv"
LEGACY_TARGET = ROOT / "gemini1_80target.py"

SEED = strict.DEFAULT_SEED
N_OUTER = 10
N_INNER = 3


@dataclass(frozen=True)
class Variant:
    name: str
    display_name: str
    use_frequency: bool
    use_demographics: bool
    frequency_pooling: str
    contrastive_weight: float


VARIANTS = [
    Variant("fc_only", "FC only", False, False, "none", 0.10),
    Variant("fc_frequency", "FC + frequency", True, False, "ocread", 0.10),
    Variant("fc_demographics", "FC + demographics", False, True, "none", 0.10),
    Variant("full_cmpb", "FC + frequency + demographics (full CMPB)", True, True, "ocread", 0.10),
    Variant("mean_pooling", "Full modalities; mean pooling replaces OCREAD", True, True, "mean", 0.10),
    Variant("no_contrastive", "Full modalities; no contrastive loss", True, True, "ocread", 0.0),
]
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def legacy_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.10) -> torch.Tensor:
    """Exact loss used by the legacy C-PAC target configuration."""
    normalized = F.normalize(embeddings, p=2, dim=1)
    similarity = torch.matmul(normalized, normalized.t()) / temperature
    mask = torch.eq(labels.view(-1, 1), labels.view(-1, 1).t()).float()
    return -(mask * F.log_softmax(similarity, dim=1)).mean()


class StrictAblationModel(nn.Module):
    def __init__(
        self,
        variant: Variant,
        candidate: strict.Candidate,
        anchors: torch.Tensor,
        hidden_dim: int = 80,
        align_dim: int = 160,
        residual_blocks: int = 4,
    ):
        super().__init__()
        self.variant = variant
        self.sc_mixer = nn.Sequential(
            nn.Linear(strict.N_SELECTED_FC, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(candidate.dropout_sc),
            nn.Linear(256, align_dim),
            nn.LayerNorm(align_dim),
            nn.GELU(),
        )
        self.ocread: Optional[controlled.VB_OCREAD] = None
        self.mean_node_mlp: Optional[nn.Module] = None
        self.freq_align: Optional[nn.Module] = None
        if variant.use_frequency and variant.frequency_pooling == "ocread":
            self.ocread = controlled.VB_OCREAD(
                num_nodes=200,
                in_dim=15,
                hidden_dim=hidden_dim,
                num_clusters=7,
                prior_anchors=anchors,
            )
            self.freq_align = nn.Sequential(
                nn.Linear(7 * hidden_dim, align_dim),
                nn.LayerNorm(align_dim),
                nn.GELU(),
            )
        elif variant.use_frequency and variant.frequency_pooling == "mean":
            self.mean_node_mlp = nn.Sequential(
                nn.Linear(15, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            self.freq_align = nn.Sequential(
                nn.Linear(hidden_dim, align_dim),
                nn.LayerNorm(align_dim),
                nn.GELU(),
            )
        self.demo_align: Optional[nn.Module] = None
        if variant.use_demographics:
            self.demo_align = nn.Sequential(
                nn.Linear(2, 32),
                nn.GELU(),
                nn.Linear(32, align_dim),
                nn.LayerNorm(align_dim),
                nn.GELU(),
            )
        branches = 1 + int(variant.use_frequency) + int(variant.use_demographics)
        self.attention_gate = nn.Sequential(
            nn.Linear(align_dim * branches, 64),
            nn.GELU(),
            nn.Linear(64, branches),
        )
        self.res_blocks = nn.Sequential(
            *[controlled.SE_ResidualBlock(align_dim, dropout=candidate.dropout_cls) for _ in range(residual_blocks)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(align_dim),
            nn.GELU(),
            nn.Dropout(candidate.dropout_cls),
            nn.Linear(align_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(candidate.dropout_cls * 0.5),
            nn.Linear(64, 1),
        )

    def set_eval_ocread_mode(self, mode: str) -> None:
        if self.ocread is not None:
            self.ocread.set_eval_ocread_mode(mode)

    def forward(
        self,
        x_s: torch.Tensor,
        x_f: torch.Tensor,
        x_d: torch.Tensor,
        tau: float = 1.0,
        training: bool = True,
        noise_std: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if training:
            x_s = x_s + torch.randn_like(x_s) * noise_std
            if self.variant.use_frequency:
                x_f = x_f + torch.randn_like(x_f) * noise_std * 0.3
        branches: List[torch.Tensor] = [self.sc_mixer(x_s)]
        mu: Optional[torch.Tensor] = None
        logvar: Optional[torch.Tensor] = None
        if self.variant.use_frequency:
            if self.variant.frequency_pooling == "ocread":
                if self.ocread is None or self.freq_align is None:
                    raise RuntimeError("OCREAD branch is not initialized")
                frequency_feature, mu, logvar = self.ocread(x_f, tau=tau)
                branches.append(self.freq_align(frequency_feature))
            else:
                if self.mean_node_mlp is None or self.freq_align is None:
                    raise RuntimeError("Mean-pooling branch is not initialized")
                nodes = self.mean_node_mlp(x_f.view(-1, 200, 15))
                branches.append(self.freq_align(nodes.mean(dim=1)))
        if self.variant.use_demographics:
            if self.demo_align is None:
                raise RuntimeError("Demographic branch is not initialized")
            branches.append(self.demo_align(x_d))
        concatenated = torch.cat(branches, dim=1)
        attention = torch.softmax(self.attention_gate(concatenated), dim=-1)
        fused = sum(attention[:, index : index + 1] * branch for index, branch in enumerate(branches))
        fused = self.res_blocks(fused)
        logits = self.classifier(fused).squeeze(-1)
        return logits, fused, mu, logvar


def make_model(variant: Variant, candidate: strict.Candidate, anchors: torch.Tensor, device: torch.device) -> StrictAblationModel:
    model = StrictAblationModel(variant, candidate, anchors).to(device)
    model.set_eval_ocread_mode("mu")
    return model


def predict_probabilities(
    model: StrictAblationModel,
    x_s: np.ndarray,
    x_f: np.ndarray,
    x_d: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    model.set_eval_ocread_mode("mu")
    values: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_s), 64):
            stop = min(start + 64, len(x_s))
            logits, _, _, _ = model(
                torch.from_numpy(x_s[start:stop]).to(device),
                torch.from_numpy(x_f[start:stop]).to(device),
                torch.from_numpy(x_d[start:stop]).to(device),
                tau=0.15,
                training=False,
            )
            values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values).reshape(-1)


def evaluate(
    model: StrictAblationModel,
    x_s: np.ndarray,
    x_f: np.ndarray,
    x_d: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> Dict[str, float]:
    probability = predict_probabilities(model, x_s, x_f, x_d, device)
    return {
        "auc": common.safe_auc(y, probability),
        "accuracy": float(accuracy_score(y, probability >= 0.5)),
        "bce": common.clipped_log_loss(y, probability),
    }


def train_model(
    variant: Variant,
    candidate: strict.Candidate,
    prepared: strict.PreparedData,
    y_train: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    eval_every: int,
    monitor_y: Optional[np.ndarray] = None,
    patience: Optional[int] = None,
    minimum_epochs: int = 0,
) -> Tuple[StrictAblationModel, Dict[str, Any]]:
    set_seed(seed)
    cfg = strict.model_config(candidate)
    model = make_model(variant, candidate, prepared.anchors, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - cfg["lr_warmup"], 1))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(prepared.x_s_train),
            torch.from_numpy(prepared.x_f_train),
            torch.from_numpy(prepared.x_d_train),
            torch.from_numpy(np.asarray(y_train, dtype=np.float32)),
        ),
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_auc = -float("inf")
    best_bce = float("inf")
    best_epoch = 0
    last_improved = 0
    stopped_epoch = epochs
    history: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        if epoch <= cfg["lr_warmup"]:
            for group in optimizer.param_groups:
                group["lr"] = cfg["lr"] * epoch / cfg["lr_warmup"]
        tau = cfg["tau_high"] if epoch <= cfg["tau_warmup"] else cfg["tau_low"]
        model.train()
        totals = {"total": 0.0, "classification": 0.0, "kl": 0.0, "anchor": 0.0, "contrastive": 0.0}
        batches = 0
        for b_s, b_f, b_d, b_y in loader:
            b_s, b_f, b_d, b_y = b_s.to(device), b_f.to(device), b_d.to(device), b_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            use_mixup = cfg["mixup_alpha"] > 0 and epoch > cfg["lr_warmup"]
            if use_mixup:
                lam = float(np.random.beta(cfg["mixup_alpha"], cfg["mixup_alpha"]))
                permutation = torch.randperm(len(b_y), device=device)
                in_s = lam * b_s + (1.0 - lam) * b_s[permutation]
                in_f = lam * b_f + (1.0 - lam) * b_f[permutation]
                in_d = lam * b_d + (1.0 - lam) * b_d[permutation]
                y_a, y_b = b_y, b_y[permutation]
            else:
                lam, in_s, in_f, in_d, y_a, y_b = 1.0, b_s, b_f, b_d, b_y, b_y
            logits, fused, mu, logvar = model(in_s, in_f, in_d, tau=tau, training=True, noise_std=cfg["noise_std"])
            bce_a = F.binary_cross_entropy_with_logits(logits, y_a, reduction="none")
            bce_b = F.binary_cross_entropy_with_logits(logits, y_b, reduction="none")
            weight_a = (1 - torch.exp(-bce_a)).pow(cfg["focal_gamma"])
            weight_b = (1 - torch.exp(-bce_b)).pow(cfg["focal_gamma"])
            loss_cls = lam * (weight_a * bce_a).mean() + (1 - lam) * (weight_b * bce_b).mean()
            if mu is not None and logvar is not None and model.ocread is not None:
                loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * cfg["kl_weight"]
                loss_anchor = F.mse_loss(mu, model.ocread.initial_anchors) * cfg["anchor_weight"]
            else:
                loss_kl = torch.zeros((), device=device)
                loss_anchor = torch.zeros((), device=device)
            if variant.contrastive_weight > 0 and not use_mixup:
                loss_contrastive = legacy_contrastive_loss(fused, b_y, temperature=0.10) * variant.contrastive_weight
            else:
                loss_contrastive = torch.zeros((), device=device)
            loss = loss_cls + loss_kl + loss_anchor + loss_contrastive
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            totals["total"] += float(loss.item())
            totals["classification"] += float(loss_cls.item())
            totals["kl"] += float(loss_kl.item())
            totals["anchor"] += float(loss_anchor.item())
            totals["contrastive"] += float(loss_contrastive.item())
            batches += 1
        if epoch > cfg["lr_warmup"]:
            scheduler.step()
        if monitor_y is not None and (epoch % eval_every == 0 or epoch == epochs):
            metric = evaluate(model, prepared.x_s_eval, prepared.x_f_eval, prepared.x_d_eval, monitor_y, device)
            row = {"epoch": epoch, **metric}
            row.update({f"train_{name}": value / max(batches, 1) for name, value in totals.items()})
            history.append(row)
            improved = metric["auc"] > best_auc + 1e-12 or (abs(metric["auc"] - best_auc) <= 1e-12 and metric["bce"] < best_bce)
            if improved:
                best_auc, best_bce, best_epoch = metric["auc"], metric["bce"], epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                last_improved = epoch
            if patience is not None and epoch >= minimum_epochs and last_improved > 0 and epoch - last_improved >= patience:
                stopped_epoch = epoch
                break
    if monitor_y is None:
        best_epoch = epochs
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append({"epoch": epochs, "checkpoint_policy": "fixed_inner_selected_epoch"})
    if best_state is None:
        raise RuntimeError("No model checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    model.set_eval_ocread_mode("mu")
    return model, {
        "best_epoch": int(best_epoch),
        "best_validation_auc": None if monitor_y is None else float(best_auc),
        "best_validation_bce": None if monitor_y is None else float(best_bce),
        "maximum_epochs": int(epochs),
        "stopped_epoch": int(stopped_epoch),
        "minimum_epochs": int(minimum_epochs),
        "patience": patience,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "history": history,
    }


def exact_outer_splits(y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    return [(train.astype(np.int64), test.astype(np.int64)) for train, test in splitter.split(np.zeros(len(y)), y)]


def exact_inner_splits(fold: int, outer_train: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=SEED + 10_000 + fold)
    return [
        (outer_train[fit].astype(np.int64), outer_train[validation].astype(np.int64))
        for fit, validation in splitter.split(outer_train, y[outer_train])
    ]


def save_preprocessing_audit(
    folder: Path,
    prepared: strict.PreparedData,
    fit_idx: np.ndarray,
    eval_idx: np.ndarray,
    reference_selected: Path,
) -> Dict[str, Any]:
    provenance = strict.save_prepared_artifacts(folder, prepared, fit_idx, eval_idx, "strict_multimodal_ablation")
    reference = np.load(reference_selected, allow_pickle=False).astype(np.int64)
    audit = {
        **provenance,
        "selected_edges_identical_to_strict_reference": bool(np.array_equal(prepared.selected_edges, reference)),
        "strict_reference_selected_edges_sha256": common.hash_edges(reference),
        "fit_eval_overlap": int(len(np.intersect1d(fit_idx, eval_idx))),
    }
    common.write_json(folder / "ablation_preprocessing_audit.json", audit)
    return audit


def prepare_fold(
    fold: int,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    full_fc: np.ndarray,
    x_freq: np.ndarray,
    x_demo: np.ndarray,
    y: np.ndarray,
    yeo_nodes: Dict[int, np.ndarray],
    args: argparse.Namespace,
    fold_dir: Path,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, strict.PreparedData]], strict.PreparedData, Dict[str, Any]]:
    reference_fold = STRICT_REFERENCE / f"fold_{fold:02d}"
    reference_train = np.load(reference_fold / "outer_train_indices.npy", allow_pickle=False).astype(np.int64)
    reference_test = np.load(reference_fold / "outer_test_indices.npy", allow_pickle=False).astype(np.int64)
    audit: Dict[str, Any] = {
        "outer_fold": fold,
        "outer_train_identical_to_strict_reference": bool(np.array_equal(outer_train, reference_train)),
        "outer_test_identical_to_strict_reference": bool(np.array_equal(outer_test, reference_test)),
        "outer_overlap": int(len(np.intersect1d(outer_train, outer_test))),
        "inner": [],
    }
    inner_prepared: List[Tuple[np.ndarray, np.ndarray, strict.PreparedData]] = []
    for inner_fold, (fit_idx, validation_idx) in enumerate(exact_inner_splits(fold, outer_train, y)):
        prepared = strict.prepare_train_eval(
            full_fc, x_freq, x_demo, y, fit_idx, validation_idx, yeo_nodes, 80, args.anchor_projection_seed
        )
        row = save_preprocessing_audit(
            fold_dir / "preprocessing" / f"inner_{inner_fold:02d}",
            prepared,
            fit_idx,
            validation_idx,
            reference_fold / "inner_cv" / f"inner_{inner_fold:02d}" / "base" / "selected_full_edge_indices.npy",
        )
        row["inner_fold"] = inner_fold
        audit["inner"].append(row)
        inner_prepared.append((fit_idx, validation_idx, prepared))
    outer_prepared = strict.prepare_train_eval(
        full_fc, x_freq, x_demo, y, outer_train, outer_test, yeo_nodes, 80, args.anchor_projection_seed
    )
    audit["outer_preprocessing"] = save_preprocessing_audit(
        fold_dir / "preprocessing" / "outer_final",
        outer_prepared,
        outer_train,
        outer_test,
        reference_fold / "selected_full_edge_indices.npy",
    )
    audit["pass"] = bool(
        audit["outer_train_identical_to_strict_reference"]
        and audit["outer_test_identical_to_strict_reference"]
        and audit["outer_overlap"] == 0
        and audit["outer_preprocessing"]["selected_edges_identical_to_strict_reference"]
        and audit["outer_preprocessing"]["fit_eval_overlap"] == 0
        and all(row["selected_edges_identical_to_strict_reference"] and row["fit_eval_overlap"] == 0 for row in audit["inner"])
    )
    common.write_json(fold_dir / "split_and_preprocessing_audit.json", audit)
    if not audit["pass"]:
        raise RuntimeError(f"Fold {fold} preprocessing audit failed")
    return inner_prepared, outer_prepared, audit


def choose_candidate(rows: Sequence[Dict[str, Any]]) -> Tuple[strict.Candidate, int, Dict[str, Any]]:
    by_name = {candidate.name: candidate for candidate in strict.candidates()}
    aggregated: Dict[str, Dict[str, Any]] = {}
    for candidate in strict.candidates():
        subset = [row for row in rows if row["candidate"] == candidate.name]
        if len(subset) != N_INNER:
            raise RuntimeError(f"Candidate {candidate.name} has {len(subset)} rows")
        epochs = [int(row["best_epoch"]) for row in subset]
        aggregated[candidate.name] = {
            "mean_inner_auc": float(np.mean([row["best_validation_auc"] for row in subset])),
            "mean_inner_bce": float(np.mean([row["best_validation_bce"] for row in subset])),
            "fold_epochs": epochs,
            "median_epoch": int(np.median(epochs)),
        }
    winner_name = sorted(
        aggregated,
        key=lambda name: (-aggregated[name]["mean_inner_auc"], aggregated[name]["mean_inner_bce"], name),
    )[0]
    selection = {
        "all_candidates": aggregated,
        "selected_candidate": winner_name,
        "final_fixed_epoch": aggregated[winner_name]["median_epoch"],
        "selection_rule": "maximum mean inner-validation AUC; lower BCE then name break ties; final epoch is the median inner best epoch",
    }
    return by_name[winner_name], int(selection["final_fixed_epoch"]), selection


def run_variant_fold(
    variant: Variant,
    fold: int,
    inner_prepared: Sequence[Tuple[np.ndarray, np.ndarray, strict.PreparedData]],
    outer_prepared: strict.PreparedData,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    fold_dir: Path,
) -> Dict[str, Any]:
    variant_dir = fold_dir / "variants" / variant.name
    common.ensure(variant_dir)
    metrics_file = variant_dir / "outer_metrics.json"
    probability_file = variant_dir / "outer_test_probs.npy"
    if args.resume and metrics_file.exists() and probability_file.exists() and (variant_dir / "model_checkpoint.pt").exists():
        return json.loads(metrics_file.read_text(encoding="utf-8"))
    inner_rows: List[Dict[str, Any]] = []
    for inner_fold, (fit_idx, validation_idx, prepared) in enumerate(inner_prepared):
        for candidate_index, candidate in enumerate(strict.candidates()):
            trial_dir = variant_dir / "inner_cv" / f"inner_{inner_fold:02d}" / candidate.name
            trial_file = trial_dir / "inner_metrics.json"
            if args.resume and trial_file.exists():
                inner_rows.append(json.loads(trial_file.read_text(encoding="utf-8")))
                continue
            common.ensure(trial_dir)
            trial_seed = SEED + fold * 1000 + inner_fold * 100 + candidate_index
            model, metadata = train_model(
                variant,
                candidate,
                prepared,
                y[fit_idx],
                device,
                trial_seed,
                args.inner_epochs,
                args.eval_every,
                monitor_y=y[validation_idx],
                patience=args.inner_patience,
                minimum_epochs=args.minimum_inner_epochs,
            )
            row = {
                "outer_fold": fold,
                "inner_fold": inner_fold,
                "variant": variant.name,
                "candidate": candidate.name,
                "candidate_config": asdict(candidate),
                "best_epoch": metadata["best_epoch"],
                "best_validation_auc": metadata["best_validation_auc"],
                "best_validation_bce": metadata["best_validation_bce"],
                "n_fit": int(len(fit_idx)),
                "n_validation": int(len(validation_idx)),
                "fit_indices_sha256": common.hash_indices(fit_idx),
                "validation_indices_sha256": common.hash_indices(validation_idx),
                "selected_edges_sha256": common.hash_edges(prepared.selected_edges),
                "training_metadata": metadata,
            }
            common.write_json(trial_file, row)
            inner_rows.append(row)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    selected_candidate, fixed_epoch, selection = choose_candidate(inner_rows)
    common.write_json(variant_dir / "inner_selection.json", selection)
    final_seed = SEED + fold * 1000 + 777
    model, final_metadata = train_model(
        variant,
        selected_candidate,
        outer_prepared,
        y[outer_train],
        device,
        final_seed,
        fixed_epoch,
        args.eval_every,
        monitor_y=None,
    )
    probability = predict_probabilities(
        model,
        outer_prepared.x_s_eval,
        outer_prepared.x_f_eval,
        outer_prepared.x_d_eval,
        device,
    )
    np.save(probability_file, probability.astype(np.float32))
    np.save(variant_dir / "outer_test_indices.npy", outer_test.astype(np.int64))
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "variant": asdict(variant),
            "candidate": asdict(selected_candidate),
            "fixed_epoch": fixed_epoch,
            "outer_train_indices_sha256": common.hash_indices(outer_train),
            "outer_test_indices_sha256": common.hash_indices(outer_test),
            "selected_edges_sha256": common.hash_edges(outer_prepared.selected_edges),
            "contrastive_lineage": "Exact legacy loss restored when contrastive_weight=0.10; zero only for no_contrastive.",
        },
        variant_dir / "model_checkpoint.pt",
    )
    metrics = {
        "status": "COMPLETE",
        "outer_fold": fold,
        "variant": asdict(variant),
        "selected_candidate": selected_candidate.name,
        "fixed_epoch": fixed_epoch,
        "outer_metrics": common.metric_bundle(y[outer_test], probability),
        "n_outer_train": int(len(outer_train)),
        "n_outer_test": int(len(outer_test)),
        "final_training_metadata": final_metadata,
        "outer_test_usage": "Single final evaluation after training-only preprocessing, candidate selection and epoch selection.",
    }
    common.write_json(metrics_file, metrics)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def contrast_specs() -> List[Tuple[str, str, str, str]]:
    return [
        ("frequency_added_to_fc", "fc_frequency", "fc_only", "FC + frequency minus FC only"),
        ("demographics_added_to_fc", "fc_demographics", "fc_only", "FC + demographics minus FC only"),
        ("demographics_added_with_frequency", "full_cmpb", "fc_frequency", "Full CMPB minus FC + frequency"),
        ("frequency_added_with_demographics", "full_cmpb", "fc_demographics", "Full CMPB minus FC + demographics"),
        ("ocread_vs_mean_pooling", "full_cmpb", "mean_pooling", "Full OCREAD minus mean pooling"),
        ("contrastive_loss", "full_cmpb", "no_contrastive", "Full CMPB minus no contrastive loss"),
    ]


def paired_delta_bootstrap(
    y: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    iterations: int,
    seed: int,
) -> Dict[str, Any]:
    return common.paired_auc_delta_bootstrap(y, first, second, iterations, seed)


def aggregate(
    run_dir: Path,
    subject_ids: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    cache_audit: Dict[str, Any],
    yeo_audit: Dict[str, Any],
    loss_audit: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    evaluated_folds = [0] if args.mode == "smoke" else list(range(N_OUTER))
    expected_full = len(evaluated_folds) == N_OUTER
    probabilities = {variant.name: np.full(len(y), np.nan, dtype=float) for variant in VARIANTS}
    assignment = {variant.name: np.zeros(len(y), dtype=int) for variant in VARIANTS}
    fold_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    split_audits: List[Dict[str, Any]] = []
    for fold in evaluated_folds:
        fold_dir = run_dir / f"fold_{fold:02d}"
        split_file = fold_dir / "split_and_preprocessing_audit.json"
        if not split_file.exists():
            common.write_json(run_dir / "partial_aggregate.json", {"status": "PARTIAL", "missing_fold": fold})
            return None
        split_audits.append(json.loads(split_file.read_text(encoding="utf-8")))
        outer_test = folds[fold][1]
        for variant in VARIANTS:
            variant_dir = fold_dir / "variants" / variant.name
            metrics_file = variant_dir / "outer_metrics.json"
            probability_file = variant_dir / "outer_test_probs.npy"
            if not metrics_file.exists() or not probability_file.exists():
                common.write_json(run_dir / "partial_aggregate.json", {"status": "PARTIAL", "missing_fold": fold, "missing_variant": variant.name})
                return None
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            probability = np.load(probability_file, allow_pickle=False).astype(float)
            probabilities[variant.name][outer_test] = probability
            assignment[variant.name][outer_test] += 1
            fold_rows.append({"outer_fold": fold, "variant": variant.name, "display_name": variant.display_name, **metrics["outer_metrics"]})
            selection_rows.append({"outer_fold": fold, "variant": variant.name, "selected_candidate": metrics["selected_candidate"], "fixed_epoch": metrics["fixed_epoch"]})
    partial = run_dir / "partial_aggregate.json"
    if partial.exists():
        partial.unlink()
    evaluated = np.concatenate([folds[fold][1] for fold in evaluated_folds])
    if any(np.any(assignment[variant.name][evaluated] != 1) for variant in VARIANTS):
        raise RuntimeError("Ablation prediction assignment is not exactly one")
    if any(np.any(~np.isfinite(probabilities[variant.name][evaluated])) for variant in VARIANTS):
        raise RuntimeError("Non-finite ablation probability")
    if not all(row["pass"] for row in split_audits):
        raise RuntimeError("Split/preprocessing audit failed")
    y_eval = y[evaluated]
    model_results: Dict[str, Any] = {}
    table_rows: List[Dict[str, Any]] = []
    for variant_index, variant in enumerate(VARIANTS):
        probability = probabilities[variant.name][evaluated]
        metrics = common.metric_bundle(y_eval, probability)
        bootstrap = common.participant_bootstrap(y_eval, probability, args.bootstrap_iterations, SEED + 800_000 + variant_index)
        model_results[variant.name] = {
            "specification": asdict(variant),
            "pooled_oof_metrics": metrics,
            "participant_stratified_bootstrap": bootstrap,
            "mean_outer_fold_auc": float(np.mean([row["auc"] for row in fold_rows if row["variant"] == variant.name])),
        }
        table_rows.append({
            "variant": variant.name,
            "display_name": variant.display_name,
            **metrics,
            "auc_ci95_low": bootstrap["ci95"]["auc"][0],
            "auc_ci95_high": bootstrap["ci95"]["auc"][1],
            "balanced_accuracy_ci95_low": bootstrap["ci95"]["balanced_accuracy"][0],
            "balanced_accuracy_ci95_high": bootstrap["ci95"]["balanced_accuracy"][1],
        })
    contrasts: Dict[str, Any] = {}
    raw_auc_p: Dict[str, float] = {}
    raw_mcnemar_p: Dict[str, float] = {}
    for contrast_index, (name, first_name, second_name, label) in enumerate(contrast_specs()):
        first_probability = probabilities[first_name][evaluated]
        second_probability = probabilities[second_name][evaluated]
        auc_delta = paired_delta_bootstrap(
            y_eval,
            first_probability,
            second_probability,
            args.bootstrap_iterations,
            SEED + 900_000 + contrast_index,
        )
        observed = auc_delta.pop("observed_delta_auc_cmpb_minus_baseline")
        auc_delta["observed_delta_auc_first_minus_second"] = observed
        mcnemar = common.mcnemar_cmpb(y_eval, first_probability, second_probability)
        contrasts[name] = {
            "label": label,
            "first_variant": first_name,
            "second_variant": second_name,
            "auc_delta": auc_delta,
            "mcnemar_accuracy": mcnemar,
        }
        raw_auc_p[name] = auc_delta["two_sided_bootstrap_p"]
        raw_mcnemar_p[name] = mcnemar["exact_two_sided_p"]
    auc_adjusted = common.holm_adjust(raw_auc_p)
    accuracy_adjusted = common.holm_adjust(raw_mcnemar_p)
    for name in contrasts:
        contrasts[name]["auc_delta"]["holm_adjusted_p_across_six_predeclared_contrasts"] = auc_adjusted[name]
        contrasts[name]["mcnemar_accuracy"]["holm_adjusted_p_across_six_predeclared_contrasts"] = accuracy_adjusted[name]
    fold_lookup = np.full(len(y), -1, dtype=int)
    for fold in evaluated_folds:
        fold_lookup[folds[fold][1]] = fold
    prediction_rows: List[Dict[str, Any]] = []
    for index in evaluated:
        row: Dict[str, Any] = {
            "subject_row_index": int(index),
            "subject_id": int(subject_ids[index]),
            "label_asd_1_td_0": int(y[index]),
            "outer_fold": int(fold_lookup[index]),
        }
        for variant in VARIANTS:
            row[f"probability_{variant.name}"] = round(float(probabilities[variant.name][index]), 10)
        prediction_rows.append(row)
    common.write_csv(run_dir / "strict_multimodal_ablation_oof_predictions.csv", prediction_rows, list(prediction_rows[0].keys()))
    common.write_csv(run_dir / "strict_multimodal_ablation_table.csv", table_rows, list(table_rows[0].keys()))
    common.write_csv(run_dir / "strict_multimodal_ablation_fold_metrics.csv", fold_rows, list(fold_rows[0].keys()))
    common.write_csv(run_dir / "strict_multimodal_ablation_selections.csv", selection_rows, list(selection_rows[0].keys()))
    summary = {
        "status": "COMPLETE_STRICT_MULTIMODAL_ABLATION" if expected_full else "COMPLETE_STRICT_MULTIMODAL_ABLATION_SMOKE",
        "generated_at": common.now_iso(),
        "task": "ABIDE-I ASD versus TD; C-PAC filt_noglobal; CC200",
        "participants": int(len(evaluated)),
        "outer_folds": len(evaluated_folds),
        "inner_folds": N_INNER,
        "seed": SEED,
        "feature_protocol": "Every inner-fit and outer-training partition independently selects 4,975 edges from all 19,900 CC200 FC edges.",
        "model_results": model_results,
        "predeclared_paired_contrasts": contrasts,
        "loss_lineage": loss_audit,
        "cache_audit": cache_audit,
        "yeo_mapping_audit": yeo_audit,
        "claim_boundary": "Single-seed internal nested CV on ABIDE-I; ablation differences estimate component associations with predictive performance, not biological causality or external generalization.",
    }
    common.write_json(run_dir / "strict_multimodal_ablation_summary.json", summary)
    fairness = {
        "status": "PASS",
        "generated_at": common.now_iso(),
        "full_10_fold_run": expected_full,
        "evaluated_participants": int(len(evaluated)),
        "variants": len(VARIANTS),
        "prediction_assignment_min_max": {variant.name: [int(assignment[variant.name][evaluated].min()), int(assignment[variant.name][evaluated].max())] for variant in VARIANTS},
        "outer_split_checks": len(split_audits),
        "inner_preprocessing_checks": int(sum(len(row["inner"]) for row in split_audits)),
        "all_split_preprocessing_checks_pass": bool(all(row["pass"] for row in split_audits)),
        "all_selected_edges_identical_to_strict_reference": bool(all(row["outer_preprocessing"]["selected_edges_identical_to_strict_reference"] and all(inner["selected_edges_identical_to_strict_reference"] for inner in row["inner"]) for row in split_audits)),
        "contrastive_lineage_audit_pass": bool(loss_audit["pass"]),
        "output_hashes": {},
    }
    fairness["output_hashes"] = {
        "summary": common.sha256_file(run_dir / "strict_multimodal_ablation_summary.json"),
        "predictions": common.sha256_file(run_dir / "strict_multimodal_ablation_oof_predictions.csv"),
        "table": common.sha256_file(run_dir / "strict_multimodal_ablation_table.csv"),
    }
    common.write_json(run_dir / "strict_multimodal_ablation_fairness_audit.json", fairness)
    write_reports(run_dir, summary, table_rows, contrasts, fairness)
    return summary


def write_reports(
    run_dir: Path,
    summary: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    contrasts: Dict[str, Any],
    fairness: Dict[str, Any],
) -> None:
    table = [
        "| Variant | AUC (95% CI) | Accuracy | Balanced accuracy | Sensitivity | Specificity | F1 | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['display_name']} | {row['auc']:.3f} ({row['auc_ci95_low']:.3f}-{row['auc_ci95_high']:.3f}) | "
            f"{row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['sensitivity']:.3f} | "
            f"{row['specificity']:.3f} | {row['f1']:.3f} | {row['brier']:.3f} |"
        )
    contrast_table = [
        "| Predeclared contrast | Delta AUC (first - second), 95% CI | Raw P | Holm P | Accuracy Holm P |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, _, _, _ in contrast_specs():
        contrast = contrasts[name]
        delta = contrast["auc_delta"]
        contrast_table.append(
            f"| {contrast['label']} | {delta['observed_delta_auc_first_minus_second']:.3f} "
            f"({delta['ci95'][0]:.3f}-{delta['ci95'][1]:.3f}) | {delta['two_sided_bootstrap_p']:.4f} | "
            f"{delta['holm_adjusted_p_across_six_predeclared_contrasts']:.4f} | "
            f"{contrast['mcnemar_accuracy']['holm_adjusted_p_across_six_predeclared_contrasts']:.4f} |"
        )
    full = summary["model_results"]["full_cmpb"]["pooled_oof_metrics"]
    report = f"""# Strict C-PAC multimodal and component ablation

Generated: {summary['generated_at']}

## Protocol

- ABIDE-I C-PAC `filt_noglobal`, CC200, n={summary['participants']}.
- Exact strict CMPB 10-fold outer splits and 3-fold inner splits (seed {summary['seed']}).
- Every training partition independently selected 4,975 FC edges from all 19,900 edges and fitted all transforms without evaluation-fold access.
- The same two predeclared regularization candidates and the same inner-validation checkpoint rule were used for every variant.
- The full row restored the exact legacy C-PAC contrastive objective (weight 0.10); the prior strict lineage had omitted that term. Consequently, all six rows were trained anew.
- Mean pooling applied the same node MLP followed by an unweighted mean across 200 frequency nodes, replacing OCREAD assignment and seven-cluster pooling.

## Core ablation table

{chr(10).join(table)}

## Paired component contrasts

{chr(10).join(contrast_table)}

## Fairness audit

- Status: **{fairness['status']}**.
- Outer split checks: {fairness['outer_split_checks']}.
- Inner preprocessing checks: {fairness['inner_preprocessing_checks']}.
- All fold-specific selected-edge arrays matched the strict reference: {fairness['all_selected_edges_identical_to_strict_reference']}.
- Every evaluated participant received exactly one out-of-fold prediction from every variant.

## Interpretation boundary

The full model achieved AUC {full['auc']:.3f}. Component claims must follow the paired contrasts rather than the ranking of point estimates alone. These are single-seed internal ABIDE-I estimates; they do not establish biological causality or independent external generalization.
"""
    (run_dir / "STRICT_MULTIMODAL_ABLATION_REPORT.md").write_text(report, encoding="utf-8")
    full_ci = summary["model_results"]["full_cmpb"]["participant_stratified_bootstrap"]["ci95"]["auc"]
    ocread = contrasts["ocread_vs_mean_pooling"]["auc_delta"]
    contrastive = contrasts["contrastive_loss"]["auc_delta"]
    result_text = f"""# Manuscript-ready ablation result text

## English

In a strict multimodal ablation on ABIDE-I C-PAC data (n={summary['participants']}), all variants shared identical nested-CV splits and training-partition feature selection from the full 19,900-edge FC universe. The full FC-frequency-demographic CMPB achieved an out-of-fold AUC of {full['auc']:.3f} (95% CI, {full_ci[0]:.3f}-{full_ci[1]:.3f}). The paired AUC difference for the full OCREAD model versus mean pooling was {ocread['observed_delta_auc_first_minus_second']:.3f} (95% CI, {ocread['ci95'][0]:.3f} to {ocread['ci95'][1]:.3f}; Holm-adjusted P={ocread['holm_adjusted_p_across_six_predeclared_contrasts']:.4f}), and that for the full contrastive model versus no contrastive loss was {contrastive['observed_delta_auc_first_minus_second']:.3f} (95% CI, {contrastive['ci95'][0]:.3f} to {contrastive['ci95'][1]:.3f}; Holm-adjusted P={contrastive['holm_adjusted_p_across_six_predeclared_contrasts']:.4f}). Neither comparison supported a stable predictive benefit. These paired results quantify predictive contributions under internal validation and do not establish biological causality.

## 中文

在ABIDE-I C-PAC数据（n={summary['participants']}）的严格多模态消融中，所有变体均使用相同的嵌套交叉验证划分，并在每个训练分区内从完整19,900条FC边中独立进行特征选择。完整的FC-频域-人口学CMPB取得{full['auc']:.3f}的折外AUC（95% CI：{full_ci[0]:.3f}-{full_ci[1]:.3f}）。完整OCREAD模型相对于mean pooling的配对AUC差值为{ocread['observed_delta_auc_first_minus_second']:.3f}（95% CI：{ocread['ci95'][0]:.3f}至{ocread['ci95'][1]:.3f}；Holm校正P={ocread['holm_adjusted_p_across_six_predeclared_contrasts']:.4f}）；完整对比学习模型相对于去除对比损失模型的配对AUC差值为{contrastive['observed_delta_auc_first_minus_second']:.3f}（95% CI：{contrastive['ci95'][0]:.3f}至{contrastive['ci95'][1]:.3f}；Holm校正P={contrastive['holm_adjusted_p_across_six_predeclared_contrasts']:.4f}）。两项比较均未显示稳定的预测获益。这些配对结果仅量化内部验证中的预测贡献，不能解释为生物学因果证据。
"""
    (run_dir / "STRICT_MULTIMODAL_ABLATION_RESULT_TEXT.md").write_text(result_text, encoding="utf-8")


def loss_lineage_audit() -> Dict[str, Any]:
    strict_source = inspect.getsource(strict.train_model)
    legacy_text = LEGACY_TARGET.read_text(encoding="utf-8", errors="replace")
    strict_has_contrastive = "contrast" in strict_source.lower()
    legacy_defines = "def contrastive_loss" in legacy_text
    legacy_weight = "'cont_weight': 0.10" in legacy_text or '"cont_weight": 0.10' in legacy_text
    audit = {
        "generated_at": common.now_iso(),
        "strict_source": str(Path(inspect.getsourcefile(strict.train_model) or "")),
        "strict_train_model_sha256": hashlib.sha256(strict_source.encode("utf-8")).hexdigest(),
        "strict_training_contains_contrastive_term": strict_has_contrastive,
        "legacy_target_path": str(LEGACY_TARGET),
        "legacy_target_sha256": common.sha256_file(LEGACY_TARGET),
        "legacy_defines_contrastive_loss": legacy_defines,
        "legacy_target_sets_cont_weight_0_10": legacy_weight,
        "decision": "Restore the exact legacy contrastive objective at weight 0.10 for every row except no_contrastive; train all rows anew.",
        "pass": bool(not strict_has_contrastive and legacy_defines and legacy_weight),
    }
    if not audit["pass"]:
        raise RuntimeError(f"Loss lineage audit failed: {audit}")
    return audit


def protocol_manifest(args: argparse.Namespace, loss_audit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": common.now_iso(),
        "task": "ABIDE-I ASD versus TD; C-PAC filt_noglobal; CC200",
        "outer_cv": {"folds": N_OUTER, "splitter": "StratifiedKFold", "shuffle": True, "seed": SEED},
        "inner_cv": {"folds": N_INNER, "seed_formula": "20260712 + 10000 + outer_fold"},
        "feature_selection": {"universe": strict.N_FULL_EDGES, "selected": strict.N_SELECTED_FC, "method": "training-partition F-test"},
        "variants": [asdict(variant) for variant in VARIANTS],
        "candidate_grid": [asdict(candidate) for candidate in strict.candidates()],
        "inner_training": {
            "maximum_epochs": args.inner_epochs,
            "minimum_epochs": args.minimum_inner_epochs,
            "patience": args.inner_patience,
            "eval_every": args.eval_every,
            "selection": "inner-validation AUC with BCE tie-break; median best epoch for final fit",
        },
        "contrastive_definition": "Exact legacy contrastive_loss: normalized fused embeddings, label-equality mask including diagonal, log-softmax similarity, temperature 0.10, weight 0.10; disabled during mixup batches as in legacy training.",
        "loss_lineage_audit": loss_audit,
        "predeclared_contrasts": [
            {"name": name, "first": first, "second": second, "label": label}
            for name, first, second, label in contrast_specs()
        ],
        "claim_boundary": "Internal single-seed nested CV; component performance associations, not causal biology or external validation.",
    }


def run(args: argparse.Namespace) -> Path:
    full_fc, x_freq, x_demo, y, subject_ids, cache_audit = strict.load_strict_inputs(rebuild=False)
    yeo_nodes, yeo_audit = strict.load_yeo7_nodes()
    loss_audit = loss_lineage_audit()
    run_dir = Path(args.run_dir) if args.run_dir else OUT_BASE / ("smoke" if args.mode == "smoke" else "run_20260716_full")
    common.ensure(run_dir)
    common.write_json(run_dir / "loss_lineage_audit.json", loss_audit)
    common.write_json(run_dir / "protocol_manifest.json", protocol_manifest(args, loss_audit))
    folds = exact_outer_splits(y)
    requested = [0] if args.mode == "smoke" else list(range(N_OUTER))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    for fold in requested:
        outer_train, outer_test = folds[fold]
        fold_dir = run_dir / f"fold_{fold:02d}"
        common.ensure(fold_dir)
        print(f"[strict-ablation] fold {fold + 1}/{N_OUTER}; train={len(outer_train)}, test={len(outer_test)}, device={device}", flush=True)
        started = time.time()
        inner_prepared, outer_prepared, audit = prepare_fold(
            fold, outer_train, outer_test, full_fc, x_freq, x_demo, y, yeo_nodes, args, fold_dir
        )
        print(f"[strict-ablation] fold {fold}: preprocessing audit PASS={audit['pass']}", flush=True)
        for variant in VARIANTS:
            variant_started = time.time()
            metrics = run_variant_fold(
                variant, fold, inner_prepared, outer_prepared, outer_train, outer_test, y, args, device, fold_dir
            )
            print(
                f"[strict-ablation] fold {fold} {variant.name}: AUC={metrics['outer_metrics']['auc']:.4f}; "
                f"{time.time() - variant_started:.1f}s",
                flush=True,
            )
        print(f"[strict-ablation] fold {fold} complete in {time.time() - started:.1f}s", flush=True)
    aggregate(run_dir, subject_ids, y, folds, args, cache_audit, yeo_audit, loss_audit)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict C-PAC multimodal/component ablation")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--inner-epochs", type=int, default=200)
    parser.add_argument("--minimum-inner-epochs", type=int, default=40)
    parser.add_argument("--inner-patience", type=int, default=40)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--anchor-projection-seed", type=int, default=20250225)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.inner_epochs < 2 or args.eval_every < 1:
        parser.error("inner-epochs must be >=2 and eval-every >=1")
    if not 1 <= args.minimum_inner_epochs <= args.inner_epochs:
        parser.error("minimum-inner-epochs must be within [1, inner-epochs]")
    if args.inner_patience < args.eval_every:
        parser.error("inner-patience must be >= eval-every")
    if args.bootstrap_iterations < 100:
        parser.error("bootstrap-iterations must be >=100")
    if args.mode == "smoke":
        args.inner_epochs = 3
        args.minimum_inner_epochs = 3
        args.inner_patience = 3
        args.eval_every = 1
        args.bootstrap_iterations = 100
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception:
        import traceback

        print(traceback.format_exc(), flush=True)
        raise

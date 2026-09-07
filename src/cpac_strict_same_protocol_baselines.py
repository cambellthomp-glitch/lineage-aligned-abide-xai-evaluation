"""Strict same-protocol baselines for the ABIDE-I C-PAC CMPB task.

All newly fitted baselines start from the complete 19,900-edge CC200 FC
universe.  The outer and inner splits are exactly those used by the completed
strict CMPB nested-CV run.  F-test selection (19,900 -> 4,975) and scaling are
fitted independently inside every applicable training partition.

The completed strict CMPB run is imported as an audited reference rather than
trained again.  Its split indices and selected-edge arrays are compared against
the independently reconstructed protocol in this script, fold by fold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import phase4_controlled_deep_artifact_rerun as controlled
import strict_nested_cv_interpretable_model as strict


ROOT = Path(__file__).resolve().parent
OUT_BASE = ROOT / "cpac_strict_same_protocol_baselines"
CMPB_RUN = ROOT / "strict_nested_cv_interpretability" / "run_20260712_strict_nested_cv"

SEED = strict.DEFAULT_SEED
N_OUTER = 10
N_INNER = 3
MODEL_ORDER = [
    "logistic_regression",
    "linear_svm",
    "rbf_svm",
    "simple_mlp",
    "fc_compact_deep",
    "cmpb_full",
]
DISPLAY_NAMES = {
    "logistic_regression": "Logistic regression",
    "linear_svm": "Linear SVM",
    "rbf_svm": "RBF-SVM",
    "simple_mlp": "Simple MLP",
    "fc_compact_deep": "FC-only compact deep model",
    "cmpb_full": "Full CMPB",
}
FC_ONLY_MODELS = MODEL_ORDER[:-1]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, value: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    ensure(path.parent)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def hash_edges(edges: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(edges, dtype=np.int64).tobytes()).hexdigest()


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


def safe_auc(y: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan")


def metric_bundle(y: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "auc": safe_auc(y, probability),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
    }


def clipped_log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    return float(log_loss(y, np.clip(probability, 1e-7, 1 - 1e-7), labels=[0, 1]))


@dataclass(frozen=True)
class ClassicalCandidate:
    name: str
    c: float
    gamma: str = "scale"


@dataclass(frozen=True)
class DeepCandidate:
    name: str
    hidden_dim: int
    dropout: float
    weight_decay: float
    residual_blocks: int = 0


def candidate_grid(model_name: str) -> List[Any]:
    if model_name == "logistic_regression":
        return [ClassicalCandidate("C_0.01", 0.01), ClassicalCandidate("C_0.1", 0.1), ClassicalCandidate("C_1", 1.0)]
    if model_name == "linear_svm":
        return [ClassicalCandidate("C_0.001", 0.001), ClassicalCandidate("C_0.01", 0.01), ClassicalCandidate("C_0.1", 0.1)]
    if model_name == "rbf_svm":
        return [ClassicalCandidate("C_0.1", 0.1), ClassicalCandidate("C_1", 1.0), ClassicalCandidate("C_10", 10.0)]
    if model_name == "simple_mlp":
        return [
            DeepCandidate("h64", 64, 0.25, 0.001),
            DeepCandidate("h128", 128, 0.35, 0.005),
        ]
    if model_name == "fc_compact_deep":
        return [
            DeepCandidate("base", 160, 0.25, 0.005, 2),
            DeepCandidate("regularized", 160, 0.35, 0.010, 2),
        ]
    raise KeyError(model_name)


@dataclass
class FCPrepared:
    selected_edges: np.ndarray
    f_scores: np.ndarray
    scaler: StandardScaler
    x_train: np.ndarray
    x_eval: np.ndarray


def prepare_fc(
    full_fc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> FCPrepared:
    selected, scores = strict.select_fc_from_training(
        full_fc[train_idx], y[train_idx], strict.N_SELECTED_FC
    )
    scaler = StandardScaler().fit(np.asarray(full_fc[train_idx][:, selected], dtype=np.float32))
    x_train = scaler.transform(np.asarray(full_fc[train_idx][:, selected], dtype=np.float32)).astype(np.float32)
    x_eval = scaler.transform(np.asarray(full_fc[eval_idx][:, selected], dtype=np.float32)).astype(np.float32)
    return FCPrepared(selected, scores, scaler, x_train, x_eval)


def save_prepared(
    folder: Path,
    prepared: FCPrepared,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    source_selected_path: Path,
) -> Dict[str, Any]:
    ensure(folder)
    np.save(folder / "fit_indices.npy", np.asarray(train_idx, dtype=np.int64))
    np.save(folder / "evaluation_indices.npy", np.asarray(eval_idx, dtype=np.int64))
    np.save(folder / "selected_full_edge_indices.npy", prepared.selected_edges.astype(np.int64))
    np.save(folder / "f_scores_full19900.npy", prepared.f_scores.astype(np.float32))
    with (folder / "fc_scaler.pkl").open("wb") as handle:
        pickle.dump(prepared.scaler, handle)
    source_selected = np.load(source_selected_path, allow_pickle=False).astype(np.int64)
    audit = {
        "fit_indices_sha256": hash_indices(train_idx),
        "evaluation_indices_sha256": hash_indices(eval_idx),
        "n_fit": int(len(train_idx)),
        "n_evaluation": int(len(eval_idx)),
        "feature_universe": int(strict.N_FULL_EDGES),
        "selected_features": int(len(prepared.selected_edges)),
        "selected_edges_sha256": hash_edges(prepared.selected_edges),
        "cmpb_selected_edges_sha256": hash_edges(source_selected),
        "selected_edges_identical_to_cmpb": bool(np.array_equal(prepared.selected_edges, source_selected)),
        "feature_selection_scope": "F-test fitted on fit_indices only",
        "scaling_scope": "StandardScaler fitted on fit_indices only after fold-specific selection",
    }
    write_json(folder / "preprocess_audit.json", audit)
    return audit


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class FCCompactDeep(nn.Module):
    """CMPB-style FC encoder and residual classifier without frequency/demo."""

    def __init__(self, input_dim: int, align_dim: int, dropout: float, blocks: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, align_dim),
            nn.LayerNorm(align_dim),
            nn.GELU(),
        )
        self.residual = nn.Sequential(
            *[controlled.SE_ResidualBlock(align_dim, dropout=dropout) for _ in range(blocks)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(align_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(align_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.residual(self.encoder(x))).squeeze(-1)


class TorchLogisticEstimator:
    """Deterministic L2 logistic regression with a full-batch LBFGS solver.

    This avoids the broken liblinear/scipy-LBFGSB binaries in the local Windows
    environment while retaining the standard convex logistic objective.
    """

    def __init__(self, c: float, seed: int):
        self.c = float(c)
        self.seed = int(seed)
        self.weight_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.n_iter_: Optional[int] = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchLogisticEstimator":
        set_seed(self.seed)
        x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
        y_tensor = torch.from_numpy(np.asarray(y, dtype=np.float32))
        layer = nn.Linear(x_tensor.shape[1], 1)
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        optimizer = torch.optim.LBFGS(
            layer.parameters(),
            lr=1.0,
            max_iter=300,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        iteration = {"count": 0}

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            logits = layer(x_tensor).squeeze(-1)
            penalty = layer.weight.square().sum() / (2.0 * self.c * len(x_tensor))
            loss = F.binary_cross_entropy_with_logits(logits, y_tensor) + penalty
            loss.backward()
            iteration["count"] += 1
            return loss

        optimizer.step(closure)
        self.weight_ = layer.weight.detach().cpu().numpy().reshape(-1).astype(np.float64)
        self.bias_ = float(layer.bias.detach().cpu().item())
        self.n_iter_ = int(iteration["count"])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weight_ is None or self.bias_ is None:
            raise RuntimeError("Estimator is not fitted")
        score = np.asarray(x, dtype=np.float64) @ self.weight_ + self.bias_
        positive = 1.0 / (1.0 + np.exp(-np.clip(score, -35.0, 35.0)))
        return np.column_stack([1.0 - positive, positive])


class TorchCalibratedSVMEstimator:
    """Linear or RBF RKHS SVM with training-only 3-fold Platt calibration."""

    def __init__(self, kernel: str, c: float, gamma: str, seed: int):
        if kernel not in {"linear", "rbf"}:
            raise ValueError(kernel)
        self.kernel = kernel
        self.c = float(c)
        self.gamma_spec = gamma
        self.seed = int(seed)
        self.weight_: Optional[np.ndarray] = None
        self.alpha_: Optional[np.ndarray] = None
        self.support_x_: Optional[np.ndarray] = None
        self.bias_: Optional[float] = None
        self.gamma_: Optional[float] = None
        self.calibrator_: Optional[TorchLogisticEstimator] = None
        self.calibration_protocol_: Optional[Dict[str, Any]] = None

    @staticmethod
    def _rbf_kernel(x: torch.Tensor, z: torch.Tensor, gamma: float) -> torch.Tensor:
        x_norm = x.square().sum(dim=1, keepdim=True)
        z_norm = z.square().sum(dim=1, keepdim=True).transpose(0, 1)
        distance = torch.clamp(x_norm + z_norm - 2.0 * (x @ z.transpose(0, 1)), min=0.0)
        return torch.exp(-gamma * distance)

    def _fit_raw(self, x: np.ndarray, y: np.ndarray, seed: int) -> Dict[str, Any]:
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
        y_sign = torch.from_numpy((2 * np.asarray(y, dtype=np.float32) - 1).astype(np.float32)).to(device)
        if self.kernel == "linear":
            weight = torch.zeros(x_tensor.shape[1], device=device, requires_grad=True)
            bias = torch.zeros((), device=device, requires_grad=True)
            optimizer = torch.optim.Adam([weight, bias], lr=0.03)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=400)
            for _ in range(400):
                optimizer.zero_grad(set_to_none=True)
                decision = x_tensor @ weight + bias
                loss = 0.5 * weight.square().sum() / len(x_tensor) + self.c * torch.relu(1.0 - y_sign * decision).mean()
                loss.backward()
                optimizer.step()
                scheduler.step()
            return {
                "weight": weight.detach().cpu().numpy().astype(np.float64),
                "bias": float(bias.detach().cpu().item()),
            }
        gamma = 1.0 / (x_tensor.shape[1] * max(float(x_tensor.var(unbiased=False).item()), 1e-12))
        kernel_matrix = self._rbf_kernel(x_tensor, x_tensor, gamma)
        alpha = torch.zeros(len(x_tensor), device=device, requires_grad=True)
        bias = torch.zeros((), device=device, requires_grad=True)
        optimizer = torch.optim.Adam([alpha, bias], lr=0.03)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
        for _ in range(500):
            optimizer.zero_grad(set_to_none=True)
            decision = kernel_matrix @ alpha + bias
            norm = 0.5 * alpha @ (kernel_matrix @ alpha) / len(x_tensor)
            loss = norm + self.c * torch.relu(1.0 - y_sign * decision).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
        return {
            "alpha": alpha.detach().cpu().numpy().astype(np.float64),
            "support_x": np.asarray(x, dtype=np.float32).copy(),
            "bias": float(bias.detach().cpu().item()),
            "gamma": float(gamma),
        }

    def _decision_from_raw(self, x: np.ndarray, raw: Dict[str, Any]) -> np.ndarray:
        if self.kernel == "linear":
            return np.asarray(x, dtype=np.float64) @ raw["weight"] + raw["bias"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
        support = torch.from_numpy(np.asarray(raw["support_x"], dtype=np.float32)).to(device)
        alpha = torch.from_numpy(np.asarray(raw["alpha"], dtype=np.float32)).to(device)
        values: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x_tensor), 256):
                kernel = self._rbf_kernel(x_tensor[start : start + 256], support, float(raw["gamma"]))
                values.append((kernel @ alpha + float(raw["bias"])).cpu().numpy())
        return np.concatenate(values).astype(np.float64)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchCalibratedSVMEstimator":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.seed + 91)
        calibration_scores = np.full(len(y), np.nan, dtype=np.float64)
        calibration_hashes: List[Dict[str, Any]] = []
        for fold, (fit_idx, calibration_idx) in enumerate(splitter.split(x, y)):
            raw = self._fit_raw(x[fit_idx], y[fit_idx], self.seed + fold)
            calibration_scores[calibration_idx] = self._decision_from_raw(x[calibration_idx], raw)
            calibration_hashes.append({
                "fold": fold,
                "fit_indices_sha256": hash_indices(fit_idx),
                "calibration_indices_sha256": hash_indices(calibration_idx),
                "overlap": int(len(np.intersect1d(fit_idx, calibration_idx))),
            })
        if np.any(~np.isfinite(calibration_scores)):
            raise RuntimeError("Incomplete SVM calibration scores")
        self.calibrator_ = TorchLogisticEstimator(c=1.0, seed=self.seed + 999).fit(calibration_scores.reshape(-1, 1), y)
        raw_final = self._fit_raw(x, y, self.seed + 777)
        self.bias_ = float(raw_final["bias"])
        if self.kernel == "linear":
            self.weight_ = raw_final["weight"]
        else:
            self.alpha_ = raw_final["alpha"]
            self.support_x_ = raw_final["support_x"]
            self.gamma_ = float(raw_final["gamma"])
        self.calibration_protocol_ = {
            "method": "training-partition 3-fold out-of-fold Platt calibration",
            "folds": calibration_hashes,
            "all_overlaps_zero": bool(all(row["overlap"] == 0 for row in calibration_hashes)),
        }
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self.bias_ is None:
            raise RuntimeError("Estimator is not fitted")
        if self.kernel == "linear":
            if self.weight_ is None:
                raise RuntimeError("Missing linear weight")
            return np.asarray(x, dtype=np.float64) @ self.weight_ + self.bias_
        if self.alpha_ is None or self.support_x_ is None or self.gamma_ is None:
            raise RuntimeError("Missing RBF parameters")
        return self._decision_from_raw(x, {
            "alpha": self.alpha_, "support_x": self.support_x_, "bias": self.bias_, "gamma": self.gamma_
        })

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.calibrator_ is None:
            raise RuntimeError("Missing Platt calibrator")
        return self.calibrator_.predict_proba(self.decision_function(x).reshape(-1, 1))


def make_deep_model(model_name: str, candidate: DeepCandidate, device: torch.device) -> nn.Module:
    if model_name == "simple_mlp":
        model = SimpleMLP(strict.N_SELECTED_FC, candidate.hidden_dim, candidate.dropout)
    elif model_name == "fc_compact_deep":
        model = FCCompactDeep(strict.N_SELECTED_FC, candidate.hidden_dim, candidate.dropout, candidate.residual_blocks)
    else:
        raise KeyError(model_name)
    return model.to(device)


def torch_probabilities(model: nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    values: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 128):
            logits = model(torch.from_numpy(x[start : start + 128]).to(device))
            values.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(values).reshape(-1)


def train_deep(
    model_name: str,
    candidate: DeepCandidate,
    x_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    eval_every: int,
    monitor: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    patience: Optional[int] = None,
    minimum_epochs: int = 0,
) -> Tuple[nn.Module, Dict[str, Any]]:
    set_seed(seed)
    model = make_deep_model(model_name, candidate, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=candidate.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32))),
        batch_size=32,
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
        model.train()
        losses: List[float] = []
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = F.binary_cross_entropy_with_logits(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        scheduler.step()
        if monitor is not None and (epoch % eval_every == 0 or epoch == epochs):
            probability = torch_probabilities(model, monitor[0], device)
            auc = safe_auc(monitor[1], probability)
            bce = clipped_log_loss(monitor[1], probability)
            history.append({"epoch": epoch, "train_bce": float(np.mean(losses)), "validation_auc": auc, "validation_bce": bce})
            improved = auc > best_auc + 1e-12 or (abs(auc - best_auc) <= 1e-12 and bce < best_bce)
            if improved:
                best_auc, best_bce, best_epoch = auc, bce, epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                last_improved = epoch
            if patience is not None and epoch >= minimum_epochs and last_improved > 0 and epoch - last_improved >= patience:
                stopped_epoch = epoch
                break
    if monitor is None:
        best_epoch = epochs
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append({"epoch": epochs, "checkpoint_policy": "fixed_inner_selected_epoch"})
    if best_state is None:
        raise RuntimeError("No deep-model state was selected")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "best_epoch": int(best_epoch),
        "best_validation_auc": None if monitor is None else float(best_auc),
        "best_validation_bce": None if monitor is None else float(best_bce),
        "stopped_epoch": int(stopped_epoch),
        "maximum_epochs": int(epochs),
        "patience": patience,
        "minimum_epochs": int(minimum_epochs),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "history": history,
    }


def fit_classical(
    model_name: str,
    candidate: ClassicalCandidate,
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> Any:
    if model_name == "logistic_regression":
        estimator = TorchLogisticEstimator(candidate.c, seed)
    elif model_name == "linear_svm":
        estimator = TorchCalibratedSVMEstimator("linear", candidate.c, candidate.gamma, seed)
    elif model_name == "rbf_svm":
        estimator = TorchCalibratedSVMEstimator("rbf", candidate.c, candidate.gamma, seed)
    else:
        raise KeyError(model_name)
    estimator.fit(x_train, y_train)
    return estimator


def select_candidate(rows: Sequence[Dict[str, Any]], model_name: str) -> Tuple[Any, Optional[int], Dict[str, Any]]:
    grid = candidate_grid(model_name)
    aggregated: Dict[str, Dict[str, Any]] = {}
    for candidate in grid:
        subset = [row for row in rows if row["candidate"] == candidate.name]
        if len(subset) != N_INNER:
            raise RuntimeError(f"{model_name}/{candidate.name} has {len(subset)} inner rows, expected {N_INNER}")
        entry = {
            "mean_inner_auc": float(np.mean([row["validation_auc"] for row in subset])),
            "mean_inner_bce": float(np.mean([row["validation_bce"] for row in subset])),
        }
        if "best_epoch" in subset[0]:
            epochs = [int(row["best_epoch"]) for row in subset]
            entry["fold_epochs"] = epochs
            entry["median_epoch"] = int(np.median(epochs))
        aggregated[candidate.name] = entry
    winner_name = sorted(
        aggregated,
        key=lambda name: (-aggregated[name]["mean_inner_auc"], aggregated[name]["mean_inner_bce"], name),
    )[0]
    by_name = {candidate.name: candidate for candidate in grid}
    fixed_epoch = aggregated[winner_name].get("median_epoch")
    selection = {
        "model": model_name,
        "all_candidates": aggregated,
        "selected_candidate": winner_name,
        "final_fixed_epoch": fixed_epoch,
        "selection_rule": "maximum mean inner-validation AUC; lower mean BCE and then candidate name break ties; deep final epoch is the median inner selected epoch",
    }
    return by_name[winner_name], fixed_epoch, selection


def exact_splits(y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=SEED)
    return [
        (train.astype(np.int64), test.astype(np.int64))
        for train, test in splitter.split(np.zeros(len(y)), y)
    ]


def inner_splits(outer_fold: int, outer_train: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=SEED + 10_000 + outer_fold)
    return [
        (outer_train[fit].astype(np.int64), outer_train[validation].astype(np.int64))
        for fit, validation in splitter.split(outer_train, y[outer_train])
    ]


def cmpb_inner_selected_path(outer_fold: int, inner_fold: int) -> Path:
    return CMPB_RUN / f"fold_{outer_fold:02d}" / "inner_cv" / f"inner_{inner_fold:02d}" / "base" / "selected_full_edge_indices.npy"


def prepare_outer_fold(
    fold: int,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    full_fc: np.ndarray,
    y: np.ndarray,
    fold_dir: Path,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray, FCPrepared]], FCPrepared, Dict[str, Any]]:
    cmpb_folder = CMPB_RUN / f"fold_{fold:02d}"
    cmpb_train = np.load(cmpb_folder / "outer_train_indices.npy", allow_pickle=False).astype(np.int64)
    cmpb_test = np.load(cmpb_folder / "outer_test_indices.npy", allow_pickle=False).astype(np.int64)
    split_audit: Dict[str, Any] = {
        "outer_fold": fold,
        "outer_train_indices_identical_to_cmpb": bool(np.array_equal(outer_train, cmpb_train)),
        "outer_test_indices_identical_to_cmpb": bool(np.array_equal(outer_test, cmpb_test)),
        "outer_train_test_index_overlap": int(len(np.intersect1d(outer_train, outer_test))),
        "outer_train_indices_sha256": hash_indices(outer_train),
        "outer_test_indices_sha256": hash_indices(outer_test),
        "inner": [],
    }
    prepared_inner: List[Tuple[np.ndarray, np.ndarray, FCPrepared]] = []
    for inner_fold, (fit_idx, val_idx) in enumerate(inner_splits(fold, outer_train, y)):
        prepared = prepare_fc(full_fc, y, fit_idx, val_idx)
        audit = save_prepared(
            fold_dir / "preprocessing" / f"inner_{inner_fold:02d}",
            prepared,
            fit_idx,
            val_idx,
            cmpb_inner_selected_path(fold, inner_fold),
        )
        audit["inner_fold"] = inner_fold
        audit["fit_validation_index_overlap"] = int(len(np.intersect1d(fit_idx, val_idx)))
        split_audit["inner"].append(audit)
        prepared_inner.append((fit_idx, val_idx, prepared))
    outer_prepared = prepare_fc(full_fc, y, outer_train, outer_test)
    outer_audit = save_prepared(
        fold_dir / "preprocessing" / "outer_final",
        outer_prepared,
        outer_train,
        outer_test,
        cmpb_folder / "selected_full_edge_indices.npy",
    )
    split_audit["outer_preprocessing"] = outer_audit
    split_audit["pass"] = bool(
        split_audit["outer_train_indices_identical_to_cmpb"]
        and split_audit["outer_test_indices_identical_to_cmpb"]
        and split_audit["outer_train_test_index_overlap"] == 0
        and outer_audit["selected_edges_identical_to_cmpb"]
        and all(item["selected_edges_identical_to_cmpb"] and item["fit_validation_index_overlap"] == 0 for item in split_audit["inner"])
    )
    write_json(fold_dir / "split_and_preprocessing_audit.json", split_audit)
    if not split_audit["pass"]:
        raise RuntimeError(f"Fairness audit failed for outer fold {fold}")
    return prepared_inner, outer_prepared, split_audit


def run_model_fold(
    model_name: str,
    fold: int,
    prepared_inner: Sequence[Tuple[np.ndarray, np.ndarray, FCPrepared]],
    outer_prepared: FCPrepared,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    y: np.ndarray,
    fold_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    model_dir = fold_dir / "models" / model_name
    ensure(model_dir)
    metrics_file = model_dir / "outer_metrics.json"
    probs_file = model_dir / "outer_test_probs.npy"
    if args.resume and metrics_file.exists() and probs_file.exists():
        return json.loads(metrics_file.read_text(encoding="utf-8"))
    inner_rows: List[Dict[str, Any]] = []
    for inner_fold, (fit_idx, val_idx, prepared) in enumerate(prepared_inner):
        for candidate_index, candidate in enumerate(candidate_grid(model_name)):
            trial_dir = model_dir / "inner_cv" / f"inner_{inner_fold:02d}" / candidate.name
            trial_file = trial_dir / "inner_metrics.json"
            if args.resume and trial_file.exists():
                inner_rows.append(json.loads(trial_file.read_text(encoding="utf-8")))
                continue
            ensure(trial_dir)
            trial_seed = SEED + fold * 1000 + inner_fold * 100 + candidate_index + 30_000 * MODEL_ORDER.index(model_name)
            started = time.time()
            if model_name in ("logistic_regression", "linear_svm", "rbf_svm"):
                estimator = fit_classical(model_name, candidate, prepared.x_train, y[fit_idx], trial_seed)
                probability = estimator.predict_proba(prepared.x_eval)[:, 1]
                training_meta = {"fit_seconds": time.time() - started}
            else:
                estimator, training_meta = train_deep(
                    model_name,
                    candidate,
                    prepared.x_train,
                    y[fit_idx],
                    device,
                    trial_seed,
                    args.inner_epochs,
                    args.eval_every,
                    monitor=(prepared.x_eval, y[val_idx]),
                    patience=args.inner_patience,
                    minimum_epochs=args.minimum_inner_epochs,
                )
                probability = torch_probabilities(estimator, prepared.x_eval, device)
            row = {
                "outer_fold": fold,
                "inner_fold": inner_fold,
                "model": model_name,
                "candidate": candidate.name,
                "candidate_config": asdict(candidate),
                "validation_auc": safe_auc(y[val_idx], probability),
                "validation_bce": clipped_log_loss(y[val_idx], probability),
                "n_fit": int(len(fit_idx)),
                "n_validation": int(len(val_idx)),
                "fit_indices_sha256": hash_indices(fit_idx),
                "validation_indices_sha256": hash_indices(val_idx),
                "selected_edges_sha256": hash_edges(prepared.selected_edges),
                "training_meta": training_meta,
            }
            if "best_epoch" in training_meta:
                row["best_epoch"] = int(training_meta["best_epoch"])
            write_json(trial_file, row)
            inner_rows.append(row)
            del estimator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    selected, fixed_epoch, selection = select_candidate(inner_rows, model_name)
    write_json(model_dir / "inner_selection.json", selection)
    final_seed = SEED + fold * 1000 + 777 + 30_000 * MODEL_ORDER.index(model_name)
    started = time.time()
    if model_name in ("logistic_regression", "linear_svm", "rbf_svm"):
        final_model = fit_classical(model_name, selected, outer_prepared.x_train, y[outer_train], final_seed)
        probability = final_model.predict_proba(outer_prepared.x_eval)[:, 1]
        with (model_dir / "model.pkl").open("wb") as handle:
            pickle.dump(final_model, handle)
        final_meta = {"fit_seconds": time.time() - started, "fixed_epoch": None}
    else:
        if fixed_epoch is None or fixed_epoch < 1:
            raise RuntimeError(f"No fixed epoch selected for {model_name}")
        final_model, final_meta = train_deep(
            model_name,
            selected,
            outer_prepared.x_train,
            y[outer_train],
            device,
            final_seed,
            fixed_epoch,
            args.eval_every,
            monitor=None,
        )
        probability = torch_probabilities(final_model, outer_prepared.x_eval, device)
        torch.save(
            {
                "model": model_name,
                "model_state_dict": {key: value.detach().cpu() for key, value in final_model.state_dict().items()},
                "candidate": asdict(selected),
                "fixed_epoch": fixed_epoch,
                "outer_train_indices_sha256": hash_indices(outer_train),
                "outer_test_indices_sha256": hash_indices(outer_test),
                "selected_edges_sha256": hash_edges(outer_prepared.selected_edges),
            },
            model_dir / "model_checkpoint.pt",
        )
    np.save(probs_file, np.asarray(probability, dtype=np.float32))
    np.save(model_dir / "outer_test_indices.npy", outer_test.astype(np.int64))
    metrics = {
        "status": "COMPLETE",
        "outer_fold": fold,
        "model": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "selected_candidate": selected.name,
        "selected_candidate_config": asdict(selected),
        "fixed_epoch": fixed_epoch,
        "outer_metrics": metric_bundle(y[outer_test], probability),
        "n_outer_train": int(len(outer_train)),
        "n_outer_test": int(len(outer_test)),
        "outer_train_indices_sha256": hash_indices(outer_train),
        "outer_test_indices_sha256": hash_indices(outer_test),
        "selected_edges_sha256": hash_edges(outer_prepared.selected_edges),
        "final_training_meta": final_meta,
        "outer_test_usage": "One deterministic evaluation after all preprocessing, candidate and epoch selection were fixed without outer-test access.",
    }
    write_json(metrics_file, metrics)
    del final_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def load_cmpb_fold(fold: int, outer_test: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    folder = CMPB_RUN / f"fold_{fold:02d}"
    source_idx = np.load(folder / "outer_test_indices.npy", allow_pickle=False).astype(np.int64)
    if not np.array_equal(source_idx, outer_test):
        raise RuntimeError(f"CMPB outer indices do not match fold {fold}")
    probability = np.load(folder / "outer_test_probs.npy", allow_pickle=False).astype(float)
    source_metrics = json.loads((folder / "fold_metrics.json").read_text(encoding="utf-8"))
    metrics = {
        "status": "COMPLETE_IMPORTED_AUDITED_REFERENCE",
        "outer_fold": fold,
        "model": "cmpb_full",
        "display_name": DISPLAY_NAMES["cmpb_full"],
        "selected_candidate": source_metrics["selected_candidate"],
        "fixed_epoch": source_metrics["final_fixed_epoch"],
        "outer_metrics": metric_bundle(y[outer_test], probability),
        "n_outer_train": int(source_metrics["n_outer_train"]),
        "n_outer_test": int(source_metrics["n_outer_test"]),
        "source_fold_metrics_sha256": sha256_file(folder / "fold_metrics.json"),
        "source_outer_probs_sha256": sha256_file(folder / "outer_test_probs.npy"),
        "source_run": str(CMPB_RUN),
    }
    return probability, metrics


def participant_bootstrap(
    y: np.ndarray,
    probability: np.ndarray,
    iterations: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == value) for value in (0, 1)]
    values: Dict[str, List[float]] = {name: [] for name in ("auc", "accuracy", "balanced_accuracy", "brier")}
    for _ in range(iterations):
        sampled = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in classes])
        rng.shuffle(sampled)
        bundle = metric_bundle(y[sampled], probability[sampled])
        for name in values:
            values[name].append(float(bundle[name]))
    return {
        "method": "participant-stratified bootstrap",
        "iterations": iterations,
        "ci95": {name: np.quantile(samples, [0.025, 0.975]).tolist() for name, samples in values.items()},
    }


def paired_auc_delta_bootstrap(
    y: np.ndarray,
    cmpb: np.ndarray,
    baseline: np.ndarray,
    iterations: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(y == value) for value in (0, 1)]
    deltas: List[float] = []
    for _ in range(iterations):
        sampled = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in classes])
        rng.shuffle(sampled)
        deltas.append(safe_auc(y[sampled], cmpb[sampled]) - safe_auc(y[sampled], baseline[sampled]))
    delta_array = np.asarray(deltas, dtype=float)
    non_positive = (np.sum(delta_array <= 0) + 1) / (iterations + 1)
    non_negative = (np.sum(delta_array >= 0) + 1) / (iterations + 1)
    return {
        "observed_delta_auc_cmpb_minus_baseline": safe_auc(y, cmpb) - safe_auc(y, baseline),
        "ci95": np.quantile(delta_array, [0.025, 0.975]).tolist(),
        "two_sided_bootstrap_p": float(min(1.0, 2 * min(non_positive, non_negative))),
        "iterations": iterations,
    }


def holm_adjust(p_values: Dict[str, float]) -> Dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: Dict[str, float] = {}
    running = 0.0
    m = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (m - rank) * float(p_values[name]))
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def mcnemar_cmpb(y: np.ndarray, cmpb: np.ndarray, baseline: np.ndarray) -> Dict[str, Any]:
    cmpb_correct = (cmpb >= 0.5).astype(int) == y
    base_correct = (baseline >= 0.5).astype(int) == y
    cmpb_only = int(np.sum(cmpb_correct & ~base_correct))
    baseline_only = int(np.sum(~cmpb_correct & base_correct))
    discordant = cmpb_only + baseline_only
    p_value = float(binomtest(cmpb_only, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "cmpb_correct_baseline_wrong": cmpb_only,
        "cmpb_wrong_baseline_correct": baseline_only,
        "discordant": discordant,
        "exact_two_sided_p": p_value,
    }


def aggregate(
    run_dir: Path,
    subject_ids: np.ndarray,
    y: np.ndarray,
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    cache_audit: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    evaluated_folds = list(range(1 if args.mode == "smoke" else N_OUTER))
    expected_full = len(evaluated_folds) == N_OUTER
    probabilities = {name: np.full(len(y), np.nan, dtype=float) for name in MODEL_ORDER}
    assignment = {name: np.zeros(len(y), dtype=int) for name in MODEL_ORDER}
    fold_rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    selection_rows: List[Dict[str, Any]] = []
    for fold in evaluated_folds:
        fold_dir = run_dir / f"fold_{fold:02d}"
        split_file = fold_dir / "split_and_preprocessing_audit.json"
        if not split_file.exists():
            return None
        split_audit = json.loads(split_file.read_text(encoding="utf-8"))
        audit_rows.append(split_audit)
        outer_test = folds[fold][1]
        for model_name in FC_ONLY_MODELS:
            model_dir = fold_dir / "models" / model_name
            metrics_file = model_dir / "outer_metrics.json"
            probs_file = model_dir / "outer_test_probs.npy"
            if not metrics_file.exists() or not probs_file.exists():
                write_json(run_dir / "partial_aggregate.json", {"status": "PARTIAL", "missing_fold": fold, "missing_model": model_name, "generated_at": now_iso()})
                return None
            probability = np.load(probs_file, allow_pickle=False).astype(float)
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            probabilities[model_name][outer_test] = probability
            assignment[model_name][outer_test] += 1
            fold_rows.append({"outer_fold": fold, "model": model_name, "display_name": DISPLAY_NAMES[model_name], **metrics["outer_metrics"]})
            selection_rows.append({"outer_fold": fold, "model": model_name, "selected_candidate": metrics["selected_candidate"], "fixed_epoch": metrics["fixed_epoch"]})
        cmpb_probability, cmpb_metrics = load_cmpb_fold(fold, outer_test, y)
        probabilities["cmpb_full"][outer_test] = cmpb_probability
        assignment["cmpb_full"][outer_test] += 1
        fold_rows.append({"outer_fold": fold, "model": "cmpb_full", "display_name": DISPLAY_NAMES["cmpb_full"], **cmpb_metrics["outer_metrics"]})
        selection_rows.append({"outer_fold": fold, "model": "cmpb_full", "selected_candidate": cmpb_metrics["selected_candidate"], "fixed_epoch": cmpb_metrics["fixed_epoch"]})
    partial = run_dir / "partial_aggregate.json"
    if partial.exists():
        partial.unlink()
    evaluated = np.concatenate([folds[fold][1] for fold in evaluated_folds])
    if any(np.any(assignment[name][evaluated] != 1) for name in MODEL_ORDER):
        raise RuntimeError("Prediction assignment is not exactly one per evaluated participant/model")
    if any(np.any(~np.isfinite(probabilities[name][evaluated])) for name in MODEL_ORDER):
        raise RuntimeError("Non-finite OOF probability detected")
    if not all(row["pass"] for row in audit_rows):
        raise RuntimeError("Split/preprocessing fairness audit failed")
    y_eval = y[evaluated]
    model_summary: Dict[str, Any] = {}
    comparison_rows: List[Dict[str, Any]] = []
    for model_index, model_name in enumerate(MODEL_ORDER):
        probability = probabilities[model_name][evaluated]
        metrics = metric_bundle(y_eval, probability)
        bootstrap = participant_bootstrap(y_eval, probability, args.bootstrap_iterations, SEED + 500_000 + model_index)
        model_summary[model_name] = {
            "display_name": DISPLAY_NAMES[model_name],
            "input": "fold-selected FC (4,975 of 19,900)" if model_name != "cmpb_full" else "fold-selected FC + frequency + demographics",
            "pooled_oof_metrics": metrics,
            "participant_stratified_bootstrap": bootstrap,
            "mean_outer_fold_auc": float(np.mean([row["auc"] for row in fold_rows if row["model"] == model_name])),
            "mean_outer_fold_balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in fold_rows if row["model"] == model_name])),
        }
        comparison_rows.append({
            "model": model_name,
            "display_name": DISPLAY_NAMES[model_name],
            **metrics,
            "auc_ci95_low": bootstrap["ci95"]["auc"][0],
            "auc_ci95_high": bootstrap["ci95"]["auc"][1],
            "balanced_accuracy_ci95_low": bootstrap["ci95"]["balanced_accuracy"][0],
            "balanced_accuracy_ci95_high": bootstrap["ci95"]["balanced_accuracy"][1],
        })
    paired: Dict[str, Any] = {}
    raw_auc_p: Dict[str, float] = {}
    raw_mcnemar_p: Dict[str, float] = {}
    cmpb = probabilities["cmpb_full"][evaluated]
    for model_index, model_name in enumerate(FC_ONLY_MODELS):
        baseline = probabilities[model_name][evaluated]
        auc_delta = paired_auc_delta_bootstrap(y_eval, cmpb, baseline, args.bootstrap_iterations, SEED + 600_000 + model_index)
        mcnemar = mcnemar_cmpb(y_eval, cmpb, baseline)
        paired[model_name] = {"auc_delta": auc_delta, "mcnemar_accuracy": mcnemar}
        raw_auc_p[model_name] = auc_delta["two_sided_bootstrap_p"]
        raw_mcnemar_p[model_name] = mcnemar["exact_two_sided_p"]
    auc_adjusted = holm_adjust(raw_auc_p)
    mcnemar_adjusted = holm_adjust(raw_mcnemar_p)
    for model_name in FC_ONLY_MODELS:
        paired[model_name]["auc_delta"]["holm_adjusted_p_across_five_cmpb_comparisons"] = auc_adjusted[model_name]
        paired[model_name]["mcnemar_accuracy"]["holm_adjusted_p_across_five_cmpb_comparisons"] = mcnemar_adjusted[model_name]
    prediction_rows: List[Dict[str, Any]] = []
    fold_lookup = np.full(len(y), -1, dtype=int)
    for fold in evaluated_folds:
        fold_lookup[folds[fold][1]] = fold
    for idx in evaluated:
        row: Dict[str, Any] = {
            "subject_row_index": int(idx),
            "subject_id": int(subject_ids[idx]),
            "label_asd_1_td_0": int(y[idx]),
            "outer_fold": int(fold_lookup[idx]),
        }
        for model_name in MODEL_ORDER:
            row[f"probability_{model_name}"] = round(float(probabilities[model_name][idx]), 10)
        prediction_rows.append(row)
    write_csv(run_dir / "strict_same_protocol_oof_predictions.csv", prediction_rows, list(prediction_rows[0].keys()))
    write_csv(run_dir / "strict_same_protocol_model_comparison.csv", comparison_rows, list(comparison_rows[0].keys()))
    write_csv(run_dir / "strict_same_protocol_fold_metrics.csv", fold_rows, list(fold_rows[0].keys()))
    write_csv(run_dir / "strict_same_protocol_selections.csv", selection_rows, list(selection_rows[0].keys()))
    fairness = {
        "status": "PASS",
        "generated_at": now_iso(),
        "full_10_fold_run": expected_full,
        "evaluated_participants": int(len(evaluated)),
        "prediction_assignment_min_max_by_model": {name: [int(assignment[name][evaluated].min()), int(assignment[name][evaluated].max())] for name in MODEL_ORDER},
        "outer_split_checks": len(audit_rows),
        "inner_preprocessing_checks": int(sum(len(row["inner"]) for row in audit_rows)),
        "all_split_and_preprocessing_checks_pass": bool(all(row["pass"] for row in audit_rows)),
        "all_selected_edge_arrays_identical_to_cmpb": bool(all(row["outer_preprocessing"]["selected_edges_identical_to_cmpb"] and all(inner["selected_edges_identical_to_cmpb"] for inner in row["inner"]) for row in audit_rows)),
        "feature_selection_universe": strict.N_FULL_EDGES,
        "features_selected_per_training_partition": strict.N_SELECTED_FC,
        "cmpb_reference_run": str(CMPB_RUN),
        "cmpb_reference_summary_sha256": sha256_file(CMPB_RUN / "strict_nested_summary.json"),
        "output_hashes": {},
    }
    summary = {
        "status": "COMPLETE_STRICT_SAME_PROTOCOL_BASELINES" if expected_full else "COMPLETE_STRICT_SAME_PROTOCOL_BASELINES_SMOKE",
        "generated_at": now_iso(),
        "task": "ABIDE-I ASD versus TD; C-PAC filt_noglobal; CC200",
        "participants": int(len(evaluated)),
        "outer_folds": len(evaluated_folds),
        "inner_folds": N_INNER,
        "outer_seed": SEED,
        "feature_protocol": "Every inner/outer training partition independently selects 4,975 FC edges by F-test from all 19,900 CC200 edges, then fits its own StandardScaler.",
        "model_results": model_summary,
        "paired_cmpb_comparisons": paired,
        "cmpb_reference_boundary": "The full CMPB column is imported from the previously completed strict nested-CV run after exact split and selected-edge identity checks; legacy globally filtered ML results are not used.",
        "claim_boundary": "Single-seed internal nested cross-validation on ABIDE-I; this comparison isolates performance under a leakage-controlled common protocol but is not external validation.",
        "cache_audit": cache_audit,
    }
    write_json(run_dir / "strict_same_protocol_summary.json", summary)
    fairness["output_hashes"] = {
        "summary": sha256_file(run_dir / "strict_same_protocol_summary.json"),
        "predictions": sha256_file(run_dir / "strict_same_protocol_oof_predictions.csv"),
        "comparison": sha256_file(run_dir / "strict_same_protocol_model_comparison.csv"),
    }
    write_json(run_dir / "strict_same_protocol_fairness_audit.json", fairness)
    write_reports(run_dir, summary, comparison_rows, paired, fairness)
    return summary


def write_reports(
    run_dir: Path,
    summary: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    paired: Dict[str, Any],
    fairness: Dict[str, Any],
) -> None:
    table_lines = [
        "| Model | Input | AUC (95% CI) | Accuracy | Balanced accuracy | Sensitivity | Specificity | F1 | Brier |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        input_name = "FC" if row["model"] != "cmpb_full" else "FC + frequency + demographics"
        table_lines.append(
            f"| {row['display_name']} | {input_name} | {row['auc']:.3f} ({row['auc_ci95_low']:.3f}-{row['auc_ci95_high']:.3f}) | "
            f"{row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['sensitivity']:.3f} | {row['specificity']:.3f} | {row['f1']:.3f} | {row['brier']:.3f} |"
        )
    delta_lines = [
        "| Baseline | Delta AUC (CMPB - baseline), 95% CI | Bootstrap P | Holm P | McNemar Holm P |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_name in FC_ONLY_MODELS:
        auc_delta = paired[model_name]["auc_delta"]
        mcnemar = paired[model_name]["mcnemar_accuracy"]
        delta_lines.append(
            f"| {DISPLAY_NAMES[model_name]} | {auc_delta['observed_delta_auc_cmpb_minus_baseline']:.3f} "
            f"({auc_delta['ci95'][0]:.3f}-{auc_delta['ci95'][1]:.3f}) | {auc_delta['two_sided_bootstrap_p']:.4f} | "
            f"{auc_delta['holm_adjusted_p_across_five_cmpb_comparisons']:.4f} | {mcnemar['holm_adjusted_p_across_five_cmpb_comparisons']:.4f} |"
        )
    cmpb = summary["model_results"]["cmpb_full"]["pooled_oof_metrics"]
    best_baseline_name = max(FC_ONLY_MODELS, key=lambda name: summary["model_results"][name]["pooled_oof_metrics"]["auc"])
    best_baseline = summary["model_results"][best_baseline_name]["pooled_oof_metrics"]
    report = f"""# Strict same-protocol C-PAC baseline comparison

Generated: {summary['generated_at']}

## Protocol

- Task: ABIDE-I ASD versus TD, C-PAC `filt_noglobal`, CC200, n={summary['participants']}.
- Outer protocol: the same {summary['outer_folds']}-fold split used by the strict CMPB run (seed {summary['outer_seed']}).
- Inner protocol: {summary['inner_folds']}-fold model/candidate selection inside each outer-training fold.
- FC selection: every inner-fit and outer-training partition independently selects exactly 4,975 edges from all 19,900 edges by training-only F-test.
- Scaling: fitted only after fold-specific selection and only on the applicable training partition.
- Threshold: fixed a priori at 0.5.
- Solvers: L2 logistic regression uses a deterministic full-batch LBFGS optimizer; linear and RBF SVMs minimize hinge loss in PyTorch, with RBF `gamma=scale` and training-partition-only 3-fold Platt probability calibration.
- Full CMPB: imported from the completed strict nested-CV run only after exact outer/inner split and selected-edge identity checks.

## Pooled out-of-fold performance

{chr(10).join(table_lines)}

## Paired comparisons against full CMPB

{chr(10).join(delta_lines)}

## Fairness audit

- Status: **{fairness['status']}**.
- Outer split checks: {fairness['outer_split_checks']}.
- Inner preprocessing checks: {fairness['inner_preprocessing_checks']}.
- Every participant/model prediction assigned exactly once: yes.
- Independently reconstructed selected-edge arrays identical to strict CMPB artifacts: {fairness['all_selected_edge_arrays_identical_to_cmpb']}.

## Interpretation boundary

Full CMPB achieved AUC {cmpb['auc']:.3f}; the strongest FC-only baseline was {DISPLAY_NAMES[best_baseline_name]} (AUC {best_baseline['auc']:.3f}). These estimates support a leakage-controlled comparison under identical C-PAC splits and fold-specific FC selection. They remain single-seed internal ABIDE-I nested-CV results and should not be described as external validation. Because CMPB additionally uses frequency and demographic inputs, the comparison establishes end-to-end predictive performance rather than attributing any difference to one module alone.
"""
    (run_dir / "STRICT_SAME_PROTOCOL_BASELINE_REPORT.md").write_text(report, encoding="utf-8")
    cmpb_ci = summary["model_results"]["cmpb_full"]["participant_stratified_bootstrap"]["ci95"]["auc"]
    best_delta = paired[best_baseline_name]["auc_delta"]
    result_text = f"""# Manuscript-ready result text

## English

Under a strict same-protocol comparison on ABIDE-I C-PAC data (n={summary['participants']}), all FC-based models used the same outer folds and independently selected 4,975 edges from the full 19,900-edge universe within every training partition. Full CMPB achieved an out-of-fold AUC of {cmpb['auc']:.3f} (participant-stratified bootstrap 95% CI, {cmpb_ci[0]:.3f}-{cmpb_ci[1]:.3f}) and a balanced accuracy of {cmpb['balanced_accuracy']:.3f}. The strongest FC-only comparator was {DISPLAY_NAMES[best_baseline_name]} (AUC, {best_baseline['auc']:.3f}); the paired AUC difference was {best_delta['observed_delta_auc_cmpb_minus_baseline']:.3f} (95% CI, {best_delta['ci95'][0]:.3f} to {best_delta['ci95'][1]:.3f}; Holm-adjusted P={best_delta['holm_adjusted_p_across_five_cmpb_comparisons']:.4f}). This comparison controls split, preprocessing and feature-selection leakage, but remains a single-seed internal validation and does not isolate the contribution of any one CMPB component.

## 中文

在ABIDE-I C-PAC数据（n={summary['participants']}）的严格同协议比较中，所有基于FC的模型均使用完全相同的外层折，并在每个训练分区内从完整的19,900条连接中独立选择4,975条边。完整CMPB获得{cmpb['auc']:.3f}的折外AUC（参与者分层Bootstrap 95% CI：{cmpb_ci[0]:.3f}-{cmpb_ci[1]:.3f}）和{cmpb['balanced_accuracy']:.3f}的平衡准确率。表现最强的FC-only比较模型为{DISPLAY_NAMES[best_baseline_name]}（AUC={best_baseline['auc']:.3f}），两者配对AUC差为{best_delta['observed_delta_auc_cmpb_minus_baseline']:.3f}（95% CI：{best_delta['ci95'][0]:.3f}至{best_delta['ci95'][1]:.3f}；Holm校正P={best_delta['holm_adjusted_p_across_five_cmpb_comparisons']:.4f}）。该结果控制了数据划分、预处理和特征选择泄漏，但仍属于单随机种子的内部验证，且不能单独归因于CMPB的某一个组件。
"""
    (run_dir / "STRICT_SAME_PROTOCOL_RESULT_TEXT.md").write_text(result_text, encoding="utf-8")


def validate_cmpb_source(y: np.ndarray) -> Dict[str, Any]:
    summary_path = CMPB_RUN / "strict_nested_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing strict CMPB summary: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = {
        "summary_status": summary.get("status"),
        "participants": summary.get("participants"),
        "outer_folds": summary.get("outer_folds"),
        "outer_seed": json.loads((CMPB_RUN / "run_configuration.json").read_text(encoding="utf-8"))["arguments"]["seed"],
        "all_fold_files_present": all(
            all((CMPB_RUN / f"fold_{fold:02d}" / name).exists() for name in ("outer_train_indices.npy", "outer_test_indices.npy", "outer_test_probs.npy", "selected_full_edge_indices.npy", "fold_metrics.json"))
            for fold in range(N_OUTER)
        ),
    }
    checks["pass"] = bool(
        checks["summary_status"] == "COMPLETE_STRICT_NESTED_CV"
        and checks["participants"] == len(y)
        and checks["outer_folds"] == N_OUTER
        and checks["outer_seed"] == SEED
        and checks["all_fold_files_present"]
    )
    if not checks["pass"]:
        raise RuntimeError(f"CMPB source validation failed: {checks}")
    return checks


def protocol_manifest(args: argparse.Namespace, cmpb_audit: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "created_at": now_iso(),
        "task": "ABIDE-I ASD versus TD; C-PAC filt_noglobal; CC200",
        "outer_protocol": {"folds": N_OUTER, "splitter": "StratifiedKFold", "shuffle": True, "seed": SEED},
        "inner_protocol": {"folds": N_INNER, "splitter": "StratifiedKFold", "seed_formula": "20260712 + 10000 + outer_fold"},
        "feature_protocol": {"universe": strict.N_FULL_EDGES, "selected": strict.N_SELECTED_FC, "method": "F-test fitted on current training partition only"},
        "scaling": "StandardScaler fitted on the current training partition after selection",
        "threshold": 0.5,
        "models": {
            name: {
                "display_name": DISPLAY_NAMES[name],
                "input": "FC only" if name != "cmpb_full" else "FC + frequency + demographics",
                "candidate_grid": [asdict(candidate) for candidate in candidate_grid(name)] if name != "cmpb_full" else [asdict(candidate) for candidate in strict.candidates()],
            }
            for name in MODEL_ORDER
        },
        "deep_training": {"max_inner_epochs": args.inner_epochs, "eval_every": args.eval_every, "minimum_inner_epochs": args.minimum_inner_epochs, "inner_patience": args.inner_patience, "selection": "inner-validation AUC, BCE tie-break; final fixed epoch is median across inner folds"},
        "classical_selection": "mean inner-validation AUC, BCE tie-break; final estimator refit on complete outer-training fold",
        "classical_solvers": {
            "logistic_regression": "Deterministic full-batch LBFGS optimization of the L2-penalized logistic objective.",
            "linear_svm": "Primal linear hinge-loss SVM optimized in PyTorch; probabilities use 3-fold out-of-fold Platt calibration entirely within the current training partition.",
            "rbf_svm": "Exact training-partition RBF kernel (gamma=scale) with RKHS hinge-loss optimization in PyTorch; probabilities use 3-fold out-of-fold Platt calibration entirely within the current training partition.",
        },
        "cmpb_reference": {"path": str(CMPB_RUN), "audit": cmpb_audit},
        "external_boundary": "Internal strict nested CV on ABIDE-I, not independent external validation.",
    }


def run(args: argparse.Namespace) -> Path:
    full_fc, _x_freq, _x_demo, y, subject_ids, cache_audit = strict.load_strict_inputs(rebuild=False)
    cmpb_audit = validate_cmpb_source(y)
    run_dir = Path(args.run_dir) if args.run_dir else OUT_BASE / ("smoke" if args.mode == "smoke" else "run_20260716_full")
    ensure(run_dir)
    write_json(run_dir / "protocol_manifest.json", protocol_manifest(args, cmpb_audit))
    folds = exact_splits(y)
    requested = [0] if args.mode == "smoke" else list(range(N_OUTER))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    for fold in requested:
        outer_train, outer_test = folds[fold]
        fold_dir = run_dir / f"fold_{fold:02d}"
        ensure(fold_dir)
        print(f"[strict-baseline] fold {fold + 1}/{N_OUTER}; train={len(outer_train)}, test={len(outer_test)}, device={device}", flush=True)
        started = time.time()
        prepared_inner, outer_prepared, split_audit = prepare_outer_fold(fold, outer_train, outer_test, full_fc, y, fold_dir)
        print(f"[strict-baseline] fold {fold}: preprocessing audit PASS={split_audit['pass']}", flush=True)
        for model_name in FC_ONLY_MODELS:
            model_started = time.time()
            metrics = run_model_fold(model_name, fold, prepared_inner, outer_prepared, outer_train, outer_test, y, fold_dir, args, device)
            print(f"[strict-baseline] fold {fold} {model_name}: AUC={metrics['outer_metrics']['auc']:.4f}; {time.time() - model_started:.1f}s", flush=True)
        _cmpb_probability, cmpb_metrics = load_cmpb_fold(fold, outer_test, y)
        write_json(fold_dir / "models" / "cmpb_full" / "outer_metrics.json", cmpb_metrics)
        print(f"[strict-baseline] fold {fold} cmpb_full: AUC={cmpb_metrics['outer_metrics']['auc']:.4f}; imported audited reference", flush=True)
        print(f"[strict-baseline] fold {fold} complete in {time.time() - started:.1f}s", flush=True)
    aggregate(run_dir, subject_ids, y, folds, args, cache_audit)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict same-protocol ABIDE-I C-PAC baseline comparison")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--inner-epochs", type=int, default=300)
    parser.add_argument("--minimum-inner-epochs", type=int, default=60)
    parser.add_argument("--inner-patience", type=int, default=60)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.inner_epochs < 2 or args.eval_every < 1:
        parser.error("inner-epochs must be >=2 and eval-every >=1")
    if not 1 <= args.minimum_inner_epochs <= args.inner_epochs:
        parser.error("minimum-inner-epochs must be between 1 and inner-epochs")
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

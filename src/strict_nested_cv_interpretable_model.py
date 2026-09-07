"""Strict nested-CV, leakage-audited interpretable ABIDE-I model.

This is a new analysis lineage.  It does not overwrite the prior controlled
rerun or its explanations, because those artifacts used a globally recovered
FC feature set and globally derived anchors.

For each outer fold:
  1) Inner CV fits FC selection, all scalers, demographic imputation and
     frequency-derived anchors using *inner-training participants only*.
  2) Inner validation participants select the regularisation configuration and
     checkpoint epoch.  The outer test fold is never inspected at this stage.
  3) The chosen configuration is retrained once on the complete outer-training
     fold.  FC selection / scalers / anchors are refit on outer-training only;
     the final checkpoint is the fixed inner-CV-selected epoch, not a test-set
     selected checkpoint.
  4) Deterministic outer-test probabilities and Grad x Input FC attributions
     are saved.  Because FC selection varies by fold, attribution is aligned to
     the complete 19,900-edge CC200 lower-triangle universe before aggregation.

The script first reconstructs a label-free full FC cache from the local CC200
time series.  It verifies that the reconstructed rows reproduce the legacy
X_features_filtered.npy under its old mask, but never uses that global mask for
the strict nested-CV model.

Examples
--------
Audit / build cache only:
  python strict_nested_cv_interpretable_model.py --mode audit

Fast structural smoke test (one outer fold, two inner folds, 3 epochs):
  python strict_nested_cv_interpretable_model.py --mode smoke

Full strict nested run (10 outer, 3 inner, two pre-declared candidates):
  python strict_nested_cv_interpretable_model.py --mode full --resume
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import pickle
import random
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_selection import f_classif
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import phase4_controlled_deep_artifact_rerun as controlled


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "abide_data" / "Outputs" / "cpac" / "filt_noglobal"
ROIS_DIR = DATA_DIR / "rois_cc200"
PHENO_CSV = DATA_DIR / "Phenotypic_V1_0b_preprocessed1.csv"
Y_PATH = DATA_DIR / "y_labels.npy"
XF_PATH = DATA_DIR / "X_freq_filtered.npy"
LEGACY_FEATURES_PATH = DATA_DIR / "X_features_filtered.npy"
LEGACY_MASK_PATH = DATA_DIR / "top_15_mask.npy"
FULL_FC_CACHE = DATA_DIR / "X_s_full19900_rebuilt_strict.npy"
FULL_FC_IDS = DATA_DIR / "X_s_full19900_rebuilt_strict_subject_ids.npy"
FULL_FC_AUDIT = DATA_DIR / "X_s_full19900_rebuilt_strict_audit.json"
YEO_MAP_CSV = ROOT / "cc200_yeo7_mapping.csv"
OUT_BASE = ROOT / "strict_nested_cv_interpretability"

N_ROI = 200
N_FULL_EDGES = N_ROI * (N_ROI - 1) // 2
N_SELECTED_FC = 4975
N_FREQ = 3000
N_DEMO = 2
DEFAULT_SEED = 20260712


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    ensure(path.parent)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    ensure(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def hash_indices(indices: np.ndarray) -> str:
    arr = np.asarray(indices, dtype=np.int64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_auc(y: np.ndarray, probs: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, probs)) if len(np.unique(y)) == 2 else float("nan")


def lower_triangle_fc(time_series: np.ndarray) -> np.ndarray:
    """Compute 19,900 Pearson FC edges without np.corrcoef / BLAS matmul.

    np.corrcoef is unstable in the local runtime.  The calculation below is the
    same Pearson correlation formula and uses einsum's non-BLAS loop instead.
    """
    if time_series.ndim != 2 or time_series.shape[1] != N_ROI:
        raise ValueError(f"Expected time series [T,{N_ROI}], got {time_series.shape}")
    values = np.asarray(time_series, dtype=np.float64)
    centered = values - np.mean(values, axis=0, keepdims=True)
    norms = np.sqrt(np.sum(centered * centered, axis=0))
    covariance = np.einsum("ti,tj->ij", centered, centered, optimize=False)
    denominator = norms[:, None] * norms[None, :]
    corr = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 0)
    ii, jj = np.tril_indices(N_ROI, k=-1)
    return np.nan_to_num(corr[ii, jj], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def phenotype_records() -> Dict[int, Dict[str, Any]]:
    records: Dict[int, Dict[str, Any]] = {}
    with PHENO_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                sid = int(float(row["SUB_ID"]))
                dx = int(float(row["DX_GROUP"]))
            except (KeyError, TypeError, ValueError):
                continue
            age_raw = row.get("AGE_AT_SCAN", "")
            try:
                age = float(age_raw)
            except (TypeError, ValueError):
                age = float("nan")
            sex_raw = row.get("SEX", "")
            try:
                sex = float(int(float(sex_raw)) == 1)
            except (TypeError, ValueError):
                sex = float("nan")
            records[sid] = {"label": 1 if dx == 1 else 0, "age": age, "sex": sex}
    if not records:
        raise RuntimeError("No valid phenotypic records were read")
    return records


def subject_id_from_filename(path: str) -> Optional[int]:
    for token in re.findall(r"\d+", Path(path).name):
        if len(token) >= 4:
            return int(token)
    return None


def build_full_fc_cache(force: bool = False) -> Dict[str, Any]:
    """Reconstruct label-free full FC and prove row alignment against legacy data."""
    inputs = [Y_PATH, XF_PATH, LEGACY_FEATURES_PATH, LEGACY_MASK_PATH, PHENO_CSV]
    missing = [str(x) for x in inputs if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))
    if not force and FULL_FC_CACHE.exists() and FULL_FC_IDS.exists() and FULL_FC_AUDIT.exists():
        audit = json.loads(FULL_FC_AUDIT.read_text(encoding="utf-8"))
        if audit.get("alignment_pass") is True:
            return audit

    phenotypes = phenotype_records()
    paths = glob.glob(str(ROIS_DIR / "*.1D"))  # deliberately matches the legacy scripts' glob order
    features: List[np.ndarray] = []
    subject_ids: List[int] = []
    labels: List[int] = []
    skipped: List[Dict[str, str]] = []
    print(f"[strict-nested] Rebuilding full FC from {len(paths)} CC200 files...", flush=True)
    for number, path in enumerate(paths, start=1):
        sid = subject_id_from_filename(path)
        if sid is None or sid not in phenotypes:
            skipped.append({"file": Path(path).name, "reason": "no phenotype match"})
            continue
        try:
            ts = np.loadtxt(path)
            if ts.ndim != 2 or ts.shape[1] != N_ROI:
                skipped.append({"file": Path(path).name, "reason": f"unexpected shape {tuple(ts.shape)}"})
                continue
            features.append(lower_triangle_fc(ts))
            subject_ids.append(sid)
            labels.append(int(phenotypes[sid]["label"]))
        except Exception as exc:
            skipped.append({"file": Path(path).name, "reason": f"read/FC error: {type(exc).__name__}: {exc}"})
        if number % 100 == 0:
            print(f"[strict-nested]   reconstructed {number}/{len(paths)} files; valid={len(features)}", flush=True)

    full = np.asarray(features, dtype=np.float32)
    ids = np.asarray(subject_ids, dtype=np.int64)
    rebuilt_y = np.asarray(labels, dtype=np.int64)
    expected_y = np.load(Y_PATH, allow_pickle=False).astype(np.int64)
    x_freq = np.load(XF_PATH, allow_pickle=False)
    if full.shape != (len(expected_y), N_FULL_EDGES):
        raise RuntimeError(f"Rebuilt FC has {full.shape}; expected ({len(expected_y)}, {N_FULL_EDGES})")
    if x_freq.shape != (len(expected_y), N_FREQ):
        raise RuntimeError(f"Frequency matrix shape is {x_freq.shape}; expected ({len(expected_y)}, {N_FREQ})")
    if not np.array_equal(rebuilt_y, expected_y):
        mismatches = np.flatnonzero(rebuilt_y != expected_y)[:10].tolist()
        raise RuntimeError(f"Rebuilt subject order/labels do not align with y_labels.npy; first mismatches={mismatches}")

    # Verification only: the legacy mask is not retained in any strict-CV fit path.
    legacy_mask = np.load(LEGACY_MASK_PATH, allow_pickle=False).astype(bool)
    legacy = np.load(LEGACY_FEATURES_PATH, allow_pickle=False)
    reconstructed_legacy = full[:, legacy_mask]
    if reconstructed_legacy.shape != legacy.shape:
        raise RuntimeError(f"Legacy mask reconstruction shape {reconstructed_legacy.shape} != saved {legacy.shape}")
    max_abs_difference = float(np.max(np.abs(reconstructed_legacy.astype(np.float64) - legacy.astype(np.float64))))
    alignment_pass = bool(max_abs_difference <= 1e-6)
    if not alignment_pass:
        raise RuntimeError(f"Legacy row-order audit failed: max |rebuilt-old|={max_abs_difference}")

    np.save(FULL_FC_CACHE, full)
    np.save(FULL_FC_IDS, ids)
    audit = {
        "created_at": now_iso(),
        "cache_path": str(FULL_FC_CACHE),
        "cache_sha256": sha256_file(FULL_FC_CACHE),
        "full_fc_shape": list(full.shape),
        "subject_ids_path": str(FULL_FC_IDS),
        "subject_ids_sha256": sha256_file(FULL_FC_IDS),
        "n_cc200_files_seen": len(paths),
        "n_valid_subjects": len(ids),
        "n_skipped": len(skipped),
        "skipped_examples": skipped[:25],
        "legacy_mask_path": str(LEGACY_MASK_PATH),
        "legacy_mask_feature_count": int(legacy_mask.sum()),
        "legacy_alignment_max_abs_difference": max_abs_difference,
        "alignment_pass": alignment_pass,
        "verification_boundary": "Legacy global mask was used only to verify raw time-series row ordering. It is never used for any strict nested-CV feature selection.",
    }
    write_json(FULL_FC_AUDIT, audit)
    return audit


def load_strict_inputs(rebuild: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    audit = build_full_fc_cache(force=rebuild)
    full_fc = np.load(FULL_FC_CACHE, mmap_mode="r")
    subject_ids = np.load(FULL_FC_IDS, allow_pickle=False).astype(np.int64)
    y = np.load(Y_PATH, allow_pickle=False).astype(np.int64)
    x_freq = np.load(XF_PATH, allow_pickle=False).astype(np.float32)
    if full_fc.shape != (len(y), N_FULL_EDGES) or x_freq.shape != (len(y), N_FREQ):
        raise RuntimeError("Cached strict inputs do not match expected row count/dimensions")
    pheno = phenotype_records()
    x_demo = np.array([[pheno[int(sid)]["age"], pheno[int(sid)]["sex"]] for sid in subject_ids], dtype=np.float32)
    if x_demo.shape != (len(y), N_DEMO):
        raise RuntimeError("Raw demographic reconstruction failed")
    return full_fc, x_freq, x_demo, y, subject_ids, audit


def load_yeo7_nodes() -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
    nodes: Dict[int, List[int]] = {n: [] for n in range(1, 8)}
    with YEO_MAP_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            roi = int(row["roi_index"])
            network = int(row["yeo_network"])
            if 1 <= network <= 7:
                nodes[network].append(roi)
    arrays = {network: np.asarray(value, dtype=np.int64) for network, value in nodes.items()}
    if any(len(value) == 0 for value in arrays.values()):
        raise RuntimeError("Yeo-7 mapping has an empty network; cannot derive all anchors")
    return arrays, {"mapping_path": str(YEO_MAP_CSV), "nodes_per_network": {str(k): int(len(v)) for k, v in arrays.items()}, "background_or_unknown_nodes": N_ROI - sum(len(v) for v in arrays.values())}


def select_fc_from_training(x_fc_train: np.ndarray, y_train: np.ndarray, count: int) -> Tuple[np.ndarray, np.ndarray]:
    """Label-aware selection fitted only on supplied training participants."""
    scores, _ = f_classif(np.asarray(x_fc_train, dtype=np.float64), np.asarray(y_train, dtype=np.int64))
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.finfo(np.float64).max, neginf=-np.inf)
    if count > scores.size:
        raise ValueError(f"Requested {count} FC features from only {scores.size}")
    # Stable sort makes ties reproducible; exactly 4,975 features in every fold.
    selected = np.argsort(-scores, kind="mergesort")[:count].astype(np.int64)
    return selected, scores.astype(np.float32)


@dataclass
class DemoTransformer:
    age_impute_value: float
    sex_impute_value: float
    scaler: StandardScaler

    @classmethod
    def fit(cls, x: np.ndarray) -> "DemoTransformer":
        values = np.asarray(x, dtype=float).copy()
        age_mean = float(np.nanmean(values[:, 0])) if np.any(np.isfinite(values[:, 0])) else 17.0
        sex_mean = float(np.nanmean(values[:, 1])) if np.any(np.isfinite(values[:, 1])) else 0.0
        values[:, 0] = np.where(np.isfinite(values[:, 0]), values[:, 0], age_mean)
        values[:, 1] = np.where(np.isfinite(values[:, 1]), values[:, 1], sex_mean)
        scaler = StandardScaler().fit(values)
        return cls(age_mean, sex_mean, scaler)

    def transform(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float).copy()
        values[:, 0] = np.where(np.isfinite(values[:, 0]), values[:, 0], self.age_impute_value)
        values[:, 1] = np.where(np.isfinite(values[:, 1]), values[:, 1], self.sex_impute_value)
        return self.scaler.transform(values).astype(np.float32)


def anchors_from_train_frequency(
    x_freq_train_scaled: np.ndarray,
    yeo_nodes: Dict[int, np.ndarray],
    hidden_dim: int,
    projection_seed: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Derive 7 anchors solely from the supplied training-fold frequency data."""
    x_nodes = torch.as_tensor(x_freq_train_scaled, dtype=torch.float32).reshape(-1, N_ROI, 15)
    mean_profiles = torch.stack([x_nodes[:, yeo_nodes[network], :].mean(dim=(0, 1)) for network in range(1, 8)])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(projection_seed)
    projection = torch.randn((15, hidden_dim), generator=generator, dtype=torch.float32) / math.sqrt(15.0)
    anchors = torch.matmul(mean_profiles, projection).detach().cpu()
    return anchors, mean_profiles.detach().cpu().numpy()


@dataclass
class Candidate:
    name: str
    dropout_sc: float
    dropout_cls: float
    weight_decay: float
    anchor_weight: float


def candidates() -> List[Candidate]:
    # Pre-declared before outer-test evaluation; inner CV selects only between these.
    return [
        Candidate("base", 0.25, 0.25, 0.005, 0.30),
        Candidate("regularized", 0.35, 0.35, 0.010, 0.30),
    ]


@dataclass
class PreparedData:
    selected_edges: np.ndarray
    f_scores: np.ndarray
    scaler_xs: StandardScaler
    scaler_xf: StandardScaler
    demo_transformer: DemoTransformer
    anchors: torch.Tensor
    anchor_profiles: np.ndarray
    x_s_train: np.ndarray
    x_f_train: np.ndarray
    x_d_train: np.ndarray
    x_s_eval: np.ndarray
    x_f_eval: np.ndarray
    x_d_eval: np.ndarray


def prepare_train_eval(
    full_fc: np.ndarray,
    x_freq: np.ndarray,
    x_demo: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    yeo_nodes: Dict[int, np.ndarray],
    hidden_dim: int,
    anchor_projection_seed: int,
) -> PreparedData:
    selected, f_scores = select_fc_from_training(full_fc[train_idx], y[train_idx], N_SELECTED_FC)
    scaler_xs = StandardScaler().fit(np.asarray(full_fc[train_idx][:, selected], dtype=np.float32))
    scaler_xf = StandardScaler().fit(x_freq[train_idx])
    demo_transformer = DemoTransformer.fit(x_demo[train_idx])
    x_s_train = scaler_xs.transform(np.asarray(full_fc[train_idx][:, selected], dtype=np.float32)).astype(np.float32)
    x_s_eval = scaler_xs.transform(np.asarray(full_fc[eval_idx][:, selected], dtype=np.float32)).astype(np.float32)
    x_f_train = scaler_xf.transform(x_freq[train_idx]).astype(np.float32)
    x_f_eval = scaler_xf.transform(x_freq[eval_idx]).astype(np.float32)
    x_d_train = demo_transformer.transform(x_demo[train_idx])
    x_d_eval = demo_transformer.transform(x_demo[eval_idx])
    anchors, profiles = anchors_from_train_frequency(x_f_train, yeo_nodes, hidden_dim, anchor_projection_seed)
    return PreparedData(selected, f_scores, scaler_xs, scaler_xf, demo_transformer, anchors, profiles, x_s_train, x_f_train, x_d_train, x_s_eval, x_f_eval, x_d_eval)


def model_config(candidate: Candidate) -> Dict[str, Any]:
    return {
        "hidden_dim": 80,
        "align_dim": 160,
        "dropout_sc": candidate.dropout_sc,
        "dropout_cls": candidate.dropout_cls,
        "num_res_blocks": 4,
        "lr": 0.0008,
        "weight_decay": candidate.weight_decay,
        "batch_size": 16,
        "kl_weight": 0.002,
        "anchor_weight": candidate.anchor_weight,
        "focal_gamma": 1.0,
        "mixup_alpha": 0.10,
        "noise_std": 0.01,
        "grad_clip": 1.0,
        "lr_warmup": 60,
        "tau_warmup": 100,
        "tau_high": 1.0,
        "tau_low": 0.15,
    }


def make_model(space_dim: int, anchors: torch.Tensor, candidate: Candidate, device: torch.device) -> torch.nn.Module:
    cfg = model_config(candidate)
    model = controlled.BrainInnovationSystem(
        space_dim=space_dim,
        prior_anchors=anchors,
        hidden_dim=cfg["hidden_dim"],
        align_dim=cfg["align_dim"],
        dropout_sc=cfg["dropout_sc"],
        dropout_cls=cfg["dropout_cls"],
        num_res_blocks=cfg["num_res_blocks"],
    ).to(device)
    model.set_eval_ocread_mode("mu")
    return model


def predict_probabilities(model: torch.nn.Module, x_s: np.ndarray, x_f: np.ndarray, x_d: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    model.set_eval_ocread_mode("mu")
    batches: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_s), 64):
            stop = min(start + 64, len(x_s))
            logits, *_ = model(
                torch.from_numpy(x_s[start:stop]).to(device),
                torch.from_numpy(x_f[start:stop]).to(device),
                torch.from_numpy(x_d[start:stop]).to(device),
                tau=0.15, training=False, return_intermediates=False,
            )
            batches.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(batches).reshape(-1)


def evaluate(model: torch.nn.Module, x_s: np.ndarray, x_f: np.ndarray, x_d: np.ndarray, y: np.ndarray, device: torch.device) -> Dict[str, float]:
    probs = predict_probabilities(model, x_s, x_f, x_d, device)
    tensor_y = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(device)
    with torch.no_grad():
        logits_parts: List[torch.Tensor] = []
        model.eval(); model.set_eval_ocread_mode("mu")
        for start in range(0, len(x_s), 64):
            stop = min(start + 64, len(x_s))
            logits, *_ = model(torch.from_numpy(x_s[start:stop]).to(device), torch.from_numpy(x_f[start:stop]).to(device), torch.from_numpy(x_d[start:stop]).to(device), tau=0.15, training=False, return_intermediates=False)
            logits_parts.append(logits)
        logits_all = torch.cat(logits_parts)
        loss = float(F.binary_cross_entropy_with_logits(logits_all, tensor_y).item())
    return {"auc": safe_auc(y, probs), "accuracy": float(accuracy_score(y, (probs >= 0.5).astype(int))), "bce": loss}


def train_model(
    x_s_train: np.ndarray,
    x_f_train: np.ndarray,
    x_d_train: np.ndarray,
    y_train: np.ndarray,
    candidate: Candidate,
    anchors: torch.Tensor,
    device: torch.device,
    seed: int,
    epochs: int,
    monitor: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    eval_every: int = 5,
    early_stopping_patience: Optional[int] = None,
    minimum_epochs: int = 0,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Train model; if monitor exists, choose checkpoint only using that monitor."""
    set_seed(seed)
    cfg = model_config(candidate)
    model = make_model(x_s_train.shape[1], anchors, candidate, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - cfg["lr_warmup"], 1))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_s_train), torch.from_numpy(x_f_train), torch.from_numpy(x_d_train), torch.from_numpy(np.asarray(y_train, dtype=np.float32))),
        batch_size=cfg["batch_size"], shuffle=True, drop_last=False,
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    best_auc, best_bce = -float("inf"), float("inf")
    last_improved_epoch = 0
    stopped_epoch = epochs
    history: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        if epoch <= cfg["lr_warmup"]:
            for group in optimizer.param_groups:
                group["lr"] = cfg["lr"] * epoch / cfg["lr_warmup"]
        tau = cfg["tau_high"] if epoch <= cfg["tau_warmup"] else cfg["tau_low"]
        model.train()
        loss_total, n_batches = 0.0, 0
        for b_s, b_f, b_d, b_y in loader:
            b_s, b_f, b_d, b_y = b_s.to(device), b_f.to(device), b_d.to(device), b_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if cfg["mixup_alpha"] > 0 and epoch > cfg["lr_warmup"]:
                lam = float(np.random.beta(cfg["mixup_alpha"], cfg["mixup_alpha"]))
                permutation = torch.randperm(len(b_y), device=device)
                in_s = lam * b_s + (1.0 - lam) * b_s[permutation]
                in_f = lam * b_f + (1.0 - lam) * b_f[permutation]
                in_d = lam * b_d + (1.0 - lam) * b_d[permutation]
                y_a, y_b = b_y, b_y[permutation]
            else:
                lam, in_s, in_f, in_d, y_a, y_b = 1.0, b_s, b_f, b_d, b_y, b_y
            logits, _, mu, logvar = model(in_s, in_f, in_d, tau=tau, training=True, noise_std=cfg["noise_std"])
            bce_a = F.binary_cross_entropy_with_logits(logits, y_a, reduction="none")
            bce_b = F.binary_cross_entropy_with_logits(logits, y_b, reduction="none")
            if cfg["focal_gamma"] > 0:
                weight_a, weight_b = (1 - torch.exp(-bce_a)).pow(cfg["focal_gamma"]), (1 - torch.exp(-bce_b)).pow(cfg["focal_gamma"])
                loss_cls = lam * (weight_a * bce_a).mean() + (1 - lam) * (weight_b * bce_b).mean()
            else:
                loss_cls = lam * bce_a.mean() + (1 - lam) * bce_b.mean()
            loss_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * cfg["kl_weight"]
            loss_anchor = F.mse_loss(mu, model.ocread.initial_anchors) * cfg["anchor_weight"]
            loss = loss_cls + loss_kl + loss_anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            loss_total += float(loss.item()); n_batches += 1
        if epoch > cfg["lr_warmup"]:
            scheduler.step()
        train_loss = loss_total / max(n_batches, 1)
        if monitor is not None and (epoch % eval_every == 0 or epoch == epochs):
            metric = evaluate(model, monitor[0], monitor[1], monitor[2], monitor[3], device)
            history.append({"epoch": epoch, "train_loss": train_loss, **metric})
            improved = metric["auc"] > best_auc + 1e-12 or (abs(metric["auc"] - best_auc) <= 1e-12 and metric["bce"] < best_bce)
            if improved:
                best_auc, best_bce, best_epoch = metric["auc"], metric["bce"], epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                last_improved_epoch = epoch
            if (
                early_stopping_patience is not None
                and epoch >= minimum_epochs
                and last_improved_epoch > 0
                and epoch - last_improved_epoch >= early_stopping_patience
            ):
                stopped_epoch = epoch
                history[-1]["early_stop_triggered"] = True
                history[-1]["early_stopping_patience"] = int(early_stopping_patience)
                history[-1]["minimum_epochs"] = int(minimum_epochs)
                break
    if monitor is None:
        best_epoch = epochs
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        history.append({"epoch": epochs, "checkpoint_policy": "fixed_inner_cv_selected_epoch"})
    if best_state is None:
        raise RuntimeError("No inner-validation checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval(); model.set_eval_ocread_mode("mu")
    return model, {"best_epoch": int(best_epoch), "best_validation_auc": float(best_auc) if monitor is not None else None, "best_validation_bce": float(best_bce) if monitor is not None else None, "history": history, "checkpoint_policy": "inner_validation_auc_then_bce" if monitor is not None else "fixed_epoch_selected_by_inner_cv", "maximum_epochs": int(epochs), "stopped_epoch": int(stopped_epoch), "early_stopping_patience": None if early_stopping_patience is None else int(early_stopping_patience), "minimum_epochs": int(minimum_epochs)}


def grad_x_input(model: torch.nn.Module, x_s: np.ndarray, x_f: np.ndarray, x_d: np.ndarray, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval(); model.set_eval_ocread_mode("mu")
    chunks: List[np.ndarray] = []
    for start in range(0, len(x_s), 32):
        stop = min(start + 32, len(x_s))
        xs = torch.from_numpy(x_s[start:stop]).to(device).detach().requires_grad_(True)
        xf = torch.from_numpy(x_f[start:stop]).to(device).detach().requires_grad_(True)
        xd = torch.from_numpy(x_d[start:stop]).to(device).detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits, *_ = model(xs, xf, xd, tau=0.15, training=False, return_intermediates=False)
        logits.sum().backward()
        chunks.append((xs.grad * xs).detach().cpu().numpy())
    values = np.concatenate(chunks, axis=0)
    return np.mean(np.abs(values), axis=0).astype(np.float32), np.mean(values, axis=0).astype(np.float32)


def save_prepared_artifacts(folder: Path, prepared: PreparedData, train_idx: np.ndarray, eval_idx: np.ndarray, kind: str) -> Dict[str, Any]:
    ensure(folder)
    np.save(folder / "selected_full_edge_indices.npy", prepared.selected_edges)
    np.save(folder / "f_scores_full19900.npy", prepared.f_scores)
    np.save(folder / "prior_anchors_train_only.npy", prepared.anchors.detach().cpu().numpy())
    np.save(folder / "anchor_mean_profiles_train_only.npy", prepared.anchor_profiles)
    np.save(folder / "fit_indices.npy", train_idx)
    np.save(folder / "evaluation_indices.npy", eval_idx)
    with (folder / "scaler_xs.pkl").open("wb") as f: pickle.dump(prepared.scaler_xs, f)
    with (folder / "scaler_xf.pkl").open("wb") as f: pickle.dump(prepared.scaler_xf, f)
    with (folder / "demo_transformer.pkl").open("wb") as f: pickle.dump(prepared.demo_transformer, f)
    provenance = {
        "kind": kind,
        "fit_indices_sha256": hash_indices(train_idx),
        "evaluation_indices_sha256": hash_indices(eval_idx),
        "n_fit": int(len(train_idx)), "n_evaluation": int(len(eval_idx)),
        "feature_selection_fit_scope": "fit_indices only",
        "scaler_fit_scope": "fit_indices only",
        "anchor_fit_scope": "fit_indices only, after frequency scaler fit on fit_indices",
        "n_selected_fc": int(len(prepared.selected_edges)),
        "selected_edge_indices_sha256": hash_indices(prepared.selected_edges),
    }
    write_json(folder / "transform_provenance.json", provenance)
    return provenance


def choose_candidate(inner_rows: List[Dict[str, Any]]) -> Tuple[Candidate, int, Dict[str, Any]]:
    aggregated: Dict[str, Dict[str, Any]] = {}
    by_name = {candidate.name: candidate for candidate in candidates()}
    for candidate in candidates():
        rows = [row for row in inner_rows if row["candidate"] == candidate.name]
        aucs = np.array([row["best_validation_auc"] for row in rows], dtype=float)
        bces = np.array([row["best_validation_bce"] for row in rows], dtype=float)
        epochs = np.array([row["best_epoch"] for row in rows], dtype=int)
        aggregated[candidate.name] = {"mean_inner_auc": float(np.mean(aucs)), "mean_inner_bce": float(np.mean(bces)), "fold_epochs": epochs.tolist(), "median_epoch": int(np.median(epochs))}
    winner_name = sorted(aggregated, key=lambda name: (-aggregated[name]["mean_inner_auc"], aggregated[name]["mean_inner_bce"], name))[0]
    winner = aggregated[winner_name]
    return by_name[winner_name], int(winner["median_epoch"]), {"all_candidates": aggregated, "selected_candidate": winner_name, "final_fixed_epoch": int(winner["median_epoch"]), "selection_rule": "maximum mean inner-validation AUC; ties resolved by lower mean inner-validation BCE; final epoch = median selected inner checkpoint epoch"}


def serialize_model_checkpoint(path: Path, model: torch.nn.Module, candidate: Candidate, fixed_epoch: int, outer_fold: int, outer_train: np.ndarray, outer_test: np.ndarray, selection: Dict[str, Any]) -> None:
    cfg = model_config(candidate)
    torch.save({
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_class_name": "BrainInnovationSystem",
        "model_config": {"space_dim": N_SELECTED_FC, "hidden_dim": cfg["hidden_dim"], "align_dim": cfg["align_dim"], "dropout_sc": cfg["dropout_sc"], "dropout_cls": cfg["dropout_cls"], "num_res_blocks": cfg["num_res_blocks"]},
        "candidate": asdict(candidate),
        "outer_fold": outer_fold,
        "fixed_epoch": fixed_epoch,
        "checkpoint_selection": "inner-CV validation selected configuration and median epoch; final model was retrained on outer training only; outer test not inspected.",
        "outer_train_indices_sha256": hash_indices(outer_train),
        "outer_test_indices_sha256": hash_indices(outer_test),
        "inner_selection": selection,
    }, path)


def outer_fold_run(
    outer_fold: int,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    full_fc: np.ndarray,
    x_freq: np.ndarray,
    x_demo: np.ndarray,
    y: np.ndarray,
    yeo_nodes: Dict[int, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
) -> Dict[str, Any]:
    fold_dir = run_dir / f"fold_{outer_fold:02d}"
    ensure(fold_dir)
    inner_rows: List[Dict[str, Any]] = []
    inner_splits = StratifiedKFold(n_splits=args.inner_splits, shuffle=True, random_state=args.seed + 10_000 + outer_fold)
    print(f"[strict-nested] outer fold {outer_fold}: inner CV ({args.inner_splits} folds × {len(candidates())} candidates)", flush=True)
    for inner_fold, (rel_fit, rel_val) in enumerate(inner_splits.split(outer_train, y[outer_train])):
        inner_fit, inner_val = outer_train[rel_fit], outer_train[rel_val]
        for cand_idx, candidate in enumerate(candidates()):
            trial_dir = fold_dir / "inner_cv" / f"inner_{inner_fold:02d}" / candidate.name
            metrics_file = trial_dir / "inner_metrics.json"
            if args.resume and metrics_file.exists():
                row = json.loads(metrics_file.read_text(encoding="utf-8"))
                inner_rows.append(row)
                continue
            prepared = prepare_train_eval(full_fc, x_freq, x_demo, y, inner_fit, inner_val, yeo_nodes, 80, args.anchor_projection_seed)
            provenance = save_prepared_artifacts(trial_dir, prepared, inner_fit, inner_val, "inner_cv")
            trial_seed = args.seed + outer_fold * 1000 + inner_fold * 100 + cand_idx
            model, train_meta = train_model(prepared.x_s_train, prepared.x_f_train, prepared.x_d_train, y[inner_fit], candidate, prepared.anchors, device, trial_seed, args.inner_epochs, (prepared.x_s_eval, prepared.x_f_eval, prepared.x_d_eval, y[inner_val]), args.eval_every)
            row = {"outer_fold": outer_fold, "inner_fold": inner_fold, "candidate": candidate.name, "best_epoch": train_meta["best_epoch"], "best_validation_auc": train_meta["best_validation_auc"], "best_validation_bce": train_meta["best_validation_bce"], "n_inner_fit": int(len(inner_fit)), "n_inner_validation": int(len(inner_val)), "fit_indices_sha256": hash_indices(inner_fit), "validation_indices_sha256": hash_indices(inner_val), "transform_provenance": provenance, "training_history": train_meta["history"]}
            write_json(metrics_file, row)
            inner_rows.append(row)
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    selected_candidate, final_epoch, selection = choose_candidate(inner_rows)
    write_json(fold_dir / "inner_cv_selection.json", selection)
    print(f"[strict-nested] outer fold {outer_fold}: selected {selected_candidate.name}, final epoch={final_epoch}", flush=True)

    # Final outer model: all train-only transforms are refit on outer_train.
    prepared = prepare_train_eval(full_fc, x_freq, x_demo, y, outer_train, outer_test, yeo_nodes, 80, args.anchor_projection_seed)
    outer_provenance = save_prepared_artifacts(fold_dir, prepared, outer_train, outer_test, "outer_final")
    final_seed = args.seed + outer_fold * 1000 + 777
    model, final_train_meta = train_model(prepared.x_s_train, prepared.x_f_train, prepared.x_d_train, y[outer_train], selected_candidate, prepared.anchors, device, final_seed, final_epoch, monitor=None, eval_every=args.eval_every)
    probs = predict_probabilities(model, prepared.x_s_eval, prepared.x_f_eval, prepared.x_d_eval, device)
    attribution_abs, attribution_signed = grad_x_input(model, prepared.x_s_eval, prepared.x_f_eval, prepared.x_d_eval, device)
    full_abs = np.full(N_FULL_EDGES, np.nan, dtype=np.float32); full_signed = np.full(N_FULL_EDGES, np.nan, dtype=np.float32)
    full_abs[prepared.selected_edges] = attribution_abs; full_signed[prepared.selected_edges] = attribution_signed
    np.save(fold_dir / "outer_test_gradxinput_abs_full19900.npy", full_abs)
    np.save(fold_dir / "outer_test_gradxinput_signed_full19900.npy", full_signed)
    np.save(fold_dir / "outer_test_probs.npy", probs.astype(np.float32))
    np.save(fold_dir / "outer_test_labels.npy", y[outer_test].astype(np.int64))
    np.save(fold_dir / "outer_test_subject_indices.npy", outer_test.astype(np.int64))
    np.save(fold_dir / "outer_train_indices.npy", outer_train.astype(np.int64))
    np.save(fold_dir / "outer_test_indices.npy", outer_test.astype(np.int64))
    serialize_model_checkpoint(fold_dir / "model_checkpoint.pt", model, selected_candidate, final_epoch, outer_fold, outer_train, outer_test, selection)
    metrics = {"outer_fold": outer_fold, "outer_test_auc": safe_auc(y[outer_test], probs), "outer_test_accuracy_threshold_0_5": float(accuracy_score(y[outer_test], (probs >= 0.5).astype(int))), "n_outer_train": int(len(outer_train)), "n_outer_test": int(len(outer_test)), "selected_candidate": selected_candidate.name, "final_fixed_epoch": final_epoch, "final_train_meta": final_train_meta, "outer_transform_provenance": outer_provenance, "outer_test_usage": "Only deterministic final evaluation and attribution; no selection, fitting, scaling, anchor derivation or checkpoint choice.", "status": "COMPLETE"}
    write_json(fold_dir / "fold_metrics.json", metrics)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return metrics


def fold_complete(fold_dir: Path) -> bool:
    required = ["model_checkpoint.pt", "fold_metrics.json", "selected_full_edge_indices.npy", "prior_anchors_train_only.npy", "outer_test_gradxinput_abs_full19900.npy", "outer_test_probs.npy", "outer_test_subject_indices.npy", "transform_provenance.json"]
    return all((fold_dir / name).exists() for name in required)


def aggregate_run(run_dir: Path, subject_ids: np.ndarray, y: np.ndarray, n_outer: int, full_cache_audit: Dict[str, Any], args: argparse.Namespace, yeo_audit: Dict[str, Any]) -> None:
    oof = np.full(len(y), np.nan, dtype=float)
    select = np.zeros((n_outer, N_FULL_EDGES), dtype=np.uint8)
    abs_maps = np.full((n_outer, N_FULL_EDGES), np.nan, dtype=np.float32)
    signed_maps = np.full((n_outer, N_FULL_EDGES), np.nan, dtype=np.float32)
    fold_rows: List[Dict[str, Any]] = []
    for fold in range(n_outer):
        folder = run_dir / f"fold_{fold:02d}"
        if not fold_complete(folder):
            continue
        metrics = json.loads((folder / "fold_metrics.json").read_text(encoding="utf-8"))
        fold_rows.append(metrics)
        idx = np.load(folder / "outer_test_subject_indices.npy", allow_pickle=False).astype(int)
        oof[idx] = np.load(folder / "outer_test_probs.npy", allow_pickle=False)
        selected = np.load(folder / "selected_full_edge_indices.npy", allow_pickle=False).astype(int)
        select[fold, selected] = 1
        abs_maps[fold] = np.load(folder / "outer_test_gradxinput_abs_full19900.npy", allow_pickle=False)
        signed_maps[fold] = np.load(folder / "outer_test_gradxinput_signed_full19900.npy", allow_pickle=False)
    if len(fold_rows) != n_outer:
        write_json(run_dir / "partial_aggregate.json", {"completed_outer_folds": len(fold_rows), "expected_outer_folds": n_outer, "generated_at": now_iso()})
        return
    if np.any(~np.isfinite(oof)):
        raise RuntimeError("Outer OOF probabilities are incomplete")
    np.save(run_dir / "outer_fold_selection_masks_full19900.npy", select)
    np.save(run_dir / "outer_fold_gradxinput_abs_full19900.npy", abs_maps)
    np.save(run_dir / "outer_fold_gradxinput_signed_full19900.npy", signed_maps)
    counts = select.sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean_abs = np.nanmean(abs_maps, axis=0)
        mean_signed = np.nanmean(signed_maps, axis=0)
    ii, jj = np.tril_indices(N_ROI, k=-1)
    rows: List[Dict[str, Any]] = []
    for edge in range(N_FULL_EDGES):
        rows.append({"full_edge_index": edge, "roi_i_index": int(ii[edge]), "roi_j_index": int(jj[edge]), "roi_i_formal_name": f"CC200_{ii[edge]+1:03d}", "roi_j_formal_name": f"CC200_{jj[edge]+1:03d}", "outer_selection_count": int(counts[edge]), "outer_selection_frequency": round(float(counts[edge] / n_outer), 6), "mean_abs_gradxinput_when_selected": "" if not np.isfinite(mean_abs[edge]) else round(float(mean_abs[edge]), 10), "mean_signed_gradxinput_when_selected": "" if not np.isfinite(mean_signed[edge]) else round(float(mean_signed[edge]), 10), "interpretation_scope": "Outer-test Grad x Input, averaged only over folds where this edge was selected using that fold's training data. Selection frequency must be reported jointly with magnitude."})
    rows.sort(key=lambda r: (-int(r["outer_selection_count"]), -(float(r["mean_abs_gradxinput_when_selected"]) if r["mean_abs_gradxinput_when_selected"] != "" else -1.0)))
    fields = list(rows[0].keys())
    write_csv(run_dir / "strict_nested_gradxinput_fc_full19900.csv", rows, fields)
    write_csv(run_dir / "strict_nested_gradxinput_fc_top200.csv", rows[:200], fields)
    oof_rows = [{"subject_row_index": i, "subject_id": int(subject_ids[i]), "label_asd_1_td_0": int(y[i]), "outer_oof_probability_asd": round(float(oof[i]), 10), "prediction_threshold_0_5": int(oof[i] >= 0.5)} for i in range(len(y))]
    write_csv(run_dir / "strict_nested_outer_oof_predictions.csv", oof_rows, list(oof_rows[0].keys()))
    summary = {"status": "COMPLETE_STRICT_NESTED_CV", "generated_at": now_iso(), "participants": int(len(y)), "outer_folds": n_outer, "inner_folds": args.inner_splits, "predeclared_candidates": [asdict(c) for c in candidates()], "inner_max_epochs": args.inner_epochs, "outer_oof_auc": safe_auc(y, oof), "outer_oof_accuracy_threshold_0_5": float(accuracy_score(y, (oof >= 0.5).astype(int))), "mean_outer_fold_auc": float(np.mean([x["outer_test_auc"] for x in fold_rows])), "mean_outer_fold_accuracy": float(np.mean([x["outer_test_accuracy_threshold_0_5"] for x in fold_rows])), "full_fc_cache_audit": full_cache_audit, "anchor_mapping_audit": yeo_audit, "strictness": {"feature_selection": "fitted separately inside each inner training fold and refit on each outer training fold", "anchors": "derived from inner-training or outer-training frequency data only", "checkpoint": "inner-validation selected epoch/configuration; final outer model retrained for that fixed epoch with no outer-test selection", "scalers_and_demographic_imputation": "fit within the relevant training fold only", "attributions": "computed on untouched outer-test participants; FC alignment performed after fold-specific feature selection"}, "claim_boundary": "Nested cross-validation controls procedural leakage but remains internal ABIDE-I validation, not an external replication or clinical biomarker study."}
    write_json(run_dir / "strict_nested_summary.json", summary)
    report = f"""# Strict nested-CV interpretable model\n\nGenerated: {summary['generated_at']}\n\n- Outer CV: {n_outer}-fold stratified; inner CV: {args.inner_splits}-fold stratified.\n- FC feature selection: exactly {N_SELECTED_FC} of {N_FULL_EDGES} edges, fitted only in the appropriate training partition.\n- Anchors: Yeo-7 overlap-derived network means from frequency features after training-partition scaling only.\n- Checkpoint: candidate/configuration and epoch selected by inner validation. The final outer model uses only outer-training data and is trained for the inner-CV-selected fixed epoch.\n- Outer OOF AUC: {summary['outer_oof_auc']:.4f}; accuracy at fixed 0.5 threshold: {summary['outer_oof_accuracy_threshold_0_5']:.4f}.\n\n## Interpretation outputs\n\n`strict_nested_gradxinput_fc_full19900.csv` reports each CC200 lower-triangle edge. Magnitude is averaged only across outer folds in which the edge was selected; always interpret it together with `outer_selection_frequency`.\n\n## Scope\n\nThis resolves the identified global feature-selection/global-anchor/checkpoint-selection leakage in the previous interpretability lineage. It does not itself provide external replication, causal inference, or a clinical biomarker.\n"""
    (run_dir / "STRICT_NESTED_CV_REPORT.md").write_text(report, encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    if args.mode == "audit":
        audit = build_full_fc_cache(force=args.rebuild_full_fc)
        print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
        return OUT_BASE
    full_fc, x_freq, x_demo, y, subject_ids, cache_audit = load_strict_inputs(rebuild=args.rebuild_full_fc)
    yeo_nodes, yeo_audit = load_yeo7_nodes()
    n_outer = 2 if args.mode == "smoke" else args.outer_splits
    outer_fold_list = [0] if args.mode == "smoke" else list(range(n_outer))
    run_dir = Path(args.run_dir) if args.run_dir else OUT_BASE / ("smoke" if args.mode == "smoke" else "run_20260712_strict_nested_cv")
    ensure(run_dir)
    write_json(run_dir / "run_configuration.json", {"created_at": now_iso(), "mode": args.mode, "arguments": vars(args), "full_fc_shape": list(full_fc.shape), "frequency_shape": list(x_freq.shape), "raw_demographic_shape": list(x_demo.shape), "n_selected_fc": N_SELECTED_FC, "device": "cuda" if torch.cuda.is_available() else "cpu", "cache_audit": cache_audit, "yeo_anchor_mapping": yeo_audit, "candidate_grid": [asdict(c) for c in candidates()]})
    write_json(run_dir / "strictness_protocol.json", {
        "protocol": "STRICT_NESTED_CV_INTERPRETABILITY",
        "outer_test_isolation": "Outer-test participants are excluded from all feature selection, frequency-anchor derivation, demographic imputation, scaling, candidate choice, checkpoint epoch choice and model fitting.",
        "inner_trial_fit_scope": "For every inner fold and candidate, FC F-scores/selectors, FC/frequency scalers, demographic imputer/scaler and frequency-derived anchors are fit on inner-training participants only.",
        "checkpoint_scope": "An inner trial checkpoint is selected by deterministic inner-validation AUC, with inner-validation BCE only as a tie breaker. No outer-test score is used.",
        "outer_final_scope": "Candidate and fixed epoch are selected from inner CV. The outer final selector/scalers/anchors are then refit on the complete outer-training fold, and the model is trained for that fixed epoch with no outer-test checkpoint selection.",
        "attribution_scope": "Grad x Input is evaluated only on each untouched outer-test fold after final training. Fold-specific selected features are mapped to the full 19,900-edge universe before aggregation.",
        "legacy_data_boundary": "The legacy global FC mask is used once only for row-order reconstruction verification; strict-CV fitting always derives a new selector from the relevant training partition.",
        "verification_artifacts": ["inner_cv/*/*/transform_provenance.json", "fold_*/transform_provenance.json", "fold_*/inner_cv_selection.json", "fold_*/model_checkpoint.pt", "fold_*/outer_test_gradxinput_abs_full19900.npy"],
    })
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=args.seed)
    for fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros(len(y)), y)):
        if fold not in outer_fold_list:
            continue
        folder = run_dir / f"fold_{fold:02d}"
        if args.resume and fold_complete(folder):
            print(f"[strict-nested] fold {fold} complete; resume skips it", flush=True)
            continue
        print(f"[strict-nested] outer fold {fold + 1}/{n_outer}: train={len(train_idx)}, test={len(test_idx)}, device={device}", flush=True)
        started = time.time()
        metrics = outer_fold_run(fold, train_idx.astype(np.int64), test_idx.astype(np.int64), full_fc, x_freq, x_demo, y, yeo_nodes, args, device, run_dir)
        print(f"[strict-nested] fold {fold} finished in {time.time()-started:.1f}s; AUC={metrics['outer_test_auc']:.4f}", flush=True)
        aggregate_run(run_dir, subject_ids, y, n_outer, cache_audit, args, yeo_audit)
    aggregate_run(run_dir, subject_ids, y, n_outer, cache_audit, args, yeo_audit)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict nested-CV interpretable ABIDE model")
    parser.add_argument("--mode", choices=["audit", "smoke", "full"], default="audit")
    parser.add_argument("--outer-splits", type=int, default=10)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--inner-epochs", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--anchor-projection-seed", type=int, default=20250225)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-full-fc", action="store_true")
    args = parser.parse_args()
    if args.inner_splits < 2 or args.outer_splits < 2:
        parser.error("outer/inner splits must be >=2")
    if args.inner_epochs < 2 or args.eval_every < 1:
        parser.error("inner epochs must be >=2 and eval interval >=1")
    if args.mode == "smoke":
        args.inner_splits, args.inner_epochs, args.eval_every = 2, 3, 1
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception:
        import traceback
        print(traceback.format_exc(), flush=True)
        raise

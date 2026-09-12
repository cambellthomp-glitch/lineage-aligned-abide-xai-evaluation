"""Leakage-audited CMPB training on ABIDE I and locked ABIDE II validation.

This adapter reuses the latest strict CMPB architecture and nested-selection
implementation while replacing its legacy C-PAC inputs with the harmonized
fMRIPrep 25.2.4 feature branch. ABIDE II is never used for feature selection,
scaling, anchor construction, candidate/epoch selection, or model fitting.

The stored FC feature is Fisher-z Pearson correlation. The strict CMPB lineage
used Pearson edges, so this script creates a versioned, reversible tanh cache.
The spectral vector follows the legacy frequency-major layout (15 x 200). The
base strict script historically reshaped that vector as 200 x 15 directly;
this adapter corrects the interpretation to reshape(15, 200).T before Yeo-7
anchor construction. Historical result folders are not modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
import sys
import time
from argparse import Namespace
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import strict_nested_cv_interpretable_model as strict  # noqa: E402


FEATURE_ROOT = PROJECT / "features"
CMPB_INPUT_ROOT = FEATURE_ROOT / "cmpb_strict_v2"
RESULT_ROOT = PROJECT / "results" / "cmpb_strict_v2"
SITE_CONFIG = PROJECT / "config" / "site_canonicalization.json"
YEO_MAP = WORKSPACE / "cc200_yeo7_mapping.csv"
SEED = 20260712
ANCHOR_PROJECTION_SEED = 20250225
FREQUENCY_LAYOUT = "frequency_major_15x200"
ANCHOR_LAYOUT_FIX = "reshape(n,15,200).transpose(0,2,1)"
THRESHOLD = 0.5
N_BOOTSTRAP = 2000


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    ensure(path.parent)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_strings(values: Iterable[str]) -> str:
    joined = "\n".join(map(str, values)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def numeric_abide_id(value: Any) -> int:
    matches = re.findall(r"[0-9]{7}", str(value))
    if not matches:
        raise ValueError(f"No seven-digit ABIDE identifier in {value!r}")
    return int(matches[-1])


def corrected_anchors_from_frequency_major(
    x_freq_train_scaled: np.ndarray,
    yeo_nodes: dict[int, np.ndarray],
    hidden_dim: int,
    projection_seed: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Construct Yeo-7 anchors from the documented 15-frequency x 200-ROI layout."""
    tensor = torch.as_tensor(x_freq_train_scaled, dtype=torch.float32)
    x_nodes = tensor.reshape(-1, 15, strict.N_ROI).transpose(1, 2)
    mean_profiles = torch.stack(
        [x_nodes[:, yeo_nodes[network], :].mean(dim=(0, 1)) for network in range(1, 8)]
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(projection_seed)
    projection = (
        torch.randn((15, hidden_dim), generator=generator, dtype=torch.float32)
        / math.sqrt(15.0)
    )
    anchors = torch.matmul(mean_profiles, projection).detach().cpu()
    return anchors, mean_profiles.detach().cpu().numpy()


# Isolate the layout correction to this process. The historical strict script
# and its already-generated result folders remain untouched on disk.
strict.anchors_from_train_frequency = corrected_anchors_from_frequency_major


@dataclass
class Cohort:
    name: str
    fc: np.ndarray
    freq: np.ndarray
    demo: np.ndarray
    subject_ids: np.ndarray
    site_ids: np.ndarray
    y_path: Path
    y: np.ndarray | None


def ensure_pearson_cache(cohort_dir: Path, cohort_name: str) -> tuple[Path, dict[str, Any]]:
    source = cohort_dir / "X_fc_raw.npy"
    target_dir = CMPB_INPUT_ROOT / cohort_name
    target = target_dir / "X_fc_full19900_pearson.npy"
    ensure(target_dir)
    z = np.load(source, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(z), strict.N_FULL_EDGES)
    rebuild = True
    if target.exists():
        try:
            cached = np.load(target, mmap_mode="r", allow_pickle=False)
            rebuild = cached.shape != expected_shape or cached.dtype != np.float32
        except Exception:
            rebuild = True
    if rebuild:
        partial = target.with_suffix(".partial.npy")
        out = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float32, shape=expected_shape
        )
        for start in range(0, len(z), 64):
            stop = min(start + 64, len(z))
            out[start:stop] = np.tanh(np.asarray(z[start:stop], dtype=np.float32))
        out.flush()
        del out
        partial.replace(target)
    pearson = np.load(target, mmap_mode="r", allow_pickle=False)
    audit = {
        "source_fisher_z_path": str(source),
        "source_fisher_z_sha256": sha256_file(source),
        "pearson_cache_path": str(target),
        "pearson_cache_sha256": sha256_file(target),
        "shape": list(pearson.shape),
        "dtype": str(pearson.dtype),
        "conversion": "numpy.tanh(Fisher-z Pearson), exactly reversing feature-builder arctanh apart from its documented clipping",
        "finite": bool(np.isfinite(pearson).all()),
        "range": [float(np.min(pearson)), float(np.max(pearson))],
    }
    return target, audit


def spotcheck_pearson_against_timeseries(
    cohort_dir: Path, pearson: np.ndarray, subject_ids: np.ndarray
) -> dict[str, Any]:
    manifest_path = cohort_dir / "feature_manifest.csv"
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(subject_ids):
        raise RuntimeError("Feature manifest does not align with feature arrays")
    indices = np.linspace(0, len(subject_ids) - 1, 5, dtype=int)
    errors: list[float] = []
    checked: list[str] = []
    for index in indices:
        row = rows[int(index)]
        if row["subject_id"] != str(subject_ids[index]):
            raise RuntimeError("Feature manifest subject ordering mismatch")
        time_series = np.load(row["time_series_path"], allow_pickle=False)
        recomputed = strict.lower_triangle_fc(time_series)
        errors.append(float(np.max(np.abs(recomputed - pearson[index]))))
        checked.append(str(subject_ids[index]))
    maximum = max(errors)
    if maximum > 1e-6:
        raise RuntimeError(f"Pearson cache spot-check failed: max error {maximum}")
    return {
        "status": "PASS",
        "subjects": checked,
        "maximum_absolute_error": maximum,
        "tolerance": 1e-6,
        "reference": "Pearson FC recomputed directly from saved CC200 time series",
    }


def load_cohort(name: str, load_labels: bool) -> tuple[Cohort, dict[str, Any]]:
    cohort_dir = FEATURE_ROOT / name
    fc_path, fc_audit = ensure_pearson_cache(cohort_dir, name)
    fc = np.load(fc_path, mmap_mode="r", allow_pickle=False)
    freq_path = cohort_dir / "X_psd.npy"
    demo_path = cohort_dir / "X_demo_raw.npy"
    ids_path = cohort_dir / "subject_ids.npy"
    sites_path = cohort_dir / "site_ids.npy"
    y_path = cohort_dir / "y_labels.npy"
    freq = np.load(freq_path, mmap_mode="r", allow_pickle=False)
    demo = np.load(demo_path, mmap_mode="r", allow_pickle=False)
    subject_ids = np.load(ids_path, allow_pickle=False).astype(str)
    site_ids = np.load(sites_path, allow_pickle=False).astype(str)
    y = np.load(y_path, allow_pickle=False).astype(np.int64) if load_labels else None
    n = len(subject_ids)
    expected = {
        "fc": (n, strict.N_FULL_EDGES),
        "freq": (n, strict.N_FREQ),
        "demo": (n, strict.N_DEMO),
        "sites": (n,),
    }
    observed = {
        "fc": fc.shape,
        "freq": freq.shape,
        "demo": demo.shape,
        "sites": site_ids.shape,
    }
    if observed != expected:
        raise RuntimeError(f"{name} input contract mismatch: {observed} != {expected}")
    if y is not None and y.shape != (n,):
        raise RuntimeError(f"{name} labels have shape {y.shape}, expected {(n,)}")
    if len(set(subject_ids.tolist())) != n:
        raise RuntimeError(f"{name} contains duplicate subject IDs")
    if not (np.isfinite(freq).all() and np.isfinite(demo).all() and np.isfinite(fc).all()):
        raise RuntimeError(f"{name} contains non-finite model inputs")
    fc_audit["timeseries_spotcheck"] = spotcheck_pearson_against_timeseries(
        cohort_dir, fc, subject_ids
    )
    audit = {
        "cohort": name,
        "n": n,
        "fc": fc_audit,
        "frequency_path": str(freq_path),
        "frequency_sha256": sha256_file(freq_path),
        "frequency_shape": list(freq.shape),
        "frequency_layout": FREQUENCY_LAYOUT,
        "anchor_layout_interpretation": ANCHOR_LAYOUT_FIX,
        "demographic_path": str(demo_path),
        "demographic_sha256": sha256_file(demo_path),
        "subject_ids_sha256": hash_strings(subject_ids),
        "site_ids_sha256": hash_strings(site_ids),
        "labels_path": str(y_path),
        "labels_loaded_during_input_preparation": load_labels,
    }
    return Cohort(name, fc, freq, demo, subject_ids, site_ids, y_path, y), audit


def load_site_config() -> dict[str, Any]:
    config = json.loads(SITE_CONFIG.read_text(encoding="utf-8"))
    return config


def canonical_sites(cohort: Cohort, config: dict[str, Any]) -> np.ndarray:
    branch = "abide_i" if cohort.name == "abide_i" else "abide_ii"
    mapping = config[branch]
    missing = sorted(set(cohort.site_ids.tolist()) - set(mapping))
    if missing:
        raise RuntimeError(f"Unmapped {cohort.name} sites: {missing}")
    if branch == "abide_i":
        return np.asarray([mapping[site] for site in cohort.site_ids], dtype=str)
    return np.asarray([mapping[site]["canonical"] for site in cohort.site_ids], dtype=str)


def input_contract_audit(dev: Cohort, ext: Cohort, audits: list[dict[str, Any]]) -> dict[str, Any]:
    config = load_site_config()
    dev_canonical = canonical_sites(dev, config)
    ext_canonical = canonical_sites(ext, config)
    shared = sorted(set(dev_canonical.tolist()) & set(ext_canonical.tolist()))
    analysis_disjoint = ~np.isin(ext_canonical, shared)
    release_disjoint = np.asarray(
        [not config["abide_ii"][site]["participated_in_abide_i_release"] for site in ext.site_ids],
        dtype=bool,
    )
    value = {
        "status": "PASS",
        "generated_utc": utc_now(),
        "cohorts": audits,
        "strict_base_module": str(WORKSPACE / "strict_nested_cv_interpretable_model.py"),
        "strict_base_module_sha256": sha256_file(WORKSPACE / "strict_nested_cv_interpretable_model.py"),
        "adapter_sha256": sha256_file(Path(__file__)),
        "site_config_path": str(SITE_CONFIG),
        "site_config_sha256": sha256_file(SITE_CONFIG),
        "yeo_mapping_path": str(YEO_MAP),
        "yeo_mapping_sha256": sha256_file(YEO_MAP),
        "shared_canonical_sites_in_analyzed_cohorts": shared,
        "analysis_cohort_site_disjoint_n": int(analysis_disjoint.sum()),
        "release_level_site_disjoint_n": int(release_disjoint.sum()),
        "external_labels_loaded": False,
        "feature_contract": {
            "fc": "19,900 Pearson CC200 lower-triangle edges, derived reversibly from stored Fisher-z edges",
            "frequency": "3,000 log1p Welch PSD values on 15 fixed physical frequencies; stored frequency-major",
            "anchors": "Yeo-7 ROI means after explicit frequency-major to ROI-major interpretation",
            "demographics": "raw age and binary sex; imputation/scaling fit inside training partitions only",
        },
    }
    write_json(RESULT_ROOT / "input_contract_audit.json", value)
    return value


def run_namespace(run_dir: Path, mode: str, resume: bool) -> Namespace:
    smoke = mode == "smoke"
    return Namespace(
        mode=mode,
        outer_splits=2 if smoke else 10,
        inner_splits=2 if smoke else 3,
        inner_epochs=3 if smoke else 300,
        eval_every=1 if smoke else 5,
        seed=SEED,
        anchor_projection_seed=ANCHOR_PROJECTION_SEED,
        run_dir=str(run_dir),
        resume=resume,
        rebuild_full_fc=False,
    )


def run_internal_nested(
    dev: Cohort,
    audit: dict[str, Any],
    mode: str,
    resume: bool,
    device: torch.device,
    yeo_nodes: dict[int, np.ndarray],
    yeo_audit: dict[str, Any],
    base_dir: Path,
) -> Path:
    if dev.y is None:
        raise RuntimeError("ABIDE I labels are required for training")
    run_dir = base_dir / "abide1_nested_cv"
    ensure(run_dir)
    args = run_namespace(run_dir, mode, resume)
    numeric_ids = np.asarray([numeric_abide_id(value) for value in dev.subject_ids], dtype=np.int64)
    write_json(
        run_dir / "run_configuration.json",
        {
            "created_utc": utc_now(),
            "protocol": "CMPB_STRICT_NESTED_V2_FMRIPREP25",
            "arguments": vars(args),
            "participants": len(dev.subject_ids),
            "class_counts": {"asd": int(dev.y.sum()), "td": int(len(dev.y) - dev.y.sum())},
            "device": str(device),
            "input_contract_audit": audit,
            "candidate_grid": [asdict(candidate) for candidate in strict.candidates()],
            "n_selected_fc": strict.N_SELECTED_FC,
            "frequency_layout": FREQUENCY_LAYOUT,
            "anchor_layout_fix": ANCHOR_LAYOUT_FIX,
        },
    )
    outer = StratifiedKFold(
        n_splits=args.outer_splits, shuffle=True, random_state=args.seed
    )
    for fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros(len(dev.y)), dev.y)):
        folder = run_dir / f"fold_{fold:02d}"
        if resume and strict.fold_complete(folder):
            print(f"[cmpb] internal fold {fold} complete; skipping", flush=True)
            continue
        started = time.time()
        print(
            f"[cmpb] internal fold {fold + 1}/{args.outer_splits}: "
            f"train={len(train_idx)}, test={len(test_idx)}, device={device}",
            flush=True,
        )
        strict.outer_fold_run(
            fold,
            train_idx.astype(np.int64),
            test_idx.astype(np.int64),
            dev.fc,
            dev.freq,
            dev.demo,
            dev.y,
            yeo_nodes,
            args,
            device,
            run_dir,
        )
        print(f"[cmpb] fold {fold} finished in {time.time() - started:.1f}s", flush=True)
        strict.aggregate_run(
            run_dir, numeric_ids, dev.y, args.outer_splits, audit, args, yeo_audit
        )
    strict.aggregate_run(
        run_dir, numeric_ids, dev.y, args.outer_splits, audit, args, yeo_audit
    )
    if not (run_dir / "strict_nested_summary.json").exists():
        raise RuntimeError("Internal strict nested CV did not complete")
    return run_dir


def prepare_external(
    dev: Cohort,
    ext: Cohort,
    yeo_nodes: dict[int, np.ndarray],
) -> strict.PreparedData:
    if dev.y is None:
        raise RuntimeError("ABIDE I labels are required")
    selected, scores = strict.select_fc_from_training(dev.fc, dev.y, strict.N_SELECTED_FC)
    scaler_xs = StandardScaler().fit(np.asarray(dev.fc[:, selected], dtype=np.float32))
    scaler_xf = StandardScaler().fit(dev.freq)
    demo_transformer = strict.DemoTransformer.fit(dev.demo)
    x_s_train = scaler_xs.transform(np.asarray(dev.fc[:, selected], dtype=np.float32)).astype(np.float32)
    x_s_eval = scaler_xs.transform(np.asarray(ext.fc[:, selected], dtype=np.float32)).astype(np.float32)
    x_f_train = scaler_xf.transform(dev.freq).astype(np.float32)
    x_f_eval = scaler_xf.transform(ext.freq).astype(np.float32)
    x_d_train = demo_transformer.transform(dev.demo)
    x_d_eval = demo_transformer.transform(ext.demo)
    anchors, profiles = corrected_anchors_from_frequency_major(
        x_f_train, yeo_nodes, 80, ANCHOR_PROJECTION_SEED
    )
    return strict.PreparedData(
        selected,
        scores,
        scaler_xs,
        scaler_xf,
        demo_transformer,
        anchors,
        profiles,
        x_s_train,
        x_f_train,
        x_d_train,
        x_s_eval,
        x_f_eval,
        x_d_eval,
    )


def save_external_transforms(
    folder: Path, prepared: strict.PreparedData, dev: Cohort, ext: Cohort
) -> dict[str, Any]:
    ensure(folder)
    np.save(folder / "selected_full_edge_indices.npy", prepared.selected_edges)
    np.save(folder / "f_scores_full19900.npy", prepared.f_scores)
    np.save(folder / "prior_anchors_abide1_only.npy", prepared.anchors.numpy())
    np.save(folder / "anchor_mean_profiles_abide1_only.npy", prepared.anchor_profiles)
    np.save(folder / "development_subject_ids.npy", dev.subject_ids)
    np.save(folder / "external_subject_ids.npy", ext.subject_ids)
    with (folder / "scaler_xs_abide1.pkl").open("wb") as handle:
        pickle.dump(prepared.scaler_xs, handle)
    with (folder / "scaler_xf_abide1.pkl").open("wb") as handle:
        pickle.dump(prepared.scaler_xf, handle)
    with (folder / "demo_transformer_abide1.pkl").open("wb") as handle:
        pickle.dump(prepared.demo_transformer, handle)
    provenance = {
        "status": "LOCKED_BEFORE_EXTERNAL_LABEL_LOAD",
        "created_utc": utc_now(),
        "development_n": len(dev.subject_ids),
        "external_n": len(ext.subject_ids),
        "development_subject_ids_sha256": hash_strings(dev.subject_ids),
        "external_subject_ids_sha256": hash_strings(ext.subject_ids),
        "selected_edges_sha256": strict.hash_indices(prepared.selected_edges),
        "n_selected_fc": len(prepared.selected_edges),
        "feature_selection_scope": "ABIDE I only",
        "scaler_imputer_scope": "ABIDE I only",
        "anchor_scope": "ABIDE I only",
        "frequency_layout": FREQUENCY_LAYOUT,
        "anchor_layout_fix": ANCHOR_LAYOUT_FIX,
        "external_labels_loaded": False,
    }
    write_json(folder / "transform_provenance.json", provenance)
    return provenance


def binary_metrics(y: np.ndarray, probs: np.ndarray, threshold: float = THRESHOLD) -> dict[str, Any]:
    y = np.asarray(y, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= threshold).astype(int)
    return {
        "n": int(len(y)),
        "asd": int(y.sum()),
        "td": int(len(y) - y.sum()),
        "auc": float(roc_auc_score(y, probs)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "brier": float(brier_score_loss(y, probs)),
        "threshold": float(threshold),
    }


METRIC_KEYS = (
    "auc",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "brier",
)


def subject_bootstrap_ci(y: np.ndarray, probs: np.ndarray, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    class_indices = {label: np.flatnonzero(y == label) for label in (0, 1)}
    draws = {key: [] for key in METRIC_KEYS}
    for _ in range(N_BOOTSTRAP):
        idx = np.concatenate(
            [rng.choice(class_indices[label], len(class_indices[label]), replace=True) for label in (0, 1)]
        )
        values = binary_metrics(y[idx], probs[idx])
        for key in METRIC_KEYS:
            draws[key].append(values[key])
    return {
        "method": "stratified participant bootstrap",
        "iterations": N_BOOTSTRAP,
        "ci95": {
            key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            for key, values in draws.items()
        },
    }


def site_cluster_bootstrap_ci(
    y: np.ndarray, probs: np.ndarray, sites: np.ndarray, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique = np.unique(sites)
    draws = {key: [] for key in METRIC_KEYS}
    valid = 0
    for _ in range(N_BOOTSTRAP):
        sampled_sites = rng.choice(unique, len(unique), replace=True)
        pieces = [np.flatnonzero(sites == site) for site in sampled_sites]
        idx = np.concatenate(pieces)
        if len(np.unique(y[idx])) != 2:
            continue
        values = binary_metrics(y[idx], probs[idx])
        valid += 1
        for key in METRIC_KEYS:
            draws[key].append(values[key])
    if valid == 0:
        raise RuntimeError("No valid two-class site-cluster bootstrap samples")
    return {
        "method": "site-cluster bootstrap",
        "requested_iterations": N_BOOTSTRAP,
        "valid_iterations": valid,
        "sites": int(len(unique)),
        "ci95": {
            key: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]
            for key, values in draws.items()
        },
    }


def evaluate_subset(
    name: str,
    mask: np.ndarray,
    y: np.ndarray,
    probs: np.ndarray,
    sites: np.ndarray,
    canonical: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    if len(np.unique(y[mask])) != 2:
        raise RuntimeError(f"{name} is not a two-class evaluation subset")
    return {
        "name": name,
        "metrics": binary_metrics(y[mask], probs[mask]),
        "subject_bootstrap": subject_bootstrap_ci(y[mask], probs[mask], seed),
        "site_cluster_bootstrap": site_cluster_bootstrap_ci(
            y[mask], probs[mask], canonical[mask], seed + 1000
        ),
        "original_site_ids": sorted(set(sites[mask].tolist())),
        "canonical_sites": sorted(set(canonical[mask].tolist())),
    }


def write_site_metrics(
    path: Path, y: np.ndarray, probs: np.ndarray, sites: np.ndarray, canonical: np.ndarray
) -> None:
    rows: list[dict[str, Any]] = []
    for site in sorted(set(sites.tolist())):
        mask = sites == site
        row: dict[str, Any] = {
            "site_id": site,
            "canonical_site": str(canonical[mask][0]),
            "n": int(mask.sum()),
            "asd": int(y[mask].sum()),
            "td": int(mask.sum() - y[mask].sum()),
        }
        if len(np.unique(y[mask])) == 2:
            row.update(binary_metrics(y[mask], probs[mask]))
        else:
            row["note"] = "single-class site; discrimination metrics not estimated"
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_final_external(
    dev: Cohort,
    ext: Cohort,
    input_audit: dict[str, Any],
    mode: str,
    resume: bool,
    device: torch.device,
    yeo_nodes: dict[int, np.ndarray],
    base_dir: Path,
) -> Path:
    if dev.y is None:
        raise RuntimeError("ABIDE I labels are required")
    final_dir = base_dir / "abide1_to_abide2_locked_external"
    ensure(final_dir)
    summary_path = final_dir / "external_validation_summary.json"
    if resume and summary_path.exists():
        print("[cmpb] locked external validation already complete; skipping", flush=True)
        return final_dir
    args = run_namespace(final_dir, mode, resume)
    inner_rows: list[dict[str, Any]] = []
    all_dev = np.arange(len(dev.y), dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=args.inner_splits, shuffle=True, random_state=SEED + 50_000
    )
    for inner_fold, (fit_idx, val_idx) in enumerate(splitter.split(np.zeros(len(dev.y)), dev.y)):
        for candidate_index, candidate in enumerate(strict.candidates()):
            trial_dir = final_dir / "abide1_inner_selection" / f"inner_{inner_fold:02d}" / candidate.name
            metrics_path = trial_dir / "inner_metrics.json"
            if resume and metrics_path.exists():
                inner_rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                continue
            prepared = strict.prepare_train_eval(
                dev.fc,
                dev.freq,
                dev.demo,
                dev.y,
                fit_idx.astype(np.int64),
                val_idx.astype(np.int64),
                yeo_nodes,
                80,
                ANCHOR_PROJECTION_SEED,
            )
            provenance = strict.save_prepared_artifacts(
                trial_dir, prepared, fit_idx, val_idx, "final_abide1_inner_selection"
            )
            model, training = strict.train_model(
                prepared.x_s_train,
                prepared.x_f_train,
                prepared.x_d_train,
                dev.y[fit_idx],
                candidate,
                prepared.anchors,
                device,
                SEED + 60_000 + inner_fold * 100 + candidate_index,
                args.inner_epochs,
                (
                    prepared.x_s_eval,
                    prepared.x_f_eval,
                    prepared.x_d_eval,
                    dev.y[val_idx],
                ),
                args.eval_every,
            )
            row = {
                "inner_fold": inner_fold,
                "candidate": candidate.name,
                "best_epoch": training["best_epoch"],
                "best_validation_auc": training["best_validation_auc"],
                "best_validation_bce": training["best_validation_bce"],
                "n_fit": int(len(fit_idx)),
                "n_validation": int(len(val_idx)),
                "transform_provenance": provenance,
                "training_history": training["history"],
            }
            write_json(metrics_path, row)
            inner_rows.append(row)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    candidate, fixed_epoch, selection = strict.choose_candidate(inner_rows)
    write_json(final_dir / "abide1_inner_selection.json", selection)

    prepared = prepare_external(dev, ext, yeo_nodes)
    provenance = save_external_transforms(final_dir / "locked_transforms", prepared, dev, ext)
    model, training = strict.train_model(
        prepared.x_s_train,
        prepared.x_f_train,
        prepared.x_d_train,
        dev.y,
        candidate,
        prepared.anchors,
        device,
        SEED + 70_000,
        fixed_epoch,
        monitor=None,
        eval_every=args.eval_every,
    )
    torch.save(
        {
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_class_name": "BrainInnovationSystem",
            "model_config": strict.model_config(candidate),
            "candidate": asdict(candidate),
            "fixed_epoch": fixed_epoch,
            "selection": selection,
            "training_scope": "ABIDE I only",
            "external_labels_used": False,
            "transform_provenance": provenance,
            "frequency_layout": FREQUENCY_LAYOUT,
            "anchor_layout_fix": ANCHOR_LAYOUT_FIX,
        },
        final_dir / "final_model_checkpoint.pt",
    )
    probabilities = strict.predict_probabilities(
        model, prepared.x_s_eval, prepared.x_f_eval, prepared.x_d_eval, device
    )
    config = load_site_config()
    ext_canonical = canonical_sites(ext, config)
    blinded_path = final_dir / "abide2_predictions_blinded.npz"
    np.savez(
        blinded_path,
        probabilities=probabilities,
        subject_ids=ext.subject_ids,
        site_ids=ext.site_ids,
        canonical_sites=ext_canonical,
    )
    write_json(
        final_dir / "prediction_lock.json",
        {
            "status": "PREDICTIONS_LOCKED_BEFORE_LABEL_LOAD",
            "created_utc": utc_now(),
            "prediction_file": str(blinded_path),
            "prediction_file_sha256": sha256_file(blinded_path),
            "external_subject_ids_sha256": hash_strings(ext.subject_ids),
            "external_labels_loaded": False,
            "selected_candidate": candidate.name,
            "fixed_epoch": fixed_epoch,
        },
    )

    # The ABIDE II labels enter only after probabilities and their lock record exist.
    y_external = np.load(ext.y_path, allow_pickle=False).astype(np.int64)
    if y_external.shape != probabilities.shape:
        raise RuntimeError("ABIDE II labels do not align with locked probabilities")
    dev_canonical = canonical_sites(dev, config)
    shared_analyzed = sorted(set(dev_canonical.tolist()) & set(ext_canonical.tolist()))
    analysis_disjoint = ~np.isin(ext_canonical, shared_analyzed)
    release_disjoint = np.asarray(
        [not config["abide_ii"][site]["participated_in_abide_i_release"] for site in ext.site_ids],
        dtype=bool,
    )
    full_mask = np.ones(len(y_external), dtype=bool)
    evaluations = {
        "full_abide_ii": evaluate_subset(
            "full_abide_ii", full_mask, y_external, probabilities, ext.site_ids, ext_canonical, SEED + 1
        ),
        "analysis_cohort_site_disjoint": evaluate_subset(
            "analysis_cohort_site_disjoint",
            analysis_disjoint,
            y_external,
            probabilities,
            ext.site_ids,
            ext_canonical,
            SEED + 2,
        ),
        "release_level_site_disjoint": evaluate_subset(
            "release_level_site_disjoint",
            release_disjoint,
            y_external,
            probabilities,
            ext.site_ids,
            ext_canonical,
            SEED + 3,
        ),
    }
    np.savez(
        final_dir / "abide2_predictions_with_labels.npz",
        probabilities=probabilities,
        y=y_external,
        subject_ids=ext.subject_ids,
        site_ids=ext.site_ids,
        canonical_sites=ext_canonical,
        analysis_cohort_site_disjoint_mask=analysis_disjoint,
        release_level_site_disjoint_mask=release_disjoint,
    )
    write_site_metrics(
        final_dir / "abide2_site_metrics.csv",
        y_external,
        probabilities,
        ext.site_ids,
        ext_canonical,
    )
    summary = {
        "status": "COMPLETE_LOCKED_EXTERNAL_VALIDATION",
        "generated_utc": utc_now(),
        "development": {
            "cohort": "ABIDE I",
            "n": len(dev.y),
            "asd": int(dev.y.sum()),
            "td": int(len(dev.y) - dev.y.sum()),
        },
        "external": {
            "cohort": "ABIDE II",
            "n": len(y_external),
            "asd": int(y_external.sum()),
            "td": int(len(y_external) - y_external.sum()),
            "labels_sha256": sha256_file(ext.y_path),
        },
        "selected_candidate": candidate.name,
        "fixed_epoch": fixed_epoch,
        "fixed_threshold": THRESHOLD,
        "selection": selection,
        "final_training": training,
        "evaluations": evaluations,
        "site_disjoint_definition": {
            "primary": config["primary_site_disjoint_rule"],
            "release_level": config["release_level_rule"],
            "shared_sites_in_analyzed_cohorts": shared_analyzed,
        },
        "selection_integrity": {
            "feature_selection": "ABIDE I only",
            "scaling_imputation": "ABIDE I only",
            "anchor_construction": "ABIDE I only",
            "candidate_and_epoch": "ABIDE I inner CV only",
            "decision_threshold": "fixed a priori at 0.5",
            "predictions_locked_before_external_label_load": True,
            "abide_ii_used_for_selection": False,
        },
        "input_contract_audit": input_audit,
    }
    write_json(summary_path, summary)
    full = evaluations["full_abide_ii"]["metrics"]
    disjoint = evaluations["analysis_cohort_site_disjoint"]["metrics"]
    release = evaluations["release_level_site_disjoint"]["metrics"]
    report = f"""# CMPB strict ABIDE I to ABIDE II external validation

Generated: {summary['generated_utc']}

## Locked protocol

- Development: ABIDE I only (n={len(dev.y)}).
- External test: ABIDE II (n={len(y_external)}).
- CMPB candidate and fixed epoch selected only by ABIDE I inner CV.
- All FC selection, scalers, demographic imputation and Yeo-7 anchors fit on ABIDE I only.
- Decision threshold fixed a priori at 0.5.
- ABIDE II probabilities were saved and hashed before labels were loaded.

## Results

| Evaluation | n | AUC | Balanced accuracy | Sensitivity | Specificity |
|---|---:|---:|---:|---:|---:|
| Full ABIDE II | {full['n']} | {full['auc']:.4f} | {full['balanced_accuracy']:.4f} | {full['sensitivity']:.4f} | {full['specificity']:.4f} |
| Site-disjoint vs analyzed ABIDE I | {disjoint['n']} | {disjoint['auc']:.4f} | {disjoint['balanced_accuracy']:.4f} | {disjoint['sensitivity']:.4f} | {disjoint['specificity']:.4f} |
| Site-disjoint vs entire ABIDE I release | {release['n']} | {release['auc']:.4f} | {release['balanced_accuracy']:.4f} | {release['sensitivity']:.4f} | {release['specificity']:.4f} |

## Site-disjoint boundary

The primary sensitivity excludes canonical institutions represented among the
actual post-QC ABIDE I development participants. The more conservative release-
level analysis also excludes ABIDE II institutions documented as ABIDE I
contributors even when no participant from that institution survived into the
analyzed ABIDE I cohort.

## Claim boundary

This is independent-cohort technical validation under harmonized fMRIPrep and
postprocessing. It does not establish clinical utility, prospective validity,
or causal biomarkers.
"""
    (final_dir / "CMPB_STRICT_EXTERNAL_VALIDATION_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "smoke", "full"], default="prepare")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-internal", action="store_true")
    args = parser.parse_args()
    ensure(RESULT_ROOT)
    dev, dev_audit = load_cohort("abide_i", load_labels=True)
    ext, ext_audit = load_cohort("abide_ii", load_labels=False)
    audit = input_contract_audit(dev, ext, [dev_audit, ext_audit])
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if args.stage == "prepare":
        return 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yeo_nodes, yeo_audit = strict.load_yeo7_nodes()
    base_dir = RESULT_ROOT / args.stage
    ensure(base_dir)
    if not args.skip_internal:
        run_internal_nested(
            dev, audit, args.stage, args.resume, device, yeo_nodes, yeo_audit, base_dir
        )
    run_final_external(
        dev, ext, audit, args.stage, args.resume, device, yeo_nodes, base_dir
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        print(traceback.format_exc(), flush=True)
        raise

"""Canonical ABIDE-I C-PAC CC200 nested training with loadable checkpoints.

The selection/training engine is reused verbatim from cpac_model_optimization_v1.
This adapter changes only the input source (the audited node-major frequency
matrix) and the persistence/audit layer around the 30 outer-final models.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cpac_leakage_free_nested_cv as audited  # noqa: E402
import cpac_model_optimization_v1 as optimization  # noqa: E402


OUTPUT_ROOT = Path(__file__).resolve().parent
CANONICAL_RUN_DIR = OUTPUT_ROOT / "canonical_run"
SMOKE_RUN_DIR = OUTPUT_ROOT / "smoke_run"
PHASE1_DIR = PROJECT_ROOT / "cpac_frequency_fix_v2"
CORRECTED_FREQUENCY_PATH = PHASE1_DIR / "X_freq_node_major_corrected_879.npy"
CORRECTED_SUBJECT_IDS_PATH = PHASE1_DIR / "subject_ids.npy"
FREQUENCY_AUDIT_PATH = PHASE1_DIR / "frequency_input_audit.json"
FREQUENCY_MANIFEST_PATH = PHASE1_DIR / "protocol_manifest.json"
SOURCE_DATA_DIR = PROJECT_ROOT / "abide_data" / "Outputs" / "cpac" / "filt_noglobal"
FULL_FC_PATH = SOURCE_DATA_DIR / audited.FULL_FC_FILENAME
SOURCE_SUBJECT_IDS_PATH = SOURCE_DATA_DIR / audited.SUBJECT_IDS_FILENAME
LABELS_PATH = SOURCE_DATA_DIR / audited.LABEL_FILENAME
PHENOTYPE_PATH = SOURCE_DATA_DIR / audited.PHENOTYPE_FILENAME
PROTOCOL_MANIFEST_PATH = OUTPUT_ROOT / "protocol_manifest.json"
CHECKPOINT_RELOAD_AUDIT_PATH = OUTPUT_ROOT / "checkpoint_reload_audit.json"
OOF_PATH = OUTPUT_ROOT / "oof_predictions.npz"
METRICS_PATH = OUTPUT_ROOT / "metrics.json"
FOLD_METRICS_PATH = OUTPUT_ROOT / "fold_metrics.csv"
LINEAGE_AUDIT_PATH = OUTPUT_ROOT / "canonical_lineage_audit.json"
HISTORICAL_METRICS_PATH = (
    PROJECT_ROOT
    / "cpac_model_optimization_v1"
    / "full_10fold_frozen_20260717_182329"
    / "one_time_evaluation_20260718"
    / "optimized_metrics.json"
)

EXPECTED_FREQUENCY_SHA256 = "04fb454092de4fd9a869289d99a4ebcf619cd9716edd5e43dba1350fdbc712ef"
EXPECTED_SUBJECT_IDS_SHA256 = "4d77aa25ecea1529f5b23e61b5e83923a1d8f002bbb355225053cb4035e2c7a8"
CHECKPOINT_RELOAD_ATOL = 1e-7
FINAL_SEEDS = (42, 43, 44)
PRIMARY_THRESHOLD = 0.5
BOOTSTRAP_SEED = 20260827
BOOTSTRAP_REPLICATES = 20_000


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    return audited.sha256_array(np.asarray(values))


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([clean(row) for row in rows])


def phase1_gate() -> dict[str, Any]:
    required = [
        CORRECTED_FREQUENCY_PATH,
        CORRECTED_SUBJECT_IDS_PATH,
        FREQUENCY_AUDIT_PATH,
        FREQUENCY_MANIFEST_PATH,
    ]
    missing = [str(path.resolve()) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Phase-1 gate FAIL: missing files: {missing}")
    frequency_audit = json.loads(FREQUENCY_AUDIT_PATH.read_text(encoding="utf-8"))
    frequency_manifest = json.loads(FREQUENCY_MANIFEST_PATH.read_text(encoding="utf-8"))
    frequency = np.load(CORRECTED_FREQUENCY_PATH, mmap_mode="r", allow_pickle=False)
    output_ids = np.load(CORRECTED_SUBJECT_IDS_PATH, allow_pickle=False)
    source_ids = np.load(SOURCE_SUBJECT_IDS_PATH, allow_pickle=False)
    frequency_hash = sha256_file(CORRECTED_FREQUENCY_PATH)
    ids_hash = sha256_file(CORRECTED_SUBJECT_IDS_PATH)
    checks = {
        "frequency_audit_pass": frequency_audit.get("status") == "PASS",
        "frequency_manifest_pass": frequency_manifest.get("status") == "PASS",
        "frequency_shape_879_3000": frequency.shape == (879, 3000),
        "frequency_all_finite": bool(np.isfinite(frequency).all()),
        "frequency_sha256_expected": frequency_hash == EXPECTED_FREQUENCY_SHA256,
        "subject_ids_sha256_expected": ids_hash == EXPECTED_SUBJECT_IDS_SHA256,
        "subject_ids_shape_879": output_ids.shape == (879,),
        "subject_ids_unique": np.unique(output_ids).size == 879,
        "subject_ids_equal_source": bool(np.array_equal(output_ids, source_ids)),
        "audit_frequency_sha_matches": (
            frequency_audit.get("arrays", {})
            .get("corrected_frequency", {})
            .get("sha256")
            == frequency_hash
        ),
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(f"Phase-1 gate FAIL: {failed}")
    return {
        "status": "PASS",
        "checked_at": now_iso(),
        "checks": checks,
        "frequency_path": CORRECTED_FREQUENCY_PATH,
        "frequency_sha256": frequency_hash,
        "subject_ids_path": CORRECTED_SUBJECT_IDS_PATH,
        "subject_ids_sha256": ids_hash,
        "frequency_audit_sha256": sha256_file(FREQUENCY_AUDIT_PATH),
        "frequency_manifest_sha256": sha256_file(FREQUENCY_MANIFEST_PATH),
    }


def canonical_input_hashes() -> dict[str, str]:
    paths = [
        CORRECTED_FREQUENCY_PATH,
        CORRECTED_SUBJECT_IDS_PATH,
        FREQUENCY_AUDIT_PATH,
        FREQUENCY_MANIFEST_PATH,
        FULL_FC_PATH,
        SOURCE_SUBJECT_IDS_PATH,
        LABELS_PATH,
        PHENOTYPE_PATH,
        optimization.PROTOCOL_PATH,
        optimization.CANDIDATE_TABLE_PATH,
        PROJECT_ROOT / "cpac_model_optimization_v1.py",
    ]
    return {str(path.resolve()): sha256_file(path) for path in paths}


def load_canonical_inputs() -> tuple[audited.FeatureBundle, audited.LabelFirewall, np.ndarray]:
    phase1_gate()
    full_fc = np.load(FULL_FC_PATH, mmap_mode="r", allow_pickle=False)
    frequency = np.load(CORRECTED_FREQUENCY_PATH, allow_pickle=False).astype(np.float32)
    subject_ids = np.load(CORRECTED_SUBJECT_IDS_PATH, allow_pickle=False).astype(np.int64)
    source_ids = np.load(SOURCE_SUBJECT_IDS_PATH, allow_pickle=False).astype(np.int64)
    if not np.array_equal(subject_ids, source_ids):
        raise RuntimeError("Corrected frequency subject IDs do not match full-FC subject IDs")
    phenotype = audited.read_phenotype_records(PHENOTYPE_PATH)
    demographics = np.asarray(
        [[phenotype[int(subject_id)]["age"], phenotype[int(subject_id)]["sex"]] for subject_id in subject_ids],
        dtype=np.float32,
    )
    labels = np.load(LABELS_PATH, allow_pickle=False).astype(np.int64)
    reconstructed = np.asarray(
        [phenotype[int(subject_id)]["label"] for subject_id in subject_ids], dtype=np.int64
    )
    if not np.array_equal(labels, reconstructed):
        raise RuntimeError("Labels do not match phenotype in corrected subject order")
    features = audited.FeatureBundle(
        full_fc=full_fc,
        frequency=frequency,
        demographics_raw=demographics,
        subject_ids=subject_ids,
        source_files={
            "full_fc": str(FULL_FC_PATH.resolve()),
            "frequency": str(CORRECTED_FREQUENCY_PATH.resolve()),
            "subject_ids": str(CORRECTED_SUBJECT_IDS_PATH.resolve()),
            "demographics_raw": str(PHENOTYPE_PATH.resolve()),
        },
    )
    features.validate()
    return features, audited.LabelFirewall(labels, subject_ids), labels


def initialize_protocol_manifest(status: str = "READY") -> dict[str, Any]:
    gate = phase1_gate()
    protocol_check = optimization.verify_implementation_protocol()
    manifest = {
        "status": status,
        "created_at": now_iso(),
        "protocol_name": "cpac_model_xai_aligned_v2",
        "scope": {
            "dataset": "ABIDE-I",
            "pipeline": "C-PAC",
            "preprocessing": "filt_noglobal",
            "atlas": "CC200",
            "cohort_size": 879,
            "frequency_layout": "node-major corrected v2",
            "excluded": ["ABIDE-II", "fMRIPrep25", "LEAP", "ADHD200", "external cohorts", "new model families"],
        },
        "phase1_gate": gate,
        "nested_design": {
            "outer_folds": 10,
            "outer_split_seed": 42,
            "inner_folds": 3,
            "inner_split_seed_formula": "42 + 10000 + outer_fold",
            "final_base_seeds": list(FINAL_SEEDS),
            "seed_probability_aggregation": "unweighted arithmetic mean",
            "candidate_registry_sha256": optimization.CANDIDATE_REGISTRY_SHA256,
            "candidate_count": len(optimization.CANDIDATES),
            "selection_and_epoch_scope": "outer-train inner-CV only",
            "outer_test_role": "prediction/evaluation and held-out XAI only",
        },
        "protocol_verification": protocol_check,
        "input_sha256": canonical_input_hashes(),
        "historical_context": {
            "old_optimized_metrics_path": HISTORICAL_METRICS_PATH,
            "old_optimized_metrics_sha256": sha256_file(HISTORICAL_METRICS_PATH),
            "old_raw_oof_auc": 0.7091714434601355,
            "old_explanation_oof_auc": 0.6891,
            "policy": "Historical numbers and explanations are context only and are not canonical-v2 results.",
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
        "paths": {
            "output_root": OUTPUT_ROOT,
            "canonical_run": CANONICAL_RUN_DIR,
            "smoke_run": SMOKE_RUN_DIR,
            "generator": Path(__file__).resolve(),
        },
        "artifact_sha256": {},
    }
    write_json(PROTOCOL_MANIFEST_PATH, manifest)
    return manifest


def update_protocol_manifest(status: str, **updates: Any) -> dict[str, Any]:
    manifest = (
        json.loads(PROTOCOL_MANIFEST_PATH.read_text(encoding="utf-8"))
        if PROTOCOL_MANIFEST_PATH.exists()
        else initialize_protocol_manifest(status)
    )
    manifest["status"] = status
    manifest["updated_at"] = now_iso()
    manifest.update(clean(updates))
    write_json(PROTOCOL_MANIFEST_PATH, manifest)
    return manifest


def _save_array(path: Path, values: np.ndarray) -> dict[str, Any]:
    np.save(path, np.asarray(values), allow_pickle=False)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "path": path.resolve(),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": sha256_file(path),
        "values_sha256": sha256_array(array),
    }


def save_preprocessing_state(
    seed_dir: Path,
    prepared: optimization.OptimizationPreparedPartition,
    candidate: optimization.OptimizationCandidate,
    features: audited.FeatureBundle,
) -> tuple[Path, dict[str, Any]]:
    seed_dir.mkdir(parents=True, exist_ok=True)
    state_path = seed_dir / "preprocessing_state.npz"
    fc_scaler = prepared.scaler_fc
    frequency_scaler = prepared.scaler_frequency
    demo = prepared.demographic_transformer
    payload = {
        "fit_indices": prepared.fit_indices.astype(np.int64),
        "evaluation_indices": prepared.evaluation_indices.astype(np.int64),
        "selected_edges": prepared.selected_edges.astype(np.int64),
        "f_scores": prepared.f_scores.astype(np.float64),
        "fc_mean": np.asarray(fc_scaler.mean_, dtype=np.float64),
        "fc_scale": np.asarray(fc_scaler.scale_, dtype=np.float64),
        "fc_var": np.asarray(fc_scaler.var_, dtype=np.float64),
        "frequency_enabled": np.asarray([int(frequency_scaler is not None)], dtype=np.int8),
        "frequency_mean": np.asarray(frequency_scaler.mean_, dtype=np.float64) if frequency_scaler is not None else np.empty(0, dtype=np.float64),
        "frequency_scale": np.asarray(frequency_scaler.scale_, dtype=np.float64) if frequency_scaler is not None else np.empty(0, dtype=np.float64),
        "frequency_var": np.asarray(frequency_scaler.var_, dtype=np.float64) if frequency_scaler is not None else np.empty(0, dtype=np.float64),
        "demographics_enabled": np.asarray([int(demo is not None)], dtype=np.int8),
        "demographic_impute": np.asarray([demo.age_impute_value, demo.sex_impute_value], dtype=np.float64) if demo is not None else np.empty(0, dtype=np.float64),
        "demographic_mean": np.asarray(demo.scaler.mean_, dtype=np.float64) if demo is not None else np.empty(0, dtype=np.float64),
        "demographic_scale": np.asarray(demo.scaler.scale_, dtype=np.float64) if demo is not None else np.empty(0, dtype=np.float64),
        "demographic_var": np.asarray(demo.scaler.var_, dtype=np.float64) if demo is not None else np.empty(0, dtype=np.float64),
        "anchors": prepared.anchors.astype(np.float32),
        "anchor_profiles": prepared.anchor_profiles.astype(np.float32),
    }
    np.savez_compressed(state_path, **payload)
    train_ids_path = seed_dir / "train_subject_ids.npy"
    test_ids_path = seed_dir / "test_subject_ids.npy"
    train_indices_path = seed_dir / "train_indices.npy"
    test_indices_path = seed_dir / "test_indices.npy"
    files = {
        "train_indices": _save_array(train_indices_path, prepared.fit_indices.astype(np.int64)),
        "test_indices": _save_array(test_indices_path, prepared.evaluation_indices.astype(np.int64)),
        "train_subject_ids": _save_array(train_ids_path, features.subject_ids[prepared.fit_indices].astype(np.int64)),
        "test_subject_ids": _save_array(test_ids_path, features.subject_ids[prepared.evaluation_indices].astype(np.int64)),
    }
    manifest = {
        "candidate_id": candidate.candidate_id,
        "trial_scope": prepared.trial_scope,
        "preprocessing_state_path": state_path.resolve(),
        "preprocessing_state_sha256": sha256_file(state_path),
        "fit_indices_sha256": audited.sha256_indices(prepared.fit_indices),
        "evaluation_indices_sha256": audited.sha256_indices(prepared.evaluation_indices),
        "selected_edges_sha256": audited.sha256_indices(prepared.selected_edges),
        "selected_edge_count": int(len(prepared.selected_edges)),
        "fit_outer_test_overlap": int(len(np.intersect1d(prepared.fit_indices, prepared.evaluation_indices))),
        "selector_fit_scope": "outer_train_only",
        "fc_scaler_fit_scope": "outer_train_only",
        "frequency_scaler_fit_scope": "outer_train_only" if frequency_scaler is not None else "not_applicable",
        "frequency_anchor_fit_scope": "outer_train_only_after_outer_train_scaling" if frequency_scaler is not None else "not_applicable",
        "demographic_imputation_scaling_scope": "outer_train_only" if demo is not None else "not_applicable",
        "files": files,
    }
    write_json(seed_dir / "preprocessing_state_manifest.json", manifest)
    return state_path, manifest


def apply_saved_preprocessing(
    state_path: Path,
    features: audited.FeatureBundle,
    partition: str,
) -> dict[str, np.ndarray]:
    with np.load(state_path, allow_pickle=False) as state:
        key = "fit_indices" if partition == "train" else "evaluation_indices"
        if partition not in {"train", "test"}:
            raise ValueError("partition must be train or test")
        indices = state[key].astype(np.int64)
        selected = state["selected_edges"].astype(np.int64)
        fc_scaler = StandardScaler()
        fc_scaler.mean_ = state["fc_mean"].astype(np.float64)
        fc_scaler.scale_ = state["fc_scale"].astype(np.float64)
        fc_scaler.var_ = state["fc_var"].astype(np.float64)
        fc_scaler.n_features_in_ = len(fc_scaler.mean_)
        x_fc = fc_scaler.transform(
            np.asarray(features.full_fc[indices][:, selected], dtype=np.float32)
        ).astype(np.float32)
        if bool(state["frequency_enabled"][0]):
            frequency_scaler = StandardScaler()
            frequency_scaler.mean_ = state["frequency_mean"].astype(np.float64)
            frequency_scaler.scale_ = state["frequency_scale"].astype(np.float64)
            frequency_scaler.var_ = state["frequency_var"].astype(np.float64)
            frequency_scaler.n_features_in_ = len(frequency_scaler.mean_)
            x_frequency = frequency_scaler.transform(
                np.asarray(features.frequency[indices], dtype=np.float32)
            ).astype(np.float32)
        else:
            x_frequency = np.zeros((len(indices), audited.N_FREQUENCY), dtype=np.float32)
        if bool(state["demographics_enabled"][0]):
            demographic = np.asarray(features.demographics_raw[indices], dtype=np.float64).copy()
            impute = state["demographic_impute"]
            demographic[:, 0] = np.where(np.isfinite(demographic[:, 0]), demographic[:, 0], impute[0])
            demographic[:, 1] = np.where(np.isfinite(demographic[:, 1]), demographic[:, 1], impute[1])
            demographic_scaler = StandardScaler()
            demographic_scaler.mean_ = state["demographic_mean"].astype(np.float64)
            demographic_scaler.scale_ = state["demographic_scale"].astype(np.float64)
            demographic_scaler.var_ = state["demographic_var"].astype(np.float64)
            demographic_scaler.n_features_in_ = len(demographic_scaler.mean_)
            x_demographic = demographic_scaler.transform(demographic).astype(np.float32)
        else:
            x_demographic = np.zeros((len(indices), audited.N_DEMOGRAPHIC), dtype=np.float32)
        return {
            "indices": indices,
            "selected_edges": selected,
            "x_fc": x_fc,
            "x_frequency": x_frequency,
            "x_demographic": x_demographic,
            "anchors": state["anchors"].astype(np.float32),
            "anchor_profiles": state["anchor_profiles"].astype(np.float32),
        }


def predict_logits_probabilities(
    model: torch.nn.Module,
    candidate: optimization.OptimizationCandidate,
    x_fc: np.ndarray,
    x_frequency: np.ndarray,
    x_demographic: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    if hasattr(model, "set_eval_ocread_mode"):
        model.set_eval_ocread_mode("mu")
    logits_parts: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_fc), 64):
            stop = min(start + 64, len(x_fc))
            xs = torch.from_numpy(x_fc[start:stop]).to(device)
            if candidate.family == "fc_only_compact":
                logits = model(xs)
            else:
                logits, _, _, _ = model(
                    xs,
                    torch.from_numpy(x_frequency[start:stop]).to(device),
                    torch.from_numpy(x_demographic[start:stop]).to(device),
                    tau=0.15,
                    training=False,
                )
            logits_parts.append(logits.detach().cpu().numpy().astype(np.float64))
    logits_array = np.concatenate(logits_parts).reshape(-1)
    probabilities = (1.0 / (1.0 + np.exp(-logits_array))).astype(np.float64)
    if not np.isfinite(logits_array).all() or not np.isfinite(probabilities).all():
        raise RuntimeError("Prediction contains NaN or Inf")
    return logits_array, probabilities


def make_model_from_checkpoint(
    checkpoint_path: Path,
    preprocessing_state_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, optimization.OptimizationCandidate, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    candidate = optimization.CANDIDATE_BY_ID[str(checkpoint["candidate_id"])]
    with np.load(preprocessing_state_path, allow_pickle=False) as state:
        prepared_stub = SimpleNamespace(anchors=state["anchors"].astype(np.float32))
    backend = optimization.TorchOptimizationBackend(device)
    model = backend._make_model(candidate, prepared_stub)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    if hasattr(model, "set_eval_ocread_mode"):
        model.set_eval_ocread_mode("mu")
    return model, candidate, checkpoint


class CanonicalCheckpointBackend(optimization.TorchOptimizationBackend):
    """Original optimizer plus outer-final persistence and immediate reload."""

    def __init__(
        self,
        device: torch.device,
        fold_directory: Path,
        features: audited.FeatureBundle,
        input_hashes: dict[str, str],
    ):
        super().__init__(device)
        self.fold_directory = fold_directory
        self.features = features
        self.input_hashes = input_hashes

    def run_inner(self, **kwargs: Any) -> optimization.InnerTrialResult:
        candidate = kwargs["candidate"]
        print(
            f"[inner] fold={kwargs['outer_fold']} candidate={candidate.candidate_id} "
            f"seed={kwargs['base_seed']} inner={kwargs['inner_fold']} "
            f"budget={kwargs['budget_epoch']}",
            flush=True,
        )
        started = time.time()
        result = super().run_inner(**kwargs)
        print(
            f"[inner-complete] fold={kwargs['outer_fold']} candidate={candidate.candidate_id} "
            f"seed={kwargs['base_seed']} inner={kwargs['inner_fold']} "
            f"best_epoch={result.best_epoch} auc={result.best_validation_auc:.6f} "
            f"seconds={time.time()-started:.1f}",
            flush=True,
        )
        return result

    def run_outer(
        self,
        *,
        candidate: optimization.OptimizationCandidate,
        prepared: optimization.OptimizationPreparedPartition,
        y_fit: np.ndarray,
        base_seed: int,
        actual_seed: int,
        outer_fold: int,
        fixed_epoch: int,
    ) -> optimization.OuterFitResult:
        print(
            f"[outer-final] fold={outer_fold} candidate={candidate.candidate_id} "
            f"seed={base_seed} epochs={fixed_epoch}",
            flush=True,
        )
        started = time.time()
        self._set_seed(actual_seed)
        model = self._make_model(candidate, prepared)
        optimizer, scheduler = self._make_optimizer_scheduler(model, candidate, fixed_epoch)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(actual_seed)
        loader = self._loader(prepared, y_fit, candidate, generator)
        model_token = f"canonical-model-fold{outer_fold:02d}-seed{base_seed}"
        optimizer_token = f"canonical-optimizer-fold{outer_fold:02d}-seed{base_seed}"
        for epoch in range(1, fixed_epoch + 1):
            self._train_epoch(model, optimizer, loader, candidate, epoch)
            if candidate.family == "fc_only_compact" or epoch > 60:
                scheduler.step()
        logits, probabilities = predict_logits_probabilities(
            model,
            candidate,
            prepared.x_fc_evaluation,
            prepared.x_frequency_evaluation,
            prepared.x_demographic_evaluation,
            self.device,
        )
        model_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        state_sha = audited.state_dict_sha256(model_state)
        seed_dir = self.fold_directory / "outer_final" / f"seed_{base_seed}"
        preprocessing_path, preprocessing_manifest = save_preprocessing_state(
            seed_dir, prepared, candidate, self.features
        )
        checkpoint_path = seed_dir / "model_checkpoint.pt"
        checkpoint = {
            "schema_version": 2,
            "lineage": "cpac_model_xai_aligned_v2",
            "outer_fold": int(outer_fold),
            "base_seed": int(base_seed),
            "actual_seed": int(actual_seed),
            "candidate_id": candidate.candidate_id,
            "candidate_config": asdict(candidate),
            "fixed_epoch": int(fixed_epoch),
            "model_state_dict": model_state,
            "model_state_dict_sha256": state_sha,
            "frequency_layout": "node-major-corrected-v2",
            "frequency_input_sha256": EXPECTED_FREQUENCY_SHA256,
            "preprocessing_state_sha256": preprocessing_manifest["preprocessing_state_sha256"],
            "outer_split_sha256": hashlib.sha256(
                np.asarray(prepared.fit_indices, dtype=np.int64).tobytes()
                + b"|"
                + np.asarray(prepared.evaluation_indices, dtype=np.int64).tobytes()
            ).hexdigest(),
        }
        torch.save(checkpoint, checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)
        outputs_path = seed_dir / "test_outputs.npz"
        np.savez_compressed(
            outputs_path,
            row_indices=prepared.evaluation_indices.astype(np.int64),
            subject_ids=self.features.subject_ids[prepared.evaluation_indices].astype(np.int64),
            logits=logits.astype(np.float64),
            probabilities=probabilities.astype(np.float64),
        )

        # Reload both persisted preprocessing and weights, build a new model,
        # and infer from raw canonical inputs rather than the in-memory arrays.
        reloaded_partition = apply_saved_preprocessing(
            preprocessing_path, self.features, "test"
        )
        reloaded_model, reloaded_candidate, reloaded_checkpoint = make_model_from_checkpoint(
            checkpoint_path, preprocessing_path, self.device
        )
        reload_logits, reload_probabilities = predict_logits_probabilities(
            reloaded_model,
            reloaded_candidate,
            reloaded_partition["x_fc"],
            reloaded_partition["x_frequency"],
            reloaded_partition["x_demographic"],
            self.device,
        )
        logits_error = float(np.max(np.abs(reload_logits - logits)))
        probability_error = float(np.max(np.abs(reload_probabilities - probabilities)))
        reload_pass = bool(
            probability_error <= CHECKPOINT_RELOAD_ATOL
            and reloaded_checkpoint["model_state_dict_sha256"] == state_sha
            and audited.state_dict_sha256(reloaded_checkpoint["model_state_dict"]) == state_sha
        )
        reload_audit = {
            "status": "PASS" if reload_pass else "FAIL",
            "outer_fold": outer_fold,
            "base_seed": base_seed,
            "candidate_id": candidate.candidate_id,
            "fixed_epoch": fixed_epoch,
            "fresh_model_instance": True,
            "preprocessing_reloaded_from_disk": True,
            "checkpoint_reloaded_from_disk": True,
            "checkpoint_path": checkpoint_path.resolve(),
            "checkpoint_file_sha256": checkpoint_sha,
            "model_state_dict_sha256": state_sha,
            "preprocessing_state_path": preprocessing_path.resolve(),
            "preprocessing_state_sha256": sha256_file(preprocessing_path),
            "test_outputs_path": outputs_path.resolve(),
            "test_outputs_sha256": sha256_file(outputs_path),
            "training_end_logits_sha256": sha256_array(logits),
            "reload_logits_sha256": sha256_array(reload_logits),
            "training_end_probabilities_sha256": sha256_array(probabilities),
            "reload_probabilities_sha256": sha256_array(reload_probabilities),
            "max_abs_logit_error": logits_error,
            "max_abs_probability_error": probability_error,
            "absolute_tolerance": CHECKPOINT_RELOAD_ATOL,
            "test_indices_sha256": audited.sha256_indices(prepared.evaluation_indices),
            "test_subject_ids_sha256": sha256_array(self.features.subject_ids[prepared.evaluation_indices].astype(np.int64)),
            "input_sha256": self.input_hashes,
        }
        write_json(seed_dir / "reload_audit.json", reload_audit)
        checkpoint_manifest = {
            "status": "PASS" if reload_pass else "FAIL",
            "outer_fold": outer_fold,
            "base_seed": base_seed,
            "actual_seed": actual_seed,
            "candidate_id": candidate.candidate_id,
            "candidate_config": asdict(candidate),
            "fixed_epoch": fixed_epoch,
            "checkpoint": {
                "path": checkpoint_path.resolve(),
                "sha256": checkpoint_sha,
                "state_dict_sha256": state_sha,
            },
            "preprocessing": preprocessing_manifest,
            "selected_fc_edges_path": (seed_dir / "preprocessing_state.npz").resolve(),
            "test_outputs": {
                "path": outputs_path.resolve(),
                "sha256": sha256_file(outputs_path),
                "logits_sha256": sha256_array(logits),
                "probabilities_sha256": sha256_array(probabilities),
            },
            "train_indices_sha256": audited.sha256_indices(prepared.fit_indices),
            "test_indices_sha256": audited.sha256_indices(prepared.evaluation_indices),
            "outer_split_sha256": checkpoint["outer_split_sha256"],
            "input_sha256": self.input_hashes,
            "reload_audit": reload_audit,
            "outer_test_labels_used_for_training_selection_or_checkpointing": False,
        }
        write_json(seed_dir / "checkpoint_manifest.json", checkpoint_manifest)
        if not reload_pass:
            raise RuntimeError(
                f"Checkpoint reload failed for fold {outer_fold}, seed {base_seed}: "
                f"probability error={probability_error}"
            )
        print(
            f"[outer-final-complete] fold={outer_fold} seed={base_seed} "
            f"reload_max={probability_error:.3g} seconds={time.time()-started:.1f}",
            flush=True,
        )
        result = optimization.OuterFitResult(
            candidate_id=candidate.candidate_id,
            base_seed=base_seed,
            fixed_epoch=fixed_epoch,
            probabilities=probabilities,
            checkpoint_state_sha256=state_sha,
            model_instance_token=model_token,
            optimizer_instance_token=optimizer_token,
            preprocessing_state_token=prepared.state_token,
        )
        del model, reloaded_model, optimizer, scheduler, loader
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result


def run_fold(
    fold: int,
    run_dir: Path,
    device_name: str,
    smoke: bool = False,
) -> dict[str, Any]:
    features, firewall, _labels = load_canonical_inputs()
    splits = firewall.make_outer_splits(optimization.OUTER_SPLIT_SEED)
    audited.validate_outer_splits(splits, len(features.subject_ids))
    if fold not in range(10):
        raise ValueError("fold must be 0..9")
    fold_dir = run_dir / "folds" / f"fold_{fold:02d}"
    if fold_dir.exists():
        raise FileExistsError(f"Refusing to overwrite fold output: {fold_dir}")
    device = audited.choose_device(device_name)
    backend = CanonicalCheckpointBackend(device, fold_dir, features, canonical_input_hashes())
    if smoke:
        candidates = (optimization.CANDIDATE_BY_ID["C01"], optimization.CANDIDATE_BY_ID["C09"])
        schedule = optimization.EngineSchedule(
            level1_rung1_epoch=1,
            level1_max_epoch=2,
            level1_minimum_epoch=2,
            level1_patience=1,
            level1_rung1_survivors=2,
            level1_final_survivors=2,
            level2_max_epoch=2,
            level2_minimum_epoch=2,
            level2_patience=1,
            evaluation_interval=1,
        )
    else:
        candidates = optimization.CANDIDATES
        schedule = optimization.EngineSchedule()
    started = time.time()
    result = optimization.run_outer_fold_optimization(
        features=features,
        label_firewall=firewall,
        outer_split=splits[fold],
        output_directory=fold_dir,
        backend=backend,
        yeo_nodes=audited.load_yeo7_nodes(),
        calibration_mode="platt",
        candidates=candidates,
        schedule=schedule,
    )
    stage_b = optimization.evaluate_frozen_outer_prediction(
        prediction_path=result["prediction_path"],
        manifest_path=result["manifest_path"],
        label_firewall=firewall,
        output_path=fold_dir / "stage_b_outer_metrics.json",
    )
    summary = {
        "status": "PASS",
        "smoke": smoke,
        "fold": fold,
        "selected_candidate": result["selected_candidate"],
        "fixed_epochs_by_seed": result["fixed_epochs_by_seed"],
        "duration_seconds": time.time() - started,
        "stage_b": stage_b,
        "checkpoint_count": len(list((fold_dir / "outer_final").glob("seed_*/model_checkpoint.pt"))),
    }
    write_json(fold_dir / "canonical_fold_completion.json", summary)
    return summary


def validate_complete_fold(fold_dir: Path, expected_fold: int) -> dict[str, Any]:
    completion_path = fold_dir / "canonical_fold_completion.json"
    if not completion_path.exists():
        raise RuntimeError(f"Existing fold directory is incomplete: {fold_dir}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != "PASS" or completion.get("fold") != expected_fold:
        raise RuntimeError(f"Fold completion audit invalid: {fold_dir}")
    for seed in FINAL_SEEDS:
        seed_dir = fold_dir / "outer_final" / f"seed_{seed}"
        reload_audit = json.loads((seed_dir / "reload_audit.json").read_text(encoding="utf-8"))
        checkpoint_manifest = json.loads((seed_dir / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        checkpoint = seed_dir / "model_checkpoint.pt"
        if reload_audit.get("status") != "PASS":
            raise RuntimeError(f"Reload audit is not PASS: {seed_dir}")
        if sha256_file(checkpoint) != checkpoint_manifest["checkpoint"]["sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
    return completion


def run_full(run_dir: Path, device_name: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "run_progress.jsonl"
    for fold in range(10):
        fold_dir = run_dir / "folds" / f"fold_{fold:02d}"
        if fold_dir.exists():
            validate_complete_fold(fold_dir, fold)
            print(f"[full] fold {fold} already complete and verified", flush=True)
            continue
        event = {"timestamp": now_iso(), "fold": fold, "status": "STARTED"}
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        summary = run_fold(fold, run_dir, device_name, smoke=False)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": now_iso(), "fold": fold, "status": "PASS", "summary": clean(summary)}, ensure_ascii=False) + "\n")
        update_protocol_manifest(
            "TRAINING_RUNNING",
            completed_outer_folds=fold + 1,
            last_completed_fold=fold,
        )
    aggregate_full_run(run_dir)


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= PRIMARY_THRESHOLD).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "n": len(labels),
        "auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "threshold": PRIMARY_THRESHOLD,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def calibration_diagnostics(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    coefficient, intercept = optimization.fit_platt_newton(probabilities, labels)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, bins - 1)
    rows = []
    ece = 0.0
    for index in range(bins):
        mask = assignments == index
        if np.any(mask):
            mean_probability = float(np.mean(probabilities[mask]))
            observed_rate = float(np.mean(labels[mask]))
            count = int(np.sum(mask))
            ece += count / len(labels) * abs(mean_probability - observed_rate)
        else:
            mean_probability = observed_rate = None
            count = 0
        rows.append({
            "bin": index + 1,
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "n": count,
            "mean_probability": mean_probability,
            "observed_positive_rate": observed_rate,
        })
    return {
        "diagnostic_fit_scope": "entire OOF labels; descriptive only, not used to alter predictions",
        "calibration_slope": coefficient,
        "calibration_intercept": intercept,
        "expected_calibration_error_10_bin": float(ece),
        "bins": rows,
    }


def fold_bootstrap_ci(rows: Sequence[dict[str, Any]], metric: str) -> list[float]:
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED + sum(ord(char) for char in metric))
    draws = rng.integers(0, len(values), size=(BOOTSTRAP_REPLICATES, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate_full_run(run_dir: Path = CANONICAL_RUN_DIR) -> dict[str, Any]:
    features, _firewall, labels = load_canonical_inputs()
    n = len(features.subject_ids)
    raw = np.full(n, np.nan, dtype=np.float64)
    calibrated = np.full(n, np.nan, dtype=np.float64)
    folds = np.full(n, -1, dtype=np.int64)
    assignment = np.zeros(n, dtype=np.int64)
    fold_rows: list[dict[str, Any]] = []
    reload_rows: list[dict[str, Any]] = []
    selected_candidates: dict[str, str] = {}
    for fold in range(10):
        fold_dir = run_dir / "folds" / f"fold_{fold:02d}"
        validate_complete_fold(fold_dir, fold)
        split = np.load(fold_dir / "outer_final" / "seed_predictions.npz", allow_pickle=False)
        frozen = np.load(fold_dir / "frozen_outer_test_predictions.npz", allow_pickle=False)
        indices = frozen["row_indices"].astype(np.int64)
        seed_probabilities = split["seed_probabilities"].astype(np.float64)
        if not np.array_equal(split["base_seeds"], np.asarray(FINAL_SEEDS, dtype=np.int64)):
            raise RuntimeError(f"Fold {fold}: seed order changed")
        expected_raw = np.mean(seed_probabilities, axis=0)
        if not np.array_equal(expected_raw, split["raw_equal_weight_mean"]):
            raise RuntimeError(f"Fold {fold}: three-seed probability mean mismatch")
        if not np.array_equal(features.subject_ids[indices], frozen["subject_ids"]):
            raise RuntimeError(f"Fold {fold}: frozen subject ID mismatch")
        raw[indices] = expected_raw
        calibrated[indices] = split["final_probabilities"].astype(np.float64)
        folds[indices] = fold
        assignment[indices] += 1
        fold_metric = probability_metrics(labels[indices], expected_raw)
        fold_rows.append({"fold": fold, **fold_metric, "n_test": len(indices)})
        stage = json.loads((fold_dir / "stage_a_summary.json").read_text(encoding="utf-8"))
        selected_candidates[str(fold)] = stage["selected_candidate"]
        for seed in FINAL_SEEDS:
            row = json.loads(
                (fold_dir / "outer_final" / f"seed_{seed}" / "reload_audit.json").read_text(encoding="utf-8")
            )
            reload_rows.append(row)
    if not np.all(assignment == 1):
        raise RuntimeError(f"OOF assignment is not exactly once: {np.unique(assignment, return_counts=True)}")
    if not np.isfinite(raw).all() or not np.isfinite(calibrated).all():
        raise RuntimeError("OOF probabilities contain NaN or Inf")
    np.savez_compressed(
        OOF_PATH,
        subject_ids=features.subject_ids.astype(np.int64),
        row_indices=np.arange(n, dtype=np.int64),
        outer_folds=folds,
        labels=labels.astype(np.int64),
        raw_equal_weight_probability=raw,
        platt_calibrated_probability=calibrated,
        assignment_count=assignment,
    )
    write_csv(FOLD_METRICS_PATH, fold_rows)
    ci_metrics = {
        metric: {
            "mean_fold_metric": float(np.mean([row[metric] for row in fold_rows])),
            "fold_bootstrap_95_ci": fold_bootstrap_ci(fold_rows, metric),
            "bootstrap_unit": "outer_fold",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED + sum(ord(char) for char in metric),
        }
        for metric in ("auc", "accuracy", "balanced_accuracy", "brier")
    }
    historical = json.loads(HISTORICAL_METRICS_PATH.read_text(encoding="utf-8"))
    metrics = {
        "status": "PASS",
        "generated_at": now_iso(),
        "cohort_size": n,
        "oof_assignment_min": int(assignment.min()),
        "oof_assignment_max": int(assignment.max()),
        "subject_ids_sha256": sha256_array(features.subject_ids.astype(np.int64)),
        "primary_canonical_probability": "unweighted arithmetic mean of seed 42/43/44 probabilities",
        "canonical_raw_equal_weight": probability_metrics(labels, raw),
        "canonical_platt_calibrated": probability_metrics(labels, calibrated),
        "calibration": {
            "raw_probability_diagnostics": calibration_diagnostics(labels, raw),
            "platt_probability_diagnostics": calibration_diagnostics(labels, calibrated),
            "platt_models_fit_scope": "each outer fold's selected-candidate three-seed inner OOF predictions only",
        },
        "fold_level": {
            "rows": fold_rows,
            "confidence_intervals": ci_metrics,
        },
        "historical_context_not_canonical": {
            "old_optimized_raw_oof_auc": historical["raw_probability"]["auc"],
            "old_optimized_calibrated_oof_auc": historical["platt_calibrated_probability"]["auc"],
            "old_explanation_model_oof_auc_approx": 0.6891,
            "policy": "Neither historical value is reported as corrected canonical-v2 performance.",
        },
        "selected_candidates_by_outer_fold": selected_candidates,
        "oof_predictions_path": OOF_PATH,
        "oof_predictions_sha256": sha256_file(OOF_PATH),
    }
    write_json(METRICS_PATH, metrics)
    all_reload_pass = (
        len(reload_rows) == 30
        and all(row["status"] == "PASS" for row in reload_rows)
        and max(row["max_abs_probability_error"] for row in reload_rows) <= CHECKPOINT_RELOAD_ATOL
    )
    reload_audit = {
        "status": "PASS" if all_reload_pass else "FAIL",
        "generated_at": now_iso(),
        "expected_checkpoint_count": 30,
        "observed_checkpoint_count": len(reload_rows),
        "all_checkpoint_files_exist": all(Path(row["checkpoint_path"]).is_file() for row in reload_rows),
        "all_reload_checks_pass": all(row["status"] == "PASS" for row in reload_rows),
        "maximum_abs_probability_error": max(row["max_abs_probability_error"] for row in reload_rows),
        "maximum_abs_logit_error": max(row["max_abs_logit_error"] for row in reload_rows),
        "absolute_tolerance": CHECKPOINT_RELOAD_ATOL,
        "models": reload_rows,
    }
    write_json(CHECKPOINT_RELOAD_AUDIT_PATH, reload_audit)
    lineage_checks = {
        "phase1_gate_pass": phase1_gate()["status"] == "PASS",
        "ten_outer_folds_complete": len(fold_rows) == 10,
        "thirty_checkpoints_complete": len(reload_rows) == 30,
        "all_checkpoints_reload": all_reload_pass,
        "oof_subjects_covered_exactly_once": bool(np.all(assignment == 1)),
        "three_seed_equal_weight_verified_each_fold": True,
        "corrected_frequency_hash_used": canonical_input_hashes()[str(CORRECTED_FREQUENCY_PATH.resolve())] == EXPECTED_FREQUENCY_SHA256,
        "historical_frequency_not_used": all(
            "X_freq_filtered.npy" not in key for key in canonical_input_hashes()
        ),
    }
    lineage_audit = {
        "status": "PASS" if all(lineage_checks.values()) else "FAIL",
        "generated_at": now_iso(),
        "checks": lineage_checks,
        "input_sha256": canonical_input_hashes(),
        "selected_candidates_by_outer_fold": selected_candidates,
        "checkpoint_reload_audit_sha256": sha256_file(CHECKPOINT_RELOAD_AUDIT_PATH),
        "metrics_sha256": sha256_file(METRICS_PATH),
        "oof_predictions_sha256": sha256_file(OOF_PATH),
    }
    write_json(LINEAGE_AUDIT_PATH, lineage_audit)
    checkpoint_hashes = {
        str(Path(row["checkpoint_path"]).resolve()): row["checkpoint_file_sha256"]
        for row in reload_rows
    }
    update_protocol_manifest(
        "TRAINING_COMPLETE_XAI_PENDING",
        training_completed_at=now_iso(),
        canonical_training={
            "status": "PASS" if all_reload_pass else "FAIL",
            "selected_candidates_by_outer_fold": selected_candidates,
            "checkpoint_count": len(reload_rows),
            "checkpoint_sha256": checkpoint_hashes,
            "checkpoint_reload_audit": CHECKPOINT_RELOAD_AUDIT_PATH,
            "oof_predictions": OOF_PATH,
            "metrics": METRICS_PATH,
        },
        artifact_sha256={
            str(OOF_PATH.resolve()): sha256_file(OOF_PATH),
            str(METRICS_PATH.resolve()): sha256_file(METRICS_PATH),
            str(FOLD_METRICS_PATH.resolve()): sha256_file(FOLD_METRICS_PATH),
            str(CHECKPOINT_RELOAD_AUDIT_PATH.resolve()): sha256_file(CHECKPOINT_RELOAD_AUDIT_PATH),
            str(LINEAGE_AUDIT_PATH.resolve()): sha256_file(LINEAGE_AUDIT_PATH),
        },
    )
    print(
        json.dumps(
            {
                "status": lineage_audit["status"],
                "raw_oof_metrics": metrics["canonical_raw_equal_weight"],
                "calibrated_oof_metrics": metrics["canonical_platt_calibrated"],
                "checkpoint_reload_max_abs_error": reload_audit["maximum_abs_probability_error"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return metrics


def equal_weight_probability_mean(seed_probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(seed_probabilities, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] != 3:
        raise ValueError("Exactly three seed probability vectors are required")
    return np.mean(values, axis=0)


def aggregate_signed_attributions(seed_attributions: np.ndarray) -> np.ndarray:
    values = np.asarray(seed_attributions)
    if values.ndim < 2 or values.shape[0] != 3:
        raise ValueError("Exactly three signed seed attribution arrays are required")
    return np.mean(values, axis=0)


def backfill_selected_edges(
    selected_values: np.ndarray,
    selected_edges: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    values = np.asarray(selected_values)
    selected = np.asarray(selected_edges, dtype=np.int64)
    if values.shape[-1] != len(selected):
        raise ValueError("Selected attribution width does not match selected edges")
    output = np.full(values.shape[:-1] + (audited.N_FULL_EDGES,), fill_value, dtype=values.dtype)
    output[..., selected] = values
    return output


def validate_xai_checkpoint_provenance(provenance: dict[str, Any]) -> bool:
    checkpoint_path = Path(provenance["checkpoint_path"])
    if not checkpoint_path.is_file():
        return False
    return sha256_file(checkpoint_path) == provenance.get("checkpoint_sha256")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical model/XAI aligned v2 training pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("gate")
    sub.add_parser("initialize")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    smoke.add_argument("--run-dir", type=Path, default=SMOKE_RUN_DIR)
    fold = sub.add_parser("run-fold")
    fold.add_argument("--fold", type=int, required=True)
    fold.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    full = sub.add_parser("run-full")
    full.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    sub.add_parser("aggregate")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "gate":
        print(json.dumps(clean(phase1_gate()), ensure_ascii=False, indent=2))
        return 0
    if args.command == "initialize":
        print(json.dumps(clean(initialize_protocol_manifest()), ensure_ascii=False, indent=2))
        return 0
    if args.command == "smoke":
        initialize_protocol_manifest("SMOKE_RUNNING")
        summary = run_fold(0, args.run_dir.resolve(), args.device, smoke=True)
        update_protocol_manifest("SMOKE_TRAINING_PASS", smoke_training=summary)
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-fold":
        if not PROTOCOL_MANIFEST_PATH.exists():
            initialize_protocol_manifest("TRAINING_RUNNING")
        summary = run_fold(args.fold, CANONICAL_RUN_DIR, args.device, smoke=False)
        print(json.dumps(clean(summary), ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-full":
        if not PROTOCOL_MANIFEST_PATH.exists():
            initialize_protocol_manifest("TRAINING_RUNNING")
        else:
            update_protocol_manifest("TRAINING_RUNNING")
        run_full(CANONICAL_RUN_DIR, args.device)
        return 0
    if args.command == "aggregate":
        aggregate_full_run(CANONICAL_RUN_DIR)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

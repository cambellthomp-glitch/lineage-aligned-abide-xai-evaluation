"""Leakage-free optimization engine for the confirmed C-PAC protocol v1.

The module intentionally separates:

* outer-train-only optimization and prediction generation (Stage A), and
* label-bearing outer evaluation after prediction freezing (Stage B).

No function in the optimization or training path accepts outer-test labels or
outer-test metrics.  The command-line entry point runs one outer fold at a
time.  Full training is never started merely by importing this module.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import cpac_leakage_free_nested_cv as audited
import cpac_strict_multimodal_ablation as ablation
import cpac_strict_same_protocol_baselines as baselines
import strict_nested_cv_interpretable_model as strict


ROOT = Path(__file__).resolve().parent
ENGINE_ROOT = ROOT / "cpac_model_optimization_v1"
PROTOCOL_PATH = ENGINE_ROOT / "optimization_protocol_draft.json"
CANDIDATE_TABLE_PATH = ENGINE_ROOT / "candidate_configuration_table.csv"
PROTECTED_BASELINE = (
    ROOT
    / "cpac_leakage_free_nested_cv"
    / "final_seed42_20260717_124500"
).resolve()
OUTER_SPLIT_SEED = 42
LEVEL1_SEED = 42
LEVEL2_SEEDS = (42, 43, 44)
EVALUATION_INTERVAL = 5


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return audited.sha256_array(np.asarray(values))


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def assert_output_path_safe(path: str | Path) -> Path:
    """Reject the protected baseline, its descendants, and its ancestors."""
    output = Path(path).resolve()
    if (
        output == PROTECTED_BASELINE
        or _is_relative_to(output, PROTECTED_BASELINE)
        or _is_relative_to(PROTECTED_BASELINE, output)
    ):
        raise RuntimeError(
            f"Optimization output path conflicts with protected baseline: {output}"
        )
    return output


FORBIDDEN_TRAINING_INPUT_TOKENS = (
    "frozen_oof",
    "outer_test_prob",
    "outer_metrics",
    "final_metrics",
    "stage_b",
    "offline_oof",
    "ensemble",
    "blend",
)


def assert_training_input_allowed(path: str | Path) -> None:
    """Extend the audited loader firewall for optimization-time inputs."""
    candidate = Path(path)
    audited.assert_input_path_allowed(candidate)
    resolved = candidate.resolve()
    if _is_relative_to(resolved, PROTECTED_BASELINE):
        raise RuntimeError(
            f"Protected baseline artifact cannot be a training input: {resolved}"
        )
    name = candidate.name.casefold()
    if any(token in name for token in FORBIDDEN_TRAINING_INPUT_TOKENS):
        raise RuntimeError(f"Forbidden optimization training input: {candidate}")
    if "oof" in name and candidate.suffix.casefold() in {
        ".npy",
        ".npz",
        ".json",
        ".csv",
    }:
        raise RuntimeError(f"Forbidden OOF training input: {candidate}")


def load_audited_inputs(
    data_directory: str | Path,
) -> tuple[audited.FeatureBundle, audited.LabelFirewall, list[str]]:
    """Load only the audited current inputs and return an access ledger."""
    accessed: list[str] = []
    original = audited.safe_load_array

    def tracked(
        path: str | Path,
        *,
        mmap_mode: Optional[str] = None,
        allow_pickle: bool = False,
    ) -> np.ndarray:
        assert_training_input_allowed(path)
        accessed.append(str(Path(path).resolve()))
        return original(
            path,
            mmap_mode=mmap_mode,
            allow_pickle=allow_pickle,
        )

    audited.safe_load_array = tracked
    try:
        features = audited.load_feature_bundle(data_directory)
        firewall = audited.load_label_firewall(data_directory)
    finally:
        audited.safe_load_array = original
    allowed_names = {
        audited.FULL_FC_FILENAME.casefold(),
        audited.FREQUENCY_FILENAME.casefold(),
        audited.SUBJECT_IDS_FILENAME.casefold(),
        audited.LABEL_FILENAME.casefold(),
    }
    unexpected = [
        value for value in accessed if Path(value).name.casefold() not in allowed_names
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected array inputs: {unexpected}")
    return features, firewall, accessed


@dataclass(frozen=True)
class OptimizationCandidate:
    candidate_id: str
    family: str
    fc_count: int
    use_frequency: bool
    use_demographics: bool
    frequency_pooling: str
    contrastive_weight: float
    regularization: str
    dropout_sc_or_encoder: float
    dropout_cls: float
    weight_decay: float
    residual_blocks: int
    loss_recipe: str
    ocread_eval_mode: str
    trainable_parameters: int

    def validate(self) -> None:
        if self.regularization not in {"base", "strong"}:
            raise ValueError(f"Invalid regularization: {self.regularization}")
        if self.fc_count not in {1000, 2500, 4975}:
            raise ValueError(f"Invalid FC count: {self.fc_count}")
        if self.use_frequency and self.frequency_pooling != "ocread":
            raise ValueError("Frequency candidates must use OCREAD in protocol v1")
        if self.use_frequency and self.ocread_eval_mode != "mu":
            raise ValueError("OCREAD evaluation must be deterministic mu")
        if self.trainable_parameters < 1:
            raise ValueError("Parameter count must be positive")


def load_candidate_registry(
    protocol_path: Path = PROTOCOL_PATH,
    table_path: Path = CANDIDATE_TABLE_PATH,
) -> tuple[OptimizationCandidate, ...]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    with table_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["candidate_id"]: row for row in rows}
    candidates: list[OptimizationCandidate] = []
    for item in protocol["candidates"]:
        row = by_id.get(item["candidate_id"])
        if row is None:
            raise RuntimeError(f"Candidate missing from CSV: {item['candidate_id']}")
        regularization = protocol["regularization_levels"][item["regularization"]]
        candidate = OptimizationCandidate(
            candidate_id=item["candidate_id"],
            family=item["family"],
            fc_count=int(item["fc_count"]),
            use_frequency=bool(item["use_frequency"]),
            use_demographics=bool(item["use_demographics"]),
            frequency_pooling=item["frequency_pooling"],
            contrastive_weight=float(item["contrastive_weight"]),
            regularization=item["regularization"],
            dropout_sc_or_encoder=float(
                regularization["dropout_sc_or_encoder"]
            ),
            dropout_cls=float(regularization["dropout_cls"]),
            weight_decay=float(regularization["weight_decay"]),
            residual_blocks=int(item["residual_blocks"]),
            loss_recipe=row["loss_recipe"],
            ocread_eval_mode=item["ocread_eval_mode"],
            trainable_parameters=int(item["trainable_parameters"]),
        )
        candidate.validate()
        if int(row["fc_count"]) != candidate.fc_count:
            raise RuntimeError(f"FC count mismatch for {candidate.candidate_id}")
        if int(row["trainable_parameters"]) != candidate.trainable_parameters:
            raise RuntimeError(
                f"Parameter count mismatch for {candidate.candidate_id}"
            )
        candidates.append(candidate)
    ids = [candidate.candidate_id for candidate in candidates]
    if len(candidates) != protocol["candidate_count"] or len(candidates) != 12:
        raise RuntimeError("Confirmed protocol must contain exactly 12 candidates")
    if len(set(ids)) != len(ids) or set(ids) != set(by_id):
        raise RuntimeError("Candidate IDs are not one-to-one across JSON and CSV")
    return tuple(candidates)


CANDIDATES = load_candidate_registry()
CANDIDATE_BY_ID = {candidate.candidate_id: candidate for candidate in CANDIDATES}
CANDIDATE_REGISTRY_SHA256 = canonical_json_sha256(
    [asdict(candidate) for candidate in CANDIDATES]
)


@dataclass(frozen=True)
class EngineSchedule:
    level1_rung1_epoch: int = 80
    level1_max_epoch: int = 160
    level1_minimum_epoch: int = 100
    level1_patience: int = 40
    level1_rung1_survivors: int = 6
    level1_final_survivors: int = 2
    level2_max_epoch: int = 300
    level2_minimum_epoch: int = 120
    level2_patience: int = 60
    evaluation_interval: int = EVALUATION_INTERVAL

    def validate(self, n_candidates: int) -> None:
        if not (
            1
            <= self.level1_rung1_epoch
            <= self.level1_max_epoch
            <= self.level2_max_epoch
        ):
            raise ValueError("Invalid epoch budgets")
        if not (
            2
            <= self.level1_final_survivors
            <= self.level1_rung1_survivors
            <= n_candidates
        ):
            raise ValueError("Invalid survivor counts")
        if self.evaluation_interval < 1:
            raise ValueError("Evaluation interval must be positive")


@dataclass
class OptimizationPreparedPartition:
    candidate_id: str
    trial_scope: str
    fit_indices: np.ndarray
    evaluation_indices: np.ndarray
    selected_edges: np.ndarray
    f_scores: np.ndarray
    scaler_fc: StandardScaler
    scaler_frequency: Optional[StandardScaler]
    demographic_transformer: Optional[audited.DemoTransformer]
    anchors: np.ndarray
    anchor_profiles: np.ndarray
    x_fc_fit: np.ndarray
    x_frequency_fit: np.ndarray
    x_demographic_fit: np.ndarray
    x_fc_evaluation: np.ndarray
    x_frequency_evaluation: np.ndarray
    x_demographic_evaluation: np.ndarray
    state_token: str = field(default_factory=lambda: uuid.uuid4().hex)


def fit_candidate_preprocessing(
    *,
    features: audited.FeatureBundle,
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    y_fit: np.ndarray,
    candidate: OptimizationCandidate,
    yeo_nodes: Dict[int, np.ndarray],
    outer_test_indices: np.ndarray,
    trial_scope: str,
    hidden_dim: int = 80,
    anchor_projection_seed: int = 20250225,
) -> OptimizationPreparedPartition:
    """Fit every candidate transform on the supplied fit indices only."""
    fit_idx = np.asarray(fit_indices, dtype=np.int64).reshape(-1)
    eval_idx = np.asarray(evaluation_indices, dtype=np.int64).reshape(-1)
    labels = np.asarray(y_fit, dtype=np.int64).reshape(-1)
    outer_test = np.asarray(outer_test_indices, dtype=np.int64).reshape(-1)
    if len(fit_idx) != len(labels):
        raise ValueError("y_fit must correspond exactly to fit_indices")
    if len(np.intersect1d(fit_idx, eval_idx)):
        raise RuntimeError("Preprocessing fit/evaluation overlap")
    if len(np.intersect1d(fit_idx, outer_test)):
        raise RuntimeError("Outer-test index entered preprocessing fit")
    if len(np.setdiff1d(np.concatenate([fit_idx, eval_idx]), np.arange(len(features.subject_ids)))):
        raise IndexError("Preprocessing indices are out of range")

    selected, f_scores = audited.select_fc_edges(
        features.full_fc[fit_idx],
        labels,
        count=candidate.fc_count,
    )
    scaler_fc = StandardScaler().fit(
        np.asarray(features.full_fc[fit_idx][:, selected], dtype=np.float32)
    )
    x_fc_fit = scaler_fc.transform(
        np.asarray(features.full_fc[fit_idx][:, selected], dtype=np.float32)
    ).astype(np.float32)
    x_fc_eval = scaler_fc.transform(
        np.asarray(features.full_fc[eval_idx][:, selected], dtype=np.float32)
    ).astype(np.float32)

    if candidate.use_frequency:
        scaler_frequency: Optional[StandardScaler] = StandardScaler().fit(
            features.frequency[fit_idx]
        )
        x_frequency_fit = scaler_frequency.transform(
            features.frequency[fit_idx]
        ).astype(np.float32)
        x_frequency_eval = scaler_frequency.transform(
            features.frequency[eval_idx]
        ).astype(np.float32)
        anchors, profiles = audited.build_frequency_anchors(
            x_frequency_fit,
            yeo_nodes,
            hidden_dim,
            anchor_projection_seed,
        )
    else:
        scaler_frequency = None
        x_frequency_fit = np.zeros(
            (len(fit_idx), audited.N_FREQUENCY), dtype=np.float32
        )
        x_frequency_eval = np.zeros(
            (len(eval_idx), audited.N_FREQUENCY), dtype=np.float32
        )
        anchors = np.zeros((7, hidden_dim), dtype=np.float32)
        profiles = np.zeros((7, audited.N_FREQUENCY_BINS), dtype=np.float32)

    if candidate.use_demographics:
        demographic_transformer: Optional[audited.DemoTransformer] = (
            audited.DemoTransformer.fit(features.demographics_raw[fit_idx])
        )
        x_demographic_fit = demographic_transformer.transform(
            features.demographics_raw[fit_idx]
        )
        x_demographic_eval = demographic_transformer.transform(
            features.demographics_raw[eval_idx]
        )
    else:
        demographic_transformer = None
        x_demographic_fit = np.zeros(
            (len(fit_idx), audited.N_DEMOGRAPHIC), dtype=np.float32
        )
        x_demographic_eval = np.zeros(
            (len(eval_idx), audited.N_DEMOGRAPHIC), dtype=np.float32
        )

    return OptimizationPreparedPartition(
        candidate_id=candidate.candidate_id,
        trial_scope=trial_scope,
        fit_indices=fit_idx.copy(),
        evaluation_indices=eval_idx.copy(),
        selected_edges=selected,
        f_scores=f_scores,
        scaler_fc=scaler_fc,
        scaler_frequency=scaler_frequency,
        demographic_transformer=demographic_transformer,
        anchors=anchors,
        anchor_profiles=profiles,
        x_fc_fit=x_fc_fit,
        x_frequency_fit=x_frequency_fit,
        x_demographic_fit=x_demographic_fit,
        x_fc_evaluation=x_fc_eval,
        x_frequency_evaluation=x_frequency_eval,
        x_demographic_evaluation=x_demographic_eval,
    )


def save_preprocessing_artifacts(
    directory: Path,
    prepared: OptimizationPreparedPartition,
    outer_train_indices: np.ndarray,
    outer_test_indices: np.ndarray,
) -> Dict[str, Any]:
    outer_train = np.asarray(outer_train_indices, dtype=np.int64)
    outer_test = np.asarray(outer_test_indices, dtype=np.int64)
    if len(np.setdiff1d(prepared.fit_indices, outer_train)):
        raise RuntimeError("Preprocessing fit indices escaped outer train")
    if len(np.intersect1d(prepared.fit_indices, outer_test)):
        raise RuntimeError("Preprocessing fit indices overlap outer test")
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / "fit_indices.npy", prepared.fit_indices)
    np.save(directory / "evaluation_indices.npy", prepared.evaluation_indices)
    np.save(directory / "selected_full_edge_indices.npy", prepared.selected_edges)
    manifest = {
        "candidate_id": prepared.candidate_id,
        "trial_scope": prepared.trial_scope,
        "state_token": prepared.state_token,
        "fit_indices_sha256": audited.sha256_indices(prepared.fit_indices),
        "evaluation_indices_sha256": audited.sha256_indices(
            prepared.evaluation_indices
        ),
        "selected_edges_sha256": audited.sha256_indices(
            prepared.selected_edges
        ),
        "fc_input_dimension": audited.N_FULL_EDGES,
        "fc_selected_dimension": int(len(prepared.selected_edges)),
        "fit_is_subset_of_outer_train": True,
        "fit_outer_test_overlap": 0,
        "selector_fit_scope": "fit_indices_only",
        "fc_scaler_fit_scope": "fit_indices_only",
        "frequency_scaler_fit_scope": (
            "fit_indices_only"
            if prepared.scaler_frequency is not None
            else "not_applicable"
        ),
        "anchors_fit_scope": (
            "fit_indices_only_after_fit_only_frequency_scaling"
            if prepared.scaler_frequency is not None
            else "not_applicable"
        ),
        "demographics_fit_scope": (
            "fit_indices_only"
            if prepared.demographic_transformer is not None
            else "not_applicable"
        ),
    }
    write_json_new(directory / "preprocessing_manifest.json", manifest)
    return manifest


@dataclass
class InnerTrialResult:
    candidate_id: str
    base_seed: int
    inner_fold: int
    validation_indices: np.ndarray
    probabilities: np.ndarray
    best_epoch: int
    best_validation_auc: float
    stopped_epoch: int
    checkpoint_state_sha256: str
    model_instance_token: str
    optimizer_instance_token: str
    preprocessing_state_token: str
    continuation: Any = None


@dataclass
class OuterFitResult:
    candidate_id: str
    base_seed: int
    fixed_epoch: int
    probabilities: np.ndarray
    checkpoint_state_sha256: str
    model_instance_token: str
    optimizer_instance_token: str
    preprocessing_state_token: str


class OptimizationBackend(Protocol):
    def run_inner(
        self,
        *,
        candidate: OptimizationCandidate,
        prepared: OptimizationPreparedPartition,
        y_fit: np.ndarray,
        y_validation: np.ndarray,
        base_seed: int,
        actual_seed: int,
        outer_fold: int,
        inner_fold: int,
        budget_epoch: int,
        planned_max_epoch: int,
        minimum_epoch: Optional[int],
        patience: Optional[int],
        evaluation_interval: int,
        continuation: Any = None,
    ) -> InnerTrialResult: ...

    def run_outer(
        self,
        *,
        candidate: OptimizationCandidate,
        prepared: OptimizationPreparedPartition,
        y_fit: np.ndarray,
        base_seed: int,
        actual_seed: int,
        outer_fold: int,
        fixed_epoch: int,
    ) -> OuterFitResult: ...


def checkpoint_auc_improved(
    current_auc: float,
    best_auc: float,
    *,
    tolerance: float = 1e-12,
) -> bool:
    """Checkpoint comparison uses AUC only; exact ties keep the earlier epoch."""
    return bool(current_auc > best_auc + tolerance)


@dataclass
class TorchContinuation:
    candidate_id: str
    prepared_state_token: str
    actual_seed: int
    planned_max_epoch: int
    current_epoch: int
    best_epoch: int
    best_auc: float
    best_probabilities: np.ndarray
    best_model_state: Dict[str, torch.Tensor]
    current_model_state: Dict[str, torch.Tensor]
    optimizer_state: Dict[str, Any]
    scheduler_state: Dict[str, Any]
    loader_generator_state: torch.Tensor
    python_random_state: object
    numpy_random_state: tuple[Any, ...]
    torch_rng_state: torch.Tensor
    cuda_rng_state: Optional[list[torch.Tensor]]
    model_instance_token: str
    optimizer_instance_token: str
    last_improved_epoch: int


def _state_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_state_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


class TorchOptimizationBackend:
    """Real backend; unit tests substitute a no-training deterministic backend."""

    def __init__(self, device: torch.device):
        self.device = device

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    @staticmethod
    def _variant(candidate: OptimizationCandidate) -> ablation.Variant:
        if candidate.family == "fc_demographics":
            return ablation.Variant(
                candidate.candidate_id,
                candidate.family,
                False,
                True,
                "none",
                candidate.contrastive_weight,
            )
        if candidate.family in {
            "full_multimodal_no_contrastive",
            "current_full_multimodal",
        }:
            return ablation.Variant(
                candidate.candidate_id,
                candidate.family,
                True,
                True,
                "ocread",
                candidate.contrastive_weight,
            )
        raise KeyError(candidate.family)

    def _make_model(
        self,
        candidate: OptimizationCandidate,
        prepared: OptimizationPreparedPartition,
    ) -> torch.nn.Module:
        if candidate.family == "fc_only_compact":
            model: torch.nn.Module = baselines.FCCompactDeep(
                candidate.fc_count,
                160,
                candidate.dropout_sc_or_encoder,
                candidate.residual_blocks,
            )
        else:
            regularization = strict.Candidate(
                candidate.candidate_id,
                candidate.dropout_sc_or_encoder,
                candidate.dropout_cls,
                candidate.weight_decay,
                0.30,
            )
            model = ablation.StrictAblationModel(
                self._variant(candidate),
                regularization,
                torch.from_numpy(prepared.anchors),
                hidden_dim=80,
                align_dim=160,
                residual_blocks=candidate.residual_blocks,
            )
            model.set_eval_ocread_mode("mu")
        model = model.to(self.device)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        if parameters != candidate.trainable_parameters:
            raise RuntimeError(
                f"{candidate.candidate_id} parameter count changed: "
                f"{parameters} != {candidate.trainable_parameters}"
            )
        return model

    @staticmethod
    def _batch_size(candidate: OptimizationCandidate) -> int:
        return 32 if candidate.family == "fc_only_compact" else 16

    def _make_optimizer_scheduler(
        self,
        model: torch.nn.Module,
        candidate: OptimizationCandidate,
        planned_max_epoch: int,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0008,
            weight_decay=candidate.weight_decay,
        )
        if candidate.family == "fc_only_compact":
            t_max = max(planned_max_epoch, 1)
        else:
            t_max = max(planned_max_epoch - 60, 1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max
        )
        return optimizer, scheduler

    def _loader(
        self,
        prepared: OptimizationPreparedPartition,
        labels: np.ndarray,
        candidate: OptimizationCandidate,
        generator: torch.Generator,
    ) -> DataLoader:
        return DataLoader(
            TensorDataset(
                torch.from_numpy(prepared.x_fc_fit),
                torch.from_numpy(prepared.x_frequency_fit),
                torch.from_numpy(prepared.x_demographic_fit),
                torch.from_numpy(np.asarray(labels, dtype=np.float32)),
            ),
            batch_size=self._batch_size(candidate),
            shuffle=True,
            drop_last=False,
            generator=generator,
        )

    def _predict(
        self,
        model: torch.nn.Module,
        candidate: OptimizationCandidate,
        prepared: OptimizationPreparedPartition,
    ) -> np.ndarray:
        model.eval()
        if hasattr(model, "set_eval_ocread_mode"):
            model.set_eval_ocread_mode("mu")
        output: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(prepared.x_fc_evaluation), 64):
                stop = min(start + 64, len(prepared.x_fc_evaluation))
                x_fc = torch.from_numpy(
                    prepared.x_fc_evaluation[start:stop]
                ).to(self.device)
                if candidate.family == "fc_only_compact":
                    logits = model(x_fc)
                else:
                    logits, _, _, _ = model(
                        x_fc,
                        torch.from_numpy(
                            prepared.x_frequency_evaluation[start:stop]
                        ).to(self.device),
                        torch.from_numpy(
                            prepared.x_demographic_evaluation[start:stop]
                        ).to(self.device),
                        tau=0.15,
                        training=False,
                    )
                output.append(torch.sigmoid(logits).cpu().numpy())
        probabilities = np.concatenate(output).astype(np.float64)
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError("Prediction contains NaN/Inf")
        return probabilities

    def _train_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        candidate: OptimizationCandidate,
        epoch: int,
    ) -> None:
        if candidate.family != "fc_only_compact" and epoch <= 60:
            for group in optimizer.param_groups:
                group["lr"] = 0.0008 * epoch / 60
        tau = 1.0 if epoch <= 100 else 0.15
        model.train()
        for x_fc, x_frequency, x_demo, labels in loader:
            x_fc = x_fc.to(self.device)
            x_frequency = x_frequency.to(self.device)
            x_demo = x_demo.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            if candidate.family == "fc_only_compact":
                logits = model(x_fc)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            else:
                use_mixup = epoch > 60
                if use_mixup:
                    mixing = float(np.random.beta(0.10, 0.10))
                    permutation = torch.randperm(len(labels), device=self.device)
                    input_fc = (
                        mixing * x_fc + (1.0 - mixing) * x_fc[permutation]
                    )
                    input_frequency = (
                        mixing * x_frequency
                        + (1.0 - mixing) * x_frequency[permutation]
                    )
                    input_demo = (
                        mixing * x_demo
                        + (1.0 - mixing) * x_demo[permutation]
                    )
                    label_a, label_b = labels, labels[permutation]
                else:
                    mixing = 1.0
                    input_fc, input_frequency, input_demo = (
                        x_fc,
                        x_frequency,
                        x_demo,
                    )
                    label_a = label_b = labels
                logits, fused, mu, logvar = model(
                    input_fc,
                    input_frequency,
                    input_demo,
                    tau=tau,
                    training=True,
                    noise_std=0.01,
                )
                bce_a = F.binary_cross_entropy_with_logits(
                    logits, label_a, reduction="none"
                )
                bce_b = F.binary_cross_entropy_with_logits(
                    logits, label_b, reduction="none"
                )
                weight_a = (1.0 - torch.exp(-bce_a)).pow(1.0)
                weight_b = (1.0 - torch.exp(-bce_b)).pow(1.0)
                classification = (
                    mixing * (weight_a * bce_a).mean()
                    + (1.0 - mixing) * (weight_b * bce_b).mean()
                )
                if (
                    mu is not None
                    and logvar is not None
                    and getattr(model, "ocread", None) is not None
                ):
                    kl = (
                        -0.5
                        * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                        * 0.002
                    )
                    anchor = (
                        F.mse_loss(mu, model.ocread.initial_anchors) * 0.30
                    )
                else:
                    kl = torch.zeros((), device=self.device)
                    anchor = torch.zeros((), device=self.device)
                if candidate.contrastive_weight > 0 and not use_mixup:
                    contrastive = (
                        ablation.legacy_contrastive_loss(
                            fused, labels, temperature=0.10
                        )
                        * candidate.contrastive_weight
                    )
                else:
                    contrastive = torch.zeros((), device=self.device)
                loss = classification + kl + anchor + contrastive
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    def run_inner(
        self,
        *,
        candidate: OptimizationCandidate,
        prepared: OptimizationPreparedPartition,
        y_fit: np.ndarray,
        y_validation: np.ndarray,
        base_seed: int,
        actual_seed: int,
        outer_fold: int,
        inner_fold: int,
        budget_epoch: int,
        planned_max_epoch: int,
        minimum_epoch: Optional[int],
        patience: Optional[int],
        evaluation_interval: int,
        continuation: Optional[TorchContinuation] = None,
    ) -> InnerTrialResult:
        del outer_fold
        if continuation is None:
            self._set_seed(actual_seed)
            model = self._make_model(candidate, prepared)
            optimizer, scheduler = self._make_optimizer_scheduler(
                model, candidate, planned_max_epoch
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(actual_seed)
            current_epoch = 0
            best_epoch = 0
            best_auc = -float("inf")
            best_probabilities = np.empty(0, dtype=np.float64)
            best_model_state: Dict[str, torch.Tensor] = {}
            last_improved = 0
            model_token = f"torch-model-{uuid.uuid4().hex}"
            optimizer_token = f"torch-optimizer-{uuid.uuid4().hex}"
        else:
            if (
                continuation.candidate_id != candidate.candidate_id
                or continuation.prepared_state_token != prepared.state_token
                or continuation.actual_seed != actual_seed
                or continuation.planned_max_epoch != planned_max_epoch
            ):
                raise RuntimeError("Continuation state does not match inner trial")
            # Reconstructing a model consumes RNG during parameter
            # initialization.  Restore the serialized RNG states only after
            # reconstruction/loading so resumed training is identical to an
            # uninterrupted continuation.
            model = self._make_model(candidate, prepared)
            model.load_state_dict(continuation.current_model_state)
            optimizer, scheduler = self._make_optimizer_scheduler(
                model, candidate, planned_max_epoch
            )
            optimizer.load_state_dict(continuation.optimizer_state)
            _optimizer_to_device(optimizer, self.device)
            scheduler.load_state_dict(continuation.scheduler_state)
            generator = torch.Generator(device="cpu")
            generator.set_state(continuation.loader_generator_state)
            random.setstate(continuation.python_random_state)
            np.random.set_state(continuation.numpy_random_state)
            torch.set_rng_state(continuation.torch_rng_state)
            if torch.cuda.is_available() and continuation.cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(continuation.cuda_rng_state)
            current_epoch = continuation.current_epoch
            best_epoch = continuation.best_epoch
            best_auc = continuation.best_auc
            best_probabilities = continuation.best_probabilities.copy()
            best_model_state = _state_to_cpu(continuation.best_model_state)
            last_improved = continuation.last_improved_epoch
            model_token = continuation.model_instance_token
            optimizer_token = continuation.optimizer_instance_token
        if budget_epoch <= current_epoch or budget_epoch > planned_max_epoch:
            raise ValueError("Inner budget must extend the current valid schedule")

        loader = self._loader(prepared, y_fit, candidate, generator)
        stopped_epoch = budget_epoch
        for epoch in range(current_epoch + 1, budget_epoch + 1):
            self._train_epoch(model, optimizer, loader, candidate, epoch)
            if candidate.family == "fc_only_compact" or epoch > 60:
                scheduler.step()
            if epoch % evaluation_interval == 0 or epoch == budget_epoch:
                probabilities = self._predict(model, candidate, prepared)
                validation_auc = audited.safe_auc(
                    y_validation, probabilities
                )
                if checkpoint_auc_improved(validation_auc, best_auc):
                    best_auc = validation_auc
                    best_epoch = epoch
                    best_probabilities = probabilities.copy()
                    best_model_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    last_improved = epoch
                if (
                    minimum_epoch is not None
                    and patience is not None
                    and epoch >= minimum_epoch
                    and last_improved > 0
                    and epoch - last_improved >= patience
                ):
                    stopped_epoch = epoch
                    break
        if best_epoch < 1 or not len(best_probabilities):
            raise RuntimeError("No AUC-selected inner checkpoint")

        resume = TorchContinuation(
            candidate_id=candidate.candidate_id,
            prepared_state_token=prepared.state_token,
            actual_seed=actual_seed,
            planned_max_epoch=planned_max_epoch,
            current_epoch=stopped_epoch,
            best_epoch=best_epoch,
            best_auc=best_auc,
            best_probabilities=best_probabilities.copy(),
            best_model_state=_state_to_cpu(best_model_state),
            current_model_state={
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            optimizer_state=_state_to_cpu(optimizer.state_dict()),
            scheduler_state=_state_to_cpu(scheduler.state_dict()),
            loader_generator_state=generator.get_state().clone(),
            python_random_state=random.getstate(),
            numpy_random_state=np.random.get_state(),
            torch_rng_state=torch.get_rng_state().clone(),
            cuda_rng_state=(
                [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            ),
            model_instance_token=model_token,
            optimizer_instance_token=optimizer_token,
            last_improved_epoch=last_improved,
        )
        checkpoint_sha = audited.state_dict_sha256(best_model_state)
        del model, optimizer, scheduler, loader
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return InnerTrialResult(
            candidate_id=candidate.candidate_id,
            base_seed=base_seed,
            inner_fold=inner_fold,
            validation_indices=prepared.evaluation_indices.copy(),
            probabilities=best_probabilities,
            best_epoch=best_epoch,
            best_validation_auc=best_auc,
            stopped_epoch=stopped_epoch,
            checkpoint_state_sha256=checkpoint_sha,
            model_instance_token=model_token,
            optimizer_instance_token=optimizer_token,
            preprocessing_state_token=prepared.state_token,
            continuation=resume,
        )

    def run_outer(
        self,
        *,
        candidate: OptimizationCandidate,
        prepared: OptimizationPreparedPartition,
        y_fit: np.ndarray,
        base_seed: int,
        actual_seed: int,
        outer_fold: int,
        fixed_epoch: int,
    ) -> OuterFitResult:
        del outer_fold
        self._set_seed(actual_seed)
        model = self._make_model(candidate, prepared)
        optimizer, scheduler = self._make_optimizer_scheduler(
            model, candidate, fixed_epoch
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(actual_seed)
        loader = self._loader(prepared, y_fit, candidate, generator)
        model_token = f"torch-model-{uuid.uuid4().hex}"
        optimizer_token = f"torch-optimizer-{uuid.uuid4().hex}"
        for epoch in range(1, fixed_epoch + 1):
            self._train_epoch(model, optimizer, loader, candidate, epoch)
            if candidate.family == "fc_only_compact" or epoch > 60:
                scheduler.step()
        # This is the single model prediction call for this outer seed.
        probabilities = self._predict(model, candidate, prepared)
        model_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        result = OuterFitResult(
            candidate_id=candidate.candidate_id,
            base_seed=base_seed,
            fixed_epoch=fixed_epoch,
            probabilities=probabilities,
            checkpoint_state_sha256=audited.state_dict_sha256(model_state),
            model_instance_token=model_token,
            optimizer_instance_token=optimizer_token,
            preprocessing_state_token=prepared.state_token,
        )
        del model, optimizer, scheduler, loader
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    auc: float
    brier: float
    trainable_parameters: int


@dataclass
class CandidateSeedEvaluation:
    candidate_id: str
    base_seed: int
    oof_indices: np.ndarray
    oof_probabilities: np.ndarray
    auc: float
    brier: float
    best_epochs: tuple[int, int, int]
    checkpoint_sha256: tuple[str, str, str]


def concatenate_inner_oof(
    *,
    candidate: OptimizationCandidate,
    base_seed: int,
    trials: Sequence[InnerTrialResult],
    outer_train_indices: np.ndarray,
    y_outer_train: np.ndarray,
) -> CandidateSeedEvaluation:
    outer_train = np.asarray(outer_train_indices, dtype=np.int64).reshape(-1)
    labels = np.asarray(y_outer_train, dtype=np.int64).reshape(-1)
    if len(trials) != audited.N_INNER:
        raise RuntimeError("Exactly three inner trials are required")
    positions = {int(index): pos for pos, index in enumerate(outer_train)}
    probabilities = np.full(len(outer_train), np.nan, dtype=np.float64)
    assignment = np.zeros(len(outer_train), dtype=np.int64)
    for trial in trials:
        if (
            trial.candidate_id != candidate.candidate_id
            or trial.base_seed != base_seed
        ):
            raise RuntimeError("Inner trial identity mismatch")
        if len(trial.validation_indices) != len(trial.probabilities):
            raise RuntimeError("Validation indices and probabilities differ")
        for index, probability in zip(
            trial.validation_indices, trial.probabilities
        ):
            position = positions.get(int(index))
            if position is None:
                raise RuntimeError("Inner OOF index escaped outer train")
            probabilities[position] = float(probability)
            assignment[position] += 1
    if not np.all(assignment == 1) or not np.all(np.isfinite(probabilities)):
        raise RuntimeError("Inner OOF coverage must be exactly once and finite")
    return CandidateSeedEvaluation(
        candidate_id=candidate.candidate_id,
        base_seed=base_seed,
        oof_indices=outer_train.copy(),
        oof_probabilities=probabilities,
        auc=audited.safe_auc(labels, probabilities),
        brier=float(brier_score_loss(labels, probabilities)),
        best_epochs=tuple(int(trial.best_epoch) for trial in trials),
        checkpoint_sha256=tuple(
            trial.checkpoint_state_sha256 for trial in trials
        ),
    )


def rank_candidate_scores(
    scores: Sequence[CandidateScore],
    *,
    auc_indifference: float = 0.005,
    brier_tolerance: float = 1e-12,
) -> list[CandidateScore]:
    """Create the confirmed deterministic AUC-tier candidate ordering."""
    if not scores:
        raise ValueError("Cannot rank an empty score set")
    remaining = list(scores)
    if len({score.candidate_id for score in remaining}) != len(remaining):
        raise ValueError("Candidate scores must have unique IDs")
    ordered: list[CandidateScore] = []
    while remaining:
        maximum_auc = max(score.auc for score in remaining)
        tier = [
            score
            for score in remaining
            if maximum_auc - score.auc < auc_indifference
        ]

        def tier_key(score: CandidateScore) -> tuple[float, int, str]:
            # Rounding only defines exact numerical Brier ties; it does not
            # introduce a Brier indifference region.
            normalized_brier = round(score.brier / brier_tolerance) * brier_tolerance
            return (
                normalized_brier,
                score.trainable_parameters,
                score.candidate_id,
            )

        tier.sort(key=tier_key)
        ordered.extend(tier)
        tier_ids = {score.candidate_id for score in tier}
        remaining = [
            score for score in remaining if score.candidate_id not in tier_ids
        ]
    return ordered


@dataclass(frozen=True)
class CalibrationModel:
    mode: str
    coefficient: float
    intercept: float
    selected_secondary_threshold: float
    fit_indices_sha256: str
    fit_indices_are_outer_train: bool
    fit_outer_test_overlap: int
    fixed_primary_threshold: float = 0.5
    secondary_metric: str = "balanced_accuracy"

    def apply(self, probabilities: np.ndarray) -> np.ndarray:
        values = np.clip(
            np.asarray(probabilities, dtype=np.float64), 1e-7, 1.0 - 1e-7
        )
        if self.mode == "none":
            return values.copy()
        if self.mode != "platt":
            raise ValueError(f"Unsupported calibration mode: {self.mode}")
        logits = np.log(values / (1.0 - values))
        calibrated_logits = self.coefficient * logits + self.intercept
        return 1.0 / (1.0 + np.exp(-calibrated_logits))


def select_secondary_inner_threshold(
    labels: np.ndarray, probabilities: np.ndarray
) -> float:
    """Predeclared secondary threshold: maximum inner OOF balanced accuracy."""
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    candidates = np.unique(np.concatenate([values, np.asarray([0.5])]))
    rows: list[tuple[float, float]] = []
    for threshold in candidates:
        predictions = (values >= threshold).astype(np.int64)
        score = float(balanced_accuracy_score(y_true, predictions))
        rows.append((float(threshold), score))
    maximum = max(score for _, score in rows)
    tied = [
        threshold
        for threshold, score in rows
        if math.isclose(score, maximum, rel_tol=0.0, abs_tol=1e-12)
    ]
    return min(tied, key=lambda value: (abs(value - 0.5), value))


def fit_platt_newton(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    l2_penalty: float = 1e-6,
    maximum_iterations: int = 100,
    tolerance: float = 1e-10,
) -> tuple[float, float]:
    """Fit two-parameter Platt scaling with deterministic Newton/IRLS.

    The local Windows SciPy L-BFGS binary is known to fail at process level.
    This small, dependency-free solver optimizes the same regularized binary
    logistic likelihood for one logit feature and an intercept.
    """
    values = np.clip(
        np.asarray(probabilities, dtype=np.float64).reshape(-1),
        1e-7,
        1.0 - 1e-7,
    )
    y_true = np.asarray(labels, dtype=np.float64).reshape(-1)
    if len(values) != len(y_true) or not len(values):
        raise ValueError("Platt probabilities and labels must be non-empty")
    if not np.all(np.isin(y_true, [0.0, 1.0])):
        raise ValueError("Platt labels must be binary")
    logits = np.log(values / (1.0 - values))
    positives = float(np.sum(y_true))
    negatives = float(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Platt calibration requires both classes")
    parameters = np.asarray(
        [0.0, math.log((positives + 1.0) / (negatives + 1.0))],
        dtype=np.float64,
    )

    def objective(beta: np.ndarray) -> float:
        linear = logits * beta[0] + beta[1]
        likelihood = np.logaddexp(0.0, linear) - y_true * linear
        return float(
            np.sum(likelihood) + 0.5 * l2_penalty * beta[0] ** 2
        )

    for _ in range(maximum_iterations):
        linear = np.clip(
            logits * parameters[0] + parameters[1], -40.0, 40.0
        )
        fitted = 1.0 / (1.0 + np.exp(-linear))
        residual = fitted - y_true
        gradient_0 = float(np.sum(logits * residual)) + (
            l2_penalty * parameters[0]
        )
        gradient_1 = float(np.sum(residual))
        weights = np.maximum(fitted * (1.0 - fitted), 1e-12)
        hessian_00 = float(np.sum(weights * logits * logits)) + l2_penalty
        hessian_01 = float(np.sum(weights * logits))
        hessian_11 = float(np.sum(weights)) + 1e-12
        determinant = hessian_00 * hessian_11 - hessian_01 * hessian_01
        if abs(determinant) < 1e-18:
            determinant = math.copysign(1e-18, determinant or 1.0)
        step = np.asarray(
            [
                (
                    hessian_11 * gradient_0
                    - hessian_01 * gradient_1
                )
                / determinant,
                (
                    -hessian_01 * gradient_0
                    + hessian_00 * gradient_1
                )
                / determinant,
            ],
            dtype=np.float64,
        )
        if float(np.max(np.abs(step))) <= tolerance:
            break
        current_objective = objective(parameters)
        scale = 1.0
        accepted = False
        while scale >= 2.0 ** -20:
            proposal = parameters - scale * step
            if objective(proposal) <= current_objective:
                parameters = proposal
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            break
    if not np.all(np.isfinite(parameters)):
        raise RuntimeError("Platt calibration produced non-finite parameters")
    return float(parameters[0]), float(parameters[1])


def fit_inner_oof_calibration(
    *,
    mode: str,
    probabilities: np.ndarray,
    labels: np.ndarray,
    oof_indices: np.ndarray,
    outer_train_indices: np.ndarray,
    outer_test_indices: np.ndarray,
) -> CalibrationModel:
    """Fit Platt calibration and the secondary threshold on outer-train OOF."""
    if mode not in {"none", "platt"}:
        raise ValueError("Calibration mode must be none or platt")
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    oof = np.asarray(oof_indices, dtype=np.int64).reshape(-1)
    outer_train = np.asarray(outer_train_indices, dtype=np.int64).reshape(-1)
    outer_test = np.asarray(outer_test_indices, dtype=np.int64).reshape(-1)
    if not (len(values) == len(y_true) == len(oof) == len(outer_train)):
        raise ValueError("Calibration inputs must cover outer train exactly")
    if not np.array_equal(oof, outer_train):
        raise RuntimeError("Calibration OOF indices must equal outer train order")
    overlap = int(len(np.intersect1d(oof, outer_test)))
    if overlap:
        raise RuntimeError("Outer-test index entered calibration fit")
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Calibration probabilities contain NaN/Inf")

    if mode == "platt":
        coefficient, intercept = fit_platt_newton(values, y_true)
    else:
        coefficient = 1.0
        intercept = 0.0
    temporary = CalibrationModel(
        mode=mode,
        coefficient=coefficient,
        intercept=intercept,
        selected_secondary_threshold=0.5,
        fit_indices_sha256=audited.sha256_indices(oof),
        fit_indices_are_outer_train=True,
        fit_outer_test_overlap=0,
    )
    calibrated = temporary.apply(values)
    threshold = select_secondary_inner_threshold(y_true, calibrated)
    return CalibrationModel(
        mode=mode,
        coefficient=coefficient,
        intercept=intercept,
        selected_secondary_threshold=threshold,
        fit_indices_sha256=audited.sha256_indices(oof),
        fit_indices_are_outer_train=True,
        fit_outer_test_overlap=0,
    )


def metrics_at_threshold(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float
) -> Dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    predictions = (values >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "auc": audited.safe_auc(y_true, values),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier": float(brier_score_loss(y_true, values)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def _score(candidate: OptimizationCandidate, evaluation: CandidateSeedEvaluation) -> CandidateScore:
    return CandidateScore(
        candidate_id=candidate.candidate_id,
        auc=evaluation.auc,
        brier=evaluation.brier,
        trainable_parameters=candidate.trainable_parameters,
    )


def _actual_inner_seed(base_seed: int, outer_fold: int, inner_fold: int) -> int:
    return base_seed + outer_fold * 1000 + inner_fold * 100


def _actual_outer_seed(base_seed: int, outer_fold: int) -> int:
    return base_seed + outer_fold * 1000 + 777


def _inner_labels(
    outer_split: audited.OuterSplit,
    y_outer_train: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    return audited.labels_for_global_indices(
        outer_split.train_idx, y_outer_train, indices
    )


def _trial_json(result: InnerTrialResult) -> Dict[str, Any]:
    return {
        "candidate_id": result.candidate_id,
        "base_seed": result.base_seed,
        "inner_fold": result.inner_fold,
        "validation_indices_sha256": audited.sha256_indices(
            result.validation_indices
        ),
        "probabilities_sha256": array_sha256(result.probabilities),
        "best_epoch": result.best_epoch,
        "best_validation_auc": result.best_validation_auc,
        "stopped_epoch": result.stopped_epoch,
        "checkpoint_state_sha256": result.checkpoint_state_sha256,
        "model_instance_token": result.model_instance_token,
        "optimizer_instance_token": result.optimizer_instance_token,
        "preprocessing_state_token": result.preprocessing_state_token,
        "checkpoint_selection": "inner_validation_auc_only_earlier_epoch_on_tie",
    }


def _evaluation_json(evaluation: CandidateSeedEvaluation) -> Dict[str, Any]:
    return {
        "candidate_id": evaluation.candidate_id,
        "base_seed": evaluation.base_seed,
        "oof_indices_sha256": audited.sha256_indices(evaluation.oof_indices),
        "oof_probabilities_sha256": array_sha256(
            evaluation.oof_probabilities
        ),
        "inner_oof_auc": evaluation.auc,
        "inner_oof_brier": evaluation.brier,
        "best_epochs": list(evaluation.best_epochs),
        "checkpoint_sha256": list(evaluation.checkpoint_sha256),
    }


def run_outer_fold_optimization(
    *,
    features: audited.FeatureBundle,
    label_firewall: audited.LabelFirewall,
    outer_split: audited.OuterSplit,
    output_directory: str | Path,
    backend: OptimizationBackend,
    yeo_nodes: Dict[int, np.ndarray],
    calibration_mode: str,
    candidates: Sequence[OptimizationCandidate] = CANDIDATES,
    schedule: EngineSchedule = EngineSchedule(),
) -> Dict[str, Any]:
    """Optimize and predict one fold without accepting outer-test labels."""
    output = assert_output_path_safe(output_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite outer-fold output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    schedule.validate(len(candidates))
    outer_split.validate(len(features.subject_ids))
    if label_firewall.n_subjects != len(features.subject_ids):
        raise ValueError("Feature and label-firewall lengths differ")
    y_outer_train = label_firewall.training_labels(
        outer_split.train_idx, outer_split.test_idx
    )
    inner_splits = audited.make_inner_splits(
        outer_split, y_outer_train, LEVEL1_SEED
    )
    candidate_map = {candidate.candidate_id: candidate for candidate in candidates}
    if len(candidate_map) != len(candidates):
        raise ValueError("Candidate IDs must be unique")

    all_preprocessing_tokens: list[str] = []
    all_model_tokens: list[str] = []
    all_optimizer_tokens: list[str] = []

    # Level 1, rung 1: all candidates, seed 42, all three inner folds.
    level1_prepared: dict[tuple[str, int], OptimizationPreparedPartition] = {}
    level1_rung1_trials: dict[str, list[InnerTrialResult]] = {}
    level1_rung1_evaluations: dict[str, CandidateSeedEvaluation] = {}
    for candidate in candidates:
        trials: list[InnerTrialResult] = []
        for inner_fold, (fit_idx, validation_idx) in enumerate(inner_splits):
            y_fit = _inner_labels(outer_split, y_outer_train, fit_idx)
            y_validation = _inner_labels(
                outer_split, y_outer_train, validation_idx
            )
            scope = (
                f"outer_{outer_split.fold:02d}/level1/{candidate.candidate_id}/"
                f"seed_42/inner_{inner_fold:02d}"
            )
            prepared = fit_candidate_preprocessing(
                features=features,
                fit_indices=fit_idx,
                evaluation_indices=validation_idx,
                y_fit=y_fit,
                candidate=candidate,
                yeo_nodes=yeo_nodes,
                outer_test_indices=outer_split.test_idx,
                trial_scope=scope,
            )
            trial_dir = (
                output
                / "inner"
                / "level1"
                / candidate.candidate_id
                / "seed_42"
                / f"inner_{inner_fold:02d}"
            )
            save_preprocessing_artifacts(
                trial_dir / "preprocessing",
                prepared,
                outer_split.train_idx,
                outer_split.test_idx,
            )
            result = backend.run_inner(
                candidate=candidate,
                prepared=prepared,
                y_fit=y_fit,
                y_validation=y_validation,
                base_seed=LEVEL1_SEED,
                actual_seed=_actual_inner_seed(
                    LEVEL1_SEED, outer_split.fold, inner_fold
                ),
                outer_fold=outer_split.fold,
                inner_fold=inner_fold,
                budget_epoch=schedule.level1_rung1_epoch,
                planned_max_epoch=schedule.level1_max_epoch,
                minimum_epoch=None,
                patience=None,
                evaluation_interval=schedule.evaluation_interval,
                continuation=None,
            )
            write_json_new(trial_dir / "rung1_result.json", _trial_json(result))
            level1_prepared[(candidate.candidate_id, inner_fold)] = prepared
            trials.append(result)
            all_preprocessing_tokens.append(prepared.state_token)
            all_model_tokens.append(result.model_instance_token)
            all_optimizer_tokens.append(result.optimizer_instance_token)
        evaluation = concatenate_inner_oof(
            candidate=candidate,
            base_seed=LEVEL1_SEED,
            trials=trials,
            outer_train_indices=outer_split.train_idx,
            y_outer_train=y_outer_train,
        )
        level1_rung1_trials[candidate.candidate_id] = trials
        level1_rung1_evaluations[candidate.candidate_id] = evaluation

    rung1_ranking = rank_candidate_scores(
        [
            _score(candidate, level1_rung1_evaluations[candidate.candidate_id])
            for candidate in candidates
        ]
    )
    rung1_survivor_ids = [
        item.candidate_id
        for item in rung1_ranking[: schedule.level1_rung1_survivors]
    ]
    write_json_new(
        output / "inner" / "level1_rung1_selection.json",
        {
            "metric_scope": "outer_train_inner_oof_only",
            "base_seed": LEVEL1_SEED,
            "ranking": [asdict(item) for item in rung1_ranking],
            "survivors": rung1_survivor_ids,
            "outer_test_metrics_available": False,
        },
    )

    # Level 1, rung 2: resume only the six survivors.
    level1_rung2_trials: dict[str, list[InnerTrialResult]] = {}
    level1_rung2_evaluations: dict[str, CandidateSeedEvaluation] = {}
    for candidate_id in rung1_survivor_ids:
        candidate = candidate_map[candidate_id]
        resumed: list[InnerTrialResult] = []
        for inner_fold, (fit_idx, validation_idx) in enumerate(inner_splits):
            prepared = level1_prepared[(candidate_id, inner_fold)]
            previous = level1_rung1_trials[candidate_id][inner_fold]
            y_fit = _inner_labels(outer_split, y_outer_train, fit_idx)
            y_validation = _inner_labels(
                outer_split, y_outer_train, validation_idx
            )
            result = backend.run_inner(
                candidate=candidate,
                prepared=prepared,
                y_fit=y_fit,
                y_validation=y_validation,
                base_seed=LEVEL1_SEED,
                actual_seed=_actual_inner_seed(
                    LEVEL1_SEED, outer_split.fold, inner_fold
                ),
                outer_fold=outer_split.fold,
                inner_fold=inner_fold,
                budget_epoch=schedule.level1_max_epoch,
                planned_max_epoch=schedule.level1_max_epoch,
                minimum_epoch=schedule.level1_minimum_epoch,
                patience=schedule.level1_patience,
                evaluation_interval=schedule.evaluation_interval,
                continuation=previous.continuation,
            )
            if (
                result.model_instance_token != previous.model_instance_token
                or result.optimizer_instance_token
                != previous.optimizer_instance_token
                or result.preprocessing_state_token
                != previous.preprocessing_state_token
            ):
                raise RuntimeError("Rung-2 continuation did not preserve trial state")
            trial_dir = (
                output
                / "inner"
                / "level1"
                / candidate.candidate_id
                / "seed_42"
                / f"inner_{inner_fold:02d}"
            )
            write_json_new(trial_dir / "rung2_result.json", _trial_json(result))
            resumed.append(result)
        evaluation = concatenate_inner_oof(
            candidate=candidate,
            base_seed=LEVEL1_SEED,
            trials=resumed,
            outer_train_indices=outer_split.train_idx,
            y_outer_train=y_outer_train,
        )
        level1_rung2_trials[candidate_id] = resumed
        level1_rung2_evaluations[candidate_id] = evaluation

    rung2_ranking = rank_candidate_scores(
        [
            _score(candidate_map[candidate_id], evaluation)
            for candidate_id, evaluation in level1_rung2_evaluations.items()
        ]
    )
    top2_ids = [
        item.candidate_id
        for item in rung2_ranking[: schedule.level1_final_survivors]
    ]
    if len(top2_ids) != 2:
        raise RuntimeError("Level 1 must produce exactly two valid candidates")
    write_json_new(
        output / "inner" / "level1_final_selection.json",
        {
            "metric_scope": "outer_train_inner_oof_only",
            "base_seed": LEVEL1_SEED,
            "ranking": [asdict(item) for item in rung2_ranking],
            "top2": top2_ids,
            "outer_test_metrics_available": False,
        },
    )

    # Level 2: fresh three-seed inner evaluation of the top two.
    level2_evaluations: dict[
        str, dict[int, CandidateSeedEvaluation]
    ] = {candidate_id: {} for candidate_id in top2_ids}
    level2_trial_records: dict[
        tuple[str, int], list[InnerTrialResult]
    ] = {}
    for candidate_id in top2_ids:
        candidate = candidate_map[candidate_id]
        for base_seed in LEVEL2_SEEDS:
            trials = []
            for inner_fold, (fit_idx, validation_idx) in enumerate(inner_splits):
                y_fit = _inner_labels(outer_split, y_outer_train, fit_idx)
                y_validation = _inner_labels(
                    outer_split, y_outer_train, validation_idx
                )
                scope = (
                    f"outer_{outer_split.fold:02d}/level2/{candidate_id}/"
                    f"seed_{base_seed}/inner_{inner_fold:02d}"
                )
                prepared = fit_candidate_preprocessing(
                    features=features,
                    fit_indices=fit_idx,
                    evaluation_indices=validation_idx,
                    y_fit=y_fit,
                    candidate=candidate,
                    yeo_nodes=yeo_nodes,
                    outer_test_indices=outer_split.test_idx,
                    trial_scope=scope,
                )
                trial_dir = (
                    output
                    / "inner"
                    / "level2"
                    / candidate_id
                    / f"seed_{base_seed}"
                    / f"inner_{inner_fold:02d}"
                )
                save_preprocessing_artifacts(
                    trial_dir / "preprocessing",
                    prepared,
                    outer_split.train_idx,
                    outer_split.test_idx,
                )
                result = backend.run_inner(
                    candidate=candidate,
                    prepared=prepared,
                    y_fit=y_fit,
                    y_validation=y_validation,
                    base_seed=base_seed,
                    actual_seed=_actual_inner_seed(
                        base_seed, outer_split.fold, inner_fold
                    ),
                    outer_fold=outer_split.fold,
                    inner_fold=inner_fold,
                    budget_epoch=schedule.level2_max_epoch,
                    planned_max_epoch=schedule.level2_max_epoch,
                    minimum_epoch=schedule.level2_minimum_epoch,
                    patience=schedule.level2_patience,
                    evaluation_interval=schedule.evaluation_interval,
                    continuation=None,
                )
                write_json_new(trial_dir / "result.json", _trial_json(result))
                trials.append(result)
                all_preprocessing_tokens.append(prepared.state_token)
                all_model_tokens.append(result.model_instance_token)
                all_optimizer_tokens.append(result.optimizer_instance_token)
            evaluation = concatenate_inner_oof(
                candidate=candidate,
                base_seed=base_seed,
                trials=trials,
                outer_train_indices=outer_split.train_idx,
                y_outer_train=y_outer_train,
            )
            level2_evaluations[candidate_id][base_seed] = evaluation
            level2_trial_records[(candidate_id, base_seed)] = trials
            write_json_new(
                output
                / "inner"
                / "level2"
                / candidate_id
                / f"seed_{base_seed}"
                / "inner_oof_evaluation.json",
                _evaluation_json(evaluation),
            )

    level2_scores: list[CandidateScore] = []
    level2_summary: dict[str, Any] = {}
    for candidate_id in top2_ids:
        candidate = candidate_map[candidate_id]
        seed_evaluations = [
            level2_evaluations[candidate_id][seed] for seed in LEVEL2_SEEDS
        ]
        mean_auc = float(np.mean([item.auc for item in seed_evaluations]))
        mean_brier = float(
            np.mean([item.brier for item in seed_evaluations])
        )
        level2_scores.append(
            CandidateScore(
                candidate_id,
                mean_auc,
                mean_brier,
                candidate.trainable_parameters,
            )
        )
        level2_summary[candidate_id] = {
            "seed_evaluations": [
                _evaluation_json(item) for item in seed_evaluations
            ],
            "mean_three_seed_inner_oof_auc": mean_auc,
            "mean_three_seed_inner_oof_brier": mean_brier,
        }
    final_ranking = rank_candidate_scores(level2_scores)
    selected_candidate_id = final_ranking[0].candidate_id
    selected_candidate = candidate_map[selected_candidate_id]
    write_json_new(
        output / "inner" / "level2_final_selection.json",
        {
            "metric_scope": "outer_train_inner_oof_only",
            "fixed_base_seeds": list(LEVEL2_SEEDS),
            "candidates": level2_summary,
            "ranking": [asdict(item) for item in final_ranking],
            "selected_candidate": selected_candidate_id,
            "seed_selection_performed": False,
            "outer_test_metrics_available": False,
        },
    )

    selected_seed_evaluations = [
        level2_evaluations[selected_candidate_id][seed]
        for seed in LEVEL2_SEEDS
    ]
    selected_inner_seed_mean = np.mean(
        np.stack(
            [item.oof_probabilities for item in selected_seed_evaluations],
            axis=0,
        ),
        axis=0,
    )
    calibration = fit_inner_oof_calibration(
        mode=calibration_mode,
        probabilities=selected_inner_seed_mean,
        labels=y_outer_train,
        oof_indices=outer_split.train_idx,
        outer_train_indices=outer_split.train_idx,
        outer_test_indices=outer_split.test_idx,
    )
    write_json_new(
        output / "inner" / "calibration.json",
        {
            **asdict(calibration),
            "fit_source": (
                "equal_weight_mean_of_selected_candidate_three_seed_inner_oof"
            ),
            "candidate_selection_used_calibrated_probabilities": False,
            "fixed_0_5_is_primary": True,
            "inner_selected_threshold_is_secondary": True,
        },
    )

    fixed_epochs = {
        seed: int(
            np.median(
                level2_evaluations[selected_candidate_id][seed].best_epochs
            )
        )
        for seed in LEVEL2_SEEDS
    }

    # Outer preprocessing is freshly refit for every seed; no state is reused.
    outer_results: list[OuterFitResult] = []
    for base_seed in LEVEL2_SEEDS:
        scope = (
            f"outer_{outer_split.fold:02d}/outer_final/"
            f"{selected_candidate_id}/seed_{base_seed}"
        )
        prepared = fit_candidate_preprocessing(
            features=features,
            fit_indices=outer_split.train_idx,
            evaluation_indices=outer_split.test_idx,
            y_fit=y_outer_train,
            candidate=selected_candidate,
            yeo_nodes=yeo_nodes,
            outer_test_indices=outer_split.test_idx,
            trial_scope=scope,
        )
        seed_dir = output / "outer_final" / f"seed_{base_seed}"
        save_preprocessing_artifacts(
            seed_dir / "preprocessing",
            prepared,
            outer_split.train_idx,
            outer_split.test_idx,
        )
        result = backend.run_outer(
            candidate=selected_candidate,
            prepared=prepared,
            y_fit=y_outer_train,
            base_seed=base_seed,
            actual_seed=_actual_outer_seed(base_seed, outer_split.fold),
            outer_fold=outer_split.fold,
            fixed_epoch=fixed_epochs[base_seed],
        )
        write_json_new(
            seed_dir / "fit_manifest.json",
            {
                "candidate_id": result.candidate_id,
                "base_seed": result.base_seed,
                "fixed_epoch": result.fixed_epoch,
                "probabilities_sha256": array_sha256(result.probabilities),
                "checkpoint_state_sha256": result.checkpoint_state_sha256,
                "model_instance_token": result.model_instance_token,
                "optimizer_instance_token": result.optimizer_instance_token,
                "preprocessing_state_token": result.preprocessing_state_token,
                "outer_test_labels_available": False,
                "outer_prediction_calls": 1,
            },
        )
        outer_results.append(result)
        all_preprocessing_tokens.append(prepared.state_token)
        all_model_tokens.append(result.model_instance_token)
        all_optimizer_tokens.append(result.optimizer_instance_token)

    if len(outer_results) != len(LEVEL2_SEEDS):
        raise RuntimeError("All three outer seed fits are required")
    seed_probabilities = np.stack(
        [result.probabilities for result in outer_results], axis=0
    )
    raw_equal_weight_mean = np.mean(seed_probabilities, axis=0)
    final_probabilities = calibration.apply(raw_equal_weight_mean)
    if not np.all(np.isfinite(final_probabilities)):
        raise RuntimeError("Final outer probabilities contain NaN/Inf")

    prediction_audit_path = output / "outer_final" / "seed_predictions.npz"
    np.savez(
        prediction_audit_path,
        base_seeds=np.asarray(LEVEL2_SEEDS, dtype=np.int64),
        seed_probabilities=seed_probabilities.astype(np.float64),
        raw_equal_weight_mean=raw_equal_weight_mean.astype(np.float64),
        final_probabilities=final_probabilities.astype(np.float64),
    )
    frozen_path = output / "frozen_outer_test_predictions.npz"
    frozen_manifest = audited.save_frozen_predictions(
        frozen_path,
        subject_ids=features.subject_ids[outer_split.test_idx],
        row_indices=outer_split.test_idx,
        probabilities=final_probabilities,
        outer_folds=np.full(
            len(outer_split.test_idx), outer_split.fold, dtype=np.int64
        ),
    )
    frozen_manifest.update(
        {
            "protocol_name": "cpac_model_optimization_v1",
            "candidate_registry_sha256": CANDIDATE_REGISTRY_SHA256,
            "outer_fold": outer_split.fold,
            "selected_candidate": selected_candidate_id,
            "base_seeds": list(LEVEL2_SEEDS),
            "seed_weights": [1.0 / 3.0] * 3,
            "seed_prediction_audit_file": str(prediction_audit_path),
            "seed_prediction_audit_sha256": audited.sha256_file(
                prediction_audit_path
            ),
            "raw_equal_weight_mean_sha256": array_sha256(
                raw_equal_weight_mean
            ),
            "calibration": asdict(calibration),
            "fixed_0_5_primary": True,
            "inner_selected_threshold_secondary": True,
            "configuration_level_fusion": False,
            "outer_metrics_computed": False,
            "outer_test_labels_used": False,
            "prediction_generated_before_evaluation": True,
        }
    )
    manifest_path = output / "frozen_prediction_manifest.json"
    write_json_new(manifest_path, frozen_manifest)
    audited.verify_frozen_prediction(frozen_path, manifest_path)

    # Rung-2 repeats its own rung-1 state by design.  Every newly initialized
    # trial and every outer seed must have a unique token.
    if len(all_preprocessing_tokens) != len(set(all_preprocessing_tokens)):
        raise RuntimeError("Preprocessing state was reused across new trials")
    if len(all_model_tokens) != len(set(all_model_tokens)):
        raise RuntimeError("Model state was reused across new trials")
    if len(all_optimizer_tokens) != len(set(all_optimizer_tokens)):
        raise RuntimeError("Optimizer state was reused across new trials")

    summary = {
        "stage": "A_OPTIMIZED_FROZEN_OUTER_PREDICTION",
        "outer_fold": outer_split.fold,
        "outer_split_sha256": outer_split.split_sha256,
        "candidate_registry_sha256": CANDIDATE_REGISTRY_SHA256,
        "level1_rung1_survivors": rung1_survivor_ids,
        "level1_top2": top2_ids,
        "selected_candidate": selected_candidate_id,
        "fixed_epochs_by_seed": {str(key): value for key, value in fixed_epochs.items()},
        "calibration": asdict(calibration),
        "frozen_prediction": frozen_manifest,
        "outer_test_labels_used": False,
        "outer_metrics_computed": False,
        "configuration_level_fusion": False,
        "seed_ensemble_only": True,
    }
    write_json_new(output / "stage_a_summary.json", summary)
    return {
        **summary,
        "prediction_path": frozen_path,
        "manifest_path": manifest_path,
        "raw_seed_probabilities": seed_probabilities,
        "raw_equal_weight_mean": raw_equal_weight_mean,
        "final_probabilities": final_probabilities,
        "level1_rung1_evaluations": level1_rung1_evaluations,
        "level1_rung2_evaluations": level1_rung2_evaluations,
        "level2_evaluations": level2_evaluations,
        "all_new_preprocessing_tokens": all_preprocessing_tokens,
        "all_new_model_tokens": all_model_tokens,
        "all_new_optimizer_tokens": all_optimizer_tokens,
    }


def evaluate_frozen_outer_prediction(
    *,
    prediction_path: str | Path,
    manifest_path: str | Path,
    label_firewall: audited.LabelFirewall,
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Stage B: verify the frozen file before requesting any labels."""
    frozen = audited.verify_frozen_prediction(prediction_path, manifest_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not manifest.get("prediction_generated_before_evaluation"):
        raise RuntimeError("Prediction-generation gate is absent")
    if manifest.get("outer_metrics_computed") is not False:
        raise RuntimeError("Prediction manifest is not a pristine Stage-A artifact")
    # This is the first authorized outer-label access in the engine.
    labels = label_firewall.labels_for_stage_b(frozen["subject_ids"])
    fixed = audited.compute_fixed_threshold_metrics(
        labels, frozen["probabilities"]
    )
    threshold = float(
        manifest["calibration"]["selected_secondary_threshold"]
    )
    secondary = metrics_at_threshold(
        labels, frozen["probabilities"], threshold
    )
    result = {
        "stage": "B_OUTER_EVALUATION_AFTER_FROZEN_PREDICTION",
        "prediction_sha256_verified_before_label_access": audited.sha256_file(
            Path(prediction_path)
        ),
        "primary": {
            "label": "fixed_0_5_primary",
            "metrics": fixed,
        },
        "secondary": {
            "label": "inner_oof_selected_threshold_secondary",
            "threshold_source": "outer_train_selected_candidate_inner_oof_only",
            "metrics": secondary,
        },
        "outer_performance_used_for_selection": False,
    }
    if output_path is not None:
        write_json_new(Path(output_path), result)
    return result


def verify_implementation_protocol() -> Dict[str, Any]:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["candidate_count"] != 12:
        raise RuntimeError("Protocol candidate count changed")
    if protocol["level_2_three_seed_selection"]["base_seeds"] != [42, 43, 44]:
        raise RuntimeError("Protocol seeds changed")
    if (
        protocol["outer_final_training_and_prediction"][
            "configuration_level_outer_oof_fusion"
        ]
        is not False
    ):
        raise RuntimeError("Configuration-level fusion must remain disabled")
    if protocol["ocread_inference_policy"]["mode"] != "mu":
        raise RuntimeError("OCREAD inference must remain deterministic mu")
    return {
        "status": "PASS",
        "candidate_count": len(CANDIDATES),
        "candidate_registry_sha256": CANDIDATE_REGISTRY_SHA256,
        "protocol_sha256": audited.sha256_file(PROTOCOL_PATH),
        "candidate_table_sha256": audited.sha256_file(CANDIDATE_TABLE_PATH),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-free C-PAC model optimization v1"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify-protocol")

    run_fold = subparsers.add_parser("run-fold")
    run_fold.add_argument("--fold", required=True, type=int)
    run_fold.add_argument("--output-dir", required=True)
    run_fold.add_argument("--data-dir", default=str(audited.DATA_DIR))
    run_fold.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    run_fold.add_argument(
        "--calibration",
        choices=("none", "platt"),
        required=True,
        help="Must be frozen before training; candidate selection remains raw.",
    )

    evaluate = subparsers.add_parser("evaluate-fold")
    evaluate.add_argument("--fold-dir", required=True)
    evaluate.add_argument("--data-dir", default=str(audited.DATA_DIR))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "verify-protocol":
        print(
            json.dumps(
                verify_implementation_protocol(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "run-fold":
        verify_implementation_protocol()
        features, firewall, _ledger = load_audited_inputs(args.data_dir)
        outer_splits = firewall.make_outer_splits(OUTER_SPLIT_SEED)
        audited.validate_outer_splits(outer_splits, len(features.subject_ids))
        if args.fold not in range(audited.N_OUTER):
            raise ValueError("--fold must be in 0..9")
        backend = TorchOptimizationBackend(audited.choose_device(args.device))
        result = run_outer_fold_optimization(
            features=features,
            label_firewall=firewall,
            outer_split=outer_splits[args.fold],
            output_directory=args.output_dir,
            backend=backend,
            yeo_nodes=audited.load_yeo7_nodes(),
            calibration_mode=args.calibration,
        )
        print(
            json.dumps(
                {
                    "status": "FROZEN_OUTER_PREDICTION",
                    "outer_fold": args.fold,
                    "selected_candidate": result["selected_candidate"],
                    "prediction_path": str(result["prediction_path"]),
                    "outer_metrics_computed": False,
                },
                ensure_ascii=False,
            )
        )
        return 0
    if args.command == "evaluate-fold":
        fold_dir = Path(args.fold_dir).resolve()
        firewall = audited.load_label_firewall(args.data_dir)
        result = evaluate_frozen_outer_prediction(
            prediction_path=fold_dir / "frozen_outer_test_predictions.npz",
            manifest_path=fold_dir / "frozen_prediction_manifest.json",
            label_firewall=firewall,
            output_path=fold_dir / "stage_b_outer_metrics.json",
        )
        print(
            json.dumps(
                {
                    "status": "OUTER_EVALUATION_COMPLETE",
                    "stage": result["stage"],
                },
                ensure_ascii=False,
            )
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

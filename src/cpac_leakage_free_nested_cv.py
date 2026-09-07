"""Independent leakage-free nested-CV pipeline for ABIDE-I C-PAC.

Protocol
--------
* Input FC always starts from all 19,900 CC200 lower-triangle edges.
* Outer CV is fixed at 10 stratified folds; inner CV is fixed at 3 folds.
* Every inner-fit and outer-training partition independently fits:
  F-score FC selection (19,900 -> 4,975), all scalers, demographic
  imputation, and frequency-derived anchors.
* Inner checkpoints are selected only by inner-validation AUC.
* The median inner best epoch is used as the fixed outer-final epoch.
* Outer-test labels are not accepted by preprocessing, training, or
  prediction functions.
* Classification uses the fixed threshold 0.5.
* No configuration-level fusion is implemented.

Stage A writes frozen predictions without true outer-test labels and records a
SHA-256 digest. Stage B is a separate entry point that verifies the digest,
loads labels, and computes metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_selection import f_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from cpac_leakage_free_model import BrainInnovationSystem


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "abide_data" / "Outputs" / "cpac" / "filt_noglobal"
OUTPUT_ROOT = ROOT / "cpac_leakage_free_nested_cv"
YEO_MAPPING_PATH = ROOT / "cc200_yeo7_mapping.csv"

FULL_FC_FILENAME = "X_s_full19900_rebuilt_strict.npy"
SUBJECT_IDS_FILENAME = "X_s_full19900_rebuilt_strict_subject_ids.npy"
FREQUENCY_FILENAME = "X_freq_filtered.npy"
LABEL_FILENAME = "y_labels.npy"
PHENOTYPE_FILENAME = "Phenotypic_V1_0b_preprocessed1.csv"

N_ROI = 200
N_FREQUENCY_BINS = 15
N_FULL_EDGES = 19_900
N_SELECTED_EDGES = 4_975
N_FREQUENCY = N_ROI * N_FREQUENCY_BINS
N_DEMOGRAPHIC = 2
N_OUTER = 10
N_INNER = 3
CLASSIFICATION_THRESHOLD = 0.5
DEFAULT_SEED = 20260717

FORBIDDEN_EXACT_INPUTS = {
    "x_features_filtered.npy",
    "x_s_4975_selected.npy",
    "x_demo.npy",
    "offline_oof_blender_results.json",
    "results_80target_ensemble.json",
    "paper_final_oof_probs.npz",
    "final_80_recipe_probs.npz",
    "final_80_recipe_results.json",
}


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def ensure_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {path}")
    path.mkdir(parents=True, exist_ok=False)


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    ensure_directory(path.parent)
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


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def sha256_indices(indices: np.ndarray) -> str:
    return sha256_array(np.asarray(indices, dtype=np.int64))


def assert_input_path_allowed(path: str | Path) -> None:
    candidate = Path(path)
    name = candidate.name.casefold()
    if name in FORBIDDEN_EXACT_INPUTS:
        raise RuntimeError(f"Forbidden legacy input requested: {candidate}")
    if name.startswith("oof_") and name.endswith(".npz"):
        raise RuntimeError(f"Forbidden legacy OOF input requested: {candidate}")
    if "blend" in name and candidate.suffix.casefold() in {".json", ".npz", ".npy"}:
        raise RuntimeError(f"Forbidden legacy fusion input requested: {candidate}")


def safe_load_array(
    path: str | Path,
    *,
    mmap_mode: Optional[str] = None,
    allow_pickle: bool = False,
) -> np.ndarray:
    candidate = Path(path)
    assert_input_path_allowed(candidate)
    if not candidate.exists():
        raise FileNotFoundError(candidate)
    return np.load(candidate, mmap_mode=mmap_mode, allow_pickle=allow_pickle)


@dataclass(frozen=True)
class FeatureBundle:
    full_fc: np.ndarray
    frequency: np.ndarray
    demographics_raw: np.ndarray
    subject_ids: np.ndarray
    source_files: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        n_subjects = len(self.subject_ids)
        expected = {
            "full_fc": (n_subjects, N_FULL_EDGES),
            "frequency": (n_subjects, N_FREQUENCY),
            "demographics_raw": (n_subjects, N_DEMOGRAPHIC),
        }
        actual = {
            "full_fc": self.full_fc.shape,
            "frequency": self.frequency.shape,
            "demographics_raw": self.demographics_raw.shape,
        }
        for key, shape in expected.items():
            if actual[key] != shape:
                raise ValueError(f"{key} shape {actual[key]} != required {shape}")
        if self.subject_ids.ndim != 1:
            raise ValueError("subject_ids must be one-dimensional")
        if len(np.unique(self.subject_ids)) != n_subjects:
            raise ValueError("subject_ids must be unique")
        for name, values in (
            ("full_fc", self.full_fc),
            ("frequency", self.frequency),
        ):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains NaN or Inf")


class LabelFirewall:
    """Owns labels and exposes only explicitly authorized partitions."""

    def __init__(self, labels: np.ndarray, subject_ids: np.ndarray):
        labels_array = np.asarray(labels, dtype=np.int64).reshape(-1).copy()
        subject_array = np.asarray(subject_ids, dtype=np.int64).reshape(-1).copy()
        if labels_array.shape != subject_array.shape:
            raise ValueError("labels and subject_ids must have identical shape")
        if not np.all(np.isin(labels_array, [0, 1])):
            raise ValueError("labels must be binary 0/1")
        if len(np.unique(subject_array)) != len(subject_array):
            raise ValueError("LabelFirewall subject IDs must be unique")
        self.__labels = labels_array
        self.__subject_ids = subject_array

    @property
    def n_subjects(self) -> int:
        return len(self.__labels)

    def make_outer_splits(self, seed: int = DEFAULT_SEED) -> List["OuterSplit"]:
        splitter = StratifiedKFold(
            n_splits=N_OUTER, shuffle=True, random_state=seed
        )
        return [
            OuterSplit(
                fold=fold,
                train_idx=train.astype(np.int64),
                test_idx=test.astype(np.int64),
            )
            for fold, (train, test) in enumerate(
                splitter.split(np.zeros(self.n_subjects), self.__labels)
            )
        ]

    def training_labels(
        self, train_indices: np.ndarray, forbidden_test_indices: np.ndarray
    ) -> np.ndarray:
        train = np.asarray(train_indices, dtype=np.int64)
        forbidden = np.asarray(forbidden_test_indices, dtype=np.int64)
        if len(np.intersect1d(train, forbidden)) != 0:
            raise RuntimeError("Label firewall rejected train/test overlap")
        if np.any(train < 0) or np.any(train >= self.n_subjects):
            raise IndexError("Training label request is out of range")
        return self.__labels[train].copy()

    def labels_for_stage_b(self, requested_subject_ids: np.ndarray) -> np.ndarray:
        lookup = {
            int(subject_id): int(label)
            for subject_id, label in zip(self.__subject_ids, self.__labels)
        }
        requested = np.asarray(requested_subject_ids, dtype=np.int64).reshape(-1)
        missing = [int(subject_id) for subject_id in requested if int(subject_id) not in lookup]
        if missing:
            raise KeyError(f"Stage-B labels missing subject IDs: {missing[:10]}")
        return np.asarray([lookup[int(subject_id)] for subject_id in requested], dtype=np.int64)

    def copied_labels_for_split_planning_only(self) -> np.ndarray:
        """Return a copy for deterministic split verification tests only."""
        return self.__labels.copy()


@dataclass(frozen=True)
class OuterSplit:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray

    def validate(self, n_subjects: int) -> None:
        train = np.asarray(self.train_idx, dtype=np.int64)
        test = np.asarray(self.test_idx, dtype=np.int64)
        if train.ndim != 1 or test.ndim != 1:
            raise ValueError("Split indices must be one-dimensional")
        if len(np.intersect1d(train, test)) != 0:
            raise ValueError(f"Outer fold {self.fold} has train/test overlap")
        combined = np.concatenate([train, test])
        if len(np.unique(combined)) != n_subjects:
            raise ValueError(f"Outer fold {self.fold} does not partition all subjects")
        if combined.min(initial=0) < 0 or combined.max(initial=-1) >= n_subjects:
            raise IndexError(f"Outer fold {self.fold} contains out-of-range indices")

    @property
    def split_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(np.asarray(self.train_idx, dtype=np.int64).tobytes())
        digest.update(b"|")
        digest.update(np.asarray(self.test_idx, dtype=np.int64).tobytes())
        return digest.hexdigest()


def validate_outer_splits(splits: Sequence[OuterSplit], n_subjects: int) -> None:
    if len(splits) != N_OUTER:
        raise ValueError(f"Required {N_OUTER} outer folds, got {len(splits)}")
    assignment = np.zeros(n_subjects, dtype=np.int64)
    for expected_fold, split in enumerate(splits):
        if split.fold != expected_fold:
            raise ValueError("Outer folds must be ordered and zero-based")
        split.validate(n_subjects)
        assignment[split.test_idx] += 1
    if not np.all(assignment == 1):
        raise ValueError(
            f"Outer-test assignment must be exactly one; min/max={assignment.min()}/{assignment.max()}"
        )


def make_inner_splits(
    outer_split: OuterSplit,
    y_outer_train: np.ndarray,
    seed: int = DEFAULT_SEED,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    outer_train = np.asarray(outer_split.train_idx, dtype=np.int64)
    labels = np.asarray(y_outer_train, dtype=np.int64).reshape(-1)
    if len(labels) != len(outer_train):
        raise ValueError("y_outer_train must correspond exactly to outer train indices")
    splitter = StratifiedKFold(
        n_splits=N_INNER,
        shuffle=True,
        random_state=seed + 10_000 + outer_split.fold,
    )
    inner: List[Tuple[np.ndarray, np.ndarray]] = []
    for relative_fit, relative_validation in splitter.split(outer_train, labels):
        fit = outer_train[relative_fit].astype(np.int64)
        validation = outer_train[relative_validation].astype(np.int64)
        if len(np.intersect1d(fit, validation)) != 0:
            raise RuntimeError("Inner fit/validation overlap")
        if len(np.setdiff1d(np.concatenate([fit, validation]), outer_train)) != 0:
            raise RuntimeError("Inner split escaped the outer-training partition")
        if len(np.intersect1d(fit, outer_split.test_idx)) != 0:
            raise RuntimeError("Outer-test subject appeared in inner fit")
        if len(np.intersect1d(validation, outer_split.test_idx)) != 0:
            raise RuntimeError("Outer-test subject appeared in inner validation")
        inner.append((fit, validation))
    return inner


@dataclass
class DemoTransformer:
    age_impute_value: float
    sex_impute_value: float
    scaler: StandardScaler

    @classmethod
    def fit(cls, values: np.ndarray) -> "DemoTransformer":
        array = np.asarray(values, dtype=np.float64).copy()
        age_impute = (
            float(np.nanmean(array[:, 0]))
            if np.any(np.isfinite(array[:, 0]))
            else 17.0
        )
        sex_impute = (
            float(np.nanmean(array[:, 1]))
            if np.any(np.isfinite(array[:, 1]))
            else 0.0
        )
        array[:, 0] = np.where(np.isfinite(array[:, 0]), array[:, 0], age_impute)
        array[:, 1] = np.where(np.isfinite(array[:, 1]), array[:, 1], sex_impute)
        return cls(age_impute, sex_impute, StandardScaler().fit(array))

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).copy()
        array[:, 0] = np.where(
            np.isfinite(array[:, 0]), array[:, 0], self.age_impute_value
        )
        array[:, 1] = np.where(
            np.isfinite(array[:, 1]), array[:, 1], self.sex_impute_value
        )
        return self.scaler.transform(array).astype(np.float32)


def select_fc_edges(
    x_full_fc_fit: np.ndarray,
    y_fit: np.ndarray,
    count: int = N_SELECTED_EDGES,
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(x_full_fc_fit, dtype=np.float64)
    labels = np.asarray(y_fit, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] != N_FULL_EDGES:
        raise ValueError(
            f"FC selector requires [n,{N_FULL_EDGES}], got {values.shape}"
        )
    if len(values) != len(labels):
        raise ValueError("FC fit features and labels have different lengths")
    scores, _ = f_classif(values, labels)
    scores = np.nan_to_num(
        scores,
        nan=-np.inf,
        posinf=np.finfo(np.float64).max,
        neginf=-np.inf,
    )
    selected = np.argsort(-scores, kind="mergesort")[:count].astype(np.int64)
    if selected.shape != (count,) or len(np.unique(selected)) != count:
        raise RuntimeError("FC selector did not produce exactly unique requested edges")
    return selected, scores.astype(np.float32)


def load_yeo7_nodes(path: str | Path = YEO_MAPPING_PATH) -> Dict[int, np.ndarray]:
    mapping_path = Path(path)
    nodes: Dict[int, List[int]] = {network: [] for network in range(1, 8)}
    with mapping_path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            roi = int(row["roi_index"])
            network = int(row["yeo_network"])
            if 0 <= roi < N_ROI and 1 <= network <= 7:
                nodes[network].append(roi)
    result = {
        network: np.asarray(indices, dtype=np.int64)
        for network, indices in nodes.items()
    }
    if any(len(indices) == 0 for indices in result.values()):
        raise RuntimeError("Yeo-7 mapping contains an empty network")
    return result


def build_frequency_anchors(
    x_frequency_fit_scaled: np.ndarray,
    yeo_nodes: Dict[int, np.ndarray],
    hidden_dim: int,
    projection_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    nodes = np.asarray(x_frequency_fit_scaled, dtype=np.float32).reshape(
        -1, N_ROI, N_FREQUENCY_BINS
    )
    profiles = np.stack(
        [
            nodes[:, yeo_nodes[network], :].mean(axis=(0, 1))
            for network in range(1, 8)
        ]
    ).astype(np.float32)
    generator = np.random.default_rng(projection_seed)
    projection = (
        generator.standard_normal((N_FREQUENCY_BINS, hidden_dim)).astype(np.float32)
        / math.sqrt(float(N_FREQUENCY_BINS))
    )
    # This is a deliberately small 7 x 15 x hidden_dim contraction.  Using an
    # explicit einsum avoids delegating the tiny operation to a process-global
    # BLAS runtime, which also keeps preprocessing state local and deterministic
    # on Windows environments with multiple numerical runtimes installed.
    anchors = np.einsum(
        "nf,fh->nh",
        profiles,
        projection,
        optimize=False,
    ).astype(np.float32)
    return anchors, profiles


@dataclass
class PreparedPartition:
    fit_indices: np.ndarray
    evaluation_indices: np.ndarray
    selected_edges: np.ndarray
    f_scores: np.ndarray
    scaler_fc: StandardScaler
    scaler_frequency: StandardScaler
    demographic_transformer: DemoTransformer
    anchors: np.ndarray
    anchor_profiles: np.ndarray
    x_fc_fit: np.ndarray
    x_frequency_fit: np.ndarray
    x_demographic_fit: np.ndarray
    x_fc_evaluation: np.ndarray
    x_frequency_evaluation: np.ndarray
    x_demographic_evaluation: np.ndarray
    state_token: str


def fit_partition_preprocessing(
    features: FeatureBundle,
    fit_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    y_fit: np.ndarray,
    *,
    hidden_dim: int,
    anchor_projection_seed: int,
    yeo_nodes: Dict[int, np.ndarray],
) -> PreparedPartition:
    fit_idx = np.asarray(fit_indices, dtype=np.int64)
    evaluation_idx = np.asarray(evaluation_indices, dtype=np.int64)
    labels = np.asarray(y_fit, dtype=np.int64).reshape(-1)
    if len(fit_idx) != len(labels):
        raise ValueError("y_fit must contain labels for fit_indices only")
    if len(np.intersect1d(fit_idx, evaluation_idx)) != 0:
        raise RuntimeError("Preprocessing fit/evaluation partitions overlap")
    selected, scores = select_fc_edges(features.full_fc[fit_idx], labels)
    if len(selected) != N_SELECTED_EDGES:
        raise RuntimeError("Required FC reduction 19,900 -> 4,975 was not achieved")

    scaler_fc = StandardScaler().fit(
        np.asarray(features.full_fc[fit_idx][:, selected], dtype=np.float32)
    )
    scaler_frequency = StandardScaler().fit(features.frequency[fit_idx])
    demographic_transformer = DemoTransformer.fit(features.demographics_raw[fit_idx])

    x_fc_fit = scaler_fc.transform(
        np.asarray(features.full_fc[fit_idx][:, selected], dtype=np.float32)
    ).astype(np.float32)
    x_fc_evaluation = scaler_fc.transform(
        np.asarray(features.full_fc[evaluation_idx][:, selected], dtype=np.float32)
    ).astype(np.float32)
    x_frequency_fit = scaler_frequency.transform(
        features.frequency[fit_idx]
    ).astype(np.float32)
    x_frequency_evaluation = scaler_frequency.transform(
        features.frequency[evaluation_idx]
    ).astype(np.float32)
    x_demographic_fit = demographic_transformer.transform(
        features.demographics_raw[fit_idx]
    )
    x_demographic_evaluation = demographic_transformer.transform(
        features.demographics_raw[evaluation_idx]
    )
    anchors, profiles = build_frequency_anchors(
        x_frequency_fit,
        yeo_nodes,
        hidden_dim,
        anchor_projection_seed,
    )
    return PreparedPartition(
        fit_indices=fit_idx.copy(),
        evaluation_indices=evaluation_idx.copy(),
        selected_edges=selected,
        f_scores=scores,
        scaler_fc=scaler_fc,
        scaler_frequency=scaler_frequency,
        demographic_transformer=demographic_transformer,
        anchors=anchors,
        anchor_profiles=profiles,
        x_fc_fit=x_fc_fit,
        x_frequency_fit=x_frequency_fit,
        x_demographic_fit=x_demographic_fit,
        x_fc_evaluation=x_fc_evaluation,
        x_frequency_evaluation=x_frequency_evaluation,
        x_demographic_evaluation=x_demographic_evaluation,
        state_token=uuid.uuid4().hex,
    )


def save_preprocessing_artifacts(
    directory: Path,
    prepared: PreparedPartition,
    *,
    scope: str,
    outer_test_indices: np.ndarray,
) -> Dict[str, Any]:
    ensure_directory(directory)
    if len(np.intersect1d(prepared.fit_indices, outer_test_indices)) != 0:
        raise RuntimeError("Outer-test index found in preprocessing fit indices")
    np.save(directory / "fit_indices.npy", prepared.fit_indices)
    np.save(directory / "evaluation_indices.npy", prepared.evaluation_indices)
    np.save(directory / "selected_full_edge_indices.npy", prepared.selected_edges)
    np.savez(
        directory / "preprocessing_state.npz",
        f_scores=prepared.f_scores,
        fc_mean=prepared.scaler_fc.mean_,
        fc_scale=prepared.scaler_fc.scale_,
        frequency_mean=prepared.scaler_frequency.mean_,
        frequency_scale=prepared.scaler_frequency.scale_,
        demographic_impute=np.asarray(
            [
                prepared.demographic_transformer.age_impute_value,
                prepared.demographic_transformer.sex_impute_value,
            ],
            dtype=np.float64,
        ),
        demographic_mean=prepared.demographic_transformer.scaler.mean_,
        demographic_scale=prepared.demographic_transformer.scaler.scale_,
        frequency_anchors=prepared.anchors,
        anchor_profiles=prepared.anchor_profiles,
    )
    manifest = {
        "scope": scope,
        "fit_indices_sha256": sha256_indices(prepared.fit_indices),
        "evaluation_indices_sha256": sha256_indices(prepared.evaluation_indices),
        "fit_evaluation_overlap": int(
            len(np.intersect1d(prepared.fit_indices, prepared.evaluation_indices))
        ),
        "fit_outer_test_overlap": int(
            len(np.intersect1d(prepared.fit_indices, outer_test_indices))
        ),
        "fc_input_dimension": N_FULL_EDGES,
        "fc_selected_dimension": int(len(prepared.selected_edges)),
        "selected_edges_sha256": sha256_indices(prepared.selected_edges),
        "feature_selection_fit_scope": "fit_indices_only",
        "scaler_fit_scope": "fit_indices_only",
        "frequency_anchor_fit_scope": "fit_indices_only_after_fit_only_scaling",
        "runtime_state_token": prepared.state_token,
    }
    write_json(directory / "preprocessing_manifest.json", manifest)
    return manifest


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 300
    eval_every: int = 5
    hidden_dim: int = 80
    align_dim: int = 160
    dropout_sc: float = 0.25
    dropout_cls: float = 0.25
    num_res_blocks: int = 4
    learning_rate: float = 0.0008
    weight_decay: float = 0.005
    batch_size: int = 16
    kl_weight: float = 0.002
    anchor_weight: float = 0.30
    contrastive_weight: float = 0.10
    contrastive_temperature: float = 0.10
    focal_gamma: float = 1.0
    mixup_alpha: float = 0.10
    noise_std: float = 0.01
    grad_clip: float = 1.0
    lr_warmup: int = 60
    tau_warmup: int = 100
    tau_high: float = 1.0
    tau_low: float = 0.15
    anchor_projection_seed: int = 20250225

    def validate(self) -> None:
        if self.epochs < 1 or self.eval_every < 1:
            raise ValueError("epochs and eval_every must be positive")
        if self.hidden_dim < 1 or self.align_dim < 1:
            raise ValueError("model dimensions must be positive")


@dataclass
class InnerTrainingResult:
    best_epoch: int
    best_validation_auc: float
    checkpoint_state_sha256: str
    state_payload: Any
    model_instance_token: str
    optimizer_instance_token: str


@dataclass
class FinalTrainingResult:
    fixed_epoch: int
    probabilities: np.ndarray
    checkpoint_state_sha256: str
    state_payload: Any
    model_instance_token: str
    optimizer_instance_token: str


class TrainingBackend(Protocol):
    def train_inner(
        self,
        prepared: PreparedPartition,
        y_fit: np.ndarray,
        y_validation: np.ndarray,
        config: TrainingConfig,
        seed: int,
    ) -> InnerTrainingResult:
        ...

    def train_fixed(
        self,
        prepared: PreparedPartition,
        y_fit: np.ndarray,
        fixed_epoch: int,
        config: TrainingConfig,
        seed: int,
    ) -> FinalTrainingResult:
        ...

    def save_checkpoint(
        self, path: Path, state_payload: Any, metadata: Dict[str, Any]
    ) -> None:
        ...


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def state_dict_sha256(state_dict: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def safe_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(y_true, dtype=np.int64)
    if len(np.unique(labels)) != 2:
        raise ValueError("AUC checkpoint selection requires both classes")
    return float(roc_auc_score(labels, probabilities))


def legacy_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    normalized = F.normalize(embeddings, p=2, dim=1)
    similarity = torch.matmul(normalized, normalized.t()) / temperature
    same_label = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
    return -(same_label * F.log_softmax(similarity, dim=1)).mean()


class TorchTrainingBackend:
    """Fresh model and optimizer per inner trial and outer-final fit."""

    def __init__(self, device: torch.device):
        self.device = device
        self._lifecycle_counter = 0

    def _fresh_model_optimizer(
        self,
        prepared: PreparedPartition,
        config: TrainingConfig,
        seed: int,
    ) -> Tuple[
        BrainInnovationSystem,
        torch.optim.Optimizer,
        str,
        str,
    ]:
        set_all_seeds(seed)
        self._lifecycle_counter += 1
        lifecycle = self._lifecycle_counter
        model = BrainInnovationSystem(
            space_dim=N_SELECTED_EDGES,
            prior_anchors=torch.from_numpy(prepared.anchors).to(self.device),
            hidden_dim=config.hidden_dim,
            align_dim=config.align_dim,
            dropout_sc=config.dropout_sc,
            dropout_cls=config.dropout_cls,
            num_res_blocks=config.num_res_blocks,
        ).to(self.device)
        model.set_eval_ocread_mode("mu")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        return (
            model,
            optimizer,
            f"torch-model-{lifecycle:06d}",
            f"torch-optimizer-{lifecycle:06d}",
        )

    def _predict(
        self, model: BrainInnovationSystem, prepared: PreparedPartition
    ) -> np.ndarray:
        model.eval()
        model.set_eval_ocread_mode("mu")
        batches: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(prepared.x_fc_evaluation), 64):
                stop = min(start + 64, len(prepared.x_fc_evaluation))
                logits, _, _, _ = model(
                    torch.from_numpy(prepared.x_fc_evaluation[start:stop]).to(
                        self.device
                    ),
                    torch.from_numpy(
                        prepared.x_frequency_evaluation[start:stop]
                    ).to(self.device),
                    torch.from_numpy(
                        prepared.x_demographic_evaluation[start:stop]
                    ).to(self.device),
                    tau=configured_tau_for_prediction(),
                    training=False,
                )
                batches.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(batches).astype(np.float64)

    def _train_epoch(
        self,
        model: BrainInnovationSystem,
        optimizer: torch.optim.Optimizer,
        loader: DataLoader,
        config: TrainingConfig,
        epoch: int,
    ) -> None:
        if epoch <= config.lr_warmup:
            for group in optimizer.param_groups:
                group["lr"] = (
                    config.learning_rate * epoch / max(config.lr_warmup, 1)
                )
        tau = config.tau_high if epoch <= config.tau_warmup else config.tau_low
        model.train()
        for x_fc, x_frequency, x_demographic, labels in loader:
            x_fc = x_fc.to(self.device)
            x_frequency = x_frequency.to(self.device)
            x_demographic = x_demographic.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad(set_to_none=True)
            use_mixup = config.mixup_alpha > 0 and epoch > config.lr_warmup
            if use_mixup:
                mixing = float(
                    np.random.beta(config.mixup_alpha, config.mixup_alpha)
                )
                permutation = torch.randperm(len(labels), device=self.device)
                input_fc = mixing * x_fc + (1.0 - mixing) * x_fc[permutation]
                input_frequency = (
                    mixing * x_frequency
                    + (1.0 - mixing) * x_frequency[permutation]
                )
                input_demographic = (
                    mixing * x_demographic
                    + (1.0 - mixing) * x_demographic[permutation]
                )
                label_a, label_b = labels, labels[permutation]
            else:
                mixing = 1.0
                input_fc = x_fc
                input_frequency = x_frequency
                input_demographic = x_demographic
                label_a = labels
                label_b = labels
            logits, fused, mu, logvar = model(
                input_fc,
                input_frequency,
                input_demographic,
                tau=tau,
                training=True,
                noise_std=config.noise_std,
            )
            bce_a = F.binary_cross_entropy_with_logits(
                logits, label_a, reduction="none"
            )
            bce_b = F.binary_cross_entropy_with_logits(
                logits, label_b, reduction="none"
            )
            weight_a = (1.0 - torch.exp(-bce_a)).pow(config.focal_gamma)
            weight_b = (1.0 - torch.exp(-bce_b)).pow(config.focal_gamma)
            classification = (
                mixing * (weight_a * bce_a).mean()
                + (1.0 - mixing) * (weight_b * bce_b).mean()
            )
            kl = (
                -0.5
                * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                * config.kl_weight
            )
            anchor = (
                F.mse_loss(mu, model.ocread.initial_anchors)
                * config.anchor_weight
            )
            if config.contrastive_weight > 0 and not use_mixup:
                contrastive = (
                    legacy_contrastive_loss(
                        fused, labels, config.contrastive_temperature
                    )
                    * config.contrastive_weight
                )
            else:
                contrastive = torch.zeros((), device=self.device)
            loss = classification + kl + anchor + contrastive
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

    @staticmethod
    def _loader(
        prepared: PreparedPartition,
        labels: np.ndarray,
        batch_size: int,
        seed: int,
    ) -> DataLoader:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        return DataLoader(
            TensorDataset(
                torch.from_numpy(prepared.x_fc_fit),
                torch.from_numpy(prepared.x_frequency_fit),
                torch.from_numpy(prepared.x_demographic_fit),
                torch.from_numpy(np.asarray(labels, dtype=np.float32)),
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )

    def train_inner(
        self,
        prepared: PreparedPartition,
        y_fit: np.ndarray,
        y_validation: np.ndarray,
        config: TrainingConfig,
        seed: int,
    ) -> InnerTrainingResult:
        model, optimizer, model_token, optimizer_token = (
            self._fresh_model_optimizer(prepared, config, seed)
        )
        loader = self._loader(prepared, y_fit, config.batch_size, seed)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(config.epochs - config.lr_warmup, 1)
        )
        best_auc = -float("inf")
        best_epoch = 0
        best_state: Optional[Dict[str, torch.Tensor]] = None
        for epoch in range(1, config.epochs + 1):
            self._train_epoch(model, optimizer, loader, config, epoch)
            if epoch > config.lr_warmup:
                scheduler.step()
            if epoch % config.eval_every == 0 or epoch == config.epochs:
                probabilities = self._predict(model, prepared)
                validation_auc = safe_auc(y_validation, probabilities)
                if validation_auc > best_auc + 1e-12:
                    best_auc = validation_auc
                    best_epoch = epoch
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
        if best_state is None:
            raise RuntimeError("No inner-validation checkpoint was selected")
        return InnerTrainingResult(
            best_epoch=best_epoch,
            best_validation_auc=best_auc,
            checkpoint_state_sha256=state_dict_sha256(best_state),
            state_payload=best_state,
            model_instance_token=model_token,
            optimizer_instance_token=optimizer_token,
        )

    def train_fixed(
        self,
        prepared: PreparedPartition,
        y_fit: np.ndarray,
        fixed_epoch: int,
        config: TrainingConfig,
        seed: int,
    ) -> FinalTrainingResult:
        if fixed_epoch < 1:
            raise ValueError("fixed_epoch must be positive")
        model, optimizer, model_token, optimizer_token = (
            self._fresh_model_optimizer(prepared, config, seed)
        )
        loader = self._loader(prepared, y_fit, config.batch_size, seed)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(fixed_epoch - config.lr_warmup, 1)
        )
        for epoch in range(1, fixed_epoch + 1):
            self._train_epoch(model, optimizer, loader, config, epoch)
            if epoch > config.lr_warmup:
                scheduler.step()
        state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        probabilities = self._predict(model, prepared)
        return FinalTrainingResult(
            fixed_epoch=fixed_epoch,
            probabilities=probabilities,
            checkpoint_state_sha256=state_dict_sha256(state),
            state_payload=state,
            model_instance_token=model_token,
            optimizer_instance_token=optimizer_token,
        )

    def save_checkpoint(
        self, path: Path, state_payload: Any, metadata: Dict[str, Any]
    ) -> None:
        ensure_directory(path.parent)
        torch.save(
            {
                "model_state_dict": state_payload,
                "metadata": metadata,
            },
            path,
        )


def configured_tau_for_prediction() -> float:
    return 0.15


def labels_for_global_indices(
    outer_train_indices: np.ndarray,
    y_outer_train: np.ndarray,
    requested_indices: np.ndarray,
) -> np.ndarray:
    positions = {
        int(global_index): position
        for position, global_index in enumerate(
            np.asarray(outer_train_indices, dtype=np.int64)
        )
    }
    requested = np.asarray(requested_indices, dtype=np.int64)
    missing = [int(index) for index in requested if int(index) not in positions]
    if missing:
        raise RuntimeError(
            f"Requested labels outside outer-training partition: {missing[:10]}"
        )
    labels = np.asarray(y_outer_train, dtype=np.int64)
    return np.asarray(
        [labels[positions[int(index)]] for index in requested],
        dtype=np.int64,
    )


def save_frozen_predictions(
    path: Path,
    *,
    subject_ids: np.ndarray,
    row_indices: np.ndarray,
    probabilities: np.ndarray,
    outer_folds: np.ndarray,
) -> Dict[str, Any]:
    ensure_directory(path.parent)
    arrays = {
        "subject_ids": np.asarray(subject_ids, dtype=np.int64),
        "row_indices": np.asarray(row_indices, dtype=np.int64),
        "probabilities": np.asarray(probabilities, dtype=np.float64),
        "outer_folds": np.asarray(outer_folds, dtype=np.int64),
        "classification_threshold": np.asarray(
            [CLASSIFICATION_THRESHOLD], dtype=np.float64
        ),
    }
    if any("label" in key.casefold() or key.casefold() == "y" for key in arrays):
        raise RuntimeError("Stage-A frozen prediction keys must not contain labels")
    np.savez(path, **arrays)
    return {
        "prediction_file": str(path),
        "prediction_sha256": sha256_file(path),
        "contains_true_test_labels": False,
        "keys": sorted(arrays),
        "n_predictions": int(len(arrays["probabilities"])),
        "threshold": CLASSIFICATION_THRESHOLD,
    }


def run_outer_fold_stage_a(
    *,
    features: FeatureBundle,
    outer_split: OuterSplit,
    y_outer_train: np.ndarray,
    yeo_nodes: Dict[int, np.ndarray],
    config: TrainingConfig,
    backend: TrainingBackend,
    fold_directory: Path,
    seed: int,
) -> Dict[str, Any]:
    """Run one outer fold without accepting or accessing outer-test labels."""
    outer_split.validate(len(features.subject_ids))
    y_train = np.asarray(y_outer_train, dtype=np.int64).reshape(-1)
    if len(y_train) != len(outer_split.train_idx):
        raise ValueError("Outer-training labels do not match outer-training indices")
    ensure_directory(fold_directory)
    inner_rows: List[Dict[str, Any]] = []
    lifecycle_tokens: List[str] = []
    preprocessing_tokens: List[str] = []

    for inner_fold, (fit_idx, validation_idx) in enumerate(
        make_inner_splits(outer_split, y_train, seed)
    ):
        y_fit = labels_for_global_indices(
            outer_split.train_idx, y_train, fit_idx
        )
        y_validation = labels_for_global_indices(
            outer_split.train_idx, y_train, validation_idx
        )
        prepared = fit_partition_preprocessing(
            features,
            fit_idx,
            validation_idx,
            y_fit,
            hidden_dim=config.hidden_dim,
            anchor_projection_seed=config.anchor_projection_seed,
            yeo_nodes=yeo_nodes,
        )
        trial_directory = fold_directory / "inner_cv" / f"inner_{inner_fold:02d}"
        preprocessing_manifest = save_preprocessing_artifacts(
            trial_directory / "preprocessing",
            prepared,
            scope="inner_fit_only",
            outer_test_indices=outer_split.test_idx,
        )
        result = backend.train_inner(
            prepared,
            y_fit,
            y_validation,
            config,
            seed + outer_split.fold * 1_000 + inner_fold * 100,
        )
        checkpoint_path = trial_directory / "best_inner_checkpoint.pt"
        checkpoint_metadata = {
            "selection_policy": "inner_validation_auc_only",
            "best_epoch": result.best_epoch,
            "best_validation_auc": result.best_validation_auc,
            "fit_indices_sha256": sha256_indices(fit_idx),
            "validation_indices_sha256": sha256_indices(validation_idx),
            "outer_test_labels_available_to_training": False,
        }
        backend.save_checkpoint(
            checkpoint_path, result.state_payload, checkpoint_metadata
        )
        row = {
            "inner_fold": inner_fold,
            "best_epoch": int(result.best_epoch),
            "best_validation_auc": float(result.best_validation_auc),
            "checkpoint_selection_policy": "inner_validation_auc_only",
            "checkpoint_state_sha256": result.checkpoint_state_sha256,
            "checkpoint_file_sha256": sha256_file(checkpoint_path),
            "fit_indices_sha256": sha256_indices(fit_idx),
            "validation_indices_sha256": sha256_indices(validation_idx),
            "n_fit": int(len(fit_idx)),
            "n_validation": int(len(validation_idx)),
            "preprocessing": preprocessing_manifest,
            "model_instance_token": result.model_instance_token,
            "optimizer_instance_token": result.optimizer_instance_token,
        }
        write_json(trial_directory / "inner_result.json", row)
        inner_rows.append(row)
        lifecycle_tokens.extend(
            [result.model_instance_token, result.optimizer_instance_token]
        )
        preprocessing_tokens.append(prepared.state_token)

    inner_epochs = np.asarray(
        [row["best_epoch"] for row in inner_rows], dtype=np.int64
    )
    if inner_epochs.shape != (N_INNER,):
        raise RuntimeError("Exactly three inner best epochs are required")
    fixed_epoch = int(np.median(inner_epochs))

    outer_prepared = fit_partition_preprocessing(
        features,
        outer_split.train_idx,
        outer_split.test_idx,
        y_train,
        hidden_dim=config.hidden_dim,
        anchor_projection_seed=config.anchor_projection_seed,
        yeo_nodes=yeo_nodes,
    )
    outer_preprocessing = save_preprocessing_artifacts(
        fold_directory / "outer_final_preprocessing",
        outer_prepared,
        scope="outer_training_only",
        outer_test_indices=outer_split.test_idx,
    )
    final_result = backend.train_fixed(
        outer_prepared,
        y_train,
        fixed_epoch,
        config,
        seed + outer_split.fold * 1_000 + 777,
    )
    lifecycle_tokens.extend(
        [
            final_result.model_instance_token,
            final_result.optimizer_instance_token,
        ]
    )
    preprocessing_tokens.append(outer_prepared.state_token)
    if len(set(lifecycle_tokens)) != len(lifecycle_tokens):
        raise RuntimeError("Model or optimizer state token was reused within a fold")
    if len(set(preprocessing_tokens)) != len(preprocessing_tokens):
        raise RuntimeError("Preprocessing state was reused within a fold")

    final_checkpoint_path = fold_directory / "outer_final_checkpoint.pt"
    final_checkpoint_metadata = {
        "training_policy": "fresh_outer_model_fixed_inner_median_epoch",
        "fixed_epoch": fixed_epoch,
        "outer_train_indices_sha256": sha256_indices(outer_split.train_idx),
        "outer_test_indices_sha256": sha256_indices(outer_split.test_idx),
        "outer_test_labels_available_to_training": False,
    }
    backend.save_checkpoint(
        final_checkpoint_path,
        final_result.state_payload,
        final_checkpoint_metadata,
    )

    frozen_path = fold_directory / "frozen_outer_test_predictions.npz"
    frozen_manifest = save_frozen_predictions(
        frozen_path,
        subject_ids=features.subject_ids[outer_split.test_idx],
        row_indices=outer_split.test_idx,
        probabilities=final_result.probabilities,
        outer_folds=np.full(
            len(outer_split.test_idx), outer_split.fold, dtype=np.int64
        ),
    )
    write_json(fold_directory / "frozen_prediction_manifest.json", frozen_manifest)

    summary = {
        "stage": "A_FROZEN_PREDICTION",
        "outer_fold": outer_split.fold,
        "outer_split_sha256": outer_split.split_sha256,
        "outer_train_indices": outer_split.train_idx.tolist(),
        "outer_test_indices": outer_split.test_idx.tolist(),
        "outer_train_subject_ids": features.subject_ids[
            outer_split.train_idx
        ].astype(int).tolist(),
        "outer_test_subject_ids": features.subject_ids[
            outer_split.test_idx
        ].astype(int).tolist(),
        "outer_train_test_overlap": int(
            len(np.intersect1d(outer_split.train_idx, outer_split.test_idx))
        ),
        "inner_results": inner_rows,
        "inner_best_epochs": inner_epochs.astype(int).tolist(),
        "outer_fixed_epoch": fixed_epoch,
        "outer_epoch_rule": "median_of_three_inner_best_epochs",
        "outer_final_preprocessing": outer_preprocessing,
        "outer_checkpoint_state_sha256": final_result.checkpoint_state_sha256,
        "outer_checkpoint_file_sha256": sha256_file(final_checkpoint_path),
        "outer_model_instance_token": final_result.model_instance_token,
        "outer_optimizer_instance_token": final_result.optimizer_instance_token,
        "frozen_prediction": frozen_manifest,
        "fixed_classification_threshold": CLASSIFICATION_THRESHOLD,
        "outer_test_labels_used": False,
        "outer_metrics_computed": False,
    }
    write_json(fold_directory / "stage_a_fold_summary.json", summary)
    return {
        **summary,
        "probabilities": np.asarray(final_result.probabilities, dtype=np.float64),
        "row_indices_array": outer_split.test_idx.copy(),
        "subject_ids_array": features.subject_ids[outer_split.test_idx].copy(),
        "preprocessing_state_tokens": preprocessing_tokens,
        "lifecycle_tokens": lifecycle_tokens,
    }


def run_stage_a(
    *,
    features: FeatureBundle,
    label_firewall: LabelFirewall,
    output_directory: Path,
    config: TrainingConfig,
    backend: TrainingBackend,
    outer_splits: Optional[Sequence[OuterSplit]] = None,
    folds_to_run: Optional[Sequence[int]] = None,
    yeo_nodes: Optional[Dict[int, np.ndarray]] = None,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Create frozen predictions; never compute outer-test metrics."""
    features.validate()
    config.validate()
    if label_firewall.n_subjects != len(features.subject_ids):
        raise ValueError("Feature and label subject counts differ")
    splits = (
        list(outer_splits)
        if outer_splits is not None
        else label_firewall.make_outer_splits(seed)
    )
    validate_outer_splits(splits, len(features.subject_ids))
    requested_folds = (
        list(range(N_OUTER))
        if folds_to_run is None
        else sorted(set(int(fold) for fold in folds_to_run))
    )
    if not requested_folds or any(fold < 0 or fold >= N_OUTER for fold in requested_folds):
        raise ValueError("folds_to_run must be a non-empty subset of 0..9")
    nodes = yeo_nodes if yeo_nodes is not None else load_yeo7_nodes()
    ensure_new_directory(output_directory)

    protocol = {
        "protocol": "CPAC_LEAKAGE_FREE_NESTED_CV",
        "created_at": now_iso(),
        "outer_folds": N_OUTER,
        "inner_folds": N_INNER,
        "requested_outer_folds": requested_folds,
        "full_fc_dimension": N_FULL_EDGES,
        "selected_fc_dimension": N_SELECTED_EDGES,
        "checkpoint_selection": "inner_validation_auc_only",
        "outer_epoch_selection": "median_inner_best_epoch",
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "configuration_level_fusion": False,
        "outer_test_label_firewall": (
            "Only outer-training labels are copied from LabelFirewall into "
            "fold training. Preprocessing, backend training, and prediction APIs "
            "do not accept outer-test labels."
        ),
        "stage_a_contains_test_labels": False,
        "training_config": asdict(config),
        "feature_shapes": {
            "full_fc": list(features.full_fc.shape),
            "frequency": list(features.frequency.shape),
            "demographics_raw": list(features.demographics_raw.shape),
            "subject_ids": list(features.subject_ids.shape),
        },
        "source_files": features.source_files,
        "forbidden_legacy_input_policy": sorted(FORBIDDEN_EXACT_INPUTS),
    }
    write_json(output_directory / "stage_a_protocol.json", protocol)
    write_json(
        output_directory / "outer_splits.json",
        {
            "outer_folds": [
                {
                    "fold": split.fold,
                    "train_indices": split.train_idx.tolist(),
                    "test_indices": split.test_idx.tolist(),
                    "split_sha256": split.split_sha256,
                }
                for split in splits
            ]
        },
    )

    fold_results: List[Dict[str, Any]] = []
    global_lifecycle_tokens: List[str] = []
    global_preprocessing_tokens: List[str] = []
    for fold in requested_folds:
        split = splits[fold]
        y_outer_train = label_firewall.training_labels(
            split.train_idx, split.test_idx
        )
        result = run_outer_fold_stage_a(
            features=features,
            outer_split=split,
            y_outer_train=y_outer_train,
            yeo_nodes=nodes,
            config=config,
            backend=backend,
            fold_directory=output_directory / f"fold_{fold:02d}",
            seed=seed,
        )
        fold_results.append(result)
        global_lifecycle_tokens.extend(result["lifecycle_tokens"])
        global_preprocessing_tokens.extend(result["preprocessing_state_tokens"])

    if len(set(global_lifecycle_tokens)) != len(global_lifecycle_tokens):
        raise RuntimeError("Model or optimizer state was reused across folds")
    if len(set(global_preprocessing_tokens)) != len(global_preprocessing_tokens):
        raise RuntimeError("Preprocessing state was reused across folds")

    assignment = np.zeros(len(features.subject_ids), dtype=np.int64)
    probabilities = np.full(len(features.subject_ids), np.nan, dtype=np.float64)
    fold_ids = np.full(len(features.subject_ids), -1, dtype=np.int64)
    for result in fold_results:
        indices = result["row_indices_array"]
        assignment[indices] += 1
        probabilities[indices] = result["probabilities"]
        fold_ids[indices] = int(result["outer_fold"])
    evaluated = np.flatnonzero(assignment)
    if np.any(assignment[evaluated] != 1):
        raise RuntimeError("OOF prediction assignment is not unique")
    if np.any(~np.isfinite(probabilities[evaluated])):
        raise RuntimeError("Frozen predictions contain NaN or Inf")
    full_run = requested_folds == list(range(N_OUTER))
    if full_run and not np.all(assignment == 1):
        raise RuntimeError("Full Stage-A run did not predict every subject exactly once")

    aggregate_name = (
        "frozen_oof_predictions.npz"
        if full_run
        else "frozen_partial_predictions.npz"
    )
    aggregate_path = output_directory / aggregate_name
    frozen_manifest = save_frozen_predictions(
        aggregate_path,
        subject_ids=features.subject_ids[evaluated],
        row_indices=evaluated,
        probabilities=probabilities[evaluated],
        outer_folds=fold_ids[evaluated],
    )
    frozen_manifest.update(
        {
            "full_10_fold_run": full_run,
            "prediction_assignment_min": int(assignment[evaluated].min()),
            "prediction_assignment_max": int(assignment[evaluated].max()),
        }
    )
    write_json(output_directory / "frozen_oof_manifest.json", frozen_manifest)
    summary = {
        "stage": "A_COMPLETE_FROZEN_PREDICTIONS",
        "full_10_fold_run": full_run,
        "requested_outer_folds": requested_folds,
        "n_frozen_predictions": int(len(evaluated)),
        "frozen_prediction": frozen_manifest,
        "outer_test_labels_written": False,
        "outer_metrics_computed": False,
        "all_model_optimizer_tokens_unique": True,
        "all_preprocessing_tokens_unique": True,
    }
    write_json(output_directory / "stage_a_summary.json", summary)
    return {
        **summary,
        "fold_results": fold_results,
        "prediction_path": aggregate_path,
        "manifest_path": output_directory / "frozen_oof_manifest.json",
        "assignment": assignment,
        "probabilities": probabilities,
    }


def verify_frozen_prediction(
    prediction_path: str | Path, manifest_path: str | Path
) -> Dict[str, np.ndarray]:
    prediction = Path(prediction_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    actual_sha = sha256_file(prediction)
    if actual_sha != manifest["prediction_sha256"]:
        raise RuntimeError(
            f"Frozen prediction SHA-256 mismatch: {actual_sha} != "
            f"{manifest['prediction_sha256']}"
        )
    with np.load(prediction, allow_pickle=False) as payload:
        keys = set(payload.files)
        if any("label" in key.casefold() or key.casefold() == "y" for key in keys):
            raise RuntimeError("Frozen Stage-A prediction contains a label field")
        required = {
            "subject_ids",
            "row_indices",
            "probabilities",
            "outer_folds",
            "classification_threshold",
        }
        if keys != required:
            raise RuntimeError(f"Unexpected frozen prediction keys: {sorted(keys)}")
        return {key: np.asarray(payload[key]).copy() for key in payload.files}


def compute_fixed_threshold_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> Dict[str, Any]:
    y_true = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if len(y_true) != len(scores):
        raise ValueError("Labels and probabilities differ in length")
    predictions = (scores >= CLASSIFICATION_THRESHOLD).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    ).ravel()
    return {
        "n": int(len(y_true)),
        "threshold": CLASSIFICATION_THRESHOLD,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "auc": safe_auc(y_true, scores),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier": float(brier_score_loss(y_true, scores)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def run_stage_b(
    *,
    prediction_path: str | Path,
    manifest_path: str | Path,
    label_firewall: LabelFirewall,
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Verify frozen predictions, then and only then load labels for metrics."""
    frozen = verify_frozen_prediction(prediction_path, manifest_path)
    labels = label_firewall.labels_for_stage_b(frozen["subject_ids"])
    metrics = compute_fixed_threshold_metrics(labels, frozen["probabilities"])
    result = {
        "stage": "B_LABELLED_EVALUATION",
        "prediction_file": str(Path(prediction_path)),
        "verified_prediction_sha256": sha256_file(Path(prediction_path)),
        "fixed_threshold": CLASSIFICATION_THRESHOLD,
        "metrics": metrics,
    }
    if output_path is not None:
        destination = Path(output_path)
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite Stage-B output: {destination}")
        write_json(destination, result)
    return result


def read_phenotype_records(path: Path) -> Dict[int, Dict[str, Any]]:
    records: Dict[int, Dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            try:
                subject_id = int(float(row["SUB_ID"]))
                diagnosis = int(float(row["DX_GROUP"]))
            except (KeyError, TypeError, ValueError):
                continue
            try:
                age = float(row.get("AGE_AT_SCAN", ""))
            except (TypeError, ValueError):
                age = float("nan")
            try:
                sex = float(int(float(row.get("SEX", ""))) == 1)
            except (TypeError, ValueError):
                sex = float("nan")
            records[subject_id] = {
                "label": 1 if diagnosis == 1 else 0,
                "age": age,
                "sex": sex,
            }
    if not records:
        raise RuntimeError(f"No phenotype records loaded from {path}")
    return records


def load_feature_bundle(data_directory: str | Path = DATA_DIR) -> FeatureBundle:
    data_path = Path(data_directory)
    full_fc_path = data_path / FULL_FC_FILENAME
    frequency_path = data_path / FREQUENCY_FILENAME
    subject_ids_path = data_path / SUBJECT_IDS_FILENAME
    phenotype_path = data_path / PHENOTYPE_FILENAME
    full_fc = safe_load_array(full_fc_path, mmap_mode="r")
    frequency = safe_load_array(frequency_path).astype(np.float32)
    subject_ids = safe_load_array(subject_ids_path).astype(np.int64)
    phenotype = read_phenotype_records(phenotype_path)
    missing = [
        int(subject_id)
        for subject_id in subject_ids
        if int(subject_id) not in phenotype
    ]
    if missing:
        raise RuntimeError(f"Phenotype missing strict subject IDs: {missing[:10]}")
    demographics = np.asarray(
        [
            [
                phenotype[int(subject_id)]["age"],
                phenotype[int(subject_id)]["sex"],
            ]
            for subject_id in subject_ids
        ],
        dtype=np.float32,
    )
    bundle = FeatureBundle(
        full_fc=full_fc,
        frequency=frequency,
        demographics_raw=demographics,
        subject_ids=subject_ids,
        source_files={
            "full_fc": str(full_fc_path),
            "frequency": str(frequency_path),
            "subject_ids": str(subject_ids_path),
            "demographics_raw": str(phenotype_path),
        },
    )
    bundle.validate()
    return bundle


def load_label_firewall(
    data_directory: str | Path = DATA_DIR,
) -> LabelFirewall:
    data_path = Path(data_directory)
    labels_path = data_path / LABEL_FILENAME
    subject_ids_path = data_path / SUBJECT_IDS_FILENAME
    phenotype_path = data_path / PHENOTYPE_FILENAME
    labels = safe_load_array(labels_path).astype(np.int64)
    subject_ids = safe_load_array(subject_ids_path).astype(np.int64)
    phenotype = read_phenotype_records(phenotype_path)
    reconstructed = np.asarray(
        [phenotype[int(subject_id)]["label"] for subject_id in subject_ids],
        dtype=np.int64,
    )
    if not np.array_equal(labels, reconstructed):
        mismatches = np.flatnonzero(labels != reconstructed)[:10].tolist()
        raise RuntimeError(
            f"Strict labels do not align with subject IDs; mismatches={mismatches}"
        )
    return LabelFirewall(labels, subject_ids)


def parse_fold_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not folds:
        raise ValueError("--folds did not contain any fold indices")
    return folds


def default_run_directory() -> Path:
    return OUTPUT_ROOT / datetime.now().strftime("run_%Y%m%d_%H%M%S")


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-free nested CV for ABIDE-I C-PAC"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Validate allowed input files and splits")
    audit.add_argument("--data-dir", default=str(DATA_DIR))

    stage_a = subparsers.add_parser(
        "stage-a", help="Train and write frozen predictions without test labels"
    )
    stage_a.add_argument("--data-dir", default=str(DATA_DIR))
    stage_a.add_argument("--output-dir", default=None)
    stage_a.add_argument("--folds", default=None, help="Comma-separated 0-based outer folds")
    stage_a.add_argument("--epochs", type=int, default=300)
    stage_a.add_argument("--eval-every", type=int, default=5)
    stage_a.add_argument("--device", default="auto")
    stage_a.add_argument("--seed", type=int, default=DEFAULT_SEED)

    stage_b = subparsers.add_parser(
        "stage-b", help="Verify frozen predictions and then compute metrics"
    )
    stage_b.add_argument("--data-dir", default=str(DATA_DIR))
    stage_b.add_argument("--prediction-file", required=True)
    stage_b.add_argument("--manifest-file", required=True)
    stage_b.add_argument("--output-file", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        features = load_feature_bundle(args.data_dir)
        firewall = load_label_firewall(args.data_dir)
        splits = firewall.make_outer_splits(DEFAULT_SEED)
        validate_outer_splits(splits, len(features.subject_ids))
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "full_fc_shape": list(features.full_fc.shape),
                    "frequency_shape": list(features.frequency.shape),
                    "demographics_raw_shape": list(
                        features.demographics_raw.shape
                    ),
                    "subject_ids_shape": list(features.subject_ids.shape),
                    "outer_folds": N_OUTER,
                    "inner_folds": N_INNER,
                    "forbidden_legacy_inputs_read": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "stage-a":
        features = load_feature_bundle(args.data_dir)
        firewall = load_label_firewall(args.data_dir)
        config = TrainingConfig(
            epochs=args.epochs,
            eval_every=args.eval_every,
        )
        output = (
            Path(args.output_dir) if args.output_dir else default_run_directory()
        )
        backend = TorchTrainingBackend(choose_device(args.device))
        result = run_stage_a(
            features=features,
            label_firewall=firewall,
            output_directory=output,
            config=config,
            backend=backend,
            folds_to_run=parse_fold_list(args.folds),
            seed=args.seed,
        )
        print(
            json.dumps(
                {
                    "status": "STAGE_A_COMPLETE",
                    "output_directory": str(output),
                    "prediction_file": str(result["prediction_path"]),
                    "manifest_file": str(result["manifest_path"]),
                    "metrics_computed": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "stage-b":
        firewall = load_label_firewall(args.data_dir)
        result = run_stage_b(
            prediction_path=args.prediction_file,
            manifest_path=args.manifest_file,
            label_firewall=firewall,
            output_path=args.output_file,
        )
        print(
            json.dumps(
                {
                    "status": "STAGE_B_COMPLETE",
                    "output_file": args.output_file,
                    "verified_prediction_sha256": result[
                        "verified_prediction_sha256"
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

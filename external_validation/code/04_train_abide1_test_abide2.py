"""Nested ABIDE-I validation followed by one locked ABIDE-II external test.

All selection objects (FC selector, scalers, C and decision threshold) are fit
on ABIDE-I data only. ABIDE-II labels enter only after final probabilities have
been generated, to calculate pre-specified metrics and confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


PROJECT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = PROJECT / "features"
RESULTS = PROJECT / "results"


@dataclass
class Cohort:
    fc: np.ndarray
    psd: np.ndarray
    demo: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    site_ids: np.ndarray


def load_cohort(name: str) -> Cohort:
    directory = FEATURE_ROOT / name.lower()
    arrays = {
        "fc": np.load(directory / "X_fc_raw.npy", allow_pickle=False),
        "psd": np.load(directory / "X_psd.npy", allow_pickle=False),
        "demo": np.load(directory / "X_demo_raw.npy", allow_pickle=False),
        "y": np.load(directory / "y_labels.npy", allow_pickle=False).reshape(-1).astype(int),
        "subject_ids": np.load(directory / "subject_ids.npy", allow_pickle=False),
        "site_ids": np.load(directory / "site_ids.npy", allow_pickle=False),
    }
    n = len(arrays["y"])
    if any(len(value) != n for value in arrays.values()):
        raise ValueError(f"{name} arrays have inconsistent subject dimensions.")
    if arrays["fc"].shape[1] != 19900 or arrays["psd"].shape[1] != 3000 or arrays["demo"].shape[1] != 2:
        raise ValueError(f"{name} feature shape mismatch: FC={arrays['fc'].shape}, PSD={arrays['psd'].shape}, demo={arrays['demo'].shape}")
    if len(np.unique(arrays["y"])) != 2:
        raise ValueError(f"{name} requires ASD and TD labels.")
    return Cohort(**arrays)


@dataclass
class Preprocessor:
    selected_fc: np.ndarray
    fc_scaler: StandardScaler
    psd_scaler: StandardScaler
    demo_scaler: StandardScaler

    def transform(self, cohort: Cohort, indices: np.ndarray) -> np.ndarray:
        return np.hstack(
            [
                self.fc_scaler.transform(cohort.fc[indices][:, self.selected_fc]),
                self.psd_scaler.transform(cohort.psd[indices]),
                self.demo_scaler.transform(cohort.demo[indices]),
            ]
        ).astype(np.float32)


def fit_preprocessor(cohort: Cohort, indices: np.ndarray, top_k: int) -> Preprocessor:
    scores, _ = f_classif(cohort.fc[indices], cohort.y[indices])
    scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    if top_k <= 0 or top_k >= cohort.fc.shape[1]:
        selected = np.arange(cohort.fc.shape[1])
    else:
        selected = np.sort(np.argpartition(scores, -top_k)[-top_k:])
    return Preprocessor(
        selected_fc=selected,
        fc_scaler=StandardScaler().fit(cohort.fc[indices][:, selected]),
        psd_scaler=StandardScaler().fit(cohort.psd[indices]),
        demo_scaler=StandardScaler().fit(cohort.demo[indices]),
    )


def train_model(x: np.ndarray, y: np.ndarray, c_value: float) -> LogisticRegression:
    return LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000, solver="liblinear", random_state=20260710).fit(x, y)


def metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "sensitivity": float(recall_score(y, predictions, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y, predictions, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y, predictions, pos_label=1, zero_division=0)),
    }


def choose_threshold(y: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.unique(np.concatenate(([0.05], probabilities, [0.95])))
    scores = [balanced_accuracy_score(y, (probabilities >= value).astype(int)) for value in candidates]
    return float(candidates[int(np.argmax(scores))])


def oof_probabilities(cohort: Cohort, indices: np.ndarray, c_value: float, top_k: int, folds: int, seed: int) -> np.ndarray:
    y = cohort.y[indices]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    output = np.full(len(indices), np.nan, dtype=float)
    for train_rel, test_rel in splitter.split(np.zeros(len(indices)), y):
        train_idx, test_idx = indices[train_rel], indices[test_rel]
        processor = fit_preprocessor(cohort, train_idx, top_k)
        model = train_model(processor.transform(cohort, train_idx), cohort.y[train_idx], c_value)
        output[test_rel] = model.predict_proba(processor.transform(cohort, test_idx))[:, 1]
    if not np.isfinite(output).all():
        raise RuntimeError("Internal OOF probability generation failed.")
    return output


def choose_c(cohort: Cohort, indices: np.ndarray, c_grid: list[float], top_k: int, folds: int, seed: int) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for c_value in c_grid:
        probabilities = oof_probabilities(cohort, indices, c_value, top_k, folds, seed)
        scores[str(c_value)] = float(roc_auc_score(cohort.y[indices], probabilities))
    selected = max(c_grid, key=lambda value: scores[str(value)])
    return float(selected), scores


def nested_cv(development: Cohort, spec: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray]:
    selection = spec["model_selection"]
    outer_folds = int(selection["outer_folds"])
    inner_folds = int(selection["inner_folds"])
    top_k = int(selection["fc_top_k"])
    c_grid = [float(value) for value in selection["logistic_c_grid"]]
    splitter = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=int(spec["random_seed"]))
    oof = np.full(len(development.y), np.nan, dtype=float)
    records: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(development.fc, development.y), start=1):
        c_value, c_scores = choose_c(development, train_idx, c_grid, top_k, inner_folds, int(spec["random_seed"]) + fold)
        train_oof = oof_probabilities(development, train_idx, c_value, top_k, inner_folds, int(spec["random_seed"]) + 100 + fold)
        threshold = choose_threshold(development.y[train_idx], train_oof)
        processor = fit_preprocessor(development, train_idx, top_k)
        model = train_model(processor.transform(development, train_idx), development.y[train_idx], c_value)
        test_probabilities = model.predict_proba(processor.transform(development, test_idx))[:, 1]
        oof[test_idx] = test_probabilities
        records.append(
            {
                "outer_fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "selected_c": c_value,
                "inner_oof_auc_by_c": c_scores,
                "selected_threshold": threshold,
                "metrics": metrics(development.y[test_idx], test_probabilities, threshold),
                "selection_scope": "ABIDE_I outer-train only",
            }
        )
    return records, oof


def bootstrap_ci(y: np.ndarray, probabilities: np.ndarray, threshold: float, n_bootstrap: int = 2000) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(20260710)
    per_class = {label: np.flatnonzero(y == label) for label in (0, 1)}
    samples: dict[str, list[float]] = {key: [] for key in ("auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1")}
    for _ in range(n_bootstrap):
        indices = np.concatenate([rng.choice(per_class[label], len(per_class[label]), replace=True) for label in (0, 1)])
        values = metrics(y[indices], probabilities[indices], threshold)
        for key, value in values.items():
            samples[key].append(value)
    return {key: {"low": float(np.quantile(value, 0.025)), "high": float(np.quantile(value, 0.975))} for key, value in samples.items()}


def external_evaluation(development: Cohort, external: Cohort, spec: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    selection = spec["model_selection"]
    top_k = int(selection["fc_top_k"])
    c_grid = [float(value) for value in selection["logistic_c_grid"]]
    folds = int(selection["inner_folds"])
    all_indices = np.arange(len(development.y))
    c_value, c_scores = choose_c(development, all_indices, c_grid, top_k, folds, int(spec["random_seed"]) + 1000)
    development_oof = oof_probabilities(development, all_indices, c_value, top_k, folds, int(spec["random_seed"]) + 2000)
    threshold = choose_threshold(development.y, development_oof)
    processor = fit_preprocessor(development, all_indices, top_k)
    model = train_model(processor.transform(development, all_indices), development.y, c_value)
    external_indices = np.arange(len(external.y))
    probabilities = model.predict_proba(processor.transform(external, external_indices))[:, 1]
    outcome = {
        "status": "DONE",
        "n_development": int(len(development.y)),
        "n_external": int(len(external.y)),
        "selected_c": c_value,
        "internal_oof_auc_by_c": c_scores,
        "selected_threshold": threshold,
        "external_metrics": metrics(external.y, probabilities, threshold),
        "external_bootstrap_95ci": bootstrap_ci(external.y, probabilities, threshold),
        "selection_integrity": {
            "FC_selector_fit_scope": "ABIDE_I only",
            "all_scalers_fit_scope": "ABIDE_I only",
            "model_C_selection_scope": "ABIDE_I only",
            "threshold_selection_scope": "ABIDE_I out-of-fold probabilities only",
            "ABIDE_II_used_for_selection": False,
        },
    }
    return outcome, probabilities


def write_site_metrics(external: Cohort, probabilities: np.ndarray, threshold: float) -> None:
    rows = []
    for site in sorted(set(map(str, external.site_ids))):
        mask = np.asarray(external.site_ids, dtype=str) == site
        if len(np.unique(external.y[mask])) == 2:
            rows.append({"site_id": site, "n": int(mask.sum()), **metrics(external.y[mask], probabilities[mask], threshold)})
        else:
            rows.append({"site_id": site, "n": int(mask.sum()), "note": "single-class site; AUC and balanced metrics are not estimable"})
    with (RESULTS / "abide2_site_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-nested-cv", action="store_true", help="For recovery only; never use this option for the primary report.")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load((PROJECT / "config" / "analysis_spec.yaml").read_text(encoding="utf-8"))
    development = load_cohort("ABIDE_I")
    external = load_cohort("ABIDE_II")
    if not args.skip_nested_cv:
        folds, oof = nested_cv(development, spec)
        nested_summary = {
            "status": "DONE",
            "outer_folds": len(folds),
            "folds": folds,
            "ABIDE_I_oof_metrics_at_0.5": metrics(development.y, oof, 0.5),
            "selection_integrity": "Every FC selector/scaler/C/threshold was fit within the applicable ABIDE-I training partition.",
        }
        (RESULTS / "abide1_nested_cv.json").write_text(json.dumps(nested_summary, indent=2), encoding="utf-8")
        np.savez(RESULTS / "abide1_nested_oof_predictions.npz", y=development.y, probabilities=oof)
    result, probabilities = external_evaluation(development, external, spec)
    (RESULTS / "abide1_to_abide2_external_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez(RESULTS / "abide2_external_predictions.npz", y=external.y, probabilities=probabilities, subject_ids=external.subject_ids, site_ids=external.site_ids)
    write_site_metrics(external, probabilities, float(result["selected_threshold"]))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

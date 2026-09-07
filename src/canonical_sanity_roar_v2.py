"""Canonical-v2 sanity checks and repeated-seed ROAR.

The run directory is always explicit.  Rankings for ROAR are derived only from
outer-training participants of the loadable canonical checkpoints.  Outer-test
labels are accessed only after each retrained condition has produced frozen
probabilities.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import canonical_pipeline_v2 as pipeline  # noqa: E402
import canonical_xai_v2 as xai  # noqa: E402


DEFAULT_OUTPUT_DIR = pipeline.OUTPUT_ROOT / "sanity_roar"
DEFAULT_PERCENTAGES = (0.05, 0.10)


def parse_ints(value: str, *, valid: set[int] | None = None) -> list[int]:
    values = sorted(set(int(item.strip()) for item in value.split(",") if item.strip()))
    if not values or (valid is not None and any(item not in valid for item in values)):
        raise argparse.ArgumentTypeError("invalid comma-separated integer list")
    return values


def parse_folds(value: str) -> list[int]:
    return parse_ints(value, valid=set(range(10)))


def parse_seeds(value: str) -> list[int]:
    return parse_ints(value, valid=set(pipeline.FINAL_SEEDS))


def parse_percentages(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(not 0.0 < item < 1.0 for item in values):
        raise argparse.ArgumentTypeError("removal percentages must lie in (0,1)")
    return values


def safe_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else float("nan")


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="mergesort")
    result = np.empty_like(order, dtype=np.float64)
    result[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    return result


def spearman(first: np.ndarray, second: np.ndarray) -> float:
    left = rank(np.asarray(first, dtype=np.float64))
    right = rank(np.asarray(second, dtype=np.float64))
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    return float(np.sum(left * right) / denominator) if denominator else float("nan")


def topk_jaccard(first: np.ndarray, second: np.ndarray, k: int) -> float:
    count = min(k, len(first), len(second))
    left = set(np.argsort(-np.asarray(first), kind="mergesort")[:count].tolist())
    right = set(np.argsort(-np.asarray(second), kind="mergesort")[:count].tolist())
    return float(len(left & right) / len(left | right)) if left | right else float("nan")


def seed_directory(run_dir: Path, fold: int, seed: int) -> Path:
    return run_dir / "folds" / f"fold_{fold:02d}" / "outer_final" / f"seed_{seed}"


def load_seed_data(
    run_dir: Path,
    fold: int,
    seed: int,
    features: pipeline.audited.FeatureBundle,
    labels: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    folder = seed_directory(run_dir, fold, seed)
    checkpoint_path = folder / "model_checkpoint.pt"
    state_path = folder / "preprocessing_state.npz"
    checkpoint_manifest = json.loads((folder / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    if checkpoint_manifest["status"] != "PASS" or pipeline.sha256_file(checkpoint_path) != checkpoint_manifest["checkpoint"]["sha256"]:
        raise RuntimeError(f"Canonical checkpoint gate failed: {folder}")
    model, candidate, checkpoint = pipeline.make_model_from_checkpoint(checkpoint_path, state_path, device)
    train = pipeline.apply_saved_preprocessing(state_path, features, "train")
    test = pipeline.apply_saved_preprocessing(state_path, features, "test")
    with np.load(folder / "test_outputs.npz", allow_pickle=False) as saved:
        saved_probabilities = saved["probabilities"].astype(np.float64)
        saved_indices = saved["row_indices"].astype(np.int64)
    if not np.array_equal(saved_indices, test["indices"]):
        raise RuntimeError(f"Saved prediction index mismatch: {folder}")
    return {
        "folder": folder,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": pipeline.sha256_file(checkpoint_path),
        "checkpoint": checkpoint,
        "model": model,
        "candidate": candidate,
        "train": train,
        "test": test,
        "y_train": labels[train["indices"]].astype(np.float32),
        "y_test": labels[test["indices"]].astype(np.int64),
        "saved_probabilities": saved_probabilities,
    }


def train_fixed_recipe(
    data: dict[str, Any],
    labels: np.ndarray,
    device: torch.device,
    removed_local: np.ndarray,
    epoch_cap: int = 0,
) -> tuple[torch.nn.Module, np.ndarray, float]:
    candidate = data["candidate"]
    checkpoint = data["checkpoint"]
    fixed_epoch = int(checkpoint["fixed_epoch"])
    epochs = min(fixed_epoch, epoch_cap) if epoch_cap > 0 else fixed_epoch
    actual_seed = int(checkpoint["actual_seed"])
    backend = pipeline.optimization.TorchOptimizationBackend(device)
    backend._set_seed(actual_seed)
    x_fc_train = data["train"]["x_fc"].copy()
    x_fc_test = data["test"]["x_fc"].copy()
    if len(removed_local):
        x_fc_train[:, removed_local] = 0.0
        x_fc_test[:, removed_local] = 0.0
    prepared = SimpleNamespace(
        anchors=data["train"]["anchors"],
        x_fc_fit=x_fc_train,
        x_frequency_fit=data["train"]["x_frequency"],
        x_demographic_fit=data["train"]["x_demographic"],
        x_fc_evaluation=x_fc_test,
        x_frequency_evaluation=data["test"]["x_frequency"],
        x_demographic_evaluation=data["test"]["x_demographic"],
    )
    model = backend._make_model(candidate, prepared)
    optimizer, scheduler = backend._make_optimizer_scheduler(model, candidate, epochs)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(actual_seed)
    loader = backend._loader(prepared, labels, candidate, generator)
    started = time.time()
    for epoch in range(1, epochs + 1):
        backend._train_epoch(model, optimizer, loader, candidate, epoch)
        if candidate.family == "fc_only_compact" or epoch > 60:
            scheduler.step()
    _, probabilities = pipeline.predict_logits_probabilities(
        model,
        candidate,
        prepared.x_fc_evaluation,
        prepared.x_frequency_evaluation,
        prepared.x_demographic_evaluation,
        device,
    )
    return model, probabilities, time.time() - started


def trained_fc_scores(data: dict[str, Any], device: torch.device) -> np.ndarray:
    fc, _, _ = xai.signed_gradient_x_input(
        data["model"],
        data["candidate"],
        data["test"]["x_fc"],
        data["test"]["x_frequency"],
        data["test"]["x_demographic"],
        device,
    )
    return np.mean(np.abs(fc), axis=0)


def run_sanity(
    run_dir: Path,
    xai_dir: Path,
    output_dir: Path,
    folds: Sequence[int],
    seeds: Sequence[int],
    device_name: str,
    epoch_cap: int = 0,
) -> dict[str, Any]:
    features, _firewall, labels = pipeline.load_canonical_inputs()
    device = pipeline.audited.choose_device(device_name)
    sanity_dir = output_dir / "sanity"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    deterministic_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    for fold in folds:
        for seed in seeds:
            data = load_seed_data(run_dir, fold, seed, features, labels, device)
            _, first = pipeline.predict_logits_probabilities(
                data["model"], data["candidate"], data["test"]["x_fc"], data["test"]["x_frequency"], data["test"]["x_demographic"], device
            )
            _, second = pipeline.predict_logits_probabilities(
                data["model"], data["candidate"], data["test"]["x_fc"], data["test"]["x_frequency"], data["test"]["x_demographic"], device
            )
            deterministic_rows.append({
                "fold": fold,
                "seed": seed,
                "checkpoint_sha256": data["checkpoint_sha256"],
                "max_abs_diff_repeat": float(np.max(np.abs(first - second))),
                "max_abs_diff_saved": float(np.max(np.abs(first - data["saved_probabilities"]))),
                "pass": bool(np.max(np.abs(first - second)) <= pipeline.CHECKPOINT_RELOAD_ATOL and np.max(np.abs(first - data["saved_probabilities"])) <= pipeline.CHECKPOINT_RELOAD_ATOL),
            })
            trained_scores = trained_fc_scores(data, device)

            # Full trainable-weight reinitialization sanity.
            random.seed(7_026_082_700 + fold * 100 + seed)
            np.random.seed((7_026_082_700 + fold * 100 + seed) % (2**32 - 1))
            torch.manual_seed(7_026_082_700 + fold * 100 + seed)
            prepared_stub = SimpleNamespace(anchors=data["test"]["anchors"])
            randomized = pipeline.optimization.TorchOptimizationBackend(device)._make_model(data["candidate"], prepared_stub)
            randomized_fc, _, _ = xai.signed_gradient_x_input(
                randomized,
                data["candidate"],
                data["test"]["x_fc"],
                data["test"]["x_frequency"],
                data["test"]["x_demographic"],
                device,
            )
            randomized_scores = np.mean(np.abs(randomized_fc), axis=0)
            weight_rows.append({
                "fold": fold,
                "seed": seed,
                "checkpoint_sha256": data["checkpoint_sha256"],
                "randomization": "all_trainable_weights_reinitialized",
                "spearman_trained_vs_randomized": spearman(trained_scores, randomized_scores),
                "top50_jaccard": topk_jaccard(trained_scores, randomized_scores, 50),
                "top100_jaccard": topk_jaccard(trained_scores, randomized_scores, 100),
            })
            del randomized

            # Coordinate-matched label randomization: canonical selector and
            # transforms remain frozen; only outer-training labels are permuted.
            rng = np.random.default_rng(8_026_082_700 + fold * 100 + seed)
            permuted_labels = rng.permutation(data["y_train"]).astype(np.float32)
            random_label_model, _, duration = train_fixed_recipe(
                data,
                permuted_labels,
                device,
                np.empty(0, dtype=np.int64),
                epoch_cap=epoch_cap,
            )
            random_fc, _, _ = xai.signed_gradient_x_input(
                random_label_model,
                data["candidate"],
                data["test"]["x_fc"],
                data["test"]["x_frequency"],
                data["test"]["x_demographic"],
                device,
            )
            random_scores = np.mean(np.abs(random_fc), axis=0)
            label_result_path = sanity_dir / "random_label_attributions" / f"fold_{fold:02d}_seed_{seed}.npz"
            label_result_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                label_result_path,
                selected_edges=data["test"]["selected_edges"],
                trained_mean_absolute=trained_scores.astype(np.float32),
                random_label_mean_absolute=random_scores.astype(np.float32),
            )
            label_rows.append({
                "fold": fold,
                "seed": seed,
                "checkpoint_sha256": data["checkpoint_sha256"],
                "epochs": min(int(data["checkpoint"]["fixed_epoch"]), epoch_cap) if epoch_cap > 0 else int(data["checkpoint"]["fixed_epoch"]),
                "runtime_seconds": duration,
                "spearman_trained_vs_random_label": spearman(trained_scores, random_scores),
                "top50_jaccard": topk_jaccard(trained_scores, random_scores, 50),
                "top100_jaccard": topk_jaccard(trained_scores, random_scores, 100),
                "selector_and_preprocessing": "canonical outer-train state frozen for coordinate matching",
                "outer_test_labels_used_for_training": False,
                "result_path": label_result_path.resolve(),
                "result_sha256": pipeline.sha256_file(label_result_path),
            })
            print(f"[sanity] fold={fold} seed={seed} complete", flush=True)
            del random_label_model, data["model"]
            if device.type == "cuda":
                torch.cuda.empty_cache()
    pipeline.write_csv(sanity_dir / "deterministic_inference.csv", deterministic_rows)
    pipeline.write_csv(sanity_dir / "weight_randomization.csv", weight_rows)
    pipeline.write_csv(sanity_dir / "label_randomization.csv", label_rows)

    # Compare the two canonical held-out methods in their common full spaces.
    xai_global_path = xai_dir / "ensemble" / "oof_three_seed_signed_attributions.npz"
    if xai_global_path.exists() and list(folds) == list(range(10)) and list(seeds) == list(pipeline.FINAL_SEEDS):
        with np.load(xai_global_path, allow_pickle=False) as attribution:
            gxi_fc = np.mean(np.abs(attribution["gradient_x_input_fc_full_signed"]), axis=0)
            ig_fc = np.mean(np.abs(attribution["integrated_gradients_fc_full_signed"]), axis=0)
            gxi_frequency = np.mean(np.abs(attribution["gradient_x_input_frequency_signed"]), axis=0)
            ig_frequency = np.mean(np.abs(attribution["integrated_gradients_frequency_signed"]), axis=0)
        method_comparison = {
            "scope": "879 held-out subjects after signed three-seed averaging",
            "fc_full19900_spearman": spearman(gxi_fc, ig_fc),
            "fc_top50_jaccard": topk_jaccard(gxi_fc, ig_fc, 50),
            "fc_top100_jaccard": topk_jaccard(gxi_fc, ig_fc, 100),
            "frequency_3000_spearman": spearman(gxi_frequency, ig_frequency),
            "frequency_top50_jaccard": topk_jaccard(gxi_frequency, ig_frequency, 50),
            "frequency_top100_jaccard": topk_jaccard(gxi_frequency, ig_frequency, 100),
        }
    else:
        method_comparison = {"scope": "not computed in partial/smoke mode"}
    checks = {
        "deterministic_inference_all_pass": all(row["pass"] for row in deterministic_rows),
        "weight_randomization_all_models_completed": len(weight_rows) == len(folds) * len(seeds),
        "label_randomization_all_models_completed": len(label_rows) == len(folds) * len(seeds),
        "all_records_point_to_existing_canonical_checkpoint": all(
            pipeline.sha256_file(seed_directory(run_dir, row["fold"], row["seed"]) / "model_checkpoint.pt") == row["checkpoint_sha256"]
            for row in weight_rows + label_rows
        ),
    }
    summary = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "generated_at": pipeline.now_iso(),
        "run_directory": run_dir.resolve(),
        "xai_directory": xai_dir.resolve(),
        "folds": list(folds),
        "seeds": list(seeds),
        "epoch_cap": epoch_cap,
        "checks": checks,
        "deterministic_inference": {
            "model_count": len(deterministic_rows),
            "maximum_abs_diff_repeat": max(row["max_abs_diff_repeat"] for row in deterministic_rows),
            "maximum_abs_diff_saved": max(row["max_abs_diff_saved"] for row in deterministic_rows),
        },
        "weight_randomization": {
            "model_count": len(weight_rows),
            "mean_spearman": float(np.mean([row["spearman_trained_vs_randomized"] for row in weight_rows])),
            "mean_top50_jaccard": float(np.mean([row["top50_jaccard"] for row in weight_rows])),
            "mean_top100_jaccard": float(np.mean([row["top100_jaccard"] for row in weight_rows])),
        },
        "label_randomization": {
            "model_count": len(label_rows),
            "mean_spearman": float(np.mean([row["spearman_trained_vs_random_label"] for row in label_rows])),
            "mean_top50_jaccard": float(np.mean([row["top50_jaccard"] for row in label_rows])),
            "mean_top100_jaccard": float(np.mean([row["top100_jaccard"] for row in label_rows])),
            "boundary": "coordinate-matched fixed-selector label sanity; not a fully re-nested permutation performance test",
        },
        "gradient_x_input_vs_integrated_gradients": method_comparison,
    }
    pipeline.write_json(sanity_dir / "sanity_summary.json", summary)
    print(json.dumps(pipeline.clean(summary), ensure_ascii=False, indent=2), flush=True)
    return summary


def training_only_rankings(
    run_dir: Path,
    output_dir: Path,
    fold: int,
    features: pipeline.audited.FeatureBundle,
    labels: np.ndarray,
    device: torch.device,
    ig_steps: int,
) -> dict[str, Any]:
    rank_dir = output_dir / "roar" / "training_only_rankings" / f"fold_{fold:02d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = rank_dir / "three_seed_training_only_rankings.npz"
    provenance_path = rank_dir / "provenance.json"
    if ranking_path.exists() and provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("status") == "PASS":
            with np.load(ranking_path, allow_pickle=False) as saved:
                return {key: saved[key] for key in saved.files} | {"provenance": provenance}
    seed_gxi: list[np.ndarray] = []
    seed_ig: list[np.ndarray] = []
    checkpoint_hashes: dict[str, str] = {}
    reference_indices = reference_selected = None
    seed_file_hashes: dict[str, str] = {}
    for seed in pipeline.FINAL_SEEDS:
        data = load_seed_data(run_dir, fold, seed, features, labels, device)
        train = data["train"]
        gxi_fc, _, _ = xai.signed_gradient_x_input(
            data["model"], data["candidate"], train["x_fc"], train["x_frequency"], train["x_demographic"], device
        )
        ig_fc, _, _ = xai.signed_integrated_gradients(
            data["model"], data["candidate"], train["x_fc"], train["x_frequency"], train["x_demographic"], device, steps=ig_steps
        )
        if reference_indices is None:
            reference_indices = train["indices"]
            reference_selected = train["selected_edges"]
        if not np.array_equal(reference_indices, train["indices"]) or not np.array_equal(reference_selected, train["selected_edges"]):
            raise RuntimeError(f"Fold {fold}: train indices/edges differ across final seeds")
        seed_path = rank_dir / f"seed_{seed}_training_signed_fc_attributions.npz"
        np.savez_compressed(
            seed_path,
            train_indices=train["indices"],
            train_subject_ids=features.subject_ids[train["indices"]],
            selected_edges=train["selected_edges"],
            gradient_x_input_fc_selected_signed=gxi_fc,
            integrated_gradients_fc_selected_signed=ig_fc,
        )
        seed_file_hashes[str(seed)] = pipeline.sha256_file(seed_path)
        checkpoint_hashes[str(seed)] = data["checkpoint_sha256"]
        seed_gxi.append(gxi_fc)
        seed_ig.append(ig_fc)
        del data["model"]
        if device.type == "cuda":
            torch.cuda.empty_cache()
    signed_gxi_mean = pipeline.aggregate_signed_attributions(np.stack(seed_gxi, axis=0))
    signed_ig_mean = pipeline.aggregate_signed_attributions(np.stack(seed_ig, axis=0))
    scores_gxi = np.mean(np.abs(signed_gxi_mean), axis=0).astype(np.float32)
    scores_ig = np.mean(np.abs(signed_ig_mean), axis=0).astype(np.float32)
    np.savez_compressed(
        ranking_path,
        train_indices=reference_indices,
        selected_edges=reference_selected,
        gradient_x_input=scores_gxi,
        integrated_gradients=scores_ig,
    )
    provenance = {
        "status": "PASS",
        "fold": fold,
        "ranking_participants": "outer_train_only",
        "outer_test_labels_or_features_used_for_ranking": False,
        "seed_aggregation": "mean signed attribution across canonical seeds 42,43,44",
        "subject_aggregation": "mean absolute attribution after signed seed aggregation",
        "ig_steps": ig_steps,
        "checkpoint_sha256": checkpoint_hashes,
        "single_seed_raw_training_attribution_sha256": seed_file_hashes,
        "ranking_path": ranking_path.resolve(),
        "ranking_sha256": pipeline.sha256_file(ranking_path),
    }
    pipeline.write_json(provenance_path, provenance)
    return {
        "train_indices": reference_indices,
        "selected_edges": reference_selected,
        "gradient_x_input": scores_gxi,
        "integrated_gradients": scores_ig,
        "provenance": provenance,
    }


def roar_condition_key(fold: int, seed: int, method: str, percentage: float | None) -> str:
    label = "baseline" if percentage is None else f"{100*percentage:g}pct"
    return f"fold{fold:02d}__seed{seed}__{method}__{label}"


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def exact_signflip_two_sided(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    null = [
        abs(float(np.mean(values * np.asarray(signs, dtype=np.float64))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def fold_bootstrap(values: np.ndarray, seed: int, replicates: int = 20_000) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[draws].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def run_roar(
    run_dir: Path,
    output_dir: Path,
    folds: Sequence[int],
    seeds: Sequence[int],
    percentages: Sequence[float],
    device_name: str,
    ig_steps: int,
    epoch_cap: int = 0,
    resume: bool = False,
) -> dict[str, Any]:
    features, _firewall, labels = pipeline.load_canonical_inputs()
    device = pipeline.audited.choose_device(device_name)
    roar_dir = output_dir / "roar"
    roar_dir.mkdir(parents=True, exist_ok=True)
    results_path = roar_dir / "repeated_roar_by_fold_seed.csv"
    rows = read_csv_rows(results_path) if resume else []
    done = {row["condition_key"] for row in rows}
    for fold in folds:
        rankings = training_only_rankings(run_dir, output_dir, fold, features, labels, device, ig_steps)
        for seed in seeds:
            data = load_seed_data(run_dir, fold, seed, features, labels, device)
            if not np.array_equal(rankings["selected_edges"], data["train"]["selected_edges"]):
                raise RuntimeError(f"Fold {fold}: ROAR ranking coordinates mismatch seed {seed}")
            baseline_key = roar_condition_key(fold, seed, "unmodified_baseline", None)
            baseline_row = next((row for row in rows if row["condition_key"] == baseline_key), None)
            if baseline_row is None:
                model, probabilities, duration = train_fixed_recipe(
                    data, data["y_train"], device, np.empty(0, dtype=np.int64), epoch_cap=epoch_cap
                )
                baseline_error = float(np.max(np.abs(probabilities - data["saved_probabilities"]))) if epoch_cap == 0 else None
                baseline_row = {
                    "condition_key": baseline_key,
                    "fold": fold,
                    "seed": seed,
                    "method": "unmodified_baseline",
                    "removal_fraction": "",
                    "removal_label": "0%",
                    "n_removed": 0,
                    "auc": safe_auc(data["y_test"], probabilities),
                    "accuracy": float(accuracy_score(data["y_test"], probabilities >= 0.5)),
                    "baseline_auc_same_fold_seed": safe_auc(data["y_test"], probabilities),
                    "delta_auc_vs_baseline": 0.0,
                    "checkpoint_sha256": data["checkpoint_sha256"],
                    "fixed_epoch": int(data["checkpoint"]["fixed_epoch"]),
                    "epochs_executed": min(int(data["checkpoint"]["fixed_epoch"]), epoch_cap) if epoch_cap > 0 else int(data["checkpoint"]["fixed_epoch"]),
                    "runtime_seconds": duration,
                    "ranking_scope": "not_applicable",
                    "outer_test_used_for_ranking_or_training": False,
                    "retrained_baseline_max_abs_error_vs_checkpoint": baseline_error,
                }
                rows.append(baseline_row)
                pipeline.write_csv(results_path, rows)
                del model
            baseline_auc = float(baseline_row["auc"])
            for percentage in percentages:
                count = min(
                    len(rankings["selected_edges"]),
                    max(1, int(math.ceil(len(rankings["selected_edges"]) * percentage))),
                )
                rng = np.random.default_rng(9_026_082_700 + fold * 100_000 + seed * 1000 + int(round(percentage * 10_000)))
                masks = {
                    "gradient_x_input": np.argsort(-rankings["gradient_x_input"], kind="mergesort")[:count].astype(np.int64),
                    "integrated_gradients": np.argsort(-rankings["integrated_gradients"], kind="mergesort")[:count].astype(np.int64),
                    "random": np.sort(rng.choice(len(rankings["selected_edges"]), size=count, replace=False)).astype(np.int64),
                }
                for method, removed in masks.items():
                    key = roar_condition_key(fold, seed, method, percentage)
                    if key in done:
                        continue
                    print(f"[ROAR] {key}: remove={count}", flush=True)
                    model, probabilities, duration = train_fixed_recipe(
                        data, data["y_train"], device, removed, epoch_cap=epoch_cap
                    )
                    auc = safe_auc(data["y_test"], probabilities)
                    row = {
                        "condition_key": key,
                        "fold": fold,
                        "seed": seed,
                        "method": method,
                        "removal_fraction": percentage,
                        "removal_label": f"{100*percentage:g}%",
                        "n_removed": count,
                        "auc": auc,
                        "accuracy": float(accuracy_score(data["y_test"], probabilities >= 0.5)),
                        "baseline_auc_same_fold_seed": baseline_auc,
                        "delta_auc_vs_baseline": auc - baseline_auc,
                        "checkpoint_sha256": data["checkpoint_sha256"],
                        "fixed_epoch": int(data["checkpoint"]["fixed_epoch"]),
                        "epochs_executed": min(int(data["checkpoint"]["fixed_epoch"]), epoch_cap) if epoch_cap > 0 else int(data["checkpoint"]["fixed_epoch"]),
                        "runtime_seconds": duration,
                        "ranking_scope": "outer_train_only__three_canonical_seed_signed_mean_then_subject_mean_absolute" if method != "random" else "seeded_equal_cardinality_random",
                        "ranking_sha256": rankings["provenance"]["ranking_sha256"] if method != "random" else "not_applicable",
                        "outer_test_used_for_ranking_or_training": False,
                    }
                    rows.append(row)
                    done.add(key)
                    pipeline.write_csv(results_path, rows)
                    pipeline.write_json(roar_dir / "progress.json", {"status": "RUNNING", "last_completed": row, "completed_conditions": len(rows), "updated_at": pipeline.now_iso()})
                    print(f"[ROAR-complete] {key}: auc={auc:.6f} delta={auc-baseline_auc:+.6f} seconds={duration:.1f}", flush=True)
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            del data["model"]
            if device.type == "cuda":
                torch.cuda.empty_cache()
    lookup = {
        (int(row["fold"]), int(row["seed"]), row["method"], row["removal_label"]): row
        for row in rows
    }
    comparisons: list[dict[str, Any]] = []
    for percentage in percentages:
        label = f"{100*percentage:g}%"
        for method in ("gradient_x_input", "integrated_gradients"):
            fold_values: list[float] = []
            raw_values: list[float] = []
            for fold in folds:
                seed_values: list[float] = []
                for seed in seeds:
                    attribution_row = lookup.get((fold, seed, method, label))
                    random_row = lookup.get((fold, seed, "random", label))
                    if attribution_row is not None and random_row is not None:
                        value = float(attribution_row["auc"]) - float(random_row["auc"])
                        seed_values.append(value)
                        raw_values.append(value)
                if len(seed_values) == len(seeds):
                    fold_values.append(float(np.mean(seed_values)))
            values = np.asarray(fold_values, dtype=np.float64)
            complete = len(values) == len(folds)
            mean = float(np.mean(values)) if complete else None
            ci = fold_bootstrap(values, 10_026_082_700 + int(percentage * 1000) + len(method)) if complete else [None, None]
            p_value = exact_signflip_two_sided(values) if complete else None
            comparisons.append({
                "method": method,
                "removal_fraction": percentage,
                "n_folds": len(values),
                "seeds_per_fold": len(seeds),
                "n_paired_runs": len(raw_values),
                "mean_auc_attribution_minus_random": mean,
                "fold_bootstrap_95_ci": ci,
                "exact_two_sided_signflip_p": p_value,
                "supports_roar_fidelity": bool(mean is not None and mean < 0 and ci[1] is not None and ci[1] < 0),
                "predeclared_positive_direction": "negative attribution-minus-random AUC",
            })
    expected = len(folds) * len(seeds) * (1 + len(percentages) * 3)
    relevant_rows = [row for row in rows if int(row["fold"]) in folds and int(row["seed"]) in seeds]
    checks = {
        "all_expected_conditions_complete": len(relevant_rows) == expected,
        "all_rankings_outer_train_only": all(
            row["ranking_scope"] in {"not_applicable", "seeded_equal_cardinality_random", "outer_train_only__three_canonical_seed_signed_mean_then_subject_mean_absolute"}
            for row in relevant_rows
        ),
        "no_outer_test_used_for_rank_or_training": all(str(row["outer_test_used_for_ranking_or_training"]).lower() in {"false", "0"} for row in relevant_rows),
        "all_checkpoint_hashes_match": all(
            pipeline.sha256_file(seed_directory(run_dir, int(row["fold"]), int(row["seed"])) / "model_checkpoint.pt") == row["checkpoint_sha256"]
            for row in relevant_rows
        ),
    }
    summary = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "generated_at": pipeline.now_iso(),
        "folds": list(folds),
        "seeds": list(seeds),
        "removal_percentages": list(percentages),
        "expected_conditions": expected,
        "completed_conditions": len(relevant_rows),
        "ig_steps_for_training_rank": ig_steps,
        "epoch_cap": epoch_cap,
        "checks": checks,
        "comparisons": comparisons,
        "claim_boundary": "Repeated ROAR is a canonical-model attribution fidelity diagnostic, not biological or causal validation.",
    }
    pipeline.write_json(roar_dir / "repeated_roar_summary.json", summary)
    pipeline.write_csv(roar_dir / "repeated_roar_comparisons.csv", comparisons)
    pipeline.write_json(roar_dir / "progress.json", {"status": summary["status"], "completed_conditions": len(relevant_rows), "updated_at": pipeline.now_iso()})
    print(json.dumps(pipeline.clean(summary), ensure_ascii=False, indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical-v2 sanity and repeated ROAR with explicit run directory")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--xai-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("sanity", "roar", "all"), default="all")
    parser.add_argument("--folds", type=parse_folds, default=list(range(10)))
    parser.add_argument("--seeds", type=parse_seeds, default=list(pipeline.FINAL_SEEDS))
    parser.add_argument("--percentages", type=parse_percentages, default=list(DEFAULT_PERCENTAGES))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--epoch-cap", type=int, default=0, help="Smoke only; 0 uses canonical fixed epochs")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.ig_steps < 2 or args.epoch_cap < 0:
        raise ValueError("invalid IG steps or epoch cap")
    sanity_summary = None
    roar_summary = None
    if args.mode in {"sanity", "all"}:
        sanity_summary = run_sanity(
            args.run_dir.resolve(), args.xai_dir.resolve(), args.output_dir.resolve(), args.folds, args.seeds, args.device, args.epoch_cap
        )
    if args.mode in {"roar", "all"}:
        roar_summary = run_roar(
            args.run_dir.resolve(), args.output_dir.resolve(), args.folds, args.seeds, args.percentages, args.device, args.ig_steps, args.epoch_cap, args.resume
        )
    statuses = [summary["status"] for summary in (sanity_summary, roar_summary) if summary is not None]
    return 0 if statuses and all(status == "PASS" for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())

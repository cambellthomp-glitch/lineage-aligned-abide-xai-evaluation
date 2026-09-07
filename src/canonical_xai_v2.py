"""Held-out Gradient×Input and Integrated Gradients for canonical v2 checkpoints.

Every attribution is produced by an outer model that did not train on the
attributed subject.  The three signed seed maps are averaged before absolute
importance is computed.  No historical strict-run directory is hard-coded;
the canonical run is an explicit command-line argument.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import canonical_pipeline_v2 as pipeline  # noqa: E402


DEFAULT_XAI_DIR = pipeline.OUTPUT_ROOT / "xai"
DEFAULT_IG_STEPS = 16
METHODS = ("gradient_x_input", "integrated_gradients")


def parse_folds(value: str) -> list[int]:
    folds = sorted(set(int(item.strip()) for item in value.split(",") if item.strip()))
    if not folds or any(fold not in range(10) for fold in folds):
        raise argparse.ArgumentTypeError("folds must be comma-separated values in 0..9")
    return folds


def _forward_probability(
    model: torch.nn.Module,
    candidate: pipeline.optimization.OptimizationCandidate,
    x_fc: torch.Tensor,
    x_frequency: torch.Tensor,
    x_demographic: torch.Tensor,
) -> torch.Tensor:
    if candidate.family == "fc_only_compact":
        logits = model(x_fc)
    else:
        logits, _, _, _ = model(
            x_fc,
            x_frequency,
            x_demographic,
            tau=0.15,
            training=False,
        )
    return torch.sigmoid(logits)


def signed_gradient_x_input(
    model: torch.nn.Module,
    candidate: pipeline.optimization.OptimizationCandidate,
    x_fc: np.ndarray,
    x_frequency: np.ndarray,
    x_demographic: np.ndarray,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    if hasattr(model, "set_eval_ocread_mode"):
        model.set_eval_ocread_mode("mu")
    output_fc: list[np.ndarray] = []
    output_frequency: list[np.ndarray] = []
    output_demographic: list[np.ndarray] = []
    for start in range(0, len(x_fc), batch_size):
        stop = min(start + batch_size, len(x_fc))
        xs = torch.from_numpy(x_fc[start:stop]).to(device).detach().requires_grad_(True)
        xf = torch.from_numpy(x_frequency[start:stop]).to(device).detach().requires_grad_(candidate.use_frequency)
        xd = torch.from_numpy(x_demographic[start:stop]).to(device).detach().requires_grad_(candidate.use_demographics)
        model.zero_grad(set_to_none=True)
        probability = _forward_probability(model, candidate, xs, xf, xd)
        probability.sum().backward()
        output_fc.append((xs * xs.grad).detach().cpu().numpy().astype(np.float32))
        if candidate.use_frequency:
            output_frequency.append((xf * xf.grad).detach().cpu().numpy().astype(np.float32))
        else:
            output_frequency.append(np.zeros_like(x_frequency[start:stop], dtype=np.float32))
        if candidate.use_demographics:
            output_demographic.append((xd * xd.grad).detach().cpu().numpy().astype(np.float32))
        else:
            output_demographic.append(np.zeros_like(x_demographic[start:stop], dtype=np.float32))
    return (
        np.concatenate(output_fc, axis=0),
        np.concatenate(output_frequency, axis=0),
        np.concatenate(output_demographic, axis=0),
    )


def signed_integrated_gradients(
    model: torch.nn.Module,
    candidate: pipeline.optimization.OptimizationCandidate,
    x_fc: np.ndarray,
    x_frequency: np.ndarray,
    x_demographic: np.ndarray,
    device: torch.device,
    steps: int = DEFAULT_IG_STEPS,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if steps < 2:
        raise ValueError("Integrated Gradients steps must be at least 2")
    model.eval()
    if hasattr(model, "set_eval_ocread_mode"):
        model.set_eval_ocread_mode("mu")
    result_fc: list[np.ndarray] = []
    result_frequency: list[np.ndarray] = []
    result_demographic: list[np.ndarray] = []
    for start in range(0, len(x_fc), batch_size):
        stop = min(start + batch_size, len(x_fc))
        original_fc = torch.from_numpy(x_fc[start:stop]).to(device)
        original_frequency = torch.from_numpy(x_frequency[start:stop]).to(device)
        original_demographic = torch.from_numpy(x_demographic[start:stop]).to(device)
        accumulated_fc = torch.zeros_like(original_fc)
        accumulated_frequency = torch.zeros_like(original_frequency)
        accumulated_demographic = torch.zeros_like(original_demographic)
        for alpha in np.linspace(1.0 / steps, 1.0, steps):
            xs = (original_fc * float(alpha)).detach().requires_grad_(True)
            xf = (original_frequency * float(alpha)).detach().requires_grad_(candidate.use_frequency)
            xd = (original_demographic * float(alpha)).detach().requires_grad_(candidate.use_demographics)
            model.zero_grad(set_to_none=True)
            probability = _forward_probability(model, candidate, xs, xf, xd)
            probability.sum().backward()
            accumulated_fc += xs.grad.detach()
            if candidate.use_frequency:
                accumulated_frequency += xf.grad.detach()
            if candidate.use_demographics:
                accumulated_demographic += xd.grad.detach()
        result_fc.append((original_fc * accumulated_fc / float(steps)).detach().cpu().numpy().astype(np.float32))
        result_frequency.append((original_frequency * accumulated_frequency / float(steps)).detach().cpu().numpy().astype(np.float32))
        result_demographic.append((original_demographic * accumulated_demographic / float(steps)).detach().cpu().numpy().astype(np.float32))
    return (
        np.concatenate(result_fc, axis=0),
        np.concatenate(result_frequency, axis=0),
        np.concatenate(result_demographic, axis=0),
    )


def compute_seed_attributions(
    run_dir: Path,
    output_dir: Path,
    fold: int,
    seed: int,
    features: pipeline.audited.FeatureBundle,
    device: torch.device,
    ig_steps: int,
    partition: str = "test",
    max_subjects: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    seed_dir = run_dir / "folds" / f"fold_{fold:02d}" / "outer_final" / f"seed_{seed}"
    checkpoint_path = seed_dir / "model_checkpoint.pt"
    state_path = seed_dir / "preprocessing_state.npz"
    checkpoint_manifest_path = seed_dir / "checkpoint_manifest.json"
    checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
    checkpoint_sha = pipeline.sha256_file(checkpoint_path)
    if checkpoint_manifest["status"] != "PASS" or checkpoint_manifest["checkpoint"]["sha256"] != checkpoint_sha:
        raise RuntimeError(f"Checkpoint manifest/hash gate failed: {seed_dir}")
    model, candidate, checkpoint = pipeline.make_model_from_checkpoint(checkpoint_path, state_path, device)
    if checkpoint.get("lineage") != "cpac_model_xai_aligned_v2":
        raise RuntimeError(f"Non-canonical checkpoint rejected: {checkpoint_path}")
    if checkpoint.get("frequency_input_sha256") != pipeline.EXPECTED_FREQUENCY_SHA256:
        raise RuntimeError(f"Wrong frequency lineage rejected: {checkpoint_path}")
    prepared = pipeline.apply_saved_preprocessing(state_path, features, partition)
    if max_subjects is not None:
        for key in ("indices", "x_fc", "x_frequency", "x_demographic"):
            prepared[key] = prepared[key][:max_subjects]
    if partition == "test":
        test_outputs_path = seed_dir / "test_outputs.npz"
        with np.load(test_outputs_path, allow_pickle=False) as saved:
            saved_indices = saved["row_indices"].astype(np.int64)
            saved_probabilities = saved["probabilities"].astype(np.float64)
        if max_subjects is not None:
            saved_indices = saved_indices[:max_subjects]
            saved_probabilities = saved_probabilities[:max_subjects]
        if not np.array_equal(prepared["indices"], saved_indices):
            raise RuntimeError(f"XAI/test output index mismatch: fold={fold} seed={seed}")
        _, current_probabilities = pipeline.predict_logits_probabilities(
            model,
            candidate,
            prepared["x_fc"],
            prepared["x_frequency"],
            prepared["x_demographic"],
            device,
        )
        prediction_error = float(np.max(np.abs(current_probabilities - saved_probabilities)))
        prediction_sha = pipeline.sha256_array(saved_probabilities)
    else:
        prediction_error = 0.0
        prediction_sha = None
    if prediction_error > pipeline.CHECKPOINT_RELOAD_ATOL:
        raise RuntimeError(f"XAI prediction/checkpoint mismatch: fold={fold} seed={seed} error={prediction_error}")

    gxi_fc, gxi_frequency, gxi_demographic = signed_gradient_x_input(
        model,
        candidate,
        prepared["x_fc"],
        prepared["x_frequency"],
        prepared["x_demographic"],
        device,
    )
    ig_fc, ig_frequency, ig_demographic = signed_integrated_gradients(
        model,
        candidate,
        prepared["x_fc"],
        prepared["x_frequency"],
        prepared["x_demographic"],
        device,
        steps=ig_steps,
    )
    arrays = {
        "row_indices": prepared["indices"].astype(np.int64),
        "subject_ids": features.subject_ids[prepared["indices"]].astype(np.int64),
        "selected_edges": prepared["selected_edges"].astype(np.int64),
        "gradient_x_input_fc_selected_signed": gxi_fc,
        "gradient_x_input_frequency_signed": gxi_frequency,
        "gradient_x_input_demographic_signed": gxi_demographic,
        "integrated_gradients_fc_selected_signed": ig_fc,
        "integrated_gradients_frequency_signed": ig_frequency,
        "integrated_gradients_demographic_signed": ig_demographic,
    }
    seed_output_dir = output_dir / ("seed_level" if partition == "test" else "training_rankings") / f"fold_{fold:02d}" / f"seed_{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)
    result_path = seed_output_dir / f"{partition}_signed_attributions.npz"
    np.savez_compressed(result_path, **arrays)
    array_hashes = {key: pipeline.sha256_array(value) for key, value in arrays.items()}
    provenance = {
        "status": "PASS",
        "lineage": "cpac_model_xai_aligned_v2",
        "partition": partition,
        "outer_fold": fold,
        "base_seed": seed,
        "candidate_id": candidate.candidate_id,
        "fixed_epoch": int(checkpoint["fixed_epoch"]),
        "input_sha256": pipeline.canonical_input_hashes(),
        "frequency_input_sha256": pipeline.EXPECTED_FREQUENCY_SHA256,
        "outer_split_sha256": checkpoint["outer_split_sha256"],
        "checkpoint_path": checkpoint_path.resolve(),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_state_dict_sha256": checkpoint["model_state_dict_sha256"],
        "checkpoint_manifest_path": checkpoint_manifest_path.resolve(),
        "checkpoint_manifest_sha256": pipeline.sha256_file(checkpoint_manifest_path),
        "prediction_sha256": prediction_sha,
        "prediction_max_abs_error_vs_saved": prediction_error,
        "xai_methods": {
            "gradient_x_input": {"target": "ASD positive-class probability", "signed": True},
            "integrated_gradients": {"target": "ASD positive-class probability", "signed": True, "baseline": "zero in outer-train standardized space", "steps": ig_steps},
        },
        "outer_test_labels_used": False,
        "result_path": result_path.resolve(),
        "result_sha256": pipeline.sha256_file(result_path),
        "array_values_sha256": array_hashes,
    }
    if not pipeline.validate_xai_checkpoint_provenance(provenance):
        raise RuntimeError(f"XAI checkpoint provenance validation failed: {result_path}")
    pipeline.write_json(seed_output_dir / "provenance.json", provenance)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return arrays, provenance


def run_xai(
    run_dir: Path,
    output_dir: Path,
    folds: Sequence[int],
    device_name: str,
    ig_steps: int,
    max_subjects: int | None = None,
) -> dict[str, Any]:
    gate = pipeline.phase1_gate()
    features, _firewall, _labels = pipeline.load_canonical_inputs()
    device = pipeline.audited.choose_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_run = list(folds) == list(range(10)) and max_subjects is None
    assignment = np.zeros(len(features.subject_ids), dtype=np.int64)
    global_arrays = {
        "gradient_x_input_fc_full_signed": np.zeros((879, pipeline.audited.N_FULL_EDGES), dtype=np.float32),
        "gradient_x_input_frequency_signed": np.zeros((879, pipeline.audited.N_FREQUENCY), dtype=np.float32),
        "gradient_x_input_demographic_signed": np.zeros((879, pipeline.audited.N_DEMOGRAPHIC), dtype=np.float32),
        "integrated_gradients_fc_full_signed": np.zeros((879, pipeline.audited.N_FULL_EDGES), dtype=np.float32),
        "integrated_gradients_frequency_signed": np.zeros((879, pipeline.audited.N_FREQUENCY), dtype=np.float32),
        "integrated_gradients_demographic_signed": np.zeros((879, pipeline.audited.N_DEMOGRAPHIC), dtype=np.float32),
    }
    provenance_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    fold_records: list[dict[str, Any]] = []
    for fold in folds:
        seed_arrays: list[dict[str, np.ndarray]] = []
        fold_provenance: list[dict[str, Any]] = []
        for seed in pipeline.FINAL_SEEDS:
            arrays, provenance = compute_seed_attributions(
                run_dir,
                output_dir,
                fold,
                seed,
                features,
                device,
                ig_steps,
                partition="test",
                max_subjects=max_subjects,
            )
            seed_arrays.append(arrays)
            fold_provenance.append(provenance)
            provenance_rows.append(provenance)
            print(f"[xai] fold={fold} seed={seed} test subjects={len(arrays['row_indices'])} complete", flush=True)
        reference_indices = seed_arrays[0]["row_indices"]
        reference_subjects = seed_arrays[0]["subject_ids"]
        selected_edges = seed_arrays[0]["selected_edges"]
        for arrays in seed_arrays[1:]:
            if not np.array_equal(arrays["row_indices"], reference_indices):
                raise RuntimeError(f"Fold {fold}: seed attribution subject mismatch")
            if not np.array_equal(arrays["selected_edges"], selected_edges):
                raise RuntimeError(f"Fold {fold}: selected edges differ across final seeds")
        ensemble: dict[str, np.ndarray] = {
            "row_indices": reference_indices,
            "subject_ids": reference_subjects,
            "selected_edges": selected_edges,
        }
        for method in METHODS:
            seed_fc = np.stack([arrays[f"{method}_fc_selected_signed"] for arrays in seed_arrays], axis=0)
            seed_frequency = np.stack([arrays[f"{method}_frequency_signed"] for arrays in seed_arrays], axis=0)
            seed_demographic = np.stack([arrays[f"{method}_demographic_signed"] for arrays in seed_arrays], axis=0)
            signed_fc = pipeline.aggregate_signed_attributions(seed_fc).astype(np.float32)
            signed_frequency = pipeline.aggregate_signed_attributions(seed_frequency).astype(np.float32)
            signed_demographic = pipeline.aggregate_signed_attributions(seed_demographic).astype(np.float32)
            full_fc = pipeline.backfill_selected_edges(signed_fc, selected_edges).astype(np.float32)
            ensemble[f"{method}_fc_full_signed"] = full_fc
            ensemble[f"{method}_frequency_signed"] = signed_frequency
            ensemble[f"{method}_demographic_signed"] = signed_demographic
            global_arrays[f"{method}_fc_full_signed"][reference_indices] = full_fc
            global_arrays[f"{method}_frequency_signed"][reference_indices] = signed_frequency
            global_arrays[f"{method}_demographic_signed"][reference_indices] = signed_demographic
        assignment[reference_indices] += 1
        fold_dir = output_dir / "fold_level" / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        ensemble_path = fold_dir / "fold_three_seed_signed_ensemble.npz"
        np.savez_compressed(ensemble_path, **ensemble)
        ranking_path = fold_dir / "fold_mean_absolute_rankings.npz"
        np.savez_compressed(
            ranking_path,
            selected_edges=selected_edges,
            gradient_x_input_fc_full_mean_absolute=np.mean(np.abs(ensemble["gradient_x_input_fc_full_signed"]), axis=0).astype(np.float32),
            integrated_gradients_fc_full_mean_absolute=np.mean(np.abs(ensemble["integrated_gradients_fc_full_signed"]), axis=0).astype(np.float32),
            gradient_x_input_frequency_mean_absolute=np.mean(np.abs(ensemble["gradient_x_input_frequency_signed"]), axis=0).astype(np.float32),
            integrated_gradients_frequency_mean_absolute=np.mean(np.abs(ensemble["integrated_gradients_frequency_signed"]), axis=0).astype(np.float32),
        )
        fold_record = {
            "fold": fold,
            "n_subjects": len(reference_indices),
            "subject_ids_sha256": pipeline.sha256_array(reference_subjects),
            "selected_edges_sha256": pipeline.sha256_array(selected_edges),
            "seed_checkpoint_sha256": {str(row["base_seed"]): row["checkpoint_sha256"] for row in fold_provenance},
            "ensemble_path": ensemble_path.resolve(),
            "ensemble_sha256": pipeline.sha256_file(ensemble_path),
            "ranking_path": ranking_path.resolve(),
            "ranking_sha256": pipeline.sha256_file(ranking_path),
            "aggregation": "mean of three signed seed attributions; ranks use across-subject mean absolute ensemble attribution",
        }
        pipeline.write_json(fold_dir / "fold_provenance.json", fold_record)
        fold_records.append(fold_record)
        for position, (row_index, subject_id) in enumerate(zip(reference_indices, reference_subjects)):
            subject_rows.append({
                "subject_id": int(subject_id),
                "row_index": int(row_index),
                "outer_fold": fold,
                "fold_ensemble_row": position,
                "checkpoint_seed_42_sha256": fold_provenance[0]["checkpoint_sha256"],
                "checkpoint_seed_43_sha256": fold_provenance[1]["checkpoint_sha256"],
                "checkpoint_seed_44_sha256": fold_provenance[2]["checkpoint_sha256"],
                "held_out_only": True,
            })

    subject_index_path = output_dir / "subject_level" / "subject_attribution_index.csv"
    subject_index_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(subject_rows[0]) if subject_rows else []
    with subject_index_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(subject_rows)
    ensemble_dir = output_dir / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    global_path = ensemble_dir / "oof_three_seed_signed_attributions.npz"
    saved_global = {
        "subject_ids": features.subject_ids.astype(np.int64),
        "assignment_count": assignment,
        **global_arrays,
    }
    np.savez_compressed(global_path, **saved_global)
    rankings_path = ensemble_dir / "oof_mean_absolute_rankings.npz"
    if complete_run:
        rank_indices = np.arange(879)
    else:
        rank_indices = np.flatnonzero(assignment == 1)
    np.savez_compressed(
        rankings_path,
        gradient_x_input_fc_full_mean_absolute=np.mean(np.abs(global_arrays["gradient_x_input_fc_full_signed"][rank_indices]), axis=0).astype(np.float32),
        integrated_gradients_fc_full_mean_absolute=np.mean(np.abs(global_arrays["integrated_gradients_fc_full_signed"][rank_indices]), axis=0).astype(np.float32),
        gradient_x_input_frequency_mean_absolute=np.mean(np.abs(global_arrays["gradient_x_input_frequency_signed"][rank_indices]), axis=0).astype(np.float32),
        integrated_gradients_frequency_mean_absolute=np.mean(np.abs(global_arrays["integrated_gradients_frequency_signed"][rank_indices]), axis=0).astype(np.float32),
    )
    checks = {
        "phase1_gate_pass": gate["status"] == "PASS",
        "all_seed_provenance_checkpoint_hashes_match": all(pipeline.validate_xai_checkpoint_provenance(row) for row in provenance_rows),
        "all_xai_targets_are_asd_probability": all(
            all(method["target"] == "ASD positive-class probability" for method in row["xai_methods"].values())
            for row in provenance_rows
        ),
        "all_xai_use_canonical_lineage": all(row["lineage"] == "cpac_model_xai_aligned_v2" for row in provenance_rows),
        "all_frequency_hashes_corrected": all(row["frequency_input_sha256"] == pipeline.EXPECTED_FREQUENCY_SHA256 for row in provenance_rows),
        "all_prediction_errors_within_tolerance": all(row["prediction_max_abs_error_vs_saved"] <= pipeline.CHECKPOINT_RELOAD_ATOL for row in provenance_rows),
        "three_seed_signed_aggregation": len(provenance_rows) == 3 * len(folds),
        "oof_assignment_exactly_once": bool(np.all(assignment == 1)) if complete_run else bool(np.all(assignment[rank_indices] == 1)),
        "oof_subject_count_879": len(subject_rows) == 879 if complete_run else True,
    }
    summary = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "generated_at": pipeline.now_iso(),
        "smoke": not complete_run,
        "run_directory": run_dir.resolve(),
        "output_directory": output_dir.resolve(),
        "folds": list(folds),
        "seeds": list(pipeline.FINAL_SEEDS),
        "methods": list(METHODS),
        "target": "ASD positive-class probability",
        "integrated_gradients_steps": ig_steps,
        "seed_aggregation": "mean signed attribution across seeds 42,43,44",
        "ranking_aggregation": "mean absolute attribution across held-out subjects after signed seed mean",
        "subject_count": len(subject_rows),
        "seed_model_count": len(provenance_rows),
        "checks": checks,
        "folds_detail": fold_records,
        "subject_level_index": subject_index_path.resolve(),
        "subject_level_index_sha256": pipeline.sha256_file(subject_index_path),
        "ensemble_attributions": global_path.resolve(),
        "ensemble_attributions_sha256": pipeline.sha256_file(global_path),
        "ensemble_rankings": rankings_path.resolve(),
        "ensemble_rankings_sha256": pipeline.sha256_file(rankings_path),
        "provenance": provenance_rows,
    }
    summary_path = output_dir / "xai_lineage_audit.json"
    pipeline.write_json(summary_path, summary)
    if complete_run:
        pipeline.update_protocol_manifest(
            "XAI_COMPLETE_SANITY_ROAR_PENDING",
            xai={
                "status": summary["status"],
                "lineage_audit": summary_path,
                "ensemble_attributions": global_path,
                "ensemble_rankings": rankings_path,
                "seed_model_count": len(provenance_rows),
            },
        )
    print(json.dumps(pipeline.clean({key: value for key, value in summary.items() if key != "provenance"}), ensure_ascii=False, indent=2), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical held-out XAI with explicit run directory")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--folds", type=parse_folds, default=list(range(10)))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ig-steps", type=int, default=DEFAULT_IG_STEPS)
    parser.add_argument("--max-subjects", type=int, default=None, help="Smoke-only per-fold limit")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.ig_steps < 2:
        raise ValueError("--ig-steps must be >=2")
    if args.max_subjects is not None and args.max_subjects < 1:
        raise ValueError("--max-subjects must be positive")
    summary = run_xai(
        args.run_dir.resolve(),
        args.output_dir.resolve(),
        args.folds,
        args.device,
        args.ig_steps,
        args.max_subjects,
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

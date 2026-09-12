#!/usr/bin/env python
"""Independently verify the completed strict CMPB handoff artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results" / "cmpb_strict_v2"
FULL = RESULTS / "full"
NESTED = FULL / "abide1_nested_cv"
EXTERNAL = FULL / "abide1_to_abide2_locked_external"
OUTPUT = FULL / "final_handoff_audit.json"


def load_json(path: Path) -> dict:
    # Windows PowerShell may write JSON with a UTF-8 BOM; utf-8-sig accepts
    # both BOM and non-BOM UTF-8 files.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    checks: dict[str, dict] = {}

    input_audit = load_json(RESULTS / "input_contract_audit.json")
    adapter_path = PROJECT / "scripts" / "05_cmpb_strict_external_validation.py"
    site_config_path = PROJECT / "config" / "site_canonicalization.json"
    strict_base_path = Path(input_audit["strict_base_module"])
    observed_hashes = {
        "adapter": sha256(adapter_path),
        "site_config": sha256(site_config_path),
        "strict_base_module": sha256(strict_base_path),
    }
    recorded_hashes = {
        "adapter": input_audit["adapter_sha256"],
        "site_config": input_audit["site_config_sha256"],
        "strict_base_module": input_audit["strict_base_module_sha256"],
    }
    checks["input_contract"] = {
        "pass": (
            input_audit.get("status") == "PASS"
            and observed_hashes == recorded_hashes
            and input_audit.get("external_labels_loaded") is False
        ),
        "status": input_audit.get("status"),
        "external_labels_loaded_during_preparation": input_audit.get(
            "external_labels_loaded"
        ),
        "recorded_hashes": recorded_hashes,
        "observed_hashes": observed_hashes,
    }

    nested = load_json(NESTED / "strict_nested_summary.json")
    fold_files = sorted(NESTED.glob("fold_*/fold_metrics.json"))
    fold_metrics = [load_json(path) for path in fold_files]
    with (NESTED / "strict_nested_outer_oof_predictions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        oof_rows = list(csv.DictReader(handle))
    oof_ids = [row["subject_id"] for row in oof_rows]
    checks["strict_nested_cv"] = {
        "pass": (
            nested.get("status") == "COMPLETE_STRICT_NESTED_CV"
            and len(fold_metrics) == 10
            and all(row.get("status") == "COMPLETE" for row in fold_metrics)
            and sum(int(row["n_outer_test"]) for row in fold_metrics) == 1024
            and len(oof_rows) == 1024
            and len(set(oof_ids)) == 1024
        ),
        "status": nested.get("status"),
        "folds": len(fold_metrics),
        "outer_test_n_sum": sum(int(row["n_outer_test"]) for row in fold_metrics),
        "oof_rows": len(oof_rows),
        "unique_oof_subjects": len(set(oof_ids)),
        "outer_oof_auc": nested.get("outer_oof_auc"),
        "outer_oof_accuracy_threshold_0_5": nested.get(
            "outer_oof_accuracy_threshold_0_5"
        ),
    }

    lock_path = EXTERNAL / "prediction_lock.json"
    blind_path = EXTERNAL / "abide2_predictions_blinded.npz"
    labeled_path = EXTERNAL / "abide2_predictions_with_labels.npz"
    lock = load_json(lock_path)
    with np.load(blind_path, allow_pickle=False) as blind:
        blind_keys = list(blind.files)
        blind_probabilities = blind["probabilities"].copy()
        blind_subject_ids = blind["subject_ids"].copy()
    forbidden = {"y", "label", "labels", "y_true"}
    checks["prediction_lock"] = {
        "pass": (
            lock.get("status") == "PREDICTIONS_LOCKED_BEFORE_LABEL_LOAD"
            and lock.get("external_labels_loaded") is False
            and lock.get("prediction_file_sha256") == sha256(blind_path)
            and not forbidden.intersection(key.lower() for key in blind_keys)
        ),
        "status": lock.get("status"),
        "blind_keys": blind_keys,
        "contains_label_key": bool(
            forbidden.intersection(key.lower() for key in blind_keys)
        ),
        "recorded_sha256": lock.get("prediction_file_sha256"),
        "observed_sha256": sha256(blind_path),
    }

    with np.load(labeled_path, allow_pickle=False) as labeled:
        labeled_probabilities = labeled["probabilities"].copy()
        probabilities_equal = np.array_equal(
            blind_probabilities, labeled_probabilities
        )
        subjects_equal = np.array_equal(blind_subject_ids, labeled["subject_ids"])
        y = labeled["y"].astype(int)
        canonical_sites = labeled["canonical_sites"].astype(str)
        primary = labeled["analysis_cohort_site_disjoint_mask"].astype(bool)
        release = labeled["release_level_site_disjoint_mask"].astype(bool)

    checks["locked_vs_labeled_predictions"] = {
        "pass": probabilities_equal and subjects_equal,
        "probabilities_identical": probabilities_equal,
        "maximum_absolute_probability_difference": float(
            np.max(np.abs(blind_probabilities - labeled_probabilities))
        ),
        "subject_ids_identical": subjects_equal,
    }

    site_config = load_json(site_config_path)
    analyzed_shared = set(input_audit["shared_canonical_sites_in_analyzed_cohorts"])
    release_shared = {
        row["canonical"]
        for row in site_config["abide_ii"].values()
        if row["participated_in_abide_i_release"]
    }
    primary_sites = set(canonical_sites[primary])
    release_sites = set(canonical_sites[release])
    checks["site_disjoint_masks"] = {
        "pass": (
            int(primary.sum()) == 343
            and int(y[primary].sum()) == 159
            and int(release.sum()) == 257
            and int(y[release].sum()) == 125
            and not primary_sites.intersection(analyzed_shared)
            and not release_sites.intersection(release_shared)
            and bool(np.all(~release | primary))
        ),
        "analysis_cohort": {
            "n": int(primary.sum()),
            "asd": int(y[primary].sum()),
            "td": int(primary.sum() - y[primary].sum()),
            "canonical_sites": sorted(primary_sites),
            "overlap_with_analyzed_abide_i": sorted(
                primary_sites.intersection(analyzed_shared)
            ),
        },
        "release_level": {
            "n": int(release.sum()),
            "asd": int(y[release].sum()),
            "td": int(release.sum() - y[release].sum()),
            "canonical_sites": sorted(release_sites),
            "overlap_with_abide_i_release": sorted(
                release_sites.intersection(release_shared)
            ),
        },
        "release_mask_is_subset_of_primary": bool(np.all(~release | primary)),
    }

    external = load_json(EXTERNAL / "external_validation_summary.json")
    evals = external["evaluations"]
    bootstrap_ok = all(
        row["subject_bootstrap"]["iterations"] == 2000
        and row["site_cluster_bootstrap"]["requested_iterations"] == 2000
        and row["site_cluster_bootstrap"]["valid_iterations"] == 2000
        for row in evals.values()
    )
    integrity = external["selection_integrity"]
    checks["external_validation"] = {
        "pass": (
            external.get("status") == "COMPLETE_LOCKED_EXTERNAL_VALIDATION"
            and bootstrap_ok
            and integrity.get("predictions_locked_before_external_label_load") is True
            and integrity.get("abide_ii_used_for_selection") is False
            and external.get("fixed_threshold") == 0.5
        ),
        "status": external.get("status"),
        "selected_candidate": external.get("selected_candidate"),
        "fixed_epoch": external.get("fixed_epoch"),
        "fixed_threshold": external.get("fixed_threshold"),
        "bootstrap_complete": bootstrap_ok,
        "metrics": {name: row["metrics"] for name, row in evals.items()},
    }

    launcher = load_json(RESULTS / "full_launcher_status.json")
    stderr_path = RESULTS / "full_run_stderr.log"
    checks["launcher"] = {
        "pass": (
            launcher.get("status") == "COMPLETE"
            and launcher.get("exit_code") == 0
            and stderr_path.stat().st_size == 0
        ),
        "status": launcher.get("status"),
        "exit_code": launcher.get("exit_code"),
        "stderr_bytes": stderr_path.stat().st_size,
    }

    report = {
        "status": "PASS" if all(row["pass"] for row in checks.values()) else "FAIL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

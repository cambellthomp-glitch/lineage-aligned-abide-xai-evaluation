from __future__ import annotations

import csv
import importlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ReleaseCandidateSmokeTests(unittest.TestCase):
    def test_core_modules_import(self) -> None:
        for name in ("cpac_leakage_free_model", "cpac_leakage_free_nested_cv", "cpac_model_optimization_v1", "canonical_pipeline_v2", "canonical_xai_v2", "canonical_sanity_roar_v2"):
            self.assertIsNotNone(importlib.import_module(name))

    def test_candidate_registry_has_twelve_rows(self) -> None:
        path = ROOT / "configs" / "candidate_configuration_table.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 12)

    def test_aggregate_files_have_no_identifier_columns(self) -> None:
        forbidden = {"subject_id", "participant_id", "file_id", "subjectkey"}
        for path in (ROOT / "aggregate_source_data").glob("*.csv"):
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = {item.strip().lower() for item in next(reader, [])}
            self.assertFalse(header & forbidden, path.name)

    def test_external_validation_public_package(self) -> None:
        required = (
            ROOT / "external_validation" / "config" / "analysis_spec.yaml",
            ROOT / "external_validation" / "config" / "site_canonicalization.json",
            ROOT / "external_validation" / "results" / "prediction_lock_public.json",
            ROOT / "external_validation" / "results" / "abide1_nested_cv_aggregate.json",
            ROOT / "external_validation" / "results" / "abide2_external_validation_aggregate.json",
            ROOT / "external_validation" / "results" / "public_release_verification.json",
        )
        for path in required:
            self.assertTrue(path.is_file(), path)

        lock = json.loads(required[2].read_text(encoding="utf-8"))
        self.assertEqual(lock["status"], "PREDICTIONS_LOCKED_BEFORE_LABEL_LOAD")
        self.assertFalse(lock["external_labels_loaded"])
        self.assertNotIn("external_subject_ids_sha256", lock)
        self.assertNotIn("subject_ids", lock)

        aggregate = json.loads(required[4].read_text(encoding="utf-8"))
        self.assertEqual(aggregate["status"], "COMPLETE_LOCKED_EXTERNAL_VALIDATION")
        self.assertEqual(aggregate["evaluations"]["full_abide_ii"]["metrics"]["n"], 883)


if __name__ == "__main__":
    unittest.main()

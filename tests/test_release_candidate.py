from __future__ import annotations

import csv
import importlib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


if __name__ == "__main__":
    unittest.main()

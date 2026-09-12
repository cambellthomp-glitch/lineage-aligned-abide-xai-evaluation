"""Join official ABIDE phenotype files to the fMRIPrep derivative universe.

The derivative subject ID preserves the original numeric ABIDE ID after `x`.
This script deliberately retains every match and records each unmatched or
incomplete record; it never filters based on imaging QC or model outcomes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
MANIFESTS = PROJECT / "manifests"
METADATA = PROJECT / "metadata"


def column(frame: pd.DataFrame, choices: tuple[str, ...]) -> str:
    for name in choices:
        if name in frame.columns:
            return name
    raise ValueError(f"None of the expected columns are present: {choices}; available: {frame.columns.tolist()}")


def read_phenotype_csv(path: Path) -> tuple[pd.DataFrame, str]:
    """Read official phenotype exports with their original legacy encodings."""
    failures = []
    for encoding in ("utf-8-sig", "cp1252", "latin1"):
        try:
            frame = pd.read_csv(path, dtype=str, encoding=encoding, low_memory=False)
            frame.columns = frame.columns.map(lambda value: str(value).strip())
            return frame, encoding
        except UnicodeDecodeError as exc:
            failures.append(f"{encoding}: {exc}")
    raise UnicodeError("Could not decode phenotype CSV. " + " | ".join(failures))


def normalize_phenotype(path: Path, cohort: str) -> tuple[pd.DataFrame, str]:
    frame, encoding = read_phenotype_csv(path)
    subject = column(frame, ("SUB_ID", "subject_id", "participant_id"))
    diagnosis = column(frame, ("DX_GROUP", "diagnosis", "DXGROUP"))
    age = column(frame, ("AGE_AT_SCAN", "AGE", "age"))
    sex = column(frame, ("SEX", "sex", "SEX_CODE"))
    site = next((name for name in ("SITE_ID", "SITE", "site_id") if name in frame.columns), None)
    output = pd.DataFrame(
        {
            "original_subject_id": pd.to_numeric(frame[subject], errors="coerce").astype("Int64"),
            "diagnosis_source": frame[diagnosis],
            "age": pd.to_numeric(frame[age], errors="coerce"),
            "sex_source": frame[sex],
            "site_name_source": frame[site] if site else pd.Series([pd.NA] * len(frame), index=frame.index),
        }
    )
    output["cohort"] = cohort
    output["label"] = output["diagnosis_source"].astype(str).str.strip().map({"1": 1, "2": 0})
    output["sex"] = output["sex_source"].astype(str).str.strip().map({"1": 1, "2": 0, "M": 1, "F": 0, "male": 1, "female": 0})
    return output.drop_duplicates(subset=["original_subject_id"], keep="first"), encoding


def original_id(subject_id: str) -> int | None:
    match = re.search(r"x(\d+)$", str(subject_id))
    return int(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=MANIFESTS / "derivative_inventory.csv")
    parser.add_argument("--abide1-phenotype", type=Path, required=True)
    parser.add_argument("--abide2-phenotype", type=Path, required=True)
    args = parser.parse_args()
    inventory = pd.read_csv(args.inventory, dtype={"subject_id": str})
    abide1, abide1_encoding = normalize_phenotype(args.abide1_phenotype, "ABIDE_I")
    abide2, abide2_encoding = normalize_phenotype(args.abide2_phenotype, "ABIDE_II")
    phenotype = pd.concat([abide1, abide2], ignore_index=True)
    inventory["original_subject_id"] = inventory["subject_id"].map(original_id).astype("Int64")
    inventory["site_id"] = inventory["subject_id"].str.extract(r"^(v[12]s\d+)x", expand=False)
    merged = inventory.merge(phenotype, on=["cohort", "original_subject_id"], how="left", validate="one_to_one")
    merged["site_id"] = merged["site_name_source"].fillna(merged["site_id"])
    complete = merged[["label", "age", "sex"]].notna().all(axis=1)
    output = merged.loc[complete, ["subject_id", "cohort", "site_id", "label", "age", "sex"]].copy()
    output["label"] = output["label"].astype(int)
    output["sex"] = output["sex"].astype(int)
    METADATA.mkdir(parents=True, exist_ok=True)
    output.to_csv(METADATA / "participants_harmonized.csv", index=False)
    exclusions = merged.loc[~complete, ["subject_id", "cohort", "original_subject_id", "diagnosis_source", "age", "sex_source"]].copy()
    exclusions["reason"] = "Missing phenotype match, diagnosis, age, or binary sex after deterministic source-ID join"
    exclusions.to_csv(METADATA / "phenotype_join_exclusions.csv", index=False)
    report = {
        "n_inventory": int(len(inventory)),
        "n_harmonized": int(len(output)),
        "n_excluded_for_missing_phenotype": int(len(exclusions)),
        "counts_by_cohort": output["cohort"].value_counts().to_dict(),
        "input_encodings": {"ABIDE_I": abide1_encoding, "ABIDE_II": abide2_encoding},
        "site_identifier": "Official phenotype site name when available; otherwise fMRIPrep encoded source-site index (v1s*/v2s*).",
    }
    (METADATA / "phenotype_harmonization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

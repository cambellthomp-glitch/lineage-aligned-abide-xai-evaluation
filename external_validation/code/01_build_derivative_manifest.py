"""Inventory an ABIDE-fMRIPrep derivative clone without fetching annex content.

The locked run rule uses the earliest session and its lowest-numbered resting
run for every participant. Additional runs are recorded for transparency but
are never used as independent subjects in this cross-sectional analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT / "manifests"
REQUIRED_PHENOTYPE_COLUMNS = {"subject_id", "cohort", "site_id", "label", "age", "sex"}


def cohort_from_subject(subject_id: str) -> str:
    if subject_id.startswith("v1"):
        return "ABIDE_I"
    if subject_id.startswith("v2"):
        return "ABIDE_II"
    return "UNKNOWN"


def subject_from_bold(path: Path) -> str:
    for part in path.parts:
        if part.startswith("sub-"):
            return part.removeprefix("sub-")
    raise ValueError(f"Could not infer BIDS subject from {path}")


def find_sidecar(bold: Path, suffix: str) -> Path | None:
    stem = bold.name
    if stem.endswith("_bold.nii.gz"):
        candidate = bold.with_name(stem.replace("_bold.nii.gz", suffix))
        if candidate.exists():
            return candidate
    matches = sorted(bold.parent.glob(f"{bold.name.split('_space-')[0]}*{suffix}"))
    return matches[0] if len(matches) == 1 else None


def build_inventory(root: Path) -> list[dict[str, str]]:
    patterns = [
        "**/*_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
        "**/*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
    ]
    bold_files = sorted({path.resolve() for pattern in patterns for path in root.glob(pattern)})
    rows: list[dict[str, str]] = []
    for bold in bold_files:
        subject_id = subject_from_bold(bold)
        confounds = find_sidecar(bold, "_desc-confounds_timeseries.tsv")
        if confounds is None:
            confounds = find_sidecar(bold, "_desc-confounds_regressors.tsv")
        bold_json = find_sidecar(bold, "_bold.json")
        rows.append(
            {
                "subject_id": subject_id,
                "cohort": cohort_from_subject(subject_id),
                "bold_path": str(bold),
                "confounds_path": "" if confounds is None else str(confounds.resolve()),
                "bold_json_path": "" if bold_json is None else str(bold_json.resolve()),
                "run_key": bold.name.removesuffix(".nii.gz"),
            }
        )
    return rows


def run_priority(row: dict[str, str]) -> tuple[int, int, str, int, str, str]:
    """Pre-specified baseline rule: earliest session, then lowest rest run."""
    name = row["run_key"]
    session_match = re.search(r"_ses-([^_]+)", name)
    run_match = re.search(r"_run-([^_]+)", name)
    session = session_match.group(1) if session_match else "1"
    run = run_match.group(1) if run_match else "1"
    session_number = int(session) if session.isdigit() else 10**9
    run_number = int(run) if run.isdigit() else 10**9
    return (0 if session == "1" else 1, session_number, session, 0 if run == "1" else 1, run_number, name)


def select_one_run(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["subject_id"], []).append(row)
    selected: list[dict[str, str]] = []
    not_selected: list[dict[str, str]] = []
    for subject_id, subject_rows in sorted(grouped.items()):
        chosen = min(subject_rows, key=run_priority)
        selected.append(chosen)
        for row in subject_rows:
            if row is not chosen:
                not_selected.append(
                    {
                        **row,
                        "reason": "Additional rest run/session not selected by the pre-specified earliest-session, lowest-run rule",
                        "selected_run_key": chosen["run_key"],
                    }
                )
    return selected, not_selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derivatives-root", type=Path, required=True)
    parser.add_argument("--phenotype", type=Path, help="Harmonized metadata CSV with required project columns")
    parser.add_argument(
        "--reuse-inventory",
        action="store_true",
        help="Reuse the existing audited derivative_inventory.csv instead of rescanning/overwriting it.",
    )
    args = parser.parse_args()
    if not args.derivatives_root.exists():
        raise SystemExit(f"Derivative root not found: {args.derivatives_root}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inventory_path = OUT_DIR / "derivative_inventory.csv"
    additional_path = OUT_DIR / "additional_runs_not_selected.csv"
    if args.reuse_inventory:
        if not inventory_path.exists():
            raise SystemExit(f"Cannot reuse missing inventory: {inventory_path}")
        inventory = pd.read_csv(inventory_path, dtype=str).to_dict(orient="records")
        not_selected = pd.read_csv(additional_path, dtype=str).to_dict(orient="records") if additional_path.exists() else []
    else:
        inventory, not_selected = select_one_run(build_inventory(args.derivatives_root))
        if not inventory:
            raise SystemExit("No fMRIPrep MNI152NLin2009cAsym preprocessed BOLD files were found.")
        pd.DataFrame(inventory).to_csv(inventory_path, index=False)
        pd.DataFrame(not_selected).to_csv(additional_path, index=False)
    report = {
        "derivatives_root": str(args.derivatives_root.resolve()),
        "n_unique_participant_candidates": len(inventory),
        "n_additional_runs_not_selected": len(not_selected),
        "run_selection_rule": "Earliest session, then lowest-numbered resting-state run; applied identically before phenotype/QC/model inspection.",
        "counts_by_cohort": pd.Series([row["cohort"] for row in inventory]).value_counts().to_dict(),
    }
    if args.phenotype:
        phenotypes = pd.read_csv(args.phenotype, dtype={"subject_id": str})
        missing = REQUIRED_PHENOTYPE_COLUMNS - set(phenotypes.columns)
        if missing:
            raise SystemExit("Phenotype file missing columns: " + ", ".join(sorted(missing)))
        queue = pd.DataFrame(inventory).merge(phenotypes, on=["subject_id", "cohort"], how="left", validate="one_to_one")
        queue["eligible_metadata"] = queue[["site_id", "label", "age", "sex"]].notna().all(axis=1)
        queue.to_csv(OUT_DIR / "streaming_queue.csv", index=False)
        report["n_with_complete_metadata"] = int(queue["eligible_metadata"].sum())
        report["n_missing_metadata"] = int((~queue["eligible_metadata"]).sum())
    (OUT_DIR / "derivative_inventory_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build compact ABIDE-I and ABIDE-II feature arrays from verified CC200 matrices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.signal import welch


PROJECT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = PROJECT / "features"
EXPECTED_COLUMNS = {"subject_id", "cohort", "site_id", "label", "age", "sex", "time_series_path", "tr_seconds", "cc200_rois"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fc_features(time_series: np.ndarray) -> np.ndarray:
    correlation = np.corrcoef(time_series.T)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    lower = correlation[np.tril_indices(200, k=-1)]
    return np.arctanh(np.clip(lower, -0.999999, 0.999999)).astype(np.float32)


def psd_features(time_series: np.ndarray, tr_seconds: float, target_frequencies: np.ndarray) -> np.ndarray:
    sampling_frequency = 1.0 / float(tr_seconds)
    nperseg = min(128, time_series.shape[0])
    frequencies, power = welch(time_series, fs=sampling_frequency, axis=0, nperseg=nperseg, detrend="linear")
    if frequencies[-1] < target_frequencies[-1]:
        raise ValueError(f"Nyquist frequency {frequencies[-1]:.4f} Hz is below requested 0.10 Hz")
    output = np.empty((len(target_frequencies), 200), dtype=np.float32)
    for roi in range(200):
        output[:, roi] = np.interp(target_frequencies, frequencies, power[:, roi])
    return np.log1p(np.maximum(output, 0.0)).reshape(-1)


def verify_metadata(index: pd.DataFrame, phenotype: Path | None) -> pd.DataFrame:
    if phenotype is None:
        return index
    metadata = pd.read_csv(phenotype, dtype={"subject_id": str})
    key = ["subject_id", "cohort"]
    merged = index.merge(metadata, on=key, how="left", suffixes=("", "_source"), validate="one_to_one")
    for column in ("site_id", "label", "age", "sex"):
        source = f"{column}_source"
        if source not in merged:
            raise ValueError(f"Harmonized phenotype is missing {column}")
        mismatched = merged[column].astype(str) != merged[source].astype(str)
        if mismatched.any():
            examples = merged.loc[mismatched, ["subject_id", "cohort", column, source]].head(5)
            raise ValueError(f"Metadata mismatch for {column}:\n{examples.to_string(index=False)}")
    return index


def write_cohort(cohort: str, table: pd.DataFrame, targets: np.ndarray) -> dict[str, object]:
    rows = []
    rejected = []
    for _, row in table.sort_values("subject_id").iterrows():
        try:
            ts = np.load(str(row["time_series_path"]), allow_pickle=False)
            if ts.ndim != 2 or ts.shape[1] != 200 or ts.shape[0] < 100:
                raise ValueError(f"invalid time series shape {ts.shape}")
            fc = fc_features(ts)
            psd = psd_features(ts, float(row["tr_seconds"]), targets)
            rows.append((row, fc, psd))
        except Exception as exc:
            rejected.append({"subject_id": str(row["subject_id"]), "reason": str(exc)})
    if not rows:
        raise RuntimeError(f"No usable {cohort} records.")
    cohort_dir = FEATURE_ROOT / cohort.lower()
    cohort_dir.mkdir(parents=True, exist_ok=True)
    np.save(cohort_dir / "X_fc_raw.npy", np.stack([item[1] for item in rows]))
    np.save(cohort_dir / "X_psd.npy", np.stack([item[2] for item in rows]))
    np.save(cohort_dir / "X_demo_raw.npy", np.asarray([[float(item[0]["age"]), float(item[0]["sex"])] for item in rows], dtype=np.float32))
    np.save(cohort_dir / "y_labels.npy", np.asarray([int(item[0]["label"]) for item in rows], dtype=np.int64))
    np.save(cohort_dir / "subject_ids.npy", np.asarray([str(item[0]["subject_id"]) for item in rows], dtype=str))
    np.save(cohort_dir / "site_ids.npy", np.asarray([str(item[0]["site_id"]) for item in rows], dtype=str))
    used = pd.DataFrame([item[0] for item in rows])
    used.to_csv(cohort_dir / "feature_manifest.csv", index=False)
    pd.DataFrame(rejected).to_csv(cohort_dir / "feature_rejections.csv", index=False)
    return {
        "n_input": int(len(table)),
        "n_included": int(len(rows)),
        "n_rejected": int(len(rejected)),
        "class_counts": {str(int(label)): int(count) for label, count in zip(*np.unique(np.asarray([item[0]["label"] for item in rows], dtype=int), return_counts=True))},
        "shapes": {"X_fc_raw": [len(rows), 19900], "X_psd": [len(rows), 3000], "X_demo_raw": [len(rows), 2]},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--phenotype", type=Path)
    args = parser.parse_args()
    if not args.index.exists():
        raise SystemExit(f"Extraction index not found: {args.index}")
    index = pd.read_csv(args.index, dtype={"subject_id": str})
    missing = EXPECTED_COLUMNS - set(index.columns)
    if missing:
        raise SystemExit("Index missing columns: " + ", ".join(sorted(missing)))
    index = verify_metadata(index, args.phenotype)
    spec_path = PROJECT / "config" / "analysis_spec.yaml"
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    targets = np.asarray(spec["features"]["spectral_frequency_hz"], dtype=float)
    if len(targets) != 15:
        raise SystemExit("Analysis spec must define exactly 15 spectral frequencies.")
    report = {
        "analysis_spec_sha256": sha256(spec_path),
        "feature_recipe": spec["features"],
        "cohorts": {},
    }
    for cohort in ("ABIDE_I", "ABIDE_II"):
        subset = index[index["cohort"] == cohort].copy()
        report["cohorts"][cohort] = write_cohort(cohort, subset, targets)
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    (FEATURE_ROOT / "feature_build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

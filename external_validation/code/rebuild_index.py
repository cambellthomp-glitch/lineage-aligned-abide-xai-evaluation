"""Rebuild extracted_timeseries/index.csv from individual provenance JSON files."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "extracted_timeseries"

prov_files = sorted(OUTPUT.rglob("*_cc200_provenance.json"))
print(f"Found {len(prov_files)} provenance files")

records = []
for pf in prov_files:
    try:
        data = json.loads(pf.read_text(encoding="utf-8"))
        records.append(data)
    except Exception as exc:
        print(f"  SKIP {pf.name}: {exc}")

if not records:
    print("No valid provenance files found!")
    raise SystemExit(1)

# Build clean index
rows = []
for rec in records:
    rows.append({
        "subject_id": rec["subject_id"],
        "cohort": rec["cohort"],
        "site_id": rec.get("site_id", ""),
        "label": rec.get("label", -1),
        "age": rec.get("age", -1.0),
        "sex": rec.get("sex", -1),
        "time_series_path": rec.get("time_series_path", ""),
        "source_bold": rec.get("source_bold", ""),
        "source_bold_url": rec.get("source_bold_url", ""),
        "source_confounds": rec.get("source_confounds", ""),
        "source_confounds_url": rec.get("source_confounds_url", ""),
        "source_bold_json": rec.get("source_bold_json", ""),
        "source_bold_json_url": rec.get("source_bold_json_url", ""),
        "source_run": rec.get("source_run", ""),
        "atlas_path": rec.get("atlas_path", ""),
        "atlas_sha256": rec.get("atlas_sha256", ""),
        "analysis_spec_sha256": rec.get("analysis_spec_sha256", ""),
        "tr_seconds": rec.get("tr_seconds", 0.0),
        "retained_volumes": rec.get("retained_volumes", 0),
        "cc200_rois": rec.get("cc200_rois", 200),
        "confound_columns": str(rec.get("confound_columns", [])),
        "source_bold_derivative_relative": rec.get("source_bold_derivative_relative", ""),
        "source_confounds_derivative_relative": rec.get("source_confounds_derivative_relative", ""),
        "source_bold_json_derivative_relative": rec.get("source_bold_json_derivative_relative", ""),
    })

df = pd.DataFrame(rows)
df = df.sort_values(["cohort", "subject_id"]).drop_duplicates(subset=["subject_id"], keep="last")

# Backup old index
old = OUTPUT / "index.csv"
if old.exists():
    bak = OUTPUT / f"index.csv.bak.{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    old.rename(bak)
    print(f"Backed up corrupted index to {bak.name}")

df.to_csv(old, index=False)
print(f"Rebuilt index: {len(df)} subjects")
print(f"  ABIDE_I: {(df['cohort']=='ABIDE_I').sum()}")
print(f"  ABIDE_II: {(df['cohort']=='ABIDE_II').sum()}")

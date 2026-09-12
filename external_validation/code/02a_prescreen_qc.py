"""Offline QC pre-screening without downloading any BOLD files.

Reads annex pointers to extract real file sizes and locally-available confounds TSVs
to pre-compute retained volumes.  Produces a sorted QC-pass queue that the streaming
extractor can consume directly, smallest files first.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
MOTION_BASES = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
MOTION_SUFFIXES = ("", "_derivative1", "_power2", "_derivative1_power2")

# ── annex pointer parsing ────────────────────────────────────────────────

_ANNEX_SIZE_RE = re.compile(r"MD5E-s(\d+)--")


def parse_annex_size(path: Path) -> int | None:
    """Return the real file size encoded in a git-annex pointer, or None."""
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    if ".git/annex/objects/" not in text.replace("\\", "/") and "MD5E-s" not in text:
        # This is a real file, not an annex pointer — return its actual size
        try:
            return path.stat().st_size
        except OSError:
            return None
    match = _ANNEX_SIZE_RE.search(text)
    if match:
        return int(match.group(1))
    return None


# ── confounds-based retained-volume pre-computation ──────────────────────


def compute_retained_volumes(confounds_path: Path, fd_threshold: float = 0.50) -> tuple[int, int, str | None]:
    """Return (retained_volumes, total_volumes, error_reason).

    Replicates the filtering logic of ``choose_confounds()`` from
    ``02_stream_extract_cc200.py`` without importing nibabel/nilearn.
    """
    if not confounds_path.exists():
        return 0, 0, "confounds file missing"

    try:
        confounds = pd.read_csv(confounds_path, sep="\t")
    except Exception as exc:
        return 0, 0, f"confounds read error: {exc}"

    total_volumes = len(confounds)
    if total_volumes == 0:
        return 0, 0, "empty confounds"

    # --- framewise displacement filter ---
    if "framewise_displacement" not in confounds.columns:
        return 0, total_volumes, "missing framewise_displacement column"

    fd = pd.to_numeric(confounds["framewise_displacement"], errors="coerce")
    retain = np.isfinite(fd.to_numpy()) & (fd.to_numpy(dtype=float) <= fd_threshold)

    # --- non-steady-state outlier filter ---
    nonsteady = [c for c in confounds.columns if c.startswith("non_steady_state_outlier_")]
    if nonsteady:
        ns_data = confounds[nonsteady].fillna(0).to_numpy(dtype=float)
        retain &= ns_data.sum(axis=1) == 0

    retained = int(retain.sum())

    # Check that required motion + aCompCor columns exist (informational only)
    missing_motion = []
    for base in MOTION_BASES:
        for suffix in MOTION_SUFFIXES:
            col = base + suffix
            if col not in confounds.columns:
                missing_motion.append(col)
    acompcor_count = len([c for c in confounds.columns if c.startswith("a_comp_cor_")])

    if missing_motion:
        return retained, total_volumes, f"missing motion columns: {missing_motion[:4]}..."
    if acompcor_count < 5:
        return retained, total_volumes, f"only {acompcor_count} aCompCor columns"

    return retained, total_volumes, None


# ── relative path recovery (reused from 02_stream_extract_cc200) ────────


def relative_from_queue(value: Any, dataset_root: Path) -> Path | None:
    """Recover a derivative-relative path even if the queue value is an
    absolute Windows path with mojibake."""
    if pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    candidate = Path(raw)
    try:
        if candidate.exists():
            return candidate.resolve().relative_to(dataset_root.resolve())
    except ValueError:
        pass
    normalized = raw.replace("\\", "/")
    marker = f"{dataset_root.name}/"
    index = normalized.lower().find(marker.lower())
    if index >= 0:
        return Path(normalized[index + len(marker) :])
    subject_index = normalized.find("sub-")
    if subject_index >= 0:
        return Path(normalized[subject_index:])
    raise ValueError(f"Cannot recover derivative-relative path from queue value: {raw}")


# ── main ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline QC pre-screening for ABIDE streaming queue")
    parser.add_argument(
        "--queue",
        type=Path,
        default=PROJECT / "manifests" / "streaming_queue.csv",
        help="Path to streaming queue CSV",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT / "data_sources" / "fmriprep-25.2",
        help="Path to fMRIPrep derivative clone",
    )
    parser.add_argument(
        "--fd-threshold",
        type=float,
        default=0.50,
        help="Framewise displacement threshold in mm",
    )
    parser.add_argument(
        "--min-retained",
        type=int,
        default=100,
        help="Minimum retained volumes for QC pass",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT / "manifests",
        help="Directory for output manifests",
    )
    args = parser.parse_args()

    if not args.queue.exists():
        raise SystemExit(f"Queue file not found: {args.queue}")
    if not args.dataset_root.exists():
        raise SystemExit(f"Dataset root not found: {args.dataset_root}")

    queue = pd.read_csv(args.queue, dtype={"subject_id": str})
    required = {"subject_id", "cohort", "bold_path", "confounds_path"}
    missing = required - set(queue.columns)
    if missing:
        raise SystemExit("Queue missing columns: " + ", ".join(sorted(missing)))

    print(f"Pre-screening {len(queue)} subjects from {args.queue.name}")
    print(f"  dataset root : {args.dataset_root}")
    print(f"  FD threshold : {args.fd_threshold} mm")
    print(f"  min retained : {args.min_retained}")
    print()

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    t_start = time.monotonic()

    for idx, (_, row) in enumerate(queue.iterrows()):
        sid = str(row["subject_id"])
        cohort = str(row.get("cohort", ""))

        # --- resolve confounds path ---
        try:
            confounds_rel = relative_from_queue(row["confounds_path"], args.dataset_root)
            confounds_path = args.dataset_root / confounds_rel if confounds_rel else None
        except Exception as exc:
            rows.append(
                {
                    "subject_id": sid,
                    "cohort": cohort,
                    "bold_size_bytes": None,
                    "bold_size_mb": None,
                    "total_volumes": None,
                    "retained_volumes": None,
                    "qc_pass": False,
                    "error": f"confounds path resolution: {exc}",
                }
            )
            errors.append({"subject_id": sid, "cohort": cohort, "reason": str(exc)})
            continue

        # --- parse annex pointer for BOLD size ---
        bold_size: int | None = None
        bold_size_mb: float | None = None
        try:
            bold_rel = relative_from_queue(row["bold_path"], args.dataset_root)
            bold_path = args.dataset_root / bold_rel if bold_rel else None
            if bold_path:
                bold_size = parse_annex_size(bold_path)
                if bold_size is not None:
                    bold_size_mb = round(bold_size / (1024 * 1024), 1)
        except Exception:
            pass

        # --- compute retained volumes from confounds ---
        retained: int | None = None
        total_vols: int | None = None
        error_reason: str | None = None
        if confounds_path is not None and confounds_path.exists():
            retained, total_vols, error_reason = compute_retained_volumes(confounds_path, args.fd_threshold)
        else:
            error_reason = "confounds file not found on disk"

        qc_pass = (retained is not None and retained >= args.min_retained) if error_reason is None else False

        def _safe_int(val: Any, default: int = -1) -> int:
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return default

        def _safe_float(val: Any, default: float = -1.0) -> float:
            try:
                return float(val)
            except (ValueError, TypeError):
                return default

        row_data = {
            "subject_id": sid,
            "cohort": cohort,
            "site_id": str(row.get("site_id", "")),
            "label": _safe_int(row.get("label", -1)),
            "age": _safe_float(row.get("age", -1.0)),
            "sex": _safe_int(row.get("sex", -1)),
            "bold_size_bytes": bold_size,
            "bold_size_mb": bold_size_mb,
            "total_volumes": total_vols,
            "retained_volumes": retained,
            "qc_pass": qc_pass,
            "error": error_reason,
            "bold_path": str(row.get("bold_path", "")),
            "confounds_path": str(row.get("confounds_path", "")),
            "bold_json_path": str(row.get("bold_json_path", "")),
            "run_key": str(row.get("run_key", "")),
            "eligible_metadata": str(row.get("eligible_metadata", "")).strip().lower() in {"true", "1", "yes"},
        }
        rows.append(row_data)
        if error_reason:
            errors.append({"subject_id": sid, "cohort": cohort, "reason": error_reason})

        # --- progress ---
        if (idx + 1) % 200 == 0:
            elapsed = time.monotonic() - t_start
            n_pass = sum(1 for r in rows if r["qc_pass"])
            print(f"  [{idx + 1:5d}/{len(queue)}]  {n_pass} QC-pass  {elapsed:.0f}s elapsed", flush=True)

    elapsed = time.monotonic() - t_start
    n_pass = sum(1 for r in rows if r["qc_pass"])
    n_fail = len(rows) - n_pass
    print(f"\nDone: {len(rows)} subjects in {elapsed:.0f}s  ({elapsed / max(len(rows), 1):.3f}s/subject)")
    print(f"  QC pass : {n_pass}")
    print(f"  QC fail : {n_fail}")
    if errors:
        print(f"  Errors  : {len(errors)}")

    manifest = pd.DataFrame(rows)
    manifest.sort_values(["cohort", "subject_id"], inplace=True)

    # --- full manifest ---
    manifest_path = args.output_dir / "prescreen_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nFull manifest written: {manifest_path}")

    # --- QC-pass sorted queue ---
    qc_pass_df = manifest[manifest["qc_pass"] == True].copy()  # noqa: E712
    # Sort by bold_size_bytes ascending (None → very large so they go last)
    qc_pass_df["_sort_key"] = qc_pass_df["bold_size_bytes"].fillna(2**63 - 1)
    qc_pass_df.sort_values("_sort_key", inplace=True)
    qc_pass_df.drop(columns=["_sort_key"], inplace=True)

    # Drop the prescreen-specific columns that aren't in the original queue format
    # but keep the original queue columns plus the new prescreen columns
    qc_pass_queue_path = args.output_dir / "streaming_queue_qc_pass_sorted.csv"
    qc_pass_df.to_csv(qc_pass_queue_path, index=False)
    print(f"QC-pass sorted queue: {qc_pass_queue_path}  ({len(qc_pass_df)} subjects)")

    # --- error log ---
    if errors:
        err_path = args.output_dir / "prescreen_errors.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False)
        print(f"Prescreen errors logged: {err_path}")

    # --- brief summary to stdout ---
    for cohort_name in ("ABIDE_I", "ABIDE_II"):
        cohort_mask = qc_pass_df["cohort"] == cohort_name
        if cohort_mask.any():
            sizes = qc_pass_df.loc[cohort_mask, "bold_size_mb"]
            print(
                f"  {cohort_name}: {cohort_mask.sum()} QC-pass, "
                f"file sizes {sizes.min():.0f}–{sizes.max():.0f} MB "
                f"(median {sizes.median():.0f} MB)"
            )

    return 0 if not errors else 0  # errors are logged but not fatal


if __name__ == "__main__":
    raise SystemExit(main())

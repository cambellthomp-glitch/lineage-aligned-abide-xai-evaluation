"""Stream fMRIPrep derivatives from GIN and extract CC200 time series.

The derivative clone is used as a lightweight file manifest plus local source
for regular metadata files. Large annexed BOLD files are downloaded directly
from GIN raw URLs into a temporary project cache, extracted, then deleted.
Start with --dry-run; --execute is required for HTTP downloads.

Enhanced with:
  - Auto-resume: skips subjects already in extracted_timeseries/index.csv
  - Parallel extraction: --jobs N (default 1; max recommended 2)
  - Checkpoint reporting: periodic progress JSON
  - Graceful interrupt handling: cleans .part files on Ctrl+C
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.parse
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT / "extracted_timeseries"
STREAM_CACHE = PROJECT / ".stream_cache"
MOTION_BASES = ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
MOTION_SUFFIXES = ("", "_derivative1", "_power2", "_derivative1_power2")

# Keep all acquisition dependencies project-local; this also makes a scheduled
# full-cohort run independent of the caller's interactive PATH configuration.
_project_tool_paths = [PROJECT / ".tools" / "mingit" / "cmd", PROJECT / ".tools" / "venv" / "Scripts"]
if all(path.exists() for path in _project_tool_paths):
    os.environ["PATH"] = os.pathsep.join([*(str(path) for path in _project_tool_paths), os.environ.get("PATH", "")])

# ── proxy auto-detection ─────────────────────────────────────────────────


def _detect_proxy() -> str | None:
    """Return proxy URL from environment variables only (safe for subprocesses)."""
    for var in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
        val = os.environ.get(var)
        if val:
            return val
    return None


def _setup_proxy() -> None:
    """Detect and log proxy settings from environment."""
    proxy = _detect_proxy()
    if proxy:
        print(f"[PROXY] Using proxy: {proxy}", flush=True)


# Set up proxy at import time (safe — only reads env vars).
_setup_proxy()

# Prevent nilearn's joblib from spawning child processes.
# With --jobs > 1 we use our own ProcessPoolExecutor; joblib workers
# would only cause CPU oversubscription and orphan processes on Windows.
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

# ── interrupt handling ───────────────────────────────────────────────────

_interrupted = False


def _on_interrupt(signum: int, frame: Any) -> None:
    global _interrupted
    _interrupted = True
    print("\n[INTERRUPT] Gracefully stopping... (press Ctrl+C again to force)", flush=True)


signal.signal(signal.SIGINT, _on_interrupt)


@dataclass(frozen=True)
class ResolvedInputs:
    bold_path: Path
    confounds_path: Path
    bold_json_path: Path | None
    bold_relative: Path
    confounds_relative: Path
    bold_json_relative: Path | None
    cache_paths: tuple[Path, ...]
    bold_url: str
    confounds_url: str
    bold_json_url: str | None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_site_key(subject_id: str) -> str:
    match = re.match(r"^(v[12]s\d+)x", subject_id)
    if not match:
        raise ValueError(f"Cannot infer fMRIPrep site index from subject ID: {subject_id}")
    return match.group(1)


def relative_from_queue(value: Any, dataset_root: Path) -> Path | None:
    """Recover a derivative-relative path even if an absolute path is mojibake."""
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


def looks_like_annex_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 2048:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return ".git/annex/objects/" in text.replace("\\", "/") or "MD5E-s" in text


def gin_raw_url(subject_id: str, relative: Path) -> str:
    site_key = infer_site_key(subject_id)
    relative_url = urllib.parse.quote(relative.as_posix(), safe="/-_.~")
    return f"https://gin.g-node.org/abide-fmriprep/{site_key}/raw/master/{relative_url}"


def cached_path(subject_id: str, suffix: str) -> Path:
    """Flat cache path to avoid Windows MAX_PATH issues with long BIDS filenames."""
    return STREAM_CACHE / f"{subject_id}_{suffix}"


def download(url: str, destination: Path, dry_run: bool) -> Path:
    print(f"GET {url}", flush=True)
    print(f" -> {destination}", flush=True)
    if dry_run:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(f"{destination.name}.{os.getpid()}.part")
    if part.exists():
        part.unlink()
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl.exe/curl is required for robust streaming downloads on Windows.")
    try:
        output_arg = str(part.resolve().relative_to(PROJECT.resolve()))
        curl_cwd = PROJECT
    except ValueError:
        output_arg = str(part)
        curl_cwd = None

    # Proxy: use env var if set, otherwise try the common clash/v2ray local proxy
    proxy = _detect_proxy()
    if not proxy:
        proxy = "http://127.0.0.1:7890"

    command = [
        curl,
        "-L",
        "--fail",
        "--show-error",
        "--connect-timeout",
        "30",
        "--speed-time",
        "300",
        "--speed-limit",
        "1024",
        "--retry",
        "10",
        "--retry-delay",
        "10",
        "--retry-max-time",
        "600",
        "--retry-all-errors",
        "--continue-at",
        "-",
        "--proxy", proxy,
        "--output",
        output_arg,
        url,
    ]
    result = subprocess.run(command, cwd=curl_cwd, check=False)
    if result.returncode != 0:
        if part.exists():
            part.unlink()
        raise RuntimeError(f"curl download failed with exit code {result.returncode}: {url}")
    if not part.exists() or part.stat().st_size <= 2048:
        if part.exists():
            part.unlink()
        raise IOError(f"Downloaded file is unexpectedly small: {url}")
    part.replace(destination)
    return destination


def ensure_available(
    subject_id: str,
    value: Any,
    dataset_root: Path,
    dry_run: bool,
    kind: str = "bold.nii.gz",
    force_download: bool = False,
) -> tuple[Path, Path | None, str, Path]:
    relative = relative_from_queue(value, dataset_root)
    if relative is None:
        raise ValueError("Missing derivative path in queue row")
    local_path = dataset_root / relative
    url = gin_raw_url(subject_id, relative)
    needs_download = force_download or not local_path.exists() or looks_like_annex_pointer(local_path)
    if not needs_download:
        return local_path, None, url, relative
    target = cached_path(subject_id, kind)
    if target.exists() and target.stat().st_size > 2048 and not looks_like_annex_pointer(target):
        return target, target, url, relative
    return download(url, target, dry_run), target, url, relative


def load_repetition_time(bold_json_path: Path | None, bold_path: Path) -> float:
    json_path = bold_json_path
    if json_path is not None and json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if "RepetitionTime" in payload:
            return float(payload["RepetitionTime"])
    zooms = nib.load(str(bold_path)).header.get_zooms()
    if len(zooms) >= 4 and float(zooms[3]) > 0:
        return float(zooms[3])
    raise ValueError("No valid RepetitionTime in BOLD JSON or NIfTI header.")


def choose_confounds(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    confounds = pd.read_csv(path, sep="\t")
    selected_motion = [base + suffix for base in MOTION_BASES for suffix in MOTION_SUFFIXES]
    missing_motion = [column for column in selected_motion if column not in confounds.columns]
    if missing_motion:
        raise ValueError("Missing required fMRIPrep motion columns: " + ", ".join(missing_motion))
    acompcor = sorted(column for column in confounds.columns if column.startswith("a_comp_cor_"))[:5]
    if len(acompcor) != 5:
        raise ValueError(f"Expected at least five aCompCor columns; found {len(acompcor)}")
    selected = [*selected_motion, *acompcor]
    matrix = confounds[selected].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    fd = pd.to_numeric(confounds.get("framewise_displacement"), errors="coerce")
    if fd is None:
        raise ValueError("fMRIPrep confounds lack framewise_displacement")
    retain = np.isfinite(fd.to_numpy()) & (fd.to_numpy(dtype=float) <= 0.50)
    nonsteady = [column for column in confounds.columns if column.startswith("non_steady_state_outlier_")]
    if nonsteady:
        retain &= (confounds[nonsteady].fillna(0).to_numpy(dtype=float).sum(axis=1) == 0)
    return matrix, np.flatnonzero(retain), selected


def fetch_paths(row: pd.Series, dataset_root: Path, dry_run: bool) -> ResolvedInputs:
    subject_id = str(row["subject_id"])
    bold_path, bold_cache, bold_url, bold_relative = ensure_available(subject_id, row["bold_path"], dataset_root, dry_run, kind="bold.nii.gz")
    confounds_path, confounds_cache, confounds_url, confounds_relative = ensure_available(
        subject_id, row["confounds_path"], dataset_root, dry_run, kind="confounds.tsv"
    )
    json_value = row.get("bold_json_path", "")
    bold_json_path: Path | None = None
    bold_json_cache: Path | None = None
    bold_json_url: str | None = None
    bold_json_relative: Path | None = None
    if pd.notna(json_value) and str(json_value).strip():
        bold_json_path, bold_json_cache, bold_json_url, bold_json_relative = ensure_available(
            subject_id, json_value, dataset_root, dry_run, kind="bold.json"
        )
    cache_paths = tuple(path for path in (bold_cache, confounds_cache, bold_json_cache) if path is not None)
    return ResolvedInputs(
        bold_path=bold_path,
        confounds_path=confounds_path,
        bold_json_path=bold_json_path,
        bold_relative=bold_relative,
        confounds_relative=confounds_relative,
        bold_json_relative=bold_json_relative,
        cache_paths=cache_paths,
        bold_url=bold_url,
        confounds_url=confounds_url,
        bold_json_url=bold_json_url,
    )


def release_paths(paths: tuple[Path, ...], dry_run: bool) -> None:
    cache_root = STREAM_CACHE.resolve()
    for path in paths:
        if dry_run or not path.exists():
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(cache_root)
        except ValueError as exc:
            raise ValueError(f"Refusing to delete a path outside stream cache: {path}") from exc
        resolved.unlink()
        for stale_part in resolved.parent.glob(f"{resolved.name}.*.part"):
            try:
                stale_part.unlink()
            except OSError:
                pass
    if STREAM_CACHE.exists() and not dry_run:
        for directory in sorted((p for p in STREAM_CACHE.rglob("*") if p.is_dir()), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass


def output_paths(row: pd.Series) -> tuple[Path, Path]:
    cohort_dir = OUTPUT_ROOT / str(row["cohort"])
    return cohort_dir / f"{row['subject_id']}_cc200.npy", cohort_dir / f"{row['subject_id']}_cc200_provenance.json"


def extract_one(row: pd.Series, inputs: ResolvedInputs, atlas: Path, analysis_spec_hash: str) -> dict[str, Any]:
    bold_path = inputs.bold_path
    confounds_path = inputs.confounds_path
    if not bold_path.exists() or not confounds_path.exists():
        raise FileNotFoundError("BOLD or confounds file is not locally available after fetch.")
    tr = load_repetition_time(inputs.bold_json_path, bold_path)
    confounds, sample_mask, selected_columns = choose_confounds(confounds_path)
    if len(sample_mask) < 100:
        raise ValueError(f"Only {len(sample_mask)} volumes pass FD/non-steady-state filtering; minimum is 100")
    masker = NiftiLabelsMasker(
        labels_img=str(atlas),
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.10,
        high_pass=0.01,
        t_r=tr,
        resampling_target="data",
        memory=str(PROJECT / ".nilearn_cache"),
        memory_level=1,
    )
    time_series = masker.fit_transform(str(bold_path), confounds=confounds, sample_mask=sample_mask)
    if time_series.ndim != 2 or time_series.shape[1] != 200:
        raise ValueError(f"Expected a [time, 200] matrix after CC200 extraction; got {time_series.shape}")
    matrix_path, provenance_path = output_paths(row)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_path, time_series.astype(np.float32))
    record = {
        "subject_id": str(row["subject_id"]),
        "cohort": str(row["cohort"]),
        "site_id": str(row["site_id"]),
        "label": int(row["label"]),
        "age": float(row["age"]),
        "sex": int(row["sex"]),
        "time_series_path": str(matrix_path.resolve()),
        "source_bold": inputs.bold_url,
        "source_bold_derivative_relative": inputs.bold_relative.as_posix(),
        "source_bold_url": inputs.bold_url,
        "source_confounds": str(confounds_path.resolve()),
        "source_confounds_derivative_relative": inputs.confounds_relative.as_posix(),
        "source_confounds_url": inputs.confounds_url,
        "source_bold_json": None if inputs.bold_json_path is None else str(inputs.bold_json_path.resolve()),
        "source_bold_json_derivative_relative": None
        if inputs.bold_json_relative is None
        else inputs.bold_json_relative.as_posix(),
        "source_bold_json_url": inputs.bold_json_url,
        "source_run": str(row["run_key"]),
        "atlas_path": str(atlas.resolve()),
        "atlas_sha256": sha256(atlas),
        "analysis_spec_sha256": analysis_spec_hash,
        "tr_seconds": tr,
        "retained_volumes": int(time_series.shape[0]),
        "cc200_rois": int(time_series.shape[1]),
        "confound_columns": selected_columns,
    }
    provenance_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def load_completed_subjects() -> set[str]:
    """Return the set of subject_ids already present in extracted_timeseries/index.csv."""
    index_path = OUTPUT_ROOT / "index.csv"
    if not index_path.exists():
        return set()
    try:
        previous = pd.read_csv(index_path, dtype={"subject_id": str})
        return set(previous["subject_id"].astype(str).str.strip())
    except Exception:
        return set()


def write_checkpoint(completed: int, failed: int, last_subject: str, elapsed: float) -> None:
    """Write per-invocation progress plus the authoritative indexed total.

    ``completed`` and ``failed`` are counts for the current invocation, not
    full-project totals.  Earlier checkpoint files did not make that scope
    explicit and could therefore look like an incomplete extraction after a
    short recovery batch.
    """
    checkpoint_path = OUTPUT_ROOT / "checkpoint.json"
    total_indexed = len(load_completed_subjects())
    payload = {
        "status": "RUN_IN_PROGRESS" if last_subject else "RUN_FINISHED",
        "completed": completed,
        "failed": failed,
        "completed_this_run": completed,
        "failed_this_run": failed,
        "total_indexed": total_indexed,
        "count_scope": "completed/failed are for this invocation; total_indexed is the full index.csv total",
        "last_subject": last_subject,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cleanup_orphan_parts() -> None:
    """Remove leftover .part files from interrupted downloads."""
    if not STREAM_CACHE.exists():
        return
    for part_file in STREAM_CACHE.rglob("*.part"):
        try:
            part_file.unlink()
            print(f"[CLEANUP] Removed orphan part: {part_file}", flush=True)
        except OSError:
            pass
    # Remove empty directories
    for directory in sorted((p for p in STREAM_CACHE.rglob("*") if p.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def _extract_one_row(
    row_dict: dict[str, Any],
    dataset_root_str: str,
    atlas_str: str,
    analysis_spec_hash: str,
    dry_run: bool,
) -> dict[str, Any]:
    """Worker function: process a single subject row (picklable for ProcessPoolExecutor).

    Each worker must set thread-control env vars to prevent CPU oversubscription
    when nilearn uses joblib internally.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    row = pd.Series(row_dict)
    dataset_root = Path(dataset_root_str)
    atlas = Path(atlas_str)
    sid = str(row.get("subject_id", ""))

    inputs: ResolvedInputs | None = None
    try:
        inputs = fetch_paths(row, dataset_root, dry_run)
        if dry_run:
            return {"subject_id": sid, "action": "dry_run", "success": True}
        record = extract_one(row, inputs, atlas, analysis_spec_hash)
        return {"subject_id": sid, "success": True, "record": record, "cache_paths": list(inputs.cache_paths)}
    except Exception as exc:
        tb = traceback.format_exc()
        return {
            "subject_id": sid,
            "success": False,
            "error": str(exc),
            "traceback": tb,
            "cache_paths": list(inputs.cache_paths) if inputs else [],
        }


def append_index(record: dict[str, Any]) -> None:
    index_path = OUTPUT_ROOT / "index.csv"
    frame = pd.DataFrame([record])
    if index_path.exists():
        previous = pd.read_csv(index_path, dtype={"subject_id": str})
        previous = previous[previous["subject_id"] != record["subject_id"]]
        frame = pd.concat([previous, frame], ignore_index=True)
    frame.sort_values(["cohort", "subject_id"]).to_csv(index_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Actually run HTTP stream/extract/cache-cleanup actions")
    parser.add_argument("--dry-run", action="store_true", help="Print planned GIN raw URLs and cache paths only")
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--cohort", choices=["ABIDE_I", "ABIDE_II"])
    parser.add_argument("--subject-id", help="Optional single BIDS subject id, without the sub- prefix")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel workers (default 1; max recommended 2)")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip already-extracted subjects")
    parser.add_argument("--report-every", type=int, default=10, help="Write checkpoint every N subjects")
    args = parser.parse_args()
    if args.execute == args.dry_run:
        raise SystemExit("Choose exactly one of --dry-run or --execute.")
    if not args.queue.exists() or not args.dataset_root.exists() or not args.atlas.exists():
        raise SystemExit("Queue, dataset root, and atlas must all exist.")

    # ── cleanup orphan .part files from previous interrupted run ─────────
    if args.execute and not args.dry_run:
        cleanup_orphan_parts()

    # ── load queue ───────────────────────────────────────────────────────
    queue = pd.read_csv(args.queue, dtype={"subject_id": str})
    required = {"subject_id", "cohort", "site_id", "label", "age", "sex", "bold_path", "confounds_path", "run_key"}
    missing = required - set(queue.columns)
    if missing:
        raise SystemExit("Queue missing columns: " + ", ".join(sorted(missing)))
    if "eligible_metadata" in queue:
        eligible = queue["eligible_metadata"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
        queue = queue[eligible]
    if "qc_pass" in queue.columns:
        queue = queue[queue["qc_pass"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})]
    if args.cohort:
        queue = queue[queue["cohort"] == args.cohort]
    if args.subject_id:
        queue = queue[queue["subject_id"] == args.subject_id]
    if args.max_subjects:
        queue = queue.head(args.max_subjects)

    # ── resume ───────────────────────────────────────────────────────────
    if not args.no_resume and args.execute:
        completed = load_completed_subjects()
        n_before = len(queue)
        queue = queue[~queue["subject_id"].isin(completed)]
        n_skipped = n_before - len(queue)
        if n_skipped:
            print(f"[RESUME] Skipping {n_skipped} already-extracted subjects ({len(queue)} remaining)")
    else:
        completed = set()

    if len(queue) == 0:
        print("All subjects already extracted. Nothing to do.")
        return 0

    # ── execute ──────────────────────────────────────────────────────────
    analysis_spec_hash = sha256(PROJECT / "config" / "analysis_spec.yaml")
    rejected: list[dict[str, str]] = []
    n_completed = 0
    n_failed = 0
    t_start = time.monotonic()

    if args.jobs > 1 and args.execute and not args.dry_run:
        # ── parallel path ────────────────────────────────────────────────
        from concurrent.futures import ProcessPoolExecutor, as_completed

        rows_as_dicts = [row.to_dict() for _, row in queue.iterrows()]
        dataset_root_str = str(args.dataset_root.resolve())
        atlas_str = str(args.atlas.resolve())

        print(f"[PARALLEL] Starting {args.jobs} workers for {len(rows_as_dicts)} subjects", flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(
                    _extract_one_row,
                    d,
                    dataset_root_str,
                    atlas_str,
                    analysis_spec_hash,
                    False,
                ): d["subject_id"]
                for d in rows_as_dicts
            }
            for future in as_completed(futures):
                if _interrupted:
                    print("[INTERRUPT] Cancelling pending futures...", flush=True)
                    for f in futures:
                        f.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                sid = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    rejected.append({"subject_id": sid, "cohort": "", "reason": f"Worker crashed: {exc}"})
                    n_failed += 1
                    print(f"FAILED {sid}: worker crash: {exc}", flush=True)
                    continue

                # Clean up cache files from worker
                for cp in result.get("cache_paths", []):
                    try:
                        release_paths((Path(cp),), dry_run=False)
                    except Exception:
                        pass

                if result["success"]:
                    n_completed += 1
                    rec = result.get("record", {})
                    append_index(rec)
                    print(
                        f"DONE {rec.get('cohort', '?')} {sid} ({rec.get('retained_volumes', '?')} volumes) "
                        f"[{n_completed}/{len(queue)}]", flush=True
                    )
                else:
                    n_failed += 1
                    rejected.append({"subject_id": sid, "cohort": "", "reason": result.get("error", "unknown")})
                    print(f"FAILED {sid}: {result.get('error', 'unknown')}", flush=True)

                if (n_completed + n_failed) % args.report_every == 0:
                    write_checkpoint(
                        n_completed, n_failed, sid, time.monotonic() - t_start
                    )
    else:
        # ── sequential path ──────────────────────────────────────────────
        for _, row in queue.iterrows():
            if _interrupted:
                print("[INTERRUPT] Stopping after current subject.", flush=True)
                break

            inputs: ResolvedInputs | None = None
            try:
                inputs = fetch_paths(row, args.dataset_root, args.dry_run)
                if args.dry_run:
                    continue
                record = extract_one(row, inputs, args.atlas, analysis_spec_hash)
                append_index(record)
                n_completed += 1
                print(
                    f"DONE {record['cohort']} {record['subject_id']} ({record['retained_volumes']} volumes) "
                    f"[{n_completed}/{len(queue)}]", flush=True
                )
            except Exception as exc:
                n_failed += 1
                rejected.append(
                    {"subject_id": str(row.get("subject_id", "")), "cohort": str(row.get("cohort", "")), "reason": str(exc)}
                )
                print(f"FAILED {row.get('subject_id', '')}: {exc}", flush=True)
            finally:
                if inputs is not None and not args.dry_run:
                    try:
                        release_paths(inputs.cache_paths, dry_run=False)
                    except Exception as exc:
                        rejected.append(
                            {
                                "subject_id": str(row.get("subject_id", "")),
                                "cohort": str(row.get("cohort", "")),
                                "reason": f"Cache cleanup failed: {exc}",
                            }
                        )

            if (n_completed + n_failed) % args.report_every == 0:
                write_checkpoint(
                    n_completed, n_failed,
                    str(row.get("subject_id", "")),
                    time.monotonic() - t_start,
                )

    # ── final report ─────────────────────────────────────────────────────
    elapsed = time.monotonic() - t_start
    rejection_path = OUTPUT_ROOT / "streaming_rejections.csv"
    if rejected:
        pd.DataFrame(rejected).to_csv(rejection_path, index=False)
    elif rejection_path.exists():
        try:
            rejection_path.unlink()
        except OSError as exc:
            print(f"WARNING: could not remove stale rejection file {rejection_path}: {exc}")

    write_checkpoint(n_completed, n_failed, "", elapsed)

    print(
        f"\nStreaming queue complete: attempted={n_completed + n_failed}, "
        f"done={n_completed}, failed={n_failed}, elapsed={elapsed:.0f}s"
    )
    if n_completed > 0:
        rate = elapsed / n_completed
        print(f"  Rate: {rate:.1f}s/subject  ({3600 / rate:.0f} subjects/hour)")
    return 0 if not rejected else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Check the local environment without downloading any ABIDE data."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
TOOLS = PROJECT / ".tools"
REQUIRED_PYTHON = ("numpy", "pandas", "nibabel", "nilearn", "scipy", "sklearn", "yaml")
REQUIRED_DIRS = ("config", "resources", "metadata", "manifests", "extracted_timeseries", "features", "results", "scripts")


def main() -> int:
    missing = []
    print(f"Project: {PROJECT}")
    print(f"Python: {sys.version.split()[0]}")
    for package in REQUIRED_PYTHON:
        try:
            module = __import__(package)
            print(f"Python package OK: {package} {getattr(module, '__version__', '')}".rstrip())
        except Exception as exc:
            missing.append(f"Python package {package}: {exc}")
    acquisition_missing = []
    project_tools = {
        "git": TOOLS / "mingit" / "cmd" / "git.exe",
        "git-annex": TOOLS / "venv" / "Scripts" / "git-annex.exe",
        "datalad": TOOLS / "venv" / "Scripts" / "datalad.exe",
    }
    for executable in ("datalad", "git-annex", "git"):
        found = str(project_tools[executable]) if project_tools[executable].exists() else shutil.which(executable)
        print(f"Command {'OK' if found else 'MISSING'}: {executable}{' -> ' + found if found else ''}")
        if not found:
            acquisition_missing.append(executable)
    for name in REQUIRED_DIRS:
        path = PROJECT / name
        print(f"Directory {'OK' if path.is_dir() else 'MISSING'}: {path}")
        if not path.is_dir():
            missing.append(f"Directory missing: {path}")
    atlas = PROJECT.parent / "downloads" / "craddock_2012" / "cc200_roi_atlas.nii.gz"
    print(f"CC200 atlas {'FOUND' if atlas.exists() else 'NOT YET PROVIDED'}: {atlas}")
    if missing:
        print("\nPreflight incomplete:\n- " + "\n- ".join(missing))
        return 1
    print("\nPython/project preflight passed.")
    if acquisition_missing:
        print("Acquisition prerequisites still needed for full-cohort streaming: " + ", ".join(acquisition_missing))
    else:
        print("Full-cohort streaming prerequisites are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate Figure 1: lineage-aligned study and validation workflow.

Python/matplotlib only. The script is non-interactive, uses a fixed declared
seed (no random drawing), and exports SVG, PDF, 600-dpi TIFF, and PNG preview.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


RANDOM_SEED = 20260905
np.random.seed(RANDOM_SEED)  # No stochastic plotting is used.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = REPOSITORY_ROOT / "aggregate_source_data" / "figure_1_workflow.csv"
OUTPUT_DIR = REPOSITORY_ROOT / "generated_figures" / "figure_1"
CAPTION_PATH = OUTPUT_DIR / "figure_1_caption.md"
BASE_NAME = "figure_1_workflow"

COLORS = {
    "navy": "#2F4B6C",
    "blue": "#4C78A8",
    "teal": "#2A9D8F",
    "orange": "#E69F00",
    "purple": "#7A5195",
    "pale_blue": "#EAF1F8",
    "pale_teal": "#E7F4F1",
    "pale_orange": "#FFF2D6",
    "pale_purple": "#F0EAF5",
    "grey": "#5B6573",
    "light_grey": "#EEF0F2",
    "dark": "#17202A",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "svg.fonttype": "none",
        "svg.hashsalt": "phase7a-main-exhibits-20260905",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def box(ax, x, y, w, h, text, face, edge, *, weight="normal", linestyle="-", fontsize=8.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=face,
        linestyle=linestyle,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=COLORS["dark"], fontweight=weight,
            transform=ax.transAxes, linespacing=1.08)
    return patch


def arrow(ax, x1, y1, x2, y2, color):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=8, linewidth=0.9,
        color=color, shrinkA=1.5, shrinkB=1.5,
    ))


def panel_frame(ax, label, title, accent):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.015, 0.02), 0.97, 0.96,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor="white", edgecolor="#C8CDD3", linewidth=0.9,
        transform=ax.transAxes,
    ))
    ax.add_patch(FancyBboxPatch(
        (0.03, 0.895), 0.94, 0.07,
        boxstyle="round,pad=0.006,rounding_size=0.015",
        facecolor=accent, edgecolor=accent, linewidth=0,
        transform=ax.transAxes,
    ))
    ax.text(0.045, 0.93, label, ha="left", va="center", fontsize=10,
            fontweight="bold", color="white", transform=ax.transAxes)
    ax.text(0.16, 0.93, title, ha="left", va="center", fontsize=8.5,
            fontweight="bold", color="white", transform=ax.transAxes)


def main() -> None:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(INPUT_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CAPTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_CSV)
    expected_panels = {"A", "B", "C", "D"}
    if set(data["panel"]) != expected_panels or data.duplicated(["panel", "order"]).any():
        raise ValueError("Figure 1 source-data panel/order keys are invalid")

    configure_style()
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 4.45))
    fig.subplots_adjust(left=0.018, right=0.988, top=0.975, bottom=0.035, wspace=0.075)

    panel_specs = [
        ("A", "Input provenance", COLORS["blue"], COLORS["pale_blue"]),
        ("B", "Nested prediction", COLORS["teal"], COLORS["pale_teal"]),
        ("C", "Held-out XAI lineage", COLORS["purple"], COLORS["pale_purple"]),
        ("D", "Validation layers", COLORS["orange"], COLORS["pale_orange"]),
    ]
    manual_wrap = {
        "Corrected node-major frequency: 3,000": "Corrected node-major\nfrequency: 3,000",
        "OOF 879/879; assignment min=max=1": "OOF 879/879\nassignment min=max=1",
        "Equal-weight mean of three seed probabilities": "Equal-weight mean of\nthree seed probabilities",
        "Subject ID → unseen outer model": "Subject ID\n↓\nunseen outer model",
        "Checkpoint hash → prediction-linked explanation": "Checkpoint hash\n↓\nprediction-linked explanation",
        "Subject / seed / fold / ensemble aggregation": "Subject / seed / fold /\nensemble aggregation",
        "Agreement: do methods rank similarly?": "Agreement\nDo methods rank similarly?",
        "Randomization sanity: dependence on learned structure?": "Randomization sanity\nDependence on learned structure?",
        "Repeated ROAR: greater retraining damage than random?": "Repeated ROAR\nGreater retraining damage\nthan random?",
        "Engineering and lineage checks: PASS": "Engineering and lineage\nchecks: PASS",
        "Pre-specified ROAR support: not obtained": "Pre-specified ROAR\nsupport: not obtained",
    }

    for ax, (panel, title, accent, pale) in zip(axes, panel_specs):
        panel_frame(ax, panel, title, accent)
        rows = data[data["panel"] == panel].sort_values("order")
        if panel != "D":
            n = len(rows)
            y_top, y_bottom, h = 0.82, 0.105, 0.086
            ys = np.linspace(y_top, y_bottom, n)
            for idx, ((_, row), y) in enumerate(zip(rows.iterrows(), ys)):
                text = manual_wrap.get(row["display_text"], row["display_text"])
                box(ax, 0.105, y, 0.79, h, text, pale, accent,
                    weight="bold" if idx in (0, n - 1) else "normal")
                if idx < n - 1:
                    arrow(ax, 0.5, y - 0.004, 0.5, ys[idx + 1] + h + 0.004, accent)
        else:
            questions = rows[rows["item_type"] == "question"]
            outcomes = rows[rows["item_type"] != "question"]
            ys = [0.74, 0.55, 0.34]
            for idx, ((_, row), y) in enumerate(zip(questions.iterrows(), ys)):
                text = manual_wrap[row["display_text"]]
                h = 0.115 if idx < 2 else 0.14
                box(ax, 0.08, y, 0.84, h, text, pale, accent, fontsize=8.0)
                if idx < 2:
                    arrow(ax, 0.5, y - 0.005, 0.5, ys[idx + 1] + (0.115 if idx + 1 < 2 else 0.14) + 0.005, accent)
            outcome_rows = list(outcomes.itertuples(index=False))
            box(ax, 0.06, 0.17, 0.88, 0.095, manual_wrap[outcome_rows[0].display_text],
                COLORS["pale_teal"], COLORS["teal"], weight="bold", fontsize=8.0)
            box(ax, 0.06, 0.055, 0.88, 0.095, manual_wrap[outcome_rows[1].display_text],
                COLORS["light_grey"], COLORS["grey"], weight="bold", linestyle="--", fontsize=8.0)

    # Cross-panel arrows reinforce the single lineage without implying one global PASS.
    for left, right in zip(axes[:-1], axes[1:]):
        p1 = left.get_position(); p2 = right.get_position()
        x1 = p1.x1 + 0.002; x2 = p2.x0 - 0.002; y = (p1.y0 + p1.y1) / 2
        fig.add_artist(FancyArrowPatch((x1, y), (x2, y), transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=9,
                                       linewidth=1.0, color=COLORS["grey"], zorder=10))

    outputs = {}
    for ext, dpi in [("svg", 600), ("pdf", 600), ("tiff", 600), ("png", 300)]:
        out = OUTPUT_DIR / f"{BASE_NAME}.{ext}"
        # Uncompressed TIFF avoids a Pillow 12.0.0 LZW encoder crash in this
        # Python 3.13 Windows runtime while preserving the required 600 dpi.
        metadata = None
        if ext == "svg":
            metadata = {"Date": "2026-09-05", "Creator": "Phase 7A Python/matplotlib"}
        elif ext == "pdf":
            metadata = {"Title": "Figure 1", "Creator": "Phase 7A Python/matplotlib",
                        "Producer": "Matplotlib", "CreationDate": None, "ModDate": None}
        elif ext == "png":
            metadata = {"Software": "Phase 7A Python/matplotlib"}
        fig.savefig(out, dpi=dpi, metadata=metadata)
        outputs[out.name] = sha256(out)
    plt.close(fig)

    env = {
        "script": str(Path(__file__).resolve()),
        "backend": "Python/matplotlib only",
        "random_seed": RANDOM_SEED,
        "random_operations": "none",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "pillow": Image.__version__,
        "input": str(INPUT_CSV),
        "input_sha256": sha256(INPUT_CSV),
        "final_size_inches": [7.2, 4.45],
        "minimum_declared_font_pt": 8.0,
        "exports": ["SVG with editable text", "PDF with TrueType text", "600 dpi TIFF", "300 dpi PNG preview"],
    }
    (OUTPUT_DIR / "execution_environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "output_sha256.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    caption = """# Figure 1 caption

**Figure 1. Lineage-aligned study and validation workflow.** **A,** The retained ABIDE-I analytical snapshot comprised 879 participants represented by C-PAC-derived CC200 inputs: 19,900 functional-connectivity (FC) edges and 3,000 corrected node-major frequency features. **B,** Diagnosis-stratified subject-wise internal evaluation used 10 outer folds and seeds 42/43/44, producing 30 final checkpoints. All 879 participants received exactly one out-of-fold (OOF) assignment, and the primary probability was the equal-weight mean of the three seed-specific probabilities. **C,** Every held-out explanation was bound by subject ID, unseen outer model, and checkpoint hash; Gradient × Input (G×I) and Integrated Gradients (IG) were retained through subject-, seed-, fold-, and ensemble-level aggregation. **D,** Agreement, randomization sanity, and repeated remove-and-retrain (ROAR) answer distinct questions. Engineering and lineage checks passed, whereas pre-specified ROAR support was not obtained. The workflow does not represent an XAI-fidelity PASS, external validation, biological validity, mechanism, or clinical utility.
"""
    CAPTION_PATH.write_text(caption, encoding="utf-8")
    print(json.dumps({"figure": 1, "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()

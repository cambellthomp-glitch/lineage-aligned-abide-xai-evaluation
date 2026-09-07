#!/usr/bin/env python3
"""Generate Figure 3: repeated-ROAR attribution-minus-random effects.

Python/matplotlib only. The forest plot uses canonical comparison fields; the
inset uses unambiguously paired fold effects after three-seed averaging.
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
from matplotlib.patches import FancyBboxPatch
from PIL import Image


RANDOM_SEED = 20260905
np.random.seed(RANDOM_SEED)  # No stochastic plotting is used.
ROOT = Path(__file__).resolve().parents[2]
SUMMARY_CSV = ROOT / "aggregate_source_data" / "figure_3_roar_summary.csv"
FOLDS_CSV = ROOT / "aggregate_source_data" / "figure_3_roar_fold_effects.csv"
OUTPUT_DIR = ROOT / "generated_figures" / "figure_3"
CAPTION_PATH = OUTPUT_DIR / "figure_3_caption.md"
BASE_NAME = "figure_3_roar_forest"

BLUE = "#4C78A8"
ORANGE = "#E69F00"
DARK = "#1D2733"
GREY = "#6C737C"
PALE = "#EEF0F3"


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
        "axes.labelsize": 8.5,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "svg.fonttype": "none",
        "svg.hashsalt": "phase7a-main-exhibits-20260905",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def add_panel_label(ax, label, x=-0.09, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="top", ha="left", color=DARK)


def main() -> None:
    for path in [SUMMARY_CSV, FOLDS_CSV]:
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CAPTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SUMMARY_CSV)
    folds = pd.read_csv(FOLDS_CSV)
    if summary.shape[0] != 4 or folds.shape[0] != 40:
        raise ValueError("Unexpected Figure 3 source-data row count")
    if summary.isna().any().any() or folds.isna().any().any():
        raise ValueError("Unexpected missing value in Figure 3 source data")
    if folds.duplicated(["method", "removal_percentage", "fold"]).any():
        raise ValueError("Ambiguous fold-level ROAR inset pairing")

    order = [("G×I", 5), ("IG", 5), ("G×I", 10), ("IG", 10)]
    summary["order"] = summary.apply(lambda r: order.index((r["method"], int(r["removal_percentage"]))), axis=1)
    summary = summary.sort_values("order").reset_index(drop=True)
    folds["order"] = folds.apply(lambda r: order.index((r["method"], int(r["removal_percentage"]))), axis=1)

    configure_style()
    fig = plt.figure(figsize=(7.2, 6.15))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.34, 1.30, 0.92, 0.48],
                          left=0.17, right=0.965, top=0.965, bottom=0.065, hspace=0.62)
    ax_key = fig.add_subplot(gs[0, 0])
    forest_gs = gs[1, 0].subgridspec(1, 2, width_ratios=[0.60, 0.40], wspace=0.06)
    ax_forest = fig.add_subplot(forest_gs[0, 0])
    ax_numbers = fig.add_subplot(forest_gs[0, 1])
    ax_inset = fig.add_subplot(gs[2, 0])
    ax_note = fig.add_subplot(gs[3, 0])

    xlim = (-0.04, 0.055)
    ax_key.set_xlim(*xlim)
    ax_key.set_ylim(0, 1)
    ax_key.axis("off")
    ax_key.axvspan(xlim[0], 0, facecolor=PALE, edgecolor="#A7ADB5", hatch="////", alpha=0.75)
    ax_key.axvline(0, color=DARK, lw=1.2)
    ax_key.annotate("Negative: pre-specified\nsupport direction", xy=(-0.002, 0.48), xytext=(-0.027, 0.48),
                    ha="center", va="center", fontsize=8, color=DARK,
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.0))
    ax_key.annotate("Positive: opposite side of\nthe pre-specified direction", xy=(0.002, 0.48), xytext=(0.030, 0.48),
                    ha="center", va="center", fontsize=8, color=DARK,
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.0))
    add_panel_label(ax_key, "A", x=-0.12, y=1.10)

    y = np.arange(4)[::-1]
    labels = [f"{r.method}, {int(r.removal_percentage)}%" for r in summary.itertuples(index=False)]
    ax_forest.axvspan(xlim[0], 0, facecolor=PALE, edgecolor="none", alpha=0.8, zorder=0)
    ax_forest.axvline(0, color=DARK, lw=1.3, zorder=1)
    for yi, row in zip(y, summary.itertuples(index=False)):
        color = BLUE if row.method == "G×I" else ORANGE
        marker = "o" if row.method == "G×I" else "s"
        ax_forest.plot([row.ci_low, row.ci_high], [yi, yi], color=color, lw=2.0, solid_capstyle="round", zorder=3)
        ax_forest.plot([row.ci_low, row.ci_low], [yi - 0.09, yi + 0.09], color=color, lw=1.0, zorder=3)
        ax_forest.plot([row.ci_high, row.ci_high], [yi - 0.09, yi + 0.09], color=color, lw=1.0, zorder=3)
        ax_forest.scatter([row.paired_difference], [yi], s=42, marker=marker,
                          facecolor=color, edgecolor="white", linewidth=0.7, zorder=4)
    ax_forest.set_xlim(*xlim)
    ax_forest.set_ylim(-0.65, 3.65)
    ax_forest.set_yticks(y, labels)
    ax_forest.set_xlabel("Attribution − random AUC")
    ax_forest.set_title("Canonical effect and 95% fold-bootstrap interval", loc="left", fontweight="bold", pad=7)
    ax_forest.grid(axis="x", color="#E1E4E8", lw=0.7)
    ax_forest.set_axisbelow(True)
    add_panel_label(ax_forest, "B")

    ax_numbers.set_xlim(0, 1)
    ax_numbers.set_ylim(-0.65, 3.65)
    ax_numbers.axis("off")
    ax_numbers.text(0.0, 1.055, "Difference [95% CI]", transform=ax_numbers.transAxes,
                    fontsize=8, fontweight="bold", ha="left", va="bottom", color=DARK)
    ax_numbers.text(1.0, 1.055, "p", transform=ax_numbers.transAxes,
                    fontsize=8, fontweight="bold", ha="right", va="bottom", color=DARK)
    for yi, row in zip(y, summary.itertuples(index=False)):
        ax_numbers.text(0.0, yi,
                        f"{row.paired_difference:+.6f} [{row.ci_low:+.6f}, {row.ci_high:+.6f}]",
                        ha="left", va="center", fontsize=8, color=DARK)
        ax_numbers.text(1.0, yi, f"{row.signflip_p:.6f}",
                        ha="right", va="center", fontsize=8, color=DARK)

    ax_inset.axvspan(xlim[0], 0, facecolor=PALE, edgecolor="none", alpha=0.8, zorder=0)
    ax_inset.axvline(0, color=DARK, lw=1.1, zorder=1)
    for idx, ((method, removal), yi) in enumerate(zip(order, y)):
        d = folds[(folds["method"] == method) & (folds["removal_percentage"] == removal)].sort_values("fold")
        color = BLUE if method == "G×I" else ORANGE
        marker = "o" if method == "G×I" else "s"
        # Small deterministic vertical offsets expose coincident fold effects.
        offset = (d["fold"].to_numpy() - 4.5) * 0.018
        ax_inset.scatter(d["difference_three_seed_mean"], np.full(10, yi) + offset,
                         s=21, marker=marker, facecolor=color, edgecolor="white",
                         linewidth=0.45, alpha=0.88, zorder=3)
        mean = float(d["difference_three_seed_mean"].mean())
        ax_inset.scatter([mean], [yi], marker="D", s=36, facecolor="white",
                         edgecolor=DARK, linewidth=0.9, zorder=4)
    ax_inset.set_xlim(*xlim)
    ax_inset.set_ylim(-0.55, 3.55)
    ax_inset.set_yticks(y, labels)
    ax_inset.set_xlabel("Fold effect after three-seed averaging")
    ax_inset.set_title("Paired outer-fold effects (n=10 per comparison)", loc="left", fontweight="bold", pad=7)
    ax_inset.grid(axis="x", color="#E1E4E8", lw=0.7)
    ax_inset.set_axisbelow(True)
    add_panel_label(ax_inset, "C")

    ax_note.axis("off")
    note = (
        "Interpretation boundary — All four comparisons failed to support the pre-specified negative direction.\n"
        "Positive estimates are not a confirmed opposite effect; intervals crossing zero are not equivalence results.\n"
        "ROAR removed selected FC only; other available branches remained unchanged.\n"
        "CI and p values summarize 10 fold effects from one fixed internal partition with overlapping training sets."
    )
    ax_note.text(0.01, 0.96, note, va="top", ha="left", transform=ax_note.transAxes,
                 fontsize=8.05, color=DARK, linespacing=1.35,
                 bbox=dict(boxstyle="round,pad=0.55", facecolor="#F7F8F9",
                           edgecolor="#B9C0C8", linewidth=0.9))

    outputs = {}
    for ext, dpi in [("svg", 600), ("pdf", 600), ("tiff", 600), ("png", 300)]:
        out = OUTPUT_DIR / f"{BASE_NAME}.{ext}"
        # Uncompressed TIFF avoids a Pillow 12.0.0 LZW encoder crash in this
        # Python 3.13 Windows runtime while preserving the required 600 dpi.
        metadata = None
        if ext == "svg":
            metadata = {"Date": "2026-09-05", "Creator": "Phase 7A Python/matplotlib"}
        elif ext == "pdf":
            metadata = {"Title": "Figure 3", "Creator": "Phase 7A Python/matplotlib",
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
        "random_operations": "none; inset offsets are deterministic functions of fold",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "pillow": Image.__version__,
        "inputs": {str(p): sha256(p) for p in [SUMMARY_CSV, FOLDS_CSV]},
        "final_size_inches": [7.2, 6.15],
        "minimum_declared_font_pt": 8.0,
        "exports": ["SVG with editable text", "PDF with TrueType text", "600 dpi TIFF", "300 dpi PNG preview"],
    }
    (OUTPUT_DIR / "execution_environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "output_sha256.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    caption = """# Figure 3 caption

**Figure 3. Repeated-ROAR attribution-minus-random effects.** **A,** Direction key. Negative attribution-minus-random area-under-the-receiver-operating-characteristic-curve (AUC) differences were the pre-specified support direction; positive values lie on the opposite side of that direction but are not interpreted as a confirmed opposite effect. **B,** Canonical mean effects with 95% intervals from 20,000 bootstrap resamples of 10 outer-fold effects and exact two-sided sign-flip p values: G×I 5%, +0.011158 [+0.000265, +0.022379], p=0.095703; IG 5%, +0.008159 [−0.003827, +0.018696], p=0.210938; G×I 10%, +0.002936 [−0.010120, +0.015646], p=0.683594; IG 10%, +0.008705 [−0.007525, +0.024834], p=0.330078. No significance stars are used. **C,** Unambiguously paired fold effects after averaging three seed-specific attribution-minus-random AUC differences within each outer fold (n=10 fold units per comparison); diamonds denote means. All four comparisons failed to support the pre-specified negative direction. Intervals crossing zero are not equivalence results. ROAR removed selected FC coordinates only, while other available input branches remained unchanged; inference summarizes one fixed internal 10-fold partition with overlapping training sets.
"""
    CAPTION_PATH.write_text(caption, encoding="utf-8")
    print(json.dumps({"figure": 3, "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate Figure 2: attribution agreement and randomization sanity.

Python/matplotlib only. Agreement panels use feature-coordinate ranks; the
sanity panel uses 30 fold-by-seed condition values per randomization type.
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
from PIL import Image


RANDOM_SEED = 20260905
np.random.seed(RANDOM_SEED)  # No stochastic plotting is used.
ROOT = Path(__file__).resolve().parents[2]
FC_CSV = ROOT / "aggregate_source_data" / "figure_2_fc_agreement.csv"
FREQ_CSV = ROOT / "aggregate_source_data" / "figure_2_frequency_agreement.csv"
SANITY_CSV = ROOT / "aggregate_source_data" / "figure_2_sanity_conditions.csv"
METADATA_CSV = ROOT / "aggregate_source_data" / "figure_2_metadata.csv"
OUTPUT_DIR = ROOT / "generated_figures" / "figure_2"
CAPTION_PATH = OUTPUT_DIR / "figure_2_caption.md"
BASE_NAME = "figure_2_agreement_sanity"

BLUE = "#4C78A8"
ORANGE = "#E69F00"
DARK = "#1D2733"
GREY = "#68717C"
LIGHT = "#E6E9ED"


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


def add_panel_label(ax, label):
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="top", ha="left", color=DARK)


def agreement_panel(ax, df, n_features, title, note_lines, annotation):
    x = df["gxi_rank_ascending"].to_numpy()
    y = df["ig_rank_ascending"].to_numpy()
    hb = ax.hexbin(x, y, gridsize=52 if n_features > 5000 else 42,
                   mincnt=1, bins="log", cmap="cividis", linewidths=0)
    ax.plot([1, n_features], [1, n_features], color=DARK, lw=1.0,
            ls=(0, (4, 3)), label="Identity")
    ax.set_xlim(1, n_features)
    ax.set_ylim(1, n_features)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("G×I aggregate rank (1 = lowest)")
    ax.set_ylabel("IG aggregate rank (1 = lowest)")
    ax.set_title(title, loc="left", fontweight="bold", pad=6)
    cb = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.025)
    # Keep the color scale vector-native in SVG/PDF and avoid undersized
    # mathtext exponents: plain-number tick labels retain the 8 pt minimum.
    cb.solids.set_rasterized(False)
    cb.formatter = mpl.ticker.FuncFormatter(lambda value, _position: f"{value:g}")
    cb.update_ticks()
    cb.set_label("Features per hexagon (log scale)", fontsize=8)
    cb.ax.tick_params(labelsize=8, length=2)
    ax.text(0.04, 0.96, annotation, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, linespacing=1.2, bbox=dict(boxstyle="round,pad=0.3",
            facecolor="white", edgecolor="#BCC4CD", alpha=0.94))
    ax.text(0.98, 0.04, "\n".join(note_lines), transform=ax.transAxes,
            va="bottom", ha="right", fontsize=8, color=DARK, linespacing=1.2,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#BCC4CD", alpha=0.94))
    return hb


def main() -> None:
    for path in [FC_CSV, FREQ_CSV, SANITY_CSV, METADATA_CSV]:
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CAPTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    fc = pd.read_csv(FC_CSV)
    freq = pd.read_csv(FREQ_CSV)
    sanity = pd.read_csv(SANITY_CSV)
    metadata = pd.read_csv(METADATA_CSV, dtype={"key": str, "value": str}).set_index("key")["value"].to_dict()
    if fc.shape[0] != 19900 or freq.shape[0] != 3000 or sanity.shape[0] != 60:
        raise ValueError("Unexpected Figure 2 source-data row count")
    if fc.isna().any().any() or freq.isna().any().any() or sanity.isna().any().any():
        raise ValueError("Unexpected missing value in Figure 2 source data")
    if sanity.duplicated(["condition", "fold", "seed"]).any():
        raise ValueError("Duplicate sanity condition key")
    required_metadata = {
        "fc_full_space", "selected_fc_per_fold", "fc_spearman", "fc_top50_jaccard", "fc_top100_jaccard",
        "frequency_features", "frequency_spearman", "frequency_top50_jaccard", "frequency_top100_jaccard",
        "frequency_absent_folds", "frequency_all_zero_subject_rows", "frequency_nonzero_subject_rows",
        "frequency_aggregation_denominator", "weight_randomization_mean_spearman", "label_randomization_mean_spearman",
    }
    if not required_metadata.issubset(metadata):
        raise ValueError("Figure 2 annotation metadata are incomplete")

    def f(key):
        return float(metadata[key])

    def i(key):
        return int(round(f(key)))

    configure_style()
    fig = plt.figure(figsize=(7.2, 6.25))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.42, 0.86],
                          left=0.075, right=0.965, top=0.955, bottom=0.09,
                          hspace=0.40, wspace=0.43)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    bottom = gs[1, :].subgridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.20)
    ax_c = fig.add_subplot(bottom[0, 0])
    ax_note = fig.add_subplot(bottom[0, 1])

    agreement_panel(
        ax_a, fc, 19900, "FC aggregate rank density",
        ["Common selector + zero back-fill", "+ deterministic tie ordering", "can elevate surface agreement"],
        f"Full FC space = {i('fc_full_space'):,}\nSelected FC = {i('selected_fc_per_fold'):,} per fold\n"
        f"Spearman ρ = {f('fc_spearman'):.6f}\nTop-50 Jaccard = {f('fc_top50_jaccard'):.6f}\n"
        f"Top-100 Jaccard = {f('fc_top100_jaccard'):.6f}",
    )
    add_panel_label(ax_a, "A")
    agreement_panel(
        ax_b, freq, 3000, "Frequency aggregate rank density",
        [f"Folds {metadata['frequency_absent_folds'].replace(';', ' and ')}: no frequency branch",
         f"{i('frequency_all_zero_subject_rows')} all-zero subject rows",
         f"{i('frequency_nonzero_subject_rows')} nonzero-contributing rows",
         f"Aggregation denominator = {i('frequency_aggregation_denominator')}"] ,
        f"Corrected node-major = {i('frequency_features'):,}\nSpearman ρ = {f('frequency_spearman'):.6f}\n"
        f"Top-50 Jaccard = {f('frequency_top50_jaccard'):.6f}\nTop-100 Jaccard = {f('frequency_top100_jaccard'):.6f}",
    )
    add_panel_label(ax_b, "B")

    conditions = ["Weight randomization", "Label randomization"]
    styles = {
        "Weight randomization": (BLUE, "o"),
        "Label randomization": (ORANGE, "s"),
    }
    for xpos, condition in enumerate(conditions):
        d = sanity[sanity["condition"] == condition].sort_values(["fold", "seed"])
        # Deterministic offsets encode fold/seed and avoid stochastic jitter.
        offsets = (d["fold"].to_numpy() - 4.5) * 0.013 + (d["seed"].to_numpy() - 43) * 0.004
        color, marker = styles[condition]
        ax_c.scatter(np.full(len(d), xpos) + offsets, d["spearman"],
                     s=24, marker=marker, facecolor=color, edgecolor="white",
                     linewidth=0.5, alpha=0.85, zorder=3, label=condition)
        mean = float(d["spearman"].mean())
        expected_mean = f("weight_randomization_mean_spearman" if condition == "Weight randomization" else "label_randomization_mean_spearman")
        if abs(mean - expected_mean) > 1e-12:
            raise ValueError(f"Sanity mean does not match frozen metadata for {condition}")
        ax_c.plot([xpos - 0.19, xpos + 0.19], [mean, mean], color=DARK, lw=1.5, zorder=4)
        ax_c.scatter([xpos], [mean], marker="D", s=35, color="white",
                     edgecolor=DARK, linewidth=0.9, zorder=5)
        ax_c.text(xpos, 0.305, f"mean = {mean:.6f}", ha="center", va="top",
                  fontsize=8, fontweight="bold", color=DARK)
    ax_c.axhline(0, color=GREY, lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax_c.set_xlim(-0.45, 1.45)
    ax_c.set_ylim(-0.025, 0.325)
    ax_c.set_xticks([0, 1], ["Weight\nrandomization", "Label\nrandomization"])
    ax_c.set_ylabel("Trained vs randomized Spearman ρ")
    ax_c.set_title("FC G×I sanity conditions", loc="left", fontweight="bold", pad=6)
    ax_c.grid(axis="y", color=LIGHT, lw=0.7)
    ax_c.set_axisbelow(True)
    add_panel_label(ax_c, "C")

    ax_note.axis("off")
    note = (
        "Sanity scope and statistical unit\n\n"
        "• Each point is one fold×seed condition\n"
        "  (n=30 per check).\n"
        "• Features are not treated as independent samples.\n"
        "• Sanity covers FC G×I only; it does not cover\n"
        "  IG or frequency attribution.\n"
        "• Label sanity is fixed-selector/preprocessing,\n"
        "  coordinate-matched.\n"
        "• It is not a fully nested permutation-performance\n"
        "  experiment.\n\n"
        "Agreement (A/B) and sanity (C) use separate axes\n"
        "because they answer different questions."
    )
    ax_note.text(0.02, 0.98, note, transform=ax_note.transAxes, va="top", ha="left",
                 fontsize=8.0, color=DARK, linespacing=1.22,
                 bbox=dict(boxstyle="round,pad=0.55", facecolor="#F6F7F9",
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
            metadata = {"Title": "Figure 2", "Creator": "Phase 7A Python/matplotlib",
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
        "random_operations": "none; point offsets are deterministic functions of fold and seed",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
        "pillow": Image.__version__,
        "inputs": {str(p): sha256(p) for p in [FC_CSV, FREQ_CSV, SANITY_CSV, METADATA_CSV]},
        "final_size_inches": [7.2, 6.25],
        "minimum_declared_font_pt": 8.0,
        "exports": ["SVG with editable text", "PDF with TrueType text", "600 dpi TIFF", "300 dpi PNG preview"],
    }
    (OUTPUT_DIR / "execution_environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "output_sha256.json").write_text(json.dumps(outputs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    caption = """# Figure 2 caption

**Figure 2. Attribution agreement and randomization sanity.** **A,** Hexagonal density of stable sequential ranks for aggregate functional-connectivity (FC) Gradient × Input (G×I) and Integrated Gradients (IG) attributions in the 19,900-coordinate full space. Each outer-fold selector retained 4,975 FC edges; unselected coordinates were zero-filled. FC Spearman ρ was 0.999925, with Top-50 and Top-100 Jaccard overlaps of 0.886792 and 0.923077. The common selector, structural-zero back-fill, and deterministic coordinate ordering of tied zeros can increase apparent agreement. **B,** Corresponding density for 3,000 corrected node-major frequency coordinates (ρ=0.917031; Top-50 and Top-100 Jaccard=0.754386 for both). Folds 8 and 9 had no frequency branch, yielding 175 all-zero subject rows; 704 participants contributed nonzero frequency rows, while the aggregation denominator remained 879. **C,** Condition-level trained-versus-randomized Spearman correlations for FC G×I only (30 fold×seed conditions per check). Mean correlations were 0.052757 after weight randomization and 0.050145 after label randomization. Label sanity retained the original fold-specific selector and preprocessing for coordinate matching. These sanity checks did not cover IG or frequency attribution and were not a fully nested permutation-performance experiment. Agreement and sanity are displayed on separate numerical scales because they have different estimands; neither establishes retraining-based faithfulness or biological meaning.
"""
    CAPTION_PATH.write_text(caption, encoding="utf-8")
    print(json.dumps({"figure": 2, "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()

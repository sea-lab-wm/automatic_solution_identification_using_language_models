"""Venn diagrams of the true positives, false negatives, and false positives of the three
best models (SVM, RoBERTa, Llama-3_ft) on the test sets of the 10 folds (EMSE revision, R2.C5).

Each region shows the number of comments in that combination of models. The diagrams are
unweighted (all regions have the same size), so the counts carry the information.

Usage: python results_analysis/plot_error_overlap_venn.py
Requires: pandas, openpyxl, matplotlib
"""

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from practical_impact import load_predictions

OUTPUT_DIR = "results_analysis/generated/error_overlap"
MODELS = [("SVM", "SVC", "#9ec5ec"), ("RoBERTa", "RoBERTa", "#f5b3a3"), ("Llama-3$_{ft}$", "LLAMA-FT", "#a9dbb4")]
TEXT = "#2b2b2b"
PANELS = [
    ("True positives", "solution comments identified", lambda y, p: (y == 1) & (p == 1)),
    ("False negatives", "solution comments missed", lambda y, p: (y == 1) & (p == 0)),
    ("False positives", "non-solution comments flagged", lambda y, p: (y == 0) & (p == 1)),
]
# Circle centers (unweighted three-set layout) and the position of each region's label
CENTERS = [(-0.55, 0.32), (0.55, 0.32), (0.0, -0.62)]
RADIUS = 1.0
REGION_POS = {
    (1, 0, 0): (-1.0, 0.65), (0, 1, 0): (1.0, 0.65), (0, 0, 1): (0.0, -1.15),
    (1, 1, 0): (0.0, 0.78), (1, 0, 1): (-0.62, -0.38), (0, 1, 1): (0.62, -0.38),
    (1, 1, 1): (0.0, 0.0),
}
LABEL_POS = [(-1.45, 1.42), (1.45, 1.42), (0.0, -1.85)]


def region_counts(sets):
    a, b, c = sets
    return {
        (1, 0, 0): len(a - b - c), (0, 1, 0): len(b - a - c), (0, 0, 1): len(c - a - b),
        (1, 1, 0): len((a & b) - c), (1, 0, 1): len((a & c) - b), (0, 1, 1): len((b & c) - a),
        (1, 1, 1): len(a & b & c),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pred = load_predictions()
    y = pred.label

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": TEXT})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    lines = []
    for ax, (title, subtitle, select) in zip(axes, PANELS):
        sets = [set(pred.index[select(y, pred[col])]) for _, col, _ in MODELS]
        counts = region_counts(sets)
        for (name, _, color), center, label_pos, s in zip(MODELS, CENTERS, LABEL_POS, sets):
            ax.add_patch(Circle(center, RADIUS, facecolor=color, alpha=0.55, edgecolor="white", linewidth=1.5))
            ax.text(*label_pos, f"{name}\n({len(s)})", ha="center", va="center", fontsize=8.5)
        for region, pos in REGION_POS.items():
            ax.text(*pos, f"{counts[region]}", ha="center", va="center", fontsize=9,
                    fontweight="bold" if region == (1, 1, 1) else "normal")
        ax.set_title(f"{title}\n", fontsize=9.5)
        ax.text(0.5, 1.0, f"({subtitle})", transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.5, color="#5a5a5a")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.15, 1.9)
        ax.set_aspect("equal")
        ax.axis("off")
        lines.append(f"{title}: " + ", ".join(f"{k}={v}" for k, v in counts.items())
                     + " | totals " + ", ".join(f"{m[0]}={len(s)}" for m, s in zip(MODELS, sets)))

    fig.tight_layout(w_pad=0.5)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTPUT_DIR, f"error_overlap_venn.{ext}"), dpi=300, bbox_inches="tight")
    with open(os.path.join(OUTPUT_DIR, "region_counts.txt"), "w") as f:
        f.write("Region keys are (SVM, RoBERTa, Llama-3_ft) memberships.\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

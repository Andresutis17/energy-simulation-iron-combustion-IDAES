"""
Script for the industrial solid-phase chain plots using the 3 reactors
"""
import math
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  

# All solids used in the reactors
SOLIDS = [("w_Fe2O3", r"$w_{Fe_2O_3}$"), ("w_Fe3O4", r"$w_{Fe_3O_4}$"),
          ("w_FeO", r"$w_{FeO}$"), ("w_Fe", r"$w_{Fe}$")]

# Doesnt plot a specie that never reaches 1 wt% in the bed
W_MIN = 0.01

# The reactors used for these plots
REACTORS = [
    ("C3a", "Reduction", "Reduction"),
    ("C3b", "wet", "Wet Oxidation"),
    ("C3c", "dry", "Dry Oxidation"),
]


def legend_below(ax, ncol, nentries):
    """
    Position the legend lower when it spans multiple rows
    """

    rows = math.ceil(nentries / ncol)
    y = -0.24 if rows == 1 else -0.33
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol)


def main():
    common.apply_style()
    for stem, reactor, title in REACTORS:
        rows, meta = common.load_axial(reactor, "ind")
        height_norm = common.cols(rows, "x_norm")
        subtitle = f"{title} (Industrial, H={meta['H']:.1f} m)"

        # What the solid is becoming along the bed
        shown = [(col, lab) for col, lab in SOLIDS
                 if max(common.cols(rows, col)) >= W_MIN]
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        for i, (col, lab) in enumerate(shown):
            ax.plot(height_norm, common.cols(rows, col), color=common.COLORS[i],
                    ls=common.LSTYLES[i], label=lab)
        ax.set_xlabel("z/H [-]")
        ax.set_ylabel("Solid mass fraction [-]")
        ax.set_title(f"Solid-phase chain: {subtitle}")
        legend_below(ax, ncol=len(shown), nentries=len(shown))
        ax.grid(True)
        fig.tight_layout()
        name = f"{stem}_chain_{reactor}"
        common.save_fig(fig, name)
        plt.close(fig)


if __name__ == "__main__":
    main()

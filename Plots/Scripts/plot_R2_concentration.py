"""
Solid mass fraction vs residence time
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

FIGS = [("reduction", "R2Wa"), ("wet", "R2Wb"), ("dry", "R2Wc")]
NAME = {"reduction": "Reduction", "wet": "Wet Oxidation",
        "dry": "Dry Oxidation"}
SOLIDS = [("w_Fe2O3", r"$w_{Fe_2O_3}$"), ("w_Fe3O4", r"$w_{Fe_3O_4}$"),
          ("w_FeO", r"$w_{FeO}$"), ("w_Fe", r"$w_{Fe}$")]

# Reaction chains
CHAIN = {"reduction": SOLIDS,
         "wet": [s for s in SOLIDS if s[0][2:] != "Fe2O3"],
         "dry": [s for s in SOLIDS if s[0][2:] in ("Fe2O3", "Fe")]}


def main():
    common.apply_style()
    for reactor, stem_w in FIGS:
        rows, meta = common.load_axial(reactor, "lab")
        
        # u_s is stored in every row 
        us = rows[0]["u_solid"]
        tau = [r["z"] / us / 60.0 for r in rows]

        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        for i, (col, lbl) in enumerate(CHAIN[reactor]):
            ax.plot(tau, [100.0 * r[col] for r in rows],
                    color=common.COLORS[i], lw=1.2, label=lbl)
        ax.set_title(f"{NAME[reactor]}: Solid mass fraction profile")
        ax.set_xlabel(r"$\tau_s$ [min]")
        ax.set_ylabel("Solid mass fraction [%]")
        ax.grid(True)
        if ax.get_legend_handles_labels()[0]:
            fig.tight_layout(rect=[0, 0.18, 1, 1])
            fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.015),
                       ncol=4, columnspacing=1.2, handlelength=1.8)
        else:
            fig.tight_layout()
        common.save_fig(fig, stem_w)
        plt.close(fig)


if __name__ == "__main__":
    main()


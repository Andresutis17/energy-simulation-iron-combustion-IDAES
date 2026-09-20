"""
Bubble diameter over column diameter plot. Bubbling and slug regime

"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

REACTORS = [("reduction", "a", "Reduction"),
            ("wet", "b", "Wet Oxidation"),
            ("dry", "c", "Dry Oxidation")]


def bed_title(panel, name, lab_meta, ind_meta):
    return (f"({panel}) {name}, "
            f"$H = {lab_meta['H']:.2f}/{ind_meta['H']:.1f}"
            f"\\,\\mathrm{{m}}$")


def main():
    common.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.7), sharex=True)
    for j, (reactor, panel, name) in enumerate(REACTORS):
        lab_rows, lab_meta = common.load_axial(reactor, "lab")
        ind_rows, ind_meta = common.load_axial(reactor, "ind")
        ax = axes[j]
        z = [r["x_norm"] for r in lab_rows]
        ax.plot(z, [r["db"] / lab_meta["D"] for r in lab_rows],
                color=common.INK, ls=common.LSTYLES[0])
        ax.plot(z, [r["db"] / ind_meta["D"] for r in ind_rows],
                color=common.INK, ls=common.LSTYLES[1])
        ax.axhline(1.0, color=common.INK2, ls=":", lw=1.0)
        if j == 0:
            ax.text(0.04, 1.12, "column diameter", fontsize=7,
                    color=common.INK2, va="bottom")
        ax.set_yscale("log")
        ax.set_ylim(0.004, 5)
        ax.set_title(bed_title(panel, name, lab_meta, ind_meta),
                     fontsize=9)
        ax.set_ylabel("$d_b/D$ [-]")
        ax.set_xlabel("z/H [-]")
        ax.grid(True)
    handles = [Line2D([], [], color=common.INK, ls=common.LSTYLES[0],
                      label="Lab"),
               Line2D([], [], color=common.INK, ls=common.LSTYLES[1],
                      label="Industrial")]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    common.save_fig(fig, "F_hydrodyn_overlay")


if __name__ == "__main__":
    main()

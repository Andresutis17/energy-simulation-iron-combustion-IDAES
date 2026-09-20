"""
Scale validation. Lab vs industrial axial profiles on the normalized
bed height z/H for all three reactors


"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

# The reactors used for these plots
REACTORS = [
    ("reduction", "a", "Reduction", r"$X_{Fe_2O_3}$"),
    ("wet", "b", "Wet Oxidation", r"$X_{Fe}$"),
    ("dry", "c", "Dry Oxidation", r"$X_{Fe}$"),
]
SPECIES = [("w_Fe2O3", r"$w_{Fe_2O_3}$"), ("w_Fe3O4", r"$w_{Fe_3O_4}$"),
           ("w_FeO", r"$w_{FeO}$"), ("w_Fe", r"$w_{Fe}$")]

# The bed heights differ per reactor, so they have it in the panel
def bed_title(panel, name, lab_meta, ind_meta):
    return (f"({panel}) {name}, "
            f"$H = {lab_meta['H']:.2f}/{ind_meta['H']:.1f}"
            f"\\,\\mathrm{{m}}$")


def conversion_figure():
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.7), sharex=True)
    for j, (reactor, panel, name, conv) in enumerate(REACTORS):
        lab_rows, lab_meta = common.load_axial(reactor, "lab")
        ind_rows, ind_meta = common.load_axial(reactor, "ind")
        ax = axes[j]
        z = [r["x_norm"] for r in lab_rows]
        ax.plot(z, common.cols(lab_rows, "X_solid"),
                color=common.INK, ls=common.LSTYLES[0])
        ax.plot(z, common.cols(ind_rows, "X_solid"),
                color=common.INK, ls=common.LSTYLES[1])
        ax.set_title(bed_title(panel, name, lab_meta, ind_meta),
                     fontsize=9)
        ax.set_ylabel(f"{conv} [%]")
        ax.set_xlabel("z/H [-]")
        ax.set_ylim(bottom=0)
        ax.grid(True)
    handles = [Line2D([], [], color=common.INK, ls=common.LSTYLES[0],
                      label="Lab"),
               Line2D([], [], color=common.INK, ls=common.LSTYLES[1],
                      label="Industrial")]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    common.save_fig(fig, "B_conversion_overlay")


def species_figure():
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.7), sharex=True)
    for j, (reactor, panel, name, conv) in enumerate(REACTORS):
        lab_rows, lab_meta = common.load_axial(reactor, "lab")
        ind_rows, ind_meta = common.load_axial(reactor, "ind")
        ax = axes[j]
        z = [r["x_norm"] for r in lab_rows]
        for i, (col, lbl) in enumerate(SPECIES):
            vals_lab = [100.0 * v for v in common.cols(lab_rows, col)]
            vals_ind = [100.0 * v for v in common.cols(ind_rows, col)]
            if max(vals_lab + vals_ind) < 1e-2:
                continue
            ax.plot(z, vals_lab, color=common.COLORS[i], lw=1.2,
                    ls=common.LSTYLES[0])
            ax.plot(z, vals_ind, color=common.COLORS[i], lw=1.2,
                    ls=common.LSTYLES[1])
        ax.set_title(bed_title(panel, name, lab_meta, ind_meta),
                     fontsize=9)
        ax.set_ylabel("$w_i$ [%]")
        ax.set_xlabel("z/H [-]")
        ax.set_ylim(bottom=0)
        ax.grid(True)
    handles = [Line2D([], [], color=common.INK, ls=common.LSTYLES[0],
                      label="Lab"),
               Line2D([], [], color=common.INK, ls=common.LSTYLES[1],
                      label="Industrial")]
    handles += [Line2D([], [], color=common.COLORS[i], lw=1.2, label=lbl)
                for i, (col, lbl) in enumerate(SPECIES)]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    common.save_fig(fig, "B_conversion_species")


def main():
    common.apply_style()
    conversion_figure()
    species_figure()


if __name__ == "__main__":
    main()

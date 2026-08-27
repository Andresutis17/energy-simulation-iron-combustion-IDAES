"""
Set of three axial plots per reactor:
Solid chain, reactant gas in emulsion vs bubble and, phase temperatures

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

# One row per reactor, the loop turns each row into 3 PNGs.
# stem names the files, reactor picks the CSV, title goes on the plot
# Gases are the curves of the plots
FIGS = [
    dict(stem="A1", reactor="dry", title="Dry Oxidation",
         gases=[("O2", r"$y_{O_2}$")]),
    dict(stem="A2", reactor="wet", title="Wet Oxidation",
         gases=[("H2O", r"$y_{H_2O}$"), ("H2", r"$y_{H_2}$")]),
    dict(stem="A3", reactor="reduction", title="Reduction",
         gases=[("H2", r"$y_{H_2}$"), ("H2O", r"$y_{H_2O}$")]),
]

# emulsion = solid line, bubble = dashed line
LS_EMUL, LS_BUB = 0, 1

# Doesnt plot a specie that never reaches 1 wt% in the bed
W_MIN = 0.01


def legend_below(ax, ncol, nentries):

    """
    Puts the legend below the plot, not inside it

    """
    rows = math.ceil(nentries / ncol)
    y = -0.24 if rows == 1 else -0.33
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol)


def new_fig():
    """
    Opens a new plot, same size for every figure
    """
    return plt.subplots(figsize=(3.4, 2.7))


def main():
    common.apply_style()
    for cfg in FIGS:
        rows, meta = common.load_axial(cfg["reactor"], "lab")
        height = common.cols(rows, "z")
        subtitle = f"{cfg['title']} (Lab, H={meta['H']:.2f} m)"

        # a) What the solid is becoming along the bed
        shown = [(col, lab) for col, lab in SOLIDS
                 if max(common.cols(rows, col), key=abs) >= W_MIN]
        fig, ax = new_fig()
        for i, (col, lab) in enumerate(shown):
            ax.plot(height, common.cols(rows, col), color=common.COLORS[i],
                    ls=common.LSTYLES[i], label=lab)
        ax.set_xlabel("Z [m]")
        ax.set_ylabel("Solid mass fraction [-]")
        ax.set_title(f"Solid-phase chain: {subtitle}")
        legend_below(ax, ncol=len(shown), nentries=len(shown))
        ax.grid(True)
        fig.tight_layout()
        name = f"{cfg['stem']}a_{cfg['reactor']}_solid_chain"
        common.save_fig(fig, name)
        plt.close(fig)

        # b) Gases mol fractions along the bed
        fig, ax = new_fig()
        for i, (specie, lab) in enumerate(cfg["gases"]):
            ax.plot(height, common.cols(rows, f"y_emul_{specie}"),
                    color=common.COLORS[i], ls=common.LSTYLES[LS_EMUL],
                    label=f"{lab} emul.")
            ax.plot(height, common.cols(rows, f"y_bub_{specie}"),
                    color=common.COLORS[i], ls=common.LSTYLES[LS_BUB],
                    label=f"{lab} bub.")
        ax.set_xlabel("Z [m]")
        ax.set_ylabel("Gas mol fraction [-]")
        ax.set_title(f"Emulsion vs Bubble: {subtitle}")
        legend_below(ax, ncol=2, nentries=2 * len(cfg["gases"]))
        ax.grid(True)
        fig.tight_layout()
        name = f"{cfg['stem']}b_{cfg['reactor']}_gas_lanes"
        common.save_fig(fig, name)
        plt.close(fig)

        # c) Emulsion, gas and, solid temperatures along the bed
        fig, ax = new_fig()
        temps = [("T_gas", r"$T_{ge}$"), ("T_bub", r"$T_{gb}$"),
                 ("T_sol", r"$T_{se}$")]
        for i, (col, lab) in enumerate(temps):
            ax.plot(height, common.cols(rows, col), color=common.COLORS[i],
                    ls=common.LSTYLES[i], label=lab)
        ax.set_xlabel("Z [m]")
        ax.set_ylabel("Temperature [K]")
        ax.set_title(f"Phase temperatures: {subtitle}")
        legend_below(ax, ncol=3, nentries=len(temps))
        ax.grid(True)
        fig.tight_layout()
        name = f"{cfg['stem']}c_{cfg['reactor']}_temperatures"
        common.save_fig(fig, name)
        plt.close(fig)




if __name__ == "__main__":
    main()

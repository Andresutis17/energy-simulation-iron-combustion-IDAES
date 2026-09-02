"""
Same inlet T for gas and solid, one PNG per lab reactor
"""

import json
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

# The same inlet pair per reactor 
EQ_T = {"reduction": 1173.9, "wet": 1173.9, "dry": 1073.0}
NAME = {"reduction": "Reduction", "wet": "Wet Oxidation",
        "dry": "Dry Oxidation"}
STEM = {"reduction": "A4a_eqT_reduction", "wet": "A4b_eqT_wet",
        "dry": "A4c_eqT_dry"}
TEMP_COLS = [("T_gas", r"$T_{ge}$"), ("T_bub", r"$T_{gb}$"),
             ("T_sol", r"$T_{se}$")]


def converged(meta):
    """
    Only a balanced optimum is data
    """
    return (str(meta.get("term")) == "optimal"
            and float(meta.get("err_mass") or 1e9) <= 1.0)


def main():
    """
    One panel per reactor
    """
    common.apply_style()
    for reactor in ("reduction", "wet", "dry"):
        meta = json.load(open(common.axial_meta(reactor, "lab", "_eqT")))
        t_in = EQ_T[reactor]
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        if converged(meta):
            rows, _ = common.load_axial(reactor, "lab", "_eqT")
            x = common.cols(rows, "x_norm")
            for i, (col, lbl) in enumerate(TEMP_COLS):
                ax.plot(x, common.cols(rows, col), color=common.COLORS[i],
                        ls=common.LSTYLES[i], label=lbl)
            ax.set_title(f"{NAME[reactor]}: Same inlet T = {t_in:g} K")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                          ncols=3)
        else:
            ax.text(0.5, 0.5, f"no solution\n({meta['term']})",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8, color=common.MUTED)
            ax.set_title(f"{NAME[reactor]}: Same inlet T = {t_in:g} K\n"
                         f"(no solution: {meta['term']})")
            ax.set_ylim(900, 1500)
        ax.set_xlabel("z/H [-]")
        ax.set_ylabel("T [K]")
        ax.grid(True)
        fig.tight_layout()
        common.save_fig(fig, STEM[reactor])
        plt.close(fig)


if __name__ == "__main__":
    main()

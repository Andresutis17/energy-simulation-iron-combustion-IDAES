"""Wet step extents vs the mean solids residence time
(appendix)"""
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

STAGE = os.path.join(common.DATA, "op_wet_R10_points.json")
MW = {"Fe": 55.845, "FeO": 71.844, "Fe3O4": 231.53}
NFE_IN = 1000.0 / MW["Fe"]            # mol Fe per kg of fresh Fe feed
TAU_XLABEL = (r"$\bar{\tau}_s = M_{s,\mathrm{bed}}/"
              r"\dot{m}_{s,\mathrm{in}}\,[\mathrm{min}]$")


def exact_tau(f):
    """The exact mean solids residence time of one point.
    f = solids-flow fraction of the baseline (f = 1 is the baseline)"""
    sfx = "_dp028" if f == 1.0 else f"_taufs{int(round(f * 10000)):04d}"
    rows, meta = common.load_axial("wet", "lab", suffix=sfx)
    ts, _, _ = common.tau_profile(rows, meta)
    return ts


def wet_extents(r):
    """Per kg of feed: (E_OW1 = Fe converted, E_OW2 = Fe3O4 formed)"""
    w_out = {k: float(r[f"w_out_{k}"]) for k in MW}
    nfe_loc = (w_out["Fe"] / MW["Fe"] + w_out["FeO"] / MW["FeO"]
               + 3.0 * w_out["Fe3O4"] / MW["Fe3O4"]) * 1000.0
    mfac = NFE_IN / nfe_loc
    n_fe = w_out["Fe"] / MW["Fe"] * 1000.0 * mfac
    n_feo = w_out["FeO"] / MW["FeO"] * 1000.0 * mfac
    n_fe34 = w_out["Fe3O4"] / MW["Fe3O4"] * 1000.0 * mfac
    e1 = NFE_IN - n_fe                    # Fe converted (OW1)
    e2 = n_fe34                           # Fe3O4 formed (OW2)
    feo_check = e1 - 3.0 * e2 - n_feo     # residual (must be ~0)
    return e1, e2, n_fe34, feo_check


def main():
    common.apply_style()
    rows = [r for r in json.load(open(STAGE)) if r.get("valid")
            and r.get("w_out_Fe") not in (None, "")]
    pts = []
    for r in rows:
        f = float(r["value"])
        e1, e2, n34, chk = wet_extents(r)
        h2 = (float(r["n_in_H2O"]) - float(r["n_out_H2O"])) \
            / float(r["solid_mass_in"])
        pts.append((exact_tau(f), f, e1, e2, h2, chk))
    pts.sort(key=lambda p: p[0])

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.plot([p[0] for p in pts], [p[2] for p in pts],
            color=common.COLORS[0], lw=1.2, marker="^", ms=4,
            label=r"WOR1: Fe$\rightarrow$FeO")
    ax.plot([p[0] for p in pts], [p[3] for p in pts],
            color=common.INK, lw=1.2, marker="s", ms=3.5,
            label=r"WOR2: 3FeO$\rightarrow$Fe$_3$O$_4$")
    base = next(p for p in pts if p[1] == 1.0)
    ax.plot([base[0]], [base[2]], marker="*", ms=11, color=common.INK,
            ls="none", mec="white", mew=0.6, zorder=5)
    ax.plot([base[0]], [base[3]], marker="*", ms=11, color=common.INK,
            ls="none", mec="white", mew=0.6, zorder=5)
    ax.set_xlabel(TAU_XLABEL)
    ax.set_ylabel(r"Step extent [mol kg$^{-1}$]")
    ax.grid(True)

    handles = [
        Line2D([], [], color=common.COLORS[0], lw=1.2, marker="^",
               ms=4, label=r"WOR1: Fe$\rightarrow$FeO"),
        Line2D([], [], color=common.INK, lw=1.2, marker="s", ms=3.5,
               label=r"WOR2: 3FeO$\rightarrow$Fe$_3$O$_4$"),
        Line2D([], [], marker="*", ms=11, color=common.INK,
               ls="none", mec="white", mew=0.6,
               label="Baseline operating point ($f = 1$)"),
    ]
    common.legend_band(fig, ncol=2, handles=handles, bottom=0.18)
    common.save_fig(fig, "R10e_wet_extents")
    plt.close(fig)


if __name__ == "__main__":
    main()

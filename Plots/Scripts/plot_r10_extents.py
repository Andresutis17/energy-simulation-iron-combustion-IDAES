"""
Reduction step extents vs the mean solids residence time
Appendix
"""
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from run_p_sweep import extents

STAGE = os.path.join(common.DATA, "op_reduction_R10_points.json")
TAU_XLABEL = (r"$\bar{\tau}_s = M_{s,\mathrm{bed}}/"
              r"\dot{m}_{s,\mathrm{in}}\,[\mathrm{min}]$")
STEPS = [
    (r"RD$_1$: Fe$_2$O$_3\rightarrow$Fe$_3$O$_4$", 0, ":", "^",
     common.COLORS[0]),
    (r"RD$_2$: Fe$_3$O$_4\rightarrow$FeO", 1, "--", "s",
     common.COLORS[1]),
    (r"RD$_3$: FeO$\rightarrow$Fe", 2, "-", "o", common.INK),
]


def exact_tau(f):

    """
    The mean solid residence time of one f, from its own axial
    f = solid flow fraction of the baseline. f = 1 is the baseline
    """
    sfx = "_lam3" if f == 1.0 else f"_taufs{int(round(f * 10000)):04d}"
    rows, meta = common.load_axial("reduction", "lab", suffix=sfx)
    ts, _, _ = common.tau_profile(rows, meta)
    return ts


def main():
    common.apply_style()
    rows = [r for r in json.load(open(STAGE)) if r.get("valid")]
    pts = []
    for r in rows:
        f = float(r["value"])
        w = {k: float(r[f"w_out_{k}"])
             for k in ("Fe2O3", "Fe3O4", "FeO", "Fe")}
        e = extents(w)
        pts.append((exact_tau(f), f, e))
    pts.sort(key=lambda p: p[0])
    xs = [p[0] for p in pts]
    e1 = [p[2][0] for p in pts]
    e2 = [p[2][1] for p in pts]
    e3 = [p[2][2] for p in pts]
    comp = [e3 / (3.0 * e2) for e2, e3 in zip(e2, e3)]
    base_i = next(i for i, p in enumerate(pts) if p[1] == 1.0)

    fig, axes = common.stack(2)

    ax = axes[0]
    for (lbl, idx, ls, mk, color), ys in zip(STEPS, (e1, e2, e3)):
        ax.plot(xs, ys, color=color, lw=1.2, marker=mk, ms=3.5,
                ls=ls, label=lbl)
    for y in (e1, e2, e3):
        ax.plot([xs[base_i]], [y[base_i]], marker="*", ms=11,
                color=common.INK, ls="none", mec="white", mew=0.6,
                zorder=5)
    ax.set_title("(a) Reduction step extents")
    ax.set_ylabel(r"Step extent [mol kg$^{-1}$]")
    ax.set_xlabel(TAU_XLABEL)
    ax.legend(fontsize=6.5, frameon=False, loc="center left")
    ax.grid(True)

    ax = axes[1]
    ax.plot(xs, comp, color=common.INK, lw=1.2, marker="o", ms=3.5)
    ax.plot([xs[base_i]], [comp[base_i]], marker="*", ms=11,
            color=common.INK, ls="none", mec="white", mew=0.6, zorder=5)
    ax.set_title("(b) Final-step completion at FeO")
    ax.set_ylabel(r"$E_{RD_3}/(3\,E_{RD_2})$ [-]")
    ax.set_xlabel(TAU_XLABEL)
    ax.grid(True)

    handles = [Line2D([], [], marker="*", ms=11, color=common.INK,
                      ls="none", mec="white", mew=0.6,
                      label="Baseline operating point ($f = 1$)")]
    common.legend_band(fig, ncol=1, handles=handles, bottom=0.10)
    common.save_fig(fig, "R10_reduction_extents")
    plt.close(fig)


if __name__ == "__main__":
    main()

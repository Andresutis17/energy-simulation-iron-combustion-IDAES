"""

Reduction operating multiplot vs lambda_H2 
The shared operating_figure builder for the reduction, wet, dry families

"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import outlet_gas as og
from plot_R3_lambda import SPECIES, PROD, CHAIN

CSV_NAME = "sens_reduction_lambda_H2_operating_refined.csv"
BASE_LAM = 1.5809047487510743   # baseline lambda

# Per reactor utilization column and label for panel C
UTIL = {"reduction": ("H2_utilization", r"$H_2$ utilization [%]"),
        "wet": ("steam_utilization", r"$H_2O$ utilization [%]"),
        "dry": ("O2_utilization", r"$O_2$ utilization [%]")}
UTIL_COLOR = "#8B4513" 


def num(row, key):
    v = row.get(key)
    return float(v) if v not in (None, "") else None


# Baseline mean solid residence time per reactor, read from the frozen anchor axial 
_TAUS = {}


def base_tau_s(reactor, lam3=None):

    """

    Baseline tau_s [min] of a reactor

    """
    if lam3 is None:
        lam3 = reactor == "reduction"
    sfx = "_lam3" if (reactor == "reduction" and lam3) else "_dp028"
    key = (reactor, sfx)
    if key not in _TAUS:
        try:
            rows, meta = common.load_axial(reactor, "lab", suffix=sfx)
            _TAUS[key] = common.tau_profile(rows, meta)[0]
        except FileNotFoundError:
            _TAUS[key] = None
    return _TAUS[key]


def put_star(ax, x, y):
    """
    The baseline star
    """
    ax.plot([x], [y], marker="*", ms=11, color=common.INK, ls="none",
            mec="white", mew=0.6, zorder=5)


def valid_rows(rows):
    """
    Split into plottable and excluded
   
    """
    ok, dead = [], []
    for r in rows:
        (ok if r.get("valid") == "True" and r.get("term") == "optimal"
         else dead).append(r)
    return ok


def operating_figure(ok, xkey, base_val, base_label, xlabel, stem,
                     vlines=(), reactor="reduction", top_scale=None,
                     top_label=None, top_func=None, gamma=False,
                     lam_note=None,
                     top_label_pad=10, title_pad_top=34,
                     util_ylim=None, util_marker_only=False):
    """
    The shared four panel builder
    """
    chain = CHAIN[reactor]
    util_key, util_lbl = UTIL[reactor]
    xs = [num(r, xkey) for r in ok]
    base = next((r for r in ok
                 if abs(num(r, xkey) - base_val) < 1e-3), None)

    def mark(ax):
        for v, _ in vlines:
            ax.axvline(v, color=common.MUTED, ls=":", lw=1.0, zorder=1)

    fig, axes = common.stack(4)
    handles = []

    # Solid outlet mass fractions, the primary solid result
    ax = axes[0]
    for i, (sp, lbl) in enumerate(chain):
        ys = [100.0 * num(r, f"w_out_{sp}") for r in ok]
        ax.plot(xs, ys, color=common.COLORS[i], lw=1.2, marker="o",
                ms=3, label=lbl)
        handles.append(ax.plot([], [], color=common.COLORS[i], lw=1.2,
                               label=lbl)[0])
    if base:
        put_star(ax, num(base, xkey),
                 100.0 * num(base, f"w_out_{PROD[reactor]}"))
    ttl = "(a) Solid outlet"
    if lam_note:
        ttl += ", " + lam_note
    ax.set_title(ttl)
    ax.set_ylabel("Solid outlet\nmass fraction [%]")
    ax.grid(True)
    mark(ax)

    # Gas outlet mole fractions
    ax = axes[1]
    prod_gas = og.GAS_SP[reactor][-1]
    for j, sp in enumerate(og.GAS_SP[reactor]):
        color = common.INK if j == 0 else og.GAS_PROD
        ys = [og.row_gas_xs(reactor, "x", r)[sp] for r in ok]
        ax.plot(xs, ys, color=color, lw=1.2, marker="o", ms=3)
        handles.append(ax.plot([], [], color=color, lw=1.2,
                               label=rf"$y_{{{sp}}}$")[0])
        if base and sp == prod_gas:
            put_star(ax, num(base, xkey),
                     og.row_gas_xs(reactor, "x", base)[sp])
    ax.set_title("(b) Gas outlet")
    ax.set_ylabel("Gas outlet\nmole fraction [%]")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(True)
    mark(ax)
    if gamma:
        # Gas oxidation degree on right axis 
        og.add_gamma(ax, list(zip(xs, ok)), reactor, "x")
        handles.append(Line2D([], [], color="#d95f02", ls="--",
                              lw=1.2, label=r"$\Gamma$"))

    # Reactant utilization
    ax = axes[2]
    ys = [100.0 * num(r, util_key) for r in ok]
    if util_marker_only:
        ax.plot(xs, ys, color=UTIL_COLOR, marker="o", ms=4, ls="none")
    else:
        ax.plot(xs, ys, color=UTIL_COLOR, lw=1.2, marker="o", ms=3)
    handles.append(ax.plot([], [], color=UTIL_COLOR, lw=1.2,
                           label=util_lbl.replace(" [%]", ""))[0])
    if base:
        put_star(ax, num(base, xkey), 100.0 * num(base, util_key))
    ax.set_title("(c) Utilization")
    ax.set_ylabel(util_lbl)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    if util_ylim is not None:
        ax.set_ylim(util_ylim)
    ax.grid(True)
    mark(ax)

    # Outlet temperatures
    ax = axes[3]
    ax.plot(xs, [num(r, "T_solid_out") for r in ok], color=common.INK,
            lw=1.2, marker="o", ms=3, label=r"$T_{solid,out}$")
    ax.plot(xs, [num(r, "T_gas_out") for r in ok], color=common.INK2,
            ls="--", lw=1.2, marker="o", ms=3, label=r"$T_{gas,out}$")
    handles.append(ax.plot([], [], color=common.INK, lw=1.2,
                           label=r"$T_{solid,out}$")[0])
    handles.append(ax.plot([], [], color=common.INK2, ls="--", lw=1.2,
                           label=r"$T_{gas,out}$")[0])
    if base:
        put_star(ax, num(base, xkey), num(base, "T_solid_out"))
    ax.set_title("(d) Outlet temperatures")
    ax.set_ylabel("Outlet temperature [K]")
    ax.set_xlabel(xlabel)
    ax.grid(True)
    mark(ax)

    if top_scale is not None and len(top_scale) == len(xs):
        # Second reading top axis, one value per point, else skipped
        sec = fig.axes[0].twiny()
        sec.set_xlim(fig.axes[0].get_xlim())
        step = max(1, len(xs) // 4)
        idx = list(range(0, len(xs), step))
        if idx[-1] != len(xs) - 1:
            idx.append(len(xs) - 1)
        span = max(xs) - min(xs)
        kept = [idx[0]]
        for i in idx[1:]:
            if xs[i] - xs[kept[-1]] > 0.14 * span:
                kept.append(i)
        idx = kept
        sec.set_xticks([xs[i] for i in idx])
        sec.set_xticklabels([f"{top_scale[i]:.3g}" for i in idx])
        sec.tick_params(axis="x", labelsize=7)
        sec.grid(False)
        # Keep numbers, axis label and panel title from sharing one strip
        sec.set_xlabel(top_label or "Throughput scale, s",
                       labelpad=top_label_pad)
        fig.axes[0].set_title(fig.axes[0].get_title(), pad=title_pad_top)
    if top_func is not None:

        # Linear second reading lambda vs u0/umf, T1 secondary-axis
        sec = fig.axes[0].secondary_xaxis("top", functions=top_func)
        sec.set_xlabel(top_label, labelpad=top_label_pad)
        sec.grid(False)
        fig.axes[0].set_title(fig.axes[0].get_title(), pad=title_pad_top)

    # Reference marks identified in the shared legend
    for _, lbl in vlines:
        handles.append(Line2D([], [], color=common.MUTED, ls=":",
                              lw=1.0, label=lbl))
    handles.append(Line2D([], [], marker="*", ms=11, color=common.INK,
                          ls="none", mec="white", mew=0.6,
                          label=base_label))
    
    # Inlet temperatures of both feeds
    def _t_span(key):
        vals = [num(r, key) for r in ok]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        lo, hi = min(vals), max(vals)
        return f"{lo:g}" if hi - lo < 0.5 else f"{lo:g} to {hi:g}"
    tg_in, ts_in = _t_span("T_gas_in"), _t_span("T_solid_in")
    if tg_in and ts_in:
        handles.append(Line2D(
            [], [], ls="none",
            label=rf"$T_{{g,in}} = {tg_in}$ K, "
                  rf"$T_{{s,in}} = {ts_in}$ K (inlet)"))
    # Baseline mean solid residence time 
    tau0 = base_tau_s(reactor)
    if tau0:
        handles.append(Line2D(
            [], [], ls="none",
            label=rf"$\bar{{\tau}}_s = {tau0:.1f}$ min (baseline)"))
    common.legend_band(fig, ncol=3, handles=handles, bottom=0.11)
    common.save_fig(fig, stem)
    plt.close(fig)


def main():
    common.apply_style()
    rows = common.load_csv(CSV_NAME)
    ok = valid_rows(rows)
    ok.sort(key=lambda r: num(r, "u_g_over_umf"))
    lam_of = lambda r: float(r.get("lambda_H2") or r.get("lambda_requested"))
    # Fixed composition and fixed u_mf: u0/umf scales 1:1 with lambda
    stoich_u = num(ok[0], "u_g_over_umf") / lam_of(ok[0])
    operating_figure(ok, "u_g_over_umf", stoich_u * BASE_LAM,
                     r"Baseline operating point, "
                     r"$\lambda_{H_2}\approx 1.58$",
                     r"Inlet $u_g/u_{mf}$ (via gas flow)",
                     "R4_operating_reduction",
                     gamma=True,
                     top_func=(lambda u: u / stoich_u,
                               lambda l: l * stoich_u),
                     top_label=r"$\lambda_{H_2}$",
                     vlines=[(stoich_u, r"Stoichiometric feed, "
                                        r"$\lambda_{H_2}=1$")])


if __name__ == "__main__":
    main()

"""
Reduction, wet and dry composition vs the representative bed
temperature, T_gas and T_solid routes overlaid

"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import outlet_gas as og
from plot_R3_lambda import CHAIN, PROD
from plot_R4_operating_reduction import (num, put_star, valid_rows,
                                         UTIL, UTIL_COLOR, base_tau_s)

XKEY = "T_solid_out"
XLABEL = r"Representative bed temperature, $T_{s,out}$ [K]"
PAIR_TOL_K = 10.0

CONFIG = {
    "reduction": {
        "csvs": [("T_gas", "sens_reduction_T_gas_lam3_operating.csv"),
                 ("T_solid", "sens_reduction_T_solid_lam3_operating.csv")],
        "anchor": (1050.0, 1173.9), "gamma": True,
        "lam_note": r"$\lambda_{H_2}=3$",
        "stem": "R5c_operating_reactortemp",
    },
    "wet": {
        "csvs": [("T_gas", "sens_wet_T_gas_operating.csv"),
                 ("T_solid", "sens_wet_T_solid_operating.csv")],
        "anchor": (1050.0, 1173.9), "gamma": True,
        "lam_note": r"$\lambda_{steam}=1.244$",
        "stem": "W5c_operating_reactortemp",
        "exclude": {"T_gas": [900.0, 923.0, 973.0]},
    },
    "dry": {
        "csvs": [("T_gas", "sens_dry_T_gas_operating.csv"),
                 ("T_solid", "sens_dry_T_solid_operating.csv"),
                 ("T_gas", "sens_dry_T_gas_overlap_operating.csv"),
                 ("T_solid", "sens_dry_T_solid_overlap_operating.csv")],
        "anchor": (600.0, 1073.0), "gamma": False,
        "lam_note": r"$\lambda_{O_2}=1.501$",
        "stem": "D5c_operating_reactortemp",
        "exclude": {"T_gas": [948.0]},
    },
}


def collapse_stats(fam_a, fam_b, util_key):

    """

    Interpolate family B onto family As bed temperatures and compare
    at the same temperature

    """

    import numpy as np
    ta = [num(r, XKEY) for r in fam_a]
    tb = [num(r, XKEY) for r in fam_b]
    lo, hi = max(min(tb), min(ta)), min(max(tb), max(ta))
    order = np.argsort(np.array(tb))
    xs_b = np.array(tb)[order]
    xg_b = np.array([num(r, "X_gas") for r in fam_b])[order]
    ut_b = np.array([100.0 * num(r, util_key) for r in fam_b])[order]
    ts, dx, du = [], [], []
    for r in fam_a:
        t = num(r, XKEY)
        if t is None or not (lo <= t <= hi):
            continue
        dx.append(abs(num(r, "X_gas")
                      - float(np.interp(t, xs_b, xg_b))))
        du.append(abs(100.0 * num(r, util_key)
                      - float(np.interp(t, xs_b, ut_b))))
        ts.append(t)
    return ts, dx, du


def run_reactor(reactor, cfg):
    common.apply_style()
    fams = []
    for label, csv_name in cfg["csvs"]:
        if not os.path.exists(os.path.join(common.DATA, csv_name)):
            continue
        rows = common.load_csv(csv_name)
        exclude = cfg.get("exclude", {}).get(label)
        if exclude:
            rows = [r for r in rows
                    if not (r.get("value")
                            and any(abs(float(r["value"]) - x) < 1e-6
                                    for x in exclude))]
        rows = valid_rows(rows)
        rows.sort(key=lambda r: num(r, XKEY))
        fams.append((label, rows, "o" if label == "T_gas" else "s",
                     "-" if label == "T_gas" else "--"))

    tg_in, ts_in = cfg["anchor"]
    anchor = None
    for _, rows, _, _ in fams:
        for r in rows:
            if abs(num(r, "Tg_in") - tg_in) < 1e-6 \
                    and abs(num(r, "Ts_in") - ts_in) < 1e-6:
                anchor = r
                break
        if anchor:
            break
    if anchor is None:
        raise SystemExit(f"[{reactor}] no anchor")

    fig, axes = common.stack(3)
    handles = []

    # Solid outlet chain
    ax = axes[0]
    for i, (sp, lbl) in enumerate(CHAIN[reactor]):
        for _, rows, mk, ls in fams:
            ax.plot([num(r, XKEY) for r in rows],
                    [100.0 * num(r, f"w_out_{sp}") for r in rows],
                    color=common.COLORS[i], lw=1.2, marker=mk, ms=3.5,
                    ls=ls)
        handles.append(ax.plot([], [], color=common.COLORS[i], lw=1.2,
                               label=lbl)[0])
    if anchor:
        put_star(ax, num(anchor, XKEY),
                 100.0 * num(anchor, f"w_out_{PROD[reactor]}"))
    ax.set_title("(a) Solid outlet, " + cfg["lam_note"])
    ax.set_ylabel("Solid outlet\nmass fraction [%]")
    ax.grid(True)

    # Gas outlet mole fractions + gamma right axis 
    ax = axes[1]
    for j, sp in enumerate(og.GAS_SP[reactor]):
        color = common.INK if j == 0 else og.GAS_PROD
        for _, rows, mk, ls in fams:
            ax.plot([num(r, XKEY) for r in rows],
                    [og.row_gas_xs(reactor, "x", r)[sp] for r in rows],
                    color=color, lw=1.2, marker=mk, ms=3.5, ls=ls)
        handles.append(ax.plot([], [], color=color, lw=1.2,
                               label=rf"$y_{{{sp}}}$")[0])
    if anchor:
        put_star(ax, num(anchor, XKEY),
                 og.row_gas_xs(reactor, "x",
                               anchor)[og.GAS_SP[reactor][-1]])
    if cfg["gamma"]:
        og.add_gamma(ax, [(num(r, XKEY), r) for _, rows, _, _ in fams
                          for r in rows], reactor, "x")
        handles.append(Line2D([], [], color="#d95f02", ls="--", lw=1.2,
                              label=r"$\Gamma$"))
    ax.set_title("(b) Gas outlet")
    ax.set_ylabel("Gas outlet\nmole fraction [%]")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(True)

    # Reactant utilization
    ax = axes[2]
    util_key, util_lbl = UTIL[reactor]
    for _, rows, mk, ls in fams:
        ax.plot([num(r, XKEY) for r in rows],
                [100.0 * num(r, util_key) for r in rows],
                color=UTIL_COLOR, lw=1.2, marker=mk, ms=3.5, ls=ls)
    handles.append(ax.plot([], [], color=UTIL_COLOR, lw=1.2,
                           label=util_lbl.replace(" [%]", ""))[0])
    if anchor:
        put_star(ax, num(anchor, XKEY), 100.0 * num(anchor, util_key))
    ax.set_title("(c) Utilization")
    ax.set_ylabel(util_lbl)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_xlabel(XLABEL)
    ax.grid(True)

    handles.append(Line2D([], [], color=common.INK, marker="o", ms=4.5,
                          ls="-", lw=1.2, label=r"via $T_{gas,in}$"))
    handles.append(Line2D([], [], color=common.INK, marker="s", ms=4,
                          ls="--", lw=1.2, label=r"via $T_{solid,in}$"))
    handles.append(Line2D([], [], marker="*", ms=11, color=common.INK,
                          ls="none", mec="white", mew=0.6,
                          label="Baseline operating point"))
    tg_in0, ts_in0 = cfg["anchor"]
    handles.append(Line2D(
        [], [], ls="none",
        label=rf"Baseline inlets: $T_{{g,in}} = {tg_in0:g}$ K, "
              rf"$T_{{s,in}} = {ts_in0:g}$ K"))
    
    # baseline mean solid residence time 
    tau0 = base_tau_s(reactor)
    if tau0:
        handles.append(Line2D(
            [], [], ls="none",
            label=rf"$\bar{{\tau}}_s = {tau0:.1f}$ min (baseline)"))
    common.legend_band(fig, ncol=3, handles=handles, bottom=0.11)
    common.save_fig(fig, cfg["stem"])
    plt.close(fig)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for reactor, cfg in CONFIG.items():
        if only and reactor != only:
            continue
        run_reactor(reactor, cfg)


if __name__ == "__main__":
    main()

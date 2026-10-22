"""

Wet step extents vs the representative bed temperature
Appendix

"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from plot_R4_operating_reduction import num, valid_rows, base_tau_s

FAMS = [("sens_wet_T_gas_operating.csv", "o", "-", [900.0, 923.0, 973.0]),
        ("sens_wet_T_solid_operating.csv", "s", "--", [])]
MW = {"Fe2O3": 0.159688, "Fe3O4": 0.231533, "FeO": 0.071844, "Fe": 0.055845}

# T_gas,in = 1050 K, T_solid,in = 1173.9 K
ANCHOR = (1050.0, 1173.9)


def extents(r):

    """
    Step extents WOR1 and WOR2 of one row [mol/s]  from the solid in/out composition
    """
    mi, mo = num(r, "solid_mass_in"), num(r, "solid_mass_out")
    n_in = {k: mi * num(r, "w_in_" + k) / MW[k] for k in MW}
    n_out = {k: mo * num(r, "w_out_" + k) / MW[k] for k in MW}
    d = {k: n_out[k] - n_in[k] for k in MW}
    return -d["Fe"], d["Fe3O4"]


def load_fam(csv_name, exclude):

    """One family ready to plot, skip the rows the figure leaves out,
    keep the valid ones, sorted by bed temperature"""
    rows = common.load_csv(csv_name)
    if exclude:
        rows = [r for r in rows
                if not (r.get("value")
                        and any(abs(float(r["value"]) - x) < 1e-6
                                for x in exclude))]
    ok = valid_rows(rows)
    ok.sort(key=lambda r: num(r, "T_solid_out"))
    return ok


def main():
    common.apply_style()
    fams = []
    for csv_name, mk, ls, exclude in FAMS:
        ok = load_fam(csv_name, exclude)
        fams.append((ok, mk, ls))

    fig, axes = common.stack(2)

    ax = axes[0]
    for ok, mk, ls in fams:
        xs = [num(r, "T_solid_out") for r in ok]
        ex = [extents(r) for r in ok]
        ax.plot(xs, [1e3 * e[0] for e in ex], color=common.COLORS[0],
                lw=1.2, marker=mk, ms=3.5, ls=ls)
        ax.plot(xs, [1e3 * e[1] for e in ex], color=common.COLORS[1],
                lw=1.2, marker=mk, ms=3.5, ls=ls)
    ax.plot([], [], color=common.COLORS[0], lw=1.2,
            label=r"WOR1: Fe$\rightarrow$FeO")
    ax.plot([], [], color=common.COLORS[1], lw=1.2,
            label=r"WOR2: 3FeO$\rightarrow$Fe$_3$O$_4$")
    ax.set_title("(a) Wet step extents")
    ax.set_ylabel("Step extent [mmol/s]")
    ax.legend(fontsize=6.5, frameon=False, loc="center right")
    ax.grid(True)

    ax = axes[1]
    for ok, mk, ls in fams:
        xs = [num(r, "T_solid_out") for r in ok]
        for k, color in (("Fe", common.INK), ("FeO", common.COLORS[1]),
                         ("Fe3O4", common.COLORS[2])):
            ax.plot(xs, [100.0 * num(r, "w_out_" + k) for r in ok],
                    color=color, lw=1.2, marker=mk, ms=3.5, ls=ls)
    ax.set_title("(b) Solid outlet")
    ax.set_ylabel("Solid outlet\nmass fraction [%]")
    ax.grid(True)

    # The anchor row, found by its inlet pair like the 5c builder
    for ax in axes:
        for ok, _, _ in fams:
            for r in ok:
                if abs(num(r, "Tg_in") - ANCHOR[0]) < 1e-6 \
                        and abs(num(r, "Ts_in") - ANCHOR[1]) < 1e-6:
                    x0 = num(r, "T_solid_out")
                    if ax is axes[0]:
                        ex = extents(r)
                        for y in (1e3 * ex[0], 1e3 * ex[1]):
                            ax.plot([x0], [y], marker="*", ms=11,
                                    color=common.INK, ls="none",
                                    mec="white", mew=0.6, zorder=5)
                    else:
                        for k in ("Fe", "FeO", "Fe3O4"):
                            ax.plot([x0], [100.0 * num(r, "w_out_" + k)],
                                    marker="*", ms=11, color=common.INK,
                                    ls="none", mec="white", mew=0.6,
                                    zorder=5)
                    break

    handles = [
        Line2D([], [], marker="o", color=common.INK, ms=4.5, ls="-",
               lw=1.2, label=r"via $T_{gas,in}$"),
        Line2D([], [], marker="s", color=common.INK, ms=4, ls="--",
               lw=1.2, label=r"via $T_{solid,in}$"),
        Line2D([], [], marker="*", ms=11, color=common.INK, ls="none",
               mec="white", mew=0.6, label="Baseline operating point"),
        Line2D([], [], color=common.INK, lw=1.2, label="Fe"),
        Line2D([], [], color=common.COLORS[1], lw=1.2, label="FeO"),
        Line2D([], [], color=common.COLORS[2], lw=1.2,
               label=r"Fe$_3$O$_4$"),
    ]
    
    # Baseline mean solid residence time 
    tau0 = base_tau_s("wet")
    if tau0:
        handles.append(Line2D(
            [], [], ls="none",
            label=rf"$\bar{{\tau}}_s = {tau0:.1f}$ min (baseline)"))
    common.legend_band(fig, ncol=4, handles=handles, bottom=0.18)
    common.save_fig(fig, "W5c_step_extents")
    plt.close(fig)


if __name__ == "__main__":
    main()

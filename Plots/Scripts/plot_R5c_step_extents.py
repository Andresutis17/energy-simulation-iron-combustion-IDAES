"""
Reduction step extents vs the representative bed temperature
Appendix
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from plot_R4_operating_reduction import num, valid_rows, base_tau_s

FAMS = [("sens_reduction_T_gas_lam3_operating.csv", "o", "-"),
        ("sens_reduction_T_solid_lam3_operating.csv", "s", "--")]
MW = {"Fe2O3": 0.159688, "Fe3O4": 0.231533, "FeO": 0.071844, "Fe": 0.055845}

# The baseline operating point T_gas,in = 1050 K  T_solid,in = 1173.9 K 
ANCHOR = (1050.0, 1173.9)


def extents(r):
    """
    Step extents RD1, RD2, RD3 [mol/s],
    from the solid in/out composition and the stoichiometry chain
    """
    mi, mo = num(r, "solid_mass_in"), num(r, "solid_mass_out")
    n_in = {k: mi * num(r, "w_in_" + k) / MW[k] for k in MW}
    n_out = {k: mo * num(r, "w_out_" + k) / MW[k] for k in MW}
    d = {k: n_out[k] - n_in[k] for k in MW}
    rd1 = -d["Fe2O3"] / 3.0
    rd2 = 2.0 * rd1 - d["Fe3O4"]
    rd3 = 3.0 * rd2 - d["FeO"]
    return rd1, rd2, rd3


def main():
    common.apply_style()
    fams = []
    for csv_name, mk, ls in FAMS:
        rows = common.load_csv(csv_name)
        ok = valid_rows(rows)
        ok.sort(key=lambda r: num(r, "T_solid_out"))
        fams.append((ok, mk, ls))

    fig, axes = common.stack(2)

    ax = axes[0]
    for ok, mk, ls in fams:
        xs = [num(r, "T_solid_out") for r in ok]
        rd = [extents(r) for r in ok]
        ax.plot(xs, [1e3 * e[0] for e in rd], color=common.COLORS[0],
                lw=1.2, marker=mk, ms=3.5, ls=":")
        ax.plot(xs, [1e3 * e[1] for e in rd], color=common.COLORS[1],
                lw=1.2, marker=mk, ms=3.5, ls="--")
        ax.plot(xs, [1e3 * e[2] for e in rd], color=common.INK, lw=1.4,
                marker=mk, ms=3.5, ls=ls)
    ax.plot([], [], color=common.COLORS[0], lw=1.2, ls=":",
            label=r"RR$_1$: Fe$_2$O$_3\rightarrow$Fe$_3$O$_4$")
    ax.plot([], [], color=common.COLORS[1], lw=1.2, ls="--",
            label=r"RR$_2$: Fe$_3$O$_4\rightarrow$FeO")
    ax.plot([], [], color=common.INK, lw=1.4,
            label=r"RR$_3$: FeO$\rightarrow$Fe")
    ax.set_title("(a) Reduction step extents")
    ax.set_ylabel("Step extent [mmol/s]")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax.grid(True)

    ax = axes[1]
    for ok, mk, ls in fams:
        xs = [num(r, "T_solid_out") for r in ok]
        comp = [e[2] / (3.0 * e[1]) for e in
                (extents(r) for r in ok)]
        ax.plot(xs, comp, color=common.INK, lw=1.2, marker=mk, ms=3.5,
                ls=ls)
    ax.set_title("(b) Final step completion at FeO")
    ax.set_ylabel(r"$\dot n_{RR_3}/(3\,\dot n_{RR_2})$ [-]")
    ax.grid(True)

    # anchor row, found by its inlet pair like the 5c builder
    for ax in axes:
        for ok, _, _ in fams:
            for r in ok:
                if abs(num(r, "Tg_in") - ANCHOR[0]) < 1e-6 \
                        and abs(num(r, "Ts_in") - ANCHOR[1]) < 1e-6:
                    x0 = num(r, "T_solid_out")
                    e = extents(r)
                    y0 = (1e3 * e[2]) if ax is axes[0] \
                        else e[2] / (3.0 * e[1])
                    ax.plot([x0], [y0], marker="*", ms=11,
                            color=common.INK, ls="none", mec="white",
                            mew=0.6, zorder=5)
                    break

    handles = [
        Line2D([], [], marker="o", color=common.INK, ms=4.5, ls="-",
               lw=1.2, label=r"via $T_{gas,in}$"),
        Line2D([], [], marker="s", color=common.INK, ms=4, ls="--",
               lw=1.2, label=r"via $T_{solid,in}$"),
        Line2D([], [], marker="*", ms=11, color=common.INK, ls="none",
               mec="white", mew=0.6, label="Baseline operating point"),
    ]
    tau0 = base_tau_s("reduction")
    if tau0:
        handles.append(Line2D(
            [], [], ls="none",
            label=rf"$\bar{{\tau}}_s = {tau0:.1f}$ min (baseline)"))
    common.legend_band(fig, ncol=2, handles=handles, bottom=0.14)
    common.save_fig(fig, "R5c_step_extents")
    plt.close(fig)


if __name__ == "__main__":
    main()

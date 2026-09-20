"""
Conversion vs gas velocity and vs solid residence time


"""
import math
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import outlet_gas as og

R_GAS = 8.314
P_ATM = 101325.0
D_LAB = 0.054
A_LAB = math.pi * D_LAB ** 2 / 4.0
U_MF = 0.039624  

FIGS = [("reduction", "T1a", "T2a"), ("wet", "T1b", "T2b"),
        ("dry", "T1c", "T2c")]
T2W_STEM = {"reduction": "T2Wa", "wet": "T2Wb", "dry": "T2Wc"}
T1W_STEM = {"reduction": "T1Wa", "wet": "T1Wb", "dry": "T1Wc"}
NAME = {"reduction": "Reduction", "wet": "Wet Oxidation",
        "dry": "Dry Oxidation"}

# Outlet mass fraction species
SPECIES = [("Fe2O3", r"$w_{Fe_2O_3}$"), ("Fe3O4", r"$w_{Fe_3O_4}$"),
           ("FeO", r"$w_{FeO}$"), ("Fe", r"$w_{Fe}$")]
CHAIN = {"reduction": SPECIES,
         "wet": [s for s in SPECIES if s[0] != "Fe2O3"],
         "dry": [s for s in SPECIES if s[0] in ("Fe2O3", "Fe")]}

# Legends
ALIAS = {"reduction": r"$X_{Fe}$ (= $X_{FeO}$)",
         "wet": r"$X_{Fe_3O_4}$ (= $X_{FeO}$)",
         "dry": r"$X_{Fe_2O_3}$ (= $X_{Fe}$)"}
GAS_LBL = {"reduction": r"$X_{H_2}$", "wet": r"$X_{H_2O}$",
           "dry": r"$X_{O_2}$"}
STAGE_LBL = {
    "reduction": [("fe23", r"$X_{\mathrm{Fe_2O_3}}$"),
                  ("fe34", r"$X_{\mathrm{Fe_3O_4}}$")],
    "wet": [("feox", r"$X_{\mathrm{Fe}}$")],
    "dry": [],
}
STAGE_C = ["#d95f02", "#8c510a"]

# legends under the x label 
LEG_BAND = {"1row": 0.12, "2row": 0.18}  


def legend_outside(fig, ax, reactor):
    """
    Two-column legend in the reserved band under the x label
    """
    band = LEG_BAND["2row" if STAGE_LBL[reactor] else "1row"]
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0, band, 1, 1])
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=2, columnspacing=1.2, handlelength=1.8)


def valid_rows(fam):
    """
    Valid points of one sensitivity family
    """
    path = os.path.join(common.DATA, f"sens_{fam}{common.SUFFIX}.csv")
    if not os.path.exists(path):
        print(f"  (missing {os.path.basename(path)} - skipped)", flush=True)
        return []
    rows = common.load_csv(f"sens_{fam}{common.SUFFIX}.csv")
    out = []
    for r in rows:
        if r["valid"].strip().lower() == "true" and r["X_prod"]:
            out.append({k: float(v) for k, v in r.items()
                        if k not in ("reactor", "knob", "term", "path",
                                     "valid", "gas_feasible", "banner_ok")
                        and v not in ("", None)})
    if fam == "wet_gas_flow":
        out = [r for r in out if abs(r["value"] - 0.85) > 1e-9]
    return out


def u0_of(rows):
    """
    Superficial velocity from the absolute molar flow and inlet T
    """
    return [r["abs_flow"] * R_GAS * r["Tg_in"] / (P_ATM * A_LAB)
            for r in rows]


def sweep_order(rows, xs):
    """
    Indices sorted by x
    """
    return sorted(range(len(rows)), key=lambda i: xs[i])


def stages(ax, rows_o, xs, reactor):
    """
    Intermediate chain links as cumulative X along the sorted sweep
    """
    for i, (key, lbl) in enumerate(STAGE_LBL[reactor]):
        xy = []
        for x, r in zip(xs, rows_o):
            if any(r.get(f"w_out_{sp}") in (None, "")
                   for sp in common.FE_SP):
                continue
            st = common.x_stages_from_w(
                reactor, {sp: r[f"w_out_{sp}"] for sp in common.FE_SP})
            if st:
                xy.append((x, st[key]))
        if xy:
            ax.plot([a for a, _ in xy], [b for _, b in xy],
                    color=STAGE_C[i % 2], lw=1.2, label=lbl, zorder=3)


def t2_points(reactor):
    """
    The solid_flow sweep mapped to tau_s, shared by T2 and T2W
    """
    rows_s = valid_rows(f"{reactor}_solid_flow")
    if not rows_s:
        return None
    base, meta = common.load_axial(reactor, "lab")
    us0 = base[0]["u_solid"]
    tau_m = meta["H"] / us0 / 60.0
    tau = [r.get("H", meta["H"]) / (us0 * r["value"]) / 60.0
           for r in rows_s]
    order = sweep_order(rows_s, tau)
    xs = [tau[i] for i in order]
    ro = [rows_s[i] for i in order]
    return xs, ro, tau_m


def t2_axes(ax, reactor, xs, ro, tau_m, stages_on=True):
    """
    Everything T2 and T2W share; product/gas conversion curves,
    and the top axis naming the feed factor behind each tau 
    """
    if stages_on:
        stages(ax, ro, xs, reactor)

    def _fac(v):
        return tau_m / np.maximum(np.asarray(v, dtype=float), 1e-9)
    sec = ax.secondary_xaxis("top", functions=(_fac, _fac))
    sec.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0])
    sec.set_xlabel("Solid flow x base [-]")
    ax.set_xlabel(r"$\tau_s$ [min]")
    ax.grid(True)


def legend_band(fig, ncol=3):

    """
    Figure level legend in the bottom band 
    """
    h, l = fig.axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=ncol, columnspacing=1.2, handlelength=1.8)


def main():
    common.apply_style()

    # T1 per reactor
    for reactor, stem, _ in FIGS:
        rows = valid_rows(f"{reactor}_gas_flow")
        if not rows:
            continue
        u0 = u0_of(rows)
        # tau_g = H/u0 in seconds
        # The superficial residence of the gas over the bed,the
        # gas_flow family sits at the match height, H is fixed
        h_bed = float(rows[0]["H"])
        tau_g = [h_bed / v for v in u0]
        order = sweep_order(rows, tau_g)
        xs = [tau_g[i] for i in order]
        ro = [rows[i] for i in order]

        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        ax.plot(xs, [r["X_prod"] for r in ro], color=common.INK,
                marker="o", ms=4.5, label=ALIAS[reactor])
        gas = [(x, r["X_gas"]) for x, r in zip(xs, ro)
               if r.get("X_gas") is not None]
        if gas:
            ax.plot([a for a, _ in gas], [b for _, b in gas],
                    color=common.INK2, marker="s", ms=4,
                    label=GAS_LBL[reactor])
        stages(ax, ro, xs, reactor)

        def ratio(v):
            return h_bed / (np.maximum(np.asarray(v, dtype=float), 1e-9)
                            * U_MF)
        sec = ax.secondary_xaxis("top", functions=(ratio, ratio))

        sec.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        sec.set_xlabel(r"$u_0/u_{mf}$ [-]")
        ax.set_title(f"{NAME[reactor]}: Conversion vs gas residence time")
        ax.set_xlabel(r"$\tau_g$ [s]")
        ax.set_ylabel(r"X [%]")
        ax.grid(True)
        legend_outside(fig, ax, reactor)
        common.save_fig(fig, stem)
        plt.close(fig)

    # T1W per reactor
    for reactor, stem, _ in FIGS:
        rows = valid_rows(f"{reactor}_gas_flow")
        if not rows:
            continue
        u0 = u0_of(rows)
        # same tau_g axis as the T1
        h_bed = float(rows[0]["H"])
        tau_g = [h_bed / v for v in u0]
        order = sweep_order(rows, tau_g)
        xs = [tau_g[i] for i in order]
        ro = [rows[i] for i in order]

        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        for i, (sp, lbl) in enumerate(CHAIN[reactor]):
            xy = sorted((x, 100.0 * r[f"w_out_{sp}"])
                        for x, r in zip(xs, ro)
                        if r.get(f"w_out_{sp}") not in (None, ""))
            if not xy or max(v for _, v in xy) < 1e-4:
                continue
            ax.plot([a for a, _ in xy], [b for _, b in xy],
                    color=common.COLORS[i], lw=1.2, label=lbl)
        og.add_gas(ax, list(zip(xs, ro)), reactor, "gas_flow")

        def ratio(v):
            return h_bed / (np.maximum(np.asarray(v, dtype=float), 1e-9)
                            * U_MF)
        sec = ax.secondary_xaxis("top", functions=(ratio, ratio))
        sec.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
        sec.set_xlabel(r"$u_0/u_{mf}$ [-]")
        ax.set_title(f"{NAME[reactor]}: outlet composition vs gas "
                     f"residence time")
        ax.set_xlabel(r"$\tau_g$ [s]")
        ax.set_ylabel("Outlet mass fraction [%]")
        legend_band(fig)
        common.save_fig(fig, T1W_STEM[reactor])
        plt.close(fig)

    # T2 per reactor: X vs tau_s along the solid_flow sweep only, in minutes
    # H is not a knob 
    for reactor, _, stem in FIGS:
        pts = t2_points(reactor)
        if not pts:
            continue
        xs, ro, tau_m = pts

        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        ax.plot(xs, [r["X_prod"] for r in ro], color=common.INK,
                marker="o", ms=4.5, label=ALIAS[reactor])
        gas = [(x, r["X_gas"]) for x, r in zip(xs, ro)
               if r.get("X_gas") is not None]
        if gas:
            ax.plot([a for a, _ in gas], [b for _, b in gas],
                    color=common.INK2, marker="s", ms=4,
                    label=GAS_LBL[reactor])
        t2_axes(ax, reactor, xs, ro, tau_m)
        ax.set_title(f"{NAME[reactor]}: Conversion vs solid "
                     f"residence time")
        ax.set_ylabel(r"X [%]")
        legend_outside(fig, ax, reactor)
        common.save_fig(fig, stem)
        plt.close(fig)

    # T2W per reactor, the same points as outlet mass fractions 
    for reactor, _, _ in FIGS:
        pts = t2_points(reactor)
        if not pts:
            continue
        xs, ro, tau_m = pts

        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        drew = set()
        for i, (sp, lbl) in enumerate(CHAIN[reactor]):
            xy = sorted((x, 100.0 * r[f"w_out_{sp}"])
                        for x, r in zip(xs, ro)
                        if r.get(f"w_out_{sp}") not in (None, ""))
            if not xy or max(v for _, v in xy) < 1e-4:
                continue
            ax.plot([a for a, _ in xy], [b for _, b in xy],
                    color=common.COLORS[i], lw=1.2, label=lbl)
            drew.add(i)
        og.add_gas(ax, list(zip(xs, ro)), reactor, "solid_flow")
        t2_axes(ax, reactor, xs, ro, tau_m, stages_on=False)
        ax.set_title(f"{NAME[reactor]}: outlet composition vs solid "
                     f"residence time")
        ax.set_ylabel("Outlet mass fraction [%]")
        legend_band(fig)
        common.save_fig(fig, T2W_STEM[reactor])
        plt.close(fig)


if __name__ == "__main__":
    main()

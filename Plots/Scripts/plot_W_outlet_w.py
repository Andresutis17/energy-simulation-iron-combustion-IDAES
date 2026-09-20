"""
Outlet composition vs the swept knob, one W figure per S figure

"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import outlet_gas as og
from plot_D_sensitivity import (TITLE, load, split, sane_bad)

# Same four species
SPECIES = [("Fe2O3", r"$w_{Fe_2O_3}$"), ("Fe3O4", r"$w_{Fe_3O_4}$"),
           ("FeO", r"$w_{FeO}$"), ("Fe", r"$w_{Fe}$")]

# Chains of the reactions
CHAIN = {"reduction": SPECIES,
         "wet": [s for s in SPECIES if s[0] != "Fe2O3"],
         "dry": [s for s in SPECIES if s[0] in ("Fe2O3", "Fe")]}

# Gamma axis for the oxidation grade
GAMMA_RX = ("reduction", "wet")


def draw(ax, series, reactor, knob):
    """
    One curve per species, solid chain in the identity colors,
    gas species in ink (reactant) and GAS_PROD purple (product) 
    """
    drew = set()
    for i, (sp, lbl) in enumerate(CHAIN[reactor]):
        for pts, ls in series:
            xy = sorted((p[0], 100.0 * float(p[3][f"w_out_{sp}"]))
                        for p in pts
                        if p[3].get(f"w_out_{sp}") not in (None, ""))
            if not xy or max(v for _, v in xy) < 1e-4:
                continue
            mk = ({"marker": "o", "ms": 3.5, "mfc": "none"}
                  if ls != "-" else {})
            ax.plot([a for a, _ in xy], [b for _, b in xy],
                    color=common.COLORS[i], ls=ls, lw=1.2, **mk)
            if i not in drew:
                ax.plot([], [], color=common.COLORS[i], ls="-",
                        label=lbl)
                drew.add(i)
    for j, sp in enumerate(og.GAS_SP[reactor]):
        color = common.INK if j == 0 else og.GAS_PROD
        for pts, ls in series:
            xy = []
            for p in pts:
                w = og.row_gas_ws(reactor, knob, p[3])
                if sp in w:
                    xy.append((p[0], w[sp]))
            xy.sort()
            if not xy or max(v for _, v in xy) < 1e-4:
                continue
            mk = ({"marker": "o", "ms": 3.5, "mfc": "none"}
                  if ls != "-" else {})
            ax.plot([a for a, _ in xy], [b for _, b in xy],
                    color=color, ls=ls, lw=1.2, **mk)
            if ("gas", j) not in drew:
                ax.plot([], [], color=color, ls="-", lw=1.2,
                        label=rf"$w_{{{sp}}}$ (gas)")
                drew.add(("gas", j))


def wfig(reactor, stem, rows_list, title, xlabel, knob, gam=False,
         xlim=None):
    """
     W renderer
    """
    if not any(pts for pts, _ in rows_list):
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    draw(ax, rows_list, reactor, knob)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Outlet mass fraction [%]")
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    if gam and reactor in GAMMA_RX:
        ok_pairs = [(p[0], p[3]) for p in rows_list[0][0]]
        og.add_gamma(ax, ok_pairs, reactor, knob)
        handles.append(Line2D([], [], color="#d95f02", ls="--", lw=1.2,
                              label=r"$\Gamma$"))
    if handles:
        fig.tight_layout(rect=[0, 0.18, 1, 1])

        fig.legend(handles=handles,
                   loc="lower center", bbox_to_anchor=(0.5, 0.015),
                   ncol=4)
    else:
        fig.tight_layout()
    common.save_fig(fig, stem)
    plt.close(fig)


def flagged(bad, max_err=1.0):
    """
    Converged but flagged rows 
    """
    return [p for p in bad if p[3].get("term") == "optimal"
            and p[3].get("err_mass") not in (None, "")
            and float(p[3]["err_mass"]) <= max_err]


def w_t_solid(reactor, stem):
    """
    W1 vs solid inlet T
    """
    rows = load(reactor, "T_solid")
    if not rows:
        return
    ok, bad, _ = split(rows, "value")
    wfig(reactor, stem, [(ok, "-"), (sane_bad(flagged(bad)), ":")],
         f"{TITLE[reactor]}: outlet composition\nvs Solid Inlet Temp",
         r"$T_{solid,in}$ [K]", "T_solid", gam=True)


def w_t_gas(reactor, stem, xlim=None):
    """W8 vs gas inlet T"""
    rows = load(reactor, "T_gas")
    if not rows:
        return
    ok, bad, _ = split(rows, "value")
    fb = flagged(bad, max_err=0.1 if reactor == "dry" else 1.0)
    ok.sort(key=lambda p: p[0])
    wfig(reactor, stem, [(ok, "-"), (sane_bad(fb), ":")],
         f"{TITLE[reactor]}: outlet composition\nvs Gas Inlet Temp",
         r"$T_{gas,in}$ [K]", "T_gas", gam=True, xlim=xlim)


def w_porosity(reactor, stem):
    """W4 vs inlet particle porosity
    """
    rows = load(reactor, "porosity")
    if not rows:
        return
    ok, bad, _ = split(rows, "value")
    if reactor == "dry":
        bad = []
    wfig(reactor, stem, [(ok, "-"), (sane_bad(bad), ":")],
         f"{TITLE[reactor]}: outlet composition\nvs Particle Porosity",
         r"Particle porosity $\varepsilon_p$ [-]", "porosity",
         gam=True, xlim=(0.10, 0.45))


def w_y(reactor, stem, knob, sp_label, xlim):
    """
    W2/W3/W9: vs feed composition
    """
    rows = load(reactor, knob)
    if not rows:
        return
    ok, bad, _ = split(rows, "value")
    if knob == "y_O2":
        ok = ok + [p for p in bad
                   if float(p[3]["T_solid_out"]) < 873.0]
        ok.sort(key=lambda p: p[0])
        bad = []
    wfig(reactor, stem, [(ok, "-"), (sane_bad(bad), ":")],
         f"{TITLE[reactor]}: outlet composition\nvs Feed {sp_label}",
         f"Feed {sp_label} [-]", knob, gam=True, xlim=xlim)


def main():
    """
    All twelve, in the S numbering
    """
    common.apply_style()
    w_t_solid("reduction", "W1a_sens_T_reduction")
    w_t_solid("wet", "W1b_sens_T_wet")
    w_t_solid("dry", "W1c_sens_T_dry")
    w_y("reduction", "W2_sens_yH2_reduction", "y_H2",
        r"$y_{H_2}$", (0.0, 1.03))
    w_y("dry", "W3_sens_yO2_dry", "y_O2",
        r"$y_{O_2}$", (0.03, 0.23))
    w_y("wet", "W9_sens_yH2O_wet", "y_H2O",
        r"$y_{H_2O}$", (0.0, 1.03))
    w_porosity("dry", "W4a_sens_porosity_dry")
    w_porosity("wet", "W4b_sens_porosity_wet")
    w_porosity("reduction", "W4c_sens_porosity_reduction")
    w_t_gas("reduction", "W8a_sens_Tgas_reduction")
    w_t_gas("wet", "W8b_sens_Tgas_wet")
    w_t_gas("dry", "W8c_sens_Tgas_dry")


if __name__ == "__main__":
    main()

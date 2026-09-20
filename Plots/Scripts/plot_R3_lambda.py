"""
Outlet composition vs the stoichiometric ratio plots
"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import outlet_gas as og
from plot_D_sensitivity import load, split, sane_bad

M_S = 6.901775147928993e-4   # lab solid feed [kg/s]
F0 = {"reduction": 0.02070532544378698, "wet": 0.02070532544378698,
      "dry": 0.06625704142011833}          # total gas feed [mol/s]
NU = {"reduction": 1.5, "wet": 4.0 / 3.0, "dry": 0.75}   # gas per Fe
N_FE = {"reduction": 8.64397e-3, "wet": 1.23588e-2,
        "dry": 1.23588e-2}                 # Fe feed [mol/s]
KNOW_Y = {"reduction": "y_H2", "wet": "y_H2O", "dry": "y_O2"}
BASE_Y = {"reduction": 0.99, "wet": 0.99, "dry": 0.21}

FIGS = [("reduction", "R3Wa"), ("wet", "R3Wb"), ("dry", "R3Wc")]
NAME = {"reduction": "Reduction", "wet": "Wet Oxidation",
        "dry": "Dry Oxidation"}
SPECIES = [("Fe2O3", r"$w_{Fe_2O_3}$"), ("Fe3O4", r"$w_{Fe_3O_4}$"),
           ("FeO", r"$w_{FeO}$"), ("Fe", r"$w_{Fe}$")]
CHAIN = {"reduction": SPECIES,
         "wet": [s for s in SPECIES if s[0] != "Fe2O3"],
         "dry": [s for s in SPECIES if s[0] in ("Fe2O3", "Fe")]}

# The chains product species carries the star(match point)
PROD = {"reduction": "Fe", "wet": "Fe3O4", "dry": "Fe2O3"}


def lam(reactor, y_in):
    """
    stoichiometric ratio of a feed composition
    """
    return F0[reactor] * y_in / (NU[reactor] * N_FE[reactor])


def rows_lam(reactor):
    """
    The y knob family as (lambda, row) pairs
    """
    rows = load(reactor, KNOW_Y[reactor])
    if not rows:
        return None
    ok, bad, _ = split(rows, "value")
    if reactor == "dry":
        ok = ok + [p for p in bad
                   if float(p[3]["T_solid_out"]) < 873.0]
        ok.sort(key=lambda p: p[0])
        bad = []
    pairs = [(lam(reactor, p[0]), p) for p in ok]
    flags = [(lam(reactor, p[0]), p) for p in sane_bad(bad)]
    return pairs, flags, rows


def star_lam(ax, pairs, reactor, y_of):
    """
    The base case star at its lambda, on the y of curve
    """
    for _, p in pairs:
        if p[3].get("X_solid") \
                and abs(float(p[3]["value"]) - BASE_Y[reactor]) < 1e-6:
            ax.plot([lam(reactor, BASE_Y[reactor])], [y_of(p)],
                    marker="*", ms=11, color=common.INK, ls="none",
                    mec="white", mew=0.6, zorder=5)
            return


def lam_axis(ax):
    """
    The shared lambda vertical line, dotted exactly stoichiometric line
    """
    ax.axvline(1.0, color=common.MUTED, ls=":", lw=1.0, zorder=1)


def legend_band(fig):
    """Legend in the bottom band"""
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=4,
               columnspacing=1.2, handlelength=1.8)


def r3(reactor, stem_w):
    """The R3W figure of one reactor from its (lambda, row) pairs
    """
    got = rows_lam(reactor)
    if not got:
        return
    pairs, flags, _ = got

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    drew = False
    for i, (sp, lbl) in enumerate(CHAIN[reactor]):
        col = common.COLORS[i]
        val = lambda p, sp=sp: p[3].get(f"w_out_{sp}")
        xy = sorted((x, 100.0 * float(val(p)))
                    for x, p in pairs if val(p) not in (None, ""))
        fxy = sorted((x, 100.0 * float(val(p)))
                     for x, p in flags if val(p) not in (None, ""))
        if not xy or max(v for _, v in xy) < 1e-4:
            continue
        ax.plot([a for a, _ in xy], [b for _, b in xy], color=col,
                lw=1.2, label=lbl)
        if fxy:
            ax.plot([a for a, _ in fxy], [b for _, b in fxy], color=col,
                    marker="o", ms=3.5, mfc="none", ls="none")
        drew = True
        if sp == PROD[reactor]:
            star_lam(ax, pairs, reactor,
                     lambda p, sp=sp: 100.0 * float(p[3][f"w_out_{sp}"]))
    for j, sp in enumerate(og.GAS_SP[reactor]):
        color = common.INK if j == 0 else og.GAS_PROD
        xy = sorted((x, og.row_gas_ws(reactor, KNOW_Y[reactor],
                                      p[3])[sp])
                    for x, p in pairs
                    if p[3].get("X_gas") not in (None, ""))
        if not xy or max(v for _, v in xy) < 1e-4:
            continue
        ax.plot([a for a, _ in xy], [b for _, b in xy],
                color=color, lw=1.2, label=rf"$w_{{{sp}}}$ (gas)")
        drew = True
    lam_axis(ax)
    ax.set_title(f"{NAME[reactor]}: outlet composition\n"
                 f"vs stoichiometric ratio")
    ax.set_xlabel(r"$\lambda$ (stoichiometric ratio)")
    ax.set_ylabel("Outlet mass fraction [%]")
    ax.grid(True)
    if drew:
        legend_band(fig)
    else:
        fig.tight_layout()
    common.save_fig(fig, stem_w)
    plt.close(fig)


def main():
    common.apply_style()
    for reactor, stem_w in FIGS:
        r3(reactor, stem_w)


if __name__ == "__main__":
    for _r, _want in (("reduction", 1.5809), ("wet", 1.2439),
                      ("dry", 1.5011)):
        _got = lam(_r, BASE_Y[_r])
        assert abs(_got - _want) < 5e-4, (_r, _got)
    main()

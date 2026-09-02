
"""
Plots for sensitivity analysis.

Reads the csv files and plots one figure per swept knob around each lab match point

"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

# Display names and axis labels
TITLE = {"reduction": "Reduction", "wet": "Wet Oxidation",
         "dry": "Dry Oxidation"}

# y axis in product basis, conversion to the last link of each chain
XS_LBL = {"reduction": r"$X_{Fe}$ [%]", "wet": r"$X_{Fe_3O_4}$ [%]",
          "dry": r"$X_{Fe_2O_3}$ [%]"}
XS_LBL_SHORT = {"reduction": r"$X_{Fe}$", "wet": r"$X_{Fe_3O_4}$",
                "dry": r"$X_{Fe_2O_3}$"}
XG_LBL = {"reduction": r"$X_{H_2}$ [%]", "wet": r"$X_{H_2O}$ [%]",
          "dry": r"$X_{O_2}$ [%]"}

# Each familys match point knob value, that row gets the "star"
BASE = {("reduction", "T_solid"): 1173.9, ("wet", "T_solid"): 1173.9,
        ("dry", "T_solid"): 1073, ("reduction", "y_H2"): 0.99,
        ("dry", "y_O2"): 0.21, ("reduction", "porosity"): 0.27,
        ("wet", "porosity"): 0.27, ("dry", "porosity"): 0.27,
        ("reduction", "flow"): 1.0, ("wet", "flow"): 1.0, ("dry", "flow"): 1.0,
        ("reduction", "T_gas"): 1050.0, ("wet", "T_gas"): 1050.0,
        ("dry", "T_gas"): 600.0, ("wet", "y_H2O"): 0.99}


def load(reactor, knob):
    """
    One familys CSV. Wet recomputes X_prod from the w_out_ columns, last component basis. 
    The frozen sweeps get re run into _fe34 copies
    """
    name = f"sens_{reactor}_{knob}_fe34.csv" if reactor == "wet" \
        else f"sens_{reactor}_{knob}.csv"
    if not os.path.exists(os.path.join(common.DATA, name)):
        if reactor == "wet" and os.path.exists(
                os.path.join(common.DATA, f"sens_wet_{knob}.csv")):
            name = f"sens_wet_{knob}.csv"  
        else:
            print(f"  missing {name}")
            return None
    rows = common.load_csv(name)
    if reactor == "wet":
        if "w_out_Fe3O4" not in rows[0]:
            print(f"  {name}: no outlet composition")
            return None
        for r in rows:
            try:
                r["X_prod"] = common.x_prod_from_w(
                    "wet", {k: float(r[f"w_out_{k}"]) for k in common.FE_SP})
            except (TypeError, ValueError):
                pass  
    return rows


def ykey(rows):
    """
    Picks the y column, the product one when the CSV has it, X_solid
    when not
    """

    return "X_prod" if rows and "X_prod" in rows[0] else "X_solid"


def split(rows, xkey):

    """
    Classifies the rows. ok = valid, bad = converged but
    flagged, stubs = died 

    """
    ok, bad, stubs = [], [], 0
    y_col = ykey(rows)
    for row in rows:
        if not row.get("X_solid"):
            stubs += 1
            continue
        pt = (float(row[xkey]), float(row[y_col]),
              float(row["X_gas"]), row)
        (ok if row.get("valid") == "True" else bad).append(pt)
    ok.sort(key=lambda p: p[0])
    bad.sort(key=lambda p: p[0])
    return ok, bad, stubs



def star(ax, rows, xkey, base):

    """
    Star the match point row of a family
    """
    y_col = ykey(rows)
    for row in rows:
        if row.get("X_solid") and abs(float(row["value"]) - base) < 1e-6:
            ax.plot([float(row[xkey])], [float(row[y_col])], marker="*",
                    ms=11, color=common.INK, ls="none",
                    mec="white", mew=0.6, zorder=5)
            return True
    return False


def two_series(ax, ok, bad, label_s, label_g):

    """
    The two conversion curves. X_solid circles, X_gas squares
    """

    if ok:
        ax.plot([p[0] for p in ok], [p[1] for p in ok], color=common.COLORS[0],
                marker="o", ms=4.5, label=label_s)
        ax.plot([p[0] for p in ok], [p[2] for p in ok], color=common.COLORS[1],
                marker="s", ms=4, label=label_g)
    if bad:  
        ax.plot([p[0] for p in bad], [p[1] for p in bad],
                color=common.COLORS[0], marker="o", ms=4.5, mfc="none",
                ls="none")
        ax.plot([p[0] for p in bad], [p[2] for p in bad],
                color=common.COLORS[1], marker="s", ms=4, mfc="none",
                ls="none")


def legend_below(ax):
    """
    Legend under the axes 
    """
    if ax.get_legend_handles_labels()[0]: 
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2)


def guard(ok, bad, tag):

    """
    A family whose every step died, so no plot
    """
    if not ok and not bad:
        print(f" {tag}: no points to plot")
        return False
    return True


def sane_bad(bad, x_is_T=False):

    """
    Drop flagged points whose numbers are too extreme to plot
    """

    keep = [pt for pt in bad
            if -100 <= pt[1] <= 200 and -100 <= pt[2] <= 200
            and (not x_is_T or 300 <= pt[0] <= 1600)]
    return keep


def fig_s1(reactor, stem):

    """
    Conversion vs solid inlet temperature for one reactor
    """

    rows = load(reactor, "T_solid")
    if not rows:
        return
    ok, bad, _ = split(rows, "value")

    # Draw a flagged point only if the solver really converged 
    bad = [p for p in bad if p[3].get("term") == "optimal"
           and float(p[3]["err_mass"]) <= 1.0]
    if not guard(ok, bad, f"{reactor} T_solid"):
        return
    bad = sane_bad(bad)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    two_series(ax, ok, bad, XS_LBL_SHORT[reactor], XG_LBL[reactor])
    star(ax, rows, "value", BASE[(reactor, "T_solid")])
    for p in ok:
        ax.annotate(f"{float(p[3]['T_solid_out']):.0f}", xy=(p[0], p[1]),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=6, color=common.MUTED)
    ax.set_title(f"{TITLE[reactor]}: {XS_LBL_SHORT[reactor]} vs Solid Inlet Temp\n"
                 r"(Point labels: Achieved bed T$_{out}$)")
    ax.set_xlabel(r"$T_{solid,in}$ [K]")
    ax.set_ylabel(XS_LBL[reactor])
    ax.grid(True)
    legend_below(ax)
    fig.tight_layout()
    common.save_fig(fig, f"{stem}_sens_T_{reactor}")
    plt.close(fig)


def fig_s2():

    """
    Conversion vs feed H2 for the reduction reactor
    """


    rows = load("reduction", "y_H2")
    if not rows:
        return
    ok, bad, _ = split(rows, "value")
    if not guard(ok, bad, "reduction y_H2"):
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.set_xlim(0.0, 1.03)
    bad = sane_bad(bad)
    two_series(ax, ok, bad, XS_LBL_SHORT["reduction"], XG_LBL["reduction"])
    star(ax, rows, "value", 0.99)
    ax.set_title(f"{TITLE['reduction']}: {XS_LBL_SHORT['reduction']} vs Feed $y_{{H_2}}$")
    ax.set_xlabel(r"Feed $y_{H_2}$ [-]")
    ax.set_ylabel(XS_LBL["reduction"])
    ax.grid(True)
    legend_below(ax)
    fig.tight_layout()
    common.save_fig(fig, "S2_sens_yH2_reduction")
    plt.close(fig)


def fig_s3():

    """
    Conversion vs feed O2 for the dry reactor
    """

    rows = load("dry", "y_O2")
    if not rows:
        return
    ok, bad, _ = split(rows, "value")
    ok += [p for p in bad if float(p[3]["T_solid_out"]) < 873.0]
    ok.sort(key=lambda p: p[0])
    bad = []
    if not guard(ok, bad, "dry y_O2"):
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.set_xlim(0.03, 0.23)
    two_series(ax, ok, bad, XS_LBL_SHORT["dry"], XG_LBL["dry"])
    star(ax, rows, "value", 0.21)
    ax.set_title(f"{TITLE['dry']}: {XS_LBL_SHORT['dry']} vs Feed $y_{{O_2}}$")
    ax.set_xlabel(r"Feed $y_{O_2}$ [-]")
    ax.set_ylabel(XS_LBL["dry"])
    ax.grid(True)
    legend_below(ax)
    fig.tight_layout()
    common.save_fig(fig, "S3_sens_yO2_dry")
    plt.close(fig)


def fig_s9():
    """
    Conversion vs feed H2O for the wet reactor
    """

    rows = load("wet", "y_H2O")
    if not rows:
        return
    ok, bad, _ = split(rows, "value")
    if not guard(ok, bad, "wet y_H2O"):
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.set_xlim(0.0, 1.03)
    bad = sane_bad(bad)
    two_series(ax, ok, bad, XS_LBL_SHORT["wet"], XG_LBL["wet"])
    star(ax, rows, "value", BASE[("wet", "y_H2O")])
    ax.set_title(f"{TITLE['wet']}: {XS_LBL_SHORT['wet']} vs Feed $y_{{H_2O}}$")
    ax.set_xlabel(r"Feed $y_{H_2O}$ [-]")
    ax.set_ylabel(XS_LBL["wet"])
    ax.grid(True)
    legend_below(ax)
    fig.tight_layout()
    common.save_fig(fig, "S9_sens_yH2O_wet")
    plt.close(fig)


def fig_s4(reactor, stem):

    """
    Conversion vs particle porosity for one reactor
    """

    rows = load(reactor, "porosity")
    if not rows:
        return
    ok, bad, _ = split(rows, "value")
    if reactor == "dry":
        
        bad = []
    if not guard(ok, bad, f"{reactor} porosity"):
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.set_xlim(0.10, 0.45)
    bad = sane_bad(bad)
    two_series(ax, ok, bad, XS_LBL_SHORT[reactor], XG_LBL[reactor])
    star(ax, rows, "value", 0.27)
    ax.set_title(f"{TITLE[reactor]}: {XS_LBL_SHORT[reactor]} vs Particle Porosity")
    ax.set_xlabel(r"Particle porosity $\varepsilon_p$ [-]")
    ax.set_ylabel(XS_LBL[reactor])
    ax.grid(True)
    legend_below(ax)
    fig.tight_layout()
    common.save_fig(fig, f"{stem}_sens_porosity_{reactor}")
    plt.close(fig)


def fig_s5(reactor, stem):
    """
    Conversion vs gas and solid flow for one reactor
    """

    rows_g = load(reactor, "gas_flow")
    rows_s = load(reactor, "solid_flow")
    if not rows_g or not rows_s:
        return
    ok_g, bad_g, _ = split(rows_g, "value")
    ok_s, bad_s, _ = split(rows_s, "value")
    if reactor == "wet":
        
        bad_g, bad_s = [], []
    elif reactor == "dry":
        ok_g += [p for p in bad_g if 0.5 < p[0] < 1.1]
        ok_g.sort(key=lambda p: p[0])
        bad_g = []
        
        ok_s += [p for p in bad_s if p[0] > 0.5]
        ok_s.sort(key=lambda p: p[0])
        bad_s = []
    if not (ok_g or bad_g or ok_s or bad_s):
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    if ok_g:
        ax.plot([p[0] for p in ok_g], [p[1] for p in ok_g],
                color=common.COLORS[0], marker="o", ms=4.5,
                label="Gas flow x")
    if ok_s:
        ax.plot([p[0] for p in ok_s], [p[1] for p in ok_s],
                color=common.COLORS[2], ls=common.LSTYLES[1], marker="s",
                ms=4, label="Solid flow x")
    
    for bad, col, mk in ((bad_g, common.COLORS[0], "o"),
                         (bad_s, common.COLORS[2], "s")):
        bad = sane_bad(bad)
        if bad:
            ax.plot([p[0] for p in bad], [p[1] for p in bad],
                    color=col, marker=mk, ms=4.5, mfc="none", ls="none")
    if not star(ax, rows_g, "value", 1.0):
        star(ax, rows_s, "value", 1.0)  

    ax.set_xlim(0.42, 2.4)
    ax.set_xticks([0.5, 0.75, 1.0, 1.5, 2.0])
    ax.set_title(f"{TITLE[reactor]}: {XS_LBL_SHORT[reactor]} vs Flow Factor")
    ax.set_xlabel("Flow factor x base [-]")
    ax.set_ylabel(XS_LBL[reactor])
    ax.grid(True)
    legend_below(ax)
    fig.tight_layout()
    common.save_fig(fig, f"{stem}_sens_flow_{reactor}")
    plt.close(fig)


def fig_s8(reactor, stem):
    """
    Conversion vs gas inlet temperature for one reactor
    """

    rows = load(reactor, "T_gas")
    if not rows:
        return
    ok, bad, _ = split(rows, "value")
    bad = [p for p in bad if p[3].get("term") == "optimal"
           and float(p[3]["err_mass"]) <= 1.0]
    if reactor == "dry":
        bad = [p for p in bad if float(p[3]["err_mass"]) <= 0.1]
    if not guard(ok, bad, f"{reactor} T_gas"):
        return
    bad = sane_bad(bad)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    if reactor == "dry":
        allp = sorted(ok + bad, key=lambda p: p[0])
        ax.plot([p[0] for p in allp], [p[1] for p in allp],
                color=common.COLORS[0], marker="o", ms=4.5,
                label=XS_LBL_SHORT["dry"])
        ax.plot([p[0] for p in allp], [p[2] for p in allp],
                color=common.COLORS[1], marker="s", ms=4,
                label=XG_LBL["dry"])
    else:
        two_series(ax, ok, bad, XS_LBL_SHORT[reactor], XG_LBL[reactor])
    star(ax, rows, "value", BASE[(reactor, "T_gas")])
    for p in ok:
        ax.annotate(f"{float(p[3]['T_solid_out']):.0f}", xy=(p[0], p[1]),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=6, color=common.MUTED)
    ax.set_title(f"{TITLE[reactor]}: {XS_LBL_SHORT[reactor]} vs Gas Inlet Temp\n"
                 r"(Point labels: Achieved bed T$_{out}$)")
    ax.set_xlabel(r"$T_{gas,in}$ [K]")
    ax.set_ylabel(XS_LBL[reactor])
    ax.grid(True)
    legend_below(ax)
    fig.tight_layout()
    common.save_fig(fig, f"{stem}_sens_Tgas_{reactor}")
    plt.close(fig)


def main():

    """
    Renders every figure
    """
    common.apply_style()
    fig_s1("reduction", "S1a")
    fig_s1("wet", "S1b")
    fig_s1("dry", "S1c")
    fig_s2()
    fig_s3()
    fig_s9()
    fig_s4("dry", "S4a")
    fig_s4("wet", "S4b")
    fig_s4("reduction", "S4c")
    fig_s5("reduction", "S5a")
    fig_s5("wet", "S5b")
    fig_s5("dry", "S5c")
    fig_s8("reduction", "S8a")
    fig_s8("wet", "S8b")
    fig_s8("dry", "S8c")


if __name__ == "__main__":
    main()

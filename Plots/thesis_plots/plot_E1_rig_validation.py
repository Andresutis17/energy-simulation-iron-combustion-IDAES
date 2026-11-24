"""
Plots for the validation of the lab reactor data
"""
import json
import os

import matplotlib.pyplot as plt

from common import (COLORS, DATA, INK, MUTED, apply_style, load_csv,
                    save_fig)


O2_EXP = 0.150  #   0.150 Nm3/h measured off gas O2 
O2_KL = 0.2013  #   0.2013 Nm3/h  model data


def main():
    apply_style()
    rows = load_csv("axial_dry_rig.csv")
    with open(os.path.join(DATA, "axial_dry_rig.meta.json")) as f:
        meta = json.load(f)

    xh = [float(r["x_norm"]) for r in rows]
    dbD = [float(r["db_over_D"]) for r in rows]
    db_top_mm = float(meta["db_top_mm"])
    db_top_ratio = db_top_mm / (float(meta["D_m"]) * 1e3)

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.9),
                                   gridspec_kw={"wspace": 0.45})

    # (a) Bubble diameter of the lab (simulation) bed over the column diameter
    axa.plot(xh, dbD, color=COLORS[0])
    axa.axhline(1.0, color=MUTED, ls="--", lw=1.0)
    axa.text(0.03, 1.06, r"$d_b = D$", color=MUTED, fontsize=7.5)
    cross = None

    # Height where db/D crosses 1.0, by linear interpolation
    for i in range(1, len(xh)):
        if dbD[i - 1] < 1.0 <= dbD[i]:
            f = (1.0 - dbD[i - 1]) / (dbD[i] - dbD[i - 1])
            cross = xh[i - 1] + f * (xh[i] - xh[i - 1])
            break
    if cross is not None:
        axa.plot([cross], [1.0], "o", ms=4.5, color=COLORS[0], mfc="white",
                 zorder=3)
    axa.text(0.97, 1.13,
             f"105 mm ($d_b/D = {db_top_ratio:.2f}$)",
             color=INK, fontsize=7.5, ha="right")
    axa.set_xlim(0.0, 1.0)
    axa.set_ylim(0.0, 2.15)
    axa.set_xlabel(r"$z/H$")
    axa.set_ylabel(r"$d_b/D$")
    axa.set_title("(a) Predicted bubble to column diameter ratio",
                  fontsize=9)

    # (b) The directly measured data vs the model at the same rig
    bars = [O2_EXP, O2_KL]
    names = ["Experiment", "Model"]
    texts = ["0.150", "0.2013"]
    axb.bar(names, bars, width=0.5, color=[INK, COLORS[0]], zorder=2)
    for x, v, t in zip([0, 1], bars, texts):
        axb.text(x, v + 0.005, t, ha="center", fontsize=8, color=INK)
    axb.set_ylim(0.0, 0.24)
    axb.set_ylabel(r"$\dot{V}_{\mathrm{O_2,out}}$ [Nm$^3\,$h$^{-1}$]")
    axb.set_title("(b) Experimental and predicted outlet O$_2$ flow",
                  fontsize=9)

    save_fig(fig, "E1_rig_validation")


if __name__ == "__main__":
    main()

"""
Dry bubble diameter along the bed at three bed states
with the emulsion O2. Appendix

"""
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

STATES = [
    ("_r8", common.INK, "Baseline, 600 K feed ($T_{s,out}$ 1083 K)"),
    ("_dp028_tg1223", "#FF0D00", "1223 K feed ($T_{s,out}$ 1467 K)"),
    ("_dp028_g155", "#006df2", r"$g = 1.55$, 600 K feed ($T_{s,out}$ 836 K)"),
]


def main():
    common.apply_style()
    fig, axes = common.stack(2)
    handles = []

    ax = axes[0]
    for sfx, color, _ in STATES:
        rows, meta = common.load_axial("dry", "lab", suffix=sfx)
        if not meta.get("valid", True):
            continue
        z = [r["z"] for r in rows]
        db = [1000.0 * r["db"] for r in rows]
        ax.plot(z, db, color=color, lw=1.2)
    ax.set_title("(a) Bubble diameter")
    ax.set_ylabel(r"Bubble diameter $d_b$ [mm]")
    ax.grid(True)

    ax = axes[1]
    handles = []
    for sfx, color, lbl in STATES:
        try:
            rows, meta = common.load_axial("dry", "lab", suffix=sfx)
        except FileNotFoundError:
            continue
        if not meta.get("valid", True):
            continue
        z = [r["z"] for r in rows]
        ye = [100.0 * r["y_emul_O2"] for r in rows]
        yb = [100.0 * r["y_bub_O2"] for r in rows]
        ax.plot(z, ye, color=color, lw=1.2)
        ax.plot(z, yb, color=color, lw=1.0, ls="--")
        base_lbl = lbl.split(" (")[0]
        handles.append(Line2D([], [], color=color, lw=1.2,
                              label=f"{base_lbl}, emulsion"))
        handles.append(Line2D([], [], color=color, lw=1.0, ls="--",
                              label=f"{base_lbl}, bubble"))
    ax.set_title("(b) Oxygen in the two lanes")
    ax.set_ylabel(r"O$_2$ mole fraction [%]")
    ax.set_xlabel("Bed height $z$ [m]")
    ax.grid(True)

    common.legend_band(fig, ncol=2, handles=handles, bottom=0.17)
    common.save_fig(fig, "E3_dry_bubbles")
    plt.close(fig)


if __name__ == "__main__":
    main()

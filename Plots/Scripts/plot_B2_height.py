"""
Bed-height sweeps, one PNG per lab reactor
"""

import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from plot_D_sensitivity import (TITLE, XS_LBL, XS_LBL_SHORT,
                                load, split, sane_bad)

# One row per reactor: name, prefix,  axis limits
# A bit wider than the data so the points do not touch the edges
PANELS = [
    ("reduction", "B2a", (0.45, 1.06)),
    ("wet", "B2b", (0.45, 1.06)),
    ("dry", "B2c", (0.45, 1.06)),
]



def fig_b2(reactor, stem, xlim):
    """
    Product conversion vs bed height for one reactor
    """
    rows = load(reactor, "H")
    if not rows:
        return
    ok, bad, _ = split(rows, "value")
    
    drawn = [p for p in bad if p[3].get("term") == "optimal"
             and p[3].get("err_mass") not in (None, "")
             and float(p[3]["err_mass"]) <= 0.1]
    bad = drawn
    if not ok and not bad:
        return
    bad = sane_bad(bad)

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    if ok:
        ax.plot([p[0] for p in ok], [p[1] for p in ok],
                color=common.COLORS[0], marker="o", ms=4.5,
                label=XS_LBL_SHORT[reactor])
    if bad:
        ax.plot([p[0] for p in bad], [p[1] for p in bad],
                color=common.COLORS[0], marker="o", ms=4.5, mfc="none",
                ls="none")
    # Star at the match height row
    h_match = common.LAB_MATCH[reactor]["H"]
    starred = False
    for row in rows:
        if row.get("X_prod") and abs(float(row["value"]) - h_match) < 1e-6 \
                and row.get("valid") == "True":
            ax.plot([float(row["value"])], [float(row["X_prod"])],
                    marker="*", ms=11, color=common.COLORS[1], ls="none",
                    mec="white", mew=0.6, zorder=5)
            starred = True

    ax.set_xlim(*xlim)
    ax.set_title(f"{TITLE[reactor]}: {XS_LBL_SHORT[reactor]} vs bed height")

    ax.set_xlabel("Bed height")
    ax.set_ylabel(XS_LBL[reactor])
    ax.grid(True)
    if ok:  
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    common.save_fig(fig, f"{stem}_height_{reactor}")
    plt.close(fig)


def main():
    """
    Three panels, one per reactor
    """
    common.apply_style()
    for reactor, stem, xlim in PANELS:
        fig_b2(reactor, stem, xlim)


if __name__ == "__main__":
    main()

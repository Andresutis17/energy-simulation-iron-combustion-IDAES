"""

Dry O2 utilization with and without pressure compensation,
plotted against T_gas,in. Appendix

"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

# The shared starting point of both sweeps. T_gas,in and its O2 utilization 
ANCHOR = (600.0, 13.848171836228552)

# The same anchors bed temperature, T_s,out [K]
ANCHOR_TS = 1083.2356736149798


def load_uncomp():

    """The uncompensated sweep. T_gas,in rises, pressure stays constant """
    
    # u: utilization
    # t: T_s,out
    u, t = [], []
    with open(os.path.join(DATA, "sens_dry_T_gas_operating.csv"),
              encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("valid") == "True":
                u.append((float(r["value"]), float(r["X_gas"])))
                t.append((float(r["value"]), float(r["T_solid_out"])))
    u.sort(); t.sort()
    return ([p[0] for p in u], [p[1] for p in u],
            [p[0] for p in t], [p[1] for p in t])


def load_comp():

    """The compensated sweep. P and T_gas,in rise together"""
    # Dicts drop the repeated solves
    # same T solved twice keeps only its latest result
    seen_u, seen_t = {}, {}
    with open(os.path.join(DATA, "sens_dry_Tg_Pcomp_operating.csv"),
              encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("valid") == "True":
                key = round(float(r["value"]), 6)
                seen_u[key] = float(r["X_gas"])
                seen_t[key] = float(r["T_solid_out"])
    xs = sorted(seen_u)
    return xs, [seen_u[x] for x in xs], xs, [seen_t[x] for x in xs]


def main():
    common.apply_style()
    fig, ax = plt.subplots(figsize=(6.0, 3.3))

    x1, y1, t1, Ts1 = load_uncomp()
    x2, y2, t2, Ts2 = load_comp()

    l1, = ax.plot(x1, y1, "-o", color="#FF0D00", ms=3.5, lw=1.4,
                  label="Utilization, uncompensated")
    l2, = ax.plot(x2, y2, "-o", color="#006df2", ms=3.5, lw=1.4,
                  label=r"Utilization, $P\propto T$")
    star, = ax.plot([ANCHOR[0]], [ANCHOR[1]], marker="*", ms=13,
                    color=common.INK, mec="white", mew=0.6, ls="none",
                    label="Shared anchor")
    ax.set_xlabel(r"$T_{gas,in}$ [K]")
    ax.set_ylabel(r"$O_2$ utilization [%]")
    ax.set_ylim(8, 20)
    ax.grid(True)

    ax2 = ax.twinx()
    d1, = ax2.plot(t1, Ts1, "--", color="#FF0D00", lw=1.6, alpha=0.9,
                   label=r"$T_{s,out}$, uncompensated")
    d2, = ax2.plot(t2, Ts2, "--", color="#006df2", lw=1.6, alpha=0.9,
                   label=r"$T_{s,out}$, $P\propto T$")
    ax2.set_ylabel(r"$T_{s,out}$ [K]")
    ax2.set_ylim(1000, 1650)
    ax2.grid(False)

    common.legend_band(fig, ncol=2,
                       handles=[l1, l2, d1, d2, star], bottom=0.30)

    common.save_fig(fig, "E2_dry_T_investigation")
    plt.close(fig)


if __name__ == "__main__":
    main()

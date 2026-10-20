"""

Wet stoichiometry/composition at fixed throughput,
plotted against lambda_steam

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from plot_R4_operating_reduction import num, operating_figure, valid_rows

CSV_NAME = "sens_wet_y_H2O_operating.csv"


def main():
    common.apply_style()
    rows = common.load_csv(CSV_NAME)
    ok = valid_rows(rows)
    ok.sort(key=lambda r: num(r, "lambda_steam"))
    operating_figure(ok, "lambda_steam", 1.2439,
                     "Baseline operating point",
                     r"$\lambda_{steam}$ (via steam mole fraction)",
                     "W4b_operating_yH2O",
                     gamma=True,
                     vlines=[(1.0, r"Stoichiometric feed, "
                                   r"$\lambda_{steam}=1$")],
                     reactor="wet")


if __name__ == "__main__":
    main()

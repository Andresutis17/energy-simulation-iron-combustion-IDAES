"""

Dry stoichiometry/composition at fixed throughput,
plotted against lambda_O2

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from plot_R4_operating_reduction import num, operating_figure, valid_rows

CSV_NAME = "sens_dry_y_O2_operating.csv"


def main():
    common.apply_style()
    rows = common.load_csv(CSV_NAME)
    ok = valid_rows(rows)
    ok.sort(key=lambda r: num(r, "lambda_O2"))
    operating_figure(ok, "lambda_O2", 1.5011,
                     "Baseline operating point",
                     r"$\lambda_{O_2}$ (via O$_2$ mole fraction)",
                     "D4b_operating_yO2", reactor="dry",
                     vlines=[(1.0, r"Stoichiometric feed, "
                                   r"$\lambda_{O_2}=1$")])


if __name__ == "__main__":
    main()

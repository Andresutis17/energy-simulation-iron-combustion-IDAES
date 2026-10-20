"""

Reduction pressure sweep as the operating multiplot,
both feeds scaled with the bed pressure

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from plot_R4_operating_reduction import operating_figure, valid_rows, num

CSV_NAME = "sens_reduction_p_sweep_operating.csv"


def main():
    common.apply_style()
    rows = common.load_csv(CSV_NAME)
    ok = valid_rows(rows)
    ok.sort(key=lambda r: num(r, "value"))
    operating_figure(ok, "value", 1.0,
                     "Baseline operating point, 1 bar",
                     r"Bed pressure $P$ [bar]",
                     "P_sweep_reduction",
                     reactor="reduction",
                     gamma=True,
                     lam_note=r"$\lambda_{H_2}=3$")


if __name__ == "__main__":
    main()

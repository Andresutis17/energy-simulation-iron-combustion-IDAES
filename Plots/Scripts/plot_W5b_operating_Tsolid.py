"""

Wet operating multiplot vs T_solid,in

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from plot_R4_operating_reduction import num, operating_figure, valid_rows

CSV_NAME = "sens_wet_T_solid_operating.csv"


def main():
    common.apply_style()
    rows = common.load_csv(CSV_NAME)
    ok = valid_rows(rows)
    ok.sort(key=lambda r: num(r, "value"))
    operating_figure(ok, "value", 1173.9, "Baseline operating point",
                     r"$T_{solid,in}$ [K]", "W5b_operating_Tsolid",
                     reactor="wet", gamma=True,
                     lam_note=r"$\lambda_{steam}=1.24$")


if __name__ == "__main__":
    main()

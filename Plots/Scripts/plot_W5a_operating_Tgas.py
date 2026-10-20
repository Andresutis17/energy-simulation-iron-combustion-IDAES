"""

Wet operating multiplot vs T_gas,in

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from plot_R4_operating_reduction import num, operating_figure, valid_rows

CSV_NAME = "sens_wet_T_gas_operating.csv"


def main():
    common.apply_style()
    rows = common.load_csv(CSV_NAME)
    ok = valid_rows(rows)
    ok.sort(key=lambda r: num(r, "value"))
    operating_figure(ok, "value", 1050.0, "Baseline operating point",
                     r"$T_{gas,in}$ [K]", "W5a_operating_Tgas",
                     reactor="wet", gamma=True,
                     lam_note=r"$\lambda_{steam}=1.24$")


if __name__ == "__main__":
    main()

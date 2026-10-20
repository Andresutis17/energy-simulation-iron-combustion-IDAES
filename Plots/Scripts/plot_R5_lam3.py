"""

Redcution operating multiplot vs T_gas,in
                              vs T_solid,in
                              
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from plot_R4_operating_reduction import num, operating_figure, valid_rows

FIGS = [
    ("T_gas", "sens_reduction_T_gas_lam3_operating.csv", 1050.0,
     r"$T_{gas,in}$ [K]", "R5a_operating_Tgas_lam3"),
    ("T_solid", "sens_reduction_T_solid_lam3_operating.csv", 1173.9,
     r"$T_{solid,in}$ [K]", "R5b_operating_Tsolid_lam3"),
]


def main():
    common.apply_style()
    for knob, csv_name, base, xlabel, stem in FIGS:
        rows = common.load_csv(csv_name)
        ok = valid_rows(rows)
        ok.sort(key=lambda r: num(r, "value"))
        operating_figure(ok, "value", base,
                         "Baseline operating point, "
                         r"$\lambda_{H_2}=3$", xlabel, stem,
                         gamma=True,
                         lam_note=r"$\lambda_{H_2}=3$")


if __name__ == "__main__":
    main()

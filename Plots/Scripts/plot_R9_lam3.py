"""
Reduction gas flow at constant lambda_H2 = 3
plotted against the velocity ratio u0/umf

"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from plot_R4_operating_reduction import num, operating_figure, valid_rows

CSV_NAME = "sens_reduction_gasflow_lam3_operating.csv"
BASE_UMF = 37.797339063   # lambda=3 anchors u0/umf


def main():
    common.apply_style()
    rows = common.load_csv(CSV_NAME)
    # g is u0/umf = 60, (60 / 37.797)=1.58741
    dropped = [r for r in rows
               if r.get("value")
               and abs(float(r["value"]) - 1.58741) < 1e-4]
    rows = [r for r in rows if r not in dropped]
    ok = valid_rows(rows)
    ok.sort(key=lambda r: num(r, "u_g_over_umf"))
    operating_figure(ok, "u_g_over_umf", BASE_UMF,
                     r"Baseline, $\lambda_{H_2}=3$",
                     r"Inlet $u_g/u_{mf}$ (via gas flow at constant "
                     r"$\lambda_{H_2}=3$)",
                     "R9_operating_gasflow_lam3", gamma=True,
                     top_scale=[float(r["value"]) for r in ok],
                     top_label="Gas flow scale, g",
                     lam_note=r"$\lambda_{H_2}=3$")


if __name__ == "__main__":
    main()

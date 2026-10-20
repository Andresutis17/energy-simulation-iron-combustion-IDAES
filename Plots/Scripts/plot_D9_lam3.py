"""

Dry gas flow at constant lambda_O2 = 1.501
plotted against the velocity ratio u0/umf

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from plot_R4_operating_reduction import num, operating_figure, valid_rows

CSV_NAME = "sens_dry_gasflow_d9_operating.csv"
BASE_UMF = 36.42148037205716   # the dry basepoints u0/umf


def main():
    common.apply_style()
    rows = common.load_csv(CSV_NAME)
    #  g is u0/umf = 60, (60 / 36.42)=1.64738

    dropped = [r for r in rows
               if r.get("value")
               and abs(float(r["value"]) - 1.64738) < 1e-4]
    rows = [r for r in rows if r not in dropped]
    ok = valid_rows(rows)
    ok.sort(key=lambda r: num(r, "u_g_over_umf"))
    operating_figure(ok, "u_g_over_umf", BASE_UMF,
                     r"Baseline, $\lambda_{O_2}=1.501$",
                     r"Inlet $u_g/u_{mf}$ (via gas flow at constant "
                     r"$\lambda_{O_2}$)",
                     "D9_operating_gasflow", reactor="dry", gamma=False,
                     top_scale=[float(r["value"]) for r in ok],
                     top_label="Gas flow scale, g",
                     lam_note=r"$\lambda_{O_2}=1.501$")


if __name__ == "__main__":
    main()

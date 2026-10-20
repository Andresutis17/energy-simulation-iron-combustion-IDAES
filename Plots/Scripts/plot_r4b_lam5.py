"""

Reduction stoichiometry/composition at fixed throughput,
plotted against lambda_H2

"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from plot_R4_operating_reduction import num, operating_figure

STAGE = os.path.join(common.DATA, "op_reduction_r4blam5_points.json")


def main():
    common.apply_style()
    rows = [r for r in json.load(open(STAGE)) if r.get("valid")]
    ok = sorted(rows, key=lambda r: num(r, "lambda_H2"))
    operating_figure(ok, "lambda_H2", 3.0,
                     r"$\lambda_{H_2} = 3$ endpoint (high-flow branch)",
                     r"$\lambda_{H_2}$ (via H$_2$ mole fraction)",
                     "R4b_lam5",
                     reactor="reduction",
                     gamma=True,
                     util_ylim=(0, 9))


if __name__ == "__main__":
    main()

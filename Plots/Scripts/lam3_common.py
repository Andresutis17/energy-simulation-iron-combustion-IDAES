"""
Shared constants and helpers for lambda_H2 = 3 
"""
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import gather_sensitivity as gs
import run_lambda_operating as rlo

gs.DP_MM = 0.28                   # Particle diameter mm

BASE_Y = 0.99                     # Reactant fraction of the feed
AREA_RATIO = (0.054 / 6.5) ** 2
BASE_FLOW = 300.0 * AREA_RATIO          # Lab gas flow, mol/s
BASE_SOLID = 10.0 * AREA_RATIO          # Lab solid flow, kg/s
BASE_LAM_RAW = 1.5809047487510743  # Lambda of the matched baseline
LAM3 = 3.0                        # The new baseline lambda

# Removable oxygen of the untouched lab solid feed
O_REM = BASE_SOLID * (3.0 * 1.0 / common.MW_S["Fe2O3"]
                      + 4.0 * 1e-5 / common.MW_S["Fe3O4"]
                      + 1e-5 / common.MW_S["FeO"])

F0_NEW = LAM3 * O_REM / BASE_Y          # Lambda=3 total gas flow mol/s
K_LAM3 = LAM3 / BASE_Y                  # Lambda = K_LAM3 * y_H2 at F0_NEW
GAS_MULT = F0_NEW / BASE_FLOW           # runner_axial gas multiplier
UMF_LO, UMF_HI = 10.0, 40.0             # u0/umf window
UMF_PER_LAM = 12.599112687974475        # u0/umf per unit lambda, measured at T_gas,in = 1050
UMF_LAM3 = UMF_PER_LAM * LAM3           # u0/umf at lambda=3 anchor

# The wet basepoint is the module base flow. y_H2O = 0.99, u0/umf = 19.918 at T_gas,in = 1050
LAM_WET0 = 1.243947375            # Wet baseline
UMF_WET0 = 19.91799707846876
# The dry basepoint flow is 960 mol/s. y_O2 = 0.21, u0/umf = 36.421, T_gas,in = 600
DRY_BASE_FLOW = 960.0 * AREA_RATIO      # 0.06625704142011833 mol/s
UMF_DRY0 = 36.42148037205716  # u0/umf of the dry basepoint

RUNNER_POINT = os.path.join(common.HERE, "runner_lam3_point.py")


def cold_point(label, value, yh2=None, gas_flow=None, solid_flow=None,
               tgas=None, tsolid=None, lamreq=None, ncont=None, log=None,
               reactor="reduction", ywater=None, lamkey="lambda_H2"):
    """
    Solves one point. Runs runner_lam3_point, reads its POINT line
    and returns the row. If it fails it returns an invalid row
    """
    if gas_flow is None:
        gas_flow = F0_NEW
    if solid_flow is None:
        solid_flow = BASE_SOLID
    tag = f"{label}_{value:g}".replace(".", "p").replace("@", "_") \
        + ("_n40" if ncont else "")
    if log is None:
        log = os.path.join(gs.LOGDIR, f"lam3_{tag}.log")
    cmd = [sys.executable, RUNNER_POINT,
           "--reactor", reactor,
           "--label", label, "--value", str(value),
           "--gas-flow", str(gas_flow), "--solid-flow", str(solid_flow),
           "--dp", str(gs.DP_MM), "--lamkey", lamkey]
    if yh2 is not None:
        cmd += ["--yh2", str(yh2)]
    if ywater is not None:
        cmd += ["--ywater", str(ywater)]
    if tgas is not None:
        cmd += ["--tgas", str(tgas)]
    if tsolid is not None:
        cmd += ["--tsolid", str(tsolid)]
    if lamreq is not None:
        cmd += ["--lamreq", str(lamreq)]
    if ncont:
        cmd += ["--ncont", str(ncont)]
    with open(log, "w") as lf:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=lf,
                           timeout=gs.timeout_of("reduction"),
                           cwd=common.HERE, text=True, check=False)
    line = next((ln for ln in p.stdout.splitlines()
                 if ln.startswith("POINT ")), None)
    if p.returncode != 0 or not line:
        stub = {"reactor": reactor, "knob": label, "value": value,
                "path": "cold", "valid": False,
                "term": f"rc={p.returncode}", "log_path": log}
        return stub
    row = json.loads(line[6:])
    row["log_path"] = log
    row["net_H2_consumption"] = (rlo.num(row, "n_in_H2")
                                 - rlo.num(row, "n_out_H2"))
    return row


def read_frozen_lam3():
    """
    The lambda=3 anchor row, read from  R4 CSV. It only reads the row back
    """
    path = os.path.join(common.DATA, rlo.CSV_NAME)
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("value")
                and abs(float(r["value"]) - LAM3) < 1e-9]
    if not rows or rows[0].get("valid") != "True":
        raise ValueError(f"missing lambda=3 anchor row in "
                         f"{rlo.CSV_NAME} - run run_lam3_base.py first")
    return rows[0]


def bootstrap_csv(csv_name, template=None):
    """
    Creates an empty CSV with the columns of an existing one.
    Use a same reactor template for wet and dry.
    """
    path = os.path.join(common.DATA, csv_name)
    if not os.path.exists(path):
        src = template or rlo.CSV_NAME
        with open(os.path.join(common.DATA, src), newline="") as f:
            fields = list(csv.DictReader(f).fieldnames)
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()


def merge_or_force(csv_name, key, stage, tag):

    """Merges the staged rows into the family CSV"""
    rlo.merge(csv_name=csv_name, key=key, stage=stage)


def umf_flag(row):

    """Label for the u0/umf of one row, it is checked row by row
    because T_gas,in moves u0"""

    u = rlo.num(row, "u_g_over_umf")
    if u == 0.0:
        return "n/a"
    if u < UMF_LO or u > UMF_HI:
        return f"out of window ({u:.2f})"
    return f"ok ({u:.2f})"

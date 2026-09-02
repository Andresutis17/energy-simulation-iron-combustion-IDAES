"""
Reruns the wet sweeps. The frozen files are never touched
"""


import concurrent.futures
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from gather_sensitivity import FIELDS

# Same knob set the frozen wet sweeps
KNOBS = ["T_solid", "gas_flow", "solid_flow", "porosity"]
TIMEOUT = 2400
RUNNER = os.path.join(common.HERE, "runner_sens.py")
WORKERS = 3  # how many solves run at once


def grid(knob):
    """
    The values to solve, the same poins tried in the original sweep
    """
    path = os.path.join(common.DATA, f"sens_wet_{knob}.csv")
    with open(path, newline="") as f:
        return [(knob, float(r["value"])) for r in csv.DictReader(f)]


def solve(point, ncont):
    """
    One wet solve in its own process. With ncont, uses the finer solver
    Returns the result row
    """
    knob, v = point
    tag = f"sens_wet_{knob}_{v:g}_fe34" + ("_ncont" if ncont else "")
    log = os.path.join(common.DATA, "logs", f"{tag}.log")
    cmd = [sys.executable, RUNNER, "--reactor", "wet", "--knob", knob,
           "--value", str(v)]
    if ncont:
        cmd += ["--ncont", "40"]
    try:
        with open(log, "w") as log_f:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=log_f,
                                  timeout=TIMEOUT, cwd=common.HERE, text=True)
        line = next((ln for ln in proc.stdout.splitlines()
                     if ln.startswith("POINT ")), None)
        if proc.returncode == 0 and line:
            return json.loads(line[6:])
        return {"knob": knob, "value": v, "term": f"rc={proc.returncode}",
                "valid": False}
    except subprocess.TimeoutExpired:
        return {"knob": knob, "value": v, "term": "TIMEOUT", "valid": False}


def run_one(point):
    """
    One solve, with one retry at the finer startup if the first one died
    """
    row = solve(point, ncont=False)
    if row.get("valid") is not True:
        retry = solve(point, ncont=True)
        if retry.get("valid") is True:
            row = retry
    print(f"  {row['knob']} {row['value']:g} "
          f"valid={row.get('valid')} X_prod={row.get('X_prod')}",
          flush=True)
    return row



def main():
    points = [point for knob in KNOBS for point in grid(knob)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(run_one, points))
    for knob in KNOBS:
        fam_rows = sorted((row for row in rows if row["knob"] == knob),
                          key=lambda row: float(row["value"]))
        path = os.path.join(common.DATA, f"sens_wet_{knob}_fe34.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows({col: row.get(col, "") for col in FIELDS}
                        for row in fam_rows)




if __name__ == "__main__":
    main()

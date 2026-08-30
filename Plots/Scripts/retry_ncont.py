"""
One more try for points the solver could not solved, with a finer
startup. A retry can only improve, never makes it worse
"""

import concurrent.futures
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  
from gather_sensitivity import FIELDS, FAMILIES, csv_path, write_csv  

# The points that need a retry 
RETRIES = [
    ("reduction", "T_solid", 1073.0),
    ("reduction", "T_solid", 923.0),
    ("reduction", "solid_flow", 0.75),
    ("reduction", "solid_flow", 0.5),
    ("wet", "porosity", 0.40),
]

# Same timeouts as in gather script
TIMEOUT = {"reduction": 1800, "wet": 2400}
RUNNER = os.path.join(common.HERE, "runner_sens.py")


def fam_of(reactor, knob):

    """
    The family name for a reactor plus knob pair
    """
    return f"{reactor}_{knob}"


def run_one(retry):
    """
    Runs one retry in its own  process and return its row,
    the solved one if it lands and a dead row if it doesnt
    """
    reactor, knob, v = retry
    tag = f"sens_{reactor}_{knob}_{v:g}_ncont"
    log = os.path.join(common.DATA, "logs", f"{tag}.log")
    cmd = [sys.executable, RUNNER, "--reactor", reactor, "--knob", knob,
           "--value", str(v), "--ncont", "40"]
    try:
        with open(log, "w") as lf:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=lf,
                               timeout=TIMEOUT[reactor], cwd=common.HERE,
                               text=True)
        line = next((ln for ln in proc.stdout.splitlines()
                     if ln.startswith("POINT ")), None)
        if proc.returncode == 0 and line:
            row = json.loads(line[6:])
            return row
        return {"reactor": reactor, "knob": knob, "value": v,
                "term": f"rc={proc.returncode}", "valid": False}
    except subprocess.TimeoutExpired:
        return {"reactor": reactor, "knob": knob, "value": v,
                "term": "TIMEOUT", "valid": False}


def main():
   
    if len(sys.argv) > 1:
        wanted = set(sys.argv[1:])
        picked = [retry for retry in RETRIES
                  if f"{retry[0]}/{retry[1]}/{retry[2]:g}" in wanted]
        if not picked:
            sys.exit(f"no retry matches {wanted}")

    else:
        picked = RETRIES
    landed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for row in ex.map(run_one, picked):
            if row.get("valid") is True or row.get("valid") == "True":
                landed.append(row)
    

    fams = sorted({fam_of(r["reactor"], r["knob"]) for r in landed})
    for fam in fams:
        new_rows = [r for r in landed if fam_of(r["reactor"], r["knob"]) == fam]
        path = csv_path(fam)
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        for row in new_rows:
            for i, old_row in enumerate(rows):
                if abs(float(old_row["value"]) - float(row["value"])) < 1e-9:
                    rows[i] = {k: row.get(k, "") for k in FIELDS}
                    break
        write_csv(fam, rows)  


if __name__ == "__main__":
    main()

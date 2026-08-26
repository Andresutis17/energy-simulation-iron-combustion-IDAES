
"""
Uses the axial solves from runner_axial to write csv and json

"""

import concurrent.futures
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  

RUNNER = os.path.join(common.HERE, "runner_axial.py")
LOGDIR = os.path.join(common.DATA, "logs")

# The 3 used models, each one with lab and industrial cases
CASES = [
    ("reduction", "lab"), ("reduction", "ind"),
    ("wet", "lab"), ("wet", "ind"),
    ("dry", "lab"), ("dry", "ind"),
]


def run_case(case):
    """
    Run one axial reactor case and save its output to a log
    """
    reactor, scale = case
    cmd = [sys.executable, RUNNER,
           "--reactor", reactor, "--scale", scale, "--mode", "axial"]
    log = os.path.join(LOGDIR, f"axial_{reactor}_{scale}.log")
    try:
        with open(log, "w") as lf:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                               timeout=2700, cwd=common.HERE)
        return reactor, scale, p.returncode, None
    except subprocess.TimeoutExpired:
        return reactor, scale, None, "TIMEOUT"


def main():
    common.ensure_dirs()
    os.makedirs(LOGDIR, exist_ok=True)
    bad = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for reactor, scale, return_code, err in executor.map(run_case, CASES):
            if err or return_code != 0 or not os.path.exists(common.axial_csv(reactor, scale)):
                bad += 1
    if bad:
        sys.exit(f"{bad}/Fail {LOGDIR}")



if __name__ == "__main__":
    main()

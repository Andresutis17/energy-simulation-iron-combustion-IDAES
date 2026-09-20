
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

# dp=28 mm 
DP_MM = None
SUFFIX = ""


def run_case(case):
    """
    Run one axial reactor case and save its output to a log
    """
    reactor, scale = case
    cmd = [sys.executable, RUNNER,
           "--reactor", reactor, "--scale", scale, "--mode", "axial"]
    if DP_MM:
        cmd += ["--dp", str(DP_MM)]
    if SUFFIX:
        cmd += ["--suffix", SUFFIX]
    log = os.path.join(LOGDIR, f"axial_{reactor}_{scale}{SUFFIX}.log")
    try:
        with open(log, "w") as lf:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                               timeout=int(2700 * (1.5 if DP_MM else 1.0)),
                               cwd=common.HERE)
        return reactor, scale, p.returncode, None
    except subprocess.TimeoutExpired:
        return reactor, scale, None, "TIMEOUT"


def main():
    global DP_MM, SUFFIX
    argv = iter(sys.argv[1:])
    dp_arg = suf_arg = None
    for a in argv:
        if a == "--dp":
            dp_arg = next(argv, None)
        elif a == "--suffix":
            suf_arg = next(argv, None)
    if (dp_arg is None) != (suf_arg is None):
        sys.exit("")
    if dp_arg is not None:
        DP_MM = float(dp_arg)
        SUFFIX = suf_arg if suf_arg.startswith("_") else f"_{suf_arg}"
        print(f"dp = {DP_MM} mm, files end in '{SUFFIX}'",
              flush=True)
    common.ensure_dirs()
    os.makedirs(LOGDIR, exist_ok=True)
    bad = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for reactor, scale, return_code, err in executor.map(run_case, CASES):
            if err or return_code != 0 or not os.path.exists(
                    common.axial_csv(reactor, scale, SUFFIX)):
                bad += 1
    if bad:
        sys.exit(f"{bad}/Fail {LOGDIR}")



if __name__ == "__main__":
    main()

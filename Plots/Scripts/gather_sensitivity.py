"""
This script runs all the sensitivity sweeps, only one knob per solve of the 3 reactors
"""
import concurrent.futures
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  

RUNNER_COLD = os.path.join(common.HERE, "runner_sens.py")
RUNNER_LADDER = os.path.join(common.HERE, "runner_dry_sens.py")
RUNNER_AXIAL = os.path.join(common.HERE, "runner_axial.py")
LOGDIR = os.path.join(common.DATA, "logs")

# The known good and real answers, if the solver misses them, then abort the solve
EXPECTED = {
    "reduction": {"X_solid": 99.06, "X_gas": 29.63, "T_solid_out": 1034.8},
    "wet": {"X_solid": 40.89, "X_gas": 26.77, "T_solid_out": 1160.1},
    "dry": {"X_solid": 20.97, "X_gas": 13.97, "T_solid_out": 1091.1},
}

# A little tolerance for the solvers, conversion and temperature
GATE_TOL = {"X": 0.05, "T": 0.5}  

# All the sweeps in run order
FAMILIES = {
    
    "reduction_y_H2": {
        "reactor": "reduction", "knob": "y_H2", "route": "cold",
        
        "grid": [0.99, 0.75, 0.50, 0.30, 0.25, 0.20, 0.10, 0.05],
    },
    "reduction_T_solid": {
        "reactor": "reduction", "knob": "T_solid", "route": "cold",
        
        "grid": [1223, 1173.9, 1123, 1073, 1023, 973, 923],
    },
    "reduction_gas_flow": {
        "reactor": "reduction", "knob": "gas_flow", "route": "cold",

        "grid": [2.0, 1.5, 1.0, 0.75, 0.5],
    },
    "reduction_solid_flow": {
        "reactor": "reduction", "knob": "solid_flow", "route": "cold",

        "grid": [2.0, 1.5, 1.0, 0.75, 0.5],
    },
    "reduction_porosity": {
        "reactor": "reduction", "knob": "porosity", "route": "cold",

        "grid": [0.27, 0.15, 0.20, 0.35, 0.40],
    },
    
    "dry_porosity": {
        "reactor": "dry", "knob": "porosity", "route": "ladder",
        
        "grid": [0.27, 0.20, 0.15, 0.27, 0.30, 0.35, 0.40],
    },
    "dry_y_O2": {
        "reactor": "dry", "knob": "y_O2", "route": "ladder",
       
        "grid": [0.21, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05, 0.21],
    },
    "dry_T_solid": {
        "reactor": "dry", "knob": "T_solid", "route": "ladder",
        
        "grid": [1073, 1023, 973, 923, 1073, 1123],
    },
    "dry_gas_flow": {
        "reactor": "dry", "knob": "gas_flow", "route": "ladder",
        
        "grid": [1.0, 0.75, 0.5, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0],
    },
    "dry_solid_flow": {
        "reactor": "dry", "knob": "solid_flow", "route": "ladder",

        "grid": [1.0, 0.75, 0.5, 1.0, 1.5, 2.0],
    },
    
    "wet_T_solid": {
        "reactor": "wet", "knob": "T_solid", "route": "cold",

        "grid": [1223, 1173.9, 1123, 1073, 1023, 973, 923],
    },
    "wet_porosity": {
        "reactor": "wet", "knob": "porosity", "route": "cold",

        "grid": [0.27, 0.15, 0.20, 0.35, 0.40],
    },
    "wet_gas_flow": {
        "reactor": "wet", "knob": "gas_flow", "route": "cold",
        
        "grid": [2.0, 1.5, 1.0, 0.9, 0.85, 0.8, 0.75, 0.5],
    },
    "wet_solid_flow": {
        "reactor": "wet", "knob": "solid_flow", "route": "cold",

        "grid": [2.0, 1.5, 1.0, 0.75, 0.5],
    },
}

# The CSV columns
FIELDS = ["reactor", "knob", "value", "abs_flow", "path", "X_solid", "X_gas",
          "T_gas_out", "T_solid_out", "Ts_in", "n_bad", "err_mass",
          "gas_feasible", "banner_ok", "valid", "term"]

# Max seconds per run
TIMEOUT_COLD = {"reduction": 1800, "wet": 2400}
TIMEOUT_LADDER = 3600


def csv_path(fam):
    """
    Which CSV file a family writes its results to
    """

    return os.path.join(common.DATA, f"sens_{FAMILIES[fam]['reactor']}"
                                     f"_{FAMILIES[fam]['knob']}.csv")


def write_csv(fam, rows):
    """
    Write one family's rows to its CSV, sorted by knob value.
    Rewrites the whole file each time 
    """

    rows = sorted(rows, key=lambda r: float(r["value"]))
    with open(csv_path(fam), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def parse_points_from_log(path, grid=None, meta=None):
    
    """
    Reads the POINT lines out of a run's log, one row per knob value.

    """

    rows, seen, drifts = [], set(), []
    if not os.path.exists(path):
        return rows, drifts
    with open(path, errors="replace") as f:
        for ln in f:
            if not ln.startswith("POINT "):
                continue
            try:
                p = json.loads(ln[6:])
            except json.JSONDecodeError:
                continue
            key = float(p["value"])
            if key in seen:
                if p.get("path") == "anchor":
                    drifts.append((p.get("anchor_drift_dX"),
                                   p.get("anchor_drift_dT")))
                continue
            seen.add(key)
            rows.append(p)
    if grid is not None:
        for v in grid:
            if float(v) not in seen:
                stub = dict(meta or {}, value=float(v), path="unvisited",
                            term="NOT_IN_LOG", valid=False)
                rows.append(stub)
    return rows, drifts


def preflight():

    """
    Health check before anything runs. Solves the three match points
    first, if theres an error nothing runs
    """
    ok = True
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        solves = {}
        for reactor in EXPECTED:
            log = os.path.join(LOGDIR, f"sens_preflight_{reactor}.log")
            cmd = [sys.executable, RUNNER_AXIAL, "--reactor", reactor,
                   "--scale", "lab", "--mode", "point"]
            solves[ex.submit(subprocess.run, cmd, stdout=subprocess.PIPE,
                           stderr=open(log, "w"), timeout=TIMEOUT_COLD.get(
                               reactor, 2400), cwd=common.HERE, text=True,
                           check=False)] = reactor
        for solve in concurrent.futures.as_completed(solves):
            reactor = solves[solve]
            try:
                p = solve.result()
            except subprocess.TimeoutExpired:
                print(f"  [{reactor}] fail", flush=True)
                ok = False
                continue
            line = next((ln for ln in p.stdout.splitlines()
                         if ln.startswith("POINT ")), None)
            if p.returncode != 0 or not line:
                print(f"  [{reactor}] rc={p.returncode} - abort", flush=True)
                ok = False
                continue
            point = json.loads(line[6:])
            expected = EXPECTED[reactor]
            dx = abs(point["X_solid"] - expected["X_solid"])
            dt = abs(point["T_solid_out"] - expected["T_solid_out"])
            dxg = abs(point["X_gas"] - expected["X_gas"])
            # X_gas is checked too, only works if the 3 numbers match, if not
            # then abort
            good = (dx <= GATE_TOL["X"] and dt <= GATE_TOL["T"]
                    and dxg <= 0.1 and point["valid"])
            print(f"  [{reactor}] X={point['X_solid']:.2f} (exp {expected['X_solid']:.2f},"
                  f" d{dx:.3f})  Xg={point['X_gas']:.2f} (d{dxg:.3f})"
                  f"  T={point['T_solid_out']:.1f} (exp"
                  f" {expected['T_solid_out']:.1f}, d{dt:.2f})  "
                  f"{'OK' if good else 'Fail'}", flush=True)
            ok = ok and good
    if not ok:
        sys.exit("Health check failed")

def run_cold_point(fam, spec, value):
    """
    Run one point, read its POINT line and finally return the row

    """
    tag = f"{fam}_{value:g}"
    cmd = [sys.executable, RUNNER_COLD, "--reactor", spec["reactor"],
           "--knob", spec["knob"], "--value", str(value)]
    log = os.path.join(LOGDIR, f"sens_{tag}.log")
    try:
        with open(log, "w") as lf:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=lf,
                               timeout=TIMEOUT_COLD[spec["reactor"]],
                               cwd=common.HERE, text=True, check=False)
        line = next((ln for ln in p.stdout.splitlines()
                     if ln.startswith("POINT ")), None)
        if p.returncode == 0 and line:
            return json.loads(line[6:])
        return {"reactor": spec["reactor"], "knob": spec["knob"], "value": value,
                "path": "cold", "term": f"rc={p.returncode}", "valid": False}
    except subprocess.TimeoutExpired:
        return {"reactor": spec["reactor"], "knob": spec["knob"], "value": value,
                "path": "cold", "term": "TIMEOUT", "valid": False}


def run_cold_family(fam): #A family is a complete sweep
    """
    Run every value of one family and write its CSV
    """

    spec = FAMILIES[fam]
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        solves = {ex.submit(run_cold_point, fam, spec, v): v
                for v in spec["grid"]}
        for solve in concurrent.futures.as_completed(solves):
            row = solve.result()
            rows.append(row)
            write_csv(fam, rows)  


def run_ladder_family(fam):
    """
    Runs one family, one process walks the grid and rows come from its log
    """
    spec = FAMILIES[fam]
    values = ",".join(f"{v:g}" for v in spec["grid"])
    log = os.path.join(LOGDIR, f"sens_{fam}.log")
    cmd = [sys.executable, RUNNER_LADDER, "--knob", spec["knob"],
           "--values", values]
    with open(log, "w") as lf:
        try:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                           timeout=TIMEOUT_LADDER, cwd=common.HERE,
                           check=False)
        except subprocess.TimeoutExpired:
            pass  
    rows, drifts = parse_points_from_log(
        log, spec["grid"], {"reactor": spec["reactor"], "knob": spec["knob"]})
    for dx, dt in drifts:
        if dx is not None and (dx > 0.05 or dt > 0.5):
            print(f"  [{fam}] drift dX={dx:.3f} pp dT={dt:.2f} K",
                  flush=True)
    write_csv(fam, rows)



def run_family(fam):
    """
    Pick the right route for this family, cold or ladder
    """

    if FAMILIES[fam]["route"] == "cold":
        run_cold_family(fam)
    else:
        run_ladder_family(fam)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    common.ensure_dirs()
    os.makedirs(LOGDIR, exist_ok=True)
    preflight()
    if which == "preflight":
        return
    if which == "all":
        targets = list(FAMILIES)
    elif which in ("reduction", "wet", "dry"):
        targets = [f for f in FAMILIES if FAMILIES[f]["reactor"] == which]
    elif which in FAMILIES:
        targets = [which]
    else:
        sys.exit(f"unknown family '{which}'")
    for fam in targets:
        run_family(fam)



if __name__ == "__main__":
    main()

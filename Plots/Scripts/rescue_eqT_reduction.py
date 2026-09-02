"""
The cold start dies at the same T point for gas and solid, so T_gas is walked up
from the lab match in small steps, each solve used for the next step
"""

import csv
import json
import os
import shutil
import sys

sys.stdout.reconfigure(line_buffering=True)

from pyomo.environ import Var

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import runner_axial as ra

# the same T target and the T_gas steps from the match up to it
TARGET_TG = 1173.9
RAMP = [1100.0, 1123.0, 1150.0, TARGET_TG]


def snapshot(m):
    """
    A save point for every variable value of the model, to use as the
    starting point of a next step
    """
    return {v.name: v.value
            for v in m.component_data_objects(Var, descend_into=True)}


def restore(m, snap):

    """
    Loads a save point back into the model
    """
    for v in m.component_data_objects(Var, descend_into=True):
        if v.name in snap:
            v.set_value(snap[v.name])


def main():
    mod = ra.load_module(common.LAB_SCRIPTS["reduction"], "reduction")
    from idaes.core.solvers import get_solver
    solver = get_solver()

    # Keep a copy of the failed cold attempt 
    archive_dir = os.path.join(common.DATA, "_archive_2026-09-01_eqT")
    os.makedirs(archive_dir, exist_ok=True)
    for name in ("axial_reduction_lab_eqT.csv",
                 "axial_reduction_lab_eqT.meta.json"):
        old_path = os.path.join(common.DATA, name)
        backup_path = os.path.join(archive_dir, name)
        if os.path.exists(old_path) and not os.path.exists(backup_path):
            shutil.copy2(old_path, backup_path)

    # Step 1: land the lab match first 
    target = dict(mod.LAB)
    target["H"] = common.LAB_MATCH["reduction"]["H"]
    target["n_orifice"] = common.LAB_MATCH["reduction"]["n_orifice"]
    m, report = mod.solve_case(target, verbose=True)
    status = report["termination"]
    if status != "optimal" or report["err_mass"] > 1.0:
        sys.exit(1)

    # Step 2: walk T_gas up to the same T target
    bed = m.fs.BFB
    snap = snapshot(m)
    t_now = target["gas_T"]
    for t_next in RAMP:
        if t_next <= t_now:
            continue
        mod.set_case(m, target["D"], target["H"], target["n_orifice"],
                     target["gas_flow_mol"], target["solid_flow_mass"],
                     t_next, target["solid_T"])
        try:
            result = solver.solve(bed, tee=False)
            status = str(result.solver.termination_condition)
        except Exception as e:
            status = f"EXC:{type(e).__name__}"
        if status == "optimal":
            t_now = t_next
            snap = snapshot(m)
            continue

        # If a step fails, go back to the save point and try halfway
        lo, hi = t_now, t_next
        landed = None
        for _ in range(3):
            mid = 0.5 * (lo + hi)
            if hi - lo < 2.0:
                break
            restore(m, snap)
            mod.set_case(m, target["D"], target["H"], target["n_orifice"],
                         target["gas_flow_mol"], target["solid_flow_mass"],
                         mid, target["solid_T"])
            try:
                result = solver.solve(bed, tee=False)
                mid_status = str(result.solver.termination_condition)
            except Exception as e:
                mid_status = f"EXC:{type(e).__name__}"
            if mid_status != "optimal":
                hi = mid
                continue
            landed = mid
            lo = mid
        if landed is not None and landed > t_now:
            t_now = landed
            snap = snapshot(m)
        if t_now < t_next - 1e-6:
            # Last try at the step target 
            restore(m, snap)
            mod.set_case(m, target["D"], target["H"], target["n_orifice"],
                         target["gas_flow_mol"], target["solid_flow_mass"],
                         t_next, target["solid_T"])
            try:
                result = solver.solve(bed, tee=False)
                retry_status = str(result.solver.termination_condition)
            except Exception as e:
                retry_status = f"EXC:{type(e).__name__}"
            if retry_status == "optimal":
                t_now = t_next
                snap = snapshot(m)
            else:
                restore(m, snap)
                sys.exit(2)

    # Step 3: check if the result is balanced 
    eqT_target = dict(target, gas_T=TARGET_TG)
    check = mod._collect(m, eqT_target, "optimal")
    ok = (check["err_mass"] <= 1.0 and check["n_bad"] == 0
          and check["gas_feasible"])
    if not ok:
        sys.exit(3)

    # Step 4: write the eqT files
    endpoint = ra.endpoint(m, "reduction")
    point = {"reactor": "reduction", "scale": "lab", "term": "optimal",
             "err_mass": check["err_mass"],
             "gas_feasible": check["gas_feasible"],
             "rescue": "warm T ramp (match to 1173.9 K)", **endpoint}
    point["valid"] = (point["n_bad"] == 0 and point["gas_feasible"]
                      and "optimal" in str(point["term"]))
    rows, _ = ra.extract_rows(m, "reduction")
    common.ensure_dirs()
    with open(common.axial_csv("reduction", "lab", "_eqT"), "w",
              newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    meta = dict(point)
    meta["source"] = common.LAB_SCRIPTS["reduction"]
    meta["gas_T"] = TARGET_TG
    meta["solid_T"] = target["solid_T"]
    with open(common.axial_meta("reduction", "lab", "_eqT"), "w") as f:
        json.dump(meta, f, indent=2)



if __name__ == "__main__":
    main()

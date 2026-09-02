"""
The dry cold start dies at the eqT target (same T for gas and solid), so Tg
is walked up from the match in small steps, each solve used for the next one.

"""
import csv
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

from pyomo.environ import Var

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import runner_axial as ra

TARGET_TG = 1073.0

# Match Tg first, then small steps up to the eqT target
RAMP = [750.0, 850.0, 923.0, 940.0, 950.0, 960.0, 973.0,
        985.0, 1000.0, 1015.0, 1030.0, 1045.0, 1060.0, TARGET_TG]

ERR_GATE = 0.1  # tolerance

# Checkpoints save every landed points, so a rerun resumes from the last one
CKPT = os.path.join(common.DATA, "logs", "rescue_eqT_dry_ckpt.json")


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
        if v.name in snap and snap[v.name] is not None:
            v.set_value(snap[v.name])


def ckpt_save(key, snap):
    """
    Saves one snapshot in the checkpoint file, under the Tg it landed at
    """
    store = {}
    if os.path.exists(CKPT):
        try:
            store = json.load(open(CKPT))
        except (json.JSONDecodeError, OSError):
            store = {}
    store[key] = snap
    with open(CKPT, "w") as f:
        json.dump(store, f)


def ckpt_load():

    """
    Loads the saved checkpoints, empty if the file is missing or broken
    """
    if not os.path.exists(CKPT):
        return {}
    try:
        return json.load(open(CKPT))
    except (json.JSONDecodeError, OSError):
        return {}


def main():
    mod = ra.load_module(common.LAB_SCRIPTS["dry"], "dry")
    solver = mod.get_solver()
    solver.options = dict(mod.SOLVER_OPTS)

    target = dict(mod.LAB)
    target["H"] = common.LAB_MATCH["dry"]["H"]
    target["n_orifice"] = common.LAB_MATCH["dry"]["n_orifice"]

    # Step 1: land the lab match first
    m, report = mod.solve_case(target, verbose=True)
    status = report["termination"]
    if "optimal" not in status or report["err_mass"] > ERR_GATE:
        sys.exit(1)
    ckpt_save("match", snapshot(m))

    bed = m.fs.BFB
    # Resume from the highest landed checkpoint of a previous run
    store = ckpt_load()
    resume = None
    for key in store:
        try:
            t_ckpt = float(key)
        except ValueError:
            continue
        if t_ckpt <= TARGET_TG and (resume is None or t_ckpt > resume):
            resume = t_ckpt
    snap = snapshot(m)
    t_now = target["gas_T"]
    if resume is not None:
        restore(m, store[str(resume) if str(resume) in store else resume])
        t_now = resume
        snap = snapshot(m)

    def attempt(t_set, from_snap=None, tag=""):

        """
        One solve at the given Tg. Restores the last good point first if given
        """
        if from_snap is not None:
            restore(m, from_snap)
        mod.set_case(m, target["D"], target["H"], target["n_orifice"],
                     target["gas_flow_mol"], target["solid_flow_mass"],
                     t_set, target["solid_T"])
        return mod._solve(bed, solver, tag, True)

    # Step 2: walk Tg up to the eqT target
    for t_next in RAMP:
        if t_next <= t_now:
            continue
        status = attempt(t_next, tag=f"ramp {t_next:.0f}")
        if not mod._ok(status):
            # If the step fails, reaction off and on again at the new
            # conditions, warm from the last good point
            status = attempt(t_next, from_snap=snap,
                             tag=f"re homotopy {t_next:.0f}")
            if not mod._ok(status):
                mod._homotopy(m, solver, True, f"eqT{t_next:g}")
                status = attempt(t_next, tag=f"homotopy final {t_next:.0f}")
        if mod._ok(status):
            t_now = t_next
            snap = snapshot(m)
            ckpt_save(str(t_next), snap)
            continue
        # Bisection from the last good point
        lo, hi = t_now, t_next
        for _ in range(3):
            mid = 0.5 * (lo + hi)
            if hi - lo < 2.0:
                break
            mid_status = attempt(mid, from_snap=snap, tag=f"mid {mid:.1f}")
            if mod._ok(mid_status):
                lo = mid
            else:
                hi = mid
    if t_now < TARGET_TG - 1e-6:
        # A straight jump to the target 
        retry_status = attempt(TARGET_TG, from_snap=snap,
                               tag=f"jump {TARGET_TG:.0f}")
        if not mod._ok(retry_status):
            sys.exit(2)

    # Step 3: Check if the result is balanced
    eqT_target = dict(target, gas_T=TARGET_TG)
    check = mod._collect(m, eqT_target, "optimal")
    ok = (check["err_mass"] <= ERR_GATE and check["n_bad"] == 0
          and check["gas_feasible"])
    if not ok:
        sys.exit(3)

    # Another path may have landed while this solve, it keep the first result
    meta_path = common.axial_meta("dry", "lab", "_eqT")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        if str(meta.get("term")) == "optimal" and meta.get("valid"):
            return

    # Step 4: write the eqT files
    endpoint = ra.endpoint(m, "dry")
    point = {"reactor": "dry", "scale": "lab", "term": "optimal",
             "err_mass": check["err_mass"],
             "gas_feasible": check["gas_feasible"],
             "rescue": "warm T ramp (match to 1073 K)", **endpoint}
    point["valid"] = (point["n_bad"] == 0 and point["gas_feasible"]
                      and "optimal" in str(point["term"]))
    rows, _ = ra.extract_rows(m, "dry")
    common.ensure_dirs()
    with open(common.axial_csv("dry", "lab", "_eqT"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    meta = dict(point)
    meta["source"] = common.LAB_SCRIPTS["dry"]
    meta["gas_T"] = TARGET_TG
    meta["solid_T"] = target["solid_T"]
    with open(common.axial_meta("dry", "lab", "_eqT"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()

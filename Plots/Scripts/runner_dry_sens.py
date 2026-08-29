"""
Script for dry oxidation sensitivity
It solves at the match point, then step the knob value by value
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  
from runner_axial import endpoint, load_module  


# Knobs: model inputs used in sweeps
KNOBS = ["T_solid", "y_O2", "porosity", "gas_flow", "solid_flow"]


def apply_knob(m, mod, base, knob, value):
    """
    Set the knob to its next value on the model, nothing
    else is touched
    """

    b = m.fs.BFB
    if knob == "T_solid":
        mod.set_case(m, base["D"], base["H"], base["n_orifice"],
                     base["gas_flow_mol"], base["solid_flow_mass"],
                     base["gas_T"], value)
    elif knob == "gas_flow":
        mod.set_case(m, base["D"], base["H"], base["n_orifice"],
                     base["gas_flow_mol"] * value, base["solid_flow_mass"],
                     base["gas_T"], base["solid_T"])
    elif knob == "solid_flow":
        mod.set_case(m, base["D"], base["H"], base["n_orifice"],
                     base["gas_flow_mol"], base["solid_flow_mass"] * value,
                     base["gas_T"], base["solid_T"])
    elif knob == "y_O2":

        # Dilute the feed, O2 takes the new value, N2 takes the rest
        b.gas_inlet.mole_frac_comp[0, "O2"].fix(value)
        b.gas_inlet.mole_frac_comp[0, "N2"].fix(1.0 - value)
    elif knob == "porosity":
        b.solid_inlet.particle_porosity[0].fix(value)
    else:
        raise ValueError(knob)


def target_with_knob(base, knob, value):
    """
    The solve recipe for this step is to use the base value + the new knob value
    """
    tgt = dict(base)
    if knob == "T_solid":
        tgt["solid_T"] = value
    elif knob == "y_O2":
        tgt["y_O2"] = value
        tgt["y_N2"] = 1.0 - value
    elif knob == "porosity":
        tgt["particle_porosity"] = value
    elif knob == "gas_flow":
        tgt["gas_flow_mol"] = base["gas_flow_mol"] * value
    elif knob == "solid_flow":
        tgt["solid_flow_mass"] = base["solid_flow_mass"] * value
    return tgt


def abs_flow_of(tgt, knob):
    """
    Returns the absolute flow the step uses for the CSV
    """
    if knob == "gas_flow":
        return tgt["gas_flow_mol"]
    if knob == "solid_flow":
        return tgt["solid_flow_mass"]
    return None


def report_step(mod, m, results, tgt, knob, v, path, drift=None):
    """
    Report one solved step, builds the point dict, check it and, print
    the json line 
    """
    report = endpoint(m, "dry")
    validity = mod._validity_banner(results)
    err_mass = results.get("err_mass")
    point = {
        "reactor": "dry", "knob": knob, "value": v,
        "abs_flow": abs_flow_of(tgt, knob), "path": path,
        "Ts_in": tgt["solid_T"], "term": results.get("termination", "?"),
        "err_mass": err_mass, "gas_feasible": results.get("gas_feasible"),
        "banner_ok": validity == "", **report,
    }
    if drift is not None:
        point["anchor_drift_dX"], point["anchor_drift_dT"] = drift
    point["valid"] = (point["n_bad"] == 0
                      and (point["gas_feasible"] is None or point["gas_feasible"])
                      and "optimal" in str(point["term"])
                      and err_mass is not None and float(err_mass) <= 0.1
                      and point["banner_ok"])
    print("POINT " + json.dumps(point), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knob", required=True, choices=KNOBS)
    ap.add_argument("--values", required=True)
    args = ap.parse_args()
    values = [float(v) for v in args.values.split(",")]

    mod = load_module(common.LAB_SCRIPTS["dry"], "sens_dry")
    base = dict(mod.LAB)
    base["H"] = common.LAB_MATCH["dry"]["H"]
    base["n_orifice"] = common.LAB_MATCH["dry"]["n_orifice"]
    base_value = {"T_solid": base["solid_T"], "y_O2": base["y_O2"],
                  "porosity": base["particle_porosity"],
                  "gas_flow": 1.0, "solid_flow": 1.0}[args.knob]
    if abs(values[0] - base_value) > 1e-9:
        sys.exit(f"first value {values[0]:g} != base {base_value:g}")

    m, results = mod.solve_case(base, verbose=True)  
    report_step(mod, m, results, base, args.knob, values[0], "cold")
    base_ep = endpoint(m, "dry")

    solver = mod.get_solver()
    solver.options = dict(mod.SOLVER_OPTS)
    b = m.fs.BFB
    dead = False
    for value in values[1:]:
        is_anchor = abs(value - base_value) <= 1e-9
        if dead and not is_anchor:
            dead_point = {"reactor": "dry", "knob": args.knob, "value": value,
                    "path": "walk", "term": "LADDER_STOPPED", "valid": False}
            print("POINT " + json.dumps(dead_point), flush=True)
            continue
        apply_knob(m, mod, base, args.knob, value)
        tag = f"sens {args.knob}={value:g}" + (" ANCHOR" if is_anchor else "")
        term = mod._solve(b, solver, tag, True)
        if not mod._ok(term):
            mod._homotopy(m, solver, True, f"sens{args.knob}{value:g}")
            term = mod._solve(b, solver, f"{tag} final", True)
        tgt = target_with_knob(base, args.knob, value)
        results = mod._collect(m, tgt, term)
        drift = None
        if is_anchor:
            ep_now = endpoint(m, "dry")
            drift = (abs(ep_now["X_solid"] - base_ep["X_solid"]),
                     abs(ep_now["T_solid_out"] - base_ep["T_solid_out"]))
            dead = False  
        report_step(mod, m, results, tgt, args.knob, value,
                    "anchor" if is_anchor else "walk", drift)
        if not mod._ok(term):
            dead = True



if __name__ == "__main__":
    main()

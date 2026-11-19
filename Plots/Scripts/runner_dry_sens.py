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
import dp_patch
from runner_axial import endpoint, load_module


# Knobs: model inputs used in sweeps
KNOBS = ["T_solid", "T_gas", "y_O2", "lambda_O2", "porosity", "gas_flow",
         "solid_flow", "throughput", "H"]

def _fe_metal_feed(target):
    """
    kg/s of Fe metal in the solid feed
    """
    return (target["solid_flow_mass"] * target.get("w_Fe", 0.0)
            / common.MW_S["Fe"])


def _set_o2_feed(target, y_o2):
    """Set the O2 fraction, N2 is the rest"""
    traces = sum(target.get(f"y_{sp}", 0.0) for sp in ("H2O", "CO2", "H2"))
    y_n2 = 1.0 - y_o2 - traces
    if y_o2 < 0.0 or y_n2 < 0.0:
        raise ValueError(f"Impossible error")
    target["y_O2"] = y_o2
    target["y_N2"] = y_n2


def _set_model_o2_feed(b, target):
    """Fix the model inlet to the target composition"""
    b.gas_inlet.mole_frac_comp[0, "O2"].fix(target["y_O2"])
    b.gas_inlet.mole_frac_comp[0, "N2"].fix(target["y_N2"])
    b.gas_inlet.mole_frac_comp[0, "H2O"].fix(target["y_H2O"])
    b.gas_inlet.mole_frac_comp[0, "CO2"].fix(target["y_CO2"])
    b.gas_inlet.mole_frac_comp[0, "H2"].fix(target["y_H2"])


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
    elif knob == "T_gas":
        mod.set_case(m, base["D"], base["H"], base["n_orifice"],
                     base["gas_flow_mol"], base["solid_flow_mass"],
                     value, base["solid_T"])
    elif knob == "gas_flow":
        mod.set_case(m, base["D"], base["H"], base["n_orifice"],
                     base["gas_flow_mol"] * value, base["solid_flow_mass"],
                     base["gas_T"], base["solid_T"])
    elif knob == "solid_flow":
        mod.set_case(m, base["D"], base["H"], base["n_orifice"],
                     base["gas_flow_mol"], base["solid_flow_mass"] * value,
                     base["gas_T"], base["solid_T"])
    elif knob == "throughput":
        # Scale both flows together. Lambda stays konstant and u0/umf changes
        mod.set_case(m, base["D"], base["H"], base["n_orifice"],
                     base["gas_flow_mol"] * value,
                     base["solid_flow_mass"] * value,
                     base["gas_T"], base["solid_T"])
    elif knob == "H":
        mod.set_case(m, base["D"], value, base["n_orifice"],
                     base["gas_flow_mol"], base["solid_flow_mass"],
                     base["gas_T"], base["solid_T"])
    elif knob == "y_O2":
        target = target_with_knob(base, knob, value)
        _set_model_o2_feed(b, target)
    elif knob == "lambda_O2":
        target = target_with_knob(base, knob, value)
        mod.set_case(m, target["D"], target["H"], target["n_orifice"],
                     target["gas_flow_mol"], target["solid_flow_mass"],
                     target["gas_T"], target["solid_T"])
        _set_model_o2_feed(b, target)
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
    elif knob == "T_gas":
        tgt["gas_T"] = value
    elif knob == "y_O2":
        _set_o2_feed(tgt, value)
    elif knob == "lambda_O2":
        # Keep the lab feed as it is set up
        stoich_o2 = 0.75 * _fe_metal_feed(tgt)
        tgt["gas_flow_mol"] = value * stoich_o2 / tgt["y_O2"]
    elif knob == "porosity":
        tgt["particle_porosity"] = value
    elif knob == "gas_flow":
        tgt["gas_flow_mol"] = base["gas_flow_mol"] * value
    elif knob == "solid_flow":
        tgt["solid_flow_mass"] = base["solid_flow_mass"] * value
    elif knob == "throughput":
        tgt["gas_flow_mol"] = base["gas_flow_mol"] * value
        tgt["solid_flow_mass"] = base["solid_flow_mass"] * value
    elif knob == "H":
        tgt["H"] = value
    return tgt


def abs_flow_of(tgt, knob):
    """
    Returns the absolute flow the step uses for the CSV
    """
    if knob == "gas_flow":
        return tgt["gas_flow_mol"]
    if knob == "lambda_O2":
        return tgt["gas_flow_mol"]
    if knob == "solid_flow":
        return tgt["solid_flow_mass"]
    if knob == "throughput":
        return tgt["gas_flow_mol"]
    return None


def report_step(mod, m, results, tgt, knob, v, path, drift=None):
    """
    Report one solved step, builds the point dict, check it and, print the json line 
    """
    report = endpoint(m, "dry")
    validity = mod._validity_banner(results)
    err_mass = results.get("err_mass")
    point = {
        "reactor": "dry", "knob": knob, "value": v,
        "abs_flow": abs_flow_of(tgt, knob), "path": path,
        "Ts_in": tgt["solid_T"], "Tg_in": tgt["gas_T"],
        "term": results.get("termination", "?"),
        "err_mass": err_mass, "gas_feasible": results.get("gas_feasible"),
        "banner_ok": validity == "", **report,
    }
    if drift is not None:
        point["anchor_drift_dX"], point["anchor_drift_dT"] = drift
    if knob == "lambda_O2":
        calc = report.get("lambda_O2")
        point["lambda_requested"] = v
        point["lambda_calculated"] = calc
        point["lambda_abs_error"] = abs(calc - v) if calc is not None else None
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
    ap.add_argument("--dp", type=float, default=None)
    args = ap.parse_args()
    values = [float(v) for v in args.values.split(",")]

    mod = load_module(common.LAB_SCRIPTS["dry"], "sens_dry")
    base = dict(mod.LAB)
    base["H"] = common.LAB_MATCH["dry"]["H"]
    base["n_orifice"] = common.LAB_MATCH["dry"]["n_orifice"]
    if args.dp is not None:
        base["particle_dia"] = dp_patch.to_m(args.dp)
    base_value = {"T_solid": base["solid_T"], "T_gas": base["gas_T"],
                  "y_O2": base["y_O2"],
                  "lambda_O2": (base["gas_flow_mol"] * base["y_O2"]
                                / (0.75 * _fe_metal_feed(base))),
                  "porosity": base["particle_porosity"],
                  "gas_flow": 1.0, "solid_flow": 1.0, "throughput": 1.0,
                  "H": base["H"]}[args.knob]
    if abs(values[0] - base_value) > 1e-9:
        sys.exit("The first value is not the base value")

    if args.dp is not None:
        m, results, anchor_path, _ = dp_patch.solve_lab_dp(
            mod, base, base["particle_dia"], "dry", verbose=True)
    else:
        m, results = mod.solve_case(base, verbose=True)
        anchor_path = "cold"
    report_step(mod, m, results, base, args.knob, values[0], anchor_path)
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

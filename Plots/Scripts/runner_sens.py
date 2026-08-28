"""
Knob script for wet and reduction lab reactors

Exactly one solve per process
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  
from runner_axial import endpoint, load_module  

# Knobs: model inputs used in sweeps
KNOBS = {
    "reduction": ["T_solid", "y_H2", "porosity", "gas_flow", "solid_flow"],
    "wet": ["T_solid", "porosity", "gas_flow", "solid_flow"],
}


def apply_knob(target, mod, knob, value):
    """
    Apply one knob in the lab dict
    """
    # Kinetics only uses the solid temperature 
    if knob == "T_solid":
        target["solid_T"] = value
    elif knob == "y_H2":
        target["y_H2"] = value
        target["y_N2"] = 1.0 - value
    elif knob == "porosity":
        target["particle_porosity"] = value
    elif knob == "gas_flow":
        target["gas_flow_mol"] = mod.LAB["gas_flow_mol"] * value
        return target["gas_flow_mol"]
    elif knob == "solid_flow":
        target["solid_flow_mass"] = mod.LAB["solid_flow_mass"] * value
        return target["solid_flow_mass"]
    else:
        raise ValueError(knob)
    return None


def main():
    """
    One model per run, it reads the knob and its value from the command line, solves it and then print the result
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--reactor", required=True, choices=sorted(KNOBS))
    ap.add_argument("--knob", required=True)
    ap.add_argument("--value", required=True, type=float)
    ap.add_argument("--ncont", type=int, default=None)
    args = ap.parse_args()
    mod = load_module(common.LAB_SCRIPTS[args.reactor], f"sens_{args.reactor}")
    target = dict(mod.LAB)
    target["H"] = common.LAB_MATCH[args.reactor]["H"]
    target["n_orifice"] = common.LAB_MATCH[args.reactor]["n_orifice"]
    abs_flow = apply_knob(target, mod, args.knob, args.value)
    solve_opts = {"n_cont": args.ncont} if args.ncont else {}
    m, results = mod.solve_case(target, verbose=False, **solve_opts)  # One solve per process
    report = endpoint(m, args.reactor)
    validity = getattr(mod, "_validity_banner", lambda res: "")(results)
    err_mass = results.get("err_mass")
    point = {
        "reactor": args.reactor, "knob": args.knob, "value": args.value,
        "abs_flow": abs_flow,
        "path": "cold+n_cont40" if args.ncont else "cold",
        "Ts_in": target["solid_T"],
        "term": results.get("termination", "?"), "err_mass": err_mass,
        "gas_feasible": results.get("gas_feasible"), "banner_ok": validity == "",
        **report,
    }
    point["valid"] = (point["n_bad"] == 0
                      and (point["gas_feasible"] is None or point["gas_feasible"])
                      and "optimal" in str(point["term"])
                      and err_mass is not None and float(err_mass) <= 0.1
                      and point["banner_ok"])
    print("POINT " + json.dumps(point), flush=True)


if __name__ == "__main__":
    main()

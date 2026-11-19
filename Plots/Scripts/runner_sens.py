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
import dp_patch
from runner_axial import endpoint, load_module

# Knobs: model inputs used in sweeps
KNOBS = {
    "reduction": ["T_solid", "T_gas", "y_H2", "lambda_H2", "porosity", "gas_flow",
                  "solid_flow", "throughput", "H"],
    "wet": ["T_solid", "T_gas", "y_H2O", "lambda_steam", "porosity", "gas_flow",
            "solid_flow", "throughput", "H"],
}

def _set_reactant_fraction(target, species, y):
    """
    Set one reactant mole fraction, N2 is the rest
    """
    traces = sum(target.get(f"y_{sp}", 0.0)
                 for sp in ("O2", "CO2", "H2O", "H2")
                 if sp != species)
    y_n2 = 1.0 - y - traces
    if y < 0.0 or y_n2 < 0.0:
        raise ValueError(f"Impossible error")
    target[f"y_{species}"] = y
    target["y_N2"] = y_n2


def _fe_metal_feed(target):
    """
    kg/s of Fe metal in the solid feed
    """
    return (target["solid_flow_mass"] * target.get("w_Fe", 0.0)
            / common.MW_S["Fe"])


def _oxygen_removal_feed(target):
    """
    mol/s of removable oxygen in the oxide feed
    """
    return target["solid_flow_mass"] * (
        3.0 * target.get("w_Fe2O3", 0.0) / common.MW_S["Fe2O3"]
        + 4.0 * target.get("w_Fe3O4", 0.0) / common.MW_S["Fe3O4"]
        + target.get("w_FeO", 0.0) / common.MW_S["FeO"])


def _set_nominal_gas(target, mod, reactor):
    """
    Keep the module's own feed composition, set the total flow
    """
    for sp in ("H2", "N2", "O2", "CO2", "H2O"):
        target[f"y_{sp}"] = mod.LAB[f"y_{sp}"]


def _lambda_key(reactor):
    """
    Which lambda column this reactor writes
    """
    return "lambda_H2" if reactor == "reduction" else "lambda_steam"


def apply_knob(target, mod, knob, value):
    """
    Apply one knob in the lab dict
    """
    # Kinetics only uses the solid temperature
    if knob == "T_solid":
        target["solid_T"] = value
    elif knob == "T_gas":
        target["gas_T"] = value
    elif knob == "y_H2":
        _set_reactant_fraction(target, "H2", value)
    elif knob == "lambda_H2":
        _set_nominal_gas(target, mod, "reduction")
        target["gas_flow_mol"] = value * _oxygen_removal_feed(target) / target["y_H2"]
        return target["gas_flow_mol"]
    elif knob == "y_H2O":
        _set_reactant_fraction(target, "H2O", value)
    elif knob == "lambda_steam":
        _set_nominal_gas(target, mod, "wet")
        target["gas_flow_mol"] = value * (4.0 / 3.0) * _fe_metal_feed(target) / target["y_H2O"]
        return target["gas_flow_mol"]
    elif knob == "porosity":
        target["particle_porosity"] = value
    elif knob == "H":
        target["H"] = value
    elif knob == "gas_flow":
        target["gas_flow_mol"] = mod.LAB["gas_flow_mol"] * value
        return target["gas_flow_mol"]
    elif knob == "solid_flow":
        target["solid_flow_mass"] = mod.LAB["solid_flow_mass"] * value
        return target["solid_flow_mass"]
    elif knob == "throughput":
        # Scales both flows together. lambda stays constant, u0/umf and the gas/solid contact times changes
        target["gas_flow_mol"] = mod.LAB["gas_flow_mol"] * value
        target["solid_flow_mass"] = mod.LAB["solid_flow_mass"] * value
        return target["gas_flow_mol"]
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
    ap.add_argument("--dp", type=float, default=None)
    args = ap.parse_args()
    mod = load_module(common.LAB_SCRIPTS[args.reactor], f"sens_{args.reactor}")
    target = dict(mod.LAB)
    target["H"] = common.LAB_MATCH[args.reactor]["H"]
    target["n_orifice"] = common.LAB_MATCH[args.reactor]["n_orifice"]
    abs_flow = apply_knob(target, mod, args.knob, args.value)
    if args.dp is not None:
        m, results, dp_path, _ = dp_patch.solve_lab_dp(
            mod, target, dp_patch.to_m(args.dp), args.reactor,
            verbose=False, ncont=args.ncont)
    else:
        dp_path = None
        solve_opts = {"n_cont": args.ncont} if args.ncont else {}
        # One solve per process
        m, results = mod.solve_case(target, verbose=False, **solve_opts)
    report = endpoint(m, args.reactor)
    validity = getattr(mod, "_validity_banner", lambda res: "")(results)
    err_mass = results.get("err_mass")
    point = {
        "reactor": args.reactor, "knob": args.knob, "value": args.value,
        "abs_flow": abs_flow,
        "path": ("cold+n_cont40" if args.ncont else "cold")
                if dp_path is None else
                dp_path + ("+n_cont40" if args.ncont else ""),
        "Ts_in": target["solid_T"], "Tg_in": target["gas_T"],
        "term": results.get("termination", "?"), "err_mass": err_mass,
        "gas_feasible": results.get("gas_feasible"), "banner_ok": validity == "",
        **report,
    }
    if args.knob in ("lambda_H2", "lambda_steam"):
        calc = report.get(_lambda_key(args.reactor))
        point["lambda_requested"] = args.value
        point["lambda_calculated"] = calc
        point["lambda_abs_error"] = abs(calc - args.value) if calc is not None else None
    point["valid"] = (point["n_bad"] == 0
                      and (point["gas_feasible"] is None or point["gas_feasible"])
                      and "optimal" in str(point["term"])
                      and err_mass is not None and float(err_mass) <= 0.1
                      and point["banner_ok"])
    print("POINT " + json.dumps(point), flush=True)


if __name__ == "__main__":
    main()

"""
Runs one reactor solve 
'Axial' writes the CSV profile. 'Point'  prints a line summary for sweeps
"""
import argparse
import contextlib
import csv
import importlib.util
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  

# Physical constants, identical in every reactor script
MW_S = {"Fe2O3": 0.15969, "Fe3O4": 0.231533, "FeO": 0.071844,   # kg/mol
        "Fe": 0.055845, "Al2O3": 0.10196}
GAS_SP = ["H2", "H2O", "N2", "O2", "CO2"]
SOLID_SP = ["Fe2O3", "Fe3O4", "FeO", "Fe", "Al2O3"]

# Reactant gases
REACTANT = {"reduction": "H2", "wet": "H2O", "dry": "O2"}


def load_module(path, name):
    """
    Imports the .py from its file path 
    
    """
    file_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(file_spec)
    file_spec.loader.exec_module(module)
    return module


def sorted_domain(b):

    """
    The z points of the bed. From the bottom to the top
    """

    return sorted(b.length_domain, key=float)


def local_solid_x(b, reactor):
    """
    Solid conversion X at each height z
    """
    flow_in = b.solid_inlet.flow_mass[0].value
    H = b.bed_height.value
    if reactor == "reduction":
        ref = flow_in * b.solid_inlet.mass_frac_comp[0, "Fe2O3"].value / MW_S["Fe2O3"]
        key = "Fe2O3"
    else:
        ref = flow_in * b.solid_inlet.mass_frac_comp[0, "Fe"].value / MW_S["Fe"]
        key = "Fe"
    out = {}
    for x in sorted_domain(b):
        solid_props = b.solid_emulsion.properties[0, x]
        n_local = solid_props.flow_mass.value * solid_props.mass_frac_comp[key].value / MW_S[key]
        out[x] = (ref - n_local) / ref * 100 if ref > 0 else 0.0
    return out, H


def local_gas_x(b, reactor):

    """
    Gas conversion X at each z from the total reactant flow, emulsion and bubble
    """
    comp = REACTANT[reactor]
    n_in = (b.gas_inlet.flow_mol[0].value
            * b.gas_inlet.mole_frac_comp[0, comp].value)
    out = {}
    for x in sorted_domain(b):
        flow_e = b.gas_emulsion.properties[0, x].flow_mol.value
        flow_b = b.bubble.properties[0, x].flow_mol.value
        y_e = b.gas_emulsion.properties[0, x].mole_frac_comp[comp].value
        y_b = b.bubble.properties[0, x].mole_frac_comp[comp].value
        n_local = flow_e * y_e + flow_b * y_b
        out[x] = (n_in - n_local) / n_in * 100 if n_in > 0 else 0.0
    return out, comp


def extract_rows(m, reactor):

    """
    One row per z point with every profile the plots use
    """
    b = m.fs.BFB
    bed_points = sorted_domain(b)
    solid_x, H = local_solid_x(b, reactor)
    gas_x, comp = local_gas_x(b, reactor)
    rows = []
    for x_norm in bed_points:
        gas_e = b.gas_emulsion.properties[0, x_norm]
        gas_b = b.bubble.properties[0, x_norm]
        solid_props = b.solid_emulsion.properties[0, x_norm]
        row = {
            "z": float(x_norm) * H, "x_norm": float(x_norm),
            "T_gas": gas_e.temperature.value, "T_bub": gas_b.temperature.value,
            "T_sol": solid_props.temperature.value, "P_emul": gas_e.pressure.value,
            "F_emul": gas_e.flow_mol.value, "F_bub": gas_b.flow_mol.value,
            "C_react": gas_e.dens_mol_comp[comp].value,
            "X_solid": solid_x[x_norm], "X_gas": gas_x[x_norm],
            "db": b.bubble_diameter[0, x_norm].value,
            "db_max": b.bubble_diameter_max[0, x_norm].value,
            "delta": b.delta[0, x_norm].value,
        }
        for specie in GAS_SP:
            row[f"y_emul_{specie}"] = gas_e.mole_frac_comp[specie].value
            row[f"y_bub_{specie}"] = gas_b.mole_frac_comp[specie].value
        for specie in SOLID_SP:
            row[f"w_{specie}"] = solid_props.mass_frac_comp[specie].value
        rows.append(row)
    return rows, H


def endpoint(m, reactor):

    """
    Outlet conversion using the inlets and outlets flows
    """
    b = m.fs.BFB
    flow_in = b.solid_inlet.flow_mass[0].value
    flow_out = b.solid_outlet.flow_mass[0].value
    if reactor == "reduction":
        n_in = flow_in * b.solid_inlet.mass_frac_comp[0, "Fe2O3"].value / MW_S["Fe2O3"]
        n_out = flow_out * b.solid_outlet.mass_frac_comp[0, "Fe2O3"].value / MW_S["Fe2O3"]
        x_solid = (n_in - n_out) / n_in * 100 if n_in > 0 else 0.0
    else:
        n_in = flow_in * b.solid_inlet.mass_frac_comp[0, "Fe"].value / MW_S["Fe"]
        n_out = flow_out * b.solid_outlet.mass_frac_comp[0, "Fe"].value / MW_S["Fe"]
        x_solid = (n_in - n_out) / n_in * 100 if n_in > 0 else 0.0
    comp = REACTANT[reactor]
    n_in = b.gas_inlet.flow_mol[0].value * b.gas_inlet.mole_frac_comp[0, comp].value
    n_out = b.gas_outlet.flow_mol[0].value * b.gas_outlet.mole_frac_comp[0, comp].value
    x_gas = (n_in - n_out) / n_in * 100 if n_in > 0 else 0.0

    n_bad = 0
    tol = 1e-4
    for x_norm in sorted_domain(b):
        for specie in m.fs.gas_properties.component_list:
            y_frac = b.gas_emulsion.properties[0, x_norm].mole_frac_comp[specie].value
            if y_frac < -tol or y_frac > 1 + tol:
                n_bad += 1
        for specie in m.fs.solid_properties.component_list:
            w_frac = b.solid_emulsion.properties[0, x_norm].mass_frac_comp[specie].value
            if w_frac < -tol:
                n_bad += 1
    return {
        "X_solid": x_solid, "X_gas": x_gas, "n_bad": n_bad,
        "T_gas_out": b.gas_outlet.temperature[0].value,
        "T_solid_out": b.solid_outlet.temperature[0].value,
        "D": b.bed_diameter.value, "H": b.bed_height.value,
    }


def solve_lab(reactor, h_override, norif_override, verbose, ncont=None):

    """
    Solves one lab case, it starts from the model's own lab inputs and
    overrides only H, n_orifice and, n_cont 
    """

    module = load_module(common.LAB_SCRIPTS[reactor], f"{reactor}")
    target = dict(module.LAB)
    if h_override is not None:
        target["H"] = h_override
    if norif_override is not None:
        target["n_orifice"] = norif_override
    if ncont is not None:
        m, r = module.solve_case(target, n_cont=ncont, verbose=verbose)
    else:
        m, r = module.solve_case(target, verbose=verbose)
    term = r.get("termination", "?")
    err_mass = r.get("err_mass", None)
    gas_feasible = r.get("gas_feasible", None)
    return m, term, err_mass, gas_feasible


def solve_ind(reactor):
    """
    Solves the industrial reference case 
    """

    mod = load_module(common.IND_SCRIPTS[reactor], f"{reactor}")
    with contextlib.redirect_stdout(io.StringIO()):
        m = mod.main()
    return m, "optimal", None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reactor", required=True,
                    choices=["reduction", "wet", "dry"])
    ap.add_argument("--scale", required=True, choices=["lab", "ind"])
    ap.add_argument("--H", type=float, default=None)
    ap.add_argument("--norif", type=float, default=None)
    ap.add_argument("--mode", default="axial", choices=["axial", "point"])
    ap.add_argument("--ncont", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()


    if args.scale == "lab":
        bed_h = args.H if args.H is not None else common.LAB_MATCH[args.reactor]["H"]
        orifice_dens = (args.norif if args.norif is not None
               else common.LAB_MATCH[args.reactor]["n_orifice"])
        model, term, err_mass, gas_feasible = solve_lab(args.reactor, bed_h, orifice_dens,
                                                    args.verbose,
                                                    ncont=args.ncont)
    else:
        bed_h, orifice_dens = None, None
        model, term, err_mass, gas_feasible = solve_ind(args.reactor)

    ep = endpoint(model, args.reactor)
    point = {"reactor": args.reactor, "scale": args.scale,
             "term": term, "err_mass": err_mass,
             "gas_feasible": gas_feasible, **ep}
    point["valid"] = (point["n_bad"] == 0
                      and (gas_feasible is None or gas_feasible)
                      and "optimal" in str(term))

    if args.mode == "axial":
        rows, H = extract_rows(model, args.reactor)
        common.ensure_dirs()
        with open(common.axial_csv(args.reactor, args.scale), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        meta = dict(point)
        meta["source"] = (common.LAB_SCRIPTS if args.scale == "lab"
                          else common.IND_SCRIPTS)[args.reactor]
        with open(common.axial_meta(args.reactor, args.scale), "w") as f:
            json.dump(meta, f, indent=2)

    print("POINT " + json.dumps(point), flush=True)


if __name__ == "__main__":
    main()

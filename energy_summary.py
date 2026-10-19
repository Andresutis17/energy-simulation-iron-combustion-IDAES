

import argparse
import csv
import json
import os
import runpy
import subprocess
import sys
import traceback

sys.stdout.reconfigure(line_buffering=True)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_layout():
    
    root = _HERE
    while root != os.path.dirname(root):
        if (os.path.isdir(os.path.join(root, "Reliable_models", "Industrial"))
                and os.path.isdir(os.path.join(root, "custom_properties"))):
            return root
        root = os.path.dirname(root)
    raise SystemExit("fail")


ROOT = _find_layout()
MODEL_DIR = os.path.join(ROOT, "Reliable_models", "Industrial")
LAB_DIR = os.path.join(ROOT, "Reliable_models", "Lab")
SCRIPTS = {
    "dry": ("Dry_Oxidation_Reactor.py", "dry_reactions"),
    "wet": ("Wet_Oxidation_Reactor.py", "wet_reactions"),
    "red": ("Reduction_Reactor.py", "reduction_reactions"),
}
LAB_MODULES = {
    "dry": ("Dry_Oxidation_Reactor_Lab.py", "dry_reactions"),
    "wet": ("Wet_Oxidation_Reactor_Lab.py", "wet_reactions"),
    "red": ("Reduction_Reactor_Lab.py", "reduction_reactions"),
}
_OUT = os.path.join(ROOT, "outputs")
os.makedirs(_OUT, exist_ok=True)
sys.path.insert(0, MODEL_DIR)

from pyomo.environ import value

AREA_RATIO = (0.054 / 6.5) ** 2   # Industrial scale to lab

# kg/mol
MW = {"Fe": 0.055845, "FeO": 0.071844, "Fe3O4": 0.231533,
      "Fe2O3": 0.159688, "Al2O3": 0.101961}
MWG = {"H2O": 0.018015, "H2": 0.002016, "N2": 0.028014,
       "O2": 0.031998, "CO2": 0.044009}

LHV_H2 = 241.8264e3   # J/mol water gas
HHV_H2 = 285.84e3     # J/mol water liquid


def _capture_output(log_path, action, *args, **kwargs):
   
    log = open(log_path, "w", buffering=1)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = log, log
    try:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        return action(*args, **kwargs)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        sys.stdout, sys.stderr = old_stdout, old_stderr
        log.close()



def dump_model(m, rxn_name, script, tag, suffix="", extra=None):
    """Reads a solved model into outputs/_energy_<tag><suffix>.json """
    bed = m.fs.BFB
    rxn = getattr(m.fs, rxn_name)

    # Package stoichiometry. Tuple dict to per-reaction nested dict
    stoich = {}
    for (rxn_id, phase, species), coeff in rxn.rate_reaction_stoichiometry.items():
        stoich.setdefault(rxn_id, {"Vap": {}, "Sol": {}})[phase][species] = float(coeff)
    dh_rxn = {r: float(value(rxn.dh_rxn[r])) for r in stoich}

    def gas_port(p):
        return {"F": float(value(p.flow_mol[0])),
                "T": float(value(p.temperature[0])),
                "y": {j: float(value(p.mole_frac_comp[0, j])) for j in MWG}}

    def solid_port(p):
        return {"F": float(value(p.flow_mass[0])),
                "T": float(value(p.temperature[0])),
                "w": {k: float(value(p.mass_frac_comp[0, k])) for k in MW}}

    axial = {"x": [], "T_emul": [], "T_bub": [], "T_sol": [], "F_emul": [],
             "F_bub": [], "m_sol": [], "cp_emul": [], "cp_bub": [], "cp_sol": [],
             "h_emul": [], "h_bub": [], "h_sol": []}
    
    for x in bed.length_domain:
        emulsion = bed.gas_emulsion.properties[0, x]
        bubble = bed.bubble.properties[0, x]
        solid = bed.solid_emulsion.properties[0, x]
        axial["x"].append(float(x))
        axial["T_emul"].append(float(value(emulsion.temperature)))
        axial["T_bub"].append(float(value(bubble.temperature)))
        axial["T_sol"].append(float(value(solid.temperature)))
        axial["F_emul"].append(float(value(emulsion.flow_mol)))
        axial["F_bub"].append(float(value(bubble.flow_mol)))
        axial["m_sol"].append(float(value(solid.flow_mass)))
        axial["cp_emul"].append(float(value(emulsion.cp_mol)))
        axial["cp_bub"].append(float(value(bubble.cp_mol)))
        axial["cp_sol"].append(float(value(solid.cp_mass)))
        axial["h_emul"].append(float(value(emulsion.enth_mol)))
        axial["h_bub"].append(float(value(bubble.enth_mol)))
        axial["h_sol"].append(float(value(solid.enth_mass)))

    data = {"tag": tag, "script": script,
            "gas_in": gas_port(bed.gas_inlet),
            "gas_out": gas_port(bed.gas_outlet),
            "solid_in": solid_port(bed.solid_inlet),
            "solid_out": solid_port(bed.solid_outlet),
            "stoich": stoich, "dh_rxn": dh_rxn, "axial": axial}
    if extra:
        data.update(extra)
    with open(os.path.join(_OUT, f"_energy_{tag}{suffix}.json"), "w") as f:
        json.dump(data, f, indent=1)


def _solve_in_subprocess(scale, tag, i, n):
    """One solve per process"""
    cmd = [sys.executable, os.path.abspath(__file__),
           "--reactor", tag, "--scale", scale, "--i", str(i), "--n", str(n)]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        log = f"_energy_{tag}{'_lab' if scale == 'lab' else ''}_run.log"
        raise SystemExit(f"\n[energy] {scale} {tag} fail "
                         f"{res.returncode})")


def run_reactor(tag, i, n):

    """Solve one industrial script and write its solved state
        to outputs/_energy_<tag>.json"""


    script, rxn_name = SCRIPTS[tag]
    log_path = os.path.join(_OUT, f"_energy_{tag}_run.log")
    script_globals = _capture_output(log_path, runpy.run_path,
                                     os.path.join(MODEL_DIR, script),
                                     run_name="__main__")
    dump_model(script_globals["m"], rxn_name, script, tag)


def lab_target(mod, tag):
    """Lab target is the modules certified match point"""
    target = dict(mod.LAB)
    if tag == "red":
        target["n_orifice"] = 2500
    if tag == "dry":
        industrial = mod.IND
        area_ratio = (target["D"] / industrial["D"]) ** 2
        target.update({"gas_flow_mol": industrial["gas_flow_mol"] * area_ratio,
                       "solid_flow_mass": industrial["solid_flow_mass"] * area_ratio,
                       "gas_T": industrial["gas_T"],
                       "solid_T": industrial["solid_T"],
                       "particle_dia": industrial["particle_dia"],
                       "velocity_mf": industrial["velocity_mf"],
                       "H": 0.90})
    return target


def run_reactor_lab(tag, i, n):
    """Solve one lab twin with its certified match recipe and write outputs/_energy_<tag>_lab.json"""

    import importlib.util
    script, rxn_name = LAB_MODULES[tag]
    log_path = os.path.join(_OUT, f"_energy_{tag}_lab_run.log")

    def solve():
        spec = importlib.util.spec_from_file_location(
            f"_energylab_{tag}", os.path.join(LAB_DIR, script))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        m, results = mod.solve_case(lab_target(mod, tag), n_cont=mod.N_CONT)
        return m, results, mod

    m, results, mod = _capture_output(log_path, solve)
    dump_model(m, rxn_name, script, tag, suffix="_lab",
               extra={"results": results,
                      "valid_banner": (mod._validity_banner(results) or ""
                                       if hasattr(mod, "_validity_banner") else "")})



def load(tag):
    with open(os.path.join(_OUT, f"_energy_{tag}.json")) as f:
        return json.load(f)


def deltas(data):
    """Species molar flow deltas out - in, mol/s. The solid comes in
       kg/s and is converted with the species molar masses"""

    d_gas = {j: data["gas_out"]["F"] * data["gas_out"]["y"][j]
             - data["gas_in"]["F"] * data["gas_in"]["y"][j] for j in MWG}
    d_sol = {k: data["solid_out"]["F"] * data["solid_out"]["w"][k] / MW[k]
             - data["solid_in"]["F"] * data["solid_in"]["w"][k] / MW[k] for k in MW}
    return d_gas, d_sol


def pct(num, den):
    """Relative difference in %"""
    return abs(num - den) / abs(den) * 100 if abs(den) > 1e-9 else 0.0


def extents(tag, d_gas, d_sol):
    """Reaction extents from species deltas mol/s """

    ext, checks = {}, []
    if tag == "dry":          # DOR1: 2Fe + 1.5O2 -> Fe2O3
        ext["R1"] = -d_sol["Fe"] / 2.0
        checks.append(("Fe2O3 formed = extent", d_sol["Fe2O3"], ext["R1"]))
        checks.append(("O2 consumed = 1.5*extent", -d_gas["O2"], 1.5 * ext["R1"]))

    elif tag == "wet":        # WOR1: Fe+H2O->FeO+H2 ; WOR": 3FeO+H2O->Fe3O4+H2
        ext["OW1"] = -d_sol["Fe"]
        ext["OW2"] = d_sol["Fe3O4"]
        checks.append(("H2 produced = OW1+OW2", d_gas["H2"], ext["OW1"] + ext["OW2"]))
        checks.append(("H2O consumed = OW1+OW2", -d_gas["H2O"], ext["OW1"] + ext["OW2"]))
        checks.append(("FeO balance = OW1-3*OW2", d_sol["FeO"], ext["OW1"] - 3 * ext["OW2"]))

    else:                     # RR1: 3Fe2O3+H2->2Fe3O4+H2O ; RR2 ; RR3
        ext["RD1"] = -d_sol["Fe2O3"] / 3.0
        ext["RD2"] = 2.0 * ext["RD1"] - d_sol["Fe3O4"]
        ext["RD3"] = 3.0 * ext["RD2"] - d_sol["FeO"]
        checks.append(("Fe formed = RD3", d_sol["Fe"], ext["RD3"]))
        checks.append(("H2 consumed = RD1+RD2+RD3", -d_gas["H2"],
                       ext["RD1"] + ext["RD2"] + ext["RD3"]))
        checks.append(("H2O produced = RD1+RD2+RD3", d_gas["H2O"],
                       ext["RD1"] + ext["RD2"] + ext["RD3"]))
    return ext, checks


def sensible(data):
    """Stream energy balance (Watt): enthalpy flow out minus in per stream."""

    axial = data["axial"]
    n = len(axial["x"]) - 1
    q_gas = ((axial["F_emul"][n] * axial["h_emul"][n]
              + axial["F_bub"][n] * axial["h_bub"][n])
             - (axial["F_emul"][0] * axial["h_emul"][0]
                + axial["F_bub"][0] * axial["h_bub"][0]))
    q_sol = (axial["m_sol"][n] * axial["h_sol"][n]
             - axial["m_sol"][0] * axial["h_sol"][0])
    return q_gas, q_sol


def q_rxn(ext, dh_rxn):
    """Reaction heat (Watt). positive = released"""
    return -sum(ext[r] * dh_rxn[r] for r in ext)


def metrics(tag):
    """Compute one reactors energy numbers """
    data = load(tag)
    d_gas, d_sol = deltas(data)
    ext, checks = extents(tag, d_gas, d_sol)
    q_gas, q_sol = sensible(data)
    heat = q_rxn(ext, data["dh_rxn"])
    return {"data": data, "ext": ext, "checks": checks, "heat": heat,
            "q_gas": q_gas, "q_sol": q_sol,
            "closure": pct(q_gas + q_sol, heat)}


def table():

    """Prints the MW table industrial and writes the CSV"""
    rows = []
    for tag in ("red", "wet", "dry"):
        met = metrics(tag)
        d_gas, _ = deltas(met["data"])
        n_h2 = (-d_gas["H2"]) if tag == "red" else (d_gas["H2"] if tag == "wet" else 0.0)
        label = {"red": "Reduction (Fe2O3 + H2)",
                 "wet": "Steam oxidation (Fe + H2O)",
                 "dry": "Air oxidation (Fe + O2)"}[tag]
        rows.append((label, met["data"], n_h2, met["heat"],
                     met["q_gas"], met["q_sol"]))

    hdr = (f"{'':34s}{'Reduction':>13s}{'Steam ox':>13s}{'Air ox':>13s}")
    print(hdr)
    gas_in_txt = []
    for _, data, *_ in rows:
        y = data["gas_in"]["y"]
        if y.get("O2", 0) > 0.1:
            gas_in_txt.append(f"{data['gas_in']['F']:.0f} mol/s air "
                              f"({y['O2'] * 100:.0f}% O2) @{data['gas_in']['T']:.0f} K")
        else:
            main = max(y, key=y.get)
            gas_in_txt.append(f"{data['gas_in']['F']:.0f} mol/s {main} "
                              f"{y[main] * 100:.0f}% @{data['gas_in']['T']:.0f} K")
    print(f"{'Gas feed':34s}{gas_in_txt[0]:>13s}{gas_in_txt[1]:>13s}{gas_in_txt[2]:>13s}")
    
    def line(name, vals, fmt="{:+8.2f}"):
        print(f"{name:34s}" + "".join(f"{fmt.format(v):>13s}" for v in vals))

    line("H2 consumed(+)/produced(-), mol/s",
         [rows[0][2], -rows[1][2], 0.0], "{:+8.1f}")
    line("H2 power, LHV, MW (+con/-prod)",
         [rows[0][2] * LHV_H2 / 1e6, -rows[1][2] * LHV_H2 / 1e6, 0.0], "{:+8.2f}")
    line("H2 power, HHV, MW (+con/-prod)",
         [rows[0][2] * HHV_H2 / 1e6, -rows[1][2] * HHV_H2 / 1e6, 0.0], "{:+8.2f}")
    line("Reaction heat (+released), MW",
         [row[3] / 1e6 for row in rows], "{:+8.2f}")
    line("Gas enthalpy change, MW",
         [row[4] / 1e6 for row in rows], "{:+8.2f}")
    line("Solid enthalpy change, MW",
         [row[5] / 1e6 for row in rows], "{:+8.2f}")

    with open(os.path.join(_OUT, "energy_summary_industrial.csv"), "w",
              newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reactor", "quantity", "value", "unit"])
        for label, data, n_h2, heat, q_gas, q_sol in rows:
            short = label.split(" ")[0]
            writer.writerow([short, "H2 consumed(+)/produced(-)", n_h2, "mol/s"])
            writer.writerow([short, "H2 power LHV", n_h2 * LHV_H2 / 1e6, "MW"])
            writer.writerow([short, "H2 power HHV", n_h2 * HHV_H2 / 1e6, "MW"])
            writer.writerow([short, "reaction heat (+released)", heat / 1e6, "MW"])
            writer.writerow([short, "gas enthalpy change", q_gas / 1e6, "MW"])
            writer.writerow([short, "solid enthalpy change", q_sol / 1e6, "MW"])


def table_lab():

    """Lab vs industrial check. Both ratios should sit at 1 within the
    1 pp conversion difference between match heights"""

    out_rows = []
    for tag in ("red", "wet", "dry"):
        ind, lab = load(tag), load(f"{tag}_lab")
        gas_i, sol_i = deltas(ind)
        gas_l, sol_l = deltas(lab)
        ext_i, _ = extents(tag, gas_i, sol_i)
        ext_l, _ = extents(tag, gas_l, sol_l)
        q_i = q_rxn(ext_i, ind["dh_rxn"])
        q_l = q_rxn(ext_l, lab["dh_rxn"])
        n_i = (-gas_i["H2"]) if tag == "red" else (gas_i["H2"] if tag == "wet" else 0.0)
        n_l = (-gas_l["H2"]) if tag == "red" else (gas_l["H2"] if tag == "wet" else 0.0)
        q_gas_l, q_sol_l = sensible(lab)
        ratio_q = q_l / (q_i * AREA_RATIO)
        ratio_h2 = (n_l / n_i) / AREA_RATIO if n_i else float("nan")
        lab_res = lab["results"]
        name = {"red": "Reduction", "wet": "Steam ox ", "dry": "Air ox   "}[tag]
        h2_txt = (f"  H2 ratio {ratio_h2:5.2f}" if n_i else "")
        print(f"  {name} (H={lab_res['target']['H']:.2f} m):  "
              f"Q_lab {q_l / 1e3:6.3f} kW   heat ratio {ratio_q:5.3f}{h2_txt}"
              f"   closure {pct(q_gas_l + q_sol_l, q_l):.3f} %")
        banner = lab.get("valid_banner", "")
        if banner:
            print(f" Validity {banner}")
        out_rows.append([tag, q_i / 1e6, q_l, ratio_q, n_i, n_l, ratio_h2])

    with open(os.path.join(_OUT, "energy_summary_lab.csv"), "w",
              newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reactor", "Q_ind_MW", "Q_lab_W", "ratio_Q_lab_over_indAR",
                         "nH2_ind_mol_s", "nH2_lab_mol_s", "ratio_nH2_lab_over_indAR"])
        writer.writerows(out_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reactor", choices=["dry", "wet", "red"])
    parser.add_argument("--scale", choices=["ind", "lab"], default="ind")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--table-lab", action="store_true")
    parser.add_argument("--i", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--n", type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if not (args.reactor or args.table or args.table_lab):
           
            for i, tag in enumerate(("red", "wet", "dry"), 1):
                _solve_in_subprocess("ind", tag, i, 6)
            table()
            for i, tag in enumerate(("red", "wet", "dry"), 4):
                _solve_in_subprocess("lab", tag, i, 6)
            table_lab()
        else:
            if args.reactor:
                if args.scale == "lab":
                    run_reactor_lab(args.reactor, args.i, args.n)
                else:
                    run_reactor(args.reactor, args.i, args.n)
            if args.table:
                table()
            if args.table_lab:
                table_lab()
    except Exception:
        print("\n Driver error:")
        traceback.print_exc()

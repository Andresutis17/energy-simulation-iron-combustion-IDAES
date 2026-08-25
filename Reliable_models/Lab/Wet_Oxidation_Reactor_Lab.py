"""
Wet oxidation BFB reactor at LAB scale
"""
import sys
import os
import json
import argparse

import sys, os
here_path = os.path.dirname(os.path.abspath(__file__))
while here_path != os.path.dirname(here_path):
    if os.path.isdir(os.path.join(here_path, "custom_properties")):
        sys.path.insert(0, here_path)
        break
    here_path = os.path.dirname(here_path)

from pyomo.environ import value, Var
from idaes.core import FlowsheetBlock
from idaes.core.util import scaling as iscale
from idaes.core.solvers import get_solver
import idaes.logger as idaeslog

idaeslog.getLogger("idaes").setLevel(idaeslog.WARNING)
from idaes.models_extra.gas_solid_contactors.unit_models.bubbling_fluidized_bed import (
    BubblingFluidizedBed,
)
from custom_properties.gas_phase_thermo import CustomGasPhaseParameterBlock
from custom_properties.solid_phase_thermo import CustomSolidPhaseParameterBlock
from custom_properties.oxi_wet_reactions import OxiWetReactionParameterBlock

#  Industrial base case
IND = {
    "D": 6.5, "H": 5.0, "n_orifice": 2500,
    "gas_flow_mol": 300.0, "solid_flow_mass": 10.0,
    "gas_T": 1050.0, "solid_T": 1173.9, "P": 1.0e5,
    "y_H2O": 0.99, "y_N2": 0.01, "y_O2": 1e-5, "y_CO2": 1e-5, "y_H2": 1e-5,
    "w_Fe": 1.0, "w_Al2O3": 0.0, "w_Fe2O3": 0.0, "w_Fe3O4": 1e-5, "w_FeO": 1e-5,
    "particle_dia": 1.5e-3, "velocity_mf": 0.039624,
    "voidage_mf": 0.45, "voidage": 0.50, "particle_porosity": 0.27,
}

# Lab reactor scale
AREA_RATIO = (0.054 / IND["D"]) ** 2          
H_MATCH = 0.70
LAB = {
    "D": 0.054, "H": H_MATCH, "n_orifice": 2500,
    "gas_flow_mol": IND["gas_flow_mol"] * AREA_RATIO,      
    "solid_flow_mass": IND["solid_flow_mass"] * AREA_RATIO,  
}

# Inputs that are identical to industrial case
LAB.update({k: IND[k] for k in [
    "gas_T", "solid_T", "P",
    "y_H2O", "y_N2", "y_O2", "y_CO2", "y_H2",
    "w_Fe", "w_Al2O3", "w_Fe2O3", "w_Fe3O4", "w_FeO",
    "particle_dia", "velocity_mf", "voidage_mf", "voidage", "particle_porosity",
]})


N_CONT = 20 # continuation steps industrial -> target


# MW and atom counts per species
MW_s = {"Fe2O3": 0.15969, "Fe3O4": 0.231533, "FeO": 0.071844,
        "Fe": 0.055845, "Al2O3": 0.10196}
MW_g = {"H2": 0.002016, "N2": 0.028, "O2": 0.032, "CO2": 0.044, "H2O": 0.018}
n_Fe = {"Fe2O3": 2, "Fe3O4": 3, "FeO": 1, "Fe": 1, "Al2O3": 0}
n_O = {"Fe2O3": 3, "Fe3O4": 4, "FeO": 1, "Fe": 0, "Al2O3": 3}


# Build model
def build_model(finite_elements=20):
    from pyomo.environ import ConcreteModel
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.gas_properties = CustomGasPhaseParameterBlock()
    m.fs.solid_properties = CustomSolidPhaseParameterBlock()
    m.fs.wet_reactions = OxiWetReactionParameterBlock(
        solid_property_package=m.fs.solid_properties,
        gas_property_package=m.fs.gas_properties,
    )
    m.fs.BFB = BubblingFluidizedBed(
        flow_type="co_current",
        finite_elements=finite_elements,
        transformation_method="dae.finite_difference",
        gas_phase_config={"property_package": m.fs.gas_properties},
        solid_phase_config={"property_package": m.fs.solid_properties,
                            "reaction_package": m.fs.wet_reactions},
    )
    return m


def fix_intensive(m, p):

    """
    Fix everything that is identical at industrial and lab scale
    """
    m.fs.solid_properties.particle_dia.fix(p["particle_dia"])
    m.fs.solid_properties.velocity_mf.fix(p["velocity_mf"])
    m.fs.solid_properties.voidage_mf.fix(p["voidage_mf"])
    m.fs.solid_properties.voidage.fix(p["voidage"])
    b = m.fs.BFB
    b.gas_inlet.pressure[0].fix(p["P"])
    b.gas_inlet.mole_frac_comp[0, "H2O"].fix(p["y_H2O"])
    b.gas_inlet.mole_frac_comp[0, "N2"].fix(p["y_N2"])
    b.gas_inlet.mole_frac_comp[0, "O2"].fix(p["y_O2"])
    b.gas_inlet.mole_frac_comp[0, "CO2"].fix(p["y_CO2"])
    b.gas_inlet.mole_frac_comp[0, "H2"].fix(p["y_H2"])
    b.solid_inlet.particle_porosity[0].fix(p["particle_porosity"])
    b.solid_inlet.mass_frac_comp[0, "Fe"].fix(p["w_Fe"])
    b.solid_inlet.mass_frac_comp[0, "Al2O3"].fix(p["w_Al2O3"])
    b.solid_inlet.mass_frac_comp[0, "Fe2O3"].fix(p["w_Fe2O3"])
    b.solid_inlet.mass_frac_comp[0, "Fe3O4"].fix(p["w_Fe3O4"])
    b.solid_inlet.mass_frac_comp[0, "FeO"].fix(p["w_FeO"])


def set_case(m, D, H, norf, gas, sol, Tg, Ts):

    """
    Fix everything that differs between industrial and lab scale
    """
    b = m.fs.BFB
    b.bed_diameter.fix(D)
    b.bed_height.fix(H)
    b.number_orifice.fix(norf)
    b.gas_inlet.flow_mol[0].fix(gas)
    b.gas_inlet.temperature[0].fix(Tg)
    b.solid_inlet.flow_mass[0].fix(sol)
    b.solid_inlet.temperature[0].fix(Ts)


def solve_case(target, n_cont=N_CONT, verbose=False, finite_elements=20):

    """
    Build a new model and solve target via industrial init + temperature
    ramp + continuation
    
    """
    solver = get_solver()
    m = build_model(finite_elements=finite_elements)
    fix_intensive(m, target)
    b = m.fs.BFB

    def _try_solve(tag):
        try:
            res = solver.solve(b, tee=False)
            term = str(res.solver.termination_condition)
        except Exception as e:  
            term = f"EXC:{type(e).__name__}"
        if verbose:
            print(f"  [{tag}] -> {term}", flush=True)
        return term

    # Stage 1: Industrial geometry and flows
    set_case(m, IND["D"], IND["H"], IND["n_orifice"],
             IND["gas_flow_mol"], IND["solid_flow_mass"],
             IND["gas_T"], IND["solid_T"])
    gas_args = {"flow_mol": IND["gas_flow_mol"], "temperature": IND["solid_T"],
                "pressure": IND["P"],
                "mole_frac": {"H2O": IND["y_H2O"], "N2": IND["y_N2"],
                              "O2": IND["y_O2"], "CO2": IND["y_CO2"],
                              "H2": IND["y_H2"]}}
    sol_args = {"flow_mass": IND["solid_flow_mass"],
                "particle_porosity": IND["particle_porosity"],
                "temperature": IND["solid_T"],
                "mass_frac": {"Fe": IND["w_Fe"], "Al2O3": IND["w_Al2O3"],
                              "Fe2O3": IND["w_Fe2O3"], "Fe3O4": IND["w_Fe3O4"],
                              "FeO": IND["w_FeO"]}}
    iscale.calculate_scaling_factors(m)
    for v in m.fs.BFB.component_data_objects(Var, descend_into=True):
        if "frac_comp" in v.name or "porosity" in v.name:
            v.setlb(0)
    b.initialize(outlvl=idaeslog.CRITICAL,
                 gas_phase_state_args=gas_args, solid_phase_state_args=sol_args)
    _try_solve("init industrial")

    # Stage 2: continuation industrial -> target 
    for step in range(1, n_cont + 1):
        path_frac = step / n_cont
        D = IND["D"] * (target["D"] / IND["D"]) ** path_frac
        H = IND["H"] + (target["H"] - IND["H"]) * path_frac
        gas = IND["gas_flow_mol"] * (target["gas_flow_mol"] / IND["gas_flow_mol"]) ** path_frac
        sol = IND["solid_flow_mass"] * (target["solid_flow_mass"] / IND["solid_flow_mass"]) ** path_frac
        norf = max(1.0, IND["n_orifice"] + (target["n_orifice"] - IND["n_orifice"]) * path_frac)
        set_case(m, D, H, norf, gas, sol, target["gas_T"], target["solid_T"])
        _try_solve(f"cont {step}/{n_cont}")

    # Stage 3: final solve 
    try:
        res = solver.solve(b, tee=False)
        term = str(res.solver.termination_condition)
    except Exception as e:  
        term = f"EXC:{type(e).__name__}"
    if verbose:
        print(f"  final  {term}", flush=True)
    results = _collect(m, target, term)
    return m, results


def _collect(m, target, term):
    """
    Read the solved model outputs and return them as a dict 
    """
    b = m.fs.BFB

    # Read inlets and outlets from the solved model
    fmin_s = value(b.solid_inlet.flow_mass[0])
    fmout_s = value(b.solid_outlet.flow_mass[0])
    fmol_g = value(b.gas_inlet.flow_mol[0])
    fmol_g_out = value(b.gas_outlet.flow_mol[0])
    y_in = {j: value(b.gas_inlet.mole_frac_comp[0, j]) for j in MW_g}
    y_out = {j: value(b.gas_outlet.mole_frac_comp[0, j]) for j in MW_g}
    w_in = {j: value(b.solid_inlet.mass_frac_comp[0, j]) for j in MW_s}
    w_out = {j: value(b.solid_outlet.mass_frac_comp[0, j]) for j in MW_s}

    # Fe atom balance
    Fe_in = sum(w_in[j] / MW_s[j] * n_Fe[j] for j in MW_s) * fmin_s
    Fe_out = sum(w_out[j] / MW_s[j] * n_Fe[j] for j in MW_s) * fmout_s
    err_Fe = abs(Fe_out - Fe_in) / Fe_in * 100 if Fe_in > 0 else 0

    # H2 consumed vs H2O produced 
    dH2O = y_in["H2O"] * fmol_g - y_out["H2O"] * fmol_g_out
    nH2 = y_out["H2"] * fmol_g_out - y_in["H2"] * fmol_g
    err_H = abs(nH2 - dH2O) / dH2O * 100 if dH2O > 1e-12 else 0

    # O balance
    O_consumed = dH2O  
    O_solid_in = sum(w_in[j] / MW_s[j] * n_O[j] for j in MW_s) * fmin_s
    O_solid_out = sum(w_out[j] / MW_s[j] * n_O[j] for j in MW_s) * fmout_s
    O_incorporated = O_solid_out - O_solid_in
    err_O = abs(O_incorporated - O_consumed) / O_consumed * 100 if O_consumed > 1e-12 else 0

    # Overall mass balance
    m_gas_in = sum(y_in[j] * MW_g[j] for j in MW_g) * fmol_g
    m_gas_out = sum(y_out[j] * MW_g[j] for j in MW_g) * fmol_g_out
    err_mass = abs((m_gas_out + fmout_s) - (m_gas_in + fmin_s)) / (m_gas_in + fmin_s) * 100

    # Conversions
    h2o_in = fmol_g * y_in["H2O"]
    h2o_out = fmol_g_out * y_out["H2O"]
    X_H2O = (h2o_in - h2o_out) / h2o_in * 100 if h2o_in > 0 else 0
    fe_in = fmin_s * w_in["Fe"] / MW_s["Fe"]
    fe_out = fmout_s * w_out["Fe"] / MW_s["Fe"]
    X_Fe = (fe_in - fe_out) / fe_in * 100 if fe_in > 0 else 0

    # Physical impossible values scan
    n_bad = 0
    tol = 1e-4
    for x in m.fs.BFB.length_domain:
        for c in m.fs.gas_properties.component_list:
            yy = value(m.fs.BFB.gas_emulsion.properties[0, x].mole_frac_comp[c])
            if yy < -tol or yy > 1 + tol:
                n_bad += 1
            Cc = value(m.fs.BFB.gas_emulsion.properties[0, x].dens_mol_comp[c])
            if Cc < -tol:
                n_bad += 1
        for c in m.fs.solid_properties.component_list:
            ww = value(m.fs.BFB.solid_emulsion.properties[0, x].mass_frac_comp[c])
            if ww < -tol:
                n_bad += 1
    # Dict
    return {
        "termination": term,
        "target": {k: target[k] for k in
                   ("D", "H", "n_orifice", "gas_flow_mol", "solid_flow_mass")},
        "T_gas_in": value(b.gas_inlet.temperature[0]),
        "T_solid_in": value(b.solid_inlet.temperature[0]),
        "T_gas_out": value(b.gas_outlet.temperature[0]),
        "T_solid_out": value(b.solid_outlet.temperature[0]),
        "P_gas_out": value(b.gas_outlet.pressure[0]),
        "gas_mol_in": fmol_g, "gas_mol_out": fmol_g_out,
        "solid_mass_in": fmin_s, "solid_mass_out": fmout_s,
        "y_out": y_out, "w_out": w_out,
        "X_H2O": X_H2O, "X_Fe": X_Fe,
        "err_Fe": err_Fe, "err_H": err_H, "err_O": err_O, "err_mass": err_mass,
        "gas_feasible": 0.999 * fmol_g <= fmol_g_out <= 1.001 * fmol_g,
        "n_bad": n_bad,
    }


def _validity_banner(r):

    """
    This function warns if the solution is physically invalid. The exothermic bed's
    multiplicity can trap the solver on bad branches. These bad results are then marked
    """

    reasons = []
    if not r["gas_feasible"]:
        reasons.append("gas flow not conserved")
    for key, label in (("err_mass", "mass"), ("err_Fe", "Fe"),
                       ("err_H", "H"), ("err_O", "O")):
        if r[key] > 2.0:
            reasons.append(f"{label} err={r[key]:.1f}%")
    t_cold_in = min(r["T_gas_in"], r["T_solid_in"])
    if r["T_gas_out"] < t_cold_in - 5.0 and r["T_solid_out"] < t_cold_in - 5.0:
        reasons.append(f"outlets below cold inlet "
                       f"(Tout={r['T_solid_out']:.0f}K < {t_cold_in:.0f}K)")
    if reasons:
        return " Invalid solutions: " + "; ".join(reasons)
    return ""


def _axial_profile(m, title):

    """
    Axial profile: reaction rates, gas reactant, and solid composition along the bed 
     
    """
    b = m.fs.BFB
    H_bed = value(b.bed_height)
    fe_in = (value(b.solid_inlet.flow_mass[0])
             * value(b.solid_inlet.mass_frac_comp[0, "Fe"]) / MW_s["Fe"])
    print(f"\n  Axial profil:")
    for x in b.length_domain:
        z = x * H_bed
        r1 = value(b.solid_emulsion.reactions[0, x].reaction_rate["OW1"])
        r2 = value(b.solid_emulsion.reactions[0, x].reaction_rate["OW2"])
        c_h2o = value(b.gas_emulsion.properties[0, x].dens_mol_comp["H2O"])
        sp = b.solid_emulsion.properties[0, x]
        w_fe = value(sp.mass_frac_comp["Fe"])
        w_feo = value(sp.mass_frac_comp["FeO"])
        w_fe3o4 = value(sp.mass_frac_comp["Fe3O4"])
        xfe = ((fe_in - value(sp.flow_mass) * w_fe / MW_s["Fe"]) / fe_in * 100
               if fe_in > 0 else 0.0)
        print(f"  z={z:.2f}: rOW1={r1:.4e} rOW2={r2:.4e} mol/m3/s, "
              f"C_H2O={c_h2o:.4f}, w_Fe={w_fe:.6f}, w_FeO={w_feo:.6f}, "
              f"w_Fe3O4={w_fe3o4:.6f}, X_Fe={xfe:.6f}")


def _print_report(m, results, model_ind=None, results_ind=None):

    """
    Report for the lab case + industrial case
        
    """
    b = m.fs.BFB
    target = results["target"]
    if results["T_solid_out"] >= 1185:
        _branch = "   Hot branch "
    elif results["T_solid_out"] <= 1175:
        _branch = "  Match branch"
    else:
        _branch = ""
    print(f" Wet oxidation lab   D={target['D']} m  H={target['H']} m  "
          f"gas={target['gas_flow_mol']:.5f} mol/s  solid={target['solid_flow_mass']:.6e} kg/s"
          f"  n_orif={target['n_orifice']}   [{results['termination']}]{_branch}")
    print(b._get_stream_table_contents().to_string())  
    print("\nPhysical sanity checks:")
    print(f"  Fe atoms      err={results['err_Fe']:.3f}%")
    print(f"  H2O->H2       err={results['err_H']:.3f}%")
    print(f"  O balance     err={results['err_O']:.3f}%")
    print(f"  Mass balance  err={results['err_mass']:.3f}%")
    print(f"  Gas flow      in={results['gas_mol_in']:.5e}  out={results['gas_mol_out']:.5e}"
          f"  feasible={results['gas_feasible']}")
    print(f"\nConversions:  X_Fe = {results['X_Fe']:.2f} %   X_H2O = {results['X_H2O']:.2f} %")
    print(f"  T_out: gas={results['T_gas_out']:.1f} K  solid={results['T_solid_out']:.1f} K")
    print("  Outlet solid (mass frac): " +
          "  ".join(f"{j}={results['w_out'][j]:.4f}" for j in ["Fe", "FeO", "Fe3O4", "Fe2O3"]))
    print(f"  n_bad = {results['n_bad']}")
    vb = _validity_banner(results)
    if vb:
        print(vb)
    _axial_profile(m, "LAB")

    if model_ind is not None and results_ind is not None:
        target_ind = results_ind["target"]
        print(f" Industrial reactor   D={target_ind['D']} m  H={target_ind['H']} m  "
              f"gas={target_ind['gas_flow_mol']:.2f} mol/s  solid={target_ind['solid_flow_mass']:.3f} kg/s"
              f"  n_orif={target_ind['n_orifice']}   [{results_ind['termination']}]")
        print(model_ind.fs.BFB._get_stream_table_contents().to_string())  
        print("\nPhysical sanity checks:")
        print(f"  Fe atoms      err={results_ind['err_Fe']:.3f}%")
        print(f"  H2O->H2       err={results_ind['err_H']:.3f}%")
        print(f"  O balance     err={results_ind['err_O']:.3f}%")
        print(f"  Mass balance  err={results_ind['err_mass']:.3f}%")
        print(f"  Gas flow      in={results_ind['gas_mol_in']:.5e}  out={results_ind['gas_mol_out']:.5e}"
              f"  feasible={results_ind['gas_feasible']}")
        print(f"\nConversions:  X_Fe = {results_ind['X_Fe']:.2f} %   X_H2O = {results_ind['X_H2O']:.2f} %")
        print(f"  T_out: gas={results_ind['T_gas_out']:.1f} K  solid={results_ind['T_solid_out']:.1f} K")
        print("  Outlet solid (mass frac): " +
              "  ".join(f"{j}={results_ind['w_out'][j]:.4f}" for j in ["Fe", "FeO", "Fe3O4", "Fe2O3"]))
        print(f"  n_bad = {results_ind['n_bad']}")
        vb_ind = _validity_banner(results_ind)
        if vb_ind:
            print(vb_ind)
        _axial_profile(model_ind, "INDUSTRIAL")
        dxfe = results["X_Fe"] - results_ind["X_Fe"]
        dh2o = results["X_H2O"] - results_ind["X_H2O"]
        dt = results["T_solid_out"] - results_ind["T_solid_out"]
        print(f"\n  -> Lab vs Industrial:  X_Fe {results['X_Fe']:.2f}% vs {results_ind['X_Fe']:.2f}%"
              f" ({dxfe:+.2f} )   X_H2O {results['X_H2O']:.2f}% vs {results_ind['X_H2O']:.2f}%"
              f" ({dh2o:+.2f} )   T_out {results['T_solid_out']:.0f} vs {results_ind['T_solid_out']:.0f} K"
              f" ({dt:+.0f})")

def _target_from_args(args):

    """
    Turn the run options into the case to solve, on top of
    the default lab case. Diameter re-scales flows by area ratio (same u_g)
    """
    t = dict(LAB)  # copy base lab target
    if args.D is not None:
        # if D changes, re-scale flows to preserve u_g (area ratio)
        t["D"] = args.D
        ar = (args.D / IND["D"]) ** 2
        t["gas_flow_mol"] = IND["gas_flow_mol"] * ar
        t["solid_flow_mass"] = IND["solid_flow_mass"] * ar
    if args.H is not None:
        t["H"] = args.H
    if args.norif is not None:
        t["n_orifice"] = args.norif
    if args.gas is not None:
        t["gas_flow_mol"] = args.gas
    if args.solid is not None:
        t["solid_flow_mass"] = args.solid
    return t


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Lab-scale wet oxidation")
    ap.add_argument("--D", type=float, default=None)
    ap.add_argument("--H", type=float, default=None)
    ap.add_argument("--norif", type=int, default=None)
    ap.add_argument("--gas", type=float, default=None, help="gas flow_mol [mol/s]")
    ap.add_argument("--solid", type=float, default=None, help="solid flow_mass [kg/s]")
    ap.add_argument("--ncont", type=int, default=N_CONT)
    ap.add_argument("--json", action="store_true", help="print one-line JSON only")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    target = _target_from_args(args)
    m, r = solve_case(target, n_cont=args.ncont, verbose=args.verbose and not args.json)

    if args.json:
        print("result json " + json.dumps(r))
    else:
        m_ind, r_ind = solve_case(dict(IND), n_cont=args.ncont, verbose=False)
        _print_report(m, r, m_ind, r_ind)


if __name__ == "__main__":
    main()

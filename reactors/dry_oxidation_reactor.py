
"""

BFB isothermal reactor with dry oxidation reactions 
Oxidizes Fe with O2 using the CORAL SCM and ZLT kinetic model

Overall lumped reaction:
2 Fe + 3/2 O2 => Fe2O3    

Flow arrangement: co-current

"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyomo.environ import ConcreteModel, value
from idaes.core import FlowsheetBlock
from idaes.core.util import scaling as iscale
from idaes.core.solvers import get_solver
import idaes.logger as idaeslog

from idaes.models_extra.gas_solid_contactors.unit_models.bubbling_fluidized_bed import (
    BubblingFluidizedBed,
)

from custom_properties.gas_phase_thermo import CustomGasPhaseParameterBlock
from custom_properties.solid_phase_thermo import CustomSolidPhaseParameterBlock
from custom_properties.oxi_dry_reactions import DryOxidationReactionParameterBlock


Rg = 8.314 #Gas constant [J/(mol*K)]


def main():
    
    # Operating variables
    n_orifice = 2500
    bed_dia = 6.5
    bed_height = 5.0
    particle_dia = 1.5e-3 


    # Operating variables
    T_op = 1073.0
    P_op = 1e5
    y_O2_in = 0.21
    y_N2_in = 1.0 - y_O2_in
    flow_mol_gas = 3000.0
    flow_mass_solid = 200.0
    porosity = 0.27
    w_Fe2O3_in = 0.05
    w_Fe3O4_in = 0.0
    w_FeO_in = 0.0
    w_Fe_in = 0.95
    w_Al2O3_in = 0.0

        

    
    print("BFB Dry Oxidation Reactor")
    print(f"T = {T_op:.0f} K, P = {P_op:.0f} Pa")
    print(f"Gas: {y_O2_in*100:.1f}% O2, {y_N2_in*100:.1f}% N2, {flow_mol_gas:.0f} mol/s")
    print(f"Solid: {flow_mass_solid:.0f} kg/s, "
          f"Fe2O3={w_Fe2O3_in*100:.1f}% Fe={w_Fe_in*100:.1f}%")
    print(f"dp = {particle_dia*1e3:.1f} mm  rp = {particle_dia/2*1e6:.0f} um")
    

    
    # Build model
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    m.fs.gas_properties = CustomGasPhaseParameterBlock()
    m.fs.solid_properties = CustomSolidPhaseParameterBlock()
    m.fs.oxi_dry_reactions = DryOxidationReactionParameterBlock(
        solid_property_package=m.fs.solid_properties,
        gas_property_package=m.fs.gas_properties,
    )

    m.fs.BFB = BubblingFluidizedBed(
        flow_type="co_current",
        finite_elements=10,
        transformation_method="dae.finite_difference",
        gas_phase_config={"property_package": m.fs.gas_properties},
        solid_phase_config={
            "property_package": m.fs.solid_properties,
            "reaction_package": m.fs.oxi_dry_reactions,
        },
    )

   
    # Fix variables
    m.fs.solid_properties.particle_dia.fix(particle_dia)
    m.fs.BFB.number_orifice.fix(n_orifice)
    m.fs.BFB.bed_diameter.fix(bed_dia)
    m.fs.BFB.bed_height.fix(bed_height)

    # Fix inlet gas variables
    m.fs.BFB.gas_inlet.flow_mol[0].fix(flow_mol_gas)
    m.fs.BFB.gas_inlet.temperature[0].fix(T_op)
    m.fs.BFB.gas_inlet.pressure[0].fix(P_op)
    m.fs.BFB.gas_inlet.mole_frac_comp[0, "CO2"].fix(0.0)
    m.fs.BFB.gas_inlet.mole_frac_comp[0, "H2O"].fix(0.0)
    m.fs.BFB.gas_inlet.mole_frac_comp[0, "H2"].fix(0.0)

    # Fix inlet solid variables
    m.fs.BFB.solid_inlet.flow_mass[0].fix(flow_mass_solid)
    m.fs.BFB.solid_inlet.particle_porosity[0].fix(porosity)
    m.fs.BFB.solid_inlet.temperature[0].fix(T_op)
    m.fs.BFB.solid_inlet.mass_frac_comp[0, "Fe2O3"].fix(w_Fe2O3_in)
    m.fs.BFB.solid_inlet.mass_frac_comp[0, "Fe3O4"].fix(w_Fe3O4_in)
    m.fs.BFB.solid_inlet.mass_frac_comp[0, "FeO"].fix(w_FeO_in)
    m.fs.BFB.solid_inlet.mass_frac_comp[0, "Fe"].fix(w_Fe_in)
    m.fs.BFB.solid_inlet.mass_frac_comp[0, "Al2O3"].fix(w_Al2O3_in)

    
    # The solve strategy is to always start at y_O2=0.21 because it converges,
    # then homotopy to target 
    
    t_start = time.time()

    solid_state_args = {
        "flow_mass": flow_mass_solid,
        "particle_porosity": porosity,
        "temperature": T_op,
        "mass_frac": {
            "Fe2O3": w_Fe2O3_in, "Fe3O4": w_Fe3O4_in,
            "FeO": w_FeO_in, "Fe": w_Fe_in, "Al2O3": w_Al2O3_in,
        },
    }

    def set_o2_inlet(y_O2):
        m.fs.BFB.gas_inlet.mole_frac_comp[0, "O2"].fix(y_O2)
        m.fs.BFB.gas_inlet.mole_frac_comp[0, "N2"].fix(1.0 - y_O2)

    def gas_state_args(y_O2):
        return {
            "flow_mol": flow_mol_gas,
            "temperature": T_op,
            "pressure": P_op,
            "mole_frac": {
                "O2": y_O2, "N2": 1.0 - y_O2,
                "CO2": 0.0, "H2O": 0.0, "H2": 0.0,
            },
        }

    solver = get_solver()

    # Step 1: Build and solve at y_O2=0.21
    set_o2_inlet(0.21)
    iscale.calculate_scaling_factors(m)
    try:
        m.fs.BFB.initialize(
            outlvl=idaeslog.CRITICAL,
            gas_phase_state_args=gas_state_args(0.21),
            solid_phase_state_args=solid_state_args,
        )
        print("Initialize works")
    except Exception as e:
        print(f"Warning: {type(e).__name__}")

    res = solver.solve(m.fs.BFB, tee=False)
    tc = str(res.solver.termination_condition)
    print(f"  Solve: {tc}")

    if tc != "optimal":
        print("Error")
        return m

    # Step 2: Homotopy to target y_O2
    y_O2_start = 0.21
    y_O2_target = y_O2_in

    if abs(y_O2_target - y_O2_start) > 1e-6:
        # 3% of change each one but with min 5 steps
        n_steps = max(5, int(abs(y_O2_target - y_O2_start) / 0.03))
        print(f"\n Step 2: Homotopy {y_O2_start:.2f} -> {y_O2_target:.2f} "
              f"({n_steps} steps)")
        
        # lineal interpolation
        for i in range(1, n_steps + 1):
            y_step = y_O2_start + (y_O2_target - y_O2_start) * i / n_steps
            y_step = round(min(y_step, y_O2_target), 4)
            set_o2_inlet(y_step)

            res = solver.solve(m.fs.BFB, tee=False)
            tc = str(res.solver.termination_condition)
            print(f"  y_O2={y_step:.4f}: {tc}")

            if tc != "optimal":
                print(f"  Homotopy stalled at y_O2={y_step:.4f}. ")
                break
    else:
        print("\n No homotopy")

    t_total = time.time() - t_start
    print(f"\n Total time: {t_total:.1f} s")

    
    # Results
    
    
    print("Results")

    try:
        stream_table = m.fs.BFB._get_stream_table_contents()
        print(stream_table)
    except Exception as e:
        print(f"Stream table error: {e}")

    x_list = sorted(value(x) for x in m.fs.BFB.length_domain)
    z_in, z_out = x_list[0], x_list[-1]

    print("\nReaction rate R1 along bed:")
    for z in x_list:
        rxn = m.fs.BFB.solid_emulsion.reactions[0, z]
        try:
            rate = value(rxn.reaction_rate["R1"])
            c_o2 = value(rxn.gas_state_ref.dens_mol_comp["O2"])
            w_fe = value(rxn.solid_state_ref.mass_frac_comp["Fe"])
            oc = value(rxn.OC_conv)
            print(f"  z={z:.1f}: rate={rate:.4e} mol/m3/s, "
                  f"C_O2={c_o2:.4f}, w_Fe={w_fe:.6f}, OC_conv={oc:.6f}")
        except Exception:
            pass

    print("\nKinetic variables :")
    for z_label, z_val in [("Inlet (z=0)", z_in), ("Outlet (z=1)", z_out)]:
        rxn = m.fs.BFB.solid_emulsion.reactions[0, z_val]
        try:
            print(f"  {z_label}:")
            print(f"    OC_conv = {value(rxn.OC_conv):.6f}")
            print(f"    k_chr   = {value(rxn.k_chr):.4e} m3/(mol.s)")
            print(f"    D_eff   = {value(rxn.D_eff):.4e} m2/s")
            print(f"    X_chr   = {value(rxn.X_chr):.4f}")
            print(f"    sigmoid_w = {value(rxn.sigmoid_w):.6f}")
            print(f"    dXdt_I  = {value(rxn.dXdt_I):.6e} 1/s")
            print(f"    dXdt_II = {value(rxn.dXdt_II):.6e} 1/s")
            print(f"    C_O2    = {value(rxn.gas_state_ref.dens_mol_comp['O2']):.4f} mol/m3")
        except Exception as e:
            print(f"  {z_label}: error - {e}")

    print("\nSolid outlet mass fractions:")
    for j in m.fs.solid_properties.component_list:
        try:
            wf = value(m.fs.BFB.solid_outlet.mass_frac_comp[0, j])
            print(f"  {j}: {wf:.6f}")
        except Exception:
            pass

    print("\nGas outlet:")
    try:
        for j in m.fs.gas_properties.component_list:
            yf = value(m.fs.BFB.gas_outlet.mole_frac_comp[0, j])
            print(f"  y_{j} = {yf:.6f}")
        print(f"  flow_mol = {value(m.fs.BFB.gas_outlet.flow_mol[0]):.2f} mol/s")
    except Exception as e:
        print(f"  error - {e}")

    # Fe balance
    _MW = {"Fe2O3": 0.15969, "Fe3O4": 0.231533, "FeO": 0.071844, "Fe": 0.055845}
    n_Fe_atoms = {"Fe2O3": 2, "Fe3O4": 3, "FeO": 1, "Fe": 1}
    flow_in = value(m.fs.BFB.solid_inlet.flow_mass[0])
    w_in = {j: value(m.fs.BFB.solid_inlet.mass_frac_comp[0, j]) for j in _MW}
    n_Fe_in = sum(w_in[j] / _MW[j] * n_Fe_atoms[j] for j in _MW) * flow_in
    print(f"\nFe balance:")
    for j in ["Fe2O3", "Fe3O4", "FeO", "Fe"]:
        n_fe = w_in[j] / _MW[j] * flow_in * n_Fe_atoms[j]
        if n_fe > 0:
            print(f"  Inlet {j}: w={w_in[j]:.4f}, n_Fe={n_fe:.2f} mol/s")
    print(f"  Total n_Fe_in = {n_Fe_in:.2f} mol/s")

    try:
        flow_out = value(m.fs.BFB.solid_outlet.flow_mass[0])
        w_out = {j: value(m.fs.BFB.solid_outlet.mass_frac_comp[0, j]) for j in _MW}
        n_Fe_out = sum(w_out[j] / _MW[j] * n_Fe_atoms[j] for j in _MW) * flow_out
        print(f"\n  Outlet: {flow_out:.2f} kg/s")
        for j in ["Fe2O3", "Fe3O4", "FeO", "Fe"]:
            n_fe = w_out[j] / _MW[j] * flow_out * n_Fe_atoms[j]
            if n_fe > 0:
                print(f"    {j}: w={w_out[j]:.6f}, n_Fe={n_fe:.2f} mol_Fe/s")
        loss_pct = (n_Fe_in - n_Fe_out) / n_Fe_in * 100
        print(f"    Total n_Fe_out = {n_Fe_out:.2f} mol_Fe/s  (loss = {loss_pct:.2f}%)")
    except Exception as e:
        print(f"  Outlet balance error: {e}")

    

    # Plots
    
    
    print("Plots")

    z_arr, C_O2, C_N2 = [], [], []
    k_chr_arr, d_eff_arr = [], []
    dXdt_I_arr, dXdt_II_arr, rate_R1_arr = [], [], []
    OC_conv_arr, X_chr_arr = [], []
    w_Fe2O3, w_Fe3O4, w_FeO, w_Fe = [], [], [], []

    for x in m.fs.BFB.length_domain:
        z = value(x)
        rxn = m.fs.BFB.solid_emulsion.reactions[0, x]
        gas_ref, sol_ref = rxn.gas_state_ref, rxn.solid_state_ref

        z_arr.append(z)
        C_O2.append(_safe(value, gas_ref.dens_mol_comp["O2"]))
        C_N2.append(_safe(value, gas_ref.dens_mol_comp["N2"]))
        k_chr_arr.append(_safe(value, rxn.k_chr))
        d_eff_arr.append(_safe(value, rxn.D_eff))
        dXdt_I_arr.append(_safe(value, rxn.dXdt_I))
        dXdt_II_arr.append(_safe(value, rxn.dXdt_II))
        rate_R1_arr.append(_safe(value, rxn.reaction_rate["R1"]))
        OC_conv_arr.append(_safe(value, rxn.OC_conv))
        X_chr_arr.append(_safe(value, rxn.X_chr))
        w_Fe2O3.append(_safe(value, sol_ref.mass_frac_comp["Fe2O3"]))
        w_Fe3O4.append(_safe(value, sol_ref.mass_frac_comp["Fe3O4"]))
        w_FeO.append(_safe(value, sol_ref.mass_frac_comp["FeO"]))
        w_Fe.append(_safe(value, sol_ref.mass_frac_comp["Fe"]))

    z_arr = np.array(z_arr)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"BFB Dry Oxidation Profiles \n"
        f"T={T_op:.0f} K, {y_O2_in*100:.1f}% O2, {flow_mol_gas:.0f} mol/s gas, "
        f"{flow_mass_solid:.0f} kg/s solid, co-current",
        fontsize=12, fontweight="bold",
    )

    ax = axes[0, 0]
    ax.plot(z_arr, C_O2, "b-o", label="C$_{O_2}$", markersize=4)
    ax.plot(z_arr, C_N2, "g-s", label="C$_{N_2}$", markersize=4)
    ax.set_xlabel("Bed height z (m)"); ax.set_ylabel("Concentration (mol/m³)")
    ax.set_title("Gas Concentrations"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(z_arr, w_Fe2O3, "b-o", label="Fe₂O₃", markersize=4)
    ax.plot(z_arr, w_Fe, "r-s", label="Fe", markersize=4)
    ax.set_xlabel("Bed height z (m)"); ax.set_ylabel("Mass fraction (-)")
    ax.set_title("Solid Composition"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.semilogy(z_arr, k_chr_arr, "b-o", label="k_chr", markersize=4)
    ax.semilogy(z_arr, d_eff_arr, "r-s", label="D_eff", markersize=4)
    ax.set_xlabel("Bed height z (m)"); ax.set_ylabel("Constant")
    ax.set_title("Kinetic Constants"); ax.legend(); ax.grid(True, alpha=0.3, which="both")

    ax = axes[1, 0]
    ax.semilogy(z_arr, np.abs(dXdt_I_arr) + 1e-30, "b-o", label="dXdt_I", markersize=4)
    ax.semilogy(z_arr, np.abs(dXdt_II_arr) + 1e-30, "r-s", label="dXdt_II", markersize=4)
    ax.semilogy(z_arr, np.abs(rate_R1_arr) + 1e-30, "k-D", label="rate_R1", markersize=4)
    ax.set_xlabel("Bed height z (m)"); ax.set_ylabel("Rate")
    ax.set_title("Reaction Rates"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3, which="both")

    ax = axes[1, 1]
    ax.plot(z_arr, OC_conv_arr, "b-o", label="OC_conv", markersize=5, linewidth=2)
    ax.plot(z_arr, X_chr_arr, "r--s", label="X_chr", markersize=4)
    ax.set_xlabel("Bed height z (m)"); ax.set_ylabel("(-)")
    ax.set_title("Conversion"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    O2_consumed = [max(C_O2[0] - c, 0) for c in C_O2]
    ax.plot(z_arr, O2_consumed, "r-o", label="O₂ consumed", markersize=4)
    ax.set_xlabel("Bed height z (m)"); ax.set_ylabel("ΔC$_{O_2}$ (mol/m³)")
    ax.set_title("Cumulative O₂ Consumption"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "dry_oxidation_profiles.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {plot_path}")

    return m




def _safe(func, *args):

    """

    Safely evaluate a Pyomo expression, returning NaN in case of failure

    """

    try:
        return func(*args)
    except Exception:
        return float("nan")


if __name__ == "__main__":
    m = main()

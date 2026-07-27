
"""

BFB reactor with dry oxidation reaction

Overall lumped reaction
2 Fe + 3/2 O2 => Fe2O3    

"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time

from pyomo.environ import ConcreteModel, value, Var, SolverFactory
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


Rg = 8.314 #[J/(mol*K)]


def main():
    
    # Operating variables
    APPLY_YBOUNDS = False  # Bounds for the compositions
    SOLVER_NAME = "ipopt"  # ipopt or ipopt_l1
    n_orifice = 2500
    bed_dia = 6.5 # [m]
    bed_height = 5 # [m]
    particle_dia = 1.5e-3  # [m]
    T_gas = 873    # [K] 
    T_solid = 1073  #  [K] 
    P_op = 1e5 # [Pa]
    y_O2_in = 0.21
    y_N2_in = 1.0 - y_O2_in
    flow_mol_gas = 1967.0 # [mol/s]
    flow_mass_solid = 500.0 # [kg/s]
    porosity = 0.45
    w_Fe2O3_in = 0.05
    w_Fe3O4_in = 0.0
    w_FeO_in = 0.0
    w_Fe_in = 0.95
    w_Al2O3_in = 0.0

        

    
  
    

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
        finite_elements=20,
        transformation_method="dae.finite_difference",
        gas_phase_config={"property_package": m.fs.gas_properties},
        solid_phase_config={
            "property_package": m.fs.solid_properties,
            "reaction_package": m.fs.oxi_dry_reactions,
        },
    )

   
    # Fix variables
    m.fs.solid_properties.particle_dia.fix(particle_dia)
    # umf = 0.0396 m/s to find convergence
    m.fs.solid_properties.velocity_mf.fix(0.0396)
    m.fs.BFB.number_orifice.fix(n_orifice)
    m.fs.BFB.bed_diameter.fix(bed_dia)
    m.fs.BFB.bed_height.fix(bed_height)

    m.fs.BFB.gas_inlet.flow_mol[0].fix(flow_mol_gas)
    m.fs.BFB.gas_inlet.temperature[0].fix(T_gas)
    m.fs.BFB.gas_inlet.pressure[0].fix(P_op)
    m.fs.BFB.gas_inlet.mole_frac_comp[0, "CO2"].fix(0.0)
    m.fs.BFB.gas_inlet.mole_frac_comp[0, "H2O"].fix(0.0)
    m.fs.BFB.gas_inlet.mole_frac_comp[0, "H2"].fix(0.0)

    
    m.fs.BFB.solid_inlet.flow_mass[0].fix(flow_mass_solid)
    m.fs.BFB.solid_inlet.particle_porosity[0].fix(porosity)
    m.fs.BFB.solid_inlet.temperature[0].fix(T_solid)
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
        "temperature": T_solid,
        "mass_frac": {
            "Fe2O3": w_Fe2O3_in, "Fe3O4": w_Fe3O4_in,
            "FeO": w_FeO_in, "Fe": w_Fe_in, "Al2O3": w_Al2O3_in,
        },
    }

    # Fix O2 mole fraction at the gas inlet. N2 is the balance 1 - O2
    def set_o2_inlet(y_O2):
        m.fs.BFB.gas_inlet.mole_frac_comp[0, "O2"].fix(y_O2)
        m.fs.BFB.gas_inlet.mole_frac_comp[0, "N2"].fix(1.0 - y_O2)

    def gas_state_args(y_O2):
        return {
            "flow_mol": flow_mol_gas,
            "temperature": T_solid, 
            "pressure": P_op,
            "mole_frac": {
                "O2": y_O2, "N2": 1.0 - y_O2,
                "CO2": 0.0, "H2O": 0.0, "H2": 0.0,
            },
        }

    solver = SolverFactory(SOLVER_NAME)

    # Step 1: Build and solve at y_O2=0.21
    set_o2_inlet(0.21)
    iscale.calculate_scaling_factors(m)

    # Applying the bounds for the compositions and porosity of the particle
    if APPLY_YBOUNDS:
        for v in m.fs.BFB.component_data_objects(Var, descend_into=True):
            if "frac_comp" in v.name or "porosity" in v.name:
                v.setlb(0)

    try:
        m.fs.BFB.initialize(
            outlvl=idaeslog.CRITICAL,
            gas_phase_state_args=gas_state_args(0.21),
            solid_phase_state_args=solid_state_args,
        )
        print("Initialize works")
    except Exception as e:
        print(f"Warning: {type(e).__name__}")


    # Solve the reactor and read the status
    res = solver.solve(m.fs.BFB, tee=False)
    terminal_condition = str(res.solver.termination_condition)
    print(f" Solve: {terminal_condition}")

    # If the status is not optimal the script stops
    if terminal_condition != "optimal":
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

            # Solve at the new y_O2 step
            res = solver.solve(m.fs.BFB, tee=False)
            terminal_condition = str(res.solver.termination_condition)
            print(f"  y_O2={y_step:.4f}: {terminal_condition}")

            # The homotopy stops if the step does not converge
            if terminal_condition != "optimal":
                print(f"  Homotopy stalled at y_O2={y_step:.4f}. ")
                break
    else:
        pass



    
    # Results
    
    # Print the stream table
    try:
        stream_table = m.fs.BFB._get_stream_table_contents()
        print(stream_table)
    except Exception as e:
        print(f"Stream table error: {e}")

    # Axial positions z inlet and z outlet
    x_list = sorted(value(x) for x in m.fs.BFB.length_domain)
    z_in, z_out = x_list[0], x_list[-1]

    # Reaction rate and composition at each axial position
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

    # Kinetic variables at the inlet and outlet
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


    # Print the solid mass fractions at the outlet
    print("\nSolid outlet mass fractions:")
    for j in m.fs.solid_properties.component_list:
        try:
            weight_fraction = value(m.fs.BFB.solid_outlet.mass_frac_comp[0, j])
            print(f"  {j}: {weight_fraction:.6f}")
        except Exception:
            pass

    # Print the gas composition and flow at the outlet           
    print("\nGas outlet:")
    try:
        for j in m.fs.gas_properties.component_list:
            yf = value(m.fs.BFB.gas_outlet.mole_frac_comp[0, j])
            print(f"  y_{j} = {yf:.6f}")
        print(f"  flow_mol = {value(m.fs.BFB.gas_outlet.flow_mol[0]):.2f} mol/s")
    except Exception as e:
        print(f"  error - {e}")

    # Fe atom balance at the inlet 
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
    print(f" Total n_Fe_in = {n_Fe_in:.2f} mol/s")

    # Fe atom balance at the outlet and mass conservation check
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

    # Looking for physically impossible values 
    n_bad = 0
    tol = 0.0001  # values that are basically zero arent used
    for x in m.fs.BFB.length_domain:
        z = value(x)
        h = z * bed_height

        # Check the emulsion gas for physically impossibles values
        for c in m.fs.gas_properties.component_list:
            y = value(m.fs.BFB.gas_emulsion.properties[0, x].mole_frac_comp[c])
            if y < -tol:
                print(f"  z={z:.2f}  h={h:.2f} m   y_{c} = {y:.6f}")
                n_bad = n_bad + 1
            if y > 1 + tol:
                print(f"  z={z:.2f}  h={h:.2f} m   y_{c} = {y:.6f}")
                n_bad = n_bad + 1
            C = value(m.fs.BFB.gas_emulsion.properties[0, x].dens_mol_comp[c])
            if C < -tol:
                print(f"  z={z:.2f}  h={h:.2f} m   C_{c} = {C:.6f} mol/m3")
                n_bad = n_bad + 1

        # Check the bubble gas for physically impossibles values
        for c in m.fs.gas_properties.component_list:
            y = value(m.fs.BFB.bubble.properties[0, x].mole_frac_comp[c])
            if y < -tol:
                print(f"  z={z:.2f}  h={h:.2f} m   y_{c} = {y:.6f}")
                n_bad = n_bad + 1
            if y > 1 + tol:
                print(f"  z={z:.2f}  h={h:.2f} m   y_{c} = {y:.6f}")
                n_bad = n_bad + 1
            C = value(m.fs.BFB.bubble.properties[0, x].dens_mol_comp[c])
            if C < -tol:
                print(f"  z={z:.2f}  h={h:.2f} m   C_{c} = {C:.6f} mol/m3")
                n_bad = n_bad + 1

        # Check the solid mass for physically impossibles values
        for c in m.fs.solid_properties.component_list:
            w = value(m.fs.BFB.solid_emulsion.properties[0, x].mass_frac_comp[c])
            if w < -tol:
                print(f"  z={z:.2f}  h={h:.2f} m   w_{c} = {w:.6f}")
                n_bad = n_bad + 1

        # Check the porosity for physically impossibles fractions
        por = value(m.fs.BFB.solid_emulsion.properties[0, x].particle_porosity)
        if por < -tol:
            print(f"  z={z:.2f}  h={h:.2f} m   porosity = {por:.6f}")
            n_bad = n_bad + 1


    return m


if __name__ == "__main__":
    m = main()

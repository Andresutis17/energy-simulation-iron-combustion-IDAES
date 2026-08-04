"""
Reduction Kuhn BFB reactor at INDUSTRIAL scale (Independent temperatures inputs)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "reduction"))

from pyomo.environ import ConcreteModel, value, Var
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
from custom_properties.reduction_kuhn_reactions import ReductionKuhnReactionParameterBlock

def main():
   # Operating variables
    n_orifice = 2500
    bed_dia = 6.5        # m
    bed_height = 5       # m
    particle_dia = 1.5e-3  # m
    T_gas = 1050         # K 
    T_solid = 1173.9     # K 
    P_op = 1e5           # Pa 
    flow_mol_gas = 300.0    # mol/s 
    flow_mass_solid = 10.0  # kg/s 
    porosity = 0.27
    y_H2_in = 0.99       
    y_N2_in = 0.01
    w_Fe2O3_in = 1.0     
    w_Al2O3_in = 0.0

   
    # Build model
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.gas_properties = CustomGasPhaseParameterBlock()
    m.fs.solid_properties = CustomSolidPhaseParameterBlock()
    m.fs.reduction_reactions = ReductionKuhnReactionParameterBlock(
        solid_property_package=m.fs.solid_properties,
        gas_property_package=m.fs.gas_properties)

    m.fs.BFB = BubblingFluidizedBed(
        flow_type="co_current",
        finite_elements=20,
        transformation_method="dae.finite_difference",
        gas_phase_config={"property_package": m.fs.gas_properties},
        solid_phase_config={"property_package": m.fs.solid_properties,
                            "reaction_package": m.fs.reduction_reactions})

    # Fix variables
    m.fs.solid_properties.particle_dia.fix(particle_dia)
    m.fs.solid_properties.velocity_mf.fix(0.039624)
    m.fs.solid_properties.voidage_mf.fix(0.45)
    m.fs.solid_properties.voidage.fix(0.50)
    m.fs.BFB.number_orifice.fix(n_orifice)
    m.fs.BFB.bed_diameter.fix(bed_dia)
    m.fs.BFB.bed_height.fix(bed_height)
    m.fs.BFB.gas_inlet.flow_mol[0].fix(flow_mol_gas)
    m.fs.BFB.gas_inlet.temperature[0].fix(1073)
    m.fs.BFB.gas_inlet.pressure[0].fix(P_op)
    m.fs.BFB.gas_inlet.mole_frac_comp[0, "H2"].fix(y_H2_in)
    m.fs.BFB.gas_inlet.mole_frac_comp[0, "N2"].fix(y_N2_in)
    m.fs.BFB.gas_inlet.mole_frac_comp[0, "O2"].fix(1e-5)
    m.fs.BFB.gas_inlet.mole_frac_comp[0, "CO2"].fix(1e-5)
    m.fs.BFB.gas_inlet.mole_frac_comp[0, "H2O"].fix(1e-5)
    m.fs.BFB.solid_inlet.flow_mass[0].fix(flow_mass_solid)
    m.fs.BFB.solid_inlet.particle_porosity[0].fix(porosity)
    m.fs.BFB.solid_inlet.temperature[0].fix(1073)
    m.fs.BFB.solid_inlet.mass_frac_comp[0, "Fe2O3"].fix(w_Fe2O3_in)
    m.fs.BFB.solid_inlet.mass_frac_comp[0, "Al2O3"].fix(w_Al2O3_in)
    m.fs.BFB.solid_inlet.mass_frac_comp[0, "Fe3O4"].fix(1e-5)
    m.fs.BFB.solid_inlet.mass_frac_comp[0, "FeO"].fix(1e-5)
    m.fs.BFB.solid_inlet.mass_frac_comp[0, "Fe"].fix(1e-5)

    # State arguments for initializing property state blocks
    gas_args = {"flow_mol": flow_mol_gas, "temperature": 1073, "pressure": P_op,
                "mole_frac": {"H2": y_H2_in, "N2": y_N2_in, "O2": 1e-5, "CO2": 1e-5, "H2O": 1e-5}}
    sol_args = {"flow_mass": flow_mass_solid, "particle_porosity": porosity, "temperature": 1073,
                "mass_frac": {"Fe2O3": w_Fe2O3_in, "Al2O3": w_Al2O3_in, "Fe3O4": 1e-5, "FeO": 1e-5, "Fe": 1e-5}}

    iscale.calculate_scaling_factors(m)

    # Lower bound compositions and porosity at 0 to prevent negative values
    for v in m.fs.BFB.component_data_objects(Var, descend_into=True):
        if "frac_comp" in v.name or "porosity" in v.name:
            v.setlb(0)

    # Hides the IPOPT log
    m.fs.BFB.initialize(outlvl=idaeslog.CRITICAL,
                        gas_phase_state_args=gas_args,
                        solid_phase_state_args=sol_args)

    solver = get_solver()
    solver.solve(m.fs.BFB, tee=False)  # solve at init point (1073/1073)

    # Isoterm ramp . Both gas+solid to T_solid
    for T in [1100, T_solid]:
        m.fs.BFB.gas_inlet.temperature[0].fix(T)
        m.fs.BFB.solid_inlet.temperature[0].fix(T)
        solver.solve(m.fs.BFB, tee=False)

    # Non iso ramp. T_gas down to target, T_solid stays at T_solid
    for Tg in [1100, T_gas]:
        m.fs.BFB.gas_inlet.temperature[0].fix(Tg)
        solver.solve(m.fs.BFB, tee=False)

    # Final solve at target
    m.fs.BFB.gas_inlet.temperature[0].fix(T_gas)
    m.fs.BFB.solid_inlet.temperature[0].fix(T_solid)
    res = solver.solve(m.fs.BFB, tee=False)
    terminal_condition = str(res.solver.termination_condition)
    print(f"Solve: {terminal_condition}")

    if terminal_condition == "optimal":
        print(m.fs.BFB._get_stream_table_contents())

    
        # MW and atom counts per species
        MW = {"Fe": 0.055845, "FeO": 0.071844, "Fe3O4": 0.231533,
              "Fe2O3": 0.159688, "Al2O3": 0.101961}
        MW_gas = {"H2O": 0.018015, "H2": 0.002016, "N2": 0.028014,
                  "O2": 0.031998, "CO2": 0.044009}
        n_Fe_atm = {"Fe": 1, "FeO": 1, "Fe3O4": 3, "Fe2O3": 2, "Al2O3": 0}
        n_O_atm  = {"Fe": 0, "FeO": 1, "Fe3O4": 4, "Fe2O3": 3, "Al2O3": 3}

        blk = m.fs.BFB

        # Read inlets and outlets from the solved model
        fmin_s = value(blk.solid_inlet.flow_mass[0])
        fmout_s = value(blk.solid_outlet.flow_mass[0])
        fmol_g = value(blk.gas_inlet.flow_mol[0])  
        y_in = {j: value(blk.gas_inlet.mole_frac_comp[0, j]) for j in MW_gas}
        y_out = {j: value(blk.gas_outlet.mole_frac_comp[0, j]) for j in MW_gas}
        w_in = {j: value(blk.solid_inlet.mass_frac_comp[0, j]) for j in MW}
        w_out = {j: value(blk.solid_outlet.mass_frac_comp[0, j]) for j in MW}

        # Fe atom balance 
        Fe_in = sum(w_in[j]/MW[j]*n_Fe_atm[j] for j in MW) * fmin_s
        Fe_out = sum(w_out[j]/MW[j]*n_Fe_atm[j] for j in MW) * fmout_s
        err_Fe = abs(Fe_out - Fe_in)/Fe_in*100 if Fe_in > 0 else 0

        # H2 consumed vs H2O produced 
        dH2 = (y_in["H2"] - y_out["H2"]) * fmol_g
        nH2O = (y_out["H2O"] - y_in["H2O"]) * fmol_g
        err_H = abs(nH2O - dH2)/dH2*100 if dH2 > 1e-10 else 0

        # O balance
        O_solid_in = sum(w_in[j]/MW[j]*n_O_atm[j] for j in MW) * fmin_s
        O_solid_out = sum(w_out[j]/MW[j]*n_O_atm[j] for j in MW) * fmout_s
        O_released = O_solid_in - O_solid_out   # positive = O lost by solid
        err_O = abs(O_released - nH2O)/nH2O*100 if nH2O > 1e-10 else 0

        # Overall mass balance
        m_gas_in = sum(y_in[j]*MW_gas[j] for j in MW_gas) * fmol_g
        m_gas_out = sum(y_out[j]*MW_gas[j] for j in MW_gas) * fmol_g
        m_total_in = m_gas_in + fmin_s
        m_total_out = m_gas_out + fmout_s
        err_mass = abs(m_total_out - m_total_in)/m_total_in*100

        # Validation
        print(f"  1. Fe atoms:    in={Fe_in:.2f}  out={Fe_out:.2f} mol_Fe/s   err={err_Fe:.2f}%")
        print(f"  2. H2->H2O:     H2 consumed={dH2:.2f}  H2O produced={nH2O:.2f} mol/s   err={err_H:.2f}%")
        print(f"  3. O balance:   O released(={O_released:.2f}  O in H2O(gas)={nH2O:.2f} mol_O/s   err={err_O:.2f}%")
        print(f"  4. Mass:        in={m_total_in:.2f}  out={m_total_out:.2f} kg/s   err={err_mass:.2f}%")
        print(f"  5. Gas mass:    in={m_gas_in:.2f}  out={m_gas_out:.2f} kg/s  (d={m_gas_out-m_gas_in:+.2f})")

        # Physical impossible values scan 
        n_bad = 0
        tol = 1e-4
        for x in m.fs.BFB.length_domain:
            for c in m.fs.gas_properties.component_list:
                y = value(m.fs.BFB.gas_emulsion.properties[0, x].mole_frac_comp[c])
                if y < -tol or y > 1 + tol:
                    n_bad += 1
                C = value(m.fs.BFB.gas_emulsion.properties[0, x].dens_mol_comp[c])
                if C < -tol:
                    n_bad += 1
            for c in m.fs.solid_properties.component_list:
                w = value(m.fs.BFB.solid_emulsion.properties[0, x].mass_frac_comp[c])
                if w < -tol:
                    n_bad += 1
        print(f"\n  n_bad= {n_bad}")
    else:
        print("\n")

    return m


if __name__ == "__main__":
    m = main()

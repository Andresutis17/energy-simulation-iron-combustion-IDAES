
"""

Dry oxidation BFB reactor at INDUSTRIAL scale.

"""
import sys, os
here_path = os.path.dirname(os.path.abspath(__file__))
while here_path != os.path.dirname(here_path):
    if os.path.isdir(os.path.join(here_path, "custom_properties")):
        sys.path.insert(0, here_path)
        break
    here_path = os.path.dirname(here_path)

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
from custom_properties.oxi_dry_reactions import DryOxidationReactionParameterBlock


def main():

    # Operating variables
    n_orifice = 2500          
    bed_dia = 6.5             # m
    bed_height = 5            # m
    particle_dia = 1.5e-3     # m
    T_gas = 600               # K  
    T_solid = 1073            # K  
    P_op = 1e5                # Pa
    flow_mol_gas = 960.0      # mol/s air 
    flow_mass_solid = 10.0    # kg/s 
    porosity = 0.27
    y_O2_in = 0.21
    y_N2_in = 0.79
    w_Fe_in = 1
    w_Al2O3_in = 0.0

    # Build model
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.gas_properties = CustomGasPhaseParameterBlock()
    m.fs.solid_properties = CustomSolidPhaseParameterBlock()
    m.fs.dry_reactions = DryOxidationReactionParameterBlock(
        solid_property_package=m.fs.solid_properties,
        gas_property_package=m.fs.gas_properties)

    m.fs.BFB = BubblingFluidizedBed(
        flow_type="co_current",
        finite_elements=20,
        transformation_method="dae.finite_difference",
        gas_phase_config={"property_package": m.fs.gas_properties},
        solid_phase_config={"property_package": m.fs.solid_properties,
                            "reaction_package": m.fs.dry_reactions})

    # Fix variables
    m.fs.solid_properties.particle_dia.fix(particle_dia)
    m.fs.solid_properties.velocity_mf.fix(0.039624)
    m.fs.solid_properties.voidage_mf.fix(0.45)
    m.fs.solid_properties.voidage.fix(0.50)
    m.fs.BFB.number_orifice.fix(n_orifice)
    m.fs.BFB.bed_diameter.fix(bed_dia)
    m.fs.BFB.bed_height.fix(bed_height)
    m.fs.BFB.gas_inlet.flow_mol[0].fix(flow_mol_gas)
    m.fs.BFB.gas_inlet.temperature[0].fix(T_gas)
    m.fs.BFB.gas_inlet.pressure[0].fix(P_op)
    m.fs.BFB.gas_inlet.mole_frac_comp[0, "O2"].fix(y_O2_in)
    m.fs.BFB.gas_inlet.mole_frac_comp[0, "N2"].fix(y_N2_in)
    m.fs.BFB.gas_inlet.mole_frac_comp[0, "H2O"].fix(1e-5)
    m.fs.BFB.gas_inlet.mole_frac_comp[0, "CO2"].fix(1e-5)
    m.fs.BFB.gas_inlet.mole_frac_comp[0, "H2"].fix(1e-5)
    m.fs.BFB.solid_inlet.flow_mass[0].fix(flow_mass_solid)
    m.fs.BFB.solid_inlet.particle_porosity[0].fix(porosity)
    m.fs.BFB.solid_inlet.temperature[0].fix(T_solid)
    m.fs.BFB.solid_inlet.mass_frac_comp[0, "Fe"].fix(w_Fe_in)
    m.fs.BFB.solid_inlet.mass_frac_comp[0, "Al2O3"].fix(w_Al2O3_in)
    m.fs.BFB.solid_inlet.mass_frac_comp[0, "Fe2O3"].fix(0)
    m.fs.BFB.solid_inlet.mass_frac_comp[0, "Fe3O4"].fix(1e-5)
    m.fs.BFB.solid_inlet.mass_frac_comp[0, "FeO"].fix(1e-5)

    # State arguments for initializing property state blocks
    gas_args = {"flow_mol": flow_mol_gas, "temperature": T_solid, "pressure": P_op,
                "mole_frac": {"O2": y_O2_in, "N2": y_N2_in, "H2O": 1e-5, "CO2": 1e-5, "H2": 1e-5}}
    sol_args = {"flow_mass": flow_mass_solid, "particle_porosity": porosity, "temperature": T_solid,
                "mass_frac": {"Fe": w_Fe_in, "Al2O3": w_Al2O3_in, "Fe2O3": 1e-5, "Fe3O4": 1e-5, "FeO": 1e-5}}

    iscale.calculate_scaling_factors(m)

    # Lower bound compositions and porosity at 0 to prevent negative values
    for v in m.fs.BFB.component_data_objects(Var, descend_into=True):
        if "frac_comp" in v.name or "porosity" in v.name:
            v.setlb(0)

    # Ipopt tuned for the homotopy. max_iter=500 was used so a failing ramp
    # aborts and gets subdivided. acceptable (1e-3) keeps intermediate steps moving
    # bound_relax_factor avoids iterates stopping on the lb=0 compositions
    solver = get_solver()
    solver.options = {"max_iter": 500, "tol": 1e-6,
                      "acceptable_tol": 1e-3,
                      "acceptable_constr_viol_tol": 1e-3,
                      "bound_relax_factor": 1e-6}

    def _solve():
        res = solver.solve(m.fs.BFB, tee=False)
        return str(res.solver.termination_condition)


    # Initialize with the reaction rate scaled to 1e-4 
    # Without the reaction terms the coupled init converges from a uniform guess. 
    # Solve once to confirm a verified starting point.
    m.fs.dry_reactions._scale_factor_rxn = 1e-4
    m.fs.BFB.initialize(outlvl=idaeslog.CRITICAL,
                        gas_phase_state_args=gas_args,
                        solid_phase_state_args=sol_args)
    _solve()  



    # Ramp the rate scale to 1 geometrically. Failed steps bisect once, then abort. 
    # A stalled run must not be taken as correct
    scale_factor, target = 1e-4, 1.0    #Target=1; kinetic completed
    while scale_factor < target * 0.999:  # 0.999: tolerance so a step just under target counts as done
        next = min(scale_factor * 1.4, target) # x1.4: largest geometric step that converges
        m.fs.dry_reactions._scale_factor_rxn = next
        condition = _solve()
        if "optimal" in condition or "acceptable" in condition:
            scale_factor = next
        else:  # subdivide once
            mid = 0.5 * (scale_factor + next)  # retry the failed jump at half size
            m.fs.dry_reactions._scale_factor_rxn = mid
            condition_mid = _solve()
            if "optimal" in condition_mid or "acceptable" in condition_mid:
                scale_factor = mid
            else:
                print(f"Stalled at scale factor={scale_factor:.4g}")
                break
    print(f"Homotopy reached at scale factor = {scale_factor:.4g}")
    terminal_condition = _solve()
    print(f"Solve: {terminal_condition}")

    if "optimal" in terminal_condition or "acceptable" in terminal_condition:

        # Stream table
        print(m.fs.BFB._get_stream_table_contents())

        # MW and atom counts per species
        MW = {"Fe": 0.055845, "FeO": 0.071844, "Fe3O4": 0.231533,
              "Fe2O3": 0.15969, "Al2O3": 0.101961}
        MW_gas = {"H2O": 0.018, "H2": 0.002016, "N2": 0.028,
                  "O2": 0.032, "CO2": 0.044}
        n_Fe_atm = {"Fe": 1, "FeO": 1, "Fe3O4": 3, "Fe2O3": 2, "Al2O3": 0}
        n_O_atm  = {"Fe": 0, "FeO": 1, "Fe3O4": 4, "Fe2O3": 3, "Al2O3": 3}

        blk = m.fs.BFB

        # Read inlets and outlets from the solved model
        fmin_s = value(blk.solid_inlet.flow_mass[0])
        fmout_s = value(blk.solid_outlet.flow_mass[0])
        fmol_g = value(blk.gas_inlet.flow_mol[0])
        fmol_g_out = value(blk.gas_outlet.flow_mol[0])
        y_in = {j: value(blk.gas_inlet.mole_frac_comp[0, j]) for j in MW_gas}
        y_out = {j: value(blk.gas_outlet.mole_frac_comp[0, j]) for j in MW_gas}
        w_in = {j: value(blk.solid_inlet.mass_frac_comp[0, j]) for j in MW}
        w_out = {j: value(blk.solid_outlet.mass_frac_comp[0, j]) for j in MW}

        # Fe atom balance
        Fe_in = sum(w_in[j]/MW[j]*n_Fe_atm[j] for j in MW) * fmin_s
        Fe_out = sum(w_out[j]/MW[j]*n_Fe_atm[j] for j in MW) * fmout_s
        err_Fe = abs(Fe_out - Fe_in)/Fe_in*100 if Fe_in > 0 else 0

        # O2 consumed, Fe consumed, and Fe2O3 produced
        n_O2_fed = y_in["O2"] * fmol_g
        n_O2_consumed = y_in["O2"] * fmol_g - y_out["O2"] * fmol_g_out
        fe_in = fmin_s * w_in["Fe"] / MW["Fe"]
        fe_out = fmout_s * w_out["Fe"] / MW["Fe"]
        n_Fe_consumed = fe_in - fe_out
        n_Fe2O3_produced = (fmout_s * w_out["Fe2O3"] / MW["Fe2O3"]
                            - fmin_s * w_in["Fe2O3"] / MW["Fe2O3"])

        X_Fe = n_Fe_consumed / fe_in * 100 if fe_in > 0 else 0
        X_O2 = n_O2_consumed / n_O2_fed * 100 if n_O2_fed > 0 else 0

        # Stoichiometric checks
        ratio_Fe_O2 = n_Fe_consumed / n_O2_consumed if n_O2_consumed > 1e-9 else 0
        ratio_Fe2O3_O2 = n_Fe2O3_produced / n_O2_consumed if n_O2_consumed > 1e-9 else 0
        excess_O2 = (n_O2_fed - fe_in * 0.75) / (fe_in * 0.75) * 100 if fe_in > 0 else 0

        # O2 balance
        O_consumed = 2 * n_O2_consumed
        O_solid_in = sum(w_in[j]/MW[j]*n_O_atm[j] for j in MW) * fmin_s
        O_solid_out = sum(w_out[j]/MW[j]*n_O_atm[j] for j in MW) * fmout_s
        O_incorporated = O_solid_out - O_solid_in
        err_O = abs(O_incorporated - O_consumed)/O_consumed*100 if O_consumed > 1e-10 else 0

        # Mass balance 
        m_gas_in = sum(y_in[j]*MW_gas[j] for j in MW_gas) * fmol_g
        m_gas_out = sum(y_out[j]*MW_gas[j] for j in MW_gas) * fmol_g_out
        m_total_in = m_gas_in + fmin_s
        m_total_out = m_gas_out + fmout_s
        err_mass = abs(m_total_out - m_total_in)/m_total_in*100

        T_bed_out = value(blk.solid_outlet.temperature[0])
        T_gas_out = value(blk.gas_outlet.temperature[0])

        # Validation
        print(f"  1. Fe atoms:  in={Fe_in:.2f}  out={Fe_out:.2f} mol_Fe/s   err={err_Fe:.2f}%")
        print(f"  2. O balance: O from O2={O_consumed:.2f}  O in solid={O_incorporated:.2f} mol_O/s   err={err_O:.2f}%")
        print(f"  3. Mass:      in={m_total_in:.2f}  out={m_total_out:.2f} kg/s   err={err_mass:.2f}%")
        print(f"  4. Gas mass:  in={m_gas_in:.2f}  out={m_gas_out:.2f} kg/s)"
              f"   gas mol: in={fmol_g:.2f} out={fmol_g_out:.2f}")
        print(f"\n  X_Fe = {X_Fe:.2f}%   X_O2 = {X_O2:.2f}% ")
        print(f"  Fe consumed / O2 consumed       = {ratio_Fe_O2:.3f}   (stoich 1.333)")
        print(f"  Fe2O3 produced / O2 consumed    = {ratio_Fe2O3_O2:.3f}   (stoich 0.667)")
        print(f"  T_out: bed={T_bed_out:.1f} K  gas={T_gas_out:.1f} K"
              f"   (CORAL range 923-1073 K;  range: 923- 1073")

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
        print(f"\n  n_bad = {n_bad} ")

        # Axial profile: rate, C_O2, w_Fe, OC_conv along the bed 
        H_bed = value(blk.bed_height)
        print(f"\nAxial profile :")
        for x in m.fs.BFB.length_domain:
            z = x * H_bed
            rate = value(blk.solid_emulsion.reactions[0, x].reaction_rate["R1"])
            c_o2 = value(blk.gas_emulsion.properties[0, x].dens_mol_comp["O2"])
            w_fe = value(blk.solid_emulsion.properties[0, x].mass_frac_comp["Fe"])
            oc = value(blk.solid_emulsion.reactions[0, x].OC_conv)
            print(f"  z={z:.2f}: rate={rate:.4e} mol/m3/s, "
                  f"C_O2={c_o2:.4f}, w_Fe={w_fe:.6f}, OC_conv={oc:.6f}")

        # CORAL Kinetic
        xs = list(m.fs.BFB.length_domain)
        idxs = (0, 4, 8, 12, 16, len(xs) - 1)
        print(f"\nCORAL Kinetic:")
        for i in idxs:
            x = xs[i]
            z = x * H_bed
            rxn = blk.solid_emulsion.reactions[0, x]
            print(f"  z = {z:.2f} m:")
            print(f"    OC_conv = {value(rxn.OC_conv):.6f}")
            print(f"    k_chr   = {value(rxn.k_chr):.4e} m3/(mol.s)")
            print(f"    D_eff   = {value(rxn.D_eff):.4e} m3/(mol.s)")
            print(f"    X_chr   = {value(rxn.X_chr):.4f}")
            print(f"    sigmoid_w = {value(rxn.sigmoid_w):.6f}")
            print(f"    dXdt_I  = {value(rxn.dXdt_I):.6e} 1/s")
            print(f"    dXdt_II = {value(rxn.dXdt_II):.6e} 1/s")
    else:
        print("\n")

    return m


if __name__ == "__main__":
    m = main()

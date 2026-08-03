"""
Industrial to Lab scale test   (Doesnt converges)

Starts from the industrial Wet Oxidation reactor and scales down to lab scale 
by area scaling.
The gas and solid flows:
gas_lab   = gas_ind   / (D_ind / D_lab)^2
solid_lab = solid_ind / (D_ind / D_lab)^2

"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pyomo.environ import ConcreteModel, value, Var
from idaes.core import FlowsheetBlock
from idaes.core.util import scaling as iscale
from idaes.core.solvers import get_solver
import idaes.logger as idaeslog
from idaes.models_extra.gas_solid_contactors.unit_models.bubbling_fluidized_bed import (
    BubblingFluidizedBed,
)
from custom_properties.gas_phase_thermo import CustomGasPhaseParameterBlock
from custom_properties.solid_phase_thermo import CustomSolidPhaseParameterBlock
from custom_properties.oxi_wet_reactions import OxiWetReactionParameterBlock

# Inputs 
D_IND = 6.5          # m industrial bed diameter
H_IND = 5.0          # m industrial bed height
H_LAB = 1.0          # m lab bed height TU Darmstadt
T_GAS = 1050         # K gas inlet temperature 
T_SOLID = 1173.9     # K solid inlet temperature
P_OP = 1e5           # Pa pressure
GAS_FLOW_IND = 300.0     # mol/s gas flow 
SOLID_FLOW_IND = 10.0    # kg/s solid flow
PHI = 0.27           # particle porosity
N_ORIFICE = 2500     # orifices per m²
GAS_IN = {"H2O": 0.99, "N2": 0.01, "O2": 1e-5, "CO2": 1e-5, "H2": 1e-5}
SOLID_IN = {"Fe": 1.0, "Al2O3": 0.0, "Fe2O3": 0.0, "Fe3O4": 1e-5, "FeO": 1e-5}

# Lab reactor scale
D_LAB = 0.054      # m TU Darmstadt bed diameter
AREA_RATIO = (D_IND / D_LAB) ** 2


def run(label, D, H, gas_flow, solid_flow):

    """
    Build, initialize, solve and report the Wet Oxidation BFB at one scale
    """

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.gas_properties = CustomGasPhaseParameterBlock()
    m.fs.solid_properties = CustomSolidPhaseParameterBlock()
    m.fs.wet_reactions = OxiWetReactionParameterBlock(
        solid_property_package=m.fs.solid_properties,
        gas_property_package=m.fs.gas_properties)
    m.fs.BFB = BubblingFluidizedBed(
        flow_type="co_current", finite_elements=20,
        transformation_method="dae.finite_difference",
        gas_phase_config={"property_package": m.fs.gas_properties},
        solid_phase_config={"property_package": m.fs.solid_properties,
                            "reaction_package": m.fs.wet_reactions})

    # Fix variables
    m.fs.solid_properties.particle_dia.fix(1.5e-3)
    m.fs.BFB.number_orifice.fix(N_ORIFICE)
    m.fs.BFB.bed_diameter.fix(D)
    m.fs.BFB.bed_height.fix(H)
    m.fs.BFB.gas_inlet.flow_mol[0].fix(gas_flow)
    m.fs.BFB.gas_inlet.temperature[0].fix(T_GAS)
    m.fs.BFB.gas_inlet.pressure[0].fix(P_OP)
    for j, v in GAS_IN.items():
        m.fs.BFB.gas_inlet.mole_frac_comp[0, j].fix(v)
    m.fs.BFB.solid_inlet.flow_mass[0].fix(solid_flow)
    m.fs.BFB.solid_inlet.particle_porosity[0].fix(PHI)
    m.fs.BFB.solid_inlet.temperature[0].fix(T_SOLID)
    for j, v in SOLID_IN.items():
        m.fs.BFB.solid_inlet.mass_frac_comp[0, j].fix(v)

    # Initialize and solve 
    iscale.calculate_scaling_factors(m)

    # Lower bound compositions and porosity at 0 to prevent negative values
    for v in m.fs.BFB.component_data_objects(Var, descend_into=True):
        if "frac_comp" in v.name or "porosity" in v.name:
            v.setlb(0)

    gas_args = {"flow_mol": gas_flow, "temperature": T_SOLID, "pressure": P_OP,
                "mole_frac": GAS_IN}
    sol_args = {"flow_mass": solid_flow, "particle_porosity": PHI, "temperature": T_SOLID,
                "mass_frac": SOLID_IN}

    try:
        solver = get_solver(); solver.options["max_cpu_time"] = 120; solver.options["max_iter"] = 4000
        m.fs.BFB.initialize(outlvl=idaeslog.CRITICAL, gas_phase_state_args=gas_args,
                            solid_phase_state_args=sol_args)
        res = solver.solve(m.fs.BFB, tee=False)
        term = str(res.solver.termination_condition)

        # Metrics
        blk = m.fs.BFB
        Fin = gas_flow                                       # inlet gas molar flow mol/s
        Fout = value(blk.gas_outlet.flow_mol[0])             # outlet gas molar flow mol/s
        yout = value(blk.gas_outlet.mole_frac_comp[0, "H2O"])  # outlet H2O mole fraction 
        H2O_conv = (Fin * GAS_IN["H2O"] - Fout * yout) / (Fin * GAS_IN["H2O"]) * 100
        s_out = value(blk.solid_outlet.flow_mass[0])         # outlet solid mass flow kg/s
        w_Fe_out = value(blk.solid_outlet.mass_frac_comp[0, "Fe"])
        Fe_conv = (solid_flow * SOLID_IN["Fe"] - s_out * w_Fe_out) / \
                  (solid_flow * SOLID_IN["Fe"]) * 100

        def db_at(ztarget):
            """
            This function finds the bubble diameter at a target axial position
            """
            x = min(blk.length_domain, key=lambda xx: abs(value(xx) - ztarget))
            return value(blk.bubble_diameter[0, x])
        Tg_out = value(blk.gas_outlet.temperature[0])

        print(f"  termination       = {term}")
        print(f"  H2O conversion     = {H2O_conv:.4f} %")
        print(f"  Fe conversion      = {Fe_conv:.4f} %")
        print(f"  db midbed, outlet  = {db_at(0.5)*1e3:.2f} / {db_at(1.0)*1e3:.2f} mm")
        print(f"  T gas out          = {Tg_out:.1f} K")
        print(blk._get_stream_table_contents())

        return dict(label=label, term=term, H2O=H2O_conv, Fe=Fe_conv,
                    db5=db_at(0.5), db1=db_at(1.0), Tg=Tg_out)
    except Exception as e:
        print(f" Crash: {type(e).__name__}")
        return dict(label=label, term="fail")


if __name__ == "__main__":
    # Scale down from industrial to lab
    gas_lab = GAS_FLOW_IND / AREA_RATIO
    solid_lab = SOLID_FLOW_IND / AREA_RATIO

    # Run both scales
    ind = run("INDUSTRIAL", D_IND, H_IND, GAS_FLOW_IND, SOLID_FLOW_IND)
    lab = run("LAB scale",  D_LAB, H_LAB, gas_lab,       solid_lab)

    # Compare
    print("\n")
    print(" Comparison")
    if lab["term"] == "fail":
        print(f"  Lab didnt converged")
    else:
        print(f"  {'':20} {'Industrial':>14} {'Lab':>14} {'Difference':>12}")
        print(f"  {'H2O conversion %':<20} {ind['H2O']:>14.4f} {lab['H2O']:>14.4f} {lab['H2O']-ind['H2O']:>+11.4f}")
        print(f"  {'Fe conversion %':<20} {ind['Fe']:>14.4f} {lab['Fe']:>14.4f} {lab['Fe']-ind['Fe']:>+11.4f}")
        print(f"  {'db(mm) midbed':<20} {ind['db5']*1e3:>14.2f} {lab['db5']*1e3:>14.2f}")
        print(f"  {'db(mm) outlet':<20} {ind['db1']*1e3:>14.2f} {lab['db1']*1e3:>14.2f}")

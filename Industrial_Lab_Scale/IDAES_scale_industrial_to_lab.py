"""
Industrial to Lab scale test 

Starts from the industrial IDAES BFB Example and scales down to lab scale 
by area scaling.
The gas and solid flows:
gas_lab   = gas_ind   / (D_ind / D_lab)^2
solid_lab = solid_ind / (D_ind / D_lab)^2

"""
from pyomo.environ import ConcreteModel, value
from idaes.core import FlowsheetBlock
from idaes.core.util import scaling as iscale
from idaes.core.solvers import get_solver
import idaes.logger as idaeslog
from idaes.models_extra.gas_solid_contactors.unit_models.bubbling_fluidized_bed import (
    BubblingFluidizedBed,
)
from idaes.models_extra.gas_solid_contactors.properties.oxygen_iron_OC_oxidation.gas_phase_thermo import (
    GasPhaseParameterBlock,
)
from idaes.models_extra.gas_solid_contactors.properties.oxygen_iron_OC_oxidation.solid_phase_thermo import (
    SolidPhaseParameterBlock,
)
from idaes.models_extra.gas_solid_contactors.properties.oxygen_iron_OC_oxidation.hetero_reactions import (
    HeteroReactionParameterBlock,
)

# Inputs
D_IND = 6.5          # m industrial bed diameter
H_IND = 5.0          # m industrial bed height
H_LAB = 1.0          # m lab bed height TU Darmstadt 
GAS_FLOW_IND = 215.45    # mol/s gas flow
SOLID_FLOW_IND = 3.2137  # kg/s solid flow
T_GAS = 400.0        # K gas inlet temperature 
T_SOLID = 1173.9     # K solid inlet temperature 
P_OP = 1.86e5        # Pa pressure
PHI = 0.27           # particle porosity
N_ORIFICE = 2500     # orifices per m² 
GAS_IN = {"O2": 0.2095, "N2": 0.7808, "CO2": 0.0004, "H2O": 0.0093}
SOLID_IN = {"Fe2O3": 0.244162011502, "Fe3O4": 0.201998299487, "Al2O3": 0.553839689011}

#  Lab reactor scale
D_LAB = 0.054      # m TU Darmstadt bed diameter
AREA_RATIO = (D_IND / D_LAB) ** 2   


def run(label, D, H, gas_flow, solid_flow):

    """
    Build, initialize, solve and report the OC BFB at one scale
    """

    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.gas_properties = GasPhaseParameterBlock()
    m.fs.solid_properties = SolidPhaseParameterBlock()
    m.fs.hetero_reactions = HeteroReactionParameterBlock(
        solid_property_package=m.fs.solid_properties,
        gas_property_package=m.fs.gas_properties)
    m.fs.BFB = BubblingFluidizedBed(
        flow_type="co_current", finite_elements=5,
        transformation_method="dae.collocation",
        gas_phase_config={"property_package": m.fs.gas_properties},
        solid_phase_config={"property_package": m.fs.solid_properties,
                            "reaction_package": m.fs.hetero_reactions})

    # Fix variables
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
    gas_args = {"flow_mol": gas_flow, "temperature": T_SOLID, "pressure": P_OP,
                "mole_frac": GAS_IN}
    sol_args = {"flow_mass": solid_flow, "particle_porosity": PHI, "temperature": T_SOLID,
                "mass_frac": SOLID_IN}
    m.fs.BFB.initialize(outlvl=idaeslog.CRITICAL, gas_phase_state_args=gas_args,
                        solid_phase_state_args=sol_args)
    res = get_solver().solve(m.fs.BFB, tee=False)
    term = str(res.solver.termination_condition)

    # Metrics
    blk = m.fs.BFB
    Fin = gas_flow # inlet gas molar flow mol/s
    Fout = value(blk.gas_outlet.flow_mol[0]) # outlet gas molar flow mol/s
    yout = value(blk.gas_outlet.mole_frac_comp[0, "O2"])  # outlet O2 mole fraction
    O2_conv = (Fin * GAS_IN["O2"] - Fout * yout) / (Fin * GAS_IN["O2"]) * 100 # O2 conversion = O2 consumed / O2 fed
    s_out = value(blk.solid_outlet.flow_mass[0]) # outlet solid mass flow kg/s
    w_Fe3O4_out = value(blk.solid_outlet.mass_frac_comp[0, "Fe3O4"]) # outlet Fe3O4 mass fraction
    Fe3O4_conv = (solid_flow * SOLID_IN["Fe3O4"] - s_out * w_Fe3O4_out) / \
                 (solid_flow * SOLID_IN["Fe3O4"]) * 100   #Fe3O4 conversion

    def db_at(ztarget):
        """
        This function finds the bubble diameter at a target axial position
        """
        x = min(blk.length_domain, key=lambda xx: abs(value(xx) - ztarget))
        return value(blk.bubble_diameter[0, x])
    Tg_out = value(blk.gas_outlet.temperature[0])

    print(f"  termination       = {term}")
    print(f"  O2 conversion     = {O2_conv:.4f} %")
    print(f"  Fe3O4 conversion  = {Fe3O4_conv:.4f} %")
    print(f"  db midbed, outlet  = {db_at(0.5)*1e3:.2f} / {db_at(1.0)*1e3:.2f} mm")
    print(f"  T gas out         = {Tg_out:.1f} K")
    print(blk._get_stream_table_contents())

    return dict(label=label, term=term, O2=O2_conv, Fe3O4=Fe3O4_conv,
                db5=db_at(0.5), db1=db_at(1.0), Tg=Tg_out)


if __name__ == "__main__":
    # Scale down from industrial to lab
    gas_lab = GAS_FLOW_IND / AREA_RATIO
    solid_lab = SOLID_FLOW_IND / AREA_RATIO

    # Run both scales
    ind = run("INDUSTRIAL", D_IND, H_IND, GAS_FLOW_IND, SOLID_FLOW_IND)
    lab = run("LAB scale",  D_LAB, H_LAB, gas_lab,   solid_lab)

    # Compare
    print("\n")
    print(" Comparison")
    print(f"  {'':20} {'Industrial':>14} {'Lab':>14} {'Difference':>12}")
    print(f"  {'O2 conversion %':<20} {ind['O2']:>14.4f} {lab['O2']:>14.4f} {lab['O2']-ind['O2']:>+11.4f}")
    print(f"  {'Fe3O4 conversion %':<20} {ind['Fe3O4']:>14.4f} {lab['Fe3O4']:>14.4f} {lab['Fe3O4']-ind['Fe3O4']:>+11.4f}")
    print(f"  {'db(mm) midbed':<20} {ind['db5']*1e3:>14.2f} {lab['db5']*1e3:>14.2f}")
    print(f"  {'db(mm) outlet':<20} {ind['db1']*1e3:>14.2f} {lab['db1']*1e3:>14.2f}")
    dO2 = abs(lab['O2'] - ind['O2'])
    

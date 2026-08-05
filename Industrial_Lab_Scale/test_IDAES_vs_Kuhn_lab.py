
"""
This script runs the IDAES example, the Kuhn wet oxidation and the Kuhn reduction at
lab scale and compares them to show why custom reaction packages cant
be run at lab scale

"""

import sys, os, subprocess, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))   

P_OP = 1.86e5
D_LAB = 0.054
UMF = 0.039624
T_OP = 1073.0


def _inlet_vars(m):

    """
    Read the inlet (x=0) bubble variables: bubble_diameter, _reform_var_3
    and superficial gas velocity
    """

    from pyomo.environ import value
    x0 = m.fs.BFB.length_domain.first()
    blk = m.fs.BFB
    return dict(
        db=value(blk.bubble_diameter[0, x0]),
        rv3=value(blk._reform_var_3[0, x0]),
        usg=value(blk.velocity_superficial_gas[0, x0]),
    )


def run_kuhn():
    """
    Run the Kuhn reduction package at lab scale 
    """
    import math
    from pyomo.environ import ConcreteModel, value, Var
    from idaes.core import FlowsheetBlock
    from idaes.core.util import scaling as iscale
    from idaes.core.solvers import get_solver
    import idaes.logger as idaeslog
    idaeslog.getLogger("idaes").setLevel(idaeslog.WARNING)
    from idaes.models_extra.gas_solid_contactors.unit_models.bubbling_fluidized_bed import BubblingFluidizedBed
    from custom_properties.gas_phase_thermo import CustomGasPhaseParameterBlock
    from custom_properties.solid_phase_thermo import CustomSolidPhaseParameterBlock
    from custom_properties.reduction_kuhn_reactions import ReductionKuhnReactionParameterBlock
    GAS_IN = {"O2": 1e-5, "N2": 0.01, "CO2": 1e-5, "H2O": 1e-5, "H2": 0.99}
    SOLID_IN = {"Fe2O3": 1.0, "Fe3O4": 1e-5, "FeO": 1e-5, "Fe": 1e-5, "Al2O3": 0.0}
    gas_lab = 0.12 * (P_OP / (8.314 * T_OP)) * math.pi * (D_LAB / 2) ** 2  # gas feed (mol/s) to operate the lab bed at u_g = 0.12 (m/s)
    m = ConcreteModel(); m.fs = FlowsheetBlock(dynamic=False)
    m.fs.gas_properties = CustomGasPhaseParameterBlock()
    m.fs.solid_properties = CustomSolidPhaseParameterBlock()
    m.fs.reduction_reactions = ReductionKuhnReactionParameterBlock(
        solid_property_package=m.fs.solid_properties, gas_property_package=m.fs.gas_properties)
    m.fs.BFB = BubblingFluidizedBed(
        flow_type="co_current", finite_elements=20, transformation_method="dae.finite_difference",
        gas_phase_config={"property_package": m.fs.gas_properties},
        solid_phase_config={"property_package": m.fs.solid_properties, "reaction_package": m.fs.reduction_reactions})

    # Fix variables
    m.fs.solid_properties.particle_dia.fix(1.5e-3)
    m.fs.solid_properties.velocity_mf.fix(0.039624)
    m.fs.solid_properties.voidage_mf.fix(0.45); m.fs.solid_properties.voidage.fix(0.50)
    m.fs.BFB.number_orifice.fix(2500)
    m.fs.BFB.bed_diameter.fix(D_LAB); m.fs.BFB.bed_height.fix(1.0)
    m.fs.BFB.gas_inlet.flow_mol[0].fix(gas_lab)
    m.fs.BFB.gas_inlet.temperature[0].fix(T_OP); m.fs.BFB.gas_inlet.pressure[0].fix(P_OP)
    for j, v in GAS_IN.items():
        m.fs.BFB.gas_inlet.mole_frac_comp[0, j].fix(v)
    m.fs.BFB.solid_inlet.flow_mass[0].fix(1.0e-4)
    m.fs.BFB.solid_inlet.particle_porosity[0].fix(0.27)
    m.fs.BFB.solid_inlet.temperature[0].fix(T_OP)
    for j, v in SOLID_IN.items():
        m.fs.BFB.solid_inlet.mass_frac_comp[0, j].fix(v)
    iscale.calculate_scaling_factors(m)

    # Lower bound compositions and porosity at 0 to prevent negative values
    for v in m.fs.BFB.component_data_objects(Var, descend_into=True):
        if "frac_comp" in v.name or "porosity" in v.name:
            v.setlb(0)
    gas_args = {"flow_mol": gas_lab, "temperature": T_OP, "pressure": P_OP, "mole_frac": GAS_IN}
    sol_args = {"flow_mass": 1.0e-4, "particle_porosity": 0.27, "temperature": T_OP, "mass_frac": SOLID_IN}
    terminal_condition = "unknown"
    try:
        m.fs.BFB.initialize(outlvl=idaeslog.CRITICAL, gas_phase_state_args=gas_args, solid_phase_state_args=sol_args)
        res = get_solver().solve(m.fs.BFB, tee=False)
        terminal_condition = str(res.solver.termination_condition)
    except Exception as e:
        terminal_condition = f"{type(e).__name__}"
    iv = _inlet_vars(m)
    return dict(label="Kuhn reduction", tc=terminal_condition, **iv)


def run_wet():

    """
    Run the Kuhn wet oxidation package at lab scale 
    """
    import math
    from pyomo.environ import ConcreteModel, value, Var
    from idaes.core import FlowsheetBlock
    from idaes.core.util import scaling as iscale
    from idaes.core.solvers import get_solver
    import idaes.logger as idaeslog
    idaeslog.getLogger("idaes").setLevel(idaeslog.WARNING)
    from idaes.models_extra.gas_solid_contactors.unit_models.bubbling_fluidized_bed import BubblingFluidizedBed
    from custom_properties.gas_phase_thermo import CustomGasPhaseParameterBlock
    from custom_properties.solid_phase_thermo import CustomSolidPhaseParameterBlock
    from custom_properties.oxi_wet_reactions import OxiWetReactionParameterBlock
    GAS_IN = {"O2": 1e-5, "N2": 0.01, "CO2": 1e-5, "H2O": 0.99, "H2": 1e-5}   
    SOLID_IN = {"Fe2O3": 1e-5, "Fe3O4": 1e-5, "FeO": 1e-5, "Fe": 1.0, "Al2O3": 0.0}  
    gas_lab = 0.12 * (P_OP / (8.314 * T_OP)) * math.pi * (D_LAB / 2) ** 2  # gas feed (mol/s) to operate the lab bed at u_g = 0.12 (m/s)
    m = ConcreteModel(); m.fs = FlowsheetBlock(dynamic=False)
    m.fs.gas_properties = CustomGasPhaseParameterBlock()
    m.fs.solid_properties = CustomSolidPhaseParameterBlock()
    m.fs.wet_reactions = OxiWetReactionParameterBlock(
        solid_property_package=m.fs.solid_properties, gas_property_package=m.fs.gas_properties)
    m.fs.BFB = BubblingFluidizedBed(
        flow_type="co_current", finite_elements=20, transformation_method="dae.finite_difference",
        gas_phase_config={"property_package": m.fs.gas_properties},
        solid_phase_config={"property_package": m.fs.solid_properties, "reaction_package": m.fs.wet_reactions})

    # Fix variables
    m.fs.solid_properties.particle_dia.fix(1.5e-3)
    m.fs.solid_properties.velocity_mf.fix(0.039624)
    m.fs.solid_properties.voidage_mf.fix(0.45); m.fs.solid_properties.voidage.fix(0.50)
    m.fs.BFB.number_orifice.fix(2500)
    m.fs.BFB.bed_diameter.fix(D_LAB); m.fs.BFB.bed_height.fix(1.0)
    m.fs.BFB.gas_inlet.flow_mol[0].fix(gas_lab)
    m.fs.BFB.gas_inlet.temperature[0].fix(T_OP); m.fs.BFB.gas_inlet.pressure[0].fix(P_OP)
    for j, v in GAS_IN.items():
        m.fs.BFB.gas_inlet.mole_frac_comp[0, j].fix(v)
    m.fs.BFB.solid_inlet.flow_mass[0].fix(1.0e-4)
    m.fs.BFB.solid_inlet.particle_porosity[0].fix(0.27)
    m.fs.BFB.solid_inlet.temperature[0].fix(T_OP)
    for j, v in SOLID_IN.items():
        m.fs.BFB.solid_inlet.mass_frac_comp[0, j].fix(v)
    iscale.calculate_scaling_factors(m)

    # Lower bound compositions and porosity at 0 to prevent negative values
    for v in m.fs.BFB.component_data_objects(Var, descend_into=True):
        if "frac_comp" in v.name or "porosity" in v.name:
            v.setlb(0)
    gas_args = {"flow_mol": gas_lab, "temperature": T_OP, "pressure": P_OP, "mole_frac": GAS_IN}
    sol_args = {"flow_mass": 1.0e-4, "particle_porosity": 0.27, "temperature": T_OP, "mass_frac": SOLID_IN}
    terminal_condition = "unknown"
    try:
        m.fs.BFB.initialize(outlvl=idaeslog.CRITICAL, gas_phase_state_args=gas_args, solid_phase_state_args=sol_args)
        res = get_solver().solve(m.fs.BFB, tee=False)
        terminal_condition = str(res.solver.termination_condition)
    except Exception as e:
        terminal_condition = f"{type(e).__name__}"
    iv = _inlet_vars(m)
    return dict(label="Kuhn oxidation", tc=terminal_condition, **iv)


def run_oc():
    from pyomo.environ import ConcreteModel, value
    from idaes.core import FlowsheetBlock
    from idaes.core.util import scaling as iscale
    from idaes.core.solvers import get_solver
    import idaes.logger as idaeslog
    idaeslog.getLogger("idaes").setLevel(idaeslog.WARNING)
    from idaes.models_extra.gas_solid_contactors.unit_models.bubbling_fluidized_bed import BubblingFluidizedBed
    from idaes.models_extra.gas_solid_contactors.properties.oxygen_iron_OC_oxidation.gas_phase_thermo import GasPhaseParameterBlock
    from idaes.models_extra.gas_solid_contactors.properties.oxygen_iron_OC_oxidation.solid_phase_thermo import SolidPhaseParameterBlock
    from idaes.models_extra.gas_solid_contactors.properties.oxygen_iron_OC_oxidation.hetero_reactions import HeteroReactionParameterBlock
    GAS_IN = {"O2": 0.2095, "N2": 0.7808, "CO2": 0.0004, "H2O": 0.0093}
    SOLID_IN = {"Fe2O3": 0.244162011502, "Fe3O4": 0.201998299487, "Al2O3": 0.553839689011}
    area_ratio = (6.5 / D_LAB) ** 2  # (D_ind/D_lab)^2
    gas_lab = 215.45 / area_ratio    # area scaled OC gas
    solid_lab = 3.2137 / area_ratio  # area scaled OC solid
    m = ConcreteModel(); m.fs = FlowsheetBlock(dynamic=False)
    m.fs.gas_properties = GasPhaseParameterBlock()
    m.fs.solid_properties = SolidPhaseParameterBlock()
    m.fs.hetero_reactions = HeteroReactionParameterBlock(
        solid_property_package=m.fs.solid_properties, gas_property_package=m.fs.gas_properties)
    m.fs.BFB = BubblingFluidizedBed(
        flow_type="co_current", finite_elements=5, transformation_method="dae.collocation",
        gas_phase_config={"property_package": m.fs.gas_properties},
        solid_phase_config={"property_package": m.fs.solid_properties, "reaction_package": m.fs.hetero_reactions})
    m.fs.BFB.number_orifice.fix(2500)
    m.fs.BFB.bed_diameter.fix(D_LAB); m.fs.BFB.bed_height.fix(1.0)
    m.fs.BFB.gas_inlet.flow_mol[0].fix(gas_lab)
    m.fs.BFB.gas_inlet.temperature[0].fix(400.0)
    m.fs.BFB.gas_inlet.pressure[0].fix(P_OP)
    for j, v in GAS_IN.items():
        m.fs.BFB.gas_inlet.mole_frac_comp[0, j].fix(v)
    m.fs.BFB.solid_inlet.flow_mass[0].fix(solid_lab)
    m.fs.BFB.solid_inlet.particle_porosity[0].fix(0.27)
    m.fs.BFB.solid_inlet.temperature[0].fix(1173.9)
    for j, v in SOLID_IN.items():
        m.fs.BFB.solid_inlet.mass_frac_comp[0, j].fix(v)
    iscale.calculate_scaling_factors(m)
    gas_args = {"flow_mol": gas_lab, "temperature": 1173.9, "pressure": P_OP, "mole_frac": GAS_IN}
    sol_args = {"flow_mass": solid_lab, "particle_porosity": 0.27, "temperature": 1173.9, "mass_frac": SOLID_IN}
    m.fs.BFB.initialize(outlvl=idaeslog.CRITICAL, gas_phase_state_args=gas_args, solid_phase_state_args=sol_args)
    res = get_solver().solve(m.fs.BFB, tee=False)
    terminal_condition = str(res.solver.termination_condition)
    iv = _inlet_vars(m)
    return dict(label="IDAES", tc=terminal_condition, **iv)


def _run_in_subprocess(arg):
    """
    It runs the given case (reduction/wet/oc) and parse its JSON
    """
    res = subprocess.run([sys.executable, os.path.abspath(__file__), arg],
                         capture_output=True, text=True)
    return json.loads(res.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in ("reduction", "wet", "oc"):
        out = {"reduction": run_kuhn, "wet": run_wet, "oc": run_oc}[sys.argv[1]]()
        print(json.dumps(out))
        sys.exit(0)

    r = _run_in_subprocess("reduction")   
    w = _run_in_subprocess("wet")    
    o = _run_in_subprocess("oc")     

    
    print(f"Lab BFB D={D_LAB} m, P={P_OP/1e5:.2f} bar, u_g=0.12 m/s")
    print(f"  {'':34} {'Reduction':>16} {'Oxidation':>16} {'IDAES OC':>14}")
    print(f"  {'cold-init':<34} {r['tc']:>16} {w['tc']:>16} {o['tc']:>14}")
    print(f"  {'bubble_diameter(x=0) [m]':<34} {r['db']:>16.3e} {w['db']:>16.3e} {o['db']:>14.3e}")
    print(f"  {'_reform_var_3(x=0)  (= db^1/4)':<34} {r['rv3']:>16.3e} {w['rv3']:>16.3e} {o['rv3']:>14.3e}")
    print(f"  {'u_sg(x=0) - umf  (gas excess)':<34} {r['usg']-UMF:>+16.3e} {w['usg']-UMF:>+16.3e} {o['usg']-UMF:>+14.3e}")
    print(f"  {'Hbe Jacobian 5*rv3^4':<34} {5*r['rv3']**4:>16.3e} {5*w['rv3']**4:>16.3e} {5*o['rv3']**4:>14.3e}")
    print()
   

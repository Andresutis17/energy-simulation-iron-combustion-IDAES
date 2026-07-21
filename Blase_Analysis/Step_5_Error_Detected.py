"""
"Initialization Step 5 Failed" Analysis. It depends on the input gas temperature
"""
import logging, io, contextlib
from pyomo.environ import ConcreteModel, value
from pyomo.opt import TerminationCondition
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

T_SOLID = 1173.9  # K

# Silence IDAES warnings and then store them in CAP to detect "Step 5 Failed"
CAP = []
class _H(logging.Handler):
    def emit(self, r): CAP.append(r.getMessage())
def _silence_idaes():
    h = _H(); h.setLevel(logging.WARNING)
    for _name in ("idaes", "idaes.init", "idaes.init.fs.BFB"):
        _lg = logging.getLogger(_name)
        _lg.handlers = [h]
        _lg.propagate = True
_silence_idaes()


def build_and_init(Tgas):
    """

    Build the lab BFB only varying T_gas

    """
    CAP.clear()
    _silence_idaes()  
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.gas_properties = GasPhaseParameterBlock()
    m.fs.solid_properties = SolidPhaseParameterBlock()
    m.fs.hetero_reactions = HeteroReactionParameterBlock(
        solid_property_package=m.fs.solid_properties,
        gas_property_package=m.fs.gas_properties)
    b = m.fs.BFB = BubblingFluidizedBed(
        flow_type="co_current", finite_elements=5,
        transformation_method="dae.collocation",
        gas_phase_config={"property_package": m.fs.gas_properties},
        solid_phase_config={"property_package": m.fs.solid_properties,
                            "reaction_package": m.fs.hetero_reactions})
    
    # Lab inputs
    b.number_orifice.fix(2500)
    b.bed_diameter.fix(0.054)
    b.bed_height.fix(1)
    b.gas_inlet.flow_mol[0].fix(0.01487)
    b.gas_inlet.temperature[0].fix(Tgas)
    b.gas_inlet.pressure[0].fix(1.86e5)
    b.gas_inlet.mole_frac_comp[0, "O2"].fix(0.2095)
    b.gas_inlet.mole_frac_comp[0, "N2"].fix(0.7808)
    b.gas_inlet.mole_frac_comp[0, "CO2"].fix(0.0004)
    b.gas_inlet.mole_frac_comp[0, "H2O"].fix(0.0093)
    b.solid_inlet.flow_mass[0].fix(2.218e-4)
    b.solid_inlet.particle_porosity[0].fix(0.27)
    b.solid_inlet.temperature[0].fix(T_SOLID)
    b.solid_inlet.mass_frac_comp[0, "Fe2O3"].fix(0.244162011502)
    b.solid_inlet.mass_frac_comp[0, "Fe3O4"].fix(0.201998299487)
    b.solid_inlet.mass_frac_comp[0, "Al2O3"].fix(0.553839689011)

    # Initialization for the gas and solid property blocks
    gas_args = {"flow_mol": 0.01487, "temperature": T_SOLID, "pressure": 1.86e5,
                "mole_frac": {"O2": 0.2095, "N2": 0.7808, "CO2": 0.0004, "H2O": 0.0093}}
    solid_args = {"flow_mass": 2.218e-4, "particle_porosity": 0.27, "temperature": T_SOLID,
                  "mass_frac": {"Fe2O3": 0.244162011502, "Fe3O4": 0.201998299487,
                                "Al2O3": 0.553839689011}}
    
    # Scale, initialize, and solve. Initialization output is captured 
    # so "Step 5 Failed" can be detected. Finally run the final solve and read out results
    iscale.calculate_scaling_factors(m)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        b.initialize(outlvl=idaeslog.WARNING,
                     gas_phase_state_args=gas_args,
                     solid_phase_state_args=solid_args)
    captured = " ".join(CAP) + " " + buf.getvalue()
    step5 = "Warning" if "Step 5 Failed" in captured else "ok"
    res = get_solver().solve(b, tee=False)
    final = str(res.solver.termination_condition)
    try:
        ts_out = value(b.solid_outlet.temperature[0])
    except Exception:
        ts_out = float("nan")
    try:
        Dcol = value(b.bed_diameter)
        db_max = max(value(b.bubble_diameter[0, x]) for x in b.length_domain)
        db_over_D = db_max / Dcol
    except Exception:
        db_over_D = float("nan")
    return step5, final, ts_out, db_over_D


def main():

    # Header of results
    print(f"{'T_gas[K]':>9}  {'Step5':>14}  {'Solve final':>12}  "
          f"{'T_s_out[K]':>11}  {'max d_b/D':>9}")

    # Points spanning 3 regimes : cold , intermediate and hot 
    rows = []
    for T in [400, 700, 1100, 1120, 1173.9]:
        try:
            s5, fin, T_solid_out, dboD = build_and_init(T)
            print(f"{T:9.1f}  {s5:>14}  {fin:>12}  {T_solid_out:11.1f}  {dboD:9.2f}")
            rows.append({"T": T, "step5": s5, "final": fin, "T_solid_out": T_solid_out, "dbD": dboD})
        except Exception:
            print(f"{T:9.1f}  {'crash':>14}  {'N/A':>12}  {'N/A':>11}  {'N/A':>9}")
            rows.append({"T": T, "step5": "crash", "final": "N/A", "T_solid_out": float("nan"),
                         "dbD": float("nan")})


if __name__ == "__main__":
    main()

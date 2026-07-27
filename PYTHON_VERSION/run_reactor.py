"""
Driver for the BFB python reactor
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   
import PYTHON_VERSION.python_bfb_reactor as reactorpy                                          
from reaction_packages import oc_oxidation, wet_oxidation, dry_oxidation, reduction   

# Inputs
GEOM = dict(bed_diameter=6.5, bed_height=5, number_orifice=2500)
CASES = {
    # IDAES basic example
    "oc": dict(
        params={**GEOM, "gas_flow_mol": 1767.79, "gas_T": 473.0, "gas_P": 1.86e5,
                "solid_flow_mass": 1230.865, "solid_T": 1173.9},
        inlet_solid={"Fe2O3": 0.244162, "Fe3O4": 0.201998, "Al2O3": 0.553840},
        inlet_gas={"O2": 0.21, "N2": 0.79},
        pkg=oc_oxidation, pdia=1.5e-3),
    # Wet iron oxidation
    "wet": dict(
        params={**GEOM, "gas_flow_mol": 300.0, "gas_T": 1073.0, "gas_P": 1.0e5,
                "solid_flow_mass": 10.0, "solid_T": 1073.0},
        inlet_solid={"Fe": 0.95, "Al2O3": 0.05},
        inlet_gas={"H2O": 0.99, "N2": 0.01},
        pkg=wet_oxidation, pdia=1.5e-3),
    # Dry iron oxidation 
    "dry": dict(
        params={**GEOM, "gas_flow_mol": 100.0, "gas_T": 1073.0, "gas_P": 1.0e5,
                "solid_flow_mass": 0.5, "solid_T": 1073.0},
        inlet_solid={"Fe": 0.95, "Al2O3": 0.05},
        inlet_gas={"O2": 0.21, "N2": 0.79},
        pkg=dry_oxidation, pdia=60e-6),
    # Iron reduction
    "reduction": dict(
        params={**GEOM, "gas_flow_mol": 100.0, "gas_T": 1073.0, "gas_P": 1.0e5,
                "solid_flow_mass": 5.0, "solid_T": 1073.0},
        inlet_solid={"Fe2O3": 0.9, "Al2O3": 0.1},
        inlet_gas={"H2": 0.5, "N2": 0.5},
        pkg=reduction, pdia=60e-6),
}


def main():
    """
    Picks the reaction package from the command line, build its reaction
    package, solve the BFB, and print the outlet 

    """
    name = sys.argv[1] if len(sys.argv) > 1 else "oc"
    if name not in CASES:
        print(f"unknown package '{name}'. choices: {list(CASES)}"); return
    c = CASES[name]
    # build the chemistry package (species, kinetics, stoichiometry, cp) from the inlets
    pkg = c["pkg"].Package(c["inlet_solid"], c["inlet_gas"],
                            particle_dia=c["pdia"], bed_pressure=c["params"]["gas_P"])
    print(f"\n {pkg.NAME}")
    g = reactorpy.solve(c["params"], pkg)    # integrate the balances along the bed
    # print the outlet quantities
    print(f"  reactive gas consumed = {g['cons_total']:8.3f} mol/s   conversion = {g['conv_pct']:.2f}%")
    print(f"  T gas out   = {g['Tg_out']:8.1f} K   T solid out = {g['Ts_out']:.1f} K"
          f"   (gas in {c['params']['gas_T']:.0f}, solid in {c['params']['solid_T']:.0f})")
    print(f"  gas molar flow: {g['fmol_in']:.1f} -> {g['fmol_out']:.1f} mol/s")
    reactorpy.stream_table(g)                          # outlet stream table


if __name__ == "__main__":
    main()
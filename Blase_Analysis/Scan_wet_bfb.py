"""
Scans if there are negative values along the wet oxidation reactor
"""
import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_TESIS_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_TESIS_ROOT, "Wet_Oxi_Prototype"))
sys.path.insert(0, _TESIS_ROOT)

from pyomo.environ import value
from Wet_Reactor_Prototype import main


m = main()



# Physical consistency scan counters
n_bad = 0 # total number of physically impossible values over all nodes
tol = 1e-4 # numerical tolerance
worst_H2O = 0.0 # most negative H2O value found
worst_any = 0.0 # most negative value across the reactor


for x in m.fs.BFB.length_domain:
    z = value(x)
    h = z * 5.0
    # Emulsion gas analysis. Mole fraction must between 0 and 1 and concentrations bigger than 0
    for c in m.fs.gas_properties.component_list:
        y = value(m.fs.BFB.gas_emulsion.properties[0, x].mole_frac_comp[c])
        C = value(m.fs.BFB.gas_emulsion.properties[0, x].dens_mol_comp[c])
        if y < -tol or y > 1 + tol or C < -tol:
            n_bad += 1
            worst_any = min(worst_any, C, y)
            if c == "H2O":
                worst_H2O = min(worst_H2O, C, y)
            print(f"  {z:.2f}   {h:.2f}   emulsion  {c:<8}   y={y:.6f}  C={C:.6f}")

    # Bubble gas analysis. Mole fraction must between 0 and 1 and concentrations bigger than 0
    for c in m.fs.gas_properties.component_list:
        y = value(m.fs.BFB.bubble.properties[0, x].mole_frac_comp[c])
        C = value(m.fs.BFB.bubble.properties[0, x].dens_mol_comp[c])
        if y < -tol or y > 1 + tol or C < -tol:
            n_bad += 1
            worst_any = min(worst_any, C, y)
            print(f"  {z:.2f}   {h:.2f}   bubble    {c:<8}   y={y:.6f}  C={C:.6f}")
            
    # Solid analysis. Mass fractiones must be positive
    for c in m.fs.solid_properties.component_list:
        w = value(m.fs.BFB.solid_emulsion.properties[0, x].mass_frac_comp[c])
        if w < -tol:
            n_bad += 1
            worst_any = min(worst_any, w)
            print(f"  {z:.2f}   {h:.2f}   solid     {c:<8}   w={w:.6f}")

print(f"\n  n_bad = {n_bad}")
print(f"  worst emulsion H2O (y or C) = {worst_H2O:.6f}")


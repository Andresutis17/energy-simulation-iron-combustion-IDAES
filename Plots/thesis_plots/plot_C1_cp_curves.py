"""

Cp curves Shomate of the  property packages

"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import apply_style, save_fig, INK2, GRID

# Shomate coefficients as in the property packages
# cp[J/mol/K] = A + B*t + C*t^2 + D*t^3 + E/t^2, t = T/1000
REPO_8 = {
    "H2":  (18.563083, 12.257357, -2.859786, 0.268238, 1.977990,
            -1.147436, 156.288133, 0.0),
    "H2O": (30.0920000, 6.8325140, 6.7934350, -2.5344800, 0.0821390,
            -250.8810000, 223.3967000, -241.8264000),
    "O2":  (30.03235, 8.772972, -3.988133, 0.788313, -0.741599,
            -11.32468, 236.1663, 0.0000),
    "N2":  (19.50583, 19.88705, -8.598535, 1.369784, 0.527601,
            -4.935202, 212.3900, 0.0000),
    "CO2": (24.9973500, 55.1869600, -33.6913700, 7.9483870, -0.1366380,
            -403.6075000, 228.2431000, -393.5224000),

    "Fe2O3": (110.9362000, 32.0471400, -9.1923330, 0.9015060, 5.4336770,
              -843.1471000, 228.3548000, -825.5032000),
    "Fe3O4": (200.8320000, 1.586435e-7, -6.661682e-8, 9.452452e-9,
              3.18602e-8, -1174.1350000, 388.0790000, -1120.8940000),
    "FeO":   (45.75120, 18.78553, -5.952201, 0.852779, -0.081265,
              -286.7429, 110.3120, -272.0441),
    "Fe":    (23.97449, 8.367750, 0.000277, -0.000086, -0.000005,
              0.268027, 62.06336, 7.788015),
    "Al2O3": (102.4290, 38.74980, -15.91090, 2.628181, -3.007551,
              -1717.930, 146.9970, -1675.690),
}


def cp_row(row5, T):
    """
    Shomate cp [J/mol/K] from one row at temperature T [K]
    """
    t = np.asarray(T, dtype=float) / 1000.0
    A, B, C, D, E = row5
    return A + B * t + C * t**2 + D * t**3 + E / t**2


def cp_repo(sp, T):
    return cp_row(REPO_8[sp][:5], T)


# Only the species that are in the simulations. CO2 and the Al2O3 are inert 
apply_style()
GAS = ["H2", "H2O", "O2", "N2"]
SOL = ["Fe2O3", "Fe3O4", "FeO", "Fe"]
COLOR5 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b06fb6"]
PANEL = {s: c for s, c in zip(GAS + SOL, COLOR5 + COLOR5)}
LABEL = {"H2": r"H$_2$", "H2O": r"H$_2$O", "O2": r"O$_2$",
         "N2": r"N$_2$", "CO2": r"CO$_2$", "Fe2O3": r"Fe$_2$O$_3$",
         "Fe3O4": r"Fe$_3$O$_4$", "FeO": "FeO", "Fe": "Fe",
         "Al2O3": r"Al$_2$O$_3$"}

T = np.linspace(600, 1500, 1801)  # Coldest reactor feed dry gas 600 K
for group, title, stem, ylim, legend_loc in (
        (GAS, "Gas Phase", "C1_cp_gas", (28, 49), "upper left"),
        (SOL, "Solid Phase", "C1_cp_solid", (20, 215), "center right")):
    fig, ax = plt.subplots(figsize=(4.4, 3.3))
    for sp in group:
        ax.plot(T, cp_repo(sp, T), lw=1.8, color=PANEL[sp],
                label=LABEL[sp])
    ax.set_xlim(600, 1500)
    ax.set_ylim(*ylim)
    ax.set_xlabel("T [K]")
    ax.set_ylabel(r"$c_p$ [J/(mol$\cdot$K)]", color=INK2)
    ax.set_title(title, color=INK2)
    ax.grid(True, color=GRID)
    ax.legend(loc=legend_loc, handlelength=1.4, labelspacing=0.25)
    save_fig(fig, stem)

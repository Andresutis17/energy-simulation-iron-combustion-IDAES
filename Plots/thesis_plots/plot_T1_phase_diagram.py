"""

Baur Glaessner phase stability diagramm 
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import apply_style, save_fig, INK, INK2, MUTED

R = 8.314462618  # J/mol/K
T_C = 840.15     # Wustite eutectoid, 567 C (OECD, Fradet Eq. 12)
X_WUS = 0.068    # 1 - 0.932 Oxygen deficit of the fixed wustite composition 

# NASA polys (Tmin, Tmax, a0, a1, a2, a3, a4, a5, a6)
# Fradet et al. 2023, Table 1
NASA = {
    "Fe2O3": [
        (298.15, 850, 2.61729398E+00, 4.20598101E-02, -2.22685705E-05,
         -3.70207634E-08, 3.83168423E-11, -1.01777798E+05, -1.56994839E+01),
        (850, 955, 1.68251519E+03, -9.19768320E+00, 1.85371494E-02,
         -1.62772004E-05, 5.28385649E-09, -3.37236385E+05, -7.55788361E+03),
        (955, 975, 6.92489897E+05, -2.14261896E+03, 2.20995541E+00,
         -7.59830854E-04, 0.0, -1.67962002E+08, -3.49260235E+06),
        (975, 1100, 2.47252937E+03, -6.93190409E+00, 6.52658863E-03,
         -2.04922273E-06, 0.0, -7.57124284E+05, -1.26975884E+04),
        (1100, 3000, 1.70916526E+01, 0.0, 0.0, 0.0, 0.0,
         -1.04383860E+05, -8.76797120E+01),
    ],
    "Fe3O4": [
        (298.15, 740, 3.82469045E+00, 5.69085403E-02, -2.70781490E-06,
         -1.26634581E-07, 1.19018732E-10, -1.37649209E+05, -2.02083041E+01),
        (740, 847.57, 2.23442015E+04, -1.19262966E+02, 2.38712553E-01,
         -2.12166584E-04, 7.06823598E-08, -3.47986100E+06, -1.01327451E+05),
        (847.57, 860, -5.30054628E+01, -6.71418489E-02, 8.19572230E-04,
         -5.91738213E-08, -7.81285922E-10, -1.41642421E+05, 2.75351760E+02),
        (860, 1040, 3.88765019E+04, -1.60652171E+02, 2.48883815E-01,
         -1.71191829E-04, 4.41088229E-08, -7.65168930E+06, -1.86255226E+05),
        (1040, 1870, 2.45369257E+01, 0.0, 0.0, 0.0, 0.0,
         -1.40723022E+05, -1.22352862E+02),
    ],
    "Fe0.932O": [
        (298.15, 1000, 4.16156123E+00, 9.86126603E-03, -1.77356756E-05,
         1.53248312E-08, -4.95273744E-12, -3.35389654E+04, -1.91701052E+01),
        (1000, 1652, 6.04388238E+00, 5.62208355E-04, 5.31547086E-08,
         0.0, 0.0, -3.38607075E+04, -2.78979839E+01),
    ],
    "Fe": [
        (298.15, 1000, 5.32161918E+00, -2.31955387E-02, 7.69124479E-05,
         -9.73594557E-08, 4.48710837E-11, -1.06396887E+03, -2.27941375E+01),
        (1000, 1043, -7.54409933E+01, 8.19911497E-02, 0.0, 0.0, 0.0,
         3.73771347E+04, 4.47128379E+02),
        (1043, 1060, 1.72886068E+06, -4.45486325E+03, 3.34563841E+00,
         -2.24235278E-04, -3.95088930E-07, -4.81566432E+08, -8.98698119E+06),
        (1060, 1185, 2.67615995E+04, -9.31264019E+01, 1.21576443E-01,
         -7.05543659E-05, 1.53562809E-08, -6.15368158E+06, -1.32837426E+05),
        (1185, 1667, 3.58685861E+00, -6.28121792E-04, 1.18014939E-06,
         -2.62802283E-10, 0.0, -1.53970727E+02, -1.61851091E+01),
    ],
    "H2": [
        (200, 1000, 3.33727920E+00, -4.94024731E-05, 4.99456778E-07,
         -1.79566394E-10, 2.00255376E-14, -9.50158922E+02, -3.20502331E+00),
        (1000, 3500, 2.34433112E+00, 7.98052075E-03, -1.94781510E-05,
         2.01572094E-08, -7.37611761E-12, -9.17935173E+02, 6.83010238E-01),
    ],
    "H2O": [
        (200, 1000, 4.19864056E+00, -2.03643410E-03, 6.52040211E-06,
         -5.48797062E-09, 1.77197817E-12, -3.02937267E+04, -8.49032208E-01),
        (1000, 3500, 3.03399249E+00, 2.17691804E-03, -1.64072518E-07,
         -9.70419870E-11, 1.68200992E-14, -3.00042971E+04, 4.96677010E+00),
    ],
    "O2": [
        (200, 1000, 3.28253784E+00, 1.48308754E-03, -7.57966669E-07,
         2.09470555E-10, -2.16717794E-14, -1.08845772E+03, 5.45323129E+00),
        (1000, 3500, 3.78245636E+00, -2.99673416E-03, 9.84730201E-06,
         -9.68129509E-09, 3.24372837E-12, -1.06394356E+03, 3.65767573E+00),
    ],
}


def _rows(name, T):
    """
    All rows valid across the T array 
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    out = []
    for row in NASA[name]:
        lo, hi = row[0], row[1]
        if np.any((T >= lo) & (T <= hi)):
            out.append(row)
    return out


def _eval(name, T):
    """
    Returns (h/RT, s/R, cp/R) 
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    h = np.full(T.shape, np.nan)
    s = np.full(T.shape, np.nan)
    for row in _rows(name, T):
        lo, hi, a0, a1, a2, a3, a4, a5, a6 = row
        m = (T >= lo) & (T <= hi)
        t = T[m]
        h[m] = a0 + a1 * t / 2 + a2 * t**2 / 3 + a3 * t**3 / 4 + a4 * t**4 / 5 + a5 / t
        s[m] = a0 * np.log(t) + a1 * t + a2 * t**2 / 2 + a3 * t**3 / 3 + a4 * t**4 / 4 + a6
    return h, s


# Reduction direction, 1 H2 -> 1 H2O each (Fradet Eqs. 1, 6, 7, 8).
# E1: 1/4 Fe3O4        + H2 -> 3/4 Fe          + H2O   (Fe3O4/Fe)
# E6: 3 Fe2O3          + H2 -> 2 Fe3O4         + H2O   (Fe2O3/Fe3O4)
# E7: (1-x)/(1-4x) Fe3O4 + H2 -> 3/(1-4x) Fe1-xO + H2O (Fe3O4/Fe1-xO)
# E8: Fe1-xO           + H2 -> (1-x) Fe        + H2O   (Fe1-xO/Fe)
C7 = 1.0 - 4.0 * X_WUS        # 0.728
def dg(T):

    """
    Reaction Gibbs energies (kJ/mol), reduction direction

    """
    def g(sp):
        hh, ss = _eval(sp, T)
        return (hh - ss) * R * T / 1000.0   # (h/RT - s/R) * RT
    return {
        "E1": 0.75 * g("Fe") + g("H2O") - 0.25 * g("Fe3O4") - g("H2"),
        "E6": 2 * g("Fe3O4") + g("H2O") - 3 * g("Fe2O3") - g("H2"),
        "E7": (3 / C7) * g("Fe0.932O") + g("H2O")
              - ((1 - X_WUS) / C7) * g("Fe3O4") - g("H2"),
        "E8": (1 - X_WUS) * g("Fe") + g("H2O") - g("Fe0.932O") - g("H2"),
    }


def x_h2(key, T):
    """
    Boundary as H2 share of H2+H2O (Fradet Eq. 5)
    """
    return 100.0 / (1.0 + np.exp(-dg(T)[key] * 1e3 / (R * T)))


apply_style()
fig, ax = plt.subplots(figsize=(5.6, 4.4))

def y_h2o(key, T):
    """
    Boundary as steam share on the x axi: 1 - x_H2/100.
    """
    return 1.0 - x_h2(key, T) / 100.0

T_below = np.linspace(600, T_C, 300)
T_above = np.linspace(T_C, 1200, 300)

ax.plot(y_h2o("E1", T_below), T_below, color=INK, lw=1.8)   # Fe3O4/Fe
ax.plot(y_h2o("E7", T_above), T_above, color=INK, lw=1.8)   # Fe3O4/FeO
ax.plot(y_h2o("E8", T_above), T_above, color=INK, lw=1.8)   # FeO/Fe

ax.set_xlim(0, 1)
ax.set_ylim(600, 1200)
ax.set_xlabel(r"$\mathrm{H_2O}/(\mathrm{H_2}+\mathrm{H_2O})$")
ax.set_ylabel(r"$T$ [K]")
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_yticks(np.arange(600, 1201, 100))
ax.grid(True)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

# Phase fields 
ax.text(0.80, 1000, r"Fe$_3$O$_4$", color=INK2, fontsize=10, ha="center")
ax.text(0.14, 990, "Fe", color=INK2, fontsize=10, ha="center")
ax.text(0.54, 1105, "FeO", color=INK2, fontsize=10, ha="center")
ax.text(0.033, 620, "Fe", color=INK2, fontsize=8, ha="center")

# eutectoid point: circle and label 
x_eut = 1.0 - float(x_h2("E1", np.array([T_C]))[0]) / 100.0
ax.plot([x_eut], [T_C], marker="o", ms=4, color=INK, zorder=5)
ax.text(x_eut - 0.027, 812, "Eutectoid Point", color=MUTED, fontsize=7,
        ha="right")

save_fig(fig, "T1_baur_glaessner")

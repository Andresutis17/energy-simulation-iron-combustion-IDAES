"""
Shared physical property data for the reaction packages.

Heat capacities use the NIST Shomate equation.
"""
import numpy as np

R = 8.314  # J/mol/K

# Molecular weights 
MW_GAS = {"O2": 0.032, "N2": 0.028, "CO2": 0.044, "H2O": 0.018, "H2": 0.002016}        # kg/mol
MW_SOLID = {"Fe2O3": 0.15969, "Fe3O4": 0.231533, "FeO": 0.071844, "Fe": 0.055845, "Al2O3": 0.10196}  # kg/mol

# Skeletal densities
RHO_SOLID = {"Fe2O3": 5250.0, "Fe3O4": 5170.0, "FeO": 5700.0, "Fe": 7874.0, "Al2O3": 3987.0}  # kg/m3

# Shomate coefficients
SHOMATE_GAS = {
    "O2":   [30.03235, 8.772972, -3.988133, 0.788313, -0.741599],
    "N2":   [19.50583, 19.88705, -8.598535, 1.369784, 0.527601],
    "CO2":  [24.99735, 55.18696, -33.69137, 7.948387, -0.136638],
    "H2O":  [30.092, 6.832514, 6.793435, -2.53448, 0.082139],
    "H2":   [18.563083, 12.257357, -2.859786, 0.268238, 1.97799],
}
SHOMATE_SOLID = {
    "Fe2O3": [110.9362, 32.04714, -9.192333, 0.901506, 5.433677],
    "Fe3O4": [200.832, 1.586435e-7, -6.661682e-8, 9.452452e-9, 3.18602e-8],
    "FeO":   [45.7512, 18.78553, -5.952201, 0.852779, -0.081265],
    "Fe":    [23.97449, 8.36775, 0.000277, -0.000086, -0.000005],
    "Al2O3": [102.429, 38.7498, -15.9109, 2.628181, -3.007551],
}


def cp_mol_shomate(T, coeffs):
    """
    Molar heat capacity [J/mol/K] from Shomate, T in K.
    """
    t = T/1000.0
    A, B, C, D, E = coeffs
    return A + B*t + C*t**2 + D*t**3 + E/t**2


def rho_skeletal(w):
    """
    Skeletal particle density [kg/m3] by inverse volume additivity
    """
    return 1.0 / sum(w[c]/RHO_SOLID[c] for c in RHO_SOLID if c in w and w[c] > 0)


def cp_solid_mass_shomate(w, T):
    """
    Solid mass heat capacity [J/kg/K] 
    """
    return sum(cp_mol_shomate(T, SHOMATE_SOLID[c]) * w[c]/MW_SOLID[c]
               for c in SHOMATE_SOLID if c in w and w[c] > 0)


def cp_gas_mol_shomate(T, y):
    """
    Gas molar heat capacity [J/mol/K] 
    """
    return sum(y[c] * cp_mol_shomate(T, SHOMATE_GAS[c])
               for c in SHOMATE_GAS if c in y and y[c] > 0)

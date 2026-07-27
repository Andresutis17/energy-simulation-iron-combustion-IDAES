"""
OC oxidation reaction package:  4 Fe3O4 + O2 -> 6 Fe2O3

Kinetics identical to the IDAES example
"""
import numpy as np

R = 8.314                    # J/mol/K

NAME = "OC oxidation: 4 Fe3O4 + O2 -> 6 Fe2O3"

PARAMS = {
    "k0":           3.1e-4,       # m/s
    "Ea":           14000.0,      # J/mol
    "rxn_order":    1.0,          # [-]
    "dens_mol_sol": 22472.0,      # mol/m3
    "a_vol":        0.28,         # [-]
    "grain_r":      2.6e-7,       # m
    "eps":          1e-8,         # mol/m3
    "nu_O2":        1.0,
    "nu_Fe3O4":     4.0,
    "nu_Fe2O3":     6.0,
    "dH":          -469.4432e3,   # J/mol_rxn
    "cp_g":         31.0,         # J/mol/K
    "MW_Fe2O3":     0.15969,      # kg/mol
    "MW_Fe3O4":     0.23153,      # kg/mol
}

_RHO_SOLID = {"Fe2O3": 5250.0, "Fe3O4": 5000.0, "Al2O3": 3987.0}   # kg/m3
_CP_SOLID = {"Fe2O3": 680.0, "Fe3O4": 650.0, "Al2O3": 1050.0}      # J/kg/K


def rho_skeletal(w_Fe2O3, w_Fe3O4, w_Al2O3):
    """
    Skeletal density of the OC particle by inverse volume additivity [kg/m3]
    """
    return 1.0 / (w_Fe2O3/_RHO_SOLID["Fe2O3"]
                  + w_Fe3O4/_RHO_SOLID["Fe3O4"]
                  + w_Al2O3/_RHO_SOLID["Al2O3"])


def cp_solid(w_Fe2O3, w_Fe3O4, w_Al2O3):
    """
    Solid heat capacity [J/kg/K] by mass fraction mixing

    """
    return (w_Fe2O3*_CP_SOLID["Fe2O3"]
            + w_Fe3O4*_CP_SOLID["Fe3O4"]
            + w_Al2O3*_CP_SOLID["Al2O3"])


def OC_conv(w_Fe2O3, w_Fe3O4, p=PARAMS):
    """
    Oxygen carrier conversion fraction 
    """
    denom = w_Fe2O3 + (p["MW_Fe2O3"]/p["MW_Fe3O4"]) * (p["nu_Fe2O3"]/p["nu_Fe3O4"]) * w_Fe3O4
    return w_Fe2O3 / denom if denom > 0 else 0.0


def k_arrhenius(T_solid, p=PARAMS):
    """
    Arrhenius rate constant[m/s]
    """
    return p["k0"] * np.exp(-p["Ea"] / (R * T_solid))


def rate(C_O2, T_solid, OC_conv_, w_Fe3O4, phi, rho_skel, p=PARAMS):
    """
    Reaction rate [mol_rxn / m3 / s]
    """
    k   = k_arrhenius(T_solid, p)
    Cs  = np.sqrt(C_O2**2 + p["eps"]**2)
    OCT = (1.0 - OC_conv_) ** (2.0/3.0)
    return (w_Fe3O4 * (1-phi) * rho_skel * (p["a_vol"]/p["MW_Fe3O4"]) * 3
            * k * Cs**p["rxn_order"] * OCT
            / (p["dens_mol_sol"] * p["grain_r"]))


class Package:
    """
    OC oxidation
    """
    NAME = "OC oxidation: 4 Fe3O4 + O2 -> 6 Fe2O3"
    GAS_REACTIVE = "O2"; GAS_INERTS = ("N2",); GAS_PRODUCTS = ()
    SOLID_INERTS = ("Al2O3",); SOLID_REACTIVE = ("Fe2O3", "Fe3O4")
    REACTIONS = ("R1",)
    STOICH = {"R1": {"O2": -1.0, "Fe3O4": -4.0, "Fe2O3": +6.0}}
    DH = {"R1": -469.4432e3}             # J/mol
    MW_SOLID = {"Fe2O3": PARAMS["MW_Fe2O3"], "Fe3O4": PARAMS["MW_Fe3O4"], "Al2O3": 0.10196}
    KBE_DIV = 5.0

    def __init__(self, inlet_solid, inlet_gas, phi=0.27, particle_dia=1.5e-3, bed_pressure=1.86e5):
        self.phi = phi
        self.particle_dia = particle_dia
        self.P = bed_pressure
        self.w_in = dict(inlet_solid)
        self.y_in = dict(inlet_gas)

    def OC_conv(self, w):
        return OC_conv(w.get("Fe2O3", 0.0), w.get("Fe3O4", 0.0))

    def rho_skeletal(self, w):
        return rho_skeletal(w.get("Fe2O3", 0.0), w.get("Fe3O4", 0.0), w.get("Al2O3", 0.0))

    def rates(self, C_reactive, T_solid, w, C_product=None):
        """Per-reaction rate [mol_rxn / m3_particle / s]."""
        oc = self.OC_conv(w)
        rsk = self.rho_skeletal(w)
        r = rate(C_reactive, T_solid, oc, w.get("Fe3O4", 0.0), self.phi, rsk)
        return np.array([r])

    def cp_solid_mass(self, w, T):
        return cp_solid(w.get("Fe2O3", 0.0), w.get("Fe3O4", 0.0), w.get("Al2O3", 0.0))

    def cp_gas_mol(self, T, y):
        return PARAMS["cp_g"]

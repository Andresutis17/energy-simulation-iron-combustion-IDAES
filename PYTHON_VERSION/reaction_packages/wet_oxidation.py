"""Wet oxidation (Kuhn 2022) reaction package

  OW1:  Fe   + H2O -> FeO   + H2
  OW2:  3FeO + H2O -> Fe3O4 + H2


"""
import numpy as np
from reaction_packages import common_properties

R = 8.314


class Package:
    NAME = "Wet oxidation (Kuhn 2022): Fe+H2O->FeO+H2; 3FeO+H2O->Fe3O4+H2"
    GAS_REACTIVE = "H2O"; GAS_INERTS = ("N2", "O2", "CO2"); GAS_PRODUCTS = ("H2",)
    SOLID_INERTS = ("Al2O3", "Fe2O3"); SOLID_REACTIVE = ("Fe", "FeO", "Fe3O4")
    REACTIONS = ("OW1", "OW2")
    STOICH = {"OW1": {"H2O": -1.0, "H2": +1.0, "Fe": -1.0, "FeO": +1.0},
              "OW2": {"H2O": -1.0, "H2": +1.0, "FeO": -3.0, "Fe3O4": +1.0}}
    DH = {"OW1": -30.214e3, "OW2": -62.914e3}                 # J/mol_rxn
    MW_SOLID = common_properties.MW_SOLID
    KBE_DIV = 8.0
    SOLID_REACTANT = {"OW1": "Fe", "OW2": "FeO"}
    PARAMS = dict(k0={"OW1": 6.5e7, "OW2": 2.2e7},            # m3/mol/s
                  Ea={"OW1": 231.0e3, "OW2": 235.0e3},        # J/mol
                  ng={"OW1": 1.2, "OW2": 1.0}, eps=1e-8, C_ref=1.0)

    def __init__(self, inlet_solid, inlet_gas, phi=0.27, particle_dia=1.5e-3, bed_pressure=1e5):
        self.phi = phi
        self.particle_dia = particle_dia
        self.P = bed_pressure
        self.w_in = dict(inlet_solid)
        self.y_in = dict(inlet_gas)

    def rho_skeletal(self, w):
        return common_properties.rho_skeletal(w)

    def cp_solid_mass(self, w, T):
        return common_properties.cp_solid_mass_shomate(w, T)

    def cp_gas_mol(self, T, y):
        return common_properties.cp_gas_mol_shomate(T, y)

    def rates(self, C_H2O, T_solid, w, C_product=None):
        """
        Reaction rate [mol_rxn / m3_particle / s] 
        """
        P = self.PARAMS
        rho_skel = common_properties.rho_skeletal(w)
        C_H2O_s = np.sqrt(C_H2O**2 + P["eps"]**2)
        out = np.zeros(2)
        for i, rxn in enumerate(self.REACTIONS):
            comp = self.SOLID_REACTANT[rxn]
            C_solid = (w.get(comp, 0.0)/common_properties.MW_SOLID[comp])*(1-self.phi)*rho_skel
            k = P["k0"][rxn]*np.exp(-P["Ea"][rxn]/(R*T_solid))
            ng = P["ng"][rxn]
            out[i] = k*C_solid*C_H2O_s*(C_H2O_s/P["C_ref"])**(ng-1.0)
        return out

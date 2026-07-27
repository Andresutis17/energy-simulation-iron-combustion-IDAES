"""
Reduction CORAL  reaction package

R1: 3 Fe2O3 + H2 -> 2 Fe3O4 + H2O    
R2:   Fe3O4 + H2 <-> 3 FeO + H2O     
R3:     FeO + H2 <-> Fe + H2O        
R4: 0.25 Fe3O4 + H2 <-> 0.75 Fe + H2O 


"""
import numpy as np
from reaction_packages import common_properties

R = 8.314


class Package:
    NAME = "Reduction CORAL: Fe2O3 -> Fe3O4 -> FeO -> Fe with H2"
    GAS_REACTIVE = "H2"; GAS_INERTS = ("N2", "O2", "CO2"); GAS_PRODUCTS = ("H2O",)
    SOLID_INERTS = ("Al2O3",); SOLID_REACTIVE = ("Fe2O3", "Fe3O4", "FeO", "Fe")
    REACTIONS = ("R1", "R2", "R3", "R4")
    STOICH = {"R1": {"H2": -1.0, "H2O": +1.0, "Fe2O3": -3.0, "Fe3O4": +2.0},
              "R2": {"H2": -1.0, "H2O": +1.0, "Fe3O4": -1.0, "FeO": +3.0},
              "R3": {"H2": -1.0, "H2O": +1.0, "FeO": -1.0, "Fe": +1.0},
              "R4": {"H2": -1.0, "H2O": +1.0, "Fe3O4": -0.25, "Fe": +0.75}}
    DH = {"R1": -7.1048e3, "R2": +62.9476e3, "R3": +30.2136e3, "R4": +38.3971e3}  # J/mol_rxn
    MW_SOLID = common_properties.MW_SOLID
    REACTANT = {"R1": "Fe2O3", "R2": "Fe3O4", "R3": "FeO", "R4": "Fe3O4"}
    NU_REACTANT = {"R1": 3.0, "R2": 1.0, "R3": 1.0, "R4": 0.25}
    PARAMS = dict(
        k0={"R1": 0.58, "R2": 1.35, "R3": 1.35, "R4": 1.35},            # m3/mol/s
        Ea={"R1": 35.6e3, "R2": 49.2e3, "R3": 49.2e3, "R4": 49.2e3},    # J/mol
        order={"R1": 1.1, "R2": 1.1, "R3": 1.1, "R4": 1.1},
        steam_order={"R1": 0.0, "R2": 0.3, "R3": 2.4, "R4": 2.2},
        psize_exp={"R1": 0.0, "R2": 0.23, "R3": 0.23, "R4": 0.23},
        reversible=("R2", "R3", "R4"),
        Keq_A={"R2": 6.6567, "R3": 0.8513, "R4": 1.5131},
        Keq_B={"R2": 6476.4, "R3": 1395.5, "R4": 1394.9},
        Keq_C={"R2": 181141.0, "R3": 253791.0, "R4": 745520.0},
        T_sinter=873.0, f_Fe_high=1.0, f_Fe_low=0.7, f_Fe_delta=30.0,
        T_regime=860.0, regime_tau=5.0, rp_ref=30e-6, eps=1e-8, C_ref=1.0)

    def __init__(self, inlet_solid, inlet_gas, phi=0.27, particle_dia=60e-6, bed_pressure=1e5):
        self.phi = phi
        self.particle_dia = particle_dia
        self.P = bed_pressure
        self.w_in = dict(inlet_solid)
        self.y_in = dict(inlet_gas)

    def rho_skeletal(self, w): return common_properties.rho_skeletal(w)
    def cp_solid_mass(self, w, T): return common_properties.cp_solid_mass_shomate(w, T)
    def cp_gas_mol(self, T, y): return common_properties.cp_gas_mol_shomate(T, y)

    def rates(self, C_H2, T_solid, w, C_product=None):
        P = self.PARAMS; eps = P["eps"]; MW = common_properties.MW_SOLID
        C_H2_s = np.sqrt(C_H2**2 + eps**2)
        rp = self.particle_dia/2.0
        C_H2O = C_product if C_product is not None else 0.0
        C_H2O_s = np.sqrt(C_H2O**2 + eps**2)
        rho_skel = common_properties.rho_skeletal(w)
        f_Fe = P["f_Fe_low"] + (P["f_Fe_high"]-P["f_Fe_low"])/(1+np.exp(-(P["T_sinter"]-T_solid)/P["f_Fe_delta"]))
        out = np.zeros(4)
        for i, r in enumerate(self.REACTIONS):
            reac = self.REACTANT[r]
            C_react = (w.get(reac, 0.0)/MW[reac])*(1-self.phi)*rho_skel
            k = P["k0"][r]*np.exp(-P["Ea"][r]/(R*T_solid))*f_Fe
            kc = k*C_H2_s*(C_H2_s/P["C_ref"])**(P["order"][r]-1.0)*(P["rp_ref"]/rp)**P["psize_exp"][r]
            if r in P["reversible"] and P["steam_order"][r] > 0:
                Keq = np.exp(P["Keq_A"][r] - P["Keq_B"][r]/T_solid - P["Keq_C"][r]/T_solid**2)
                xi = max(1.0 - C_H2O_s/(C_H2_s*Keq + 1e-12), 0.0)
                kc *= xi**P["steam_order"][r]
            out[i] = (2.0/self.NU_REACTANT[r])*C_react*kc
        w_reg = 0.5 + np.arctan((T_solid-P["T_regime"])/P["regime_tau"])/np.pi
        out[2] *= w_reg
        out[3] *= 1.0-w_reg
        return out

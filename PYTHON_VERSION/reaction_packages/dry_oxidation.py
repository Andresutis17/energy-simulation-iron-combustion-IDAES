"""
Dry oxidation CORAL

R1:  2 Fe + 3/2 O2 -> Fe2O3

"""
import numpy as np
from reaction_packages import common_properties

R = 8.314


class Package:
    NAME = "Dry oxidation CORAL: 2 Fe + 1.5 O2 -> Fe2O3"
    GAS_REACTIVE = "O2"; GAS_INERTS = ("N2", "CO2", "H2O", "H2"); GAS_PRODUCTS = ()
    SOLID_INERTS = ("Al2O3", "Fe3O4", "FeO"); SOLID_REACTIVE = ("Fe", "Fe2O3")
    REACTIONS = ("R1",)
    STOICH = {"R1": {"O2": -1.5, "Fe": -2.0, "Fe2O3": +1.0}}
    DH = {"R1": -825.5032e3}                                   # J/mol_rxn
    MW_SOLID = common_properties.MW_SOLID
    PARAMS = dict(
        k_chr_0=0.561, E_chr=45500.0, n_chr=1.0, n_k_rp=0.63,
        D_g_0=2.0e-6, E_g=10000.0, D_s_0=7.28e13, E_s=367300.0, n_dif=1.0, n_D_rp=2.0,
        Xchr_O2_a=0.8351, Xchr_O2_b=-2073.0,
        a_O2_0=2.127e-4, a_O2_a=4.28e-9, a_O2_b=11560.0,
        b_O2_a=40.0, b_O2_b=-3060.0, n_X_rp=0.6,
        rp_ref=30e-6, delta_smooth=0.02, eps=1e-8)

    def __init__(self, inlet_solid, inlet_gas, phi=0.27, particle_dia=60e-6, bed_pressure=1e5):
        self.phi = phi
        self.particle_dia = particle_dia
        self.P = bed_pressure
        self.w_in = dict(inlet_solid)
        self.y_in = dict(inlet_gas)
        rho_skel_in = common_properties.rho_skeletal(self.w_in)
        self.C0_Fe = (self.w_in.get("Fe", 0.0)/common_properties.MW_SOLID["Fe"])*(1-phi)*rho_skel_in

    def OC_conv(self, w):
        MW_F2 = common_properties.MW_SOLID["Fe2O3"]; MW_Fe = common_properties.MW_SOLID["Fe"]
        denom = w.get("Fe2O3", 0.0)*2.0/MW_F2 + w.get("Fe", 0.0)/MW_Fe
        return w.get("Fe2O3", 0.0)*2.0/MW_F2/denom if denom > 0 else 0.0

    def rho_skeletal(self, w): return common_properties.rho_skeletal(w)
    def cp_solid_mass(self, w, T): return common_properties.cp_solid_mass_shomate(w, T)
    def cp_gas_mol(self, T, y): return common_properties.cp_gas_mol_shomate(T, y)

    def rates(self, C_O2, T_solid, w, C_product=None):
        P = self.PARAMS; X = self.OC_conv(w)
        C_O2_s = np.sqrt(C_O2**2 + P["eps"]**2)
        rp = self.particle_dia/2.0
        k_chr = P["k_chr_0"]*(P["rp_ref"]/rp)**P["n_k_rp"]*np.exp(-P["E_chr"]/(R*T_solid))
        dXdt_I = 3.0*k_chr*C_O2_s**P["n_chr"]*max(1.0-X, 0.0)**(2.0/3.0)
        D_eff = (P["D_g_0"]*np.exp(-P["E_g"]/(R*T_solid))
                 + P["D_s_0"]*np.exp(-P["E_s"]/(R*T_solid)))*(P["rp_ref"]/rp)**P["n_D_rp"]
        Xchr_O2 = P["Xchr_O2_a"]*np.exp(P["Xchr_O2_b"]/T_solid)
        a_O2 = P["a_O2_0"] + P["a_O2_a"]*np.exp(P["a_O2_b"]/T_solid)
        b_O2 = P["b_O2_a"]*np.exp(P["b_O2_b"]/T_solid)
        X_chr = (Xchr_O2 + a_O2*np.exp(b_O2*C_O2_s))*(P["rp_ref"]/rp)**P["n_X_rp"]
        if X_chr < 1.0 and X > X_chr:
            X_dif = (X-X_chr)/(1.0-X_chr); omX = max(1.0-X_dif, 1e-6)
            dXdt_II = ((1.0-X_chr)*1.5*D_eff*C_O2_s**P["n_dif"]
                       * omX**(5.0/3.0)/(1.0-omX**(1.0/3.0)+1e-8))
        else:
            dXdt_II = 0.0
        sigmoid_w = 0.5 + np.arctan((X-X_chr)/P["delta_smooth"]*np.pi/2.0)/np.pi
        dXdt = (1.0-sigmoid_w)*dXdt_I + sigmoid_w*dXdt_II
        return np.array([dXdt*self.C0_Fe/2.0])      # mol_rxn/m3_particle/s

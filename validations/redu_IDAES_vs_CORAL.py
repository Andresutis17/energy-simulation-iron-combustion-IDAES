"""
Validates CORAL reduction conversion vs time using rate expressions from reduction_reactions.py
Integrates batch reactor ODEs same with TGA conditions and compares with CORAL analytical curves
The approach reads kinetic parameters directly from HeteroReactionParameterBlock.
Uses the same rate expressions as the BFB model, integrates dw/dt via scipy to get X(t), 
and compares IDAES X(t) with CORAL analytical X(t)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import exp

from pyomo.environ import ConcreteModel, value
from custom_properties.reduction_reactions import ReductionReactionParameterBlock
from custom_properties.gas_phase_thermo import CustomGasPhaseParameterBlock
from custom_properties.solid_phase_thermo import CustomSolidPhaseParameterBlock


# Package parameters

m_params = ConcreteModel()
m_params.gas_props = CustomGasPhaseParameterBlock()
m_params.sol_props = CustomSolidPhaseParameterBlock()
m_params.hetero_rxns = ReductionReactionParameterBlock(
    solid_property_package=m_params.sol_props,
    gas_property_package=m_params.gas_props,
)

# Kinetic parameters
rxns = m_params.hetero_rxns

kinetic = {}
for r in rxns.rate_reaction_idx:
    kinetic[r] = {
        "k0": value(rxns.k0_rxn[r]),
        "Ea": value(rxns.energy_activation[r]),
        "n": value(rxns.rxn_order[r]),
        "s": value(rxns.rxn_steam_order[r]),
        "r_size": value(rxns.particle_size_exp[r]),
    }

keq_coeff = {}
for r in rxns.reversible_rxn_idx:
    keq_coeff[r] = {
        "A": value(rxns.Keq_A[r]),
        "B": value(rxns.Keq_B[r]),
        "C": value(rxns.Keq_C[r]),
    }

dh_rxn = {}
for r in rxns.rate_reaction_idx:
    dh_rxn[r] = value(rxns.dh_rxn[r])

stoich = rxns.rate_reaction_stoichiometry

# Solid properties
sol = m_params.sol_props
mw = {j: value(sol.mw_comp[j]) for j in sol.component_list}
RHO_SKEL = {j: value(sol.dens_mass_comp_skeletal[j]) for j in sol.component_list}


particle_diam = value(sol.particle_dia)  # m
particle_porosity = 0.27  
rp_ref = value(rxns.rp_ref)  # 30e-6 m

comps = ["Fe2O3", "Fe3O4", "FeO", "Fe"]

comps_koeff_abs = {"R1": 3.0, "R2": 1.0, "R3": 1.0, "R4": 0.25}


# C0 = (1-porosity) * rho_skel*(1/mw)
c0_ref = {}
for comp in ["Fe2O3", "Fe3O4", "FeO"]:
    c0_ref[comp] = (1.0 / mw[comp]) * (1.0 - particle_porosity) * RHO_SKEL[comp]

# Molecular weights
print(f"  mw: Fe2O3={mw['Fe2O3']:.5f}, Fe3O4={mw['Fe3O4']:.5f}, FeO={mw['FeO']:.5f}, Fe={mw['Fe']:.5f}")
print(f"  Kinetic params: {list(rxns.rate_reaction_idx)}")
for r in rxns.rate_reaction_idx:
    print(f"    {r}: k0={kinetic[r]['k0']}, Ea={kinetic[r]['Ea']}, n={kinetic[r]['n']}, s={kinetic[r]['s']}, r_sz={kinetic[r]['r_size']}")
print(f"  dh_rxn: {dh_rxn}")
print(f"  rp_ref: {rp_ref:.2e} m")
print(f"  c0_ref: Fe2O3={c0_ref['Fe2O3']:.0f}, Fe3O4={c0_ref['Fe3O4']:.0f}, FeO={c0_ref['FeO']:.0f} mol/m3")



# Same speed functions as in hetero_reactions.py

def keq_calc(T, r_id):

    """

    Compute equilibrium constant

    """
    c = keq_coeff[r_id]
    return exp(c["A"] - c["B"] / T - c["C"] / T**2)


def kc_intrinsic(T, r_id, active_fe=1.0):

    """

    Arrhenius rate constant [m^(3n)/mol^n/s]

    """
    Rg = 8.314
    k = kinetic[r_id]["k0"] * exp(-kinetic[r_id]["Ea"] / (Rg * T)) * active_fe
    return k


def xi_smooth(C_H2O, C_H2, Keq, eps=1e-8):

    """

    Equilibrium driving force with smoothing

    """
    ratio = np.sqrt(C_H2O**2 + eps**2) / (np.sqrt(C_H2**2 + eps**2) * Keq)
    xi_raw = 1.0 - ratio
    return 0.5 * (xi_raw - eps + np.sqrt((xi_raw - eps)**2 + eps**2)) + eps


def C_smooth(C):

    """

    Smooth concentration

    """
    eps = 1e-8
    return np.sqrt(C**2 + eps**2)



# IDAES ReactionBlock

from pyomo.environ import Set
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from idaes.core import FlowsheetBlock

m_val = ConcreteModel()
m_val.fs = FlowsheetBlock(dynamic=False)

m_val.fs.gas_state = m_params.gas_props.state_block_class(
    [0], parameters=m_params.gas_props, defined_state=True,
)
m_val.fs.solid_state = m_params.sol_props.state_block_class(
    [0], parameters=m_params.sol_props, defined_state=True,
)

rxn_blk = m_params.hetero_rxns.build_reaction_block(
    [0],
    gas_state_block=m_val.fs.gas_state,
    solid_state_block=m_val.fs.solid_state,
    has_equilibrium=False,
)
rxn_blk.construct()
b = rxn_blk[0]


# f_Fe sigmoid parameters 
_fFe_sinter = value(m_params.hetero_rxns.T_sinter)
_fFe_delta = value(m_params.hetero_rxns.f_Fe_delta)
_fFe_low = value(m_params.hetero_rxns.f_Fe_low)
_fFe_high = value(m_params.hetero_rxns.f_Fe_high)


def f_Fe_sigmoid(T_val):

    """

    Sigmoid with f_Fe values

    """
    return _fFe_low + (_fFe_high - _fFe_low) / (1.0 + exp(-(_fFe_sinter - T_val) / _fFe_delta))


def set_idaes_state(T_val, C_gas_dict, rp_um):

    """

    Fix state variables and evaluate derived properties

    """
    rp_m = rp_um * 1e-6
    gs = b.gas_state_ref

    gs.temperature.fix(T_val)
    gs.pressure.fix(101325.0)
    gs.flow_mol.fix(1.0)
    for j in m_params.gas_props.component_list:
        y = C_gas_dict.get(j, 0.0) * 8.314 * T_val / 101325.0
        gs.mole_frac_comp[j].fix(max(y, 1e-20))

    b.solid_state_ref.temperature.fix(T_val)
    b.solid_state_ref.particle_porosity.fix(0.27)
    b.solid_state_ref.flow_mass.fix(200.0)
    for j in m_params.sol_props.component_list:
        b.solid_state_ref.mass_frac_comp[j].fix(1.0 if j == "Fe2O3" else 0.0)

    
    b.solid_state_ref.params.particle_dia.set_value(rp_m)

    
    if not hasattr(gs, "ideal_gas"):
        try:
            _ = gs.dens_mol  # triggers _dens_mol builds ideal_gas constraint
        except Exception:
            pass
    if not hasattr(gs, "comp_conc_eqn"):
        try:
            _ = gs.dens_mol_comp  # triggers _dens_mol_comp builds comp_conc_eqn
        except Exception:
            pass

    # Evaluation of gas derived properties 
    if hasattr(gs, "ideal_gas"):
        calculate_variable_from_constraint(gs.dens_mol, gs.ideal_gas)
    if hasattr(gs, "comp_conc_eqn"):
        for j in m_params.gas_props.component_list:
            calculate_variable_from_constraint(gs.dens_mol_comp[j], gs.comp_conc_eqn[j])


def eval_idaes_block(T_val, C_gas_dict, rp_um, X_step_dict, active_fe=1.0):

    """

    Evaluate all reaction block variables at given conditions
    
    """
    set_idaes_state(T_val, C_gas_dict, rp_um)

    # k_rxn 
    for j in m_params.hetero_rxns.rate_reaction_idx:
        calculate_variable_from_constraint(b.k_rxn[j], b.rate_constant_eqn[j])

    # Keq_red
    for j in m_params.hetero_rxns.reversible_rxn_idx:
        calculate_variable_from_constraint(b.Keq_red[j], b.Keq_red_eqn[j])

    # xi_red
    for j in m_params.hetero_rxns.reversible_rxn_idx:
        calculate_variable_from_constraint(b.xi_red[j], b.xi_red_eqn[j])

    # kc_full 
    _rxn_to_step = {"R1": "I", "R2": "II", "R3": "III", "R4": "II"}
    _c0_reactant = {"R1": "Fe2O3", "R2": "Fe3O4", "R3": "FeO", "R4": "Fe3O4"}
    
    for s in m_params.hetero_rxns.step_idx:
        r_id = {"I": "R1", "II": "R2", "III": "R3"}[s]
        _n_val = value(m_params.hetero_rxns.rxn_order[r_id])
        _s_val = value(m_params.hetero_rxns.rxn_steam_order[r_id])
        _rp_val = value(m_params.hetero_rxns.particle_size_exp[r_id])
        _rp_ref_val = value(m_params.hetero_rxns.rp_ref)
        _dia_val = value(b.solid_state_ref.params.particle_dia)
        _eps_val = value(m_params.hetero_rxns.eps)
        _C_H2_val = value(b.gas_state_ref.dens_mol_comp["H2"])
        _C_H2_s = (_C_H2_val**2 + _eps_val**2)**0.5
        _eq_val = 1.0
        if r_id in m_params.hetero_rxns.reversible_rxn_idx and _s_val > 0:
            _eq_val = value(b.xi_red[r_id]) ** _s_val
        _sz_val = (_rp_ref_val / (_dia_val / 2)) ** _rp_val
        b.kc_full[s].set_value(
            value(b.k_rxn[r_id]) * _C_H2_s ** _n_val * _eq_val * _sz_val
        )

    # Fix X_conv for rate evaluation
    b.X_conv_I.fix(X_step_dict.get("I", 0.01))
    b.X_conv_II.fix(X_step_dict.get("II", 0.001))
    b.X_conv_III.fix(X_step_dict.get("III", 0.001))

    # Fix c0 reference concentrations
    b.C0_Fe2O3.fix(c0_ref["Fe2O3"])
    b.C0_Fe3O4.fix(c0_ref["Fe3O4"])
    b.C0_FeO.fix(c0_ref["FeO"])

    # Reaction_rate 
    _eps_rate = 1e-8  # Smoothing constant
    for j in m_params.hetero_rxns.rate_reaction_idx:
        step = _rxn_to_step[j]
        C0 = value(getattr(b, f"C0_{_c0_reactant[j]}"))
        kc_val = value(b.kc_full[step])
        X_val = value(getattr(b, f"X_conv_{step}"))
        avr_val = (_eps_rate + (-np.log(1.0 - X_val + _eps_rate)))**0.5
        b.reaction_rate[j].set_value(
            (2.0 / comps_koeff_abs[j]) * C0 * kc_val * (1.0 - X_val) * avr_val
        )


    # Collect values
    result = {
        "k_rxn": {j: value(b.k_rxn[j]) for j in m_params.hetero_rxns.rate_reaction_idx},
        "keq_red": {j: value(b.Keq_red[j]) for j in m_params.hetero_rxns.reversible_rxn_idx},
        "xi_red": {j: value(b.xi_red[j]) for j in m_params.hetero_rxns.reversible_rxn_idx},
        "kc_full": {s: value(b.kc_full[s]) for s in m_params.hetero_rxns.step_idx},
        "reaction_rate": {j: value(b.reaction_rate[j]) for j in m_params.hetero_rxns.rate_reaction_idx},
    }
    return result


def idaes_kc_per_step(T_val, C_gas_dict, rp_um, regime="3step"):

    """

    Evaluate kc for each reduction step using IDAES parameters
    Computes k_arr manually (f_Fe sigmoid isnt applied)
    Reads gas state and xi_red via Pyomo Constraints
    Returns dict {"I": kc_I, "II": kc_II, "III": kc_III}

    """
    _step_to_rxn_regime = {
        "I": "R1",
        "II": "R2" if regime == "3step" else "R4",
        "III": "R3",
    }
    Rg = 8.314
    eps_val = 1e-8

    set_idaes_state(T_val, C_gas_dict, rp_um)

    # Evaluation of Keq and xi_red 
    for j in m_params.hetero_rxns.reversible_rxn_idx:
        calculate_variable_from_constraint(b.Keq_red[j], b.Keq_red_eqn[j])
    for j in m_params.hetero_rxns.reversible_rxn_idx:
        calculate_variable_from_constraint(b.xi_red[j], b.xi_red_eqn[j])

    result = {}
    for step in ["I", "II", "III"]:
        r = _step_to_rxn_regime[step]
        # Arrhenius without f_Fe sigmoid
        k_arr = kinetic[r]["k0"] * exp(-kinetic[r]["Ea"] / (Rg * T_val))
        # H2 concentration term
        C_H2 = value(b.gas_state_ref.dens_mol_comp["H2"])
        C_H2_s = (C_H2**2 + eps_val**2)**0.5
        h2_term = C_H2_s ** kinetic[r]["n"]
        # Equilibrium driving force
        eq_term = 1.0
        if r in keq_coeff and kinetic[r]["s"] > 0:
            xi = value(b.xi_red[r])
            eq_term = xi ** kinetic[r]["s"]
        # Particle size correction
        dia_val = value(b.solid_state_ref.params.particle_dia)
        size_term = (rp_ref / (dia_val / 2)) ** kinetic[r]["r_size"]

        result[step] = k_arr * h2_term * eq_term * size_term

    return result


def numpy_reference(T_val, C_gas_dict, rp_um, X_step_dict, active_fe=None):

    """

    Uses numpy to evalute reference values, same as reduction_reactions
    Returns the same structure as eval_idaes_block

    """
    rp_m = rp_um * 1e-6
    f_Fe = f_Fe_sigmoid(T_val)
    result = {"k_rxn": {}, "keq_red": {}, "xi_red": {}, "kc_full": {}, "reaction_rate": {}}

    for r in kinetic:
        result["k_rxn"][r] = kc_intrinsic(T_val, r, f_Fe)

    for r in keq_coeff:
        result["keq_red"][r] = keq_calc(T_val, r)

    for r in keq_coeff:
        Keq = result["keq_red"][r]
        result["xi_red"][r] = xi_smooth(
            C_gas_dict.get("H2O", 0), C_gas_dict.get("H2", 0), Keq
        )

    for sn in ["I", "II", "III"]:
        result["kc_full"][sn] = kc_step_calc(
            T_val, C_gas_dict.get("H2", 0), C_gas_dict.get("H2O", 0), rp_um, sn, f_Fe
        )

    eps = 1e-8
    _rxn_to_step = {"R1": "I", "R2": "II", "R3": "III", "R4": "II"}
    _koeff_abs = {"R1": 3.0, "R2": 1.0, "R3": 1.0, "R4": 0.25}
    _c0_reactant = {"R1": "Fe2O3", "R2": "Fe3O4", "R3": "FeO", "R4": "Fe3O4"}

    for r in kinetic:
        step = _rxn_to_step[r]
        C0 = c0_ref[_c0_reactant[r]]
        kc = result["kc_full"][step]
        X_step = X_step_dict.get(step, 0.001)
        avr = np.sqrt(eps + max(-np.log(1.0 - X_step + eps), eps))
        result["reaction_rate"][r] = (2.0 / _koeff_abs[r]) * C0 * kc * (1.0 - X_step) * avr

    return result


def step_conversion_from_w(w, step):

    """

    Evaluate per step conversion from mass fractions
    X_I   = 1 - n_Fe2O3 / (n_Fe2O3 + n_Fe3O4 + n_FeO + n_Fe)
    X_II  = 1 - n_Fe3O4 / (n_Fe3O4 + n_FeO + n_Fe)
    X_III = 1 - n_FeO   / (n_FeO + n_Fe)
    wn_comp = w_comp * N_Fe / mw

    """
    eps = 1e-8
    n_Fe2O3 = w["Fe2O3"] * 2.0 / mw["Fe2O3"]
    n_Fe3O4 = w["Fe3O4"] * 3.0 / mw["Fe3O4"]
    n_FeO = w["FeO"] * 1.0 / mw["FeO"]
    n_Fe = w["Fe"] * 1.0 / mw["Fe"]

    if step == "I":
        n_total = n_Fe2O3 + n_Fe3O4 + n_FeO + n_Fe + eps
        return 1.0 - n_Fe2O3 / n_total
    elif step == "II":
        n_total = n_Fe3O4 + n_FeO + n_Fe + eps
        return 1.0 - n_Fe3O4 / n_total
    elif step == "III":
        n_total = n_FeO + n_Fe + eps
        return 1.0 - n_FeO / n_total
    return 0.0


def avrami_factor(X_step, eps=1e-8):

    """

    Evaluate Avrami nucleation factor sqrt(eps + -ln(1 - X_step + eps))
    Characteristic S-shape of the 2D nuclei model

    """
    return np.sqrt(eps + np.maximum(-np.log(1.0 - X_step + eps), eps))


def reaction_rate_idaes(w, C_gas, T, rp_m, active_fe=1.0):

    """

    Evaluate reaction rates, rate = (2/|v|) * C0 * kc_full * (1-X) * sqrt(-ln(1-X))

    """
    eps = 1e-8
    rates = {}

    
    solid_reactant = {"R1": "Fe2O3", "R2": "Fe3O4", "R3": "FeO", "R4": "Fe3O4"}
    avrami_step   = {"R1": "I",   "R2": "II",  "R3": "III", "R4": "II"}

    for r_id in kinetic:
        reactant = solid_reactant[r_id]

        k_int = kc_intrinsic(T, r_id, active_fe)
        C_H2_smoothed = C_smooth(C_gas["H2"])
        H2_term = C_H2_smoothed ** kinetic[r_id]["n"]

        # Equilibrium driving force
        s = kinetic[r_id]["s"]
        if s > 0 and r_id in keq_coeff:
            Keq = keq_calc(T, r_id)
            xi = xi_smooth(C_gas["H2O"], C_gas["H2"], Keq, eps)
            eq_term = xi ** s
        else:
            eq_term = 1.0

        # Particle size
        rp_ratio = rp_ref / (rp_m / 2)
        size_term = rp_ratio ** kinetic[r_id]["r_size"]

        kc_full = k_int * H2_term * eq_term * size_term

        c0 = c0_ref[reactant]
        X_step = step_conversion_from_w(w, avrami_step[r_id])
        avr = avrami_factor(X_step)

        rates[r_id] = (2.0 / comps_koeff_abs[r_id]) * c0 * kc_full * (1.0 - X_step) * avr

    return rates



# CORAL per step conversion ODEs 


def kc_step_calc(T, C_H2, C_H2O, rp_um, step, active_fe=1.0):

    """

    Evaluate the apparent kinetic constant kc [s^-1] for a reduction step
    
    """
    Rg = 8.314
    rp_m = rp_um * 1e-6
    eps = 1e-8

    step_to_rxn = {"I": "R1", "II": "R2", "II_direct": "R4", "III": "R3"}
    r_id = step_to_rxn.get(step)
    if r_id is None:
        return 0.0

    k0    = kinetic[r_id]["k0"]
    Ea    = kinetic[r_id]["Ea"]
    n     = kinetic[r_id]["n"]
    s_exp = kinetic[r_id]["s"]
    r_exp = kinetic[r_id]["r_size"]

    arrhenius = k0 * exp(-Ea / (Rg * T)) * active_fe

    
    C_H2_s = C_smooth(C_H2)
    H2_term = C_H2_s ** n

    
    eq_term = 1.0
    if s_exp > 0 and r_id in keq_coeff:
        Keq = keq_calc(T, r_id)
        xi = xi_smooth(C_H2O, C_H2, Keq, eps)
        eq_term = xi ** s_exp

    size_term = (rp_ref / (rp_m / 2)) ** r_exp
    return arrhenius * H2_term * eq_term * size_term


def coral_conversion_ode(t, y, T, y_H2, y_H2O, rp_um, active_fe=1.0, regime="3step"):

    """

    CORAL per step conversion ODEs:
    dX_I/dt   = 2 * kc_I   * (1 - X_I)   * sqrt(-ln(1 - X_I))
    dX_II/dt  = 2 * kc_II  * (1 - X_II)  * sqrt(-ln(1 - X_II))
    dX_III/dt = 2 * kc_III * (1 - X_III) * sqrt(-ln(1 - X_III))

    Gas conditions are constant 

    """
    P_atm = 1.0
    C_total = P_atm * 101325 / (8.314 * T)
    C_H2 = y_H2 * C_total
    C_H2O = y_H2O * C_total

    X_I, X_II, X_III = np.clip(y, 1e-15, 1.0 - 1e-8)
    eps = 1e-8

    # Avrami factor
    avr = lambda X: np.sqrt(eps + np.maximum(-np.log(1.0 - X + eps), eps))

    if regime == "3step":
        kc_I = kc_step_calc(T, C_H2, C_H2O, rp_um, "I", active_fe)
        kc_II = kc_step_calc(T, C_H2, C_H2O, rp_um, "II", active_fe)
        kc_III = kc_step_calc(T, C_H2, C_H2O, rp_um, "III", active_fe)
        dydt = np.array([
            2 * kc_I * (1 - X_I) * avr(X_I),
            2 * kc_II * (1 - X_II) * avr(X_II),
            2 * kc_III * (1 - X_III) * avr(X_III),
        ])
    elif regime == "2step":
        kc_I = kc_step_calc(T, C_H2, C_H2O, rp_um, "I", active_fe)
        kc_II = kc_step_calc(T, C_H2, C_H2O, rp_um, "II_direct", active_fe)
        kc_III = 0.0
        dydt = np.array([
            2 * kc_I * (1 - X_I) * avr(X_I),
            2 * kc_II * (1 - X_II) * avr(X_II),
            0.0,
        ])

    return dydt


def mass_fractions_from_X(X_I, X_II, X_III):

    """

    Obtain solid mass fractions from per step conversions.
    Maps CORAL per step conversions to physical composition
    Based on stoichiometric mass balance for 3 mol Fe2O3 initial

    """

    # Moles of each species. 3 moles of Fe2O3, so 6 Fe atoms
    m_Fe2O3 = 3 * (1 - X_I) * mw["Fe2O3"]
    m_Fe3O4 = 2 * X_I * (1 - X_II) * mw["Fe3O4"]
    m_FeO = 6 * X_I * X_II * (1 - X_III) * mw["FeO"]
    m_Fe = 6 * X_I * X_II * X_III * mw["Fe"]

    m_total = m_Fe2O3 + m_Fe3O4 + m_FeO + m_Fe

    if m_total < 1e-30:
        return {"Fe2O3": 1.0, "Fe3O4": 0.0, "FeO": 0.0, "Fe": 0.0}

    return {
        "Fe2O3": m_Fe2O3 / m_total,
        "Fe3O4": m_Fe3O4 / m_total,
        "FeO": m_FeO / m_total,
        "Fe": m_Fe / m_total,
    }


def overall_conversion_from_X(X_I, X_II, X_III):

    """

    Evaluate complete conversion X 
    X = ΔXI·XI + ΔXII·XII + ΔX_III·XIII

    """
    return 0.11 * X_I + 0.22 * X_II + 0.67 * X_III


def conversion_from_w(w, w_Al2O3=0.0):

    """

    Evaluate complete oxygen removal conversion from mass fractions
    X = 1 - (O_current / O_initial), based on O/Fe atom ratio.

    """

    w_total_oxides = sum(w[j] for j in ["Fe2O3", "Fe3O4", "FeO", "Fe"])
    if w_total_oxides < 1e-30:
        return 0.0

    # Moles of O from each oxide
    # Fe2O3 3 O per mole, Fe3O4 4 per mole, FeO 1 per mol, Fe has 0
    n_O_total = (
        w["Fe2O3"] / mw["Fe2O3"] * 3.0
        + w["Fe3O4"] / mw["Fe3O4"] * 4.0
        + w["FeO"] / mw["FeO"] * 1.0
    )

    # Total Fe atoms 
    n_Fe_total = (
        w["Fe2O3"] * 2.0 / mw["Fe2O3"]
        + w["Fe3O4"] * 3.0 / mw["Fe3O4"]
        + w["FeO"] * 1.0 / mw["FeO"]
        + w["Fe"] * 1.0 / mw["Fe"]
    )

    if n_Fe_total < 1e-30:
        return 0.0

    # Fe2O3: O/Fe = 3/2 = 1.5. Fe: O/Fe = 0
    O_per_Fe = n_O_total / n_Fe_total
    return max(0.0, 1.0 - O_per_Fe / 1.5)


def batch_ode(t, y, C_gas, T, rp_m, active_fe, use_reactions):

    """

    Evaluates how the solid mass fractions of Fe2O3, Fe3O4, FeO and Fe change over time in a batch reactor
    The fractions are normalized to sum to 1 because losing oxygen reduces the total solid mass

    """

    w = {}
    for i, comp in enumerate(comps):
        w[comp] = max(y[i], 1e-15)  

    # Normalize mass fractions 
    w_sum = sum(w[j] for j in comps)
    for comp in comps:
        w[comp] = w[comp] / w_sum

    # Skeletal density 
    rho_skel = 1.0 / sum(w[j] / RHO_SKEL[j] for j in comps)

    # rho_bed in kg/m3 
    rho_bed = (1 - particle_porosity) * rho_skel

    # Reaction rates 
    rates = reaction_rate_idaes(w, C_gas, T, rp_m, active_fe)

    # Total solid mass loss rate from all reactions [kg/m3/s]
    mass_loss = 0.0
    for r_id in use_reactions:
        for comp in comps:
            nu = stoich[(r_id, "Sol", comp)]
            mass_loss += nu * mw[comp] * rates[r_id]

    # Normalized mass balance dw/dt = (production - w * mass_loss) / rho_bed
    # Units 1/s. This ensures sum(dw/dt) = 0
    dydt = np.zeros(4)
    for i, comp in enumerate(comps):
        raw_dw = 0.0
        for r_id in use_reactions:
            nu = stoich[(r_id, "Sol", comp)]
            raw_dw += nu * mw[comp] * rates[r_id]
        dydt[i] = (raw_dw - w[comp] * mass_loss) / rho_bed

    return dydt



# Analytical curves 

DX_I = 0.11
DX_II = 0.22
DX_III = 0.67


def X_analytical(t_arr, T, y_H2, y_H2O, rp_um, P_atm=1.0, active_fe_override=None):

    """

    Analytical conversion X(t) using 2D Nuclei model
    Uses kc_step_calc(), same reduction_reactions params + smoothing as reaction_rate_idaes
    so it matches ODE and mass-action curves exactly

    """

    C_total = P_atm * 101325 / (8.314 * T)
    C_H2 = y_H2 * C_total
    C_H2O = y_H2O * C_total

    if T > 860:
        active_fe = active_fe_override if active_fe_override else 1
        # 3 steps route
        kc1 = kc_step_calc(T, C_H2, C_H2O, rp_um, "I", active_fe)
        kc2 = kc_step_calc(T, C_H2, C_H2O, rp_um, "II", active_fe)
        kc3 = kc_step_calc(T, C_H2, C_H2O, rp_um, "III", active_fe)
        X = DX_I * (1 - np.exp(-(kc1*t_arr)**2)) + \
            DX_II * (1 - np.exp(-(kc2*t_arr)**2)) + \
            DX_III * (1 - np.exp(-(kc3*t_arr)**2))
        
    else:
        active_fe = active_fe_override if active_fe_override else 1.0
        # 2 steps route
        kc1 = kc_step_calc(T, C_H2, C_H2O, rp_um, "I", active_fe)
        kc2 = kc_step_calc(T, C_H2, C_H2O, rp_um, "II_direct", active_fe)
        X = DX_I * (1 - np.exp(-(kc1*t_arr)**2)) + \
            0.89 * (1 - np.exp(-(kc2*t_arr)**2))

    return X



# ODE with ReactionBlock rates


def idaes_block_ode(t, y, C_gas, T, rp_um, active_fe, use_reactions):

    """

    ODE using ReactionBlock constraints
    Re evaluates k_rxn, kc_full, reaction_rate at each step via Pyomo.

    """
    w = {comps[i]: max(y[i], 1e-30) for i in range(4)}
    w_Al2O3 = 0.0

    # Compute per step conversions from mass fractions
    X_step = {
        "I": step_conversion_from_w(w, "I"),
        "II": step_conversion_from_w(w, "II"),
        "III": step_conversion_from_w(w, "III"),
    }

    # Evaluate block at current conditions
    idaes_vals = eval_idaes_block(T, C_gas, rp_um, X_step, active_fe)
    rates = idaes_vals["reaction_rate"]

    # Same mass balance as batch_ode
    rho_skel = 1.0 / sum(w[j] / rho_skel[j] for j in comps)
    rho_bed = (1 - particle_porosity) * rho_skel * 1000

    mass_loss = 0.0
    for r_id in use_reactions:
        st = rxns.rate_reaction_stoichiometry
        for j in comps:
            nu = st.get((r_id, "Sol", j), 0)
            mass_loss += nu * mw[j] * rates[r_id]  # kg/s per m3

    # dw_j/dt = mass_loss_j / rho_bed (normalized)
    dw = np.zeros(4)
    for i, j in enumerate(comps):
        nu_sum = 0.0
        for r_id in use_reactions:
            nu_sum += rxns.rate_reaction_stoichiometry.get((r_id, "Sol", j), 0)
        dw[i] = nu_sum * mw[j] * sum(rates.values()) / (rho_bed * 1000.0)

    # Normalize to sum = 1
    w_sum = sum(w[j] for j in comps) + w_Al2O3
    for i in range(4):
        dw[i] = dw[i] / w_sum if w_sum > 0 else 0.0
        y[i] = max(y[i] + dw[i] * 1.0, 0.0) 

    return dw



# Run batch and compare

def run_batch(T, y_H2, y_H2O, rp_um, active_fe=1.0, t_max=1800, label=""):
    
    """

    Runs batch simulation, returns t and 3 X curves for comparison: IDAES (mass action ODE),
    CORAL ODE and CORAL analytical Avrami
      
    """

    # Gas concentrations 
    P_atm = 1.0
    C_total = P_atm * 101325 / (8.314 * T)
    C_gas = {
        "O2": 0.0,
        "N2": (1 - y_H2 - y_H2O) * C_total,
        "CO2": 0.0, 
        "H2O": y_H2O * C_total, 
        "H2": y_H2 * C_total,
    }

    rp_m = rp_um * 1e-6

    # Initial mass fractions, TGA conditions
    rp_m = rp_um * 1e-6

    # Select regime 3 steps if T >= 860K , 2 steps if not
    regime = "3step" if T > 860 else "2step"

    # Time
    # Time
    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, 500)

    # Curve 1: IDAES perstep ODE 
    # Same per step ODE formulation as CORAL ODE, but kc evaluated
    # through IDAES parameter block (Pyomo). f_Fe sigmoid isnt applied
    kc_idaes = idaes_kc_per_step(T, C_gas, rp_um, regime)

    y0_idaes = np.array([1e-15, 1e-15, 1e-15])
    def idaes_ode_func(t, y):
        X_I, X_II, X_III = np.clip(y, 1e-15, 1.0 - 1e-8)
        eps = 1e-8
        avr = lambda X: np.sqrt(eps + np.maximum(-np.log(1.0 - X + eps), eps))
        if regime == "3step":
            return np.array([
                2 * kc_idaes["I"] * (1 - X_I) * avr(X_I),
                2 * kc_idaes["II"] * (1 - X_II) * avr(X_II),
                2 * kc_idaes["III"] * (1 - X_III) * avr(X_III),
            ])
        else:
            return np.array([
                2 * kc_idaes["I"] * (1 - X_I) * avr(X_I),
                2 * kc_idaes["II"] * (1 - X_II) * avr(X_II),
                0.0,
            ])

    try:
        sol_idaes = solve_ivp(idaes_ode_func, t_span, y0_idaes, t_eval=t_eval,
                             method="BDF", rtol=1e-10, atol=1e-12, max_step=5.0)
    except Exception:
        sol_idaes = solve_ivp(idaes_ode_func, t_span, y0_idaes, t_eval=t_eval,
                             method="RK45", rtol=1e-10, atol=1e-12)

    # X_global
    X_IDAES = np.array([
        0.11 * sol_idaes.y[0, k] + 0.22 * sol_idaes.y[1, k] + 0.67 * sol_idaes.y[2, k]
        if regime == "3step"
        else 0.11 * sol_idaes.y[0, k] + 0.89 * sol_idaes.y[1, k]
        for k in range(sol_idaes.y.shape[1])
    ])

    # Curve 2: CORAL per step ODE 
    y0_coral = np.array([1e-15, 1e-15, 1e-15])  

    def coral_ode_func(t, y):
        return coral_conversion_ode(t, y, T, y_H2, y_H2O, rp_um, active_fe, regime)

    try:
        sol_coral = solve_ivp(coral_ode_func, t_span, y0_coral, t_eval=t_eval,
                              method="BDF", rtol=1e-10, atol=1e-12, max_step=5.0)
    except Exception:
        sol_coral = solve_ivp(coral_ode_func, t_span, y0_coral, t_eval=t_eval,
                              method="RK45", rtol=1e-10, atol=1e-12)


    # 3-step: X = 0.11*X_I + 0.22*X_II + 0.67*X_III
    # 2-step: X = 0.11*X_I + 0.89*X_II  (X_III=0, X_II is Fe3O4 to Fe direct)
    x_coral_ode = np.array([
        overall_conversion_from_X(sol_coral.y[0, k], sol_coral.y[1, k], sol_coral.y[2, k])
        if regime == "3step"
        else 0.11 * sol_coral.y[0, k] + 0.89 * sol_coral.y[1, k]
        for k in range(sol_coral.y.shape[1])
    ])


    # Curve 3: CORAL analytical 
    x_coral_analytical = X_analytical(t_eval, T, y_H2, y_H2O, rp_um, P_atm, active_fe)

    return t_eval, X_IDAES, x_coral_ode, x_coral_analytical



# Plots

def main():
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle(
        "Validation",
        fontsize=13, fontweight="bold",
    )

    # Plot 1: Temperature Effect
    ax = axes[0, 0]
    temps = [723, 773, 823, 873]
    colors = {723: "#1C6FAA", 773: "#ff7f0e", 823: "#2ca02c", 873: "#b31f1f"}
    for T in temps:
        # f_Fe isnt applied. Normalized (0-1) data
        t, X_id, X_co, X_ca = run_batch(T, 0.15, 0.0, 60, active_fe=1.0)
        lbl = f"{T} K"
        ax.plot(t, X_id, color=colors[T], linewidth=2.5, label=f"IDAES {lbl}")
        ax.plot(t, X_co, color=colors[T], linewidth=1.5, linestyle="-.", alpha=0.8, label=f"CORAL-ODE {lbl}")
        ax.plot(t, X_ca, color=colors[T], linewidth=1.0, linestyle="--", alpha=0.5, label=f"CORAL-Analy {lbl}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Conversion X (-)")
    ax.set_title("Temperature effect\n(15% H₂, 0% H₂O, rp=60 µm)")
    ax.legend(fontsize=5.5, ncol=3)
    ax.set_xlim(0, 1800); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

    # Plot 2: H2 Effect
    ax = axes[0, 1]
    H2_fracs = [0.05, 0.15, 0.30, 0.60]
    colors_H2 = {0.05: "#1C6FAA", 0.15: "#ff7f0e", 0.30: "#2ca02c", 0.60: "#b31f1f"}
    for yH2 in H2_fracs:
        t, X_id, X_co, X_ca= run_batch(823, yH2, 0.0, 60, active_fe=1.0)
        lbl = f"{int(yH2*100)}% H₂"
        ax.plot(t, X_id, color=colors_H2[yH2], linewidth=2.5, label=f"IDAES {lbl}")
        ax.plot(t, X_co, color=colors_H2[yH2], linewidth=1.5, linestyle="-.", alpha=0.8, label=f"CORAL-ODE {lbl}")
        ax.plot(t, X_ca, color=colors_H2[yH2], linewidth=1.0, linestyle="--", alpha=0.5, label=f"CORAL-Analy {lbl}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Conversion X (-)")
    ax.set_title("H2 Effect\n(T=823 K, 0% H2O, rp=60 um)")
    ax.legend(fontsize=5.5, ncol=3)
    ax.set_xlim(0, 1800); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

    # Plot 3: H2O Effect
    ax = axes[1, 0]
    H2O_fracs = [0.0, 0.05, 0.10, 0.15]
    colors_H2O = {0.0: "#1C6FAA", 0.05: "#ff7f0e", 0.10: "#2ca02c", 0.15: "#b31f1f"}
    for yH2O in H2O_fracs:
        t, X_id, X_co, X_ca = run_batch(873, 0.60, yH2O, 60, active_fe=1)
        lbl = f"{int(yH2O*100)}% H2O"
        ax.plot(t, X_id, color=colors_H2O[yH2O], linewidth=2.5, label=f"IDAES {lbl}")
        ax.plot(t, X_co, color=colors_H2O[yH2O], linewidth=1.5, linestyle="-.", alpha=0.8, label=f"CORAL-ODE {lbl}")
        ax.plot(t, X_ca, color=colors_H2O[yH2O], linewidth=1.0, linestyle="--", alpha=0.5, label=f"CORAL-Analy {lbl}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Conversion X (-)")
    ax.set_title("H2O Effect\n(T=873 K, 60% H2, rp=60 um)")
    ax.legend(fontsize=5.5, ncol=3)
    ax.set_xlim(0, 1800); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

    # Plot 4: Size Effect
    ax = axes[1, 1]
    dp_vals = [60, 115, 175, 225, 350]
    colors_dp = {60: "#1C6FAA", 115: "#ff7f0e", 175: "#2ca02c", 225: "#b31f1f", 350: "#9467bd"}
    for dp in dp_vals:
        t, X_id, X_co, X_ca= run_batch(773, 0.60, 0.0, dp, active_fe=1.0)
        lbl = f"dp={dp}um"
        ax.plot(t, X_id, color=colors_dp[dp], linewidth=2.5, label=f"IDAES {lbl}")
        ax.plot(t, X_co, color=colors_dp[dp], linewidth=1.5, linestyle="-.", alpha=0.8, label=f"CORAL-ODE {lbl}")
        ax.plot(t, X_ca, color=colors_dp[dp], linewidth=1.0, linestyle="--", alpha=0.5, label=f"CORAL-Anal {lbl}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Conversion X (-)")
    ax.set_title("Size Effect\n(T=773 K, 60% H2, 0% H2O)")
    ax.legend(fontsize=5.5, ncol=3)
    ax.set_xlim(0, 1800); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = r"c:\Users\habbo\OneDrive\Desktop\TESIS_CODE\redu_IDAES_vs_CORAL.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    

    # RMS Error Table
    
    print("Validation: RMS: IDAES | CORAL ODE | CORAL Analytical")
    
    conditions = [
        ("Fig 4.2", 723, 0.15, 0.0, 60, 1.0),
        ("Fig 4.2", 773, 0.15, 0.0, 60, 1.0),
        ("Fig 4.2", 823, 0.15, 0.0, 60, 1.0),
        ("Fig 4.2", 873, 0.15, 0.0, 60, 1.0),
        ("Fig 4.3", 823, 0.05, 0.0, 60, 1.0),
        ("Fig 4.3", 823, 0.60, 0.0, 60, 1.0),
        ("Fig 4.4", 873, 0.60, 0.0, 60, 1.0),
        ("Fig 4.4", 873, 0.60, 0.15, 60, 1.0),
        ("Fig 4.5", 773, 0.60, 0.0, 60, 1.0),
        ("Fig 4.5", 773, 0.60, 0.0, 350, 1.0),
    ]

    header = (f"{'Caso':>10} | {'T(K)':>6} | {'H2%':>5} | {'H2O%':>5} | {'dp':>5} | "
              f"{'X_IDAES':>10} | {'X_ODE':>10} | {'X_Anal':>10} | "
              f"{'RMS_ID-OD':>11} | {'RMS_ID-An':>11} | {'RMS_OD-An':>11}")
    print(header)
    print("-" * len(header))

    for label, T, yH2, yH2O, dp, af in conditions:
        t, X_id, X_co, X_ca= run_batch(T, yH2, yH2O, dp, active_fe=af)
        rms_id_od = np.sqrt(np.mean((X_id - X_co)**2))
        rms_id_an = np.sqrt(np.mean((X_id - X_ca)**2))
        rms_od_an = np.sqrt(np.mean((X_co - X_ca)**2))
        print(f"{label:>10} | {T:>6} | {yH2*100:>5.1f} | {yH2O*100:>5.1f} | {dp:>5} | "
              f"{X_id[-1]:>10.4f} | {X_co[-1]:>10.4f} | {X_ca[-1]:>10.4f} | "
              f"{rms_id_od:>11.6f} | {rms_id_an:>11.6f} | {rms_od_an:>11.6f}")


    
    # ReactionBlock vs numpy 
    # Compare k_rxn, keq_red, xi_red, kc_full, and reaction_rate for each variable

    print("Point to point validation: IDAES ReactionBlock (Pyomo) vs numpy")
   
    test_conditions = [
        ("T=723K, 15%H2", 723, 0.15, 0.0, 60, 1.0),
        ("T=823K, 15%H2", 823, 0.15, 0.0, 60, 1.0),
        ("T=873K, 15%H2", 873, 0.15, 0.0, 60, 1.0),
        ("T=873K, 60%H2", 873, 0.60, 0.0, 60, 1.0),
        ("T=873K, 15%H2O", 873, 0.60, 0.15, 60, 1.0),
        ("T=773K, dp=350", 773, 0.60, 0.0, 350, 1.0),
    ]

    all_errors = []  
    var_names = ["k_rxn", "kc_full", "reaction_rate"]

    for case_label, T_val, yH2, yH2O, dp, af in test_conditions:
        P_atm = 1.0
        C_total = P_atm * 101325 / (8.314 * T_val)
        C_gas = {
            "O2": 0.0, "N2": (1 - yH2 - yH2O) * C_total,
            "CO2": 0.0, "H2O": yH2O * C_total, "H2": yH2 * C_total,
        }
        X_step = {"I": 0.3, "II": 0.05, "III": 0.01}

        idaes_vals = eval_idaes_block(T_val, C_gas, dp, X_step, af)
        numpy_vals = numpy_reference(T_val, C_gas, dp, X_step, af)

        print(f"\n  Caso: {case_label}")
        print(f"  {'Variable':>16} | {'ID':>4} | {'IDAES (Pyomo)':>16} | {'numpy':>16} | {'Diff':>12} | {'Rel Err %':>10}")
        print(f"  {'-'*80}")

        for var in var_names:
            for key in idaes_vals[var]:
                v_id = idaes_vals[var][key]
                v_np = numpy_vals[var][key]
                diff = abs(v_id - v_np)
                rel = 100.0 * diff / max(abs(v_np), 1e-30)
                status = "good" if rel < 0.01 else ("warn" if rel < 1.0 else "fail")
                if rel > 0.001:
                    print(f"  {var+'['+str(key)+']':>16} | {key:>4} | {v_id:>16.8e} | {v_np:>16.8e} | {diff:>12.2e} | {rel:>9.4f}% {status}")
                all_errors.append((case_label, var, key, rel))

        # Check keq and xi 
        for key in idaes_vals["keq_red"]:
            v_id = idaes_vals["keq_red"][key]
            v_np = numpy_vals["keq_red"][key]
            rel = 100.0 * abs(v_id - v_np) / max(abs(v_np), 1e-30)
            if rel > 0.001:
                print(f"  {'keq_red['+str(key)+']':>16} | {key:>4} | {v_id:>16.8e} | {v_np:>16.8e} | {abs(v_id-v_np):>12.2e} | {rel:>9.4f}%")
            all_errors.append((case_label, "keq_red", key, rel))
        for key in idaes_vals["xi_red"]:
            v_id = idaes_vals["xi_red"][key]
            v_np = numpy_vals["xi_red"][key]
            rel = 100.0 * abs(v_id - v_np) / max(abs(v_np), 1e-30)
            if rel > 0.001:
                print(f"  {'xi_red['+str(key)+']':>16} | {key:>4} | {v_id:>16.8e} | {v_np:>16.8e} | {abs(v_id-v_np):>12.2e} | {rel:>9.4f}%")
            all_errors.append((case_label, "xi_red", key, rel))

    # Summary
    max_err = max(e[3] for e in all_errors) if all_errors else 0
    mean_err = np.mean([e[3] for e in all_errors]) if all_errors else 0
    print(f"\n  Summary: max relative error = {max_err:.6f}%, mean = {mean_err:.6f}%")
    if max_err < 0.01:
        print("Really good")
    elif max_err < 0.1:
        print("Good")
    else:
        print("No good")


if __name__ == "__main__":
    main()

"""
Validacion CORAL oxidation reaction package
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import exp as m_exp

from pyomo.environ import ConcreteModel, value
from custom_properties.oxi_dry_reactions import DryOxidationReactionParameterBlock
from custom_properties.gas_phase_thermo import CustomGasPhaseParameterBlock
from custom_properties.solid_phase_thermo import CustomSolidPhaseParameterBlock


# Package parameters


m_params = ConcreteModel()
m_params.gas_props = CustomGasPhaseParameterBlock()
m_params.sol_props = CustomSolidPhaseParameterBlock()
m_params.hetero_rxns = DryOxidationReactionParameterBlock(
    solid_property_package=m_params.sol_props,
    gas_property_package=m_params.gas_props,
)

rxns = m_params.hetero_rxns

# Parameters
PARAMS = {
    "k_chr_0": value(rxns.k_chr_0),
    "E_chr": value(rxns.E_chr),
    "n_chr": value(rxns.n_chr),
    "n_k_rp": value(rxns.n_k_rp),
    "D_g_0": value(rxns.D_g_0),
    "E_g": value(rxns.E_g),
    "D_s_0": value(rxns.D_s_0),
    "E_s": value(rxns.E_s),
    "n_dif": value(rxns.n_dif),
    "n_D_rp": value(rxns.n_D_rp),
    "Xchr_O2_a": value(rxns.Xchr_O2_a),
    "Xchr_O2_b": value(rxns.Xchr_O2_b),
    "a_O2_0": value(rxns.a_O2_0),
    "a_O2_a": value(rxns.a_O2_a),
    "a_O2_b": value(rxns.a_O2_b),
    "b_O2_a": value(rxns.b_O2_a),
    "b_O2_b": value(rxns.b_O2_b),
    "n_X_rp": value(rxns.n_X_rp),
    "rp_ref": value(rxns.rp_ref),
}


for k, v in PARAMS.items():
    print(f"    {k}: {v}")

# Solid properties
sol = m_params.sol_props
MW = {j: value(sol.mw_comp[j]) for j in sol.component_list}
RHO_SKEL = {j: value(sol.dens_mass_comp_skeletal[j]) for j in sol.component_list}
particle_porosity = 0.27



# Referency numpy functions

Rg = 8.314


def C_smooth(C):
    eps = 1e-8
    return np.sqrt(C**2 + eps**2)



""" 

k_chr = k_chr_0 * (rp_ref/rp)^n_k_rp * exp(-E_chr/(Rg*T))
rp = radius = rp_m/2

"""
def calc_k_chr(T, rp_m):
    rp = rp_m / 2.0
    return (PARAMS["k_chr_0"]
            * (PARAMS["rp_ref"] / rp) ** PARAMS["n_k_rp"]
            * m_exp(-PARAMS["E_chr"] / (Rg * T)))


"""

D_eff = (D_g + D_s) * (rp_ref/rp)^n_D_rp
rp = radius = rp_m/2

"""

def calc_D_eff(T, rp_m):
    rp = rp_m / 2.0
    D_g = PARAMS["D_g_0"] * m_exp(-PARAMS["E_g"] / (Rg * T))
    D_s = PARAMS["D_s_0"] * m_exp(-PARAMS["E_s"] / (Rg * T))
    return (D_g + D_s) * (PARAMS["rp_ref"] / rp) ** PARAMS["n_D_rp"]


"""

X_chr_O2 = Xchr_O2_a * exp(Xchr_O2_b / T)
a_O2 = a_O2_0 + a_O2_a * exp(a_O2_b / T)
b_O2 = b_O2_a * exp(b_O2_b / T)
X_chr = (Xchr_O2 + a_O2 * exp(b_O2 * C_O2)) * (rp_ref / rp_m) ** n_X_rp
rp = radius = rp_m/2
   
"""

def calc_X_chr(T, C_O2, rp_m, use_smooth=True):
    
    rp = rp_m / 2.0
    Xchr_O2 = PARAMS["Xchr_O2_a"] * m_exp(PARAMS["Xchr_O2_b"] / T)
    a_O2 = PARAMS["a_O2_0"] + PARAMS["a_O2_a"] * m_exp(PARAMS["a_O2_b"] / T)
    b_O2 = PARAMS["b_O2_a"] * m_exp(PARAMS["b_O2_b"] / T)
    C = C_smooth(C_O2) if use_smooth else C_O2
    return (Xchr_O2 + a_O2 * m_exp(b_O2 * C)) * (
        PARAMS["rp_ref"] / rp
    ) ** PARAMS["n_X_rp"]


"""

Two-step rate, SCM + ZLT 

"""

def calc_dXdt(X, T, C_O2, rp_m):
    
    eps_x = 1e-8
    delta = 0.02

    # Step I
    k_chr = calc_k_chr(T, rp_m)
    C_s = C_smooth(C_O2)
    tau_chr = 1.0 / (k_chr * C_s ** PARAMS["n_chr"])
    dXdt_I = (3.0 / tau_chr) * max(1 - X + eps_x, eps_x) ** (2.0 / 3.0)

    # Step II
    D_eff = calc_D_eff(T, rp_m)
    tau_dif = 1.0 / (D_eff * C_s ** PARAMS["n_dif"])
    X_chr = calc_X_chr(T, C_O2, rp_m)

    X_dif_raw = (X - X_chr) / (1 - X_chr + eps_x)
    X_dif = np.sqrt(X_dif_raw**2 + (eps_x * 0.1)**2)
    omXdif = 1 - X_dif + eps_x
    dXdif_dt = (
        (3.0 / (2.0 * tau_dif))
        * omXdif ** (5.0 / 3.0)
        / (1 - omXdif ** (1.0 / 3.0) + eps_x)
    )
    dXdt_II = (1 - X_chr) * dXdif_dt

    # Sigmoid blending
    w = 0.5 * (1 + np.tanh((X - X_chr) / delta))
    return (1 - w) * dXdt_I + w * dXdt_II



"""

Mixture skeletal density
1/rho_mix = sum(w_j/rho_j).


"""

def calc_dens_mass_skeletal(w_Fe):
    w_Fe2O3 = 1.0 - w_Fe
    return 1.0 / (w_Fe / RHO_SKEL["Fe"] + w_Fe2O3 / RHO_SKEL["Fe2O3"])



"""

reaction_rate = dX/dt * n_Fe_vol / |nu_Fe|

"""
def calc_reaction_rate(X, w_Fe, T, C_O2, rp_m):
    
    dXdt = calc_dXdt(X, T, C_O2, rp_m)
    rho_mix = calc_dens_mass_skeletal(w_Fe)
    n_Fe_vol = (1 - particle_porosity) * rho_mix * w_Fe / MW["Fe"]
    return dXdt * n_Fe_vol / 2.0



# ReactionBlock IDAES 

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


"""

State variables are fixed and evaluation of all derived properties

"""
def set_idaes_state(T_val, y_O2, rp_m, X_val=0.01, w_Fe=0.95):
    
    gs = b.gas_state_ref
    P = 101325.0
    C_total = P / (Rg * T_val)

    gs.temperature.fix(T_val)
    gs.pressure.fix(P)
    gs.flow_mol.fix(1.0)
    y_O2_val = max(y_O2 * C_total * Rg * T_val / P, 1e-20)
    for j in m_params.gas_props.component_list:
        y = y_O2_val if j == "O2" else max((1 - y_O2) * C_total * Rg * T_val / P, 1e-20)
        gs.mole_frac_comp[j].fix(y)

    b.solid_state_ref.temperature.fix(T_val)
    b.solid_state_ref.particle_porosity.fix(particle_porosity)
    b.solid_state_ref.flow_mass.fix(200.0)
    for j in m_params.sol_props.component_list:
        if j == "Fe":
            b.solid_state_ref.mass_frac_comp[j].fix(w_Fe)
        elif j == "Fe2O3":
            b.solid_state_ref.mass_frac_comp[j].fix(1.0 - w_Fe)
        else:
            b.solid_state_ref.mass_frac_comp[j].fix(0.0)

    b.solid_state_ref.params.particle_dia.set_value(rp_m)



    # Evaluate mixture skeletal density from constraint 
    if not hasattr(b.solid_state_ref, "density_skeletal_constraint"):
        try:
            _ = b.solid_state_ref.dens_mass_skeletal
        except Exception:
            pass
    if hasattr(b.solid_state_ref, "density_skeletal_constraint"):
        calculate_variable_from_constraint(
            b.solid_state_ref.dens_mass_skeletal,
            b.solid_state_ref.density_skeletal_constraint,
        )

    # Trigger gas property construction
    if not hasattr(gs, "ideal_gas"):
        try:
            _ = gs.dens_mol
        except Exception:
            pass
    if not hasattr(gs, "comp_conc_eqn"):
        try:
            _ = gs.dens_mol_comp
        except Exception:
            pass

    if hasattr(gs, "ideal_gas"):
        calculate_variable_from_constraint(gs.dens_mol, gs.ideal_gas)
    if hasattr(gs, "comp_conc_eqn"):
        for j in m_params.gas_props.component_list:
            calculate_variable_from_constraint(
                gs.dens_mol_comp[j], gs.comp_conc_eqn[j]
            )


    # Initialize intermediate vars. OC_conv first, then dependents

    calculate_variable_from_constraint(b.C_O2_smooth, b.C_O2_smooth_eqn)
    calculate_variable_from_constraint(b.k_chr, b.k_chr_eqn)
    calculate_variable_from_constraint(b.D_eff, b.D_eff_eqn)
    calculate_variable_from_constraint(b.X_chr, b.X_chr_eqn)


    # OC_conv from mass fractions

    if hasattr(b, "OC_conv_eqn"):
        calculate_variable_from_constraint(b.OC_conv, b.OC_conv_eqn)
    b.OC_conv.fix(X_val)


    # Dependents of OC_conv 

    calculate_variable_from_constraint(b.dXdt_I, b.dXdt_I_eqn)
    calculate_variable_from_constraint(b.dXdt_II, b.dXdt_II_eqn)
    calculate_variable_from_constraint(b.sigmoid_w, b.sigmoid_w_eqn)


    # Reaction_rate

    calculate_variable_from_constraint(
        b.reaction_rate["R1"], b.gen_rate_expression["R1"]
    )



"""

Evaluate all IDAES block variables

"""
def eval_idaes(T, y_O2, rp_m, X_val=0.01, w_Fe=0.95):
    
    set_idaes_state(T, y_O2, rp_m, X_val, w_Fe)
    return {
        "C_O2_smooth": value(b.C_O2_smooth),
        "k_chr": value(b.k_chr),
        "D_eff": value(b.D_eff),
        "X_chr": value(b.X_chr),
        "dXdt_I": value(b.dXdt_I),
        "dXdt_II": value(b.dXdt_II),
        "sigmoid_w": value(b.sigmoid_w),
        "OC_conv": value(b.OC_conv),
        "reaction_rate": value(b.reaction_rate["R1"]),
    }



"""

Evaluation with numpy functions

"""
def eval_numpy(T, y_O2, rp_m, X_val=0.01, w_Fe=0.95):
    
    eps_x = 1e-8
    C_total = 101325.0 / (Rg * T)
    C_O2 = y_O2 * C_total
    C_s = C_smooth(C_O2)

    k_chr = calc_k_chr(T, rp_m)
    D_eff = calc_D_eff(T, rp_m)
    X_chr = calc_X_chr(T, C_O2, rp_m, use_smooth=True)

    # Raw dXdtI 
    tau_chr = 1.0 / (k_chr * C_s ** PARAMS["n_chr"])
    dXdt_I = (3.0 / tau_chr) * max(1 - X_val + eps_x, eps_x) ** (2.0 / 3.0)

    # Raw dXdtII 
    tau_dif = 1.0 / (D_eff * C_s ** PARAMS["n_dif"])
    X_dif_raw = (X_val - X_chr) / (1 - X_chr + eps_x)
    X_dif = np.sqrt(X_dif_raw**2 + (eps_x * 0.1)**2)
    omXdif = 1 - X_dif + eps_x
    dXdif_dt = (
        (3.0 / (2.0 * tau_dif))
        * omXdif ** (5.0 / 3.0)
        / (1 - omXdif ** (1.0 / 3.0) + eps_x)
    )
    dXdt_II = (1 - X_chr) * dXdif_dt

    # Sigmoid 
    w = 0.5 * (1 + np.tanh((X_val - X_chr) / 0.02))

    # Reaction rate
    rho_mix = calc_dens_mass_skeletal(w_Fe)
    n_Fe_vol = (1 - particle_porosity) * rho_mix * w_Fe / MW["Fe"]
    blended_dXdt = (1 - w) * dXdt_I + w * dXdt_II

    return {
        "C_O2_smooth": C_s,
        "k_chr": k_chr,
        "D_eff": D_eff,
        "X_chr": X_chr,
        "dXdt_I": dXdt_I,
        "dXdt_II": dXdt_II,
        "sigmoid_w": w,
        "OC_conv": X_val,
        "reaction_rate": blended_dXdt * n_Fe_vol / 2.0,
    }



# Validation


print("Validation")


test_conditions = [
    ("T=923K, 5%O2", 923, 0.05, 60e-6, 0.01, 0.95),
    ("T=923K, 21%O2", 923, 0.21, 60e-6, 0.01, 0.95),
    ("T=1073K, 5%O2", 1073, 0.05, 60e-6, 0.01, 0.95),
    ("T=1073K, 21%O2", 1073, 0.21, 60e-6, 0.30, 0.65),
    ("T=973K, 10%O2", 973, 0.10, 60e-6, 0.10, 0.85),
    ("T=973K, 21%O2", 973, 0.21, 200e-6, 0.05, 0.90),
]

all_errors = []
for case_label, T, y_O2, rp_m, X_val, w_Fe in test_conditions:
    id = eval_idaes(T, y_O2, rp_m, X_val, w_Fe)
    np_val = eval_numpy(T, y_O2, rp_m, X_val, w_Fe)

    print(f"\n  Case: {case_label}")
    print(f"  {'Variable':>16} | {'IDAES (Pyomo)':>16} | {'numpy':>16} | {'Rel Err %':>10}")
    

    for var in id:
        v_id = id[var]
        v_np = np_val[var]
        diff = abs(v_id - v_np)
        rel = 100.0 * diff / max(abs(v_np), 1e-30)
        status = "ok" if rel < 0.01 else ("warn" if rel < 1.0 else "fail")
        if rel > 0.001:
            print(f"  {var:>16} | {v_id:>16.8e} | {v_np:>16.8e} | {rel:>9.4f}% {status}")
        all_errors.append((case_label, var, rel))

max_err = max(e[2] for e in all_errors) if all_errors else 0
mean_err = np.mean([e[2] for e in all_errors]) if all_errors else 0
print(f"\n  Summary: max error = {max_err:.6f}%, mean = {mean_err:.6f}%")
if max_err < 0.01:
    print("Really good")
elif max_err < 0.5:
    print("Good")
else:
    print("No good")




# 5. Verificacion de X_chr


print(" X_chr verification with CORAL condition references")
print("=" * 120)

print(f"\n  b_O2_a = {PARAMS['b_O2_a']} m3/mol")
print(f"  rp_ref = {PARAMS['rp_ref']*1e6:.0f} um")


for T in [923, 973, 1023, 1073]:
    for y_O2 in [0.05, 0.10, 0.21]:
        C_total = 101325.0 / (Rg * T)
        C_O2 = y_O2 * C_total
        rp_m = 60e-6
        X_chr = calc_X_chr(T, C_O2, rp_m)
        print(f"  T={T:>4}K, {y_O2*100:5.1f}% O2: C_O2={C_O2:.3f}, X_chr={X_chr:.4f}")



# Curves

print("Comparison of curves")



def batch_ode(t, y, T, y_O2, rp_m):
    X = max(y[0], 1e-15)
    return [calc_dXdt(X, T, y_O2 * 101325.0 / (Rg * T), rp_m)]



"""

Run batch integration 

"""

def run_batch(T, y_O2, rp_um, t_max=4000):
    rp_m = rp_um * 1e-6
    t_eval = np.linspace(0, t_max, 500)
    y0 = [1e-15]

    try:
        sol = solve_ivp(
            lambda t, y: batch_ode(t, y, T, y_O2, rp_m),
            (0, t_max), y0, t_eval=t_eval,
            method="BDF", rtol=1e-8, atol=1e-12, max_step=5.0,
        )
    except Exception:
        sol = solve_ivp(
            lambda t, y: batch_ode(t, y, T, y_O2, rp_m),
            (0, t_max), y0, t_eval=t_eval,
            method="RK45", rtol=1e-8, atol=1e-10,
        )

    return sol.t, sol.y[0]


# Temperature effect plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Oxidation Kinetics Validation Curves\n",
    fontsize=13, fontweight="bold",
)

# Temperature effect at 21% O2
ax = axes[0, 0]
colors_T = {923: "#1f77b4", 973: "#ff7f0e", 1023: "#2ca02c", 1073: "#d62728"}
for T in [923, 973, 1023, 1073]:
    t, X = run_batch(T, 0.21, 60, t_max=4000)
    ax.plot(t, X, color=colors_T[T], lw=2.5, label=f"{T} K")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Conversion X (-)")
ax.set_title("Temperature effect (21% O2, dp=60 um)")
ax.legend(fontsize=9)
ax.set_xlim(0, 4000)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# O2 concentration effect at 1073K
ax = axes[0, 1]
colors_O2 = {0.05: "#1f77b4", 0.10: "#ff7f0e", 0.15: "#2ca02c", 0.21: "#d62728"}
for yO2 in [0.05, 0.10, 0.15, 0.21]:
    t, X = run_batch(1073, yO2, 60, t_max=4000)
    ax.plot(t, X, color=colors_O2[yO2], lw=2.5, label=f"{int(yO2*100)}% O2")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Conversion X (-)")
ax.set_title("O2 concentration effect (1073 K, dp=60 um)")
ax.legend(fontsize=9)
ax.set_xlim(0, 4000)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# Particle size effect at 1073K, 21% O2
ax = axes[1, 0]
colors_dp = {60: "#1f77b4", 150: "#ff7f0e", 250: "#2ca02c", 350: "#d62728"}
for dp in [60, 150, 250, 350]:
    t, X = run_batch(1073, 0.21, dp, t_max=4000)
    ax.plot(t, X, color=colors_dp[dp], lw=2.5, label=f"dp={dp} um")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Conversion X (-)")
ax.set_title("Particle size effect (1073 K, 21% O2)")
ax.legend(fontsize=9)
ax.set_xlim(0, 4000)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

# Particle size effect at 973K, 10% O2
ax = axes[1, 1]
for dp in [60, 150, 250, 350]:
    t, X = run_batch(973, 0.10, dp, t_max=4000)
    ax.plot(t, X, color=colors_dp[dp], lw=2.5, label=f"dp={dp} um")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Conversion X (-)")
ax.set_title("Particle size effect (973 K, 10% O2)")
ax.legend(fontsize=9)
ax.set_xlim(0, 4000)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(
    os.path.dirname(__file__), "oxi_IDAES_vs_CORAL.png"
)
fig.savefig(output_path, dpi=150, bbox_inches="tight")



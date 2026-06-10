

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Constants

Rg = 8.314  # J/(mol*K)
P_ATM = 1.01325e5  # Pa (1 atm, TGA)



# Support functions

def C_gas(y_frac, T, P=P_ATM):
    """
    Molar concentration [mol/m³] for the molar fraction at T and P
    """
    return y_frac * P / (Rg * T)


def get_regime(T):

    return "high" if T > 860 else "low"



# Equilibrium


# Keq = exp(A - B/T - C/T^2)
Keq_params = {
    "R2": {"A": 6.6567, "B": 6476.4, "C": 181_141.0},   # Fe3O4/FeO (T>860K)
    "R3": {"A": 0.8513, "B": 1395.5, "C": 253_791.0},    # FeO/Fe (T>860K)
    "R4": {"A": 1.5131, "B": 1394.9, "C": 745_520.0},    # Fe3O4/Fe (T<860K)
}


def calc_Keq(rx, T):
    p = Keq_params[rx]
    return np.exp(p["A"] - p["B"] / T - p["C"] / T**2)


def calc_xi(C_H2, C_H2O, Keq_val):
    """xi = 1 - C_H2O/(C_H2 * Keq)"""
    if Keq_val is None or C_H2 < 1e-12:
        return 1.0
    ratio = C_H2O / (C_H2 * Keq_val)
    return max(1e-10, 1.0 - ratio)



# Kinetic parameters


# dX, the total sum is 1
dX_3step = {"I": 0.11, "II": 0.22, "III": 0.67}
dX_2step = {"I": 0.11, "II": 0.89, "III": 0.00}

# Sintering
f_Fe_map = {"low": 1.0, "high": 0.7}

# 2D nuclei
params_2D = {
    "k0":     {"I": 0.58,  "II": 1.35, "III": 1.35},
    "Ea":     {"I": 35_600, "II": 49_200, "III": 49_200},   # J/mol
    "n":      {"I": 1.1,   "II": 1.1,   "III": 1.1},
    "s":      {"I": {"low": 0.0, "high": 0.0}, 
               "II": {"low": 2.2, "high": 0.3},
               "III": {"low": None, "high": 2.4}}, # No s at T<860K
    "r":      {"I": 0.0,   "II": 0.23,  "III": 0.23},
}

# 3D diffusion
params_3D = {
    "k0":     {"I": 1.10,  "II": 0.58,  "III": 0.58},
    "Ea":     {"I": 42_100, "II": 47_500, "III": 47_500},   # J/mol
    "n":      {"I": 1.0,   "II": 1.0,   "III": 1.0},
    "s":      {"I": {"low": 0.0, "high": 0.0},
               "II": {"low": 2.2, "high": 0.3},
               "III": {"low": None, "high": 2.3}}, # No s at T<860K
    "r":      {"I": 0.0,   "II": 0.25,  "III": 0.25},
}

RP_REF = 30.0  # um



# Calculation of kc


def calc_kc(params, step, T, C_H2, C_H2O, rp_um=60.0):

    """
    kc_i = k0 * exp(-Ea/Rg/T) * C_H2^n * xi^s * (30/rp)^r

    """
    regime = get_regime(T)

    # Arrhenius 
    k_arr = params["k0"][step] * np.exp(-params["Ea"][step] / (Rg * T))

    # H2
    h2_term = C_H2 ** params["n"][step]

    # Equilibrium
    s_val = params["s"][step][regime]
    if s_val is None:
        return 0.0
    if step == "I":
        xi_term = 1.0   # Irreversible
    elif regime == "high":
        if step == "II":
            xi_val = calc_xi(C_H2, C_H2O, calc_Keq("R2", T))
        else:
            xi_val = calc_xi(C_H2, C_H2O, calc_Keq("R3", T))
        xi_term = xi_val ** s_val
    else:  # low
        if step == "II":
            xi_val = calc_xi(C_H2, C_H2O, calc_Keq("R4", T))
            xi_term = xi_val ** s_val
        else:
            xi_term = 1.0

    # Size
    size_term = (RP_REF / rp_um) ** params["r"][step]

    return k_arr * h2_term * xi_term * size_term


def calc_kc_all(params, T, C_H2, C_H2O, rp_um=60.0):
    """dict {I: kc_I, II: kc_II, III: kc_III}."""
    regime = get_regime(T)
    steps = ["I", "II"] if regime == "low" else ["I", "II", "III"]
    return {s: calc_kc(params, s, T, C_H2, C_H2O, rp_um) for s in steps}



# Models


def X_2D_nuclei(kc_val, t):

    """2D nuclei Avrami"""

    return 1.0 - np.exp(-(kc_val * t) ** 2)


def X_3D_diffusion(kc_val, t):

    """3D diffusion Jander"""

    u2 = (kc_val * t) ** 2
    return 1.0 - (1.0 - np.minimum(u2, 1.0 - 1e-12)) ** 3 # Range between 0 and 1


def X_overall(kc_dict, t, model="2D"):

    """
    Global conversion

    """
    regime = "high" if "III" in kc_dict else "low"
    weights = dX_3step if regime == "high" else dX_2step
    func = X_2D_nuclei if model == "2D" else X_3D_diffusion
    return sum(weights[s] * func(kc_dict[s], t) for s in kc_dict)



# Plots


def plot_fig4_10():
    
    t = np.linspace(0, 1800, 500)
    rp = 60.0  # um 

    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    fig.suptitle(
        
        "Conversion vs Time (TGA-like)",
        fontsize=12, fontweight="bold", y=0.98,
    )
    axes = axes.flatten()

    # Colors
    colors_4 = ["#1675E2", "#E78A58", "#40C947", "#A332E9"]  
    colors_5 = ["#1675E2", "#E78A58", "#40C947", "#A332E9", "#FF0303"]  

    # Tag
    bbox_kw = dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.5)

    def label_at_X(ax, t_arr, X_curve, x_target, label, color, side="right"):
        """Tag placer"""
        idx = np.argmin(np.abs(X_curve - x_target))
        t_pos = t_arr[idx]
        x_pos = X_curve[idx]
        if side == "right":
            ax.annotate(label, xy=(t_pos, x_pos),
                        xytext=(8, 0), textcoords="offset points",
                        fontsize=9, color=color, fontweight="bold",
                        va="center", ha="left", bbox=bbox_kw)
        else:
            ax.annotate(label, xy=(t_pos, x_pos),
                        xytext=(-8, 0), textcoords="offset points",
                        fontsize=9, color=color, fontweight="bold",
                        va="center", ha="right", bbox=bbox_kw)

    # Temperature effect
    ax = axes[0]
    ax.set_title("(a) Temperature effect")
    x_targets_a = [0.90, 0.70, 0.50, 0.30]  
    for i, T_val in enumerate([723, 773, 823, 873]):
        C_H2 = C_gas(0.15, T_val)
        kc = calc_kc_all(params_2D, T_val, C_H2, 0.0, rp)
        X_curve = np.array([X_overall(kc, ti, "2D") for ti in t])
        ax.plot(t, X_curve, color=colors_4[i], lw=2)
        label_at_X(ax, t, X_curve, x_targets_a[i], f"{T_val} K", colors_4[i])

    ax.set_ylabel("Conversion (-)")
    ax.set_xlim(0, 2100)
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.04, "15% H2, dp=60 $\\mu$m",
            transform=ax.transAxes, fontsize=7, color="gray", ha="right")

    # Effect of H2 at 773 K
    ax = axes[1]
    ax.set_title("(b) Effect of H2 at 773 K")
    T_val = 773
    x_targets_b = [0.30, 0.50, 0.70, 0.90]
    for i, yh2 in enumerate([5, 15, 30, 60]):
        C_H2 = C_gas(yh2 / 100.0, T_val)
        kc = calc_kc_all(params_2D, T_val, C_H2, 0.0, rp)
        X_curve = np.array([X_overall(kc, ti, "2D") for ti in t])
        ax.plot(t, X_curve, color=colors_4[i], lw=2)
        label_at_X(ax, t, X_curve, x_targets_b[i], f"{yh2}% H$_2$", colors_4[i])

    ax.set_ylabel("Conversion (-)")
    ax.set_xlim(0, 2100)
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.04, "d$_p$=60 $\\mu$m",
            transform=ax.transAxes, fontsize=7, color="gray", ha="right")

    # Effect of H2 at 823 K
    ax = axes[2]
    ax.set_title("(c) Effect of H2 at 823 K ")
    T_val = 823
    x_targets_c = [0.30, 0.50, 0.70, 0.90]
    for i, yh2 in enumerate([5, 15, 30, 60]):
        C_H2 = C_gas(yh2 / 100.0, T_val)
        kc = calc_kc_all(params_2D, T_val, C_H2, 0.0, rp)
        X_curve = np.array([X_overall(kc, ti, "2D") for ti in t])
        ax.plot(t, X_curve, color=colors_4[i], lw=2)
        label_at_X(ax, t, X_curve, x_targets_c[i], f"{yh2}% H2", colors_4[i])

    ax.set_ylabel("Conversion (-)")
    ax.set_xlim(0, 2100)
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.04, "d$_p$=60 $\\mu$m",
            transform=ax.transAxes, fontsize=7, color="gray", ha="right")

    # Effect of H2O at 873 K
    ax = axes[3]
    ax.set_title("(d) Effect of H2O at 873 K")
    T_val = 873
    C_H2_base = C_gas(0.60, T_val)
    x_targets_d = [0.85, 0.65, 0.45, 0.25]
    for i, yh2o in enumerate([0, 5, 10, 15]):
        C_H2O = C_gas(yh2o / 100.0, T_val)
        kc = calc_kc_all(params_2D, T_val, C_H2_base, C_H2O, rp)
        X_curve = np.array([X_overall(kc, ti, "2D") for ti in t])
        ax.plot(t, X_curve, color=colors_4[i], lw=2)
        label_at_X(ax, t, X_curve, x_targets_d[i], f"{yh2o}% H2O", colors_4[i])

    ax.set_ylabel("Conversion (-)")
    ax.set_xlim(0, 2100)
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.04, "60% H2, dp=60 $\\mu$m",
            transform=ax.transAxes, fontsize=7, color="gray", ha="right")

    # Particle size effect at 773 K
    ax = axes[4]
    ax.set_title("(e) Particle size effect at 773 K")
    T_val = 773
    C_H2 = C_gas(0.15, T_val)
    x_targets_e = [0.90, 0.70, 0.50, 0.30, 0.15]
    for i, dp_val in enumerate([60, 115, 175, 225, 350]):
        kc = calc_kc_all(params_2D, T_val, C_H2, 0.0, dp_val)
        X_curve = np.array([X_overall(kc, ti, "2D") for ti in t])
        ax.plot(t, X_curve, color=colors_5[i], lw=2)
        label_at_X(ax, t, X_curve, x_targets_e[i], f"{dp_val} $\\mu$m", colors_5[i])

    ax.set_ylabel("Conversion (-)")
    ax.set_xlim(0, 2100)
    ax.set_ylim(0, 1.08)
    ax.grid(True, alpha=0.3)
    ax.annotate("15% H2", xy=(0.98, 0.04),
                xycoords="axes fraction", fontsize=7, color="gray", ha="right")

    # Empty

    ax = axes[5]
    ax.axis("off")

    for ax_i in axes[:5]:
        ax_i.set_xlabel("time (s)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig



# Keq and equilibrium


def plot_equilibrium():
    """Baur-Glaessner and Keq vs T"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Thermodynamic equilibrium", fontsize=13, fontweight="bold")

    # Baur-Glaessner H2O/(H2+H2O) vs T
    ax = ax1
    T_range = np.linspace(600, 1200, 300)
    for rx, label, color, ls in [
        ("R2", "Fe$_3$O$_4$/FeO", "r", "-"),
        ("R3", "FeO/Fe", "g", "-"),
        ("R4", "Fe$_3$O$_4$/Fe", "m", "--"),
    ]:
        keq_vals = [calc_Keq(rx, T_) for T_ in T_range]
        # Keq = P_H2O/P_H2 = y_H2O/y_H2, so y_H2O/(y_H2+y_H2O) = Keq/(1+Keq)
        y_ratio = [kv / (1.0 + kv) for kv in keq_vals]
        ax.plot(T_range, y_ratio, color=color, linestyle=ls, lw=2, label=label)

    ax.axvline(x=860, color="gray", linestyle="--", alpha=0.5, label="T=860 K")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("H$_2$O / (H$_2$ + H$_2$O)")
    ax.set_title("(a) Baur-Glaessner diagram")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    # Keq vs T (log scale)

    ax = ax2
    for rx, label, color, ls in [
        ("R2", "K$_{eq,R2}$ (Fe$_3$O$_4$/FeO)", "r", "-"),
        ("R3", "K$_{eq,R3}$ (FeO/Fe)", "g", "-"),
        ("R4", "K$_{eq,R4}$ (Fe$_3$O$_4$/Fe)", "m", "--"),
    ]:
        keq_vals = [calc_Keq(rx, T_) for T_ in T_range]
        ax.plot(T_range, keq_vals, color=color, linestyle=ls, lw=2, label=label)

    ax.axhline(y=1.0, color="k", linestyle=":", alpha=0.5, label="Keq = 1")
    ax.axvline(x=860, color="gray", linestyle="--", alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Keq (-)")
    ax.set_title("Equilibrium constants")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig



def main():
    print("=" * 65)
    print("CORAL Kinetic model")
    print("=" * 65)

    # Parameters
    print("\n Kinetic Parametes")
    print(f"\n  2D Nuclei Model ")
    for s in ["I", "II", "III"]:
        print(f"    Step {s}: k0={params_2D['k0'][s]}, "
              f"Ea={params_2D['Ea'][s]/1000:.1f} kJ/mol, "
              f"n={params_2D['n'][s]}, "
              f"r={params_2D['r'][s]}")
    print(f"\n  3D Diffusion Model ")
    for s in ["I", "II", "III"]:
        print(f"    Step {s}: k0={params_3D['k0'][s]}, "
              f"Ea={params_3D['Ea'][s]/1000:.1f} kJ/mol, "
              f"n={params_3D['n'][s]}, "
              f"r={params_3D['r'][s]}")

    # kc under reference conditions (15% H2, 773 K, dp=60um)

    print(f"\n--- kc under TGA conditions ---")
    T_ref = 773.0
    C_H2_ref = C_gas(0.15, T_ref)
    rp_ref = 60.0
    print(f"  T={T_ref} K, 15% H2 (C_H2={C_H2_ref:.2f} mol/m3), dp={rp_ref} um")
    for model_name, params in [("2D", params_2D), ("3D", params_3D)]:
        kc = calc_kc_all(params, T_ref, C_H2_ref, 0.0, rp_ref)
        print(f"  {model_name}: " + ", ".join(f"kc_{s}={kc[s]:.4e}" for s in kc))

    # Equilibrium
    print(f"\n Equilibrium at different temperatures ")
    for T_val in [723, 773, 823, 873, 973, 1073]:
        regime = "3-step" if T_val > 860 else "2-step"
        keq_r2 = calc_Keq("R2", T_val)
        keq_r3 = calc_Keq("R3", T_val)
        keq_r4 = calc_Keq("R4", T_val)
        print(f"  T={T_val} K ({regime}): "
              f"Keq_R2={keq_r2:.4f}, Keq_R3={keq_r3:.4f}, Keq_R4={keq_r4:.4f}")

    # Plots maker
    print(f"\n--- Generando graficos ---")

    fig1 = plot_fig4_10()
    fig1.savefig("coral_fig4_10.png", dpi=150, bbox_inches="tight")
    print(" Figure 1 ")

    fig2 = plot_equilibrium()
    fig2.savefig("coral_equilibrium.png", dpi=150, bbox_inches="tight")
    print(" Figure 2 ")

    plt.show()
    


if __name__ == "__main__":
    main()

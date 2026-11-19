"""
CORAL kinetics for the dry bed: rates and regime maps 

"""
import csv
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")


# Same numbers as oxi_dry_reactions.py
Rg = 8.314462618       # Gas constant J/(mol K)
rp = 50e-6             # Particle radius m
rp_ref = 30e-6         # Reference particle radius of the CORAL fit m
k_chr_0 = 0.561        # Regime I rate constant, pre exponential factor
E_chr = 45500.0        # Regime I activation energy J/mol 
n_k_rp = 0.63          # Particle size exponent of the rate constant
n_chr = 1.0            # Reaction order in O2, regime I
D_g_0 = 2.0e-6         # Gas diffusivity, pre exponential factor
E_g = 10000.0          # Gas diffusivity activation energy J/mol
D_s_0 = 7.28e13        # Solid diffusivity, pre exponential factor
E_s = 367300.0         # Solid diffusivity activation energy  J/mol 
n_dif = 1.0            # Reaction order in O2, regime II
n_D_rp = 2.0           # Particle size exponent of the diffusivity
Xchr_O2_a = 0.8351     # X_chr correlation, main factor
Xchr_O2_b = -2073.0    # X_chr correlation, T term
a_O2_0 = 2.127e-4      # X_chr O2 activity, base term
a_O2_a = 4.28e-9       # X_chr O2 activity, first T term
a_O2_b = 11560.0       # X_chr O2 activity, first T coefficient
b_O2_a = 40.0          # X_chr O2 activity, second T term
b_O2_b = -3060.0       # X_chr O2 activity, second T coefficient
n_X_rp = 0.6           # Particle size exponent of X_chr
eps = 1e-8             # Smoothing floor of the O2 term
eps_x = 1e-8           # Smoothing floor of the conversion terms
delta = 2e-3           # Smoothing width of the regime blend
delta_smooth = 0.02    # Smoothing width of the regime weight w


def k_chr_of(T):
    """
    CORAL intrinsic rate constant at T
    """
    return k_chr_0 * (rp_ref / rp) ** n_k_rp * np.exp(-E_chr / (Rg * T))


def D_eff_of(T):
    """
    Effective diffusivity in the grain at T
    """
    return (D_g_0 * np.exp(-E_g / (Rg * T))
            + D_s_0 * np.exp(-E_s / (Rg * T))) * (rp_ref / rp) ** n_D_rp


def X_chr_of(T, C):
    """
    Grain scale conversion reached in the residence time
    """
    return (Xchr_O2_a * np.exp(Xchr_O2_b / T)
            + (a_O2_0 + a_O2_a * np.exp(a_O2_b / T))
            * np.exp(b_O2_a * np.exp(b_O2_b / T) * C)) * (rp_ref / rp) ** n_X_rp


def C_smooth(C):
    """
    The smoothed O2 term of the rate
    """
    return 0.5 * (C + np.sqrt(C * C + eps * eps))


def dXdt_I_of(T, C, X):
    """
    Rate of regime I 
    """
    tau = 1.0 / (k_chr_of(T) * C_smooth(C) ** n_chr)
    return (3.0 / tau) * (1.0 - X + eps_x) ** (2.0 / 3.0)


def dXdt_II_of(T, C, X, Xc=None):
    """
    Rate of regime II 
    """
    tau = 1.0 / (D_eff_of(T) * C_smooth(C) ** n_dif)
    if Xc is None:
        Xc = X_chr_of(T, C)
    X_dr = (X - Xc) / (1.0 - Xc + eps_x)
    X_dif = np.sqrt(X_dr ** 2 + (eps_x * 0.1) ** 2)
    om = 1.0 - X_dif + eps_x
    om = (om + np.sqrt(om * om + delta * delta)) / 2.0
    dXdif = (3.0 / (2.0 * tau)) * om ** (5.0 / 3.0) \
        / (1.0 - om ** (1.0 / 3.0) + eps_x)
    return (1.0 - Xc) * dXdif


def w_of(X, Xc):
    """
    How much of each regime is active at X
    """
    return 0.5 + np.arctan((X - Xc) / delta_smooth * math.pi / 2.0) / math.pi


def dXdt_of(T, C, X, Xc=None, w=None):
    """
    Combined rate dX/dt at (T, X)
    """
    if Xc is None:
        Xc = X_chr_of(T, C)
    if w is None:
        w = w_of(X, Xc)
    return (1.0 - w) * dXdt_I_of(T, C, X) + w * dXdt_II_of(T, C, X, Xc)


def self_test():
    """
    Checks the recipe at T=1073 K and C=4.0664 mol/m3
    against fixed reference values
    """
    assert abs(k_chr_of(1073.0) - 2.4789e-3) <= 5e-4 * 2.4789e-3
    assert abs(D_eff_of(1073.0) - 3.4771e-05) <= 5e-4 * 3.4771e-05
    assert abs(X_chr_of(1073.0, 4.0664) - 3.7687) <= 5e-4 * 3.7687
    assert abs(dXdt_I_of(1073.0, 4.0664, 0.0) - 3.024066e-2) <= 5e-4 * 3.024066e-2


# Bed helpers, the exact IDAES bubbling bed closures

def D_ab_fuller(T, P, y=0.21):
    """O2 in N2 binary diffusivity by Fuller, then the Stefan inversion the property package applies"""
    V_O2, V_N2 = 16.6, 17.9    # Diffusion volumes of O2 and N2, Fuller correlation
    M_O2, M_N2 = 0.032, 0.028  # Molar masses of O2 and N2, kg/mol
    d_bin = 1.43e-7 * T ** 1.75 * math.sqrt((M_O2 + M_N2)
                                            / (2.0 * M_O2 * M_N2)) \
        / (P * (V_O2 ** (1.0 / 3.0) + V_N2 ** (1.0 / 3.0)) ** 2)
    # diffus_comp["O2"]*(y_O2/D + y_N2/D) = 1 - y_O2, same D for the pair
    return (1.0 - y) / ((y + (1.0 - y)) / d_bin)


def Kbe_of(db, u_mf, D_ab, g=9.80665):
    """
    IDAES bubble_cloud_mass_trans_coeff form
    Kbe = 1.62*u_mf/db + 5.85*sqrt(D_ab)*g**0.25/db**1.25
    """
    return 1.62 * u_mf / db + 5.85 * math.sqrt(D_ab) * g ** 0.25 / db ** 1.25


def load_axial(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(row, key, default=None):
    v = row.get(key)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def kin_curves():
    """
    Intrinsic kinetics response of the blended CORAL rate,
    local X fixed, over the bed temperatures sweep
    """
    print("Kinetics curves, X=0.115")
    X_fix = 0.115           # Baseline mid bed solid conversion
    Ts = np.array([836.0, 900.0, 973.0, 1023.0, 1083.2, 1112.0,
                   1200.0, 1333.0, 1467.1])
    P = 1.0e5   # Pressure Pa
    cases = {
        "C=4.21 (baseline inlet)": lambda T: 4.2107,
        "C=2.07 (hot path inlet)": lambda T: 2.0658,
        "p_O2=0.21 bar": lambda T: 0.21 * P / (Rg * T),
    }
    out = {}
    print("     T[K]" + "".join(f"{k:>28s}" for k in cases) + "  [1/s]")
    for T in Ts:
        line = f"  {T:7.1f}"
        for k, cf in cases.items():
            v = float(dXdt_of(T, cf(T), X_fix))
            line += f"  {v:26.6e}"
            out.setdefault(k, {})[str(T)] = v
        print(line)
    print("\n  intrinsic gain factors (blended rate):")
    gains = {}
    for k, cf in cases.items():
        hot = float(dXdt_of(1467.1, cf(1467.1), X_fix)
                    / dXdt_of(1083.2, cf(1083.2), X_fix))
        cold = float(dXdt_of(836.0, cf(836.0), X_fix)
                     / dXdt_of(1083.2, cf(1083.2), X_fix))
        stepI = float(dXdt_I_of(1467.1, cf(1467.1), X_fix)
                      / dXdt_I_of(1083.2, cf(1083.2), X_fix))
        gains[k] = {"hot_1083_to_1467": hot, "cold_1083_to_836": cold,
                    "stepI_only_hot": stepI}
        print(f"    {k:<26s} hot x{hot:8.3f}   cold x{cold:8.3f}"
              f"   (step I only, hot: x{stepI:6.3f})")

    print("\n  blended gain, hot path, C=4.21, vs X:")
    for X in (0.05, 0.115, 0.20):
        g = float(dXdt_of(1467.1, 4.2107, X) / dXdt_of(1083.2, 4.2107, X))
        print(f"    X={X:5.3f}: x{g:8.3f}")

    print("\n  X_chr(T, C) table [the distributor region is X_chr>1]:")
    print("      T[K]    C=0.5    C=1.0    C=2.0    C=4.21")
    xchr = {}
    for T in (836.0, 973.0, 1083.2, 1200.0, 1333.0, 1467.1):
        row = [float(X_chr_of(T, C)) for C in (0.5, 1.0, 2.0, 4.2107)]
        xchr[str(T)] = row
        print(f"      {T:7.1f} " + "".join(f"{v:8.3f}" for v in row))

    with open(os.path.join(DATA, "op_dry_kinetics.json"), "w") as f:
        json.dump({"gains": gains, "X_chr_table": xchr}, f, indent=1)
    return gains


def regime_map(axial_name, tag):
    """
    Per element regime decomposition of an axial profile
    """
    rows = load_axial(os.path.join(DATA, axial_name))
    print(f"Regime map on {axial_name}")
    out = []
    for r in rows:
        T = fnum(r, "T_sol")
        C = fnum(r, "C_react")
        X = fnum(r, "X_solid") / 100.0
        Xc = X_chr_of(T, C)
        w = w_of(X, Xc)
        out.append({"z": fnum(r, "z"), "T_sol": T, "C_react": C,
                    "X_frac": X, "X_chr": float(Xc), "w": float(w),
                    "dXdt_I": float(dXdt_I_of(T, C, X)),
                    "dXdt_II": float(dXdt_II_of(T, C, X, Xc)),
                    "dXdt_blend": float(dXdt_of(T, C, X, Xc, w)),
                    "y_emul_O2": fnum(r, "y_emul_O2")})
    dz = out[1]["z"] - out[0]["z"]
    pos = [o for o in out if o["dXdt_blend"] > 0]
    tot = sum(o["dXdt_blend"] for o in pos) * dz
    chem = sum((1.0 - o["w"]) * o["dXdt_blend"] for o in pos) * dz
    chem_frac = chem / tot if tot else float("nan")
    w_mean = float(np.mean([o["w"] for o in out]))
    n_chem = sum(1 for o in out if o["w"] < 0.25)
    n_diff = sum(1 for o in out if o["w"] > 0.75)
    starve = next((o["z"] for o in out if o["y_emul_O2"] is not None
                   and o["y_emul_O2"] < 0.02), None)
    print(f"  mean w = {w_mean:.3f}; elements w<0.25: {n_chem}/{len(out)}, "
          f"w>0.75: {n_diff}/{len(out)}")
    print(f"  rate weighted chemical share of the dXdt integral: "
          f"{chem_frac*100:.1f}%")
    print(f"  emulsion O2 < 2% from z = {starve} m")
    print("      z     X%     X_chr      w    dXdt_I      dXdt_II     blend")
    for o in out[::2]:
        print(f"  {o['z']:6.3f} {o['X_frac']*100:6.2f} {o['X_chr']:8.3f}  "
              f"{o['w']:6.3f}  {o['dXdt_I']:10.3e} {o['dXdt_II']:10.3e} "
              f"{o['dXdt_blend']:10.3e}")
    csv_path = os.path.join(DATA, f"dryT_regime_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(out)
    return {"w_mean": w_mean, "chem_share": float(chem_frac),
            "z_starve": starve}


if __name__ == "__main__":
    self_test()
    gains = kin_curves()
    reg0 = regime_map("axial_dry_lab_r8.csv", "base")
    with open(os.path.join(DATA, "op_dry_regime.json"), "w") as f:
        json.dump({"regime_base": reg0}, f, indent=1)

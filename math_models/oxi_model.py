import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
Rg = 8.314          # J/(mol·K)
P = 101325          # Pa (1 atm)

# Parameters Phase I
n_chr = 1.0
n_k_rp = 0.63
k_chr_o = 5.61e-1   # m^3/(mol·s)
E_chr = 45500       # J/mol

# Parameters Phase II 
n_dif = 1.0
n_D_rp = 2.0
Dg_o = 2.0e-6       # m^3/(mol·s)
Eg = 10000          # J/mol
Ds_o = 7.28e13      # m^3/(mol·s)
Es = 367300         # J/mol

# Parameters X_chr
n_X_rp = 0.6
Xchr_O2_a = 8.351e-1
Xchr_O2_b = -2073.0
a_O2_0 = 2.127e-4
a_O2_a = 4.28e-9
a_O2_b = 11560.0
b_O2_a = 0.4           # m^3/mol
b_O2_b = -3060.0

def concentration_O2(y_O2, T):
    return (y_O2 * P) / (Rg * T)

def k_chr(T, rp_um):
    factor_size = (30.0 / rp_um) ** n_k_rp
    return k_chr_o * factor_size * np.exp(-E_chr / (Rg * T))

def D_eff(T, rp_um):
    Dg = Dg_o * np.exp(-Eg / (Rg * T))
    Ds = Ds_o * np.exp(-Es / (Rg * T))
    factor_size = (30.0 / rp_um) ** n_D_rp
    return (Dg + Ds) * factor_size

def X_chr(T, C_O2, rp_um):
    X_O2 = Xchr_O2_a * np.exp(Xchr_O2_b / T)
    a_O2 = a_O2_0 + a_O2_a * np.exp(a_O2_b / T)
    b_O2 = b_O2_a * np.exp(b_O2_b / T)
    factor_size = (30.0 / rp_um) ** n_X_rp
    return (X_O2 + a_O2 * np.exp(b_O2 * C_O2)) * factor_size

def oxidation_simulation(T, y_O2, rp_um, t_max=4000):
    C_O2 = concentration_O2(y_O2, T)
    Xc = X_chr(T, C_O2, rp_um)
   # print("C_O2 =", C_O2)
    #print("X_chr =", Xc)
   # print("k_chr =", k_chr(T,rp_um))
    #print("D_eff =", D_eff(T,rp_um))
    if Xc >= 1.0:
        print(f"Warning: X_chr = {Xc:.3f} >= 1. Only Phase I will be used.")
        Xc = 0.999

    # Phase I
    def dXdt_I(t, X):
        tau = 1.0 / (k_chr(T, rp_um) * (C_O2 ** n_chr))
        return (3.0 / tau) * (1.0 - X) ** (2.0/3.0)

    # Event: Stops when X is Xc
    event = lambda t, X: X[0] - Xc
    event.terminal = True
    event.direction = 1   

    t_eval_I = np.linspace(0, t_max, 500)
    sol_I = solve_ivp(dXdt_I, [0, t_max], [0.0], t_eval=t_eval_I,
                      events=event, method='RK45', rtol=1e-6, atol=1e-8)

    if sol_I.t_events[0].size == 0:
        # Xc was not reached within the given time
        print("X_chr was not reached within the maximum time. Phase I only.")
        return sol_I.t, sol_I.y[0]

    t_chr = sol_I.t_events[0][0]
    X_chr_val = sol_I.y_events[0][0][0]
    print("Transition time =", t_chr)
    print("X transition =", X_chr_val)

    # Retain the part of Stage I up to t_chr.
    mask = sol_I.t <= t_chr
    t_I = sol_I.t[mask]
    X_I = sol_I.y[0][mask]

    # Phase II
    # Add a small epsilon to avoid singularity at Xd=0
    epsilon = 1e-9
    X_start = X_chr_val + epsilon
    if X_start >= 1.0:
        print("X_chr very close to 1; phase II isnt performed.")
        return t_I, X_I

    # Transformation of variables
    def dXdt_II(t, X):
        # X: global conversion
        Xd = (X - X_chr_val) / (1.0 - X_chr_val)
        
        if Xd <= 0:
            Xd = epsilon
        tau = 1.0 / (D_eff(T, rp_um) * (C_O2 ** n_dif))
        # Denominator: (1-Xd)^{-1/3} - 1
        denom = (1.0 - Xd) ** (-1.0/3.0) - 1.0
        if denom < 1e-12:
            denom = 1e-12
        dX_dif_dt = (3.0 / (2.0 * tau)) * (1.0 - Xd) ** (4.0/3.0) / denom
        return (1.0 - X_chr_val) * dX_dif_dt

    t_span_II = (t_chr, t_max)
    t_eval_II = np.linspace(t_chr, t_max, 500)
    sol_II = solve_ivp(dXdt_II, t_span_II, [X_start], t_eval=t_eval_II,
                       method='RK45', rtol=1e-6, atol=1e-8)

    if sol_II.success and len(sol_II.t) > 0:
        # Combine results
        t_II = sol_II.t[1:] if len(sol_II.t) > 1 else []
        X_II = sol_II.y[0][1:] if len(sol_II.y[0]) > 1 else []
        t_total = np.concatenate([t_I, t_II])
        X_total = np.concatenate([X_I, X_II])
    else:
        print("Phase II failed")
        return t_I, X_I

    return t_total, X_total


t, X = oxidation_simulation(T=1073, y_O2=0.21, rp_um=30.0, t_max=4000)

plt.figure(figsize=(8,5))
plt.plot(t, X, label=f'T=1073 K, 21% O₂, dp=60 µm', linewidth=2)
plt.yticks(np.arange(0, 1.05, 0.1))
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Conversion X', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()


# Particle size

T = 1073
y_O2 = 0.21
particle_sizes = [50, 100, 270, 310]

plt.figure(figsize=(8,5))

for rp in particle_sizes:
    t, X = oxidation_simulation(T, y_O2, rp, t_max=4000)
    plt.plot(t, X, label=f'{rp} μm')

plt.xlabel('Time (s)')
plt.ylabel('Conversion X')
plt.title('Effect of particle size')
plt.grid(True)
plt.legend()
plt.show()


# Temperature

rp_um = 30
y_O2 = 0.21
temperatures = [923, 973, 1023, 1073]

plt.figure(figsize=(8,5))

for T in temperatures:
    t, X = oxidation_simulation(T, y_O2, rp_um, t_max=4000)
    plt.plot(t, X, label=f'{T} K')

plt.xlabel('Time (s)')
plt.ylabel('Conversion X')
plt.title('Effect of the temperature')
plt.grid(True)
plt.legend()
plt.show()


# Oxygen

T = 1073
rp_um = 30
oxygen_levels = [0.05, 0.10, 0.15, 0.21]

plt.figure(figsize=(8,5))

for y_O2 in oxygen_levels:
    t, X = oxidation_simulation(T, y_O2, rp_um, t_max=4000)
    plt.plot(t, X, label=f'{y_O2*100:.0f}% O₂')

plt.xlabel('Time (s)')
plt.ylabel('Conversion X')
plt.title('Effect of the oxygen')
plt.grid(True)
plt.legend()
plt.show()
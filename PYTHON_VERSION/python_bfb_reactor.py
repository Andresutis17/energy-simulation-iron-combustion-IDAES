"""
Python BFB

Implements the Kunii Levenspiel model. The gas splits into a bubble
phase and an emulsion phase, and the solid moves as plug flow. Everything
is integrated along the reactor with an ODE solver.

The hydrodynamics and structure are ported from the IDAES BubblingFluidizedBed so that this model
reproduces IDAES. The two calibration constants KBE_DIV (mass transfer) and AH_SE (heat transfer) are used.

solve(p, pkg) reads all the chemistry from the package object (reactive gas,
reactions, stoichiometry, dH, kinetics, cp) 

The four balances solved are:
1. Gas mass   (bubble + emulsion, bubble to emulsion transfer + reaction sink)
2. Solid mass (stoichiometric consumption and production of each solid)
3. Energy     (reaction heat vs gas heat uptake)
4. Momentum   (bed pressure drop, hydrostatic)
    
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.integrate import solve_ivp

R = 8.314; g = 9.81
AH_SE = 14.0      # Heat transfer parameter
KBE_DIV = 5.0     # Mass transfer parameter
UMF = 0.039624    # minimum fluidization velocity [m/s]
KD = 1.0          # bulk gas permeation coefficient [m/s], ported from IDAES
DG = {"O2": 2e-4, "N2": 2e-4, "CO2": 1.6e-5, "H2O": 2e-4, "H2": 5e-4}  # gas diffusivity [m2/s]
HAS_PRESSURE_CHANGE = True    # Pressure change option


def setup(p, pkg):
    """
    Builds the reactor from the inputs and the reaction package
    """
    D = p["bed_diameter"]; H = p["bed_height"]; A = np.pi*(D/2)**2
    rho_skel_in = pkg.rho_skeletal(pkg.w_in)
    rho_particle = rho_skel_in*(1.0 - getattr(pkg, "phi", 0.27))   # particle density 
    kbe_div = getattr(pkg, "KBE_DIV", KBE_DIV)   # per-package calibration 
    return dict(p=p, pkg=pkg, A=A, H=H, P=p["gas_P"], T_g=p["gas_T"], T_s=p["solid_T"],
                Ms_in=p["solid_flow_mass"], aorf=1.0/p["number_orifice"], emf=0.45,
                umf=UMF, n_gas=p["gas_flow_mol"],
                y_reactive=pkg.y_in[pkg.GAS_REACTIVE], rho_mol_in=p["gas_P"]/(R*p["gas_T"]),
                rho_particle=rho_particle, deltaP_orifice=0.0, kbe_div=kbe_div,
                has_pressure_change=HAS_PRESSURE_CHANGE)


def _hydro(z, Tb, P, rc):

    """
    IDAES hydrodynamics at z given local bubble gas T and pressure P
    """
    p = rc["p"]; A = rc["A"]; D = p["bed_diameter"]; umf = rc["umf"]
    rho_mol_b = P/(R*Tb)                                  # bubble gas molar density (ideal gas)
    ug = rc["n_gas"]/(rho_mol_b*A)                        # superficial gas velocity [m/s]
    ub_excess = ug - umf                                  # gas flowing as bubbles (excess over umf)
    # Mori-Wen bubble diameter [m], ported from IDAES
    dbmax = (2.59**5/g*(ub_excess*A)**2)**0.2
    db0 = (1.38**5/g*(ub_excess*rc["aorf"])**2)**0.2      # db0 depends on area per orifice
    db = dbmax - (dbmax-db0)*np.exp(-0.3*z/D)             # Mori-Wen growth of db along the bed
    vbr = 0.711*np.sqrt(g*db)                             # Davidson single bubble rise velocity
    v_bubble = ub_excess + vbr                            # absolute bubble phase velocity
    delta = ub_excess/v_bubble                            # volume fraction of the bed in bubbles

    # Kbe is the bubble to emulsion mass transfer coefficient from the classic Kunii-Levenspiel
    # correlation (Fluidization Engineering), scaled by kbe_div to match IDAES's form
    Kbe = (3.0*umf/db + 5.85*(2e-4**0.5*g**0.25)/db**1.25)/rc["kbe_div"]
    return dict(db=db, ug=ug, delta=delta, v_bubble=v_bubble, rho_mol_b=rho_mol_b,
                Kbe=Kbe, bubble_area=A*delta, P=P)


def solve(p, pkg, ah_se=AH_SE):
    """
    Solves the BFB for operating inputs and reaction package
    """
    rc = setup(p, pkg)
    P = rc["P"]; A = rc["A"]; H = rc["H"]; T_g = rc["T_g"]; T_s = rc["T_s"]
    umf = rc["umf"]; emf = rc["emf"]; n_gas = rc["n_gas"]
    y_reactive = rc["y_reactive"]; Ms_in = rc["Ms_in"]
    reactive_solids = list(pkg.SOLID_REACTIVE); inert_solids = list(pkg.SOLID_INERTS)
    n_react = len(reactive_solids)
    MW = pkg.MW_SOLID; STOICH = pkg.STOICH; DH = pkg.DH
    RXNS = pkg.REACTIONS; GAS_REACTIVE = pkg.GAS_REACTIVE
    n_rxn = len(RXNS)
    has_gas_product = len(pkg.GAS_PRODUCTS) > 0
    Fm_react_in = np.array([Ms_in*pkg.w_in.get(s, 0.0) for s in reactive_solids])
    Fm_inert = np.array([Ms_in*pkg.w_in.get(s, 0.0) for s in inert_solids])

    def ode(z, y):
        """
        BFB ODE system. Returns dy/dz at axial position z
        for the state vector y (quantities vector)
        """
        # Unpack the state vector y = [Nb, Ne, Fm..., Tb, Te, Ts, P] 
        Nb, Ne = y[0], y[1]; Fm = y[2:2+n_react]; Tb, Te, Ts, P = y[-4], y[-3], y[-2], y[-1]
        # Total solid mass flow and mass fractions w 
        Ms = Fm.sum() + Fm_inert.sum()
        w = {reactive_solids[i]: Fm[i]/Ms for i in range(n_react)}
        w.update({inert_solids[i]: Fm_inert[i]/Ms for i in range(len(inert_solids))})

        # local hydrodynamics at height z, ported from IDAES
        h = _hydro(z, Tb, P, rc)
        delta = h["delta"]; Kbe = h["Kbe"]; barea = h["bubble_area"]; ug = h["ug"]
        dens_mol_e = h["P"]/(R*Te) # emulsion gas molar density (ideal gas)
        # bubble / emulsion gas volumetric flows [m3/s]. The bubble gets the excess gas
        # (ug-umf), the emulsion flows at minimum fluidization, Kunii-Levenspiel.
        Fb = (ug-umf)*A*h["rho_mol_b"]; Fe = umf*A*dens_mol_e
        # Reactive gas concentration in each phase [mol/m3] = molar flow / volumetric flow
        Cb = Nb/((ug-umf)*A); Ce = Ne/(umf*A)

        # Reaction rate from the reaction package 
        # Evaluated at the EMULSION concentration Ce and SOLID temperature Ts
        r = pkg.rates(Ce, Ts, w)
        cons = r*(1-delta)*(1-emf)*A         # reaction rate per reactor length [mol_rxn/s/m]
                                             # (1-delta)*(1-emf)*A = solids volume per bed length

        # Gas mass balance
        Ttr = barea*Kbe*(Cb-Ce)    # bubble to emulsion mass transfer [mol/s/m]
        cons_reactive = sum(cons[j]*abs(STOICH[RXNS[j]][GAS_REACTIVE]) for j in range(n_rxn))
        dNb = -Ttr            # bubble: only loses gas to the emulsion
        dNe = Ttr - cons_reactive   # emulsion: gains by transfer, loses by reaction

        # Solid mass balance, stoichiometry
        # dFm_i/dz = sum_reactions (rate * stoich * MW). Each solid changes by what the
        # reactions produce or consume of it.
        dFm = np.zeros(n_react)
        for i, s in enumerate(reactive_solids):
            for j in range(n_rxn):
                dFm[i] += cons[j]*STOICH[RXNS[j]].get(s, 0.0)*MW[s]

        # Energy balance
        cp_g_b = pkg.cp_gas_mol(Tb, pkg.y_in); cp_g_e = pkg.cp_gas_mol(Te, pkg.y_in)
        cp_s = pkg.cp_solid_mass(w, Ts)
        # Bubble and emulsion heat by contact with the next phase. AH_SE calibrated
        dTb = ah_se*(Te-Tb); dTe = ah_se*(Ts-Te)
        Q_rxn = sum(cons[j]*(-DH[RXNS[j]]) for j in range(n_rxn))   # reaction heat source [W/m]
        Q_gas = Fb*cp_g_b*dTb + Fe*cp_g_e*dTe            # heat carried away by the gas [W/m]
        dTs = (Q_rxn - Q_gas)/(Ms*cp_s)                   # solid: reaction heat not taken by the gas

        # Momentum balance, Pressure
        # IDAES dP/dz = -g*(1-voidage_avg)*dens_mass_particle; voidage_avg includes bubbles.
        voidage_avg = delta + (1-delta)*emf
        dP = -g*(1-voidage_avg)*rc["rho_particle"] if rc.get("has_pressure_change") else 0.0

        return [dNb, dNe] + list(dFm) + [dTb, dTe, dTs, dP]

    # Initial conditions at z = 0 
    P0 = rc["P"] - rc["deltaP_orifice"]
    h0 = _hydro(0.0, T_g, P0, rc)
    Nb0 = (h0["ug"]-umf)*A*y_reactive*h0["rho_mol_b"]     # reactive gas entering as bubbles
    Ne0 = umf*A*(P0/(R*T_g))*y_reactive                   # reactive gas entering the emulsion
    y0 = [Nb0, Ne0] + list(Fm_react_in) + [T_g, T_g, T_s, P0]   # both gas phases start at T_g
    # Integrate the four balances along the reactor (BDF, tight tolerances, small max step)
    # integrate the stiff Backward Differentiation Formula system along the reactor, stable for fast kinetics.
    sol = solve_ivp(ode, [0, H], y0, method="BDF", dense_output=True, rtol=1e-7, atol=1e-9, max_step=0.05)

    # Read the outlet
    NbH, NeH = sol.y[0, -1], sol.y[1, -1]
    FmH = sol.y[2:2+n_react, -1]
    TbH, TeH, TsH, P_out = sol.y[-4, -1], sol.y[-3, -1], sol.y[-2, -1], sol.y[-1, -1]
    MsH = FmH.sum() + Fm_inert.sum()
    reactive_in = n_gas*y_reactive
    cons_total = reactive_in - (NbH + NeH)     # gas consumed = in - (bubble+emulsion out)
    fmol_out = n_gas if has_gas_product else n_gas - cons_total   # 1:1 product keeps gas moles
    y_out = max(reactive_in - cons_total, 0.0)/max(fmol_out, 1e-9)
    w_out = {reactive_solids[i]: FmH[i]/MsH for i in range(n_react)}
    w_out.update({inert_solids[i]: Fm_inert[i]/MsH for i in range(len(inert_solids))})
    hH = _hydro(H, TbH, P_out, rc)
    FbH = (hH["ug"]-umf)*A*hH["rho_mol_b"]; FeH = umf*A*(P_out/(R*TeH))
    Tg_out = (FbH*TbH + FeH*TeH)/(FbH+FeH)
    return dict(tag=pkg.NAME, sol=sol, rc=rc,
                cons_total=cons_total, conv_pct=(y_reactive-y_out)/y_reactive*100,
                y_out=y_out, Tg_out=float(Tg_out), Ts_out=float(TsH),
                fmol_in=n_gas, fmol_out=fmol_out, w_out=w_out, P_out=float(P_out),
                reactive_solids=reactive_solids, inert_solids=inert_solids,
                Fm_inert=Fm_inert, Ms_in=Ms_in)


def stream_table(res):
    """

    Print an IDAES style stream table

    """
    from reaction_packages._common import MW_GAS
    pkg = res["rc"]["pkg"]; rc = res["rc"]
    n_gas = res["fmol_in"]; fmol_out = res["fmol_out"]; cons = res["cons_total"]
    reactive = pkg.GAS_REACTIVE; products = pkg.GAS_PRODUCTS; inerts = pkg.GAS_INERTS
    y_in = pkg.y_in
    # Rebuild gas outlet mole fractions
    y_out = {reactive: res["y_out"]}
    for sp in inerts:
        y_out[sp] = y_in.get(sp, 0.0)*n_gas/fmol_out
    for sp in products:
        y_out[sp] = y_in.get(sp, 0.0) + cons/fmol_out  # 1 mol product per consumed mol reactive 
    s = sum(y_out.values());  y_out = {k: v/s for k, v in y_out.items()}
    MWg_in  = sum(y_in.get(k, 0.0)*MW_GAS.get(k, 0.0) for k in set(list(y_in)+list(MW_GAS)))
    MWg_out = sum(y_out.get(k, 0.0)*MW_GAS.get(k, 0.0) for k in set(list(y_out)+list(MW_GAS)))
    # Solid flows
    Ms_in = rc["Ms_in"]
    Fm_final = res["sol"].y[2:2+len(pkg.SOLID_REACTIVE), -1]
    Ms_out = float(Fm_final.sum() + res["Fm_inert"].sum())
    w_in = pkg.w_in; w_out = res["w_out"]
    cols = ["Gas Inlet", "Gas Outlet", "Solid Inlet", "Solid Outlet"]
    rows = [("flow_mol [mol/s]", f"{n_gas:.2f}", f"{fmol_out:.2f}", "-", "-"),
            ("flow_mass [kg/s]", "-", "-", f"{Ms_in:.2f}", f"{Ms_out:.2f}"),
            ("T [K]", f"{rc['T_g']:.1f}", f"{res['Tg_out']:.1f}", f"{rc['T_s']:.1f}", f"{res['Ts_out']:.1f}"),
            ("P [Pa]", f"{rc['P']:.0f}", f"{res.get('P_out', rc['P']):.0f}", "-", "-")]
    for sp in [reactive] + list(inerts) + list(products):
        rows.append((f"y_{sp}", f"{y_in.get(sp,0):.4f}", f"{y_out.get(sp,0):.4f}", "-", "-"))
    for sp in list(pkg.SOLID_REACTIVE) + list(pkg.SOLID_INERTS):
        rows.append((f"w_{sp}", "-", "-", f"{w_in.get(sp,0):.4f}", f"{w_out.get(sp,0):.4f}"))
    print("\n" + "="*78 + f"\n  STREAM TABLE — {res['tag']}\n" + "="*78)
    print(f"{'':<20}" + "".join(f"{c:>14}" for c in cols))
    for name, *vals in rows:
        print(f"{name:<20}" + "".join(f"{v:>14}" for v in vals))
    print("="*78)
    dp = (rc['P'] - res.get('P_out', rc['P']))/rc['P']*100
    print(f"  Gas pressure drop across bed = {dp:+.2f}%   "
          f"Reactive-gas conversion = {res['conv_pct']:.2f}%   "
          f"Solid mass change = {(Ms_out-Ms_in)/Ms_in*100:+.2f}%")
    return dict(Ms_in=Ms_in, Ms_out=Ms_out, MWg_in=MWg_in, MWg_out=MWg_out, y_out=y_out)

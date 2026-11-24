"""
Validation of the lab dry oxidation reactor against the TU Darmstadt data
"""
import sys
import os
import math

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),
        "Reliable_models", "Lab")
)

from pyomo.environ import value, Var
from idaes.core.util import scaling as iscale
from idaes.core.solvers import get_solver
import idaes.logger as idaeslog

idaeslog.getLogger("idaes").setLevel(idaeslog.WARNING)

from Dry_Oxidation_Reactor_Lab import (
    IND, SOLVER_OPTS, build_model, fix_intensive, set_case,
    _solve, _ok, _homotopy, _collect,
)

# TU Darmstadt lab data
AIR_M3H = 1.2          # m3/h air in
O2OUT_M3H = 0.15       # m3/h O2 in the exit gas
FE_GMIN = 13.31        # g/min metallic Fe in

VM_A = 22.414e-3       # Molar volume m3/mol at 0 C, 1 atm
GRAV = 9.81            # m/s2
MW_FE = 55.845e-3      # kg/mol
RHO_SKEL = {"Fe": 7874.0}
FE_KGH = FE_GMIN * 60.0 / 1000.0        # 0.7986 kg/h


def exp_targets():
    """
    Measured values of the lab comparison, from the oxygen balance
    X_O2 40.48 %, X_Fe 42.43 %
    """
    n_air = AIR_M3H / VM_A
    n_o2_cons = 0.21 * n_air - O2OUT_M3H / VM_A
    fe_molh = FE_KGH / MW_FE
    return {
        "n_air": n_air,
        "X_O2": n_o2_cons / (0.21 * n_air) * 100.0,
        "X_Fe": n_o2_cons / 0.75 / fe_molh * 100.0,
    }


def wen_yu_umf(dp, T=1023.15, P=1.0e5, rho_p=5748.0):
    """
    Wen and Yu correlation for umf
    """
    R = 8.314
    rho_g = P * 0.028965 / (R * T)
    mu = 1.72e-5 * (T / 273.15) ** 0.7
    Ar = GRAV * dp ** 3 * rho_p * rho_g / mu ** 2
    Re_mf = math.sqrt(33.7 ** 2 + 0.0408 * Ar) - 33.7
    return Re_mf * mu / (rho_g * dp)


def rho_particle(w, porosity=0.27):
    """
    Particle density from the skeleton densities and the porosity
    """
    inv = sum(w[j] / RHO_SKEL[j] for j in w)
    return (1.0 - porosity) / inv


NORIF = 2500  # orifice density 1/m2


def rig_target(H=0.65, tgas=298.15, tsolid=298.15, dp_um=100.0,
               rp_um=50.0, norif=2500):
    """
    All model inputs for one lab case
    """
    tg = exp_targets()
    rho_in = rho_particle({"Fe": 1.0})
    dp = dp_um * 1e-6
    return {
        "D": 0.054, "H": H, "n_orifice": norif,
        "gas_flow_mol": tg["n_air"] / 3600.0,
        "solid_flow_mass": FE_KGH / 3600.0,
        "gas_T": tgas, "solid_T": tsolid, "P": 1.0e5,
        "y_O2": 0.21, "y_N2": 0.79, "y_H2O": 1e-5, "y_CO2": 1e-5, "y_H2": 1e-5,
        "w_Fe": 1.0, "w_Al2O3": 0.0, "w_Fe2O3": 0.0,
        "w_Fe3O4": 1e-5, "w_FeO": 1e-5,
        "particle_dia": dp, "velocity_mf": wen_yu_umf(dp, rho_p=rho_in),
        "voidage_mf": 0.45, "voidage": 0.50, "particle_porosity": 0.27,
        "grain_radius": rp_um * 1e-6,
        "rho_p_in": rho_in,
    }


def _interp(a, b, s, how):
    return a * (b / a) ** s if how == "log" else a + (b - a) * s


def _stiff_step(m, b, solver, verbose, tag):
    """
    Extra attempts when a solve step fails. First without the O2 clamp, then with slower kinetics
    """
    rx = m.fs.dry_reactions
    t = "skipped"
    for lam in (0.0, 0.5, 1.0):
        rx.clamp_lambda = lam
        t = _solve(b, solver, f"{tag} lam{lam}", verbose)
        if not _ok(t):
            break
    rx.clamp_lambda = 1.0
    if not _ok(t):
        for sf in (0.3, 0.1, 0.03, 0.01):
            rx._scale_factor_rxn = sf
            td = _solve(b, solver, f"{tag} dip{sf}", verbose)
            if not _ok(td):
                continue
            t = _homotopy(m, solver, verbose, tag, sf0=sf)
            if _ok(t):
                break
    if not _ok(t):
        rx._scale_factor_rxn = 1.0
    return t


def _continue(m, solver, src, dst, n_cont, verbose, tag):
    """
    The model goes from case to another in small steps and solve again at each step. 
    If a step fails, the step is split into smaller ones
    """
    b = m.fs.BFB
    rp_src = src.get("grain_radius", 50e-6)

    def _fix_at(s):
        D = _interp(src["D"], dst["D"], s, "log")
        H = _interp(src["H"], dst["H"], s, "lin")
        gas = _interp(src["gas_flow_mol"], dst["gas_flow_mol"], s, "log")
        sol = _interp(src["solid_flow_mass"], dst["solid_flow_mass"], s, "log")
        Tg = _interp(src["gas_T"], dst["gas_T"], s, "lin")
        Ts = _interp(src["solid_T"], dst["solid_T"], s, "lin")
        umf = _interp(src["velocity_mf"], dst["velocity_mf"], s, "log")
        dp = _interp(src["particle_dia"], dst["particle_dia"], s, "log")
        rp = _interp(rp_src, dst["grain_radius"], s, "log")
        wfe = _interp(src["w_Fe"], dst["w_Fe"], s, "lin")
        wal = _interp(src["w_Al2O3"], dst["w_Al2O3"], s, "lin")
        norf = _interp(src["n_orifice"], dst["n_orifice"], s, "log")
        m.fs.solid_properties.velocity_mf.fix(umf)
        m.fs.solid_properties.particle_dia.fix(dp)
        m.fs.dry_reactions.grain_radius.fix(rp)
        b.solid_inlet.mass_frac_comp[0, "Fe"].fix(wfe)
        b.solid_inlet.mass_frac_comp[0, "Al2O3"].fix(wal)
        set_case(m, D, H, norf, gas, sol, Tg, Ts)

    def _try(s, sub):
        _fix_at(s)
        t = _solve(b, solver, f"{tag}{sub}", verbose)
        if not _ok(t):
            t = _stiff_step(m, b, solver, verbose, f"{tag}{sub}")
        return _ok(t)

    # s = How far along to the target case it is. s_prev = the last point that solved
    s_prev = 0.0
    for k in range(1, n_cont + 1):
        s = k / n_cont
        if _try(s, f" {k}/{n_cont}"):
            s_prev = s
            continue

        # Split the failed step into smaller steps from the last optimal point 
        landed = False
        for depth, nsub in ((1, 2), (2, 4)):
            ok_all = True
            for j in range(1, nsub + 1):
                sj = s_prev + (s - s_prev) * j / nsub
                if not _try(sj, f" {k}/{n_cont} d{depth}.{j}"):
                    ok_all = False
                    break
            if ok_all:
                landed = True
                break
        if landed:
            s_prev = s
        elif verbose:
            print(" step failed")
    # Solve one last time at the exact target values, so a failed last step cannot leave at an intermediate point 
    _fix_at(1.0)


def solve_rig(target, n_cont=40, verbose=False):
    """
    Solve one lab sim case, start from the industrial point, then it goes to a case with dp = 100 um, then it changes only dp to the target
    """
    solver = get_solver()
    solver.options = dict(SOLVER_OPTS)
    m = build_model()
    b = m.fs.BFB
    b.deltaP_orifice.fix(3400.0)  # orifice pressure drop, fixed value

    # Stage 1: Start from the industrial case
    fix_intensive(m, IND)
    m.fs.dry_reactions.grain_radius.fix(target["grain_radius"])
    set_case(m, IND["D"], IND["H"], IND["n_orifice"],
             IND["gas_flow_mol"], IND["solid_flow_mass"],
             IND["gas_T"], IND["solid_T"])
    gas_args = {"flow_mol": IND["gas_flow_mol"], "temperature": IND["solid_T"],
                "pressure": IND["P"],
                "mole_frac": {"O2": IND["y_O2"], "N2": IND["y_N2"],
                              "H2O": IND["y_H2O"], "CO2": IND["y_CO2"],
                              "H2": IND["y_H2"]}}
    sol_args = {"flow_mass": IND["solid_flow_mass"],
                "particle_porosity": IND["particle_porosity"],
                "temperature": IND["solid_T"],
                "mass_frac": {"Fe": IND["w_Fe"], "Al2O3": IND["w_Al2O3"],
                              "Fe2O3": IND["w_Fe2O3"], "Fe3O4": IND["w_Fe3O4"],
                              "FeO": IND["w_FeO"]}}
    iscale.calculate_scaling_factors(m)
    for v in m.fs.BFB.component_data_objects(Var, descend_into=True):
        if "frac_comp" in v.name or "porosity" in v.name:
            v.setlb(0)
    m.fs.dry_reactions._scale_factor_rxn = 1e-4
    b.initialize(outlvl=idaeslog.CRITICAL,
                 gas_phase_state_args=gas_args, solid_phase_state_args=sol_args)
    # The O2 clamp makes the first solves hard, so the reaction rate sf is increased slowly with the clamp off, 
    # and then the clamp is switched back
    rxn = m.fs.dry_reactions
    rxn.clamp_lambda = 0.0
    t = _homotopy(m, solver, verbose, "ind")
    for ratio in (1.6, 1.25):
        sf_now = value(rxn._scale_factor_rxn)
        if _ok(t) or sf_now >= 0.999:
            break
        # If the rate ramp gets stuck, its done again with smaller steps
        back = max(1e-4, sf_now / ratio ** 2)
        t = _homotopy(m, solver, verbose, f"ind-r{ratio}", sf0=back, ratio=ratio)
    if value(rxn._scale_factor_rxn) < 0.999 or not _ok(t):
        raise RuntimeError(
            f"reaction rate error ")
    for lam in (0.25, 0.5, 0.75, 1.0):
        rxn.clamp_lambda = lam
        tl = _solve(b, solver, f"ind lam{lam}", verbose)
        if not _ok(tl):
            tl = _stiff_step(m, b, solver, verbose, f"ind lam{lam}")
        if not _ok(tl):
            raise RuntimeError(
                f" O2 clamp error")

    # Stage 2: solve a lab case with dp = 100 um first and then change only the particle diameter to the target.
    warm = dict(target)
    warm["n_orifice"] = NORIF
    base = dict(warm)
    base["particle_dia"] = 100e-6
    # u_mf of 100 um 
    base["velocity_mf"] = wen_yu_umf(100e-6, rho_p=warm["rho_p_in"])
    _continue(m, solver, IND, base, n_cont, verbose, "rig")
    _continue(m, solver, base, warm, max(30, n_cont), verbose,
              "dpleg")
    if target["n_orifice"] != NORIF:
        _continue(m, solver, warm, target, max(15, n_cont // 2), verbose,
                  "norif")

    # Stage 3: Final solve, if it doesnt converge, the results arent used
    m.fs.dry_reactions.clamp_lambda = 1.0
    m.fs.dry_reactions._scale_factor_rxn = 1.0
    term = _solve(b, solver, "FINAL", verbose)
    if not _ok(term):
        term = _stiff_step(m, b, solver, verbose, "FINAL")
    sf_final = value(m.fs.dry_reactions._scale_factor_rxn)
    if not _ok(term) or sf_final < 0.999:
        raise RuntimeError(
            f"the solve didnt converge ")
    r = _collect(m, target, term)
    r.update(read_dP(m))
    return m, r


def read_dP(m):
    """
    Bed pressure drop from the model pressures
    """
    b = m.fs.BFB
    P0 = value(b.gas_emulsion.properties[0, 0].pressure)
    P1 = value(b.gas_emulsion.properties[0, 1].pressure)
    xs = list(b.length_domain)
    return {
        "dP_bed_Pa": P0 - P1, "dP_bed_mbar": (P0 - P1) / 100.0,
        "delta_avg": sum(value(b.delta[0, x]) for x in xs) / len(xs),
        "db_top_m": value(b.bubble_diameter[0, 1]),
        "ug_bottom_mps": value(b.velocity_superficial_gas[0, 0]),
        "ug_top_mps": value(b.velocity_superficial_gas[0, 1]),
    }

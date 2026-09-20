"""
dp = 0.28 mm update for every runner


"""
import contextlib
import io

from idaes.core.util import scaling as iscale
from idaes.core.solvers import get_solver

DP_MM = 0.28
DP_M = 2.8e-4
LABEL = "dp = 0.28 mm"


def to_m(dp_mm):
    """
    mm -> m
    """
    return dp_mm * 1e-3


def refix_rescale(m, dp_m):
    """
    Re fix particle diameter and redo the scaling.
    dp sits in the scaling of the emulsion Reynolds reformulation and
    both convective heat coefficients
    """
    m.fs.solid_properties.particle_dia.fix(dp_m)
    iscale.calculate_scaling_factors(m)


def snapshot(m):
    """
    Every Var value by name, so a failed solve doesnt affect the next one
    """
    from pyomo.environ import Var
    return {v.name: v.value for v in m.component_data_objects(
        Var, descend_into=True)}


def restore(m, memo):
    from pyomo.environ import Var
    for v in m.component_data_objects(Var, descend_into=True):
        if v.name in memo:
            v.set_value(memo[v.name])


def looks_ignited(ep):
    """
    Balance/physics check for a 28 mm solve

    """
    if ep is None:
        return False, ""
    checks = [
        (ep["n_bad"] == 0, "n_bad>0"),
        (ep["err"] is None or ep["err"] <= 0.1, "err_mass>0.1%"),
        (-0.02 <= ep["delta_min"] and ep["delta_max"] <= 1.0, "delta out of range"),
        (400.0 <= ep["T_solid_out"] <= 1600.0, "T_solid_out out of range"),
        (0.0 <= ep["X_solid"] <= 100.0, "X_solid out of range"),
        (-1e-3 <= ep["X_gas"] <= 100.0, "X_gas out of range"),
    ]
    bad = [msg for ok, msg in checks if not ok]
    return (not bad), ";".join(bad)


def probe(m, reactor, mod=None, results=None):
    """
    Endpoint + delta range + err_mass
    """
    from runner_axial import endpoint
    ep = endpoint(m, reactor)
    b = m.fs.BFB
    deltas = [b.delta[0, x].value for x in sorted(b.length_domain, key=float)]
    ep["delta_min"] = min(deltas)
    ep["delta_max"] = max(deltas)
    ep["err"] = (results or {}).get("err_mass")
    return ep


def _solver(mod):
    solver = mod.get_solver() if hasattr(mod, "get_solver") else get_solver()
    if hasattr(mod, "SOLVER_OPTS"):
        solver.options = dict(mod.SOLVER_OPTS)
    return solver


def _warm_solve(mod, m, dp_m, verbose):
    """
    One warm solve after refix_rescale. Only the dry reactor has _solve,
    wet and reduction just get a plain idaes solve 
    """
    solver = _solver(mod)
    tag = f"dp_replug {dp_m*1e6:.0f}um"
    try:
        return mod._solve(m.fs.BFB, solver, tag, verbose)
    except AttributeError:
        res = solver.solve(m.fs.BFB, tee=verbose)
        return str(res.solver.termination_condition)


def solve_lab_dp(mod, target, dp_m, reactor, verbose=True, ncont=None):
    """
    One lab solve at dp= 28 mm. Cold dict injection first, warm replug
    as the fallback}
    """
    opts = {"n_cont": ncont} if ncont else {}
    tgt = dict(target, particle_dia=dp_m)
    try:
        m, results = mod.solve_case(tgt, verbose=verbose, **opts)
        ep = probe(m, reactor, mod, results)
        ok, msg = looks_ignited(ep)
        if ok and "optimal" in str(results.get("termination", "?")):
            return m, results, "cold_dp028", msg
        why = msg or str(results.get("termination", "?"))
        if verbose:
            print(f" fail ({why}),",
                  flush=True)
    except Exception as e:
        why = f"EXC:{type(e).__name__}"
        if verbose:
            print(f"  fail ({why})",
                  flush=True)

    # A finer geometric startup, cold solve and then replug
    base = dict(target, particle_dia=1.5e-3)
    why = "no attempt"
    for extra in ({}, {"n_cont": 40}):
        call_opts = dict(opts, **extra)
        try:
            m, results = mod.solve_case(base, verbose=verbose, **call_opts)
        except Exception as e:
            why = f"EXC:{type(e).__name__} in base solve"
            if verbose:
                print(f" fail ({why})"
                      if not extra else
                      f"  fail({why})", flush=True)
            continue
        refix_rescale(m, dp_m)
        term = _warm_solve(mod, m, dp_m, verbose)
        results = dict(results, termination=term)
        ep = probe(m, reactor, mod, results)
        ok, msg = looks_ignited(ep)
        if ok:
            return m, results, "warm_replug_from_1.5e-3", msg
        why = f"{msg} ({term})"
        if verbose:
            print(f"  replug fail ({why})"
                  if not extra else f"  replug fail({why})",
                  flush=True)
    raise RuntimeError(f"replug fail {why}")


def solve_ind_dp(mod, dp_m, reactor, verbose=False):
    """
    Industrial solve at dp= 28 m. main() lands at 1.5 mm, then replug.

    """
    with contextlib.redirect_stdout(io.StringIO()):
        m = mod.main()
    refix_rescale(m, dp_m)
    term = _warm_solve(mod, m, dp_m, verbose)
    ep = probe(m, reactor)
    ok, msg = looks_ignited(ep)
    if not ok:
        raise RuntimeError(f"Replug fail {msg} ({term})")
    return m, term, "warm_replug_from_1.5e-3"

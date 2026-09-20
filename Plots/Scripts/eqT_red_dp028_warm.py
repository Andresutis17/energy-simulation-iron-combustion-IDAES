"""
EqT at a small dp via a gas_T ramp

"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import dp_patch
from runner_axial import extract_rows, load_module

ERR_GATE = 0.1  # Tolerance


def solve_once(b, solver):
    """
    One solve
    """
    try:
        res = solver.solve(b, tee=False)
        return str(res.solver.termination_condition)
    except Exception as e:
        return f"EXC:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reactor", required=True)
    ap.add_argument("--tgas", type=float, required=True)
    ap.add_argument("--dp", type=float, default=0.28)
    ap.add_argument("--suffix", default="_eqT_dp028")
    args = ap.parse_args()
    reactor = args.reactor

    mod = load_module(common.LAB_SCRIPTS[reactor], f"{reactor}_eqT_warm")
    dp_m = dp_patch.to_m(args.dp)
    target = dict(mod.LAB)
    target["H"] = common.LAB_MATCH[reactor]["H"]
    target["n_orifice"] = common.LAB_MATCH[reactor]["n_orifice"]

    # The match point
    m, results, path, msg = dp_patch.solve_lab_dp(
        mod, target, dp_m, reactor, verbose=True)
    print(f"match via {path} ({msg})", flush=True)

    solver = dp_patch._solver(mod)
    b = m.fs.BFB
    ts = target["solid_T"]
    snap = dp_patch.snapshot(m)

    def attempt(t_set, tag=""):
        """
        One solve at the given Tg, restored to the last good point 
        """
        dp_patch.restore(m, snap)
        mod.set_case(m, target["D"], target["H"], target["n_orifice"],
                     target["gas_flow_mol"], target["solid_flow_mass"],
                     t_set, ts)
        if hasattr(mod, "_solve"):
            return mod._solve(b, solver, tag, True)
        return solve_once(b, solver)

    def good(term):
        if hasattr(mod, "_ok"):
            return mod._ok(term)
        return "optimal" in str(term)

    # Walk from the match gas T to the eqT value
    tg0 = float(target["gas_T"])
    t_end = float(args.tgas)
    steps, t = [], tg0
    while (t_end - t) * (1 if t_end >= tg0 else -1) > 1e-9:
        step = 100.0 if t < 900.0 else 15.0
        t = (min if t_end >= tg0 else max)(t_end, t + step)
        steps.append(t)
    t_now = tg0
    term = "not solved"
    for t_next in steps:
        term = attempt(t_next, tag=f"ramp {t_next:.0f}")
        if not good(term):
            term = attempt(t_next, tag=f"re-warm {t_next:.0f}")
        if not good(term) and hasattr(mod, "_homotopy"):
            mod._homotopy(m, solver, True, f"eqT{t_next:g}")
            term = attempt(t_next, tag=f"homotopy final {t_next:.0f}")
        if good(term):
            t_now = t_next
            snap = dp_patch.snapshot(m)
            continue
        lo, hi = t_now, t_next
        for _ in range(3):
            mid = 0.5 * (lo + hi)
            if hi - lo < 2.0:
                break
            mid_term = attempt(mid, tag=f"mid {mid:.1f}")
            if not good(mid_term):
                mid_term = attempt(mid, tag=f"mid re-warm {mid:.1f}")
            if good(mid_term):
                lo = mid
                term = mid_term
            else:
                hi = mid
        if good(term) and lo > t_now:
            t_now = lo
            snap = dp_patch.snapshot(m)
        if not good(term) or t_now < t_next - 1e-6:
            break

    # When the cold branch dies below the target only the hot one is left
    # One straight solve at the target from the last good point
    if not good(term) or t_now < t_end - 1e-6:
        for tag in (f"branch jump {t_end:.0f}",
                    f"branch jump warm {t_end:.0f}"):
            term = attempt(t_end, tag=tag)
            if good(term):
                t_now = t_end
                snap = dp_patch.snapshot(m)
                break

    # Collect 
    tgt_eqT = dict(target, gas_T=float(args.tgas))
    landed = good(term) and t_now >= float(args.tgas) - 1e-6
    results = mod._collect(m, tgt_eqT, term if landed else "maxIterations")
    ep = dp_patch.probe(m, reactor, mod, results)
    gate_ok, gate_msg = dp_patch.looks_ignited(ep)
    valid = (landed and gate_ok
             and float(results.get("err_mass") or 1e9) <= ERR_GATE
             and results.get("gas_feasible") is not False)

    point = {"reactor": reactor, "scale": "lab", "term": term,
             "err_mass": results.get("err_mass"),
             "gas_feasible": results.get("gas_feasible"),
             "dp_mm": args.dp, "dp_path": f"{path}+Tg_ramp", **ep}
    point["valid"] = valid

    common.ensure_dirs()
    if valid:
        rows, H = extract_rows(m, reactor)
        with open(common.axial_csv(reactor, "lab", args.suffix), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        print(f"eqT fail {gate_msg or term} at Tg={t_now:g} -")
    meta = dict(point)
    meta["source"] = common.LAB_SCRIPTS[reactor]
    meta["gas_T"] = float(args.tgas)
    meta["solid_T"] = ts
    with open(common.axial_meta(reactor, "lab", args.suffix), "w") as f:
        json.dump(meta, f, indent=2)
    print("POINT " + json.dumps(point), flush=True)


if __name__ == "__main__":
    main()

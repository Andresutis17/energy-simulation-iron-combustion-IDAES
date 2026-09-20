"""
Dry eqT at dp = 0.28 mm via a dp homotopy down from 1.5 mm

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

ERR_GATE = 0.1
# dp ladder 
DP_RATIO = 0.82


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
    ap.add_argument("--reactor", default="dry")
    ap.add_argument("--tgas", type=float, default=1073.0)
    ap.add_argument("--dp", type=float, default=0.28)
    ap.add_argument("--suffix", default="_eqT_dp028")
    args = ap.parse_args()
    reactor = args.reactor

    mod = load_module(common.LAB_SCRIPTS[reactor], f"{reactor}_eqT_dpdown")
    dp_end = dp_patch.to_m(args.dp)
    target = dict(mod.LAB)
    target["H"] = common.LAB_MATCH[reactor]["H"]
    target["n_orifice"] = common.LAB_MATCH[reactor]["n_orifice"]
    ts = target["solid_T"]
    t_end = float(args.tgas)

    solver = dp_patch._solver(mod)
    b = None
    m = None
    snap = None

    def attempt(t_set, dp_m, tag=""):
        """
        One solve at the given Tg and dp, restored to the last good
        point first
        """
        dp_patch.restore(m, snap)
        m.fs.solid_properties.particle_dia.fix(dp_m)
        mod.set_case(m, target["D"], target["H"], target["n_orifice"],
                     target["gas_flow_mol"], target["solid_flow_mass"],
                     t_set, ts)
        return mod._solve(b, solver, tag, True)

    def good(term):
        return mod._ok(term)

    # Phase 1: eqT at 1.5 mm
    # Same ladder as the warm script, 100 K steps, 15 K past 900
    target["particle_dia"] = 1.5e-3
    m, results, path, msg = dp_patch.solve_lab_dp(
        mod, target, 1.5e-3, reactor, verbose=True)
    print(f"dp1.5 match landed {path} ({msg})", flush=True)
    b = m.fs.BFB
    snap = dp_patch.snapshot(m)

    tg0 = float(target["gas_T"])
    t_now = tg0
    term = "not solved"
    
    while t_now < t_end - 1e-6:
        steps, t = [], t_now
        while (t_end - t) * (1 if t_end >= tg0 else -1) > 1e-9:
            step = 100.0 if (t + 100.0) <= 900.0 else 15.0
            t = (min if t_end >= tg0 else max)(t_end, t + step)
            steps.append(t)
        advanced = died = False
        for t_next in steps:
            term = attempt(t_next, 1.5e-3, tag=f"ramp {t_next:.0f}")
            if not good(term):
                term = attempt(t_next, 1.5e-3, tag=f"re-warm {t_next:.0f}")
            if not good(term):
                mod._homotopy(m, solver, True, f"eqT{t_next:g}")
                term = attempt(t_next, 1.5e-3,
                               tag=f"homotopy final {t_next:.0f}")
            if good(term):
                t_now = t_next
                snap = dp_patch.snapshot(m)
                continue
            lo, hi = t_now, t_next
            for _ in range(3):
                mid = 0.5 * (lo + hi)
                if hi - lo < 2.0:
                    break
                mid_term = attempt(mid, 1.5e-3, tag=f"mid {mid:.1f}")
                if not good(mid_term):
                    mid_term = attempt(mid, 1.5e-3,
                                       tag=f"mid re warm {mid:.1f}")
                if good(mid_term):
                    lo = mid
                    term = mid_term
                else:
                    hi = mid
            if good(term) and lo > t_now:
                t_now = lo
                snap = dp_patch.snapshot(m)
                advanced = True
                break
            died = True
            break
        if died or (not advanced and t_now < t_end - 1e-6):
            break
    if not good(term) or t_now < t_end - 1e-6:
        term = attempt(t_end, 1.5e-3, tag=f"branch jump {t_end:.0f}")
        if good(term):
            t_now = t_end
            snap = dp_patch.snapshot(m)
    if not good(term) or t_now < t_end - 1e-6:
        sys.exit(f" failed Tg={t_now:g} ({term}) -")
    print(f"landed Tg={t_end:g} ({term})", flush=True)

    # Phase 2: shrink dp 
    dp_m = 1.5e-3
    while dp_m > dp_end * (1 + 1e-9):
        dp_next = max(dp_end, dp_m * DP_RATIO)
        snap = dp_patch.snapshot(m)
        term = attempt(t_end, dp_next, tag=f"dp {dp_next*1e3:.3f}mm")
        if not good(term):
            term = attempt(t_end, dp_next, tag=f" {dp_next*1e3:.3f}")
        if not good(term):
            sys.exit(f"dp homotopy fail {dp_next*1e3:.3f} mm ({term})")
        dp_m = dp_next
        print(f"dp landed  {dp_m*1e3:.3f} mm ({term})", flush=True)

    # Collect 
    tgt_eqT = dict(target, gas_T=t_end, particle_dia=dp_end)
    results = mod._collect(m, tgt_eqT, term)
    ep = dp_patch.probe(m, reactor, mod, results)
    gate_ok, gate_msg = dp_patch.looks_ignited(ep)
    valid = (gate_ok
             and float(results.get("err_mass") or 1e9) <= ERR_GATE
             and results.get("gas_feasible") is not False)

    point = {"reactor": reactor, "scale": "lab", "term": term,
             "err_mass": results.get("err_mass"),
             "gas_feasible": results.get("gas_feasible"),
             "dp_mm": args.dp,
             "dp_path": "dp1.5_eqT+dp_hot_branch_down", **ep}
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
        print(f"eqT fail {gate_msg or term} ",
              flush=True)
    meta = dict(point)
    meta["source"] = common.LAB_SCRIPTS[reactor]
    meta["gas_T"] = t_end
    meta["solid_T"] = ts
    with open(common.axial_meta(reactor, "lab", args.suffix), "w") as f:
        json.dump(meta, f, indent=2)
    print("POINT " + json.dumps(point), flush=True)


if __name__ == "__main__":
    main()

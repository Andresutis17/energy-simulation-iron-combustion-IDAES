"""
Reduction, wet and dry operating multiplots,
mean solids residence time at constant stoichiometry
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from plot_R4_operating_reduction import operating_figure

FIGS = [("reduction", "R10a", r"$\lambda_{H_2}=3$", True),
        ("wet", "R10b", r"$\lambda_{\mathrm{steam}}=1.24$", True),
        ("dry", "R10c", r"$\lambda_{O_2}=1.50$", False)]
SWEEP = {"reduction": "sens_reduction_tau_scale_lam3_operating.csv",
         "wet": "sens_wet_tau_scale_operating.csv",
         "dry": "sens_dry_tau_scale_operating.csv"}
BASE_AXIAL = {"reduction": "_lam3", "wet": "_dp028", "dry": "_dp028"}
TAU_XLABEL = (r"$\bar{\tau}_s = M_{s,\mathrm{bed}}/"
              r"\dot{m}_{s,\mathrm{in}}\,[\mathrm{min}]$")


def f_suffix(f):
    return f"_taufs{int(round(f * 10000)):04d}"


def sweep_points(reactor):
    rows = common.load_csv(SWEEP[reactor])
    out, dropped = [], []
    skip = ("reactor", "knob", "term", "path", "valid", "gas_feasible",
            "banner_ok", "solver_status", "validation_issues",
            "validation_ok", "log_path")
    for r in rows:
        if (r.get("valid", "").strip().lower() == "true"
                and "optimal" in r.get("term", "") and r.get("X_prod")):
            row = {}
            for k, v in r.items():
                if k in skip:
                    continue
                try:
                    row[k] = float(v) if v not in ("", None) else None
                except (TypeError, ValueError):
                    row[k] = None
            out.append(row)
        else:
            dropped.append(f"f={float(r['value']):g}")
    return out, dropped


def exact_tau(reactor, f):
    sfx = BASE_AXIAL[reactor] if f == 1.0 else f_suffix(f)
    try:
        rows, meta = common.load_axial(reactor, "lab", suffix=sfx)
    except FileNotFoundError:
        return None
    if not meta.get("valid", True):
        return None
    ts, _, _ = common.tau_profile(rows, meta)
    return ts


def main():
    common.apply_style()

    for reactor, stem, note, gamma in FIGS:
        if not os.path.exists(os.path.join(common.DATA, SWEEP[reactor])):
            continue
        pts, dropped = sweep_points(reactor)
        xy, miss = [], []
        for r in pts:
            ts = exact_tau(reactor, r["value"])
            if ts is None:
                miss.append(f"f={r['value']:g}")
            else:
                r["tau_s"] = ts
                xy.append((ts, r))
        miss
        if not xy:
            continue
        xy.sort(key=lambda p: p[0])
        tau0 = next(ts for ts, r in xy if r["value"] == 1.0)
        def _f_of_tau(t, t0=tau0):
            t = np.asarray(t, dtype=float)
            return np.where(t > 0, t0 / np.where(t > 0, t, 1.0), np.nan)

        def _tau_of_f(x, t0=tau0):
            x = np.asarray(x, dtype=float)
            return np.where(x > 0, t0 / np.where(x > 0, x, 1.0), np.nan)

        operating_figure([r for _, r in xy], "tau_s", tau0,
                         "Baseline operating point ($f=1$)",
                         TAU_XLABEL, stem,
                         reactor=reactor,
                         gamma=gamma,
                         lam_note=note,
                         top_func=(_f_of_tau, _tau_of_f),
                         top_label=r"$f$",
                         top_label_pad=3,
                         title_pad_top=24)
        taus = ", ".join(f"f={r['value']:g}:{ts:.1f}" for ts, r in xy)


if __name__ == "__main__":
    main()

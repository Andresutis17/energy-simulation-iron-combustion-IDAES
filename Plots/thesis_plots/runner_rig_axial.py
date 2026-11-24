"""
Writes the axial profile the E1 figure plots
"""

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import common  
import validate_oxi_dry_exp_v19 as V19  

def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    target = V19.rig_target(H=0.828, tgas=298.15, tsolid=1023.15,
                            dp_um=110.0, rp_um=50.0, norif=2500)
    m, r = V19.solve_rig(target, n_cont=40)
    b = m.fs.BFB
    H = b.bed_height.value
    D = b.bed_diameter.value

    xs = sorted(b.length_domain, key=float)
    rows = []
    for x in xs:
        db = b.bubble_diameter[0, x].value
        rows.append({
            "x_norm": float(x),
            "z_m": float(x) * H,
            "db_m": db,
            "db_max_m": b.bubble_diameter_max[0, x].value,
            "db_over_D": db / D,
        })

    point = {
        "X_O2_pct": r["X_O2"], "X_Fe_pct": r["X_Fe"],
        "dP_bed_mbar": r["dP_bed_mbar"], "db_top_mm": r["db_top_m"] * 1e3,
        "termination": r["termination"], "delta_avg": r["delta_avg"],
        "D_m": D, "H_m": H,
    }

    common.ensure_dirs()
    csv_path = os.path.join(common.DATA, "axial_dry_rig.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = dict(point)
    with open(os.path.join(common.DATA, "axial_dry_rig.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("POINT " + json.dumps(point), flush=True)


if __name__ == "__main__":
    main()

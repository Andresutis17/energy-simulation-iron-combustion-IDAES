
"""
Merges refined walk rows into a family CSV  for each knob value,
the best row stays
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  
from gather_sensitivity import FAMILIES, csv_path, parse_points_from_log, write_csv


def is_valid(r):

    """
    True if the row passed its quality checks 
    """

    return r.get("valid") == "True" or r.get("valid") is True


def rank(row, source):
    """
    The score for picking the best candidate, lower error wins
    
    """
    e_m = row.get("err_mass")
    e_m = float(e_m) if e_m not in (None, "") else float("inf")
    return (0 if is_valid(row) else 1, e_m, source)


def main():
    family, suffixes = sys.argv[1], sys.argv[2:] or ["refined"]
    recipe = FAMILIES[family]

    by_value = {}
    csv_rows = []
    if os.path.exists(csv_path(family)):
        with open(csv_path(family), newline="") as f:
            csv_rows = list(csv.DictReader(f))
    for row in csv_rows:
        by_value.setdefault(float(row["value"]), []).append((0, row))

    used_logs = []
    for i, suffix in enumerate(suffixes, start=1):
        log = os.path.join(common.DATA, "logs", f"sens_{family}_{suffix}.log")
        if not os.path.exists(log):
            continue
        rows, drifts = parse_points_from_log(
            log, meta={"reactor": recipe["reactor"], "knob": recipe["knob"]})
        if not rows:
            continue
        used_logs.append((log, drifts))
        for row in rows:
            by_value.setdefault(float(row["value"]), []).append((i, row))
    if not used_logs:
        sys.exit("Nothing to merge")

    merged = []
    for _, candidates in sorted(by_value.items()):
        _, row = min(candidates, key=lambda c: rank(c[1], c[0]))
        merged.append(row)
    write_csv(family, merged)

    for log, drifts in used_logs:
        for dx, dt in drifts:
            if dx is not None:
                flag = "Drift" if (dx > 0.05 or dt > 0.5) else ""
                print(f"  {os.path.basename(log)} drift "
                      f"dX={dx:.3f} pp dT={dt:.2f} K{flag}")


if __name__ == "__main__":
    main()

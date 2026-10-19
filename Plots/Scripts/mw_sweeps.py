""" 
MW per sweep point
MW(point) = anchor MW x lab metric(point) / lab metric(anchor)

Anchors are the repos own solves, industrial MW from
outputs/_energy_<tag>.json, lab baseline metric from
outputs/_energy_<tag>_lab.json. Reads CSVs/JSONs
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "outputs")

LHV_H2 = 241.8264e3   # J/mol water gas

# label, data file, reactor
SWEEPS = [
    ("R4b lam5 stoich  red y_H2 ", "op_reduction_r4blam5_points.json", "red"),
    ("W4b       stoich  wet y_H2O", "sens_wet_y_H2O_operating.csv", "wet"),
    ("D4b       stoich  dry y_O2  ", "sens_dry_y_O2_operating.csv", "dry"),
    ("R9        gasflow red       ", "sens_reduction_gasflow_lam3_operating.csv", "red"),
    ("W9        gasflow wet       ", "sens_wet_gasflow_w9_operating.csv", "wet"),
    ("D9        gasflow dry       ", "sens_dry_gasflow_d9_operating.csv", "dry"),
    ("R5a       Tgas    red       ", "sens_reduction_T_gas_lam3_operating.csv", "red"),
    ("W5a       Tgas    wet       ", "sens_wet_T_gas_operating.csv", "wet"),
    ("D5a       Tgas    dry       ", "sens_dry_T_gas_operating.csv", "dry"),
    ("R5b       Tsolid  red       ", "sens_reduction_T_solid_lam3_operating.csv", "red"),
    ("W5b       Tsolid  wet       ", "sens_wet_T_solid_operating.csv", "wet"),
    ("D5b       Tsolid  dry       ", "sens_dry_T_solid_operating.csv", "dry"),
    ("D5c       overlap dry       ", "sens_dry_T_solid_overlap_operating.csv", "dry"),
    ("R10       tau     red       ", "sens_reduction_tau_scale_lam3_operating.csv", "red"),
    ("W10       tau     wet       ", "sens_wet_tau_scale_operating.csv", "wet"),
    ("D10       tau     dry       ", "sens_dry_tau_scale_operating.csv", "dry"),
]


def load_rows(path):

    full = os.path.join(DATA, path)
    if path.endswith(".json"):
        with open(full) as f:
            return json.load(f)
    with open(full, newline="") as f:
        return list(csv.DictReader(f))


def net_flow(row, species, produced=False):

    """mol/s of one gas species consumed """

    fin = float(row["gas_mol_in"]) * float(row[f"y_in_{species}"])
    fout = float(row["gas_mol_out"]) * float(row[f"y_out_{species}"])
    return (fout - fin) if produced else (fin - fout)


def row_metric(row, reactor):

    """The beds converted quantity in mol/s (H2, or DOR1 extent for dry)"""

    if reactor == "dry":
        return net_flow(row, "O2") / 1.5
    return net_flow(row, "H2", produced=(reactor == "wet"))


def pseudo_row(gas_in, gas_out):

    """
    Build one CSV like row from the gas in/out ports of the energy
    outputs, so row_metric works on CSV rows and anchors
    """

    row = {"gas_mol_in": gas_in["F"], "gas_mol_out": gas_out["F"]}
    for side, port in (("in", gas_in), ("out", gas_out)):
        for sp, y in port["y"].items():
            row[f"y_{side}_{sp}"] = y
    return row


def anchor(reactor):

    """Industrial anchor MW and lab baseline metric from the outputs"""

    tag = {"red": "red", "wet": "wet", "dry": "dry"}[reactor]
    with open(os.path.join(OUT, f"_energy_{tag}.json")) as f:
        ind = json.load(f)
    with open(os.path.join(OUT, f"_energy_{tag}_lab.json")) as f:
        lab = json.load(f)

    lab_metric = row_metric(pseudo_row(lab["gas_in"], lab["gas_out"]), reactor)
    ind_row = pseudo_row(ind["gas_in"], ind["gas_out"])
    if reactor == "dry":
        dh = abs(ind["dh_rxn"]["R1"])
        ind_mw = row_metric(ind_row, reactor) * dh / 1e6
    else:
        ind_mw = row_metric(ind_row, reactor) * LHV_H2 / 1e6
    return ind_mw, lab_metric


def main():

    anchors = {rx: anchor(rx) for rx in ("red", "wet", "dry")}
    for rx, (mw, lab_metric) in anchors.items():
        print(f"anchor {rx:4s}  industrial {mw:5.2f} MW   "
              f"lab baseline {lab_metric * 1e3:.3f} mmol/s")
    print()

    for label, path, reactor in SWEEPS:
        ind_mw, lab_metric = anchors[reactor]
        mws = []
        for row in load_rows(path):
            if str(row.get("valid", row.get("validation_ok", True))
                   ).lower() in ("false", "0"):
                continue
            met = row_metric(row, reactor)
            mws.append((float(row["value"]), ind_mw * met / lab_metric))
        if not mws:
            print(f"{label}  (no rows)")
            continue
        lo = min(m for _, m in mws)
        hi = max(m for _, m in mws)
        vs = [v for v, _ in mws]
        print(f"{label} {len(mws):2d} pts   MW {lo:5.2f} -> {hi:5.2f}"
              f"   (value {min(vs):g} .. {max(vs):g})")
        for v, m in mws:
            print(f"    value {v:9.5g}   {m:6.2f} MW")


if __name__ == "__main__":
    main()

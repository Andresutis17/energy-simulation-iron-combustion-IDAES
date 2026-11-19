"""
Dry temperature budget, how much of each utilization change comes from kinetics vs from a hotter bed
"""
import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dry_T_kinetics as ko

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")

Rg = 8.314462618       # gas constant J/(mol K)
UMF = 0.039624         # minimum fluidization velocity m/s
P0 = 1.0e5             # reference pressure Pa
T_GAS_IN0 = 600.0      # baseline gas inlet temperature K
Y_O2_0 = 0.21          # O2 fraction of the baseline feed

LEGS = {
    "hot (T_gas,in 600-1223, fixed flow+y)":
        "sens_dry_T_gas_operating.csv",
    "cold (g 1-1.55, fixed O2 feed)":
        "sens_dry_gasflow_d9_operating.csv",
    "T_solid,in (923-1173.9, fixed gas)":
        "sens_dry_T_solid_operating.csv",
    "T_solid,in overlap (warm ladder)":
        "sens_dry_T_solid_overlap_operating.csv",
}


def fnum(row, key):
    """
    float() that returns None for missing or invalid values
    """
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return None


def leg_rows(csv_name):
    """
    Valid rows of one temperature path
    """
    path = os.path.join(DATA, csv_name)
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("valid") == "True"]
    out = []
    for r in rows:
        u = fnum(r, "O2_utilization")
        ts = fnum(r, "T_solid_out")
        if not u or not ts or u <= 0:
            continue
        out.append({
            "value": fnum(r, "value"),
            "Tg_in": fnum(r, "Tg_in") or fnum(r, "T_gas_in"),
            "Ts_out": ts, "u0_umf": fnum(r, "u_g_over_umf"),
            "y_in_O2": fnum(r, "y_in_O2"),
            "gas_in": fnum(r, "gas_mol_in"),
            "util": u,
        })
    out.sort(key=lambda r: r["Ts_out"])
    return out


def slope_ln_vs_invT(pts):
    """Least squares slope of ln(y) against 1/T"""
    n = len(pts)                                  # Number of points
    if n < 2:
        return None                               # A line needs two points
    xs = [1.0 / p["Ts_out"] for p in pts]         # x: 1/T of each point
    ys = [math.log(p["util"]) for p in pts]       # y: ln of the utilization
    xm, ym = sum(xs) / n, sum(ys) / n             # Averages of x and y
    sxy = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    sxx = sum((x - xm) ** 2 for x in xs)
    return sxy / sxx if sxx else None             # The slope


def main():
    results = {}
    for name, csv_name in LEGS.items():
        pts = leg_rows(csv_name)
        if len(pts) < 2:
            continue
        print(f"\n  {name}  [{csv_name}]")
        print("     value   Tg_in  Ts_out  u0/umf  y_inO2   util%    "
              "dlnU(dT)")
        for p in pts:
            print(f"   {p['value']:8.4g} {p['Tg_in']:7.1f} {p['Ts_out']:7.1f} "
                  f"{p['u0_umf']:7.2f} {p['y_in_O2']:7.4f} "
                  f"{p['util']*100:7.3f}")
        sl = slope_ln_vs_invT(pts)
        e_app = -Rg * sl / 1000.0 if sl is not None else None  # Apparent activation energy kJ/mol
        # Same fit against the velocity ratio
        n = len(pts)
        xm = sum(math.log(p["u0_umf"]) for p in pts) / n   # Average of ln(u0/umf)
        ym = sum(math.log(p["util"]) for p in pts) / n     # Average of ln(util)
        suu = sum((math.log(p["u0_umf"]) - xm) * (math.log(p["util"]) - ym)
                  for p in pts) / n                        # Numerator of the slope
        suu2 = sum((math.log(p["u0_umf"]) - xm) ** 2 for p in pts) / n
        expo = suu / suu2 if suu2 else None                 # Exponent of util  (u0/umf)^expo
        results[csv_name] = {"E_app_kJmol": e_app, "util_vs_u0umf_exp": expo,
                             "n": n}
        if e_app is not None:
            print(f"   E_app = {e_app:+.1f} kJ/mol "
                  f"(d lnU / d(1/T_s,out) = {sl:+.0f} K)")

    # Hot channel split. a = baseline, b = hot end
    print("\nHot leg channel")
    pts = leg_rows(LEGS["hot (T_gas,in 600-1223, fixed flow+y)"])
    a, b = pts[0], pts[-1]
    dlnU = math.log(b["util"] / a["util"])                                # Measured change in ln(utilization)
    cin_a = Y_O2_0 * P0 / (Rg * T_GAS_IN0)                                # Inlet O2 concentration, baseline
    cin_b = Y_O2_0 * P0 / (Rg * b["Tg_in"])                               # Inlet O2 concentration, leg end
    dlnC = math.log(cin_b / cin_a)                                        # The feed dilution (C drops as 1/T)
    # The Arrhenius gain at the local concentration and the feed dilution
    dln_k = math.log(ko.k_chr_of(b["Ts_out"]) / ko.k_chr_of(a["Ts_out"]))  # The Arrhenius gain from the hotter bed
    dlnX_geom = 0.0                     # (1-X)^(2/3) moves approx. 1% over the path
    resid = dlnU - (dln_k + dlnC + dlnX_geom)
    print(f"  d lnU measured            = {dlnU:+.3f}")
    print(f"  d ln k_chr                = {dln_k:+.3f}")
    print(f"  d ln C_in                 = {dlnC:+.3f}")
    print(f"  contact/hydro residue     = {resid:+.3f}")

    # d lnU = 0 needs dln_k = -(dlnC + resid), with the same residue
    e_req = Rg * (-(dlnC + resid + dlnX_geom)) \
        / (1.0 / a["Ts_out"] - 1.0 / b["Ts_out"]) / 1000.0
    results["hot_split"] = {"dlnU": dlnU, "dlnk": dln_k,
                            "dlnC": dlnC, "resid": resid,
                            "E_req_kJmol": e_req}

    # Cold path split 
    print("\nCold leg channel")
    pts = leg_rows(LEGS["cold (g 1-1.55, fixed O2 feed)"])
    a = pts[-1]
    b = next(p for p in pts if abs(p["value"] - 1.55) < 1e-6)
    dlnU = math.log(b["util"] / a["util"])                                # Measured change in ln(utilization)
    cin_a = Y_O2_0 * P0 / (Rg * T_GAS_IN0)                                # Inlet O2 concentration, baseline
    cin_b = fnum({"y_in_O2": b["y_in_O2"]}, "y_in_O2") * P0 / (Rg * T_GAS_IN0)   # Inlet O2 concentration, diluted feed
    dlnC = math.log(cin_b / cin_a)                                        # The feed dilution
    dln_k = math.log(ko.k_chr_of(b["Ts_out"]) / ko.k_chr_of(a["Ts_out"]))
    resid = dlnU - (dln_k + dlnC)
    print(f"  d lnU measured            = {dlnU:+.3f}")
    print(f"  d ln k_chr                = {dln_k:+.3f}")
    print(f"  d ln C_in                 = {dlnC:+.3f}")
    print(f"  contact/hydro residue     = {resid:+.3f}")
    results["cold_split"] = {"dlnU": dlnU, "dlnk": dln_k,
                             "dlnC": dlnC, "resid": resid}

    # T_solid path
    pts = leg_rows(LEGS["T_solid,in (923-1173.9, fixed gas)"])
    dlnU = math.log(pts[-1]["util"] / pts[0]["util"])
    dTs = pts[-1]["Ts_out"] - pts[0]["Ts_out"]
    dlnk = math.log(ko.k_chr_of(pts[-1]["Ts_out"])
                    / ko.k_chr_of(pts[0]["Ts_out"]))
    print("\nT_solid channel")
    print(f"  T_s,out {pts[0]['Ts_out']:.1f}->{pts[-1]['Ts_out']:.1f} K "
          f"({dTs:+.1f} K), d lnU = {dlnU:+.4f} "
          f"({(dlnU / dTs) * 100:+.2f} pp/100K)")
    results["tsolid_leg"] = {"dlnU": dlnU, "dlnk": dlnk,
                             "dTs": dTs}

    # Contact curves from the axials
    print("Kbe / Contact curves from axials")
    axial = {}
    for tag, fname in [("base(dp028)", "axial_dry_lab_r8.csv"),
                       ("tg1223", "axial_dry_lab_dp028_tg1223.csv"),
                       ("g155", "axial_dry_lab_dp028_g155.csv")]:
        path = os.path.join(DATA, fname)
        if not os.path.exists(path):
            continue
        rows = ko.load_axial(path)
        r = rows[len(rows) // 2]
        T = ko.fnum(r, "T_gas")
        Pe = ko.fnum(r, "P_emul")
        db = ko.fnum(r, "db")
        dab = ko.D_ab_fuller(T, Pe)
        kbe = ko.Kbe_of(db, UMF, dab)
        A = math.pi * (0.054 / 2.0) ** 2              # Bed cross sectional area m2
        C_e = P0 and Pe / (Rg * T)                    # Emulsion gas concentration mol/m3
        ue = ko.fnum(r, "F_emul") / (A * C_e)         # Gas velocity in the emulsion m/s
        ub = ko.fnum(r, "F_bub") / (A * ko.fnum(r, "delta")
                                    * (Pe / (Rg * ko.fnum(r, "T_bub"))))   # Gas velocity in the bubbles m/s
        tau_b = 0.9 / ub                              # Bubble residence time over the bed s
        axial[tag] = {"db_mid": db, "Kbe_mid": kbe, "u_e_mid": ue,
                      "u_b_mid": ub, "tau_bub_mid_s": tau_b,
                      "delta_mid": ko.fnum(r, "delta")}
        print(f"  {tag:12s} db_mid={db*1000:6.1f} mm  Kbe_mid={kbe:7.3f} 1/s"
              f"  u_e={ue:6.4f}  u_b={ub:6.3f} m/s  tau_bub={tau_b:5.2f} s"
              f"  delta={ko.fnum(r, 'delta'):.3f}")

    with open(os.path.join(DATA, "op_dry_budget.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()

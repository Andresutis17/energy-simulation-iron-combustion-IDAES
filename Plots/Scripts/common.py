"""
Helpers for all the plot scripts: paths, style, loaders and
the quantity formulas 

"""

import json
import os


# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(os.path.dirname(HERE), "data")
FIGS = os.path.join(os.path.dirname(HERE), "Figs")
MODELS = os.path.join(REPO, "Reliable_models")

# Path to the lab models
LAB_SCRIPTS = {
    "reduction": os.path.join(MODELS, "Lab", "Reduction_Reactor_Lab.py"),
    "wet": os.path.join(MODELS, "Lab", "Wet_Oxidation_Reactor_Lab.py"),
    "dry": os.path.join(MODELS, "Lab", "Dry_Oxidation_Reactor_Lab.py"),
}

# Path to the industrial models
IND_SCRIPTS = {
    "reduction": os.path.join(MODELS, "Industrial", "Reduction_Reactor.py"),
    "wet": os.path.join(MODELS, "Industrial", "Wet_Oxidation_Reactor.py"),
    "dry": os.path.join(MODELS, "Industrial", "Dry_Oxidation_Reactor.py"),
}

# Lab operating points that match the industrial reactors
LAB_MATCH = {
    "reduction": {"H": 0.90, "n_orifice": 2500}, 
    "wet": {"H": 0.70, "n_orifice": 2500},
    "dry": {"H": 0.90, "n_orifice": 2500},
}

# PLOT_ERA=_dp028 
SUFFIX = os.environ.get("PLOT_ERA", "")


def axial_csv(reactor, scale, suffix=None):
    """
    Path to the axial profiles CSV file

    """

    return os.path.join(DATA, f"axial_{reactor}_{scale}{suffix or ''}.csv")


def axial_meta(reactor, scale, suffix=None):
    """
    Path to the metadata JSON file for axial profiles
    """
    return os.path.join(DATA, f"axial_{reactor}_{scale}{suffix or ''}.meta.json")


def ensure_dirs():
    """
    Create DATA and FIGS directories if they are missing
    """
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)


# Plots whose y axis is a conversion X 
CONVERSION_PREFIXES = ("S", "T1a", "T1b", "T1c", "T2a", "T2b", "T2c",
                       "B_conversion")


def save_fig(fig, name):

    """
    One png per figure, routed to Figs/Conversion when the y axis
    is a conversion X
    """
    ensure_dirs()
    out = os.path.join(FIGS, "Conversion"
                       if name.startswith(CONVERSION_PREFIXES) else "")
    os.makedirs(out, exist_ok=True)
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=300,
                bbox_inches="tight")


def load_axial(reactor, scale, suffix=None):
    """
    Return rows and meta for one solved case

    """
    import csv
    sfx = SUFFIX if suffix is None else suffix
    with open(axial_csv(reactor, scale, sfx), newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            r[k] = float(r[k]) if r[k] != "" else None
    with open(axial_meta(reactor, scale, sfx)) as f:
        meta = json.load(f)
    return rows, meta


def load_csv(name):
    """
    Reads csv
    """
    import csv
    with open(os.path.join(DATA, name), newline="") as f:
        return list(csv.DictReader(f))


# Colors used in the plots
COLORS = ["#006df2", "#FF0D00", "#229801", "#9801CF"]
LSTYLES = ["-", "--", "-.", ":"]
INK = "#0b0b0b"
INK2 = "#5c5c5b"
MUTED = "#7B7A7A"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# The outlet figures. Solid curves are mass fractions w, gas curves are mole fractions y
OUTLET_LBL = "Outlet fraction [%]\n(solid: mass, gas: molar)"


def apply_style():

    """
    One style in common for every plot
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "axes.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def cols(rows, key):
    """
    Pulls one column out of the dict as a plain list
    """
    return [r[key] for r in rows]


# Solid molar masses in kg/mol and how many Fe atoms each species has
MW_S = {"Fe2O3": 0.15969, "Fe3O4": 0.231533, "FeO": 0.071844,
        "Fe": 0.055845, "Al2O3": 0.10196}
N_FE = {"Fe2O3": 2, "Fe3O4": 3, "FeO": 1, "Fe": 1}
FE_SP = ["Fe2O3", "Fe3O4", "FeO", "Fe"]


def x_prod_from_w(reactor, mass_frac):

    """
    Solid conversion to the last product of the chain, from mass fractions
    """
    fe_mol = {sp: mass_frac[sp] / MW_S[sp] * N_FE[sp] for sp in FE_SP}
    tot = sum(fe_mol.values())
    if tot <= 1e-12:
        return None
    if reactor == "reduction":
        fe_prod = fe_mol["Fe"]
    elif reactor == "wet":
        fe_prod = fe_mol["Fe3O4"]
    else:
        fe_prod = fe_mol["Fe2O3"]
    return 100.0 * fe_prod / tot


def x_stages_from_w(reactor, mass_frac):

    """
    The chain links before the product, as acumulative Fe conversions
    on the same basis as x_prod_from_w
    """
    fe_mol = {sp: mass_frac[sp] / MW_S[sp] * N_FE[sp] for sp in FE_SP}
    tot = sum(fe_mol.values())
    if tot <= 1e-12:
        return {}
    if reactor == "reduction":
        return {"fe23": 100.0 * (tot - fe_mol["Fe2O3"]) / tot,
                "fe34": 100.0 * (fe_mol["FeO"] + fe_mol["Fe"]) / tot}
    if reactor == "wet":
        return {"feox": 100.0 * (fe_mol["FeO"] + fe_mol["Fe3O4"]) / tot}
    return {}


# Gas species used 
GAS_SP = ["O2", "N2", "CO2", "H2O", "H2"]
MW_G = {"O2": 0.032, "N2": 0.028, "CO2": 0.044,
        "H2O": 0.018, "H2": 0.002016}


def _safe_value(obj, default=None):
    """
    Read a Pyomo value 
    """
    try:
        from pyomo.environ import value
        return float(value(obj))
    except Exception:
        try:
            return float(obj.value)
        except Exception:
            return default


def _component_values(indexed, species, prefix=()):
    return {sp: _safe_value(indexed[prefix + (sp,)], 0.0)
            for sp in species}


def gas_component_flows(flow_mol, mole_frac):
    """
    Molar component flows mol/s from total molar flow and mole fractions
    """
    return {sp: flow_mol * mole_frac.get(sp, 0.0) for sp in GAS_SP}




def fe_atom_flow(flow_mass, mass_frac):
    """
    Total Fe atom flow in the solid mol Fe/s
    """
    return sum(flow_mass * mass_frac.get(sp, 0.0) / MW_S[sp] * N_FE.get(sp, 0.0)
               for sp in FE_SP)


def removable_oxygen_flow(flow_mass, mass_frac):
    """
    O atoms removable by H2 to metallic Fe mol O/s
    """
    n_o = {"Fe2O3": 3.0, "Fe3O4": 4.0, "FeO": 1.0, "Fe": 0.0,
           "Al2O3": 0.0}
    return sum(flow_mass * mass_frac.get(sp, 0.0) / MW_S[sp] * n_o[sp]
               for sp in n_o)


# Oxygen atoms per Fe atom of each oxide
O_PER_FE = {"Fe2O3": 1.5, "Fe3O4": 4.0 / 3.0, "FeO": 1.0, "Fe": 0.0}


def dof_from_w(mass_frac):

    """
    Degree of oxidation of the solid on the Fe basis,
    (O/Fe)_out / (O/Fe)_Fe2O3 from 0 to 1
    """
    fe_mol = {sp: mass_frac[sp] / MW_S[sp] * N_FE[sp] for sp in FE_SP}
    tot = sum(fe_mol.values())
    if tot <= 1e-12:
        return None
    o_fe = sum(fe_mol[sp] * O_PER_FE[sp] for sp in FE_SP) / tot
    return o_fe / O_PER_FE["Fe2O3"]


def _superficial_gas_velocity(b):
    """
    The beds superficial gas velocity m/s. The models own value
    when the block computes it, else inlet volumetric flow / area
    """
    if hasattr(b, "velocity_superficial_gas"):
        try:
            return _safe_value(b.velocity_superficial_gas[0])
        except Exception:
            pass
    import math
    flow = _safe_value(b.gas_inlet.flow_mol[0], 0.0)
    temp = _safe_value(b.gas_inlet.temperature[0], 0.0)
    press = _safe_value(b.gas_inlet.pressure[0], 1.0)
    diam = _safe_value(b.bed_diameter, 0.0)
    area = math.pi * diam ** 2 / 4.0 if diam > 0 else 0.0
    return flow * 8.314 * temp / press / area if area > 0 else None


def extract_operating_kpis(m, reactor):
    """
    Reads the solved models report numbers: conversions, utilizations,
    lambda, flows and outlet temperatures, from the inlet and outlet ports
    """
    b = m.fs.BFB
    gas_species = list(m.fs.gas_properties.component_list)
    solid_species = list(m.fs.solid_properties.component_list)

    gas_in = _safe_value(b.gas_inlet.flow_mol[0], 0.0)
    gas_out = _safe_value(b.gas_outlet.flow_mol[0], 0.0)
    solid_in = _safe_value(b.solid_inlet.flow_mass[0], 0.0)
    solid_out = _safe_value(b.solid_outlet.flow_mass[0], 0.0)
    y_in = _component_values(b.gas_inlet.mole_frac_comp, gas_species, (0,))
    y_out = _component_values(b.gas_outlet.mole_frac_comp, gas_species, (0,))
    w_in = _component_values(b.solid_inlet.mass_frac_comp, solid_species, (0,))
    w_out = _component_values(b.solid_outlet.mass_frac_comp, solid_species, (0,))
    n_in = gas_component_flows(gas_in, y_in)
    n_out = gas_component_flows(gas_out, y_out)

    out = {
        "gas_mol_in": gas_in,
        "gas_mol_out": gas_out,
        "solid_mass_in": solid_in,
        "solid_mass_out": solid_out,
        "T_gas_in": _safe_value(b.gas_inlet.temperature[0]),
        "T_gas_out": _safe_value(b.gas_outlet.temperature[0]),
        "T_solid_in": _safe_value(b.solid_inlet.temperature[0]),
        "T_solid_out": _safe_value(b.solid_outlet.temperature[0]),
    }
    out["deltaT_gas"] = out["T_gas_out"] - out["T_gas_in"]
    out["deltaT_solid"] = out["T_solid_out"] - out["T_solid_in"]
    out["y_in_sum"] = sum(y_in.values())
    out["y_out_sum"] = sum(y_out.values())
    out["w_in_sum"] = sum(w_in.values())
    out["w_out_sum"] = sum(w_out.values())
    for sp in gas_species:
        out[f"y_in_{sp}"] = y_in[sp]
        out[f"y_out_{sp}"] = y_out[sp]
        out[f"n_in_{sp}"] = n_in.get(sp, 0.0)
        out[f"n_out_{sp}"] = n_out.get(sp, 0.0)
    for sp in solid_species:
        out[f"w_in_{sp}"] = w_in[sp]
        out[f"w_out_{sp}"] = w_out[sp]

    u_g = _superficial_gas_velocity(b)
    u_mf = _safe_value(m.fs.solid_properties.velocity_mf)
    out["velocity_superficial_gas"] = u_g
    out["velocity_mf"] = u_mf
    out["u_g_over_umf"] = u_g / u_mf if u_g is not None and u_mf and u_mf > 0 else None
    if bool(getattr(b.config, "has_pressure_change", False)):
        out["deltaP"] = (_safe_value(b.gas_inlet.pressure[0])
                         - _safe_value(b.gas_outlet.pressure[0]))

    fe_in = fe_atom_flow(solid_in, w_in)
    fe_out = fe_atom_flow(solid_out, w_out)
    if reactor == "dry":
        n_fe_metal_in = solid_in * w_in.get("Fe", 0.0) / MW_S["Fe"]
        n_fe_metal_out = solid_out * w_out.get("Fe", 0.0) / MW_S["Fe"]
        stoich_o2 = 0.75 * n_fe_metal_in
        out["lambda_O2"] = n_in["O2"] / stoich_o2 if stoich_o2 > 0 else None
        out["lambda_stoich"] = out["lambda_O2"]
        out["O2_utilization"] = ((n_in["O2"] - n_out["O2"]) / n_in["O2"]
                                  if n_in["O2"] > 0 else None)
        out["reactant_utilization"] = out["O2_utilization"]
        out["Fe_conversion"] = ((n_fe_metal_in - n_fe_metal_out) / n_fe_metal_in
                                 if n_fe_metal_in > 0 else None)
    elif reactor == "wet":
        n_fe_metal_in = solid_in * w_in.get("Fe", 0.0) / MW_S["Fe"]
        n_fe_metal_out = solid_out * w_out.get("Fe", 0.0) / MW_S["Fe"]
        stoich_h2o = (4.0 / 3.0) * n_fe_metal_in
        out["lambda_steam"] = n_in["H2O"] / stoich_h2o if stoich_h2o > 0 else None
        out["lambda_stoich"] = out["lambda_steam"]
        out["steam_utilization"] = ((n_in["H2O"] - n_out["H2O"]) / n_in["H2O"]
                                     if n_in["H2O"] > 0 else None)
        out["reactant_utilization"] = out["steam_utilization"]
        out["net_H2_production"] = n_out["H2"] - n_in["H2"]
        out["Fe_conversion"] = ((n_fe_metal_in - n_fe_metal_out) / n_fe_metal_in
                                 if n_fe_metal_in > 0 else None)
    elif reactor == "reduction":
        o_rem_in = removable_oxygen_flow(solid_in, w_in)
        o_rem_out = removable_oxygen_flow(solid_out, w_out)
        out["lambda_H2"] = n_in["H2"] / o_rem_in if o_rem_in > 0 else None
        out["lambda_stoich"] = out["lambda_H2"]
        out["H2_utilization"] = ((n_in["H2"] - n_out["H2"]) / n_in["H2"]
                                  if n_in["H2"] > 0 else None)
        out["reactant_utilization"] = out["H2_utilization"]
        out["net_H2O_production"] = n_out["H2O"] - n_in["H2O"]
        out["reduction_degree"] = ((o_rem_in - o_rem_out) / o_rem_in
                                    if o_rem_in > 0 else None)
        out["metallization_degree"] = ((solid_out * w_out.get("Fe", 0.0) / MW_S["Fe"])
                                        / fe_out if fe_out > 0 else None)
        fe2o3_in = solid_in * w_in.get("Fe2O3", 0.0) / MW_S["Fe2O3"]
        fe2o3_out = solid_out * w_out.get("Fe2O3", 0.0) / MW_S["Fe2O3"]
        out["Fe2O3_conversion"] = ((fe2o3_in - fe2o3_out) / fe2o3_in
                                    if fe2o3_in > 0 else None)
    return out


# Emulsion voidage
EPS_MF = 0.45


def tau_profile(rows, meta):

    """
    Mean solids residence time and the tau(z) profile
    The solids live in the emulsion region only,
    so the mean residence time is that inventory over the solids feed.
    u_s from the model is a superficial velocity over the cross
    section u_s * bed_area * dens_mass_particle = flow_mass

    """
    from math import pi
    zs = [float(r["z"]) for r in rows]
    de = [float(r["delta_e"]) for r in rows]
    rho = [float(r["dens_mass_particle"]) for r in rows]
    area = pi * float(meta["D"]) ** 2 / 4.0
    mdot = float(meta["solid_mass_in"])
    eps_e = float(meta.get("voidage_mf", EPS_MF))
    hold = [area * d * (1.0 - eps_e) * r for d, r in zip(de, rho)]
    tau_cum = [0.0]
    for i in range(1, len(zs)):
        tau_cum.append(tau_cum[-1] + 0.5 * (hold[i] + hold[i - 1])
                       * (zs[i] - zs[i - 1]) / mdot)
    tau_mean = tau_cum[-1]
    return tau_mean / 60.0, zs, [t / 60.0 for t in tau_cum]


def stack(nrows, w=3.4, row_h=2.2):

    """
    Multiplot with one shared x axis
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows, 1, figsize=(w, row_h * nrows + 0.3),
                             sharex=True)
    return fig, axes


def legend_band(fig, ncol=3, handles=None, bottom=0.18):

    """
    Figure level legend in the bottom band under the axes
    """
    if handles is None:
        handles, _ = fig.axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, bottom, 1, 1])
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=ncol, columnspacing=1.2, handlelength=1.8)


if __name__ == "__main__":
    # PLOT_ERA=_dp028 
    #rx: reactor
    #lo: low limit
    #hi: high limit
    #rws: rows
    #mt: metadata
    for rx, lo, hi in (("reduction", 40.0, 48.0), ("wet", 55.0, 63.0),
                       ("dry", 42.0, 50.0)):
        rws, mt = load_axial(rx, "lab")
        tm, _, _ = tau_profile(rws, mt)
        assert lo <= tm <= hi, (rx, tm)

    # Certified reduction anchor outlet, this mix holds all four
    # species, so it contains the whole O/Fe table. DoO here: 0.552
    base = {"Fe2O3": 0.012857024703548237, "Fe3O4": 0.5322702616185936,
            "FeO": 0.1834995083404017, "Fe": 0.271373205337458}
    assert 0.50 < dof_from_w(base) < 0.60

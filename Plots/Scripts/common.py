

"""
Shared paths and figure-saving for the plots

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
    "reduction": {"H": 1, "n_orifice": 2500}, 
    "wet": {"H": 0.70, "n_orifice": 2500},        
    "dry": {"H": 0.90, "n_orifice": 2500},        
}

# Conversion reports
REPORT_KEYS = {
    "reduction": ("X_Fe2O3", "X_H2"),
    "wet": ("X_Fe", "X_H2O"),
    "dry": ("X_Fe", "X_O2"),
}


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


def save_fig(fig, name):

    """
    One png per figure
    """
    ensure_dirs()
    fig.savefig(os.path.join(FIGS, f"{name}.png"), dpi=300,
                bbox_inches="tight")


def load_axial(reactor, scale, suffix=None):
    """
    Return rows and meta for one solved case
    """
    import csv
    with open(axial_csv(reactor, scale, suffix), newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            r[k] = float(r[k]) if r[k] != "" else None
    with open(axial_meta(reactor, scale, suffix)) as f:
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

# Skeletal densities in kg/m3
RHO_SKELETAL = {"Fe2O3": 5250.0, "Fe3O4": 5170.0, "FeO": 5700.0,
                "Fe": 7874.0, "Al2O3": 3987.0}


def x_prod_from_w(reactor, mass_frac):

    """
    Product basis solid conversion from mass fractions
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


def x_prod_col(reactor, rows):

    """
    Conversion of every row, recomputed when the file has the data
    """


    if all(f"w_{sp}" in rows[0] for sp in FE_SP):
        return [x_prod_from_w(reactor, {sp: row[f"w_{sp}"] for sp in FE_SP})
                for row in rows]
    return [row.get("X_prod") for row in rows]

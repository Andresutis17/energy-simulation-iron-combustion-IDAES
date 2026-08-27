

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


def axial_csv(reactor, scale):
    """
    Path to the axial profiles CSV file

    """

    return os.path.join(DATA, f"axial_{reactor}_{scale}.csv")


def axial_meta(reactor, scale):
    """
    Path to the metadata JSON file for axial profiles
    """
    return os.path.join(DATA, f"axial_{reactor}_{scale}.meta.json")


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


def load_axial(reactor, scale):
    """
    Return rows and meta for one solved case
    """
    import csv
    with open(axial_csv(reactor, scale), newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in r:
            r[k] = float(r[k])
    with open(axial_meta(reactor, scale)) as f:
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

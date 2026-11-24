"""
Shared paths for the plots

"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
FIGS = os.path.join(HERE, "figs")
MODELS_LAB = os.path.join(REPO, "Reliable_models", "Lab")
MODELS_IND = os.path.join(REPO, "Reliable_models", "Industrial")


LAB_SCRIPTS = {
    "reduction": os.path.join(MODELS_LAB, "Reduction_Reactor_Lab.py"),
    "wet": os.path.join(MODELS_LAB, "Wet_Oxidation_Reactor_Lab.py"),
    "dry": os.path.join(MODELS_LAB, "Dry_Oxidation_Reactor_Lab.py"),
}
IND_SCRIPTS = {
    "reduction": os.path.join(MODELS_IND, "Reduction_Reactor.py"),
    "wet": os.path.join(MODELS_IND, "Wet_Oxidation_Reactor.py"),
    "dry": os.path.join(MODELS_IND, "Dry_Oxidation_Reactor.py"),
}

# Lab verified operating points that match the industrial reactors 
LAB_MATCH = {
    "reduction": {"H": 0.90, "n_orifice": 2500},  
    "wet": {"H": 0.70, "n_orifice": 2500},        
    "dry": {"H": 0.90, "n_orifice": 2500},        
}

REPORT_KEYS = {
    "reduction": ("X_Fe2O3", "X_H2"),
    "wet": ("X_Fe", "X_H2O"),
    "dry": ("X_Fe", "X_O2"),
}


def axial_csv(reactor, scale):
    return os.path.join(DATA, f"axial_{reactor}_{scale}.csv")


def axial_meta(reactor, scale):
    return os.path.join(DATA, f"axial_{reactor}_{scale}.meta.json")


def ensure_dirs():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)


def save_fig(fig, name):
    ensure_dirs()
    fig.savefig(os.path.join(FIGS, f"{name}.png"), dpi=300,
                bbox_inches="tight")


def load_axial(reactor, scale):
    """
    Return rows, meta for one solved case, rows are dicts keyed by column
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
    import csv
    with open(os.path.join(DATA, name), newline="") as f:
        return list(csv.DictReader(f))


COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LSTYLES = ["-", "--", "-.", ":"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def apply_style():
    """
    One style for all figures
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
    return [r[key] for r in rows]

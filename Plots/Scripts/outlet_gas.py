"""
Outlet gas curves + Gamma oxidation grade from X_gas

"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  

MW_G = {"H2": 2.016e-3, "H2O": 18.015e-3, "N2": 28.014e-3,
        "O2": 31.998e-3}
# Mole fraction floor so a fully swept out product still draws a curve
FLOOR = 1e-5
KNOW_Y = {"reduction": "y_H2", "wet": "y_H2O", "dry": "y_O2"}
BASE_Y = {"reduction": 0.99, "wet": 0.99, "dry": 0.21}

# The gas species the figures draw reactant first, product second
GAS_SP = {"reduction": ["H2", "H2O"], "wet": ["H2O", "H2"],
          "dry": ["O2"]}

GAS_PROD = "#762a83"


def y_in_of(reactor, knob, value):
    """
    Feed mole fraction of the reacting gas. In the y knob sweeps
    the row value is the feed
    """
    if knob == KNOW_Y[reactor]:
        return float(value)
    return BASE_Y[reactor]


def y_out(reactor, x_gas, y_in):
    """
    Outlet mole fractions from X_gas [%]
    """
    x = x_gas / 100.0
    if reactor == "dry":
        den = 1.0 - y_in * x
        return {"O2": y_in * (1.0 - x) / den,
                "N2": (1.0 - y_in) / den}
    if reactor == "reduction":
        return {"H2": y_in * (1.0 - x),
                "H2O": FLOOR + y_in * x,
                "N2": 1.0 - y_in}
    return {"H2O": y_in * (1.0 - x),
            "H2": FLOOR + y_in * x,
            "N2": 1.0 - y_in}


def gas_ws(reactor, x_gas, y_in):
    """
    Outlet gas mass fractions on the same basis as the
    solid w_out_* columns
    """
    y = y_out(reactor, x_gas, y_in)
    tot = sum(yi * MW_G[sp] for sp, yi in y.items())
    w = {sp: 100.0 * yi * MW_G[sp] / tot for sp, yi in y.items()}
    assert abs(sum(w.values()) - 100.0) < 1e-6
    return w


def row_gas_ws(reactor, knob, row):
    """
    Gas_ws for one sens CSV row
    """
    xg = row.get("X_gas")
    if xg in (None, ""):
        return {}
    return gas_ws(reactor, float(xg), y_in_of(reactor, knob, row["value"]))


def row_gamma(reactor, knob, row):
    """
    Gas oxidation degree y_H2O/(y_H2O + y_H2). Reduction is
    X_gas/100, wet is 1 - X_gas/100 
    """
    if reactor == "dry":
        return None
    xg = row.get("X_gas")
    if xg in (None, ""):
        return None
    y = y_out(reactor, float(xg), y_in_of(reactor, knob, row["value"]))
    return y["H2O"] / (y["H2O"] + y["H2"])


def add_gas(ax, pairs, reactor, knob):
    """
    Draws the reacting gas species over pairs 
    """
    hs = []
    for j, sp in enumerate(GAS_SP[reactor]):
        color = common.INK if j == 0 else GAS_PROD
        xy = sorted((x, row_gas_ws(reactor, knob, row)[sp])
                    for x, row in pairs
                    if row.get("X_gas") not in (None, ""))
        if not xy or max(v for _, v in xy) < 1e-4:
            continue
        ax.plot([a for a, _ in xy], [b for _, b in xy],
                color=color, ls="-", lw=1.2)
        hs.append(ax.plot([], [], color=color, ls="-", lw=1.2,
                          label=rf"$w_{{{sp}}}$ (gas)")[0])
    return hs


def add_gamma(ax, pairs, reactor, knob, color="#d95f02"):
    """Gas oxidation degree on right axis"""
    ax2 = ax.twinx()
    xy = sorted((x, g) for x, row in pairs
                if (g := row_gamma(reactor, knob, row)) is not None)
    if xy:
        ax2.plot([a for a, _ in xy], [b for _, b in xy],
                 color=color, ls="--", lw=1.2)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel(r"$\Gamma = y_{H_2O}/(y_{H_2}+y_{H_2O})$ [-]",
                   color=color)
    ax2.tick_params(axis="y", colors=color)
    ax2.spines["right"].set_color(color)
    ax2.grid(False)
    return ax2


if __name__ == "__main__":
    w = gas_ws("reduction", 28.311, 0.99)
    assert abs(w["H2"] - 21.17) < 0.02, w
    assert abs(w["H2O"] - 74.68) < 0.02, w
    assert abs(row_gamma("reduction", "y_H2",
                         {"value": 0.99, "X_gas": 28.311})
               - 0.28311) < 1e-4
    g = row_gamma("wet", "y_H2O", {"value": 0.99, "X_gas": 26.74})
    assert abs(g - 0.7326) < 1e-4, g
    wd = gas_ws("dry", 13.85, 0.21)
    assert abs(wd["O2"] - 20.73) < 0.02, wd
    wn = gas_ws("dry", 13.85, 0.21)
    y = y_out("dry", 13.85, 0.21)
    assert abs(y["O2"] / y["N2"] - 0.21 * (1 - 0.1385) / 0.79) < 1e-12
    print("outlet_gas self-test OK")
    print(f"  red base  w: H2 {w['H2']:.2f} H2O {w['H2O']:.2f} "
          f"N2 {w['N2']:.2f}  Gamma {0.28311:.5f}")
    print(f"  wet base  Gamma {g:.5f}")
    print(f"  dry base  w: O2 {wd['O2']:.2f} N2 {wd['N2']:.2f}")

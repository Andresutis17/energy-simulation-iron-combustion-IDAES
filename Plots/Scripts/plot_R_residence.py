"""
Residence time of every solid species in the lab beds 

tau_i = axial inventory of i / outlet flow of i, with
inventory M_i(z) = A * (1-delta) * (1-voidage_mf) * w_i(z) * rho_p(z)
"""
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy
from matplotlib.ticker import FuncFormatter, NullFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

VOIDAGE_MF = 0.45  # emulsion voidage at minimum fluidization, lab constant

# Every solid species
SPECIES = [("Fe2O3", r"$Fe_2O_3$"), ("Fe3O4", r"$Fe_3O_4$"),
           ("FeO", r"$FeO$"), ("Fe", r"$Fe$"), ("Al2O3", r"$Al_2O_3$")]

# Used reactors
REACTORS = [("reduction", "Reduction", "R1a_residence_reduction"),
            ("wet", "Wet Oxidation", "R1b_residence_wet"),
            ("dry", "Dry Oxidation", "R1c_residence_dry")]


def main():

    """
    One bar figure per reactor
    """
    common.apply_style()
    for reactor, title, stem in REACTORS:
        rows, meta = common.load_axial(reactor, "lab")
        if "u_solid" not in rows[0] or "solid_mass_out" not in meta:
            sys.exit(f"{reactor}: residence columns missing")
        area = math.pi * meta["D"] ** 2 / 4.0
        u_s = rows[0]["u_solid"]
        tau_hydro = meta["H"] / u_s

        z = common.cols(rows, "z")
        taus = {}
        for key, _ in SPECIES:
            kg_per_m3 = [(1.0 - row["delta"]) * (1.0 - VOIDAGE_MF)
                      * row[f"w_{key}"] * row["dens_mass_particle"]
                      for row in rows]
            m_i = area * numpy.trapezoid(kg_per_m3, z)
            # Outlet composition = last axial row 
            mdot_i = meta["solid_mass_out"] * rows[-1][f"w_{key}"]
            # Absent species = exact zero 
            taus[key] = (m_i / mdot_i
                         if mdot_i > meta["solid_mass_out"] * 1e-6 else None)

        # Absent species get no bar
        plot_species = [(k, lbl) for k, lbl in SPECIES if taus[k] is not None]


        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        keys = [k for k, _ in plot_species]
        vals = [taus[k] / 60.0 for k in keys]
        y = range(len(keys))
        ax.barh(list(y), vals, color=common.COLORS[:len(keys)], height=0.62)
        ax.set_yticks(list(y))
        ax.set_yticklabels([lbl for _, lbl in plot_species])
        ax.invert_yaxis()
        ax.set_xlabel(r"$\tau_i$ [min]")
        ax.set_title(title)
        ax.grid(True, axis="x")
        if max(vals) / min(vals) > 10:
            ax.set_xscale("log")
            lo = math.floor(math.log10(min(vals)))
            hi = math.ceil(math.log10(max(vals)))
            ax.set_xticks([10.0 ** k for k in range(lo, hi + 1)])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(NullFormatter())
        fig.tight_layout()
        common.save_fig(fig, stem)
        plt.close(fig)


if __name__ == "__main__":
    main()

"""
Scale validation. Lab vs industrial axial profiles on the normalized
bed height z/H for all three reactors


"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common 

# The reactors used for these plots
REACTORS = [
    ("reduction", r"(a) Reduction", r"$X_{Fe_2O_3}$"),
    ("wet", r"(b) Wet Oxidation", r"$X_{Fe}$"),
    ("dry", r"(c) Dry Oxidation", r"$X_{Fe}$"),
]


def main():
    common.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6), sharex=True)

    legend_lines = None
    for j, (reactor, title, conv) in enumerate(REACTORS):
        lab_rows, lab_meta = common.load_axial(reactor, "lab")
        ind_rows, ind_meta = common.load_axial(reactor, "ind")

        # Solid conversion vs normalized height
        ax = axes[j]
        height_norm = [r["x_norm"] for r in lab_rows]
        (lab_line,) = ax.plot(height_norm, common.cols(lab_rows, "X_solid"),
                        color=common.COLORS[0], ls=common.LSTYLES[0],
                        label=f"Lab (D=0.054 m, H={lab_meta['H']:.2f} m)")
        (ind_line,) = ax.plot(height_norm, common.cols(ind_rows, "X_solid"),
                        color=common.COLORS[1], ls=common.LSTYLES[1],
                        label=f"Industrial (D=6.5 m, H={ind_meta['H']:.1f} m)")
        if legend_lines is None:
            legend_lines = (lab_line, ind_line)
        ax.set_title(title)
        ax.set_ylabel(f"{conv} [%]")
        ax.set_xlabel("z/H [-]")
        ax.set_ylim(0, 105 if reactor == "reduction" else None)
        ax.grid(True)

    fig.legend(handles=legend_lines, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    common.save_fig(fig, "B_conversion_overlay")


if __name__ == "__main__":
    main()

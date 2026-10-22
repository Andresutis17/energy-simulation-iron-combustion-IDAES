"""
Local solid reaction rates along the bed

"""
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

REACTORS = [
    ("reduction", "R8a", "Reduction",
     [("rate_RD1", r"RR$_1$: Fe$_2$O$_3\to$Fe$_3$O$_4$"),
      ("rate_RD2", r"RR$_2$: Fe$_3$O$_4\to$FeO"),
      ("rate_RD3", r"RR$_3$: FeO$\to$Fe")]),
    ("wet", "R8b", "Wet Oxidation",
     [("rate_OW1", r"WOR$_1$: Fe$\to$FeO"),
      ("rate_OW2", r"WOR$_2$: FeO$\to$Fe$_3$O$_4$")]),
    ("dry", "R8c", "Dry Oxidation",
     [("rate_R1", r"DOR: Fe$\to$Fe$_2$O$_3$")]),
]


def main():
    common.apply_style()

    for reactor, stem, name, rxns in REACTORS:
        try:
            rows, meta = common.load_axial(reactor, "lab", suffix="_r8")
        except FileNotFoundError:
            sys.exit(f"missing axial_{reactor}_lab_r8.csv")
        if not meta.get("valid", True):
            sys.exit(f" {reactor} r8 axial is not valid")
        if rxns[0][0] not in rows[0]:
            sys.exit(f" {reactor} r8 axial has no rate columns")
        xs = [r["x_norm"] for r in rows]

        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        for i, (col, lbl) in enumerate(rxns):
            ys = [r[col] for r in rows]
            ax.plot(xs, ys, color=common.COLORS[i], lw=1.2, label=lbl)

        ax.set_title(f"{name}: local reaction rates\nalong the bed")
        ax.set_xlabel("z/H [-]")

        # The rate is per m3 of solid particles  not per m3 of bed.
        # The bed balance multiplies it by the solid actually held in the emulsion
        ax.set_ylabel(r"$r$ [mol m$^{-3}_\mathrm{solid}$ s$^{-1}$]")
        ax.set_xlim(0.0, 1.0)
        ax.set_yscale("log")
        ax.grid(True)
        fig.tight_layout(rect=[0, 0.18, 1, 1])
        fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.015),
                   ncol=2, columnspacing=1.2, handlelength=1.8)
        common.save_fig(fig, stem)
        plt.close(fig)


if __name__ == "__main__":
    main()

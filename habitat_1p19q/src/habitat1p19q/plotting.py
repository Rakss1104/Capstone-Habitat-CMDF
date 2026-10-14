"""Plot style: one colour per class / cohort / model, fixed order, used identically in every figure."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_ORDER = ["M0", "M1_clin", "M1", "M2", "M3", "M4", "M5"]


def apply_style():
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.6,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#2b2b2a", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "font.size": 10, "legend.frameon": False, "lines.linewidth": 2})


class Palette:
    def __init__(self, cfg):
        c = cfg["colors"]
        self.cls = {0: c["non-codeleted"], 1: c["codeleted"]}
        self.cohort = {"UTSW": c["UTSW"], "UCSF": c["UCSF"]}
        self.model = dict(zip(MODEL_ORDER, c["models"]))
        self.series = c["models"]
        self.codeleted = c["codeleted"]

    def model_style(self, m):
        # Colour follows the model, never its position in a list; "+cohort" variants reuse the base colour, dashed.
        return {"color": self.model[m.replace("+cohort", "")], "ls": "--" if m.endswith("+cohort") else "-"}

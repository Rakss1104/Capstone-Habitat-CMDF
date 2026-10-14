"""Output handling: tables, figures, console reporting, the check registry and environment capture."""
from __future__ import annotations

import hashlib
import importlib
import logging
import platform
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("habitat1p19q")

SMOKE_BANNER = "SYNTHETIC SMOKE TEST — NOT STUDY RESULTS"


class Reporter:
    """Writes tables/figures under the output directory and records every automated check."""

    def __init__(self, output_dir, smoke=False):
        self.out = Path(output_dir)
        self.smoke = smoke
        self.checks = OrderedDict()

    def banner(self):
        if self.smoke:
            self.say("=" * 78 + "\n" + SMOKE_BANNER + "\n" + "=" * 78)

    def say(self, text=""):
        print(text, flush=True)

    def section(self, title):
        self.say("\n" + "#" * 78 + f"\n# {title}\n" + "#" * 78)

    def show(self, df, title=None, round_to=None):
        if title:
            self.say(title)
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if round_to is not None:
            df = df.round(round_to)
        with pd.option_context("display.width", 220, "display.max_columns", 60, "display.max_colwidth", 70):
            self.say(df.to_string())
        self.say()

    def save_table(self, df, name):
        path = self.out / "tables" / f"{name}.csv"
        df.to_csv(path, index=True)
        return path

    def save_fig(self, fig, name):
        import matplotlib.pyplot as plt
        path = self.out / "figures" / f"{name}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return path

    def record_check(self, name, passed, detail="", status=None):
        status = status or ("PASS" if passed else "FAIL")
        self.checks[name] = {"status": status, "detail": detail}
        self.say(f"{status} {name}" + (f" — {detail}" if detail else ""))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pkg_version(name):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", None) or (mod.Version_VersionString() if name == "SimpleITK" else "unknown")
    except Exception:
        return "not installed"


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            **{p: pkg_version(p) for p in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "SimpleITK",
                                           "radiomics", "joblib"]}}


def jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)

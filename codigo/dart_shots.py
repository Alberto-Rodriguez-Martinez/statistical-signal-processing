# -*- coding: utf-8 -*-
"""
dart_shots.py — minimal utilities for the darts teaching example.
Used to simulate shots and save them to folder

Functions:
- Generation:
  - make_gaussian_independent, make_gaussian_correlated, make_laplace_heavy_tails
  - write_csv, generate_datasets (convenience)

All distances are assumed to be in millimeters (consistent with the teaching example).

@author: alrom
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle


# =========================
# Data generation utilities
# =========================

def make_gaussian_independent(n: int, sigma_x: float, sigma_y: float,
                              rng: np.random.Generator) -> np.ndarray:
    """
    Generate independent Gaussian errors for X and Y.
    Returns array of shape (n, 2).
    """
    x = rng.normal(0.0, sigma_x, size=n)
    y = rng.normal(0.0, sigma_y, size=n)
    return np.column_stack([x, y])

def make_gaussian_correlated(n: int, sigma: float, rho: float,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Generate correlated Gaussian errors with correlation rho (|rho|<1) and same std in both axes.
    Returns array of shape (n, 2).
    """
    cov = np.array([[sigma**2, rho * sigma**2],
                    [rho * sigma**2, sigma**2]], dtype=float)
    mean = np.array([0.0, 0.0], dtype=float)
    data = rng.multivariate_normal(mean, cov, size=n)
    return data

def make_laplace_heavy_tails(n: int, b_x: float, b_y: float,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Generate heavy-tailed Laplace (double-exponential) errors (independent across axes).
    Returns array of shape (n, 2).
    """
    x = rng.laplace(0.0, b_x, size=n)
    y = rng.laplace(0.0, b_y, size=n)
    return np.column_stack([x, y])

def write_csv(path: Path | str, data: np.ndarray) -> None:
    """
    Write data (shape (n,2)) to CSV with header X,Y.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, delimiter=",", header="X,Y", comments="", fmt="%.6f")

@dataclass
class GenConfig:
    n: int = 500
    seed: int = 2025
    A_sigma_x: float = 20.0
    A_sigma_y: float = 20.0
    B_sigma: float = 15.0
    B_rho: float = 0.75
    C_bx: float = 10.0
    C_by: float = 10.0

def generate_datasets(outdir: Path | str,
                      prefix: str = "darts",
                      cfg: GenConfig = GenConfig(),
                      write_meta: bool = False) -> Dict[str, Path]:
    """
    Generate three datasets A (Gaussian independent), B (Gaussian correlated), C (Laplace).
    Returns dict mapping dataset labels to file paths.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    A = make_gaussian_independent(cfg.n, cfg.A_sigma_x, cfg.A_sigma_y, rng)
    B = make_gaussian_correlated(cfg.n, cfg.B_sigma, cfg.B_rho, rng)
    C = make_laplace_heavy_tails(cfg.n, cfg.C_bx, cfg.C_by, rng)

    paths = {
        "A": out / f"{prefix}_A.csv",
        "B": out / f"{prefix}_B.csv",
        "C": out / f"{prefix}_C.csv",
    }
    write_csv(paths["A"], A)
    write_csv(paths["B"], B)
    write_csv(paths["C"], C)

    if write_meta:
        meta = {
            "seed": cfg.seed,
            "n": cfg.n,
            "files": {k: str(v) for k, v in paths.items()},
            "params": {
                "A": {"type": "gaussian_independent", "sigma_x": cfg.A_sigma_x, "sigma_y": cfg.A_sigma_y},
                "B": {"type": "gaussian_correlated", "sigma": cfg.B_sigma, "rho": cfg.B_rho},
                "C": {"type": "laplace_heavy_tails", "b_x": cfg.C_bx, "b_y": cfg.C_by},
            },
        }
        import json
        (out / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return paths
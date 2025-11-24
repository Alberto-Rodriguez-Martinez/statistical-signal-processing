
# -*- coding: utf-8 -*-
"""
dart_shots.py — extended with configurable means per axis.
This version keeps the original API but adds optional mean parameters.
If means are not provided, they default to zero for backward compatibility.
All distances are in millimeters.

New parameters:
- make_gaussian_independent(..., mu_x=0.0, mu_y=0.0)
- make_gaussian_correlated(..., mu_x=0.0, mu_y=0.0)
- make_laplace_heavy_tails(..., mu_x=0.0, mu_y=0.0)

GenConfig adds fields:
- A_mu_x, A_mu_y
- B_mu_x, B_mu_y
- C_mu_x, C_mu_y
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

# =========================
# Data generation utilities
# =========================

def make_gaussian_independent(n: int, sigma_x: float, sigma_y: float,
                              rng: np.random.Generator,
                              mu_x: float = 0.0, mu_y: float = 0.0) -> np.ndarray:
    """
    Generate independent Gaussian samples with configurable means.
    Returns array of shape (n, 2) with columns [X, Y].
    """
    x = rng.normal(mu_x, sigma_x, size=n)
    y = rng.normal(mu_y, sigma_y, size=n)
    return np.column_stack([x, y])

def make_gaussian_correlated(n: int, sigma: float, rho: float,
                             rng: np.random.Generator,
                             mu_x: float = 0.0, mu_y: float = 0.0) -> np.ndarray:
    """
    Generate correlated Gaussian samples with correlation rho (|rho|<1),
    same std in both axes, and configurable means.
    Returns array of shape (n, 2).
    """
    cov = np.array([[sigma**2, rho * sigma**2],
                    [rho * sigma**2, sigma**2]], dtype=float)
    mean = np.array([mu_x, mu_y], dtype=float)
    data = rng.multivariate_normal(mean, cov, size=n)
    return data

def make_laplace_heavy_tails(n: int, b_x: float, b_y: float,
                             rng: np.random.Generator,
                             mu_x: float = 0.0, mu_y: float = 0.0) -> np.ndarray:
    """
    Generate independent Laplace (double-exponential) samples with configurable location (mean) per axis.
    Returns array of shape (n, 2).
    Note: For Laplace, the mean equals the location parameter when finite.
    """
    x = rng.laplace(mu_x, b_x, size=n)
    y = rng.laplace(mu_y, b_y, size=n)
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
    # Player A (Gaussian independent)
    A_sigma_x: float = 20.0
    A_sigma_y: float = 20.0
    A_mu_x: float = 0.0
    A_mu_y: float = 0.0
    # Player B (Gaussian correlated)
    B_sigma: float = 15.0
    B_rho: float = 0.75
    B_mu_x: float = 0.0
    B_mu_y: float = 0.0
    # Player C (Laplace heavy-tails)
    C_bx: float = 10.0
    C_by: float = 10.0
    C_mu_x: float = 0.0
    C_mu_y: float = 0.0

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

    A = make_gaussian_independent(cfg.n, cfg.A_sigma_x, cfg.A_sigma_y, rng,
                                  mu_x=cfg.A_mu_x, mu_y=cfg.A_mu_y)
    B = make_gaussian_correlated(cfg.n, cfg.B_sigma, cfg.B_rho, rng,
                                 mu_x=cfg.B_mu_x, mu_y=cfg.B_mu_y)
    C = make_laplace_heavy_tails(cfg.n, cfg.C_bx, cfg.C_by, rng,
                                 mu_x=cfg.C_mu_x, mu_y=cfg.C_mu_y)

    paths = {
        "A": Path(out) / f"{prefix}_A.csv",
        "B": Path(out) / f"{prefix}_B.csv",
        "C": Path(out) / f"{prefix}_C.csv",
    }
    write_csv(paths["A"], A)
    write_csv(paths["B"], B)
    write_csv(paths["C"], C)

    if write_meta:
        import json
        meta = {
            "seed": cfg.seed,
            "n": cfg.n,
            "files": {k: str(v) for k, v in paths.items()},
            "params": {
                "A": {"type": "gaussian_independent",
                      "mu_x": cfg.A_mu_x, "mu_y": cfg.A_mu_y,
                      "sigma_x": cfg.A_sigma_x, "sigma_y": cfg.A_sigma_y},
                "B": {"type": "gaussian_correlated",
                      "mu_x": cfg.B_mu_x, "mu_y": cfg.B_mu_y,
                      "sigma": cfg.B_sigma, "rho": cfg.B_rho},
                "C": {"type": "laplace_heavy_tails",
                      "mu_x": cfg.C_mu_x, "mu_y": cfg.C_mu_y,
                      "b_x": cfg.C_bx, "b_y": cfg.C_by},
            },
        }
        (out / f"{prefix}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return paths

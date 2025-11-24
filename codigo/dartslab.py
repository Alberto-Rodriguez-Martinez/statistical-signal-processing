# -*- coding: utf-8 -*-


"""
dartslab.py — minimal utilities for the darts teaching example.

Functions:
- Generation:
  - make_gaussian_independent, make_gaussian_correlated, make_laplace_heavy_tails
  - write_csv, generate_datasets (convenience)
- IO:
  - load_xy, load_many
- Plotting:
  - draw_dartboard, compute_limits, scatter_player, scatter_all

All distances are assumed to be in millimeters (consistent with the teaching example).
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
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection
from matplotlib import cm, colors

# ===========
# IO helpers
# ===========

def load_xy(path: Path | str) -> pd.DataFrame:
    """
    Load CSV with columns X,Y as a DataFrame of shape (n, 2).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    if not {"X", "Y"}.issubset(df.columns):
        raise ValueError(f"File {path} must contain columns 'X' and 'Y'.")
    return df[["X", "Y"]].copy()

def load_many(paths: Dict[str, Path | str]) -> Dict[str, pd.DataFrame]:
    """
    Load multiple CSVs keyed by dataset label.
    """
    return {k: load_xy(v) for k, v in paths.items()}


# ===============
# Plotting layer
# ===============

def draw_dartboard(ax: Axes,
                   max_radius: float,
                   ring_step: float = 10.0,
                   grid_color: str = "#d0d0d0") -> None:
    """
    Draw concentric rings and cross axes to mimic a dartboard.
    """
    r = ring_step
    while r <= max_radius:
        ax.add_patch(Circle((0, 0), r, fill=False, lw=1, edgecolor=grid_color))
        r += ring_step
    ax.axhline(0, lw=0.8, color=grid_color)
    ax.axvline(0, lw=0.8, color=grid_color)

def compute_limits(dfs: Dict[str, pd.DataFrame], padding: float = 5.0) -> Tuple[float, float, float, float, float]:
    """
    Compute square limits and max radius based on all datasets.
    Returns (x_min, x_max, y_min, y_max, rmax).
    """
    xs, ys = [], []
    for df in dfs.values():
        xs.append(df["X"].to_numpy())
        ys.append(df["Y"].to_numpy())
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    r_all = np.sqrt(x_all**2 + y_all**2)
    rmax = float(np.max(r_all)) + padding
    return (-rmax, rmax, -rmax, rmax, rmax)

def scatter_player(df: pd.DataFrame,
                   title: str = "Player - Throws",
                   limits: Optional[Tuple[float, float, float, float, float]] = None,
                   ring_step: float = 10.0,
                   point_size: float = 14,
                   alpha: float = 0.75) -> Tuple[Figure, Axes]:
    """
    Plot one player's throws on a dartboard background.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    if limits is None:
        limits = compute_limits({"_": df})
    x_min, x_max, y_min, y_max, rmax = limits
    draw_dartboard(ax, max_radius=rmax, ring_step=ring_step, grid_color="#d0d0d0")
    ax.scatter(df["X"], df["Y"], s=point_size, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(False)
    plt.tight_layout()
    return fig, ax

def scatter_all(datasets: Dict[str, pd.DataFrame],
                colors: Optional[Dict[str, str]] = None,
                title: str = "All Players - Color-coded Throws",
                ring_step: float = 10.0,
                point_size: float = 14,
                alpha: float = 0.75) -> Tuple[Figure, Axes]:
    """
    Plot all players together on a single dartboard, color-coded by label.
    """
    if colors is None:
        colors = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c"}

    x_min, x_max, y_min, y_max, rmax = compute_limits(datasets)
    fig, ax = plt.subplots(figsize=(4, 4))
    draw_dartboard(ax, max_radius=rmax, ring_step=ring_step, grid_color="#d0d0d0")

    for name, df in datasets.items():
        ax.scatter(df["X"], df["Y"], s=point_size, alpha=alpha,
                   label=f"Player {name}", color=colors.get(name, None))
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(False)
    ax.legend()
    plt.tight_layout()
    return fig, ax

# ---------------------------------------------
# Scatter plot: joint behavior of (X, Y)
# ---------------------------------------------
def scatter_plot(df: pd.DataFrame, title: str, 
                 point_size: float = 12, alpha: float = 0.6):
    """
    Visualize the *joint* distribution of X and Y.
    What to read from this plot:
      - Overall shape (circular vs. elongated ellipse) -> hints at independence vs. linear dependence.
      - Orientation (tilt) -> sign of correlation (up-right = positive; up-left = negative).
      - Center (mean) -> bias if not near (0,0).
      - Spread along axes -> different variances in X and Y (anisotropy).
      - Outliers/clusters -> potential data issues or multimodality.
      - Fan shapes -> heteroscedasticity (variance changes with level).
    """
    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Scatter shows the joint structure. Alpha < 1 helps reveal density when points overlap.
    ax.scatter(x, y, s=point_size, alpha=alpha)

    # Equal aspect ratio avoids misleading interpretations of dependence/shape.
    ax.set_aspect("equal", adjustable="box")

    # Axes and labels
    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

    # Light guides to orient the viewer to the origin (board center).
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.axvline(0, color="#cccccc", lw=0.8)

    # Optional: consistent limits across players could be set outside this function
    # e.g., ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    ax.grid(False)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# Histograms: marginal behavior of X and Y separately
# -------------------------------------------------
def histograms(df: pd.DataFrame, title_prefix: str, bins: int = 30):
    """
    Visualize *marginal* distributions f_X and f_Y using histograms.
    What to read from these plots:
      - Symmetry vs. skewness -> does a Normal model make sense?
      - Tail heaviness -> heavy tails suggest Laplace/Student-t rather than Normal.
      - Multiple peaks -> mixtures (e.g., two throwing modes).
      - Scale (std) differences between X and Y -> anisotropy.
    Practical tip:
      - Try multiple 'bins' values (e.g., 20, 30, 50); binning strongly affects the look.
    """
    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Overlaid histograms let you compare spreads and shapes along each axis.
    # density=False shows counts; switch to density=True for approximate PDFs.
    ax.hist(x, bins=bins, alpha=0.7, label="X", edgecolor="white")
    ax.hist(y, bins=bins, alpha=0.7, label="Y", edgecolor="white")

    ax.set_title(f"{title_prefix}: Histograms of X and Y")
    ax.set_xlabel("mm")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()

def scatter_and_hist(
    df: pd.DataFrame,
    title: str = "Scatter (left) and Histograms (right)",
    bins: int = 30,
    density: bool = False,
    ring_guides: bool = True,
    equal_limits: bool = True,
    pad: float = 5.0,
    point_size: float = 12,
    alpha_points: float = 0.6
):
    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()

    # Turn on constrained layout; do NOT call tight_layout() later
    fig, (ax_scatter, ax_hist) = plt.subplots(
        1, 2, figsize=(10, 5), gridspec_kw={"wspace": 0.25}, constrained_layout=True
    )
    fig.suptitle(title)

    # --- Left: scatter (joint) ---
    ax_scatter.scatter(x, y, s=point_size, alpha=alpha_points)
    ax_scatter.set_xlabel("X (mm)")
    ax_scatter.set_ylabel("Y (mm)")
    ax_scatter.set_aspect("equal", adjustable="box")
    if ring_guides:
        ax_scatter.axhline(0, color="#cccccc", lw=0.8)
        ax_scatter.axvline(0, color="#cccccc", lw=0.8)
    if equal_limits and x.size and y.size:
        r = np.sqrt(x**2 + y**2)
        rmax = float(np.max(r)) + pad
        ax_scatter.set_xlim(-rmax, rmax)
        ax_scatter.set_ylim(-rmax, rmax)

    # --- Right: histograms (marginals) ---
    ax_hist.hist(x, bins=bins, density=density, alpha=0.7, label="X", edgecolor="white")
    ax_hist.hist(y, bins=bins, density=density, alpha=0.7, label="Y", edgecolor="white")
    ax_hist.set_title("Marginals (X & Y)")
    ax_hist.set_xlabel("mm")
    ax_hist.set_ylabel("density" if density else "count")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.25)

    plt.show()


def hist3d_bars(
    df: pd.DataFrame,
    bins: int | tuple[int, int] = 20,
    density: bool = False,
    title: str = "3D Histogram (bars)",
    elev: float = 30,
    azim: float = -60,
    alpha: float = 0.9
):
    """
    3D bar histogram of the joint distribution of (X, Y).

    Parameters
    ----------
    df : DataFrame with columns 'X','Y'
    bins : int or (int,int), number of bins per axis (or pair)
    density : if True, show probability density; if False, show counts
    title : figure title
    elev, azim : 3D view angles (degrees)
    alpha : bar opacity
    """
    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()

    # 2D histogram on a regular grid
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=density)

    # Bin centers (for bar positions)
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])

    # Widths of each bin along x and y
    dx = np.diff(xedges)
    dy = np.diff(yedges)

    # Create a grid of centers; flatten for bar3d
    Xc, Yc = np.meshgrid(xcenters, ycenters, indexing="ij")
    xpos = Xc.ravel()
    ypos = Yc.ravel()
    zpos = np.zeros_like(xpos)

    # Match widths to each bar
    dx_rep = np.repeat(dx, len(ycenters))
    dy_tile = np.tile(dy, len(xcenters))

    dz = H.ravel()

    # Colors mapped to bar heights
    norm = colors.Normalize(vmin=dz.min(), vmax=dz.max() if dz.max() > 0 else 1.0)
    facecolors = cm.viridis(norm(dz))

    # Plot
    fig = plt.figure(figsize=(9, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    ax.bar3d(xpos, ypos, zpos, dx_rep, dy_tile, dz, shade=True, color=facecolors, alpha=alpha)

    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("density" if density else "count")
    plt.show()
    

def hist3d_surface(
    df: pd.DataFrame,
    bins: int | tuple[int, int] = 40,
    density: bool = True,
    title: str = "3D Surface of Binned Density",
    elev: float = 40,
    azim: float = -55
):
    """
    Render the 2D histogram as a 3D surface (plot_surface).
    Same information as a histogram, but continuous-looking between bin centers.
    """
    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()

    H, xedges, yedges = np.histogram2d(x, y, bins=bins, density=density)

    # Use bin centers for the surface grid
    xcenters = 0.5 * (xedges[:-1] + xedges[1:])
    ycenters = 0.5 * (yedges[:-1] + yedges[1:])
    Xc, Yc = np.meshgrid(xcenters, ycenters, indexing="ij")

    fig = plt.figure(figsize=(9, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    surf = ax.plot_surface(Xc, Yc, H, cmap=cm.viridis, linewidth=0, antialiased=True)
    fig.colorbar(surf, shrink=0.7, aspect=16, pad=0.1, label=("density" if density else "count"))

    ax.set_title(title)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("density" if density else "count")
    plt.show()
    

def simulate_darts_timevarying(
    N=1500,
    rho=0,
    muX_amp=8.0, muY_amp=-6.0,
    muX_freq=1/10000, muY_freq=1/6000,   # slow drifts (cycles per sample)
    sigX_base=12.0, sigY_base=10.0,
    sigX_amp=4.0,  sigY_amp=3.0,      # slow variance modulation
    seed=2025
):
    """
    Simulate a non-stationary darts process: (X[n], Y[n]) with slowly varying mean and variance.
    Returns DataFrame with columns X, Y and also the underlying muX, muY, sigX, sigY.
    """
    rng = np.random.default_rng(seed)
    n = np.arange(N)

    # Slow drifts in mean (sinusoidal for pedagogy)
    muX = muX_amp * 4*np.sin(2*np.pi*muX_freq * n)
    muY = muY_amp * 0.5*np.sin(2*np.pi*muY_freq * n)

    # Slow changes in standard deviation (stay positive)
    sigX = sigX_base + sigX_amp * np.sin(2*np.pi*(muX_freq/2) * n + 1.1)
    sigY = sigY_base + 4*sigY_amp * np.sin(2*np.pi*(muY_freq/2) * n - 0.7)
    sigX = np.clip(sigX, 1.0, None)
    sigY = np.clip(sigY, 1.0, None)

    # Correlated standard normal innovations [U, V] with corr=rho
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)  # 2x2
    Z = rng.normal(size=(N, 2)) @ L.T  # rows ~ N(0, cov)
    U, V = Z[:,0], Z[:,1]

    # Build X, Y with time-varying mean and variance
    X = muX + sigX * U
    Y = muY + sigY * V

    return pd.DataFrame({"X": X, "Y": Y, "muX": muX, "muY": muY, "sigX": sigX, "sigY": sigY})

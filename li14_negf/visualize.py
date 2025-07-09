"""Visualization utilities for coupling matrices and self-energies.

Matplotlib-based helper functions so users can quickly inspect

* absolute values of coupling blocks V_L_ED and V_ED_R extracted from
  ``partition.extract_pl_couplings``;
* spectral behaviour of the retarded self-energy Σ(E).

All routines return the created *matplotlib* figure to allow further tweaking
or saving by callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import matplotlib.pyplot as plt

from .partition import extract_pl_couplings
from .lead_self_energy import compute_sigma_grid, load_lead_blocks

# -----------------------------------------------------------------------------
# Coupling-matrix heatmaps
# -----------------------------------------------------------------------------


def plot_couplings(info: dict, *, save: str | Path | None = None):
    """Heat-map of |V_L_ED| and |V_ED_R|.

    Parameters
    ----------
    info
        Dictionary returned by :func:`partition.partition_system`.
    save
        Optional path; if provided, figure is saved (PNG) instead of shown.
    """

    coup = extract_pl_couplings(info)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

    for ax, key, title in zip(
        axes,
        ("V_L_ED", "V_ED_R"),
        ("|V_L_ED|", "|V_ED_R|"),
    ):
        mat = np.abs(coup[key])
        im = ax.imshow(mat, cmap="viridis", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("Column index")
        ax.set_ylabel("Row index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if save:
        fig.savefig(Path(save), dpi=150)
        plt.close(fig)
    else:
        plt.show(block=False)
    return fig


# -----------------------------------------------------------------------------
# Self-energy spectrum
# -----------------------------------------------------------------------------


def plot_self_energy(
    energies: Iterable[float] | np.ndarray,
    Sigma: np.ndarray | None = None,
    *,
    lead_json: str | Path | None = None,
    mode: Literal["imag_trace", "real_trace", "gamma_trace"] = "imag_trace",
    save: str | Path | None = None,
):
    """Plot selected scalar quantity derived from Σ(E).

    If *Sigma* is not provided it will be computed/loaded via
    :func:`lead_self_energy.compute_sigma_grid` (requires *lead_json*).

    Parameters
    ----------
    energies
        Real energy grid (Ha).
    Sigma
        Pre-computed Σ(E) array of shape (N_E, N, N).
    lead_json
        Path to *lead_data.json* (used only when *Sigma* is None).
    mode
        Which scalar to plot:
        * ``imag_trace`` – Im Tr Σ(E)
        * ``real_trace`` – Re Tr Σ(E)
        * ``gamma_trace`` – Γ(E) = –2 Im Tr Σ(E)
    save
        Optional path to save PNG instead of showing.
    """

    energies_arr = np.asarray(energies, dtype=float)

    if Sigma is None:
        if lead_json is None:
            raise ValueError("Either Sigma or lead_json must be provided")
        Sigma = compute_sigma_grid(energies_arr, lead_json=lead_json)

    if Sigma.ndim != 3:
        raise ValueError("Sigma array must have shape (N_E, N, N)")

    trace = np.trace(Sigma, axis1=1, axis2=2)

    if mode == "imag_trace":
        y = np.imag(trace)
        ylabel = "Im Tr Σ(E)"
    elif mode == "real_trace":
        y = np.real(trace)
        ylabel = "Re Tr Σ(E)"
    elif mode == "gamma_trace":
        y = -2 * np.imag(trace)
        ylabel = "Γ(E) = –2 Im Tr Σ(E)"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(energies_arr, y)
    ax.set_xlabel("Energy (Ha)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if save:
        fig.savefig(Path(save), dpi=150)
        plt.close(fig)
    else:
        plt.show(block=False)
    return fig


# -----------------------------------------------------------------------------
# Σ(E) diagnostics (Task 3.4)
# -----------------------------------------------------------------------------


def sigma_diagnostics_summary(
    energies: Iterable[float] | np.ndarray,
    Sigma: np.ndarray,
    *,
    tol: float = 1e-8,
) -> dict:
    """Return and print basic Hermiticity / Γ-positivity diagnostics.

    For each energy this routine computes

    • *herm_res*  = max‖Σ − Σ†‖ (max-abs element)
    • *gamma_min* = min λ[Γ] where Γ(E) = −2 Im Σ(E)

    A perfectly formed retarded self-energy should satisfy herm_res → 0 and
    Γ positive-semidefinite ⇒ gamma_min ≥ 0 (within numeric tolerance).
    """

    energies_arr = np.asarray(energies, dtype=float)
    Sigma_arr = np.asarray(Sigma)

    if Sigma_arr.ndim != 3 or Sigma_arr.shape[0] != energies_arr.size:
        raise ValueError("Sigma shape must be (N_E, N, N) and match energies")

    herm_res = np.empty(energies_arr.size)
    gamma_min = np.empty_like(herm_res)

    for i, Sig in enumerate(Sigma_arr):
        herm_res[i] = np.max(np.abs(Sig - Sig.conj().T))
        Gamma = -2 * np.imag(Sig)
        eigvals = np.linalg.eigvalsh(Gamma)
        gamma_min[i] = eigvals.min()

    # Print concise report
    print("[Σ Diagnostics] max |Σ-Σ†| over grid: {:.2e}".format(herm_res.max()))
    if herm_res.max() > tol:
        print("  WARNING: Hermiticity residual exceeds tol={:.1e}".format(tol))

    print("[Σ Diagnostics] min eigenvalue of Γ over grid: {:.2e}".format(gamma_min.min()))
    if gamma_min.min() < -tol:
        print("  WARNING: Γ is not positive-semidefinite within tol={:.1e}".format(tol))

    return {
        "herm_res": herm_res,
        "gamma_min": gamma_min,
    }


def plot_sigma_hermiticity(
    energies: Iterable[float] | np.ndarray,
    Sigma: np.ndarray,
    *,
    save: str | Path | None = None,
):
    """Plot max element-wise Hermiticity residual |Σ−Σ†| versus energy."""

    energies_arr = np.asarray(energies, dtype=float)
    Sigma_arr = np.asarray(Sigma)
    if Sigma_arr.ndim != 3:
        raise ValueError("Sigma array must have shape (N_E, N, N)")

    herm = [np.max(np.abs(S - S.conj().T)) for S in Sigma_arr]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.semilogy(energies_arr, herm)
    ax.set_xlabel("Energy (Ha)")
    ax.set_ylabel("max |Σ−Σ†|")
    ax.grid(True, which="both", alpha=0.3)

    if save:
        fig.savefig(Path(save), dpi=150)
        plt.close(fig)
    else:
        plt.show(block=False)
    return fig


def plot_gamma_eigenvalues(
    energies: Iterable[float] | np.ndarray,
    Sigma: np.ndarray,
    *,
    save: str | Path | None = None,
):
    """Plot minimum eigenvalue of Γ(E)=−2 Im Σ(E) vs energy."""

    energies_arr = np.asarray(energies, dtype=float)
    Sigma_arr = np.asarray(Sigma)

    mins = []
    for S in Sigma_arr:
        Gamma = -2 * np.imag(S)
        mins.append(np.linalg.eigvalsh(Gamma).min())

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(energies_arr, mins)
    ax.set_xlabel("Energy (Ha)")
    ax.set_ylabel("min eig Γ(E)")
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.grid(True, alpha=0.3)

    if save:
        fig.savefig(Path(save), dpi=150)
        plt.close(fig)
    else:
        plt.show(block=False)
    return fig 
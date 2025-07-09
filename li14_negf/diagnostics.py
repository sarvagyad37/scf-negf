"""Diagnostics utilities for SCF NEGF loop (Tasks 4.6a,b).

Functions implemented:
    gamma_from_sigma(Σ)     – Return broadening matrix Γ = i(Σ − Σ†).
    trace_gamma_S(Γ, S)     – Compute Tr[Γ S] (charge/current measure).
    current_conservation(Γ_L, Γ_R, S, tol) – Check |Tr[Γ_L S] − Tr[Γ_R S]| <= tol.
    min_eigen_gamma(Γ)      – Smallest eigenvalue of Γ (should be ≤ 0 for retarded Σ).
    save_potential_profile(potential, iteration) – Persist iteration-resolved potential profiles.
    validate_sigma(Gamma_L, Gamma_R, S_d, G_EF) – Validate Σ based on current conservation and channel count.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Sequence, Tuple

__all__ = [
    "gamma_from_sigma",
    "trace_gamma_S",
    "current_conservation",
    "min_eigen_gamma",
    "gf_hermiticity",
    "gf_identity_residual",
    "validate_device_gf",
    "save_potential_profile",
    "validate_sigma",
]


def gamma_from_sigma(Sigma: np.ndarray) -> np.ndarray:
    """Return Γ = i(Σ − Σ†) given retarded self‐energy Σ."""

    return 1j * (Sigma - Sigma.conj().T)


def trace_gamma_S(Gamma: np.ndarray, S: np.ndarray) -> float:
    """Return real part of Tr[Γ S] (electrons)."""

    return float(np.trace(Gamma @ S).real)


def current_conservation(
    Gamma_L: np.ndarray,
    Gamma_R: np.ndarray,
    S: np.ndarray,
    *,
    tol: float = 1e-3,
) -> bool:
    """Check current conservation |Tr[Γ_L S] − Tr[Γ_R S]| <= tol (electrons)."""

    qL = trace_gamma_S(Gamma_L, S)
    qR = trace_gamma_S(Gamma_R, S)
    return abs(qL - qR) <= tol


def min_eigen_gamma(Gamma: np.ndarray) -> float:
    """Return smallest eigenvalue of Γ (should be ≤ 0 for retarded Σ)."""

    eigs = np.linalg.eigvalsh((Gamma + Gamma.conj().T) / 2.0)  # ensure Hermitian
    return float(np.min(eigs).real)


# -----------------------------------------------------------------------------
# NEW: Device Green-function consistency checks (literature-backed)
# -----------------------------------------------------------------------------

def gf_hermiticity(G: np.ndarray, *, tol: float = 1e-8) -> float:
    """Return max‖G − G†‖ (element-wise abs).

    A strictly retarded Green function should satisfy G = G††.  Numerical
    integration/linear-algebra noise may lead to deviations on the order of
    1e-12–1e-10.  Values above *tol* indicate a potential loss of precision or
    an asymmetric Σ embedding.
    """

    res = np.max(np.abs(G - G.conj().T))
    return float(res)


def gf_identity_residual(
    E: complex,
    G: np.ndarray,
    H: np.ndarray,
    S: np.ndarray,
    Sigma_tot: np.ndarray,
    *,
    tol: float = 1e-6,
) -> float:
    """Return max| (E S − H − Σ) G − I | – Dyson identity residual.

    The matrix identity

        (E S − H − Σ) G = I

    must hold for a *retarded* Green function.  The maximum absolute element of
    the residual provides a direct measure of numerical accuracy.  Literature
    (e.g. Papior *et al.*, CPC 212, 8-24 (2017)) recommends 10⁻⁶–10⁻⁸ as a
    practical threshold for double-precision workflows.
    """

    eye = np.eye(G.shape[0], dtype=np.complex128)
    A = E * S - H - Sigma_tot
    resid = np.max(np.abs(A @ G - eye))
    return float(resid)


def validate_device_gf(
    E: complex,
    G: np.ndarray,
    H: np.ndarray,
    S: np.ndarray,
    Sigma_L: np.ndarray,
    Sigma_R: np.ndarray,
    *,
    tol_herm: float = 1e-8,
    tol_dyson: float = 1e-6,
) -> bool:
    """Return *True* if both Hermiticity and Dyson residual pass tolerances."""

    herm_ok = gf_hermiticity(G) < tol_herm

    Sigma_tot = Sigma_L + Sigma_R
    dyson_ok = gf_identity_residual(E, G, H, S, Sigma_tot) < tol_dyson

    return herm_ok and dyson_ok


# -----------------------------------------------------------------------------
# 4.6c – Persist iteration-resolved potential profiles
# -----------------------------------------------------------------------------

def save_potential_profile(
    potential: np.ndarray,
    iteration: int,
    *,
    out_dir: str | Path = "outputs",
) -> Path:
    """Save *potential* array to ``out_dir/potential_iter{iteration}.npy``.

    Creates *out_dir* if it does not exist.  Returns the ``Path`` written.
    """

    out_path = Path(out_dir).expanduser().absolute()
    out_path.mkdir(parents=True, exist_ok=True)

    fname = out_path / f"potential_iter{iteration}.npy"
    np.save(fname, potential)
    return fname


# -----------------------------------------------------------------------------
# Σ validation helper (Task 3.1d)
# -----------------------------------------------------------------------------

def validate_sigma(
    Gamma_L: np.ndarray,
    Gamma_R: np.ndarray,
    S_d: np.ndarray,
    G_EF: np.ndarray,
    *,
    tol: float = 1e-2,
) -> bool:
    """Return True if current conservation and channel-count tests pass.

    Criteria:
        1. |Tr[Γ_L S] − Tr[Γ_R S]| ≤ *tol* (electrons)
        2. |T(E_F) − N_open| ≤ *tol* where N_open = Tr[Γ S] / (2π)
    """

    # Current conservation
    if not current_conservation(Gamma_L, Gamma_R, S_d, tol=tol):
        return False

    # Channel count estimate from Γ_L
    n_open = trace_gamma_S(Gamma_L, S_d) / (2.0 * np.pi)

    # Zero-bias transmission at E_F
    T_EF = float(np.trace(Gamma_L @ G_EF @ Gamma_R @ G_EF.conj().T).real)

    return abs(T_EF - n_open) <= tol 
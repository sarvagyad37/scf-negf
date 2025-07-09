"""Potential-alignment utilities (shift-1) – Task 4.4a.

Shift-1 aligns the outermost principal-layer on-site Hamiltonian blocks in the
extended device (ED) with the bulk lead reference so that the self-energies and
device Hamiltonian share a common zero of potential.  The routine operates in
AO basis and assumes core Hamiltonian/Fock matrices are passed in.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

__all__ = ["compute_shift1", "apply_shift", "compute_shift2"]


def _mean_diag(mat: np.ndarray) -> float:
    """Return mean of real diagonal elements of *mat*."""

    return float(np.trace(mat).real) / mat.shape[0]


def compute_shift1(
    info: Dict[str, object],
    *,
    H_bulk_onsite: np.ndarray,
) -> float:
    """Compute constant Δ such that ED PL on-sites match *H_bulk_onsite*.

    Parameters
    ----------
    info
        Dictionary returned by :func:`partition.partition_system`.
    H_bulk_onsite
        On-site Hamiltonian block (H00) of the bulk principal layer, complex
        array with the same dimension as a PL block in ED.

    Returns
    -------
    delta_shift : float
        Value (Hartree) to *add* to the entire ED Hamiltonian.
    """

    from .partition import extract_pl_couplings  # local import

    coup = extract_pl_couplings(info)

    ao_pl_l = coup["ao_pl_l"]
    ao_pl_r = coup["ao_pl_r"]

    H_core: np.ndarray = info["H_core"]  # type: ignore[assignment]

    # On-site blocks inside ED
    H_LL = H_core[np.ix_(ao_pl_l, ao_pl_l)]
    H_RR = H_core[np.ix_(ao_pl_r, ao_pl_r)]

    mean_L = _mean_diag(H_LL)
    mean_R = _mean_diag(H_RR)
    mean_bulk = _mean_diag(H_bulk_onsite)

    delta_shift = mean_bulk - 0.5 * (mean_L + mean_R)

    return delta_shift


def apply_shift(H: np.ndarray, delta: float) -> np.ndarray:
    """Return a new matrix with *delta* added to the diagonal (in-place safe)."""

    H_shifted = H.copy()
    np.fill_diagonal(H_shifted, np.diag(H_shifted) + delta)
    return H_shifted


# -----------------------------------------------------------------------------
# Shift-2 – global potential shift ensuring charge neutrality in ED
# -----------------------------------------------------------------------------

def compute_shift2(
    D: np.ndarray,
    S: np.ndarray,
    n_electrons: float,
    *,
    proportionality: float = 0.5,
) -> float:
    """Return global potential shift Δ enforcing charge neutrality.

    We follow a simple proportional controller:

        Δ = −κ (Q − N_e) ,    Q = Tr[D S]

    where *κ* (``proportionality``) approximates the inverse electronic
    compressibility (∂Q/∂μ).  Literature (TranSIESTA manual §5.2) suggests κ in
    the range 0.1–1.0 Ha⋅e⁻¹ depending on device size; 0.5 works well for small
    Li chains.  The NEGF self-consistent loop will apply this Δ to all on-site
    terms every iteration until |Q−N_e| falls below tolerance.

    Parameters
    ----------
    D, S
        Density matrix and AO overlap of the *extended device*.
    n_electrons
        Target electron count (typically number of valence electrons in ED).
    proportionality
        Controller gain κ (Ha per electron).
    """

    Q = float(np.trace(D @ S).real)
    delta = -proportionality * (Q - n_electrons)
    return delta 
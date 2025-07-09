"""Kerker preconditioner for charge-sloshing control (Task 4.5b).

This module provides a single function :func:`kerker_precondition` that can be
plugged into :class:`mixing.MixerDIIS` to damp long-wavelength components of
the density residual in extended metallic devices.

The implementation is intentionally lightweight: we use the Mulliken charge
residual per AO and subtract the average (q=0) component, which corresponds to
the most problematic long-wavelength mode.  More elaborate schemes (FFT-based
k-dependent filters) can be added later if needed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["kerker_precondition"]


def kerker_precondition(
    residual_dm: np.ndarray,
    S: np.ndarray | None = None,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Return preconditioned residual.

    ``alpha`` (< 1) applies additional damping on top of the uniform-charge
    removal.  Setting ``alpha=0`` would completely freeze the residual (not
    useful except for debugging); typical values lie in the 0.3–1.0 range.

    Parameters
    ----------
    residual_dm
        Density-matrix residual *ΔD* (AO basis).
    S
        AO overlap matrix.  If ``None``, identity is assumed.  The Mulliken
        charge residual is computed as ``q = diag(ΔD S)``.
    alpha
        Extra damping factor after the Kerker filter is applied.

    Notes
    -----
    The standard Kerker filter in plane-wave DFT multiplies each Fourier mode
    by  \( f(k) = k^2 / (k^2 + k_0^2) \).  In a tight-binding/AO context we
    lack a well-defined \(k\) grid, but the worst offender is always the *k←0*
    component (uniform charge shift).  Removing that component already helps
    convergence dramatically and is what we implement here.
    """

    if S is None:
        S = np.eye(residual_dm.shape[0])

    # Mulliken charge residual per AO
    charges = np.real(np.einsum("ij,ji->i", residual_dm, S))
    q_avg = charges.mean()

    # Nothing to do if already charge-neutral
    if abs(q_avg) < 1e-12:
        return residual_dm

    # Subtract uniform component from the diagonal elements only (least invasive)
    correction = np.zeros_like(residual_dm)
    np.fill_diagonal(correction, q_avg / np.diag(S).real)

    filtered = residual_dm - correction

    if alpha != 1.0:
        filtered *= alpha

    return filtered 
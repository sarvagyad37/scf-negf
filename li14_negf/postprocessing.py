"""Post-processing helpers: transmission and zero-bias conductance (Task 5.1).

Functions
---------
transmission(G, Gamma_L, Gamma_R)  – Landauer formula T(E) = Tr[Γ_L G Γ_R G†]
conductance(T_EF)                 – G = (2e^2/h) T(E_F)  (returns in units of G0)
"""

from __future__ import annotations

import numpy as np

__all__ = ["transmission", "conductance"]

_G0_SI = 7.748091729e-5  # S  (2e^2/h)


def transmission(G: np.ndarray, Gamma_L: np.ndarray, Gamma_R: np.ndarray) -> float:
    """Return Landauer transmission T(E) for given Green function & Γ's."""

    T = np.trace(Gamma_L @ G @ Gamma_R @ G.conj().T).real
    return float(T)


def conductance(T_EF: float, *, in_units: str = "G0") -> float:
    """Return zero-bias conductance.

    Parameters
    ----------
    T_EF
        Transmission at the Fermi level.
    in_units
        "G0"  – multiples of the conductance quantum 2e^2/h (default)
        "S"   – Siemens (absolute value).
    """

    if in_units == "G0":
        return float(T_EF)
    elif in_units == "S":
        return float(T_EF * _G0_SI)
    else:
        raise ValueError("in_units must be 'G0' or 'S'")


# -----------------------------------------------------------------------------
# Density of states
# -----------------------------------------------------------------------------

def density_of_states(G: np.ndarray, S: np.ndarray | None = None) -> float:
    """Return total DOS(E) = -(1/π) Im Tr[G S]  (or Tr[G] if S is None).

    Parameters
    ----------
    G
        Retarded Green function at energy E.
    S
        Overlap matrix (optional).  If omitted, identity is assumed.
    """

    if S is None:
        trace = np.trace(G)
    else:
        trace = np.trace(G @ S)

    return -float(trace.imag) / np.pi


__all__.append("density_of_states")

# -----------------------------------------------------------------------------
# MO isosurface cube (Task 5.3)
# -----------------------------------------------------------------------------

def write_mo_cube(mol, mo_coeff, mo_idx: int, filename: str, *, n_grid: int = 80):
    """Write a Gaussian cube file for molecular orbital *mo_idx*.

    Parameters
    ----------
    mol
        PySCF ``Mole`` object.
    mo_coeff
        Molecular orbital coefficient matrix (AOs × MOs).
    mo_idx
        Index of MO to export (0-based).
    filename
        Output .cube path.
    n_grid
        Cube grid density (points per axis).
    """

    try:
        from pyscf.tools import cubegen
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PySCF cube generation unavailable") from exc

    cubegen.orbital(mol, filename, mo_coeff[:, mo_idx], nx=n_grid, ny=n_grid, nz=n_grid)

__all__.append("write_mo_cube") 
"""Geometry utilities for Li14 DFT + NEGF project.

Currently provides helpers to construct a linear lithium chain suitable for
PySCF calculations.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from pyscf import gto

__all__ = ["build_li_chain"]


def build_li_chain(
    n_atoms: int = 22,
    spacing: float = 3.0,
    basis: str | dict = "sto-3g",
    unit: str = "Angstrom",
) -> Tuple[gto.Mole, np.ndarray]:
    """Construct a linear Li chain along the *z*-axis.

    Parameters
    ----------
    n_atoms
        Total number of Li atoms. *Must be even* to keep spin compensation
        (see docs/project-brain.md Mandatory Rules).
    spacing
        Nearest-neighbor distance between Li nuclei in *Angstrom* by default.
    basis
        Basis-set descriptor understood by PySCF, e.g. ``"sto-3g"`` or a
        custom basis dictionary.
    unit
        Distance unit for *spacing* and returned MOL coordinates – accepted
        values are ``"Angstrom"`` (default) or ``"Bohr"``.

    Returns
    -------
    mol
        A **built** ``pyscf.gto.Mole`` instance ready for SCF calculations.
    coords
        ``(n_atoms, 3)`` NumPy array of nuclear positions in **Bohr** (PySCF
        internal units).
    """

    if n_atoms % 2 != 0:
        raise ValueError("Only even n_atoms recommended to maintain spin neutrality.")

    # Build coordinates: chain aligned with z-axis
    coords = np.zeros((n_atoms, 3))
    coords[:, 2] = np.arange(n_atoms) * spacing  # 0, spacing, 2*spacing, ...

    # Prepare PySCF molecule
    mol = gto.Mole()
    mol.atom = [["Li", *coord] for coord in coords]
    mol.unit = unit  # PySCF will convert to Bohr internally
    mol.basis = basis
    mol.charge = 0
    mol.spin = 0  # 2 * S; even-electron closed-shell assumption

    # Use moderate grid level; can be tuned later if accuracy demands
    mol.build(verbose=4)  # verbose for debugging geometry step

    # Convert coords to Bohr (PySCF internal) for downstream users
    coords_bohr = np.asarray(mol.atom_coords(unit="Bohr"))

    return mol, coords_bohr 
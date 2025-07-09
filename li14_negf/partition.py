"""Partitioning utilities for Li14 DFT + NEGF project.

This module divides the full lithium chain into lead/device regions and
provides convenience functions to slice overlap and Hamiltonian matrices into
blocks required for NEGF calculations.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from pyscf import gto

__all__ = [
    "partition_system",
    "print_partition_summary",
    "extract_pl_couplings",
]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _atom_ao_ranges(mol: gto.Mole) -> List[Tuple[int, int]]:
    """Return (start, stop) AO index range for each atom in *mol*.

    The ranges are half-open [start, stop) and follow PySCF AO ordering.
    """
    aoslices = mol.aoslice_by_atom()
    ranges: List[Tuple[int, int]] = []
    for start, stop in aoslices[:, [2, 3]]:  # cols 2: nao_start, 3: nao_end
        ranges.append((int(start), int(stop)))
    return ranges


def _collect_ao_indices(ranges: List[Tuple[int, int]], atom_idx: List[int]) -> np.ndarray:
    """Return concatenated AO indices for the specified *atom_idx* list."""
    idx: List[int] = []
    for a in atom_idx:
        start, stop = ranges[a]
        idx.extend(range(start, stop))
    return np.asarray(idx, dtype=int)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def partition_system(
    mol: gto.Mole,
    *,
    n_left: int = 8,
    n_device: int = 6,
    n_right: int = 8,
    lead_unit: int = 4,
) -> Dict[str, object]:
    """Partition a built PySCF **mol** into lead/device/extended-device blocks.

    Parameters
    ----------
    mol
        Built PySCF ``Mole`` containing the *full* Li-14 chain (14 atoms).
    n_left, n_device, n_right
        Atom counts in left lead fragment, central device fragment, and right
        lead fragment, respectively.
    lead_unit
        Number of atoms that constitute a single principal layer of the lead.
        Used later for self-energy calculations.

    Returns
    -------
    info
        Dictionary containing:
        - ``S``: full AO-overlap matrix (numpy.ndarray)
        - ``H_core``: one-electron core Hamiltonian (kinetic + nuclear)
        - ``S_blocks``: region-specific blocks of S (dict of numpy.ndarrays)
        - ``H_blocks``: region-specific blocks of H_core (dict of numpy.ndarrays)
        - ``ao_idx``: mapping of region → AO indices (numpy arrays)
        - ``atom_idx``: mapping of region → atom indices
        - ``meta``: misc metadata (slice boundaries, counts)
        - ``mol``: keep reference for downstream AO mapping
    """

    # Calculate fundamental one-electron integrals (AO basis)
    S = mol.intor("int1e_ovlp")
    T = mol.intor("int1e_kin")
    V = mol.intor("int1e_nuc")
    H_core = T + V

    total_atoms = mol.natm
    if total_atoms != (n_left + n_device + n_right):
        raise ValueError(
            f"Mismatch: mol has {total_atoms} atoms but partition counts sum to {n_left + n_device + n_right}."
        )

    # Determine atom index ranges for each region
    idx_left = list(range(0, n_left))
    idx_device = list(range(n_left, n_left + n_device))
    idx_right = list(range(n_left + n_device, total_atoms))

    # Extended device includes +lead_unit atoms on both sides
    idx_ext_device = list(range(n_left - lead_unit, n_left + n_device + lead_unit))

    # Build AO index arrays
    ao_ranges = _atom_ao_ranges(mol)
    ao_left = _collect_ao_indices(ao_ranges, idx_left)
    ao_device = _collect_ao_indices(ao_ranges, idx_device)
    ao_right = _collect_ao_indices(ao_ranges, idx_right)
    ao_ext_device = _collect_ao_indices(ao_ranges, idx_ext_device)

    ao_idx = {
        "left": ao_left,
        "device": ao_device,
        "right": ao_right,
        "extended_device": ao_ext_device,
    }

    atom_idx = {
        "left": np.asarray(idx_left),
        "device": np.asarray(idx_device),
        "right": np.asarray(idx_right),
        "extended_device": np.asarray(idx_ext_device),
    }

    meta = {
        "n_left": n_left,
        "n_device": n_device,
        "n_right": n_right,
        "lead_unit": lead_unit,
    }

    # ------------------------------------------------------------------
    # Slice region-specific blocks (square) for convenience.
    # We keep the **original ordering** of AOs inside each region so that
    # downstream code (self-energy, device GF) can work in that reduced
    # sub-space without reindexing.
    # ------------------------------------------------------------------

    def _sub(mat: np.ndarray, idx: np.ndarray) -> np.ndarray:  # helper
        return mat[np.ix_(idx, idx)]

    S_blocks = {reg: _sub(S, idx) for reg, idx in ao_idx.items()}
    H_blocks = {reg: _sub(H_core, idx) for reg, idx in ao_idx.items()}

    return {
        "S": S,
        "H_core": H_core,
        "S_blocks": S_blocks,
        "H_blocks": H_blocks,
        "ao_idx": ao_idx,
        "atom_idx": atom_idx,
        "meta": meta,
        "mol": mol,  # keep reference for downstream AO mapping
    }


# -----------------------------------------------------------------------------
# Convenience: pretty-print partition diagnostics
# -----------------------------------------------------------------------------

def print_partition_summary(info: Dict[str, object]) -> None:
    """Print human-readable summary of atom counts and matrix dimensions.

    Parameters
    ----------
    info
        The dictionary returned by :func:`partition_system`.
    """

    meta = info["meta"]
    print("\n=== Partition Summary ===")
    print(
        f"Left atoms : {meta['n_left']},  Device atoms : {meta['n_device']},  Right atoms : {meta['n_right']}"
    )
    for reg in ("left", "device", "right", "extended_device"):
        s_block = info["S_blocks"][reg]
        print(f"  {reg:>14s}  —  AO dim: {s_block.shape[0]}")
    print("========================\n")


# -----------------------------------------------------------------------------
# Coupling extraction helpers (Task 3.3)
# -----------------------------------------------------------------------------

def extract_pl_couplings(
    info: Dict[str, object],
) -> Dict[str, np.ndarray]:
    """Return coupling Hamiltonian blocks V_L_ED and V_ED_R.

    The extended device (ED) contains *lead_unit* atoms from each lead.  We
    define:

    • **PL_L** – the left-most *lead_unit* atoms inside ED.
    • **PL_R** – the right-most *lead_unit* atoms inside ED.

    V_L_ED is the block coupling PL_L ↔ (ED \ PL_L).  Likewise V_ED_R couples
    (ED \ PL_R) ↔ PL_R.

    These matrices are required to assemble the full device Green-function and
    to compute Γ = i(Σ – Σ†).  They use the *bare* one-electron Hamiltonian
    stored in the *info* dictionary (currently H_core; can be replaced by Fock
    later in the SCF loop).
    """

    H = info["H_core"]  # one-electron Hamiltonian in AO basis
    ao_idx: Dict[str, np.ndarray] = info["ao_idx"]
    meta = info["meta"]

    lead_unit = int(meta["lead_unit"])

    # Atom indices of ED and left/right principal layers inside ED
    idx_ext_atoms: np.ndarray = info["atom_idx"]["extended_device"]

    # Determine PL_L / PL_R atoms via positions inside extended list
    pl_l_atoms = idx_ext_atoms[:lead_unit]
    pl_r_atoms = idx_ext_atoms[-lead_unit:]

    # Build AO index arrays for those atoms
    mol = info["mol"]
    ao_ranges = _atom_ao_ranges(mol)
    ao_pl_l = _collect_ao_indices(ao_ranges, list(pl_l_atoms))
    ao_pl_r = _collect_ao_indices(ao_ranges, list(pl_r_atoms))

    ao_ed = ao_idx["extended_device"]

    # Complement sets (ED \ PL_{L,R})
    ao_ed_no_l = np.setdiff1d(ao_ed, ao_pl_l, assume_unique=True)
    ao_ed_no_r = np.setdiff1d(ao_ed, ao_pl_r, assume_unique=True)

    # Coupling blocks (Hamiltonian off-diagonals)
    def _block(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        return H[np.ix_(rows, cols)]

    V_L_ED = _block(ao_pl_l, ao_ed_no_l)
    V_ED_R = _block(ao_ed_no_r, ao_pl_r)

    return {
        "V_L_ED": V_L_ED,
        "V_ED_R": V_ED_R,
        "ao_pl_l": ao_pl_l,
        "ao_pl_r": ao_pl_r,
    } 
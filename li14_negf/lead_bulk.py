"""Bulk lead (4-atom principal layer) DFT utilities.

This module performs a one-shot, finite-temperature LDA calculation on a
**4-atom lithium principal layer** and extracts the Hamiltonian/overlap blocks
required for surface Green-function self-energies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from pyscf import dft, gto, scf
from pyscf.scf import addons as scf_addons
from pyscf.scf.addons import _smearing_optimize, _fermi_smearing_occ

from .geometry import build_li_chain
from .partition import _atom_ao_ranges, _collect_ao_indices  # local import

__all__ = [
    "compute_bulk_lead",
    "save_lead_data",
]

TMP_PATH = Path("outputs/lead_data.json")


def compute_bulk_lead(
    spacing: float = 3.0,
    basis: str = "sto-3g",
    smear_sigma: float = 0.001,  # Ha (~300 K)
) -> Dict[str, np.ndarray]:
    """Run single-shot LDA DFT for 4-atom Li principal layer.

    Parameters
    ----------
    spacing
        Li–Li distance in Å.
    basis
        PySCF basis spec (default STO-3G).
    smear_sigma
        Gaussian/fermi smearing width in Hartree (~ 0.00095 Ha ≈ 300 K).

    Returns
    -------
    data
        Dict with keys ``S00``, ``S01``, ``H00``, ``H01``, ``fermi``.
    """

    # ------------------------------------------------------------------
    # Strategy: build **two** principal layers (8 atoms) so we can extract
    # an explicit inter-layer coupling H01 and S01.  The first 4 atoms = PL0,
    # the next 4 atoms = PL1.
    # ------------------------------------------------------------------

    mol, _coords = build_li_chain(n_atoms=8, spacing=spacing, basis=basis)

    # Unrestricted/Restricted: Lithium chain is metallic-ish but closed-shell
    # is fine for LDA ground state.
    mf = dft.RKS(mol, xc="lda,vwn")
    mf.conv_tol = 1e-9
    mf.max_cycle = 150

    # Add Fermi–Dirac smearing
    mf = scf_addons.smearing(mf, sigma=smear_sigma, method="fermi")

    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Bulk lead DFT did not converge – adjust settings.")

    # Full-overlap and Fock in AO basis (mf.get_ovlp caches integrals)
    S_full = mf.get_ovlp()
    H_full = mf.get_fock()

    # Slice AO indices for the two 4-atom principal layers
    ao_ranges = _atom_ao_ranges(mol)
    idx_pl0 = _collect_ao_indices(ao_ranges, list(range(4)))
    idx_pl1 = _collect_ao_indices(ao_ranges, list(range(4, 8)))

    # Extract blocks
    H00 = H_full[np.ix_(idx_pl0, idx_pl0)]
    H01 = H_full[np.ix_(idx_pl0, idx_pl1)]

    S00 = S_full[np.ix_(idx_pl0, idx_pl0)]
    S01 = S_full[np.ix_(idx_pl0, idx_pl1)]

    # Use the same Fermi–Dirac occupancy routine employed by PySCF smearing
    mo_e = mf.mo_energy
    nelec = mol.nelectron
    sigma_fd = smear_sigma

    fermi, _ = _smearing_optimize(
        _fermi_smearing_occ,
        mo_e,
        nelec,
        sigma_fd,
    )

    fermi = float(fermi)

    return {
        "S00": S00,
        "S01": S01,
        "H00": H00,
        "H01": H01,
        "fermi": fermi,
    }


def save_lead_data(data: Dict[str, np.ndarray | float], path: Path | str = TMP_PATH) -> None:
    """Serialize lead matrices + Fermi level to JSON (NumPy lists)."""
    out = {
        "S00": data["S00"].tolist(),
        "S01": data["S01"].tolist(),
        "H00": data["H00"].tolist(),
        "H01": data["H01"].tolist(),
        "fermi": data["fermi"],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2) 
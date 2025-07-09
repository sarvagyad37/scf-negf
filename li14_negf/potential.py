"""Potential analysis utilities.

Provides a helper to approximate the Hartree/effective potential profile
along the lithium chain by averaging the on-site AO potential matrix blocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft, scf
from pyscf.scf import addons as scf_addons

from .geometry import build_li_chain
from .partition import _atom_ao_ranges  # reuse internal helper

plt.rcParams.update({"figure.dpi": 150})

__all__ = [
    "compute_site_potential",
    "plot_site_potential",
]


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def compute_site_potential(
    n_atoms: int = 22,
    spacing: float = 3.0,
    basis: str = "sto-3g",
    smear_sigma: float = 0.001,  # Ha (~300 K)
) -> Tuple[np.ndarray, np.ndarray]:
    """Return z positions (Bohr) and average on-site potentials (Hartree).

    The potential is approximated by averaging the diagonal elements of the
    effective Kohn–Sham potential matrix V_eff for the AO block belonging to
    each atom.
    """

    mol, _ = build_li_chain(n_atoms=n_atoms, spacing=spacing, basis=basis)

    # Finite-T LDA SCF
    mf = dft.RKS(mol, xc="lda,vwn")
    scf_addons.smearing_(mf, sigma=smear_sigma)
    mf.kernel()

    veff = mf.get_veff(mol)

    ao_ranges = _atom_ao_ranges(mol)
    z_coords = mol.atom_coords()[:, 2]  # Å
    z_coords_bohr = z_coords / 0.52917721092  # convert to Bohr for consistency

    site_pot = np.empty(n_atoms)
    for i, (s, e) in enumerate(ao_ranges):
        block = veff[s:e, s:e]
        site_pot[i] = np.trace(block).real / (e - s)

    return z_coords_bohr, site_pot


def plot_site_potential(z: np.ndarray, pot: np.ndarray, filename: str | Path | None = None) -> None:
    """Plot potential vs z and optionally save PNG to *filename*."""

    plt.figure(figsize=(6, 3))
    plt.plot(z, pot, marker="o")
    plt.xlabel("z (Bohr)")
    plt.ylabel("Average on-site V_eff (Ha)")
    plt.title("Effective potential profile along Li chain")
    plt.grid(True)
    if filename is not None:
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
    plt.show() 
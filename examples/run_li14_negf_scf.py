"""Self-consistent DFT + NEGF calculation for the Li chain.

This script executes a *minimal* self-consistent loop that now embeds the
left/right electrode self-energies Σ_L/R(E) into the Green-function and
updates the Fock matrix with PySCF at every cycle.

Workflow
--------
1. Build a 22-atom Li chain (customisable) and partition it into
   L(8) – C(6) – R(8).
2. Compute Δ₁ alignment against an 8-atom bulk-lead calculation
   (saved/loaded from ``outputs/lead_data.json``).
3. Construct equilibrium contour and a matching Σ_L/R(E) grid on all
   contour points; embed Σ into the extended-device space.
4. Call :pyfunc:`li14_negf.scf.run_scf_loop` with the Σ grids so every
   iteration integrates *embedded* Green-functions.
5. Save converged density / Fock matrices and a convergence log.

Usage::

    python examples/run_li14_negf_scf.py --max-iter 50  # default 22 atoms

Notes
-----
* Shift-2 neutrality correction **is now active**; its magnitude is
  scaled by the proportionality constant κ (CLI flag `--shift2-kappa`).
* The Σ(E) cache is automatically refreshed whenever the alignment shift
  Δ₁ changes by more than `--shift-tol` (Ha).
* For debugging, add `--probe-every N` to save an unconverged transmission
  spectrum every *N* SCF cycles (plots + data in the output directory).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from datetime import datetime

import numpy as np

import li14_negf as ln
from li14_negf import (
    geometry,
    partition,
    alignment,
    lead_self_energy,
    scf,
)

from pyscf import gto

# -----------------------------------------------------------------------------
# Command-line interface
# -----------------------------------------------------------------------------

def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--n-atoms", type=int, default=22, help="Total number of Li atoms (even).")
    p.add_argument("--spacing", type=float, default=3.0, help="Li–Li spacing in Å.")
    p.add_argument("--max-iter", type=int, default=300, help="Maximum SCF iterations.")
    p.add_argument("--tol-dm", type=float, default=1e-5, help="Convergence threshold on density matrix.")
    p.add_argument("--tol-e", type=float, default=1e-6, help="Convergence threshold on total energy (Ha).")
    p.add_argument("--eta", type=float, default=1e-6, help="Green-function broadening (Ha).")
    p.add_argument("--out", type=Path, default=Path("outputs"), help="Output directory.")
    p.add_argument("--transmission", action="store_true", help="Compute and save T(E) after SCF convergence.")
    p.add_argument("--e-min", type=float, default=-0.2, help="Min energy window (Ha) for T(E) plot.")
    p.add_argument("--e-max", type=float, default=0.2, help="Max energy window (Ha) for T(E) plot.")
    p.add_argument("--n-e", type=int, default=401, help="Number of energy points for transmission.")
    p.add_argument("--shift2-kappa", type=float, default=1e-4, help="Proportionality constant κ controlling the magnitude of the Shift-2 neutrality correction (0 disables Shift-2).")
    p.add_argument("--shift-tol", type=float, default=1e-4, help="Δ₁ change tolerance (Ha) that triggers a Σ(E) grid refresh.")
    p.add_argument("--align-freeze-dm", type=float, default=1e-3, help="Once max|ΔD| falls below this, Δ₁ alignment is frozen.")
    p.add_argument("--align-freeze-e", type=float, default=1e-3, help="Once |ΔE| falls below this, Δ₁ alignment is frozen.")
    p.add_argument("--probe-every", type=int, default=1, help="If >0, compute transmission spectrum every N SCF iterations.")
    p.add_argument("--probe-first", type=int, default=5, help="Compute transmission in each of the first N iterations before switching to the periodic schedule.")
    p.add_argument("--diis-warmup", type=int, default=3, help="Number of initial iterations that use linear mixing before enabling DIIS (warm-up phase).")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Helper – embed Sigma(E) grids into full ED dimension
# -----------------------------------------------------------------------------

def _embed_sigma_grid(info: dict, sigma_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Σ_L_full, Σ_R_full) each of shape (N_E, N_d, N_d)."""

    idx_ed_global = info["ao_idx"]["extended_device"]
    coup = partition.extract_pl_couplings(info)
    idx_pl_l = coup["ao_pl_l"]
    idx_pl_r = coup["ao_pl_r"]

    lookup = {ao: i for i, ao in enumerate(idx_ed_global)}
    idx_pl_l_loc = np.array([lookup[i] for i in idx_pl_l], dtype=int)
    idx_pl_r_loc = np.array([lookup[i] for i in idx_pl_r], dtype=int)

    n_e, n_pl, _ = sigma_grid.shape
    n_d = len(idx_ed_global)

    sigma_l = np.zeros((n_e, n_d, n_d), dtype=np.complex128)
    sigma_r = np.zeros_like(sigma_l)

    for k in range(n_e):
        sig = sigma_grid[k]
        sigma_l[k][np.ix_(idx_pl_l_loc, idx_pl_l_loc)] = sig
        sigma_r[k][np.ix_(idx_pl_r_loc, idx_pl_r_loc)] = sig

    return sigma_l, sigma_r


def _make_submol(mol_full: gto.Mole, atom_idx: np.ndarray) -> gto.Mole:
    """Return a new PySCF Mole consisting of *atom_idx* slice of mol_full."""

    atoms = [(mol_full.atom_symbol(i), mol_full.atom_coord(i, unit="Angstrom")) for i in atom_idx]
    submol = gto.Mole()
    submol.atom = atoms
    submol.basis = mol_full.basis
    submol.unit = "Angstrom"
    submol.charge = 0
    submol.spin = 0
    submol.build(verbose=0)
    return submol


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main() -> None:
    args = _parse_cli()

    # ------------------------------------------------------------------
    # Create a unique timestamped sub-directory inside the requested
    # output root so that consecutive runs never overwrite each other.
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    out_root = args.out
    out_dir = out_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=False)

    # Re-bind args.out so the rest of the script saves into the new dir
    args.out = out_dir

    # 1. Build full chain and partition ------------------------------------------------
    mol_full, _ = geometry.build_li_chain(n_atoms=args.n_atoms, spacing=args.spacing)
    info = partition.partition_system(mol_full)
    partition.print_partition_summary(info)

    # 2. Alignment against bulk lead ---------------------------------------------------
    lead_json = args.out / "lead_data.json"
    if not lead_json.exists():
        print("[run_negf_scf] No cached lead data → computing bulk lead …")
        lead_data = ln.lead_bulk.compute_bulk_lead()
        ln.lead_bulk.save_lead_data(lead_data, lead_json)
    lead_blocks = lead_self_energy.load_lead_blocks(lead_json)
    # If the stored JSON lacks overlap blocks (legacy runs) rebuild bulk lead
    if lead_blocks[2] is None or lead_blocks[3] is None:
        print("[run_negf_scf] Legacy lead_data detected – rebuilding with overlaps …")
        lead_data = ln.lead_bulk.compute_bulk_lead()
        ln.lead_bulk.save_lead_data(lead_data, lead_json)
        lead_blocks = lead_self_energy.load_lead_blocks(lead_json)

    h00 = lead_blocks[0]

    delta1 = alignment.compute_shift1(info, H_bulk_onsite=h00)
    print(f"[alignment] Δ₁ = {delta1:.6f} Ha  ({delta1*27.2114:.3f} eV)")

    # Slice ED overlap matrix once ------------------------------------------------------
    idx_ed = info["ao_idx"]["extended_device"]
    S_full = info["S"]
    S_d = S_full[np.ix_(idx_ed, idx_ed)]

    # 3. Contour and Sigma grid ---------------------------------------------------------
    contour = scf.build_contour(mode="equilibrium")
    energies_cont = contour["E_c"]

    print("[run_negf_scf] Σ grids will be handled inside SCF loop.")

    # 4. Run SCF loop -------------------------------------------------------------------
    mol_ed = _make_submol(mol_full, info["atom_idx"]["extended_device"])

    dm, fock, E_tot, n_done = scf.run_scf_loop(
        mol_ed,
        info=info,
        lead_json=lead_json,
        H_bulk_onsite=h00,
        contour=contour,
        n_iter=args.max_iter,
        tol_dm=args.tol_dm,
        tol_E=args.tol_e,
        shift_tol=args.shift_tol,
        align_freeze_dm=args.align_freeze_dm,
        align_freeze_E=args.align_freeze_e,
        verbose=True,
        shift2_kappa=args.shift2_kappa,
        transmission_probe=(
            {
                "every": args.probe_every,
                "first_n": args.probe_first,
                "E_grid": np.linspace(args.e_min, args.e_max, args.n_e),
                "out_dir": args.out,
            }
            if args.probe_every > 0
            else None
        ),
        diis_warmup=args.diis_warmup,
    )

    # ------------------------------------------------------------------
    # Alignment & Σ diagnostics (literature check-list)
    # ------------------------------------------------------------------
    from li14_negf.alignment import compute_shift1 as _cshift1, _mean_diag
    from li14_negf.partition import extract_pl_couplings as _extract_cpl
    from li14_negf.lead_self_energy import compute_sigma_grid as _sig_grid
    from li14_negf.visualize import sigma_diagnostics_summary as _sig_diag

    delta1_after = _cshift1(info, H_bulk_onsite=h00)
    print("\n[DIAG] Δ₁ recomputed after SCF: {:.6e} Ha".format(delta1_after))

    # On-site block comparison (should be ~0 after shift)
    coup = _extract_cpl(info)
    ao_pl_l = coup["ao_pl_l"]
    ao_pl_r = coup["ao_pl_r"]

    ao_ed_global = info["ao_idx"]["extended_device"]
    lookup_ed = {ao: i for i, ao in enumerate(ao_ed_global)}
    idx_pl_l_local = np.array([lookup_ed[i] for i in ao_pl_l], dtype=int)
    idx_pl_r_local = np.array([lookup_ed[i] for i in ao_pl_r], dtype=int)

    H_LL = fock[np.ix_(idx_pl_l_local, idx_pl_l_local)]
    H_RR = fock[np.ix_(idx_pl_r_local, idx_pl_r_local)]

    diff_L = _mean_diag(h00) - _mean_diag(H_LL)
    diff_R = _mean_diag(h00) - _mean_diag(H_RR)
    print("[DIAG] mean(H00_bulk) – mean(H_LL_ED) = {:+.2e} Ha".format(diff_L))
    print("[DIAG] mean(H00_bulk) – mean(H_RR_ED) = {:+.2e} Ha".format(diff_R))

    # Σ(E) shift equivalence test at E=0
    E_test = np.array([0.0])
    Sigma_shift_arg = _sig_grid(E_test, lead_json=lead_json, potential_shift=delta1_after, eta=args.eta, overwrite=False)[0]
    Sigma_energy_shift = _sig_grid(E_test - delta1_after, lead_json=lead_json, potential_shift=0.0, eta=args.eta, overwrite=False)[0]
    sig_diff = np.max(np.abs(Sigma_shift_arg - Sigma_energy_shift))
    print("[DIAG] max|Σ(E,Δ) − Σ(E−Δ)| = {:.2e}".format(sig_diff))

    # Σ diagnostics over a small energy grid
    E_grid_small = np.linspace(-0.5, 0.5, 11)
    Sigma_small = _sig_grid(E_grid_small, lead_json=lead_json, potential_shift=delta1_after, eta=args.eta, overwrite=False)
    diag = _sig_diag(E_grid_small, Sigma_small)
    print("[DIAG] Σ Hermiticity max  = {:.2e}".format(diag["herm_res"].max()))
    print("[DIAG] min eig Γ          = {:.2e}\n".format(diag["gamma_min"].min()))

    # 5. Save results -------------------------------------------------------------------
    np.save(args.out / "dm_final.npy", dm)
    np.save(args.out / "fock_final.npy", fock)
    with open(args.out / "energy.txt", "w") as fh:
        fh.write(f"E_tot = {E_tot:.10f} Ha\n")
    print(
        f"[run_negf_scf] Converged in {n_done} iterations  →  outputs written to {args.out}"
    )

    # ------------------------------------------------------------------
    # Optional transmission calculation on real energy grid
    # ------------------------------------------------------------------
    if args.transmission:
        print("[run_negf_scf] Computing transmission T(E) …")
        E_grid = np.linspace(args.e_min, args.e_max, args.n_e)

        # Recompute Σ(E) on real grid (fast vectorised path)
        Sigma_real = lead_self_energy.compute_sigma_grid(
            E_grid,
            lead_json=lead_json,
            eta=args.eta,
            potential_shift=delta1,
            overwrite=False,
        )

        Sigma_L_real, Sigma_R_real = _embed_sigma_grid(info, Sigma_real)

        T = np.empty_like(E_grid)
        from li14_negf.postprocessing import transmission as _Tfunc

        for k, E in enumerate(E_grid):
            G_E = scf.device_green_function(
                E,
                fock,  # use converged Fock
                S_d,
                Sigma_L=Sigma_L_real[k],
                Sigma_R=Sigma_R_real[k],
                eta=args.eta,
            )
            Gamma_L_E = 1j * (Sigma_L_real[k] - Sigma_L_real[k].conj().T)
            Gamma_R_E = 1j * (Sigma_R_real[k] - Sigma_R_real[k].conj().T)
            T[k] = _Tfunc(G_E, Gamma_L_E, Gamma_R_E)

        np.save(args.out / "T_vs_E.npy", np.vstack([E_grid, T]))
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(5, 3))
            plt.plot(E_grid * 27.2114, T)
            plt.xlabel("Energy – μ (eV)")
            plt.ylabel("T(E)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.out / "transmission_scf.png", dpi=150)
            print("[run_negf_scf] Transmission plot saved to transmission_scf.png")
        except ImportError:
            pass


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1) 
"""Self-consistent DFT + NEGF driver utilities.

The module now supports the full *embedded* workflow used in the Li-chain
demonstrator:

1.  Build an initial density matrix from neutral-atom superposition and obtain
    a **one-shot Fock** (no expensive SCF) – `initialize_density`.
2.  Construct equilibrium or bias contours for Green-function integration –
    `build_contour`, `device_green_function`, `integrate_density_matrix`.
3.  Fetch or compute lead self-energy grids Σ_L/R(E) with caching –
    `get_sigma_grid`.
4.  Iterate `scf_step` inside `run_scf_loop`, where each step:
      • embeds Σ_L/R(E) into the extended-device space,
      • integrates the density via NEGF,
      • applies Δ₁ alignment and optional Shift-2 neutrality correction,
      • updates the Fock with a single PySCF call,
      • performs DIIS + Kerker mixing.

The routines therefore cover the complete Σ-aware SCF cycle; only advanced
features such as bias transport and many-body self-energies remain future
work.

The API will grow as Tasks 4.2–4.9 are implemented; for now we expose:

* ``initialize_density(mol, xc="lda,vwn", smear_sigma=0.001)`` – returns
  ``dm0`` and ``fock0`` suitable for the first Green-function integration.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from pyscf import dft, scf, gto
from pyscf.scf import addons as scf_addons

__all__ = [
    "initialize_density",
    "get_sigma_grid",
    "build_equilibrium_contour",
    "build_real_axis_supplement",
    "build_contour",
    "device_green_function",
    "integrate_density_matrix",
    "update_fock",
    "has_converged",
    "scf_step",
    "run_scf_loop",
]


def _atomic_superposition_dm(mol: gto.Mole) -> np.ndarray:
    """Return AO-basis density matrix from neutral-atom superposition.

    PySCF provides this helper in ``pyscf.scf.hf``; we wrap it for clarity and
    to keep a single import location.
    """

    from pyscf.scf import hf as scf_hf  # local import to avoid polluting top-level

    dm = scf_hf.init_guess_by_atom(mol)
    dm = np.asarray(dm, dtype=np.float64)
    return dm


def initialize_density(
    mol: gto.Mole,
    *,
    xc: str = "lda,vwn",
    smear_sigma: float = 0.001,  # Ha  ≈ 300 K
    verbose: int = 4,
    max_memory: int = 10000,
    max_cycle: int = 150,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return initial (dm0, fock0) for the extended device.

    Parameters
    ----------
    mol
        *PySCF* ``Mole`` containing the *extended device* atoms only.
    xc
        Exchange–correlation functional (default LDA VWN).
    smear_sigma
        Electronic temperature (Gaussian/Fermi smearing) in Hartree.

    Returns
    -------
    dm0
        Superposition density matrix (AO basis).
    fock0
        One-electron Fock matrix evaluated at *dm0* (includes XC, Hartree,
        smearing).  Use as the starting Hamiltonian in the first NEGF
        iteration.
    """

    # --- Superposition density -------------------------------------------------
    dm0 = _atomic_superposition_dm(mol)

    # --- Build corresponding Fock matrix --------------------------------------
    mf = dft.RKS(mol, xc=xc)
    mf.verbose = verbose
    mf.max_memory = max_memory
    mf.max_cycle = max_cycle

    # finite-T smearing speeds up later convergence and matches literature
    scf_addons.smearing_(mf, sigma=smear_sigma)

    fock0 = mf.get_fock(dm=dm0) # TODO: plot it

    return dm0, fock0


# -----------------------------------------------------------------------------
# Lead self-energy fetcher (Task 4.1c)
# -----------------------------------------------------------------------------


def get_sigma_grid(
    energies: np.ndarray,
    *,
    lead_json: str | Path,
    potential_shift: float = 0.0,
    eta: float = 1e-6,
) -> np.ndarray:
    """Return Σ(E) for the specified *energies*, using cache where possible.

    This thin wrapper around :func:`lead_self_energy.compute_sigma_grid` sets
    the *potential_shift* and ensures consistency with the cache file.
    """

    from pathlib import Path

    from .lead_self_energy import compute_sigma_grid

    Sigma = compute_sigma_grid(
        energies,
        lead_json=lead_json,
        eta=eta,
        potential_shift=potential_shift,
        overwrite=True,  # ensure fresh grid with correct Σ sign convention
    )

    return Sigma


# -----------------------------------------------------------------------------
# Contour integration helpers (Task 4.2)
# -----------------------------------------------------------------------------


def _fermi_dirac(E: np.ndarray | float, mu: float = 0.0, kT: float = 0.001) -> np.ndarray | float:
    """Fermi–Dirac distribution for diagnostic/testing purposes (not used yet)."""

    return 1.0 / (np.exp((E - mu) / kT) + 1.0)


def build_equilibrium_contour(
    mu: float = 0.0,
    radius: float = 5.0,  # Ha – semi-ellipse radius on imag axis
    n_poles: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Return energies **(complex)** and weights for the upper-half semi-ellipse contour.

    Following TranSIESTA and ATK practice we discretise the equilibrium Green-
    function integral along a semi-circular path in the upper half-plane plus a
    line parallel to the imaginary axis (semi-ellipse).  For now we implement a
    simple semi-circle parameterisation

        E(θ) = μ + R e^{i θ},   θ ∈ [0, π]

    Using *n_poles* Gauss–Legendre points mapped from [-1,1] → [0, π].

    Returns
    -------
    E_c : np.ndarray (complex)
        Contour energies.
    w_c : np.ndarray (real)
        Quadrature weights (already include the Jacobian R i e^{i θ}).
    """

    # Gauss-Legendre on [-1,1]
    x, w = np.polynomial.legendre.leggauss(n_poles)
    theta = 0.5 * (x + 1.0) * np.pi  # map to [0, π]

    # Energies on semi-circle
    E_c = mu + radius * np.exp(1j * theta)

    # Jacobian dE/dθ = i R e^{i θ}
    jac = 1j * radius * np.exp(1j * theta)
    w_c = w * jac * 0.5 * np.pi  # account for interval scaling

    return E_c, w_c


def build_real_axis_supplement(
    mu: float = 0.0,
    window: float = 5.0,  # Ha; integrate from μ−window → μ+window
    n_points: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Return energies (real) and weights for Ozaki real-axis supplement.

    Uses Gauss–Legendre quadrature on the symmetric interval around μ.
    """

    x, w = np.polynomial.legendre.leggauss(n_points)
    # Map x ∈ [-1,1] → E ∈ [μ-window, μ+window]
    E_r = mu + x * window
    w_r = w * window  # Jacobian (window)
    return E_r, w_r


# -----------------------------------------------------------------------------
# Unified contour builder
# -----------------------------------------------------------------------------


def build_contour(
    *,
    mode: str = "equilibrium",  # "equilibrium" | "bias"
    mu: float = 0.0,
    bias: float = 0.0,  # applied bias (V) → shifts μ_L/R by ±eV/2 (Ha)
    radius: float | None = None,
    n_poles: int | None = None,
    window: float | None = None,
    n_real: int | None = None,
) -> dict[str, np.ndarray]:
    """Return integration energies & weights for NEGF contour integration.

    Parameters
    ----------
    mode
        "equilibrium" – single chemical potential *mu*.
        "bias"        – two potentials: μ_L/R = μ ± bias/2 (bias given in Hartree).
    bias
        Electrochemical bias (Ha).  Ignored in *equilibrium* mode.

    Returns
    -------
    dict with keys:
        "E_c", "w_c" – complex contour (upper half-plane semi-circle)
        "E_r", "w_r" – real-axis supplement (array can contain both left/right if bias)
        "E": E_all – combined energies
        "w": w_all – combined weights
        "mu_L", "mu_R" – chemical potentials used
    """

    from scipy.constants import elementary_charge as e_c  # 1.602e-19 J
    from . import config as _cfg

    # Translate bias voltage (V) to Hartree (Ha): 1 V = 1/e_c J per electron 
    # but we assume bias already in Ha for simplicity. If user passes volts they
    # should convert.

    if mode not in {"equilibrium", "bias"}:
        raise ValueError("mode must be 'equilibrium' or 'bias'")

    mu_L = mu_R = mu
    if mode == "bias":
        half = bias / 2.0
        mu_L = mu + half
        mu_R = mu - half

    if radius is None:
        radius = _cfg.get_param("contour_radius")
    if n_poles is None:
        n_poles = _cfg.get_param("contour_n_poles")
    if window is None:
        window = _cfg.get_param("realaxis_window")
    if n_real is None:
        n_real = _cfg.get_param("realaxis_n_points")

    # Build contour around average μ (same complex path)
    E_c, w_c = build_equilibrium_contour(mu=mu, radius=radius, n_poles=n_poles)

    # Real‐axis supplement: separate sets for left/right if bias ≠ 0
    if n_real == 0:
        E_r = np.empty(0, dtype=float)
        w_r = np.empty(0, dtype=float)
    elif mode == "equilibrium":
        E_r, w_r = build_real_axis_supplement(mu=mu, window=window, n_points=n_real)
    else:
        E_r_L, w_r_L = build_real_axis_supplement(mu=mu_L, window=window, n_points=n_real)
        E_r_R, w_r_R = build_real_axis_supplement(mu=mu_R, window=window, n_points=n_real)
        E_r = np.concatenate([E_r_L, E_r_R])
        w_r = np.concatenate([w_r_L, w_r_R])

    E_all = np.concatenate([E_c, E_r])
    w_all = np.concatenate([w_c, w_r])

    return {
        "E_c": E_c,
        "w_c": w_c,
        "E_r": E_r,
        "w_r": w_r,
        "E": E_all,
        "w": w_all,
        "mu_L": mu_L,
        "mu_R": mu_R,
    }


# -----------------------------------------------------------------------------
# Device Green-function and density integration (Task 4.3)
# -----------------------------------------------------------------------------


def device_green_function(
    E: complex,
    H_d: np.ndarray,
    S_d: np.ndarray,
    *,
    Sigma_L: np.ndarray | None = None,
    Sigma_R: np.ndarray | None = None,
    eta: float = 1e-6,
) -> np.ndarray:
    """Return retarded device Green-function G(E).

    Parameters follow standard NEGF notation.  Σ_L/R default to 0 if not
    provided so the routine can be unit-tested before full embedding is wired
    in.
    """

    N = H_d.shape[0]
    eye = np.eye(N, dtype=np.complex128)

    Sigma_tot = 0.0
    if Sigma_L is not None:
        Sigma_tot += Sigma_L
    if Sigma_R is not None:
        Sigma_tot += Sigma_R

    A = (E + 1j * eta) * S_d - H_d - Sigma_tot
    G = np.linalg.inv(A)
    return G


def integrate_density_matrix(
    contour: dict[str, np.ndarray],
    *,
    H_d: np.ndarray,
    S_d: np.ndarray,
    Sigma_L_grid: np.ndarray | None = None,
    Sigma_R_grid: np.ndarray | None = None,
    eta: float = 1e-6,
) -> np.ndarray:
    """Integrate Green-functions over contour to obtain density matrix D.

    Very simplified (equilibrium) implementation:  uses

        D = (1/2πi) Σ w_c G(E_c)

    where (E_c, w_c) come from :func:`build_equilibrium_contour`.  Real-axis
    supplement is presently ignored for speed; will be added in 4.3b once f(E)
    weighting is needed.
    """

    if "E" not in contour or "w" not in contour:
        raise KeyError("Contour dictionary must contain unified 'E' and 'w' arrays. Ensure build_contour() >= real-axis implementation is used.")

    E_all = contour["E"]
    w_all = contour["w"]

    if Sigma_L_grid is None:
        Sigma_L_grid = [None] * len(E_all)
    if Sigma_R_grid is None:
        Sigma_R_grid = [None] * len(E_all)

    D = np.zeros_like(H_d, dtype=np.complex128)

    for Ei, wi, sL, sR in zip(E_all, w_all, Sigma_L_grid, Sigma_R_grid):
        G = device_green_function(Ei, H_d, S_d, Sigma_L=sL, Sigma_R=sR, eta=eta)
        D += wi * G

    # Density matrix must be Hermitian; take Hermitian part / (2π i)
    D = (D / (2j * np.pi)).real
    D = 0.5 * (D + D.T)  # enforce symmetry numerically

    return D


def update_fock(
    mol: gto.Mole,
    density_matrix: np.ndarray,
    *,
    xc: str = "lda,vwn",
    smear_sigma: float = 0.001,
    verbose: int = 4,
    max_memory: int = 10000,
    max_cycle: int = 150,
) -> np.ndarray:
    """Return new Fock matrix given *density_matrix* using PySCF one-shot call."""

    mf = dft.RKS(mol, xc=xc)
    mf.verbose = verbose
    mf.max_memory = max_memory
    mf.max_cycle = max_cycle
    scf_addons.smearing_(mf, sigma=smear_sigma)
    fock = mf.get_fock(dm=density_matrix)
    return fock


# -----------------------------------------------------------------------------
# Convergence utilities (Task 4.5c)
# -----------------------------------------------------------------------------


def has_converged(
    delta_dm: np.ndarray | float,
    delta_E: float,
    *,
    tol_dm: float = 1e-5,
    tol_E: float = 1e-6,
) -> bool:
    """Return ``True`` when SCF convergence criteria are satisfied.

    Parameters
    ----------
    delta_dm
        Density‐matrix change since last iteration (*matrix* or pre-computed
        max‐norm).  If an ``ndarray`` is passed we compute
        ``max(abs(delta_dm))``.
    delta_E
        Total‐energy change (Hartree).
    tol_dm, tol_E
        Convergence thresholds for ``max|ΔD|`` (electrons) and ``|ΔE|`` (Ha).
    """

    if isinstance(delta_dm, np.ndarray):
        dD = float(np.max(np.abs(delta_dm)))
    else:
        dD = abs(delta_dm)

    dE = abs(delta_E)

    return dD < tol_dm and dE < tol_E 

# -----------------------------------------------------------------------------
# High-level SCF driver (very first minimal implementation – Task 4.5 prototype)
# -----------------------------------------------------------------------------

from pathlib import Path
from .mixing import MixerDIIS
from .kerker import kerker_precondition
from . import config as _cfg

# Optional import guard – lead self-energy not yet wired into contour loop
try:
    from .lead_self_energy import compute_sigma_grid  # noqa: WPS433
except ModuleNotFoundError:  # pragma: no cover – doctest environments
    compute_sigma_grid = None  # type: ignore[assignment]


def scf_step(
    dm_in: np.ndarray,
    fock_in: np.ndarray,
    *,
    mol: gto.Mole,
    contour: dict[str, np.ndarray] | None = None,
    Sigma_L_grid: np.ndarray | None = None,
    Sigma_R_grid: np.ndarray | None = None,
    H_core: np.ndarray | None = None,
    S: np.ndarray | None = None,
    eta: float = 1e-6,
    apply_shift2: bool = True,
    shift2_kappa: float = 0.5,
    delta1_shift: float = 0.0,
    mixer: "MixerDIIS" | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (dm_out, fock_out, E_tot) after **one Σ-embedded NEGF iteration**.

    The function performs one full NEGF SCF step **with** lead self-energies if
    `Sigma_L_grid` / `Sigma_R_grid` are provided (as they are in
    `run_scf_loop`).  If those arguments are `None` the routine gracefully
    degrades to an isolated-device DFT step – handy for unit tests.

    Workflow inside the step
    ------------------------
    1. Integrate the density matrix on the supplied contour using the current
       Fock **plus** Σ_L/R(E).
    2. Mix the new density with DIIS + Kerker (if a mixer object is passed).
    3. Build an updated Fock via a one-shot PySCF call.
    4. Apply Δ₁ alignment shift and optional Shift-2 neutrality correction so
       that the Hamiltonian and self-energies remain on a common reference and
       charge neutrality is maintained.
    5. Return the new (density, Fock, total KS energy).

    Parameters
    ----------
    dm_in, fock_in
        Density matrix and Fock from previous iteration.
    mol
        PySCF ``Mole`` object of the extended device.
    contour
        Integration contour dictionary from :func:`build_contour`.  If *None*,
        a default equilibrium contour is constructed using project defaults.
    H_core, S
        One-electron core Hamiltonian and overlap.  If omitted they are computed
        from *mol* on-demand (cached inside the function for speed).
    eta
        Imaginary broadening for the Green-function.
    apply_shift2
        Whether to apply Shift-2 neutrality correction.
    shift2_kappa
        Proportionality constant for Shift-2 correction.

    Notes
    -----
    The *total energy* returned is the simple Kohn–Sham expression

        E_tot = ½ Tr[ D ( H_core + F ) ]   (Ha),

    which is sufficient for convergence monitoring (ΔE).  True grand-canonical
    NEGF energies with lead contributions and double-count corrections are not
    yet implemented, but they do not affect the SCF convergence logic.
    """

    # ------------------------------------------------------------------
    # Lazy compute H_core and S if not provided (expensive integrals!)
    # ------------------------------------------------------------------
    if H_core is None or S is None:
        H_core = mol.intor("int1e_kin") + mol.intor("int1e_nuc") if H_core is None else H_core
        S = mol.intor("int1e_ovlp") if S is None else S

    # ------------------------------------------------------------------
    # 1. Green-function integration – obtains *unmixed* new density matrix
    # ------------------------------------------------------------------
    if contour is None:
        contour = build_contour(mode="equilibrium")

    D_raw = integrate_density_matrix(
        contour,
        H_d=fock_in,  # use last Fock as effective Hamiltonian
        S_d=S,
        Sigma_L_grid=Sigma_L_grid,
        Sigma_R_grid=Sigma_R_grid,
        eta=eta,
    )

    # ------------------------------------------------------------------
    # 2. Mixing (simple linear for now; DIIS wired for later)
    # ------------------------------------------------------------------
    if mixer is None:
        beta = _cfg.get_param("mixing_beta")
        size = _cfg.get_param("diis_size")
        k_alpha = _cfg.get_param("kerker_alpha")
        mixer_local = MixerDIIS(
            beta=beta,
            size=size,
            precond=lambda r: kerker_precondition(r, S, alpha=k_alpha),
        )
    else:
        mixer_local = mixer

    dm_out = mixer_local.update(dm_in, D_raw)

    # ------------------------------------------------------------------
    # 3. New Fock from PySCF one-shot call
    # ------------------------------------------------------------------
    fock_out = update_fock(mol, dm_out)

    # ------------------------------------------------------------------
    # 3a. Apply constant alignment shift Δ₁ so device and leads share
    #      the same reference.  This must precede Shift-2 so the latter
    #      acts on the already aligned Hamiltonian.
    # ------------------------------------------------------------------
    if abs(delta1_shift) > 1e-16:
        np.fill_diagonal(fock_out, np.diag(fock_out) + delta1_shift)
        if H_core is not None:
            np.fill_diagonal(H_core, np.diag(H_core) + delta1_shift)

    # ------------------------------------------------------------------
    # 3b. Global Shift-2 neutrality correction (optional)
    # ------------------------------------------------------------------
    if apply_shift2:
        from .alignment import compute_shift2 as _cs2

        n_elec_target = float(mol.nelectron)
        delta2 = _cs2(dm_out, S, n_elec_target, proportionality=shift2_kappa)

        # Apply shift to Fock and H_core for downstream iterations
        np.fill_diagonal(fock_out, np.diag(fock_out) + delta2)
        if H_core is not None:
            np.fill_diagonal(H_core, np.diag(H_core) + delta2)
        if __debug__:
            print(f"[Shift-2] Δ₂ = {delta2:.2e} Ha applied for charge neutrality")

    # ------------------------------------------------------------------
    # 4. Total-energy diagnostic (KS expression)
    # ------------------------------------------------------------------
    E_tot = 0.5 * float(np.trace(dm_out @ (H_core + fock_out)).real)

    return dm_out, fock_out, E_tot


# -----------------------------------------------------------------------------
# Simple driver that repeats *scf_step* until convergence (isolated device)
# -----------------------------------------------------------------------------


def _embed_sigma_full(info: dict, sigma_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Σ_L_full, Σ_R_full embedded in ED AO dimension."""

    from .partition import extract_pl_couplings # TODO: plot it

    ao_ed = info["ao_idx"]["extended_device"]
    coup = extract_pl_couplings(info)
    ao_pl_l = coup["ao_pl_l"]
    ao_pl_r = coup["ao_pl_r"]

    lookup = {ao: i for i, ao in enumerate(ao_ed)}
    idx_l = np.array([lookup[i] for i in ao_pl_l], dtype=int)
    idx_r = np.array([lookup[i] for i in ao_pl_r], dtype=int)

    n_e, n_pl, _ = sigma_grid.shape
    n_d = len(ao_ed)

    Sigma_L = np.zeros((n_e, n_d, n_d), dtype=np.complex128)
    Sigma_R = np.zeros_like(Sigma_L)

    for k in range(n_e):
        Sig = sigma_grid[k]
        Sigma_L[k][np.ix_(idx_l, idx_l)] = Sig
        Sigma_R[k][np.ix_(idx_r, idx_r)] = Sig

    return Sigma_L, Sigma_R


def run_scf_loop(
    mol: gto.Mole,
    *,
    info: dict,
    lead_json: str | Path,
    H_bulk_onsite: np.ndarray,
    contour: dict[str, np.ndarray],
    n_iter: int = 50,
    tol_dm: float = 1e-5,
    tol_E: float = 1e-6,
    shift_tol: float = 1e-3,
    # Alignment-freeze thresholds: when both are satisfied, we stop
    # recomputing Δ₁ and refreshing Σ to avoid noise in late SCF cycles.
    align_freeze_dm: float = 1e-3,   # max|ΔD| threshold (e)
    align_freeze_E: float = 1e-3,    # |ΔE| threshold (Ha)
    diis_warmup: int = 3,
    verbose: bool = True,
    transmission_probe: dict | None = None,
    **step_kwargs,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Run a minimal SCF loop on the isolated extended device.

    Returns
    -------
    (dm, fock, E_tot, n_done)
        Final density matrix, Fock, total energy and number of iterations actually performed.
    """

    from .alignment import compute_shift1, apply_shift as _apply_shift
    from .lead_self_energy import compute_sigma_grid
    from .diagnostics import gf_hermiticity, gf_identity_residual  # local import for probe only

    # Precompute energy list once
    energies = contour["E"]

    # Initial guess
    dm, fock = initialize_density(mol)

    # ------------------------------------------------------------------
    # Keep info['H_core'] in sync with the latest Fock so compute_shift1
    # always sees the current on-site energies when evaluating Δ₁.
    # ------------------------------------------------------------------
    info_H = info["H_core"]
    info_H[np.ix_(info["ao_idx"]["extended_device"], info["ao_idx"]["extended_device"])] = fock

    delta1_prev = compute_shift1(info, H_bulk_onsite=H_bulk_onsite)

    # Apply initial Δ₁ alignment shift now that we have delta1_prev
    fock = _apply_shift(fock, delta1_prev)

    align_frozen = False  # will turn True once SCF is near convergence

    sigma_block = compute_sigma_grid(energies, lead_json=lead_json, potential_shift=delta1_prev)
    Sigma_L, Sigma_R = _embed_sigma_full(info, sigma_block) # TODO: plot it

    step_kwargs = dict(step_kwargs)  # copy
    step_kwargs.update({
        "contour": contour,
    })

    ao_ed = info["ao_idx"]["extended_device"]
    H_core = info["H_core"][np.ix_(ao_ed, ao_ed)]
    E_prev = 0.5 * float(np.trace(dm @ (H_core + fock)).real)

    # --------------------------------------------------------------
    # Pre-slice overlap matrix for the extended device once
    # --------------------------------------------------------------
    ao_ed_global = info["ao_idx"]["extended_device"]
    S_ed_full = info["S"][np.ix_(ao_ed_global, ao_ed_global)]

    # --------------------------------------------------------------
    # Diagnostics CSV path (enabled only when transmission_probe active)
    # --------------------------------------------------------------
    out_dir_global: Path | None = None

    # Precompute data for optional transmission probe
    if transmission_probe is not None:
        E_probe = transmission_probe.get("E_grid")
        probe_every = int(transmission_probe.get("every", 5))
        probe_first_n = int(transmission_probe.get("first_n", 5))
        out_dir = Path(transmission_probe.get("out_dir", "."))
        eta_probe = step_kwargs.get("eta", 1e-6)

        out_dir_global = out_dir

        # Slice overlap once for Green-function evaluations
        S_probe = S_ed_full

        # Helper closure
        # Return tuple with HOMO, LUMO, and gap in eV units
        def _do_probe(iter_idx: int, delta1_val: float) -> tuple[float, float, float, float, float, float, float, float, float]:
            from .lead_self_energy import compute_sigma_grid as _sig_grid

            Sigma_real = _sig_grid(
                E_probe,
                lead_json=lead_json,
                potential_shift=delta1_val,
                overwrite=False,
            )
            Sigma_L_real, Sigma_R_real = _embed_sigma_full(info, Sigma_real)

            T = np.empty_like(E_probe, dtype=float)
            for k, E in enumerate(E_probe):
                G_E = device_green_function(
                    E,
                    fock,
                    S_probe,
                    Sigma_L=Sigma_L_real[k],
                    Sigma_R=Sigma_R_real[k],
                    eta=eta_probe,
                )
                Gamma_L_E = 1j * (Sigma_L_real[k] - Sigma_L_real[k].conj().T)
                Gamma_R_E = 1j * (Sigma_R_real[k] - Sigma_R_real[k].conj().T)
                T[k] = float(np.real(np.trace(Gamma_L_E @ G_E @ Gamma_R_E @ G_E.conj().T)))

            fname_np = out_dir / f"T_vs_E_iter{iter_idx:03d}.npy"
            np.save(fname_np, np.vstack([E_probe, T]))
            if verbose:
                print(f"[probe] Saved transmission spectrum → {fname_np.name}")

            # Diagnostics at energy closest to μ (≈ 0 Ha)
            ef_idx = int(np.argmin(np.abs(E_probe)))
            Gamma_L_EF = 1j * (Sigma_L_real[ef_idx] - Sigma_L_real[ef_idx].conj().T)
            Gamma_R_EF = 1j * (Sigma_R_real[ef_idx] - Sigma_R_real[ef_idx].conj().T)

            min_eig_L = float(np.min(np.linalg.eigvalsh(Gamma_L_EF.real)))
            min_eig_R = float(np.min(np.linalg.eigvalsh(Gamma_R_EF.real)))

            # Channel count proxy: Tr[Γ_L S] / (2π)
            N_open = float(np.real(np.trace(Gamma_L_EF @ S_probe)) / (2 * np.pi))

            max_T_val = float(np.max(T))

            # ------------------------------------------------------
            # New: Green-function diagnostics at E ≈ μ (ef_idx)
            # ------------------------------------------------------
            G_EF = device_green_function(
                E_probe[ef_idx],
                fock,
                S_probe,
                Sigma_L=Sigma_L_real[ef_idx],
                Sigma_R=Sigma_R_real[ef_idx],
                eta=eta_probe,
            )

            Sigma_tot_EF = Sigma_L_real[ef_idx] + Sigma_R_real[ef_idx]
            herm_res = gf_hermiticity(G_EF)
            dyson_res = gf_identity_residual(
                E_probe[ef_idx],
                G_EF,
                fock,
                S_probe,
                Sigma_tot_EF,
            )

            # ------------------------------------------------------
            # HOMO / LUMO diagnostics (generalised eigenproblem)
            # ------------------------------------------------------
            try:
                from scipy.linalg import eigh  # type: ignore

                # Generalised eigenvalues F C = S C ε (RKS: double occupancy)
                eps = eigh(fock, S_probe, eigvals_only=True)

                n_occ = mol.nelectron // 2  # assuming closed-shell RKS
                # Convert to eV
                _eV_per_Ha = 27.2114
                homo_Ha = eps[n_occ - 1] if n_occ >= 1 else np.nan
                lumo_Ha = eps[n_occ] if n_occ < len(eps) else np.nan

                homo = float(homo_Ha * _eV_per_Ha)
                lumo = float(lumo_Ha * _eV_per_Ha)
                gap = float((lumo_Ha - homo_Ha) * _eV_per_Ha) if np.isfinite(homo_Ha) and np.isfinite(lumo_Ha) else float('nan')
            except Exception:
                homo = float('nan')
                lumo = float('nan')
                gap = float('nan')

            try:
                from matplotlib import pyplot as plt  # optional
            except ImportError:
                return (
                    max_T_val,
                    min_eig_L,
                    min_eig_R,
                    N_open,
                    herm_res,
                    dyson_res,
                    homo,
                    lumo,
                    gap,
                )

            plt.figure(figsize=(5, 3))
            plt.plot(E_probe * 27.2114, T)
            plt.xlabel("Energy – μ (eV)")
            plt.ylabel("T(E)")
            plt.title(f"Iteration {iter_idx}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            png_path = out_dir / f"transmission_iter{iter_idx:03d}.png"
            plt.savefig(png_path, dpi=150)
            plt.close()
            if verbose:
                print(f"[probe] Transmission plot saved to {png_path.name}")

            return (
                max_T_val,
                min_eig_L,
                min_eig_R,
                N_open,
                herm_res,
                dyson_res,
                homo,
                lumo,
                gap,
            )

        # Initial probe before first SCF iteration
        maxT0, minEigL0, minEigR0, Nopen0, herm0, dyson0, homo0, lumo0, gap0 = _do_probe(0, delta1_prev)
        # delta2 for iter 0 is not defined yet (nan)
        diag_csv = out_dir / "scf_log.csv"
        out_dir_global = out_dir  # record for later relative paths
        with open(diag_csv, "w") as fh:
            fh.write(
                "iter,E_tot,delta_E,max_dD,delta1,delta2,maxT,minEigL,minEigR,Nopen,hermG,dysonRes,HOMO_eV,LUMO_eV,GAP_eV\n"
            )
        with open(diag_csv, "a") as fh:
            fh.write(
                f"0,{E_prev:.8f},0.0,0.0,{delta1_prev:.6e},nan,{maxT0:.6e},{minEigL0:.4e},{minEigR0:.4e},{Nopen0:.4e},nan,nan,{homo0:.6e},{lumo0:.6e},{gap0:.6e}\n"
            )

    # ------------------------------------------------------------------
    # Prepare persistent mixer (DIIS) and spike-detection vars
    # ------------------------------------------------------------------

    beta = _cfg.get_param("mixing_beta")
    size = _cfg.get_param("diis_size")
    k_alpha = _cfg.get_param("kerker_alpha")
    mixer = MixerDIIS(
        beta=beta,
        size=size,
        precond=lambda r: kerker_precondition(r, S_ed_full, alpha=k_alpha),
    )

    delta_dm_prev = float("inf")

    for it in range(1, n_iter + 1):
        dm_prev = dm.copy()
        use_mixer = mixer if it > diis_warmup else None

        dm, fock, E_tot = scf_step(
            dm,
            fock,
            mol=mol,
            Sigma_L_grid=Sigma_L,
            Sigma_R_grid=Sigma_R,
            S=S_ed_full,
            **step_kwargs,
            delta1_shift=delta1_prev,
            mixer=use_mixer,
        )

        # Compute density-matrix change *before* any alignment-freeze logic so
        # the latest value is available when evaluating the freeze criteria.
        delta_dm = np.max(np.abs(dm - dm_prev))

        # ------------------------------------------------------------------
        # Keep info['H_core'] in sync with the *latest* Fock so that any
        # subsequent compute_shift1() call (inside this loop or by the caller)
        # sees the current on-site energies.  Missing this update led to a
        # ~1 Ha mismatch in the diagnostics block because Δ₁ was evaluated
        # against an outdated Hamiltonian.
        # ------------------------------------------------------------------
        info["H_core"][np.ix_(ao_ed, ao_ed)] = fock

        # ------------------------------------------------------------------
        # Δ₁ update / Σ-refresh controller (with freeze)
        # ------------------------------------------------------------------
        if not align_frozen:
            delta1_curr = compute_shift1(info, H_bulk_onsite=H_bulk_onsite)

            if abs(delta1_curr - delta1_prev) > shift_tol:
                if verbose:
                    print(
                        f"[Σ-refresh] |Δ₁ change| = {(delta1_curr - delta1_prev):+.2e} Ha → recompute Σ"
                    )
                sigma_block = compute_sigma_grid(
                    energies,
                    lead_json=lead_json,
                    potential_shift=delta1_curr,
                    overwrite=False,
                )
                Sigma_L, Sigma_R = _embed_sigma_full(info, sigma_block)

                # Consistently update current Fock matrix by the Δ₁ change so
                # the Hamiltonian keeps the same reference as the new Σ grid.
                delta_shift_diff = delta1_curr - delta1_prev
                if abs(delta_shift_diff) > 1e-16:
                    np.fill_diagonal(fock, np.diag(fock) + delta_shift_diff)
                delta1_prev = delta1_curr

            # Evaluate freeze criteria now that *delta_dm* and ΔE for this
            # iteration are known.
            if it > 1 and delta_dm < align_freeze_dm and abs(E_tot - E_prev) < align_freeze_E:
                align_frozen = True
                if verbose:
                    print(
                        f"[alignment] Criteria met (|ΔE|<{align_freeze_E}, max|ΔD|<{align_freeze_dm}) – Δ₁ frozen."
                    )

        # delta_dm already computed earlier
        # ----------------------------------------------------------
        # Detect divergence spike (ΔD grows >2×) and reset DIIS
        # ----------------------------------------------------------
        if it > diis_warmup and delta_dm_prev < float("inf") and delta_dm > 2.0 * delta_dm_prev:
            if verbose:
                print(
                    f"[DIIS] |ΔD| spike ({delta_dm:.2e} > 2×{delta_dm_prev:.2e}) – resetting mixer history."
                )
            mixer.reset()

        delta_E_val = E_tot - E_prev
        print(
            f"[SCF] iter {it:02d}  E_tot = {E_tot:.6f}  ΔE = {delta_E_val:+.3e}  max|ΔD| = {delta_dm:.2e}"
        )

        # Record for next iteration/spike detection
        delta_dm_prev = delta_dm
        E_prev_old = E_prev
        E_prev = E_tot  # update after computing ΔE for CSV

        if has_converged(delta_dm, delta_E_val, tol_dm=tol_dm, tol_E=tol_E):
            if verbose:
                print("[SCF] Converged.")

            # Final sync – make absolutely sure the caller sees the converged
            # Fock embedded back into info['H_core'] before we return.
            info["H_core"][np.ix_(ao_ed, ao_ed)] = fock

            return dm, fock, E_tot, it

        # --------------------------------------------------------------
        # Mid-loop transmission spectrum (debugging aid)
        # --------------------------------------------------------------
        maxT_iter = float('nan')
        minEigL_iter = float('nan')
        minEigR_iter = float('nan')
        Nopen_iter = float('nan')
        herm_iter = float('nan')
        dyson_iter = float('nan')
        homo_iter = float('nan')
        lumo_iter = float('nan')
        gap_iter = float('nan')

        if transmission_probe is not None and (it <= probe_first_n or (probe_every > 0 and it % probe_every == 0)):
            (
                maxT_iter,
                minEigL_iter,
                minEigR_iter,
                Nopen_iter,
                herm_iter,
                dyson_iter,
                homo_iter,
                lumo_iter,
                gap_iter,
            ) = _do_probe(it, delta1_prev)

        # Compute delta2 value for diagnostics
        from .alignment import compute_shift2 as _cs2
        n_elec_target = float(mol.nelectron)
        shift2_kappa = step_kwargs.get("shift2_kappa", 0.5)
        delta2_curr = _cs2(dm, S_ed_full, n_elec_target, proportionality=shift2_kappa)

        # Append row only when CSV diagnostics enabled
        if out_dir_global is not None:
            diag_csv = out_dir_global / "scf_log.csv"
            with open(diag_csv, "a") as fh:
                fh.write(
                    f"{it},{E_tot:.8f},{delta_E_val:.6e},{delta_dm:.6e},{delta1_prev:.6e},{delta2_curr:.6e},{maxT_iter:.6e},{minEigL_iter:.4e},{minEigR_iter:.4e},{Nopen_iter:.4e},{herm_iter:.3e},{dyson_iter:.3e},{homo_iter:.6e},{lumo_iter:.6e},{gap_iter:.6e}\n"
                )

    raise RuntimeError("SCF did not converge within the maximum number of iterations") 
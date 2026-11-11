"""Sector levels, variational truncated-CI bounds, contaminant weights and energy certificates (F41, F42).

Tags and formulas F1-F19 as in :mod:`thc_engine_cqe.ft.primitives` (docs/m3-cost-model.md); F39
and F40 are in :mod:`thc_engine_cqe.ft.acceptance`, F41 and F42 here. Energies in Hartree,
variances Ha^2, costs Toffolis. H' is the selected QPE encoding's model Hamiltonian, E_0' its
ground energy, E1lb the lower bound on the excited levels that rule B (F40) and the filters
(F43) rely on.

The sector of E1lb (F41) [derived]
---------------------------------
Rule B is sound if every eigenstate psi_j of H' (j >= 1) that the hand-off state populates lies
at or above E1lb; p_j = 0 outside the symmetry sector that both the preparation and the walk
conserve. :func:`sector_levels` reads the candidates off the model spectrum of H'
(``ft.acceptance.model_spectrum``):

* HF (a closed-shell determinant) and the exact iterates (spin-adapted generators, exact
  exponentials, the masked residual of F2a) conserve (N, S_z, S^2, irrep): E1lb refers to the
  lowest excited singlet of the ground state's irrep, M3's E1 on H' ("e1_singlet").
* Compiled elementwise iterates (F1, F3) are products of b-bit rounded spin-orbital excitation
  rotations and break S^2. Either E1lb refers to the lowest excited level of the irrep of any S
  ("e1_any_spin"), or the contaminant weight q_c, the weight on eigenstates of another spin than
  psi_0 below the threshold (:func:`contaminant_weight`), is budgeted:
  eps_id <- eps_id + q_c / (p_0 K_0) (``ft.acceptance.acceptance_stats(contaminant=q_c)`` with the
  singlet E1lb; the full-sector kernel term of ``acceptance_stats`` already holds every spin).
* THC breaks the point group at the 1e-5 level [design]: with THC encodings eps_id uses the
  lowest excited singlet of any irrep ("e1_any_irrep"). ``sector_levels(det_irreps=...)``
  measures the breaking as the largest weight of an eigenvector outside its dominant irrep.
* E1lb = E_1(H') - 0.1 mHa where the model spectrum is computed (N <= 8), else
  E_1(H) - 1 mHa (:func:`e1_lower_bound`) [assumption: |E_1(H') - E_1(H)| <= 1 mHa; observed
  [build check] +0.75, +0.56 and -0.06 mHa on n2_r2.8, n2_r2.4 and n2s with the selected SF
  encodings (L = 13, 16, 17)].

Observed [build check] (tests/test_ft_certify.py, mHa above E_0): on the exact H, n2s
(n2_r2.0) has the same-irrep singlet at 191.734 and the lowest same-irrep level of any spin, a
quintet, at 20.892; h6s 104.666 and 88.557 (a triplet). On stretched N2 that quintet falls to
4.35 (n2_r2.4) and 1.21 mHa (n2_r2.8) above E_0' of the selected SF encodings: a spin-broken
compiled state there has no E1lb of any spin with a margin, and only the contaminant budget
remains. The compiled spin-orbital Psi~_1 of n2s (classical first step) has q_c / p_0 = 1.19e-4
below theta = CISD(H') + dE.

Truncated CI (F42) [derived]
---------------------------
E_CI(l) is the lowest eigenvalue of P_l H P_l, P_l the projector on the determinants of the
(n_alpha, n_beta) sector whose excitation level relative to HF (the number of electrons, both
spins, in orbitals outside the HF occupation; :func:`excitation_levels`) is <= l: l = 2 is CISD,
l = 3 CISDT (:func:`truncated_ci_energy`). By the variational principle E_CI(l) >= E_0, the
lowest eigenvalue of the sector, and E_CI(l) <= E_HF = E_CI(0). The level is a function of the
spatial occupation numbers, which S_+ and S_- conserve, so P_l commutes with S_z, S^2 and the
point group, and its eigenvectors are spin and symmetry eigenstates; the lowest need not be a
singlet. As a certificate of E_0 any of them counts (the variational principle needs no spin),
and the lowest is the tightest. On stretched N2 CAS(6,6) the CISD minimum is a quintet
(S^2 = 6) and the CISDT minimum the septet of six open shells, all of whose determinants lie at
level 3 (mHa above E_0' of H', in brackets the exact H [build check]):

    system    CISD               singlet CISD       CISDT (septet)
    n2_r2.0   134.215 (133.608)  159.641 (159.610)  54.752 (54.135)
    n2_r2.4   225.686 (224.810)  263.969 (263.394)   9.241 (7.494)
    n2_r2.8   293.495 (292.418)  331.066 (330.576)   2.455 (0.535)

On n2_r2.4 the singlet CISD lies above E_1' (230.863 mHa), the quintet CISD 5.18 mHa below it.
h6s: CISD 54.506, CISDT 51.399 mHa, both singlets.

The eigenvalue is found by Lanczos with full reorthogonalization on the P space, the start
vector HF plus a seeded random vector of norm 1/2 (every spin and irrep sector present, so the
minimum over all of them is found), H applied in the full CI space (``fci_ops.Hamiltonian``; one
application per iteration), so N = 10 is feasible (876 CISD determinants of 63,504 at N = 10).
Iteration stops when the Lanczos residual ||P H P x - theta x|| <= tol, so theta lies within tol
of an eigenvalue of P H P (and ~tol^2/gap from it), the lowest once the Krylov space has resolved
it, which the start vector's random part ensures at the default tol (tested against dense solves).
The returned energy is the Rayleigh quotient of the Ritz vector x (one more application): a
variational upper bound on E_0 at any tol.

Energy certificates (F42) [derived; F13, F17]
---------------------------------------------
A certificate is an upper bound U >= E_0' with its cost (:class:`EnergyCertificate`,
:func:`energy_certificates`):

    source        U                         extra cost
    classical     E_CISD(H') (CISDT opt.)   0
    sampled       E^_j + z sigma            S_j (C(Psi_j) + C_rot), S_j = (2z)^2 V_E(Psi_j) / m_j^2
    ae            as sampled                N_j = 4 pi kappa lambda z / m_j applications of
                                            C(Psi_j) + C_W + Q_A / 2 [estimate: z-scaling of a
                                            non-Gaussian error]
    loop_energy   E^_j + z eps_j            none beyond C_loop(j + 1)
    oracle        E_0'                      0 (reported only)

with sigma = m_j / (2z) and m_j = E1lb - 2 dE - E_j, so that U = E_j + m_j/2 lies in
[E_j, E1lb - 2 dE] whenever |E^_j - E_j| <= z sigma (the planned U is reported, E^_j = E_j). The
realized bound is U^ = U + delta, delta ~ N(0, sigma^2): the certificate carries sigma
(``EnergyCertificate.sigma``), and rule B costs the verification runs it then needs by their
expectation over U^ (``ft.acceptance.expected_verification_runs``, ``ft.handoff.rule_b_table``).
The sampled energy is F13's DF-grouped estimator (``ft.residual.energy_sampling`` with
sigma in place of eps_ls), the AE energy F13's option (``ft.residual.energy_ae``,
Q_A = Q(Psi_j) + Q_walk - 2N + 1). E_j is the record's energy of H; the energy of H' in Psi_j
differs by <Psi_j|H' - H|Psi_j>, which the certificate does not budget [assumption: below the
dE margins for the selected encodings, |de_fact| <= 0.5 mHa by F9]. A certificate is returned
only if U <= E1lb - 2 dE, the weakest condition under which any form uses U (the window-model
bracket of B_U is non-empty; B_L's and the E1lb-referenced forms' termination conditions are
stricter). ``loop_iterate`` is the iterate up to which the loop must run before U exists:
0 (classical, oracle), j (dedicated energies of Psi_j), j + 1 (loop energies, conservative for
rules that measure E_j in iteration j - 1). With CISD, rule B on n2_r2.0 is B_U with one run
(margin E1lb - U - 2 dE = 55.6 mHa), on n2_r2.4 B_L (CISD's margin 3.08 mHa needs three runs
under B_U), and n2_r2.8 has no CISD certificate (``ft.acceptance.acceptance_bracket``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import scipy.linalg

from ..fci_ops import Hamiltonian
from ..integrals import ActiveSpace
from .acceptance import ModelSpectrum
from .qpe import BlockEncoding
from .residual import PrepParams, energy_ae, energy_sampling
from .trajectory import TrajectoryRecord

__all__ = [
    "CERT_SOURCES",
    "E1_MARGIN_MODEL",
    "E1_MARGIN_EXACT",
    "e1_lower_bound",
    "sector_levels",
    "excitation_levels",
    "truncated_ci_energy",
    "contaminant_weight",
    "EnergyCertificate",
    "energy_certificates",
]

CERT_SOURCES = ("classical", "sampled", "ae", "loop_energy", "oracle")
#: F41: E1lb = E_1(H') - E1_MARGIN_MODEL with the model spectrum, E_1(H) - E1_MARGIN_EXACT without it.
E1_MARGIN_MODEL = 1e-4
E1_MARGIN_EXACT = 1e-3
# <S^2> below this is a singlet (as ft.acceptance).
_SINGLET_TOL = 1e-6
# Seed of the random part of the Lanczos start vector.
_LANCZOS_SEED = 20261108
# Largest number of Lanczos iterations (the P space is exhausted earlier below this dimension).
_LANCZOS_MAX = 1000
# Relative size of a Lanczos beta below which the Krylov space is invariant.
_BREAKDOWN_RTOL = 1e-13
# Sources whose certificate costs nothing beyond the loop.
_FREE_SOURCES = ("classical", "loop_energy", "oracle")
_LEVEL_KEYS = ("e1_singlet", "e1_any_spin", "e1_any_irrep", "e1_any")


# ------------------------------------------------------------------ validation helpers
def _finite(value, name: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _positive(value, name: str) -> float:
    v = float(value)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return v


def _nonnegative(value, name: str) -> float:
    v = float(value)
    if not (math.isfinite(v) and v >= 0.0):
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return v


def _int_at_least(value, low: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < low:
        raise ValueError(f"{name} must be an integer >= {low}, got {value!r}")
    return int(value)


def _check_spec(spec) -> ModelSpectrum:
    if not isinstance(spec, ModelSpectrum):
        raise ValueError(f"spec must be a ModelSpectrum (ft.acceptance.model_spectrum), got {type(spec).__name__}")
    return spec


# ------------------------------------------------------------------ the sector of E1lb (F41)
def e1_lower_bound(e1: float, *, model_spectrum: bool) -> float:
    """F41's E1lb: E_1(H') - 0.1 mHa from the model spectrum, else E_1(H) - 1 mHa [assumption]."""
    return _finite(e1, "e1") - (E1_MARGIN_MODEL if model_spectrum else E1_MARGIN_EXACT)


def _level(spec: ModelSpectrum, mask: np.ndarray) -> int | None:
    idx = np.flatnonzero(mask)
    idx = idx[idx >= 1]
    return int(idx[0]) if idx.size else None


def sector_levels(spec: ModelSpectrum, *, det_irreps: np.ndarray | None = None) -> dict:
    """The candidate levels of E1lb on the model spectrum of H' (F41, module docstring).

    Args:
        spec: the model spectrum of H' in the hand-off sector (``ft.acceptance.model_spectrum``).
        det_irreps: the irrep of every determinant of the sector, shape ``spec.shape``
            (``ft.symmetry.determinant_irreps(norb, nelec, irreps)``); with it the eigenvectors'
            irrep weights are measured.

    Returns:
        A dict with ``e0``, ``s2_e0`` and ``irrep_e0`` (psi_0 = eigenvector 0) and, for each
        level -- ``e1_singlet`` (lowest excited singlet of psi_0's irrep, M3's E1 on H'),
        ``e1_any_spin`` (lowest excited level of that irrep, any S), ``e1_any_irrep`` (lowest
        excited singlet, any irrep) and ``e1_any`` (lowest excited level) -- its energy (nan if
        none), ``gap_<level>`` = level - e0, ``s2_<level>``, ``irrep_<level>`` and
        ``index_<level>`` (None if none). With ``det_irreps``: ``irrep_weight_<level>`` and
        ``irrep_weight_e0`` (the eigenvector's weight in its dominant irrep) and
        ``irrep_impurity`` = max_j (1 - that weight) over every eigenvector, 0 to round-off for a
        symmetry-conserving H'. Irreps are the dominant ones of ``spec``.
    """
    spec = _check_spec(spec)
    e, s2, ir = spec.energies, spec.s2, spec.irreps
    singlet = s2 < _SINGLET_TOL
    same = ir == ir[0]
    masks = {"e1_singlet": singlet & same, "e1_any_spin": same, "e1_any_irrep": singlet,
             "e1_any": np.ones(e.size, dtype=bool)}
    out: dict = {"e0": float(e[0]), "s2_e0": float(s2[0]), "irrep_e0": int(ir[0])}
    weights = None
    if det_irreps is not None:
        det_ir = np.asarray(det_irreps)
        if det_ir.shape not in (spec.shape, (spec.shape[0] * spec.shape[1],)) or det_ir.dtype.kind not in "iu" \
                or np.any((det_ir < 0) | (det_ir > 7)):
            raise ValueError(f"det_irreps must be integer irrep ids 0..7 of shape {spec.shape}")
        det_ir = det_ir.ravel()
        sq = spec.vectors ** 2
        per_irrep = np.stack([sq[det_ir == g].sum(axis=0) for g in range(8)])
        weights = per_irrep.max(axis=0) / np.maximum(per_irrep.sum(axis=0), 1e-300)
        out["irrep_weight_e0"] = float(weights[0])
        out["irrep_impurity"] = float(max(0.0, 1.0 - float(weights.min())))
    for key in _LEVEL_KEYS:
        j = _level(spec, masks[key])
        out[key] = math.nan if j is None else float(e[j])
        out[f"gap_{key}"] = math.nan if j is None else float(e[j] - e[0])
        out[f"s2_{key}"] = math.nan if j is None else float(s2[j])
        out[f"irrep_{key}"] = None if j is None else int(ir[j])
        out[f"index_{key}"] = j
        if weights is not None:
            out[f"irrep_weight_{key}"] = math.nan if j is None else float(weights[j])
    return out


def contaminant_weight(spec: ModelSpectrum, ci: np.ndarray, theta: float, *, s2_tol: float = 1e-4) -> float:
    """F41's q_c: the weight of ``ci`` on eigenstates of H' below ``theta`` whose <S^2> differs from psi_0's.

    q_c = sum_j p_j over E_j < theta and |<S^2>_j - <S^2>_0| > ``s2_tol`` (non-singlet levels for
    the singlet ground states of every system here), p_j = ``spec.weights(ci)``; it enters
    eps_id as q_c / (p_0 K_0) when E1lb is the singlet E_1 (module docstring).
    """
    spec = _check_spec(spec)
    theta = _finite(theta, "theta")
    s2_tol = _nonnegative(s2_tol, "s2_tol")
    w = spec.weights(ci)
    other = np.abs(spec.s2 - spec.s2[0]) > s2_tol
    return float(np.sum(w[other & (spec.energies < theta)]))


# ------------------------------------------------------------------ truncated CI (F42)
def excitation_levels(norb: int, nelec: Sequence[int]) -> np.ndarray:
    """Excitation level of every determinant relative to HF, as an (n_alpha strings, n_beta strings) array.

    The level counts the electrons of both spins in orbitals outside the HF occupation (the lowest
    n_alpha and n_beta orbitals, PySCF address 0), in PySCF's string order.
    """
    from pyscf.fci import cistring

    n = _int_at_least(norb, 1, "norb")
    ne = tuple(int(x) for x in nelec)
    if len(ne) != 2 or min(ne) < 0 or max(ne) > n:
        raise ValueError(f"nelec must be (n_alpha, n_beta) with 0 <= n <= norb = {n}, got {nelec!r}")
    levels = []
    for k in ne:
        strings = cistring.make_strings(range(n), k)
        occupied = (1 << k) - 1
        levels.append(np.array([bin(int(s) & ~occupied).count("1") for s in strings], dtype=np.int64))
    return levels[0][:, None] + levels[1][None, :]


def _lanczos_lowest(matvec, v0: np.ndarray, tol: float, max_iter: int) -> tuple[np.ndarray, float, int]:
    """Lowest Ritz pair of a symmetric operator by Lanczos with full reorthogonalization.

    Returns (Ritz vector, residual norm estimate, iterations). Stops when beta_j |y_j| <= tol or
    the Krylov space becomes invariant.
    """
    n = v0.size
    m_max = min(max_iter, n)
    basis = np.zeros((m_max, n))
    alpha: list[float] = []
    beta: list[float] = []
    basis[0] = v0 / np.linalg.norm(v0)
    scale = 0.0
    y = np.ones(1)
    res = math.inf
    for j in range(m_max):
        w = matvec(basis[j])
        a = float(basis[j] @ w)
        alpha.append(a)
        w = w - a * basis[j]
        if j > 0:
            w = w - beta[-1] * basis[j - 1]
        for _ in range(2):                                  # full reorthogonalization, twice
            w = w - basis[:j + 1].T @ (basis[:j + 1] @ w)
        b = float(np.linalg.norm(w))
        scale = max(scale, abs(a), b)
        if j == 0:
            y = np.ones(1)
        else:
            _, vec = scipy.linalg.eigh_tridiagonal(np.array(alpha), np.array(beta), select="i",
                                                   select_range=(0, 0))
            y = vec[:, 0]
        res = b * abs(float(y[-1]))
        if res <= tol or b <= _BREAKDOWN_RTOL * max(scale, 1.0) or j == m_max - 1:
            return basis[:j + 1].T @ y, res, j + 1
        beta.append(b)
        basis[j + 1] = w / b
    return basis[:1].T @ y, res, 1                          # pragma: no cover (m_max >= 1)


def truncated_ci_energy(active: ActiveSpace, max_level: int = 2, *, tol: float = 1e-10) -> float:
    """Variational truncated-CI energy (F42): the lowest eigenvalue of P H P over levels <= ``max_level``.

    ``max_level`` = 2 is CISD, 3 CISDT (0 returns E_HF; a level covering the sector returns its
    lowest eigenvalue). The lowest over every spin and irrep of the (n_alpha, n_beta) sector
    (module docstring): an upper bound on its lowest eigenvalue E_0 and at most E_HF, ecore
    included (Hartree). Lanczos with full reorthogonalization stops at the residual ``tol``
    (Hartree); RuntimeError if 1000 iterations do not reach it.
    """
    if not isinstance(active, ActiveSpace):
        raise ValueError(f"active must be an ActiveSpace, got {type(active).__name__}")
    level = _int_at_least(max_level, 0, "max_level")
    tol = _positive(tol, "tol")
    ham = Hamiltonian(active)
    shape = ham.space.shape
    sel = np.flatnonzero(excitation_levels(active.norb, active.nelec).ravel() <= level)
    full = np.zeros(shape[0] * shape[1])

    def matvec(x: np.ndarray) -> np.ndarray:
        full[:] = 0.0
        full[sel] = x
        return np.asarray(ham.apply(full.reshape(shape))).ravel()[sel]

    rng = np.random.default_rng(_LANCZOS_SEED)
    v0 = rng.standard_normal(sel.size)
    v0 *= 0.5 / np.linalg.norm(v0)                          # norm 1/2: v0 cannot cancel HF's unit entry
    v0[0] += 1.0                                            # HF is address 0, the first selected determinant
    x, res, _ = _lanczos_lowest(matvec, v0, tol, _LANCZOS_MAX)
    if res > tol and sel.size > _LANCZOS_MAX:
        raise RuntimeError(f"Lanczos did not converge: residual {res:.2e} > tol = {tol:.1e} "
                           f"after {_LANCZOS_MAX} iterations ({sel.size} determinants)")
    hx = matvec(x)
    return float(x @ hx / (x @ x)) + float(active.ecore)


# ------------------------------------------------------------------ energy certificates (F42)
@dataclass(frozen=True)
class EnergyCertificate:
    """A certified upper bound U >= E_0' and its cost beyond the loop (F42, module docstring).

    Attributes:
        source: one of :data:`CERT_SOURCES`.
        upper: U (Hartree): the planned bound E_j + m_j/2 for dedicated energies.
        j: the iterate whose energy certifies (None for classical and oracle certificates).
        toffolis: Toffolis beyond C_loop (0 for classical, loop-energy and oracle sources).
        shots: samples of a dedicated sampled energy (0 otherwise).
        applications: applications of a dedicated AE energy (0 otherwise).
        label: a description ("cisd", "sampled E(Psi_3)", ...).
        sigma: the standard deviation of the realized bound U^ about ``upper`` (Hartree): m_j/(2z)
            for dedicated sampled and AE energies; None for a deterministic bound (classical,
            loop-energy and oracle sources, or a dedicated energy costed at its planned U).
    """

    source: str
    upper: float
    j: int | None
    toffolis: float
    shots: float
    applications: float
    label: str
    sigma: float | None = None

    def __post_init__(self) -> None:
        if self.source not in CERT_SOURCES:
            raise ValueError(f"source must be one of {CERT_SOURCES}, got {self.source!r}")
        object.__setattr__(self, "upper", _finite(self.upper, "upper"))
        if self.sigma is not None:
            if self.source not in ("sampled", "ae"):
                raise ValueError(f"a {self.source} certificate is deterministic (sigma = None), got {self.sigma!r}")
            object.__setattr__(self, "sigma", _positive(self.sigma, "sigma"))
        if self.source in ("classical", "oracle"):
            if self.j is not None:
                raise ValueError(f"a {self.source} certificate has no iterate (j = None), got {self.j!r}")
        else:
            object.__setattr__(self, "j", _int_at_least(self.j, 0, "j"))
        for name in ("toffolis", "shots", "applications"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        if self.source in _FREE_SOURCES and (self.toffolis or self.shots or self.applications):
            raise ValueError(f"a {self.source} certificate costs nothing beyond the loop")
        if self.source == "sampled" and (self.shots <= 0.0 or self.applications != 0.0):
            raise ValueError("a sampled certificate has shots > 0 and no applications")
        if self.source == "ae" and (self.applications <= 0.0 or self.shots != 0.0):
            raise ValueError("an ae certificate has applications > 0 and no shots")
        if not isinstance(self.label, str):
            raise ValueError(f"label must be a string, got {self.label!r}")

    @property
    def loop_iterate(self) -> int:
        """The iterate up to which the loop must run before U exists: 0, j, or j + 1 (loop energies)."""
        if self.j is None:
            return 0
        return self.j + 1 if self.source == "loop_energy" else self.j


def _prep_arrays(prep_cumulative, prep_qubits) -> tuple[np.ndarray, np.ndarray | None]:
    """C(Psi_j) and Q(Psi_j) from a PrepCost or from plain sequences."""
    if hasattr(prep_cumulative, "cumulative") and hasattr(prep_cumulative, "logical_qubits"):
        cum = np.asarray(prep_cumulative.cumulative, dtype=float)
        qub = np.asarray(prep_cumulative.logical_qubits) if prep_qubits is None else np.asarray(prep_qubits)
    else:
        cum = np.asarray(prep_cumulative, dtype=float)
        qub = None if prep_qubits is None else np.asarray(prep_qubits)
    if cum.ndim != 1 or cum.size == 0 or not np.all(np.isfinite(cum)) or np.any(cum < 0.0):
        raise ValueError("prep_cumulative must be a non-empty 1-D array of finite, non-negative Toffoli counts")
    if qub is not None:
        if qub.shape != cum.shape or qub.dtype.kind not in "iu" or np.any(qub < 1):
            raise ValueError("prep_qubits must hold one positive integer Q(Psi_j) per entry of prep_cumulative")
    return cum, qub


def energy_certificates(record: TrajectoryRecord, prep_cumulative, *, e1_lower: float, de: float,
                        classical: Mapping[str, float] | None = None,
                        energy_variances: Sequence[float] | None = None, readout_toffolis: int = 0,
                        walk: BlockEncoding | None = None,
                        loop_energies: Mapping[int, tuple[float, float]] | None = None,
                        z: float = 3.0, kappa: float = 1.0, include_oracle: bool = False,
                        prep_qubits: Sequence[int] | None = None,
                        e0_model: float | None = None) -> list[EnergyCertificate]:
    """Every admissible certificate (U <= E1lb - 2 dE) of one record (F42, module docstring).

    Args:
        record: the trajectory (energies E_j of H, the iterates j = 0..K).
        prep_cumulative: C(Psi_j) (a sequence, or a ``ft.stateprep.PrepCost``, which also gives
            Q(Psi_j)).
        e1_lower: E1lb (F41).
        de: the QPE error dE (Hartree).
        classical: {label: variational energy of H'} (e.g. CISD(H'), CISDT(H')).
        energy_variances: V_E(Psi_j) per shot (Ha^2) for sampled certificates (nan entries are
            skipped); None: none.
        readout_toffolis: C_rot of the energy readout (``ft.stateprep.readout_rotation_toffolis``).
        walk: the selected encoding (lambda, C_W, Q_walk) for AE certificates; they need Q(Psi_j)
            (a PrepCost or ``prep_qubits``).
        loop_energies: {j: (E^_j, eps_j)} measured by a rule with energies.
        z: the confidence multiple (z = 3).
        kappa: the AE constant (F11).
        include_oracle: add U = E_0' (``e0_model``; ``record.e0``, the E_0 of H, if None), for
            reporting only.
        prep_qubits: Q(Psi_j) when ``prep_cumulative`` is a plain sequence.

    Order: classical, then per j the sampled and AE certificates, then loop energies, then the
    oracle. Certificates of iterates beyond ``prep_cumulative`` are skipped.
    """
    if not isinstance(record, TrajectoryRecord):
        raise ValueError(f"record must be a TrajectoryRecord, got {type(record).__name__}")
    e1 = _finite(e1_lower, "e1_lower")
    de = _positive(de, "de")
    z = _positive(z, "z")
    kappa = _positive(kappa, "kappa")
    c_rot = _int_at_least(readout_toffolis, 0, "readout_toffolis")
    prep, qubits = _prep_arrays(prep_cumulative, prep_qubits)
    thr = e1 - 2.0 * de
    energies = np.asarray(record.energies, dtype=float)
    n_states = min(energies.size, prep.size)
    n_sys = 2 * int(record.norb)
    out: list[EnergyCertificate] = []
    for label, u in (classical or {}).items():
        u = _finite(u, f"classical[{label!r}]")
        if u <= thr:
            out.append(EnergyCertificate("classical", u, None, 0.0, 0.0, 0.0, str(label)))
    var = None
    if energy_variances is not None:
        var = np.asarray(energy_variances, dtype=float).ravel()
        if np.any(var[np.isfinite(var)] < 0.0):
            raise ValueError("energy_variances must be >= 0 (nan entries are skipped)")
    if walk is not None:
        if not isinstance(walk, BlockEncoding):
            raise ValueError(f"walk must be a BlockEncoding, got {type(walk).__name__}")
        if qubits is None:
            raise ValueError("AE certificates need Q(Psi_j): pass a PrepCost or prep_qubits")
    for j in range(n_states):
        margin = thr - float(energies[j])
        if not margin > 0.0:
            continue
        sigma = margin / (2.0 * z)
        upper = float(energies[j]) + z * sigma
        if var is not None and j < var.size and math.isfinite(float(var[j])):
            q_run = int(qubits[j]) if qubits is not None else n_sys
            cost = energy_sampling(float(var[j]), float(prep[j]) + c_rot, q_run, eps=sigma)
            if cost.n_prep > 0.0:
                out.append(EnergyCertificate("sampled", upper, j, float(cost.toffolis), float(cost.n_prep), 0.0,
                                             f"sampled E(Psi_{j})", sigma=sigma))
        if walk is not None:
            cost = energy_ae(walk.walk_params(), PrepParams(float(prep[j]), int(qubits[j])), eps=sigma,
                             kappa=kappa, norb=int(record.norb))
            out.append(EnergyCertificate("ae", upper, j, float(cost.toffolis), 0.0, float(cost.n_prep),
                                         f"ae E(Psi_{j})", sigma=sigma))
    for j, pair in sorted((loop_energies or {}).items()):
        j = _int_at_least(j, 0, "loop_energies key")
        e_hat, eps_j = pair
        u = _finite(e_hat, f"loop_energies[{j}] energy") + z * _nonnegative(eps_j, f"loop_energies[{j}] eps")
        if u <= thr:
            out.append(EnergyCertificate("loop_energy", u, j, 0.0, 0.0, 0.0, f"loop energy E(Psi_{j})"))
    if include_oracle:
        u = float(record.e0) if e0_model is None else _finite(e0_model, "e0_model")
        if u <= thr:
            out.append(EnergyCertificate("oracle", u, None, 0.0, 0.0, 0.0,
                                         "E0 of H" if e0_model is None else "E0'"))
    return out

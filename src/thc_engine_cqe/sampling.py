"""Finite-shot noise models for the CQE, with shot accounting (M2 workflow C).

The models are estimator hooks of :func:`thc_engine_cqe.cqe.run_cqe`:

* :class:`ResidualNoise` (``residual_estimator``): Gaussian errors of the residual
  elements of a measurement route ("direct", "aux"), or isotropic noise;
* :class:`SnapshotResidual` (``residual_estimator``): the direct-route estimate from
  simulated matchgate-shadow snapshots (dense state; H4-sized systems);
* :class:`EnergyNoise` (``energy_estimator``): Gaussian energy errors of DF basis
  rotations or of a Pauli-level scheme;
* :class:`ShotLedger`: the shots spent per CQE iteration;
* :func:`noise_factor`: the matrix F of ``LinearizedCQE.noise_floor`` for a
  ResidualNoise model; :func:`element_variances`: the per-shot variances v_e.

Real states and integrals (M2 scope). Conventions: docs/conventions.md and the
element sets, targets and routes of :mod:`thc_engine_cqe.measure.targets` and
:mod:`thc_engine_cqe.measure.routes`. All randomness comes from the numpy
Generator passed as ``rng`` (or built from a seed); ``model.seed`` records the
entropy and spawn key of its SeedSequence. Give independent models independent
streams, e.g. ``np.random.SeedSequence(seed).spawn(2)``.

Residual noise model (ResidualNoise)
------------------------------------
Elements. A route estimates the K independent elements e of the residual
(``targets.element_set``): spin-orbital 2A^{ij}_{kl}, K = 2 C(C(N,2),2) + C(N^2,2)
(150 for N = 4, 840 for N = 6), or spin-free A[p,q,r,s], K = N^2 (N^2 - 1)/4 (60,
315). Every element has one Hermitian target: X_e = [Gamma_e - Gamma_e^dagger, H]/2
on Psi (direct route) or -Q_e/delta, Q_e = (Gamma_e - Gamma_e^dagger)/(2i), on the
auxiliary state(s) (aux route).

What is Gaussian. The vector of element estimates a_hat (K,) from S shots is

    a_hat = a + b + eta,   eta ~ N(0, diag(v) / S),

with a the exact element values, b the route bias (deterministic, state
dependent) and v_e the per-shot variance of element e under the scheme and its
shot allocation (``VarianceResult.per_target``: S shots in total give
Var(a_hat_e) = v_e / S). Different elements have independent errors in this model;
real shadow estimates are correlated (SnapshotResidual below; a correlated
Gaussian is available through ``per_element_covariance``).

Fill. The driver needs the spin-free pair-order tensor, a linear function L of
the element values, A = L(a). Hence

    A_est = L(a_hat) = A + L(b + eta),   A = rd.A,

so only the error is mapped and the exact spin-orbital residual is never formed.
The returned tensor is passed through ``optimizers.symmetrize_residual``, which
removes round-off only (the filled tensors are structured up to round-off) and
makes pair symmetry and anti-Hermiticity exact, so run_cqe's relative structure
check holds even for A_est at round-off level. L is (tested):

* spin-free kind (``targets.spin_free_tensor``): element x = (p,q,r,s) fills its
  orbit, +v at x and (r,s,p,q), -v at (q,p,s,r) and (s,r,q,p); 4 entries, or 2 for
  x = (p,q,p,q), p != q (N(N-1)/2 elements). The filled error has squared norm
  sum_e m_e eta_e^2 with m_e from ``targets.element_multiplicities``.
* spin-orbital kind (``targets.contract_spin``): A[p,q,r,s] = sum_{sigma tau}
  2A^{(p sigma)(r tau)}_{(q sigma)(s tau)}. An element stands for 8 entries of the
  (2N)^4 Mazziotti-order tensor (``targets.spin_orbital_tensor``); the contraction
  keeps the entries in which each creator shares its spin with its annihilator
  (spin orbitals (p,a) and (p,b) for alpha and beta):
  - alpha-beta element ((p,a),(r,b),(q,a),(s,b)): +v at (p,q,r,s) and (r,s,p,q),
    -v at (q,p,s,r) and (s,r,q,p); +2v at (p,q,p,q) and -2v at (q,p,q,p) if p = r
    and q = s; nothing if p = s and q = r: the N(N-1)/2 elements
    2A^{(p,a)(q,b)}_{(q,a)(p,b)} feed only the entries A[p,q,q,p], which vanish
    for real states, so their errors drop out;
  - same-spin element ((p,c),(r,c),(q,c),(s,c)), c = a or b, p < r, q < s: the
    orbit of (p,q,r,s) with signs (+, +, -, -) and the orbit of the exchanged
    (r,q,p,s) with the opposite signs (8 entries).
  A spin-free entry is the sum of up to four spin-orbital elements (alpha-alpha,
  beta-beta and the two alpha-beta orders), so its error variance is the sum of
  theirs, and error combinations in the kernel of L do not reach the CQE. H4:
  squared column norms 0 (6 columns), 4 (108) and 8 (36).

The filled error L(eta) is Gaussian on the residual's symmetry subspace
(``diagnostics.linearized.structured_basis``) with covariance L diag(v/S) L^T.

Routes:

* "direct": X_e on Psi, unbiased (b = 0).
* "aux": -Q_e/delta on the states Lambda_l of ``routes.aux_states(active, Psi, delta,
  aux_method, df=df)``. The bias of the current state is recomputed at every call:
  the infinite-shot aux estimate minus A. Its fill is the same tensor for both
  element kinds,

      L(b) = -sum_l Im dm2(Lambda_l) / delta - A,  dm2[p,q,r,s] = <Lambda|e_{pq,rs}|Lambda>,

  because the spin-free Q_e is the spin sum of spin-orbital ones and both obey the
  antisymmetries of the fill as operator identities. It equals the fill of
  ``routes.aux_residual(...) - routes.exact_residual(...)`` (both kinds, every
  method; tested to 1e-13 relative) and costs one real ``trans_rdm12`` per state:
  Im <Lambda|e_{pq,rs}|Lambda> = D[p,q,r,s] - D[q,p,s,r] with D = <Re Lambda|e|Im Lambda>.

  - "exact" (Lambda = exp(-i delta H) Psi): an eigenstate only acquires a phase, so
    b(Psi_0) = 0 and the route moves no fixed point. With Psi = Psi_0 + sum_k d_k Psi_k
    (diagnostics/linearized.py: gaps w_k, couplings M_k), to first order in d and to
    all orders in delta, -Im <Gamma_e>_Lambda / delta = sum_k d_k w_k sinc(delta w_k) M_k,
    sinc x = sin(x)/x (tested): the route damps the residual of every excitation by
    sinc(delta w_k). Fixed-step SD then maps y by I - (eps/2) K diag(sinc(delta w)), and
    relaxation slows; observed on H4, the sorted eigenvalues of K diag(sinc(delta w))
    are 0.98-1.00 (delta = 0.1), 0.86-0.99 (0.3) and 0.51-0.96 (0.6) times the sorted
    stiffnesses lambda.
  - "df-trotter" and "df-fragments": the fragment propagators do not evolve an
    eigenstate by a phase, so b(Psi_0) != 0, of order delta^2, which moves the
    fixed point of fixed-step SD to y* = -B^+ b (``LinearizedCQE.bias_offset``).
    Observed on H4 at FCI: ||b|| = 9.1e-3 (df-trotter) and 5.5e-3 (df-fragments)
    at delta = 0.1, offsets 4.0e-6 and 7.6e-6 Ha (tested for df-trotter), scaling as
    delta^2 and delta^4 to within 7% for delta = 0.01-0.3. The three softest modes
    (lambda <= 0.98) hold 0.03% of the df-trotter offset, while 74% of the
    df-fragments offset lies in the softest mode (lambda = 0.054, relaxation time
    2/(eps lambda) = 740 steps at eps = 0.05; observed).

  ``bias=False`` keeps the aux variances and drops the bias.
* "isotropic": eta = (sigma / sqrt(S)) U z with U = ``structured_basis(N)`` and
  z ~ N(0, I_K), K = N^2 (N^2 - 1)/4; with S = 1 this is the noise sigma U z of the
  linearized theory. ``kind`` and ``scheme`` are not used.

Shots. ``shots`` = S is the number of shots of one residual estimate, made once per
CQE iteration, for all K elements together (the scheme's optimal allocation over
its settings, and for "df-fragments" over the L + 1 auxiliary states, as in
``VarianceResult``). ``shots=inf`` switches the noise off; the aux bias stays.

Frozen variances. v_e is given (``per_element_variance``) or computed once by
:func:`element_variances` at ``reference_ci`` (default: the FCI ground state, the
linearization point of the noise-floor theory); it does not follow the state.
The systematic error of freezing, observed along the exact SD trajectory from HF
(iterates 0, 2, 4, 5, 7) relative to FCI:

* matchgate shadows, direct and aux (delta = 0.1) routes, both kinds: totals within
  0.4% (H4) and 0.1% (H6 at 1.0 A), single elements 0.95-1.02 and 0.97-1.02 (the
  matchgate variances are dominated by the state-independent 1/lambda_k);
* Pauli shadows (H4 direct and aux, H6 at 1.0 A aux): totals within 0.4%, single
  elements 0.80-1.06;
* QWC: totals within 3% (H4 aux, spin-free; 1.5% for H6 at 1.0 A and the H4 direct
  route), but single aux-route elements change by factors up to 170 (ratio 0.006 at
  HF, a single determinant); 0.85-1.08 for the H4 direct route.

Correlated Gaussian (extension): ``per_element_covariance`` C (K, K, per shot) gives
eta = C^{1/2} z / sqrt(S), e.g. with C from :meth:`SnapshotResidual.element_covariance`.

Mapping onto the linearized CQE (noise_factor)
----------------------------------------------
``LinearizedCQE.noise_floor(eps, F)`` takes the residual error of one step as F z,
z ~ N(0, I_r), drawn afresh every step (for real states g = conj(A_est) = A_est).
The model's filled error is L G z with G = diag(sqrt(v / S)) (or C^{1/2} / sqrt(S)),
so F = L G: column e is the filled pair-order tensor of a unit error in element e
(through the spin contraction for spin-orbital kinds) times sqrt(v_e / S); for the
isotropic route F = (sigma / sqrt(S)) U. The prediction uses exactly the simulated
noise, with the variances frozen at the linearization point; only F F^T matters.
With an aux bias, the stationary mean energy of the linear model is the offset
||y*||^2 plus the noise floor tr C (they add).

Validated on H4 (1.8 bohr), fixed-step SD from FCI with the direct route and
matchgate shadows, 200 + 1500 steps: the time-averaged energy of the fast modes is
0.98-1.05 of ``noise_floor`` and the total E - E_FCI 0.97-1.03 of it (3 seeds x
eps in {0.03, 0.05} x S in {1e5, 1e6} x both kinds; predicted floors 1.1e-5 to
2.7e-4 Ha; tested within 15%); at S = 5000 (floor 5.4e-3 Ha, eps = 0.05, 100 + 500
steps) still 0.98-1.01, and 0.98-1.00 for the correlated Gaussian (observed;
"Snapshot residual"). The aux-route fixed point (df-trotter,
delta = 0.1, S = inf, eps = 0.05) is 1.014 of ``bias_offset`` (tested within 5%; 1.13
at delta = 0.3). The difference is the state dependence of the bias,
b(Psi) = b(Psi_0) + J d with J = O(delta^2): with J the linear prediction agrees to
1e-4 (3e-3 at delta = 0.3; observed). For df-fragments only the relaxed modes
agree (to 0.6-5%); in the softest mode, whose first-order residual is small, the
nonlinear CQE departs from the linear prediction (after 1500 steps 0.92 of it at
delta = 0.1 and 0.44 at delta = 0.3; observed).

Snapshot residual (SnapshotResidual)
------------------------------------
A drop-in ``residual_estimator`` that simulates the direct route with matchgate
shadows (Zhao, Rubin, Miyake, PRL 127, 110504 (2021)) instead of drawing Gaussian
errors. Each snapshot draws a uniformly random perfect matching pi of the 2n
Majoranas (``schemes.random_matchings``) and measures its n commuting pair
operators i gamma_a gamma_b one after the other on the dense state (Born rule,
projective). The post-measurement state phi is the joint eigenvector of the n
pair operators, so every Pauli string P_j of the targets covered by pi (a union of
pairs of pi; enumerated as the unions of at most k_max = 3 pairs, the degree of the
three-body X_e, and looked up among the target strings) has a sharp value
v_j = (P_j phi)[b] / phi[b] = +-1, read at the largest amplitude b. The element
estimates from S snapshots are those of the matchgate estimator of
:mod:`thc_engine_cqe.measure.schemes`,

    a_hat_e = sum_j W_ej lambda_j^{-1} (1/S) sum_snapshots [P_j covered] v_j,

with X_e = sum_j W_ej P_j and lambda_j = C(n,k)/C(2n,2k) for a 2k-Majorana string;
then A_est = L(a_hat), without the exact residual. No Gaussian assumption is made,
and the elements are correlated because they are read off the same snapshots. The
estimator is linear in the targets, so the same snapshots give the same filled
tensor for both element kinds (tested).

Observed on H4 (FCI and SD iterate 2, 1e5 snapshots): unbiased; per-element
variances 0.98-1.02 of the exact matchgate variances, i.e. of the Gaussian model's
diagonal (tested); off-diagonal correlation coefficients RMS 0.049 (spin-orbital)
and 0.096 (spin-free), max |rho| 0.42 and 0.73. The correlations raise the SD
floor. Exactly, through the matchgate variances of the mode targets
Y_m = sum_e c_me X_e that drive the linearized dynamics, the correlated floor is
1.32 (eps = 0.03) and 1.41 (eps = 0.05) times the independent-element floor, for
both kinds; the empirical covariance of 1e5 snapshots gives the same ratios. In
the real CQE (fixed-step SD from FCI, S = 5000 snapshots per iteration, eps = 0.05,
100 + 500 steps; observed), the mean E - E_FCI is 7.4e-3 and 7.8e-3 Ha with
snapshots (2 seeds), 7.4e-3 to 7.6e-3 with the correlated Gaussian
(``per_element_covariance`` from 2e5 snapshots; prediction 7.6e-3) and 5.3e-3 to
5.4e-3 with independent elements (prediction 5.4e-3): the snapshot floor is that
of the correlated Gaussian, 1.4 times the independent-element floor, with no
visible non-Gaussian effect.

Energy noise (EnergyNoise)
--------------------------
E_est = E + sqrt(V(Psi) / S) z, z ~ N(0, 1) per call: unbiased, S = ``shots`` per
energy estimate. V is the per-shot variance of the scheme: DF basis rotations
(``schemes.df_variance``; ``shift=True`` by default, the leaves in
E~_pq = E_pq - delta_pq, 1.23 Ha^2 for H6 at 1.0 A at FCI against 5.02 unshifted,
see measure/schemes.py), evaluated at the estimated state itself ("exact"), or V
frozen at ``reference_ci`` (QWC, Pauli shadows, matchgate shadows, whose energy
variances cost 0.5-0.6 s per state at H6; see "Timings"). The shifted DF variance
changes along the SD trajectory: at HF it is 0.79-0.81 of its FCI value and at SD
iterate 2 0.94-0.95 (H4, H6 at 1.0 A; observed), so a frozen V would overstate the
early noise by up to 25%. run_cqe calls the
estimator once with "current" per iteration and once per nonzero candidate; the
zero candidate reuses the current estimate (no call, no shots). The chosen
candidate's estimate is not reused as the next "current" estimate: run_cqe draws
a fresh one, which also avoids the downward selection bias of the minimum over
candidates. A truncated factorization's bias <H_DF> - <H> is not added.

Shot ledger (ShotLedger)
------------------------
One record per CQE iteration: residual shots, current-state energy shots and
candidate energy shots, with call counts. The models charge the ledger they are
given (default: their own, ``model.ledger``); give the residual and energy models
of one run the same ledger. ``cumulative()[k]`` is the number of shots spent to
reach ``CQEResult.energies[k + 1]``.

Timings
-------
Observed, one thread on a shared 4-core machine, ms per CQE iteration (fixed-step SD
from FCI, one candidate unless noted; H4 at 1.8 bohr / H6 at 1.0 A):

* exact residual and energies: 4-5 / 6-6.5 (11 candidates: 4.4 / 6.5);
* ResidualNoise direct, spin-orbital or spin-free: 4.5-5.3 / 6.5-7.3; isotropic
  3.5 / 6.6;
* ResidualNoise aux, delta = 0.1 (bias recomputed every call): "exact" 6.4 / 14,
  "df-trotter" 35 / 86, "df-fragments" 48 / 90 (``routes.aux_states`` propagates every
  DF fragment by a Taylor series);
* EnergyNoise, 11 candidates (12 estimates per iteration): DF variance at every state
  22 / 46, i.e. about 1.5 / 3.3 ms per estimate; frozen variance 3.9 / 8.7;
* SnapshotResidual: 69 ms / 1.6 s per 1000 snapshots (about 1.4e4 / 6e2 snapshots per
  second; dense states of 2^8 / 2^12 amplitudes);
* setup of computed variances (direct route, matchgate shadows, spin-orbital, at FCI):
  2.5 s / 17 s (exact pair sums / Rao-Blackwellized with 2e4 settings); per-state energy
  variances (``EnergyNoise.variance_of``) 2-5 ms at H4, at H6 3 ms for DF and 0.5-0.6 s
  for QWC, Pauli and matchgate shadows.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from .fci_ops import FCISpace
from .integrals import ActiveSpace
from .optimizers import structure_error, symmetrize_residual
from .residual import ResidualData, compute_residual, trans_rdm12

__all__ = [
    "ROUTES",
    "ENERGY_SCHEMES",
    "ShotLedger",
    "element_variances",
    "ResidualNoise",
    "EnergyNoise",
    "noise_factor",
    "SnapshotResidual",
]

ROUTES = ("direct", "aux", "isotropic")
ENERGY_SCHEMES = ("df", "qwc", "pauli_shadow", "matchgate_shadow")
_CONTEXTS = ("current", "candidate")
# Relative tolerance of the internal structure check of A_est (the driver allows 1e-10).
_STRUCTURE_RTOL = 1e-12


# ================================================================ helpers
def _as_rng(rng) -> np.random.Generator:
    """A numpy Generator: passed through, or ``default_rng(rng)`` for a seed / SeedSequence."""
    if isinstance(rng, np.random.Generator):
        return rng
    if rng is None:
        raise ValueError("pass rng: a numpy Generator or a seed (runs must be reproducible)")
    return np.random.default_rng(rng)


def _seed_record(rng: np.random.Generator) -> dict | None:
    """Entropy and spawn key of the Generator's SeedSequence (None if it has none)."""
    seq = getattr(rng.bit_generator, "seed_seq", None)
    if seq is None or not hasattr(seq, "entropy"):
        return None
    return {"entropy": seq.entropy, "spawn_key": tuple(getattr(seq, "spawn_key", ()))}


def _check_shots(shots, *, integer: bool = False) -> float:
    s = float(shots)
    if not s > 0:                       # also rejects NaN
        raise ValueError(f"shots must be positive (or inf for no noise), got {shots!r}")
    if integer and (not math.isfinite(s) or s != int(s)):
        raise ValueError(f"snapshots must be a positive integer, got {shots!r}")
    return s


def _real_ci(ci, space: FCISpace) -> np.ndarray:
    """Normalized real CI array; complex input must be real up to a global phase (1e-12)."""
    c = np.asarray(ci)
    if c.size != space.size:
        raise ValueError(f"CI vector has {c.size} entries, expected {space.size}")
    c = c.reshape(space.shape)
    if np.iscomplexobj(c):
        big = c.flat[int(np.argmax(np.abs(c)))]
        if big == 0:
            raise ValueError("zero CI vector")
        c = c * (np.conj(big) / abs(big))
        if np.max(np.abs(c.imag)) > 1e-12 * np.max(np.abs(c)):
            raise ValueError("the noise models are implemented for real states only (M2 scope)")
        c = c.real
    c = np.asarray(c, dtype=float)
    nrm = float(np.linalg.norm(c))
    if nrm == 0:
        raise ValueError("zero CI vector")
    return c / nrm


def _real_residual(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if np.iscomplexobj(a):
        scale = float(np.max(np.abs(a))) if a.size else 0.0
        if a.size and float(np.max(np.abs(a.imag))) > 1e-12 * max(scale, 1e-300):
            raise ValueError("complex residual: the noise models are implemented for real states only")
        a = a.real
    return np.asarray(a, dtype=float)


def _structured(a: np.ndarray) -> np.ndarray:
    """A_est with exact structure: the projection only removes round-off, then the structure is asserted.

    ``optimizers.symmetrize_residual`` makes the pair symmetry and the antisymmetry
    A[q,p,s,r] = -A[p,q,r,s] hold bit for bit, so run_cqe's relative structure check
    also passes when A_est is itself at round-off level (an eigenstate, shots = inf).
    """
    out = symmetrize_residual(a)
    if not np.all(np.isfinite(out)):
        raise RuntimeError("internal error: non-finite residual estimate")
    pair, herm = structure_error(out)
    tol = _STRUCTURE_RTOL * float(np.abs(out).max(initial=0.0))
    if pair > tol or herm > tol:
        raise RuntimeError(f"internal error: A_est lost the structure of A (defects {pair:.2e}, {herm:.2e})")
    return out


def _fci_ci(active: ActiveSpace) -> tuple[float, np.ndarray]:
    e, ci = active.fci()
    return float(e), np.asarray(ci)


# ================================================================ ledger
class ShotLedger:
    """Shots charged per CQE iteration: residual, current-state energy and candidate energies.

    Records are opened in call order, which follows run_cqe's hook order
    (residual_estimator, energy_estimator("current"), then the candidates):

    * :meth:`charge_residual` always opens a new record (one per iteration);
    * :meth:`charge_energy` with context "current" joins the latest record if that
      record has a residual charge and no energy charge yet, and opens a new record
      otherwise (runs with exact residuals: one record per "current" call);
    * "candidate" charges join the latest record.

    So one ledger shared by a ResidualNoise and an EnergyNoise of the same run
    has one record per CQEStep, in order. Use a fresh ledger per run (or
    :meth:`reset`). Shots are floats (``inf`` for noise-free estimates).
    """

    def __init__(self) -> None:
        self._records: list[dict] = []

    def reset(self) -> None:
        self._records = []

    def _new(self, iteration) -> dict:
        rec = {"iteration": len(self._records) if iteration is None else int(iteration),
               "residual": 0.0, "n_residual": 0,
               "energy_current": 0.0, "n_current": 0,
               "energy_candidate": 0.0, "n_candidate": 0}
        self._records.append(rec)
        return rec

    def charge_residual(self, shots: float, iteration: int | None = None) -> None:
        rec = self._new(iteration)
        rec["residual"] += float(shots)
        rec["n_residual"] += 1

    def charge_energy(self, shots: float, context: str) -> None:
        if context not in _CONTEXTS:
            raise ValueError(f"context must be one of {_CONTEXTS}, got {context!r}")
        last = self._records[-1] if self._records else None
        if context == "current":
            if last is not None and last["n_residual"] > 0 and last["n_current"] == 0 \
                    and last["n_candidate"] == 0:
                rec = last
            else:
                rec = self._new(None)
            rec["energy_current"] += float(shots)
            rec["n_current"] += 1
        else:
            rec = last if last is not None else self._new(None)
            rec["energy_candidate"] += float(shots)
            rec["n_candidate"] += 1

    # ---------------------------------------------------------------- views
    @property
    def records(self) -> list[dict]:
        """Copies of the per-iteration records."""
        return [dict(r) for r in self._records]

    @property
    def n_iterations(self) -> int:
        return len(self._records)

    def _column(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self._records], dtype=float)

    @property
    def residual_shots(self) -> np.ndarray:
        """Residual shots per iteration."""
        return self._column("residual")

    @property
    def energy_shots(self) -> np.ndarray:
        """Energy shots per iteration (current state plus candidates)."""
        return self._column("energy_current") + self._column("energy_candidate")

    @property
    def total_shots(self) -> np.ndarray:
        """All shots per iteration."""
        return self.residual_shots + self.energy_shots

    def cumulative(self) -> np.ndarray:
        """Cumulative shots after each iteration: entry k pays for CQEResult.energies[k + 1]."""
        return np.cumsum(self.total_shots)

    def summary(self) -> dict:
        """Totals and call counts (a JSON-friendly dict of floats and ints)."""
        res = self.residual_shots
        cur = self._column("energy_current")
        cand = self._column("energy_candidate")
        return {
            "n_iterations": self.n_iterations,
            "residual_shots": float(res.sum()),
            "energy_shots_current": float(cur.sum()),
            "energy_shots_candidate": float(cand.sum()),
            "energy_shots": float(cur.sum() + cand.sum()),
            "total_shots": float(res.sum() + cur.sum() + cand.sum()),
            "n_residual_estimates": int(sum(r["n_residual"] for r in self._records)),
            "n_energy_estimates_current": int(sum(r["n_current"] for r in self._records)),
            "n_energy_estimates_candidate": int(sum(r["n_candidate"] for r in self._records)),
        }


# ================================================================ variances
def element_variances(active: ActiveSpace, ci=None, *, route: str = "direct",
                      scheme: str = "matchgate_shadow", kind: str = "spin_orbital",
                      delta: float | None = None, aux_method: str = "exact", df=None, mapping=None,
                      **scheme_options):
    """Per-shot variances of the independent residual elements of one route and scheme.

    Returns the ``schemes.VarianceResult`` of the vector target, whose
    ``per_target`` are the v_e of the noise model:

    * ``route="direct"``: X_e = ``targets.direct_targets(elements, active, mapping, kind)``
      on Psi = ``QubitState.from_ci(ci)``;
    * ``route="aux"``: [-Q_e / delta] (``targets.aux_targets``) on each auxiliary state of
      ``routes.aux_states(active, ci, delta, aux_method, df=df)``; the L + 1 states of
      "df-fragments" are combined with ``schemes.combine_independent`` (optimal shot
      allocation across the states, total (sum_l sqrt V_l)^2).

    This is the definition of ``routes.direct_residual_variance`` and
    ``routes.aux_residual_variance`` (built here from targets and schemes, so this
    module does not depend on those functions). ``ci`` defaults to the FCI ground state;
    elements are ``targets.element_set(N, kind)``; ``mapping`` defaults to the blocked
    order; ``scheme_options`` go to ``schemes.scheme_variance`` (e.g. ``method``,
    ``n_settings``, ``seed``; matchgate shadows use the exact pair sums up to 2e7 pairs
    and Rao-Blackwellized Monte Carlo above, ``measure.schemes``).
    """
    from .measure import schemes
    from .measure.pauli import QubitMapping, QubitState
    from .measure.routes import aux_states
    from .measure.targets import aux_targets, direct_targets, element_set

    if route not in ("direct", "aux"):
        raise ValueError(f"route must be 'direct' or 'aux' for element variances, got {route!r}")
    if scheme not in schemes.SCHEMES:
        raise ValueError(f"scheme must be one of {schemes.SCHEMES}, got {scheme!r}")
    if route == "aux" and delta is None:
        raise ValueError("the aux route needs delta")
    norb = active.norb
    space = FCISpace.of(active)
    mapping = QubitMapping(norb) if mapping is None else mapping
    if mapping.norb != norb:
        raise ValueError(f"mapping has norb={mapping.norb}, active space {norb}")
    el = element_set(norb, kind)
    psi = _real_ci(_fci_ci(active)[1] if ci is None else ci, space)
    opts = dict(scheme_options)
    if route == "direct":
        ops = direct_targets(el, active, mapping, kind)
        return schemes.scheme_variance(ops, QubitState.from_ci(norb, active.nelec, psi, mapping), scheme, **opts)
    states = aux_states(active, psi, float(delta), aux_method, df=df)
    ops = [(-q) / float(delta) for q in aux_targets(el, mapping, kind)]
    if scheme == "qwc" and opts.get("grouping") is None:
        opts["grouping"] = schemes.qwc_grouping(ops)
    if len(states) > 1 and "seed" in opts and not isinstance(opts["seed"], np.random.Generator):
        opts["seed"] = np.random.default_rng(opts["seed"])
    parts = [schemes.scheme_variance(ops, QubitState.from_ci(norb, active.nelec, lam, mapping), scheme, **opts)
             for lam in states]
    return parts[0] if len(parts) == 1 else schemes.combine_independent(parts)


# ================================================================ residual noise
class _ElementFill:
    """The linear map from independent-element values (K,) to the spin-free pair-order tensor."""

    def __init__(self, norb: int, kind: str):
        from .measure.targets import element_set

        self.norb = int(norb)
        self.kind = kind
        self.elements = element_set(self.norb, kind)
        self.K = int(self.elements.shape[0])
        self._matrix: np.ndarray | None = None

    def __call__(self, values: np.ndarray) -> np.ndarray:
        from .measure.targets import contract_spin, spin_free_tensor

        if self.kind == "spin_orbital":
            return contract_spin(values, self.elements, self.norb)
        return spin_free_tensor(values, self.elements, self.norb)

    def matrix(self) -> np.ndarray:
        """Dense (N^4, K) matrix whose column e is the filled tensor of a unit value on e."""
        if self._matrix is None:
            eye = np.eye(self.K)
            self._matrix = np.ascontiguousarray(
                np.array([self(eye[e]).ravel() for e in range(self.K)]).T)
        return self._matrix


class ResidualNoise:
    """Gaussian finite-shot model of an estimated residual: a ``residual_estimator`` for run_cqe.

    See the module docstring ("Residual noise model"). ``model(rd, ci, it)``
    charges ``shots`` to the ledger and returns A_est = A + fill(b + eta), a
    fresh real pair-order tensor with the exact structure of A (module
    docstring); A = rd.A is the exact spin-free residual, so the exact
    spin-orbital residual is never formed.

    Args:
        active: the active space (real integrals).
        route: "direct" (X_e on Psi; unbiased), "aux" (-Q_e/delta on the auxiliary
            state(s); biased) or "isotropic" (eta = sigma/sqrt(shots) U z with
            U = ``structured_basis(N)``; for theory checks).
        scheme: the measurement scheme of the computed per-element variances
            ("qwc", "pauli_shadow", "matchgate_shadow"); not used by "isotropic"
            or with given variances.
        kind: element set, "spin_orbital" (K = 2 C(C(N,2),2) + C(N^2,2)) or
            "spin_free" (K = N^2 (N^2 - 1)/4); not used by "isotropic".
        shots: shots per residual estimate (one per CQE iteration) for the whole
            vector of K elements; inf switches the noise off (the aux bias stays).
        delta, aux_method, df: aux route only: the step delta, the method of
            ``routes.aux_states`` ("exact", "df-trotter", "df-fragments") and the
            double factorization of the DF methods (default: untruncated).
        per_element_variance: the per-shot v_e (K,), or a ``schemes.VarianceResult``
            (its ``per_target``); default: :func:`element_variances` at
            ``reference_ci``.
        per_element_covariance: instead of the variances, a full per-shot
            covariance (K, K) of the element estimates (e.g. from
            :meth:`SnapshotResidual.element_covariance`): eta = C^{1/2} z / sqrt(shots).
            An extension beyond the independent-element model, for separating
            correlation from non-Gaussian effects.
        reference_ci: the state at which computed variances are frozen (default: FCI).
        sigma: isotropic route only: per-shot standard deviation per orthonormal
            direction of the residual's symmetry subspace.
        bias: aux route only: False drops the route bias (variance only).
        rng: a numpy Generator or a seed; every draw comes from it (K standard
            normals per call).
        ledger: a :class:`ShotLedger` to charge (default: a new one, ``self.ledger``).
        mapping: the qubit mapping of the variance computation (default blocked).
        variance_options: keyword arguments of :func:`element_variances` (scheme options).
    """

    def __init__(self, active: ActiveSpace, *, route: str = "direct", scheme: str = "matchgate_shadow",
                 kind: str = "spin_orbital", shots: float, delta: float | None = None,
                 aux_method: str = "exact", df=None, per_element_variance=None, per_element_covariance=None,
                 reference_ci=None, sigma: float | None = None, bias: bool = True, rng,
                 ledger: ShotLedger | None = None, mapping=None, variance_options: Mapping | None = None):
        if route not in ROUTES:
            raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
        self.active = active
        self.norb = int(active.norb)
        self.space = FCISpace.of(active)
        self.route = route
        self.shots = _check_shots(shots)
        self.rng = _as_rng(rng)
        self.seed = _seed_record(self.rng)
        self.ledger = ShotLedger() if ledger is None else ledger
        self.info: dict = {}
        self.delta = None
        self.aux_method = None
        self.bias = False
        self._df = None
        self._sqrt_cov: np.ndarray | None = None      # per-shot C^{1/2} (K, K) of a full covariance
        if route == "isotropic":
            from .diagnostics.linearized import structured_basis

            if sigma is None or not float(sigma) >= 0:
                raise ValueError("the isotropic route needs sigma >= 0")
            if per_element_variance is not None or per_element_covariance is not None:
                raise ValueError("the isotropic route takes sigma, not per-element variances")
            self.sigma = float(sigma)
            self.kind = None
            self.scheme = None
            self._basis = structured_basis(self.norb)
            self.K = int(self._basis.shape[1])
            self.variances = np.full(self.K, self.sigma ** 2)
            self._fill = None
            self.info.update(variance_source="isotropic")
            return
        if sigma is not None:
            raise ValueError("sigma is only used by the isotropic route")
        self.sigma = None
        self.kind = kind
        self.scheme = scheme
        self._fill = _ElementFill(self.norb, kind)
        self.K = self._fill.K
        if route == "aux":
            from .measure.routes import METHODS

            if delta is None or not np.isfinite(float(delta)) or float(delta) == 0:
                raise ValueError("the aux route needs a finite nonzero delta")
            if aux_method not in METHODS:
                raise ValueError(f"aux_method must be one of {METHODS}, got {aux_method!r}")
            self.delta = float(delta)
            self.aux_method = aux_method
            self.bias = bool(bias)
            if aux_method != "exact":
                from .factor import double_factorize

                self._df = double_factorize(np.asarray(active.eri)) if df is None else df
        if per_element_variance is not None and per_element_covariance is not None:
            raise ValueError("pass per_element_variance or per_element_covariance, not both")
        if per_element_covariance is not None:
            c = np.asarray(per_element_covariance, dtype=float)
            if c.shape != (self.K, self.K) or not np.all(np.isfinite(c)):
                raise ValueError(f"per_element_covariance must be a finite ({self.K}, {self.K}) matrix")
            c = 0.5 * (c + c.T)
            w, u = np.linalg.eigh(c)
            if w.min(initial=0.0) < -1e-8 * max(float(w.max(initial=0.0)), 1e-300):
                raise ValueError("per_element_covariance must be positive semidefinite")
            self._sqrt_cov = (u * np.sqrt(np.clip(w, 0.0, None))[None, :]) @ u.T
            v = np.diag(c).copy()
            self.info.update(variance_source="covariance")
        elif per_element_variance is None:
            ref = _fci_ci(active)[1] if reference_ci is None else reference_ci
            res = element_variances(active, ref, route=route, scheme=scheme, kind=kind, delta=self.delta,
                                    aux_method=self.aux_method or "exact", df=self._df, mapping=mapping,
                                    **dict(variance_options or {}))
            v = res.per_target
            self.info.update(variance_source="computed", variance_result=res)
        else:
            res = per_element_variance
            v = getattr(res, "per_target", res)
            self.info.update(variance_source="given")
        v = np.asarray(v, dtype=float)
        if v.shape != (self.K,):
            raise ValueError(f"per_element_variance must have shape ({self.K},) for kind {kind!r}, got {v.shape}")
        if not np.all(np.isfinite(v)) or np.any(v < 0):
            raise ValueError("per_element_variance must be finite and non-negative")
        self.variances = v

    # ------------------------------------------------------------ properties
    @property
    def elements(self) -> np.ndarray | None:
        """The (K, 4, 2) or (K, 4) element set (None for the isotropic route)."""
        return None if self._fill is None else self._fill.elements

    @property
    def std(self) -> np.ndarray:
        """Standard deviations sqrt(v_e / shots) of the element errors (zeros for inf shots)."""
        if math.isinf(self.shots):
            return np.zeros(self.K)
        return np.sqrt(self.variances / self.shots)

    @property
    def total_variance(self) -> float:
        """Per-shot sum_e v_e (isotropic: K sigma^2)."""
        return float(self.variances.sum())

    def element_factor(self) -> np.ndarray:
        """(K, K) matrix G with element errors eta = G z, z ~ N(0, I_K): diag(std) or C^{1/2}/sqrt(shots)."""
        if self._sqrt_cov is None or math.isinf(self.shots):
            return np.diag(self.std)
        return self._sqrt_cov / math.sqrt(self.shots)

    # ------------------------------------------------------------ the model
    def fill(self, values) -> np.ndarray:
        """Spin-free pair-order tensor of independent-element values (K,) (isotropic: U @ values)."""
        values = np.asarray(values, dtype=float)
        if values.shape != (self.K,):
            raise ValueError(f"values must have shape ({self.K},), got {values.shape}")
        if self._fill is None:
            return (self._basis @ values).reshape((self.norb,) * 4)
        return self._fill(values)

    def fill_matrix(self) -> np.ndarray:
        """Dense (N^4, K) matrix of :meth:`fill` (isotropic: ``structured_basis(N)``)."""
        if self._fill is None:
            return self._basis
        return self._fill.matrix()

    def bias_tensor(self, ci, A: np.ndarray | None = None) -> np.ndarray:
        """Filled route bias b(Psi) at the state ``ci`` (zeros unless the aux route with bias=True).

        Aux route: -sum_l Im dm2(Lambda_l) / delta - A, with the auxiliary states of
        ``routes.aux_states`` and dm2 the spin-summed pair-order 2-RDM; equal to the fill
        of ``aux_residual(...) - exact_residual(...)`` for both element kinds (module
        docstring). ``A`` (the exact residual of ``ci``) is computed when not given.
        """
        n = self.norb
        if self.route != "aux" or not self.bias:
            return np.zeros((n,) * 4)
        from .measure.routes import aux_states

        psi = _real_ci(ci, self.space)
        if A is None:
            A = compute_residual(self.active, psi).A
        A = _real_residual(A)
        est = np.zeros((n,) * 4)
        for lam in aux_states(self.active, psi, self.delta, self.aux_method, df=self._df):
            # Im <Lam|e_{pq,rs}|Lam> = D[p,q,r,s] - D[q,p,s,r], D = <Re Lam|e|Im Lam> (real H, real e)
            d = trans_rdm12(self.space, np.ascontiguousarray(lam.real), np.ascontiguousarray(lam.imag))[1]
            est -= (d - d.transpose(1, 0, 3, 2)) / self.delta
        return est - A

    def sample_element_errors(self) -> np.ndarray:
        """One draw of the element errors eta (K,) (isotropic: the coefficients of U), no bias."""
        if math.isinf(self.shots):
            return np.zeros(self.K)
        z = self.rng.standard_normal(self.K)
        if self._sqrt_cov is None:
            return self.std * z
        return (self._sqrt_cov @ z) / math.sqrt(self.shots)

    def sample_error(self) -> np.ndarray:
        """One draw of the filled noise fill(eta) (no bias), from this model's Generator."""
        return self.fill(self.sample_element_errors())

    def __call__(self, rd: ResidualData, ci: np.ndarray, it: int) -> np.ndarray:
        self.ledger.charge_residual(self.shots, it)
        A = _real_residual(rd.A)
        out = A + self.sample_error()
        if self.route == "aux" and self.bias:
            out = out + self.bias_tensor(ci, A)
        return _structured(out)

    def noise_factor(self) -> np.ndarray:
        """(N^4, K) matrix F = fill_matrix() G with filled error F z, z ~ N(0, I_K) (:func:`noise_factor`)."""
        if self._sqrt_cov is None or math.isinf(self.shots):
            return self.fill_matrix() * self.std[None, :]
        return self.fill_matrix() @ self.element_factor()


def noise_factor(model: ResidualNoise, norb: int | None = None) -> np.ndarray:
    """The noise factor F of ``LinearizedCQE.noise_floor`` for a :class:`ResidualNoise` model.

    Column e is the filled pair-order tensor of a unit error in element e (the
    spin contraction for spin-orbital kinds) times sqrt(v_e / shots); for the
    isotropic route F = sigma / sqrt(shots) U. Then the model's filled error is
    exactly F z with z ~ N(0, I) (module docstring). Shape (N^4, K); zeros for
    shots = inf. ``norb`` is checked against the model's.
    """
    if not isinstance(model, ResidualNoise):
        raise TypeError(f"noise_factor needs a ResidualNoise model, got {type(model).__name__}")
    if norb is not None and int(norb) != model.norb:
        raise ValueError(f"norb={norb} does not match the model's {model.norb}")
    return model.noise_factor()


# ================================================================ energy noise
class EnergyNoise:
    """Gaussian finite-shot energy estimates: an ``energy_estimator`` for run_cqe.

    ``model(e_exact, ci, context)`` charges ``shots`` to the ledger (contexts
    "current" and "candidate") and returns E + sqrt(V / shots) z, z ~ N(0, 1),
    with V the per-shot variance of the scheme. See the module docstring.

    Args:
        active: the active space.
        scheme: "df" (DF basis rotations, ``schemes.df_variance``), "qwc",
            "pauli_shadow" or "matchgate_shadow" (``schemes.scheme_variance`` of
            ``targets.energy_target``).
        shots: shots per energy estimate; inf switches the noise off.
        variance: "exact" (V of the estimated state itself), "frozen" (V at
            ``reference_ci``, or ``per_shot_variance``) or "auto" (default:
            "exact" for DF, "frozen" for the other schemes, whose variances
            cost 0.5-0.6 s per state at H6; observed).
        factorization: the double factorization (default: untruncated
            ``double_factorize(active.eri)``, computed once).
        shift: DF leaves in E~ = E - delta (``df_variance(..., shift=True)``; default).
        rng, ledger: as for :class:`ResidualNoise`.
        reference_ci: state of the frozen variance (default: FCI).
        per_shot_variance: a given frozen V (implies variance="frozen").
        mapping, scheme_options: qubit mapping and options of the non-DF schemes.
    """

    def __init__(self, active: ActiveSpace, *, scheme: str = "df", shots: float, variance: str = "auto",
                 factorization=None, shift: bool = True, rng, ledger: ShotLedger | None = None,
                 reference_ci=None, per_shot_variance: float | None = None, mapping=None,
                 scheme_options: Mapping | None = None):
        if scheme not in ENERGY_SCHEMES:
            raise ValueError(f"scheme must be one of {ENERGY_SCHEMES}, got {scheme!r}")
        if variance not in ("auto", "exact", "frozen"):
            raise ValueError(f"variance must be 'auto', 'exact' or 'frozen', got {variance!r}")
        if per_shot_variance is not None:
            if variance == "exact":
                raise ValueError("per_shot_variance is a frozen variance; use variance='frozen'")
            variance = "frozen"
        elif variance == "auto":
            variance = "exact" if scheme == "df" else "frozen"
        self.active = active
        self.space = FCISpace.of(active)
        self.scheme = scheme
        self.shots = _check_shots(shots)
        self.variance_mode = variance
        self.shift = bool(shift)
        self.rng = _as_rng(rng)
        self.seed = _seed_record(self.rng)
        self.ledger = ShotLedger() if ledger is None else ledger
        self.mapping = mapping
        self.scheme_options = dict(scheme_options or {})
        self.factorization = None
        if scheme == "df":
            from .factor import double_factorize

            self.factorization = double_factorize(np.asarray(active.eri)) if factorization is None \
                else factorization
        self._target = None
        self.frozen_variance = None
        if variance == "frozen":
            if per_shot_variance is not None:
                v = float(per_shot_variance)
                if not (np.isfinite(v) and v >= 0):
                    raise ValueError("per_shot_variance must be finite and non-negative")
            else:
                v = self.variance_of(_fci_ci(active)[1] if reference_ci is None else reference_ci)
            self.frozen_variance = v

    def variance_of(self, ci) -> float:
        """Per-shot energy variance V of the scheme at the state ``ci``."""
        from .measure import schemes

        psi = _real_ci(ci, self.space)
        if self.scheme == "df":
            return float(schemes.df_variance(self.active, psi, factorization=self.factorization,
                                             shift=self.shift).total)
        from .measure.pauli import QubitMapping, QubitState
        from .measure.targets import energy_target

        mapping = QubitMapping(self.active.norb) if self.mapping is None else self.mapping
        if self._target is None:
            self._target = energy_target(self.active, mapping)
        st = QubitState.from_ci(self.active.norb, self.active.nelec, psi, mapping)
        return float(schemes.scheme_variance(self._target, st, self.scheme, **self.scheme_options).total)

    def __call__(self, e_exact: float, ci: np.ndarray, context: str) -> float:
        if context not in _CONTEXTS:
            raise ValueError(f"context must be one of {_CONTEXTS}, got {context!r}")
        self.ledger.charge_energy(self.shots, context)
        if math.isinf(self.shots):
            return float(e_exact)
        v = self.frozen_variance if self.variance_mode == "frozen" else self.variance_of(ci)
        return float(e_exact) + math.sqrt(v / self.shots) * float(self.rng.standard_normal())


# ================================================================ snapshots
class SnapshotResidual:
    """Direct-route residual estimates from simulated matchgate-shadow snapshots (small systems).

    Drop-in replacement for :class:`ResidualNoise`: ``model(rd, ci, it)`` charges
    ``shots`` to the ledger and returns A_est = fill(A_hat), where A_hat_e is the
    average over ``shots`` snapshots of the matchgate-shadow estimator of <X_e>
    (module docstring, "Snapshot residual"); rd is not used. No Gaussian
    assumption; the element errors are correlated because all elements are read
    from the same snapshots. The dense state has 2^(2N) amplitudes (H4: 256,
    H6: 4096).

    Args:
        active: the active space.
        kind: element set ("spin_orbital" or "spin_free").
        shots: snapshots per residual estimate (a positive integer).
        rng, ledger: as for :class:`ResidualNoise`.
        mapping: qubit mapping (default blocked; the estimator's statistics do not
            depend on it).
        batch: snapshots per vectorized block (default max(32, 2^17 / 2^n): 512 for H4).
    """

    def __init__(self, active: ActiveSpace, *, kind: str = "spin_orbital", shots: int, rng,
                 ledger: ShotLedger | None = None, mapping=None, batch: int | None = None):
        from itertools import combinations

        from .measure.pauli import QubitMapping, coefficient_matrix, pauli_to_majorana
        from .measure.schemes import matchgate_lambda
        from .measure.targets import direct_targets

        self.active = active
        self.norb = int(active.norb)
        self.space = FCISpace.of(active)
        self.kind = kind
        self.shots = _check_shots(shots, integer=True)
        self.rng = _as_rng(rng)
        self.seed = _seed_record(self.rng)
        self.ledger = ShotLedger() if ledger is None else ledger
        self.mapping = QubitMapping(self.norb) if mapping is None else mapping
        if self.mapping.norb != self.norb:
            raise ValueError(f"mapping has norb={self.mapping.norb}, active space {self.norb}")
        self.n_qubits = self.mapping.n_qubits
        if self.n_qubits > 14:
            raise ValueError("SnapshotResidual simulates the dense state: at most 14 qubits")
        self._fill = _ElementFill(self.norb, kind)
        self.K = self._fill.K
        ops = direct_targets(self._fill.elements, active, self.mapping, kind)
        xs, zs, w = coefficient_matrix(ops)
        if w.nnz and float(np.max(np.abs(w.data.imag))) > 1e-12 * float(np.max(np.abs(w.data))):
            raise RuntimeError("internal error: X_e has non-real Pauli coefficients")
        w = w.real.tocsc()
        ident = (xs == 0) & (zs == 0)
        self._const = np.asarray(w[:, np.flatnonzero(ident)].sum(axis=1)).ravel()
        nid = np.flatnonzero(~ident)
        masks, _ = pauli_to_majorana(xs[nid], zs[nid])
        masks = np.atleast_1d(np.asarray(masks, dtype=np.uint64))
        deg = np.bitwise_count(masks).astype(np.int64)
        if np.any(deg % 2):
            raise RuntimeError("internal error: X_e contains odd Majorana monomials")
        lam = matchgate_lambda(self.n_qubits, deg // 2)
        order = np.argsort(masks)
        self._masks = masks[order]
        self._xs = xs[nid][order]
        self._zs = zs[nid][order]
        # W'_ej = W_ej / lambda_j over the sorted non-identity strings: A_hat = W' (mean covered values) + const
        self._wprime = (w[:, nid[order]] @ _diag_sparse(1.0 / lam[order])).tocsr()
        self.n_strings = int(nid.size)
        kmax = int(deg.max() // 2) if deg.size else 0
        self._subsets = [np.array(list(combinations(range(self.n_qubits), k)), dtype=np.intp)
                         for k in range(1, kmax + 1)]
        self.batch = int(batch) if batch is not None else max(32, (1 << 17) // (1 << self.n_qubits))
        self._pair_ops: dict = {}
        self.info = {"n_strings": self.n_strings, "k_max": kmax, "max_eigenvalue_deviation": 0.0}

    @property
    def elements(self) -> np.ndarray:
        return self._fill.elements

    def fill(self, values) -> np.ndarray:
        return self._fill(np.asarray(values, dtype=float))

    # ------------------------------------------------------------ sampling
    def _dense(self, ci) -> np.ndarray:
        from .measure.pauli import QubitState

        psi = _real_ci(ci, self.space)
        return QubitState.from_ci(self.norb, self.active.nelec, psi, self.mapping).to_dense()

    def _pair_operator(self, a: int, b: int):
        """(src, phase) with (i gamma_a gamma_b phi)[c] = phase[c] phi[src[c]] (cached per pair)."""
        from .measure.pauli import majorana_to_pauli

        key = (int(a), int(b))
        hit = self._pair_ops.get(key)
        if hit is None:
            one = np.uint64(1)
            ph, px, pz = majorana_to_pauli((one << np.uint64(a)) | (one << np.uint64(b)))   # gamma_a gamma_b = ph P
            coef = complex(1j * ph)
            if abs(coef.imag) > 1e-12 or abs(abs(coef.real) - 1.0) > 1e-12:
                raise RuntimeError("internal error: i gamma_a gamma_b is not +-P(x, z)")
            px, pz = np.uint64(px), np.uint64(pz)
            c = np.arange(1 << self.n_qubits, dtype=np.uint64)
            src = c ^ px
            sign = 1.0 - 2.0 * (np.bitwise_count(src & pz) & 1)
            ipow = (1, 1j, -1, -1j)[int(np.bitwise_count(px & pz)) & 3]
            hit = (src.astype(np.intp), (coef.real * ipow) * sign)
            self._pair_ops[key] = hit
        return hit

    def _sample_batch(self, psi: np.ndarray, r: int):
        """(rows, cols, values) of the covered strings of r snapshots (values +-1).

        Each snapshot draws a uniformly random perfect matching of the 2n Majoranas
        and measures its n commuting pair operators i gamma_a gamma_b one after the
        other (projective, Born probabilities; the state is kept unnormalized with
        its squared norm tracked). The pairs of each matching are measured in the
        order of their smaller Majorana, so that at every step many snapshots
        measure the same pair and share one vectorized update.
        """
        from .measure.schemes import random_matchings

        n = self.n_qubits
        one = np.uint64(1)
        mt = random_matchings(n, r, self.rng)
        mt = np.take_along_axis(mt, np.argsort(mt[:, :, 0], axis=1)[:, :, None], axis=1)
        phi = np.broadcast_to(psi, (r, psi.size)).astype(np.complex128)
        nrm2 = np.full(r, float(np.vdot(psi, psi).real))
        pair_masks = (one << mt[:, :, 0].astype(np.uint64)) | (one << mt[:, :, 1].astype(np.uint64))
        m2 = 2 * n
        for t in range(n):
            pid = mt[:, t, 0] * m2 + mt[:, t, 1]
            order = np.argsort(pid, kind="stable")
            uniq, start = np.unique(pid[order], return_index=True)
            stops = np.r_[start[1:], r]
            u = self.rng.random(r)
            for key, lo, hi in zip(uniq, start, stops):
                rows = order[lo:hi]
                src, phase = self._pair_operator(key // m2, key % m2)
                sub = phi[rows]
                op = sub[:, src] * phase
                ev = np.einsum("rb,rb->r", np.conj(sub), op).real / nrm2[rows]
                s = np.where(u[rows] < 0.5 * (1.0 + ev), 1.0, -1.0)
                phi[rows] = 0.5 * (sub + s[:, None] * op)
                nrm2[rows] *= 0.5 * (1.0 + s * ev)
        # covered strings: unions of 1..k_max pairs of the matching, looked up among the targets' strings
        rows, cols = [], []
        for sub in self._subsets:
            u = np.bitwise_or.reduce(pair_masks[:, sub], axis=2)    # (r, C(n, k))
            idx = np.searchsorted(self._masks, u)
            idx = np.minimum(idx, self._masks.size - 1)
            hit = self._masks[idx] == u
            rr, cc = np.nonzero(hit)
            rows.append(rr)
            cols.append(idx[rr, cc])
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        # phi_r is a joint eigenvector of every covered string: read the eigenvalue off one amplitude
        b = np.argmax(np.abs(phi), axis=1).astype(np.uint64)
        x, z = self._xs[cols], self._zs[cols]
        src = b[rows] ^ x
        sign = 1.0 - 2.0 * (np.bitwise_count(src & z) & 1)
        ipow = np.array([1, 1j, -1, -1j])[np.bitwise_count(x & z) & 3]
        v = ipow * sign * phi[rows, src.astype(np.intp)] / phi[rows, b[rows].astype(np.intp)]
        dev = float(np.max(np.abs(v - np.round(v.real)), initial=0.0))
        self.info["max_eigenvalue_deviation"] = max(self.info["max_eigenvalue_deviation"], dev)
        return rows, cols, np.round(v.real)

    def string_sums(self, ci, n_snapshots: int) -> np.ndarray:
        """Sum over ``n_snapshots`` snapshots of [covered] x measured value, per target string."""
        psi = self._dense(ci)
        sums = np.zeros(self.n_strings)
        done = 0
        while done < n_snapshots:
            r = min(self.batch, n_snapshots - done)
            _, cols, vals = self._sample_batch(psi, r)
            sums += np.bincount(cols, weights=vals, minlength=self.n_strings)
            done += r
        return sums

    def estimate(self, ci, n_snapshots: int | None = None) -> np.ndarray:
        """Element estimates A_hat (K,) from ``n_snapshots`` snapshots (default: ``shots``)."""
        s = int(self.shots if n_snapshots is None else _check_shots(n_snapshots, integer=True))
        return self._wprime @ (self.string_sums(ci, s) / s) + self._const

    def snapshot_estimates(self, ci, n_snapshots: int) -> np.ndarray:
        """Single-snapshot estimates x_{r,e}, shape (n_snapshots, K) (for statistics)."""
        import scipy.sparse as sp

        psi = self._dense(ci)
        out = []
        done = 0
        while done < n_snapshots:
            r = min(self.batch, n_snapshots - done)
            rows, cols, vals = self._sample_batch(psi, r)
            v = sp.csr_matrix((vals, (rows, cols)), shape=(r, self.n_strings))
            out.append(np.asarray((v @ self._wprime.T).todense()) + self._const[None, :])
            done += r
        return np.concatenate(out, axis=0)

    def element_covariance(self, ci, n_snapshots: int, mean=None) -> tuple[np.ndarray, np.ndarray]:
        """Empirical per-snapshot (mean, covariance (K, K)) of the element estimators at ``ci``.

        The covariance is taken about ``mean`` if given (e.g. the exact residual on the
        element set, which the estimator is unbiased for), else about the sample mean
        (with the 1/(n - 1) normalization). Accumulated in batches; does not charge the
        ledger. Input for ``ResidualNoise(per_element_covariance=...)``.
        """
        n_snapshots = int(_check_shots(n_snapshots, integer=True))
        s1 = np.zeros(self.K)
        s2 = np.zeros((self.K, self.K))
        center = None if mean is None else np.asarray(mean, dtype=float)
        done = 0
        while done < n_snapshots:
            r = min(max(self.batch, 1024), n_snapshots - done)
            x = self.snapshot_estimates(ci, r)
            if center is not None:
                x = x - center[None, :]
            s1 += x.sum(axis=0)
            s2 += x.T @ x
            done += r
        if center is not None:
            return center + s1 / n_snapshots, s2 / n_snapshots
        m = s1 / n_snapshots
        return m, (s2 - n_snapshots * np.outer(m, m)) / max(n_snapshots - 1, 1)

    def __call__(self, rd: ResidualData, ci: np.ndarray, it: int) -> np.ndarray:
        self.ledger.charge_residual(self.shots, it)
        return _structured(self.fill(self.estimate(ci)))


def _diag_sparse(d: np.ndarray):
    import scipy.sparse as sp

    return sp.diags(np.asarray(d, dtype=float), format="csc")

"""Windowed QPE outcome laws, tails, the model spectrum of H' and the bracketed acceptance rule B (F39, F40).

Tags and formulas F1-F19 as in :mod:`thc_engine_cqe.ft.primitives` (docs/m3-cost-model.md); F39 and
F40 continue that numbering. Energies in Hartree, phases in radians, distances in grid units.

Windowed outcome law (F39) [derived; the Fejer case is ``ft.walk``; the sine window cited:
Berry and Wiseman, PRL 85, 5098 (2000)]
------------------------------------------------------------------------------------------
An M-point control register prepared in sum_x a_x |x> (sum_x a_x^2 = 1, a real), controlled W^x
and the inverse QFT over Z_M return m with amplitude

    A(m | phi) = M^{-1/2} sum_{x=0}^{M-1} a_x exp(i x (phi - 2 pi m / M))

on a walk eigenvector of eigenphase phi. The walk W block-encodes (H' - c)/lambda, and an
eigenstate psi_j of H' splits into the two walk eigenvectors of phases +-theta_j,
theta_j = arccos((E_j - c)/lambda), with weight 1/2 each, so

    K_j(m) = (|A(m | theta_j)|^2 + |A(m | -theta_j)|^2) / 2,   P(m) = sum_j p_j K_j(m),
    E(m) = c + lambda cos(2 pi m / M),

with p_j = |<psi_j|Psi>|^2 (:func:`eigen_kernel`, :func:`windowed_outcome_law`). Windows
(:func:`phase_window`): "rect" a_x = M^{-1/2} (the Fejer kernel of ``ft.walk.qpe_outcome_law``,
equal to it at M = 2^t), "sine" a_x ~ sin(pi (x + 1)/(M + 1)) (the RMS-optimal register state),
"kaiser" a_x ~ I_0(pi alpha (1 - (2x/(M - 1) - 1)^2)^{1/2}). For a real window
|A(m | -theta)|^2 = |A(-m | theta)|^2, so the mass of K_j on any set of outcomes symmetric under
m -> -m (every energy window {m : E(m) in I} is) is the mass of one branch. Kernels are evaluated
by an FFT of a_x exp(2 pi i x f / M) with f the fractional part of M theta / 2 pi; the integer
part is a cyclic shift of the outcomes, applied exactly (as ``ft.walk``'s Fejer rows).

Assumptions of the costing:

* (A2) c is the identity coefficient of the Jordan-Wigner form of H', ecore included
  (:func:`walk_offset`) [assumption; exact for ``ft.walk.PauliLCUWalk``; OpenFermion's LCUs may use
  another offset, which moves theta_0 only]. In closed form [derived: Tr a+_i a_j = 2^{2N-1}
  delta_ij, Tr n_i n_j = 2^{2N-2} for i != j]

      c = Tr H' / 2^{2N} = ecore + sum_p h_pp + 1/2 sum_pr (pp|rr) - 1/4 sum_pr (pr|rp).

* (A3) One C_QPE buys M = 2 n_iter phase points, since a controlled walk step kicks back
  2 theta (:func:`qpe_points`) [assumption, consistent with OpenFermion's
  n_iter = ceil(pi lambda / (2 dE)) [OF]: the sine window's RMS energy error at M = 2 n_iter is
  then dE sin(theta_0), tested]. Under the other reading every QPE length doubles.

The energy grid spacing at the phase theta is g = 2 pi lambda sin(theta) / M, i.e.
g = 2 dE sin(theta) at M = 2 n_iter; F40 evaluates it at theta_0 (:meth:`ModelSpectrum.grid`).
The exact phase distance of a level differs from its energy distance over g by the ratio of
sin(theta_0) to sin between the two (0.997-0.998 from E_0' to E_1' on stretched N2). The costs keep
M3's window model (an outcome within dE of E_j with probability w p_j, w = 1); the windowed law
enters the identification error and the reported window masses only.

Tails (:func:`tail_mass`, :func:`fejer_tail_bound`) [derived]
------------------------------------------------------------
tau_W(D) is the supremum over phase offsets of the one-sided tail of one branch, the mass at
grid distance >= D from its eigenphase on one side (up to M/2). A level D grid units above a
threshold puts at most tau_W(D) of its kernel below the threshold when it lies below the mirror
energy 2c - theta (the outcome set below the threshold then lies within one half circle of its
eigenphase; above the mirror energy, i.e. >= 2(c - theta) above the threshold, at most 2 tau_W).
With the outcomes at distances x + k (k >= 0) from the eigenphase, x in [D, D + 1) set by the phase
offset, the tail is T(x) = sum_k K(x + k), and T(x) - T(x + 1) = K(x) >= 0, so the supremum over
offsets is the supremum of T over [D, D + 1); where the offset puts a lattice point at distance
exactly D (x = D) the tail jumps up, and near the main lobe and at the n_v steps the supremum sits
there. The code evaluates the finite-register law (M = 4096) at 64 equispaced offsets refined around
the largest and at the offset of that jump point; its continuum limit is the lattice law of F34
[derived: the Fourier transform of sqrt(2) sin(pi xi) on [0, 1]],

    P(delta) = 8 cos^2(pi delta) / (pi^2 (1 - 4 delta^2)^2) = (1/2) sinc^2(delta - 1/2) / (delta + 1/2)^2,
    sum_{k >= 0} P(x + k) = (8 cos^2(pi x) / pi^2) [(psi_1(x - 1/2) + psi_1(x + 1/2))/16
                                                     - 1/(8 (x - 1/2))],

by partial fractions of (1 - 4y^2)^{-2} and the telescoping sum of the simple poles (psi_1 the
trigamma function; sinc(u) = sin(pi u)/(pi u), regular at delta = 1/2), so tau_sine(D) =
sup_{x in [D, D + 1)} sum_k P(x + k): 1.84e-4 at D = 5 and 7.3e-7 at D = 28.7
(tests/test_ft_acceptance.py). For the sine window ``tail_mass`` returns the larger of the
finite-register supremum and this continuum supremum (x = D included), which bounds the tails of the
longer registers of the hand-off QPE (M = 2 n_iter > 4096) as well [build check: at D = 1.30 the
finite-register value with the jump point is 9.352e-3 and the continuum 9.389e-3, against 9.264e-3
from the equispaced offsets alone; at D = 26.03 the continuum value 1.0055e-6 needs two runs where the
M = 4096 law, 9.989e-7, needs one]. Fejer bound: the rect kernel is
sin^2(pi u)/(M^2 sin^2(pi u / M)) <= 1/(4 u^2) for |u| <= M/2 (sin x >= 2x/pi on [0, pi/2]), and
the outcomes at circle distance >= D sit at D + a + n (a in [0, 1), n >= 0) on each side, so their
mass is <= 2 [1/(4D^2) + int_D^inf du/(4u^2)] = (1 + 1/D)/(2D) (two-sided, every phase).

Model spectrum of H' (:func:`model_spectrum`)
--------------------------------------------
Every eigenpair of the selected model H' (``qpe.encoding.model(active)``) in the (n_alpha,
n_beta) sector of the hand-off state, by dense diagonalization (the sector matrix is PySCF's
``direct_spin1.pspace`` over all determinants, ecore added; tested against ``fci_ops``), with
<S^2> (``pyscf.fci.spin_op.spin_square0``) and the dominant irrep of every eigenvector.
Within a degenerate level (energies within 1e-10 Ha) the eigenvectors are rotated to diagonalize
the irrep label and then S^2, so every vector of a symmetry-conserving H' is irrep- and
spin-pure. Where H' breaks the point group the irreps are dominant ones: THC fits, and SF
truncations that cut through a degenerate eigenspace of the ERI supermatrix [build check:
n2_r2.4's selected SF L = 16 has forbidden integrals up to 1.3e-5 Ha and mixes near-degenerate
pairs of different irreps, eigenvector irrep weights down to 0.9945 (the pair 235.2 mHa above
E_0'), while HF's weight outside its irrep stays at round-off; h4, n2_r2.0 and n2_r2.8 conserve
D2h to round-off]. The kernel terms of rule B sum over the whole sector, so they do not depend on
the labels.
E_1 variants, all above the ground state E_0' (index 0): ``e1_singlet``, the lowest excited
singlet of the ground state's irrep (M3's E1 on H'); ``e1_any_spin``, the lowest excited level of
that irrep of any S (the sector of compiled iterates, F41); ``e1_any_irrep``, the lowest excited
singlet of any irrep (THC encodings). Dense matrices limit it to N <= 8 (4900 determinants at
N = 8: about 0.5 s for the matrix, 15 s for eigh and 11 s for S^2 on one thread).

Rule B: the bracketed threshold (F40) [derived]
-----------------------------------------------
Rule: fix theta and n_v >= 1; repeat independent QPE runs on Psi until an outcome has
E(m) < theta, then run n_v - 1 further QPEs (verification runs) on the post-measurement state;
accept if every one of them returns E(m) < theta, else start again from Psi.

* Soundness. QPE maps every walk eigenvector to itself, so after outcome m the walk state is
  proportional to sum_{j, +-} (p_j/2)^{1/2} A(m | +-theta_j) |w_j^{+-}>, and n runs return
  (m_1 .. m_n) with probability sum_{j, +-} (p_j/2) prod_i |A(m_i | +-theta_j)|^2. The set
  A_theta = {m : E(m) < theta} is symmetric under m -> -m, so an attempt passes with probability
  sum_j p_j K_j(A_theta)^n and the accepted state, distributed as one attempt conditioned on
  passing, carries the weight

      eps_id(theta, n_v) = sum_{j>=1} p_j K_j^{n_v} / sum_j p_j K_j^{n_v},   K_j = K_j(A_theta),

  outside the walk eigenvectors of psi_0 (:func:`acceptance_stats`). If every populated excited
  level lies at or above E1lb (and below the mirror energy 2c - theta, see Tails),

      eps_id <= [(1 - p_0)/p_0] [tau_W(D) / K_0]^{n_v},   D = (E1lb - theta)/g

  (:func:`identification_bound`). It grows as 1/p_0, so soundness is stated for every input state
  with p_0 >= p_min. In the window model eps_id = 0 for every theta <= E1lb - dE.
* Verification runs (:func:`verification_runs`): n_v is the smallest n with
  [(1 - p_min)/p_min] tau_W(D)^n <= eps_id at the design floor p_min = 1e-3 and eps_id = 1e-3
  (:data:`P_MIN`, :data:`EPS_ID`) [assumption: the hand-off is designed for p_0 >= p_min]: for the
  sine window n_v = 1 for D > 26, 2 for 3 < D <= 26 and 3 for 1.3 <~ D <= 3 (tau_W is a step
  function of D beyond the main lobe: its supremum puts a lattice point on the next integer). It
  needs no p_0, so it is realizable. Each verification run costs C_QPE / K_0.
* Termination: the number of attempts is geometric with mean <= 1/(p_0 K_0^{n_v}); in the window
  model K_0 >= w once theta >= E_0' + dE.
* Forms (:func:`acceptance_bracket`): B_U, theta = U + dE with a certified U >= E_0' (F42),
  D_U = (E1lb - U - dE)/g, whose bracket [U + dE, E1lb - dE] is non-empty iff U <= E1lb - 2dE
  (the window-model certificate; M3's rule E is B_U with U = E_k + dE); B_L without U,
  theta = E1lb - D_L g with D_L = 5, hence n_v = 2, sound to eps_id <= 1e-3 for every input state
  with p_0 >= tau_W(5)^2/(eps_id + tau_W(5)^2) = 3.4e-5 (with one run only for p_0 >= 0.155). B_L
  terminates iff E_0' + dE <= theta; that is certified only where a certified U has
  U + dE <= theta, else the flag "termination_not_certified". The hand-off takes the form with
  the smaller n_v, B_U on ties.
* A dedicated energy certificate (F42) is a random bound: its realized value is U^ = U + delta,
  delta ~ N(0, sigma^2) about the planned U, so D(U^) = (E1lb - U^ - dE)/g and n_v(U^) are random.
  The realizable protocol fixes n_v after U^ is known (:func:`expected_verification_runs`): B_U
  with n_v(U^) runs where U^ <= E1lb - 2 dE (admissible), or B_L's n_l runs where that is fewer or
  U^ is inadmissible, so with the thresholds t_n = min{D : n_v(D) <= n}
  (:func:`verification_run_threshold`) and a = dE/g

      E[n_v] = 1 + sum_{n=1}^{n_l - 1} P(D(U^) < max(a, t_n)),   P(D(U^) < t) = Phi((t - D(U)) g / sigma);

  where B_L does not terminate, inadmissible outcomes repeat the certificate (on average
  1/P(D(U^) >= a) times) and the runs are E[n_v | D(U^) >= a].
* Logical failures (:func:`failure_budget`, ``acceptance_stats(p_run=...)``): a failed run
  returns an arbitrary outcome and state, which the threshold accepts with probability f <= 1
  (f = 1 - theta_0/pi for a uniformly random register, the share of outcomes below E_0'; f = 1 in
  the costing). An unverified post-selected run enriches accepted failures, which contribute
  f p_run / (P_acc + f p_run); a later correct QPE run on the returned state accepts the excited
  part of a failed run's state with probability <= tau_W, which removes the enrichment, and the
  re-testing run itself adds f p_run. Every scheme is costed at the output failure
  eps_fail = 1e-3 (:data:`EPS_FAIL`): unverified post-selected runs at p_run = eps_fail p_min / f
  (p_min in place of the unknown acceptance probability), every other run at
  p_run = eps_fail / (2f). The kernel term and the failure term are budgeted separately, so the
  total identification error is <= 2e-3; ``AcceptanceStats`` reports both.

The per-level masses K_j(A_theta) (and below theta outside |E(m) - E_0'| <= dE) depend on the
spectrum, the register (M, window, alpha), theta and dE only, not on the state: ``acceptance_stats``
keeps them per model spectrum (a weak reference; the levels computed so far, per register and
threshold), so the states of one system share them. The values are those of a fresh evaluation
(each level's kernel is its own FFT row).

Observed [build check] (tests/test_ft_acceptance.py; selected SF encodings, sine at
M = 2 n_iter unless noted): the rect law equals ``ft.walk.qpe_outcome_law`` on H4 for
M = 2^10 .. 2^14; on n2_r2.4 (SF L = 16, lambda = 7.5908 Ha, n_iter = 11924)
E_1' - E_0' = 230.863 mHa, E_0' - E_0 = -0.368 mHa, the sine RMS over 8 phase offsets is
0.9856 mHa against dE sin theta_0 = 0.9857 mHa (half of it at M = 4 n_iter), and HF accepted
by B_U with U = CISD(H') has eps_id = 2.38e-4 (Kaiser(3) at M = 4 n_iter: 3.76e-8); n2_r2.0 HF
under the literal theta = E_1' - dE: eps_id = 0.180; n2_r2.8 HF under B_L (D_L = 5): 1.6e-4 with
one run, 1.0e-8 with two.
"""

from __future__ import annotations

import functools
import math
import weakref
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import scipy.special

from ..integrals import ActiveSpace
from .qpe import QPECost
from .symmetry import determinant_irreps

__all__ = [
    "WINDOWS",
    "P_MIN",
    "EPS_ID",
    "EPS_FAIL",
    "FAILURE_KINDS",
    "BRACKET_FORMS",
    "phase_window",
    "walk_offset",
    "qpe_points",
    "eigen_kernel",
    "windowed_outcome_law",
    "tail_mass",
    "fejer_tail_bound",
    "verification_runs",
    "verification_run_threshold",
    "ExpectedRuns",
    "expected_verification_runs",
    "failure_budget",
    "ModelSpectrum",
    "model_spectrum",
    "Bracket",
    "acceptance_bracket",
    "AcceptanceStats",
    "acceptance_stats",
    "identification_bound",
]

WINDOWS = ("rect", "sine", "kaiser")
#: F40: design floor p_min of the soundness statements, kernel target eps_id, output failure eps_fail.
P_MIN = 1e-3
EPS_ID = 1e-3
EPS_FAIL = 1e-3
#: Run classes of the logical-failure budget (F40): unverified post-selected runs, and all others.
FAILURE_KINDS = ("unverified", "retested")
BRACKET_FORMS = ("B_U", "B_L")

# Eigenstates with weight <= _WMIN are left out of acceptance_stats (as windowed_outcome_law's default).
_WMIN = 1e-14
# Relative tolerance of |E - c| <= lambda (the walk must bound the spectrum of H' - c).
_LAMBDA_RTOL = 1e-12
# Energies within this distance (Hartree) form one degenerate level of the model spectrum: eigh splits exact
# degeneracies by ~1e-13 Ha; a symmetry-breaking H' splits levels by more, and they keep H''s eigenvectors.
_DEGENERACY_TOL = 1e-10
# <S^2> below this is a singlet.
_SINGLET_TOL = 1e-6
# Largest number of runs verification_runs searches.
_MAX_RUNS = 64
# verification_runs evaluates tau_W at min(D, _TAIL_DMAX): tau_W is non-increasing in D, so this bounds it.
_TAIL_DMAX = 1000.0
# Complex entries of one FFT batch of kernels (memory about 16 bytes each).
_BATCH = 1 << 22
# Level masses of acceptance_stats per model spectrum: {(M, window, alpha, theta, dE): (k_acc, k_far)} with nan for
# levels not computed yet; at most _MASS_KEYS registers and thresholds per spectrum (the oldest dropped).
_LEVEL_MASSES: "weakref.WeakKeyDictionary[ModelSpectrum, dict]" = weakref.WeakKeyDictionary()
_MASS_KEYS = 64
# Sub-offsets of the local refinement of tail_mass around its largest offset.
_REFINE = 16
# Points of [D, D + 1) at which the continuum sine tail is scanned (then refined around the largest, 2 _REFINE + 1
# points over two grid spacings).
_CONTINUUM_POINTS = 2048
# Bisection steps of verification_run_threshold (the bracket [0, 1000] halves to 1e-12 grid units).
_THRESHOLD_STEPS = 60


# ------------------------------------------------------------------ validation helpers
def _int_at_least(value, low: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < low:
        raise ValueError(f"{name} must be an integer >= {low}, got {value!r}")
    return int(value)


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


def _probability(value, name: str, *, open_low: bool = True, open_high: bool = True) -> float:
    v = float(value)
    lo_ok = v > 0.0 if open_low else v >= 0.0
    hi_ok = v < 1.0 if open_high else v <= 1.0
    if not (math.isfinite(v) and lo_ok and hi_ok):
        lo, hi = ("(" if open_low else "["), (")" if open_high else "]")
        raise ValueError(f"{name} must lie in {lo}0, 1{hi}, got {value!r}")
    return v


def _check_window(window: str, alpha: float) -> float:
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {WINDOWS}, got {window!r}")
    a = float(alpha)
    if window == "kaiser" and not (math.isfinite(a) and a >= 0.0):
        raise ValueError(f"the Kaiser alpha must be finite and >= 0, got {alpha!r}")
    return a


def _walk_x(energies: np.ndarray, lam: float, c: float) -> np.ndarray:
    """(E - c)/lambda, clipped to [-1, 1] after checking that lambda bounds |E - c|."""
    x = (np.asarray(energies, dtype=float) - c) / lam
    if x.size and float(np.max(np.abs(x))) > 1.0 + _LAMBDA_RTOL:
        raise ValueError(f"lam = {lam:g} does not bound |E - c| (max {float(np.max(np.abs(x))) * lam:g} Ha)")
    return np.clip(x, -1.0, 1.0)


# ------------------------------------------------------------------ windows and kernels
def phase_window(n_points: int, window: str = "sine", alpha: float = 3.0) -> np.ndarray:
    """Real, normalized control-register amplitudes a_x, x = 0..M-1, of ``window`` (F39).

    "rect": a_x = M^{-1/2}; "sine": a_x ~ sin(pi (x + 1)/(M + 1)); "kaiser": a_x ~
    I_0(pi alpha (1 - (2x/(M - 1) - 1)^2)^{1/2}) (alpha = 0 is rect; evaluated in the log domain,
    so large alpha does not overflow). ``alpha`` is ignored for the other windows. M >= 2.
    """
    m = _int_at_least(n_points, 2, "n_points")
    alpha = _check_window(window, alpha)
    x = np.arange(m, dtype=float)
    if window == "rect":
        a = np.ones(m)
    elif window == "sine":
        a = np.sin(np.pi * (x + 1.0) / (m + 1.0))
    else:
        u = 2.0 * x / (m - 1) - 1.0
        z = np.pi * alpha * np.sqrt(np.clip(1.0 - u * u, 0.0, None))
        log_i0 = z + np.log(scipy.special.i0e(z))
        a = np.exp(log_i0 - log_i0.max())
    return a / np.linalg.norm(a)


def walk_offset(model_active: ActiveSpace) -> float:
    """(A2): c = Tr H'/2^{2N}, the identity coefficient of the JW form of H', ecore included (Hartree).

    Closed form of the module docstring; equals ``ft.walk.PauliLCUWalk(model_active).c0`` (tested).
    """
    if not isinstance(model_active, ActiveSpace):
        raise ValueError(f"model_active must be an ActiveSpace, got {type(model_active).__name__}")
    h1, eri = model_active.h1, model_active.eri
    coulomb = float(np.einsum("pprr->", eri))
    exchange = float(np.einsum("prrp->", eri))
    return float(model_active.ecore + np.trace(h1) + 0.5 * coulomb - 0.25 * exchange)


def qpe_points(qpe: QPECost, multiple: int = 2) -> int:
    """(A3): M = multiple * n_iter control-register points; one C_QPE buys multiple = 2."""
    if not isinstance(qpe, QPECost):
        raise ValueError(f"qpe must be a QPECost, got {type(qpe).__name__}")
    k = _int_at_least(multiple, 1, "multiple")
    return k * _int_at_least(qpe.iterations, 1, "qpe.iterations")


def _branch_kernels(thetas: np.ndarray, a: np.ndarray) -> np.ndarray:
    """|A(m | theta)|^2 for each theta (rows) and m = 0..M-1 (columns); each row sums to 1.

    A(m | theta) = M^{-1/2} FFT[a_x exp(2 pi i x f / M)](m - n) with M theta / 2 pi = n + f,
    n the nearest integer: the integer part is a cyclic shift, so the phase argument stays below pi.
    """
    m = a.size
    y = np.asarray(thetas, dtype=float) * (m / (2.0 * np.pi))
    n = np.rint(y)
    f = y - n
    x = np.arange(m, dtype=float)
    out = np.empty((y.size, m))
    rows = max(1, _BATCH // m)
    for i0 in range(0, y.size, rows):
        sl = slice(i0, min(y.size, i0 + rows))
        b = a[None, :] * np.exp((2j * np.pi / m) * np.outer(f[sl], x))
        amp = np.fft.fft(b, axis=1)
        k = (amp.real ** 2 + amp.imag ** 2) / m
        shift = n[sl].astype(np.int64) % m
        idx = (np.arange(m)[None, :] - shift[:, None]) % m
        out[sl] = np.take_along_axis(k, idx, axis=1)
    return out


def _mirror(k: np.ndarray) -> np.ndarray:
    """k(-m mod M) along the last axis."""
    return np.roll(k[..., ::-1], 1, axis=-1)


def eigen_kernel(energy: float, lam: float, c: float, a: np.ndarray) -> np.ndarray:
    """K_j(m) of one eigenstate at ``energy`` for the window amplitudes ``a`` (F39); sums to 1.

    K_j(m) = (|A(m | theta_j)|^2 + |A(m | -theta_j)|^2)/2, theta_j = arccos((E_j - c)/lambda);
    ValueError unless lambda bounds |E_j - c|.
    """
    lam = _positive(lam, "lam")
    c = _finite(c, "c")
    e = _finite(energy, "energy")
    aw = np.asarray(a, dtype=float).ravel()
    if aw.size < 2 or not np.all(np.isfinite(aw)) or abs(float(aw @ aw) - 1.0) > 1e-10:
        raise ValueError("a must be a finite, normalized window of length >= 2 (phase_window)")
    theta = float(np.arccos(_walk_x(np.array([e]), lam, c))[0])
    k = _branch_kernels(np.array([theta]), aw)[0]
    return 0.5 * (k + _mirror(k))


def _check_spectrum_input(energies, weights) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(energies, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if e.shape != w.shape:
        raise ValueError(f"energies and weights differ in length ({e.size}, {w.size})")
    if not (np.all(np.isfinite(e)) and np.all(np.isfinite(w))):
        raise ValueError("energies and weights must be finite")
    if np.any(w < 0.0):
        raise ValueError(f"weights must be >= 0, got min {float(w.min()):g}")
    return e, w


def windowed_outcome_law(energies: Sequence[float] | np.ndarray, weights: Sequence[float] | np.ndarray,
                         lam: float, c: float, n_points: int, *, window: str = "sine", alpha: float = 3.0,
                         wmin: float = 1e-14) -> tuple[np.ndarray, np.ndarray]:
    """(P(m), E(m)) of an M-point windowed QPE on the walk of (H' - c)/lambda (F39).

    Args:
        energies: eigenvalues E_j of H' in the sector of the state (Hartree), bounded by lambda.
        weights: p_j >= 0 (not renormalized); eigenstates with p_j <= ``wmin`` are left out, so P
            sums to the sum of the kept weights.
        lam, c: the walk block-encodes (H' - c)/lambda.
        n_points: M >= 2; "rect" at M = 2^t is ``ft.walk.qpe_outcome_law`` with t bits.

    E(m) = c + lambda cos(2 pi m / M). Cost: one length-M FFT per kept eigenstate.
    """
    e, w = _check_spectrum_input(energies, weights)
    lam = _positive(lam, "lam")
    c = _finite(c, "c")
    wmin = float(wmin)
    if not (math.isfinite(wmin) and wmin >= 0.0):
        raise ValueError(f"wmin must be finite and >= 0, got {wmin!r}")
    a = phase_window(n_points, window, alpha)
    m = a.size
    keep = w > wmin
    theta = np.arccos(_walk_x(e[keep], lam, c))
    q = np.zeros(m)
    wk = w[keep]
    rows = max(1, _BATCH // m)
    for i0 in range(0, theta.size, rows):
        q += wk[i0:i0 + rows] @ _branch_kernels(theta[i0:i0 + rows], a)
    return 0.5 * (q + _mirror(q)), c + lam * np.cos(2.0 * np.pi * np.arange(m) / m)


def tail_mass(distance: float, *, window: str, alpha: float = 3.0, n_points: int = 4096,
              n_offsets: int = 64) -> float:
    """tau_W(D): the largest one-sided tail of one branch at grid distance >= D (module docstring).

    The supremum over ``n_offsets`` equispaced phase offsets, refined on a finer grid around the
    largest, and over the offset that puts a lattice point at distance exactly D, of the mass at
    distances D <= delta <= M/2 from the eigenphase, at M = ``n_points``; for the sine window the
    larger of that and the continuum supremum sup_{x in [D, D + 1)} sum_k P(x + k) of F34's lattice
    law (x = D included), so that the value bounds the longer registers too. 0 <= D < M/2 - 1.
    Values are cached per argument tuple (the function is pure).
    """
    d = float(distance)
    m = _int_at_least(n_points, 8, "n_points")
    n_off = _int_at_least(n_offsets, 1, "n_offsets")
    if not (math.isfinite(d) and 0.0 <= d < m / 2 - 1):
        raise ValueError(f"distance must lie in [0, {m / 2 - 1:g}) grid units, got {distance!r}")
    alpha = _check_window(window, alpha)
    return _tail_mass_cached(d, window, alpha if window == "kaiser" else 0.0, m, n_off)


@functools.lru_cache(maxsize=4096)
def _tail_mass_cached(d: float, window: str, alpha: float, m: int, n_off: int) -> float:
    finite = _finite_tail_sup(d, window, alpha, m, n_off)
    return max(finite, _continuum_tail_sup(d)) if window == "sine" else finite


def _finite_tail_sup(d: float, window: str, alpha: float, m: int, n_off: int) -> float:
    """The finite-register supremum of tail_mass: equispaced offsets, their refinement and the jump point."""
    a = phase_window(m, window, alpha)

    def tails(s: np.ndarray) -> np.ndarray:
        # eigenphase at -s grid units: outcome m sits at distance m + s
        k = _branch_kernels(-2.0 * np.pi * s / m, a)
        lo = np.ceil(d - s).astype(np.int64)
        hi = np.floor(m / 2 - s).astype(np.int64)
        cs = np.concatenate([np.zeros((s.size, 1)), np.cumsum(k, axis=1)], axis=1)
        r = np.arange(s.size)
        return cs[r, hi + 1] - cs[r, lo]

    s0 = np.arange(n_off) / n_off
    t0 = tails(s0)
    best = float(s0[int(np.argmax(t0))])
    s1 = np.mod(best + np.linspace(-1.0, 1.0, 2 * _REFINE + 1) / n_off, 1.0)
    # the jump point: s = D - floor(D) is exact in floating point, so ceil(D - s) = floor(D) and the lattice
    # point at distance exactly D is counted
    jump = tails(np.array([d - math.floor(d)]))
    return float(max(np.max(t0), np.max(tails(s1)), jump[0]))


def _lattice_law(delta: np.ndarray) -> np.ndarray:
    """P(delta) of F34's continuum sine-window lattice law, delta >= 0, regular at delta = 1/2 (module docstring)."""
    delta = np.asarray(delta, dtype=float)
    return 0.5 * np.sinc(delta - 0.5) ** 2 / (delta + 0.5) ** 2


def _continuum_tail(x: np.ndarray) -> np.ndarray:
    """T(x) = sum_{k>=0} P(x + k) for x >= 0: the trigamma closed form at arguments >= 1, P(x) added below."""
    x = np.asarray(x, dtype=float)
    low = x < 1.0
    y = np.where(low, x + 1.0, x)
    psi = scipy.special.polygamma(1, y - 0.5) + scipy.special.polygamma(1, y + 0.5)
    closed = (8.0 * np.cos(np.pi * y) ** 2 / np.pi ** 2) * (psi / 16.0 - 1.0 / (8.0 * (y - 0.5)))
    return np.where(low, _lattice_law(x) + closed, closed)


def _continuum_tail_sup(d: float) -> float:
    """sup_{x in [D, D + 1)} T(x): a scan of _CONTINUUM_POINTS points from x = D, refined around the largest."""
    h = 1.0 / _CONTINUUM_POINTS
    x0 = d + h * np.arange(_CONTINUUM_POINTS)
    t0 = _continuum_tail(x0)
    best = float(x0[int(np.argmax(t0))])
    x1 = np.clip(best + h * np.linspace(-1.0, 1.0, 2 * _REFINE + 1), d, d + 1.0 - 0.5 * h / _REFINE)
    return float(max(np.max(t0), np.max(_continuum_tail(x1))))


def fejer_tail_bound(distance: float) -> float:
    """(1 + 1/D)/(2D): the rect-window (Fejer) mass at circle distance >= D, both sides, any phase (D > 0)."""
    d = _positive(distance, "distance")
    return (1.0 + 1.0 / d) / (2.0 * d)


def verification_runs(distance: float, *, p_min: float = P_MIN, eps_id: float = EPS_ID, window: str = "sine",
                      alpha: float = 3.0) -> int:
    """F40's n_v: the smallest n >= 1 with [(1 - p_min)/p_min] tau_W(D)^n <= eps_id (D > 0, grid units).

    tau_W from :func:`tail_mass` (M = 4096, 64 offsets and the jump point; the continuum supremum for
    the sine window); beyond D = 1000 the non-increasing tail is bounded by tau_W(1000). ValueError if
    no n <= 64 suffices.
    """
    d = _positive(distance, "distance")
    p_min = _probability(p_min, "p_min")
    eps_id = _probability(eps_id, "eps_id")
    tau = tail_mass(min(d, _TAIL_DMAX), window=window, alpha=alpha)
    odds = (1.0 - p_min) / p_min
    for n in range(1, _MAX_RUNS + 1):
        if odds * tau ** n <= eps_id:
            return n
    raise ValueError(f"tau_W({d:g}) = {tau:.3e}: no n <= {_MAX_RUNS} runs reach eps_id = {eps_id:g} "
                     f"at p_min = {p_min:g}")


def verification_run_threshold(n_runs: int, *, p_min: float = P_MIN, eps_id: float = EPS_ID, window: str = "sine",
                               alpha: float = 3.0) -> float:
    """t_n = min{D : verification_runs(D) <= n}, the distance (grid units) above which n runs suffice (F40).

    verification_runs(D) <= n iff [(1 - p_min)/p_min] tau_W(D)^n <= eps_id, which is monotone in D
    (tau_W is non-increasing), so t_n is found by bisection on [0, 1000] to 1e-12 grid units (for the
    sine window first on the continuum supremum alone, which is cheap and decides wherever it
    dominates the finite-register value). math.inf where n runs suffice at no distance (beyond
    D = 1000 verification_runs bounds the tail by tau_W(1000)), as for one run of the rect window,
    whose tail falls only as 1/D. Cached per argument tuple. [build check: t_1 = 26.036, t_2 = 3.032,
    t_3 = 1.295 for the sine window at the default p_min and eps_id; rect t_1 = inf, t_2 = 101.51.]
    """
    n = _int_at_least(n_runs, 1, "n_runs")
    if n > _MAX_RUNS:
        raise ValueError(f"n_runs must be <= {_MAX_RUNS}, got {n_runs!r}")
    alpha = _check_window(window, alpha)
    return _run_threshold(n, _probability(p_min, "p_min"), _probability(eps_id, "eps_id"), window,
                          alpha if window == "kaiser" else 0.0)


@functools.lru_cache(maxsize=1024)
def _run_threshold(n: int, p_min: float, eps_id: float, window: str, alpha: float) -> float:
    odds = (1.0 - p_min) / p_min

    def bisect(ok, lo: float, hi: float) -> float:
        for _ in range(_THRESHOLD_STEPS):
            mid = 0.5 * (lo + hi)
            if ok(mid):
                hi = mid
            else:
                lo = mid
        return hi

    def full(dist: float) -> bool:
        return dist > 0.0 and odds * tail_mass(min(dist, _TAIL_DMAX), window=window, alpha=alpha) ** n <= eps_id

    if not full(_TAIL_DMAX):
        return math.inf                                 # n runs suffice nowhere: tau_W(1000) bounds the tail beyond
    lo, hi = 0.0, _TAIL_DMAX
    if window == "sine":
        t_c = bisect(lambda dist: odds * _continuum_tail_sup(dist) ** n <= eps_id, lo, hi)
        if full(t_c):
            return t_c                                  # below t_c the continuum part alone exceeds the target
        lo = t_c
    return bisect(full, lo, hi)


@dataclass(frozen=True)
class ExpectedRuns:
    """F40's runs at a random certified bound U^ ~ N(U, sigma^2) (:func:`expected_verification_runs`).

    Attributes:
        runs: E[n_v] (with a fallback form), or E[n_v | U^ admissible] (without one).
        admissible: P(U^ <= E1lb - 2 dE).
        planned: n_v at the planned U (D(U) = (E1lb - U - dE)/g).
        fallback: the fallback's n_v (B_L's n_l), None without one.
    """

    runs: float
    admissible: float
    planned: int
    fallback: int | None


def expected_verification_runs(u_upper: float, sigma: float, e1_lower: float, de: float, *, grid: float,
                               fallback_runs: int | None = None, p_min: float = P_MIN, eps_id: float = EPS_ID,
                               window: str = "sine", alpha: float = 3.0) -> ExpectedRuns:
    """F40's n_v at the realized bound U^ = U + delta, delta ~ N(0, sigma^2), of a dedicated certificate (F42).

    The protocol decides after U^ is known: B_U at U^ + dE with n_v(U^) = verification_runs(D(U^))
    runs, D(U^) = (E1lb - U^ - dE)/g, where U^ <= E1lb - 2 dE (D(U^) >= a = dE/g); with
    ``fallback_runs`` = n_l (B_L terminates) the smaller of n_v(U^) and n_l, and n_l for inadmissible
    U^, so E[n_v] = 1 + sum_{n < n_l} P(D(U^) < max(a, t_n)) (:func:`verification_run_threshold`);
    without a fallback (None) inadmissible outcomes repeat the certificate and ``runs`` is
    E[n_v | D(U^) >= a] = 1 + sum_n P(a <= D(U^) < t_n)/P(D(U^) >= a). Closed-form sums of normal
    probabilities, P(D(U^) < t) = Phi((t - D(U)) g / sigma), with P = 1 where t_n = inf (n runs suffice
    nowhere). The planned U must be admissible (U <= E1lb - 2 dE); without a fallback, ValueError where
    an admissible outcome needs more than 64 runs (``verification_runs`` at D = a).
    """
    u = _finite(u_upper, "u_upper")
    s = _positive(sigma, "sigma")
    e1 = _finite(e1_lower, "e1_lower")
    de = _positive(de, "de")
    g = _positive(grid, "grid")
    if u > e1 - 2.0 * de:
        raise ValueError(f"the planned U = {u!r} lies above E1lb - 2 dE = {e1 - 2.0 * de!r}")
    kw = dict(p_min=p_min, eps_id=eps_id, window=window, alpha=alpha)
    d0 = (e1 - u - de) / g
    a = de / g
    planned = verification_runs(d0, **kw)
    nl = None if fallback_runs is None else _int_at_least(fallback_runs, 1, "fallback_runs")

    def below(t: float) -> float:                       # P(D(U^) < t)
        return float(scipy.special.ndtr((t - d0) * g / s))

    p_adm = 1.0 - below(a)
    if nl is not None:
        runs = 1.0 + sum(below(max(a, verification_run_threshold(n, **kw))) for n in range(1, nl))
    else:
        edge = verification_runs(a, **kw)               # the most runs an admissible outcome needs
        tail = sum(below(verification_run_threshold(n, **kw)) - below(a) for n in range(1, edge))
        runs = 1.0 + max(0.0, tail) / p_adm
    return ExpectedRuns(runs=float(runs), admissible=float(p_adm), planned=int(planned), fallback=nl)


def failure_budget(kind: str, *, p_min: float = P_MIN, eps_fail: float = EPS_FAIL, f: float = 1.0) -> float:
    """F40's per-run logical-failure budget p_run for an output failure ``eps_fail``.

    "unverified" (a post-selected run whose state is not re-tested: n_v = 1 with rep, aa, bbht or
    fp): eps_fail p_min / f; "retested" (every other run): eps_fail / (2 f). f in (0, 1] is the
    probability that the threshold accepts a failed run (f = 1 in the costing).
    """
    if kind not in FAILURE_KINDS:
        raise ValueError(f"kind must be one of {FAILURE_KINDS}, got {kind!r}")
    p_min = _probability(p_min, "p_min")
    eps_fail = _probability(eps_fail, "eps_fail")
    f = _probability(f, "f", open_high=False)
    return eps_fail * p_min / f if kind == "unverified" else eps_fail / (2.0 * f)


def identification_bound(p0: float, tau: float, k0: float, n_runs: int = 1) -> float:
    """F40: eps_id <= [(1 - p0)/p0] (tau / K_0)^{n_runs} when every populated excited level is >= E1lb."""
    p0 = _probability(p0, "p0", open_high=False)
    tau = _probability(tau, "tau", open_low=False, open_high=False)
    k0 = _probability(k0, "k0", open_high=False)
    n = _int_at_least(n_runs, 1, "n_runs")
    return (1.0 - p0) / p0 * (tau / k0) ** n


# ------------------------------------------------------------------ model spectrum
def _as_frozen(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False)
class ModelSpectrum:
    """Every eigenpair of H' in the (n_alpha, n_beta) sector with <S^2> and irreps (module docstring).

    Attributes:
        energies: (n,) eigenvalues of H' (ecore included), ascending.
        vectors: (dim, n) eigenvectors as raveled PySCF CI arrays (columns).
        s2: (n,) <S^2> of each eigenvector.
        irreps: (n,) dominant irrep id of each eigenvector.
        shape: the CI array shape (n_alpha strings, n_beta strings); dim = shape[0] shape[1].
        lam, c: lambda of the selected encoding and the walk offset (A2).
        sin_theta0: sin(theta_0), theta_0 = arccos((E_0' - c)/lambda).
        e0: E_0' = energies[0].
        e1_singlet: lowest excited singlet of the ground state's irrep (nan if none).
        e1_any_spin: lowest excited level of that irrep, any S (nan if none).
        e1_any_irrep: lowest excited singlet of any irrep (nan if none).

    The arrays are read-only copies.
    """

    energies: np.ndarray
    vectors: np.ndarray
    s2: np.ndarray
    irreps: np.ndarray
    shape: tuple
    lam: float
    c: float
    sin_theta0: float
    e0: float
    e1_singlet: float
    e1_any_spin: float
    e1_any_irrep: float

    def __post_init__(self) -> None:
        e = np.asarray(self.energies, dtype=float)
        if e.ndim != 1 or e.size < 1 or not np.all(np.isfinite(e)):
            raise ValueError("energies must be a non-empty finite 1-D array")
        if np.any(np.diff(e) < -1e-12 * max(1.0, float(np.max(np.abs(e))))):
            raise ValueError("energies must be ascending")
        shape = tuple(int(s) for s in self.shape)
        if len(shape) != 2 or min(shape) < 1:
            raise ValueError(f"shape must be (n_alpha strings, n_beta strings), got {self.shape!r}")
        v = np.asarray(self.vectors, dtype=float)
        if v.shape != (shape[0] * shape[1], e.size) or not np.all(np.isfinite(v)):
            raise ValueError(f"vectors must be a finite ({shape[0] * shape[1]}, {e.size}) array, got {v.shape}")
        s2 = np.asarray(self.s2, dtype=float)
        if s2.shape != e.shape or not np.all(np.isfinite(s2)) or np.any(s2 < -1e-8):
            raise ValueError("s2 must be a finite, non-negative array with one entry per eigenvalue")
        ir = np.asarray(self.irreps)
        if ir.shape != e.shape or ir.dtype.kind not in "iu" or np.any((ir < 0) | (ir > 7)):
            raise ValueError("irreps must be integer ids 0..7, one per eigenvalue")
        lam = _positive(self.lam, "lam")
        c = _finite(self.c, "c")
        st = float(self.sin_theta0)
        if not (math.isfinite(st) and 0.0 <= st <= 1.0):
            raise ValueError(f"sin_theta0 must lie in [0, 1], got {self.sin_theta0!r}")
        e0 = _finite(self.e0, "e0")
        if abs(e0 - e[0]) > 1e-12 * max(1.0, abs(e0)):
            raise ValueError(f"e0 = {e0!r} is not the lowest eigenvalue {e[0]!r}")
        for name in ("e1_singlet", "e1_any_spin", "e1_any_irrep"):
            val = float(getattr(self, name))
            if not (math.isnan(val) or (math.isfinite(val) and val >= e0 - 1e-12 * max(1.0, abs(e0)))):
                raise ValueError(f"{name} must be nan or a finite energy >= e0, got {val!r}")
            object.__setattr__(self, name, val)
        object.__setattr__(self, "energies", _as_frozen(e))
        object.__setattr__(self, "vectors", _as_frozen(v))
        object.__setattr__(self, "s2", _as_frozen(s2))
        object.__setattr__(self, "irreps", _as_frozen(ir.astype(np.int64)))
        object.__setattr__(self, "shape", shape)
        for name, val in (("lam", lam), ("c", c), ("sin_theta0", st), ("e0", e0)):
            object.__setattr__(self, name, val)

    @property
    def theta0(self) -> float:
        """theta_0 = arccos((E_0' - c)/lambda) (radians)."""
        return float(math.acos(max(-1.0, min(1.0, (self.e0 - self.c) / self.lam))))

    def grid(self, n_points: int) -> float:
        """g = 2 pi lambda sin(theta_0) / M (Hartree): 2 dE sin(theta_0) at M = 2 n_iter (F39)."""
        return 2.0 * math.pi * self.lam * self.sin_theta0 / _int_at_least(n_points, 2, "n_points")

    def weights(self, ci: np.ndarray) -> np.ndarray:
        """p_j = |<psi_j|Psi>|^2 / <Psi|Psi> for a CI array (or its ravel) of this sector."""
        c = np.asarray(ci)
        dim = self.shape[0] * self.shape[1]
        if c.shape not in (self.shape, (dim,)):
            raise ValueError(f"ci has shape {c.shape}, expected {self.shape} or ({dim},)")
        c = c.ravel()
        nrm = float(np.vdot(c, c).real)
        if not (math.isfinite(nrm) and nrm > 0.0):
            raise ValueError("ci must be finite and nonzero")
        amp = self.vectors.T @ c
        return (amp.real ** 2 + amp.imag ** 2) / nrm if np.iscomplexobj(amp) else amp ** 2 / nrm


def _sector_matrix(active: ActiveSpace) -> np.ndarray:
    """Dense (n_alpha, n_beta) sector Hamiltonian with ecore, in PySCF's determinant order (raveled CI)."""
    from pyscf.fci import cistring, direct_spin1

    na, nb = active.nelec
    dim = cistring.num_strings(active.norb, na) * cistring.num_strings(active.norb, nb)
    addr, h0 = direct_spin1.pspace(active.h1, active.eri, active.norb, active.nelec, np=dim)
    if not np.array_equal(np.asarray(addr), np.arange(dim)):
        raise RuntimeError("pspace did not return the full determinant space in address order")
    h = 0.5 * (h0 + h0.T)
    h[np.diag_indices(dim)] += active.ecore
    return h


def _contract_ss(vec: np.ndarray, norb: int, nelec: tuple[int, int], shape: tuple) -> np.ndarray:
    from pyscf.fci import spin_op

    return np.asarray(spin_op.contract_ss(vec.reshape(shape), norb, nelec)).ravel()


def _resolve_degenerate(e: np.ndarray, v: np.ndarray, labels: np.ndarray, norb: int, nelec: tuple[int, int],
                        shape: tuple) -> np.ndarray:
    """Rotate eigenvectors within degenerate levels to diagonalize the irrep label, then S^2."""
    v = v.copy()
    i, n = 0, e.size
    while i < n:
        j = i + 1
        while j < n and e[j] - e[j - 1] <= _DEGENERACY_TOL:
            j += 1
        if j - i > 1:
            block = v[:, i:j]
            lab = block.T @ (labels[:, None] * block)
            w, u = np.linalg.eigh(0.5 * (lab + lab.T))
            block = block @ u
            a = 0
            while a < w.size:
                b = a + 1
                while b < w.size and w[b] - w[b - 1] <= 1e-6:
                    b += 1
                if b - a > 1:
                    sub = block[:, a:b]
                    ss = sub.T @ np.column_stack([_contract_ss(sub[:, t], norb, nelec, shape)
                                                  for t in range(b - a)])
                    _, u2 = np.linalg.eigh(0.5 * (ss + ss.T))
                    block[:, a:b] = sub @ u2
                a = b
            v[:, i:j] = block
        i = j
    return v


def _lowest(e: np.ndarray, mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    idx = idx[idx >= 1]
    return float(e[idx[0]]) if idx.size else math.nan


def model_spectrum(active: ActiveSpace, qpe: QPECost, irreps: np.ndarray, *, max_dim: int = 5000) -> ModelSpectrum:
    """The model spectrum of the selected encoding's H' in the sector of ``active`` (module docstring).

    Args:
        active: the active space of H (its sector and orbitals); H' is ``qpe.encoding.model(active)``.
        qpe: the selected QPE (F9); its encoding must carry ``eri_model``; lambda = encoding.lam.
        irreps: orbital irrep ids of ``active`` (``ft.symmetry.orbital_irreps``).
        max_dim: largest sector dimension diagonalized densely (4900 at N = 8); larger sectors
            raise ValueError (no model spectrum, e.g. N = 10).

    ValueError also if lambda does not bound |E_j - c| on the sector.
    """
    from pyscf.fci import cistring, spin_op

    if not isinstance(qpe, QPECost):
        raise ValueError(f"qpe must be a QPECost, got {type(qpe).__name__}")
    model = qpe.encoding.model(active)
    norb, nelec = model.norb, tuple(model.nelec)
    shape = (cistring.num_strings(norb, nelec[0]), cistring.num_strings(norb, nelec[1]))
    dim = shape[0] * shape[1]
    if dim > _int_at_least(max_dim, 1, "max_dim"):
        raise ValueError(f"the sector has {dim} determinants > max_dim = {max_dim}: no model spectrum")
    det_ir = determinant_irreps(norb, nelec, irreps).ravel()
    e, v = np.linalg.eigh(_sector_matrix(model))
    v = _resolve_degenerate(e, v, det_ir.astype(float) + 1.0, norb, nelec, shape)
    weights = np.zeros((8, e.size))
    for g in np.unique(det_ir):
        weights[g] = np.sum(v[det_ir == g] ** 2, axis=0)
    irr = np.argmax(weights, axis=0).astype(np.int64)
    s2 = np.array([spin_op.spin_square0(v[:, j].reshape(shape), norb, nelec)[0] for j in range(e.size)])
    s2 = np.where(np.abs(s2) < 1e-12, 0.0, s2)
    lam = float(qpe.encoding.lam)
    c = walk_offset(model)
    x = _walk_x(e, lam, c)
    singlet = s2 < _SINGLET_TOL
    same = irr == irr[0]
    return ModelSpectrum(energies=e, vectors=v, s2=s2, irreps=irr, shape=shape, lam=lam, c=c,
                         sin_theta0=float(math.sqrt(max(0.0, 1.0 - x[0] ** 2))), e0=float(e[0]),
                         e1_singlet=_lowest(e, singlet & same), e1_any_spin=_lowest(e, same),
                         e1_any_irrep=_lowest(e, singlet))


# ------------------------------------------------------------------ rule B
@dataclass(frozen=True)
class Bracket:
    """The threshold of rule B and its form (F40).

    Attributes:
        theta: the threshold (Hartree): U + dE (B_U) or E1lb - D_L g (B_L).
        form: "B_U" or "B_L".
        lower: U + dE (nan without U), the lowest threshold whose termination U certifies.
        upper: E1lb - D_L g, the B_L threshold (E1lb - dE in the window model).
        certified: termination certified: for B_U the window-model bracket condition
            U <= E1lb - 2 dE (for U = E_k + dE, M3's rule-E certificate); for B_L a certified U
            with U + dE <= theta.
        distance: D = (E1lb - theta)/g in grid units (g = dE in the window model).
        n_runs: n_v of F40 (1 in the window model).
        flags: "termination_not_certified" (B_L without such a U), "no_certificate" (no U),
            "u_outside_bracket" (a U whose B_U threshold is not admissible: U + dE >= E1lb, or more
            than 64 runs; in the window model U > E1lb - 2 dE), "window_model".
    """

    theta: float
    form: str
    lower: float
    upper: float
    certified: bool
    distance: float
    n_runs: int
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.form not in BRACKET_FORMS:
            raise ValueError(f"form must be one of {BRACKET_FORMS}, got {self.form!r}")
        theta = _finite(self.theta, "theta")
        upper = _finite(self.upper, "upper")
        lower = float(self.lower)
        if not (math.isnan(lower) or math.isfinite(lower)):
            raise ValueError(f"lower must be finite or nan, got {self.lower!r}")
        if self.form == "B_U" and math.isnan(lower):
            raise ValueError("a B_U bracket needs lower = U + dE")
        d = _positive(self.distance, "distance")
        n = _int_at_least(self.n_runs, 1, "n_runs")
        if not isinstance(self.certified, (bool, np.bool_)):
            raise ValueError(f"certified must be a bool, got {self.certified!r}")
        flags = tuple(self.flags)
        if not all(isinstance(fl, str) for fl in flags):
            raise ValueError("flags must be strings")
        for name, val in (("theta", theta), ("upper", upper), ("lower", lower), ("distance", d)):
            object.__setattr__(self, name, val)
        object.__setattr__(self, "n_runs", n)
        object.__setattr__(self, "certified", bool(self.certified))
        object.__setattr__(self, "flags", flags)


def acceptance_bracket(u_upper: float | None, e1_lower: float, de: float, *, grid: float | None,
                       d_l: int = 5, p_min: float = P_MIN, eps_id: float = EPS_ID) -> Bracket:
    """Rule B's threshold: the form with the smaller n_v, B_U on ties (F40, module docstring).

    Args:
        u_upper: a certified U >= E_0' (F42), or None.
        e1_lower: E1lb, a lower bound on every populated excited level (F41).
        de: the QPE error dE (Hartree).
        grid: g = 2 pi lambda sin(theta_0)/M of the hand-off QPE (2 dE sin(theta_0) at
            M = 2 n_iter): n_v from the sine-window tails; None: M3's window model (outcomes
            within dE of their level, no tails, n_v = 1, B_L at E1lb - dE).
        d_l: B_L's top margin D_L in grid units.
        p_min, eps_id: the design floor and target of :func:`verification_runs`.
    """
    e1 = _finite(e1_lower, "e1_lower")
    de = _positive(de, "de")
    d_l = _positive(d_l, "d_l")
    u = None if u_upper is None else _finite(u_upper, "u_upper")
    flags: list[str] = [] if u is not None else ["no_certificate"]
    lower = math.nan if u is None else u + de
    if grid is None:
        flags.append("window_model")
        upper = e1 - de
        if u is not None and lower <= upper:
            return Bracket(theta=lower, form="B_U", lower=lower, upper=upper, certified=True,
                           distance=(e1 - lower) / de, n_runs=1, flags=tuple(flags))
        if u is not None:
            flags.append("u_outside_bracket")
        return Bracket(theta=upper, form="B_L", lower=lower, upper=upper, certified=False, distance=1.0,
                       n_runs=1, flags=tuple(flags + ["termination_not_certified"]))
    g = _positive(grid, "grid")
    upper = e1 - d_l * g
    n_l = verification_runs(d_l, p_min=p_min, eps_id=eps_id)
    if u is not None and lower < e1:
        d_u = (e1 - lower) / g
        try:
            n_u = verification_runs(d_u, p_min=p_min, eps_id=eps_id)
        except ValueError:          # more than 64 runs: B_U is not admissible
            n_u = None
        if n_u is not None and n_u <= n_l:
            return Bracket(theta=lower, form="B_U", lower=lower, upper=upper, certified=u <= e1 - 2.0 * de,
                           distance=d_u, n_runs=n_u, flags=tuple(flags))
        if n_u is None:
            flags.append("u_outside_bracket")
    elif u is not None:
        flags.append("u_outside_bracket")
    certified = u is not None and lower <= upper
    if not certified:
        flags.append("termination_not_certified")
    return Bracket(theta=upper, form="B_L", lower=lower, upper=upper, certified=certified, distance=d_l,
                   n_runs=n_l, flags=tuple(flags))


@dataclass(frozen=True)
class AcceptanceStats:
    """Per-attempt statistics of rule B on one state (F40, F41).

    Attributes:
        accept: P_acc = sum_j p_j K_j^n, the probability that an attempt passes all n runs.
        ground: p_0 K_0^n, passing with the state in the walk eigenvectors of psi_0.
        identification_error: eps_id of the accepted state: the kernel term
            sum_{j>=1} p_j K_j^n / P_acc plus the contaminant term q_c/(p_0 K_0^n) plus ``logical``
            (nan if P_acc = 0).
        energy_error: P(|E(m) - E_0'| > dE) of the returned (last) outcome from the kernels, plus the
            same contaminant and logical terms (nan if P_acc = 0).
        k0: K_0(A_theta), the single-run ground-state mass below theta.
        logical: the logical-failure term, f p_run/(P_acc + f p_run) for one run and f p_run for
            n >= 2, f = 1 - theta_0/pi.
    """

    accept: float
    ground: float
    identification_error: float
    energy_error: float
    k0: float
    logical: float

    def __post_init__(self) -> None:
        tol = 1e-9
        acc = float(self.accept)
        if not (math.isfinite(acc) and -tol <= acc <= 1.0 + tol):
            raise ValueError(f"accept must lie in [0, 1], got {self.accept!r}")
        gr = float(self.ground)
        if not (math.isfinite(gr) and -tol <= gr <= acc + tol):
            raise ValueError(f"ground must lie in [0, accept], got {self.ground!r}")
        k0 = float(self.k0)
        if not (math.isfinite(k0) and -tol <= k0 <= 1.0 + tol):
            raise ValueError(f"k0 must lie in [0, 1], got {self.k0!r}")
        lg = float(self.logical)
        if not (math.isfinite(lg) and 0.0 <= lg <= 1.0):
            raise ValueError(f"logical must lie in [0, 1], got {self.logical!r}")
        for name in ("identification_error", "energy_error"):
            val = float(getattr(self, name))
            if math.isnan(val):
                if acc > 0.0:
                    raise ValueError(f"{name} is nan although accept > 0")
            elif not val >= -tol:
                raise ValueError(f"{name} must be >= 0, got {val!r}")
            object.__setattr__(self, name, val)
        for name, val in (("accept", acc), ("ground", gr), ("k0", k0), ("logical", lg)):
            object.__setattr__(self, name, val)


def _level_masses(spec: ModelSpectrum, idx: np.ndarray, theta: float, n_points: int, window: str, alpha: float,
                  de: float) -> tuple[np.ndarray, np.ndarray]:
    """(K_j(A_theta), the part of it outside |E(m) - E_0'| <= dE) for the levels idx (others nan), cached per spectrum."""
    key = (int(n_points), window, float(alpha) if window == "kaiser" else None, float(theta), float(de))
    store = _LEVEL_MASSES.setdefault(spec, {})
    entry = store.pop(key, None)
    if entry is None:
        entry = (np.full(spec.energies.size, np.nan), np.full(spec.energies.size, np.nan))
        while len(store) >= _MASS_KEYS:
            store.pop(next(iter(store)))
    store[key] = entry                  # most recently used last
    k_acc, k_far = entry
    todo = idx[np.isnan(k_acc[idx])]
    if todo.size:
        a = phase_window(n_points, window, alpha)
        m = a.size
        grid_e = spec.c + spec.lam * np.cos(2.0 * np.pi * np.arange(m) / m)
        below = grid_e < theta
        near = np.abs(grid_e - spec.e0) <= de
        thetas = np.arccos(_walk_x(spec.energies[todo], spec.lam, spec.c))
        rows = max(1, _BATCH // m)
        for i0 in range(0, todo.size, rows):
            k = _branch_kernels(thetas[i0:i0 + rows], a)
            k_acc[todo[i0:i0 + rows]] = k[:, below].sum(axis=1)
            k_far[todo[i0:i0 + rows]] = k[:, below & ~near].sum(axis=1)
    return k_acc, k_far


def acceptance_stats(spec: ModelSpectrum, weights: np.ndarray, theta: float, *, n_points: int,
                     window: str = "sine", alpha: float = 3.0, de: float = 1e-3, n_runs: int = 1,
                     contaminant: float = 0.0, p_run: float = 0.0) -> AcceptanceStats:
    """F40 exactly on the model spectrum: the kernels below ``theta`` to the power ``n_runs``.

    The ground state is the eigenvector of index 0 (E_0'); every other eigenvector counts as an
    identification error.

    Args:
        spec: the model spectrum of H'.
        weights: p_j on ``spec``'s eigenvectors (``spec.weights(ci)``; sum <= 1); eigenstates with
            p_j <= 1e-14 are left out.
        theta: the threshold (accept E(m) < theta).
        n_points: M, the control-register length (``qpe_points``).
        window, alpha: the register window.
        de: dE of the energy-error window |E(m) - E_0'| <= dE.
        n_runs: n_v (the first run and n_v - 1 verification runs).
        contaminant: F41's q_c, weight on levels the kernel term does not represent (0 with the
            full-sector weights of ``spec.weights``, which hold every spin and irrep).
        p_run: the per-run logical-failure probability (``failure_budget``).
    """
    if not isinstance(spec, ModelSpectrum):
        raise ValueError(f"spec must be a ModelSpectrum, got {type(spec).__name__}")
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape != spec.energies.shape or not np.all(np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError(f"weights must be {spec.energies.size} finite non-negative numbers")
    if float(w.sum()) > 1.0 + 1e-9:
        raise ValueError(f"weights must sum to <= 1, got {float(w.sum())!r}")
    theta = _finite(theta, "theta")
    de = _positive(de, "de")
    n = _int_at_least(n_runs, 1, "n_runs")
    q_c = float(contaminant)
    if not (math.isfinite(q_c) and 0.0 <= q_c <= 1.0):
        raise ValueError(f"contaminant must lie in [0, 1], got {contaminant!r}")
    p_run = _probability(p_run, "p_run", open_low=False, open_high=False)
    use = w > _WMIN
    use[0] = True
    idx = np.flatnonzero(use)
    k_acc, k_far = _level_masses(spec, idx, theta, n_points, window, alpha, de)
    k_acc = np.where(use, k_acc, 0.0)
    k_far = np.where(use, k_far, 0.0)   # mass below theta and outside |E(m) - E_0'| <= dE
    kn = k_acc ** n
    accept = float(w @ kn)
    ground = float(w[0] * kn[0])
    k0 = float(k_acc[0])
    f = 1.0 - spec.theta0 / math.pi
    logical = (f * p_run / (accept + f * p_run) if accept + f * p_run > 0.0 else 0.0) if n == 1 else f * p_run
    if accept > 0.0:
        cont = 0.0 if q_c == 0.0 else (q_c / ground if ground > 0.0 else math.inf)
        kernel = float(w[1:] @ kn[1:]) / accept
        energy = float(w @ (k_acc ** (n - 1) * k_far)) / accept
        ident, energy = kernel + cont + logical, energy + cont + logical
    else:
        ident = energy = math.nan
    return AcceptanceStats(accept=accept, ground=ground, identification_error=ident, energy_error=energy,
                           k0=k0, logical=logical)

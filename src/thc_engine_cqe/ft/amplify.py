"""QPE-side schemes of the hand-off: amplitude amplification, step filters, two-stage QPE, scheme costs (F43, F44).

Tags and formulas F1-F19 as in :mod:`thc_engine_cqe.ft.primitives` (docs/m3-cost-model.md); F39,
F40 in :mod:`thc_engine_cqe.ft.acceptance`, F41, F42 in :mod:`thc_engine_cqe.ft.certify`. Costs are
Toffolis or C_QPE units as stated, energies Hartree, phases radians. Scheme costs take plain
numbers (overlaps, Toffoli counts), so they apply to any trajectory or closed-loop record.

Amplitude amplification (F43) [derived; cited as noted]
-------------------------------------------------------
A call of A = (coherent QPE with the comparator flag E(m) < theta) Prep(Psi) costs C(Psi) + C_QPE;
the comparator (about 2t Toffolis) and the reflections (about Q) are neglected [estimate]. With the
success probability a (a = p in M3's window model) and phi = asin(sqrt(a)):

* known a [cited: Brassard, Hoyer, Mosca, Tapp, Contemp. Math. 305, 53 (2002)]: repeat until
  success with m Grover iterations, N*(a) = min_m (2m + 1)/sin^2((2m + 1) phi) expected calls
  (:func:`grover_calls`). With x = (2m + 1) phi, x/sin^2 x is least at tan x = 2x, x = 1.165561,
  so N* sqrt(a) -> 1.380050 as a -> 0 [derived]; N* < 1/a iff a < 0.32. N* is not monotone: on
  a in (0.25, 0.32) the m = 1 branch is past its maximum ((2m + 1) phi > pi/2), so
  N*(0.25) = 3 < N*(0.32) = 3.125. An oracle (it needs a to choose m).
* exponential search with unknown a [cited: Boyer, Brassard, Hoyer, Tapp, Fortschr. Phys. 46, 493
  (1998)] (:func:`exponential_search_calls`): stage r draws j uniformly from [0, M_r),
  M_r = ceil(growth^r), growth = 6/5, and makes 2j + 1 calls, E[2j + 1] = M_r; the stage succeeds
  with P(M) = (1/M) sum_{j<M} sin^2((2j + 1) phi) = 1/2 - sin(4 M phi)/(4 M sin 2 phi) [derived],
  so E[N] = sum_r M_r prod_{s<r} (1 - P(M_s)); E[N] sqrt(a) = 1.8-2.9 for a <= 0.1 [build check: 1.796
  at a = 0.1 to 2.899 at a = 5e-6].
* fixed point with a certified a_lb [cited: Yoder, Low, Chuang, PRL 113, 210501 (2014)]
  (:func:`fixed_point_calls`, :func:`fixed_point_success`): L = 2l + 1 calls with
  P_L(a) = 1 - delta^2 T_L(T_{1/L}(1/delta) sqrt(1 - a))^2 >= 1 - delta^2 for every a >= a_lb iff
  L >= acosh(1/delta)/atanh(sqrt(a_lb)).

An AA sequence is one coherent run of N (C(Psi) + C_QPE) Toffolis: it does not parallelize (F19)
and its failure budget is that of the whole run (F40's output rule).

Step filter (F43) [cited: Lin and Tong, Quantum 4, 372 (2020); GQSP as F8]
-------------------------------------------------------------------------
An even trigonometric polynomial g of degree d in the walk phase theta (a polynomial of degree d
in x = cos theta = (E - c)/lambda) with |g| <= 1, |1 - g| <= eps_f on [theta_pass, pi] (energies
<= U) and |g| <= eps_f on [0, theta_stop] (energies >= E1lb), theta_pass = arccos((U - c)/lambda),
theta_stop = arccos((E1lb - c)/lambda), dtheta = theta_pass - theta_stop. The construction
(:func:`step_filter_degree`, :func:`step_filter_polynomial`) is a Gaussian-smoothed step in theta
centred in the transition band, Fourier-truncated at degree d, scaled to max |g| <= 1 on the grid,
its width optimized over 28 values in [0.05, 0.5] dtheta, with the band errors evaluated on a
periodic grid of at least 2^14 and >= 512/dtheta points and at the two band edges exactly, and
bisection on d. The edges matter: the error of the smoothed step is largest there, and on the grid
alone it is missed by up to 11% at eps_f = 1e-3 (by < 1% at 1e-2) [build check: a 16x denser
grid]; between grid points |g| exceeds 1 by <= 1.1e-6, which a rescaling by as much absorbs. The
degree is d = ceil(c_f(eps_f)/dtheta) from the table :data:`FILTER_TABLE` (:func:`filter_degree`)
[design]: c_f(eps_f) = 1.01 x max over theta_pass in {pi/2 + 0.17, pi/2 + 0.6} and dtheta_ref in
{0.1, 0.05} of the constructed d dtheta_ref, on the grid eps_f = 10^(-j/4), j = 8..40, with eps_f
rounded down to the grid; for dtheta > 0.1 or eps_f < 1e-10 the construction itself is used.
[build check: c_f = 9.999 at 1e-2, 18.48 at 1e-3, 28.58 at 1e-4, 44.84 at 1e-6, 67.06 at 1e-8; the
constructed d dtheta is 8.92 at dtheta = 0.02 and 9.85 at 0.05 for eps_f = 1e-2, 18.2 at both for
1e-3, so the table bounds the construction from above, by <= 13% at 1e-2 (tests/test_ft_amplify.py).]
Cost (F8's GQSP count, 2d controlled walk calls):

    C_filt = 2d C_W + 3(2d + 1)(beta - 2)  on Q_walk + 1 logical qubits,
    C_filt / C_QPE ~ (4 c_f / pi) sin(theta_bar) dE / (E1lb - U)      (:func:`filter_cost`),

the closed form for the walk part 2d C_W (to 0.05% on stretched N2), to which the GQSP rotations add
3(beta - 2)/C_W, 5.1-5.2% at C_W = 768-777 [build check].

The leakage is eps_f^2 in probability per unit excited weight.

Two-stage QPE (F43) [design]
---------------------------
A coarse sine-window QPE of M_c points (C_c = (M_c/2) C_W, assumption A3 of F39) repeated until its
outcome lies below a threshold between U and E1lb, then one full QPE. M_c is the smallest length at
which every level at or above E1lb passes the threshold with probability <= eps_p and every level
at or below U with probability >= 1 - eps_p (:func:`coarse_qpe_points`), the threshold the best
cut between grid phases. Levels "at or above E1lb" are all the excited levels the bound covers:
with only the two levels exactly at E1lb and U the condition is a lattice lottery, since the
sine-window lattice law of F34 vanishes at half-integer distances (at theta_U = pi/2 + 0.17,
dtheta = 0.1, eps_p = 1e-4 it holds at M = 182, at no M from 183 to 246 and at a third of the
lengths up to 697, while every level beyond the edges needs 704 [build check]). With every level
beyond each edge the worst level within one grid spacing beyond it decides (the one-sided tail
obeys T(D + 1) <= T(D)), and the condition needs D_1 + D_U = dtheta M/(2 pi) + 1 grid units split
into two tails >= D_p, tau_W(D_p) = eps_p (D_p = 6.0 at 1e-4): it holds for every M with
dtheta M/(2 pi) >= 2 D_p and for some M down to 2 D_p - 1 (where the cut quantization is
favourable), so

    C_c / C_QPE ~ 4 sin(theta_0) D_p dE / (E1lb - U)                  (:func:`coarse_qpe_toffolis`)

within -8%/+0% [build check: 0.92 on stretched N2]. The search scans M upward from
2 pi (2 D_p - 1.5)/dtheta; up to M = 4096 it evaluates the finite-register sine law (FFT kernels,
32 sub-offsets per grid spacing), above it the continuum lattice law of F34 (closed-form tails with
trigamma functions, as ``ft.acceptance``), which the finite-register tails approach (8e-4 relative
at D = 5, M = 4096; tests/test_ft_acceptance.py).

E1lb-referenced forms (F43) [derived; B_L's status]
--------------------------------------------------
In lt, lt_bbht and 2s, U only places the pass edge (termination); soundness comes from the stop edge
at E1lb. lt_bbht_L: the step filter with pass edge E1lb - W_L and stop edge E1lb (eps_f = 1e-2);
2s_L: the coarse QPE between E1lb - W_L and E1lb (eps_p = 1e-4); W_L = 100 mHa (:data:`W_L`). Both
terminate iff E_0' <= E1lb - W_L (checked in the model with the exact E_0'; rows that fail are
"not_terminating" and excluded) and are certified only where a certified U <= E1lb - W_L exists,
else "termination_not_certified".

Hand-off cost per scheme (F43) [derived]
---------------------------------------
At iterate k with c = C(Psi_k), C_loop = C_loop(k), V = (n_v - 1) C_QPE (F40's verification runs;
the filtered schemes need none, one final QPE is sound at p_min for D >= 1.3):

    rep        C_loop + (c + C_QPE)/p + V             realizable (B_U; B_L flagged)
    aa         C_loop + N*(p)(c + C_QPE) + V          oracle (p known)
    bbht       C_loop + E_BBHT(p)(c + C_QPE) + V      realizable
    fp         C_loop + L(p_lb, delta)(c + C_QPE) + V realizable with a certified p_lb > 0 (F15)
    lt         C_loop + N*(p)(c + C_filt(U)) + C_QPE  oracle; needs a certified U
    lt_bbht    C_loop + E_BBHT(p)(c + C_filt(U)) + C_QPE   realizable; certified U
    2s         C_loop + (c + C_c(U))/p + C_QPE        realizable; certified U
    lt_bbht_L  C_loop + E_BBHT(p)(c + C_filt(W_L)) + C_QPE realizable (termination_not_certified)
    2s_L       C_loop + (c + C_c(W_L))/p + C_QPE      realizable (termination_not_certified)

(:func:`scheme_costs`, :func:`scheme_table`). Without a certified U, lt, lt_bbht and 2s are reported
with the oracle U = E_0' + 2 dE as lower bounds under the keys "<scheme>@oracle_U" (flag "oracle_U"),
never in the best realizable scheme s* = argmin over :data:`REALIZABLE` (:func:`best_realizable`).
With n_v = 1 the "rep" column is M3's rule-E expected cost T_E (``ft.handoff.HandoffRow.cost_expected``).

Per state, :func:`state_scheme_costs` assembles the table from the hand-off's inputs (p on psi_0' of
H', rule B's option, the filter costs of a certified U, of the pass edge E1lb - W_L and of the oracle
U) and attaches the flags of each scheme: rep, aa, bbht and fp carry rule B's bracket flags (B_L
without U: "termination_not_certified", "no_certificate"); lt_bbht_L and 2s_L
"termination_not_certified" unless a certified U <= E1lb - W_L exists; lt, lt_bbht and 2s none where
their final QPE is B_U with a certified U at D_U >= 1.3, "termination_not_certified" where it is B_L.
The flags of s* qualify every verdict of record.

Sensitivity (never in s*): exponential search makes E_BBHT(p) > 1/p calls above p = 0.30 [build
check: E_BBHT(p) p = 1.0027 at 0.30, 1.5785 at 0.5, 1.1565 at 0.918; > 1 for 0.29892 < p < 1, = 1 at
p = 1], its schedule overshooting where a single call often succeeds, so there the step filter followed by
plain repetition, the m = 0 branch of lt, which needs no knowledge of p, is cheaper than lt_bbht and
lt_bbht_L:

    lt_rep     C_loop + (c + C_filt(U))/p + C_QPE       certified U
    lt_rep_L   C_loop + (c + C_filt(W_L))/p + C_QPE     where the L forms terminate

(:data:`SENSITIVITY_SCHEMES`); with C_filt < C_c they also undercut 2s and 2s_L at every p [build
check: C_filt/C_c = 0.59-0.64 on the pass edges of the hand-off benchmark]. They lie outside the
scheme set of s* (the verdict of record); :func:`best_extended` and :func:`extended_verdict` report the
optimum and the verdict over the set extended by them.

What a better state is worth (F44) [derived]
-------------------------------------------
With T_s(k) = C_loop(k) + H_s(p_k, c_k), iteration k pays under s iff
C_iter(k) < mu_k^s = H_s(p_k, c_k) - H_s(p_{k+1}, c_{k+1}) (:func:`scheme_saving`, the hook of
``ft.scaling.cost_ratio``), i.e. iff T_s(k + 1) < T_s(k) (:func:`iteration_verdict`). At the best
realizable scheme the minimizing scheme may change with the state: the iteration pays iff
min_s T_s(k + 1) < min_s T_s(k) over the realizable schemes with a cost, i.e. iff C_iter(k) <
mu*_k = min_s H_s(p_k, c_k) - min_s H_s(p_{k+1}, c_{k+1}) (:func:`best_realizable_saving`; F40's
verification runs do not cancel here, since they enter rep, bbht and fp but not the filtered
schemes). The most a perfect state can save is H_s(p_0, 0) - H_s(1, 0)
(:func:`saving_budget`): C_QPE(1/p_0 - 1) (rep), C_QPE(N*(p_0) - 1) ~ C_QPE(1.38/sqrt(p_0) - 1) (aa),
(N*(p_0) - 1) C_filt (lt), E_BBHT in place of N* for bbht, lt_bbht and lt_bbht_L (E_BBHT(1) = 1),
C_c(1/p_0 - 1) for 2s and 2s_L; F40's verification runs add (n_v - 1) C_QPE to both states and
cancel. The break-even overlap of an iteration of cost C_it that lifts p_0 to p_1
(:func:`break_even_overlap`) is the largest p_0 with H_s(p_0, 0) - H_s(p_1, c_1) >= C_it; for
C_it >> C_QPE, p_0* ~ C_QPE/C_it (rep), (1.38 C_QPE/C_it)^2 (aa), (1.38 C_filt/C_it)^2 (lt) and
(kappa_B C_filt/C_it)^2 with kappa_B = E_BBHT sqrt(p) (lt_bbht, lt_bbht_L).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import scipy.special

from .acceptance import _branch_kernels, phase_window
from .primitives import FTParameters, gqsp_rotation_toffolis
from .qpe import BlockEncoding

__all__ = [
    "SCHEMES",
    "REALIZABLE",
    "ORACLE_U_SCHEMES",
    "W_L",
    "FILTER_TABLE",
    "grover_calls",
    "exponential_search_calls",
    "fixed_point_calls",
    "fixed_point_success",
    "step_filter_degree",
    "step_filter_polynomial",
    "build_filter_table",
    "filter_degree",
    "FilterCost",
    "filter_cost",
    "coarse_qpe_points",
    "coarse_qpe_toffolis",
    "scheme_costs",
    "SENSITIVITY_SCHEMES",
    "FINAL_D_MIN",
    "final_qpe_form",
    "state_scheme_costs",
    "SchemeRow",
    "scheme_table",
    "best_realizable",
    "best_extended",
    "extended_verdict",
    "saving_budget",
    "break_even_overlap",
    "scheme_saving",
    "best_realizable_saving",
    "iteration_verdict",
]

SCHEMES = ("rep", "aa", "bbht", "fp", "lt", "lt_bbht", "2s", "lt_bbht_L", "2s_L")
REALIZABLE = ("rep", "bbht", "fp", "lt_bbht", "2s", "lt_bbht_L", "2s_L")
#: The U-referenced schemes reported with the oracle U = E_0' + 2 dE where no certified U exists.
ORACLE_U_SCHEMES = ("lt", "lt_bbht", "2s")
# Schemes whose H_s scales with C_QPE at fixed C(Psi); the filtered ones (lt, lt_bbht, lt_bbht_L, 2s, 2s_L) carry
# C_QPE once.
_QPE_SCALED = ("rep", "aa", "bbht", "fp")
#: The step filter followed by plain repetition (module docstring): sensitivity keys, never in s*.
SENSITIVITY_SCHEMES = ("lt_rep", "lt_rep_L")
#: Hartree: the pass edge E1lb - W_L of the E1lb-referenced forms (F43).
W_L = 0.1
#: Grid units: the smallest D_U at which the filtered schemes' final QPE is B_U with one run (F40, F43).
FINAL_D_MIN = 1.3
# F43 filter table: grid exponents j of eps_f = 10^(-j/4), the reference placements and widths.
_TABLE_J = tuple(range(8, 41))
_TABLE_THETA = (math.pi / 2 + 0.17, math.pi / 2 + 0.6)
_TABLE_DTHETA = (0.1, 0.05)
_TABLE_SAFETY = 1.01
_TABLE_DTHETA_MAX = 0.1
_TABLE_EPS_MIN = 10.0 ** (-_TABLE_J[-1] / 4.0)
# Gaussian widths of the step construction, as fractions of the transition width: 22 in [0.08, 0.5] and
# six narrower ones, which the smallest eps_f of the table need (the edge error of a width f dtheta is
# erfc(1/(2 sqrt(2) f))/2, 1.6e-10 at f = 0.08) and which do not win at eps_f >= 1e-8.
_FILTER_WIDTHS = np.concatenate([np.linspace(0.05, 0.075, 6), np.linspace(0.08, 0.5, 22)])
# Largest filter degree the construction searches.
_FILTER_D_MAX = 1 << 21
# Coarse QPE: largest M evaluated with the finite-register law; sub-offsets of the level positions.
_COARSE_EXACT_MAX = 4096
_COARSE_SUB = 32
_COARSE_M_MAX = 1 << 34
# Exponential search: stop when the probability of reaching the next stage is below this.
_BBHT_TAIL = 1e-16
# Overlaps above 1 - this are read as 1 (no amplification needed).
_ONE_TOL = 1e-15

#: F43: (eps_f, c_f) on eps_f = 10^(-j/4), j = 8..40; ``build_filter_table()`` rebuilds it (slow test).
FILTER_TABLE: tuple[tuple[float, float], ...] = (
    (0.01, 9.999),
    (0.005623413251903491, 13.736000000000002),
    (0.0031622776601683794, 15.655),
    (0.0017782794100389228, 16.867),
    (0.001, 18.483),
    (0.0005623413251903491, 20.9575),
    (0.00031622776601683794, 22.270500000000002),
    (0.00017782794100389227, 27.472000000000005),
    (0.0001, 28.583000000000002),
    (5.623413251903491e-05, 29.694000000000003),
    (3.1622776601683795e-05, 30.805),
    (1.778279410038923e-05, 32.118),
    (1e-05, 39.895),
    (5.623413251903491e-06, 41.107000000000006),
    (3.162277660168379e-06, 42.42),
    (1.778279410038923e-06, 43.632000000000005),
    (1e-06, 44.84400000000001),
    (5.62341325190349e-07, 46.106500000000004),
    (3.162277660168379e-07, 51.813),
    (1.7782794100389227e-07, 60.398),
    (1e-07, 61.913000000000004),
    (5.6234132519034905e-08, 63.226),
    (3.162277660168379e-08, 64.539),
    (1.7782794100389228e-08, 65.8015),
    (1e-08, 67.06400000000001),
    (5.623413251903491e-09, 68.478),
    (3.1622776601683795e-09, 69.69),
    (1.7782794100389228e-09, 70.902),
    (1e-09, 72.215),
    (5.623413251903491e-10, 73.4775),
    (3.1622776601683795e-10, 76.55799999999999),
    (1.778279410038923e-10, 80.69900000000001),
    (1e-10, 81.91100000000002),
)


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


def _cost(value, name: str) -> float:
    """A Toffoli count: >= 0, +inf allowed (an infeasible route), nan not."""
    v = float(value)
    if math.isnan(v) or v < 0.0:
        raise ValueError(f"{name} must be >= 0 (inf allowed), got {value!r}")
    return v


def _overlap(value, name: str = "p") -> float:
    v = float(value)
    if not (math.isfinite(v) and 0.0 < v <= 1.0 + 1e-12):
        raise ValueError(f"{name} must lie in (0, 1], got {value!r}")
    return min(v, 1.0)


def _unit_open(value, name: str) -> float:
    v = float(value)
    if not (math.isfinite(v) and 0.0 < v < 1.0):
        raise ValueError(f"{name} must lie in (0, 1), got {value!r}")
    return v


def _int_at_least(value, low: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < low:
        raise ValueError(f"{name} must be an integer >= {low}, got {value!r}")
    return int(value)


def _phases(theta_pass, theta_stop) -> tuple[float, float]:
    tp = _finite(theta_pass, "theta_pass")
    ts = _finite(theta_stop, "theta_stop")
    if not (0.0 <= ts < tp <= math.pi):
        raise ValueError(f"need 0 <= theta_stop < theta_pass <= pi, got {theta_stop!r}, {theta_pass!r}")
    return tp, ts


# ------------------------------------------------------------------ amplitude amplification
def grover_calls(a: float) -> tuple[float, int]:
    """(N*(a), m*): expected calls of repeat-until-success AA with m* iterations, a known (F43).

    N*(a) = min_m (2m + 1)/sin^2((2m + 1) asin sqrt(a)); a in (0, 1]; N*(1) = 1 (m* = 0). The
    minimum lies at (2m + 1) phi near 1.165561 (tan x = 2x): the integers m near it, and m = 0
    and 1, are compared, the range limited to (2m + 1) phi < pi (tested against a full scan).
    """
    a = _overlap(a, "a")
    if a >= 1.0 - _ONE_TOL:
        return 1.0, 0
    phi = math.asin(math.sqrt(a))
    centre = (1.1655611852072113 / phi - 1.0) / 2.0
    top = int(math.floor((math.pi / phi - 1.0) / 2.0))
    cands = {0, 1, max(0, int(math.floor(centre)) - 1), int(math.floor(centre)), int(math.floor(centre)) + 1,
             int(math.floor(centre)) + 2}
    best = (math.inf, 0)
    for m in sorted(cands):
        if m < 0 or (m > top and m > 0):
            continue
        s = math.sin((2 * m + 1) * phi) ** 2
        if s > 0.0:
            best = min(best, ((2 * m + 1) / s, m))
    return best


def _stage_success(n: int, phi: float) -> float:
    """P(M) = (1/M) sum_{j<M} sin^2((2j + 1) phi), by the direct sum for M <= 64, else the closed form."""
    if n <= 64:
        j = np.arange(n)
        return float(np.mean(np.sin((2 * j + 1) * phi) ** 2))
    return 0.5 - math.sin(4.0 * n * phi) / (4.0 * n * math.sin(2.0 * phi))


def exponential_search_calls(a: float, growth: float = 1.2) -> float:
    """E[N] of the exponential search with unknown a (F43): sum_r M_r prod_{s<r} (1 - P(M_s)); E(1) = 1."""
    a = _overlap(a, "a")
    lam = float(growth)
    if not (math.isfinite(lam) and 1.0 < lam <= 4.0 / 3.0 + 1e-12):
        raise ValueError(f"growth must lie in (1, 4/3] (the BBHT range), got {growth!r}")
    if a >= 1.0 - _ONE_TOL:
        return 1.0
    phi = math.asin(math.sqrt(a))
    total, alive, r = 0.0, 1.0, 0
    while alive > _BBHT_TAIL:
        m = math.ceil(lam ** r - 1e-12)
        total += alive * m
        alive *= 1.0 - _stage_success(m, phi)
        r += 1
        if r > 100000:                                    # pragma: no cover (a >= 1e-300 needs < 4000 stages)
            raise RuntimeError("exponential search did not terminate")
    return total


def fixed_point_calls(a_lb: float, delta: float = 0.01) -> int:
    """L, the smallest odd number of calls with L >= acosh(1/delta)/atanh(sqrt(a_lb)) (F43, YLC)."""
    a_lb = _overlap(a_lb, "a_lb")
    delta = _unit_open(delta, "delta")
    if a_lb >= 1.0 - _ONE_TOL:
        return 1
    n = math.acosh(1.0 / delta) / math.atanh(math.sqrt(a_lb))
    big = math.ceil(n - 1e-12)
    big = max(big, 1)
    return big if big % 2 == 1 else big + 1


def _chebyshev(n: float, x: float) -> float:
    """T_n(x) for x >= 0 (n real): cos(n acos x) on [0, 1], cosh(n acosh x) above."""
    if x <= 1.0:
        return math.cos(n * math.acos(max(-1.0, x)))
    return math.cosh(n * math.acosh(x))


def fixed_point_success(a: float, n_calls: int, delta: float) -> float:
    """P_L(a) = 1 - delta^2 T_L(T_{1/L}(1/delta) sqrt(1 - a))^2 of the Yoder-Low-Chuang sequence (F43)."""
    a = float(a)
    if not (math.isfinite(a) and 0.0 <= a <= 1.0):
        raise ValueError(f"a must lie in [0, 1], got {a!r}")
    n = _int_at_least(n_calls, 1, "n_calls")
    if n % 2 == 0:
        raise ValueError(f"n_calls must be odd (L = 2l + 1), got {n}")
    delta = _unit_open(delta, "delta")
    x = math.sqrt(1.0 - a) * _chebyshev(1.0 / n, 1.0 / delta)
    return 1.0 - delta ** 2 * _chebyshev(float(n), x) ** 2


# ------------------------------------------------------------------ step filters
class _StepConstruction:
    """The Gaussian-smoothed steps of one band edge pair, Fourier-transformed once for all degrees."""

    def __init__(self, theta_pass: float, theta_stop: float) -> None:
        width = theta_pass - theta_stop
        n = 1 << max(14, int(math.ceil(math.log2(64.0 / width))) + 3)
        t = 2.0 * np.pi * np.arange(n) / n
        folded = np.where(t <= np.pi, t, 2.0 * np.pi - t)
        self.n = n
        self.theta = folded
        self.pass_band = folded >= theta_pass
        self.stop_band = folded <= theta_stop
        self.edges = np.array([theta_pass, theta_stop])
        mid = 0.5 * (theta_pass + theta_stop)
        self.coefs = []
        for frac in _FILTER_WIDTHS:
            sigma = frac * width
            f = 0.5 * (1.0 + scipy.special.erf((folded - mid) / (math.sqrt(2.0) * sigma)))
            self.coefs.append(np.fft.rfft(f) / n)

    def _values(self, coef: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
        """g on the grid and at the two band edges (exactly, from the cosine coefficients), normalized."""
        c = coef.copy()
        c[d + 1:] = 0.0
        g = np.fft.irfft(c * self.n, n=self.n)
        a = 2.0 * c[:d + 1].real
        a[0] /= 2.0
        g_edge = np.cos(np.outer(self.edges, np.arange(d + 1))) @ a
        scale = max(1.0, float(np.max(np.abs(g))))
        return g / scale, g_edge / scale

    def error(self, d: int) -> tuple[float, int]:
        """(the best band error over the widths, the index of that width) at degree d; grid and band edges."""
        best, arg = math.inf, 0
        for i, coef in enumerate(self.coefs):
            g, g_edge = self._values(coef, d)
            err = max(float(np.max(np.abs(1.0 - g[self.pass_band]))) if self.pass_band.any() else 0.0,
                      float(np.max(np.abs(g[self.stop_band]))) if self.stop_band.any() else 0.0,
                      abs(1.0 - float(g_edge[0])), abs(float(g_edge[1])))
            if err < best:
                best, arg = err, i
        return best, arg

    def cosine_coefficients(self, d: int, width_index: int) -> np.ndarray:
        """a_0..a_d with g(theta) = sum_k a_k cos(k theta) (the normalized truncation)."""
        coef = self.coefs[width_index]
        c = coef.copy()
        c[d + 1:] = 0.0
        g = np.fft.irfft(c * self.n, n=self.n)
        scale = max(1.0, float(np.max(np.abs(g))))
        a = 2.0 * c[:d + 1].real / scale
        a[0] /= 2.0
        return a


def _degree_search(cons: _StepConstruction, eps: float) -> int:
    lo, hi = 1, 8
    top = min(_FILTER_D_MAX, cons.n // 2)
    while cons.error(hi)[0] > eps:
        if hi >= top:
            raise ValueError(f"no filter of degree <= {top} reaches eps = {eps:g}")
        lo, hi = hi, min(2 * hi, top)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if cons.error(mid)[0] <= eps:
            hi = mid
        else:
            lo = mid
    return hi


def step_filter_polynomial(theta_pass: float, theta_stop: float, eps: float) -> tuple[int, np.ndarray]:
    """(d, a_0..a_d): the constructed step filter g(theta) = sum_k a_k cos(k theta) (module docstring).

    |1 - g| <= eps on [theta_pass, pi] and |g| <= eps on [0, theta_stop] (on the construction's grid
    and at the band edges), |g| <= 1 on the grid; ``step_filter_degree`` is its d. One construction
    takes 0.1-3 s, about 1/dtheta.
    """
    tp, ts = _phases(theta_pass, theta_stop)
    eps = _unit_open(eps, "eps")
    cons = _StepConstruction(tp, ts)
    d = _degree_search(cons, eps)
    return d, cons.cosine_coefficients(d, cons.error(d)[1])


def step_filter_degree(theta_pass: float, theta_stop: float, eps: float) -> int:
    """The degree d of the constructed step filter (the reference of :func:`filter_degree`; F43)."""
    tp, ts = _phases(theta_pass, theta_stop)
    return _degree_search(_StepConstruction(tp, ts), _unit_open(eps, "eps"))


def build_filter_table() -> tuple[tuple[float, float], ...]:
    """FILTER_TABLE from the construction: c_f(eps_f) = 1.01 max d dtheta_ref over the reference placements."""
    out = []
    for j in _TABLE_J:
        eps = 10.0 ** (-j / 4.0)
        worst = max(step_filter_degree(th, th - dth, eps) * dth for th in _TABLE_THETA for dth in _TABLE_DTHETA)
        out.append((eps, _TABLE_SAFETY * worst))
    return tuple(out)


def _table_index(eps: float) -> int:
    """Index of eps rounded down to the grid 10^(-j/4) (eps > 1e-2 rounds to 1e-2; eps >= 1e-10)."""
    j = min(_TABLE_J[-1], max(_TABLE_J[0], math.ceil(-4.0 * math.log10(eps) - 1e-9)))
    return j - _TABLE_J[0]


def filter_degree(theta_pass: float, theta_stop: float, eps: float) -> int:
    """F43's degree: ceil(c_f(eps rounded down to the grid)/dtheta); the construction for dtheta > 0.1 or eps < 1e-10."""
    tp, ts = _phases(theta_pass, theta_stop)
    eps = _unit_open(eps, "eps")
    dth = tp - ts
    if dth > _TABLE_DTHETA_MAX or eps < _TABLE_EPS_MIN:
        return step_filter_degree(tp, ts, eps)
    if not FILTER_TABLE:                                    # pragma: no cover (the constant is filled)
        raise RuntimeError("FILTER_TABLE is empty")
    return int(math.ceil(FILTER_TABLE[_table_index(eps)][1] / dth - 1e-12))


def _walk_phase(energy: float, lam: float, c: float, name: str) -> float:
    x = (float(energy) - c) / lam
    if not -1.0 - 1e-12 <= x <= 1.0 + 1e-12:
        raise ValueError(f"{name} = {energy!r} lies outside the walk's spectrum [c - lambda, c + lambda]")
    return math.acos(max(-1.0, min(1.0, x)))


@dataclass(frozen=True)
class FilterCost:
    """The step filter between U and E1lb and its cost (F43, module docstring).

    Attributes:
        degree: d (``filter_degree``).
        toffolis: C_filt = 2d C_W + 3(2d + 1)(beta - 2).
        logical_qubits: Q_walk + 1.
        u_upper: the pass edge (U, or E1lb - W_L for the E1lb-referenced forms; Hartree).
        e1_lower: the stop edge E1lb (Hartree).
        eps: eps_f.
    """

    degree: int
    toffolis: float
    logical_qubits: int
    u_upper: float
    e1_lower: float
    eps: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "degree", _int_at_least(self.degree, 1, "degree"))
        object.__setattr__(self, "toffolis", _positive(self.toffolis, "toffolis"))
        object.__setattr__(self, "logical_qubits", _int_at_least(self.logical_qubits, 1, "logical_qubits"))
        u = _finite(self.u_upper, "u_upper")
        e1 = _finite(self.e1_lower, "e1_lower")
        if not u < e1:
            raise ValueError(f"the pass edge must lie below the stop edge, got U = {u!r}, E1lb = {e1!r}")
        object.__setattr__(self, "u_upper", u)
        object.__setattr__(self, "e1_lower", e1)
        object.__setattr__(self, "eps", _unit_open(self.eps, "eps"))


def filter_cost(enc: BlockEncoding, c: float, u_upper: float, e1_lower: float, *, eps: float = 1e-2,
                params: FTParameters = FTParameters()) -> FilterCost:
    """The step filter of ``enc``'s walk with pass edge U and stop edge E1lb (F43).

    ``c`` is the walk offset (``ft.acceptance.walk_offset``); theta = arccos((E - c)/lambda). The
    E1lb-referenced forms pass ``u_upper = e1_lower - W_L``.
    """
    if not isinstance(enc, BlockEncoding):
        raise ValueError(f"enc must be a BlockEncoding, got {type(enc).__name__}")
    c = _finite(c, "c")
    lam = _positive(enc.lam, "enc.lam")
    u = _finite(u_upper, "u_upper")
    e1 = _finite(e1_lower, "e1_lower")
    if not u < e1:
        raise ValueError(f"u_upper must lie below e1_lower, got {u!r} >= {e1!r}")
    tp = _walk_phase(u, lam, c, "u_upper")
    ts = _walk_phase(e1, lam, c, "e1_lower")
    d = filter_degree(tp, ts, eps)
    calls = 2 * d
    tof = calls * int(enc.toffoli_per_step) + gqsp_rotation_toffolis(calls, params.beta)
    return FilterCost(degree=d, toffolis=float(tof), logical_qubits=int(enc.ancilla_walk) + 1, u_upper=u,
                      e1_lower=e1, eps=eps)


# ------------------------------------------------------------------ two-stage QPE
def _lattice_tail(x: np.ndarray) -> np.ndarray:
    """T(x) = sum_{k>=0} P(x + k) of the continuum sine-window lattice law (F34), x > 1/2 (ft.acceptance)."""
    x = np.asarray(x, dtype=float)
    psi = scipy.special.polygamma(1, x - 0.5) + scipy.special.polygamma(1, x + 0.5)
    return (8.0 * np.cos(np.pi * x) ** 2 / np.pi ** 2) * (psi / 16.0 - 1.0 / (8.0 * (x - 0.5)))


def _sup_tail(d: float) -> float:
    """tau(D) = sup_{x >= D} T(x) = sup over [D, D + 1) (continuum sine window), D >= 1."""
    s = np.linspace(0.0, 1.0, 2049)[:-1]
    return float(np.max(_lattice_tail(d + s)))


def _tail_distance(eps: float) -> float:
    """D_p = min{D >= 1 : tau(D) <= eps} by bisection (tau is non-increasing)."""
    lo, hi = 1.0, 2.0
    while _sup_tail(hi) > eps:
        lo, hi = hi, 2.0 * hi
        if hi > 1e7:
            raise ValueError(f"eps_p = {eps:g} needs a tail distance above 1e7 grid units")
    if _sup_tail(lo) <= eps:
        return lo
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _sup_tail(mid) <= eps:
            hi = mid
        else:
            lo = mid
    return hi


def _coarse_ok_exact(m: int, theta_u: float, theta_1: float, eps: float) -> bool:
    """The finite-register condition at length m (module docstring): FFT kernels, the best cut."""
    a = phase_window(m, "sine")
    step = 2.0 * math.pi / m
    s = np.arange(_COARSE_SUB + 1) / _COARSE_SUB
    k_lo = _branch_kernels(theta_1 - s * step, a)            # levels at or above E1lb: phases <= theta_1
    k_hi = _branch_kernels(theta_u + s * step, a)            # levels at or below U: phases >= theta_u
    ph = 2.0 * np.pi * np.arange(m) / m
    ph = np.where(ph <= np.pi, ph, 2.0 * np.pi - ph)
    order = np.argsort(-ph, kind="stable")
    phs = ph[order]
    last = np.r_[phs[:-1] != phs[1:], True]                 # cuts between distinct folded phases
    leak = np.cumsum(k_lo[:, order], axis=1)[:, last].max(axis=0)
    keep = np.cumsum(k_hi[:, order], axis=1)[:, last].min(axis=0)
    i = int(np.searchsorted(leak, eps, side="right")) - 1   # the largest acceptance set with leak <= eps
    return i >= 0 and bool(keep[i] >= 1.0 - eps)


def _coarse_ok_continuum(m: int, theta_u: float, theta_1: float, d_p: float) -> bool:
    """Continuum condition: a first accepted grid index a with a - x_1 >= D_p and x_U - (a - 1) >= D_p."""
    x1 = theta_1 * m / (2.0 * math.pi)
    xu = theta_u * m / (2.0 * math.pi)
    lo = math.ceil(x1 + d_p - 1e-12)
    hi = math.floor(xu - d_p + 1.0 + 1e-12)
    return lo <= hi and lo <= m // 2


def coarse_qpe_points(theta_u: float, theta_1: float, eps_p: float, *, window: str = "sine") -> int:
    """M_c of the coarse QPE between the phases of U (theta_u) and E1lb (theta_1 < theta_u) (F43).

    The smallest M at which every level at or above E1lb passes the best threshold with probability
    <= eps_p and every level at or below U with probability >= 1 - eps_p (module docstring): a scan
    upward from 2 pi (2 D_p - 1.5)/dtheta, the finite-register sine law up to M = 4096 and the
    continuum lattice law above. Only the sine window is supported.
    """
    if window != "sine":
        raise ValueError(f"only the sine window is supported, got {window!r}")
    tu, t1 = _phases(theta_u, theta_1)
    eps = _unit_open(eps_p, "eps_p")
    if eps >= 0.5:
        raise ValueError(f"eps_p must lie below 1/2, got {eps_p!r}")
    d_p = _tail_distance(eps)
    dth = tu - t1
    m = max(4, int(math.floor(2.0 * math.pi * (2.0 * d_p - 1.5) / dth)))
    while m <= _COARSE_M_MAX:
        ok = _coarse_ok_exact(m, tu, t1, eps) if m <= _COARSE_EXACT_MAX else _coarse_ok_continuum(m, tu, t1, d_p)
        if ok:
            return m
        m += 1
    raise ValueError(f"no coarse length up to {_COARSE_M_MAX} separates the edges")    # pragma: no cover


def coarse_qpe_toffolis(enc: BlockEncoding, c: float, u_upper: float, e1_lower: float, *,
                        eps_p: float = 1e-4) -> float:
    """C_c = (M_c/2) C_W of the coarse QPE between U and E1lb on ``enc``'s walk (offset ``c``; F43)."""
    if not isinstance(enc, BlockEncoding):
        raise ValueError(f"enc must be a BlockEncoding, got {type(enc).__name__}")
    c = _finite(c, "c")
    lam = _positive(enc.lam, "enc.lam")
    u = _finite(u_upper, "u_upper")
    e1 = _finite(e1_lower, "e1_lower")
    if not u < e1:
        raise ValueError(f"u_upper must lie below e1_lower, got {u!r} >= {e1!r}")
    m = coarse_qpe_points(_walk_phase(u, lam, c, "u_upper"), _walk_phase(e1, lam, c, "e1_lower"), eps_p)
    return 0.5 * m * float(enc.toffoli_per_step)


# ------------------------------------------------------------------ scheme costs
def _h(scheme: str, p: float, c: float, c_qpe: float, *, c_filt: float | None, c_coarse: float | None,
       p_lb: float | None = None, delta: float = 0.01) -> float | None:
    """H_s(p, c): the hand-off cost beyond C_loop and the verification runs (None if an input is missing)."""
    if scheme == "rep":
        return (c + c_qpe) / p
    if scheme == "aa":
        return grover_calls(p)[0] * (c + c_qpe)
    if scheme == "bbht":
        return exponential_search_calls(p) * (c + c_qpe)
    if scheme == "fp":
        if p_lb is None or not p_lb > 0.0:
            return None
        return fixed_point_calls(min(p_lb, 1.0), delta) * (c + c_qpe)
    if scheme in ("lt", "lt_bbht", "lt_bbht_L"):
        if c_filt is None:
            return None
        n = grover_calls(p)[0] if scheme == "lt" else exponential_search_calls(p)
        return n * (c + c_filt) + c_qpe
    if scheme in ("2s", "2s_L"):
        if c_coarse is None:
            return None
        return (c + c_coarse) / p + c_qpe
    raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")


def _optional_cost(value, name: str) -> float | None:
    return None if value is None else _cost(value, name)


def scheme_costs(p: float, c_prep: float, c_loop: float, c_qpe: float, *, c_filt: float | None = None,
                 c_coarse: float | None = None, c_filt_l: float | None = None, c_coarse_l: float | None = None,
                 p_lb: float | None = None, delta: float = 0.01, n_verify: int = 1) -> dict[str, float | None]:
    """F43's table for one state: {scheme: T_s} over :data:`SCHEMES` (module docstring).

    Args:
        p: the overlap p_k (the success probability of the flagged QPE).
        c_prep, c_loop, c_qpe: C(Psi_k), C_loop(k) and C_QPE (any common unit).
        c_filt, c_coarse: C_filt(U) and C_c(U) with a certified U (None: lt, lt_bbht, 2s get None).
        c_filt_l, c_coarse_l: C_filt(W_L) and C_c(W_L) of the E1lb-referenced forms (None where they
            do not terminate).
        p_lb: a certified overlap bound for fp (None or <= 0: fp gets None).
        delta: fp's delta (success >= 1 - delta^2).
        n_verify: n_v of F40 for rep, aa, bbht and fp.
    """
    p = _overlap(p)
    c = _cost(c_prep, "c_prep")
    loop = _cost(c_loop, "c_loop")
    q = _positive(c_qpe, "c_qpe")
    nv = _int_at_least(n_verify, 1, "n_verify")
    delta = _unit_open(delta, "delta")
    cf, cc = _optional_cost(c_filt, "c_filt"), _optional_cost(c_coarse, "c_coarse")
    cfl, ccl = _optional_cost(c_filt_l, "c_filt_l"), _optional_cost(c_coarse_l, "c_coarse_l")
    plb = None if p_lb is None else float(p_lb)
    if plb is not None and not (math.isfinite(plb) or math.isnan(plb)):
        raise ValueError(f"p_lb must be finite or nan, got {p_lb!r}")
    if plb is not None and (math.isnan(plb) or plb <= 0.0):
        plb = None
    verify = (nv - 1) * q
    out: dict[str, float | None] = {}
    for s in SCHEMES:
        filt = cfl if s == "lt_bbht_L" else cf
        coarse = ccl if s == "2s_L" else cc
        h = _h(s, p, c, q, c_filt=filt, c_coarse=coarse, p_lb=plb, delta=delta)
        if h is None:
            out[s] = None
        else:
            out[s] = loop + h + (verify if s in ("rep", "aa", "bbht", "fp") else 0.0)
    return out


def final_qpe_form(u_upper: float, e1_lower: float, de: float, *, grid: float | None) -> str:
    """The form of the filtered schemes' final QPE with a certified U (F43): "B_U" iff D_U >= 1.3, else "B_L".

    D_U = (E1lb - U - dE)/g; in M3's window model (``grid=None``) "B_U" iff U <= E1lb - 2 dE.
    """
    u = _finite(u_upper, "u_upper")
    e1 = _finite(e1_lower, "e1_lower")
    de = _positive(de, "de")
    if grid is None:
        return "B_U" if u <= e1 - 2.0 * de else "B_L"
    return "B_U" if (e1 - u - de) / _positive(grid, "grid") >= FINAL_D_MIN else "B_L"


def _filter_pair(value, name: str) -> tuple[float, float] | None:
    """(C_filt, C_c) of a filter mapping with keys "c_filt" and "c_coarse", or None."""
    if value is None:
        return None
    try:
        return _cost(value["c_filt"], f"{name}['c_filt']"), _cost(value["c_coarse"], f"{name}['c_coarse']")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{name} must map 'c_filt' and 'c_coarse' to Toffoli counts, got {value!r}") from exc


def state_scheme_costs(p: float, c_prep: float, c_loop: float, c_qpe: float, *,
                       filt_u: Mapping | None = None, filt_wl: Mapping | None = None,
                       filt_oracle_u: Mapping | None = None, n_verify: float = 1.0, rule_b_terminates: bool = True,
                       rule_b_flags: Sequence[str] = (), rule_b_extra: float = 0.0, u_extra: float = 0.0,
                       u_final: str = "B_U", b_l_terminates: bool = True, e0_model: float | None = None,
                       e1_lower: float | None = None, w_l_certified: bool = False, p_lb: float | None = None,
                       delta: float = 0.01) -> dict:
    """Every scheme of F43 for one state with its flags (module docstring), in the unit of the cost arguments.

    Args:
        p: the overlap with psi_0' of H' (the QPE's Hamiltonian), the success probability of the flagged QPE.
        c_prep, c_loop, c_qpe: C(Psi_k), C_loop(k) and C_QPE.
        filt_u: {"c_filt", "c_coarse"} of the certified U_k (None: no certified U).
        filt_wl: the same at the pass edge E1lb - W_L (None where the E1lb-referenced forms do not terminate).
        filt_oracle_u: the same at the oracle U = E_0' + 2 dE: the "<s>@oracle_U" lower bounds, reported only
            without ``filt_u``.
        n_verify: F40's n_v of rule B's option for rep, aa, bbht and fp (real >= 1: the expected runs of a
            dedicated certificate, ``ft.handoff.RuleBRow.n_runs_expected``).
        rule_b_terminates: whether rule B's option terminates (else rep, aa, bbht, fp have no cost).
        rule_b_flags: the option's bracket flags (``ft.acceptance.Bracket.flags``), the flags of rep, aa, bbht, fp.
        rule_b_extra: the option's cost beyond C_loop(k) and its runs: C_loop(max(k, j_U)) - C_loop(k) + C_U.
        u_extra: the certificate cost of U_k, added to lt, lt_bbht, 2s and lt_rep.
        u_final: the U-referenced forms' final QPE (:func:`final_qpe_form`): "B_U", or "B_L" (one run,
            termination not certified; no cost where ``b_l_terminates`` is False).
        e0_model, e1_lower: E_0' and E1lb; given both, the E1lb-referenced forms need E_0' <= E1lb - W_L.
        w_l_certified: a certified U <= E1lb - W_L exists (their termination is certified).
        p_lb: a certified overlap bound for fp (None or <= 0: no fp).
        delta: fp's delta.

    Returns:
        {"costs": {key: T or None} over :data:`SCHEMES`, the "<s>@oracle_U" bounds (s in
        :data:`ORACLE_U_SCHEMES`, only where given) and :data:`SENSITIVITY_SCHEMES`, "flags": {key: [flag, ...]}
        for every key ("not_terminating", "not_terminating_L", "no_certificate", "no_overlap_bound" where a key has
        no cost; "oracle_U" on the bounds), "best_realizable": s* (:func:`best_realizable`, None if no realizable
        scheme has a cost), "best_extended": :func:`best_extended`}.
    """
    p = _overlap(p)
    c = _cost(c_prep, "c_prep")
    loop = _cost(c_loop, "c_loop")
    q = _positive(c_qpe, "c_qpe")
    nv = float(n_verify)
    if not (math.isfinite(nv) and nv >= 1.0):
        raise ValueError(f"n_verify must be finite and >= 1, got {n_verify!r}")
    extra_b = _cost(rule_b_extra, "rule_b_extra")
    extra_u = _cost(u_extra, "u_extra")
    if u_final not in ("B_U", "B_L"):
        raise ValueError(f"u_final must be 'B_U' or 'B_L', got {u_final!r}")
    fu, fl = _filter_pair(filt_u, "filt_u"), _filter_pair(filt_wl, "filt_wl")
    fo = _filter_pair(filt_oracle_u, "filt_oracle_u")
    flags_b = [str(f) for f in rule_b_flags]
    if (e0_model is None) != (e1_lower is None):
        raise ValueError("pass e0_model and e1_lower together")
    l_terminates = fl is not None and (e0_model is None
                                       or _finite(e0_model, "e0_model") <= _finite(e1_lower, "e1_lower") - W_L)
    costs: dict[str, float | None] = {}
    flags: dict[str, list[str]] = {}
    # rep, aa, bbht, fp: rule B's acceptance (its option's surcharge and runs)
    base = scheme_costs(p, c, loop + extra_b, q, p_lb=p_lb, delta=delta)
    for s in ("rep", "aa", "bbht", "fp"):
        if not rule_b_terminates:
            costs[s], flags[s] = None, ["not_terminating"]
        elif base[s] is None:
            costs[s], flags[s] = None, ["no_overlap_bound"]
        else:
            costs[s], flags[s] = base[s] + (nv - 1.0) * q, list(flags_b)
    # lt, lt_bbht, 2s, lt_rep: the certified U_k (its certificate cost added)
    u_keys = ORACLE_U_SCHEMES + ("lt_rep",)
    if fu is None:
        for s in u_keys:
            costs[s], flags[s] = None, ["no_certificate"]
    elif u_final == "B_L" and not b_l_terminates:
        for s in u_keys:
            costs[s], flags[s] = None, ["not_terminating"]
    else:
        cs = scheme_costs(p, c, loop + extra_u, q, c_filt=fu[0], c_coarse=fu[1])
        for s in ORACLE_U_SCHEMES:
            costs[s] = cs[s]
        costs["lt_rep"] = loop + extra_u + (c + fu[0]) / p + q
        for s in u_keys:
            flags[s] = [] if u_final == "B_U" else ["termination_not_certified"]
    # lt_bbht_L, 2s_L, lt_rep_L: the pass edge E1lb - W_L
    l_keys = ("lt_bbht_L", "2s_L", "lt_rep_L")
    if not l_terminates:
        for s in l_keys:
            costs[s], flags[s] = None, ["not_terminating_L"]
    else:
        cs = scheme_costs(p, c, loop, q, c_filt_l=fl[0], c_coarse_l=fl[1])
        costs["lt_bbht_L"], costs["2s_L"] = cs["lt_bbht_L"], cs["2s_L"]
        costs["lt_rep_L"] = loop + (c + fl[0]) / p + q
        for s in l_keys:
            flags[s] = [] if w_l_certified else ["termination_not_certified"]
    # the oracle-U lower bounds, never in s*
    if fu is None and fo is not None:
        orc = scheme_costs(p, c, loop, q, c_filt=fo[0], c_coarse=fo[1])
        for s in ORACLE_U_SCHEMES:
            costs[f"{s}@oracle_U"], flags[f"{s}@oracle_U"] = orc[s], ["oracle_U"]
    ordered = {s: costs[s] for s in SCHEMES}
    ordered.update((s, costs[s]) for s in costs if s not in ordered and s not in SENSITIVITY_SCHEMES)
    ordered.update((s, costs[s]) for s in SENSITIVITY_SCHEMES)
    return {"costs": ordered, "flags": {s: flags[s] for s in ordered},
            "best_realizable": _best_or_none(best_realizable, ordered),
            "best_extended": _best_or_none(best_extended, ordered)}


@dataclass(frozen=True)
class SchemeRow:
    """The hand-off of iterate k under every scheme (F43, module docstring).

    Attributes:
        k, p, c_prep, c_loop: the iterate, its overlap, C(Psi_k) and C_loop(k).
        costs: {scheme: T_s} for :data:`SCHEMES` (None where a scheme needs a missing certificate or
            does not terminate), the oracle-U lower bounds under "<scheme>@oracle_U" and optionally the
            sensitivity keys :data:`SENSITIVITY_SCHEMES`.
        best_realizable: s* = :func:`best_realizable` of ``costs``.
        flags: "oracle_U" (lower bounds reported), "termination_not_certified", "not_terminating", ...
    """

    k: int
    p: float
    c_prep: float
    c_loop: float
    costs: dict[str, float | None]
    best_realizable: str
    flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "k", _int_at_least(self.k, 0, "k"))
        object.__setattr__(self, "p", _overlap(self.p))
        object.__setattr__(self, "c_prep", _cost(self.c_prep, "c_prep"))
        object.__setattr__(self, "c_loop", _cost(self.c_loop, "c_loop"))
        allowed = set(SCHEMES) | {f"{s}@oracle_U" for s in ORACLE_U_SCHEMES} | set(SENSITIVITY_SCHEMES)
        costs = dict(self.costs)
        if not set(costs) <= allowed:
            raise ValueError(f"unknown cost keys {sorted(set(costs) - allowed)}")
        for key, val in costs.items():
            if val is not None:
                costs[key] = _cost(val, f"costs[{key!r}]")
        object.__setattr__(self, "costs", costs)
        if self.best_realizable not in REALIZABLE:
            raise ValueError(f"best_realizable must be one of {REALIZABLE}, got {self.best_realizable!r}")
        if costs.get(self.best_realizable) is None:
            raise ValueError(f"best_realizable {self.best_realizable!r} has no cost")
        flags = tuple(self.flags)
        if not all(isinstance(f, str) for f in flags):
            raise ValueError("flags must be strings")
        object.__setattr__(self, "flags", flags)


def best_realizable(costs: Mapping[str, float | None]) -> str:
    """s* = argmin of T_s over :data:`REALIZABLE` keys with a cost (ties: the order of REALIZABLE; F43, U0).

    Never an oracle scheme (aa, lt), an "@oracle_U" key or a scheme without a cost (None: a missing
    certificate, fp without p_lb > 0, a non-terminating form). ValueError if none has a cost.
    """
    return _argmin(costs, REALIZABLE)


def _argmin(costs: Mapping[str, float | None], keys: Sequence[str]) -> str:
    best, arg = math.inf, None
    for s in keys:
        v = costs.get(s)
        if v is None:
            continue
        v = float(v)
        if math.isnan(v):
            continue
        if arg is None or v < best:
            best, arg = v, s
    if arg is None:
        raise ValueError("no realizable scheme has a cost")
    return arg


def best_extended(costs: Mapping[str, float | None]) -> str:
    """argmin of T_s over :data:`REALIZABLE` and :data:`SENSITIVITY_SCHEMES` with a cost (ties: that order).

    A sensitivity of s* to the scheme set (module docstring), never the verdict of record.
    ValueError if none has a cost.
    """
    return _argmin(costs, REALIZABLE + SENSITIVITY_SCHEMES)


def extended_verdict(costs_k: Mapping[str, float | None], costs_k1: Mapping[str, float | None]) -> dict:
    """The verdict of :func:`iteration_verdict` at :func:`best_extended` in place of s* (a sensitivity).

    Returns {"best_k", "best_k1", "saving_best", "relative_best", "pays_best"} as iteration_verdict.
    """
    bk, bk1 = _best_or_none(best_extended, costs_k), _best_or_none(best_extended, costs_k1)
    out = {"best_k": bk, "best_k1": bk1, "saving_best": None, "relative_best": None, "pays_best": False}
    if bk is not None and bk1 is not None:
        c = _compare(costs_k[bk], costs_k1[bk1])
        if c is not None:
            out.update(saving_best=c["saving"], relative_best=c["relative"], pays_best=c["pays"])
    return out


def _best_or_none(fn: Callable, costs: Mapping) -> str | None:
    try:
        return fn(costs)
    except ValueError:
        return None


def _compare(a, b) -> dict | None:
    """{"saving": a - b, "relative": (a - b)/a, "pays": a > b}; None if a cost is missing, nan, or both are inf."""
    if a is None or b is None:
        return None
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b) or (math.isinf(a) and math.isinf(b)):
        return None
    saving = a - b
    rel = saving / a if math.isfinite(a) and a > 0.0 else math.nan
    return {"saving": saving, "relative": rel, "pays": bool(saving > 0.0)}


def _per_row(value, n: int, name: str) -> list:
    """A scalar (or None) for every row, or one value per row."""
    if value is None or np.isscalar(value):
        return [value] * n
    vals = list(value)
    if len(vals) != n:
        raise ValueError(f"{name} has {len(vals)} entries for {n} rows")
    return vals


def _row_fields(row) -> dict:
    if isinstance(row, Mapping):
        return {"k": row["k"], "p": row["p"], "c_prep": row.get("c_prep", 0.0), "c_loop": row.get("c_loop", 0.0),
                "p_lb": row.get("p_lb"), "flags": tuple(row.get("flags", ()))}
    return {"k": row.k, "p": row.p, "c_prep": row.c_prep, "c_loop": row.c_loop, "p_lb": getattr(row, "p_lb", None),
            "flags": ()}


def scheme_table(rows: Sequence, c_qpe: float, *, c_filt=None, c_coarse=None, c_filt_l=None, c_coarse_l=None,
                 p_lb=None, n_verify=1, c_filt_oracle=None, c_coarse_oracle=None,
                 flags: Sequence[str] = ()) -> list[SchemeRow]:
    """F43's table over hand-off rows (``ft.handoff.HandoffRow`` or mappings with k, p, c_prep, c_loop).

    Every keyword is a scalar (or None) for all rows or a sequence with one value per row:
    ``c_filt``/``c_coarse`` with a certified U_k (``filter_cost``, ``coarse_qpe_toffolis``),
    ``c_filt_l``/``c_coarse_l`` of the E1lb-referenced forms, ``p_lb`` (default: a HandoffRow's
    Eckart-Temple ``p_lb``, or a mapping's "p_lb"), ``n_verify`` (F40's n_v). Where ``c_filt`` and
    ``c_coarse`` are None, ``c_filt_oracle``/``c_coarse_oracle`` (U = E_0' + 2 dE) give the
    "<scheme>@oracle_U" lower bounds and the flag "oracle_U". ``flags`` (and a mapping's "flags") are
    attached to the rows. With n_v = 1 "rep" equals ``HandoffRow.cost_expected``.
    """
    fields = [_row_fields(r) for r in rows]
    n = len(fields)
    cols = {name: _per_row(val, n, name) for name, val in (
        ("c_filt", c_filt), ("c_coarse", c_coarse), ("c_filt_l", c_filt_l), ("c_coarse_l", c_coarse_l),
        ("p_lb", p_lb), ("n_verify", n_verify), ("c_filt_oracle", c_filt_oracle),
        ("c_coarse_oracle", c_coarse_oracle))}
    extra = tuple(flags)
    out = []
    for i, f in enumerate(fields):
        plb = cols["p_lb"][i] if p_lb is not None else f["p_lb"]
        costs = scheme_costs(f["p"], f["c_prep"], f["c_loop"], c_qpe, c_filt=cols["c_filt"][i],
                             c_coarse=cols["c_coarse"][i], c_filt_l=cols["c_filt_l"][i],
                             c_coarse_l=cols["c_coarse_l"][i], p_lb=plb, n_verify=cols["n_verify"][i])
        row_flags = list(f["flags"]) + [fl for fl in extra if fl not in f["flags"]]
        if cols["c_filt"][i] is None and cols["c_coarse"][i] is None and (
                cols["c_filt_oracle"][i] is not None or cols["c_coarse_oracle"][i] is not None):
            oracle = scheme_costs(f["p"], f["c_prep"], f["c_loop"], c_qpe, c_filt=cols["c_filt_oracle"][i],
                                  c_coarse=cols["c_coarse_oracle"][i])
            for s in ORACLE_U_SCHEMES:
                if oracle[s] is not None:
                    costs[f"{s}@oracle_U"] = oracle[s]
            if "oracle_U" not in row_flags:
                row_flags.append("oracle_U")
        out.append(SchemeRow(k=f["k"], p=f["p"], c_prep=f["c_prep"], c_loop=f["c_loop"], costs=costs,
                             best_realizable=best_realizable(costs), flags=tuple(row_flags)))
    return out


# ------------------------------------------------------------------ savings and break-even (F44)
def _scheme_inputs(scheme: str, c_filt, c_coarse) -> tuple[float | None, float | None]:
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")
    cf = _optional_cost(c_filt, "c_filt")
    cc = _optional_cost(c_coarse, "c_coarse")
    if scheme in ("lt", "lt_bbht", "lt_bbht_L") and cf is None:
        raise ValueError(f"scheme {scheme!r} needs c_filt")
    if scheme in ("2s", "2s_L") and cc is None:
        raise ValueError(f"scheme {scheme!r} needs c_coarse")
    return cf, cc


def _h_scheme(scheme: str, p: float, c: float, c_qpe: float, cf, cc) -> float:
    """H_s(p, c); fp with p_lb = p (the most a certified bound can be)."""
    return _h(scheme, p, c, c_qpe, c_filt=cf, c_coarse=cc, p_lb=p)


def saving_budget(scheme: str, p0: float, *, c_qpe: float, c_filt: float | None = None,
                  c_coarse: float | None = None) -> float:
    """F44: H_s(p_0, 0) - H_s(1, 0), the most a perfect state can save under ``scheme``.

    lt_bbht_L takes C_filt(W_L) and 2s_L C_c(W_L) as ``c_filt``/``c_coarse``; fp uses p_lb = p.
    """
    cf, cc = _scheme_inputs(scheme, c_filt, c_coarse)
    q = _positive(c_qpe, "c_qpe")
    p0 = _overlap(p0, "p0")
    return _h_scheme(scheme, p0, 0.0, q, cf, cc) - _h_scheme(scheme, 1.0, 0.0, q, cf, cc)


def break_even_overlap(scheme: str, c_iter: float, p1: float, *, c_qpe: float, c_prep1: float = 0.0,
                       c_filt: float | None = None, c_coarse: float | None = None) -> float:
    """F44: the largest p_0 with H_s(p_0, 0) - H_s(p_1, c_1) >= C_it (an iteration from HF-like p_0 pays below it).

    Found by stepping down from p_1 by factors of 1.1 (so that the bump of N* on (0.25, 0.32) and the
    ripples of E_BBHT are resolved) to the first p with the condition, then bisection; for C_it >> C_QPE
    the asymptotes of the module docstring. ``c_iter`` > 0.
    """
    cf, cc = _scheme_inputs(scheme, c_filt, c_coarse)
    q = _positive(c_qpe, "c_qpe")
    c_it = _positive(c_iter, "c_iter")
    p1 = _overlap(p1, "p1")
    c1 = _cost(c_prep1, "c_prep1")
    target = c_it + _h_scheme(scheme, p1, c1, q, cf, cc)

    def pays(p: float) -> bool:
        return _h_scheme(scheme, p, 0.0, q, cf, cc) >= target

    hi = p1
    lo = p1 / 1.1
    while not pays(lo):
        hi, lo = lo, lo / 1.1
        if lo < 1e-300:
            raise ValueError("no break-even overlap above 1e-300")    # pragma: no cover
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if pays(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _cached_saving(h: Callable[[float, float], float]) -> Callable:
    """mu = h(p_k, c_k) - h(p_k1, c_k1) with the values of h kept per (p, c) argument; scalars give a float, arrays
    (broadcast) an array of the same shape, evaluated elementwise. The kept values serve the break-even scans of
    ``ft.scaling.break_even_size``, which evaluate the same extrapolated overlaps for every case of a bracket."""
    known: dict[tuple[float, float], float] = {}

    def value(p: float, c: float, p_name: str, c_name: str) -> float:
        key = (float(p), float(c))
        v = known.get(key)
        if v is None:
            v = known[key] = h(_overlap(p, p_name), _cost(c, c_name))
        return v

    def saving(p_k, p_k1, c_k=0.0, c_k1=0.0):
        args = (p_k, p_k1, c_k, c_k1)
        if all(np.ndim(x) == 0 for x in args):
            return value(p_k, c_k, "p_k", "c_k") - value(p_k1, c_k1, "p_k1", "c_k1")
        a, b, ca, cb = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in args))
        out = [value(x, y, "p_k", "c_k") - value(u, v, "p_k1", "c_k1")
               for x, u, y, v in zip(a.ravel(), b.ravel(), ca.ravel(), cb.ravel())]
        return np.asarray(out, dtype=float).reshape(a.shape)

    return saving


def scheme_saving(scheme: str, *, c_qpe: float = 1.0, c_filt: float | None = None,
                  c_coarse: float | None = None) -> Callable[[float, float, float, float], float]:
    """mu_k^s as a function (p_k, p_k1, c_k, c_k1) -> H_s(p_k, c_k) - H_s(p_k1, c_k1) (F44; ``ft.scaling.cost_ratio``).

    Arrays are accepted (broadcast, elementwise); the values of H_s are kept per (p, c), so the scans of
    ``ft.scaling.break_even_size`` evaluate it once per extrapolated overlap. The attribute ``qpe_scaled`` is True
    where H_s scales with C_QPE at fixed C(Psi) (rep, aa, bbht, fp) and False for the filtered schemes, which carry
    C_QPE once: ``ft.scaling.break_even_size`` lengthens only the former beyond the certification window.
    """
    cf, cc = _scheme_inputs(scheme, c_filt, c_coarse)
    q = _positive(c_qpe, "c_qpe")
    hook = _cached_saving(lambda p, c: _h_scheme(scheme, p, c, q, cf, cc))
    hook.qpe_scaled = scheme in _QPE_SCALED
    return hook


def best_realizable_saving(*, schemes: Sequence[str] = REALIZABLE, c_qpe: float = 1.0, c_filt: float | None = None,
                           c_coarse: float | None = None, c_filt_l: float | None = None,
                           c_coarse_l: float | None = None, n_verify: int = 1,
                           delta: float = 0.01) -> Callable[[float, float, float, float], float]:
    """mu*_k = min_s H_s(p_k, c_k) - min_s H_s(p_k1, c_k1) as a function (p_k, p_k1, c_k, c_k1) (F44 at the best
    realizable scheme; the ``saving`` hook of ``ft.scaling.cost_ratio`` and ``break_even_size``).

    The minima run over the schemes of ``schemes`` (a subset of :data:`REALIZABLE`) that have a cost with the
    given inputs (:func:`scheme_costs` at C_loop = 0: lt_bbht and 2s need ``c_filt``/``c_coarse`` of a certified
    U, lt_bbht_L and 2s_L ``c_filt_l``/``c_coarse_l``; fp has no cost, since it needs a certified p_lb of the
    state, which an overlap alone does not give), each with F40's (n_verify - 1) c_qpe for rep and bbht, which
    does not cancel between schemes. The minimizing scheme may differ between the two iterates; iteration k pays
    at the best realizable scheme iff c_iter(k) < mu*_k, i.e. min_s T_s(k + 1) < min_s T_s(k)
    (:func:`iteration_verdict`). ValueError if no scheme of ``schemes`` has a cost. Arrays are accepted
    (broadcast, elementwise; values kept per (p, c) as in :func:`scheme_saving`). ``qpe_scaled`` (as in
    :func:`scheme_saving`) is set where the schemes with a cost are all of one kind; a mixed set leaves it unset.
    """
    chosen = tuple(schemes)
    if not chosen or any(s not in REALIZABLE for s in chosen):
        raise ValueError(f"schemes must be a non-empty subset of {REALIZABLE}, got {schemes!r}")
    q = _positive(c_qpe, "c_qpe")
    nv = _int_at_least(n_verify, 1, "n_verify")
    kw = dict(c_filt=c_filt, c_coarse=c_coarse, c_filt_l=c_filt_l, c_coarse_l=c_coarse_l, delta=delta, n_verify=nv)
    probe = scheme_costs(1.0, 0.0, 0.0, q, **kw)
    if all(probe[s] is None for s in chosen):
        raise ValueError(f"none of {chosen} has a cost with the given inputs")

    def h_best(p: float, c: float) -> float:
        costs = scheme_costs(p, c, 0.0, q, **kw)
        return float(costs[best_realizable({s: costs[s] for s in chosen})])

    hook = _cached_saving(h_best)
    kinds = {s in _QPE_SCALED for s in chosen if probe[s] is not None}
    if len(kinds) == 1:
        hook.qpe_scaled = kinds.pop()
    return hook


def iteration_verdict(costs_k: Mapping[str, float | None], costs_k1: Mapping[str, float | None]) -> dict:
    """F44's decision rule on one trajectory: iteration k pays under scheme s iff T_s(k + 1) < T_s(k).

    Args:
        costs_k, costs_k1: {scheme: T_s} at Psi_k and Psi_{k+1} in one unit, C_loop included
            (:func:`scheme_costs`, ``SchemeRow.costs``; None where a scheme has no cost). Every key with a cost at
            both iterates is compared, the oracle schemes and the "<scheme>@oracle_U" bounds included; they never
            enter the verdict of record.

    Returns:
        {"by_scheme": {s: {"saving": T_s(k) - T_s(k + 1), "relative": saving / T_s(k), "pays": saving > 0}},
        "best_k", "best_k1": s* = :func:`best_realizable` at each iterate (None if no realizable scheme has a
        cost), "saving_best", "relative_best": T_{s*(k)}(k) - T_{s*(k+1)}(k + 1) and its share of T_{s*(k)}(k), and
        "pays_best": the verdict of record (the best realizable hand-off after the iteration is cheaper than the
        best before it; False without a realizable cost at both iterates)}. An infinite T_s(k) with a finite
        T_s(k + 1) pays (saving inf); infinite costs at both iterates are not compared.
    """
    by_scheme = {}
    for s in dict.fromkeys(list(costs_k) + list(costs_k1)):
        c = _compare(costs_k.get(s), costs_k1.get(s))
        if c is not None:
            by_scheme[s] = c
    bk, bk1 = _best_or_none(best_realizable, costs_k), _best_or_none(best_realizable, costs_k1)
    out = {"by_scheme": by_scheme, "best_k": bk, "best_k1": bk1, "saving_best": None, "relative_best": None,
           "pays_best": False}
    if bk is not None and bk1 is not None:
        c = _compare(costs_k[bk], costs_k1[bk1])
        if c is not None:
            out.update(saving_best=c["saving"], relative_best=c["relative"], pays_best=c["pays"])
    return out

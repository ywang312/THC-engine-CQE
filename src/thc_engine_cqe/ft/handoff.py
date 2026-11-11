"""The CQE -> QPE hand-off (F15) and the cost of the CQE energy alone (F17).

Tags and formula numbers are those of :mod:`thc_engine_cqe.ft.primitives`. Costs are
Toffolis, energies Hartree, variances Ha^2.

Hand-off (F15) [derived]
------------------------
QPE on a qubitized walk returns an outcome within dE of E_0 with probability w p_k,
p_k = |<psi_0|Psi_k>|^2, where the window factor w (not the amplitude-estimation constant
kappa of F11) is set by where the ground-state phase falls on the QPE grid and is nearly
state-independent (``ft.walk``: on H4 with DF H' at threshold 1e-3 and t = 15, the setup of
the design notes [design], w = 0.986 and the ratio varies by 1e-8 along the L-BFGS iterates;
tests/test_ft_walk.py::test_window_probability_over_p_is_k_independent); w and the success
probability of a QPE run, 1 - p_fail_qpe, rescale every QPE term alike and are both set to
1, as in F15. Handing off the CQE iterate Psi_k costs

    T_E(k) = C_loop(k) + (C(Psi_k) + C_QPE) / p_k                 rule E, expected cost
    T_d(k) = C_loop(k) + R_d(p_lb,k) (C(Psi_k) + C_QPE),  R_d(p) = ceil(ln d / ln(1 - p))

with C(Psi_k) the state preparation (``ft.stateprep.PrepCost.cumulative``), C_QPE one
phase estimation (``ft.qpe.QPECost.toffolis``) and C_loop(k) = sum_{j<k} C_iter(j) the
measured iterations that produced d_0 .. d_{k-1} (F14):

* "measured": every iteration costs;
* "classical_first_step": C_iter(0) = 0, since A_0 and eps_0 = eps_2 are classical at HF
  (F16). The record should then come from ``exact_trajectory(..., first_step="second_order")``,
  whose Psi_1 is the classical step; with an exact-line-search record the rows k >= 1 carry
  the flag "exact_first_step" (Psi_1 is exp(eps* G_0) HF; on the stretched h6s and n2s,
  eps_2/eps* = 1.14 and 0.80, the classical Psi_1 lies 3.7 and 11.9 mHa higher, ft.trajectory);
* "free": C_loop = 0, the bound for generators that cost nothing (e.g. replayed
  classical-ACSE generators).

C_iter(j) may be +inf (an infeasible route, F10): T_E and T_d are then inf from k = j + 1
on. Rows reach the last iterate covered by the trajectory, by C(Psi_k) and by the
iteration costs (the classical first step needs none for k <= 1).

Rule E accepts the first outcome below E_k + 2 dE; it is certified iff E1 - E_k > 3 dE,
for a lower bound E1 on the first excited energy of the state's symmetry sector
(:func:`rule_e_certifiable`); otherwise an excited state below E_k + 2 dE can be accepted.
Rule d (fixed confidence d = ``params.delta_conf``) makes R_d runs and keeps the lowest
outcome, so it needs a certified p_lb <= p_k; runs with the exact p_k are reported as the
oracle column, which is not certified. :func:`optimal_handoff` mirrors this: "expected" minimizes T_E over
the certified rows only (k_opt of rule E), "delta" the certified T_d, and
"expected_oracle" and "delta_oracle" minimize over every row, as if the ground-state
outcomes could be recognized (rule E) or p_k were known (rule d). The certificate is the
Eckart-Temple bound [cited: Eckart, Phys. Rev. 36, 878 (1930) (volume unverified); Temple
1928 (unverified)] (:func:`overlap_lower_bound`):

    p_k >= p_lb = 1 / (1 + sigma_k^2 / (E1 - E_k)^2)      for E_k < E1 (nan otherwise),

re-derived: with q = 1 - p_k and E_j >= E1 > E_k for j >= 1, q (E1 - E_k) <=
sum_{j>=1} p_j (E_j - E_k) = p_k (E_k - E_0), and by Cauchy-Schwarz
sigma^2 >= p_k (E_k - E_0)^2 / q >= q (E1 - E_k)^2 / p_k. It is exact for two-level states,
and a lower bound on E1 keeps it valid. ``use_e1="same_irrep"`` takes the same-irrep E1
of the trajectory (F15), "any_irrep" the any-irrep one. Observed (exact L-BFGS records,
the selected DF C_QPE, C_iter = 100 C_QPE per measured iteration [assumption: the design
estimate, ">= 10^2 C_QPE"]): the cheapest T_E row of the measured loop is HF for h6s
and n2s (1.74 and 3.96 C_QPE), where rule E is not certified (E_HF - E_0 = 0.282 and
0.566 Ha against E1 - E_0 = 0.105 and 0.192 Ha); the certified optimum is k = 1
(101 C_QPE) and k = 2 (203 C_QPE).

Per hand-off iterate k, :class:`HandoffRow` reports p_k, E_k - E_0, C(Psi_k), C_loop(k),
1/p_k, T_E, p_lb, R_d(p_lb) and T_d (None when p_lb is nan or 0), R_d(p_k) and its cost,
the certification of rule E, the QPE-side saving against HF,

    saving_vs_hf(k) = C_QPE / p_0 - (C(Psi_k) + C_QPE) / p_k    <= C_QPE (1/p_0 - 1)  (budget),

and the break-even HF overlap p_0*(k) = C_QPE / T_E(k): handing off Psi_k beats HF iff
p_HF < p_0*(k). Step k pays (T_E(k+1) < T_E(k)) iff the marginal condition
C_iter(k) < (C(Psi_k) + C_QPE)/p_k - (C(Psi_{k+1}) + C_QPE)/p_{k+1} holds (tested).
Overlaps that round above 1 are read as 1, so saving_vs_hf(0) = 0 exactly.

CQE energy alone (F17) [design]
-------------------------------
k_v is the first iterate with E_k - E_0 <= ``params.e_tol_energy`` (1 mHa). The final
energy with exact integrals takes S_fin = V_E / ``params.sigma_final``^2 shots of
C(Psi_kv) + C_rot (V_E: the DF-grouped energy variance, C_rot: the readout rotation,
``ft.stateprep.readout_rotation_toffolis``), so

    C = C_loop(k_v) + S_fin (C(Psi_kv) + C_rot),

with the loop convention of the hand-off (default: classical first step), compared with
the HF hand-off C_QPE / p_0 (:func:`cqe_only_cost`). This is a lower bound: finite-shot
trajectories stop on noise floors [M2].

Rule B: the E1-referenced hand-off (F40, F42) [derived]
-------------------------------------------------------
Rule B accepts the first QPE outcome below a threshold theta in the bracket between a certified
upper bound U >= E_0' (``ft.certify.EnergyCertificate``) and E1lb (F41), then n_v - 1
verification runs; its forms B_U (theta = U + dE) and B_L (theta = E1lb - D_L g, no U) and n_v
are those of ``ft.acceptance.acceptance_bracket`` (the form with the smaller n_v, B_U on ties).
In M3's window model (``grid=None``: n_v = 1, B_L at E1lb - dE) its expected cost is

    T_B(k) = C_loop(max(k, j_U)) + C_U + (C(Psi_k) + C_QPE)/(p_k K_0) + (n_v - 1) C_QPE/K_0,   K_0 = 1,

i.e. M3's oracle T_E(k) plus the certificate (j_U = ``EnergyCertificate.loop_iterate``, the
iterate up to which the loop must run before U exists, and its Toffolis C_U; 0 and 0 for
CISD(H')) and the verification runs (:func:`rule_b_table`). Per row the cheapest option among the
certificates (oracle certificates excluded) and no certificate at all (B_L) is taken, ties to
certified termination, then B_U, then no certificate; B_U terminates (U >= E_0'), B_L iff
E_0' + dE <= theta, checked in the model with E_0' (``e0_model``, default the record's E_0 of
H), and is certified only by a U with U + dE <= theta (flag "termination_not_certified"); rows
where nothing terminates cost inf (flag "not_terminating"). Each certificate's form is the
realizable choice of ``acceptance_bracket`` (no E_0'); where that is B_L, B_U at U + dE with its
own n_v (more runs than B_L's) is a second option of the certificate, so a row whose B_L does not
terminate still has a terminating hand-off wherever a U exists (:func:`rule_b_bracket` is the same
choice for a state with a classical U only). A dedicated sampled or AE energy (sigma not None) is a
random bound U^ ~ N(U, sigma^2): its B_U option costs C_U + C_QPE (E[n_v] - 1), with n_v fixed after
U^ is known (fewer than B_L's n_l, else B_L; ``ft.acceptance.expected_verification_runs``), and
where B_L does not terminate C_U / P(U^ admissible) + C_QPE (E[n_v | admissible] - 1), the
certificate repeated until U^ <= E1lb - 2 dE; the row reports the planned U, theta and n_v and the
expected runs and certificate cost. M3's rule E is B_U with
U = E_k + dE and n_v = 1: in the window model T_B(k) = T_E(k), certified iff E1 - E_k >= 3 dE
(:func:`rule_e_certifiable` with the edge included; tested). :func:`optimal_rule_b` takes the
cheapest row ("bracketed", realizable, flagged rows included) or M3's oracle T_E
("bracketed_oracle"). The runtime of rule B is a quantile, not a certificate: the number of
attempts is geometric with success probability q >= p_0 K_0^{n_v}, so a deadline met with
probability >= 1 - delta needs R_delta(q) = ceil(ln delta / ln(1 - q)) attempts
(:func:`runtime_quantile`).
"""

from __future__ import annotations

import decimal
import fractions
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .primitives import FTParameters
from .qpe import QPECost
from .trajectory import TrajectoryRecord, first_iterate_below

if TYPE_CHECKING:                                       # rule B's types; imported at call time below
    from .acceptance import AcceptanceStats, Bracket
    from .certify import EnergyCertificate

__all__ = [
    "LOOPS",
    "RULES",
    "runs_for_confidence",
    "overlap_lower_bound",
    "rule_e_certifiable",
    "HandoffRow",
    "handoff_table",
    "optimal_handoff",
    "CQEOnlyCost",
    "cqe_only_cost",
    "RULES_B",
    "RuleBRow",
    "rule_b_table",
    "rule_b_bracket",
    "optimal_rule_b",
    "runtime_quantile",
]

LOOPS = ("measured", "classical_first_step", "free")
RULES = ("expected", "expected_oracle", "delta", "delta_oracle")
_E1_CHOICES = ("same_irrep", "any_irrep")
# runs_for_confidence: the float quotient ln(delta)/ln(1 - p) is accurate to a few ulps, so its
# ceiling is exact unless it lies within this relative distance of an integer; up to
# _EXACT_RUNS_LIMIT runs such cases are decided exactly (_runs_near_integer).
_NEAR_INTEGER_RTOL = 1e-12
_EXACT_RUNS_LIMIT = 2 ** 50
# Decimal digits of the first exact decision (doubled until it is resolved). An exact tie
# (1 - p)^n = delta between dyadic rationals with delta >= 2^-1074 needs n <= 1074.
_DECIMAL_DIGITS = 60
_MAX_TIE_RUNS = 1074


def _runs_near_integer(p: float, delta: float, n: int) -> int:
    """R_d(p) when ln(delta)/ln(1 - p) lies within _NEAR_INTEGER_RTOL (relative) of the integer n >= 1.

    R is n if (1 - p)^n <= delta and n + 1 otherwise. The sign of n ln(1 - p) - ln(delta) is
    evaluated in decimal arithmetic on the exact binary values of p and delta, from 60 digits
    and doubling until |gap| exceeds its rounding error (n + |ln delta|) 10^(1 - digits) by 100x;
    an unresolved gap is either an exact tie (1 - p)^n = delta, checked in rationals, or needs
    more digits.
    """
    dp, dd = decimal.Decimal(p), decimal.Decimal(delta)
    digits = _DECIMAL_DIGITS
    while True:
        with decimal.localcontext() as ctx:
            ctx.prec = digits
            log_d = dd.ln()
            gap = n * (1 - dp).ln() - log_d
            resolved = abs(gap) > (n + abs(log_d)) * decimal.Decimal(10) ** (3 - digits)
        if resolved:
            return n if gap < 0 else n + 1
        if n <= _MAX_TIE_RUNS and (1 - fractions.Fraction(p)) ** n == fractions.Fraction(delta):
            return n
        digits *= 2


def runs_for_confidence(p: float, delta: float = 0.01) -> int:
    """R_d(p) = min{R >= 1 : (1 - p)^R <= delta}, i.e. ceil(ln delta / ln(1 - p)); 1 if p >= 1 - delta.

    Exact for the binary values of p and delta up to R = 2^50. ln(1 - p) is log1p(-p) (1 - p
    itself rounds to 1 below p = 1.1e-16), and the float ceiling is used unless the quotient
    lies within 1e-12 (relative) of an integer n. There a float rule can fall one run short:
    on inputs within 3 ulps of a tie (1 - p)^n = delta it did for 8% (also when the ceiling
    is corrected until R ln(1 - p) <= ln delta < (R - 1) ln(1 - p) holds in floats), and
    never on 20000 random inputs (measured); (1 - p)^n <= delta is then decided in decimal
    and rational arithmetic (``_runs_near_integer``, about 60 us a call against 3 us). The
    case R = 1 is decided in rationals, so (p, delta) = (0.99, 0.01) needs 2 runs: in binary
    1 - 0.99 = 0.0100000000000000089 > 0.01. Above 2^50 runs R is the float ceiling (relative
    accuracy ~1e-15). Raises ValueError for p outside (0, 1], delta outside (0, 1), or p so
    small that R overflows a float (p < 2.6e-308 at delta = 0.01).
    """
    p, delta = float(p), float(delta)
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must lie in (0, 1), got {delta!r}")
    if not 0.0 < p <= 1.0:
        raise ValueError(f"p must lie in (0, 1], got {p!r}")
    if 1 - fractions.Fraction(p) <= fractions.Fraction(delta):
        return 1
    quotient = math.log(delta) / math.log1p(-p)
    if not math.isfinite(quotient):
        raise ValueError(f"p = {p!r} is too small: the number of runs overflows")
    n = round(quotient)
    if quotient < _EXACT_RUNS_LIMIT and abs(quotient - n) <= _NEAR_INTEGER_RTOL * quotient:
        return _runs_near_integer(p, delta, n)
    return math.ceil(quotient)


def overlap_lower_bound(energy: float, variance: float, e1_lower: float) -> float:
    """Eckart-Temple certificate p_lb = 1/(1 + sigma^2/(E1 - E)^2) for E < E1, else nan (module docstring).

    Args:
        energy: E_k of the state (Hartree).
        variance: sigma_k^2 = <H^2> - <H>^2 (Ha^2).
        e1_lower: a lower bound on the first excited energy of the state's sector (Hartree).
    """
    if variance < 0.0:
        raise ValueError(f"variance must be >= 0, got {variance!r}")
    gap = float(e1_lower) - float(energy)
    if not gap > 0.0:
        return math.nan
    return 1.0 / (1.0 + float(variance) / gap ** 2)


def rule_e_certifiable(energy: float, e1_lower: float, de_qpe: float) -> bool:
    """Rule E (accept the first outcome below E + 2 dE) is certified iff E1 - E > 3 dE."""
    return bool(float(e1_lower) - float(energy) > 3.0 * float(de_qpe))


@dataclass(frozen=True)
class HandoffRow:
    """Hand-off of the CQE iterate Psi_k to QPE (module docstring); costs in Toffolis.

    Attributes:
        k: the iterate.
        p: p_k = |<psi_0|Psi_k>|^2.
        e_err: E_k - E_0 (Hartree).
        c_prep: C(Psi_k).
        c_loop: C_loop(k) under the table's loop convention.
        runs_expected: 1/p_k (rule E).
        cost_expected: T_E(k).
        p_lb: certified overlap (nan if E_k >= E1).
        runs_delta, cost_delta: R_d(p_lb) and T_d(k); None without a positive p_lb.
        runs_delta_oracle: R_d(p_k), with the exact overlap (the oracle column, not certified).
        certifiable: rule E is certified (E1 - E_k > 3 dE).
        saving_vs_hf: C_QPE/p_0 - (C(Psi_k) + C_QPE)/p_k.
        p0_break_even: C_QPE / T_E(k).
        cost_delta_oracle: C_loop(k) + R_d(p_k) (C(Psi_k) + C_QPE).
        flags: "exact_first_step" (classical-first-step loop on an exact-line-search
            record, k >= 1) and "second_order_first_step" (measured loop on a classical-step
            record, k >= 1).
    """

    k: int
    p: float
    e_err: float
    c_prep: float
    c_loop: float
    runs_expected: float
    cost_expected: float
    p_lb: float
    runs_delta: int | None
    cost_delta: float | None
    runs_delta_oracle: int
    certifiable: bool
    saving_vs_hf: float
    p0_break_even: float
    cost_delta_oracle: float
    flags: tuple[str, ...] = ()


def _qpe_terms(qpe: QPECost | float, params: FTParameters) -> tuple[float, float]:
    """(C_QPE, dE) from a QPECost, or from a Toffoli count with dE = params.de_qpe."""
    if isinstance(qpe, QPECost):
        c, de = float(qpe.toffolis), float(qpe.de_qpe)
    else:
        c, de = float(qpe), float(params.de_qpe)
    if not (c > 0.0 and math.isfinite(c)):
        raise ValueError(f"C_QPE must be positive and finite, got {c!r}")
    return c, de


def _overlap(record: TrajectoryRecord, k: int) -> float:
    """p_k, with round-off above 1 read as 1; ValueError if p_k is not positive."""
    p = min(float(record.overlaps[k]), 1.0)
    if not p > 0.0:
        raise ValueError(f"overlap p_{k} = {p!r}: the iterate cannot be handed off")
    return p


def _covered_iterations(values: Sequence[float] | None, loop: str) -> int | None:
    """The last iterate k whose C_loop(k) the iteration costs determine (None: no limit)."""
    if loop not in LOOPS:
        raise ValueError(f"loop must be one of {LOOPS}, got {loop!r}")
    if loop == "free":
        return None
    n = 0 if values is None else len(values)
    if loop == "measured":
        if values is None:
            raise ValueError("the measured loop needs the per-iteration costs")
        return n
    return max(n, 1)                            # classical first step: C_iter(0) = 0 is known


def _loop_costs(values: Sequence[float] | None, loop: str, n: int) -> np.ndarray:
    """C_iter(j), j = 0..n-1, under the loop convention: 0 for the free loop and for j = 0 of the classical first step.

    Entries may be +inf (infeasible); nan and negative entries raise ValueError.
    """
    limit = _covered_iterations(values, loop)
    c = np.zeros(n)
    if limit is not None and n > limit:
        raise ValueError(f"iteration costs cover C_loop(k) up to k = {limit}, k = {n} was requested")
    first = 0 if loop == "measured" else 1
    if loop == "free" or n <= first:
        return c
    used = np.asarray(values, dtype=float).ravel()[first:n]
    if np.any(np.isnan(used)) or np.any(used < 0.0):
        raise ValueError("iteration costs must be >= 0 (+inf allowed), without nan")
    c[first:n] = used
    return c


def _row_flags(record: TrajectoryRecord, loop: str, k: int) -> tuple[str, ...]:
    """The flags of HandoffRow for iterate k of ``record`` under ``loop``."""
    flags = []
    if k >= 1 and loop == "classical_first_step" and record.first_step != "second_order":
        flags.append("exact_first_step")
    if k >= 1 and loop == "measured" and record.first_step == "second_order":
        flags.append("second_order_first_step")
    return tuple(flags)


def handoff_table(record: TrajectoryRecord, prep_cumulative: np.ndarray,
                  iteration_toffolis: Sequence[float] | None, qpe: QPECost | float, *,
                  loop: str = "classical_first_step", use_e1: str = "same_irrep",
                  params: FTParameters = FTParameters()) -> list[HandoffRow]:
    """F15 for every hand-off iterate k = 0..k_max (module docstring).

    Args:
        record: the exact trajectory (energies, variances, overlaps, E1); for the
            classical-first-step loop one with ``first_step="second_order"``.
        prep_cumulative: C(Psi_k), k = 0..; ``ft.stateprep.PrepCost.cumulative``.
        iteration_toffolis: C_iter(j) of the iteration at Psi_j, j = 0.. (+inf allowed);
            ignored by the "free" loop and C_iter(0) by the classical first step. None is
            allowed for those two loops (the classical first step then covers k <= 1).
        qpe: the QPE (``ft.qpe.QPECost``, whose ``de_qpe`` sets the certification) or C_QPE in
            Toffolis (then dE = ``params.de_qpe``).
        loop: "measured", "classical_first_step" or "free".
        use_e1: "same_irrep" (``record.e1``) or "any_irrep" (``record.e1_any``).
        params: ``delta_conf`` for rule d.

    k_max is the last iterate covered by the trajectory, by ``prep_cumulative`` and (except
    for the free loop) by the iteration costs, which C_loop(k) needs for j < k.
    """
    if use_e1 not in _E1_CHOICES:
        raise ValueError(f"use_e1 must be one of {_E1_CHOICES}, got {use_e1!r}")
    c_qpe, de = _qpe_terms(qpe, params)
    e1 = record.e1 if use_e1 == "same_irrep" else record.e1_any
    prep = np.asarray(prep_cumulative, dtype=float)
    k_max = min(len(record.energies), prep.size) - 1
    limit = _covered_iterations(iteration_toffolis, loop)
    if limit is not None:
        k_max = min(k_max, limit)
    if k_max < 0:
        raise ValueError("the trajectory and prep_cumulative must cover at least k = 0")
    c_loop = np.concatenate([[0.0], np.cumsum(_loop_costs(iteration_toffolis, loop, k_max))])
    p0 = _overlap(record, 0)
    rows = []
    for k in range(k_max + 1):
        p = _overlap(record, k)
        run = prep[k] + c_qpe
        cost_e = c_loop[k] + run / p
        p_lb = overlap_lower_bound(record.energies[k], record.variances[k], e1)
        r_d = runs_for_confidence(p_lb, params.delta_conf) if p_lb > 0.0 else None
        r_o = runs_for_confidence(p, params.delta_conf)
        rows.append(HandoffRow(
            k=k, p=p, e_err=float(record.energies[k] - record.e0), c_prep=float(prep[k]),
            c_loop=float(c_loop[k]), runs_expected=1.0 / p, cost_expected=float(cost_e), p_lb=p_lb,
            runs_delta=r_d, cost_delta=None if r_d is None else float(c_loop[k] + r_d * run),
            runs_delta_oracle=r_o, certifiable=rule_e_certifiable(record.energies[k], e1, de),
            saving_vs_hf=float(c_qpe / p0 - run / p), p0_break_even=float(c_qpe / cost_e),
            cost_delta_oracle=float(c_loop[k] + r_o * run), flags=_row_flags(record, loop, k)))
    return rows


def optimal_handoff(rows: Sequence[HandoffRow], rule: str = "expected") -> HandoffRow:
    """The row of lowest total cost under ``rule`` (module docstring).

    "expected": T_E over the rows where rule E is certified (k_opt of rule E);
    "expected_oracle": T_E over every row; "delta": the certified T_d (rows with a p_lb);
    "delta_oracle": T_d with the exact p_k. Ties go to the smaller k. Raises ValueError if
    no row qualifies.
    """
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
    key = {"expected": "cost_expected", "expected_oracle": "cost_expected", "delta": "cost_delta",
           "delta_oracle": "cost_delta_oracle"}[rule]
    ok = [r for r in rows if getattr(r, key) is not None and (rule != "expected" or r.certifiable)]
    if not ok:
        raise ValueError(f"no row qualifies under rule {rule!r} "
                         f"({'no certified row' if rule in ('expected', 'delta') else 'no rows'})")
    return min(ok, key=lambda r: (getattr(r, key), r.k))


@dataclass(frozen=True)
class CQEOnlyCost:
    """F17: the CQE energy without QPE (module docstring).

    Attributes:
        k_v: the first iterate with E_k - E_0 <= e_tol_energy (None if not reached).
        e_err: E_kv - E_0 (Hartree); the last iterate's error if k_v is None.
        shots: loop shots plus S_fin (inf if k_v is None).
        toffolis: C_loop(k_v) + S_fin (C(Psi_kv) + C_rot) (inf if k_v is None).
        ratio_to_hf_qpe: toffolis / (C_QPE / p_0).
        final_shots: S_fin = V_E / sigma_final^2.
        flags: the HandoffRow flags of iterate k_v.
    """

    k_v: int | None
    e_err: float
    shots: float
    toffolis: float
    ratio_to_hf_qpe: float
    final_shots: float
    flags: tuple[str, ...] = ()


def cqe_only_cost(record: TrajectoryRecord, prep_cumulative: np.ndarray,
                  iteration_toffolis: Sequence[float] | None, iteration_shots: Sequence[float] | None,
                  energy_variance: float, readout_toffolis: int, qpe: QPECost | float, *,
                  params: FTParameters = FTParameters(), loop: str = "classical_first_step") -> CQEOnlyCost:
    """F17 (module docstring): the measured loop to k_v plus a final energy at sigma_final.

    Args:
        record: the exact trajectory.
        prep_cumulative: C(Psi_k).
        iteration_toffolis, iteration_shots: C_iter(j) and the shots of iteration j, j < k_v
            (None as in :func:`handoff_table`).
        energy_variance: V_E per shot of the final energy (Ha^2).
        readout_toffolis: C_rot of the energy readout.
        qpe: the QPE of the HF hand-off it is compared with (QPECost or C_QPE).
        loop: the loop convention of :func:`handoff_table` (zeroes the shots too).
    """
    c_qpe, _ = _qpe_terms(qpe, params)
    if energy_variance < 0.0:
        raise ValueError(f"energy_variance must be >= 0, got {energy_variance!r}")
    s_fin = float(energy_variance) / params.sigma_final ** 2
    k_v = first_iterate_below(record, params.e_tol_energy)
    if k_v is None:
        return CQEOnlyCost(k_v=None, e_err=float(record.energies[-1] - record.e0), shots=math.inf,
                           toffolis=math.inf, ratio_to_hf_qpe=math.inf, final_shots=s_fin)
    prep = np.asarray(prep_cumulative, dtype=float)
    if prep.size <= k_v:
        raise ValueError(f"prep_cumulative must cover k_v = {k_v}")
    loop_tof = float(np.sum(_loop_costs(iteration_toffolis, loop, k_v)))
    loop_shots = float(np.sum(_loop_costs(iteration_shots, loop, k_v)))
    tof = loop_tof + s_fin * (prep[k_v] + float(readout_toffolis))
    return CQEOnlyCost(k_v=k_v, e_err=float(record.energies[k_v] - record.e0), shots=loop_shots + s_fin,
                       toffolis=tof, ratio_to_hf_qpe=tof / (c_qpe / _overlap(record, 0)),
                       final_shots=s_fin, flags=_row_flags(record, loop, k_v))


# ------------------------------------------------------------------ rule B (F40, F42)
RULES_B = ("bracketed", "bracketed_oracle")
# Flags of a rule-B row whose E1lb is the fallback E_1(H) - 1 mHa (no e1_lower given).
_FLAG_E1_FALLBACK = "e1lb_from_h"


@dataclass(frozen=True)
class RuleBRow:
    """Hand-off of Psi_k under rule B (module docstring); costs in Toffolis.

    Attributes:
        k, p, e_err, c_prep, c_loop: as :class:`HandoffRow` (C_loop(k) of the table's loop).
        form: "B_U" or "B_L" (``ft.acceptance.acceptance_bracket``).
        certified: termination certified (B_U within the window-model bracket; B_L by a U with
            U + dE <= theta).
        certificate: the certificate used (None: B_L without U).
        theta: the threshold (Hartree).
        n_runs: n_v of F40 (1 in the window model).
        cost: T_B(k), the certificate and the verification runs included (inf if no option
            terminates).
        cost_oracle_e: M3's oracle T_E(k) (``HandoffRow.cost_expected``).
        identification_error: eps_id of the accepted state from ``stats`` (None without).
        flags: the bracket's flags, M3's row flags, "not_terminating", "e1lb_from_h".
        n_runs_expected: the expected n_v of the option (``n_runs`` for a deterministic bound; for a
            dedicated certificate's B_U the expectation over the realized bound; None: ``n_runs``).
        certificate_cost: the expected certificate cost of the option (C_U, or C_U over the
            probability of an admissible bound where B_L does not terminate; None: the
            certificate's Toffolis, 0 without one).
        bracket_flags: the chosen bracket's own flags (``ft.acceptance.Bracket.flags``).
    """

    k: int
    p: float
    e_err: float
    c_prep: float
    c_loop: float
    form: str
    certified: bool
    certificate: "EnergyCertificate | None"
    theta: float
    n_runs: int
    cost: float
    cost_oracle_e: float
    identification_error: float | None = None
    flags: tuple[str, ...] = ()
    n_runs_expected: float | None = None
    certificate_cost: float | None = None
    bracket_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        from .acceptance import BRACKET_FORMS
        from .certify import EnergyCertificate

        k = self.k
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 0:
            raise ValueError(f"k must be an integer >= 0, got {k!r}")
        object.__setattr__(self, "k", int(k))
        p = float(self.p)
        if not (math.isfinite(p) and 0.0 < p <= 1.0):
            raise ValueError(f"p must lie in (0, 1], got {self.p!r}")
        if self.form not in BRACKET_FORMS:
            raise ValueError(f"form must be one of {BRACKET_FORMS}, got {self.form!r}")
        if not isinstance(self.certified, (bool, np.bool_)):
            raise ValueError(f"certified must be a bool, got {self.certified!r}")
        object.__setattr__(self, "certified", bool(self.certified))
        if self.certificate is not None and not isinstance(self.certificate, EnergyCertificate):
            raise ValueError("certificate must be an EnergyCertificate or None")
        if self.form == "B_U" and self.certificate is None:
            raise ValueError("a B_U row needs its certificate")
        n = self.n_runs
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
            raise ValueError(f"n_runs must be an integer >= 1, got {n!r}")
        object.__setattr__(self, "n_runs", int(n))
        for name in ("e_err", "theta"):
            val = float(getattr(self, name))
            if not math.isfinite(val):
                raise ValueError(f"{name} must be finite, got {val!r}")
            object.__setattr__(self, name, val)
        for name in ("c_prep", "c_loop", "cost", "cost_oracle_e"):
            val = float(getattr(self, name))
            if math.isnan(val) or val < 0.0:
                raise ValueError(f"{name} must be >= 0 (inf allowed), got {val!r}")
            object.__setattr__(self, name, val)
        if self.identification_error is not None:
            ide = float(self.identification_error)
            if not (math.isnan(ide) or ide >= 0.0):
                raise ValueError(f"identification_error must be >= 0 or nan, got {ide!r}")
            object.__setattr__(self, "identification_error", ide)
        flags = tuple(self.flags)
        if not all(isinstance(f, str) for f in flags):
            raise ValueError("flags must be strings")
        object.__setattr__(self, "flags", flags)
        object.__setattr__(self, "p", p)
        runs = float(self.n_runs) if self.n_runs_expected is None else float(self.n_runs_expected)
        if not (math.isfinite(runs) and runs >= 1.0):
            raise ValueError(f"n_runs_expected must be finite and >= 1, got {self.n_runs_expected!r}")
        object.__setattr__(self, "n_runs_expected", runs)
        c_u = (0.0 if self.certificate is None else float(self.certificate.toffolis)) \
            if self.certificate_cost is None else float(self.certificate_cost)
        if math.isnan(c_u) or c_u < 0.0:
            raise ValueError(f"certificate_cost must be >= 0, got {self.certificate_cost!r}")
        object.__setattr__(self, "certificate_cost", c_u)
        bflags = tuple(self.bracket_flags)
        if not all(isinstance(f, str) for f in bflags):
            raise ValueError("bracket_flags must be strings")
        object.__setattr__(self, "bracket_flags", bflags)


def rule_b_table(record: TrajectoryRecord, prep_cumulative, iteration_toffolis: Sequence[float] | None,
                 qpe: QPECost | float, *, certificates: "Sequence[EnergyCertificate]",
                 loop: str = "classical_first_step", e1_lower: float | None = None, grid: float | None = None,
                 stats: "Sequence[AcceptanceStats] | None" = None, params: FTParameters = FTParameters(),
                 e0_model: float | None = None) -> list[RuleBRow]:
    """T_B(k) under rule B for every row of :func:`handoff_table` (same coverage; module docstring).

    Args:
        record, prep_cumulative, iteration_toffolis, qpe, loop, params: as :func:`handoff_table`.
        certificates: candidate certificates (``ft.certify.energy_certificates``); oracle ones are
            ignored (the oracle bound is ``cost_oracle_e``).
        e1_lower: E1lb (F41); None takes E_1(H) - 1 mHa from the record (flag "e1lb_from_h").
        grid: g = 2 pi lambda sin(theta_0)/M of the hand-off QPE (2 dE sin(theta_0) at
            M = 2 n_iter; ``ft.acceptance.ModelSpectrum.grid``); None: M3's window model.
        stats: per-row ``ft.acceptance.AcceptanceStats`` at the rows' thresholds and n_v, whose
            identification errors are attached (one per row).
        e0_model: E_0' for B_L's termination check (None: ``record.e0``).

    The options are those of :func:`_rule_b_options` (each certificate's ``acceptance_bracket``, its
    B_U form where that bracket is B_L, and B_L without U); a dedicated certificate with ``sigma``
    is costed by the expectation over its realized bound (module docstring) wherever ``grid`` is
    given and its planned U is admissible (U <= E1lb - 2 dE).
    """
    from .acceptance import AcceptanceStats, expected_verification_runs
    from .certify import EnergyCertificate, e1_lower_bound

    rows = handoff_table(record, np.asarray(prep_cumulative, dtype=float), iteration_toffolis, qpe, loop=loop,
                         params=params)
    c_qpe, de = _qpe_terms(qpe, params)
    flags_e1: tuple[str, ...] = ()
    if e1_lower is None:
        e1 = e1_lower_bound(record.e1, model_spectrum=False)
        flags_e1 = (_FLAG_E1_FALLBACK,)
    else:
        e1 = float(e1_lower)
        if not math.isfinite(e1):
            raise ValueError(f"e1_lower must be finite, got {e1_lower!r}")
    e0 = float(record.e0) if e0_model is None else float(e0_model)
    if not math.isfinite(e0):
        raise ValueError(f"e0_model must be finite, got {e0_model!r}")
    certs = list(certificates)
    if not all(isinstance(c, EnergyCertificate) for c in certs):
        raise ValueError("certificates must be EnergyCertificate instances")
    certs = [c for c in certs if c.source != "oracle"]
    if stats is not None:
        stats = list(stats)
        if len(stats) != len(rows) or not all(isinstance(st, AcceptanceStats) for st in stats):
            raise ValueError(f"stats must hold one AcceptanceStats per row ({len(rows)})")
    k_max = rows[-1].k
    c_loop = np.array([r.c_loop for r in rows])
    brackets = _rule_b_options(certs, e1, de, grid)
    b_l = brackets[0][1]
    b_l_terminates = e0 + de <= b_l.theta
    # the terminating options with their loop iterate j_U, expected certificate cost and expected runs
    priced = []
    for cert, br in brackets:
        terminates = br.form == "B_U" or e0 + de <= br.theta
        j_u = 0 if cert is None else cert.loop_iterate
        if not terminates or j_u > k_max:
            continue
        c_u, runs = (0.0 if cert is None else float(cert.toffolis)), float(br.n_runs)
        if (cert is not None and cert.sigma is not None and br.form == "B_U" and grid is not None
                and cert.upper <= e1 - 2.0 * de):
            ex = expected_verification_runs(cert.upper, cert.sigma, e1, de, grid=grid,
                                            fallback_runs=b_l.n_runs if b_l_terminates else None)
            runs = ex.runs
            if not b_l_terminates:
                c_u /= ex.admissible
        priced.append((cert, br, j_u, c_u, runs))
    out = []
    for i, r in enumerate(rows):
        run = (r.c_prep + c_qpe) / r.p
        best = None
        for cert, br, j_u, c_u, runs in priced:
            cost = float(c_loop[max(r.k, j_u)] + c_u + run + (runs - 1.0) * c_qpe)
            key = (cost, not br.certified, br.form != "B_U", cert is not None,
                   math.inf if cert is None else cert.upper)
            if best is None or key < best[0]:
                best = (key, cert, br, c_u, runs)
        if best is None:
            (cert, br), c_u, runs = brackets[0], 0.0, float(brackets[0][1].n_runs)
            cost, extra = math.inf, ("not_terminating",)
        else:
            _, cert, br, c_u, runs = best
            cost, extra = best[0][0], ()
        flags = tuple(dict.fromkeys(br.flags + r.flags + extra + flags_e1))
        out.append(RuleBRow(k=r.k, p=r.p, e_err=r.e_err, c_prep=r.c_prep, c_loop=r.c_loop, form=br.form,
                            certified=br.certified and best is not None, certificate=cert, theta=br.theta,
                            n_runs=br.n_runs, cost=cost, cost_oracle_e=r.cost_expected,
                            identification_error=None if stats is None else stats[i].identification_error,
                            flags=flags, n_runs_expected=runs, certificate_cost=c_u, bracket_flags=br.flags))
    return out


def _b_u_bracket(u: float, e1: float, de: float, grid: float, d_l: int = 5) -> "Bracket | None":
    """B_U at theta = U + dE with its own n_v (F40), None if more than 64 runs are needed or U + dE >= E1lb."""
    from .acceptance import Bracket, verification_runs

    if not u + de < e1:
        return None
    d_u = (e1 - u - de) / grid
    try:
        n_u = verification_runs(d_u)
    except ValueError:
        return None
    return Bracket(theta=u + de, form="B_U", lower=u + de, upper=e1 - d_l * grid, certified=u <= e1 - 2.0 * de,
                   distance=d_u, n_runs=n_u)


def _rule_b_options(certs, e1: float, de: float, grid: float | None) -> list:
    """[(certificate or None, Bracket)]: B_L without U first, then per certificate its ``acceptance_bracket`` and,
    where that is B_L (it needs fewer runs than B_U), its B_U form as a second option (grid given)."""
    from .acceptance import acceptance_bracket

    out = [(None, acceptance_bracket(None, e1, de, grid=grid))]
    for c in certs:
        br = acceptance_bracket(c.upper, e1, de, grid=grid)
        out.append((c, br))
        if grid is not None and br.form == "B_L":
            alt = _b_u_bracket(c.upper, e1, de, grid)
            if alt is not None:
                out.append((c, alt))
    return out


def rule_b_bracket(u_upper: float | None, e1_lower: float, de: float, *, grid: float | None,
                   e0_model: float) -> "Bracket | None":
    """Rule B's form for a state with a classical (deterministic, free) U or none: :func:`rule_b_table`'s choice.

    The fewest runs among the terminating options of :func:`_rule_b_options` (B_U terminates; B_L iff
    E_0' + dE <= theta), ties to certified termination, then B_U, then no certificate; None if none
    terminates. ``acceptance_bracket`` stays the realizable choice without E_0'; this adds the
    termination check of the model.
    """
    from .certify import EnergyCertificate

    e1 = float(e1_lower)
    e0 = float(e0_model)
    de = float(de)
    if not (math.isfinite(e1) and math.isfinite(e0)):
        raise ValueError(f"e1_lower and e0_model must be finite, got {e1_lower!r}, {e0_model!r}")
    certs = [] if u_upper is None else [EnergyCertificate("classical", float(u_upper), None, 0.0, 0.0, 0.0, "u")]
    best = None
    for cert, br in _rule_b_options(certs, e1, de, grid):
        if not (br.form == "B_U" or e0 + de <= br.theta):
            continue
        key = (br.n_runs, not br.certified, br.form != "B_U", cert is not None)
        if best is None or key < best[0]:
            best = (key, br)
    return None if best is None else best[1]


def optimal_rule_b(rows: Sequence[RuleBRow], rule: str = "bracketed") -> RuleBRow:
    """The row of lowest cost: "bracketed" T_B over the terminating rows (flagged rows included),
    "bracketed_oracle" M3's oracle T_E; ties to the smaller k. ValueError if no row qualifies."""
    if rule not in RULES_B:
        raise ValueError(f"rule must be one of {RULES_B}, got {rule!r}")
    key = "cost" if rule == "bracketed" else "cost_oracle_e"
    ok = [r for r in rows if math.isfinite(getattr(r, key))]
    if not ok:
        raise ValueError(f"no row qualifies under rule {rule!r}")
    return min(ok, key=lambda r: (getattr(r, key), r.k))


def runtime_quantile(q: float, delta: float) -> int:
    """Attempts that rule B needs to finish with probability >= 1 - delta: R_delta(q) = runs_for_confidence(q, delta).

    q is the per-attempt acceptance probability (>= p_0 K_0^{n_v}); P(R > n) = (1 - q)^n.
    """
    return runs_for_confidence(q, delta)

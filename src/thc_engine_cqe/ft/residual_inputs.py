"""Inputs of the residual-cost routes from the M2 measure layer (F2a, F10-F13).

:func:`build_problem` assembles a :class:`~thc_engine_cqe.ft.residual.ResidualProblem`
for one state from the pieces below. The per-shot variances and the normalizations are
computed once per system and reused for every iterate [design]; the
exact residual and the aux-route bias are computed per state. States are real CI vectors.

Element sets (:func:`allowed_elements`; F2a)
--------------------------------------------
The rows of ``targets.element_set`` whose four orbital irreps multiply to the totally
symmetric irrep (``symmetry.element_mask``). The others vanish exactly at every state of
the HF irrep, so they are not measured: their variance and their K^(3/2) share of the
AE cost are saved, and ||A||_m on the allowed set is the full-tensor norm (tested).
Without irreps every element is kept, with a warning.

Normalizations (:func:`element_normalizations`)
-----------------------------------------------
lambda_Q,e and alpha_X,e are the Pauli 1-norms (identity excluded, ``PauliSum.norm1(False)``)
of Q_e = (Gamma_e - Gamma_e^dagger)/(2i) (``targets.aux_targets``) and of
X_e = i [Q_e, H] (``targets.direct_targets``), and L_X,e the number of non-identity
Pauli terms of X_e. They are the LCU normalizations of the AE routes b1-b3 of
``ft.residual``. Both operators are traceless (Gamma_e is a real matrix in the
Jordan-Wigner basis, so tr Q_e = Im tr Gamma_e = 0, and X_e is a commutator), so
``targets`` gives them no identity term and the exclusion removes nothing [derived;
tested]. A Pauli 1-norm of a fermionic operator does not depend on the
Jordan-Wigner order, because reordering the modes only changes the signs of its
Majorana coefficients [derived]. Every spin-orbital Q_e has lambda_Q = 1/2: with four
distinct modes Gamma_e has 16 Pauli terms of modulus 1/16, Gamma_e - Gamma_e^dagger
keeps the 8 anti-Hermitian ones, doubled, and 1/(2i) halves them again (a repeated mode
gives 4 terms of modulus 1/8) [derived; tested on every H4 element].
Observed (spin-free, allowed set, means): lambda_Q = 1.5 and alpha_X = 4.79 Ha with
L_X = 698 (H4, 28 elements); lambda_Q = 1.67 and alpha_X = 13.0 Ha with L_X = 4995
(H6 at 1.0 A, 153 elements). The 1-norm is loose: ||X_e|| on the (2, 2) sector is
0.16-0.50 alpha_X (median 0.25) for H4. Dense Pauli-LCU block encodings of H2 with these
normalizations reproduce A_e through the Hadamard tests of b1, b2 and b3 to 1e-10
(tests/test_ft_residual_inputs.py).

Per-shot variances (:func:`element_variances`; F10) [M2]
-------------------------------------------------------
* v_direct: ``routes.direct_residual_variance`` under matchgate shadows, on the weighted
  targets sqrt(m_e) X_e, so the scheme's total is V_m = sum_e m_e v_e, the numerator of
  the direct route's shot count, and its standard error is that of V_m, correlations
  between the elements included. Exact pair sums are used when the pair count
  sum_e L_e (L_e + 1) / 2 (the count of the schemes' ``method="auto"``) is at most 5e7,
  Rao-Blackwellized sampling of ``params.n_settings`` matchings (seed ``params.seed``)
  above. On the allowed spin-free sets: H4 7.6e6 pairs and N2 2.8e7 (exact), LiH 3.5e8 and
  H6 2.2e9 (RB). The design specification had exact sums for N <= 4 only; that would give
  N2 a seed-dependent estimate (3730 +- 50 at seed 7, against the exact 3751.1) to save 2 s,
  so the pair count decides instead (changed from the design specification;
  docs/m3-results.md section 11; tests/test_ft_residual_inputs.py::test_n2_variances_use_exact_pair_sums).
* w_aux = delta^2 x ``routes.aux_residual_variance(delta=0.1, "exact", ...,
  "matchgate_shadow")``: the aux-route variance at delta is w_aux / delta^2 to O(delta^2)
  (``measure/routes.py``, "Shot cost of the routes").

Observed at SD iterate 1 (spin-free, allowed set, one thread), V_m and W_m = sum_e m_e w_e:
H4 1216.7 Ha^2 and 816.2 (exact, 1.3 s); N2 3751.1 Ha^2 and 2117.7 (exact, 3.5 s); LiH
37783 +- 396 Ha^2 and 7037 (RB, 5000 settings, seed 7, 1.8 s); H6 at 1.0 A 25126 +- 113 Ha^2
and 11173 (RB, 2.9 s), against the exact 25288.5 (1.1e9 commuting pairs, 272 s).
Computed once per system: the exact V_m at HF and SD iterates 1, 3 and 5 is 1220.2 /
1216.7 / 1218.5 / 1218.8 for H4 and 3761.1 / 3751.1 / 3750.7 / 3750.7 for N2, and v_e
moves by at most 1.9% (H4) and 2.5% (N2) per element from iterate 1 to HF; a test checks
H4 iterates 1 and 4 against 3%.

Subset totals (:func:`variance_totals_subset`; F24 (ii)) [estimate]
------------------------------------------------------------------
Where the per-element variances of every allowed element are too slow (N = 10), V_m and
W_m are estimated from a uniform random subset S of K_s elements (``numpy`` default_rng
with ``seed``, without replacement): V^_m = (K/K_s) sum_{e in S} m_e v_e, with the
sampling standard error (K/K_s) sqrt(K_s) sd(m_e v_e) sqrt(1 - K_s/K) (finite population,
sd with ddof = 1), and the same for W_m. The routes receive the uniform per-element values
v_e = V^_m / sum_e m_e and w_e = W^_m / sum_e m_e, which reproduce the totals and hence
the direct_mg and aux_mg costs exactly; the AE routes use alpha_X, not v_e. The estimate
is unbiased over subsets (exactly so for exact pair sums; tests/test_ft_scaling.py). The
variances are usually taken at the HF determinant (F24 (i)).

Biases
------
* Aux route (:func:`aux_bias_curve`): ||routes.aux_residual(delta, "exact") -
  routes.exact_residual||_m, the exact O(delta^2) bias (exp(-i delta H) applied by
  ``fci_ops.expm_apply``) on a delta grid. :class:`AuxBiasCurve` interpolates it
  log-log for the routes. Observed relative to ||A||_m at H4 SD iterate 2: 0.0034 at
  delta = 0.1, 0.082 at 0.5, 0.29 at 1.0.
* Factorization (:func:`factorization_bias`): ||A[H'](Psi) - A[H](Psi)||_m for a model
  H' = ``active.with_eri(eri_model)``. A is linear in the Hamiltonian, so this is
  ||A[V](Psi)||_m with V = H' - H: linear in V, and at the ground state of H the M1
  floor ||A[H'](Psi_0)|| (``diagnostics.thc_floors``, "Exact relations").

Energy variances (:func:`energy_variances`; F13)
------------------------------------------------
The per-shot variance of DF basis-rotation grouping (``schemes.df_variance``) in the
four splits of M2 (:data:`DF_VARIANTS`: leaves from pivoted Cholesky or from the
supermatrix eigendecomposition, each unshifted or shifted to E_pq - delta_pq), and the
lowest. F13 picks the variant at HF and keeps it along the trajectory. Observed
(cholesky / eigh / cholesky_shift / eigh_shift, Ha^2), at HF and at SD iterate 1:

    H4         3.153 / 2.093 / 0.487 / 0.461 (eigh_shift)   2.509 / 1.677 / 0.631 / 0.610
    H6 1.0 A   6.558 / 4.168 / 0.992 / 1.022 (chol._shift)  5.384 / 3.403 / 1.230 / 1.214
    N2         0.573 / 0.657 / 0.573 / 0.657 (chol._shift)  0.993 / 0.918 / 0.827 / 0.906
    LiH        2.216 / 1.559 / 0.619 / 0.962 (chol._shift)  2.192 / 1.564 / 0.672 / 1.006

The HF choice is not always the lowest later (H6: eigh_shift is 0.7-3.2% lower along the
SD and L-BFGS iterates 1-6). For N2 the two Cholesky splits tie exactly at HF; a tie goes
to the shifted split, which is 14-20% lower along those iterates, where the dict order of
the splits would have kept the unshifted one.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..fci_ops import FCISpace
from ..integrals import ActiveSpace
from ..measure import routes, schemes, targets
from ..measure.pauli import PauliSum, QubitMapping
from . import symmetry
from .primitives import FTParameters
from .residual import ResidualProblem, _as_real, _real

__all__ = [
    "DF_VARIANTS",
    "AuxBiasCurve",
    "allowed_elements",
    "element_normalizations",
    "element_variances",
    "variance_totals_subset",
    "aux_bias_curve",
    "factorization_bias",
    "energy_variances",
    "build_problem",
]

#: The four DF splits of M2 (benchmarks/run_measurement.py): name -> (double_factorize options, shift).
DF_VARIANTS: dict[str, tuple[dict, bool]] = {
    "cholesky": ({}, False),
    "eigh": ({"first": "eigh"}, False),
    "cholesky_shift": ({}, True),
    "eigh_shift": ({"first": "eigh"}, True),
}
# energy_variances: variances within this relative tolerance of the lowest tie; ties go to
# the first split in _TIE_ORDER (shifted first).
_TIE_RTOL = 1e-9
_TIE_ORDER = ("cholesky_shift", "eigh_shift", "cholesky", "eigh")

# delta at which the aux-route variance is evaluated (w_aux = delta^2 v_aux) [design].
_W_DELTA = 0.1
# Direct-route variances by exact matchgate pair sums up to this many term pairs
# (sum_e L_e (L_e + 1) / 2, the count of the schemes' method="auto"), Rao-Blackwellized above.
_EXACT_MAX_PAIRS = 5e7
# Floor of log-log interpolation (an exactly zero bias interpolates to ~0).
_TINY = np.finfo(float).tiny


# ------------------------------------------------------------------------- helpers
def _elements(active: ActiveSpace, elements, kind: str) -> np.ndarray:
    el = np.asarray(elements)
    if el.size == 0:
        raise ValueError("need at least one element")
    targets.element_multiplicities(el, kind, active.norb)      # validates shape, range and orbits
    return el


def _norm_m(x: np.ndarray, m: np.ndarray) -> float:
    return math.sqrt(float(np.sum(m * np.asarray(x, dtype=float) ** 2)))


def _non_identity_terms(op: PauliSum) -> int:
    return int(np.count_nonzero((op.xs | op.zs) != 0))


@dataclass(frozen=True, eq=False)
class AuxBiasCurve:
    """delta -> ||b(delta)||_m (Hartree), log-log interpolation of exact values on a delta grid.

    Between grid points log b is linear in log delta; outside the grid the first or last
    segment is extended (a power law); a one-point grid is extended with the leading
    order b proportional to delta^2 [derived: ``measure/targets.py``]. Zero values are
    floored at the smallest positive float before taking logs.

    Attributes:
        deltas: (n,) increasing evolution times (1/Hartree).
        values: (n,) ||b(delta)||_m at those times (Hartree).
    """

    deltas: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        d = _as_real("deltas", self.deltas).ravel()
        v = _as_real("values", self.values).ravel()
        if d.size == 0 or d.shape != v.shape:
            raise ValueError(f"deltas and values must be non-empty and of one shape, got {d.shape} and {v.shape}")
        if not (np.all(np.isfinite(d)) and np.all(d > 0) and np.all(np.isfinite(v)) and np.all(v >= 0)):
            raise ValueError("deltas must be positive and values non-negative, all finite")
        order = np.argsort(d)
        d, v = d[order], v[order]
        if np.any(np.diff(d) == 0):
            raise ValueError("deltas must be distinct")
        for a in (d, v):
            a.setflags(write=False)
        object.__setattr__(self, "deltas", d)
        object.__setattr__(self, "values", v)

    def __call__(self, delta: float) -> float:
        delta = _real("delta", delta, positive=True)
        x, y = np.log(self.deltas), np.log(np.maximum(self.values, _TINY))
        t = math.log(delta)
        if x.size == 1:
            return float(math.exp(y[0] + 2.0 * (t - x[0])))
        if t < x[0]:
            seg = 0
        elif t > x[-1]:
            seg = x.size - 2
        else:
            return float(math.exp(np.interp(t, x, y)))
        slope = (y[seg + 1] - y[seg]) / (x[seg + 1] - x[seg])
        return float(math.exp(y[seg] + slope * (t - x[seg])))


# ------------------------------------------------------------------------- elements
def allowed_elements(active: ActiveSpace, kind: str = "spin_free",
                     irreps: np.ndarray | Sequence[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Symmetry-allowed rows of ``targets.element_set(N, kind)`` (F2a).

    Returns (elements, mask): the allowed rows and the boolean mask over the full
    element set. ``irreps`` are the orbital irrep ids (``symmetry.orbital_irreps``);
    None keeps every element and warns that the forbidden ones (exactly zero at
    symmetric states) are then measured too.
    """
    el = targets.element_set(active.norb, kind)
    if irreps is None:
        warnings.warn("no orbital irreps given: every element is kept, including the point-group-forbidden "
                      "ones, which vanish exactly and cost shots (pass irreps=symmetry.orbital_irreps(active))",
                      UserWarning, stacklevel=2)
        mask = np.ones(el.shape[0], dtype=bool)
    else:
        mask = symmetry.element_mask(irreps, el, kind)
    return el[mask], mask


def element_normalizations(active: ActiveSpace, elements: np.ndarray, kind: str,
                           mapping: QubitMapping | None = None) -> dict:
    """LCU normalizations of the AE routes: {"lam_q", "alpha_x", "n_terms_x", "order"}.

    ``lam_q`` (K,, dimensionless) and ``alpha_x`` (K,, Hartree) are the Pauli 1-norms without
    the identity term of Q_e (``targets.aux_targets``) and X_e (``targets.direct_targets``),
    ``n_terms_x`` (K,) the numbers of non-identity Pauli terms of X_e; ``order`` is the qubit
    order of ``mapping`` (default blocked; the norms do not depend on it).
    """
    el = _elements(active, elements, kind)
    mapping = QubitMapping(active.norb) if mapping is None else mapping
    qs = targets.aux_targets(el, mapping, kind)
    xs = targets.direct_targets(el, active, mapping, kind)
    return {"lam_q": np.array([q.norm1(False) for q in qs]),
            "alpha_x": np.array([x.norm1(False) for x in xs]),
            "n_terms_x": np.array([_non_identity_terms(x) for x in xs], dtype=np.int64),
            "order": mapping.order}


def element_variances(active: ActiveSpace, ci: np.ndarray, elements: np.ndarray, kind: str, *,
                      params: FTParameters = FTParameters()) -> dict:
    """Per-shot variances of the two residual routes under matchgate shadows at the state ``ci``.

    Returns {"v_direct", "stderr", "variance_m", "stderr_m", "w_aux", "method", "n_settings",
    "seed", "delta_aux"}:

    * ``v_direct`` (K,): the direct-route variances (Ha^2), and ``stderr`` (K,) their
      standard errors;
    * ``variance_m``: V_m = sum_e m_e v_e (Ha^2), the numerator of the direct route's shot
      count, and ``stderr_m`` its standard error, correlations between the elements
      included;
    * ``method``: "exact" (pair sums; standard errors 0) when the pair count
      sum_e L_e (L_e + 1) / 2 is at most 5e7, else "rb" over ``params.n_settings``
      matchings with ``params.seed`` (``n_settings`` the settings used and ``seed``; both
      None for exact sums);
    * ``w_aux`` (K,): delta^2 times the aux-route variances at delta = ``delta_aux`` = 0.1
      (dimensionless).

    The direct route is evaluated on the weighted targets sqrt(m_e) X_e, so the scheme's
    total is V_m and its standard error that of V_m (from the per-setting totals); the
    per-element values are divided by m_e.
    """
    el = _elements(active, elements, kind)
    m = targets.element_multiplicities(el, kind, active.norb).astype(float)
    mapping = QubitMapping(active.norb)
    weighted = [x * math.sqrt(w) for x, w in zip(targets.direct_targets(el, active, mapping, kind), m)]
    dv = routes.direct_residual_variance(active, ci, el, kind, "matchgate_shadow", mapping=mapping,
                                         targets=weighted, method="auto", max_pairs=_EXACT_MAX_PAIRS,
                                         n_settings=params.n_settings, seed=params.seed)
    av = routes.aux_residual_variance(active, ci, _W_DELTA, "exact", el, kind, "matchgate_shadow")
    rb = dv.method == "rb"
    return {"v_direct": np.asarray(dv.per_target, dtype=float) / m,
            "stderr": np.asarray(dv.per_target_stderr, dtype=float) / m if rb else np.zeros(m.size),
            "variance_m": float(dv.total), "stderr_m": float(dv.stderr) if rb else 0.0,
            "w_aux": _W_DELTA ** 2 * np.asarray(av.per_target, dtype=float),
            "method": dv.method, "n_settings": int(dv.n_samples) if rb else None,
            "seed": params.seed if rb else None, "delta_aux": _W_DELTA}


def variance_totals_subset(active: ActiveSpace, ci: np.ndarray, elements: np.ndarray, kind: str, *,
                           k_subset: int = 100, seed: int = 11,
                           params: FTParameters = FTParameters()) -> dict:
    """F24 (ii): V_m and W_m from the per-shot variances of a uniform random subset of ``elements``.

    The subset is ``np.random.default_rng(seed).choice(K, min(k_subset, K), replace=False)``
    (sorted); its variances come from :func:`element_variances` at ``ci`` (with ``params``).
    Returns the format of :func:`element_variances`, with uniform per-element values, so the
    dict serves as ``variances`` of :func:`build_problem` (module docstring):

    * ``v_direct`` (K,) = V^_m / sum_e m_e and ``w_aux`` (K,) = W^_m / sum_e m_e;
    * ``variance_m`` = V^_m = (K/K_s) sum_{e in S} m_e v_e (Ha^2) and ``stderr_m`` its sampling
      standard error (K/K_s) sqrt(K_s) sd(m_e v_e) sqrt(1 - K_s/K) (0 for K_s = K);
      ``stderr`` (K,) = stderr_m / sum_e m_e;
    * ``w_aux_m`` = W^_m and ``w_stderr_m`` its sampling standard error;
    * ``stderr_m_estimator``: (K/K_s) times the subset's own ``stderr_m`` (Rao-Blackwellized
      sampling of its variances; 0 for exact pair sums), not included in ``stderr_m``;
    * ``method`` = "subset" and ``subset_method`` the subset's "exact" or "rb";
      ``n_settings``, ``seed`` and ``delta_aux`` as in :func:`element_variances`;
    * ``subset`` (K_s,) the element indices, ``k_subset``, ``subset_seed`` and ``n_elements``.
    """
    el = _elements(active, elements, kind)
    k = el.shape[0]
    k_s = int(k_subset)
    if k_s != k_subset or k_s < 2:
        raise ValueError(f"k_subset must be an integer >= 2 (a standard deviation), got {k_subset!r}")
    k_s = min(k_s, k)
    m = targets.element_multiplicities(el, kind, active.norb).astype(float)
    sub = np.sort(np.random.default_rng(seed).choice(k, size=k_s, replace=False))
    vs = element_variances(active, ci, el[sub], kind, params=params)
    scale, fpc = k / k_s, math.sqrt(max(0.0, 1.0 - k_s / k))
    x_v = m[sub] * vs["v_direct"]
    x_w = m[sub] * vs["w_aux"]
    vm, wm = scale * float(np.sum(x_v)), scale * float(np.sum(x_w))
    if k_s == k:                    # the full set: no sampling error
        se_v = se_w = 0.0
    else:
        se_v = scale * math.sqrt(k_s) * float(np.std(x_v, ddof=1)) * fpc
        se_w = scale * math.sqrt(k_s) * float(np.std(x_w, ddof=1)) * fpc
    m_total = float(np.sum(m))
    return {"v_direct": np.full(k, vm / m_total), "stderr": np.full(k, se_v / m_total),
            "variance_m": vm, "stderr_m": se_v, "w_aux": np.full(k, wm / m_total), "w_aux_m": wm,
            "w_stderr_m": se_w, "stderr_m_estimator": scale * float(vs["stderr_m"]),
            "method": "subset", "subset_method": vs["method"], "n_settings": vs["n_settings"],
            "seed": vs["seed"], "delta_aux": vs["delta_aux"], "subset": sub, "k_subset": k_s,
            "subset_seed": seed, "n_elements": k}


# ------------------------------------------------------------------------- biases
def _aux_bias_values(active: ActiveSpace, ci: np.ndarray, el: np.ndarray, kind: str, deltas: Sequence[float],
                     a: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.array([_norm_m(routes.aux_residual(active, ci, d, "exact", el, kind) - a, m) for d in deltas])


def aux_bias_curve(active: ActiveSpace, ci: np.ndarray, elements: np.ndarray, kind: str,
                   deltas: Sequence[float]) -> np.ndarray:
    """||routes.aux_residual(delta, "exact") - routes.exact_residual||_m per delta (Hartree).

    The exact bias of the aux route on exp(-i delta H) Psi, weighted by the multiplicities;
    ``deltas`` in 1/Hartree.
    """
    el = _elements(active, elements, kind)
    m = targets.element_multiplicities(el, kind, active.norb).astype(float)
    a = routes.exact_residual(active, ci, el, kind)
    return _aux_bias_values(active, ci, el, kind, np.atleast_1d(np.asarray(deltas, dtype=float)), a, m)


def factorization_bias(active: ActiveSpace, active_model: ActiveSpace, ci: np.ndarray, elements: np.ndarray,
                       kind: str) -> float:
    """||A[H'](Psi) - A[H](Psi)||_m (Hartree) for the model ``active_model`` (e.g. ``active.with_eri(eri_DF)``)."""
    if active_model.norb != active.norb or active_model.nelec != active.nelec:
        raise ValueError("active_model must have the orbitals and electrons of active")
    el = _elements(active, elements, kind)
    m = targets.element_multiplicities(el, kind, active.norb).astype(float)
    diff = routes.exact_residual(active_model, ci, el, kind) - routes.exact_residual(active, ci, el, kind)
    return _norm_m(diff, m)


# ------------------------------------------------------------------------- energies
def energy_variances(active: ActiveSpace, ci: np.ndarray) -> dict:
    """Per-shot energy variances (Ha^2) of the four DF splits of :data:`DF_VARIANTS`, and the lowest.

    Returns {"cholesky", "eigh", "cholesky_shift", "eigh_shift": ``schemes.df_variance(...).total``,
    "best": the name of the lowest, "variance": its value}. Variances within 1e-9 relative
    of the lowest tie, and a tie goes to a shifted split (cholesky_shift, then eigh_shift):
    at HF the shift can change nothing (N2 CAS(6,6): every totally symmetric leaf is
    diagonal, HF is its eigenstate, and cholesky and cholesky_shift agree to the last bit),
    while away from HF the shifted split is the lower one (module docstring).
    """
    space = FCISpace(active.norb, active.nelec)
    c = np.asarray(ci).reshape(space.shape)
    c = c / np.linalg.norm(c)
    out: dict = {name: float(schemes.df_variance(active, c, shift=shift, **opts).total)
                 for name, (opts, shift) in DF_VARIANTS.items()}
    lowest = min(out.values())
    best = next(name for name in _TIE_ORDER if out[name] <= lowest * (1.0 + _TIE_RTOL))
    out.update(best=best, variance=out[best])
    return out


# ------------------------------------------------------------------------- problem
def build_problem(active: ActiveSpace, ci: np.ndarray, elements: np.ndarray, kind: str, *, variances: dict,
                  normalizations: dict, deltas: Sequence[float] | None = None) -> ResidualProblem:
    """The :class:`~thc_engine_cqe.ft.residual.ResidualProblem` of the state ``ci``.

    A comes from ``routes.exact_residual``; ``variances`` (:func:`element_variances`) and
    ``normalizations`` (:func:`element_normalizations`) must belong to the same
    ``elements`` and ``kind``; ``aux_bias`` is the :class:`AuxBiasCurve` of
    :func:`aux_bias_curve` on ``deltas`` (default ``FTParameters().aux_deltas``).
    """
    el = _elements(active, elements, kind)
    m = targets.element_multiplicities(el, kind, active.norb).astype(float)
    a = routes.exact_residual(active, ci, el, kind)
    deltas = FTParameters().aux_deltas if deltas is None else tuple(float(d) for d in deltas)
    curve = AuxBiasCurve(np.array(deltas), _aux_bias_values(active, ci, el, kind, deltas, a, m))
    return ResidualProblem(kind=kind, elements=el, m=m, A=a, v_direct=variances["v_direct"],
                           w_aux=variances["w_aux"], lam_q=normalizations["lam_q"],
                           alpha_x=normalizations["alpha_x"], n_terms_x=normalizations["n_terms_x"],
                           aux_bias=curve, norb=active.norb)

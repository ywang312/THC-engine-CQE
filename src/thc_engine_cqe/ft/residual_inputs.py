"""Inputs of the residual-cost routes from the M2 measure layer (F2a, F10-F13).

:func:`build_problem` assembles a :class:`~thc_engine_cqe.ft.residual.ResidualProblem`
for one state from the pieces below. The per-shot variances and the normalizations are
computed once per system and reused for every iterate (specification section 6.7); the
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
``ft.residual``. A Pauli 1-norm of a fermionic operator does not depend on the
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
(tests/test_ft_residual_inputs.py, slow).

Per-shot variances (:func:`element_variances`; F10) [M2]
-------------------------------------------------------
* v_direct: ``routes.direct_residual_variance`` under matchgate shadows, by exact pair
  sums for N <= 4 and by Rao-Blackwellized sampling of ``params.n_settings`` matchings
  (seed ``params.seed``) otherwise; the standard errors are returned.
* w_aux = delta^2 x ``routes.aux_residual_variance(delta=0.1, "exact", ...,
  "matchgate_shadow")``: the aux-route variance at delta is w_aux / delta^2 to O(delta^2)
  (``measure/routes.py``, "Shot cost of the routes").

Observed at SD iterate 1 (spin-free, allowed set, one thread): V_m = sum_e m_e v_e =
1216.7 Ha^2 and W_m = sum_e m_e w_e = 816.2 for H4 (exact, 2.1 s); V_m = 25126 Ha^2
(RB, 5000 settings, standard error 33) and W_m = 11173 for H6 at 1.0 A (2.5 s). Memo 2
had 1219 / 816 and 25138 / 11174 at SD iterate 5, so the variances are
state-independent to about 0.2% there; a test checks H4 iterates 1 and 4 against 3%.

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
lowest. F13 picks the variant at HF and keeps it along the trajectory. Observed at H4
SD iterate 1: 2.51 / 1.68 / 0.631 / 0.610 Ha^2 (eigh_shift lowest); H6 at 1.0 A:
5.38 / 3.40 / 1.230 / 1.214 Ha^2.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from ..fci_ops import FCISpace
from ..integrals import ActiveSpace
from ..measure import routes, schemes, targets
from ..measure.pauli import PauliSum, QubitMapping
from . import symmetry
from .primitives import FTParameters
from .residual import ResidualProblem

__all__ = [
    "DF_VARIANTS",
    "AuxBiasCurve",
    "allowed_elements",
    "element_normalizations",
    "element_variances",
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

# delta at which the aux-route variance is evaluated (w_aux = delta^2 v_aux; M3 specification 6.7).
_W_DELTA = 0.1
# Exact pair sums for the direct-route variances up to this many orbitals, RB above.
_EXACT_MAX_NORB = 4
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
        d = np.array(self.deltas, dtype=float).ravel()
        v = np.array(self.values, dtype=float).ravel()
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
        delta = float(delta)
        if not (math.isfinite(delta) and delta > 0):
            raise ValueError(f"delta must be positive and finite, got {delta!r}")
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
def allowed_elements(active: ActiveSpace, kind: str = "spin_free", irreps=None) -> tuple[np.ndarray, np.ndarray]:
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


def element_normalizations(active: ActiveSpace, elements, kind: str,
                           mapping: QubitMapping | None = None) -> dict:
    """LCU normalizations of the AE routes: {"lam_q", "alpha_x", "n_terms_x", "order"}.

    ``lam_q`` (K,) and ``alpha_x`` (K,) (Hartree) are the Pauli 1-norms without the identity
    term of Q_e (``targets.aux_targets``) and X_e (``targets.direct_targets``), ``n_terms_x``
    (K,) the numbers of non-identity Pauli terms of X_e; ``order`` is the qubit order of
    ``mapping`` (default blocked; the norms do not depend on it).
    """
    el = _elements(active, elements, kind)
    mapping = QubitMapping(active.norb) if mapping is None else mapping
    qs = targets.aux_targets(el, mapping, kind)
    xs = targets.direct_targets(el, active, mapping, kind)
    return {"lam_q": np.array([q.norm1(False) for q in qs]),
            "alpha_x": np.array([x.norm1(False) for x in xs]),
            "n_terms_x": np.array([_non_identity_terms(x) for x in xs], dtype=np.int64),
            "order": mapping.order}


def element_variances(active: ActiveSpace, ci: np.ndarray, elements, kind: str, *,
                      params: FTParameters = FTParameters()) -> dict:
    """Per-shot variances of the two residual routes under matchgate shadows at the state ``ci``.

    Returns {"v_direct", "w_aux", "stderr", "stderr_total", "method", "n_settings", "seed",
    "delta_aux"}: ``v_direct`` (K,) the direct-route variances (Ha^2; exact pair sums for
    N <= 4, else RB over ``params.n_settings`` matchings with ``params.seed``), ``stderr``
    (K,) and ``stderr_total`` their standard errors (0 for exact sums), ``w_aux`` (K,)
    delta^2 times the aux-route variances at delta = ``delta_aux`` = 0.1 (dimensionless).
    """
    el = _elements(active, elements, kind)
    if active.norb <= _EXACT_MAX_NORB:
        opts = {"method": "exact"}
    else:
        opts = {"method": "rb", "n_settings": params.n_settings, "seed": params.seed}
    dv = routes.direct_residual_variance(active, ci, el, kind, "matchgate_shadow", **opts)
    av = routes.aux_residual_variance(active, ci, _W_DELTA, "exact", el, kind, "matchgate_shadow")
    k = el.shape[0]
    stderr = np.zeros(k) if dv.per_target_stderr is None else np.asarray(dv.per_target_stderr, dtype=float)
    return {"v_direct": np.asarray(dv.per_target, dtype=float),
            "w_aux": _W_DELTA ** 2 * np.asarray(av.per_target, dtype=float),
            "stderr": stderr, "stderr_total": 0.0 if dv.stderr is None else float(dv.stderr),
            "method": dv.method, "n_settings": opts.get("n_settings"), "seed": opts.get("seed"),
            "delta_aux": _W_DELTA}


# ------------------------------------------------------------------------- biases
def _aux_bias_values(active: ActiveSpace, ci: np.ndarray, el: np.ndarray, kind: str, deltas,
                     a: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.array([_norm_m(routes.aux_residual(active, ci, d, "exact", el, kind) - a, m) for d in deltas])


def aux_bias_curve(active: ActiveSpace, ci: np.ndarray, elements, kind: str, deltas) -> np.ndarray:
    """||routes.aux_residual(delta, "exact") - routes.exact_residual||_m per delta (Hartree).

    The exact bias of the aux route on exp(-i delta H) Psi, weighted by the multiplicities;
    ``deltas`` in 1/Hartree.
    """
    el = _elements(active, elements, kind)
    m = targets.element_multiplicities(el, kind, active.norb).astype(float)
    a = routes.exact_residual(active, ci, el, kind)
    return _aux_bias_values(active, ci, el, kind, np.atleast_1d(np.asarray(deltas, dtype=float)), a, m)


def factorization_bias(active: ActiveSpace, active_model: ActiveSpace, ci: np.ndarray, elements,
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
    "best": the name of the lowest, "variance": its value}.
    """
    space = FCISpace(active.norb, active.nelec)
    c = np.asarray(ci).reshape(space.shape)
    c = c / np.linalg.norm(c)
    out: dict = {name: float(schemes.df_variance(active, c, shift=shift, **opts).total)
                 for name, (opts, shift) in DF_VARIANTS.items()}
    best = min(DF_VARIANTS, key=out.__getitem__)
    out.update(best=best, variance=out[best])
    return out


# ------------------------------------------------------------------------- problem
def build_problem(active: ActiveSpace, ci: np.ndarray, elements, kind: str, *, variances: dict,
                  normalizations: dict, deltas=None) -> ResidualProblem:
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

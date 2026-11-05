"""Residual routes: auxiliary-state (finite-difference) estimators of the ACSE residual.

The targets and their post-processing are defined in
:mod:`thc_engine_cqe.measure.targets`; this module prepares the auxiliary states,
evaluates the infinite-shot estimates (the bias of each route), computes the
per-shot variances of the routes under the measurement schemes of
:mod:`thc_engine_cqe.measure.schemes` (:func:`aux_residual_variance`,
:func:`direct_residual_variance`) and converts per-shot variances into shot counts.

States are real CI vectors (exact CQE iterates with real orbitals).
:func:`aux_residual` and :func:`exact_residual` accept a real vector times a global
phase (the phase is removed; every quantity here is quadratic in Psi) and reject
genuinely complex input (the complex-state extension is described in :mod:`.targets`).

Auxiliary states (:func:`aux_states`)
-------------------------------------
With real double-factorized ERIs (pq|rs) = sum_l L_l[p,q] L_l[r,s]
(``factor.double_factorize``; untruncated by default, a truncated factorization is
passed as ``df``),

    H - ecore = H_0 + sum_{l=1}^{L} H_l,   H_0 = sum_pq h'_pq E_pq,   h' = h - 1/2 sum_r (pr|rq),
    H_l = 1/2 F_l^2,   F_l = sum_pq L_l[p,q] E_pq,

an operator identity for the factorized Hamiltonian H[eri_DF]: h' uses the
factorized ERIs (as ``schemes.df_variance`` does), so the fragments sum to
H[eri_DF] - ecore, which is H - ecore for the untruncated default (observed
max |eri - eri_DF| = 3.5e-12 for H6 at 1.0 A; the sum is tested against H on the
whole Fock space for H4, and against H[eri_DF] for a truncated factorization).
Fragment 0 is the one-body part, fragments 1..L are the leaves in the order of the
factorization (decreasing Frobenius norm). The methods are:

* ``"exact"``: Lambda = exp(-i delta H) Psi (one state; ``df`` is not used);
* ``"df-trotter"``: Lambda = U_0 U_1 ... U_L Psi with U_l = exp(-i delta H_l), the
  first-order product of the exact fragment propagators (U_L acts first; one state);
* ``"df-fragments"``: Lambda_l = U_l Psi for l = 0..L (L + 1 states).

Every exponential is applied with ``fci_ops.expm_apply``: a Taylor series in
substeps h with |h| ||H_l|| <= 1/2 (power-iteration norm estimate), each summed
until the next term is below ``tol`` ||v|| (``tol = 1e-15`` by default). Agreement
with ``scipy.linalg.expm`` is tested to 1e-13 on H4 (all methods); observed
<= 3e-15 for H6 at 1.0 A (all methods, delta up to 0.6). The global phase
exp(-i delta ecore) is omitted; it does not change any expectation value.

Bias (:func:`aux_residual`)
---------------------------
The estimate is A_hat_e = -sum_l <Q_e>_{Lambda_l} / delta (one term for "exact" and
"df-trotter"), Q_e = (Gamma_e - Gamma_e^dagger)/(2i). Parity argument: for real
Psi, real factors (real h' and L_l, so every H_l is a real symmetric matrix in the
determinant basis) and real Gamma_e, each propagator obeys conj U_l(delta) =
U_l(-delta). Hence conj Lambda(delta) = Lambda(-delta), and
f(delta) = <Gamma_e>_{Lambda(delta)} obeys conj f(delta) = f(-delta): Im f is odd
in delta, and the delta-expansion of <Q_e>_Lambda = Im f has only odd powers.
With f'(0) = -i <[Gamma_e, H]> = -i A_e (for the product, the first-order terms
of the factors add up), A_hat_e = A_e + O(delta^2), with no O(delta) term. For
"exact" the leading term is (delta^2/6) <[H,[H,[H,Gamma_e]]]>; each fragment of
"df-fragments" gives <[Gamma_e, H_l]> + (delta^2/6) <ad_{H_l}^3(Gamma_e)> + O(delta^4),
and the first-order terms add up to A_e. These statements are tested in
tests/test_routes.py (log-log slopes, leading coefficients and dense
exponentials on H4).

Shot cost of the routes
-----------------------
Every auxiliary state carries the full per-shot variance of the target -Q_e/delta.
The schemes of ``measure/schemes.py`` (QWC, Pauli and matchgate shadows) build
their per-shot (per-snapshot) variances from blocks, one per QWC group or one
for a shadow ensemble, of the form sum_jk w_jk <P_j P_k>_Lambda minus a quadratic
form in the means <P_j>_Lambda, with real weights w_jk and commuting pairs
(P_j, P_k) of the Pauli strings of the Q_e. Q_e = -i (Gamma_e - Gamma_e^T)/2 is an
imaginary matrix with real Pauli coefficients, so its strings are imaginary and
every such P_j P_k is real symmetric. By conj Lambda(delta) = Lambda(-delta)
(parity argument above), <P_j P_k>_Lambda is even in delta and <P_j>_Lambda is
odd: every block, and with it the per-shot variance on Lambda, equals its value
on Psi up to O(delta^2), however small the fragment that turns Psi into Lambda.
Hence:

* ``"exact"`` and ``"df-trotter"`` cost about the variance V(Psi) of one state.
* ``"df-fragments"`` measures L + 1 states, each with about V(Psi); with the
  optimal allocation over the states, total = (sum_l sqrt V_l)^2 ~ (L + 1)^2 V(Psi).
  A leaf's first-order term <[Gamma_e, H_l]> shrinks with the leaf, its share of
  the variance does not. This is tested on H4 (L = 10) with each Q_e measured
  alone in its eigenbasis. With the same proxy, observed for H6 at 1.0 A (CQE
  iterate 2, L = 20): (sum_l sqrt V_l)^2 / V_exact = 441.0 at delta = 0.01 and
  438-441 up to delta = 0.6, against (L + 1)^2 = 441, for both element sets.
  With the exact variances of ``measure/schemes.py`` (:func:`aux_residual_variance`),
  (sum_l sqrt V_l)^2 / ((L + 1)^2 V_exact) is 1 within 0.03 delta^2 on H4 (tested
  for QWC, Pauli and matchgate shadows, delta = 0.1 and 0.6, both element sets;
  observed 0.9948-1.0000), and 0.9999 for H6 at 1.0 A (QWC, delta = 0.1;
  observed). The first-order terms of single fragments are larger
  than A and cancel in the sum (observed ||a_1|| = 14 ||A|| for the largest leaf,
  H6 spin-orbital set).
* Truncating the factorization removes leaves that add variance and almost no
  signal. Observed with ``df=double_factorize(active.eri, tol=1e-6)`` (Cholesky
  stopped at a largest remaining diagonal of 1e-6): H6 keeps L = 15 of 20 leaves
  (1.0 A and 1.8 bohr), so (L + 1)^2 drops from 441 to 256, and the first-order
  truncation bias ||A[eri_DF] - A|| is 4.1e-6 to 5.0e-6 ||A|| (both element sets,
  iterate 2); the route bias of "df-trotter" and "df-fragments" changes by at most
  0.35% at delta >= 0.01, and by +60% to +104% at delta = 1e-3, where the
  truncation bias is comparable to it. H4 at 1.8 bohr keeps 9 of 10 leaves; LiH
  (21 leaves) and N2 CAS(6,6) (20) keep all.
  ``aux_residual(..., df=df) - exact_residual(...)`` includes the truncation bias,
  because :func:`exact_residual` uses the exact ERIs.
* Fragment-specific step sizes (a larger delta_l for a smaller leaf) would cut
  the cost of the small leaves; they are not implemented.

Per-shot variances of the routes (:func:`aux_residual_variance`, :func:`direct_residual_variance`)
-----------------------------------------------------------------------------------------------
Both functions return a ``schemes.VarianceResult`` whose ``total`` is the per-shot
sum_e Var(A_hat_e) and whose ``expectation`` is the infinite-shot estimate.

* Aux route: the vector target [-Q_e / delta] (post-processing folded in, so
  every variance carries 1/delta^2) on ``QubitState.from_ci`` of each auxiliary
  state, with ``schemes.scheme_variance``. For "df-fragments" the L + 1 states
  are independent experiments whose estimates add up; ``schemes.combine_independent``
  gives the optimal allocation across them, total = (sum_l sqrt V_l)^2, nested
  with the optimal allocation over the settings of each state. The expectation
  equals :func:`aux_residual` (tested).
* Direct route: the vector target X_e of ``targets.direct_targets`` on
  ``QubitState.from_ci`` of Psi; the expectation is the exact residual (tested).

The scheme options pass through; the scheme's ``method`` is passed as
``scheme_method`` (``scheme_method="mc"`` simulates the actual measurements, which
validates the wiring end to end on H4; tested). The scheme code builds a dense
table of <P(x, z)> for every z once per call whenever it fits (``measure.schemes``,
"Expectation values"); for the 840 spin-orbital and 315 spin-free H6 Q_e this table
(1024 x 4096 entries) takes 0.2-0.8 s per call, against 4-16 ms for the pair sums
themselves (observed, one thread, any of the three schemes; the table time varies
with the machine load), so a df-fragments call costs about L + 1 times one state.

Shots (:func:`shots_for_relative_error`)
----------------------------------------
For an unbiased estimator with per-shot variances v_e, S shots give the mean
squared error sum_e v_e / S, so the RMS criterion sqrt(sum_e v_e / S) = rel ||A||
over the element set gives S = sum_e v_e / (rel^2 ||A||^2). With a bias b
(aux route), ||b||^2 + sum_e v_e / S = rel^2 ||A||^2 gives
S = sum_e v_e / (rel^2 ||A||^2 - ||b||^2), which diverges as ||b|| -> rel ||A||. The
target is unreachable (inf) when ||b|| >= rel ||A||, and also when the difference
is within round-off of zero (at most 1e-12 rel^2 ||A||^2). Norms are over the
element set; per-element ``weights`` (e.g. ``targets.element_multiplicities``)
switch to the Frobenius norm of the full tensor.
"""

from __future__ import annotations

import dataclasses
from typing import Sequence

import numpy as np

from ..fci_ops import FCISpace, Hamiltonian, expm_apply, one_body
from ..residual import compute_residual
from . import schemes
from .pauli import QubitMapping, QubitState, coefficient_matrix
from .targets import _as_elements, _check_kind, aux_estimates, aux_targets, direct_targets, element_set

__all__ = [
    "METHODS",
    "df_fragments",
    "aux_states",
    "aux_residual",
    "exact_residual",
    "shots_for_relative_error",
    "aux_residual_variance",
    "direct_residual_variance",
]

METHODS = ("exact", "df-trotter", "df-fragments")


def _check_method(method: str) -> str:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    return method


def _check_delta(delta) -> float:
    d = float(delta)
    if d == 0 or not np.isfinite(d):
        raise ValueError(f"delta must be finite and nonzero, got {delta!r}")
    return d


def _real_ci(ci, space: FCISpace) -> np.ndarray:
    """Normalized real CI array of a real state, possibly given times a global phase.

    Complex input is divided by the phase of its largest-modulus amplitude and must
    then be real to 1e-13 relative (max |Im| <= 1e-13 max |c|); every route quantity
    is quadratic in Psi, so the phase does not matter. A ValueError for genuinely
    complex states (the routes assume real Psi) and for the zero vector.
    """
    c = np.asarray(ci)
    if c.size != space.size:
        raise ValueError(f"CI vector has {c.size} entries, expected {space.size}")
    c = c.reshape(space.shape)
    if np.iscomplexobj(c):
        big = c.flat[int(np.argmax(np.abs(c)))]
        if big == 0:
            raise ValueError("zero CI vector")
        c = c * (np.conj(big) / abs(big))
        if np.max(np.abs(c.imag)) > 1e-13 * np.max(np.abs(c)):
            raise ValueError("the residual routes assume a real CI vector (up to a global phase); for "
                             "complex states see the complex-state extension in measure.targets")
        c = c.real
    c = np.asarray(c, dtype=float)
    nrm = float(np.linalg.norm(c))
    if nrm == 0:
        raise ValueError("zero CI vector")
    return c / nrm


def _check_states(states, space: FCISpace) -> list[np.ndarray]:
    """CI arrays of reused auxiliary states: CI-space size and unit norm (1e-10) are checked."""
    out = []
    for l, lam in enumerate(states):
        lam = np.asarray(lam)
        if lam.size != space.size:
            raise ValueError(f"state {l} has {lam.size} entries, expected {space.size} (the CI space of active)")
        nrm = float(np.linalg.norm(lam))
        if not abs(nrm - 1.0) <= 1e-10:
            raise ValueError(f"state {l} has norm {nrm!r}; pass the normalized output of aux_states")
        out.append(lam.reshape(space.shape))
    if not out:
        raise ValueError("states is empty")
    return out


def df_fragments(active, df=None) -> tuple[np.ndarray, np.ndarray]:
    """(h', leaves) of H[eri_DF] - ecore = sum h'_pq E_pq + 1/2 sum_l (sum_pq L_l[p,q] E_pq)^2.

    ``df`` is a ``factor.DoubleFactorization`` (default: ``double_factorize(active.eri)``,
    untruncated; pass e.g. ``double_factorize(active.eri, tol=1e-6)`` or ``n_leaves=...``
    to drop small leaves, see "Shot cost" in the module docstring). ``leaves`` has shape
    (L, N, N) (``df.leaves()``), so "df-fragments" has L + 1 states, and
    h' = h1 - 1/2 einsum('prrq->pq', eri_DF) uses the factorized ERIs.
    """
    from ..factor import double_factorize

    df = double_factorize(np.asarray(active.eri)) if df is None else df
    leaves = df.leaves()
    if leaves.shape[1:] != np.asarray(active.h1).shape:
        raise ValueError(f"factorization has norb={leaves.shape[1]}, active space {active.norb}")
    h_prime = np.asarray(active.h1) - 0.5 * np.einsum("prrq->pq", df.to_eri())
    return h_prime, leaves


def aux_states(active, ci, delta: float, method: str = "exact", *, df=None,
               tol: float = 1e-15) -> list[np.ndarray]:
    """Auxiliary states of the aux route as complex CI arrays (module docstring).

    Returns [exp(-i delta H) Psi] (``"exact"``), [U_0 U_1 ... U_L Psi]
    (``"df-trotter"``) or [U_0 Psi, U_1 Psi, ..., U_L Psi] (``"df-fragments"``),
    U_l = exp(-i delta H_l), for Psi = ci / ||ci|| (any CI array; real states are
    assumed only by the estimators). ``df`` as in :func:`df_fragments` (not used by
    ``"exact"``); ``tol`` is the Taylor-series tolerance of ``fci_ops.expm_apply``.
    """
    _check_method(method)
    delta = _check_delta(delta)
    ham = Hamiltonian(active)
    space = ham.space
    psi = np.asarray(ci).reshape(space.shape).astype(np.complex128)
    nrm = float(np.linalg.norm(psi))
    if nrm == 0:
        raise ValueError("zero CI vector")
    psi = psi / nrm
    if method == "exact":
        return [expm_apply(lambda v: -1j * ham.apply(v), psi, delta, tol=tol)]
    h_prime, leaves = df_fragments(active, df)

    def u_one(v):
        return expm_apply(lambda u: -1j * one_body(space, h_prime, u), v, delta, tol=tol)

    def u_leaf(leaf, v):
        def matvec(u):
            return -0.5j * one_body(space, leaf, one_body(space, leaf, u))
        return expm_apply(matvec, v, delta, tol=tol)

    if method == "df-trotter":
        v = psi
        for leaf in leaves[::-1]:       # U_L acts first, U_0 last
            v = u_leaf(leaf, v)
        return [u_one(v)]
    return [u_one(psi)] + [u_leaf(leaf, psi) for leaf in leaves]


def aux_residual(active, ci, delta: float, method: str = "exact", elements=None,
                 kind: str = "spin_orbital", *, mapping: QubitMapping | None = None, df=None,
                 states: Sequence[np.ndarray] | None = None, per_state: bool = False) -> np.ndarray:
    """Infinite-shot aux-route estimate A_hat_e = -sum_l <Q_e>_{Lambda_l} / delta (real Psi).

    ``elements`` defaults to ``targets.element_set(N, kind)``; the result has one
    real entry per element, in the same order (shape (K,)). The <Q_e> are exact
    expectation values of the qubit operators ``targets.aux_targets`` on
    ``QubitState.from_ci`` of the auxiliary states, so ``aux_residual(...) -
    exact_residual(...)`` is the bias of the route (including the truncation bias of a
    truncated ``df``). ``per_state=True`` returns the per-state contributions
    -<Q_e>_{Lambda_l} / delta, shape (n_states, K).

    ``states`` reuses the output of ``aux_states(active, ci, delta, method, df=df)``
    for the same active space, state (up to normalization and global phase), delta,
    method and df; then ``method`` and ``df`` are not used. Only the CI-space size and
    the unit norm (to 1e-10) of each state are checked: states of another ci or delta
    are not detected.
    """
    space = FCISpace(active.norb, active.nelec)
    ci = _real_ci(ci, space)
    delta = _check_delta(delta)
    _check_method(method)
    _check_kind(kind)
    norb = active.norb
    el = element_set(norb, kind) if elements is None else _as_elements(elements, kind, norb)
    mapping = QubitMapping(norb) if mapping is None else mapping
    if mapping.norb != norb:
        raise ValueError(f"mapping has norb={mapping.norb}, active space {norb}")
    if states is None:
        states = aux_states(active, ci, delta, method, df=df)
    else:
        states = _check_states(states, space)
    ops = aux_targets(el, mapping, kind)
    q = np.zeros((len(states), len(ops)))
    if ops and any(o.n_terms for o in ops):
        xs, zs, w = coefficient_matrix(ops)
        for l, lam in enumerate(states):
            st = QubitState.from_ci(norb, active.nelec, lam, mapping)
            q[l] = (w @ st.expectations(xs, zs)).real
    if per_state:
        return -q / delta
    return aux_estimates(q, delta)


def _mazziotti_elements(blocks: np.ndarray, el: np.ndarray) -> np.ndarray:
    """<a+_i a+_j a_l a_k> from the spin blocks.

    blocks[sigma, tau][p,q,r,s] = <a+_{p sigma} a+_{r tau} a_{s tau} a_{q sigma}> (PySCF's trans_rdm12s).
    """
    (pi, si), (pj, sj), (pk, sk), (pl, sl) = (el[:, c].T for c in range(4))
    direct = (si == sk) & (sj == sl)
    cross = ~direct & (si == sl) & (sj == sk)            # a_l a_k = -a_k a_l
    out = np.zeros(el.shape[0])
    out[direct] = blocks[si, sj, pi, pk, pj, pl][direct]
    out[cross] = -blocks[si, sj, pi, pl, pj, pk][cross]
    return out


def exact_residual(active, ci, elements=None, kind: str = "spin_orbital") -> np.ndarray:
    """Exact residual on an element set (real Psi): 2A^{ij}_{kl} or A[p,q,r,s], shape (K,).

    Spin-free: ``residual.compute_residual(active, ci).A`` at the representatives.
    Spin-orbital: from PySCF's spin-resolved transition 2-RDMs of (Psi, H Psi),
    T^{ij}_{kl} = <Psi| a+_i a+_j a_l a_k H |Psi> and A^{ij}_{kl} = T^{ij}_{kl} - T^{kl}_{ij}
    (real states). Independent of the qubit layer.
    """
    from pyscf.fci import direct_spin1

    ham = Hamiltonian(active)
    space = ham.space
    ci = _real_ci(ci, space)
    _check_kind(kind)
    norb = active.norb
    el = element_set(norb, kind) if elements is None else _as_elements(elements, kind, norb)
    if kind == "spin_free":
        a = compute_residual(active, ci, ham).A
        return np.ascontiguousarray(np.real(a[tuple(el.T)]), dtype=float)
    hci = np.asarray(ham.apply(ci), dtype=float)
    _, (d2aa, d2ab, d2ba, d2bb) = direct_spin1.trans_rdm12s(ci, hci, norb, active.nelec)
    blocks = np.array([[d2aa, d2ab], [d2ba, d2bb]])
    t_e = _mazziotti_elements(blocks, el)
    t_bar = _mazziotti_elements(blocks, el[:, [2, 3, 0, 1]])
    return t_e - t_bar


def shots_for_relative_error(total_variance, A, rel: float, bias=None, *, weights=None) -> float:
    """Shots S for a relative RMS (Frobenius) error ``rel`` of the residual estimate.

    ``total_variance`` is the per-shot sum_e v_e (a scalar) or the per-element
    v_e; ``A`` the exact values on the element set (or the scalar ||A||); ``bias``
    None, the per-element bias, or the scalar ||bias||. Returns
    S = sum v / (rel^2 ||A||^2 - ||bias||^2), which is sum v / (rel^2 ||A||^2)
    without bias; S is a real number (not rounded up). ``weights`` (per-element,
    e.g. ``targets.element_multiplicities``) weight every sum, which gives the
    criterion for the full tensor; then every other input must be per-element too.

    The target is unreachable (inf) when the budget rel^2 ||A||^2 - ||bias||^2 is
    at most 1e-12 rel^2 ||A||^2: when ||bias|| >= rel ||A||, and when the two agree
    to round-off (S diverges as ||bias|| -> rel ||A|| from below). The per-element
    inputs (and ``weights``) must be 1-D with one common shape, every input finite,
    the variances real and non-negative, the weights non-negative and ``rel``
    positive and finite (ValueError otherwise; TypeError for non-numeric input).
    """
    rel = float(rel)
    if not (np.isfinite(rel) and rel > 0):
        raise ValueError(f"rel must be positive and finite, got {rel!r}")
    named = {"total_variance": total_variance, "A": A}
    if bias is not None:
        named["bias"] = bias
    if weights is not None:
        named["weights"] = weights
    arrays = {}
    for name, x in named.items():
        x = np.asarray(x)
        if x.dtype.kind not in "biufc":
            raise TypeError(f"{name} must be numeric, got dtype {x.dtype}")
        if x.ndim > 1:
            raise ValueError(f"{name} must be a scalar or a per-element 1-D array, got shape {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{name} must be finite")
        arrays[name] = x
    for name in ("total_variance", "weights"):
        if name in arrays:
            x = arrays[name]
            if np.iscomplexobj(x) or np.any(x < 0):
                raise ValueError(f"{name} must be real and non-negative")
    shapes = {name: x.shape for name, x in arrays.items() if x.ndim == 1}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"per-element inputs have different shapes: {shapes}")
    if weights is not None:
        scalars = [name for name, x in arrays.items() if x.ndim == 0]
        if scalars:
            raise ValueError(f"weights need per-element {', '.join(scalars)}")
        w = arrays["weights"].astype(float)
    else:
        w = None

    def wsum(x):                                # weighted sum (x per-element) or the scalar itself
        if x.ndim == 0:
            return float(x)
        return float(np.sum(x if w is None else w * x))

    v_tot = wsum(arrays["total_variance"].astype(float))
    target = rel * rel * wsum(np.abs(arrays["A"]) ** 2)
    budget = target - (wsum(np.abs(arrays["bias"]) ** 2) if bias is not None else 0.0)
    if budget <= 1e-12 * target:
        return float("inf")
    return v_tot / budget


# ---------------------------------------------------------------------------
# Scheme wiring (spec_B.md B3): per-shot variances of the two routes under the
# measurement schemes of measure/schemes.py ("Per-shot variances of the routes"
# in the module docstring).
# ---------------------------------------------------------------------------
def _check_scheme(scheme: str) -> str:
    if scheme not in schemes.SCHEMES:
        raise ValueError(f"scheme must be one of {schemes.SCHEMES} for the residual routes (DF basis "
                         f"rotation measures the energy only: schemes.df_variance), got {scheme!r}")
    return scheme


def _check_mapping(mapping: QubitMapping | None, norb: int) -> QubitMapping:
    mapping = QubitMapping(norb) if mapping is None else mapping
    if not isinstance(mapping, QubitMapping):
        raise TypeError(f"mapping must be a QubitMapping, got {type(mapping).__name__}")
    if mapping.norb != norb:
        raise ValueError(f"mapping has norb={mapping.norb}, active space {norb}")
    return mapping


def _shared_rng(opts: dict, n_states: int) -> dict:
    """One Generator for all states, so that their Monte Carlo draws are independent."""
    if n_states > 1 and "seed" in opts and not isinstance(opts["seed"], np.random.Generator):
        opts = dict(opts, seed=np.random.default_rng(opts["seed"]))
    return opts


def _scheme_options(opts: dict) -> dict:
    """The scheme's keywords, with ``scheme_method`` renamed to the scheme's ``method``.

    ``aux_residual_variance`` has a positional ``method`` (the route method), so the
    scheme's method reaches it only as ``scheme_method``; ``direct_residual_variance``
    has no route method and also accepts ``method``.
    """
    opts = dict(opts)
    if "scheme_method" in opts:
        if "method" in opts:
            raise TypeError("pass the scheme's method once (scheme_method or method)")
        opts["method"] = opts.pop("scheme_method")
    return opts


def aux_residual_variance(active, ci, delta, method, elements, kind, scheme, **opts):
    """Per-shot variance of the aux-route estimator A_hat_e = -Q_hat_e / delta under a scheme.

    The vector target is [-Q_e / delta] over ``elements`` (``targets.aux_targets(elements,
    mapping, kind)``, each divided by -delta, so the post-processing is folded in), and
    each auxiliary state Lambda_l of :func:`aux_states` is measured as
    ``QubitState.from_ci(norb, nelec, Lambda_l, mapping)`` with
    ``schemes.scheme_variance(targets, state, scheme, **scheme_options)``:

    * ``"exact"`` and ``"df-trotter"``: one state; the result is that call's
      :class:`~.schemes.VarianceResult`.
    * ``"df-fragments"``: L + 1 independent experiments whose estimates are summed.
      ``schemes.combine_independent`` allocates the shots across them optimally, so
      ``total`` = (sum_l sqrt V_l)^2 with V_l the per-state totals (for QWC the global
      optimum over all (L + 1) n_G settings), ``per_target`` = sum_l v_{l,e} / f_l,
      ``allocation`` = the fractions f_l per state, ``n_settings`` = the sum over the
      states, and ``scheme`` = "combined"; ``info["parts"]`` holds the per-state
      results (and their per-setting allocations). The total is about (L + 1)^2 times
      that of one state ("Shot cost of the routes"); report L (``info["n_leaves"]``)
      with every df-fragments number.

    ``total`` and ``per_target`` are per shot (S shots give sum_e Var(A_hat_e) =
    total / S) and include the 1/delta^2 of the post-processing. ``expectation`` is the
    infinite-shot estimate: it equals :func:`aux_residual` (tested), so
    ``expectation - exact_residual(...)`` is the route bias.

    Args:
        active, ci, delta, method, elements, kind: as for :func:`aux_residual`
            (``elements=None``: ``targets.element_set(N, kind)``). Real Psi only, up to
            a global phase (the parity argument and the estimator assume it).
            ``method`` is the route method (``METHODS``).
        scheme: one of ``schemes.SCHEMES`` ("qwc", "pauli_shadow", "matchgate_shadow").
        **opts: the route options ``mapping`` (QubitMapping, default blocked; QWC and
            Pauli shadows depend on the order), ``df`` and ``states`` (as for
            :func:`aux_residual`; ``states`` must hold one state for "exact" and
            "df-trotter", and L + 1 for "df-fragments" when ``df`` is given).
            ``scheme_method`` is the scheme's ``method`` ("exact"/"auto" by default;
            "mc" simulates the measurements, n <= 14 for the shadows, n <= 20 for
            QWC). Every other keyword goes to the scheme unchanged, e.g.
            ``n_shots``/``n_snapshots``/``n_settings`` and ``seed``. For QWC the
            grouping of the targets is computed once (unless a ``grouping`` is
            passed) and used for every state. With several states a ``seed`` that
            is not a Generator becomes one Generator shared by the states, so their
            Monte Carlo draws are independent.

    ``info`` additions: ``route`` ("aux"), ``route_method``, ``delta``, ``kind``,
    ``scheme``, ``order`` (of the mapping), ``n_states`` and ``n_leaves`` (L of the
    factorization; None for "exact", and for "df-trotter" with reused ``states``
    and no ``df``).
    """
    space = FCISpace(active.norb, active.nelec)
    ci = _real_ci(ci, space)
    delta = _check_delta(delta)
    _check_method(method)
    _check_kind(kind)
    _check_scheme(scheme)
    norb = active.norb
    el = element_set(norb, kind) if elements is None else _as_elements(elements, kind, norb)
    if el.shape[0] == 0:
        raise ValueError("need at least one element")
    mapping = _check_mapping(opts.pop("mapping", None), norb)
    df = opts.pop("df", None)
    states = opts.pop("states", None)
    opts = _scheme_options(opts)
    if states is not None:
        states = _check_states(states, space)              # any iterable of CI arrays -> list
    n_leaves = None
    if method != "exact":
        if states is None and df is None:
            from ..factor import double_factorize

            df = double_factorize(np.asarray(active.eri))      # the default of aux_states
        if df is not None:
            n_leaves = int(df.n_leaves)
        elif method == "df-fragments":
            n_leaves = len(states) - 1
    if states is None:
        states = aux_states(active, ci, delta, method, df=df)
    else:
        want = 1 if method != "df-fragments" else (None if df is None else n_leaves + 1)
        if want is not None and len(states) != want:
            raise ValueError(f"method {method!r} needs {want} auxiliary state(s), got {len(states)}")
    ops = [(-q) / delta for q in aux_targets(el, mapping, kind)]
    if scheme == "qwc" and opts.get("grouping") is None:
        opts["grouping"] = schemes.qwc_grouping(ops)
    opts = _shared_rng(opts, len(states))
    parts = [schemes.scheme_variance(ops, QubitState.from_ci(norb, active.nelec, lam, mapping), scheme, **opts)
             for lam in states]
    res = parts[0] if len(parts) == 1 else schemes.combine_independent(parts)
    info = dict(res.info)
    info.update(route="aux", route_method=method, delta=delta, kind=kind, scheme=scheme, order=mapping.order,
                n_states=len(parts), n_leaves=n_leaves)
    return dataclasses.replace(res, info=info)


def direct_residual_variance(active, ci, elements, kind, scheme, **opts):
    """Per-shot variance of the direct-route estimator A_hat_e = X_hat_e under a scheme.

    The vector target X_e = [Gamma_e - Gamma_e^dagger, H]/2
    (``targets.direct_targets(elements, active, mapping, kind)``; identity
    post-processing) measured on Psi = ``QubitState.from_ci(norb, nelec, ci, mapping)``:
    the :class:`~.schemes.VarianceResult` of ``schemes.scheme_variance(X, Psi, scheme,
    **scheme_options)``. ``expectation`` = <X_e>_Psi = A_e, the exact residual (tested).
    Real Psi only, up to a global phase (``_real_ci``): for complex Psi, <X_e> is Re A_e.

    Cost: the X_e have up to three-body terms (H6: 1.6e6 Pauli terms over 36,474
    strings, 1.68e9 term pairs for the 840 spin-orbital X_e; observed). With the
    scheme's default ``method="auto"``, matchgate shadows use the Rao-Blackwellized
    estimate above 2e7 pairs (default 2e4 settings; standard errors in ``stderr`` and
    ``per_target_stderr``), and Pauli shadows always use the exact pair sums (61-99 s
    for the 840 spin-orbital X_e at H6, 160-263 s for the 315 spin-free ones; observed,
    one thread, depending on the machine load), because their Rao-Blackwellized
    estimate is heavy-tailed (``measure.schemes``).

    Args:
        active, ci, elements, kind: as for :func:`exact_residual` (``elements=None``:
            ``targets.element_set(N, kind)``).
        scheme: one of ``schemes.SCHEMES``.
        **opts: the route options ``mapping`` (QubitMapping, default blocked),
            ``hamiltonian`` (a prebuilt qubit Hamiltonian, passed to ``direct_targets``)
            and ``targets`` (the X_e of ``direct_targets(elements, active, mapping,
            kind)``, to reuse them for several states; only their number and qubit
            count are checked). A PauliSum does not record its qubit order, so
            ``hamiltonian`` and ``targets`` each require ``mapping``, the mapping they
            were built with (ValueError without it; X_e of another order would give
            wrong variances and a wrong ``expectation``, and a mismatched mapping is
            not detected).
            ``scheme_method`` (or ``method``) is the scheme's method; every other
            keyword goes to the scheme unchanged (``n_settings``, ``rel_stderr``,
            ``seed``, ``max_pairs``, ...).

    ``info`` additions: ``route`` ("direct"), ``kind``, ``scheme`` and ``order``.
    """
    space = FCISpace(active.norb, active.nelec)
    ci = _real_ci(ci, space)
    _check_kind(kind)
    _check_scheme(scheme)
    norb = active.norb
    el = element_set(norb, kind) if elements is None else _as_elements(elements, kind, norb)
    if el.shape[0] == 0:
        raise ValueError("need at least one element")
    for name in ("hamiltonian", "targets"):
        if opts.get(name) is not None and opts.get("mapping") is None:
            raise ValueError(f"pass the mapping the {name} {'was' if name == 'hamiltonian' else 'were'} built "
                             "with (a PauliSum does not record its qubit order)")
    mapping = _check_mapping(opts.pop("mapping", None), norb)
    hamiltonian = opts.pop("hamiltonian", None)
    ops = opts.pop("targets", None)
    opts = _scheme_options(opts)
    if ops is None:
        ops = direct_targets(el, active, mapping, kind, hamiltonian=hamiltonian)
    else:
        ops = list(ops)
        if len(ops) != el.shape[0] or any(o.n_qubits != mapping.n_qubits for o in ops):
            raise ValueError(f"targets must be the {el.shape[0]} X_e of direct_targets(elements, active, "
                             f"mapping, kind) on {mapping.n_qubits} qubits")
    res = schemes.scheme_variance(ops, QubitState.from_ci(norb, active.nelec, ci, mapping), scheme, **opts)
    info = dict(res.info)
    info.update(route="direct", kind=kind, scheme=scheme, order=mapping.order)
    return dataclasses.replace(res, info=info)

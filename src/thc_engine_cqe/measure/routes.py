"""Residual routes: auxiliary-state (finite-difference) estimators of the ACSE residual.

The targets and their post-processing are defined in
:mod:`thc_engine_cqe.measure.targets`; this module prepares the auxiliary states,
evaluates the infinite-shot estimates (the bias of each route) and converts
per-shot variances into shot counts. The per-shot variances of the routes under
the measurement schemes (``aux_residual_variance``, ``direct_residual_variance``)
are wired to ``measure/schemes.py`` separately; see the stubs at the end.

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
For each scheme of ``measure/schemes.py`` (QWC, Pauli and matchgate shadows) the
per-shot variance on a state Lambda is sum_jk w_jk <P_j P_k>_Lambda minus a
quadratic form in the means <P_j>_Lambda, with real weights w_jk and only
commuting pairs (P_j, P_k) of Pauli strings of the Q_e. Q_e = -i (Gamma_e -
Gamma_e^T)/2 is an imaginary matrix with real Pauli coefficients, so its strings
are imaginary and every such P_j P_k is real symmetric. By conj Lambda(delta) =
Lambda(-delta) (parity argument above), <P_j P_k>_Lambda is even in delta and
<P_j>_Lambda is odd: the per-shot variance on Lambda equals that on Psi up to
O(delta^2), however small the fragment that turns Psi into Lambda. Hence:

* ``"exact"`` and ``"df-trotter"`` cost about the variance V(Psi) of one state.
* ``"df-fragments"`` measures L + 1 states, each with about V(Psi); with the
  optimal allocation over the states, total = (sum_l sqrt V_l)^2 ~ (L + 1)^2 V(Psi).
  A leaf's first-order term <[Gamma_e, H_l]> shrinks with the leaf, its share of
  the variance does not. Observed for H6 at 1.0 A (CQE iterate 2, L = 20, each Q_e
  measured alone in its eigenbasis): (sum_l sqrt V_l)^2 / V_exact = 441.0 at
  delta = 0.01 and 438-441 up to delta = 0.6, against (L + 1)^2 = 441, for both
  element sets. The first-order terms of single fragments are larger than A
  (observed ||a_1|| = 14 ||A||) and cancel in the sum. The relation is tested on
  H4 (L = 10).
* Truncating the factorization removes leaves that add variance and almost no
  signal. Observed with ``df=double_factorize(active.eri, tol=1e-6)``: H6 keeps
  L = 15 of 20 leaves (1.0 A and 1.8 bohr), so (L + 1)^2 drops from 441 to 256,
  with a first-order truncation bias of 4e-6 to 5e-6 ||A|| (spin-orbital set,
  iterate 2) and a route bias that changes by 0.3% or less at delta >= 0.01;
  H4 keeps 9 of 10 leaves; LiH (21 leaves) and N2 CAS(6,6) (20) keep all.
  ``aux_residual(..., df=df) - exact_residual(...)`` includes the truncation
  bias, because :func:`exact_residual` uses the exact ERIs.

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

from typing import Sequence

import numpy as np

from ..fci_ops import FCISpace, Hamiltonian, expm_apply, one_body
from ..residual import compute_residual
from .pauli import QubitMapping, QubitState, coefficient_matrix
from .targets import _as_elements, _check_kind, aux_estimates, aux_targets, element_set

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
    positive and finite (ValueError otherwise).
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
# Scheme wiring (benchmark agent, spec_B.md B3). These need measure/schemes.py,
# which this module does not import yet. The stubs fix the names and contracts.
# ---------------------------------------------------------------------------
def aux_residual_variance(active, ci, delta, method, elements, kind, scheme, **opts):
    """STUB (benchmark agent): per-shot variance of the aux-route estimator under a scheme.

    Contract (spec_B.md B3): returns the ``schemes.VarianceResult`` of
    A_hat_e = -Q_hat_e / delta over ``elements``. For ``"exact"`` and
    ``"df-trotter"``: the vector target [-Q_e / delta] (``targets.aux_targets``)
    on the auxiliary state ``QubitState.from_ci(norb, nelec, aux_states(...)[0],
    mapping)``. For ``"df-fragments"``: the same target on every fragment state,
    with the per-fragment totals V_l combined by the optimal shot allocation
    across fragments, total = (sum_l sqrt(V_l))^2, about (L + 1)^2 times the
    single-state total ("Shot cost" in the module docstring; report L, e.g.
    ``df.n_leaves``, with every df-fragments number). ``df`` (``opts``) is passed
    to :func:`aux_states`. Real Psi only (``_real_ci``).
    """
    raise NotImplementedError("aux_residual_variance is added by the benchmark agent "
                              "(spec_B.md B3); it wires routes.py to measure/schemes.py")


def direct_residual_variance(active, ci, elements, kind, scheme, **opts):
    """STUB (benchmark agent): per-shot variance of the direct-route estimator A_hat_e = X_hat_e.

    Contract (spec_B.md B3): the ``schemes.VarianceResult`` of the vector target
    ``targets.direct_targets(elements, active, mapping, kind)`` on
    ``QubitState.from_ci(norb, nelec, ci, mapping)`` (real Psi), exact for small
    targets and Rao-Blackwellized Monte Carlo for H6-size targets (about 1.6e6
    Pauli terms, observed).
    """
    raise NotImplementedError("direct_residual_variance is added by the benchmark agent "
                              "(spec_B.md B3); it wires routes.py to measure/schemes.py")

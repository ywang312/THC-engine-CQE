"""Residual routes: auxiliary-state (finite-difference) estimators of the ACSE residual.

The targets and their post-processing are defined in
:mod:`thc_engine_cqe.measure.targets`; this module prepares the auxiliary states,
evaluates the infinite-shot estimates (the bias of each route) and converts
per-shot variances into shot counts. The per-shot variances of the routes under
the measurement schemes (``aux_residual_variance``, ``direct_residual_variance``)
are wired to ``measure/schemes.py`` separately; see the stubs at the end.

States are real CI vectors (exact CQE iterates with real orbitals);
:func:`aux_residual` and :func:`exact_residual` reject complex input (the
complex-state extension is described in :mod:`.targets`).

Auxiliary states (:func:`aux_states`)
-------------------------------------
With real double-factorized ERIs (pq|rs) = sum_l L_l[p,q] L_l[r,s]
(``factor.double_factorize``, untruncated by default),

    H - ecore = H_0 + sum_{l=1}^{L} H_l,   H_0 = sum_pq h'_pq E_pq,   h' = h - 1/2 sum_r (pr|rq),
    H_l = 1/2 F_l^2,   F_l = sum_pq L_l[p,q] E_pq,

an operator identity (h' uses the factorized ERIs, so the sum is the factorized
Hamiltonian, which equals H for the untruncated default; observed
max |eri - eri_DF| = 3.5e-12 for H6 at 1.0 A). Fragment 0 is the one-body part,
fragments 1..L are the leaves in the order of the factorization (decreasing
Frobenius norm). The methods are:

* ``"exact"``: Lambda = exp(-i delta H) Psi (one state);
* ``"df-trotter"``: Lambda = U_0 U_1 ... U_L Psi with U_l = exp(-i delta H_l), the
  first-order product of the exact fragment propagators (U_L acts first; one state);
* ``"df-fragments"``: Lambda_l = U_l Psi for l = 0..L (L + 1 states).

Every exponential is applied exactly (``fci_ops.expm_apply``, Taylor series with
substeps, to 1e-15 of the norm by default). The global phase exp(-i delta ecore)
is omitted; it does not change any expectation value.

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

Shots (:func:`shots_for_relative_error`)
----------------------------------------
For an unbiased estimator with per-shot variances v_e, S shots give the mean
squared error sum_e v_e / S, so the RMS criterion sqrt(sum_e v_e / S) = rel ||A||
over the element set gives S = sum_e v_e / (rel^2 ||A||^2). With a bias b
(aux route), ||b||^2 + sum_e v_e / S = rel^2 ||A||^2 gives
S = sum_e v_e / (rel^2 ||A||^2 - ||b||^2), unreachable (inf) when ||b|| >= rel ||A||.
Norms are over the element set; per-element ``weights`` (e.g.
``targets.element_multiplicities``) switch to the Frobenius norm of the full tensor.
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
    """Normalized real CI array; a ValueError for complex states (the routes assume real Psi)."""
    c = np.asarray(ci)
    if c.size != space.size:
        raise ValueError(f"CI vector has {c.size} entries, expected {space.size}")
    c = c.reshape(space.shape)
    if np.iscomplexobj(c):
        if np.max(np.abs(c.imag)) > 1e-13 * np.max(np.abs(c)):
            raise ValueError("the residual routes assume a real CI vector; for complex states "
                             "see the complex-state extension in measure.targets")
        c = c.real
    c = np.asarray(c, dtype=float)
    nrm = float(np.linalg.norm(c))
    if nrm == 0:
        raise ValueError("zero CI vector")
    return c / nrm


def df_fragments(active, df=None) -> tuple[np.ndarray, np.ndarray]:
    """(h', leaves) of H - ecore = sum h'_pq E_pq + 1/2 sum_l (sum_pq L_l[p,q] E_pq)^2.

    ``df`` is a ``factor.DoubleFactorization`` (default: ``double_factorize(active.eri)``,
    untruncated). ``leaves`` has shape (L, N, N) (``df.leaves()``) and
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
    U_l = exp(-i delta H_l), for the normalized ``ci`` (any CI array; real states
    are assumed only by the estimators). ``df`` as in :func:`df_fragments`;
    ``tol`` is the Taylor-series tolerance of ``fci_ops.expm_apply``.
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
        return expm_apply(lambda u: -0.5j * one_body(space, leaf, one_body(space, leaf, u)), v, delta, tol=tol)

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
    exact_residual(...)`` is the bias of the route. ``states`` reuses the output of
    :func:`aux_states` (then ``method`` and ``df`` are not used); ``per_state=True``
    returns the per-state contributions -<Q_e>_{Lambda_l} / delta, shape (n_states, K).
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
    """<a+_i a+_j a_l a_k> from the spin blocks b[sigma, tau][p,q,r,s] = <a+_{p sigma} a+_{r tau} a_{s tau} a_{q sigma}>."""
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
    without bias, and inf ("unreachable") when ||bias|| >= rel ||A||. ``weights``
    (per-element, e.g. ``targets.element_multiplicities``) weight every sum, which
    gives the criterion for the full tensor; they need per-element arrays.
    S is a real number (not rounded up).
    """
    rel = float(rel)
    if not rel > 0:
        raise ValueError("rel must be positive")

    def wsum2(x, name):
        x = np.asarray(x)
        if x.ndim == 0:
            if weights is not None:
                raise ValueError(f"weights need per-element {name}")
            return float(abs(x)) ** 2
        w = np.ones(x.shape) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != x.shape:
            raise ValueError(f"weights and {name} have different shapes {w.shape}, {x.shape}")
        return float(np.sum(w * np.abs(x) ** 2))

    v = np.asarray(total_variance, dtype=float)
    if np.any(v < 0):
        raise ValueError("variances must be non-negative")
    if v.ndim == 0:
        if weights is not None:
            raise ValueError("weights need per-element variances")
        v_tot = float(v)
    else:
        w = np.ones(v.shape) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != v.shape:
            raise ValueError(f"weights and variances have different shapes {w.shape}, {v.shape}")
        v_tot = float(np.sum(w * v))
    budget = rel * rel * wsum2(A, "A") - (0.0 if bias is None else wsum2(bias, "bias"))
    if budget <= 0:
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
    across fragments, total = (sum_l sqrt(V_l))^2. Real Psi only (``_real_ci``).
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

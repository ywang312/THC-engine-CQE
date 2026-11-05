"""Tests of the residual routes (``thc_engine_cqe.measure.routes``): auxiliary states, infinite-shot
estimates (bias), shot counts, and the scheme wiring (per-shot variances of both routes).

References independent of the module: dense Fock-space matrices and ``scipy.linalg.expm`` (H4),
the fragment operators re-derived here from the DF leaves, Fock-oracle commutators, and
<[H,[H,[H,Gamma]]]> from PySCF transition RDMs of H^k Psi. The wiring tests rebuild the targets and
states from ``targets``/``aux_states`` and call ``measure.schemes`` directly, check the expectations
against the infinite-shot estimates and the exact residual, the 1/delta^2 scaling and the (L + 1)^2
cost of df-fragments, and simulate the measurements end to end (``scheme_method="mc"``).

Parity argument (bias O(delta^2) with real factors and real Psi): every fragment Hamiltonian is a
real symmetric matrix in the determinant basis, so conj U(delta) = U(-delta) for each exact
propagator and their products; with real Psi, conj Lambda(delta) = Lambda(-delta), and since
Gamma_e is real, f(delta) = <Gamma_e>_{Lambda(delta)} obeys conj f(delta) = f(-delta). Hence
<Q_e>_Lambda = Im f has only odd powers of delta, and A_hat_e = -Im f / delta = A_e + O(delta^2).
``test_aux_states_obey_the_parity_relation`` checks conj Lambda(delta) = Lambda(-delta) directly.
"""

import numpy as np
import pytest
import scipy.linalg

from thc_engine_cqe import residual, systems
from thc_engine_cqe.cqe import run_cqe
from thc_engine_cqe.factor import double_factorize
from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian
from thc_engine_cqe.measure import routes as rt
from thc_engine_cqe.measure import schemes as sc
from thc_engine_cqe.measure import targets as tg
from thc_engine_cqe.measure.pauli import QubitMapping, QubitState
from thc_engine_cqe.testing.fock import FockSpace

DELTAS = np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1])


@pytest.fixture(scope="module")
def h4():
    """H4 at 1.8 bohr after 2 exact SD CQE iterations: (active, ci)."""
    act = systems.hydrogen_chain(4, 1.8, "bohr")
    return act, run_cqe(act, max_iter=2).ci


@pytest.fixture(scope="module")
def h6():
    """H6 at 1.0 A after 2 exact SD CQE iterations (19.6 mHa above FCI, observed): (active, ci)."""
    act = systems.hydrogen_chain(6, 1.0, "angstrom")
    return act, run_cqe(act, max_iter=2).ci


@pytest.fixture(scope="module")
def h4_fock(h4):
    """Fock-space matrices of H4: oracle, state vector, H, the DF fragments H_0..H_L (re-derived from
    the leaves), and Gamma_e for both element sets."""
    act, ci = h4
    n = act.norb
    fs = FockSpace(n)
    leaves = double_factorize(act.eri).leaves()
    eri_df = np.einsum("lpq,lrs->pqrs", leaves, leaves)
    h_prime = act.h1 - 0.5 * np.einsum("prrq->pq", eri_df)
    frags = [fs.one_body(h_prime).toarray()]
    for leaf in leaves:
        f = fs.one_body(leaf).toarray()
        frags.append(0.5 * f @ f)
    gam = {}
    for kind in tg.KINDS:
        mats = []
        for e in tg.element_set(n, kind).tolist():
            if kind == "spin_orbital":
                (pi, si), (pj, sj), (pk, sk), (pl, sl) = e
                mats.append((fs.adag(pi, si) @ fs.adag(pj, sj) @ fs.a(pl, sl) @ fs.a(pk, sk)).toarray())
            else:
                mats.append(fs.e2(*e).toarray())
        gam[kind] = mats
    return {"fs": fs, "v": fs.from_ci(ci, act.nelec),
            "H": fs.hamiltonian(act.h1, act.eri, act.ecore).toarray(),
            "frags": frags, "gamma": gam, "h_prime": h_prime, "leaves": leaves}


def fock_expect_comm(v, g, m):
    """<v|[g, m]|v> for dense matrices."""
    return v.conj() @ (g @ (m @ v)) - v.conj() @ (m @ (g @ v))


AD3 = ((3, 0, 1), (2, 1, -3), (1, 2, 3), (0, 3, -1))   # ad_m^3(g) = m^3 g - 3 m^2 g m + 3 m g m^2 - g m^3


def fock_ad3(v, gs, m):
    """<v|[m,[m,[m, g]]]|v> for every g in gs (m Hermitian): sum_c c <phi_a|g|phi_b>, phi_k = m^k v."""
    phi = [v]
    for _ in range(3):
        phi.append(m @ phi[-1])
    return np.array([sum(c * (phi[a].conj() @ (g @ phi[b])) for a, b, c in AD3) for g in gs])


def so_transition(act, bra, ket):
    """Full Mazziotti tensor <bra| a+_i a+_j a_l a_k |ket> (mode indices) from PySCF's trans_rdm12s."""
    from pyscf.fci import direct_spin1

    n = act.norb
    _, (aa, ab, ba, bb) = direct_spin1.trans_rdm12s(bra, ket, n, act.nelec)
    t = np.zeros((2 * n,) * 4)
    sl = lambda s: slice(s * n, (s + 1) * n)
    for (s, u), blk in {(0, 0): aa, (0, 1): ab, (1, 0): ba, (1, 1): bb}.items():
        t[sl(s), sl(u), sl(s), sl(u)] = blk.transpose(0, 2, 1, 3)
        if s != u:
            t[sl(s), sl(u), sl(u), sl(s)] = -blk.transpose(0, 2, 3, 1)
    return t


def slope(deltas, values):
    return np.polyfit(np.log(deltas), np.log(values), 1)[0]


# ======================================================= fragments and states
def test_df_fragments_reproduce_the_hamiltonian(h4, h4_fock):
    """df_fragments: h' = h - 1/2 sum_r (pr|rq) (1e-10 with the exact ERIs) and the DF leaves; the
    fragments sum to H - ecore as an operator on the whole Fock space (1e-12)."""
    act, _ = h4
    h_prime, leaves = rt.df_fragments(act)
    assert np.abs(h_prime - (act.h1 - 0.5 * np.einsum("prrq->pq", act.eri))).max() <= 1e-10
    assert np.array_equal(leaves, h4_fock["leaves"])
    assert np.abs(sum(h4_fock["frags"]) + act.ecore * np.eye(h4_fock["fs"].dim) - h4_fock["H"]).max() <= 1e-12


def test_truncated_factorization_is_passed_through(h4, h4_fock):
    """A truncated factorization passed as ``df`` (5 of the 10 leaves of H4) defines the route: the
    fragments rebuilt here from its leaves, with h' from its ERIs, sum to the Fock-space H[eri_DF]
    (1e-12); aux_states and aux_residual ("df-trotter", "df-fragments", both element sets) equal the
    dense references built from them (1e-13 and 1e-12); as delta -> 0 the estimate tends to the
    residual of the model H[eri_DF] (relative 1e-4 at delta = 1e-3; observed 3e-6), so it differs
    from the untruncated route by the truncation bias (observed 0.6 ||A||). "exact" ignores df.
    """
    act, ci = h4
    fs, v, gam = h4_fock["fs"], h4_fock["v"], h4_fock["gamma"]
    df5 = double_factorize(act.eri, n_leaves=5)
    leaves = df5.leaves()
    eri_df = np.einsum("lpq,lrs->pqrs", leaves, leaves)
    hp_ref = act.h1 - 0.5 * np.einsum("prrq->pq", eri_df)
    h_prime, got = rt.df_fragments(act, df5)
    assert got.shape == (5, act.norb, act.norb) and np.array_equal(got, leaves)
    assert np.abs(h_prime - hp_ref).max() <= 1e-14
    frags = [fs.one_body(hp_ref).toarray()]
    for leaf in leaves:
        f = fs.one_body(leaf).toarray()
        frags.append(0.5 * f @ f)
    h_model = fs.hamiltonian(act.h1, eri_df, act.ecore).toarray()
    assert np.abs(sum(frags) + act.ecore * np.eye(fs.dim) - h_model).max() <= 1e-12
    model = act.with_eri(eri_df)
    for d in (0.3, 1e-3):
        u = [scipy.linalg.expm(-1j * d * m) for m in frags]
        w = v.astype(complex)
        for ul in u[::-1]:
            w = ul @ w
        for method, ref in (("df-trotter", [w]), ("df-fragments", [ul @ v for ul in u])):
            states = rt.aux_states(act, ci, d, method, df=df5)
            assert len(states) == len(ref)
            for st, r in zip(states, ref):
                assert np.abs(fs.from_ci(st, act.nelec) - r).max() <= 1e-13
            for kind in tg.KINDS:
                el = tg.element_set(act.norb, kind)
                est_ref = -sum(np.array([np.vdot(r, g @ r).imag for g in gam[kind]]) for r in ref) / d
                est = rt.aux_residual(act, ci, d, method, el, kind, df=df5)
                assert np.abs(est - est_ref).max() <= 1e-12, (method, kind, d)
                if d == 1e-3:
                    a_model = rt.exact_residual(model, ci, el, kind)
                    assert np.linalg.norm(est - a_model) <= 1e-4 * np.linalg.norm(a_model)
                    default = rt.aux_residual(act, ci, d, method, el, kind)
                    assert np.linalg.norm(est - default) > 0.3 * np.linalg.norm(rt.exact_residual(act, ci, el, kind))
    exact_states = rt.aux_states(act, ci, 0.3, "exact", df=df5)
    assert np.array_equal(exact_states[0], rt.aux_states(act, ci, 0.3, "exact")[0])


@pytest.mark.parametrize("method", rt.METHODS)
@pytest.mark.parametrize("delta", [0.3, -0.17])
def test_aux_states_match_dense_exponentials(h4, h4_fock, method, delta):
    """aux_states equals scipy.linalg.expm of the dense Fock-space operators (1e-13; observed
    <= 1.5e-15): exp(-i delta (H - ecore)), the product U_0 U_1 ... U_L (U_L applied first) and the
    separate U_l Psi, U_l = exp(-i delta H_l)."""
    act, ci = h4
    fs, v, frags = h4_fock["fs"], h4_fock["v"], h4_fock["frags"]
    u = [scipy.linalg.expm(-1j * delta * m) @ np.eye(fs.dim) for m in frags]
    if method == "exact":
        ref = [scipy.linalg.expm(-1j * delta * (h4_fock["H"] - act.ecore * np.eye(fs.dim))) @ v]
    elif method == "df-trotter":
        w = v.astype(complex)
        for ul in u[::-1]:
            w = ul @ w
        ref = [w]
    else:
        ref = [ul @ v for ul in u]
    states = rt.aux_states(act, ci, delta, method)
    assert len(states) == len(ref) == (1 if method != "df-fragments" else 1 + len(h4_fock["leaves"]))
    for st, r in zip(states, ref):
        assert st.dtype == np.complex128 and st.shape == FCISpace(act.norb, act.nelec).shape
        assert np.abs(fs.from_ci(st, act.nelec) - r).max() <= 1e-13
        assert abs(np.linalg.norm(st) - 1) <= 1e-13


@pytest.mark.parametrize("method", rt.METHODS)
def test_aux_states_obey_the_parity_relation(h4, method):
    """conj Lambda(delta) = Lambda(-delta) for real Psi and real factors (1e-13), so
    A_hat(-delta) = A_hat(delta)."""
    act, ci = h4
    plus = rt.aux_states(act, ci, 0.4, method)
    minus = rt.aux_states(act, ci, -0.4, method)
    for a, b in zip(plus, minus):
        assert np.abs(np.conj(a) - b).max() <= 1e-13
    for kind in tg.KINDS:
        a_plus = rt.aux_residual(act, ci, 0.4, method, kind=kind, states=plus)
        a_minus = rt.aux_residual(act, ci, -0.4, method, kind=kind, states=minus)
        assert np.allclose(a_plus, a_minus, atol=1e-13, rtol=0)


@pytest.mark.parametrize("method", rt.METHODS)
def test_aux_states_propagate_the_normalized_state(h4, method):
    """aux_states propagates Psi = ci / ||ci||: 3 ci gives the same states (1e-14), and a global phase
    of ci multiplies every state by it (the propagators are linear)."""
    act, ci = h4
    ref = rt.aux_states(act, ci, 0.3, method)
    for st, r in zip(rt.aux_states(act, 3.0 * ci, 0.3, method), ref):
        assert np.abs(st - r).max() <= 1e-14
    phase = np.exp(0.7j)
    for st, r in zip(rt.aux_states(act, phase * ci, 0.3, method), ref):
        assert np.abs(st - phase * r).max() <= 1e-14


# ================================================================= bias
@pytest.mark.parametrize("system", ["h4", "h6"])
@pytest.mark.parametrize("method", ["exact", "df-trotter"])
def test_aux_bias_is_second_order(request, system, method):
    """||A_hat(delta) - A|| has log-log slope 2 +- 0.1 over delta in [1e-3, 1e-1] for "exact" and "df-trotter"
    (real factors, real Psi; parity argument in the module docstring), for both element sets.

    Observed relative bias on the H6 iterate at delta = 1e-3: 3.7e-7 (exact) and 2.8e-6 (df-trotter)
    for the spin-orbital set; at delta = 0.1: 3.7e-3 and 2.8e-2.
    """
    act, ci = request.getfixturevalue(system)
    for kind in tg.KINDS:
        exact = rt.exact_residual(act, ci, kind=kind)
        bias = [np.linalg.norm(rt.aux_residual(act, ci, d, method, kind=kind) - exact) for d in DELTAS]
        s = slope(DELTAS, bias)
        assert abs(s - 2) <= 0.1, (system, method, kind, s, bias)
        assert bias[0] / np.linalg.norm(exact) < 1e-5


def test_exact_route_leading_bias_coefficient(h6):
    """(A_hat(delta) - A)/delta^2 -> <[H,[H,[H,Gamma_e]]]>/6 on the H6 iterate (relative 1e-5 at
    delta = 1e-3 for both element sets; observed 2.6e-7), with <ad_H^3 Gamma> =
    <phi_3|Gamma|phi_0> - 3 <phi_2|Gamma|phi_1> + 3 <phi_1|Gamma|phi_2> - <phi_0|Gamma|phi_3>,
    phi_k = H^k Psi, from PySCF transition RDMs."""
    act, ci = h6
    ham = Hamiltonian(act)
    space = ham.space
    phi = [ci / np.linalg.norm(ci)]
    for _ in range(3):
        phi.append(ham.apply(phi[-1]))
    for kind in tg.KINDS:
        el = tg.element_set(act.norb, kind)
        if kind == "spin_free":
            ad3 = sum(c * residual.trans_rdm12(space, phi[a], phi[b])[1] for a, b, c in AD3)
            ref = ad3[tuple(el.T)] / 6
        else:
            ad3 = sum(c * so_transition(act, phi[a], phi[b]) for a, b, c in AD3)
            ref = np.array([ad3[tuple(int(p + s * act.norb) for p, s in row)] for row in el.tolist()]) / 6
        exact = rt.exact_residual(act, ci, el, kind)
        for d, tol in ((1e-3, 1e-5), (1e-2, 1e-4)):
            coeff = (rt.aux_residual(act, ci, d, "exact", el, kind) - exact) / d ** 2
            assert np.linalg.norm(coeff - ref) <= tol * np.linalg.norm(ref), (kind, d)


@pytest.mark.parametrize("kind", tg.KINDS)
def test_fragment_estimates_sum_to_the_first_order_term(h4, h4_fock, kind):
    """df-fragments on H4: the per-fragment first-order terms a_l = <[Gamma_e, H_l]> (Fock oracle) sum to
    A (1e-12); the per-fragment estimates -<Q_e>_{Lambda_l}/delta approach a_l with remainders
    r_l = (delta^2/6) <ad_{H_l}^3(Gamma_e)> + O(delta^4) (coefficients to 1e-5 at delta = 1e-3), and the
    summed remainder (the bias of the route) has slope 2 +- 0.1.

    Observed remainders on the H4 iterate, relative to ||A||: ||sum_l r_l|| = 1.6e-4 (delta = 0.01),
    1.6e-2 (0.1), 0.14 (0.3) for the spin-orbital set (1.7e-4, 1.7e-2, 0.15 spin-free); the largest
    single-fragment ||r_l|| is 1.5 times the summed one.
    """
    act, ci = h4
    v, frags, gam = h4_fock["v"], h4_fock["frags"], h4_fock["gamma"][kind]
    el = tg.element_set(act.norb, kind)
    first = np.array([[fock_expect_comm(v, g, m) for g in gam] for m in frags]).real
    exact = np.array([fock_expect_comm(v, g, h4_fock["H"]) for g in gam]).real
    assert np.abs(first.sum(axis=0) - exact).max() <= 1e-12
    assert np.abs(rt.exact_residual(act, ci, el, kind) - exact).max() <= 1e-12
    coeff_ref = np.array([fock_ad3(v, gam, m) for m in frags]).real / 6
    summed = []
    for d in DELTAS:
        per = rt.aux_residual(act, ci, d, "df-fragments", el, kind, per_state=True)
        assert per.shape == first.shape
        assert np.allclose(per.sum(axis=0), rt.aux_residual(act, ci, d, "df-fragments", el, kind),
                           atol=1e-15, rtol=1e-13)
        rem = per - first
        summed.append(np.linalg.norm(rem.sum(axis=0)))
        if d == DELTAS[0]:
            assert np.linalg.norm(rem / d ** 2 - coeff_ref) <= 1e-5 * np.linalg.norm(coeff_ref)
    assert abs(slope(DELTAS, summed) - 2) <= 0.1
    rel = np.array(summed) / np.linalg.norm(exact)
    assert 1e-4 < rel[2] < 3e-4 and 1e-2 < rel[-1] < 3e-2


# ============================================================== shot cost
@pytest.mark.parametrize("kind", tg.KINDS)
def test_every_auxiliary_state_carries_the_variance_of_psi(h4, h4_fock, kind):
    """"Shot cost" in the module docstring, for every Q_e measured alone in its eigenbasis:
    V(Lambda) = sum_e Var_Lambda(Q_e) / delta^2 equals V(Psi) up to O(delta^2) on the "exact",
    "df-trotter" and every "df-fragments" state, so the optimal allocation over the L + 1 fragment
    states costs (sum_l sqrt V_l)^2 = (L + 1)^2 V_exact up to O(delta^2) (H4: L = 10), including for
    the last leaf, whose first-order term is negligible. Dense Fock matrices and scipy.linalg.expm;
    delta = 0.1 and 0.6.

    Observed (spin-orbital / spin-free): max_l |V_l / V_Psi - 1| / delta^2 = 0.10 / 0.17;
    |(sum_l sqrt V_l)^2 / ((L + 1)^2 V_exact) - 1| / delta^2 <= 0.014 / 0.037; the last leaf has
    ||<[Gamma_e, H_L]>|| = 2e-6 ||A|| (tested <= 1e-4 ||A||), and V_L = V_Psi to 1e-6 (tested).
    """
    act, ci = h4
    v, frags, gam = h4_fock["v"], h4_fock["frags"], h4_fock["gamma"][kind]
    h_el = h4_fock["H"] - act.ecore * np.eye(h4_fock["fs"].dim)
    first = np.array([[fock_expect_comm(v, g, m) for g in gam] for m in frags]).real
    norm_a = np.linalg.norm(first.sum(axis=0))
    for d in (0.1, 0.6):
        u = [scipy.linalg.expm(-1j * d * m) for m in frags]
        w = v.astype(complex)
        for ul in u[::-1]:
            w = ul @ w
        lams = np.stack([v.astype(complex), scipy.linalg.expm(-1j * d * h_el) @ v, w] + [ul @ v for ul in u],
                        axis=1)
        var = np.zeros(lams.shape[1])
        for g in gam:                                   # Q = (Gamma - Gamma^T) / 2i, Gamma real
            ql = (g @ lams - g.T @ lams) / 2j
            var += np.einsum("as,as->s", ql.conj(), ql).real - np.einsum("as,as->s", lams.conj(), ql).real ** 2
        var /= d ** 2
        v_psi, v_exact, v_trotter, v_frag = var[0], var[1], var[2], var[3:]
        assert abs(v_exact / v_psi - 1) <= 0.05 * d ** 2
        assert abs(v_trotter / v_psi - 1) <= 0.2 * d ** 2
        assert np.all(np.abs(v_frag / v_psi - 1) <= 0.3 * d ** 2)
        total = np.sum(np.sqrt(v_frag)) ** 2
        assert abs(total / (len(frags) ** 2 * v_exact) - 1) <= 0.1 * d ** 2
        assert np.linalg.norm(first[-1]) <= 1e-4 * norm_a and abs(v_frag[-1] / v_psi - 1) <= 1e-6


# ===================================================== element-set consistency
@pytest.mark.parametrize("method", rt.METHODS)
def test_spin_free_estimate_is_the_spin_contraction_of_the_spin_orbital_one(h6, method):
    """The spin-free aux estimate is the spin contraction of the spin-orbital one at every delta (1e-12):
    both are linear in Gamma and the (anti)symmetries hold exactly for -Im<Gamma>_Lambda / delta.
    The exact residuals obey the same relation, with compute_residual's A."""
    act, ci = h6
    n = act.norb
    so, sf = tg.spin_orbital_elements(n), tg.spin_free_elements(n)
    a_ref = residual.compute_residual(act, ci).A
    a_so = rt.exact_residual(act, ci, so, "spin_orbital")
    assert np.abs(tg.contract_spin(a_so, so, n) - a_ref).max() <= 1e-12
    for d in (0.1, 0.6):
        states = rt.aux_states(act, ci, d, method)
        a_so = rt.aux_residual(act, ci, d, elements=so, kind="spin_orbital", states=states)
        a_sf = rt.aux_residual(act, ci, d, elements=sf, kind="spin_free", states=states)
        assert np.abs(a_sf).max() > 1e-2
        assert np.abs(tg.contract_spin(a_so, so, n) - tg.spin_free_tensor(a_sf, sf, n)).max() <= 1e-12


def test_exact_residual_vs_fock_oracle(h4, h4_fock, rng):
    """exact_residual equals <Psi|[Gamma_e, H]|Psi> from Fock-oracle matrices for both element sets on the
    H4 iterate and on a random real CI vector (1e-12), and accepts elements in any member order."""
    act, ci = h4
    fs, hf = h4_fock["fs"], h4_fock["H"]
    for psi in (ci, FCISpace(act.norb, act.nelec).random(rng)):
        v = fs.from_ci(psi, act.nelec)
        for kind in tg.KINDS:
            ref = np.array([fock_expect_comm(v, g, hf) for g in h4_fock["gamma"][kind]]).real
            assert np.abs(rt.exact_residual(act, 3.0 * psi, kind=kind) - ref).max() <= 1e-12
    so = tg.spin_orbital_elements(act.norb)
    swapped = so[:, [1, 0, 2, 3]]               # a+_j a+_i = -a+_i a+_j
    assert np.allclose(rt.exact_residual(act, ci, swapped), -rt.exact_residual(act, ci, so), atol=1e-15)
    crossed = so[:, [2, 3, 0, 1]]               # 2A^{kl}_{ij} = -2A^{ij}_{kl} (real state)
    assert np.allclose(rt.exact_residual(act, ci, crossed), -rt.exact_residual(act, ci, so), atol=1e-15)


@pytest.mark.parametrize("method", rt.METHODS)
def test_aux_residual_does_not_depend_on_the_qubit_order(h4, method):
    """Blocked and interleaved mappings give the same estimates (1e-13), and reusing the states and
    the per-state split reproduce the default call."""
    act, ci = h4
    states = rt.aux_states(act, ci, 0.3, method)
    for kind in tg.KINDS:
        ref = rt.aux_residual(act, ci, 0.3, method, kind=kind)
        interleaved = QubitMapping(act.norb, "interleaved")
        other = rt.aux_residual(act, ci, 0.3, method, kind=kind, mapping=interleaved)
        assert np.abs(ref - other).max() <= 1e-13
        assert np.array_equal(rt.aux_residual(act, ci, 0.3, kind=kind, states=states), ref)
        per = rt.aux_residual(act, ci, 0.3, kind=kind, states=states, per_state=True)
        assert per.shape == (len(states), len(ref))
        assert np.allclose(per.sum(axis=0), ref, atol=1e-15, rtol=1e-13)


def test_h6_bias_preview(h6):
    """Relative aux-route bias ||A_hat(inf) - A|| / ||A|| on the H6 iterate (1.0 A, after 2 SD iterations),
    within 2% of the observed values below; ||A|| over the element sets is 0.13259 (840 spin-orbital)
    and 0.25568 (315 spin-free), tested to 1e-4.

    Observed (spin-orbital / spin-free), delta = 0.01, 0.03, 0.1, 0.3, 0.6:
      exact         3.67e-5 / 4.30e-5, 3.30e-4 / 3.87e-4, 3.66e-3 / 4.29e-3, 3.23e-2 / 3.79e-2, 0.122 / 0.142
      df-trotter    2.78e-4 / 2.80e-4, 2.50e-3 / 2.52e-3, 2.77e-2 / 2.79e-2, 0.240 / 0.241,     0.848 / 0.850
      df-fragments  3.35e-4 / 3.53e-4, 3.01e-3 / 3.18e-3, 3.34e-2 / 3.52e-2, 0.295 / 0.311,     1.111 / 1.174
    """
    act, ci = h6
    observed = {
        "exact": ((3.667e-5, 3.300e-4, 3.659e-3, 3.232e-2, 0.1216),
                  (4.299e-5, 3.869e-4, 4.289e-3, 3.786e-2, 0.1420)),
        "df-trotter": ((2.778e-4, 2.500e-3, 2.766e-2, 0.2398, 0.8482),
                       (2.799e-4, 2.518e-3, 2.786e-2, 0.2413, 0.8502)),
        "df-fragments": ((3.345e-4, 3.010e-3, 3.338e-2, 0.2950, 1.111),
                         (3.530e-4, 3.176e-3, 3.522e-2, 0.3114, 1.174)),
    }
    norms = {"spin_orbital": 0.13259, "spin_free": 0.25568}
    exact = {kind: rt.exact_residual(act, ci, kind=kind) for kind in tg.KINDS}
    for kind in tg.KINDS:
        assert abs(np.linalg.norm(exact[kind]) - norms[kind]) <= 1e-4 * norms[kind]
    for method, ref in observed.items():
        for i, d in enumerate((0.01, 0.03, 0.1, 0.3, 0.6)):
            states = rt.aux_states(act, ci, d, method)
            for k, kind in enumerate(tg.KINDS):
                est = rt.aux_residual(act, ci, d, kind=kind, states=states)
                rel = np.linalg.norm(est - exact[kind]) / np.linalg.norm(exact[kind])
                assert abs(rel - ref[k][i]) <= 0.02 * ref[k][i], (method, kind, d, rel)


# ================================================================== shots
def test_shots_for_relative_error():
    """S = sum v / (rel^2 ||A||^2), and with a bias S = sum v / (rel^2 ||A||^2 - ||b||^2) (inf when
    ||b|| >= rel ||A||): at S shots the mean squared error ||b||^2 + sum v / S equals (rel ||A||)^2."""
    a = np.array([1.2, -1.6])                    # ||A|| = 2
    v = np.array([4.0, 6.0])                     # sum v = 10
    assert np.isclose(rt.shots_for_relative_error(v, a, 0.3), 10 / (0.09 * 4))
    assert np.isclose(rt.shots_for_relative_error(10.0, 2.0, 0.3), 10 / 0.36)
    b = np.array([0.18, 0.24])                   # ||b|| = 0.3
    s = rt.shots_for_relative_error(v, a, 0.3, bias=b)
    assert np.isclose(s, 10 / (0.36 - 0.09))
    assert np.isclose(np.sum(b ** 2) + v.sum() / s, (0.3 * 2) ** 2)
    assert np.isclose(rt.shots_for_relative_error(10.0, 2.0, 0.3, bias=0.3), s)
    assert rt.shots_for_relative_error(v, a, 0.3, bias=0.6) == np.inf
    assert rt.shots_for_relative_error(v, a, 0.3, bias=0.7) == np.inf
    assert rt.shots_for_relative_error(v, np.zeros(2), 0.3) == np.inf
    assert rt.shots_for_relative_error(np.zeros(2), a, 0.3) == 0.0
    w = np.array([8.0, 2.0])
    assert np.isclose(rt.shots_for_relative_error(v, a, 0.3, weights=w),
                      (8 * 4 + 2 * 6) / (0.09 * (8 * 1.44 + 2 * 2.56)))
    with pytest.raises(ValueError):
        rt.shots_for_relative_error(v, a, 0.0)
    with pytest.raises(ValueError):
        rt.shots_for_relative_error(10.0, a, 0.3, weights=w)
    with pytest.raises(ValueError):
        rt.shots_for_relative_error(-v, a, 0.3)


def test_shots_weighted_bias_threshold_and_validation(rng):
    """The weights apply to the bias too (closed form, 1e-13); a bias with ||b|| = rel ||A|| to
    round-off is unreachable (inf, not a huge finite S): a budget rel^2 ||A||^2 - ||b||^2 of
    1e-14 rel^2 ||A||^2, positive beyond round-off but below the 1e-12 threshold, gives inf, while
    1e-10 and a bias 0.9999 rel ||A|| give the finite closed form; per-element inputs of different
    shapes, 2-D or non-finite inputs, complex variances and negative weights raise."""
    a = rng.standard_normal(50)
    v = rng.random(50) * 3
    b = 0.05 * rng.standard_normal(50)
    w = rng.integers(1, 9, 50).astype(float)
    s = rt.shots_for_relative_error(v, a, 0.3, bias=b, weights=w)
    assert np.isclose(s, np.sum(w * v) / (0.09 * np.sum(w * a ** 2) - np.sum(w * b ** 2)), rtol=1e-13, atol=0)
    at = b / np.linalg.norm(b) * 0.3 * np.linalg.norm(a)
    assert rt.shots_for_relative_error(v, a, 0.3, bias=at) == np.inf
    assert rt.shots_for_relative_error(v, a, 0.3, bias=1.0001 * at) == np.inf
    assert rt.shots_for_relative_error(v, a, 0.3, bias=at * np.sqrt(1 - 1e-14)) == np.inf
    near = rt.shots_for_relative_error(v, a, 0.3, bias=at * np.sqrt(1 - 1e-10))
    assert np.isclose(near, v.sum() / (1e-10 * 0.09 * np.sum(a ** 2)), rtol=1e-4, atol=0)
    below = rt.shots_for_relative_error(v, a, 0.3, bias=0.9999 * at)
    assert np.isclose(below, v.sum() / (0.09 * np.sum(a ** 2) - np.sum((0.9999 * at) ** 2)), rtol=1e-9, atol=0)
    bad = [((v[:3], a), {}), ((v, a), {"bias": b[:3]}), ((v.sum(), a), {"bias": b[:3]}),
           ((v, a), {"weights": w[:3]}), ((np.ones((2, 3)), np.ones((2, 3))), {}),
           ((np.r_[np.nan, v[1:]], a), {}), ((v, np.r_[np.inf, a[1:]]), {}),
           ((v, a), {"bias": np.r_[np.nan, b[1:]]}), ((v, a), {"weights": np.r_[np.nan, w[1:]]}),
           ((v, a), {"weights": -w}), ((v + 0j, a), {}), ((v, a, np.inf), {}), ((v, a, np.nan), {})]
    for args, kw in bad:
        args = args if len(args) == 3 else args + (0.3,)
        with pytest.raises(ValueError):
            rt.shots_for_relative_error(*args, **kw)


# ============================================================== validation
def test_a_global_phase_of_a_real_state_is_accepted(h4, rng):
    """aux_residual and exact_residual remove a global phase of a real CI vector (every route quantity
    is quadratic in Psi): ci e^{0.3i} and -2i ci give the real-ci results (1e-13). Imaginary noise of
    1e-14 relative is accepted, 1e-12 and a genuinely complex state are rejected. The phase is read
    from the largest amplitude: with a first amplitude of 1e-9, noise of 1e-15 is still accepted (its
    own phase would be off by about 1e-6 and the state would be rejected)."""
    act, ci = h4
    for kind in tg.KINDS:
        ref = rt.exact_residual(act, ci, kind=kind)
        for scaled in (np.exp(0.3j) * ci, -2j * ci):
            assert np.abs(rt.exact_residual(act, scaled, kind=kind) - ref).max() <= 1e-13
        for method in rt.METHODS:
            ref = rt.aux_residual(act, ci, 0.1, method, kind=kind)
            assert np.abs(rt.aux_residual(act, np.exp(0.3j) * ci, 0.1, method, kind=kind) - ref).max() <= 1e-13
    noise = rng.standard_normal(ci.shape)
    assert np.abs(rt.exact_residual(act, ci + 1e-14j * noise) - rt.exact_residual(act, ci)).max() <= 1e-13
    small = ci.copy()
    small.flat[0] = 1e-9
    small /= np.linalg.norm(small)
    noisy = np.exp(0.3j) * (small + 1e-15j * noise)
    assert np.abs(rt.exact_residual(act, noisy) - rt.exact_residual(act, small)).max() <= 1e-13
    for bad in (ci + 1e-12j * noise, ci + 0.1j * noise):
        with pytest.raises(ValueError, match="real"):
            rt.exact_residual(act, bad)
        with pytest.raises(ValueError, match="real"):
            rt.aux_residual(act, np.exp(0.3j) * bad, 0.1)


def test_reused_states_are_checked(h4):
    """aux_residual(states=...) checks the CI-space size and the unit norm (1e-10) of every state."""
    act, ci = h4
    states = rt.aux_states(act, ci, 0.1, "df-fragments")
    assert np.array_equal(rt.aux_residual(act, ci, 0.1, states=[s.ravel() for s in states]),
                          rt.aux_residual(act, ci, 0.1, states=states))
    with pytest.raises(ValueError, match="norm"):
        rt.aux_residual(act, ci, 0.1, states=[states[0], 2.0 * states[1]])
    with pytest.raises(ValueError, match="norm"):
        rt.aux_residual(act, ci, 0.1, states=[states[0] * (1 + 1e-9)])
    with pytest.raises(ValueError, match="entries"):
        rt.aux_residual(act, ci, 0.1, states=[states[0].ravel()[:-1]])
    with pytest.raises(ValueError, match="empty"):
        rt.aux_residual(act, ci, 0.1, states=[])
    assert np.allclose(rt.aux_residual(act, ci, 0.1, states=[states[0] * (1 + 1e-12)]),
                       rt.aux_residual(act, ci, 0.1, states=states[:1]), atol=1e-11, rtol=0)


def test_route_input_validation(h4, rng):
    act, ci = h4
    cplx = ci + 0.1j * rng.standard_normal(ci.shape)
    with pytest.raises(ValueError, match="real"):
        rt.aux_residual(act, cplx, 0.1)
    with pytest.raises(ValueError, match="real"):
        rt.exact_residual(act, cplx)
    assert np.allclose(rt.aux_residual(act, ci.astype(complex), 0.1), rt.aux_residual(act, ci, 0.1))
    with pytest.raises(ValueError, match="method"):
        rt.aux_states(act, ci, 0.1, "trotter")
    with pytest.raises(ValueError, match="delta"):
        rt.aux_residual(act, ci, 0.0)
    with pytest.raises(ValueError, match="kind"):
        rt.aux_residual(act, ci, 0.1, kind="spinfree")
    with pytest.raises(ValueError, match="norb"):
        rt.aux_residual(act, ci, 0.1, mapping=QubitMapping(3))
    with pytest.raises(ValueError):
        rt.exact_residual(act, ci[:3])
    assert rt.METHODS == ("exact", "df-trotter", "df-fragments")
    for method in rt.METHODS:
        n_states = len(rt.aux_states(act, ci, 0.05, method))
        assert n_states == (1 if method != "df-fragments" else 1 + double_factorize(act.eri).n_leaves)


# ========================================================= scheme wiring
def scheme_reference(act, states, ops, scheme, mapping=None):
    """schemes.scheme_variance of the vector target ``ops`` on each CI array of ``states``."""
    mapping = QubitMapping(act.norb) if mapping is None else mapping
    return [sc.scheme_variance(ops, QubitState.from_ci(act.norb, act.nelec, lam, mapping), scheme)
            for lam in states]


@pytest.mark.parametrize("method", rt.METHODS)
def test_aux_variance_is_the_scheme_variance_on_the_auxiliary_states(h4, method):
    """aux_residual_variance = schemes.scheme_variance of [-Q_e / delta] on QubitState.from_ci of each
    auxiliary state, rebuilt here from aux_states and aux_targets (rtol 1e-12), for every scheme and
    both element sets; "df-fragments" combines the L + 1 = 11 states with the optimal allocation
    f_l = sqrt(V_l) / sum sqrt(V_m): total = (sum_l sqrt V_l)^2, per_target = sum_l v_l / f_l, QWC
    n_settings = (L + 1) n_G. The expectation is aux_residual (1e-13), and reused ``states`` give
    the same result."""
    act, ci = h4
    d = 0.3
    states = rt.aux_states(act, ci, d, method)
    n_leaves = None if method == "exact" else double_factorize(act.eri).n_leaves
    for kind in tg.KINDS:
        el = tg.element_set(act.norb, kind)
        ops = [(-q) / d for q in tg.aux_targets(el, QubitMapping(act.norb), kind)]
        est = rt.aux_residual(act, ci, d, method, kind=kind)
        for scheme in sc.SCHEMES:
            res = rt.aux_residual_variance(act, ci, d, method, None, kind, scheme)
            ref = scheme_reference(act, states, ops, scheme)
            v = np.array([r.total for r in ref])
            if method != "df-fragments":
                assert res.scheme == scheme and res.n_targets == len(el)
                assert np.isclose(res.total, v[0], rtol=1e-12, atol=0)
                assert np.allclose(res.per_target, ref[0].per_target, rtol=1e-12, atol=0)
            else:
                f = np.sqrt(v) / np.sqrt(v).sum()
                assert res.scheme == "combined" and len(res.info["parts"]) == n_leaves + 1
                assert np.isclose(res.total, np.sqrt(v).sum() ** 2, rtol=1e-12, atol=0)
                assert np.allclose(res.allocation, f, rtol=1e-12, atol=0)
                per = sum(r.per_target / fl for r, fl in zip(ref, f))
                assert np.allclose(res.per_target, per, rtol=1e-12, atol=0)
            if scheme == "qwc":
                n_g = sc.qwc_grouping(ops).n_groups
                assert res.n_settings == len(states) * n_g
            assert np.abs(res.expectation - est).max() <= 1e-13
            assert res.info["route"] == "aux" and res.info["route_method"] == method
            assert res.info["kind"] == kind and res.info["scheme"] == scheme and res.info["delta"] == d
            assert res.info["order"] == "blocked" and res.info["n_states"] == len(states)
            assert res.info["n_leaves"] == n_leaves
        again = rt.aux_residual_variance(act, ci, d, method, el, kind, "matchgate_shadow", states=states)
        assert np.isclose(again.total, rt.aux_residual_variance(act, ci, d, method, el, kind,
                                                                "matchgate_shadow").total, rtol=1e-13, atol=0)


def test_aux_variance_scales_as_one_over_delta_squared(h4):
    """Every auxiliary state carries the variance of Psi ("Shot cost" in the module docstring): for the
    "exact" route, delta^2 times the per-shot variance of A_hat_e = -Q_hat_e/delta equals the variance of
    Q_e on Psi up to O(delta^2), for every scheme and both qubit orders: |delta^2 V(delta) / V_Q(Psi) - 1|
    <= 0.05 delta^2 for the total and max_e |delta^2 v_e(delta) - v_e^Q(Psi)| <= 0.1 delta^2 max_e v_e^Q
    (delta = 1e-3, 1e-2, 0.1; observed normalized deviations <= 0.0045 and <= 0.023)."""
    act, ci = h4
    for order in ("blocked", "interleaved"):
        m = QubitMapping(act.norb, order)
        psi = QubitState.from_ci(act.norb, act.nelec, ci, m)
        for kind in tg.KINDS:
            q = tg.aux_targets(tg.element_set(act.norb, kind), m, kind)
            for scheme in sc.SCHEMES:
                ref = sc.scheme_variance(q, psi, scheme)
                for d in (1e-3, 1e-2, 0.1):
                    res = rt.aux_residual_variance(act, ci, d, "exact", None, kind, scheme, mapping=m)
                    assert res.info["order"] == order
                    assert abs(res.total * d * d / ref.total - 1) <= 0.05 * d * d, (order, kind, scheme, d)
                    dev = np.abs(res.per_target * d * d - ref.per_target).max()
                    assert dev <= 0.1 * d * d * ref.per_target.max(), (order, kind, scheme, d)


def test_df_fragments_cost_about_L_plus_1_squared_times_one_state(h4):
    """With the schemes' exact variances on H4 (L = 10): total(df-fragments) / ((L + 1)^2 total(exact)) and
    total(df-trotter) / total(exact) are 1 within 0.03 delta^2 at delta = 0.1 and 0.6, for every scheme
    and both element sets (observed: fragments 0.9948-1.0000, Trotter 1.0000-1.0056). A truncated
    factorization (5 leaves) is passed through: 6 states, n_leaves = 5, and the expectation is the
    truncated route's aux_residual (1e-13)."""
    act, ci = h4
    n_leaves = double_factorize(act.eri).n_leaves
    assert n_leaves == 10
    for kind in tg.KINDS:
        for d in (0.1, 0.6):
            for scheme in sc.SCHEMES:
                tot = {m: rt.aux_residual_variance(act, ci, d, m, None, kind, scheme).total for m in rt.METHODS}
                ratio = tot["df-fragments"] / ((n_leaves + 1) ** 2 * tot["exact"])
                assert abs(ratio - 1) <= 0.03 * d * d, (kind, d, scheme, ratio)
                assert abs(tot["df-trotter"] / tot["exact"] - 1) <= 0.03 * d * d, (kind, d, scheme)
    df5 = double_factorize(act.eri, n_leaves=5)
    res = rt.aux_residual_variance(act, ci, 0.3, "df-fragments", None, "spin_free", "pauli_shadow", df=df5)
    assert res.info["n_leaves"] == 5 and res.info["n_states"] == 6 and len(res.allocation) == 6
    est = rt.aux_residual(act, ci, 0.3, "df-fragments", kind="spin_free", df=df5)
    assert np.abs(res.expectation - est).max() <= 1e-13


def test_aux_variance_monte_carlo_end_to_end(h4):
    """scheme_method="mc" simulates the measurements of the folded targets -Q_e/delta on the auxiliary
    states (delta = 0.3): the sampled per-shot totals agree with the exact ones within 4 standard
    errors (QWC on "exact" and "df-fragments", matchgate shadows on "exact"), and the sampled
    estimates of A_hat_e (QWC, a uniform-shot campaign) agree with aux_residual within 4.5 standard
    errors for every element. With several states an integer seed becomes one shared Generator: the
    first fragment's draws equal a single-state call with default_rng(seed), the second's differ
    from a fresh default_rng(seed), and the call is reproducible."""
    act, ci = h4
    d = 0.3
    for kind in tg.KINDS:
        exact = rt.aux_residual_variance(act, ci, d, "exact", None, kind, "qwc")
        mc = rt.aux_residual_variance(act, ci, d, "exact", None, kind, "qwc", scheme_method="mc",
                                      n_shots=4000, seed=11)
        assert mc.method == "mc" and abs(mc.total - exact.total) <= 4 * mc.stderr
        z = (np.asarray(mc.info["sample_mean"]) - exact.expectation) / np.asarray(mc.info["sample_mean_stderr"])
        assert np.abs(z).max() <= 4.5
        assert np.abs(exact.expectation - rt.aux_residual(act, ci, d, "exact", kind=kind)).max() <= 1e-13
    exact = rt.aux_residual_variance(act, ci, d, "exact", None, "spin_orbital", "matchgate_shadow")
    mc = rt.aux_residual_variance(act, ci, d, "exact", None, "spin_orbital", "matchgate_shadow",
                                  scheme_method="mc", n_snapshots=5000, seed=12)
    assert mc.method == "mc" and abs(mc.total - exact.total) <= 4 * mc.stderr
    exact = rt.aux_residual_variance(act, ci, d, "df-fragments", None, "spin_orbital", "qwc")
    kw = dict(scheme_method="mc", n_shots=500, seed=7)
    mc = rt.aux_residual_variance(act, ci, d, "df-fragments", None, "spin_orbital", "qwc", **kw)
    assert mc.method == "mc" and abs(mc.total - exact.total) <= 4 * mc.stderr
    assert rt.aux_residual_variance(act, ci, d, "df-fragments", None, "spin_orbital", "qwc", **kw).total == mc.total
    states = rt.aux_states(act, ci, d, "df-fragments")
    ops = [(-q) / d for q in tg.aux_targets(tg.element_set(act.norb), QubitMapping(act.norb))]
    grouping = sc.qwc_grouping(ops)
    single = [sc.qwc_variance(ops, QubitState.from_ci(act.norb, act.nelec, states[l], QubitMapping(act.norb)),
                              grouping=grouping, method="mc", n_shots=500, seed=np.random.default_rng(7))
              for l in (0, 1)]
    parts = mc.info["parts"]
    assert parts[0].total == single[0].total and parts[1].total != single[1].total


def test_direct_variance_is_the_scheme_variance_of_the_direct_targets(h4):
    """direct_residual_variance = schemes.scheme_variance of targets.direct_targets on Psi (rtol 1e-12),
    with expectation = exact_residual (1e-12), for every scheme and both element sets (subsets of about
    20 elements, and the full spin-free set with Pauli shadows); reused ``targets`` and a prebuilt
    ``hamiltonian`` (with its mapping) give the same result, and a model Hamiltonian (5 of the 10 DF
    leaves) gives the model's residual (1e-12); matchgate shadows do not depend on the qubit order
    (rtol 1e-12); scheme_method="rb" passes through (seeded, n_settings settings, within 4 standard
    errors of the exact value)."""
    act, ci = h4
    m = QubitMapping(act.norb)
    psi = QubitState.from_ci(act.norb, act.nelec, ci, m)
    for kind in tg.KINDS:
        full = tg.element_set(act.norb, kind)
        for scheme in sc.SCHEMES:
            el = full if (kind, scheme) == ("spin_free", "pauli_shadow") else full[::len(full) // 20]
            xs = tg.direct_targets(el, act, m, kind)
            ref = sc.scheme_variance(xs, psi, scheme)
            res = rt.direct_residual_variance(act, ci, None if el is full else el, kind, scheme)
            assert np.isclose(res.total, ref.total, rtol=1e-12, atol=0)
            assert np.allclose(res.per_target, ref.per_target, rtol=1e-12, atol=0)
            assert np.abs(res.expectation - rt.exact_residual(act, ci, el, kind)).max() <= 1e-12
            assert res.info["route"] == "direct" and res.info["kind"] == kind and res.info["scheme"] == scheme
        el = full[:12]
        xs = tg.direct_targets(el, act, m, kind)
        res = rt.direct_residual_variance(act, ci, el, kind, "matchgate_shadow")
        reuse = rt.direct_residual_variance(act, ci, el, kind, "matchgate_shadow", targets=xs)
        prebuilt = rt.direct_residual_variance(act, ci, el, kind, "matchgate_shadow", mapping=m,
                                               hamiltonian=tg.energy_target(act, m))
        il = QubitMapping(act.norb, "interleaved")
        other = rt.direct_residual_variance(act, ci, el, kind, "matchgate_shadow", mapping=il)
        assert other.info["order"] == "interleaved"
        for r in (reuse, prebuilt, other):
            assert np.isclose(r.total, res.total, rtol=1e-12, atol=0)
            assert np.allclose(r.per_target, res.per_target, rtol=1e-12, atol=0)
        model = act.with_eri(double_factorize(act.eri, n_leaves=5).to_eri())
        res_model = rt.direct_residual_variance(act, ci, el, kind, "matchgate_shadow", mapping=m,
                                                hamiltonian=tg.energy_target(model, m))
        a_model = rt.exact_residual(model, ci, el, kind)
        assert np.abs(res_model.expectation - a_model).max() <= 1e-12
        assert np.abs(a_model - res.expectation).max() > 1e-3
        rb = rt.direct_residual_variance(act, ci, el, kind, "matchgate_shadow", scheme_method="rb",
                                         n_settings=1000, seed=3)
        assert rb.method == "rb" and rb.n_samples == 1000
        assert abs(rb.total - res.total) <= 4 * rb.stderr
        assert rt.direct_residual_variance(act, ci, el, kind, "matchgate_shadow", method="rb",
                                           n_settings=1000, seed=3).total == rb.total


def test_variance_wiring_validation(h4):
    """Unknown or energy-only schemes, a mapping of another size, reused states of the wrong number,
    size or norm, a scheme method given twice, a hamiltonian without its mapping, reused targets of the
    wrong number, empty element sets and genuinely complex states raise. Reused states may be any
    iterable (a generator gives the same result as the list; without ``df`` the state count sets L)."""
    act, ci = h4
    so = tg.element_set(act.norb)
    with pytest.raises(ValueError, match="residual routes"):         # before any state is prepared
        rt.aux_residual_variance(act, ci, 0.1, "exact", None, "spin_orbital", "df")
    with pytest.raises(ValueError, match="scheme"):
        rt.direct_residual_variance(act, ci, None, "spin_orbital", "classical")
    with pytest.raises(ValueError, match="norb"):
        rt.aux_residual_variance(act, ci, 0.1, "exact", None, "spin_orbital", "qwc", mapping=QubitMapping(3))
    with pytest.raises(TypeError, match="QubitMapping"):
        rt.direct_residual_variance(act, ci, so[:3], "spin_orbital", "qwc", mapping="blocked")
    frag = rt.aux_states(act, ci, 0.1, "df-fragments")
    with pytest.raises(ValueError, match="1 auxiliary state"):
        rt.aux_residual_variance(act, ci, 0.1, "exact", so[:3], "spin_orbital", "qwc", states=frag[:2])
    with pytest.raises(ValueError, match="11 auxiliary state"):
        rt.aux_residual_variance(act, ci, 0.1, "df-fragments", so[:3], "spin_orbital", "qwc",
                                 states=frag[:5], df=double_factorize(act.eri))
    short = rt.aux_residual_variance(act, ci, 0.1, "df-fragments", so[:3], "spin_orbital", "qwc", states=frag[:5])
    assert short.info["n_states"] == 5 and short.info["n_leaves"] == 4
    lazy = rt.aux_residual_variance(act, ci, 0.1, "df-fragments", so[:3], "spin_orbital", "qwc",
                                    states=(s for s in frag[:5]))              # any iterable of states
    assert lazy.info["n_states"] == 5 and lazy.info["n_leaves"] == 4 and lazy.total == short.total
    with pytest.raises(ValueError, match="entries"):                           # reused states are validated
        rt.aux_residual_variance(act, ci, 0.1, "exact", so[:3], "spin_orbital", "qwc", states=[np.ones(5)])
    with pytest.raises(ValueError, match="norm"):
        rt.aux_residual_variance(act, ci, 0.1, "exact", so[:3], "spin_orbital", "qwc", states=[2 * frag[0]])
    with pytest.raises(TypeError, match="once"):
        rt.direct_residual_variance(act, ci, so[:3], "spin_orbital", "qwc", method="exact", scheme_method="mc")
    with pytest.raises(ValueError, match="mapping"):
        rt.direct_residual_variance(act, ci, so[:3], "spin_orbital", "qwc", hamiltonian=tg.energy_target(act))
    xs = tg.direct_targets(so[:3], act)
    with pytest.raises(ValueError, match="targets"):
        rt.direct_residual_variance(act, ci, so[:4], "spin_orbital", "qwc", targets=xs)
    for fn, args in ((rt.aux_residual_variance, (act, ci, 0.1, "exact")), (rt.direct_residual_variance, (act, ci))):
        with pytest.raises(ValueError, match="element"):
            fn(*args, so[:0], "spin_orbital", "qwc")
        with pytest.raises(ValueError, match="real"):
            fn(act, ci + 0.1j * np.roll(ci, 1), *args[2:], so[:3], "spin_orbital", "qwc")

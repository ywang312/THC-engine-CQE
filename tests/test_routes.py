"""Tests of the residual routes (``thc_engine_cqe.measure.routes``): auxiliary states, infinite-shot
estimates (bias) and shot counts.

References independent of the module: dense Fock-space matrices and ``scipy.linalg.expm`` (H4),
the fragment operators re-derived here from the DF leaves, Fock-oracle commutators, and
<[H,[H,[H,Gamma]]]> from PySCF transition RDMs of H^k Psi.

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
from thc_engine_cqe.measure import targets as tg
from thc_engine_cqe.measure.pauli import QubitMapping
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
    return {"fs": fs, "v": fs.from_ci(ci, act.nelec), "H": fs.hamiltonian(act.h1, act.eri, act.ecore).toarray(),
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


@pytest.mark.parametrize("method", rt.METHODS)
@pytest.mark.parametrize("delta", [0.3, -0.17])
def test_aux_states_match_dense_exponentials(h4, h4_fock, method, delta):
    """aux_states equals scipy.linalg.expm of the dense Fock-space operators (1e-12): exp(-i delta (H - ecore)),
    the product U_0 U_1 ... U_L (U_L applied first) and the separate U_l Psi, U_l = exp(-i delta H_l)."""
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
        assert np.abs(fs.from_ci(st, act.nelec) - r).max() <= 1e-12
        assert abs(np.linalg.norm(st) - 1) <= 1e-13


@pytest.mark.parametrize("method", rt.METHODS)
def test_aux_states_obey_the_parity_relation(h4, method):
    """conj Lambda(delta) = Lambda(-delta) for real Psi and real factors (1e-13), so A_hat(-delta) = A_hat(delta)."""
    act, ci = h4
    plus = rt.aux_states(act, ci, 0.4, method)
    minus = rt.aux_states(act, ci, -0.4, method)
    for a, b in zip(plus, minus):
        assert np.abs(np.conj(a) - b).max() <= 1e-13
    for kind in tg.KINDS:
        assert np.allclose(rt.aux_residual(act, ci, 0.4, method, kind=kind, states=plus),
                           rt.aux_residual(act, ci, -0.4, method, kind=kind, states=minus), atol=1e-13, rtol=0)


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
    assert np.abs(tg.contract_spin(rt.exact_residual(act, ci, so, "spin_orbital"), so, n) - a_ref).max() <= 1e-12
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
        other = rt.aux_residual(act, ci, 0.3, method, kind=kind, mapping=QubitMapping(act.norb, "interleaved"))
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
        "exact": ((3.667e-5, 3.300e-4, 3.659e-3, 3.232e-2, 0.1216), (4.299e-5, 3.869e-4, 4.289e-3, 3.786e-2, 0.1420)),
        "df-trotter": ((2.778e-4, 2.500e-3, 2.766e-2, 0.2398, 0.8482), (2.799e-4, 2.518e-3, 2.786e-2, 0.2413, 0.8502)),
        "df-fragments": ((3.345e-4, 3.010e-3, 3.338e-2, 0.2950, 1.111), (3.530e-4, 3.176e-3, 3.522e-2, 0.3114, 1.174)),
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


# ============================================================== validation
def test_route_input_validation(h4):
    act, ci = h4
    with pytest.raises(ValueError, match="real"):
        rt.aux_residual(act, ci * (1 + 0.5j), 0.1)
    with pytest.raises(ValueError, match="real"):
        rt.exact_residual(act, ci * 1j)
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

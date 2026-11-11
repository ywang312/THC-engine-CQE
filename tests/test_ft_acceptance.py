"""Tests of ft.acceptance: windowed QPE outcome laws, tails, the model spectrum of H' and rule B (F39, F40).

References: ``ft.walk.qpe_outcome_law`` (the Fejer case), direct DFT sums of the register
amplitudes, the closed-form one-sided tail of the sine-window lattice law (trigamma functions),
PySCF FCI of H', eigenpair residuals with the CI Hamiltonian of ``fci_ops``, the Jordan-Wigner
identity coefficient (``measure.pauli``, ``ft.walk``), a dense CISD solve of H', Monte Carlo
sampling of QPE outcomes and of the post-measurement state, and reference numbers of the M4 cost
model [build check] with their tolerances. Systems (``ft.symmetry.benchmark_system``, F20 names):
M3's h4 and n2s (= n2_r2.0), n2_r2.4, n2_r2.8 and h4_r4.0. The QPE of each is its selected
encoding (F9's rule over DF, THC and SF), an SF factorization whose rank is pinned here and
checked by the slow test_selected_encodings.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.special
from pyscf.fci import cistring

from thc_engine_cqe.cqe import run_cqe
from thc_engine_cqe.diagnostics.thc_floors import ExactReference
from thc_engine_cqe.fci_ops import Hamiltonian, normalize
from thc_engine_cqe.ft import acceptance as acc
from thc_engine_cqe.ft import certify
from thc_engine_cqe.ft import qpe as ftqpe
from thc_engine_cqe.ft import symmetry, walk
from thc_engine_cqe.ft.trajectory import second_order_step
from thc_engine_cqe.measure.pauli import jw_hamiltonian

DE = 1e-3
# SF ranks of the selected encodings (F9) [build check]
_SELECTED = {"h4": 7, "n2s": 17, "n2_r2.4": 16, "n2_r2.8": 13, "h4_r4.0": 7}
_CACHE: dict = {}


def _dense(active) -> np.ndarray:
    """Dense sector Hamiltonian (ecore included) from fci_ops, column by column."""
    ham = Hamiltonian(active)
    sp = ham.space
    h = np.column_stack([ham.apply(np.eye(sp.size)[i].reshape(sp.shape)).ravel() for i in range(sp.size)])
    return 0.5 * (h + h.T) + active.ecore * np.eye(sp.size)


def _cisd(active) -> float:
    """Variational CISD: the lowest eigenvalue of H on determinants of excitation level <= 2 from HF."""
    na, nb = active.nelec
    lev = []
    for ne in (na, nb):
        strings = cistring.make_strings(range(active.norb), ne)
        lev.append(np.array([bin(int(s) & ~((1 << ne) - 1)).count("1") for s in strings]))
    sel = (lev[0][:, None] + lev[1][None, :]).ravel() <= 2
    h = _dense(active)
    return float(np.linalg.eigvalsh(h[np.ix_(sel, sel)])[0])


def _setup(name: str) -> dict:
    """active, irreps, selected QPE, model spectrum, CISD(H'), E_0 of H, M = 2 n_iter and 4 n_iter, the grid g."""
    if name not in _CACHE:
        a = symmetry.benchmark_system(name)
        ir = symmetry.orbital_irreps(a)
        enc = ftqpe.sf_encoding(a, _SELECTED[name], reference=ExactReference.build(a))
        q = ftqpe.qpe_cost(enc, DE)
        spec = acc.model_spectrum(a, q, ir)
        m2 = acc.qpe_points(q)
        _CACHE[name] = {"active": a, "irreps": ir, "qpe": q, "spec": spec, "model": enc.model(a),
                        "u": _cisd(enc.model(a)), "e0": float(a.fci()[0]), "m2": m2,
                        "m4": acc.qpe_points(q, 4), "grid": spec.grid(m2)}
    return _CACHE[name]


def _hf(spec) -> np.ndarray:
    ci = np.zeros(spec.shape)
    ci[0, 0] = 1.0
    return ci


def _states(name: str) -> dict:
    """HF, the classical spin-orbital Psi_1 (F16) and the noise-free unit-step Psi_2 (L-BFGS, masked residual)."""
    key = ("states", name)
    if key not in _CACHE:
        s = _setup(name)
        a, ir = s["active"], s["irreps"]
        step = second_order_step(a, metric="spin_orbital", irreps=ir)
        unit = run_cqe(a, max_iter=2, optimizer="lbfgs", metric="spin_orbital", first_eps=step["eps2"],
                       eps_candidates=(1.0,), residual_estimator=lambda rd, ci, it: symmetry.mask_pair_tensor(rd.A, ir))
        _CACHE[key] = {"HF": _hf(s["spec"]), "Psi1": np.asarray(step["state"]), "Psi2": np.asarray(unit.ci)}
    return _CACHE[key]


def _one_branch(theta: float, a: np.ndarray) -> np.ndarray:
    """|A(m|theta)|^2 by a plain FFT of a_x exp(i x theta) (no offset splitting)."""
    amp = np.fft.fft(a * np.exp(1j * theta * np.arange(a.size)))
    return np.abs(amp) ** 2 / a.size


# ------------------------------------------------------------------ the windowed law (F39)
def test_rect_window_equals_ft_walk():
    """H4 (selected SF H'): the rect law at M = 2^t equals ft.walk.qpe_outcome_law for t = 10..14 (<= 1e-10).

    Per eigenstate (eigen_kernel against a one-state law) and for HF and random weights on all 36
    eigenstates; the energy grids agree to round-off.
    """
    s = _setup("h4")
    spec = s["spec"]
    e = spec.energies
    rng = np.random.default_rng(7)
    weights = [spec.weights(_hf(spec)), rng.uniform(0.0, 1.0, e.size) / e.size]
    for t in range(10, 15):
        m = 1 << t
        a = acc.phase_window(m, "rect")
        for w in weights:
            p, grid = acc.windowed_outcome_law(e, w, spec.lam, spec.c, m, window="rect", wmin=0.0)
            ref, ref_grid = walk.qpe_outcome_law(e, w, spec.lam, spec.c, t)
            assert np.max(np.abs(p - ref)) <= 1e-10
            assert np.max(np.abs(grid - ref_grid)) <= 1e-12
        for j in (0, 1, 17, e.size - 1):
            ref, _ = walk.qpe_outcome_law(e[j:j + 1], [1.0], spec.lam, spec.c, t)
            assert np.max(np.abs(acc.eigen_kernel(e[j], spec.lam, spec.c, a) - ref)) <= 1e-10


@pytest.mark.parametrize("window,alpha", [("rect", 0.0), ("sine", 0.0), ("kaiser", 3.0), ("kaiser", 0.0)])
def test_kernels_normalized(window, alpha):
    """Every window is unit-norm and every K_j sums to 1, for M from 2 to 2 n_iter and phases across [0, pi].

    Kaiser(alpha = 0) is the rect window; the sine window is symmetric under x -> M - 1 - x.
    """
    rng = np.random.default_rng(3)
    for m in (2, 3, 64, 1000, 4097, 23848):
        a = acc.phase_window(m, window, alpha)
        assert a.shape == (m,) and abs(float(a @ a) - 1.0) < 1e-13
        if window == "sine":
            assert np.allclose(a, a[::-1], rtol=0, atol=1e-15)
        if window == "kaiser" and alpha == 0.0:
            assert np.allclose(a, acc.phase_window(m, "rect"), rtol=0, atol=1e-15)
        for x in [-1.0, 1.0, math.cos(2 * math.pi * 3 / m), *rng.uniform(-1, 1, 4)]:
            k = acc.eigen_kernel(0.3 + 2.0 * x, 2.0, 0.3, a)
            assert k.shape == (m,) and np.all(k >= 0.0)
            assert abs(float(k.sum()) - 1.0) < 1e-12
    w = rng.uniform(0, 1, 5)
    p, _ = acc.windowed_outcome_law(rng.uniform(-1.5, 1.5, 5), w, 1.6, -0.1, 1000, window=window, alpha=alpha)
    assert abs(float(p.sum()) - float(w.sum())) < 1e-12


@pytest.mark.parametrize("window,alpha", [("sine", 0.0), ("kaiser", 3.0), ("rect", 0.0)])
def test_kernels_against_direct_dft_sums(window, alpha):
    """K_j(m) = (|A(m|theta)|^2 + |A(m|-theta)|^2)/2 with A(m|phi) = M^{-1/2} sum_x a_x e^{ix(phi - 2 pi m/M)}.

    Direct sums at M = 16 and 61 for random phases and phases on and next to the grid; the FFT
    with the integer part of the phase rolled out must reproduce them to 1e-13.
    """
    rng = np.random.default_rng(5)
    for m in (16, 61):
        a = acc.phase_window(m, window, alpha)
        x = np.arange(m)
        for theta in [*rng.uniform(0, math.pi, 5), 2 * math.pi * 5 / m, 2 * math.pi * (5 + 1e-9) / m, 0.0, math.pi]:
            ref = np.zeros(m)
            for phi in (theta, -theta):
                amp = np.array([np.sum(a * np.exp(1j * x * (phi - 2 * math.pi * mm / m))) for mm in range(m)])
                ref += 0.5 * np.abs(amp) ** 2 / m
            k = acc.eigen_kernel(0.2 + 1.5 * math.cos(theta), 1.5, 0.2, a)
            assert np.max(np.abs(k - ref)) < 1e-13


def test_sine_rms_is_de_sin_theta():
    """[build check] n2_r2.4: the sine window's RMS energy error at M = 2 n_iter is dE sin(theta_0) (A3).

    Averaged over 8 phase offsets of a level near E_0': 0.9856 mHa against dE sin(theta_0) =
    0.9857 mHa, within 0.5%; at M = 4 n_iter half of it.
    """
    s = _setup("n2_r2.4")
    spec, q = s["spec"], s["qpe"]
    assert spec.sin_theta0 == pytest.approx(0.9857, abs=1e-4)
    for mult, scale in ((2, 1.0), (4, 0.5)):
        m = mult * q.iterations
        a = acc.phase_window(m, "sine")
        grid = spec.c + spec.lam * np.cos(2 * np.pi * np.arange(m) / m)
        rms = []
        for off in np.linspace(0.0, 1.0, 8, endpoint=False):
            e_t = spec.c + spec.lam * math.cos(spec.theta0 + 2 * math.pi * off / m)
            k = acc.eigen_kernel(e_t, spec.lam, spec.c, a)
            rms.append(math.sqrt(float(k @ (grid - e_t) ** 2)))
        mean = float(np.mean(rms))
        if mult == 2:
            assert mean == pytest.approx(0.9856e-3, abs=1e-7)
        assert mean == pytest.approx(scale * DE * spec.sin_theta0, rel=5e-3)


# ------------------------------------------------------------------ tails
def _sine_tail_closed(d: float) -> float:
    """sup_{s in [0, 1)} sum_{k>=0} P(d + s + k) of the continuum sine-window lattice law, in closed form."""
    x = d + np.linspace(0.0, 1.0, 4001)[:-1]
    s = (scipy.special.polygamma(1, x - 0.5) + scipy.special.polygamma(1, x + 0.5)) / 16.0 - 1.0 / (8.0 * (x - 0.5))
    return float(np.max(8.0 * np.cos(np.pi * x) ** 2 / np.pi ** 2 * s))


def test_fejer_tail_bound():
    """The rect mass at circle distance >= D (both sides) stays below (1 + 1/D)/(2D) for random phases.

    M = 2048, 40 random phases, D in {1, 2, 5, 17, 100}; the one-sided tail_mass is below half the bound.
    """
    rng = np.random.default_rng(3)
    m = 2048
    a = acc.phase_window(m, "rect")
    for _ in range(40):
        theta = rng.uniform(0.3, 2.8)
        k = _one_branch(theta, a)
        dist = np.abs((np.arange(m) - m * theta / (2 * math.pi) + m / 2) % m - m / 2)
        for d in (1, 2, 5, 17, 100):
            assert float(k[dist >= d].sum()) <= acc.fejer_tail_bound(d)
    for d in (1, 2, 5, 17, 100):
        assert acc.tail_mass(d, window="rect") <= 0.5 * acc.fejer_tail_bound(d)
    with pytest.raises(ValueError):
        acc.fejer_tail_bound(0.0)


def test_tail_mass_against_the_closed_form():
    """[derived] tail_mass(sine) at M = 4096 equals the continuum closed form within 3% at D = 1.3 .. 28.7.

    The closed form (module docstring) is checked against a direct lattice sum first; tau_W is
    non-increasing, and the Kaiser(3) tail is far below the sine tail beyond its main lobe.
    """
    x = 5.37
    direct = float(np.sum(8 * np.cos(np.pi * (x + np.arange(2_000_000))) ** 2
                          / (np.pi ** 2 * (1 - 4 * (x + np.arange(2_000_000)) ** 2) ** 2)))
    s = (scipy.special.polygamma(1, x - 0.5) + scipy.special.polygamma(1, x + 0.5)) / 16.0 - 1.0 / (8.0 * (x - 0.5))
    assert 8 * math.cos(math.pi * x) ** 2 / math.pi ** 2 * s == pytest.approx(direct, rel=1e-9)
    assert _sine_tail_closed(5.0) == pytest.approx(1.84e-4, abs=5e-7)
    taus = []
    for d in (1.3, 2.07, 3.0, 5.0, 9.0, 26.0, 28.7):
        tau = acc.tail_mass(d, window="sine")
        assert tau == pytest.approx(_sine_tail_closed(d), rel=0.03)
        taus.append(tau)
    assert all(t1 >= t2 for t1, t2 in zip(taus, taus[1:]))
    assert acc.tail_mass(5.0, window="kaiser", alpha=3.0) < 1e-6 < acc.tail_mass(5.0, window="sine")


def test_verification_runs():
    """[build check] n_v = 1, 2, 3 at D = 28.7, 5, 2.07 (sine, p_min = 1e-3, eps_id = 1e-3), minimal by its definition.

    [(1 - p_min)/p_min] tau_W(D)^{n_v} <= 1e-3 < [(1 - p_min)/p_min] tau_W(D)^{n_v - 1}; tau_W(5) within
    3% of 1.84e-4; B_L's p_0 floors tau/(eps + tau) = 0.155 (one run) and tau^2/(eps + tau^2) = 3.4e-5
    (two runs) [derived]; beyond D = 1000 one run suffices.
    """
    odds = (1 - acc.P_MIN) / acc.P_MIN
    for d, n_ref in ((28.7, 1), (5.0, 2), (2.07, 3)):
        n = acc.verification_runs(d)
        assert n == n_ref
        tau = acc.tail_mass(d, window="sine")
        assert odds * tau ** n <= acc.EPS_ID < odds * tau ** (n - 1)
    tau5 = acc.tail_mass(5.0, window="sine")
    assert tau5 == pytest.approx(1.84e-4, rel=0.03)
    assert tau5 / (acc.EPS_ID + tau5) == pytest.approx(0.155, abs=5e-3)
    assert tau5 ** 2 / (acc.EPS_ID + tau5 ** 2) == pytest.approx(3.4e-5, rel=0.03)
    for d, n_ref in ((26.5, 1), (25.5, 2), (3.2, 2), (2.9, 3), (1.4, 3), (1.15, 5)):     # n_v's steps
        assert acc.verification_runs(d) == n_ref
    assert _sine_tail_closed(28.7) == pytest.approx(7.30e-7, rel=0.01)
    assert acc.verification_runs(5000.0) == 1
    assert acc.verification_runs(5.0, p_min=0.5) == 1           # odds 1: tau_W(5) <= 1e-3
    assert acc.verification_runs(5.0, window="rect") > 2
    with pytest.raises(ValueError):
        acc.verification_runs(0.1)                               # tau_W = 0.85: more than 64 runs


def _jump_tail(d: float, m: int = 4096) -> float:
    """The one-sided sine tail at M points with a lattice point at distance exactly d, by a plain FFT.

    The eigenphase sits s = d - floor(d) grid units below outcome 0, so outcome j lies at distance j + s and
    the outcomes j = floor(d) .. floor(M/2 - s) are those at distances d .. M/2.
    """
    s = d - math.floor(d)
    k = _one_branch(-2.0 * math.pi * s / m, acc.phase_window(m, "sine"))
    return float(k[int(math.floor(d)):int(math.floor(m / 2 - s)) + 1].sum())


def test_tail_mass_is_the_supremum_with_the_jump_point():
    """tau_W(D) bounds the tail at the offset that puts a lattice point at distance D, and for the sine window the
    continuum supremum over x in [D, D + 1) (x = D included): at D = 1.3 the M = 4096 tail at that offset is
    9.352e-3 and the continuum 9.389e-3 (an equispaced offset grid alone gives 9.26e-3); the n_v steps then sit at
    the continuum thresholds, n_v(26.03) = 2 (999 tau = 1.0045e-3), n_v(26.04) = 1, n_v(3.03) = 3, n_v(3.035) = 2;
    the reference values of F40 stay (28.7 -> 1, 5 -> 2, 2.07 -> 3)."""
    jump = _jump_tail(1.3)
    assert jump == pytest.approx(9.352e-3, rel=1e-3)
    assert acc.tail_mass(1.3, window="sine") >= jump
    assert acc.tail_mass(1.3, window="sine") == pytest.approx(_sine_tail_closed(1.3), rel=5e-3)
    assert acc.tail_mass(1.3, window="sine") == pytest.approx(9.389e-3, rel=1e-3)
    for d in (1.15, 2.07, 5.0, 13.2, 26.03, 26.05, 28.7):
        tau = acc.tail_mass(d, window="sine")
        assert tau >= _jump_tail(d) and tau >= _sine_tail_closed(d) * (1 - 1e-6)
    for window, alpha in (("rect", 0.0), ("kaiser", 3.0)):          # the jump point counts for every window
        for d in (1.3, 5.0):
            s = d - math.floor(d)
            k = _one_branch(-2.0 * math.pi * s / 4096, acc.phase_window(4096, window, alpha))
            assert acc.tail_mass(d, window=window, alpha=alpha) >= float(k[int(math.floor(d)):2048].sum()) * (1 - 1e-12)
    odds = (1 - acc.P_MIN) / acc.P_MIN
    assert odds * _sine_tail_closed(26.03) == pytest.approx(1.0045e-3, rel=1e-3)
    for d, n_ref in ((26.03, 2), (26.04, 1), (3.03, 3), (3.035, 2), (28.7, 1), (5.0, 2), (2.07, 3)):
        assert acc.verification_runs(d) == n_ref, d


_T_REF = {1: 26.036, 2: 3.032, 3: 1.295}       # n_v thresholds of the sine window [build check]


def _bisected_threshold(n: int) -> float:
    """min{D : verification_runs(D) <= n} by bisection of verification_runs itself, from a 0.02 bracket."""
    key = ("threshold", n)
    if key not in _CACHE:
        lo, hi = _T_REF[n] - 0.01, _T_REF[n] + 0.01
        assert acc.verification_runs(hi) <= n < acc.verification_runs(lo)
        for _ in range(26):
            mid = 0.5 * (lo + hi)
            if acc.verification_runs(mid) <= n:
                hi = mid
            else:
                lo = mid
        _CACHE[key] = hi
    return _CACHE[key]


def test_verification_run_thresholds():
    """t_n = min{D : verification_runs(D) <= n} against a bisection of verification_runs itself: 26.036, 3.032,
    1.295 for n = 1, 2, 3 (sine, p_min = 1e-3, eps_id = 1e-3); n_v steps from n + 1 to n at t_n."""
    for n in (1, 2, 3):
        t = acc.verification_run_threshold(n)
        assert t == pytest.approx(_bisected_threshold(n), abs=1e-6) and t == pytest.approx(_T_REF[n], abs=1e-3)
        assert acc.verification_runs(t + 1e-6) <= n < acc.verification_runs(t - 1e-6)
    assert acc.verification_run_threshold(1, window="kaiser") < acc.verification_run_threshold(1)
    with pytest.raises(ValueError):
        acc.verification_run_threshold(0)
    with pytest.raises(ValueError):
        acc.verification_run_threshold(65)


def _expected_runs_by_quadrature(u, sigma, e1, de, g, fallback, n: int = 400_001) -> tuple[float, float]:
    """(E[n_v], P(admissible)) by a midpoint rule over delta ~ N(0, sigma^2) on [-12 sigma, 12 sigma].

    n_v(D) = 1 + #{n <= 3 : D < t_n} from the bisected thresholds (the cases put a weight below 1e-6 under
    t_3); B_L's fallback runs where they are fewer or U^ > E1lb - 2 dE.
    """
    edges = np.linspace(-12.0 * sigma, 12.0 * sigma, n)
    delta = 0.5 * (edges[1:] + edges[:-1])
    w = np.exp(-0.5 * (delta / sigma) ** 2)
    w /= w.sum()
    d = (e1 - (u + delta) - de) / g
    adm = u + delta <= e1 - 2.0 * de
    runs = 1.0 + sum((d < _bisected_threshold(k)).astype(float) for k in (1, 2, 3))
    if fallback is None or fallback > 3:
        assert float(w[adm & (d < _bisected_threshold(3))].sum()) < 1e-6
    p_adm = float(w[adm].sum())
    if fallback is not None:
        runs = np.where(adm, np.minimum(runs, fallback), fallback)
        return float(w @ runs), p_adm
    return float(w[adm] @ runs[adm]) / p_adm, p_adm


def test_expected_verification_runs():
    """F40's runs at a realized bound U^ ~ N(U, sigma^2) (F42): against a direct quadrature over delta with n_v from
    independently bisected thresholds (with and without B_L's fallback); a planned D_U at the threshold t_1 with
    fallback n_l = 2 gives exactly 1.5 runs; at n2_r2.8's planned D_U = 26.52 grid units (sigma = 17.1 mHa) a second
    run is needed with probability 0.478, so the certificate's 0.564 C_QPE plus 0.478 C_QPE exceeds the C_QPE of
    B_L's second run."""
    de, g, e1 = 1e-3, 1.9714e-3, -1.0
    u = e1 - de - _bisected_threshold(1) * g
    ex = acc.expected_verification_runs(u, 5e-3, e1, de, grid=g, fallback_runs=2)
    assert ex.runs == pytest.approx(1.5, rel=1e-6) and ex.planned == 1 and ex.fallback == 2
    cases = [(e1 - de - 33.41 * g, 21.6e-3, 2), (e1 - de - 26.52 * g, 17.1e-3, 2), (e1 - de - 9.0 * g, 2.5e-3, 2),
             (e1 - de - 9.0 * g, 2.5e-3, None), (e1 - de - 3.5 * g, 0.85e-3, None), (e1 - de - 2.2 * g, 0.5e-3, 3)]
    for u, sigma, fallback in cases:
        ex = acc.expected_verification_runs(u, sigma, e1, de, grid=g, fallback_runs=fallback)
        runs, p_adm = _expected_runs_by_quadrature(u, sigma, e1, de, g, fallback)
        assert ex.runs == pytest.approx(runs, rel=2e-4, abs=2e-4), (u, sigma, fallback)
        assert ex.admissible == pytest.approx(p_adm, rel=1e-4)
        assert ex.planned == acc.verification_runs((e1 - u - de) / g)
    ex = acc.expected_verification_runs(e1 - de - 26.52 * g, 17.1e-3, e1, de, grid=g, fallback_runs=2)
    assert ex.planned == 1 and ex.runs - 1.0 == pytest.approx(0.478, abs=1e-3) and 0.564 + ex.runs - 1.0 > 1.0
    with pytest.raises(ValueError):
        acc.expected_verification_runs(e1 - 1.5 * de, 1e-3, e1, de, grid=g)          # U > E1lb - 2 dE
    with pytest.raises(ValueError):
        acc.expected_verification_runs(e1 - 0.05, 0.0, e1, de, grid=g)


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def test_expected_runs_where_a_run_count_suffices_nowhere():
    """Run counts without a threshold. The rect window's one-sided tail at D = 1000 is at least that of an eigenphase
    midway between outcomes, sum_{j = 1000}^{M/2 - 1} 1/(M^2 sin^2(pi (j + 1/2)/M)) = 8.06e-5 at M = 4096, and
    999 x 8.06e-5 > eps_id: one run suffices at no distance (beyond D = 1000 the tail is bounded by tau_W(1000)), so
    t_1 = inf and every bound needs a second run: with B_L's three runs as the fallback E[n_v] = 2 + P(D(U^) < t_2),
    t_2 from a bisection of verification_runs. With a fallback the expectation needs no run count at the admissible
    edge a = dE/g: at g = 10 dE (a = 0.1, where no n <= 64 suffices) E[n_v] = 1 + Phi((t_1 - D(U)) g / sigma);
    without one, the edge's run count is needed and the call fails."""
    m = 4096
    j = np.arange(1000, m // 2)
    tail_mid = float(np.sum(1.0 / (m ** 2 * np.sin(np.pi * (j + 0.5) / m) ** 2)))
    assert tail_mid == pytest.approx(8.06e-5, rel=1e-3)
    assert (1 - acc.P_MIN) / acc.P_MIN * tail_mid > acc.EPS_ID
    assert acc.tail_mass(1000.0, window="rect") >= tail_mid * (1 - 1e-9)            # FFT rounding: 4e-12
    assert acc.verification_run_threshold(1, window="rect") == math.inf
    assert acc.verification_runs(5000.0, window="rect") == 2
    lo, hi = 50.0, 200.0
    assert acc.verification_runs(hi, window="rect") == 2 < acc.verification_runs(lo, window="rect")
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if acc.verification_runs(mid, window="rect") <= 2:
            hi = mid
        else:
            lo = mid
    de, g, e1, sigma = 1e-3, 2e-3, -1.0, 5e-3
    for d_plan in (hi, hi + 3.0):
        ex = acc.expected_verification_runs(e1 - de - d_plan * g, sigma, e1, de, grid=g, fallback_runs=3,
                                            window="rect")
        assert ex.runs == pytest.approx(2.0 + _phi((hi - d_plan) * g / sigma), abs=1e-6)
        assert (ex.planned, ex.fallback) == (2, 3)
    g = 10 * de
    with pytest.raises(ValueError):
        acc.verification_runs(de / g)
    u = e1 - de - 27.0 * g
    ex = acc.expected_verification_runs(u, 4e-3, e1, de, grid=g, fallback_runs=2)
    assert ex.runs == pytest.approx(1.0 + _phi((_bisected_threshold(1) - 27.0) * g / 4e-3), abs=1e-6)
    assert ex.planned == 1 and ex.admissible == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        acc.expected_verification_runs(u, 4e-3, e1, de, grid=g)


# ------------------------------------------------------------------ the model spectrum
@pytest.mark.parametrize("name,n_roots", [("h4", 4), ("n2s", 4), ("n2_r2.4", 2)])
def test_model_spectrum_against_fci(name, n_roots):
    """E_0' and E_1' (same-irrep singlet) against PySCF FCI of H' to 1e-8 Ha; eigenpairs, S^2, irreps, c.

    The eigenpairs satisfy ||H' v - E v|| <= 1e-9 with the fci_ops Hamiltonian; <S^2> lies within
    1e-10 of S(S+1); every eigenvector carries its dominant irrep, and on the point-group conserving
    models (h4, n2s) it is irrep-pure (degenerate levels are resolved). [build check] n2_r2.4's SF
    L = 16 model breaks D2h: forbidden integrals up to 1.3e-5 Ha mix near-degenerate pairs of
    different irreps (weights down to 0.9945), while HF's weight outside its irrep stays below 1e-15.
    c equals the Jordan-Wigner identity coefficient of H' to 1e-12; HF's weights sum to 1; n2_r2.4:
    E_1' - E_0' = 230.863 mHa, E_0' - E_0 = -0.368 mHa.
    E_1' is root 1 of every case (FCI converged to 1e-14 Ha: at the default 1e-12 PySCF's root 1 of
    n2_r2.4 lies 2.4e-8 Ha above the dense eigenvalue).
    """
    s = _setup(name)
    spec, model, ir = s["spec"], s["model"], s["irreps"]
    es, cis = model.fci(nroots=n_roots, spin_square=0.0, conv_tol=1e-14)
    det_ir = symmetry.determinant_irreps(model.norb, model.nelec, ir).ravel()
    g0 = symmetry.ci_irrep(cis[0], model.norb, model.nelec, ir)[0]
    e1 = None
    i = 1
    while i < len(es) and e1 is None:                       # M3's rule: degenerate root groups by projection
        j = i + 1
        while j < len(es) and es[j] - es[j - 1] <= 1e-6:
            j += 1
        c = np.column_stack([np.asarray(cis[r]).ravel() for r in range(i, j)])
        c = c[det_ir == g0]
        if np.linalg.eigvalsh(c.T @ c)[-1] > 0.5:
            e1 = float(es[i])
        i = j
    assert spec.e0 == pytest.approx(float(es[0]), abs=1e-8)
    assert spec.e1_singlet == pytest.approx(e1, abs=1e-8)
    assert spec.e1_any_spin <= spec.e1_singlet and spec.e1_any_irrep <= spec.e1_singlet
    ham = Hamiltonian(model)
    for j in range(spec.energies.size):
        v = spec.vectors[:, j]
        hv = ham.apply(v.reshape(spec.shape)).ravel() + model.ecore * v
        assert np.linalg.norm(hv - spec.energies[j] * v) <= 1e-9
        g, weight = symmetry.ci_irrep(v, model.norb, model.nelec, ir)
        assert g == spec.irreps[j] and weight >= (0.994 if name == "n2_r2.4" else 1.0 - 1e-10)
    forbidden = float(np.max(np.abs(model.eri[~symmetry.pair_mask(ir)])))
    w_hf = spec.weights(_hf(spec))
    off_irrep = float(w_hf[spec.irreps != spec.irreps[0]].sum())
    assert off_irrep <= 1e-15
    if name == "n2_r2.4":
        assert 1e-5 <= forbidden <= 1.5e-5
        assert min(symmetry.ci_irrep(spec.vectors[:, j], model.norb, model.nelec, ir)[1]
                   for j in range(spec.energies.size)) <= 0.995
    else:
        assert forbidden <= 1e-14
    allowed = np.array([0.0, 2.0, 6.0, 12.0, 20.0])
    assert np.max(np.min(np.abs(spec.s2[:, None] - allowed[None, :]), axis=1)) <= 1e-10
    op = jw_hamiltonian(model)
    c_jw = float(op.coeffs.real[(op.xs == 0) & (op.zs == 0)].sum())
    assert acc.walk_offset(model) == pytest.approx(c_jw, abs=1e-12) and spec.c == acc.walk_offset(model)
    if name == "h4":
        assert spec.c == pytest.approx(walk.PauliLCUWalk(model).c0, abs=1e-12)
    w = spec.weights(_hf(spec))
    assert float(w.sum()) == pytest.approx(1.0, abs=1e-12)
    assert spec.weights(2.0 * _hf(spec).ravel()) == pytest.approx(w, abs=1e-15)
    assert spec.sin_theta0 == pytest.approx(math.sin(spec.theta0), abs=1e-14)
    if name == "n2_r2.4":
        assert 1e3 * (spec.e1_singlet - spec.e0) == pytest.approx(230.863, abs=1e-3)
        assert 1e3 * (spec.e0 - s["e0"]) == pytest.approx(-0.368, abs=1e-3)


def test_e1_any_irrep_against_fci():
    """The any-irrep singlet E_1' (E1lb of THC and point-group-broken encodings, F41) where it differs from the
    same-irrep one: h4 (SF L = 7) has PySCF FCI singlet root 1 of H' 563.347 mHa above E_0' in another irrep than
    the ground state, below the same-irrep 565.666 mHa; ModelSpectrum and sector_levels equal that root to 1e-8 Ha."""
    s = _setup("h4")
    spec, model, ir = s["spec"], s["model"], s["irreps"]
    es, cis = model.fci(nroots=4, spin_square=0.0, conv_tol=1e-14)
    g0 = symmetry.ci_irrep(cis[0], model.norb, model.nelec, ir)[0]
    assert symmetry.ci_irrep(cis[1], model.norb, model.nelec, ir)[0] != g0
    assert spec.e1_any_irrep == pytest.approx(float(es[1]), abs=1e-8)
    assert 1e3 * (spec.e1_any_irrep - spec.e0) == pytest.approx(563.347, abs=1e-3)
    assert 1e3 * (spec.e1_singlet - spec.e0) == pytest.approx(565.666, abs=1e-3)
    assert spec.e1_any_irrep < spec.e1_singlet - 2e-3
    lv = certify.sector_levels(spec)
    assert lv["e1_any_irrep"] == pytest.approx(float(es[1]), abs=1e-8) and lv["irrep_e1_any_irrep"] != lv["irrep_e0"]
    assert lv["e1_singlet"] == spec.e1_singlet


def test_model_spectrum_reference_numbers():
    """[build check] n2_r2.4's selected QPE, CISD(H') and HF overlap; the gaps and CISD margins of n2_r2.0, 2.8.

    n2_r2.4: lambda = 7.59 Ha, n_iter = 11924, C_QPE = 9,157,632 Toffolis, CISD(H') - E_0' = 225.686 mHa,
    p_0(HF) = 0.11201 on H'. n2_r2.0: E_1' - E_0' = 191.907, CISD(H') - E_0' = 134.215 mHa; n2_r2.8:
    244.441 and 293.495 mHa (no bracket).
    """
    s = _setup("n2_r2.4")
    spec, q = s["spec"], s["qpe"]
    assert (q.encoding.lam, q.iterations, q.toffolis) == (pytest.approx(7.5908, abs=1e-4), 11924, 9157632)
    assert 1e3 * (s["u"] - spec.e0) == pytest.approx(225.686, abs=1e-3)
    assert spec.weights(_hf(spec))[0] == pytest.approx(0.11201, abs=1e-5)
    for name, gap, cisd in (("n2s", 191.907, 134.215), ("n2_r2.8", 244.441, 293.495)):
        t = _setup(name)
        assert 1e3 * (t["spec"].e1_singlet - t["spec"].e0) == pytest.approx(gap, abs=1e-3)
        assert 1e3 * (t["u"] - t["spec"].e0) == pytest.approx(cisd, abs=1e-3)


def test_model_spectrum_limits():
    """Sectors above max_dim have no model spectrum; an encoding without model integrals is rejected."""
    s = _setup("h4")
    with pytest.raises(ValueError):
        acc.model_spectrum(s["active"], s["qpe"], s["irreps"], max_dim=35)
    bare = ftqpe.QPECost(encoding=ftqpe.BlockEncoding(**{**s["qpe"].encoding.__dict__, "eri_model": None}),
                         de_qpe=DE, iterations=s["qpe"].iterations, toffolis=s["qpe"].toffolis,
                         logical_qubits=s["qpe"].logical_qubits)
    with pytest.raises(ValueError):
        acc.model_spectrum(s["active"], bare, s["irreps"])
    with pytest.raises(ValueError):
        s["spec"].weights(np.zeros(s["spec"].shape))
    with pytest.raises(ValueError):
        s["spec"].weights(np.ones(7))


# ------------------------------------------------------------------ rule B (F40)
def test_identification_errors():
    """[build check] eps_id of HF under exact windowed laws on the model spectrum of H' (2% relative unless noted).

    n2_r2.4 under B_U (theta = CISD(H') + dE): sine at 2 n_iter 2.38e-4, Kaiser(3) at 4 n_iter 3.76e-8;
    n2_r2.0 under the literal theta = E_1' - dE: sine at 2 n_iter 0.180; n2_r2.8 under B_L
    (theta = E1lb - 5g, E1lb = E_1' - 0.1 mHa): sine at 2 n_iter 1.6e-4 with one run and 1.0e-8 with
    two (two digits).
    """
    s = _setup("n2_r2.4")
    w = s["spec"].weights(_hf(s["spec"]))
    theta = s["u"] + DE
    assert acc.acceptance_stats(s["spec"], w, theta, n_points=s["m2"]).identification_error == pytest.approx(
        2.38e-4, rel=0.02)
    assert acc.acceptance_stats(s["spec"], w, theta, n_points=s["m4"], window="kaiser",
                                alpha=3.0).identification_error == pytest.approx(3.76e-8, rel=0.02)
    s = _setup("n2s")
    w = s["spec"].weights(_hf(s["spec"]))
    assert acc.acceptance_stats(s["spec"], w, s["spec"].e1_singlet - DE,
                                n_points=s["m2"]).identification_error == pytest.approx(0.180, rel=0.02)
    s = _setup("n2_r2.8")
    spec = s["spec"]
    w = spec.weights(_hf(spec))
    theta = spec.e1_singlet - 1e-4 - 5 * s["grid"]
    one = acc.acceptance_stats(spec, w, theta, n_points=s["m2"]).identification_error
    two = acc.acceptance_stats(spec, w, theta, n_points=s["m2"], n_runs=2).identification_error
    assert (f"{one:.1e}", f"{two:.1e}") == ("1.6e-04", "1.0e-08")


# F40's table, two significant digits: state -> {law: eps_id}; "lit" is theta = E_1' - dE.
_F40_TABLE = {
    ("n2s", "HF"): {"BU_sine": 7.5e-7, "BU_kaiser": 2.3e-9, "BU_rect": 1.3e-3, "BL1": 1.1e-4, "BL2": 9.6e-9,
                    "lit": 0.180},
    ("n2s", "Psi2"): {"BU_sine": 4.9e-7, "BU_kaiser": 1.5e-9, "BU_rect": 6.9e-4, "lit": 0.126},
    ("n2_r2.4", "HF"): {"BU_sine": 2.4e-4, "BU_kaiser": 3.8e-8, "BU_rect": 9.8e-2, "BL1": 1.6e-5, "BL2": 1.0e-10,
                        "lit": 1.9e-3},
    ("n2_r2.4", "Psi1"): {"BU_sine": 1.9e-4, "BU_kaiser": 3.0e-8, "BU_rect": 8.1e-2, "BL1": 1.3e-5,
                          "BL2": 8.5e-11, "lit": 1.5e-3},
    ("n2_r2.4", "Psi2"): {"BU_sine": 1.5e-4, "BU_kaiser": 2.3e-8, "BU_rect": 6.4e-2, "lit": 1.2e-3},
    ("n2_r2.8", "HF"): {"BL1": 1.6e-4, "BL2": 1.0e-8, "lit": 0.322},
    ("n2_r2.8", "Psi1"): {"BL1": 1.3e-4, "BL2": 8.5e-9, "lit": 0.281},
    ("h4_r4.0", "HF"): {"BL1": 4.9e-6, "BL2": 6.9e-10},
}
_P0 = {("n2s", "HF"): 0.25390, ("n2s", "Psi1"): 0.39761, ("n2s", "Psi2"): 0.47300, ("n2_r2.4", "HF"): 0.11201,
       ("n2_r2.4", "Psi1"): 0.21252, ("n2_r2.4", "Psi2"): 0.28103, ("n2_r2.8", "HF"): 0.07815,
       ("n2_r2.8", "Psi1"): 0.16468, ("n2_r2.8", "Psi2"): 0.22942, ("h4_r4.0", "HF"): 0.4341}


def test_identification_error_table():
    """[build check] F40's table: eps_id of HF, the classical Psi_1 and the unit Psi_2 per law, to two digits (5%).

    B_U at theta = CISD(H') + dE (sine 2 n_iter, Kaiser(3) 4 n_iter, rect 2 n_iter), B_L at
    E1lb - 5g with one and two runs, the literal theta = E_1' - dE; overlaps on H' to 1e-4.
    Spectrum-weighted values for these states, not bounds. U = CISD(H') brackets n2_r2.0 and
    n2_r2.4 only.
    """
    for (name, state), row in _F40_TABLE.items():
        s = _setup(name)
        spec = s["spec"]
        w = spec.weights(_states(name)[state])
        assert w[0] == pytest.approx(_P0[name, state], abs=1e-4)
        e1lb = spec.e1_singlet - 1e-4
        assert (s["u"] + DE < e1lb) == (name in ("n2s", "n2_r2.4"))
        laws = {"BU_sine": (s["u"] + DE, s["m2"], "sine", 1), "BU_kaiser": (s["u"] + DE, s["m4"], "kaiser", 1),
                "BU_rect": (s["u"] + DE, s["m2"], "rect", 1), "BL1": (e1lb - 5 * s["grid"], s["m2"], "sine", 1),
                "BL2": (e1lb - 5 * s["grid"], s["m2"], "sine", 2), "lit": (spec.e1_singlet - DE, s["m2"], "sine", 1)}
        for law, ref in row.items():
            theta, m, window, n = laws[law]
            got = acc.acceptance_stats(spec, w, theta, n_points=m, window=window, alpha=3.0,
                                       n_runs=n).identification_error
            assert got == pytest.approx(ref, rel=0.05), (name, state, law, got)
    for name in ("n2s", "n2_r2.4", "n2_r2.8"):
        for state in ("Psi1", "Psi2"):
            assert _setup(name)["spec"].weights(_states(name)[state])[0] == pytest.approx(_P0[name, state], abs=1e-4)


def test_acceptance_matches_monte_carlo():
    """n2_r2.4 HF, 2e5 sampled hand-offs: acceptance and identification within 4 sigma, n_runs = 1 and 2.

    A draw picks an eigenstate j ~ p_j and a walk branch +-theta_j with probability 1/2, then the
    outcome of the first run from |A(m|+-theta_j)|^2; the verification run on the post-measurement
    state draws from the same branch, which reproduces the joint law
    sum_{j,+-} (p_j/2) |A(m_1|+-theta_j)|^2 |A(m_2|+-theta_j)|^2. Thresholds: B_U, the literal
    E_1' - dE and E_1' + dE (where about half of E_1''s outcomes pass).
    """
    s = _setup("n2_r2.4")
    spec, m = s["spec"], s["m2"]
    w = spec.weights(_hf(spec))
    a = acc.phase_window(m, "sine")
    grid = spec.c + spec.lam * np.cos(2 * np.pi * np.arange(m) / m)
    rng = np.random.default_rng(11)
    n_draw = 200_000
    js = np.flatnonzero(w > 1e-14)
    pj = np.repeat(w[js] / w[js].sum(), 2) / 2.0
    counts = rng.multinomial(n_draw, pj / pj.sum())
    draws = []                              # (j, m_1, m_2) per draw
    for idx, cnt in enumerate(counts):
        if cnt == 0:
            continue
        j, sign = js[idx // 2], (1.0 if idx % 2 == 0 else -1.0)
        k = _one_branch(sign * math.acos((spec.energies[j] - spec.c) / spec.lam), a)
        k = k / k.sum()
        draws.append((np.full(cnt, j), rng.choice(m, size=cnt, p=k), rng.choice(m, size=cnt, p=k)))
    jj, m1, m2 = (np.concatenate(col) for col in zip(*draws))
    for theta in (s["u"] + DE, spec.e1_singlet - DE, spec.e1_singlet + DE):
        for n_runs in (1, 2):
            passed = grid[m1] < theta if n_runs == 1 else (grid[m1] < theta) & (grid[m2] < theta)
            st = acc.acceptance_stats(spec, w, theta, n_points=m, n_runs=n_runs)
            p_acc = float(passed.mean())
            assert abs(p_acc - st.accept) <= 4 * math.sqrt(st.accept * (1 - st.accept) / n_draw)
            n_acc = int(passed.sum())
            eps = st.identification_error
            wrong = int(np.sum(passed & (jj != 0)))
            assert abs(wrong / n_acc - eps) <= 4 * math.sqrt(max(eps * (1 - eps), 1.0 / n_acc) / n_acc)
            assert st.ground == pytest.approx(w[0] * st.k0 ** n_runs, rel=1e-12)


def test_identification_bound_holds():
    """F40: exact eps_id <= [(1 - p_0)/p_0] (tau_W(D)/K_0)^n on HF and states down to p_0 = p_min.

    HF populates singlets of the ground irrep only, all at or above E1lb = E_1' - 0.1 mHa; D is the
    exact phase distance of E1lb from theta in grid units. Overlap-scan states
    p psi_0 + (1 - p) HF's excited part at p = 0.1, 0.01, 1e-3: the realizable n_v of the bracket
    keeps eps_id <= 1e-3 at p_0 >= p_min, as F40 states, and one B_L run would not at p_min.
    """
    for name in ("n2s", "n2_r2.4", "n2_r2.8", "h4_r4.0"):
        s = _setup(name)
        spec, m = s["spec"], s["m2"]
        w_hf = spec.weights(_hf(spec))
        e1lb = spec.e1_singlet - 1e-4
        assert np.all(spec.energies[1:][w_hf[1:] > 1e-14] >= e1lb)
        bracket = acc.acceptance_bracket(s["u"] if s["u"] + DE < e1lb else None, e1lb, DE, grid=s["grid"])
        exc = w_hf.copy()
        exc[0] = 0.0
        exc /= exc.sum()
        for p in (None, 0.1, 0.01, 1e-3):
            w = w_hf if p is None else (1 - p) * exc + p * np.eye(exc.size)[0]
            for theta in {bracket.theta, e1lb - 5 * s["grid"]}:
                d = (math.acos((theta - spec.c) / spec.lam) - math.acos((e1lb - spec.c) / spec.lam)) * m / (2 * math.pi)
                tau = acc.tail_mass(d, window="sine")
                for n in (1, 2, 3):
                    st = acc.acceptance_stats(spec, w, theta, n_points=m, n_runs=n)
                    assert st.identification_error <= acc.identification_bound(w[0], tau, st.k0, n)
            st = acc.acceptance_stats(spec, w, bracket.theta, n_points=m, n_runs=bracket.n_runs)
            if p is not None:
                assert st.identification_error <= acc.EPS_ID
        if bracket.form == "B_L":
            w = (1 - acc.P_MIN) * exc + acc.P_MIN * np.eye(exc.size)[0]
            assert acc.acceptance_stats(spec, w, bracket.theta, n_points=m).identification_error > acc.EPS_ID


def test_identification_bound_closed_form():
    """[(1 - p_0)/p_0] (tau/K_0)^n: (0.8/0.2)(1e-3/0.5)^2 = 1.6e-5 for two runs and 8e-3 for one."""
    assert acc.identification_bound(0.2, 1e-3, 0.5, 2) == pytest.approx(1.6e-5, rel=1e-14)
    assert acc.identification_bound(0.2, 1e-3, 0.5, 1) == pytest.approx(8e-3, rel=1e-14)
    assert acc.identification_bound(0.5, 0.1, 1.0, 3) == pytest.approx(1e-3, rel=1e-14)


def _k_below(spec, energy: float, theta: float, m: int) -> float:
    """K_j(A_theta) of one level from eigen_kernel on the outcome grid E(m) = c + lambda cos(2 pi m/M)."""
    grid = spec.c + spec.lam * np.cos(2 * np.pi * np.arange(m) / m)
    return float(acc.eigen_kernel(energy, spec.lam, spec.c, acc.phase_window(m))[grid < theta].sum())


def test_contaminant_term_uses_the_k0_power():
    """F41's contaminant term is q_c/(p_0 K_0^n) for n runs: h4 HF at theta = E_0' + g/2, where K_0 < 0.9, so the
    power matters (rel 1e-9, K_0 from eigen_kernel)."""
    s = _setup("h4")
    spec, m = s["spec"], s["m2"]
    w = spec.weights(_hf(spec))
    theta = spec.e0 + 0.5 * s["grid"]
    k0 = _k_below(spec, spec.e0, theta, m)
    assert 0.1 < k0 < 0.9
    for n in (1, 2):
        base = acc.acceptance_stats(spec, w, theta, n_points=m, n_runs=n)
        sc = acc.acceptance_stats(spec, w, theta, n_points=m, n_runs=n, contaminant=1e-5)
        assert sc.identification_error - base.identification_error == pytest.approx(1e-5 / (w[0] * k0 ** n), rel=1e-9)


def test_small_weights_enter_the_identification_error():
    """Levels with weights in (1e-14, 1e-8] count: 1e-9 on the five lowest excited levels (h4, 258-805 mHa above
    E_0', below theta = E_0' + 0.9 Ha), eps_id = sum_{j>=1} p_j K_j^n / sum_j p_j K_j^n with K_j from eigen_kernel
    (rel 1e-9), n = 1, 2."""
    s = _setup("h4")
    spec, m = s["spec"], s["m2"]
    theta = spec.e0 + 0.9
    low = np.flatnonzero(spec.energies[1:] < theta - 0.02)[:5] + 1
    assert low.size == 5
    w = np.zeros(spec.energies.size)
    w[0] = 0.5
    w[low] = 1e-9
    k = {j: _k_below(spec, float(spec.energies[j]), theta, m) for j in np.r_[0, low]}
    for n in (1, 2):
        exc = sum(1e-9 * k[j] ** n for j in low)
        ref = exc / (0.5 * k[0] ** n + exc)
        st = acc.acceptance_stats(spec, w, theta, n_points=m, n_runs=n)
        assert ref > 1e-9 and st.identification_error == pytest.approx(ref, rel=1e-9)


def test_bracket_forms():
    """F40's forms: B_U certified iff U <= E1lb - 2dE (window model); B_L flagged unless U + dE <= theta.

    Rule E is B_U with U = E_k + dE (theta = E_k + 2dE) and n_v = 1, certified as M3's rule E; the
    bracket takes the form with the smaller n_v, B_U on ties: n2_r2.0 B_U with D_U = 28.7 and one run,
    n2_r2.4 B_L with two runs (B_U at D_U = 2.07 would need three).
    """
    from thc_engine_cqe.ft.handoff import rule_e_certifiable

    e1 = -1.0
    for gap in (2.5e-3, 3.0e-3 - 1e-12, 3.0e-3 + 1e-12, 3.5e-3, 0.2):
        e_k = e1 - gap
        b = acc.acceptance_bracket(e_k + DE, e1, DE, grid=None)
        assert b.n_runs == 1 and "window_model" in b.flags
        if rule_e_certifiable(e_k, e1, DE):
            assert (b.form, b.certified) == ("B_U", True) and b.theta == pytest.approx(e_k + 2 * DE, abs=1e-15)
        else:
            assert (b.form, b.certified) == ("B_L", False) and b.theta == pytest.approx(e1 - DE, abs=1e-15)
            assert "termination_not_certified" in b.flags
    b = acc.acceptance_bracket(None, e1, DE, grid=None)
    assert (b.form, b.theta, b.certified) == ("B_L", e1 - DE, False) and "no_certificate" in b.flags
    # the kernel-aware brackets of the reference systems
    for name, form, n_v, d in (("n2s", "B_U", 1, 28.7), ("n2_r2.4", "B_L", 2, 5.0)):
        s = _setup(name)
        e1lb = s["spec"].e1_singlet - 1e-4
        b = acc.acceptance_bracket(s["u"], e1lb, DE, grid=s["grid"])
        assert (b.form, b.n_runs) == (form, n_v) and b.distance == pytest.approx(d, abs=0.05)
        assert b.lower == pytest.approx(s["u"] + DE, abs=1e-15) and b.upper == pytest.approx(e1lb - 5 * s["grid"])
        if form == "B_U":
            assert b.certified and b.theta == b.lower and not b.flags
        else:
            assert not b.certified and "termination_not_certified" in b.flags
            assert acc.verification_runs((e1lb - s["u"] - DE) / s["grid"]) == 3
    s = _setup("n2_r2.8")
    b = acc.acceptance_bracket(s["u"], s["spec"].e1_singlet - 1e-4, DE, grid=s["grid"])
    assert b.form == "B_L" and {"u_outside_bracket", "termination_not_certified"} <= set(b.flags)
    # synthetic: B_U wins ties; n_v is non-increasing in D, so a U with U + dE <= E1lb - D_L g has
    # n_v(D_U) <= n_v(D_L) and B_U is taken: a chosen B_L is never certified by that U
    g = 2e-3
    b = acc.acceptance_bracket(e1 - 0.06, e1, DE, grid=g)           # D_U = 29.5: one run
    assert (b.form, b.n_runs, b.certified) == ("B_U", 1, True)
    b = acc.acceptance_bracket(e1 - 0.012, e1, DE, grid=g)          # D_U = 5.5: n_v = 2 = B_L's, tie -> B_U
    assert (b.form, b.n_runs, b.certified) == ("B_U", 2, True)
    b = acc.acceptance_bracket(e1 - 0.006, e1, DE, grid=g)          # D_U = 2.5: three runs; B_L with two
    assert (b.form, b.n_runs, b.certified) == ("B_L", 2, False) and "termination_not_certified" in b.flags
    b = acc.acceptance_bracket(e1 - 0.0018, e1, DE, grid=g)         # U > E1lb - 2dE, D_U = 0.4: 27 runs
    assert b.form == "B_L" and b.lower == pytest.approx(e1 - 0.0008)
    b = acc.acceptance_bracket(e1 - 0.0018, e1, DE, grid=g, d_l=0.3)   # D_L = 0.3 needs 38 runs
    assert (b.form, b.certified, b.n_runs) == ("B_U", False, 27)
    b = acc.acceptance_bracket(e1 - 0.0011, e1, DE, grid=g)         # D_U = 0.05: no n <= 64, B_U inadmissible
    assert b.form == "B_L" and "u_outside_bracket" in b.flags


def test_logical_failure_term():
    """F40: f = 1 - theta_0/pi is the share of a uniformly random register below E_0' (n2_r2.4: 0.446).

    The term is f p_run/(P_acc + f p_run) for one run and f p_run for n_runs >= 2, added to eps_id
    and the energy error; failure_budget gives eps_fail p_min/f and eps_fail/(2f); the contaminant
    weight enters as q_c/(p_0 K_0^n).
    """
    s = _setup("n2_r2.4")
    spec, m = s["spec"], s["m2"]
    f = 1 - spec.theta0 / math.pi
    grid = spec.c + spec.lam * np.cos(2 * np.pi * np.arange(m) / m)
    assert f == pytest.approx(0.446, abs=5e-4)
    assert f == pytest.approx(float(np.mean(grid < spec.e0)), abs=2.0 / m)
    w = spec.weights(_hf(spec))
    theta = s["u"] + DE
    for n in (1, 2):
        base = acc.acceptance_stats(spec, w, theta, n_points=m, n_runs=n)
        p_run = acc.failure_budget("unverified" if n == 1 else "retested", f=f)
        st = acc.acceptance_stats(spec, w, theta, n_points=m, n_runs=n, p_run=p_run)
        term = f * p_run / (base.accept + f * p_run) if n == 1 else f * p_run
        assert base.logical == 0.0 and st.logical == pytest.approx(term, rel=1e-12)
        assert st.identification_error == pytest.approx(base.identification_error + term, rel=1e-12)
        assert st.energy_error == pytest.approx(base.energy_error + term, rel=1e-12)
        assert (st.accept, st.ground, st.k0) == (base.accept, base.ground, base.k0)
        q_c = 1e-5
        sc = acc.acceptance_stats(spec, w, theta, n_points=m, n_runs=n, contaminant=q_c)
        assert sc.identification_error == pytest.approx(base.identification_error + q_c / base.ground, rel=1e-12)
    assert acc.failure_budget("unverified", f=f) == pytest.approx(acc.EPS_FAIL * acc.P_MIN / f, rel=1e-15)
    assert acc.failure_budget("retested", f=f) == pytest.approx(acc.EPS_FAIL / (2 * f), rel=1e-15)
    assert acc.failure_budget("unverified") == pytest.approx(1e-6) and acc.failure_budget("retested") == 5e-4
    # at the unverified budget the enriched term is <= eps_fail for every P_acc >= p_min
    for p_acc in (acc.P_MIN, 0.01, 0.5):
        p_run = acc.failure_budget("unverified", f=f)
        assert f * p_run / (p_acc + f * p_run) <= acc.EPS_FAIL


def test_energy_error_and_stats_identities():
    """Energy error = 1 - P(accepted with |E(m) - E_0'| <= dE)/P_acc from the full law; K_0 below theta.

    Checked against windowed_outcome_law on n2_r2.4 HF (one run); the two-run energy error weights the
    last run's window by K_j^{n-1}.
    """
    s = _setup("n2_r2.4")
    spec, m = s["spec"], s["m2"]
    w = spec.weights(_hf(spec))
    theta = spec.e1_singlet - DE
    p, grid = acc.windowed_outcome_law(spec.energies, w, spec.lam, spec.c, m)
    st = acc.acceptance_stats(spec, w, theta, n_points=m)
    below = grid < theta
    assert st.accept == pytest.approx(float(p[below].sum()), rel=1e-12)
    right = float(p[below & (np.abs(grid - spec.e0) <= DE)].sum())
    assert st.energy_error == pytest.approx(1 - right / st.accept, rel=1e-9, abs=1e-15)
    k0 = acc.eigen_kernel(spec.e0, spec.lam, spec.c, acc.phase_window(m))
    assert st.k0 == pytest.approx(float(k0[below].sum()), rel=1e-12)
    st2 = acc.acceptance_stats(spec, w, theta, n_points=m, n_runs=2)
    kj = np.array([acc.eigen_kernel(e, spec.lam, spec.c, acc.phase_window(m)) if wj > 1e-14 else np.zeros(m)
                   for e, wj in zip(spec.energies, w)])
    kb = kj[:, below].sum(axis=1)
    kr = kj[:, below & (np.abs(grid - spec.e0) <= DE)].sum(axis=1)
    assert st2.accept == pytest.approx(float(w @ kb ** 2), rel=1e-12)
    assert st2.energy_error == pytest.approx(1 - float(w @ (kb * kr)) / st2.accept, rel=1e-9)


# ------------------------------------------------------------------ small pieces and validation
def test_qpe_points_and_grid():
    """M = multiple n_iter (A3); g = 2 pi lambda sin(theta_0)/M = 2 dE sin(theta_0) at M = 2 n_iter."""
    s = _setup("n2_r2.4")
    q, spec = s["qpe"], s["spec"]
    assert acc.qpe_points(q) == 2 * q.iterations and acc.qpe_points(q, 4) == 4 * q.iterations
    assert spec.grid(acc.qpe_points(q)) == pytest.approx(2 * DE * spec.sin_theta0, rel=1e-4)
    for name in ("n2s", "n2_r2.4", "n2_r2.8"):                 # exact phase distance E_0' -> E_1' over (E_1' - E_0')/g
        t = _setup(name)
        sp = t["spec"]
        d_exact = (sp.theta0 - math.acos((sp.e1_singlet - sp.c) / sp.lam)) * t["m2"] / (2 * math.pi)
        assert 0.997 <= d_exact / ((sp.e1_singlet - sp.e0) / t["grid"]) <= 0.998
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            acc.qpe_points(q, bad)
    with pytest.raises(ValueError):
        acc.qpe_points(q.encoding)


def test_argument_validation():
    """Invalid inputs raise ValueError; the frozen dataclasses validate their fields."""
    s = _setup("h4")
    spec = s["spec"]
    w = spec.weights(_hf(spec))
    a = acc.phase_window(64)
    bad_calls = [
        lambda: acc.phase_window(1), lambda: acc.phase_window(64, "hann"), lambda: acc.phase_window(64, "kaiser", -1.0),
        lambda: acc.eigen_kernel(3.0, 1.0, 0.0, a), lambda: acc.eigen_kernel(0.1, 0.0, 0.0, a),
        lambda: acc.eigen_kernel(0.1, 1.0, 0.0, 2 * a),
        lambda: acc.windowed_outcome_law([0.1, 0.2], [1.0], 1.0, 0.0, 64),
        lambda: acc.windowed_outcome_law([0.1], [-1.0], 1.0, 0.0, 64),
        lambda: acc.windowed_outcome_law([np.nan], [1.0], 1.0, 0.0, 64),
        lambda: acc.windowed_outcome_law([2.0], [1.0], 1.0, 0.0, 64),
        lambda: acc.tail_mass(-1.0, window="sine"), lambda: acc.tail_mass(3000.0, window="sine"),
        lambda: acc.tail_mass(5.0, window="sine", n_offsets=0),
        lambda: acc.verification_runs(0.0), lambda: acc.verification_runs(5.0, p_min=0.0),
        lambda: acc.verification_runs(5.0, eps_id=1.0),
        lambda: acc.failure_budget("other"), lambda: acc.failure_budget("retested", f=0.0),
        lambda: acc.failure_budget("retested", f=1.5), lambda: acc.failure_budget("unverified", eps_fail=0.0),
        lambda: acc.identification_bound(0.0, 1e-4, 1.0), lambda: acc.identification_bound(0.1, 2.0, 1.0),
        lambda: acc.identification_bound(0.1, 1e-4, 0.0), lambda: acc.identification_bound(0.1, 1e-4, 1.0, 0),
        lambda: acc.acceptance_bracket(None, np.nan, DE, grid=None),
        lambda: acc.acceptance_bracket(None, -1.0, 0.0, grid=None),
        lambda: acc.acceptance_bracket(None, -1.0, DE, grid=-1.0),
        lambda: acc.acceptance_bracket(None, -1.0, DE, grid=1e-3, d_l=0),
        lambda: acc.acceptance_stats(spec, w[:-1], -1.0, n_points=64),
        lambda: acc.acceptance_stats(spec, 2 * w, -1.0, n_points=64),
        lambda: acc.acceptance_stats(spec, w, -1.0, n_points=64, n_runs=0),
        lambda: acc.acceptance_stats(spec, w, -1.0, n_points=64, p_run=1.5),
        lambda: acc.acceptance_stats(spec, w, -1.0, n_points=64, contaminant=-1e-3),
        lambda: acc.acceptance_stats(object(), w, -1.0, n_points=64),
        lambda: acc.walk_offset(object()),
        lambda: acc.Bracket(theta=-1.0, form="B_X", lower=np.nan, upper=-1.0, certified=False, distance=1.0,
                            n_runs=1),
        lambda: acc.Bracket(theta=-1.0, form="B_U", lower=np.nan, upper=-1.0, certified=True, distance=1.0,
                            n_runs=1),
        lambda: acc.Bracket(theta=-1.0, form="B_L", lower=np.nan, upper=-1.0, certified=False, distance=0.0,
                            n_runs=1),
        lambda: acc.Bracket(theta=-1.0, form="B_L", lower=np.nan, upper=-1.0, certified=False, distance=5.0,
                            n_runs=0),
        lambda: acc.AcceptanceStats(accept=0.5, ground=0.6, identification_error=0.1, energy_error=0.1, k0=1.0,
                                    logical=0.0),
        lambda: acc.AcceptanceStats(accept=0.5, ground=0.4, identification_error=np.nan, energy_error=0.1, k0=1.0,
                                    logical=0.0),
        lambda: acc.AcceptanceStats(accept=1.5, ground=0.4, identification_error=0.1, energy_error=0.1, k0=1.0,
                                    logical=0.0),
        lambda: acc.AcceptanceStats(accept=0.5, ground=0.4, identification_error=0.1, energy_error=0.1, k0=1.0,
                                    logical=-1.0),
    ]
    for call in bad_calls:
        with pytest.raises(ValueError):
            call()
    fields = {f: getattr(spec, f) for f in ("energies", "vectors", "s2", "irreps", "shape", "lam", "c", "sin_theta0",
                                            "e0", "e1_singlet", "e1_any_spin", "e1_any_irrep")}
    acc.ModelSpectrum(**fields)
    for key, val in (("energies", spec.energies[::-1]), ("vectors", spec.vectors[:, :3]), ("s2", -spec.s2 - 1.0),
                     ("irreps", spec.irreps + 8), ("shape", (6,)), ("lam", 0.0), ("sin_theta0", 1.5),
                     ("e0", spec.e0 + 1.0), ("e1_singlet", spec.e0 - 1.0), ("e1_any_irrep", np.inf)):
        with pytest.raises(ValueError):
            acc.ModelSpectrum(**{**fields, key: val})
    assert not spec.energies.flags.writeable and not spec.vectors.flags.writeable
    ok = acc.AcceptanceStats(accept=0.0, ground=0.0, identification_error=np.nan, energy_error=np.nan, k0=0.0,
                             logical=0.0)
    assert math.isnan(ok.identification_error)


@pytest.mark.slow
def test_selected_encodings():
    """F9's selection (DF, THC and SF scans) picks the pinned SF ranks: h4 7, n2_r2.0 17, n2_r2.4 16, n2_r2.8 13."""
    for name in ("h4", "n2s", "n2_r2.4", "n2_r2.8", "h4_r4.0"):
        a = symmetry.benchmark_system(name)
        q = ftqpe.select_qpe(ftqpe.scan_encodings(a))
        assert (q.encoding.kind, q.encoding.params["L"]) == ("sf", _SELECTED[name])
        assert q.toffolis == _setup(name)["qpe"].toffolis


@pytest.mark.slow
def test_dense_sector_against_fci_ops_at_n8():
    """h8 (4900 determinants): the model spectrum's lowest eigenpairs against fci_ops and FCI of H' (1e-8 Ha).

    The densest sector of the M4 set; HF's weights sum to 1 and the eigenvalues lie within the walk's
    [c - lambda, c + lambda].
    """
    a = symmetry.benchmark_system("h8")
    ir = symmetry.orbital_irreps(a)
    enc = ftqpe.df_encoding(a, 1e-3, reference=ExactReference.build(a))
    q = ftqpe.qpe_cost(enc, DE)
    spec = acc.model_spectrum(a, q, ir)
    model = enc.model(a)
    assert spec.energies.size == 4900
    assert spec.e0 == pytest.approx(float(model.fci()[0]), abs=1e-8)
    ham = Hamiltonian(model)
    for j in (0, 1, 2, 100, 4899):
        v = spec.vectors[:, j]
        hv = ham.apply(v.reshape(spec.shape)).ravel() + model.ecore * v
        assert np.linalg.norm(hv - spec.energies[j] * v) <= 1e-8
    assert float(spec.weights(normalize(_hf(spec))).sum()) == pytest.approx(1.0, abs=1e-10)
    assert np.all(np.abs(spec.energies - spec.c) <= spec.lam)

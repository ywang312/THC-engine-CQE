"""Tests of ft.qpe: OpenFermion's block-encoding costs, the QR clamp, QPE costs, selection, F8 evolution.

References: OpenFermion's own regression tuples and functions (``compute_cost``,
``utils.QR``, ``sf.compute_lambda``), brute-force minimization, direct FCI of the model
Hamiltonians, and the QPE numbers of spec Appendix A (memo 3 scan).
"""

from __future__ import annotations

import dataclasses
import math
import warnings

import numpy as np
import pytest
import scipy.special

pytest.importorskip("openfermion")

from thc_engine_cqe.diagnostics.thc_floors import ExactReference  # noqa: E402
from thc_engine_cqe.ft import primitives, qpe, symmetry  # noqa: E402
from thc_engine_cqe.ft.primitives import FTParameters  # noqa: E402

# OpenFermion 1.8.1 tests (df/thc/sf compute_cost_*_test.py): inputs and (C_W, total, qubits).
OF_CASES = {
    "df_reiher": (("df", 108, 294.8), dict(n_leaves=360, n_eigenvectors=13031, beta=16),
                  (21753, 10073183463, 3725)),
    "df_li": (("df", 152, 1171.2), dict(n_leaves=394, n_eigenvectors=20115, beta=20), (35011, 64410331887, 6405)),
    "thc_reiher": (("thc", 108, 306.3), dict(nthc=350, beta=16), (10912, 5250145120, 2142)),
    "thc_li": (("thc", 152, 1201.5), dict(nthc=450, beta=20), (16923, 31938980976, 2196)),
    "sf_reiher": (("sf", 108, 4258.0), dict(n_leaves=200), (14184, 94868988984, 3320)),
    "sf_li": (("sf", 152, 3071.8), dict(n_leaves=275), (24396, 117714920508, 3628)),
}
# Appendix A (dE = 1 mHa, selection rule): threshold, L, Xi, lambda_DF, C_W, total, logical qubits.
APPENDIX_DF = {
    "h4": (3e-3, 7, 27, 4.935, 868, 6.73e6, 188),
    "h6a": (3e-3, 11, 63, 9.726, 1309, 2.00e7, 233),
    "n2": (3e-3, 20, 82, 9.423, 1420, 2.10e7, 235),
    "lih": (3e-4, 18, 68, 9.259, 1366, 1.99e7, 235),
}

_CACHE: dict = {}


def _system(name: str):
    if name not in _CACHE:
        a = symmetry.benchmark_system(name)
        _CACHE[name] = (a, ExactReference.build(a))
    return _CACHE[name]


def _df_scan(name: str):
    key = ("df", name)
    if key not in _CACHE:
        a, ref = _system(name)
        _CACHE[key] = qpe.scan_encodings(a, kinds=("df",), reference=ref)
    return _CACHE[key]


@pytest.mark.parametrize("case", sorted(OF_CASES))
def test_openfermion_reference_tuples(case):
    """openfermion_cost (two passes, under the clamp) returns OpenFermion's regression tuples.

    The tuples are asserted in OpenFermion's own tests for these inputs (Reiher and Li
    FeMoco, dE = 1 mHa, chi = 10); none of them needs the clamp.
    """
    (kind, n, lam), kw, expected = OF_CASES[case]
    step, total, qubits, clamped = qpe.openfermion_cost(kind, n, lam, 1e-3, chi=10, **kw)
    assert (step, total, qubits) == expected
    assert clamped is False


def test_qr_clamp_restores_and_flags():
    """The clamp patches QR in the three cost modules only inside the block and restores it.

    Inside: L >= M returns utils.QR's own result with the flag down; L < M returns
    (0, ceil(L)) and raises the flag. After the block (also after an exception) every module
    holds utils.QR again; outside, utils.QR still calls sys.exit for L < M (OpenFermion).
    """
    import openfermion.resource_estimates.utils as of_utils
    from openfermion.resource_estimates.df import compute_cost_df
    from openfermion.resource_estimates.sf import compute_cost_sf
    from openfermion.resource_estimates.thc import compute_cost_thc

    modules = (compute_cost_df, compute_cost_thc, compute_cost_sf)
    original = of_utils.QR
    assert all(m.QR is original for m in modules)
    with qpe.openfermion_qr_clamp() as state:
        assert all(m.QR is not original for m in modules) and of_utils.QR is original
        for L, M in [(10, 3), (100, 7), (4096, 64), (64, 64), (1e6, 17.0)]:
            assert compute_cost_df.QR(L, M) == original(L, M)
        assert state["active"] is False
        assert compute_cost_thc.QR(31, 64) == (0, 31)
        assert state["active"] is True
    assert all(m.QR is original for m in modules)
    with pytest.raises(RuntimeError, match="boom"):
        with qpe.openfermion_qr_clamp():
            raise RuntimeError("boom")
    assert all(m.QR is original for m in modules)
    with pytest.raises(SystemExit):
        original(31, 64)


def test_qr_clamp_value_is_the_integer_optimum():
    """For L < M the clamp's (0, ceil(L)) is the brute-force minimum of L/2^k + M(2^k - 1) over k >= 0."""
    from openfermion.resource_estimates.df import compute_cost_df

    rng = np.random.default_rng(3)
    with qpe.openfermion_qr_clamp():
        for _ in range(200):
            M = int(rng.integers(2, 5000))
            L = float(rng.integers(1, M))
            costs = [L / 2 ** k + M * (2 ** k - 1) for k in range(0, 20)]
            k_best = int(np.argmin(costs))
            assert compute_cost_df.QR(L, M) == (k_best, math.ceil(costs[k_best]))


def test_qpe_cost_matches_openfermion_at_several_errors():
    """iterations = ceil(pi lambda / (2 dE)), toffolis = C_W iterations, qubits = OpenFermion's ancilla_cost.

    The reference for every dE is a fresh OpenFermion call at that dE (DF and SF on h4),
    so the control-register bookkeeping of ancilla_walk is tested on both formulas.
    """
    a, ref = _system("h4")
    for enc in (qpe.df_encoding(a, 3e-3, reference=ref), qpe.sf_encoding(a, 7, reference=ref)):
        kw = (dict(n_leaves=enc.params["L"], n_eigenvectors=enc.params["Lxi"]) if enc.kind == "df"
              else dict(n_leaves=enc.params["L"]))
        for de in (1e-3, 1.6e-3, 3e-4, 1e-2):
            c = qpe.qpe_cost(enc, de)
            assert c.iterations == math.ceil(math.pi * enc.lam / (2 * de))
            assert c.toffolis == enc.toffoli_per_step * c.iterations
            step, total, qubits, _ = qpe.openfermion_cost(enc.kind, 2 * a.norb, enc.lam, de, chi=10, beta=16, **kw)
            assert (step, total, qubits) == (enc.toffoli_per_step, c.toffolis, c.logical_qubits)
        assert qpe.qpe_cost(enc, enc.de_qpe).logical_qubits == enc.ancilla_qpe
    with pytest.raises(ValueError):
        qpe.qpe_cost(enc, 0.0)


@pytest.mark.parametrize("name", ["h4", "h6a", "n2", "lih"])
def test_select_qpe_df_reproduces_appendix_a(name):
    """select_qpe(kind="df") picks Appendix A's threshold, L, Xi, C_W, total (+-0.5%), qubits, lambda (1e-3).

    Appendix A is the memo-3 scan, consistent with OpenFermion; h6a is the spec's named
    regression (3e-3, L = 11, Xi = 63, 1309 per step, 2.00e7). n2 runs on the
    symmetry-adapted orbitals (the spec allows 2%; it reproduces exactly).
    """
    thr, L, Xi, lam, step, total, qubits = APPENDIX_DF[name]
    sel = qpe.select_qpe(_df_scan(name), kind="df")
    e = sel.encoding
    assert e.params["threshold"] == thr and e.params["L"] == L and e.params["Lxi"] == Xi
    assert abs(e.lam - lam) <= 1e-3
    assert e.toffoli_per_step == step and sel.logical_qubits == qubits
    assert abs(sel.toffolis / total - 1.0) <= 5e-3
    assert e.qr_clamped is True and abs(e.de_fact) <= 5e-4 and e.infidelity <= 1e-4


def test_selection_rule_brute_force():
    """select_qpe equals the brute-force minimum over the admissible encodings, per kind and overall.

    The admissible set and the minimum are recomputed here from the BlockEncoding fields.
    Synthetic DF encodings (copies of a scanned one with 1 Toffoli per step, so any
    admissible one wins) test both criteria and their limits: infidelity 2e-4 with
    de_fact 1e-4, and de_fact = -6e-4, must be skipped (in no scan here does an encoding
    with |de_fact| <= 5e-4 have infidelity > 1e-4); de_fact = -5e-4 with infidelity 1e-4
    (both limits, inclusive) is admissible. An empty admissible set raises.
    """
    a, ref = _system("h4")
    real = _df_scan("h4") + [qpe.sf_encoding(a, L, reference=ref) for L in (6, 7, 9)]
    base = real[0]
    skipped = [dataclasses.replace(base, toffoli_per_step=1, de_fact=1e-4, infidelity=2e-4),
               dataclasses.replace(base, toffoli_per_step=1, de_fact=-6e-4, infidelity=0.0)]
    encs = real + skipped
    p = FTParameters()

    def total(e):
        return e.toffoli_per_step * math.ceil(math.pi * e.lam / (2 * p.de_qpe))

    best = {}
    for kind in (None, "df", "sf"):
        ok = [e for e in real if (kind is None or e.kind == kind) and abs(e.de_fact) <= p.de_fact
              and e.infidelity <= p.infid_tol]
        best[kind] = min(ok, key=total)
        assert qpe.select_qpe(encs, kind=kind).encoding is best[kind]
    edge = dataclasses.replace(base, toffoli_per_step=1, de_fact=-5e-4, infidelity=1e-4)
    assert qpe.select_qpe(encs + [edge]).encoding is edge
    assert qpe.select_qpe(encs + [edge], kind="sf").encoding is best["sf"]
    with pytest.raises(ValueError):
        qpe.select_qpe(encs, kind="thc")
    with pytest.raises(ValueError):
        qpe.select_qpe([e for e in encs if e.kind == "df"], params=FTParameters(de_fact=1e-9))


@pytest.mark.parametrize("kind", ["df", "sf"])
def test_control_register_at_powers_of_two(kind, monkeypatch):
    """qpe_cost's qubits equal OpenFermion's ancilla_cost at n_iter = 2^13 - 1, 2^13, 2^13 + 1 (h4).

    Only at powers of two do the DF/THC register 2 ceil(log2(n_iter + 1)) - 1 and the SF
    register 2 ceil(log2 n_iter) - 1 differ. The reference calls OpenFermion's compute_cost
    directly in its two passes (stps = 20000, then the first pass's step cost), with QR
    patched by this test's own k = 0 rule for L < M (h4 needs it), not through ft.qpe.
    """
    import openfermion.resource_estimates.utils as of_utils
    from openfermion.resource_estimates.df import compute_cost_df
    from openfermion.resource_estimates.sf import compute_cost_sf

    a, ref = _system("h4")
    enc = qpe.df_encoding(a, 3e-3, reference=ref) if kind == "df" else qpe.sf_encoding(a, 7, reference=ref)
    mod = compute_cost_df if kind == "df" else compute_cost_sf
    original = of_utils.QR
    monkeypatch.setattr(mod, "QR", lambda L, M: (0, math.ceil(L)) if L < M else original(L, M))
    n, L = 2 * a.norb, enc.params["L"]

    def call(de, stps):
        if kind == "df":
            return mod.compute_cost(n, enc.lam, de, L, enc.params["Lxi"], 10, 16, stps)
        return mod.compute_cost(n, enc.lam, de, L, 10, stps)

    for n_iter in (8191, 8192, 8193):
        de = math.pi * enc.lam / (2 * n_iter) * (1 + 1e-12)
        c = qpe.qpe_cost(enc, de)
        assert c.iterations == n_iter
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            step, total, qubits = call(de, int(call(de, 20000)[0]))
        assert (step, total, qubits) == (enc.toffoli_per_step, c.toffolis, c.logical_qubits)


def test_sf_scan_covers_the_full_supermatrix_rank():
    """The default SF scan runs L = 1 .. the rank of the ERI supermatrix, and the full-rank model is exact.

    The rank is counted here from eigvalsh of the (N^2, N^2) supermatrix (eigenvalues above
    1e-12 of the largest); at full rank eri_model reproduces the exact ERIs and de_fact vanishes.
    """
    a, ref = _system("h4")
    w = np.linalg.eigvalsh(a.eri.reshape(a.norb ** 2, a.norb ** 2))
    rank = int(np.sum(w > 1e-12 * w.max()))
    scan = qpe.scan_encodings(a, kinds=("sf",), reference=ref)
    assert [e.params["L"] for e in scan] == list(range(1, rank + 1))
    assert np.max(np.abs(scan[-1].eri_model - a.eri)) < 1e-10 and abs(scan[-1].de_fact) < 1e-9


def test_reference_must_match_the_integrals():
    """An ExactReference of other integrals is rejected: same N and electrons, but another h1, ERIs or ecore.

    Cases: h4 with only h1, only the ERIs or only ecore changed, and h6s against the
    reference of h6a (another geometry). A reference of the wrong Hamiltonian would give
    de_fact and the infidelity against the wrong ground state.
    """
    a, ref = _system("h4")
    variants = [dataclasses.replace(a, h1=1.01 * a.h1), a.with_eri(1.01 * a.eri),
                dataclasses.replace(a, ecore=a.ecore + 1e-3)]
    h6a, h6s = symmetry.benchmark_system("h6a"), symmetry.benchmark_system("h6s")
    assert (h6s.norb, tuple(h6s.nelec)) == (h6a.norb, tuple(h6a.nelec))
    ref6 = ExactReference.build(h6a)
    builds = [lambda v=v: qpe.df_encoding(v, 1e-2, reference=ref) for v in variants]
    builds += [lambda: qpe.sf_encoding(variants[0], 3, reference=ref),
               lambda: qpe.df_encoding(h6s, 1e-2, reference=ref6),
               lambda: qpe.thc_encoding(h6s, 6, penalty=0.0, reference=ref6)]
    for build in builds:
        with pytest.raises(ValueError, match="same active space"):
            build()


def test_encoding_fields_against_direct_fci():
    """de_fact and the model integrals: E_0(H') - E_0(H) recomputed by FCI of active.with_eri(eri_model).

    Also lambda against the DF/SF formulas of the factor module (T' from the exact ERIs) and
    the model method; this is independent of diagnostics.thc_floors, which the module uses.
    """
    a, ref = _system("h4")
    e0 = a.fci()[0]
    for enc in (qpe.df_encoding(a, 1e-2, reference=ref), qpe.sf_encoding(a, 4, reference=ref)):
        e0m = enc.model(a).fci()[0]
        assert abs((e0m - e0) - enc.de_fact) < 1e-9
        assert abs(enc.lam - enc.lam_one_body - enc.lam_two_body) < 1e-12
    with pytest.raises(ValueError):
        qpe.sf_encoding(a, 99, reference=ref)
    with pytest.raises(ValueError):
        qpe.df_encoding(a, 1e3, reference=ref)
    with pytest.raises(ValueError):
        qpe.df_encoding(symmetry.benchmark_system("h6a"), 1e-3, reference=ref)


def test_thc_h6a_appendix_a():
    """THC on h6a at M = 15 with the auto penalty: lambda 9.125 (1e-3), 1035 per step, 1.48e7 total (+-0.5%).

    Appendix A's THC row (memo 3); the encoding is admissible under the selection rule.
    """
    a, ref = _system("h6a")
    enc = qpe.thc_encoding(a, 15, penalty="auto", reference=ref)
    c = qpe.qpe_cost(enc)
    assert abs(enc.lam - 9.125) <= 1e-3
    assert enc.toffoli_per_step == 1035 and abs(c.toffolis / 1.48e7 - 1.0) <= 5e-3
    assert abs(enc.de_fact) <= 5e-4 and enc.infidelity <= 1e-4 and not enc.qr_clamped


@pytest.mark.slow
def test_thc_scan_h6a_selects_m15():
    """The default THC scan of h6a (ranks 6-21, penalties 0 and auto) selects M = 15 with the auto penalty."""
    a, ref = _system("h6a")
    sel = qpe.select_qpe(qpe.scan_encodings(a, kinds=("thc",), reference=ref), kind="thc")
    assert sel.encoding.params["M"] == 15 and sel.encoding.params["penalty"] == "auto"
    assert abs(sel.toffolis / 1.48e7 - 1.0) <= 5e-3


def test_sf_lambda_matches_openfermion_on_random_factors(monkeypatch):
    """sf_lambda equals OpenFermion's own ``sf.compute_lambda`` on random real and complex factors (h4 integrals).

    OpenFermion takes |L| elementwise, the modulus for complex factors. Its module imports
    ``molecule.pyscf_to_cas``, which this environment lacks (it needs jax); it is stubbed to
    return the h4 integrals, so OpenFermion's code computes T and both lambda terms itself.
    Factor layouts are (N, N, n) there and (n, N, N) here.
    """
    import importlib
    import sys

    import openfermion.resource_estimates.molecule as of_molecule
    import openfermion.resource_estimates.sf as of_sf

    a, _ = _system("h4")
    name = "openfermion.resource_estimates.sf.compute_lambda_sf"
    monkeypatch.setattr(of_molecule, "pyscf_to_cas", lambda mf: (a.h1, a.eri, a.ecore) + a.nelec, raising=False)
    monkeypatch.delitem(sys.modules, name, raising=False)
    try:
        of_lambda = importlib.import_module(name).compute_lambda
        rng = np.random.default_rng(5)
        for rank in (1, 3, 7):
            for f in (rng.standard_normal((a.norb, a.norb, rank)),
                      rng.standard_normal((a.norb, a.norb, rank)) + 1j * rng.standard_normal((a.norb, a.norb, rank))):
                lam, lam_t, lam_w = qpe.sf_lambda(a.h1, a.eri, f.transpose(2, 0, 1))
                assert abs(lam - of_lambda(None, f)) <= 1e-12 * lam
                assert abs(lam - lam_t - lam_w) < 1e-12
    finally:
        sys.modules.pop(name, None)                 # do not leave the stubbed module behind
        if hasattr(of_sf, "compute_lambda_sf"):
            delattr(of_sf, "compute_lambda_sf")
    with pytest.raises(ValueError):
        qpe.sf_lambda(a.h1, a.eri, f)


def test_evolution_cost_and_walk_params():
    """F8 on a DF encoding: R from primitives, 2R calls, C_evol, Q_walk + 1; walk_params carries C_W and Q_walk.

    The references are the primitives' formula and, at t = 1.4 (tau = 6.9), a hand Bessel
    sum: R = min{R : 2 sum_{j>R} |J_j(tau)| <= eps_evol} and C_evol = 2R C_W + 3(2R + 1)(beta - 2).
    WalkParams is task B's type, imported only if ft.residual is present.
    """
    a, ref = _system("h4")
    enc = qpe.df_encoding(a, 3e-3, reference=ref)
    p = FTParameters()
    for t in (0.0, 0.01, 0.1, 1.4):
        ev = qpe.evolution_cost(enc, t)
        R = primitives.jacobi_anger_degree(enc.lam * t, p.eps_evol)
        assert ev["degree"] == R and ev["walk_calls"] == 2 * R
        assert ev["toffolis"] == primitives.evolution_toffolis(enc.lam, t, enc.toffoli_per_step,
                                                               delta=p.eps_evol, beta=p.beta)
        assert ev["logical_qubits"] == enc.ancilla_walk + 1
    assert enc.ancilla_walk == enc.ancilla_qpe - (2 * math.ceil(math.log2(qpe.qpe_cost(enc).iterations + 1)) - 1)
    t = 1.4                                       # by hand: R from the Bessel tail, then C_evol of F8
    tail = 2.0 * np.cumsum(np.abs(scipy.special.jv(np.arange(300), enc.lam * t))[::-1])[::-1]
    R = int(np.flatnonzero(np.append(tail[1:], 0.0) <= p.eps_evol)[0])
    ev = qpe.evolution_cost(enc, t)
    assert ev["degree"] == R and ev["toffolis"] == 2 * R * enc.toffoli_per_step + 3 * (2 * R + 1) * (p.beta - 2)
    pytest.importorskip("thc_engine_cqe.ft.residual")
    w = enc.walk_params(fact_bias_m=0.25)
    assert (w.lam, w.toffoli_per_step, w.logical_qubits, w.kind, w.fact_bias_m) == (
        enc.lam, enc.toffoli_per_step, enc.ancilla_walk, "df", 0.25)


def test_argument_checks_and_hashing():
    """Invalid arguments raise ValueError before OpenFermion is called; encodings and QPE costs are hashable.

    Covered: unknown kinds, missing, zero, non-integral or boolean ranks, odd or non-integral
    2N, lambda or dE that are zero, negative, infinite or nan (qpe_cost at dE = inf used to
    return 0 iterations), non-integral SF and THC ranks (2.7 used to be read as 2), a nan
    DF threshold. Equal encodings hash equal (params and eri_model are not hashed).
    """
    a, ref = _system("h4")
    enc = qpe.df_encoding(a, 3e-3, reference=ref)
    df_kw = dict(n_leaves=7, n_eigenvectors=27)
    with pytest.raises(ValueError):
        qpe.openfermion_cost("cd", 8, 1.0, 1e-3)
    with pytest.raises(ValueError):
        qpe.openfermion_cost("df", 8, 1.0, 1e-3, n_leaves=3)
    with pytest.raises(ValueError):
        qpe.openfermion_cost("thc", 8, 1.0, 1e-3)
    for bad in (0.0, -1e-3, math.inf, math.nan):
        with pytest.raises(ValueError):
            qpe.qpe_cost(enc, bad)
        with pytest.raises(ValueError):
            qpe.openfermion_cost("df", 8, 4.9, bad, **df_kw)
        with pytest.raises(ValueError):
            qpe.openfermion_cost("df", 8, bad, 1e-3, **df_kw)
        with pytest.raises(ValueError):
            qpe.qpe_cost(dataclasses.replace(enc, lam=bad), 1e-3)
    for n in (9, 7.5, 0):
        with pytest.raises(ValueError):
            qpe.openfermion_cost("df", n, 4.9, 1e-3, **df_kw)
    for rank in (0, 2.5, True):
        with pytest.raises(ValueError):
            qpe.openfermion_cost("sf", 8, 4.9, 1e-3, n_leaves=rank)
        with pytest.raises(ValueError):
            qpe.openfermion_cost("thc", 8, 4.9, 1e-3, nthc=rank)
        with pytest.raises(ValueError):
            qpe.openfermion_cost("df", 8, 4.9, 1e-3, n_leaves=7, n_eigenvectors=rank)
    for bad in (0, 2.7, 99):
        with pytest.raises(ValueError):
            qpe.sf_encoding(a, bad, reference=ref)
    with pytest.raises(ValueError):
        qpe.thc_encoding(a, 2.5, penalty=0.0, reference=ref)
    with pytest.raises(ValueError):
        qpe.df_encoding(a, math.nan, reference=ref)
    with pytest.raises(ValueError):
        qpe.scan_encodings(symmetry.benchmark_system("h4"), kinds=("cd",))
    twin = dataclasses.replace(enc)
    assert twin == enc and hash(twin) == hash(enc) and len({enc, twin}) == 1
    assert hash(qpe.qpe_cost(enc)) == hash(qpe.qpe_cost(twin))

"""Tests of the scaling layer (``thc_engine_cqe.ft.scaling``; F20-F25, F44, F45) and of its inputs (F24).

F24 (ii): ``ft.residual_inputs.variance_totals_subset`` against the per-element variances of the full
element set (``element_variances``, exact pair sums on H4): the scaled subset sum, its finite-population
standard error and the uniform per-element values are recomputed by hand from the returned subset.
F21: ``spectrum`` against the gaps and overlaps of the design [design], against single-root FCI
solves and against M3's records (``exact_trajectory`` with its own roots, bit for bit); the point-group
selection rule of the square H4. The stage cache against its digest and atomic-write rules (on tmp_path);
``to_json`` against hand-built strict JSON.
F22, F21 (fits), F23: synthetic exponentials, geometric drifts (the Aitken limit exactly), power laws and 1/n
gaps (gamma from the two largest sizes); the H chains at R = 3.0 bohr [design]; the H10 holdout from n <= 8
(slow) [design]. F21/F40: the one-run B_L margin D_L(p) by its definition and the one-run window against a direct
scan of its condition (closing, constant and growing gaps). F44: the marginal condition against
``handoff.handoff_table`` on random cases. F45: against an independent pure-Python scan (the closed-form ln
rho_k(n) without preparation terms; mu = H_s(p_k, c_k) - H_s(p_k+1, c_k+1) with the hand-off QPE lengthened
beyond the window otherwise), the preparation terms, both rates, the crossed corners, the filtered schemes,
overlaps below 1e-300, the window of each case's own gap model; no n* without gain. The report's inputs from
stand-in records (U2's fit window, p_1 there and the B_L side column; U4's alpha, cost and preparation fits).
Costs: one residual problem per iterate for every r equals ``pipeline.iteration_costs`` and
``ft.steprules.step_rule_iteration_costs``; the closed-form direct route equals M3's direct_mg. F25: the 1/s law
on the h4_r4.0 exact-line-search step [design] and the slice rule up to s = 16 from its own state against direct
``circuits.compiled_step`` calls. The per-system scan (cache reuse, the N = 10 path on a small system) and the
report U1-U4 against the records. Regression pins at N = 8 and 10 (slow) [design].
"""

import dataclasses
import functools
import json
import math
import os
import pathlib
import pickle
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from thc_engine_cqe.fci_ops import FCISpace
from thc_engine_cqe.ft import residual_inputs as ri
from thc_engine_cqe.ft import scaling
from thc_engine_cqe.ft import symmetry as sy
from thc_engine_cqe.ft.primitives import FTParameters
from thc_engine_cqe.ft.trajectory import exact_trajectory
from thc_engine_cqe.measure import targets
from thc_engine_cqe.steprules import StepEvent, StepRule, run_step_rule


@pytest.fixture(scope="module")
def h4_hf_variances():
    act = sy.benchmark_system("h4_r4.0")
    ir = sy.orbital_irreps(act)
    el, _ = ri.allowed_elements(act, "spin_free", ir)
    hf = FCISpace.of(act).hf()
    return act, el, hf, ri.element_variances(act, hf, el, "spin_free")


@pytest.mark.parametrize("k_subset, seed", [(10, 11), (17, 3), (27, 5)])
def test_subset_totals_formula(h4_hf_variances, k_subset, seed):
    """V^_m = (K/K_s) sum_S m_e v_e and W^_m likewise, from the full set's v_e on the returned subset (exact pair
    sums: a subset's per-element variances are the full set's); the standard error is
    (K/K_s) sqrt(K_s) sd(m_e v_e) sqrt(1 - K_s/K); the uniform per-element values reproduce the totals; the
    subset is default_rng(seed).choice(K, K_s, replace=False), sorted."""
    act, el, hf, full = h4_hf_variances
    k = len(el)
    m = targets.element_multiplicities(el, "spin_free", act.norb).astype(float)
    out = ri.variance_totals_subset(act, hf, el, "spin_free", k_subset=k_subset, seed=seed)
    sub = np.sort(np.random.default_rng(seed).choice(k, size=k_subset, replace=False))
    assert np.array_equal(out["subset"], sub)
    assert (out["k_subset"], out["subset_seed"], out["n_elements"], out["method"], out["subset_method"]) == \
        (k_subset, seed, k, "subset", "exact")
    x_v, x_w = m[sub] * full["v_direct"][sub], m[sub] * full["w_aux"][sub]
    assert out["variance_m"] == pytest.approx(k / k_subset * x_v.sum(), rel=1e-12)
    assert out["w_aux_m"] == pytest.approx(k / k_subset * x_w.sum(), rel=1e-12)
    fpc = math.sqrt(1.0 - k_subset / k)
    assert out["stderr_m"] == pytest.approx(k / k_subset * math.sqrt(k_subset) * x_v.std(ddof=1) * fpc, rel=1e-12)
    assert out["w_stderr_m"] == pytest.approx(k / k_subset * math.sqrt(k_subset) * x_w.std(ddof=1) * fpc, rel=1e-12)
    assert out["stderr_m_estimator"] == 0.0
    assert out["v_direct"].shape == out["w_aux"].shape == (k,)
    assert np.sum(m * out["v_direct"]) == pytest.approx(out["variance_m"], rel=1e-12)
    assert np.sum(m * out["w_aux"]) == pytest.approx(out["w_aux_m"], rel=1e-12)
    assert np.ptp(out["v_direct"]) == 0.0 and np.ptp(out["w_aux"]) == 0.0


def test_subset_totals_full_set_and_mean(h4_hf_variances):
    """K_s >= K is the full set (no sampling error); over all 28 leave-one-out subsets (K_s = K - 1) the mean of
    the scaled sums is V_m exactly (the estimator is unbiased over uniform subsets)."""
    act, el, hf, full = h4_hf_variances
    k = len(el)
    out = ri.variance_totals_subset(act, hf, el, "spin_free", k_subset=10 * k)
    assert out["k_subset"] == k and out["stderr_m"] == 0.0 and out["w_stderr_m"] == 0.0
    assert out["variance_m"] == pytest.approx(full["variance_m"], rel=1e-12)
    m = targets.element_multiplicities(el, "spin_free", act.norb).astype(float)
    x = m * full["v_direct"]
    loo = [k / (k - 1) * (x.sum() - x[j]) for j in range(k)]
    assert np.mean(loo) == pytest.approx(full["variance_m"], rel=1e-12)


def test_hf_variances():
    """F24 (i): on h4_r4.0 (exact pair sums) V_m at the HF determinant, 418.7 Ha^2, lies within 2% of V_m at
    iterate 1 of the exact spin-free L-BFGS record, 420.5 Ha^2 (M3's state) [design]."""
    act = sy.benchmark_system("h4_r4.0")
    ir = sy.orbital_irreps(act)
    el, _ = ri.allowed_elements(act, "spin_free", ir)
    psi1 = exact_trajectory(act, irreps=ir, max_iter=1).states[1]
    v_hf = ri.element_variances(act, FCISpace.of(act).hf(), el, "spin_free")
    v_1 = ri.element_variances(act, psi1, el, "spin_free")
    assert v_hf["method"] == v_1["method"] == "exact"
    assert v_hf["variance_m"] == pytest.approx(418.7, abs=0.05) and v_1["variance_m"] == pytest.approx(420.5, abs=0.05)
    assert v_hf["variance_m"] == pytest.approx(v_1["variance_m"], rel=0.02)


@pytest.mark.slow
def test_subset_totals_unbiased():
    """h6_r3.0 at HF (153 allowed elements, Rao-Blackwellized per-element variances): over 20 subset seeds
    (K_s = 100) the mean of the scaled totals lies within 2 standard errors of the full V_m."""
    act = sy.benchmark_system("h6_r3.0")
    el, _ = ri.allowed_elements(act, "spin_free", sy.orbital_irreps(act))
    hf = FCISpace.of(act).hf()
    full = ri.element_variances(act, hf, el, "spin_free")
    est = np.array([ri.variance_totals_subset(act, hf, el, "spin_free", k_subset=100, seed=seed)["variance_m"]
                    for seed in range(20)])
    assert len(el) == 153 and full["method"] == "rb"
    assert abs(est.mean() - full["variance_m"]) <= 2.0 * est.std(ddof=1) / math.sqrt(est.size)


def test_subset_totals_validation(h4_hf_variances):
    act, el, hf, _ = h4_hf_variances
    for bad in (1, 0, 2.5):
        with pytest.raises(ValueError, match="k_subset"):
            ri.variance_totals_subset(act, hf, el, "spin_free", k_subset=bad)


# ------------------------------------------------------------------------- system sets (F20)
def test_system_sets():
    """The sets of the scaling protocol: 16 chains (n = 4..10 at R = 2.5..4.0 bohr), five N2 CAS(6,6) and four
    CAS(10,8) bond lengths, and every name builds a system spec (M3 names or F20 patterns)."""
    assert len(scaling.SIZE_AXIS) == 16 == len(set(scaling.SIZE_AXIS))
    assert scaling.SIZE_AXIS[:4] == ("h4_r2.5", "h6_r2.5", "h8_r2.5", "h10_r2.5")
    assert scaling.OVERLAP_AXIS == ("n2_r1.098", "n2_r1.6", "n2_r2.0", "n2_r2.4", "n2_r2.8")
    assert len(scaling.ACTIVE_SPACE_AXIS) == 4 and len(scaling.PROPERTIES) == 13
    assert set(scaling.QUICK) <= set(scaling.SIZE_AXIS) | set(scaling.OVERLAP_AXIS)
    names = set().union(scaling.SIZE_AXIS, scaling.OVERLAP_AXIS, scaling.ACTIVE_SPACE_AXIS, scaling.EQUILIBRIUM,
                        scaling.CLOSED_LOOP, scaling.PROPERTIES, scaling.HOLDOUT, scaling.QUICK)
    for name in names:
        assert name in sy.BENCHMARK_SYSTEMS or sy.system_spec(name)


# ------------------------------------------------------------------------- spectrum (F21)
@pytest.fixture(scope="module")
def spectra():
    """Builds and spectra, computed once per (name, options)."""
    cache = {}

    def get(name, **kw):
        key = (name, tuple(sorted(kw.items())))
        if key not in cache:
            act = sy.benchmark_system(name)
            ir = sy.orbital_irreps(act)
            cache[key] = (act, ir, scaling.spectrum(act, ir, **kw))
        return cache[key]

    return get


def test_spectrum_gaps_and_overlaps(spectra):
    """h6_r3.0: Delta_1 = 104.67 mHa, same and any irrep; h4_r1.8: 565.72 / 563.34 mHa; p_HF(h4_r3.0) = 0.7012
    +- 1e-4 [design]; E0 and p_HF agree with an independent single-root FCI (to 1e-10 Ha and 1e-8)."""
    _, _, h6 = spectra("h6_r3.0")
    assert 1e3 * h6.gap == pytest.approx(104.67, abs=6e-3) and 1e3 * h6.gap_any == pytest.approx(104.67, abs=6e-3)
    _, _, h4 = spectra("h4_r1.8")
    assert 1e3 * h4.gap == pytest.approx(565.72, abs=6e-3) and 1e3 * h4.gap_any == pytest.approx(563.34, abs=6e-3)
    act, _, h43 = spectra("h4_r3.0")
    assert h43.p_hf == pytest.approx(0.7012, abs=1e-4)
    e0, ci0 = act.fci(nroots=1, spin_square=0.0)
    assert h43.e0 == pytest.approx(e0, abs=1e-10)
    assert h43.p_hf == pytest.approx(ci0[0, 0] ** 2 / np.sum(ci0 ** 2), abs=1e-8)
    for sp in (h6, h4, h43):
        assert sp.flags == () and sp.spin_tol == 1e-6 and sp.n_roots == 12 and sp.ground_irrep == sp.hf_irrep == 0
        assert sp.e0 < sp.e1 and sp.e0 < sp.e_hf and sp.dim == sp.psi0.size


@pytest.mark.parametrize("name", ["h4", "n2s"])
def test_spectrum_reproduces_m3_records(spectra, name):
    """exact_trajectory(roots=spectrum.roots) is the record with its own roots (M3's call) bit for bit; p_HF, E_HF,
    E1, E1_any and the ground irrep are the record's overlaps[0], energies[0], e1, e1_any, ground_irrep."""
    act, ir, sp = spectra(name)
    a = exact_trajectory(act, irreps=ir, max_iter=3, roots=sp.roots, n_roots=sp.n_roots)
    b = exact_trajectory(act, irreps=ir, max_iter=3)
    for f in ("energies", "variances", "overlaps", "eps"):
        assert np.array_equal(getattr(a, f), getattr(b, f))
    assert all(np.array_equal(x, y) for x, y in zip(a.states, b.states))
    assert (sp.p_hf, sp.e_hf, sp.e1, sp.e1_any, sp.e0, sp.ground_irrep) == \
        (b.overlaps[0], b.energies[0], b.e1, b.e1_any, b.e0, b.ground_irrep)
    assert not sp.psi0.flags.writeable and not sp.roots[0].flags.writeable
    assert not any(c.flags.writeable for c in sp.roots[1])


def test_spectrum_flags_and_few_roots(spectra):
    """The square H4 (B1g ground state) is flagged, with p_HF = 0 to round-off; with two roots on h4_r1.8, whose second
    root lies in another irrep, E1 is the highest root, a lower bound ("e1_lower_bound"); one root gives E1 None and
    E1_any nan."""
    _, _, sq = spectra("h4rect_1.8x1.8")
    assert sq.flags == ("ground_state_not_in_hf_irrep",) and sq.p_hf < 1e-24 and sq.hf_irrep == 0
    assert sq.ground_irrep == 1
    _, _, two = spectra("h4_r1.8", n_roots=2)
    _, _, full = spectra("h4_r1.8")
    assert two.flags == ("e1_lower_bound",) and two.e1 == two.e1_any == two.roots[0][-1]
    assert two.e1 < full.e1 and two.e1_any == pytest.approx(full.e1_any, abs=1e-10)
    _, _, one = spectra("h4_r3.0", n_roots=1)
    assert one.e1 is None and math.isnan(one.e1_any) and math.isnan(one.gap) and one.n_roots == 1
    assert one.e0 == pytest.approx(spectra("h4_r3.0")[2].e0, abs=1e-10)


def test_spectrum_validation(spectra):
    act, ir, sp = spectra("h4_r3.0")
    good = {f.name: getattr(sp, f.name) for f in dataclasses.fields(sp)}
    es, cis = sp.roots
    for bad in ({"psi0": 2.0 * sp.psi0}, {"psi0": sp.psi0.ravel()[:10]}, {"p_hf": 1.5}, {"e1": sp.e0 - 1.0},
                {"e_hf": sp.e0 - 1.0}, {"e1_any": float(es[2])}, {"flags": ("ground_state_not_in_hf_irrep",)},
                {"flags": ("unknown",)}, {"roots": (es[:3], cis[:2])}, {"roots": (es[::-1], cis)},
                {"nelec": (2,)}, {"nelec": (2, 9)}, {"dim": 35}, {"spin_tol": 0.0}, {"ground_irrep": 9},
                {"e1": None}, {"norb": 0}, {"e0": math.nan}):
        with pytest.raises(ValueError):
            scaling.Spectrum(**{**good, **bad})
    assert scaling.Spectrum(**good).e1 == sp.e1
    with pytest.raises(ValueError):
        sp.psi0[0, 0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        sp.e0 = 0.0
    for kw in ({"n_roots": 0}, {"n_roots": 2.5}, {"spin_tols": (1e-4, 1e-6)}, {"spin_tols": ()},
               {"spin_tols": (0.0,)}):
        with pytest.raises(ValueError):
            scaling.spectrum(act, ir, **kw)
    with pytest.raises(ValueError):
        scaling.spectrum(act, ir[:2])


@pytest.mark.slow
def test_spin_tolerance_fallback():
    """n2v_r2.4 (CAS(10,8), 3136 determinants): no penalty shift brings its 12 lowest singlet roots within 1e-6 of
    <S^2> = 0; the 1e-4 fallback carries "spin_tol_relaxed"; p_HF = 0.1112 [design]."""
    act = sy.benchmark_system("n2v_r2.4")
    sp = scaling.spectrum(act, sy.orbital_irreps(act))
    assert sp.flags == ("spin_tol_relaxed",) and sp.spin_tol == 1e-4 and sp.dim == 3136
    assert sp.p_hf == pytest.approx(0.1112, abs=1e-4)


# ------------------------------------------------------------------------- the stage cache
def test_cache_roundtrip_and_selection(tmp_path):
    """One computation per (name, stage, key); later gets load the pickle; a different name, stage or key is a
    different entry; refresh recomputes and overwrites; a corrupt entry is recomputed; entry files are
    <name>__<stage>__<sha256>.pkl and no temporary file is left."""
    root = tmp_path / "cache"
    cache = scaling.Cache(root)
    calls = []

    def compute(tag="v"):
        calls.append(tag)
        return {"x": np.arange(3.0), "tag": tag}

    v1 = cache.get("h4_r1.8", "spectrum", compute)
    v2 = cache.get("h4_r1.8", "spectrum", compute)
    assert calls == ["v"] and (cache.hits, cache.misses) == (1, 1) and np.array_equal(v1["x"], v2["x"])
    (entry,) = root.iterdir()
    digest = cache.digest("h4_r1.8", "spectrum")
    assert entry.name == f"h4_r1.8__spectrum__{digest}.pkl" and len(digest) == 64
    assert entry == cache.path("h4_r1.8", "spectrum")
    key = {"params": FTParameters(r=1.0), "k_cost": 3}
    for name, stage, kw in (("h4_r1.8", "spectrum", {"key": key}), ("h4_r1.8", "qpe", {}), ("h4_r3.0", "spectrum", {}),
                            ("fits", "spectrum", {})):
        cache.get(name, stage, compute, **kw)
        cache.get(name, stage, compute, **kw)
    assert len(calls) == 5 and len(list(root.iterdir())) == 5
    assert cache.digest("h4_r1.8", "spectrum", key=key) != cache.digest(
        "h4_r1.8", "spectrum", key={"params": FTParameters(r=0.3), "k_cost": 3})
    assert scaling.Cache(root).digest("h4_r1.8", "spectrum", key=key) == cache.digest("h4_r1.8", "spectrum", key=key)
    fresh = scaling.Cache(root, refresh=True)
    assert fresh.get("h4_r1.8", "spectrum", lambda: compute("new"))["tag"] == "new"
    assert cache.get("h4_r1.8", "spectrum", compute)["tag"] == "new"
    entry.write_bytes(b"not a pickle")
    assert cache.get("h4_r1.8", "spectrum", lambda: compute("again"))["tag"] == "again"
    with open(entry, "rb") as fh:
        assert pickle.load(fh)["tag"] == "again"
    assert not [p for p in root.iterdir() if p.name.startswith(".")]
    for bad in (("../h4", "s"), ("h4", ""), ("h4", "a/b"), (".h4", "s"), (3, "s")):
        with pytest.raises(ValueError):
            cache.get(*bad, compute)
    with pytest.raises(ValueError):
        cache.get("h4", "s", compute, key=[("r", 1.0)])
    with pytest.raises(ValueError):
        cache.get("h4", "s", 3)


def test_cache_writes_are_atomic(tmp_path, monkeypatch):
    """A write goes to a temporary file in the cache directory, renamed onto the entry by os.replace; a value that
    cannot be pickled leaves neither an entry nor a temporary file."""
    root = tmp_path / "c"
    replaced = []
    real = os.replace

    def spy(src, dst):
        replaced.append((pathlib.Path(src), pathlib.Path(dst)))
        real(src, dst)

    monkeypatch.setattr(scaling.os, "replace", spy)
    cache = scaling.Cache(root)
    cache.get("n2_r2.4", "records", lambda: [1, 2])
    ((src, dst),) = replaced
    assert src.parent == dst.parent == root and src.name.startswith(".") and not src.exists()
    assert dst == cache.path("n2_r2.4", "records") and dst.exists()
    with pytest.raises((pickle.PicklingError, AttributeError, TypeError)):
        cache.get("n2_r2.4", "other", lambda: (lambda: 0))
    assert [p.name for p in root.iterdir()] == [dst.name] and len(replaced) == 1


def test_cache_digest_rules(tmp_path, monkeypatch):
    """The digest changes with the package sources, the dependency versions and the build spec of the name; the tree
    digest covers every *.py file by relative path and content and ignores __pycache__ and other files."""
    cache = scaling.Cache(tmp_path)
    d0 = cache.digest("n2_r2.4", "records")
    monkeypatch.setattr(scaling, "source_digest", lambda: "0" * 64)
    d1 = cache.digest("n2_r2.4", "records")
    monkeypatch.setattr(scaling, "dependency_versions", lambda: {"numpy": "0"})
    d2 = cache.digest("n2_r2.4", "records")
    monkeypatch.setattr(scaling, "system_spec", lambda name: {"atom": "changed"})
    d3 = cache.digest("n2_r2.4", "records")
    assert len({d0, d1, d2, d3}) == 4
    monkeypatch.undo()
    assert cache.digest("n2_r2.4", "records") == d0
    assert set(scaling.dependency_versions()) == {"numpy", "scipy", "pyscf", "openfermion"}
    tree = tmp_path / "pkg"
    (tree / "sub" / "__pycache__").mkdir(parents=True)
    (tree / "a.py").write_text("x = 1\n")
    (tree / "sub" / "b.py").write_text("y = 2\n")
    t0 = scaling._tree_digest(tree)
    (tree / "readme.txt").write_text("ignored")
    (tree / "sub" / "__pycache__" / "b.py").write_text("ignored")
    assert scaling._tree_digest(tree) == t0
    (tree / "sub" / "b.py").write_text("y = 3\n")
    t1 = scaling._tree_digest(tree)
    (tree / "sub" / "b.py").rename(tree / "sub" / "c.py")
    assert len({t0, t1, scaling._tree_digest(tree)}) == 3
    assert len(scaling.source_digest()) == 64 and scaling.source_digest() == scaling.source_digest()


# ------------------------------------------------------------------------- strict JSON
def test_to_json_is_strict(spectra):
    """nan -> None, +-inf -> "inf"/"-inf", numpy scalars and arrays, str keys, named tuples as dicts; dataclasses
    without CI vectors and directions (a Spectrum keeps its root energies); json.dumps(allow_nan=False) round-trips;
    complex numbers and callables raise TypeError."""
    data = {"a": math.nan, "b": math.inf, "c": -math.inf, "d": np.float64(1.5), "e": np.arange(3), "f": (1, 2.0),
            0.3: [np.bool_(True), None, np.int64(7)], "g": StepEvent(1, "energy", 0.5, 1e-3, True)}
    out = scaling.to_json(data)
    assert out == {"a": None, "b": "inf", "c": "-inf", "d": 1.5, "e": [0, 1, 2], "f": [1, 2.0],
                   "0.3": [True, None, 7], "g": {"iteration": 1, "kind": "energy", "eps": 0.5, "sigma": 1e-3,
                                                 "used_for_step": True}}
    assert json.loads(json.dumps(out, allow_nan=False)) == out
    act, ir, sp = spectra("h4_r3.0")
    js = scaling.to_json(sp)
    assert "psi0" not in js and "roots" not in js and js["root_energies"] == sp.roots[0].tolist()
    assert js["p_hf"] == sp.p_hf and js["flags"] == [] and js["nelec"] == [2, 2]
    rec = exact_trajectory(act, irreps=ir, max_iter=2, roots=sp.roots, n_roots=sp.n_roots)
    jr = scaling.to_json(rec)
    assert "states" not in jr and "directions" not in jr and jr["energies"] == rec.energies.tolist()
    json.dumps(jr, allow_nan=False)
    assert scaling.to_json(FTParameters())["kappa_g"] == [1.0, 10.0, 100.0]
    run = run_step_rule(act, StepRule("unit"), max_iter=2, irreps=ir, keep_states=True)
    jrun = scaling.to_json(run)
    assert "states" not in jrun and "directions" not in jrun and jrun["eps"] == run.eps
    assert "generator" not in jrun["steps"][1] and jrun["steps"][1]["slope_est"] == run.steps[1].slope_est
    assert jrun["events"][0] == {"iteration": 1, "kind": "residual", "eps": 0.0, "sigma": None, "used_for_step": True}
    json.dumps(jrun, allow_nan=False)
    for bad in (1j, np.array([1j]), (lambda: 0), {1, 2}):
        with pytest.raises(TypeError):
            scaling.to_json(bad)


# ------------------------------------------------------------------------- F22: rate fits
def _geometric_rates(alpha_inf, c, q, ns):
    """Successive rates alpha^(n) = alpha_inf - c q^(n/2): their drifts are geometric with ratio q."""
    return np.array([alpha_inf - c * q ** (n / 2) for n in ns[1:]])


def test_rate_fits_synthetic():
    """F22 on synthetic chains: ln p = a - alpha n gives a, alpha and every successive rate to 1e-12; with
    geometric drifts (ratio q) the Aitken limit is alpha_inf exactly; the fallbacks and flags of three and two
    sizes and of q outside (0, 1); extrapolation and the size at an overlap invert each other."""
    ns = np.array([4, 6, 8, 10])
    a, alpha = 0.083, 0.1064
    fit = scaling.fit_log_overlap(ns, np.exp(a - alpha * ns), R=3.0, record="hf", k=0)
    assert fit.a == pytest.approx(a, abs=1e-12) and fit.alpha == pytest.approx(alpha, abs=1e-12)
    assert np.allclose(fit.successive, alpha, atol=1e-12) and abs(fit.drift) < 1e-12
    assert fit.ns == (4, 6, 8, 10) and fit.R == 3.0 and fit.record == "hf"
    rates = _geometric_rates(0.24, 0.05, 0.6, ns)
    lp = np.concatenate([[-0.3], -0.3 - 2.0 * np.cumsum(rates)])
    geo = scaling.fit_log_overlap(ns, np.exp(lp))
    assert np.allclose(geo.successive, rates, atol=1e-13) and geo.flags == ()
    assert geo.drift == pytest.approx(rates[-1] - rates[-2], abs=1e-13)
    assert geo.alpha_aitken == pytest.approx(0.24, abs=1e-12)
    assert geo.rate("last") == geo.alpha_last == geo.successive[-1] and geo.rate("aitken") == geo.alpha_aitken
    # n = 12 by both rates: p(10) e^(-2 alpha)
    for rate in scaling.RATES:
        assert scaling.extrapolate_overlap(geo, 12, rate=rate) == pytest.approx(
            math.exp(lp[-1] - 2.0 * geo.rate(rate)), rel=1e-12)
        n_t = scaling.size_at_overlap(geo, 1e-3, rate=rate)
        assert scaling.extrapolate_overlap(geo, n_t, rate=rate) == pytest.approx(1e-3, rel=1e-10)
    three = scaling.fit_log_overlap(ns[:3], np.exp(lp[:3]), window=(4, 6, 8))
    assert three.flags == ("aitken_fallback",) and three.alpha_aitken == pytest.approx(rates[1] + (rates[1] - rates[0]))
    two = scaling.fit_log_overlap(ns[:2], np.exp(lp[:2]), window=(4, 6))
    assert two.flags == ("no_drift",) and math.isnan(two.drift) and two.alpha_aitken == two.alpha_last
    up = scaling.fit_log_overlap(ns, np.exp(np.concatenate([[-0.3], -0.3 - 2.0 * np.cumsum([0.1, 0.12, 0.13])])))
    assert up.flags == () and up.alpha_aitken == pytest.approx(0.13 + 0.01 * 0.5 / 0.5)       # q = 1/2
    flat = scaling.fit_log_overlap(ns, np.exp(np.concatenate([[-0.3], -0.3 - 2.0 * np.cumsum([0.1, 0.12, 0.15])])))
    assert flat.flags == ("aitken_fallback",) and flat.alpha_aitken == pytest.approx(0.15 + 0.03)  # q = 1.5
    assert scaling.extrapolate_overlap(fit, 0) == 1.0              # exp(a) = 1.087 is capped at 1
    assert scaling.size_at_overlap(scaling.fit_log_overlap([4, 6], [0.5, 0.6], window=(4, 6)), 0.1) == math.inf
    assert np.allclose(scaling.successive_rates([8, 4, 6], [0.4, 0.7, 0.55]),
                       [math.log(0.7 / 0.55) / 2, math.log(0.55 / 0.4) / 2], atol=1e-15)


def test_rate_fit_validation():
    good = scaling.fit_log_overlap([4, 6, 8], [0.7, 0.57, 0.46])
    fields = {f.name: getattr(good, f.name) for f in dataclasses.fields(good)}
    for bad in ({"successive": (0.1, 0.2)}, {"p": (0.7, 0.57, 1.5)}, {"ns": (4, 4, 8)}, {"k": -1},
                {"drift": math.nan}, {"alpha": math.inf}, {"flags": ("other",)}, {"R": math.inf}, {"record": 3}):
        with pytest.raises(ValueError):
            scaling.RateFit(**{**fields, **bad})
    for args in (([4], [0.5]), ([4, 6], [0.5, 0.0]), ([4, 6], [0.5, 1.2]), ([4, 4.5], [0.5, 0.4]), ([4, 6], [0.5])):
        with pytest.raises(ValueError):
            scaling.fit_log_overlap(*args, window=(4, 6))
    with pytest.raises(ValueError, match="window"):
        scaling.fit_log_overlap([4, 6, 8], [0.7, 0.57, 0.46], window=(10, 12))
    for bad in ({"rate": "fit"},):
        with pytest.raises(ValueError):
            scaling.extrapolate_overlap(good, 10, **bad)
    with pytest.raises(ValueError):
        scaling.size_at_overlap(good, 0.0)


def test_rates_on_chains():
    """F22 on the H chains at R = 3.0 bohr: the HF successive rates are 0.1007 and 0.1049 per atom (n = 4, 6, 8)
    [design] (p_HF from single-root FCI, as ``spectrum`` with n_roots = 1)."""
    ps = []
    for n in (4, 6, 8):
        act = sy.benchmark_system(f"h{n}_r3.0")
        ps.append(scaling.spectrum(act, sy.orbital_irreps(act), n_roots=1).p_hf)
    rates = scaling.successive_rates([4, 6, 8], ps)
    assert rates == pytest.approx([0.1007, 0.1049], abs=5e-4)


# ------------------------------------------------------------------------- F21: gap and window
def test_gap_window():
    """F21: a power-law gap A n^-gamma is recovered exactly from its two largest sizes, A' = n_max Delta(n_max);
    the window [A'/m, (A/m)^(1/gamma)] solves Delta(n) = m in both models; a 1/n gap closes both ends; the margins
    of the three rules. On a gap that is no power law (138.96, 104.67, 83.32 mHa at n = 4, 6, 8) gamma comes from the
    two largest sizes only: 0.792972, A = 433.386 mHa, A' = 666.56 mHa, n_cert(3 mHa) = [222.187, 529.197]. A power
    law that does not close has the end inf if it grows or stays at or above m, and 0 if it stays below m."""
    g468 = scaling.fit_gap((4, 6, 8), (0.13896, 0.10467, 0.08332))
    gamma = math.log(0.10467 / 0.08332) / math.log(8 / 6)
    assert g468.gamma == pytest.approx(gamma, rel=1e-12) and gamma == pytest.approx(0.792972, abs=1e-6)
    assert g468.A == pytest.approx(0.08332 * 8 ** gamma, rel=1e-12) and g468.A == pytest.approx(0.433386, rel=1e-5)
    assert g468.A_inv == pytest.approx(0.66656, rel=1e-12)
    assert scaling.certification_window(g468, 3e-3) == pytest.approx((222.187, 529.197), rel=1e-5)
    assert scaling.certification_window(g468, 3e-3)[1] == pytest.approx(
        (0.08332 * 8 ** gamma / 3e-3) ** (1 / gamma), rel=1e-12)
    ns = np.array([4, 6, 8, 10])
    A, gamma = 0.48, 0.84
    g = scaling.fit_gap(ns, A * ns ** -gamma)
    assert g.A == pytest.approx(A, rel=1e-12) and g.gamma == pytest.approx(gamma, rel=1e-12)
    assert g.A_inv == pytest.approx(10 * A * 10 ** -gamma, rel=1e-12)
    lo, hi = scaling.certification_window(g, 3e-3)
    assert g.gap(lo, "inverse") == pytest.approx(3e-3, rel=1e-12) and g.gap(hi, "power") == pytest.approx(3e-3, rel=1e-12)
    assert lo < hi
    inv = scaling.fit_gap(ns, 0.7 / ns)
    assert inv.gamma == pytest.approx(1.0, abs=1e-12)
    w = scaling.certification_window(inv, 1e-2)
    assert w[0] == pytest.approx(70.0, rel=1e-12) and w[1] == pytest.approx(70.0, rel=1e-12)
    assert g.as_power_law("inverse").gap(37.0) == pytest.approx(g.gap(37.0, "inverse"), rel=1e-12)
    m = scaling.acceptance_margins(1e-3, 0.9857, 1e-4)
    assert m == pytest.approx({"E": 3e-3, "B": 2.1e-3, "BL": 1e-3 * (1 + 10 * 0.9857) + 1e-4}, rel=1e-12)
    assert scaling.acceptance_margins(1e-3, 0.94, 1e-3)["BL"] == pytest.approx(11.4e-3, rel=1e-12)
    for bad in ((0.0, 0.9, 0.0), (1e-3, 1.2, 0.0), (1e-3, 0.9, -1e-3)):
        with pytest.raises(ValueError):
            scaling.acceptance_margins(*bad)
    with pytest.raises(ValueError):
        scaling.GapFit(A=0.5, gamma=0.8, A_inv=0.1, ns=(8, 10), gaps=(0.09, 0.08))
    with pytest.raises(ValueError):
        scaling.certification_window(g, 0.0)
    # gaps that do not close: growing (gamma < 0) and constant (gamma = 0) power laws; A'/n still closes at A'/m
    grow, flat = scaling.fit_gap([8, 10], [0.002, 0.0025]), scaling.fit_gap([8, 10], [0.02, 0.02])
    assert grow.gamma == pytest.approx(-1.0, abs=1e-12) and flat.gamma == 0.0
    assert scaling.certification_window(grow, 3e-3) == pytest.approx((0.025 / 3e-3, math.inf), rel=1e-12)
    assert scaling.certification_window(flat, 1e-2) == pytest.approx((20.0, math.inf), rel=1e-12)
    assert scaling.certification_window(flat, 0.02) == pytest.approx((10.0, math.inf), rel=1e-12)
    assert scaling.certification_window(flat, 0.03) == pytest.approx((0.2 / 0.03, 0.0), rel=1e-12)


@functools.lru_cache(maxsize=None)
def _sine_tails(d_max=60):
    """tau_W(D) of the sine window (ft.acceptance.tail_mass) for D = 0..d_max."""
    from thc_engine_cqe.ft import acceptance

    return tuple(acceptance.tail_mass(float(d), window="sine") for d in range(d_max + 1))


def _ref_d_l(p, eps=1e-3, d_max=60):
    """D_L(p) = min{D in N : tau_W(D) <= eps p/(1 - p)} by its definition (D <= d_max)."""
    if p >= 1.0:
        return 0
    return next(d for d, tau in enumerate(_sine_tails(d_max)) if tau <= eps * p / (1.0 - p))


def test_one_run_margin():
    """F40: the one-run B_L margin D_L(p) is 5, 6, 7, 9, 13, 27 grid units at p = 0.25, 0.1, 0.078, 0.03, 0.01, 1e-3
    (the definition, evaluated directly), 0 at p = 1; the two-run floor tau_W(5)^2/(eps_id + tau_W(5)^2) = 3.37e-5,
    within the band of tau_W(5) = 1.84e-4 (F40) to its printed precision."""
    from thc_engine_cqe.ft import acceptance

    ps = (0.25, 0.1, 0.078, 0.03, 0.01, 1e-3)
    assert [scaling.d_l_one_run(p) for p in ps] == [_ref_d_l(p) for p in ps] == [5, 6, 7, 9, 13, 27]
    assert scaling.d_l_one_run(1.0) == 0
    tau = acceptance.tail_mass(5.0, window="sine")
    assert scaling.two_run_overlap_floor() == pytest.approx(tau ** 2 / (1e-3 + tau ** 2), rel=1e-12)
    floor = scaling.two_run_overlap_floor()
    assert floor == pytest.approx(3.369e-5, abs=1e-8)
    assert 1.835e-4 ** 2 / (1e-3 + 1.835e-4 ** 2) <= floor <= 1.845e-4 ** 2 / (1e-3 + 1.845e-4 ** 2)
    for bad in (0.0, 1.5, math.nan):
        with pytest.raises(ValueError):
            scaling.d_l_one_run(bad)


def _ref_admitted(gap, fit, rate, model, de=1e-3, s=1.0, em=1e-3, n_min=20.0, n_max=45.0, step=1e-3, d_max=60):
    """The grid of ``step`` on [n_min, n_max] and where Delta_1(n) >= dE (1 + 2 D_L(p(n)) s) + em holds on it: p(n) =
    p(n_last) e^(-alpha (n - n_last)) capped at 1, D_L(p) the first D <= d_max with tau_W(D) <= eps p/(1 - p),
    Delta_1 = A n^-gamma or A'/n."""
    n = np.arange(int(round(n_min / step)), int(round(n_max / step)) + 1) * step
    p = np.minimum(1.0, fit.p[-1] * np.exp(-fit.rate(rate) * (n - fit.ns[-1])))
    thr = np.full(n.shape, np.inf)
    thr[p < 1.0] = 1e-3 * p[p < 1.0] / (1.0 - p[p < 1.0])
    meets = np.asarray(_sine_tails(d_max))[None, :] <= thr[:, None]
    assert meets.any(axis=1).all(), "D_L lies beyond the table"
    d_l = np.argmax(meets, axis=1)
    delta = gap.A * n ** -gap.gamma if model == "power" else gap.A_inv / n
    return n, delta >= de * (1.0 + 2.0 * d_l * s) + em


def _ref_one_run_window(gap, fit, rate, model, **kw):
    """The largest admitted n of :func:`_ref_admitted` (None if no grid point is admitted)."""
    n, ok = _ref_admitted(gap, fit, rate, model, **kw)
    return float(n[ok][-1]) if ok.any() else None


def _p_ref(fit, n, rate="last"):
    return min(1.0, fit.p[-1] * math.exp(-fit.rate(rate) * (n - fit.ns[-1])))


def _hf_like_fit(p10=0.37463):
    """R = 3.0 HF-like overlaps: p(10) = 0.37463, successive rates 0.1007 and 0.1078 (Aitken fallback 0.1149)."""
    lp10 = math.log(p10)
    return scaling.fit_log_overlap([6, 8, 10], np.exp([lp10 + 2 * 0.1078 + 2 * 0.1007, lp10 + 2 * 0.1078, lp10]),
                                   R=3.0, record="hf")


def test_one_run_window_jointly_with_overlap():
    """F21's one-run B_L window (D_L = D_L(p(n)), no verification run) on R = 3.0-like HF inputs: p(10) = 0.37463 with
    successive rates 0.1007, 0.1078 (Aitken fallback 0.1149), the gap through (8, 83.32) and (10, 69.04) mHa,
    sin theta_0 = 1, E1 - E1lb = 1 mHa. The largest admitted n is 37 / 35 (power law, last / Aitken) and 34 / 32
    (A'/n), and the real window end agrees with a direct scan of the condition to the scan's grid; it lies below the
    two-run window. A power law that does not close (gamma <= 0) under the decaying overlap: a gap growing as
    n^0.176 (dE = 10 mHa) and a constant one of 30.5 mHa (levels up to D_L = 14) end where D_L steps up, as the scan;
    a gap below the margin everywhere gives 0, one above the margin of D_L = 1000 raises. Overlaps that do not decay
    (rate 0, D_L = D_L(0.3); rate < 0, D_L -> 0): inf if the gap reaches the large-n margin, else 0, as the scan
    up to n = 200."""
    fit = _hf_like_fit()
    assert fit.alpha_last == pytest.approx(0.1078, abs=1e-12) and fit.alpha_aitken == pytest.approx(0.1149, abs=1e-12)
    gap = scaling.fit_gap([8, 10], [0.08332, 0.06904])
    kw = dict(de_qpe=1e-3, sin_theta0=1.0, e1_margin=1e-3)
    got = {(model, rate): scaling.certification_window_overlap(gap, fit, rate=rate, model=model, **kw)
           for model in scaling.GAP_MODELS for rate in scaling.RATES}
    assert {key: math.floor(v) for key, v in got.items()} == {("power", "last"): 37, ("power", "aitken"): 35,
                                                             ("inverse", "last"): 34, ("inverse", "aitken"): 32}
    for (model, rate), v in got.items():
        assert v == pytest.approx(_ref_one_run_window(gap, fit, rate, model), abs=1.5e-3)
        assert v < _window_end_ref(gap, 12e-3, model)
    slow = scaling.fit_gap([8, 10], [0.25, 0.26])
    const = scaling.fit_gap([8, 10], [0.0305, 0.0305])
    assert slow.gamma == pytest.approx(math.log(0.25 / 0.26) / math.log(1.25), rel=1e-12) and const.gamma == 0.0
    for g, de in ((slow, 1e-2), (const, 1e-3)):
        for rate in scaling.RATES:
            v = scaling.certification_window_overlap(g, fit, de_qpe=de, sin_theta0=1.0, e1_margin=1e-3, rate=rate)
            ref = _ref_one_run_window(g, fit, rate, "power", de=de, n_min=1e-3, n_max=60.0)
            assert ref is not None and v == pytest.approx(ref, abs=1.5e-3)
            d_in, d_out = (_ref_d_l(_p_ref(fit, x, rate)) for x in (v - 1e-6, v + 1e-6))
            assert d_out == d_in + 1
    tiny = scaling.fit_gap([8, 10], [1e-4, 1.2e-4])
    assert scaling.certification_window_overlap(tiny, fit, **kw) == 0.0
    assert _ref_one_run_window(tiny, fit, "last", "power", n_min=1e-3, n_max=60.0) is None
    with pytest.raises(ValueError):
        scaling.certification_window_overlap(scaling.fit_gap([8, 10], [0.5, 0.6]), fit, **kw)
    grow = scaling.fit_gap([8, 10], [0.05, 0.06])
    flat = scaling.fit_log_overlap([8, 10], [0.3, 0.3])
    rise = scaling.fit_log_overlap([8, 10], [0.2, 0.3])
    assert flat.rate("last") == flat.rate("aitken") == 0.0 and rise.rate("last") < 0.0 and rise.rate("aitken") < 0.0
    for f, m_inf in ((flat, 1e-3 * (1.0 + 2.0 * _ref_d_l(0.3)) + 1e-3), (rise, 2e-3)):
        for g, expected in ((grow, math.inf), (scaling.fit_gap([8, 10], [1.05 * m_inf] * 2), math.inf),
                            (scaling.fit_gap([8, 10], [0.95 * m_inf] * 2), 0.0)):
            assert [scaling.certification_window_overlap(g, f, rate=r, **kw) for r in scaling.RATES] == [expected] * 2
            last = _ref_one_run_window(g, f, "last", "power", n_min=1e-2, n_max=200.0, step=1e-2)
            assert last == 200.0 if expected == math.inf else last is None
    with pytest.raises(ValueError):
        scaling.certification_window_overlap(gap, fit, rate="mean", **kw)
    with pytest.raises(ValueError):
        scaling.certification_window_overlap(gap, fit, model="exp", **kw)


def test_one_run_window_growing_gap():
    """A gap growing as n^0.817 (fit through (8, 50) and (10, 60) mHa) never closes, but D_L(p(n)) of the decaying
    HF-like overlap outgrows it: at the last rate the one-run window (dE = E1 - E1lb = 1 mHa, sin theta_0 = 1) ends at
    n = 126.51, where D_L steps from 237 to 238, equal to a direct scan of the condition on n = 100..130. With p(10) =
    0.01 (same rates) the lowest levels of D_L lie at n <= 0 and the next ones fall short of the margin, so no n below
    2.9 is admitted, and the window ends at 83.70 (D_L 169 -> 170), as the scan on n = 0.001..95."""
    grow = scaling.fit_gap([8, 10], [0.05, 0.06])
    assert grow.gamma == pytest.approx(math.log(5 / 6) / math.log(1.25), rel=1e-12)
    for p10, (n_min, n_max), end, levels in ((0.37463, (100.0, 130.0), 126.51, [237, 238]),
                                             (0.01, (1e-3, 95.0), 83.70, [169, 170])):
        fit = _hf_like_fit(p10)
        v = scaling.certification_window_overlap(grow, fit, de_qpe=1e-3, sin_theta0=1.0, e1_margin=1e-3)
        n, ok = _ref_admitted(grow, fit, "last", "power", n_min=n_min, n_max=n_max, d_max=280)
        assert v == pytest.approx(float(n[ok][-1]), abs=1.5e-3) and v == pytest.approx(end, abs=1e-2)
        assert [_ref_d_l(_p_ref(fit, x), d_max=280) for x in (v - 1e-6, v + 1e-6)] == levels
        delta = grow.A * v ** -grow.gamma
        assert 1e-3 * (1 + 2 * levels[0]) + 1e-3 <= delta < 1e-3 * (1 + 2 * levels[1]) + 1e-3
        if p10 == 0.01:
            assert not ok[0] and n[ok][0] > 2.9


def _window_end_ref(gap, m, model):
    return (gap.A / m) ** (1.0 / gap.gamma) if model == "power" else gap.A_inv / m


def test_power_law_fit():
    """F23: successive exponents of an exact power law are its exponent; beta_last and beta_mean; predict."""
    x = np.array([6.0, 8.0, 10.0])
    f = scaling.fit_power_law(x[::-1], 3.0 * x[::-1] ** 8.6)
    assert np.allclose(f.successive, 8.6, atol=1e-12) and f.beta_last == pytest.approx(8.6, abs=1e-12)
    assert f.predict(20.0) == pytest.approx(3.0 * 20.0 ** 8.6, rel=1e-12)
    g = scaling.fit_power_law([6, 8, 10], [1.0, 2.0, 3.0])
    assert g.beta_mean == pytest.approx(np.mean([math.log(2) / math.log(8 / 6), math.log(1.5) / math.log(10 / 8)]))
    for bad in (([6, 6], [1, 2]), ([6, 8], [1, -2]), ([6], [1])):
        with pytest.raises(ValueError):
            scaling.fit_power_law(*bad)
    with pytest.raises(ValueError):
        scaling.PowerFit(x=(6.0, 8.0), y=(1.0, 2.0), successive=(1.0,), beta_last=1.0, beta_mean=1.0)


# ------------------------------------------------------------------------- F44 under repetition
def _record(p, energies, e1=-1.0):
    """A synthetic one-step TrajectoryRecord with overlaps p (handoff_table uses energies, variances, p, E1)."""
    from thc_engine_cqe.ft.trajectory import TrajectoryRecord

    k = len(p)
    return TrajectoryRecord(system="synthetic", optimizer="lbfgs", norb=4, nelec=(2, 2), e0=-2.0, e1=e1, e1_any=e1,
                            energies=np.asarray(energies, dtype=float), variances=np.full(k, 1e-3),
                            residual_norms=np.ones(k), overlaps=np.asarray(p, dtype=float), eps=np.ones(k - 1),
                            directions=[], states=[], irreps=None, first_step="exact")


def test_break_even_overlap_matches_marginal_condition():
    """F44 against M3's hand-off table on 300 random one-step cases (measured loop): T_E(1) < T_E(0) iff
    p_0 < p_0* iff rho_0 < 1, and mu_0 C_QPE = T_E(0) - T_E(1) + C_iter(0)."""
    from thc_engine_cqe.ft.handoff import handoff_table

    rng = np.random.default_rng(20261108)
    checked = 0
    for _ in range(300):
        p0 = rng.uniform(0.01, 0.95)
        p1 = min(1.0, p0 * rng.uniform(0.5, 4.0))
        c_qpe = rng.uniform(1e6, 1e7)
        c_prep = [0.0, rng.uniform(0.0, 0.2) * c_qpe]
        c_iter = rng.uniform(0.0, 2.0) * c_qpe
        rows = handoff_table(_record([p0, p1], [-1.5, -1.8]), np.array(c_prep), [c_iter], c_qpe, loop="measured")
        mu = scaling.marginal_saving(p0, p1, c_prep[0] / c_qpe, c_prep[1] / c_qpe)
        assert mu * c_qpe == pytest.approx(rows[0].cost_expected - rows[1].cost_expected + c_iter, rel=1e-9, abs=1e-3)
        rho = scaling.cost_ratio(c_iter / c_qpe, p0, p1, c_prep[0] / c_qpe, c_prep[1] / c_qpe)
        p_star = scaling.break_even_overlap(c_iter / c_qpe, p1 / p0, c_prep[0] / c_qpe, c_prep[1] / c_qpe)
        if abs(rho - 1.0) < 1e-9:
            continue
        pays = rows[1].cost_expected < rows[0].cost_expected
        assert pays == (rho < 1.0) == (p0 < p_star)
        checked += 1
    assert checked >= 295
    assert scaling.break_even_overlap(0.0, 2.0) == math.inf and scaling.break_even_overlap(0.0, 0.5) == 0.0
    assert scaling.cost_ratio(1.0, 0.5, 0.4) == math.inf and scaling.cost_ratio(math.inf, 0.2, 0.5) == math.inf
    assert scaling.cost_ratio(1.0, 0.2, 0.5, saving=lambda a, b, c, d: 4.0) == 0.25
    assert np.allclose(scaling.marginal_saving(np.array([0.2, 0.5]), np.array([0.4, 0.5])), [2.5, 0.0])
    for bad in ((0.0, 0.5), (0.5, 1.5), (0.5, 0.5, -1.0)):
        with pytest.raises(ValueError):
            scaling.marginal_saving(*bad)


# ------------------------------------------------------------------------- F45: break-even size
def _fit_from(a, alpha, ns=(4, 6, 8, 10)):
    ns = np.asarray(ns)
    return scaling.fit_log_overlap(ns, np.exp(a - alpha * ns))


def _geometric_fit(lp4, alpha_inf, c, q, ns=(4, 6, 8, 10)):
    """A fit with successive rates alpha_inf - c q^(n/2) (geometric drifts, Aitken limit alpha_inf), ln p(4) = lp4."""
    rates = _geometric_rates(alpha_inf, c, q, np.asarray(ns))
    return scaling.fit_log_overlap(list(ns), np.exp(np.concatenate([[lp4], lp4 - 2.0 * np.cumsum(rates)])))


def _ln_p(fit, n, rate):
    """ln p(n) = ln p(n_last) - alpha (n - n_last) (F22, capped at 0), by hand."""
    return min(0.0, math.log(fit.p[-1]) - fit.rate(rate) * (n - fit.ns[-1]))


def _rep(p, c, f):
    """Repetition's hand-off cost with the QPE lengthened by f, C_QPE(n) units: (c + f)/p."""
    return (c + f) / p


def _brute_n_star(fk, fk1, c0, n0, beta, crossover=None, gap=None, margin=None, rate="last", n_max=3000, prep=None,
                  rate_k1=None, h=None):
    """Independent reference of F45, a pure-Python scan over even n >= n0 that stops where p < 1e-300.

    Repetition without preparation terms (``prep`` and ``h`` None): the closed form ln rho(n) = ln c(n) - ln f(n) -
    (alpha_k n - a_k) - ln(1 - e^((a_k - a_k1) - (alpha_k - alpha_k1) n)), a_k = ln p_k(n_last) + alpha_k n_last.
    Otherwise mu = h(p_k, c_k, f) - h(p_k1, c_k1, f) with the hand-off cost h(p, c, f) of a QPE lengthened by f
    (default :func:`_rep`) and c_j = c0_j (n/n0_j)^beta_j of ``prep``. p_k at ``rate``, p_k1 at ``rate_k1`` (default
    the same)."""
    rk, rk1 = rate, rate if rate_k1 is None else rate_k1
    al_k, al_k1 = fk.rate(rk), fk1.rate(rk1)
    ak_ = math.log(fk.p[-1]) + al_k * fk.ns[-1]
    ak1_ = math.log(fk1.p[-1]) + al_k1 * fk1.ns[-1]
    for n in range(n0, n_max + 1, 2):
        if crossover is None or n <= max(crossover[0], n0):
            c = c0 * (n / n0) ** beta
        else:
            nc = max(crossover[0], n0)
            c = c0 * (nc / n0) ** beta * (n / nc) ** crossover[1]
        f = 1.0 if gap is None else max(1.0, margin / (gap.A * n ** -gap.gamma))
        lpk, lpk1 = _ln_p(fk, n, rk), _ln_p(fk1, n, rk1)
        if min(lpk, lpk1) < math.log(1e-300):
            return None
        if prep is None and h is None:
            if al_k * n - ak_ < 0.0 or al_k1 * n - ak1_ < 0.0:      # the extrapolated p would exceed 1 (capped)
                continue
            x = (ak_ - ak1_) - (al_k - al_k1) * n
            if x >= 0.0:
                continue
            if math.log(c) - math.log(f) - (al_k * n - ak_) - math.log(-math.expm1(x)) < 0.0:
                return n
            continue
        ck, ck1 = (0.0, 0.0) if prep is None else (cj * (n / nj) ** bj for cj, nj, bj in prep)
        hh = _rep if h is None else h
        mu = hh(math.exp(lpk), ck, f) - hh(math.exp(lpk1), ck1, f)
        if mu > 0.0 and c / mu < 1.0:
            return n
    return None


_BRUTE_CASES = [dict(beta=9.4), dict(beta=8.9, crossover=(20, 2.0)), dict(beta=9.0, crossover=(20, 4.0)),
                dict(beta=9.4, gap="power", margin=3e-3), dict(beta=8.9, crossover=(20, 2.0), gap="inverse",
                                                                margin=11.4e-3), dict(beta=9.4, rate="aitken")]


def _brute_inputs(kw):
    gap = scaling.fit_gap([8, 10], [0.017, 0.0141])
    kw = dict(kw)
    if "gap" in kw:
        kw["gap"] = gap.as_power_law(kw["gap"])
    return _fit_from(-0.09, 0.0591), _geometric_fit(-0.05, 0.054, 0.01, 0.5), kw


@pytest.mark.parametrize("case", range(6))
def test_break_even_size_brute_force(case):
    """F45 without preparation terms (prep=None) against a pure-Python scan of the closed-form ln rho_k(n) on
    synthetic fits: measured exponents, a crossover, the gap rescaling beyond the window, the Aitken rate."""
    fk, fk1, kw = _brute_inputs(_BRUTE_CASES[case])
    c0, n0 = 166.0, 10
    got = scaling.break_even_size(fk, fk1, c0, n0, prep=None, **kw)
    assert got == _brute_n_star(fk, fk1, c0, n0, **kw) and got is not None
    assert got == scaling.break_even_size(fk, fk1, c0, n0, **kw)


def test_break_even_size_rates():
    """The rate enters both fits: with Aitken limits away from the last rates (successive rates 0.07 - 0.05 0.6^(n/2),
    ln p(4) = -0.09, and 0.054 - 0.01 0.5^(n/2), ln p(4) = -0.05; c_0 = 166 at n_0 = 10, beta = 9.4) n* is 682 at the
    last rates and 634 at the Aitken limits, each equal to the reference scan."""
    fk, fk1 = _geometric_fit(-0.09, 0.07, 0.05, 0.6), _geometric_fit(-0.05, 0.054, 0.01, 0.5)
    assert fk.alpha_aitken == pytest.approx(0.07, abs=1e-12) and abs(fk.alpha_last - 0.07) > 3e-3
    got = {rate: scaling.break_even_size(fk, fk1, 166.0, 10, 9.4, rate=rate) for rate in scaling.RATES}
    assert got == {rate: _brute_n_star(fk, fk1, 166.0, 10, 9.4, rate=rate) for rate in scaling.RATES}
    assert got == {"last": 682, "aitken": 634}


def _two_point_fit(p_last, alpha, n_last=8):
    """A fit on (n_last - 2, n_last) with one successive rate alpha and p(n_last) = p_last."""
    return scaling.fit_log_overlap([n_last - 2, n_last], [p_last * math.exp(2.0 * alpha), p_last],
                                   window=(n_last - 2, n_last))


def test_break_even_size_preparation_terms():
    """F45 with the preparation terms c_j(n) = C(Psi_j)(n)/C_QPE(n) on R = 4.0-like spin-free inputs (p_1(8) = 0.5930,
    p_2(8) = 0.7794, rates 0.05133 / 0.05053, c_0 = 23.009 at n_0 = 8, beta = 9.3673, no window): n* = 938 without them
    (F45's c_k << 1 form), none with c_1 = 2.3413e-4 (n/8)^2.2701 and c_2 = 2.3398e-3 (n/8)^2.3384 (C(Psi_2) grows
    faster than the overlap gain repays it), 2534 with c_2 at c_1's exponent; each equal to the reference scan of
    mu = (1 + c_1)/p_1 - (1 + c_2)/p_2. A PowerFit gives the same power law as (y_last, x_last, beta_last)."""
    fk, fk1 = _two_point_fit(0.5930, 0.05133), _two_point_fit(0.7794, 0.05053)
    args = (fk, fk1, 23.009, 8, 9.3673)
    c1 = (2.3413e-4, 8, 2.2701)
    for prep, expected in ((None, 938), ((c1, (2.3398e-3, 8, 2.3384)), None), ((c1, (2.3398e-3, 8, 2.2701)), 2534)):
        got = scaling.break_even_size(*args, prep=prep)
        assert got == _brute_n_star(*args, prep=prep) == expected
    pf = scaling.fit_power_law([6, 8], [2.3413e-4 * 0.75 ** 2.2701, 2.3413e-4])
    assert scaling.break_even_size(*args, prep=(pf, (2.3398e-3, 8, 2.2701))) == 2534
    for bad in (((1.0, 8),), ((1.0, 8, 2.0), (-1.0, 8, 2.0)), ((1.0, 0.0, 2.0), (1.0, 8, 2.0)), "prep"):
        with pytest.raises(ValueError):
            scaling.break_even_size(*args, prep=bad)


@pytest.mark.parametrize("case", range(4))
def test_break_even_size_preparation_window(case):
    """Beyond the window the hand-off QPE is lengthened and the preparation terms are not: mu = (f + c_k)/p_k -
    (f + c_k1)/p_k1 (c_1 = 1e-3 (n/10)^2.2, c_2 = 1e-2 (n/10)^2.3) against the reference scan; the preparation terms
    move n*, and lengthening them too (mu = f[(1 + c_k)/p_k - (1 + c_k1)/p_k1]) would give another n* where f > 1."""
    fk, fk1, kw = _brute_inputs([dict(beta=9.4), dict(beta=9.4, gap="power", margin=3e-3),
                                 dict(beta=8.9, crossover=(20, 2.0), gap="inverse", margin=11.4e-3),
                                 dict(beta=9.4, gap="power", margin=3e-3, rate="aitken")][case])
    prep = ((1e-3, 10, 2.2), (1e-2, 10, 2.3))
    got = scaling.break_even_size(fk, fk1, 166.0, 10, prep=prep, **kw)
    assert got is not None and got == _brute_n_star(fk, fk1, 166.0, 10, prep=prep, **kw)
    assert got != scaling.break_even_size(fk, fk1, 166.0, 10, **kw)
    if "gap" in kw:
        g = kw["gap"]
        assert kw["margin"] / g.gap(got) > 1.0
        lengthened = _brute_n_star(fk, fk1, 166.0, 10, prep=prep, h=lambda p, c, f: f * (1.0 + c) / p, **kw)
        assert lengthened != got


def test_break_even_size_underflow():
    """Overlaps that fall below 1e-300 before n = 3000 (rates 0.30 and 0.29 per site) end the scan there: no warning and
    no error, n* = 92 under repetition (the reference scan), an n* or None under the F44 hooks of ft.amplify. Where
    p_k+1 falls below the floor first (rates 0.25 and 0.31 per site, no gain) the scan ends there too: None under
    repetition, bbht and the best realizable scheme, as the reference scan."""
    from thc_engine_cqe.ft import amplify

    ns = np.array([4, 6, 8, 10])
    fa = scaling.fit_log_overlap(ns, np.exp(-0.1 - 0.30 * ns))
    fb = scaling.fit_log_overlap(ns, np.exp(-0.05 - 0.29 * ns))
    assert scaling.extrapolate_overlap(fa, 3000) == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert scaling.break_even_size(fa, fb, 1e3, 10, 9.0) == _brute_n_star(fa, fb, 1e3, 10, 9.0) == 92
        for sv in (amplify.scheme_saving("bbht"), amplify.best_realizable_saving(schemes=("rep", "bbht"), n_verify=2)):
            got = scaling.break_even_size(fa, fb, 1e3, 10, 9.0, saving=sv)
            assert got is None or (isinstance(got, int) and got % 2 == 0)
        # at c_0 = 1e280, rho < 1 only where p_k < 1e-300 (n > 2302): none, as the reference scan
        assert scaling.break_even_size(fa, fb, 1e280, 10, 9.0) is None and _brute_n_star(fa, fb, 1e280, 10, 9.0) is None
        # p_k+1 (0.31 per site) falls below 1e-300 first (n > 2228, p_k at n > 2763), with no gain at any n: none
        fc = scaling.fit_log_overlap(ns, np.exp(-0.05 - 0.25 * ns))
        fd = scaling.fit_log_overlap(ns, np.exp(-0.1 - 0.31 * ns))
        assert scaling.extrapolate_overlap(fd, 2230) < 1e-300 < scaling.extrapolate_overlap(fc, 2230)
        assert scaling.extrapolate_overlap(fd, 2500) == 0.0
        assert scaling.break_even_size(fc, fd, 1e3, 10, 9.0) is None and _brute_n_star(fc, fd, 1e3, 10, 9.0) is None
        for sv in (amplify.scheme_saving("bbht"), amplify.best_realizable_saving(schemes=("rep", "bbht"), n_verify=2)):
            assert scaling.break_even_size(fc, fd, 1e3, 10, 9.0, saving=sv) is None
    n = 2330
    ln_rho = math.log(1e280) + 9.0 * math.log(n / 10) - (0.1 + 0.30 * n) - math.log(-math.expm1(-0.05 - 0.01 * n))
    assert ln_rho < 0.0 and -0.1 - 0.30 * n < math.log(1e-300)


def test_break_even_size_schemes_beyond_window():
    """The F44 hooks beyond the window. QPE-scaled schemes take f inside: bbht with preparation terms equals the
    reference scan of mu = E(p_k)(c_k + f) - E(p_k1)(c_k1 + f), the repetition hook the default path. The filtered
    lt_bbht_L carries C_QPE once (it cancels in mu) and C_filt = 0.125 C_QPE(n) does not depend on dE: with
    qpe_scaled=False (or a hook marked qpe_scaled = False) n* is that of no window, the reference scan of mu =
    E(p_k)(c_k + C_filt) - E(p_k1)(c_k1 + C_filt); scaled by f its saving would be overstated (an earlier n*)."""
    from thc_engine_cqe.ft import amplify

    fk, fk1 = _fit_from(-0.09, 0.0591), _geometric_fit(-0.05, 0.054, 0.01, 0.5)
    gap = scaling.fit_gap([8, 10], [0.017, 0.0141])
    prep = ((1e-3, 10, 2.2), (1e-2, 10, 2.3))
    args, win = (fk, fk1, 166.0, 10, 9.4), dict(gap=gap, margin=3e-3)
    e = amplify.exponential_search_calls
    bbht = scaling.break_even_size(*args, saving=amplify.scheme_saving("bbht"), prep=prep, **win)
    assert bbht is not None and bbht == _brute_n_star(*args, prep=prep, h=lambda p, c, f: e(p) * (c + f), **win)
    assert scaling.break_even_size(*args, saving=amplify.scheme_saving("rep"), prep=prep, **win) == \
        scaling.break_even_size(*args, prep=prep, **win)
    lt = amplify.scheme_saving("lt_bbht_L", c_filt=0.125)
    plain = scaling.break_even_size(*args, saving=lt, prep=prep)
    assert plain is not None and plain == _brute_n_star(*args, prep=prep, h=lambda p, c, f: e(p) * (c + 0.125) + f)
    assert scaling.break_even_size(*args, saving=lt, prep=prep, qpe_scaled=False, **win) == plain

    def marked(p_k, p_k1, c_k=0.0, c_k1=0.0):
        return lt(p_k, p_k1, c_k, c_k1)
    marked.qpe_scaled = False
    assert scaling.break_even_size(*args, saving=marked, prep=prep, **win) == plain
    assert scaling.break_even_size(*args, saving=lt, prep=prep, qpe_scaled=True, **win) < plain
    marked.qpe_scaled = "no"
    with pytest.raises(ValueError):
        scaling.break_even_size(*args, saving=marked, **win)
    with pytest.raises(ValueError):
        scaling.break_even_size(*args, qpe_scaled=False)


def test_break_even_size_finds_and_misses():
    """A cheap iteration with a growing gain pays at a finite even n; n* is the first n with rho < 1 (checked at
    n* - 2); without a cost it pays at n0; validation."""
    fk, fk1 = _fit_from(0.0, 0.06), _fit_from(0.0, 0.03)
    n_star = scaling.break_even_size(fk, fk1, 1.0, 10, 2.0)
    assert n_star is not None and n_star % 2 == 0 and n_star > 10

    def rho(n):
        pk, pk1 = scaling.extrapolate_overlap(fk, n), scaling.extrapolate_overlap(fk1, n)
        return (n / 10) ** 2 / (1 / pk - 1 / pk1)
    assert rho(n_star) < 1.0 <= rho(n_star - 2)
    assert scaling.break_even_size(fk, fk1, 0.0, 10, 9.0) == 10
    cost = scaling.fit_power_law([6, 8, 10], [0.01, 0.1, 0.6])
    gap = scaling.fit_gap([8, 10], [0.04, 0.032])
    with pytest.raises(ValueError):
        scaling.break_even_bracket(fk, fk1, cost, 0.6, 10, gap=gap)
    with pytest.raises(ValueError):
        scaling.break_even_size(fk, fk1, 1.0, 10, 2.0, gap=gap)
    with pytest.raises(ValueError):
        scaling.break_even_size(fk, fk1, 1.0, 10, 2.0, rate="mean")


def _case_args(case, cost_fit):
    return {"beta": case["beta_value"], "crossover": None if case["crossover"] is None else tuple(case["crossover"])}


def test_break_even_bracket_cases_and_summaries():
    """The bracket on a gap with gamma != 1 (fit through (8, 17), (10, 14.1) mHa) and preparation terms: 252 cases
    (126 matched, 126 crossed); every A'/n case equals break_even_size with A'/n written as a power law, every case
    the reference scan, and at least one A'/n case differs from its power-law twin; each windowed case's n_cert is
    the window end of its own gap model and it is inside iff n* <= n_cert (a closing gap); p_k, p_k1, c_k, c_k1 and f
    at n*; "overall" is the range over the windowed cases, "no_window" over the f = 1 cases; the c_k = 0 check equals
    the bracket without preparation terms."""
    fk, fk1 = _fit_from(-0.09, 0.0591), _geometric_fit(-0.05, 0.054, 0.01, 0.5)
    cost = scaling.fit_power_law([6, 8, 10], [0.4, 12.0, 166.0])
    gap = scaling.fit_gap([8, 10], [0.017, 0.0141])
    margins = {"E": 3e-3, "BL": 11.4e-3}
    prep = ((1e-3, 10, 2.2), (1e-2, 10, 2.3))
    br = scaling.break_even_bracket(fk, fk1, cost, 166.0, 10, gap=gap, margins=margins, prep=prep)
    plain = scaling.break_even_bracket(fk, fk1, cost, 166.0, 10, gap=gap, margins=margins)
    cases = br["cases"]
    assert len(cases) == 2 * 3 * 3 * (1 + 2 * 2) and len(br["crossed"]["cases"]) == len(cases)
    assert br["prep"] == {"c_k": {"c0": 1e-3, "n0": 10.0, "beta": 2.2}, "c_k1": {"c0": 1e-2, "n0": 10.0, "beta": 2.3}}
    ends = {"power": (gap.A / 3e-3) ** (1 / gap.gamma), "inverse": gap.A_inv / 3e-3}
    twins = 0
    for case in cases:
        kw = _case_args(case, cost)
        if case["gap_model"] is not None:
            kw.update(gap=gap.as_power_law(case["gap_model"]), margin=margins[case["margin"]])
        ref = _brute_n_star(fk, fk1, 166.0, 10, rate=case["rate"], prep=prep, **kw)
        assert case["n_star"] == ref == scaling.break_even_size(fk, fk1, 166.0, 10, rate=case["rate"], prep=prep, **kw)
        assert case["rate_k"] == case["rate_k1"] == case["rate"]
        if case["gap_model"] == "inverse":
            twin = next(c for c in cases if c["gap_model"] == "power" and all(
                c[key] == case[key] for key in ("rate", "beta", "crossover", "margin")))
            twins += twin["n_star"] != case["n_star"]
        if case["margin"] == "E":
            assert case["n_cert"] == pytest.approx(ends[case["gap_model"]], rel=1e-12)
        if case["gap_model"] is None:
            assert case["n_cert"] is None and case["inside"] is None
        else:
            assert case["inside"] == (case["n_star"] is not None and case["n_star"] <= case["n_cert"])
        if case["n_star"] is not None:
            n = case["n_star"]
            assert case["c_k"] == pytest.approx(1e-3 * (n / 10) ** 2.2, rel=1e-12)
            assert case["c_k1"] == pytest.approx(1e-2 * (n / 10) ** 2.3, rel=1e-12)
            f = 1.0 if case["gap_model"] is None else max(1.0, margins[case["margin"]] / gap.gap(n, case["gap_model"]))
            assert case["f"] == pytest.approx(f, rel=1e-12)
            assert case["p_k"] == pytest.approx(math.exp(_ln_p(fk, n, case["rate"])), rel=1e-12)
            assert case["p_k1"] == pytest.approx(math.exp(_ln_p(fk1, n, case["rate"])), rel=1e-12)
    assert twins >= 1
    windowed = [c for c in cases if c["margin"] is not None]
    hits = [c for c in windowed if c["n_star"] is not None]
    assert br["overall_scope"] == "windowed" and br["overall"]["n_cases"] == len(windowed) == 72
    assert br["overall"]["n_star"] == [min(c["n_star"] for c in hits), max(c["n_star"] for c in hits)]
    lo = min(hits, key=lambda c: c["n_star"])
    assert br["overall"]["p_k"][0] == lo["p_k"] and br["overall"]["c_k"][0] == lo["c_k"]
    nw = [c["n_star"] for c in cases if c["margin"] is None]
    assert br["no_window"]["n_cases"] == 18 and br["no_window"]["n_star"] == [min(nw), max(nw)]
    for label, m in margins.items():
        s = br["by_margin"][label]
        sub = [c for c in cases if c["margin"] == label]
        assert s["n_cert"] == sorted(scaling.certification_window(gap, m))
        assert s["n_inside"] == sum(c["inside"] for c in sub)
        assert s["inside_any"] == (s["n_inside"] > 0) and s["inside_all"] == (s["n_inside"] == len(sub))
        for model in scaling.GAP_MODELS:
            assert s["n_star_min_by_model"][model] == min(c["n_star"] for c in sub if c["gap_model"] == model)
    # the c_k = 0 check is the bracket without the preparation terms
    assert [c["n_star_c0"] for c in cases] == [c["n_star"] for c in plain["cases"]]
    assert br["overall"]["n_star_c0"] == plain["overall"]["n_star"] and plain["prep"] is None
    assert all(c["c_k"] == 0.0 for c in plain["cases"] if c["n_star"] is not None)


def test_inside_window_per_gap_model():
    """Each windowed case is compared with the window of its own gap model. Inputs: fk = e^(-0.06 n), fk1 = e^(-0.03 n)
    on n = 4..10, c_iter = c_0 (N/10)^beta through (6, c_0/60), (8, c_0/6), (10, c_0), the gap through (8, 83.32) and
    (10, 69.04) mHa (gamma = 0.8425), m_E = 3 mHa: n_cert = [230.13, 413.58]. At c_0 = 50 every case has an n*
    (240-694), the smallest per gap model 242 (power) and 240 (A'/n): power-law cases reach their window, the A'/n
    cases do not. At c_0 = 7e5 the power-law minimum lies beyond (A/m)^(1/gamma) and the A'/n minimum between A'/m and
    (A/m)^(1/gamma): no case lies inside its own window, although the smallest n* lies below the larger end."""
    ns = np.array([4, 6, 8, 10])
    fk, fk1 = scaling.fit_log_overlap(ns, np.exp(-0.06 * ns)), scaling.fit_log_overlap(ns, np.exp(-0.03 * ns))
    gap = scaling.fit_gap([8, 10], [0.08332, 0.06904])
    gamma = math.log(0.08332 / 0.06904) / math.log(10 / 8)
    a_pow, a_inv = 0.06904 * 10 ** gamma, 10 * 0.06904
    ends = {"power": (a_pow / 3e-3) ** (1 / gamma), "inverse": a_inv / 3e-3}
    assert gap.gamma == pytest.approx(gamma, rel=1e-12) and gamma == pytest.approx(0.84252, abs=1e-5)
    assert ends["inverse"] == pytest.approx(230.1333, abs=1e-4) and ends["power"] == pytest.approx(413.578, abs=1e-3)
    betas = {"last": math.log(6.0) / math.log(10 / 8),
             "mean": (math.log(10.0) / math.log(8 / 6) + math.log(6.0) / math.log(10 / 8)) / 2, "fixed_9": 9.0}
    for c0, n_star, minima, n_inside in ((50.0, [240, 694], {"power": 242, "inverse": 240}, None),
                                         (7e5, None, None, 0)):
        cost = scaling.fit_power_law([6, 8, 10], [c0 / 60.0, c0 / 6.0, c0])
        br = scaling.break_even_bracket(fk, fk1, cost, c0, 10, gap=gap, margins={"E": 3e-3})
        s = br["by_margin"]["E"]
        ref = []
        for case in (c for c in br["cases"] if c["margin"] == "E"):
            n = _brute_n_star(fk, fk1, c0, 10, betas[case["beta"]], rate=case["rate"],
                              crossover=None if case["crossover"] is None else tuple(case["crossover"]),
                              gap=gap.as_power_law(case["gap_model"]), margin=3e-3)
            assert case["n_star"] == n
            ref.append((case["gap_model"], n, n is not None and n <= ends[case["gap_model"]]))
        per_model = {m: min(n for g, n, _ in ref if g == m) for m in scaling.GAP_MODELS}
        assert s["n_star_min_by_model"] == per_model and s["n_none"] == sum(n is None for _, n, _ in ref) == 0
        assert s["n_inside"] == sum(inside for _, _, inside in ref)
        assert s["inside_any"] == (s["n_inside"] > 0) and s["inside_all"] == (s["n_inside"] == 36)
        assert s["n_cert"] == pytest.approx([ends["inverse"], ends["power"]], rel=1e-12)
        if n_star is not None:
            assert s["n_star"] == n_star and per_model == minima and s["inside_any"]
        if n_inside is not None:
            assert per_model["power"] > ends["power"] and ends["inverse"] < per_model["inverse"] <= ends["power"]
            assert s["n_inside"] == n_inside and not s["inside_any"]


def test_inside_window_open_gap():
    """A case is inside iff Delta_1(n*) >= m in its own gap model (f = 1 at n*), also where the power law does not
    close. Delta_1 = 2.5e-4 n (fit through (8, 2) and (10, 2.5) mHa, gamma = -1) reaches m_E = 3 mHa at n = 12, with
    n_cert = inf; a constant 2 mHa (gamma = 0) never does, n_cert = 0; A'/n closes at A'/m (8.33 and 6.67). Inputs:
    fk = e^(-0.06 n), fk1 = e^(-0.03 n), c_iter = c_0 (N/10)^beta through (6, c_0/60), (8, c_0/6), (10, c_0). At
    c_0 = 1e-3 every n* is n_0 = 10, outside every window (f = 1.2 or 1.5); at c_0 = 10 the power-law cases of the
    growing gap are inside and all others outside."""
    ns = np.array([4, 6, 8, 10])
    fk, fk1 = scaling.fit_log_overlap(ns, np.exp(-0.06 * ns)), scaling.fit_log_overlap(ns, np.exp(-0.03 * ns))
    grow, flat = scaling.fit_gap([8, 10], [0.002, 0.0025]), scaling.fit_gap([8, 10], [0.002, 0.002])
    deltas = {"grow": {"power": lambda n: 2.5e-4 * n, "inverse": lambda n: 0.025 / n},
              "flat": {"power": lambda n: 0.002, "inverse": lambda n: 0.02 / n}}
    ends = {"grow": {"power": math.inf, "inverse": 0.025 / 3e-3}, "flat": {"power": 0.0, "inverse": 0.02 / 3e-3}}
    for name, g in (("grow", grow), ("flat", flat)):
        for c0 in (1e-3, 10.0):
            cost = scaling.fit_power_law([6, 8, 10], [c0 / 60.0, c0 / 6.0, c0])
            br = scaling.break_even_bracket(fk, fk1, cost, c0, 10, gap=g, margins={"E": 3e-3})
            cases = [c for c in br["cases"] if c["margin"] == "E"]
            for case in cases:
                n, model = case["n_star"], case["gap_model"]
                delta = deltas[name][model](n)
                assert case["f"] == pytest.approx(max(1.0, 3e-3 / delta), rel=1e-12)
                assert case["inside"] == (delta >= 3e-3) == (case["f"] == 1.0)
                assert case["n_cert"] == pytest.approx(ends[name][model], rel=1e-12)
                if c0 == 1e-3:
                    assert n == 10 and not case["inside"]
                else:
                    assert case["inside"] == (name == "grow" and model == "power") and n > 12
            s = br["by_margin"]["E"]
            assert s["n_inside"] == sum(c["inside"] for c in cases) == (18 if (name, c0) == ("grow", 10.0) else 0)
            assert s["n_cert_by_model"] == pytest.approx(ends[name], rel=1e-12)


# Spin-free inputs of the H chains (r = 1), with the overlap fits of the R1 records at n = 4, 6, 8 (p_1, p_2), the
# k = 1 costs T_res(1)/C_QPE at N = 6, 8 (n_0 = 8) and Delta_1 at n = 8, 10 (Ha), without preparation terms.
_SF_CHAIN_INPUTS = {
    "3.0": dict(p1=(0.9469740553435715, 0.8830993121229678, 0.8165522851380947),
                p2=(0.9814969855415664, 0.9470441772047921, 0.9002202314362651),
                c=(1.806615405253286, 27.028931504233057), gap=(0.08331873279164093, 0.0690440375208432)),
    "2.5": dict(p1=(0.9589042611811888, 0.9139365991374815, 0.8682822272394396),
                p2=(0.9897986977970322, 0.9665543884637923, 0.9365209610838182),
                c=(2.072922897246114, 32.31254522912679), gap=(0.15927333520628117, 0.13209058981797916)),
}


@pytest.mark.parametrize("R, n_inside, minima", [("3.0", 1, {"power": 410, "inverse": 396}),
                                                 ("2.5", 6, {"power": 696, "inverse": 678})])
def test_inside_window_regression_values(R, n_inside, minima):
    """The window flag per case on H-chain inputs (spin-free, r = 1, m_E = 3 mHa): every case's n* equals the reference
    scan with the rates, exponents and gap models computed by hand, and a case is inside iff n* <= the window end of
    its own gap model. At R = 3.0 one case of 36 is inside (Aitken rate, beta = 9, crossover (20, 2), power law: n* =
    410 against (A/m)^(1/gamma) = 414.2), although the smallest n* (396, an A'/n case) lies below the larger end; at
    R = 2.5 six cases are inside, all power-law cases with the crossover to N^2."""
    x = _SF_CHAIN_INPUTS[R]
    ns = (4, 6, 8)
    f1 = scaling.fit_log_overlap(ns, x["p1"])
    f2 = scaling.fit_log_overlap(ns, x["p2"])
    cost = scaling.fit_power_law([6, 8], x["c"])
    gap = scaling.fit_gap([8, 10], x["gap"])
    br = scaling.break_even_bracket(f1, f2, cost, x["c"][1], 8, gap=gap, margins={"E": 3e-3})

    def rates(ps):            # successive rates on n = 4, 6, 8; two rates: the Aitken fallback 2 a_8 - a_6
        a6, a8 = math.log(ps[0] / ps[1]) / 2, math.log(ps[1] / ps[2]) / 2
        return {"last": a8, "aitken": 2 * a8 - a6}

    r1, r2 = rates(x["p1"]), rates(x["p2"])
    beta = math.log(x["c"][1] / x["c"][0]) / math.log(8 / 6)
    gamma = math.log(x["gap"][0] / x["gap"][1]) / math.log(10 / 8)
    a_pow, a_inv = x["gap"][1] * 10 ** gamma, 10 * x["gap"][1]
    models = {"power": SimpleNamespace(A=a_pow, gamma=gamma), "inverse": SimpleNamespace(A=a_inv, gamma=1.0)}
    ends = {"power": (a_pow / 3e-3) ** (1 / gamma), "inverse": a_inv / 3e-3}
    cases = [c for c in br["cases"] if c["margin"] == "E"]
    inside = []
    for case in cases:
        fk = SimpleNamespace(p=x["p1"], ns=ns, rate=r1.__getitem__)
        fk1 = SimpleNamespace(p=x["p2"], ns=ns, rate=r2.__getitem__)
        b = {"last": beta, "mean": beta, "fixed_9": 9.0}[case["beta"]]
        n = _brute_n_star(fk, fk1, x["c"][1], 8, b, rate=case["rate"],
                          crossover=None if case["crossover"] is None else tuple(case["crossover"]),
                          gap=models[case["gap_model"]], margin=3e-3)
        assert case["n_star"] == n and n is not None
        assert case["n_cert"] == pytest.approx(ends[case["gap_model"]], rel=1e-12)
        assert case["inside"] == (n <= ends[case["gap_model"]])
        if case["inside"]:
            inside.append(case)
    s = br["by_margin"]["E"]
    assert s["n_inside"] == len(inside) == n_inside and s["n_cases"] == len(cases) == 36
    assert s["n_star_min_by_model"] == minima and s["inside_any"] and not s["inside_all"]
    assert all(c["gap_model"] == "power" and c["crossover"] == [20, 2.0] for c in inside)
    assert min(c["n_star"] for c in cases) < ends["power"]
    if R == "3.0":
        (c,) = inside
        assert (c["rate"], c["beta"], c["n_star"]) == ("aitken", "fixed_9", 410)
        assert ends["power"] == pytest.approx(414.21, abs=5e-3) and ends["inverse"] == pytest.approx(230.15, abs=5e-3)


def test_break_even_bracket_crossed_corners():
    """The crossed corners, p_k at the last rate with p_k+1 at the Aitken limit and the reverse, are a labelled
    sensitivity: every crossed case equals the reference scan with the two rates and reports p_k and p_k+1 at n* at
    its own two rates, and the matched summaries hold the matched cases only."""
    fk, fk1 = _geometric_fit(-0.09, 0.07, 0.05, 0.6), _geometric_fit(-0.05, 0.054, 0.01, 0.5)
    cost = scaling.fit_power_law([6, 8, 10], [0.4, 12.0, 166.0])
    gap = scaling.fit_gap([8, 10], [0.017, 0.0141])
    br = scaling.break_even_bracket(fk, fk1, cost, 166.0, 10, gap=gap, margins={"E": 3e-3})
    crossed = br["crossed"]["cases"]
    assert {(c["rate_k"], c["rate_k1"]) for c in crossed} == {("last", "aitken"), ("aitken", "last")}
    assert {c["rate"] for c in crossed} == {"last/aitken", "aitken/last"}
    differs = 0
    for case in crossed:
        kw = _case_args(case, cost)
        if case["gap_model"] is not None:
            kw.update(gap=gap.as_power_law(case["gap_model"]), margin=3e-3)
        assert case["n_star"] == _brute_n_star(fk, fk1, 166.0, 10, rate=case["rate_k"], rate_k1=case["rate_k1"], **kw)
        if case["n_star"] is not None:
            n = case["n_star"]
            assert case["p_k"] == pytest.approx(math.exp(_ln_p(fk, n, case["rate_k"])), rel=1e-12)
            assert case["p_k1"] == pytest.approx(math.exp(_ln_p(fk1, n, case["rate_k1"])), rel=1e-12)
        matched = [c["n_star"] for c in br["cases"] if all(c[k] == case[k] for k in ("beta", "crossover", "margin",
                                                                                      "gap_model"))]
        differs += case["n_star"] not in matched
    assert differs >= 1
    hits = [c["n_star"] for c in br["cases"] if c["margin"] is not None and c["n_star"] is not None]
    assert br["overall"]["n_star"] == [min(hits), max(hits)] and br["overall"]["n_cases"] == 36
    chits = [c["n_star"] for c in crossed if c["margin"] is not None and c["n_star"] is not None]
    assert br["crossed"]["overall"]["n_star"] == [min(chits), max(chits)]
    assert br["crossed"]["by_margin"]["E"]["n_cases"] == 36


def test_no_break_even_without_gain():
    """If alpha_k+1 >= alpha_k and a_k+1 <= a_k the gain p_k+1/p_k <= 1 at every n: no n* for any cost, exponent,
    crossover, window or rate."""
    fk, fk1 = _fit_from(-0.05, 0.05), _fit_from(-0.08, 0.055)
    gap = scaling.fit_gap([8, 10], [0.017, 0.0141])
    for c0, beta, cross in ((1e-9, 0.0, None), (1e-6, 2.0, (20, 2.0)), (1.0, 9.0, None)):
        for rate in scaling.RATES:
            assert scaling.break_even_size(fk, fk1, c0, 10, beta, crossover=cross, rate=rate) is None
            assert scaling.break_even_size(fk, fk1, c0, 10, beta, gap=gap, margin=3e-3, rate=rate) is None
    br = scaling.break_even_bracket(fk, fk1, scaling.fit_power_law([6, 8, 10], [0.1, 1.0, 5.0]), 5.0, 10)
    assert br["overall"]["n_star"] is None and br["overall"]["n_none"] == br["overall"]["n_cases"] == 18
    assert br["overall_scope"] == "no_window" and "by_margin" not in br


# ------------------------------------------------------------------------- costs, cheapest iteration, F25
@pytest.fixture(scope="module")
def h4_inputs():
    from thc_engine_cqe.ft import closedloop

    return closedloop.system_inputs("h4_r3.0")


def test_costs_by_r_equal_m3_and_f30(h4_inputs):
    """One residual problem per iterate for every r: the M3 costs equal ``pipeline.iteration_costs`` at each r and
    the unit-rule costs ``ft.steprules.step_rule_iteration_costs``, field for field (strict JSON compared)."""
    from thc_engine_cqe.ft import pipeline
    from thc_engine_cqe.ft import steprules as ftsr
    from thc_engine_cqe.ft.stateprep import prep_cost, readout_rotation_toffolis

    inp = h4_inputs
    act, ir, walk = inp.active, inp.irreps, inp.qpe.encoding
    res = {"elements": inp.elements, "variances": inp.variances, "normalizations": inp.normalizations}
    readout = readout_rotation_toffolis(act.norb)
    rs = (0.3, 1.0)
    rec = exact_trajectory(act, irreps=ir, max_iter=3, roots=inp.spectrum.roots, n_roots=inp.spectrum.n_roots,
                           first_step="second_order")
    pc = prep_cost(rec.directions, rec.eps, irreps=ir)
    got = scaling._iteration_costs_by_r(act, rec, pc, walk, res, r_values=rs, k_max=2, line_search=True,
                                        energy_split=inp.energy_split, readout_toffolis=readout)
    for r in rs:
        ref = pipeline.iteration_costs(act, rec, pc, walk, elements=inp.elements, variances=inp.variances,
                                       normalizations=inp.normalizations, energy_variance=inp.energy_split,
                                       readout_toffolis=readout, k_max=2, params=FTParameters(r=r))
        assert scaling.to_json(got[r]) == scaling.to_json(ref)
    run = run_step_rule(act, StepRule("unit"), max_iter=3, metric="spin_orbital", irreps=ir, keep_states=True)
    r1 = ftsr.to_trajectory_record(run, system="h4_r3.0", active=act, irreps=ir, spectrum=inp.spectrum)
    pc1 = prep_cost(r1.directions, r1.eps, irreps=ir)
    got1 = scaling._iteration_costs_by_r(act, r1, pc1, walk, res, r_values=rs, k_max=2, line_search=False,
                                         energy_split=inp.energy_split, readout_toffolis=readout, result=run)
    for r in rs:
        ref1 = ftsr.step_rule_iteration_costs(act, r1, run, pc1, walk, energy_variance=inp.energy_split,
                                              readout_toffolis=readout, r=r, k_max=2, **res)
        assert scaling.to_json(got1[r]) == scaling.to_json(ref1)
    cand = run_step_rule(act, StepRule("candidates"), max_iter=2, irreps=ir, keep_states=True)
    rc = ftsr.to_trajectory_record(cand, system="h4_r3.0", active=act, irreps=ir, spectrum=inp.spectrum)
    with pytest.raises(ValueError, match="ledger"):
        scaling._iteration_costs_by_r(act, rc, prep_cost(rc.directions, rc.eps, irreps=ir), walk, res, r_values=rs,
                                      k_max=1, line_search=False, energy_split=inp.energy_split,
                                      readout_toffolis=readout, result=cand)


def test_cheapest_iteration_routes(h4_inputs):
    """The cheapest iteration (k = 1 of the unit loop) is the R1 cost at k = 1; the closed-form direct route of
    F24 (ii) equals M3's direct_mg route exactly; T scales as r^-2; validation."""
    from thc_engine_cqe.ft.stateprep import prep_cost

    inp = h4_inputs
    act, ir, walk = inp.active, inp.irreps, inp.qpe.encoding
    res = {"elements": inp.elements, "variances": inp.variances, "normalizations": inp.normalizations}
    run = run_step_rule(act, StepRule("unit"), max_iter=2, metric="spin_free", irreps=ir, keep_states=True)
    from thc_engine_cqe.ft import steprules as ftsr
    rec = ftsr.to_trajectory_record(run, system="h4_r3.0", active=act, irreps=ir, spectrum=inp.spectrum)
    pc = prep_cost(rec.directions, rec.eps, irreps=ir)
    allr = scaling.cheapest_iteration(act, rec, pc, walk, res, r_values=(0.3, 1.0, 3.0))
    direct = scaling.cheapest_iteration(act, rec, pc, walk, {"elements": inp.elements, "variances": inp.variances},
                                        r_values=(0.3, 1.0, 3.0), routes="direct")
    costs = ftsr.step_rule_iteration_costs(act, rec, run, pc, walk, energy_variance=inp.energy_split,
                                           readout_toffolis=inp.readout_toffolis, r=1.0, k_max=1, **res)
    assert allr["by_r"]["1"]["toffolis"] == costs[1].toffolis and allr["by_r"]["1"]["best"] == costs[1].best
    for r in ("0.3", "1", "3"):
        assert direct["by_r"][r]["toffolis"] == allr["by_r"][r]["route_toffolis"]["direct_mg"]
        assert set(direct["by_r"][r]["route_toffolis"]) == {"direct_mg"}
    assert direct["by_r"]["0.3"]["toffolis"] == pytest.approx(direct["by_r"]["3"]["toffolis"] * 100.0, rel=1e-12)
    assert allr["c_qpe"] == inp.c_qpe and allr["c_prep"] == pc.cumulative[1] and allr["variance_m"] == inp.variance_m
    assert math.isnan(direct["fact_bias_m"]) and allr["fact_bias_m"] >= 0.0
    assert direct["norm_a_m"] == pytest.approx(allr["norm_a_m"], rel=1e-12)
    for kw in ({"routes": "aux"}, {"k": 5}, {"r_values": ()}, {"r_values": (1.0, -1.0)}):
        with pytest.raises(ValueError):
            scaling.cheapest_iteration(act, rec, pc, walk, res, **kw)


def test_slicing_law():
    """F25 on the h4_r4.0 exact-line-search first step (eps* = 0.52): the compiled error halves with the number of
    slices (s = 1 / s = 2 in [1.8, 2.2]) and p~_1 lies within 1% of p_1 at s = 2 [design]; the global and step errors
    of the first step coincide; the F25 rule's s; the cost is s C_step(d, eps/s)."""
    from thc_engine_cqe.ft.stateprep import elementwise_step_cost

    act = sy.benchmark_system("h4_r4.0")
    ir = sy.orbital_irreps(act)
    rec = exact_trajectory(act, irreps=ir, max_iter=1)
    psi0 = rec.states[0] * 0.0
    sp = scaling.spectrum(act, ir)
    out = scaling.compiled_overlaps(act, rec, sp.psi0, k_max=1, slices=(1, 2, 4))
    rows = {r["s"]: r for r in out["rows"]}
    assert rec.eps[0] == pytest.approx(0.522, abs=1e-3)
    assert 1.8 <= rows[1]["error"] / rows[2]["error"] <= 2.2
    assert abs(rows[2]["p_compiled"] / rows[2]["p"] - 1.0) <= 0.01
    for s, r in rows.items():
        assert r["error"] == pytest.approx(r["step_error"], abs=1e-12) and r["norm_deficit"] < 1e-12
        assert r["c_step"] == s * elementwise_step_cost(rec.directions[0], rec.eps[0] / s, irreps=ir).toffolis
    first_ok = min(s for s, r in rows.items() if r["step_error"] <= scaling.ETA_SLICE)
    (rule,) = out["rule"]
    assert (rule["k"], rule["s"], rule["flags"]) == (1, first_ok, [])
    assert {key: rule[key] for key in rows[first_ok]} == rows[first_ok]
    assert psi0.shape == sp.psi0.shape
    with pytest.raises(ValueError):
        scaling.compiled_overlaps(act, rec, sp.psi0, slices=(1, 1))


def _f25_rule_reference(act, ir, rec, k_max):
    """F25's rule by direct ``circuits.compiled_step`` calls: from HF, per step the smallest s in (1, 2, 4, 8, 16) with
    ||Psi~^(s) - exp(eps D) Psi~|| <= 0.05 (both normalized) from the rule's current state, 16 if none; [(s, error)]."""
    from thc_engine_cqe.fci_ops import Hamiltonian, normalize
    from thc_engine_cqe.ft.circuits import compiled_step
    from thc_engine_cqe.ft.trajectory import propagate

    space = FCISpace.of(act)
    ham = Hamiltonian(act)
    state = normalize(space.hf())
    out = []
    for k in range(k_max):
        d, eps = rec.directions[k], float(rec.eps[k])
        exact = propagate(ham, d, state, eps)
        for s in (1, 2, 4, 8, 16):
            psi = state
            for _ in range(s):
                psi = compiled_step(act, psi, d, eps / s, irreps=ir, rounded=True)
            psi = np.asarray(psi).reshape(space.shape)
            psi = psi / np.linalg.norm(psi)
            err = float(np.linalg.norm(psi - exact))
            if err <= 0.05:
                break
        out.append((s, err))
        state = psi
    return out


def _classical_so_record(name, max_iter):
    act = sy.benchmark_system(name)
    ir = sy.orbital_irreps(act)
    sp = scaling.spectrum(act, ir)
    rec = exact_trajectory(act, irreps=ir, max_iter=max_iter, metric="spin_orbital", first_step="second_order",
                           roots=sp.roots, n_roots=sp.n_roots)
    return act, ir, sp, rec


@pytest.mark.parametrize("name, step_error, capped", [("h4_r4.0", 0.0332, False), ("h6_r4.0", 0.0507, True)])
def test_slice_rule_continues_to_16(name, step_error, capped):
    """F25's rule on the overshooting classical spin-orbital first step of the stretched chains: s = 1, 2, 4, 8 miss
    eta_s = 0.05 and s = 16 meets it on h4_r4.0 (step error 0.0332) and misses it on h6_r4.0 (0.0507, flag
    "slice_cap") [build check]; the rule equals a direct evaluation with circuits.compiled_step, its cost is
    16 C_step(d, eps/16)."""
    from thc_engine_cqe.ft.stateprep import elementwise_step_cost

    act, ir, sp, rec = _classical_so_record(name, 1)
    out = scaling.compiled_overlaps(act, rec, sp.psi0, k_max=1)
    ((s_ref, err_ref),) = _f25_rule_reference(act, ir, rec, 1)
    (rule,) = out["rule"]
    assert rule["s"] == s_ref == 16 and rule["step_error"] == pytest.approx(err_ref, rel=1e-9, abs=1e-12)
    assert rule["step_error"] == pytest.approx(step_error, abs=5e-5)
    assert rule["flags"] == (["slice_cap"] if capped else [])
    assert rule["c_step"] == 16 * elementwise_step_cost(rec.directions[0], rec.eps[0] / 16, irreps=ir).toffolis
    assert all(r["step_error"] > scaling.ETA_SLICE for r in out["rows"])


def test_slice_rule_follows_its_own_state():
    """From k = 2 on the rule starts from its own compiled state (h4_r4.0, classical spin-orbital record: s = 16, 8
    and 1 at k = 1, 2, 3): its slice counts and step errors equal the direct evaluation from that state, while no
    same-s trajectory of s = 1, 2, 4 meets eta_s at k = 2 and the s = 1 trajectory's step error at k = 3 is that of
    another state; its cumulative cost is the sum of its steps' costs."""
    act, ir, sp, rec = _classical_so_record("h4_r4.0", 3)
    out = scaling.compiled_overlaps(act, rec, sp.psi0, k_max=3)
    ref = _f25_rule_reference(act, ir, rec, 3)
    assert [r["s"] for r in out["rule"]] == [s for s, _ in ref] == [16, 8, 1]
    assert [r["step_error"] for r in out["rule"]] == pytest.approx([e for _, e in ref], rel=1e-9, abs=1e-12)
    assert [r["flags"] for r in out["rule"]] == [[], [], []]
    assert out["rule"][2]["c_prep"] == sum(r["c_step"] for r in out["rule"])
    assert all(r["step_error"] > scaling.ETA_SLICE for r in out["rows"] if r["k"] == 2)
    row = next(r for r in out["rows"] if r["k"] == 3 and r["s"] == 1)
    assert abs(row["step_error"] - out["rule"][2]["step_error"]) > 1e-4


# ------------------------------------------------------------------------- the per-system scan and the report
def test_scan_system_and_cache(tmp_path):
    """scan_system on h4_r3.0: every record, costs at every r for k <= k_cost (k = 0 free on the classical-step
    records), the cheapest iteration of both metrics, compiled overlaps; a second scan only loads from the cache
    and gives the same strict JSON; the closed loop's inputs are the same cache entries; scan_stage fills a stage."""
    from thc_engine_cqe.ft import closedloop

    cache = scaling.Cache(tmp_path / "cache")
    rec = scaling.scan_system("h4_r3.0", cache=cache, k_cost=2, n_steps=4, r_values=(1.0, 3.0))
    assert set(rec.records) == set(scaling.RECORD_KEYS) and rec.flags == ()
    for key, r in rec.records.items():
        assert r.n_steps == 4 and len(rec.prep[key].cumulative) == 5
        assert set(rec.costs[key]) == {1.0, 3.0} and all(len(c) == 3 for c in rec.costs[key].values())
        classical = r.first_step == "second_order"
        assert (rec.costs[key][1.0][0].best == "classical") == classical
        assert (rec.costs[key][1.0][1].line_search is None) == key.startswith("r1_")
    assert rec.cheapest["sf"]["record"] == "r1_sf" and rec.cheapest["so"]["record"] == "r1_so"
    assert rec.cheapest["so"]["by_r"]["1"]["toffolis"] == rec.costs["r1_so"][1.0][1].toffolis
    assert set(rec.compiled) == {"r1_so", "lbfgs/second_order", "lbfgs_so/second_order"}
    misses = cache.misses
    again = scaling.scan_system("h4_r3.0", cache=cache, k_cost=2, n_steps=4, r_values=(1.0, 3.0))
    assert cache.misses == misses
    js = scaling.record_summary(rec)
    js2 = scaling.record_summary(again)
    assert js.pop("timings").keys() >= {"inputs", "records", "costs", "compiled"} and js2.pop("timings")
    assert json.loads(json.dumps(js, allow_nan=False)) == js2
    assert js["qpe"]["c_qpe"] == rec.c_qpe and js["records"]["r1_so"]["costs"]["1"][0]["best"] == "classical"
    assert "states" not in js["records"]["r1_so"] and js["records"]["lbfgs/exact"]["k_cert"] == 1
    hits = cache.hits
    closedloop.system_inputs("h4_r3.0", cache=cache)
    assert cache.misses == misses and cache.hits == hits + 3
    timings = scaling.scan_stage("h4_r4.0", "records", cache=cache, n_steps=4)
    assert "records" in timings
    with pytest.raises(ValueError):
        scaling.scan_stage("h4_r4.0", "fits", cache=cache)


def test_scan_large_n_path(monkeypatch):
    """The N = 10 protocol, exercised on h4_r3.0 by lowering the dense limit: 2-root spectrum, F24 (ii) totals at HF
    (the full set at K_s >= K), the closed-form direct route only, second-order records and r1_so; its direct cost at
    k = 1 equals the dense path's direct_mg route."""
    dense = scaling.scan_system("h4_r3.0", k_cost=1, n_steps=3, r_values=(1.0,))
    monkeypatch.setattr(scaling, "_N_DENSE_MAX", 3)
    big = scaling.scan_system("h4_r3.0", k_cost=1, n_steps=3, r_values=(1.0,))
    assert set(big.records) == {"lbfgs/second_order", "lbfgs_so/second_order", "r1_so"}
    assert big.spectrum.n_roots == 2 and {"no_model_spectrum", "direct_route_only", "subset_variances"} <= set(big.flags)
    assert big.cheapest["sf"]["record"] == "lbfgs/second_order" and big.cheapest["so"]["variance_method"] == "subset"
    for key in big.records:
        assert all(c.best in ("classical", "direct_mg") for c in big.costs[key][1.0])
    # T = V_m C(Psi_1)/B^2: the paths differ only by the variance state (HF here, M3's iterate 1 there; F24 (i))
    d_route = dense.cheapest["so"]["by_r"]["1"]["route_toffolis"]["direct_mg"]
    ratio = big.cheapest["so"]["variance_m"] / dense.cheapest["so"]["variance_m"]
    assert big.cheapest["so"]["by_r"]["1"]["toffolis"] == pytest.approx(d_route * ratio, rel=1e-9)
    assert ratio == pytest.approx(480.7 / 478.1, abs=1e-3)                                   # [design]
    # psi_0 of the 2-root and the 12-root FCI solves agree to the solver's convergence
    assert big.records["r1_so"].overlaps[1] == pytest.approx(dense.records["r1_so"].overlaps[1], abs=1e-9)


def _num(x):
    """A number of the strict JSON: "inf" -> inf."""
    return math.inf if x == "inf" else (-math.inf if x == "-inf" else x)


def test_scaling_report_tables():
    """scaling_report and markdown_tables on two H4 chains: U1 to U4 rows with the repetition columns, strict JSON,
    every table rendered. Every U4 row against its ScalingRecord: p_k, p_k+1 and c_iter(k) of the R1 record, c_k =
    C(Psi_k)/C_QPE and c_k+1 = C(Psi_k+1)/C_QPE of its preparation cost, mu = (1 + c_k)/p_k - (1 + c_k+1)/p_k+1, p_k* =
    [(1 + c_k) - (1 + c_k+1) p_k/p_k+1]/c_iter(k) and rho_LS = (T_res(k)/C_QPE)/mu_LS with the overlaps and preparation
    costs of the exact second-order record (residual only: the optimistic rho at zero step cost); U3's C(Psi_2) with
    the record of its Psi_2; the U2 and U4 captions state the model bracket and the model's scope."""
    recs = [scaling.scan_system(n, k_cost=2, n_steps=3, r_values=(1.0,)) for n in ("h4_r3.0", "h4_r4.0")]
    rep = scaling.scaling_report(recs)
    assert json.loads(json.dumps(rep, allow_nan=False)) == rep
    assert [r["system"] for r in rep["U1"]] == ["h4_r3.0", "h4_r4.0"]
    assert rep["U1"][0]["k_cert"]["lbfgs/exact"] == 1
    rows = [r for r in rep["U4"]["rows"] if r["system"] == "h4_r3.0" and r["metric"] == "sf"]
    assert [r["k"] for r in rows] == [0, 1, 2] and rows[0]["c_iter"] == 0.0 and rows[0]["rho"] == 0.0
    r1 = rows[1]
    assert r1["rho"] == pytest.approx(r1["c_iter"] / r1["mu"], rel=1e-12) and r1["pays"] == (r1["rho"] < 1.0)
    assert r1["mu"] == pytest.approx(0.0367, abs=5e-4) and r1["rho"] == pytest.approx(1.45, abs=0.02)   # [design]
    assert r1["p_star"] == pytest.approx(0.650, abs=1e-3)
    by_name = {r.name: r for r in recs}
    keys = {"sf": ("r1_sf", "lbfgs/second_order"), "so": ("r1_so", "lbfgs_so/second_order")}
    assert len(rep["U4"]["rows"]) == 2 * 2 * 3
    for row in rep["U4"]["rows"]:
        rec, k = by_name[row["system"]], row["k"]
        r1_key, exact_key = keys[row["metric"]]
        c_qpe, tr, pc = rec.c_qpe, rec.records[r1_key], rec.prep[r1_key]
        p, p1 = float(tr.overlaps[k]), float(tr.overlaps[k + 1])
        ck, ck1 = pc.cumulative[k] / c_qpe, pc.cumulative[k + 1] / c_qpe
        c_it = rec.costs[r1_key][row["r"]][k].toffolis / c_qpe
        assert (row["p"], row["p_next"], row["c_k"], row["c_k1"]) == (p, p1, ck, ck1)
        assert row["c_iter"] == pytest.approx(c_it, rel=1e-12)
        mu = (1.0 + ck) / p - (1.0 + ck1) / p1
        assert row["mu"] == pytest.approx(mu, rel=1e-12) and row["gain"] == pytest.approx(p1 / p, rel=1e-12)
        num = (1.0 + ck) - (1.0 + ck1) * p / p1
        p_star = num / c_it if c_it > 0.0 else (math.inf if num > 0.0 else 0.0)
        assert _num(row["p_star"]) == pytest.approx(p_star, rel=1e-12)
        er, epc = rec.records[exact_key], rec.prep[exact_key]
        res = rec.costs[exact_key][row["r"]][k].residual_toffolis / c_qpe
        mu_ls = ((1.0 + epc.cumulative[k] / c_qpe) / er.overlaps[k]
                 - (1.0 + epc.cumulative[k + 1] / c_qpe) / er.overlaps[k + 1])
        assert _num(row["rho_ls"]) == pytest.approx(res / mu_ls if mu_ls > 0.0 else math.inf, rel=1e-12)
    rec = by_name["h4_r4.0"]
    row2 = next(r for r in rep["U4"]["rows"] if r["system"] == "h4_r4.0" and r["metric"] == "so" and r["k"] == 2)
    p2, p3 = rec.records["r1_so"].overlaps[2], rec.records["r1_so"].overlaps[3]
    c2, c3 = (rec.prep["r1_so"].cumulative[j] / rec.c_qpe for j in (2, 3))
    assert row2["mu"] == pytest.approx((1.0 + c2) / p2 - (1.0 + c3) / p3, rel=1e-12)
    for row in rep["U3"]["rows"]:
        for m in ("sf", "so"):
            x = row[m]
            assert x["c_psi2_record"] == x["record"]
            assert x["c_psi2"] == by_name[row["system"]].prep[x["record"]].cumulative[2]
    assert rep["U2"]["overlap"] == [] and rep["U2"]["gap"] == [] and rep["U4"]["n_star"] == []
    md = scaling.markdown_tables(rep)
    for title in ("## U1", "## U2", "## U3", "## U4"):
        assert title in md
    for phrase in ("model bracket", "canonical orbitals", "elementwise compilation", "full allowed residual"):
        assert phrase in md
    assert "C(Psi_2) (record)" in md and "(r1_sf)" in md
    with pytest.raises(ValueError):
        scaling.scaling_report([])


def _stand_in(n, *, p_hf=0.5, gap=None, classical=None, r1=None, cheapest=None, c_qpe=1e6, R="3.0"):
    """A stand-in for the report's fits: h<n>_r<R> with the spin-free classical-step record (p_1, C(Psi_1)), the R1
    record (p_1, p_2, C(Psi_1), C(Psi_2), T_res(1)) and the cheapest iteration at the classical Psi_1
    (T_res(1)/C_QPE)."""
    records, prep, costs, cheap = {}, {}, {}, {}
    if classical is not None:
        records["lbfgs/second_order"] = SimpleNamespace(overlaps=[p_hf, classical[0]], n_steps=1)
        prep["lbfgs/second_order"] = SimpleNamespace(cumulative=[0.0, classical[1]])
    if r1 is not None:
        p1, p2, c1, c2, t1 = r1
        records["r1_sf"] = SimpleNamespace(overlaps=[p_hf, p1, p2], n_steps=2)
        prep["r1_sf"] = SimpleNamespace(cumulative=[0.0, c1, c2])
        costs["r1_sf"] = {1.0: [SimpleNamespace(toffolis=0.0), SimpleNamespace(toffolis=t1)]}
    if cheapest is not None:
        cheap["sf"] = {"k": 1, "first_step": "second_order", "record": "lbfgs/second_order",
                       "by_r": {"1": {"over_qpe": cheapest}}}
    spectrum = SimpleNamespace(p_hf=p_hf, e1=None if gap is None else gap, e0=0.0, flags=(), gap=gap)
    return SimpleNamespace(name=f"h{n}_r{R}", records=records, prep=prep, costs=costs, cheapest=cheap, c_qpe=c_qpe,
                           spectrum=spectrum)


def _by_hand_rates(ns, ps):
    """F22 by hand: the successive rates, the last one and the Aitken limit (or its fallback)."""
    lp = [math.log(p) for p in ps]
    succ = [(lp[i] - lp[i + 1]) / (ns[i + 1] - ns[i]) for i in range(len(ns) - 1)]
    if len(succ) >= 3:
        q = (succ[-1] - succ[-2]) / (succ[-2] - succ[-3])
        aitken = succ[-1] + (succ[-1] - succ[-2]) * q / (1 - q) if 0 < q < 1 else 2 * succ[-1] - succ[-2]
    else:
        aitken = 2 * succ[-1] - succ[-2]
    return succ, succ[-1], aitken


def test_n_star_inputs_from_records():
    """The report's F45 inputs from stand-in records at R = 3.0. With both records at n = 4, 6, 8: the cost points are
    the R1 k = 1 costs on N >= 6 (N = 4 excluded), n_0 = 8, c_0 = c(8) and one successive exponent
    ln(c_8/c_6)/ln(8/6). Adding n = 10 with the second-order record and the cheapest iteration only (no R1 record, as
    sf at N = 10): alpha_1 comes from the four-point classical series (its last rate and Aitken limit), alpha_2 from the
    R1 records at n <= 8, the n = 10 cost is the cheapest iteration's (n_0 = 10, two successive exponents), c_1 is a
    power law through N = 8, 10 and c_2 through N = 6, 8, each anchored at its last size."""
    rates = [0.05 - 0.02 * 0.6 ** (n / 2) for n in (6, 8, 10)]
    lp = [math.log(0.95)]
    for a in rates:
        lp.append(lp[-1] - 2 * a)
    p_cl = [math.exp(x) for x in lp]
    p2 = [0.98, 0.95, 0.90]
    t1 = {4: 5e4, 6: 1.8e6, 8: 2.7e7}
    c_psi1 = {4: 160.0, 6: 969.0, 8: 3528.0, 10: 9000.0}
    c_psi2 = {4: 1352.0, 6: 9451.0, 8: 33828.0}
    base = [_stand_in(n, classical=(p_cl[i], c_psi1[n]), r1=(p_cl[i], p2[i], c_psi1[n], c_psi2[n], t1[n]))
            for i, n in enumerate((4, 6, 8))]
    (b,) = scaling._n_star({"3.0": base}, {"3.0": None}, {})
    c = {n: t / 1e6 for n, t in t1.items()}
    assert (b["n0"], b["c0"], b["metric"], b["r"], b["k"]) == (8, c[8], "sf", 1.0, 1)
    assert b["cost_successive"] == pytest.approx([math.log(c[8] / c[6]) / math.log(8 / 6)], rel=1e-12)
    assert b["inputs"]["cost"] == {"ns": [6, 8], "c_iter": [c[6], c[8]], "source": ["r1_sf", "r1_sf"]}
    _, last3, ait3 = _by_hand_rates([4, 6, 8], p_cl[:3])
    assert b["alpha_k"] == pytest.approx([last3, ait3], rel=1e-12)
    assert "by_margin" not in b and b["overall_scope"] == "no_window"
    ten = _stand_in(10, classical=(p_cl[3], c_psi1[10]), cheapest=209.0)
    (b10,) = scaling._n_star({"3.0": base + [ten]}, {"3.0": None}, {})
    _, last4, ait4 = _by_hand_rates([4, 6, 8, 10], p_cl)
    assert ait4 == pytest.approx(0.05, abs=1e-12)
    assert b10["alpha_k"] == pytest.approx([last4, ait4], rel=1e-12)
    _, last2, ait2 = _by_hand_rates([4, 6, 8], p2)
    assert b10["alpha_k1"] == pytest.approx([last2, ait2], rel=1e-12)
    assert (b10["n0"], b10["c0"]) == (10, 209.0) and len(b10["cost_successive"]) == 2
    assert b10["cost_successive"][1] == pytest.approx(math.log(209.0 / c[8]) / math.log(10 / 8), rel=1e-12)
    assert b10["inputs"]["cost"]["source"] == ["r1_sf", "r1_sf", "cheapest:lbfgs/second_order"]
    assert b10["inputs"]["alpha_k"]["ns"] == [4, 6, 8, 10] and b10["inputs"]["alpha_k1"]["ns"] == [4, 6, 8]
    prep = b10["prep"]
    assert prep["c_k"]["n0"] == 10.0 and prep["c_k"]["c0"] == pytest.approx(9000.0 / 1e6, rel=1e-12)
    assert prep["c_k"]["beta"] == pytest.approx(math.log(9000.0 / 3528.0) / math.log(10 / 8), rel=1e-12)
    assert prep["c_k1"]["n0"] == 8.0 and prep["c_k1"]["beta"] == pytest.approx(math.log(33828.0 / 9451.0)
                                                                                / math.log(8 / 6), rel=1e-12)
    # the bracket is break_even_bracket of these inputs
    fk = scaling.fit_log_overlap([4, 6, 8, 10], p_cl)
    fk1 = scaling.fit_log_overlap([4, 6, 8], p2, window=(4, 6, 8))
    case = b10["cases"][0]
    assert case["n_star"] == _brute_n_star(fk, fk1, 209.0, 10, case["beta_value"], rate=case["rate"],
                                           prep=[(p["c0"], p["n0"], p["beta"]) for p in (prep["c_k"], prep["c_k1"])])


def test_u2_fit_window_p1_and_one_run_column():
    """U2 from stand-in chains (n = 4, 6, 8; ln p not linear in n): the classical series' alpha is the slope on n = 6,
    8 only (0.10720 for p = 0.7, 0.57, 0.46), not the fit over every size; p_1 at the HF's n(p = 1e-3) is extrapolated
    at the rate of each end (last, then Aitken); the gap row carries the one-run B_L window of HF and the classical
    Psi_1 (certification_window_overlap per gap model and rate), p at both ends of the two-run window n_cert(m_BL),
    the states below the floor 3.36e-5 there and the two-run window capped where p falls to the floor."""
    p_hf, p_cl, gaps = [0.5, 0.3, 0.17], [0.7, 0.57, 0.46], [0.2, 0.15, 0.12]
    recs = [_stand_in(n, p_hf=p_hf[i], gap=gaps[i], classical=(p_cl[i], 100.0)) for i, n in enumerate((4, 6, 8))]
    margins = scaling.acceptance_margins(1e-3, 1.0, 1e-3)
    u2 = scaling._u2({"3.0": recs}, margins, de_qpe=1e-3, sin_theta0=1.0, e1_margin=1e-3)
    rows = {r["record"]: r for r in u2["overlap"]}
    assert set(rows) == {"hf", "classical_sf"}
    assert rows["classical_sf"]["alpha"] == pytest.approx(math.log(0.57 / 0.46) / 2, rel=1e-12)
    assert rows["classical_sf"]["alpha"] == pytest.approx(0.10720, abs=5e-6)
    assert rows["classical_sf"]["alpha"] != pytest.approx(
        scaling.fit_log_overlap([4, 6, 8], p_cl, window=(4, 6, 8)).alpha, abs=1e-3)
    _, hf_last, hf_ait = _by_hand_rates([4, 6, 8], p_hf)
    _, cl_last, cl_ait = _by_hand_rates([4, 6, 8], p_cl)
    n_at = [8 + math.log(0.17 / 1e-3) / a for a in (hf_last, hf_ait)]
    assert rows["hf"]["n_at_p"] == pytest.approx(n_at, rel=1e-12)
    expected_p1 = [0.46 * math.exp(-a * (n - 8)) for a, n in zip((cl_last, cl_ait), n_at)]
    assert rows["hf"]["p1_at_n"]["sf"] == pytest.approx(expected_p1, rel=1e-12)
    (g,) = u2["gap"]
    fits = u2["_fits"]
    gfit = fits[("3.0", "gap")]
    for state in ("hf", "classical_sf"):
        for model in scaling.GAP_MODELS:
            assert g["bl_one_run"][state][model] == [scaling.certification_window_overlap(
                gfit, fits[("3.0", state)], de_qpe=1e-3, sin_theta0=1.0, e1_margin=1e-3, rate=rate, model=model)
                for rate in scaling.RATES]
    assert set(g["bl_one_run"]) == {"hf", "classical_sf"}
    gamma = math.log(0.15 / 0.12) / math.log(8 / 6)
    lo, hi = 8 * 0.12 / 12e-3, (0.12 * 8 ** gamma / 12e-3) ** (1 / gamma)
    two = g["bl_two_run"]
    assert two["n_cert"] == pytest.approx([lo, hi], rel=1e-12)
    floor = scaling.two_run_overlap_floor()
    below = []
    for state, p8, rates in (("hf", 0.17, (hf_last, hf_ait)), ("classical_sf", 0.46, (cl_last, cl_ait))):
        ends = {"inverse": [p8 * math.exp(-a * (lo - 8)) for a in rates],
                "power": [p8 * math.exp(-a * (hi - 8)) for a in rates]}
        for model, ps in ends.items():
            assert two["p_at_n_cert"][state][model] == pytest.approx(ps, rel=1e-12)
        n_floor = [8 + math.log(p8 / floor) / a for a in rates]
        assert two["n_cert_sound"][state]["inverse"] == pytest.approx([min(lo, n) for n in n_floor], rel=1e-12)
        assert two["n_cert_sound"][state]["power"] == pytest.approx([min(hi, n) for n in n_floor], rel=1e-12)
        if any(p < floor for ps in ends.values() for p in ps):
            below.append(state)
    assert two["below_floor"] == below and "hf" in below and two["n_cert_sound"]["hf"]["power"][1] < hi
    md = scaling.markdown_tables({"settings": {**{"margins": margins, "sin_theta0": 1.0, "e1_margin": 1e-3, "d_l": 5},
                                               "p_floor_two_run": floor},
                                  "U1": [], "U2": {"overlap": u2["overlap"], "gap": scaling.to_json(u2["gap"])},
                                  "U3": {"rows": [], "exponents": []}, "U4": {"rows": [], "n_star": []}})
    assert "B_L without the verification run" in md and "(capped at " in md and "power, aitken: " in md


@pytest.mark.slow
def test_scaling_report_fits(monkeypatch):
    """The fit and break-even path of the report on h4_r3.0 and h6_r3.0 (cost exponents from N >= 4 here): rates
    and the gap fit per R, the cost exponents and an n* bracket per metric and r, rendered; break_even_sizes gives
    the same brackets under repetition and, under a scheme's F44 hook (bbht), the cases of break_even_size with
    alpha_1 from the classical-step series, alpha_2 from the R1 record and the preparation terms C(Psi_1)/C_QPE
    (classical-step record) and C(Psi_2)/C_QPE (R1 record) as power laws."""
    monkeypatch.setattr(scaling, "_N_COST_MIN", 4)
    recs = [scaling.scan_system(n, k_cost=1, n_steps=3, r_values=(1.0,)) for n in ("h4_r3.0", "h6_r3.0")]
    rep = scaling.scaling_report(recs)
    hf = next(r for r in rep["U2"]["overlap"] if r["record"] == "hf")
    assert hf["successive"] == pytest.approx([0.1007], abs=5e-4) and hf["ns"] == [4, 6]
    (g,) = rep["U2"]["gap"]
    assert g["gaps"] == pytest.approx([0.13896, 0.10467], abs=1e-5)
    assert {e["quantity"] for e in rep["U3"]["exponents"]} >= {"c_qpe", "K", "t_res1_over_qpe_sf_r1"}
    assert {(b["metric"], b["r"]) for b in rep["U4"]["n_star"]} == {("sf", 1.0), ("so", 1.0)}
    assert all(set(b["by_margin"]) == {"E", "B", "BL"} for b in rep["U4"]["n_star"])
    assert "Break-even size" in scaling.markdown_tables(rep)
    # break_even_sizes: the report's brackets with saving None and with repetition as an F44 hook; under bbht
    # every case equals a direct break_even_size call with that hook
    from thc_engine_cqe.ft import amplify

    assert scaling.break_even_sizes(recs) == rep["U4"]["n_star"]
    assert scaling.break_even_sizes(recs, saving=amplify.scheme_saving("rep"), scheme="rep") == rep["U4"]["n_star"]
    sv = amplify.scheme_saving("bbht")
    under = scaling.break_even_sizes(recs, saving=sv, scheme="bbht")
    assert [(b["R"], b["metric"], b["r"]) for b in under] == [(b["R"], b["metric"], b["r"]) for b in rep["U4"]["n_star"]]
    exact_keys = {"sf": "lbfgs/second_order", "so": "lbfgs_so/second_order"}
    for b in under:
        assert b["scheme"] == "bbht"
        r1, ex = "r1_" + b["metric"], exact_keys[b["metric"]]
        ns = [int(r.name[1]) for r in recs]
        fk = scaling.fit_log_overlap(ns, [r.records[ex].overlaps[1] for r in recs], window=tuple(ns))
        fk1 = scaling.fit_log_overlap(ns, [r.records[r1].overlaps[2] for r in recs], window=tuple(ns))
        assert b["alpha_k"] == [fk.alpha_last, fk.alpha_aitken] and b["alpha_k1"] == [fk1.alpha_last, fk1.alpha_aitken]
        c1 = scaling.fit_power_law(ns, [r.prep[ex].cumulative[1] / r.c_qpe for r in recs])
        c2 = scaling.fit_power_law(ns, [r.prep[r1].cumulative[2] / r.c_qpe for r in recs])
        assert b["prep"] == {"c_k": {"c0": c1.y[-1], "n0": c1.x[-1], "beta": c1.beta_last},
                             "c_k1": {"c0": c2.y[-1], "n0": c2.x[-1], "beta": c2.beta_last}}
        for case in b["cases"][:12]:
            if case["margin"] is not None:
                continue
            ref = scaling.break_even_size(fk, fk1, b["c0"], b["n0"], case["beta_value"],
                                          crossover=None if case["crossover"] is None else tuple(case["crossover"]),
                                          saving=sv, rate=case["rate"], prep=(c1, c2))
            assert case["n_star"] == ref
    with pytest.raises(ValueError):
        scaling.break_even_sizes([])


# ------------------------------------------------------------------------- slow regressions (N = 8, 10)
@pytest.fixture(scope="session")
def h10_solutions():
    """H10 at R = 3.0 (2 singlet roots: E0, psi_0, Delta_1) and R = 4.0 (1 root: overlaps), with the classical
    first step of both metrics (F16) for the holdouts."""
    from thc_engine_cqe.ft.trajectory import f16, propagate

    out = {}
    for name, n_roots in (("h10_r3.0", 2), ("h10_r4.0", 1)):
        act = sy.benchmark_system(name)
        ir = sy.orbital_irreps(act)
        sp = scaling.spectrum(act, ir, n_roots=n_roots)
        out[name] = {"spectrum": sp, "p1": _classical_overlaps(act, ir, sp.psi0)}
    return out


def _classical_overlaps(act, ir, psi0):
    """p_1 after the classical F16 step from HF, per metric."""
    from thc_engine_cqe.fci_ops import Hamiltonian, inner, normalize
    from thc_engine_cqe.ft.trajectory import f16, propagate

    ham = Hamiltonian(act)
    hf = normalize(ham.space.hf())
    out = {}
    for label, metric in (("sf", "spin_free"), ("so", "spin_orbital")):
        _, d, slope, d2e = f16(act, hf, ham, metric, ir)
        out[label] = abs(complex(inner(psi0, propagate(ham, d, hf, -slope / d2e)))) ** 2
    return out


@pytest.mark.slow
def test_holdout_h10_from_h8(h10_solutions):
    """F22 holdout: p_HF and the classical p_1 (both metrics) of H10 lie within 3% of the extrapolation from
    n = 4, 6, 8 at R = 3.0 and 4.0 bohr, at both rates of the bracket [design]."""
    for R in ("3.0", "4.0"):
        ps = {"hf": [], "sf": [], "so": []}
        for n in (4, 6, 8):
            act = sy.benchmark_system(f"h{n}_r{R}")
            ir = sy.orbital_irreps(act)
            sp = scaling.spectrum(act, ir, n_roots=1)
            p1 = _classical_overlaps(act, ir, sp.psi0)
            ps["hf"].append(sp.p_hf)
            ps["sf"].append(p1["sf"])
            ps["so"].append(p1["so"])
        h10 = h10_solutions[f"h10_r{R}"]
        computed = {"hf": h10["spectrum"].p_hf, **h10["p1"]}
        for label, values in ps.items():
            fit = scaling.fit_log_overlap([4, 6, 8], values, window=(4, 6, 8))
            for rate in scaling.RATES:
                assert scaling.extrapolate_overlap(fit, 10, rate=rate) == pytest.approx(computed[label], rel=0.03), \
                    (R, label, rate)


@pytest.mark.slow
def test_regression_pins(h10_solutions):
    """[design] p_HF and Delta_1 of h8_r4.0 (0.1685, 16.98 mHa) and h10_r3.0 (0.3746, 69.04 mHa); eps_2/eps* of the
    spin-free classical step on h8_r4.0 (1.598); C_QPE(h8_r3.0) = 18,661,951 Toffolis (THC M = 16)."""
    from thc_engine_cqe.diagnostics.thc_floors import ExactReference
    from thc_engine_cqe.ft.qpe import scan_encodings, select_qpe
    from thc_engine_cqe.ft.trajectory import second_order_step

    act = sy.benchmark_system("h8_r4.0")
    ir = sy.orbital_irreps(act)
    sp = scaling.spectrum(act, ir)
    assert sp.p_hf == pytest.approx(0.1685, abs=5e-5) and 1e3 * sp.gap == pytest.approx(16.98, abs=5e-3)
    assert second_order_step(act, metric="spin_free", irreps=ir)["ratio"] == pytest.approx(1.598, abs=5e-4)
    h10 = h10_solutions["h10_r3.0"]["spectrum"]
    assert h10.p_hf == pytest.approx(0.3746, abs=5e-5) and 1e3 * h10.gap == pytest.approx(69.04, abs=5e-3)
    assert "e1_lower_bound" not in h10.flags
    a8 = sy.benchmark_system("h8_r3.0")
    sel = select_qpe(scan_encodings(a8, reference=ExactReference.build(a8)))
    assert sel.encoding.kind == "thc" and sel.encoding.params["M"] == 16 and sel.toffolis == 18_661_951

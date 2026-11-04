"""THC floors: how a factorized two-body tensor changes the quantities the CQE uses.

Checks the relations stated in thc_engine_cqe/diagnostics/thc_floors.py:
the variational sandwich, the floor identities (sigma^2[H'](Psi0) = Var_Psi0(V),
A[H'](Psi0) = A[V](Psi0)), the perturbative orders under a scaled error, the H4
scan at an exact and a truncated THC rank, the spin-sector guard and the
OpenFermion cost wrapper.

Reproducibility note: refined THC fits are reproducible only when L-BFGS-B converges
to a minimum with an objective well above zero. Fits that stop at the iteration limit,
or that creep towards an exact fit, end wherever round-off steers them. The scan test
therefore uses M = 1.5 N on H4. Its unit-gauge refinement converges (2460-2730
iterations observed, f = 1.585046e-4) and gave the same floors to all printed digits
for 1 and 4 threads. M = 2N on H4 has 68 parameters for 55 unique integrals, and its
refinement creeps towards an exact fit. The other tests use least-squares fits only.
"""

import argparse
import functools
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest

from conftest import random_active_space
from thc_engine_cqe import systems
from thc_engine_cqe.diagnostics.thc_floors import (FT_DEFAULTS, NOT_CONVERGED_MARK,
                                                    ExactReference, auto_ranks, format_markdown,
                                                    lowest_state, model_floors, spin_square,
                                                    thc_floor_scan, thc_ft_cost)
from thc_engine_cqe.factor.df import double_factorize
from thc_engine_cqe.factor.thc import fit_thc
from thc_engine_cqe.fci_ops import Hamiltonian, inner
from thc_engine_cqe.integrals import ActiveSpace
from thc_engine_cqe.residual import compute_residual

ROOT = pathlib.Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=None)
def h4():
    return systems.hydrogen_chain(4, 1.8, "bohr")


@functools.lru_cache(maxsize=None)
def h4_reference():
    return ExactReference.build(h4())


@functools.lru_cache(maxsize=None)
def h4_scan():
    """THC at M = 1.5 N (refined to convergence, truncated) and M = N(N+1)/2 (exact),
    plus DF rows. maxiter leaves a margin over the observed 2460-2730 iterations."""
    return thc_floor_scan(h4(), [6, 10], reference=h4_reference(),
                          fit_kwargs={"maxiter": 20000})


@functools.lru_cache(maxsize=None)
def h4_ls_error(nthc=8):
    """Error tensor dV = eri_THC - eri of a least-squares (unrefined) THC fit."""
    act = h4()
    fac = fit_thc(act.eri, nthc, mol=act.mol, mo_coeff=act.mo_coeff, refine=False)
    return fac.to_eri() - act.eri


def _operator_space(active, eri):
    """The operator V = 1/2 sum eri e (no one-body part, no constant)."""
    n = active.norb
    return ActiveSpace(h1=np.zeros((n, n)), eri=eri, ecore=0.0, nelec=active.nelec)


def _by_label(rows):
    return {r["label"]: r for r in rows}


# ------------------------------------------------------------------- the scan
def test_scan_exact_rank_floors_vanish():
    """At M = N(N+1)/2 LS-THC is exact, so every floor is zero up to the solver noise."""
    ref = h4_reference()
    assert ref.residual.residual_norm < 1e-6 and ref.residual.variance < 1e-12
    row = _by_label(h4_scan())["THC M=10"]
    assert row["kind"] == "thc" and row["M"] == 10 and row["M_over_N"] == 2.5
    assert row["eri_max_abs_err"] < 1e-10
    for key in ("de_fci", "de_pt1", "de_exact_at_model", "de_model_at_exact"):
        assert abs(row[key]) < 1e-9, key
    assert row["res_noise"] < 1e-6
    # A[H'](Psi0) = A[H](Psi0) + A[V](Psi0) and ||A[V]|| ~ ||dV||, so both residual
    # floors equal the solver noise of the FCI vectors.
    assert row["res_model_at_exact"] <= row["res_noise"] + 1e-9
    assert row["res_exact_at_model"] <= row["res_noise"] + 1e-9
    assert row["var_model_at_exact"] < 1e-12 and row["var_exact_at_model"] < 1e-12
    assert row["infidelity"] < 1e-12
    assert row["spin_ok"] and abs(row["s2_model"]) < 1e-8 and row["spin_shift"] is None


def test_scan_truncated_rank_is_second_order_and_floors_are_positive():
    """M = 1.5 N: the exact energy of the THC ground state is far below the first-order
    error, matches the other second-order quantity, and the floors are well above noise."""
    row = _by_label(h4_scan())["THC M=6"]
    details = row["details"]
    assert details["refine_success"] is True and details["refine_nit"] < 20000
    assert np.isclose(details["refine_f"], 1.585046e-4, rtol=1e-4)  # converged minimum
    assert details["penalty"] == 0.0 and details["penalty_rule"] == "fixed"
    assert 1e-4 < row["eri_max_abs_err"] < 1e-1
    # observed: de_fci = -1.869e-2, de_exact_at_model = 3.35e-5 (ratio 0.0018)
    assert abs(row["de_fci"]) > 1e-5
    assert 0.0 < row["de_exact_at_model"] < 0.2 * abs(row["de_fci"])
    # variational sandwich: de_fci <= <V>_Psi0 and E_H'(Psi0) >= E0'
    assert row["de_fci"] <= row["de_pt1"] + 1e-10
    assert row["de_model_at_exact"] >= -1e-10
    # the two second-order energies agree to leading order (observed 3.349e-5, 3.404e-5)
    assert np.isclose(row["de_exact_at_model"], row["de_model_at_exact"], rtol=0.1)
    # floors are positive and far above the solver noise
    noise, vnoise = row["res_noise"], row["var_noise"]
    assert row["res_model_at_exact"] > 1e3 * noise and row["res_exact_at_model"] > 1e3 * noise
    assert row["var_model_at_exact"] > 1e3 * max(vnoise, 1e-15)
    assert row["var_exact_at_model"] > 1e3 * max(vnoise, 1e-15)
    assert 0.0 < row["infidelity"] < 1e-2
    # the two residual floors (and the two variance floors) agree to leading order
    assert np.isclose(row["res_model_at_exact"], row["res_exact_at_model"], rtol=0.1)
    assert np.isclose(row["var_model_at_exact"], row["var_exact_at_model"], rtol=0.1)
    # lambda bookkeeping (OpenFermion convention: T' from the exact ERIs)
    assert np.isclose(row["lambda"], row["lambda_one_body"] + row["lambda_two_body"])
    assert row["details"]["guess"] == "grid" and row["details"]["refine"] is True
    assert row["n_params"] == 6 * 4 + 6 * 7 // 2


def test_scan_df_rows_and_serialization():
    rows = h4_scan()
    df = [r for r in rows if r["kind"] == "df"]
    thc = [r for r in rows if r["kind"] == "thc"]
    assert [r["M"] for r in thc] == [6, 10] and rows[:2] == thc  # THC rows first, in order
    assert len(df) >= 3
    thr = [r["eig_threshold"] for r in df]
    assert thr == sorted(thr, reverse=True)
    keys = {(r["n_leaves"], r["n_eigenvectors"]) for r in df}
    assert len(keys) == len(df)  # duplicate factorizations are skipped
    lam_t = {round(r["lambda_one_body"], 12) for r in rows}
    assert len(lam_t) == 1  # T' uses the exact ERIs for every model
    for r in rows:
        assert set(r) >= {"de_fci", "de_pt1", "de_exact_at_model", "res_model_at_exact",
                          "var_model_at_exact", "res_exact_at_model", "eri_fro_rel_err"}
        assert r["de_exact_at_model"] >= -1e-10 and r["de_model_at_exact"] >= -1e-10
        assert r["de_fci"] <= r["de_pt1"] + 1e-10
        assert r["spin_ok"]
        if r["kind"] == "df":
            assert r["n_params"] == r["n_eigenvectors"] * 5 and r["M"] is None
    assert df[-1]["eri_max_abs_err"] < 1e-4  # smallest threshold: nearly exact
    json.loads(json.dumps(rows))  # plain JSON types only
    # dedup rationale: equal (L, Xi) at two thresholds means identical factors
    kw = dict(first="eigh", tol=0.0, threshold_rule="openfermion")
    a = double_factorize(h4().eri, eig_threshold=1e-3, **kw)
    b = double_factorize(h4().eri, eig_threshold=3e-4, **kw)
    assert (a.n_leaves, a.n_eigenvectors) == (b.n_leaves, b.n_eigenvectors)
    assert np.array_equal(a.to_eri(), b.to_eri())


def test_model_floors_rejects_unsymmetric_tensor():
    act, ref = h4(), h4_reference()
    bad = act.eri.copy()
    bad[0, 1, 2, 3] += 1e-3
    with pytest.raises(ValueError):
        model_floors(ref, bad)


# -------------------------------------------------------------- exact relations
def test_floor_identities_and_variational_sandwich():
    """sigma^2[H'](Psi0) = Var_Psi0(V), A[H'](Psi0) = A[H](Psi0) + A[V](Psi0),
    sigma^2[H](Psi0') = Var_Psi0'(V), and <V>_Psi0' <= E0' - E0 <= <V>_Psi0."""
    act, ref = h4(), h4_reference()
    dv = h4_ls_error(8)
    row, ci_m = model_floors(ref, act.eri + dv, return_state=True)
    vspace = _operator_space(act, dv)
    ham_v = Hamiltonian(vspace)
    noise, vnoise = row["res_noise"], row["var_noise"]

    def var_and_mean(ci):
        hv = ham_v.apply(ci)
        mean = inner(ci, hv).real
        return inner(hv, hv).real - mean ** 2, mean

    var0, v0 = var_and_mean(ref.ci)
    var1, v1 = var_and_mean(ci_m)
    # (H' - <H'>) Psi0 = (V - <V>) Psi0 + r0 with ||r0||^2 = sigma^2[H](Psi0) <= var_noise.
    # <H^2> - <H>^2 cannot resolve ||r0||^2 below round-off, hence the 1e-13 floor.
    vn = max(vnoise, 1e-13)
    assert abs(row["var_model_at_exact"] - var0) <= 2 * np.sqrt(var0 * vn) + vn
    assert abs(row["var_exact_at_model"] - var1) <= 2 * np.sqrt(var1 * vn) + vn
    assert row["var_model_at_exact"] > 1e-5  # a real floor, not noise
    # residuals: linear in the Hamiltonian, A[H](Psi0) is the solver noise
    model = act.with_eri(act.eri + dv)
    a_m0 = compute_residual(model, ref.ci).A
    a_v0 = compute_residual(vspace, ref.ci).A
    assert np.allclose(a_m0, ref.residual.A + a_v0, atol=1e-12)
    assert abs(row["res_model_at_exact"] - np.linalg.norm(a_v0)) <= noise + 1e-12
    a_v1 = compute_residual(vspace, ci_m).A
    assert abs(row["res_exact_at_model"] - np.linalg.norm(a_v1)) <= noise + 1e-12
    # tensor level: A[H](Psi0') + A[V](Psi0') = A[H'](Psi0'), the solver noise of Psi0'
    a_0m = compute_residual(act, ci_m).A
    assert np.linalg.norm(a_0m + a_v1) <= noise + 1e-12
    assert np.linalg.norm(a_0m) > 1e3 * noise
    # energies: de_pt1 is <V>_Psi0; the sandwich holds
    assert np.isclose(row["de_pt1"], v0, atol=1e-12)
    assert v1 - 1e-10 <= row["de_fci"] <= v0 + 1e-10
    assert np.isclose(row["de_exact_at_model"], row["de_fci"] - v1, atol=1e-10)
    # infidelity and model energy from the returned state
    assert np.isclose(row["infidelity"], 1 - abs(inner(ref.ci, ci_m)) ** 2, atol=1e-14)
    assert np.isclose(row["e_fci_model"], model.fci()[0], atol=1e-9)
    assert abs(spin_square(ci_m, act.norb, act.nelec)) < 1e-8


def test_perturbative_orders_under_scaled_error():
    """V = t W with W an LS-THC error tensor: O(t) and O(t^2) quantities halve and
    quarter when t halves, and the two second-order energies agree to leading order."""
    act, ref = h4(), h4_reference()
    w = h4_ls_error(8)
    ts = (0.2, 0.1, 0.05)
    rows = [model_floors(ref, act.eri + t * w) for t in ts]

    def ratios(key):
        v = [r[key] for r in rows]
        return np.array([v[i] / v[i + 1] for i in range(len(v) - 1)])

    # Exactly linear / quadratic: these are evaluated in the fixed state Psi0. Only the
    # solver residual of Psi0 (~1e-7) perturbs them (observed deviations: 2e-12, 4e-8,
    # 3e-7). The O(t) quantities below deviate by > 2e-3 at these t.
    assert np.allclose(ratios("de_pt1"), 2.0, rtol=1e-9)
    assert np.allclose(ratios("res_model_at_exact"), 2.0, rtol=1e-4)
    assert np.allclose(ratios("var_model_at_exact"), 4.0, rtol=1e-3)
    # first order with O(t) corrections (observed 2.02, 2.01)
    for key in ("de_fci", "res_exact_at_model"):
        assert np.all(np.abs(ratios(key) - 2.0) < 0.05), key
    # second order with O(t) corrections (observed 3.97-3.99)
    for key in ("de_exact_at_model", "de_model_at_exact", "var_exact_at_model", "infidelity"):
        assert np.all(np.abs(ratios(key) - 4.0) < 0.1), key
    # leading-order equality de_exact_at_model = de_model_at_exact = -E^(2); the
    # relative difference is O(t), so it halves with t (observed 0.0053, 0.0027, 0.0013)
    dev = np.array([abs(1 - r["de_exact_at_model"] / r["de_model_at_exact"]) for r in rows])
    assert dev[-1] < 0.01
    assert np.all((dev[1:] / dev[:-1] > 0.4) & (dev[1:] / dev[:-1] < 0.6))
    # de_fci - de_pt1 = -de_model_at_exact is second order
    assert all(np.isclose(r["de_fci"] - r["de_pt1"], -r["de_model_at_exact"], atol=1e-12)
               for r in rows)


def test_second_order_sums_match_exact_spectrum():
    """Leading-order formulas of the module docstring against the exact spectrum of H4:
    with w_k = <k|W|Psi0>, de_exact_at_model / t^2 and de_model_at_exact / t^2 tend to
    sum |w_k|^2 / (E_k - E0), infidelity / t^2 to sum |w_k|^2 / (E_k - E0)^2, and
    var_model_at_exact / t^2 = sum |w_k|^2 for every t."""
    act, ref = h4(), h4_reference()
    w = h4_ls_error(8)
    ham = ref.ham
    shape = ham.space.shape
    dim = shape[0] * shape[1]
    hmat = np.array([ham.apply(np.eye(dim)[i].reshape(shape)).ravel() for i in range(dim)]).T
    assert np.allclose(hmat, hmat.T, atol=1e-12)
    e, u = np.linalg.eigh(hmat)
    k0 = int(np.argmax(np.abs(u.T @ ref.ci.ravel())))
    assert k0 == 0 and np.isclose(e[0] + act.ecore, ref.energy, atol=1e-12)
    wk = u.T @ Hamiltonian(_operator_space(act, w)).apply(ref.ci).ravel()
    exc = np.arange(dim) != k0
    gaps = e[exc] - e[k0]
    assert gaps.min() > 0.1  # non-degenerate ground state (observed gap 0.258 Ha)
    s_energy = np.sum(wk[exc] ** 2 / gaps)
    s_fidelity = np.sum(wk[exc] ** 2 / gaps ** 2)
    s_var = np.sum(wk[exc] ** 2)
    devs = []
    for t in (0.05, 0.02):
        r = model_floors(ref, act.eri + t * w)
        dev = np.array([r["de_exact_at_model"] / (t * t * s_energy),
                        r["de_model_at_exact"] / (t * t * s_energy),
                        r["infidelity"] / (t * t * s_fidelity)]) - 1.0
        devs.append(np.abs(dev))
        assert np.isclose(r["var_model_at_exact"], t * t * s_var, rtol=1e-4)
    # observed relative deviations: 2.7e-3, 1.4e-3, 2.9e-3 at t = 0.05 and 2.5x smaller at 0.02
    assert np.all(devs[1] < 0.01)
    assert np.all(devs[1] < devs[0])


# ------------------------------------------------------------------ spin guard
def _high_spin_toy(norb=3):
    """Degenerate orbitals with strong exchange: the triplet lies 0.8 Ha below the
    lowest singlet, more than PySCF's default spin penalty (0.1 S^2) can lift."""
    U, V, K, t = 1.0, 0.2, 0.5, 0.02
    h1 = -t * (np.ones((norb, norb)) - np.eye(norb))
    eri = np.zeros((norb,) * 4)
    for p in range(norb):
        eri[p, p, p, p] = U
        for q in range(norb):
            if p != q:
                eri[p, p, q, q] = V
                eri[p, q, p, q] = eri[p, q, q, p] = K
    act = ActiveSpace(h1=h1, eri=eri, ecore=0.0, nelec=(1, 1), label="high-spin toy")
    act.check_symmetry()
    return act


def test_lowest_state_recovers_singlet_when_penalty_is_too_weak():
    act = _high_spin_toy()
    ham = Hamiltonian(act)
    shape = ham.space.shape
    dim = shape[0] * shape[1]
    hmat = np.array([ham.apply(np.eye(dim)[i].reshape(shape)).ravel() for i in range(dim)]).T
    w, v = np.linalg.eigh(hmat)
    s2 = np.array([spin_square(v[:, k].reshape(shape), act.norb, act.nelec) for k in range(dim)])
    e_singlet = w[np.abs(s2) < 1e-6].min()
    e_triplet = w[np.abs(s2 - 2) < 1e-6].min()
    assert e_triplet < e_singlet - 0.5
    # PySCF's default penalty (0.1 S^2) alone returns the triplet, so the core
    # solver refuses it when restricted to that shift ...
    with pytest.raises(RuntimeError, match="S\\^2"):
        act.fci(shifts=(0.1,))
    # ... and by default retries with larger shifts and returns the singlet,
    # with the energy of H itself (no penalty)
    e_core, ci_core = act.fci()
    assert abs(spin_square(ci_core, act.norb, act.nelec)) < 1e-8
    assert np.isclose(e_core, e_singlet, atol=1e-9)
    # without a spin target the true ground state (the triplet) is returned
    e_free, ci_free = act.fci(spin_square=None)
    assert np.isclose(e_free, e_triplet, atol=1e-9)
    assert np.isclose(spin_square(ci_free, act.norb, act.nelec), 2.0, atol=1e-6)
    # lowest_state agrees (its own retries are now a second line of defence)
    st = lowest_state(act)
    assert st.spin_ok and abs(st.s2) < 1e-8
    assert np.isclose(st.energy, e_singlet, atol=1e-9)
    # an explicit target selects the triplet instead
    st3 = lowest_state(act, s2_target=2.0)
    assert st3.spin_ok and np.isclose(st3.energy, e_triplet, atol=1e-9)


def test_scan_without_molecule_falls_back_to_random_guess():
    act = random_active_space(4, (2, 2), seed=3)
    assert act.mol is None
    rows = thc_floor_scan(act, [10], include_df=False, ft_cost=False, refine=False)
    (row,) = rows
    assert row["details"]["guess"] == "random"
    assert row["eri_max_abs_err"] < 1e-8 and abs(row["de_fci"]) < 1e-8
    assert row["res_model_at_exact"] <= row["res_noise"] + 1e-7
    assert "ft_toffoli_per_step" not in row
    with pytest.raises(ValueError):
        thc_floor_scan(act, [4], guess="bogus")


# ------------------------------------------------------------------ FT costing
def test_ft_cost_reproduces_openfermion_reference_values():
    """Both cases of openfermion/resource_estimates/thc/compute_cost_thc_test.py."""
    pytest.importorskip("openfermion")
    reiher = thc_ft_cost(54, 306.3, 350, dE=0.001, chi=10, beta=16)
    if reiher is None:
        pytest.skip("openfermion.resource_estimates.thc.compute_cost is not importable")
    assert (reiher["ft_toffoli_per_step"], reiher["ft_toffoli_total"],
            reiher["ft_logical_qubits"]) == (10912, 5250145120, 2142)
    li = thc_ft_cost(76, 1201.5, 450, dE=0.001, chi=10, beta=20)
    assert (li["ft_toffoli_per_step"], li["ft_toffoli_total"],
            li["ft_logical_qubits"]) == (16923, 31938980976, 2196)
    assert FT_DEFAULTS == {"dE": 1.6e-3, "chi": 10, "beta": 16}


def test_ft_cost_outside_model_range_does_not_exit():
    """M = 4 on 4 orbitals: d = 14 < m = 18, where OpenFermion's QR() calls sys.exit."""
    pytest.importorskip("openfermion")
    out = thc_ft_cost(4, 5.0, 4)
    if out is None:
        pytest.skip("openfermion.resource_estimates.thc.compute_cost is not importable")
    assert out["ft_toffoli_per_step"] is None and "d = 14 < output width m = 18" in out["ft_note"]
    # this is the case in which OpenFermion itself exits
    from openfermion.resource_estimates.thc import compute_cost
    with pytest.raises(SystemExit):
        compute_cost(8, 5.0, 1.6e-3, 10, 16, 4, 20000)
    ok = thc_ft_cost(4, 5.0, 5)  # d = 19 >= m = 18
    assert ok["ft_note"] is None and ok["ft_toffoli_per_step"] > 0 and ok["ft_logical_qubits"] > 8


def test_scan_rows_carry_ft_cost():
    pytest.importorskip("openfermion")
    rows = h4_scan()
    thc = [r for r in rows if r["kind"] == "thc"]
    if "ft_toffoli_per_step" not in thc[0]:
        pytest.skip("openfermion.resource_estimates.thc.compute_cost is not importable")
    for r in thc:
        ref = thc_ft_cost(r["norb"], r["lambda"], r["M"])
        assert r["ft_toffoli_per_step"] == ref["ft_toffoli_per_step"] > 0
        assert (r["ft_dE"], r["ft_chi"], r["ft_beta"]) == (1.6e-3, 10, 16)
        # phase estimation total = step cost * ceil(pi lambda / (2 dE))
        iters = int(np.ceil(np.pi * r["lambda"] / (2 * r["ft_dE"])))
        assert r["ft_toffoli_total"] == r["ft_toffoli_per_step"] * iters
    assert all("ft_toffoli_per_step" not in r for r in rows if r["kind"] == "df")


# ----------------------------------------------------------- ranks and output
def test_auto_ranks():
    assert auto_ranks(4) == [4, 6, 8, 10]
    assert auto_ranks(6) == [6, 9, 12, 18, 21]
    assert auto_ranks(8) == [8, 12, 16, 24, 32, 36]
    assert auto_ranks(5) == [5, 8, 10, 15]
    assert auto_ranks(1) == [1] and auto_ranks(2) == [2, 3]
    assert auto_ranks(6, max_rank=10) == [6, 9, 10]


def test_format_markdown():
    rows = h4_scan()
    md = format_markdown(rows)
    table = [ln for ln in md.splitlines() if ln.startswith("|")]
    assert len(table) == len(rows) + 2
    assert len({ln.replace("\\|", "").count("|") for ln in table}) == 1  # same column count
    assert "THC M=6" in md and "E₀ = " in md and "solver noise" in md
    # converged rows carry no mark; the legend states that the THC fits are unregularized
    assert NOT_CONVERGED_MARK not in md
    assert "refined without a λ penalty, unlike OpenFermion's default pipeline" in md
    # FT columns are dropped when no row has them; None prints as a dash
    slim = [{k: v for k, v in r.items() if not k.startswith("ft_")} for r in rows]
    md2 = format_markdown(slim)
    assert "Tof/step" not in md2 and "| – |" in md2
    custom = format_markdown(rows, columns=[("model", "label", "s"), ("λ", "lambda", ".3f")])
    assert custom.splitlines()[0] == "| model | λ |"


def test_format_markdown_marks_unconverged_refinement():
    """A THC row whose L-BFGS-B run did not converge gets a dagger and a legend line
    with nit and the message; LS-only and penalized rows are described in the legend."""
    rows = [dict(r, details=dict(r["details"])) for r in h4_scan()]
    m6 = next(r for r in rows if r["label"] == "THC M=6")
    m6["details"].update(refine_success=False, refine_nit=5000,
                         refine_message="STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT")
    md = format_markdown(rows)
    table = [ln for ln in md.splitlines() if ln.startswith("|")]
    assert any(ln.startswith(f"| THC M=6 {NOT_CONVERGED_MARK} |") for ln in table)
    assert not any(NOT_CONVERGED_MARK in ln for ln in table if not ln.startswith("| THC M=6"))
    legend = [ln for ln in md.splitlines() if ln.startswith(f"- {NOT_CONVERGED_MARK}")]
    assert len(legend) == 1
    assert "THC M=6 (nit = 5000, STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT)" in legend[0]
    m6["details"].update(penalty_rule="auto", penalty=0.05)
    ls_row = next(r for r in rows if r["label"] == "THC M=10")
    ls_row["details"].update(refine=False, refine_success=None)
    md = format_markdown(rows)
    assert "OpenFermion's automatic λ penalty" in md and "least-squares only" in md
    assert "unlike OpenFermion's default pipeline" not in md


def test_scan_rejects_reserved_fit_kwargs_and_records_penalty():
    act = random_active_space(4, (2, 2), seed=3)
    for key, value in (("guess", "random"), ("refine", False), ("seed", 1), ("mol", None)):
        with pytest.raises(ValueError, match="fit_kwargs"):
            thc_floor_scan(act, [10], include_df=False, ft_cost=False, refine=False,
                           fit_kwargs={key: value})
    (row,) = thc_floor_scan(act, [6], include_df=False, ft_cost=False,
                            fit_kwargs={"penalty": "auto", "maxiter": 50})
    d = row["details"]
    assert d["penalty_rule"] == "auto" and d["penalty"] > 0
    assert d["refine_nit"] <= 50 and isinstance(d["refine_f"], float)
    json.loads(json.dumps(row))


def test_benchmark_cli_smoke(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)  # keep benchmarks/ free of caches
    spec = importlib.util.spec_from_file_location("run_thc_floors",
                                                  ROOT / "benchmarks" / "run_thc_floors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.parse_ranks("auto") is None and mod.parse_ranks("4, 8") == [4, 8]
    assert mod.parse_penalty("auto") == "auto" and mod.parse_penalty("0.5") == 0.5
    for bad in ("-1", "x"):
        with pytest.raises(argparse.ArgumentTypeError):
            mod.parse_penalty(bad)
    code = mod.main(["--systems", "h4", "--ranks", "8,10", "--no-refine", "--out", str(tmp_path),
                     "--quiet", "--penalty", "auto"])
    assert code == 0
    data = json.loads((tmp_path / "thc_floors_h4.json").read_text(encoding="utf-8"))
    assert data["system"] == "h4" and data["norb"] == 4 and data["settings"]["refine"] is False
    assert data["settings"]["fit_kwargs"] == {"penalty": "auto"}
    assert data["environment"]["integral_threads"] == 1
    # build_system: single-threaded build, thread count restored, same integrals
    from pyscf import lib

    nthreads = lib.num_threads()
    act = mod.build_system("h4")
    assert lib.num_threads() == nthreads
    assert act.nelec == h4().nelec and abs(act.ecore - h4().ecore) < 1e-10
    # orbital signs may differ between builds; |(pq|rs)| and diag(h1) do not
    assert np.abs(np.abs(act.eri) - np.abs(h4().eri)).max() < 1e-10
    assert np.abs(np.diag(act.h1) - np.diag(h4().h1)).max() < 1e-10
    thc = [r for r in data["rows"] if r["kind"] == "thc"]
    assert [r["M"] for r in thc] == [8, 10]
    assert all(r["details"]["penalty_rule"] == "auto" for r in thc)
    assert any(r["kind"] == "df" for r in data["rows"])
    out = capsys.readouterr().out
    assert "## h4" in out and "| THC M=8 |" in out and "least-squares only" in out

"""Tests of ft.properties: property targets, bias bounds, estimators, routes A and B, Hellmann-Feynman QPE (F46-F53).

References: ``residual.rdm12`` and dense tensor algebra for the targets; the JW Pauli sums of
``measure.pauli`` against the CI-space operators of ``fci_ops``; PySCF's ``dip_moment`` for the LiH
dipole; exact CQE records (``ft.trajectory.exact_trajectory`` and the R1 rule of
``steprules.run_step_rule``) for F47; M3's ``ft.residual`` routes (``sampling_direct``, ``ae_x_lcu``) and
hand-written closed forms for the estimator costs; brute-force minimizations over the field h for F51;
snapshot Monte Carlo of ``measure.schemes``; finite-field FCI; ``ft.circuits.replay_elementwise``; and
the reference numbers of the M4 cost model [build check] with their tolerances. Systems
(``ft.symmetry.benchmark_system``): M3's h4, lih, n2, n2s (= n2_r2.0) and h6s (= h6_r3.0), with the
selected SF encodings pinned by their ranks (F9 [build check]: h4 L = 7, lih L = 16, n2s L = 17).
"""

from __future__ import annotations

import copy
import dataclasses
import math

import numpy as np
import pytest

from thc_engine_cqe import steprules
from thc_engine_cqe.diagnostics.thc_floors import ExactReference, model_floors
from thc_engine_cqe.fci_ops import FCISpace, normalize
from thc_engine_cqe.ft import acceptance as acc
from thc_engine_cqe.ft import amplify, certify
from thc_engine_cqe.ft import properties as P
from thc_engine_cqe.ft import qpe as ftqpe
from thc_engine_cqe.ft import scaling, symmetry
from thc_engine_cqe.ft import steprules as ftsteprules
from thc_engine_cqe.ft.circuits import replay_elementwise
from thc_engine_cqe.ft.primitives import FTParameters
from thc_engine_cqe.ft.residual import PrepParams, ResidualProblem, ae_allocation, ae_x_lcu, sampling_direct
from thc_engine_cqe.ft.stateprep import prep_cost
from thc_engine_cqe.ft.trajectory import exact_trajectory
from thc_engine_cqe.measure import schemes
from thc_engine_cqe.measure.pauli import QubitMapping, QubitState
from thc_engine_cqe.residual import rdm12

DE = 1e-3
_RANK = {"h4": 7, "lih": 16, "n2s": 17}
_CACHE: dict = {}


def _system(name: str) -> dict:
    """active, irreps, psi_0 (lowest singlet), its RDMs and the six targets (the dipole on all-active builds)."""
    if name not in _CACHE:
        a = symmetry.benchmark_system(name)
        ir = symmetry.orbital_irreps(a)
        roots = a.fci(nroots=12, spin_square=0.0)
        psi = normalize(np.asarray(roots[1][0]))
        dm1, dm2 = rdm12(FCISpace.of(a), psi)
        t = {"rdm1": P.rdm1_target(a, ir), "rdm2": P.rdm2_target(a, ir), "occupations": P.occupation_target(a, dm1),
             "cumulant": P.cumulant_target(a, ir, dm1), "cumulant_norm": P.cumulant_norm_target(a, ir, dm1, dm2)}
        if a.n_electrons == a.mol.nelectron and a.mo_coeff.shape[1] == a.mol.nao_nr():
            t["dipole"] = P.dipole_target(a)
        _CACHE[name] = {"active": a, "irreps": ir, "roots": roots, "psi": psi, "dm1": dm1, "dm2": dm2, "targets": t}
    return _CACHE[name]


def _record(name: str, kind: str, n: int = 20):
    """Exact spin-orbital records with the classical first step: "ls" (exact line search) or "r1" (unit steps)."""
    key = ("record", name, kind, n)
    if key not in _CACHE:
        s = _system(name)
        a, ir = s["active"], s["irreps"]
        if kind == "ls":
            rec = exact_trajectory(a, optimizer="lbfgs", metric="spin_orbital", max_iter=n, system=name, irreps=ir,
                                   n_roots=12, first_step="second_order", roots=s["roots"])
        else:
            res = steprules.run_step_rule(a, steprules.StepRule("unit"), max_iter=n, metric="spin_orbital",
                                          irreps=ir, keep_states=True)
            rec = ftsteprules.to_trajectory_record(res, system=name, active=a, irreps=ir,
                                                   spectrum=scaling.spectrum(a, ir))
        _CACHE[key] = rec
    return _CACHE[key]


def _encoding(name: str) -> dict:
    """The pinned SF encoding of H', its QPE, walk offset, model spectrum, E1lb and CISD(H')."""
    key = ("enc", name)
    if key not in _CACHE:
        s = _system(name)
        a = s["active"]
        enc = ftqpe.sf_encoding(a, _RANK[name], reference=ExactReference.build(a))
        q = ftqpe.qpe_cost(enc, DE)
        model = enc.model(a)
        ms = acc.model_spectrum(a, q, s["irreps"])
        _CACHE[key] = {"enc": enc, "qpe": q, "c": acc.walk_offset(model), "model": model, "spectrum": ms,
                       "e1lb": certify.e1_lower_bound(ms.e1_singlet, model_spectrum=True),
                       "u": certify.truncated_ci_energy(model, 2)}
    return _CACHE[key]


def _inputs(name: str, kind: str) -> P.PropertyInputs:
    key = ("inputs", name, kind)
    if key not in _CACHE:
        s = _system(name)
        _CACHE[key] = P.property_inputs(s["active"], s["psi"], s["targets"][kind])
    return _CACHE[key]


def _synthetic_inputs(norm: str, k: int, seed: int = 3) -> P.PropertyInputs:
    """A hand-built target on N = 2 with random variances, 1-norms, term counts and spreads."""
    rng = np.random.default_rng(seed)
    if norm == "scalar":
        k = 1
        el = np.zeros((0,), dtype=np.int64)
    else:
        el = np.zeros((k, 2), dtype=np.int64)
    m = np.ones(1) if norm == "scalar" else rng.integers(1, 5, size=k).astype(float)
    f = np.eye(2)
    t = P.PropertyTarget("rdm1" if norm != "scalar" else "dipole", norm, el, m, (f,) * k, (None,) * k)
    return P.PropertyInputs(target=t, v_shadow=rng.uniform(0.5, 40.0, k), alpha=rng.uniform(0.2, 3.0, k),
                            n_terms=rng.integers(1, 40, size=k).astype(float), sigma0=rng.uniform(0.0, 1.0, k),
                            r0=rng.uniform(0.5, 4.0, k))


# ------------------------------------------------------------------ targets (F46)
@pytest.mark.parametrize("name", ["h4", "lih"])
def test_targets_reproduce_rdm12(name):
    s = _system(name)
    a, t = s["active"], s["targets"]
    rec = _record(name, "ls", 4)
    for ci in (s["psi"], rec.states[2], FCISpace.of(a).hf()):
        dm1, dm2 = rdm12(FCISpace.of(a), normalize(np.asarray(ci)))
        e1, e2 = t["rdm1"].elements, t["rdm2"].elements
        ref1, ref2 = dm1[e1[:, 0], e1[:, 1]], dm2[tuple(e2.T)]
        assert np.max(np.abs(t["rdm1"].expectations(a, ci) - ref1)) <= 1e-12
        assert np.max(np.abs(t["rdm2"].expectations(a, ci) - ref2)) <= 1e-12
        assert np.max(np.abs(P.property_values(t["rdm1"], a, ci) - ref1)) <= 1e-12
        assert np.max(np.abs(P.property_values(t["rdm2"], a, ci) - ref2)) <= 1e-12
        delta = dm2 - np.einsum("pq,rs->pqrs", dm1, dm1) + 0.5 * np.einsum("ps,rq->pqrs", dm1, dm1)
        assert np.max(np.abs(P.property_values(t["cumulant"], a, ci) - delta[tuple(e2.T)])) <= 1e-12
        assert abs(P.property_values(t["cumulant_norm"], a, ci)[0] - np.linalg.norm(delta)) <= 1e-12
        assert np.max(np.abs(P.property_values(t["occupations"], a, ci)
                             - np.sort(np.linalg.eigvalsh(dm1))[::-1])) <= 1e-12
    # the linearized occupations equal the NONs at the reference (the natural orbitals diagonalize gamma_0)
    assert np.max(np.abs(t["occupations"].expectations(a, s["psi"])
                         - P.property_values(t["occupations"], a, s["psi"]))) <= 1e-12


@pytest.mark.parametrize("name", ["h4", "lih"])
def test_two_constructions_agree(name):
    s = _system(name)
    a = s["active"]
    psi = _record(name, "ls", 4).states[2]                    # not an eigenstate
    mp = QubitMapping(a.norb)
    st = QubitState.from_ci(a.norb, a.nelec, psi, mp)
    for kind, t in s["targets"].items():
        jw = np.array([st.expectation(op, normalize=True).real for op in t.operators(mp)])
        assert np.max(np.abs(jw - t.expectations(a, psi))) <= 1e-12, kind


def test_rdm2_orbits():
    for name, n_masked in (("h4", 44), ("h6s", 189), ("lih", 141), ("n2", 69)):
        a = symmetry.benchmark_system(name)
        n = a.norb
        full = P.rdm2_target(a, np.zeros(n, dtype=np.int64))
        assert full.n_elements == (n ** 4 + 3 * n ** 2) // 4 == {4: 76, 6: 351}[n]
        assert full.m.sum() == n ** 4
        ir = symmetry.orbital_irreps(a)
        t = P.rdm2_target(a, ir)
        assert t.n_elements == n_masked, name
        # independent count: orbits as frozensets of the allowed index tuples; sum m = the allowed entries
        allowed = np.argwhere(symmetry.pair_mask(ir))
        orbits = {frozenset({(p, q, r, s), (r, s, p, q), (q, p, s, r), (s, r, q, p)}) for p, q, r, s in allowed}
        assert len(orbits) == n_masked and t.m.sum() == allowed.shape[0]
    # sum_e m_e x_e^2 is the Frobenius norm of the full tensor
    s = _system("lih")
    t = s["targets"]["rdm2"]
    x = t.expectations(s["active"], s["psi"])
    assert math.isclose(float(np.sum(t.m * x ** 2)), float(np.sum(s["dm2"] ** 2)), rel_tol=1e-12)
    t1 = s["targets"]["rdm1"]
    x1 = t1.expectations(s["active"], s["psi"])
    assert math.isclose(float(np.sum(t1.m * x1 ** 2)), float(np.sum(s["dm1"] ** 2)), rel_tol=1e-12)


@pytest.mark.parametrize("name", ["h4", "lih"])
def test_cumulant_vanishes_for_hf_and_trace_rules(name):
    s = _system(name)
    a, t = s["active"], s["targets"]
    hf = FCISpace.of(a).hf()
    assert np.max(np.abs(P.property_values(t["cumulant"], a, hf))) <= 1e-14
    assert P.property_values(t["cumulant_norm"], a, hf)[0] <= 1e-14
    dm1, dm2 = rdm12(FCISpace.of(a), hf)
    with pytest.raises(ValueError):
        P.cumulant_norm_target(a, s["irreps"], dm1, dm2)
    ne = a.n_electrons
    for ci in (s["psi"], _record(name, "ls", 4).states[3]):
        g, d = rdm12(FCISpace.of(a), normalize(np.asarray(ci)))
        assert abs(np.trace(g) - ne) <= 1e-12
        assert abs(np.einsum("ppqq->", d) - ne * (ne - 1)) <= 1e-11
    # the linearized operators are first-order exact along a symmetry-conserving path from psi_0 (chi in the
    # ground irrep; LiH's degenerate pi natural orbitals belong to different irreps, which a symmetry-breaking
    # chi would mix at first order): the error falls as h^2
    rng = np.random.default_rng(5)
    det_ir = symmetry.determinant_irreps(a.norb, a.nelec, s["irreps"])
    chi = np.where(det_ir == det_ir.ravel()[0], rng.standard_normal(s["psi"].shape), 0.0)
    chi -= np.vdot(s["psi"], chi) * s["psi"]
    chi /= np.linalg.norm(chi)
    lin_err = {}
    for kind in ("occupations", "cumulant", "cumulant_norm"):
        tg = t[kind]
        x0, y0 = tg.expectations(a, s["psi"]), P.property_values(tg, a, s["psi"])
        errs = []
        for h in (1e-3, 5e-4):
            ci = normalize(s["psi"] + h * chi)
            d_lin = tg.expectations(a, ci) - x0
            d_true = P.property_values(tg, a, ci) - y0
            errs.append(float(np.max(np.abs(d_lin - d_true))))
            assert np.max(np.abs(d_true)) > 20.0 * errs[-1]
        lin_err[kind] = errs[0] / errs[1]
    assert all(3.5 < v < 4.5 for v in lin_err.values()), lin_err


def test_dipole_matches_pyscf():
    from pyscf import scf
    from pyscf.fci import direct_spin1

    s = _system("lih")
    a, t = s["active"], s["targets"]["dipole"]
    hf = FCISpace.of(a).hf()
    mu_fci = P.property_values(t, a, s["psi"])[0]
    mu_hf = P.property_values(t, a, hf)[0]
    assert abs(mu_fci - (-1.8177157)) <= 1e-6 and abs(mu_hf - (-1.9110889)) <= 1e-6      # [build check]
    c = a.mo_coeff
    dm_hf = 2.0 * c[:, :a.nelec[0]] @ c[:, :a.nelec[0]].T
    mf = scf.RHF(a.mol)
    assert abs(mf.dip_moment(a.mol, dm_hf, unit="AU", verbose=0)[2] - mu_hf) <= 1e-10
    dm_mo = direct_spin1.make_rdm1(np.asarray(s["psi"]), a.norb, a.nelec)
    assert abs(mf.dip_moment(a.mol, c @ dm_mo @ c.T, unit="AU", verbose=0)[2] - mu_fci) <= 1e-10
    with pytest.raises(ValueError):
        P.dipole_target(_system("n2s")["active"])                              # frozen core
    with pytest.raises(ValueError):
        P.dipole_target(a, axis=3)


def test_target_validation():
    f = np.eye(2)
    with pytest.raises(ValueError):
        P.PropertyTarget("spin", "fro", np.zeros((1, 2), int), np.ones(1), (f,), (None,))
    with pytest.raises(ValueError):
        P.PropertyTarget("rdm1", "l1", np.zeros((1, 2), int), np.ones(1), (f,), (None,))
    with pytest.raises(ValueError):
        P.PropertyTarget("rdm1", "fro", np.zeros((2, 2), int), np.ones(1), (f,), (None,))
    with pytest.raises(ValueError):
        P.PropertyTarget("rdm1", "fro", np.zeros((1, 2), int), np.zeros(1), (f,), (None,))
    with pytest.raises(ValueError):
        P.PropertyTarget("rdm1", "fro", np.zeros((1, 2), int), np.ones(1), (None,), (None,))
    with pytest.raises(ValueError):
        P.PropertyTarget("rdm1", "fro", np.full((1, 2), 2), np.ones(1), (f,), (None,))
    with pytest.raises(ValueError):
        P.PropertyTarget("dipole", "scalar", np.zeros((1, 2), int), np.ones(1), (f,), (None,))
    with pytest.raises(ValueError):
        P.PropertyTarget("rdm1", "fro", np.zeros((1, 2), int), np.ones(1), (np.eye(3),), (np.zeros((2,) * 4),))
    t = P.PropertyTarget("dipole", "scalar", (), np.ones(1), (f,), (None,), constant=1.5)
    assert t.elements.shape == (0,) and t.norb == 2 and t.n_elements == 1
    with pytest.raises(ValueError):
        t.expectations(_system("h4")["active"], _system("h4")["psi"])          # 2 orbitals against 4
    with pytest.raises(ValueError):
        P.rdm1_target(_system("h4")["active"], np.zeros(3, dtype=np.int64))


# ------------------------------------------------------------------ bias bound (F47)
def _f47_inputs(target: P.PropertyTarget, active, psi) -> P.PropertyInputs:
    """sigma_0 and r_O of a target at psi in a PropertyInputs (the estimator fields unused here)."""
    k = target.n_elements
    return P.PropertyInputs(target=target, v_shadow=np.zeros(k), alpha=np.zeros(k), n_terms=np.zeros(k),
                            sigma0=target.spreads(active, psi), r0=target.ranges(active, psi))


@pytest.mark.parametrize("name", ["h4", "lih"])
def test_bias_bound_on_records(name):
    """F47 against the property errors of the record iterates (the nonlinear functionals, as U11 compares them):
    a theorem for the linear targets and per element for the linearized operators; on these records it also
    holds for the nonlinear targets (where it is first order), and the cumulant norm's all-orders form lies
    above both the linearized bound and the error."""
    s = _system(name)
    a, psi = s["active"], s["psi"]
    t = s["targets"]
    f47 = {tk: _f47_inputs(tg, a, psi) for tk, tg in t.items()}
    parts = (f47["cumulant"], f47["rdm1"])
    worst = {tk: math.inf for tk in t}
    worst_all_orders = math.inf
    n_rows = 0
    for kind in ("ls", "r1"):
        rec = _record(name, kind, 20)
        gap = rec.e1 - rec.e0
        for tk, tg in t.items():
            sig, rng = f47[tk].sigma0, f47[tk].r0
            x0 = tg.expectations(a, psi)
            for k, ci in enumerate(rec.states):
                err = P.property_error(tg, a, ci, psi)
                if err <= 1e-9:
                    continue
                n_rows += 1
                q = 1.0 - float(rec.overlaps[k])
                de = float(rec.energies[k] - rec.e0)
                bq = P.bias_bound(tg, a, psi, q=q, sigma0=sig, ranges=rng)
                be = P.bias_bound(tg, a, psi, energy_error=de, gap=gap, sigma0=sig, ranges=rng)
                worst[tk] = min(worst[tk], bq / err, be / err)
                d = tg.expectations(a, ci) - x0
                per = 2.0 * sig * math.sqrt(q * (1.0 - q)) + q * rng          # per element (linearized)
                big = np.abs(d) > 1e-9
                assert np.all(per[big] >= np.abs(d[big])), (tk, k)
                if tk in ("rdm1", "rdm2", "dipole"):                         # linear: the error is the operators'
                    assert math.isclose(err, P._norm(tg, d), rel_tol=1e-9, abs_tol=1e-12)
                if tk == "cumulant_norm":
                    for b_lin, kw in ((bq, {"q": q}), (be, {"energy_error": de, "gap": gap})):
                        b_all = P.bias_bound(tg, a, psi, cumulant_parts=parts, **kw)
                        assert b_all >= b_lin and b_all >= err, (k, b_all, b_lin, err)
                        worst_all_orders = min(worst_all_orders, b_all / err)
    assert n_rows > 100
    assert all(v >= 1.0 for v in worst.values()), worst
    assert worst_all_orders >= 1.0
    # the default (computed) spreads and ranges give the same bound
    tr = t["rdm1"]
    assert math.isclose(P.bias_bound(tr, a, psi, q=0.01),
                        P.bias_bound(tr, a, psi, q=0.01, sigma0=tr.spreads(a, psi), ranges=tr.ranges(a, psi)))
    with pytest.raises(ValueError):
        P.bias_bound(tr, a, psi)
    with pytest.raises(ValueError):
        P.bias_bound(tr, a, psi, q=0.1, energy_error=1e-3, gap=0.1)
    with pytest.raises(ValueError):
        P.bias_bound(tr, a, psi, energy_error=1e-3)
    with pytest.raises(ValueError):
        P.bias_bound(tr, a, psi, q=0.1, cumulant_parts=parts)                    # only for the cumulant norm
    with pytest.raises(ValueError):
        P.bias_bound(t["cumulant_norm"], a, psi, q=0.1, cumulant_parts=parts[::-1])


@pytest.mark.parametrize("name", ["h4", "lih"])
def test_cumulant_norm_bound_to_all_orders(name):
    """The decomposition behind the all-orders bound: Delta_Psi - Delta_0 = L + Q with L the linearized cumulant
    elements (exactly) and ||Q||_F <= 3/2 ||d gamma||_F^2; the bound B_cum + 3/2 B_1^2 then covers states of
    psi_0's sector far from it, where the linearized F47 need not."""
    s = _system(name)
    a, psi, ir = s["active"], s["psi"], s["irreps"]
    t = s["targets"]
    space = FCISpace.of(a)
    rng = np.random.default_rng(11)
    det_ir = symmetry.determinant_irreps(a.norb, a.nelec, ir)
    parts = (_f47_inputs(t["cumulant"], a, psi), _f47_inputs(t["rdm1"], a, psi))
    full_c = P.cumulant_target(a, np.zeros(a.norb, dtype=np.int64), s["dm1"])    # every element, no mask
    g0, d0 = s["dm1"], s["dm2"]
    delta0 = P.cumulant(g0, d0)
    for q in (0.05, 0.3, 0.7):
        chi = np.where(det_ir == det_ir.ravel()[0], rng.standard_normal(psi.shape), 0.0)
        chi -= np.vdot(psi, chi) * psi
        chi /= np.linalg.norm(chi)
        ci = math.sqrt(1.0 - q) * psi + math.sqrt(q) * chi
        g, d = rdm12(space, ci)
        dg = g - g0
        lin = (d - d0) - np.einsum("pq,rs->pqrs", g0, dg) - np.einsum("pq,rs->pqrs", dg, g0) \
            + 0.5 * (np.einsum("ps,rq->pqrs", g0, dg) + np.einsum("ps,rq->pqrs", dg, g0))
        quad = -np.einsum("pq,rs->pqrs", dg, dg) + 0.5 * np.einsum("ps,rq->pqrs", dg, dg)
        assert np.max(np.abs(P.cumulant(g, d) - delta0 - lin - quad)) <= 1e-12
        x = full_c.expectations(a, ci) - full_c.expectations(a, psi)              # the linearized elements
        assert np.max(np.abs(x - lin[tuple(full_c.elements.T)])) <= 1e-12
        assert np.linalg.norm(quad) <= 1.5 * np.linalg.norm(dg) ** 2 * (1 + 1e-12)
        # the masked targets carry the whole tensors on the sector
        tc, t1 = t["cumulant"], t["rdm1"]
        d_c = tc.expectations(a, ci) - tc.expectations(a, psi)
        d_1 = P.property_values(t1, a, ci) - P.property_values(t1, a, psi)
        assert math.isclose(P._norm(tc, d_c), np.linalg.norm(lin), rel_tol=1e-10)
        assert math.isclose(P._norm(t1, d_1), np.linalg.norm(dg), rel_tol=1e-10)
        err = P.property_error(t["cumulant_norm"], a, ci, psi)
        bound = P.bias_bound(t["cumulant_norm"], a, psi, q=q, cumulant_parts=parts)
        assert err <= np.linalg.norm(lin) + np.linalg.norm(quad) <= bound
    # the bound is B_cum + 3/2 B_1^2 with F47's bounds of the two linear targets (both forms); PropertyTarget parts
    # (spreads and ranges computed) give the same value
    for kw in ({"q": 0.3}, {"energy_error": 0.02, "gap": 0.5}):
        b_c = P.bias_bound(t["cumulant"], a, psi, sigma0=parts[0].sigma0, ranges=parts[0].r0, **kw)
        b_1 = P.bias_bound(t["rdm1"], a, psi, sigma0=parts[1].sigma0, ranges=parts[1].r0, **kw)
        b_all = P.bias_bound(t["cumulant_norm"], a, psi, cumulant_parts=parts, **kw)
        assert math.isclose(b_all, b_c + 1.5 * b_1 ** 2, rel_tol=1e-12)
        assert math.isclose(b_all, P.bias_bound(t["cumulant_norm"], a, psi, cumulant_parts=(t["cumulant"], t["rdm1"]),
                                                **kw), rel_tol=1e-12)


def test_ranges_are_sector_norms():
    """r_O equals the largest |eigenvalue - <O>| of a dense sector matrix (h4 and lih, every target)."""
    for name in ("h4", "lih"):
        s = _system(name)
        a, psi = s["active"], s["psi"]
        space = FCISpace.of(a)
        for kind in ("rdm1", "occupations", "cumulant_norm") + (("dipole",) if "dipole" in s["targets"] else ()):
            t = s["targets"][kind]
            rng = t.ranges(a, psi)
            for e in range(min(t.n_elements, 4)):
                mat = np.column_stack([t._apply(space, e, col.reshape(space.shape)).ravel()
                                       for col in np.eye(space.size)])
                mean = float(psi.ravel() @ mat @ psi.ravel())
                w = np.linalg.eigvalsh(0.5 * (mat + mat.T))
                assert math.isclose(rng[e], max(abs(w[-1] - mean), abs(w[0] - mean)), rel_tol=1e-8), (name, kind, e)


# ------------------------------------------------------------------ estimators (F48)
def test_shadow_cost_and_crossover():
    prep = PrepParams(12345.0, 70)
    for norm in ("fro", "max", "scalar"):
        inp = _synthetic_inputs(norm, 9)
        t = inp.target
        eps, bias = 2e-3, 1.2e-3
        b2 = eps ** 2 - bias ** 2
        var = np.sum(t.m * inp.v_shadow) if norm == "fro" else np.max(inp.v_shadow)
        c = P.shadow_cost(inp, eps, bias, prep)
        assert math.isclose(c.n_prep, var / b2, rel_tol=1e-12)
        assert math.isclose(c.toffolis, var / b2 * 12345.0, rel_tol=1e-12)
        assert c.coherent_toffolis == 12345.0 and c.logical_qubits == 70 and c.n_runs == c.n_prep
        assert "infeasible" in P.shadow_cost(inp, eps, eps, prep).flags
        # crossover: the same preparations at B*, AE fewer below it, shadows fewer above it
        b_star = P.crossover_accuracy(inp)
        ae = P.ae_cost(inp, b_star, 0.0, prep)
        assert math.isclose(P.shadow_cost(inp, b_star, 0.0, prep).n_prep, ae.n_prep, rel_tol=1e-12)
        assert P.ae_cost(inp, 0.5 * b_star, 0.0, prep).n_prep < P.shadow_cost(inp, 0.5 * b_star, 0.0, prep).n_prep
        assert P.ae_cost(inp, 2.0 * b_star, 0.0, prep).n_prep > P.shadow_cost(inp, 2.0 * b_star, 0.0, prep).n_prep
    # against M3's direct route with the same variances and budget B = r ||A||_m
    inp = _inputs("h4", "rdm2")
    t = inp.target
    a_vec = np.linspace(0.1, 1.0, t.n_elements)
    norm_a = math.sqrt(float(np.sum(t.m * a_vec ** 2)))
    prob = ResidualProblem(kind="spin_free", elements=t.elements, m=t.m, A=a_vec, v_direct=inp.v_shadow,
                           w_aux=np.ones(t.n_elements), lam_q=np.ones(t.n_elements), alpha_x=inp.alpha,
                           n_terms_x=inp.n_terms, aux_bias=None, norb=4)
    m3 = sampling_direct(prob, prep, params=FTParameters(r=0.01))
    mine = P.shadow_cost(inp, 0.01 * norm_a, 0.0, prep)
    assert math.isclose(m3.n_prep, mine.n_prep, rel_tol=1e-12) and math.isclose(m3.toffolis, mine.toffolis, rel_tol=1e-12)


def test_ae_cost_uses_f11():
    params = FTParameters()
    prep = PrepParams(5000.0, 66)
    # fro: equals M3's ae_x route on a residual problem with the same 1-norms, terms and budget
    inp = _inputs("h4", "rdm2")
    t = inp.target
    a_vec = np.linspace(0.2, 1.0, t.n_elements)
    norm_a = math.sqrt(float(np.sum(t.m * a_vec ** 2)))
    prob = ResidualProblem(kind="spin_free", elements=t.elements, m=t.m, A=a_vec, v_direct=inp.v_shadow,
                           w_aux=np.ones(t.n_elements), lam_q=np.ones(t.n_elements), alpha_x=inp.alpha,
                           n_terms_x=inp.n_terms, aux_bias=None, norb=4)
    for kappa in (1.0, 2.0):
        m3 = ae_x_lcu(prob, prep, kappa=kappa, params=dataclasses.replace(params, r=0.02))
        mine = P.ae_cost(inp, 0.02 * norm_a, 0.0, prep, kappa=kappa)
        for field in ("n_prep", "toffolis", "coherent_toffolis", "n_runs"):
            assert math.isclose(getattr(m3, field), getattr(mine, field), rel_tol=1e-12), field
        assert m3.logical_qubits == mine.logical_qubits
        assert np.allclose(m3.run_toffolis, mine.run_toffolis, rtol=1e-12)
    # closed forms: N_A = 2 pi kappa Gamma / B (fro), N_e = 2 pi kappa alpha_e / B (max, scalar)
    for norm in ("fro", "max", "scalar"):
        inp = _synthetic_inputs(norm, 7)
        t = inp.target
        eps, bias = 3e-3, 1e-3
        b = math.sqrt(eps ** 2 - bias ** 2)
        c = P.ae_cost(inp, eps, bias, prep)
        if norm == "fro":
            gamma = float(np.sum(np.cbrt(t.m) * np.cbrt(inp.alpha) ** 2)) ** 1.5
            assert math.isclose(c.n_prep, 2 * math.pi * gamma / b, rel_tol=1e-12)
            eps_e = ae_allocation(inp.alpha, t.m, b)[1]
            assert math.isclose(float(np.sum(t.m * eps_e ** 2)), b * b, rel_tol=1e-12)
        else:
            eps_e = np.full(t.n_elements, b)
            assert math.isclose(c.n_prep, 2 * math.pi * float(np.sum(inp.alpha)) / b, rel_tol=1e-12)
        n_e = 2 * math.pi * inp.alpha / eps_e
        q_a = 66 + np.ceil(np.log2(inp.n_terms)) + params.chi + 4 + 1
        run = n_e * (5000.0 + 2 * inp.n_terms + 0.5 * q_a)
        assert math.isclose(c.toffolis, float(run.sum()), rel_tol=1e-12)
        assert math.isclose(c.coherent_toffolis, float(run.max()), rel_tol=1e-12)
        assert c.logical_qubits == int(np.max(q_a + np.ceil(np.log2(n_e / 2))))
        assert "infeasible" in P.ae_cost(inp, eps, 2 * eps, prep).flags


def test_gradient_cost_k1_equals_ae():
    prep = PrepParams(800.0, 60)
    for name, kind in (("lih", "dipole"), ("h4", "cumulant_norm")):
        inp = _inputs(name, kind)
        ae = P.ae_cost(inp, 4e-3, 1e-3, prep, kappa=1.0)
        ge = P.gradient_cost(inp, 4e-3, 1e-3, prep, kappa_g=1.0)
        assert math.isclose(ge.n_prep, ae.n_prep, rel_tol=1e-12)
        assert math.isclose(ge.toffolis, ge.n_prep * (800.0 + 2 * 16 + 2), rel_tol=1e-12)
    for norm in ("fro", "max"):
        inp = _synthetic_inputs(norm, 11)
        t = inp.target
        b = math.sqrt(4e-6 - 1e-6)
        ge = P.gradient_cost(inp, 2e-3, 1e-3, prep, kappa_g=10.0, b_theta=8)
        scale = math.sqrt(float(np.sum(t.m * inp.alpha ** 2))) if norm == "fro" else float(np.max(inp.alpha))
        n = 2 * math.pi * 10.0 * math.sqrt(11) * scale / b
        assert math.isclose(ge.n_prep, n, rel_tol=1e-12)
        assert math.isclose(ge.toffolis, n * (800.0 + 11 * 18), rel_tol=1e-12)
        assert {"estimate", "assumption"} <= set(ge.flags)


def test_property_inputs_on_h4():
    inp = _inputs("h4", "rdm2")
    s = _system("h4")
    t = inp.target
    assert inp.method == "exact"
    assert math.isclose(float(np.sum(t.m * inp.v_shadow)), 1511.0, rel_tol=1e-3)       # [build check] 1510.9
    ops = t.operators()
    for e in (0, 7, 30):
        ident = (ops[e].xs == 0) & (ops[e].zs == 0)
        assert math.isclose(inp.alpha[e], float(np.sum(np.abs(ops[e].coeffs[~ident]))))
        assert inp.n_terms[e] == np.count_nonzero(~ident)
    assert np.allclose(inp.sigma0, t.spreads(s["active"], s["psi"]))
    assert inp.sigma_norm == pytest.approx(math.sqrt(float(np.sum(t.m * inp.sigma0 ** 2))))
    assert inp.range_norm == pytest.approx(math.sqrt(float(np.sum(t.m * inp.r0 ** 2))))
    with pytest.raises(ValueError):
        P.PropertyInputs(target=t, v_shadow=inp.v_shadow[:-1], alpha=inp.alpha, n_terms=inp.n_terms,
                         sigma0=inp.sigma0)
    with pytest.raises(ValueError):
        P.PropertyInputs(target=t, v_shadow=-inp.v_shadow, alpha=inp.alpha, n_terms=inp.n_terms, sigma0=inp.sigma0)


# ------------------------------------------------------------------ route B copies (F50)
def _n_ex_brute(a: float) -> int:
    """The smallest odd n with n asin(sqrt(a)) >= pi/2 (exact AA reaches certainty with n calls)."""
    phi = math.asin(math.sqrt(a))
    n = 1
    while n * phi < math.pi / 2 - 1e-12:
        n += 2
    return n


def test_copy_counts():
    for a, n in ((0.434, 3), (0.254, 3), (0.2125, 5), (0.112, 5), (0.078, 7)):       # [build check]
        assert P.exact_aa_calls(a) == n == _n_ex_brute(a)
    assert P.exact_aa_calls(0.25) == 3 and P.exact_aa_calls(0.2499) == 5
    edge = math.sin(math.pi / 10) ** 2
    assert P.exact_aa_calls(edge) == 5 and P.exact_aa_calls(edge * 0.999) == 7
    assert P.exact_aa_calls(1.0) == 1
    for a in np.geomspace(1e-4, 0.999, 60):
        assert P.exact_aa_calls(a) == _n_ex_brute(a)
    # ground copies on n2s's encoding: C_U = min(2/a, N_ex)(C_s + C_filt), the ratio estimator above a = 2/3
    e = _encoding("n2s")
    inp = _inputs("n2s", "occupations")
    for p, c_s in ((0.2539, 0.0), (0.3976, 7.7e3), (0.7, 2e4)):
        gc = P.ground_copy(e["enc"], e["c"], p=p, start_toffolis=c_s, start_qubits=40, u_upper=e["u"],
                           e1_lower=e["e1lb"], eps=1e-3, sigma0=inp.sigma_norm, r_o=inp.range_norm, norb=6)
        per = c_s + gc.filter.toffolis
        assert gc.aa_calls == P.exact_aa_calls(p)
        assert math.isclose(gc.unitary_toffolis, min(2.0 / p, gc.aa_calls) * per, rel_tol=1e-12)
        assert math.isclose(gc.copy_toffolis, min(1.0 / p, amplify.exponential_search_calls(p)) * per, rel_tol=1e-12)
        assert gc.coherent == ("ratio" if p > 2.0 / 3.0 else "exact_aa")
        assert (gc.a_estimate_toffolis > 0.0) == (gc.coherent == "exact_aa")
        assert gc.logical_qubits == e["enc"].ancilla_walk + 1 + 40 - 12
        eps_f = min(1e-2, 0.1 * 1e-3 * math.sqrt(p) / (2 * inp.sigma_norm * math.sqrt(1 - p)))
        assert math.isclose(gc.eps_f, eps_f, rel_tol=1e-12)
        w = eps_f ** 2 * (1 - p) / (p * (1 - eps_f) ** 2)
        assert math.isclose(gc.leak_bound, 2 * inp.sigma_norm * math.sqrt(w * (1 - w)) + w * inp.range_norm,
                            rel_tol=1e-12)
        # the AA leakage fills the leakage share: 2 sigma sqrt(w_AA) + w_AA r = 0.1 eps
        x = 0.5 * gc.aa_calls * gc.delta_a / math.sqrt(p * (1 - p))
        w_aa = math.sin(x) ** 2
        assert math.isclose(2 * inp.sigma_norm * math.sqrt(w_aa) + w_aa * inp.range_norm, 1e-4, rel_tol=1e-9)
        assert math.isclose(gc.exact_aa_estimate_toffolis, math.pi * math.sqrt(p * (1 - p)) / gc.delta_a * per,
                            rel_tol=1e-12)
        assert gc.exact_aa_leak_bound > gc.leak_bound
        assert "model_bias_bound" in gc.flags and gc.model_bias > 0.0
    # the E1lb-referenced edge, its termination and the exact model bias
    gc = P.ground_copy(e["enc"], e["c"], p=0.25, start_toffolis=0.0, start_qubits=12, u_upper=None,
                       e1_lower=e["e1lb"], eps=1e-2, sigma0=0.5, norb=6, e0_model=e["spectrum"].e0, model_bias=1e-4)
    assert gc.u_upper == e["e1lb"] - amplify.W_L and "termination_not_certified" in gc.flags
    assert gc.terminating and gc.model_bias == 1e-4 and "leak_first_order" in gc.flags
    gc = P.ground_copy(e["enc"], e["c"], p=0.25, start_toffolis=0.0, start_qubits=12, u_upper=None,
                       e1_lower=e["spectrum"].e0 + 0.05, eps=1e-2, sigma0=0.5, norb=6, e0_model=e["spectrum"].e0)
    assert not gc.terminating
    with pytest.raises(ValueError):
        P.ground_copy(e["enc"], e["c"], p=0.25, start_toffolis=0.0, start_qubits=12, u_upper=e["e1lb"] + 1e-3,
                      e1_lower=e["e1lb"], eps=1e-2, sigma0=0.5, norb=6)
    with pytest.raises(ValueError):
        P.ground_copy(e["enc"], e["c"], p=0.0, start_toffolis=0.0, start_qubits=12, u_upper=None,
                      e1_lower=e["e1lb"], eps=1e-2, sigma0=0.5, norb=6)
    with pytest.raises(ValueError):
        dataclasses.replace(gc, coherent="ratio", a_estimate_toffolis=1.0)
    with pytest.raises(ValueError):
        dataclasses.replace(gc, u_upper=gc.e1_lower)
    with pytest.raises(ValueError):
        dataclasses.replace(gc, eps_f=1.5)


def _n2s_route_b(kind: str = "occupations", n_steps: int = 8) -> dict:
    """The n2s inputs of route B: encoding, R1 record, preparation costs, the target's inputs, T_res and C_loop."""
    e = _encoding("n2s")
    rec = _record("n2s", "r1", n_steps)
    prep = prep_cost(rec.directions, rec.eps, irreps=rec.irreps)
    tres = {"meas_r1": np.r_[0.0, np.full(rec.n_steps, 2e6)]}
    c_loop = {"free": np.zeros(rec.n_steps + 1), "meas_r1": np.r_[0.0, np.cumsum(tres["meas_r1"][:rec.n_steps])]}
    return {"e": e, "rec": rec, "prep": prep, "inp": _inputs("n2s", kind), "tres": tres, "c_loop": c_loop,
            "e0m": e["spectrum"].e0}


def _copy_of(d: dict, r, u, e1_lower: float, model_bias: float = 2e-4, eps: float = 1e-2):
    """The ground copy of a route-B row's start and pass edge, recomputed."""
    rec, prep, inp, e = d["rec"], d["prep"], d["inp"], d["e"]
    return P.ground_copy(e["enc"], e["c"], p=float(rec.overlaps[r.k]), start_toffolis=float(prep.cumulative[r.k]),
                         start_qubits=int(prep.logical_qubits[r.k]), u_upper=u, e1_lower=e1_lower, eps=eps,
                         sigma0=inp.sigma_norm, r_o=inp.range_norm, norb=6, e0_model=d["e0m"], model_bias=model_bias)


def test_route_b_rows_add_the_estimate_once():
    """Every selectable row recomputed from its copy: shadows on the destructive copy min(1/a, E_BBHT)(C_s +
    C_filt), AE and GE on the coherent copy (ratio or exact AA with the estimate of a paid once), C_loop and the
    certificate cost of the pass edge in loop_toffolis, and the bias b_m + the copy's leak bound."""
    d = _n2s_route_b()
    e, rec, prep, inp = d["e"], d["rec"], d["prep"], d["inp"]
    rows = P.route_b_rows(rec, prep, d["tres"], inp, 1e-2, e["enc"], e["c"], u_classical=e["u"], e1_lower=e["e1lb"],
                          e0_model=d["e0m"], model_bias=2e-4)
    assert {r.route for r in rows} == {"B0", "Bk"}
    assert {r.loop for r in rows} == {"free", "meas_r1"}
    certs = certify.energy_certificates(rec, prep.cumulative, e1_lower=e["e1lb"], de=DE, walk=e["enc"],
                                        prep_qubits=prep.logical_qubits, e0_model=d["e0m"])
    checked = {"shadow": 0, "ae": 0, "ge": 0}
    for r in rows:
        if not r.selectable:
            continue
        edge = _flag_value(r, "edge:")
        if edge == "U_classical":
            u, cert_cost = e["u"], 0.0
        elif edge == "E1lb":
            u, cert_cost = None, 0.0
        else:
            match = [c for c in certs if c.j == r.k and f"U_{c.source}" == edge]
            assert len(match) == 1
            u, cert_cost = match[0].upper, match[0].toffolis
        gc = _copy_of(d, r, u, e["e1lb"])
        loop = d["c_loop"][r.loop][r.k] + cert_cost
        if r.estimator == "shadow":
            assert "destructive" in r.flags and r.per_prep == gc.copy_toffolis
            cost = P.shadow_cost(inp, 1e-2, 2e-4 + gc.leak_bound, PrepParams(gc.copy_toffolis, gc.logical_qubits))
            extra, leak = 0.0, gc.leak_bound
        else:
            if "coherent:exact_aa" in r.flags:
                c_u, extra, leak = gc.exact_aa_toffolis, gc.exact_aa_estimate_toffolis, gc.exact_aa_leak_bound
            else:
                assert "coherent:ratio" in r.flags
                c_u, extra, leak = gc.ratio_toffolis, 0.0, gc.leak_bound
            prep_u = PrepParams(c_u, gc.logical_qubits)
            cost = (P.ae_cost(inp, 1e-2, 2e-4 + leak, prep_u) if r.estimator == "ae" else
                    P.gradient_cost(inp, 1e-2, 2e-4 + leak, prep_u, kappa_g=float(r.estimator[2:])))
        checked[r.estimator[:2] if r.estimator.startswith("ge") else r.estimator] += 1
        assert math.isclose(r.loop_toffolis, loop + extra, rel_tol=1e-12)
        assert math.isclose(r.toffolis, loop + extra + cost.toffolis, rel_tol=1e-12)
        assert math.isclose(r.bias, 2e-4 + leak, rel_tol=1e-12)
    assert min(checked.values()) >= 2, checked
    # Bk is never dearer than B0 in the same loop and estimator
    for lp in ("free", "meas_r1"):
        for est in ("shadow", "ae"):
            b0 = [r for r in rows if r.route == "B0" and r.loop == lp and r.estimator == est][0]
            bk = [r for r in rows if r.route == "Bk" and r.loop == lp and r.estimator == est][0]
            assert bk.toffolis <= b0.toffolis


def test_route_b_rows_pass_edges_and_termination():
    """Pass edges: with E_0' > E1lb - W_L the E1lb-referenced copy does not terminate and is never kept; the
    iterates' AE certificates then carry every row (their Toffolis in loop_toffolis); without certificates no
    row is usable. Where E_0' <= E1lb - W_L the E1lb edge alone gives finite rows."""
    d = _n2s_route_b()
    e, rec, prep, inp, e0m = d["e"], d["rec"], d["prep"], d["inp"], d["e0m"]
    e1 = e0m + 0.05
    assert e0m > e1 - amplify.W_L
    certs = [certify.EnergyCertificate(source="ae", upper=e0m + 0.01 + 1e-3 * k, j=k, toffolis=1e5, shots=0.0,
                                       applications=1e3, label=f"AE E(Psi_{k})") for k in (0, 1, 3)]
    common = dict(u_classical=None, e1_lower=e1, e0_model=e0m, model_bias=2e-4)
    rows = P.route_b_rows(rec, prep, d["tres"], inp, 1e-2, e["enc"], e["c"], certificates=certs, **common)
    assert all("not_terminating" not in r.flags for r in rows)
    sel = [r for r in rows if r.selectable]
    assert {(r.route, r.loop) for r in sel if r.estimator == "ae"} == {(b, lp) for b in ("B0", "Bk")
                                                                    for lp in ("free", "meas_r1")}
    for r in sel:
        assert "edge:U_ae" in r.flags and r.k in (0, 1, 3) and (r.route == "Bk" or r.k <= 1)
        gc = _copy_of(d, r, e0m + 0.01 + 1e-3 * r.k, e1)
        est = gc.exact_aa_estimate_toffolis if "coherent:exact_aa" in r.flags else 0.0
        assert math.isclose(r.loop_toffolis, d["c_loop"][r.loop][r.k] + 1e5 + est, rel_tol=1e-12)
        assert "termination_not_certified" not in r.flags
    # no certificate: only the non-terminating E1lb edge is left, and no row is usable
    none = P.route_b_rows(rec, prep, d["tres"], inp, 1e-2, e["enc"], e["c"], certificates=[], **common)
    assert none and all(r.toffolis == math.inf and "not_usable" in r.flags and not r.selectable for r in none)
    # E_0' <= E1lb - W_L: the E1lb-referenced edge terminates and serves every row
    assert e0m <= e["e1lb"] - amplify.W_L
    rows_l = P.route_b_rows(rec, prep, d["tres"], inp, 1e-2, e["enc"], e["c"], u_classical=None,
                            e1_lower=e["e1lb"], e0_model=e0m, model_bias=2e-4, certificates=[])
    sel_l = [r for r in rows_l if r.selectable]
    assert len(sel_l) == len(rows_l) and all("edge:E1lb" in r.flags and "termination_not_certified" in r.flags
                                             for r in sel_l)
    for r in sel_l:
        gc = _copy_of(d, r, None, e["e1lb"])
        assert gc.u_upper == e["e1lb"] - amplify.W_L and gc.terminating


def test_route_b_rows_b0_start_and_exact_aa():
    """B0 takes the cheaper of HF and Psi_1 (Psi_1 on n2s), and an exact-AA row charges its larger leak bound
    (F47 at w_f + w_AA) in the bias and the estimate of a in loop_toffolis."""
    d = _n2s_route_b("rdm1")
    e, rec, prep, inp, e0m = d["e"], d["rec"], d["prep"], d["inp"], d["e0m"]
    rows = P.route_b_rows(rec, prep, d["tres"], inp, 1e-2, e["enc"], e["c"], u_classical=e["u"], e1_lower=e["e1lb"],
                          e0_model=e0m, model_bias=2e-4, certificates=[])
    # B0 free (AE): the cheapest of every start k <= 1, pass edge and coherent option, recomputed
    totals = {}
    for k in (0, 1):
        for label, u in (("U_classical", e["u"]), ("E1lb", None)):
            gc = P.ground_copy(e["enc"], e["c"], p=float(rec.overlaps[k]), start_toffolis=float(prep.cumulative[k]),
                               start_qubits=int(prep.logical_qubits[k]), u_upper=u, e1_lower=e["e1lb"], eps=1e-2,
                               sigma0=inp.sigma_norm, r_o=inp.range_norm, norb=6, e0_model=e0m, model_bias=2e-4)
            for mode, c_u, extra, leak in (("ratio", gc.ratio_toffolis, 0.0, gc.leak_bound),
                                           ("exact_aa", gc.exact_aa_toffolis, gc.exact_aa_estimate_toffolis,
                                            gc.exact_aa_leak_bound)):
                ae = P.ae_cost(inp, 1e-2, 2e-4 + leak, PrepParams(c_u, gc.logical_qubits))
                totals[(k, label, mode)] = extra + ae.toffolis
    best = min(totals, key=totals.get)
    b0 = [r for r in rows if r.route == "B0" and r.loop == "free" and r.estimator == "ae"][0]
    assert best[0] == 1 and totals[best] < min(v for key, v in totals.items() if key[0] == 0)
    assert b0.k == best[0] and f"edge:{best[1]}" in b0.flags and f"coherent:{best[2]}" in b0.flags
    assert math.isclose(b0.toffolis, totals[best], rel_tol=1e-12)
    # exact-AA rows: bias b_m + F47(w_f + w_AA) and the estimate of a once
    xs = [r for r in rows if r.estimator == "ae" and r.selectable and "coherent:exact_aa" in r.flags]
    assert xs
    for r in xs:
        edge = _flag_value(r, "edge:")
        gc = _copy_of(d, r, e["u"] if edge == "U_classical" else None, e["e1lb"])
        assert gc.exact_aa_leak_bound > gc.leak_bound
        assert math.isclose(r.bias, 2e-4 + gc.exact_aa_leak_bound, rel_tol=1e-12)
        assert math.isclose(r.loop_toffolis, d["c_loop"][r.loop][r.k] + gc.exact_aa_estimate_toffolis, rel_tol=1e-12)


def test_default_model_bias_is_f47_at_the_infidelity():
    """Without the exact model bias, ground copies and HF-QPE rows charge F47 with q = 1 - F (the infidelity)."""
    e = _encoding("lih")
    inp = _inputs("lih", "dipole")
    q = float(e["enc"].infidelity)
    sig, rng = inp.sigma_norm, inp.range_norm
    f47 = 2.0 * sig * math.sqrt(q * (1.0 - q)) + q * rng
    gc = P.ground_copy(e["enc"], e["c"], p=0.9, start_toffolis=0.0, start_qubits=12, u_upper=e["u"],
                       e1_lower=e["e1lb"], eps=4e-3, sigma0=sig, r_o=rng, norb=6, e0_model=e["spectrum"].e0)
    assert "model_bias_bound" in gc.flags and math.isclose(gc.model_bias, f47, rel_tol=1e-12)
    hs = _lih_hf_setup()
    rows = P.route_b_rows(hs["rec"], hs["prep"], {}, inp, 4e-3, e["enc"], e["c"], u_classical=e["u"],
                          e1_lower=e["e1lb"], e0_model=e["spectrum"].e0, hf_derivatives=hs["derivs"], qpe=e["qpe"],
                          certificates=[])
    hf = [r for r in rows if r.route == "HF-QPE"]
    assert len(hf) == 1 and "model_bias_bound" in hf[0].flags and hf[0].selectable
    ref = P.hellmann_feynman_cost(e["qpe"], 4e-3, d3=hs["derivs"]["d3"], d5=hs["derivs"]["d5"],
                                  stencil=_flag_value(hf[0], "stencil:"), fixed_bias=f47)
    assert math.isclose(hf[0].bias, f47 + ref["stencil_bias"], rel_tol=1e-12)
    # above eps the bound makes HF-QPE infeasible (the LiH dipole at 1e-3 a.u.: F47 = 3.3e-3 a.u.)
    rows1 = P.route_b_rows(hs["rec"], hs["prep"], {}, inp, 1e-3, e["enc"], e["c"], u_classical=e["u"],
                           e1_lower=e["e1lb"], e0_model=e["spectrum"].e0, hf_derivatives=hs["derivs"], qpe=e["qpe"],
                           certificates=[])
    hf1 = [r for r in rows1 if r.route == "HF-QPE"]
    assert f47 > 1e-3 and len(hf1) == 1 and "infeasible" in hf1[0].flags and not hf1[0].selectable


def test_hf_qpe_acceptance_on_the_long_qpe():
    """HF-QPE's rule B: n_v of F40 on the long QPE's grid 2 sigma_E sin(theta_0); B_L (no certified U) takes
    D_L = 5 and n_v = 2 under repetition, (C_s + L)/p + L per energy; the _L pre-projections only where
    E_0' <= E1lb - W_L."""
    e = _encoding("lih")
    inp = _inputs("lih", "dipole")
    hs = _lih_hf_setup()
    rec, prep, derivs, b_model = hs["rec"], hs["prep"], hs["derivs"], hs["b_model"]
    e0m = e["spectrum"].e0
    c_qpe, de = float(e["qpe"].toffolis), e["qpe"].de_qpe
    sin0 = math.sqrt(1.0 - ((e0m - e["c"]) / e["enc"].lam) ** 2)
    common = dict(e0_model=e0m, model_bias=b_model, hf_derivatives=derivs, qpe=e["qpe"], certificates=[])

    def hf_rows(**kw):
        rows = P.route_b_rows(rec, prep, {}, inp, 4e-3, e["enc"], e["c"], **common, **kw)
        out = [r for r in rows if r.route == "HF-QPE"]
        assert len(out) == 1
        return out[0]

    def sigma_e(r):
        return c_qpe * de / (r.longest_run - float(prep.cumulative[r.k]))

    # no certified U and E1lb 50 mHa above E_0': repetition only, B_L with n_v = 2
    r = hf_rows(u_classical=None, e1_lower=e0m + 0.05)
    assert _flag_value(r, "scheme:") == "rep" and int(_flag_value(r, "n_verify:")) == 2 == acc.verification_runs(5)
    assert "termination_not_certified" in r.flags
    length = r.longest_run - float(prep.cumulative[r.k])
    p, c_s = float(rec.overlaps[r.k]), float(prep.cumulative[r.k])
    assert math.isclose(r.per_prep, (c_s + length) / p + length, rel_tol=1e-12)
    assert math.isclose(r.toffolis, {"central": 2, "five_point": 4}[_flag_value(r, "stencil:")] * r.per_prep,
                        rel_tol=1e-12)
    # a certified U 20 mHa below E1lb: D_U is ~500 units of the long QPE's grid (n_v = 1), ~10 of a 1-mHa QPE's
    u = e["e1lb"] - 0.02
    r = hf_rows(u_classical=u, e1_lower=e["e1lb"])
    sig = sigma_e(r)
    n_v = int(_flag_value(r, "n_verify:"))
    assert n_v == acc.acceptance_bracket(u, e["e1lb"], sig, grid=2.0 * sig * sin0).n_runs == 1
    assert acc.acceptance_bracket(u, e["e1lb"], de, grid=2.0 * de * sin0).n_runs == 2
    # with the _L forms allowed (E_0' <= E1lb - W_L) and no U, the cheapest pre-projection is E1lb-referenced
    r = hf_rows(u_classical=None, e1_lower=e["e1lb"])
    assert _flag_value(r, "scheme:") in ("lt_bbht_L", "2s_L") and "termination_not_certified" in r.flags
    # near F40's threshold (n_v = 1 above D = 26.03 grid units, 2 below) n_v follows the row's own sigma_E, the
    # model bias included: U at D_U = 27.5 units of the central stencil's grid
    ref = P.hellmann_feynman_cost(e["qpe"], 4e-3, d3=derivs["d3"], d5=derivs["d5"], fixed_bias=b_model)
    sig_c = ref["sigma_e"]
    u = e["e1lb"] - sig_c - 27.5 * 2.0 * sig_c * sin0
    r = hf_rows(u_classical=u, e1_lower=e["e1lb"])
    assert _flag_value(r, "stencil:") == "central" and math.isclose(sigma_e(r), sig_c, rel_tol=1e-12)
    assert int(_flag_value(r, "n_verify:")) == acc.acceptance_bracket(u, e["e1lb"], sig_c,
                                                                      grid=2.0 * sig_c * sin0).n_runs == 1
    sig_0 = P.hellmann_feynman_cost(e["qpe"], 4e-3, d3=derivs["d3"])["sigma_e"]       # without the model bias
    assert acc.acceptance_bracket(u, e["e1lb"], sig_0, grid=2.0 * sig_0 * sin0).n_runs == 2


def test_bias_bound_scalar_and_max_forms():
    """F47's per-element form for scalar and max-norm targets: max_e (2 sigma_e s + r_e t)."""
    s = _system("lih")
    t = s["targets"]["dipole"]
    q = 0.3
    b = P.bias_bound(t, None, None, q=q, sigma0=np.array([1.144]), ranges=np.array([8.674]))
    assert math.isclose(b, 2 * 1.144 * math.sqrt(0.21) + 0.3 * 8.674, rel_tol=1e-12)
    assert math.isclose(b, 3.650694, rel_tol=1e-6)
    be = P.bias_bound(t, None, None, energy_error=2e-3, gap=0.13, sigma0=np.array([1.144]), ranges=np.array([8.674]))
    x = 2e-3 / 0.13
    assert math.isclose(be, 2 * 1.144 * math.sqrt(x) + 8.674 * x, rel_tol=1e-12)
    f = np.eye(2)
    tm = P.PropertyTarget("occupations", "max", np.array([[0, 0], [1, 1]]), np.ones(2), (f, f), (None, None))
    sig, rng = np.array([0.4, 0.1]), np.array([0.2, 3.0])
    for qq in (1e-3, 0.3):
        per = 2 * sig * math.sqrt(qq * (1 - qq)) + qq * rng
        assert math.isclose(P.bias_bound(tm, None, None, q=qq, sigma0=sig, ranges=rng), float(per.max()),
                            rel_tol=1e-12)
    assert np.argmax(2 * sig * math.sqrt(0.3 * 0.7) + 0.3 * rng) == 1          # the range term decides at q = 0.3


def _lih_model_bias() -> float:
    """b_fact of the LiH dipole on the pinned SF L = 16 encoding (exact psi_0' against psi_0)."""
    key = ("lih_model_bias",)
    if key not in _CACHE:
        s = _system("lih")
        a = s["active"]
        _, psi_m = model_floors(ExactReference.build(a), _encoding("lih")["enc"].eri_model, return_state=True)
        _CACHE[key] = P.property_error(s["targets"]["dipole"], a, psi_m, s["psi"])
    return _CACHE[key]


def _lih_hf_setup(n_steps: int = 8) -> dict:
    """The LiH dipole inputs of the HF-QPE rows: record, preparation costs, F51 derivatives (21 fields), b_fact."""
    key = ("lih_hf", n_steps)
    if key not in _CACHE:
        s = _system("lih")
        rec = _record("lih", "r1", n_steps)
        ff = P.finite_field_derivatives(s["active"], s["targets"]["dipole"], n_points=21)
        _CACHE[key] = {"rec": rec, "prep": prep_cost(rec.directions, rec.eps, irreps=rec.irreps),
                       "derivs": {"d3": ff["d3"], "d5": ff["d5"]}, "b_model": _lih_model_bias()}
    return _CACHE[key]


def _flag_value(row, prefix: str) -> str:
    return [f for f in row.flags if f.startswith(prefix)][0][len(prefix):]


def _hf_projections(e) -> dict:
    """The pre-projection costs of the HF-QPE rows with a certified U below E1lb and E_0' <= E1lb - W_L."""
    return {"c_filt": amplify.filter_cost(e["enc"], e["c"], e["u"], e["e1lb"], eps=1e-2).toffolis,
            "c_coarse": amplify.coarse_qpe_toffolis(e["enc"], e["c"], e["u"], e["e1lb"]),
            "c_filt_l": amplify.filter_cost(e["enc"], e["c"], e["e1lb"] - amplify.W_L, e["e1lb"], eps=1e-2).toffolis,
            "c_coarse_l": amplify.coarse_qpe_toffolis(e["enc"], e["c"], e["e1lb"] - amplify.W_L, e["e1lb"])}


def test_route_b_hellmann_feynman_rows_on_lih():
    """HF-QPE rows: one per loop, F51's cost at the chosen start with the model bias added linearly (bias^2 +
    variance <= eps^2 with the row's own bias); with the exact model bias they beat AE on copies."""
    e = _encoding("lih")
    inp = _inputs("lih", "dipole")
    hs = _lih_hf_setup()
    rec, prep, derivs, b_model = hs["rec"], hs["prep"], hs["derivs"], hs["b_model"]
    assert abs(b_model - 4.7e-4) <= 0.05e-4                                       # [build check] 4.71e-4
    tres = {"meas_r1": np.r_[0.0, np.full(rec.n_steps, 5e8)]}
    eps = 4e-3
    rows = P.route_b_rows(rec, prep, tres, inp, eps, e["enc"], e["c"], u_classical=e["u"], e1_lower=e["e1lb"],
                          e0_model=e["spectrum"].e0, model_bias=b_model, hf_derivatives=derivs, qpe=e["qpe"])
    hf_rows = [r for r in rows if r.route == "HF-QPE"]
    assert sorted(r.loop for r in hf_rows) == ["free", "meas_r1"]
    for r in hf_rows:
        assert r.estimator == "qpe" and r.selectable
        scheme, stencil = _flag_value(r, "scheme:"), _flag_value(r, "stencil:")
        n_v = int(_flag_value(r, "n_verify:"))
        proj = _hf_projections(e)
        c_s = float(prep.cumulative[r.k])
        ref_cost = P.hellmann_feynman_cost(e["qpe"], eps, d3=derivs["d3"], d5=derivs["d5"], stencil=stencil,
                                           p=float(rec.overlaps[r.k]), start_toffolis=c_s, projections=proj,
                                           n_verify=n_v, fixed_bias=b_model)
        assert ref_cost["scheme"] == scheme and n_v == 1                          # CISD certifies B_U
        assert math.isclose(r.toffolis - r.loop_toffolis, ref_cost["toffolis"], rel_tol=1e-12)
        # the row's bias is the model bias plus the stencil bias, and bias^2 + variance <= eps^2
        assert math.isclose(r.bias, b_model + ref_cost["stencil_bias"], rel_tol=1e-12)
        length = r.longest_run - c_s                                              # one long QPE, C_QPE dE/sigma_E
        sigma_e = float(e["qpe"].toffolis) * e["qpe"].de_qpe / length
        h = ref_cost["h"]
        sigma_d = sigma_e / (math.sqrt(2.0) * h) if stencil == "central" else math.sqrt(130.0) * sigma_e / (12.0 * h)
        assert r.bias ** 2 + sigma_d ** 2 <= eps ** 2 * (1.0 + 1e-12)
        assert r.bias ** 2 + sigma_d ** 2 >= eps ** 2 * (1.0 - 1e-9)             # the budget is used
        ae = [x for x in rows if x.route in ("B0", "Bk") and x.loop == r.loop and x.estimator == "ae"]
        assert all(r.toffolis < x.toffolis for x in ae)
        assert 95.0 < r.toffolis / e["qpe"].toffolis < 105.0                      # [build check] 99.2 C_QPE
        assert "stencil_bias_checked" not in r.flags
    # with a stencil-bias callable (here 3% above the leading order) every HF-QPE row takes the refined step
    def model(h, st):
        return 1.03 * (derivs["d3"] * h ** 2 / 6.0 if st == "central" else -derivs["d5"] * h ** 4 / 30.0)

    rows_s = P.route_b_rows(rec, prep, tres, inp, eps, e["enc"], e["c"], u_classical=e["u"], e1_lower=e["e1lb"],
                            e0_model=e["spectrum"].e0, model_bias=b_model, hf_derivatives=derivs, qpe=e["qpe"],
                            hf_stencil_bias=model)
    for r in [x for x in rows_s if x.route == "HF-QPE"]:
        assert "stencil_bias_checked" in r.flags
        plain = [x for x in hf_rows if x.loop == r.loop][0]
        stencil = _flag_value(r, "stencil:")
        ref = P.hellmann_feynman_cost(e["qpe"], eps, d3=derivs["d3"], d5=derivs["d5"], stencil=stencil,
                                      p=float(rec.overlaps[r.k]), start_toffolis=float(prep.cumulative[r.k]),
                                      projections=proj, n_verify=int(_flag_value(r, "n_verify:")),
                                      fixed_bias=b_model, stencil_bias=model)
        assert ref["h"] < ref["h_leading"] and abs(model(ref["h"], stencil)) <= ref["stencil_bias"]
        assert math.isclose(r.toffolis - r.loop_toffolis, ref["toffolis"], rel_tol=1e-12)
        assert r.toffolis > plain.toffolis and r.bias == plain.bias


def test_hf_qpe_rows_without_an_admissible_field_step():
    """HF-QPE rows when a stencil has no admissible field step: the other stencil serves them (flag
    "no_field_step:<stencil>"); with neither, they are infeasible and not usable, and routes B0 and Bk are unchanged."""
    e = _encoding("lih")
    inp = _inputs("lih", "dipole")
    hs = _lih_hf_setup()
    rec, prep, derivs, b_model = hs["rec"], hs["prep"], hs["derivs"], hs["b_model"]
    eps = 1e-3
    common = dict(u_classical=e["u"], e1_lower=e["e1lb"], e0_model=e["spectrum"].e0, model_bias=b_model,
                  hf_derivatives=derivs, qpe=e["qpe"], certificates=[])

    def central_only(h, st):                                     # the five-point stencil never meets b
        return 1.03 * derivs["d3"] * h ** 2 / 6.0 if st == "central" else 1.0

    plain = P.route_b_rows(rec, prep, {}, inp, eps, e["enc"], e["c"], **common)
    one = P.route_b_rows(rec, prep, {}, inp, eps, e["enc"], e["c"], hf_stencil_bias=central_only, **common)
    none = P.route_b_rows(rec, prep, {}, inp, eps, e["enc"], e["c"], hf_stencil_bias=lambda h, st: 1.0, **common)
    hf_plain = [r for r in plain if r.route == "HF-QPE"]
    assert len(hf_plain) == 1 and _flag_value(hf_plain[0], "stencil:") == "five_point"     # five-point wins at 1e-3
    hf = [r for r in one if r.route == "HF-QPE"]
    assert len(hf) == 1 and hf[0].selectable and _flag_value(hf[0], "stencil:") == "central"
    assert "no_field_step:five_point" in hf[0].flags and "no_field_step:central" not in hf[0].flags
    r = hf[0]
    ref = P.hellmann_feynman_cost(e["qpe"], eps, d3=derivs["d3"], d5=derivs["d5"], stencil="central",
                                  p=float(rec.overlaps[r.k]), start_toffolis=float(prep.cumulative[r.k]),
                                  projections=_hf_projections(e), n_verify=int(_flag_value(r, "n_verify:")),
                                  fixed_bias=b_model, stencil_bias=central_only)
    assert ref["feasible"] and math.isclose(r.toffolis, ref["toffolis"], rel_tol=1e-12)
    assert r.toffolis > hf_plain[0].toffolis
    hf0 = [r for r in none if r.route == "HF-QPE"]
    assert len(hf0) == 1 and not hf0[0].selectable and hf0[0].toffolis == math.inf
    assert {"infeasible", "not_usable", "no_field_step:central", "no_field_step:five_point"} <= set(hf0[0].flags)
    assert [repr(r) for r in none if r.route != "HF-QPE"] == [repr(r) for r in plain if r.route != "HF-QPE"]


# ------------------------------------------------------------------ Hellmann-Feynman QPE (F51)
def _hf_brute(eps, d3, d5, stencil, c_qpe, de, p, c_s, n_verify=1):
    """min over h of n_E ((C_s + C_QPE dE/sigma_E)/p + (n_v - 1) C_QPE dE/sigma_E) on a fine grid."""
    hs = np.geomspace(1e-5, 1.0, 400001)
    if stencil == "central":
        bias, w, n_e = abs(d3) * hs ** 2 / 6.0, 1.0 / (math.sqrt(2.0) * hs), 2
    else:
        bias, w, n_e = abs(d5) * hs ** 4 / 30.0, math.sqrt(130.0) / (12.0 * hs), 4
    ok = bias < eps
    sig = np.sqrt(eps ** 2 - bias[ok] ** 2) / w[ok]
    length = c_qpe * de / sig
    cost = n_e * ((c_s + length) / p + (n_verify - 1) * length)
    i = int(np.argmin(cost))
    return float(cost[i]), float(hs[ok][i])


def test_hellmann_feynman_optimum():
    s = _system("lih")
    e = _encoding("lih")
    q = e["qpe"]
    ff = P.finite_field_derivatives(s["active"], s["targets"]["dipole"], n_points=21)
    assert math.isclose(ff["d1"], P.property_values(s["targets"]["dipole"], s["active"], s["psi"])[0], abs_tol=1e-5)
    d3, d5 = ff["d3"], ff["d5"]
    assert 560.0 < d3 < 600.0                                                    # [build check] 581.6
    p_hf = float(np.vdot(s["psi"].ravel(), FCISpace.of(s["active"]).hf().ravel()) ** 2)
    eps = 4e-3
    r = P.hellmann_feynman_cost(q, eps, d3=d3, p=p_hf)
    h_closed = math.sqrt(6 * eps / (math.sqrt(3) * abs(d3)))
    assert math.isclose(r["h"], h_closed, rel_tol=1e-12) and abs(r["h"] - 4.88e-3) <= 0.02e-3   # [build check]
    assert math.isclose(r["bias"], eps / math.sqrt(3), rel_tol=1e-12)
    assert math.isclose(r["sigma_e"], 2 / math.sqrt(3) * h_closed * eps, rel_tol=1e-12)
    t_closed = math.sqrt(3) * q.toffolis * DE / (h_closed * eps * p_hf)
    assert math.isclose(r["toffolis"], t_closed, rel_tol=1e-12)
    assert abs(r["c_qpe_units"] - 91.0) <= 5.0                                    # [build check] 91.1
    brute, h_b = _hf_brute(eps, d3, d5, "central", q.toffolis, DE, p_hf, 0.0)
    assert math.isclose(brute, r["toffolis"], rel_tol=1e-6) and math.isclose(h_b, r["h"], rel_tol=1e-3)
    # with a start cost and verification runs: T = n_E ((C_s + L)/p + (n_v - 1) L) at the same h
    r2 = P.hellmann_feynman_cost(q, eps, d3=d3, p=0.6, start_toffolis=3e5, n_verify=2)
    brute2, _ = _hf_brute(eps, d3, d5, "central", q.toffolis, DE, 0.6, 3e5, n_verify=2)
    assert math.isclose(r2["toffolis"], brute2, rel_tol=1e-6)
    assert math.isclose(r2["toffolis"], 2 * ((3e5 + r2["qpe_toffolis"]) / 0.6 + r2["qpe_toffolis"]), rel_tol=1e-12)
    # five-point stencil
    r5 = P.hellmann_feynman_cost(q, 1e-3, d3=d3, d5=d5, p=p_hf, stencil="five_point")
    brute5, h5 = _hf_brute(1e-3, d3, d5, "five_point", q.toffolis, DE, p_hf, 0.0)
    assert math.isclose(r5["toffolis"], brute5, rel_tol=1e-6) and math.isclose(r5["h"], h5, rel_tol=1e-3)
    assert math.isclose(r5["bias"], 1e-3 / math.sqrt(5), rel_tol=1e-12)
    # pre-projection: the minimum over the schemes is <= repetition, and each scheme follows F51
    c_filt = amplify.filter_cost(e["enc"], e["c"], e["u"], e["e1lb"], eps=1e-2).toffolis
    c_coarse = amplify.coarse_qpe_toffolis(e["enc"], e["c"], e["u"], e["e1lb"])
    proj = {"c_filt": c_filt, "c_coarse": c_coarse}
    for p in (p_hf, 0.3, 0.05):
        rp = P.hellmann_feynman_cost(q, eps, d3=d3, p=p, start_toffolis=1e4, projections=proj)
        rr = P.hellmann_feynman_cost(q, eps, d3=d3, p=p, start_toffolis=1e4)
        assert rp["toffolis"] <= rr["toffolis"] and math.isclose(rp["h"], rr["h"])
        length = rp["qpe_toffolis"]
        assert math.isclose(rp["schemes"]["lt_bbht"],
                            amplify.exponential_search_calls(p) * (1e4 + c_filt) + length, rel_tol=1e-12)
        assert math.isclose(rp["schemes"]["2s"], (1e4 + c_coarse) / p + length, rel_tol=1e-12)
    assert P.hellmann_feynman_cost(q, eps, d3=d3, p=0.05, projections=proj)["scheme"] != "rep"
    with pytest.raises(ValueError):
        P.hellmann_feynman_cost(q, eps, d3=0.0)
    with pytest.raises(ValueError):
        P.hellmann_feynman_cost(q, eps, d3=d3, stencil="five_point")
    with pytest.raises(ValueError):
        P.hellmann_feynman_cost(q, eps, d3=d3, projections={"c_filter": 1.0})


def _hf_brute_fixed(eps, b_m, k, stencil, c_qpe, de, p, c_s):
    """min over h of n_E (C_s + C_QPE dE/sigma_E)/p with the bias b_m + |E^(n)| h^n/c_n added linearly."""
    hs = np.geomspace(1e-4, 0.1, 600001)
    if stencil == "central":
        bias, w, n_e = b_m + k * hs ** 2 / 6.0, 1.0 / (math.sqrt(2.0) * hs), 2
    else:
        bias, w, n_e = b_m + k * hs ** 4 / 30.0, math.sqrt(130.0) / (12.0 * hs), 4
    ok = bias < eps
    sig = np.sqrt(eps ** 2 - bias[ok] ** 2) / w[ok]
    cost = n_e * (c_s + c_qpe * de / sig) / p
    i = int(np.argmin(cost))
    return float(cost[i]), float(hs[ok][i])


def test_hellmann_feynman_fixed_bias():
    """A fixed (model) bias adds linearly to the stencil bias: the stencil bias maximizes h(b) sigma_d(b) under
    (b_m + b)^2 + sigma_d^2 = eps^2 (roots of the stationarity quadratics, brute force over h); b_m = 0 is F51."""
    q = _encoding("lih")["qpe"]
    c_qpe = float(q.toffolis)
    d3, d5, b_m = 581.571, 6.96513e6, 4.7146e-4                     # the LiH dipole (a.u.) [build check]
    for eps, stencil, n_e in ((4e-3, "central", 2), (1e-3, "five_point", 4)):
        r = P.hellmann_feynman_cost(q, eps, d3=d3, d5=d5, stencil=stencil, fixed_bias=b_m)
        # the stationarity condition of b^(1/2) (eps^2 - (b_m + b)^2)^(1/2) (central) or b^(1/4) (...) (five-point)
        coef = (3.0, 4.0 * b_m, b_m ** 2 - eps ** 2) if stencil == "central" else (5.0, 6.0 * b_m, b_m ** 2 - eps ** 2)
        b = max(np.roots(coef).real)
        h = math.sqrt(6.0 * b / d3) if stencil == "central" else (30.0 * b / d5) ** 0.25
        w = 1.0 / (math.sqrt(2.0) * h) if stencil == "central" else math.sqrt(130.0) / (12.0 * h)
        sigma_e = math.sqrt(eps ** 2 - (b_m + b) ** 2) / w
        assert math.isclose(r["stencil_bias"], b, rel_tol=1e-10) and math.isclose(r["h"], h, rel_tol=1e-10)
        assert math.isclose(r["bias"], b_m + b, rel_tol=1e-10) and r["fixed_bias"] == b_m
        assert math.isclose(r["sigma_e"], sigma_e, rel_tol=1e-10)
        assert math.isclose(r["qpe_toffolis"], c_qpe * q.de_qpe / sigma_e, rel_tol=1e-10)
        assert math.isclose(r["bias"] ** 2 + (r["sigma_e"] * w) ** 2, eps ** 2, rel_tol=1e-12)
        assert r["feasible"] and r["stencil_bias_checked"] is None and r["h"] == r["h_leading"]
        brute, h_b = _hf_brute_fixed(eps, b_m, d3 if stencil == "central" else d5, stencil, c_qpe, q.de_qpe, 0.9, 2e4)
        rr = P.hellmann_feynman_cost(q, eps, d3=d3, d5=d5, stencil=stencil, fixed_bias=b_m, p=0.9,
                                     start_toffolis=2e4)
        assert math.isclose(rr["toffolis"], brute, rel_tol=1e-4) and math.isclose(rr["h"], h_b, rel_tol=1e-3)
        assert rr["toffolis"] <= brute * (1.0 + 1e-12)                            # the closed form is the minimum
        # the stencil bias added in quadrature would overspend the budget
        b_quad = math.sqrt(eps ** 2 - b_m ** 2) / math.sqrt(3.0 if stencil == "central" else 5.0)
        assert (b_m + b_quad) ** 2 + (eps ** 2 - b_m ** 2 - b_quad ** 2) > eps ** 2
    # the LiH values [build check]: 4e-3 central b = 2.0004e-3, h = 4.543e-3, L = 49.49 C_QPE; 1e-3 five-point
    # b = 2.0247e-4, h = 5.434e-3, L = 236.7 C_QPE
    r4 = P.hellmann_feynman_cost(q, 4e-3, d3=d3, fixed_bias=b_m)
    r1 = P.hellmann_feynman_cost(q, 1e-3, d3=d3, d5=d5, stencil="five_point", fixed_bias=b_m)
    assert abs(r4["stencil_bias"] - 2.0004e-3) <= 1e-7 and abs(r4["h"] - 4.5429e-3) <= 1e-7
    assert abs(r4["qpe_toffolis"] / c_qpe - 49.49) <= 0.02
    assert abs(r1["stencil_bias"] - 2.0247e-4) <= 1e-8 and abs(r1["h"] - 5.4343e-3) <= 1e-7
    assert abs(r1["qpe_toffolis"] / c_qpe - 236.66) <= 0.02
    # b_m = 0 reproduces F51's optimum exactly
    for stencil in ("central", "five_point"):
        r0 = P.hellmann_feynman_cost(q, 2e-3, d3=d3, d5=d5, stencil=stencil, fixed_bias=0.0)
        assert r0["stencil_bias"] == 2e-3 / math.sqrt(3.0 if stencil == "central" else 5.0) == r0["bias"]
        assert r0 == P.hellmann_feynman_cost(q, 2e-3, d3=d3, d5=d5, stencil=stencil)
    # infeasible iff b_m >= eps
    bad = P.hellmann_feynman_cost(q, 1e-3, d3=d3, fixed_bias=1e-3)
    assert not bad["feasible"] and bad["toffolis"] == math.inf and bad["bias"] == 1e-3
    for stencil in P.STENCILS:
        worse = P.hellmann_feynman_cost(q, 1e-3, d3=d3, d5=d5, stencil=stencil, fixed_bias=1.5e-3)
        assert not worse["feasible"] and worse["toffolis"] == math.inf and math.isnan(worse["h"])
    assert P.hellmann_feynman_cost(q, 1e-3, d3=d3, fixed_bias=0.999e-3)["feasible"]
    with pytest.raises(ValueError):
        P.hellmann_feynman_cost(q, 1e-3, d3=d3, fixed_bias=-1e-4)
    with pytest.raises(ValueError):
        P.hellmann_feynman_cost(q, 1e-3, d3=d3, fixed_bias=math.nan)


def test_hellmann_feynman_step_refinement():
    """With a stencil-bias callable, h is the largest step (to the bisection's resolution) whose actual stencil
    bias is within the allocated b; sigma_E follows at the refined h."""
    q = _encoding("lih")["qpe"]
    d3, d5, b_m = 581.571, 6.96513e6, 4.7146e-4
    d7 = 1.8e11                                                                  # a next term that matters
    calls = []

    def model(h, stencil):                                                       # D(h) - f'(0), Taylor series
        calls.append((h, stencil))
        if stencil == "central":
            return d3 * h ** 2 / 6.0 + d5 * h ** 4 / 120.0
        return -(d5 * h ** 4 / 30.0 + d7 * h ** 6 / 252.0)

    for eps, stencil in ((4e-3, "central"), (1e-3, "five_point")):
        lead = P.hellmann_feynman_cost(q, eps, d3=d3, d5=d5, stencil=stencil, fixed_bias=b_m)
        r = P.hellmann_feynman_cost(q, eps, d3=d3, d5=d5, stencil=stencil, fixed_bias=b_m, stencil_bias=model)
        b = lead["stencil_bias"]
        from scipy.optimize import brentq

        h_star = brentq(lambda x: abs(model(x, stencil)) - b, 1e-4, lead["h"], xtol=1e-15)
        assert r["h_leading"] == lead["h"] and r["stencil_bias"] == b and r["bias"] == lead["bias"]
        assert abs(model(r["h"], stencil)) <= b
        assert h_star * (1.0 - 1e-3) <= r["h"] <= h_star
        assert r["stencil_bias_checked"] == model(r["h"], stencil) and r["bias_evaluations"] >= 13
        w = 1.0 / (math.sqrt(2.0) * r["h"]) if stencil == "central" else math.sqrt(130.0) / (12.0 * r["h"])
        assert math.isclose(r["sigma_e"], math.sqrt(eps ** 2 - lead["bias"] ** 2) / w, rel_tol=1e-12)
        assert r["toffolis"] > lead["toffolis"]
    # a callable below the allocation leaves F51's h unchanged after one evaluation
    calls.clear()
    r = P.hellmann_feynman_cost(q, 1e-3, d3=d3, d5=d5, stencil="five_point", fixed_bias=b_m,
                                stencil_bias=lambda h, st: 0.9 * d5 * h ** 4 / 30.0)
    assert r["h"] == r["h_leading"] and r["bias_evaluations"] == 1 and not r["no_field_step"]


def test_hellmann_feynman_without_an_admissible_field_step():
    """A stencil bias above the allocation b at every h (a convergence-noise floor n/h on top of the leading term)
    leaves no admissible field step: the estimate is infeasible after 12 halvings of h, not an error."""
    q = _encoding("lih")["qpe"]
    d3, b_m, eps = 581.571, 4.7146e-4, 1e-3
    b = (math.sqrt(b_m ** 2 + 3.0 * eps ** 2) - 2.0 * b_m) / 3.0                  # central allocation
    h0 = math.sqrt(6.0 * b / d3)
    for noise, admissible in ((1e-6, False), (1e-12, True)):
        def model(h, st):
            return d3 * h ** 2 / 6.0 + noise / h

        # min over h of d3 h^2/6 + n/h is 3 (d3/6)^(1/3) (n/2)^(2/3), at h = (3 n/d3)^(1/3)
        floor = 3.0 * (d3 / 6.0) ** (1.0 / 3.0) * (noise / 2.0) ** (2.0 / 3.0)
        grid = np.geomspace(h0 / 2.0 ** 12, h0, 20001)
        assert (floor > b) == (not admissible) == bool(np.all(np.abs(model(grid, "central")) > b))
        r = P.hellmann_feynman_cost(q, eps, d3=d3, fixed_bias=b_m, stencil_bias=model)
        assert math.isclose(r["h_leading"], h0, rel_tol=1e-12) and math.isclose(r["stencil_bias"], b, rel_tol=1e-12)
        assert r["no_field_step"] == (not admissible) and r["feasible"] == admissible
        if admissible:
            assert abs(model(r["h"], "central")) <= b and r["h"] < h0 and math.isfinite(r["toffolis"])
            continue
        assert math.isnan(r["h"]) and r["toffolis"] == math.inf and r["c_qpe_units"] == math.inf
        assert r["bias_evaluations"] == 13                                       # h0 and 12 halvings
        assert r["stencil_bias_checked"] == model(r["h_leading"] / 2.0 ** 12, "central")
        assert math.isclose(r["bias"], b_m + b, rel_tol=1e-12)


# ------------------------------------------------------------------ route A and selection (F49, F53)
def test_route_a_fixed_budgets():
    name = "h4"
    s = _system(name)
    a, psi = s["active"], s["psi"]
    rec = _record(name, "r1", 30)
    prep = prep_cost(rec.directions, rec.eps, irreps=rec.irreps)
    t = s["targets"]["occupations"]
    inp = _inputs(name, "occupations")
    errs = np.array([P.property_error(t, a, ci, psi) for ci in rec.states])
    k_max = rec.n_steps
    tres = np.r_[0.0, 1e6 * 4.0 ** np.arange(k_max)]
    rc = {"meas_r1": tres, "meas_r03": 10 * tres}
    eps = 1e-3
    rows = P.route_a_rows(rec, prep, rc, errs, inp, eps)
    ests = ["shadow", "ae", "ge1", "ge10", "ge100"]
    for lp in ("free", "meas_r1", "meas_r03"):
        for est in ests:
            sel = [r for r in rows if r.loop == lp and r.estimator == est]
            assert sorted(r.k for r in sel if r.route == "A") == [2, 5, 10, 20, 60]
            oracle = [r for r in sel if r.route == "A-oracle"]
            assert len(oracle) == 1 and "oracle_k" in oracle[0].flags
            for r in sel:
                kk = min(r.k, k_max)
                ended = r.route == "A" and r.k > k_max
                assert ("not_usable" in r.flags) == (errs[kk] >= eps or (ended and lp != "free"))
                assert ("record_ended" in r.flags) == ended
    # one row by hand: AE at k = 10 in the measured loop
    r = [x for x in rows if x.route == "A" and x.loop == "meas_r1" and x.estimator == "ae" and x.k == 10][0]
    ae = P.ae_cost(inp, eps, errs[10], PrepParams(float(prep.cumulative[10]), int(prep.logical_qubits[10])))
    assert math.isclose(r.loop_toffolis, float(np.sum(tres[:10])), rel_tol=1e-12)
    assert math.isclose(r.toffolis, float(np.sum(tres[:10])) + ae.toffolis, rel_tol=1e-12)
    assert r.bias == errs[10]
    # the A-oracle row is the minimum over every k, and never selected
    for lp in ("free", "meas_r1"):
        orc = [x for x in rows if x.route == "A-oracle" and x.loop == lp and x.estimator == "ae"][0]
        fixed = [x for x in rows if x.route == "A" and x.loop == lp and x.estimator == "ae" and x.selectable]
        assert all(orc.toffolis <= x.toffolis for x in fixed)
        best = P.best_property_row(rows, loop=lp)
        assert best.route == "A" and "oracle_k" not in best.flags and best.selectable
        assert best.toffolis == min(x.toffolis for x in fixed)
    # an oracle k between the budgets that beats every fixed budget is still not selected
    assert any(x.route == "A-oracle" and x.k not in (2, 5, 10, 20, 60) for x in rows)
    with pytest.raises(ValueError):
        P.best_property_row([x for x in rows if x.route == "A-oracle"], loop="free")
    with pytest.raises(ValueError):
        P.route_a_rows(rec, prep, rc, errs[:-1], inp, eps)
    with pytest.raises(ValueError):
        P.route_a_rows(rec, prep, {}, errs, inp, eps)


def test_route_a_budgets_beyond_the_record():
    """A budget past a converged record: the free loop costs C(Psi_K) (selectable); a measured loop would run
    k - K more measured iterations (C_loop(K) is a lower bound): reported, flagged and never selected. Past a
    record that did not converge no row is usable."""
    name = "h4"
    s = _system(name)
    a, psi = s["active"], s["psi"]
    t = s["targets"]["occupations"]
    inp = _inputs(name, "occupations")
    rec = _record(name, "r1", 60)
    assert rec.converged and rec.n_steps < 60                                    # [build check] K = 29
    k_last = rec.n_steps
    prep = prep_cost(rec.directions, rec.eps, irreps=rec.irreps)
    errs = np.array([P.property_error(t, a, ci, psi) for ci in rec.states])
    eps = 1e-2
    assert errs[10] < eps and errs[k_last] < eps
    tres = {"meas_r1": np.r_[0.0, np.full(k_last, 1e5)]}
    rows = P.route_a_rows(rec, prep, tres, errs, inp, eps, loops=("free", "meas_r1"), budgets=(10, 60))
    for est in ("ae", "shadow"):
        free = [r for r in rows if r.route == "A" and r.loop == "free" and r.k == 60 and r.estimator == est][0]
        meas = [r for r in rows if r.route == "A" and r.loop == "meas_r1" and r.k == 60 and r.estimator == est][0]
        assert "record_ended" in free.flags and free.selectable and free.loop_toffolis == 0.0
        cost = (P.ae_cost if est == "ae" else P.shadow_cost)(inp, eps, errs[k_last],
                                                             PrepParams(float(prep.cumulative[k_last]),
                                                                        int(prep.logical_qubits[k_last])))
        assert math.isclose(free.toffolis, cost.toffolis, rel_tol=1e-12)
        assert {"record_ended", "not_usable"} <= set(meas.flags) and not meas.selectable
        assert math.isclose(meas.loop_toffolis, 1e5 * (k_last - 1), rel_tol=1e-12)   # T_res(0) = 0
        best = P.best_property_row(rows, loop="meas_r1", estimator=est)
        assert best.k == 10 and best.selectable
        assert P.best_property_row(rows, loop="free", estimator=est).k in (10, 60)
    # a record that stopped before converging: no row past it is usable, whatever the loop
    short = _record(name, "r1", 8)
    assert not short.converged and short.n_steps == 8
    prep8 = prep_cost(short.directions, short.eps, irreps=short.irreps)
    errs8 = np.array([P.property_error(t, a, ci, psi) for ci in short.states])
    rows8 = P.route_a_rows(short, prep8, {"meas_r1": np.r_[0.0, np.full(8, 1e5)]}, errs8, inp, 0.5,
                           loops=("free", "meas_r1"), budgets=(2, 20))
    for r in rows8:
        if r.route == "A" and r.k == 20:
            assert {"record_ended", "not_usable"} <= set(r.flags) and not r.selectable
        elif r.route == "A":
            assert r.selectable
    # the same 8 steps as a record that ended by converging: budgets (2, 20) with one measured loop
    done = dataclasses.replace(short, converged=True)
    tres8 = {"meas_r1": np.r_[0.0, np.full(8, 1e5)]}
    rows_c = P.route_a_rows(done, prep8, tres8, errs8, inp, 0.5, loops=("free", "meas_r1"), budgets=(2, 20))
    for est in ("ae", "shadow"):
        free = [r for r in rows_c if r.route == "A" and r.loop == "free" and r.k == 20 and r.estimator == est][0]
        meas = [r for r in rows_c if r.route == "A" and r.loop == "meas_r1" and r.k == 20 and r.estimator == est][0]
        cost = (P.ae_cost if est == "ae" else P.shadow_cost)(inp, 0.5, errs8[8], PrepParams(
            float(prep8.cumulative[8]), int(prep8.logical_qubits[8])))
        assert free.selectable and "record_ended" in free.flags and free.loop_toffolis == 0.0
        assert math.isclose(free.toffolis, cost.toffolis, rel_tol=1e-12)
        assert not meas.selectable and {"record_ended", "not_usable"} <= set(meas.flags)
        best = P.best_property_row(rows_c, loop="meas_r1", estimator=est)
        assert best.k == 2 and best.selectable


def test_best_property_row_rules():
    common = dict(system="x", target="rdm1", eps=1e-3, estimator="ae", loop="free", bias=1e-4, loop_toffolis=0.0,
                  per_prep=1.0, n_prep=1.0, longest_run=1.0, logical_qubits=10)
    rows = [P.PropertyRow(route="A-oracle", k=3, toffolis=1.0, flags=("oracle_k",), **common),
            P.PropertyRow(route="A", k=2, toffolis=2.0, flags=("not_usable",), **common),
            P.PropertyRow(route="B0", k=0, toffolis=3.0, flags=("not_terminating",), **common),
            P.PropertyRow(route="Bk", k=4, toffolis=math.inf, flags=(), **common),
            P.PropertyRow(route="A", k=5, toffolis=7.0, flags=(), **common),
            P.PropertyRow(route="Bk", k=1, toffolis=6.0, flags=("termination_not_certified",), **common),
            P.PropertyRow(**dict(common, estimator="qpe"), route="HF-QPE", k=0, toffolis=5.0, flags=()),
            P.PropertyRow(**dict(common, estimator="shadow"), route="A", k=2, toffolis=0.5, flags=())]
    assert P.best_property_row(rows, loop="free").route == "HF-QPE"
    assert P.best_property_row(rows, loop="free", estimator="shadow").toffolis == 0.5
    assert P.best_property_row(rows[:6], loop="free").toffolis == 6.0
    with pytest.raises(ValueError):
        P.best_property_row(rows, loop="meas_r1")
    with pytest.raises(ValueError):
        P.PropertyRow(route="C", k=0, toffolis=1.0, **common)
    with pytest.raises(ValueError):
        P.PropertyRow(route="A", k=-1, toffolis=1.0, **common)
    with pytest.raises(ValueError):
        P.PropertyRow(**dict(common, estimator="mc"), route="A", k=0, toffolis=1.0)


def test_run_layout_and_the_no_layout_flag():
    """F53 lays out every run (F18): no factory fails below 3.6e-16 per Toffoli [OF], so a single coherent run at
    p_fail_ae = 1e-4 has no layout beyond 1e-4/3.6e-16 = 2.78e11 Toffolis (3.95e11 on 261 qubits: none; 2.5e11, whose
    distillation term is 9.0e-5: a layout), and a shadow snapshot, at p_fail_shot = 1e-5 with its readout, none
    beyond 2.78e10. The row builders flag such rows "no_layout" and keep them selectable: with p_fail_ae below every
    factory's failure per Toffoli each finite coherent row of route A and route B is flagged, the shadow rows (at
    p_fail_shot) are not, and the best rows do not change."""
    from thc_engine_cqe.ft import physical

    params = FTParameters()
    f_min = min(f.failure for f in physical._known_factories(physical.SurfaceCodeModel().p_phys))
    assert math.isclose(f_min, 3.6e-16, rel_tol=1e-12)
    common = dict(system="x", target="rdm1", eps=1e-3, route="Bk", loop="free", k=3, bias=1e-4, toffolis=1e12,
                  loop_toffolis=0.0, per_prep=1.0, n_prep=1.0, logical_qubits=261)
    for est, budget in (("ae", params.p_fail_ae), ("qpe", params.p_fail_ae), ("ge10", params.p_fail_ae),
                        ("shadow", params.p_fail_shot)):
        limit = budget / f_min
        below = P.PropertyRow(estimator=est, longest_run=0.9 * limit, **common)
        above = P.PropertyRow(estimator=est, longest_run=1.01 * limit, **common)
        lay = P.run_layout(below, params=params)
        assert lay is not None and lay.distance <= physical.SurfaceCodeModel().distances[-1], est
        assert P.run_layout(above, params=params) is None, est
    ae = dict(common, estimator="ae")
    assert P.run_layout(P.PropertyRow(longest_run=3.95e11, **ae)) is None
    assert P.run_layout(P.PropertyRow(longest_run=2.5e11, **ae)) is not None
    assert P.run_layout(P.PropertyRow(longest_run=math.inf, **dict(ae, toffolis=math.inf))) is None
    # the builders: route A on h4 and route B on n2s with a budget no factory meets
    tight = dataclasses.replace(params, p_fail_ae=1e-17)
    s = _system("h4")
    rec = _record("h4", "r1", 30)
    prep = prep_cost(rec.directions, rec.eps, irreps=rec.irreps)
    t = s["targets"]["occupations"]
    errs = np.array([P.property_error(t, s["active"], ci, s["psi"]) for ci in rec.states])
    rc = {"meas_r1": np.r_[0.0, 1e6 * 4.0 ** np.arange(rec.n_steps)]}
    inp = _inputs("h4", "occupations")
    d = _n2s_route_b()
    e = d["e"]
    for build in (lambda pr: P.route_a_rows(rec, prep, rc, errs, inp, 1e-2, loops=("free", "meas_r1"), params=pr),
                  lambda pr: P.route_b_rows(d["rec"], d["prep"], d["tres"], d["inp"], 1e-2, e["enc"], e["c"],
                                            u_classical=e["u"], e1_lower=e["e1lb"], e0_model=d["e0m"],
                                            model_bias=2e-4, params=pr)):
        plain, flagged = build(params), build(tight)
        assert len(plain) == len(flagged)
        n_flagged = 0
        for a, b in zip(plain, flagged):
            finite = math.isfinite(b.toffolis) and math.isfinite(b.longest_run)
            want = finite and b.estimator != "shadow"
            assert ("no_layout" in b.flags) == want, (b.route, b.estimator, b.loop, b.k)
            assert ("no_layout" in a.flags) == (finite and P.run_layout(a, params=params) is None)
            assert tuple(f for f in b.flags if f != "no_layout") == tuple(f for f in a.flags if f != "no_layout")
            assert b.selectable == a.selectable and b.toffolis == a.toffolis
            n_flagged += want
        assert n_flagged > 0
        for lp in ("free", "meas_r1"):
            assert P.best_property_row(flagged, loop=lp).toffolis == P.best_property_row(plain, loop=lp).toffolis


def _row(route, k, toffolis, *, loop="free", estimator="ae", flags=(), bias=1e-4, system="s", target="rdm1",
         eps=1e-3):
    return P.PropertyRow(system=system, target=target, eps=eps, route=route, estimator=estimator, loop=loop, k=k,
                         bias=bias, toffolis=toffolis, loop_toffolis=0.0, per_prep=1.0, n_prep=1.0, longest_run=1.0,
                         logical_qubits=10, flags=tuple(flags))


def _row_json(row: P.PropertyRow, c_qpe: float = 1.0) -> dict:
    out = scaling.to_json(row)
    out["toffolis_over_qpe"] = scaling.to_json(row.toffolis / c_qpe)
    out["selectable"] = bool(row.selectable)
    return out


def _pair_json(name: str, win_k: int, *, compiled_usable: bool, floors: dict, sens_route: str | None = None,
               sens_free_cost: float | None = None) -> dict:
    """A properties_<system>.json with one rdm1 pair at eps = 1e-3 (accuracies 1e-2, 1e-3): route A beats the
    Bk row (5 Toffolis) only at k = win_k; the compiled route-A rows are usable or not."""
    eps = 1e-3
    rows_a, rows_c, rows_b = [], [], []
    for lp in ("free", "meas_r1"):
        for k in (2, 5, 10, 20, 60):
            rows_a.append(_row("A", k, 1.0 if k == win_k else 10.0, loop=lp, system=name, eps=eps))
            rows_c.append(_row("A", k, 1.5 if k == win_k else 12.0, loop=lp, system=name, eps=eps,
                               bias=1e-4 if compiled_usable else 2e-3,
                               flags=() if compiled_usable else ("not_usable",)))
        rows_a.append(_row("A-oracle", win_k, 1.0, loop=lp, system=name, eps=eps, flags=("oracle_k",)))
        rows_b.append(_row("Bk", 3, 5.0, loop=lp, system=name, eps=eps, flags=("edge:U_classical",)))
        rows_b.append(_row("B0", 1, math.inf, loop=lp, system=name, eps=eps, bias=math.inf,
                           flags=("not_usable", "infeasible")))
    best = {lp: {"ae": {"route": "A", "estimator": "ae", "k": win_k, "toffolis_over_qpe": 1.0, "bias": 1e-4,
                        "flags": []}, "shadow": None} for lp in ("free", "meas_r1")}
    sens = {lp: dict(v) for lp, v in best.items()}
    if sens_route is not None:
        sens["meas_r1"] = {"ae": {"route": sens_route, "estimator": "ae", "k": 3, "toffolis_over_qpe": 7.0,
                                  "bias": 2e-4, "flags": []}, "shadow": None}
    if sens_free_cost is not None:
        sens["free"] = {"ae": dict(best["free"]["ae"], toffolis_over_qpe=sens_free_cost), "shadow": None}
    blk = {"accuracies": [1e-2, eps], "floors": {lab: {"min": v} for lab, v in floors.items()},
           "rows": {str(eps): {"route_a": [_row_json(r) for r in rows_a], "route_b": [_row_json(r) for r in rows_b],
                               "route_a_compiled": [_row_json(r) for r in rows_c]}},
           "best": {str(eps): best}, "model_bias_sensitivity": {str(eps): {"best": sens}}}
    return {"system": name, "targets": {"rdm1": blk},
            "records": {"compiled_sliced": {"slices": [2, 1, 1] + [1] * 5, "slice_errors": [0.0249, 0.0046, 0.0007]
                                            + [None] * 5}}}


def test_route_tally_and_q6_comparison():
    """The route comparison of U0's Q6 from benchmark JSON: best of budgets, a single pre-declared budget per k,
    route A on compiled iterates, compiled bias-limited pairs and the cells resting on the exact model bias."""
    pj = {"s1": _pair_json("s1", 5, compiled_usable=False, floors={"r1_so": 1e-5, "compiled_s1": 3e-2,
                                                                     "compiled_sliced": 2e-2}, sens_free_cost=1.5),
          "s2": _pair_json("s2", 20, compiled_usable=True, floors={"r1_so": 2e-5, "compiled_s1": 2e-3,
                                                                    "compiled_sliced": 5e-4}, sens_route="Bk")}
    # s2's first step capped at s = 16 above eta_s
    pj["s2"]["records"]["compiled_sliced"].update(slices=[16, 1, 1] + [1] * 5, flags=["slice_cap:step0"],
                                                  slice_errors=[0.06, 0.0046, 0.0007] + [None] * 5)
    pairs = P.property_pairs(pj)
    assert [(p["system"], p["target"], p["eps"]) for p in pairs] == [("s1", "rdm1", 1e-3), ("s2", "rdm1", 1e-3)]
    assert all(isinstance(r, P.PropertyRow) for p in pairs for r in p["rows"])
    # the JSON round trip of a row
    r = _row("B0", 1, math.inf, bias=math.inf, flags=("not_usable", "infeasible"))
    assert P.row_from_json(_row_json(r)) == r
    cmp = P.q6_comparison(pj)
    ex = cmp["exact"]
    assert ex["pairs"] == 2
    for lp in ("free", "meas_r1"):
        assert ex["best"][lp] == {"A": 2}
        assert {k: v.get("A", 0) for k, v in ex["fixed_k"][lp].items()} == {2: 0, 5: 1, 10: 0, 20: 1, 60: 0}
        assert {k: v.get("Bk", 0) for k, v in ex["fixed_k"][lp].items()} == {2: 2, 5: 1, 10: 2, 20: 1, 60: 2}
        # the compiled tally drops s1's route-A win (its compiled iterates are not usable)
        assert cmp["compiled"]["best"][lp] == {"Bk": 1, "A": 1}
    assert P.best_at_budget(pairs[0]["rows"], loop="free", budget=20).route == "Bk"
    assert P.best_at_budget(pairs[0]["rows"], loop="free").k == 5
    assert P.best_at_budget([x for x in pairs[0]["rows"] if x.route == "B0"], loop="free") is None
    lim = cmp["compiled_bias_limited"]
    assert [(x["system"], x["r1_so"], x["compiled_s1"], x["compiled_sliced"]) for x in lim] == [
        ("s1", 1e-5, 3e-2, 2e-2)]
    assert cmp["largest_slice_error"] == ("s1", 0, 0.0249) and cmp["slice_caps"] == [("s2", 0, 0.06)]
    assert cmp["slices"]["s2"] == {"steps": [0, 1, 2], "slices": [16, 1, 1], "errors": [0.06, 0.0046, 0.0007],
                                   "capped": [0]}
    sens = cmp["model_bias_sensitivity"]
    assert [(x["system"], x["loop"], x["change"], x["exact"]["route"], x["f47"]["route"]) for x in sens] == [
        ("s1", "free", "cost", "A", "A"), ("s2", "meas_r1", "route", "A", "Bk")]
    assert sens[0]["ratio"] == 1.5 and sens[1]["ratio"] == 7.0
    st = P.q6_statements(cmp)
    text = " ".join(st["verdict"])
    assert "k = 2 / 5 / 10 / 20 / 60" in text and "0 / 1 / 0 / 1 / 0 with a free loop" in text
    assert "reproduced by no single budget" in text and "of the 2 pairs" in text
    assert "eps on 1 pair (floors R1 / compiled s = 1 / compiled sliced): s1 rdm1 (1e-5 / 0.03 / 0.02)" in text
    assert "1 with a free loop" in text
    unc = " ".join(st["uncertainty"])
    assert ("state error (up to 0.0249 where it meets eta_s, s1's step 0 at s = 2; 0.06 at the cap s = 16, s2's "
            "step 0), not the property error") in unc
    assert "the best route changes on 1 cell, " in unc and "s2 rdm1 meas_r1 (A 1 -> Bk 7)" in unc
    assert "1 more cell keeps its route at another cost (up to x1.5, s1 rdm1 free)" in unc
    # the largest error that meets eta_s on an unsliced step (s = 1) is named with its own s
    pj3 = copy.deepcopy(pj)
    pj3["s1"]["records"]["compiled_sliced"]["slice_errors"][1] = 0.045
    cmp3 = P.q6_comparison(pj3)
    assert cmp3["largest_slice_error"] == ("s1", 1, 0.045)
    assert "(up to 0.045 where it meets eta_s, s1's step 1 at s = 1; 0.06 at the cap" in " ".join(
        P.q6_statements(cmp3)["uncertainty"])
    # a cell without a selectable row under F47's bound
    pj["s2"]["targets"]["rdm1"]["model_bias_sensitivity"]["0.001"]["best"]["free"]["ae"] = None
    cmp2 = P.q6_comparison(pj)
    assert [(x["loop"], x["change"], x["f47"]["route"], x["ratio"]) for x in cmp2["model_bias_sensitivity"]
            if x["system"] == "s2"] == [("free", "route", "none", math.inf), ("meas_r1", "route", "Bk", 7.0)]
    unc2 = " ".join(P.q6_statements(cmp2)["uncertainty"])
    assert "the best route changes on 2 cells, " in unc2 and "s2 rdm1 free (A 1 -> none)" in unc2
    # route "none" where nothing is selectable
    t = P.route_tally([[x for x in pairs[0]["rows"] if x.route == "B0"]], loops=("free",), budgets=(5,))
    assert t == {"pairs": 1, "best": {"free": {"none": 1}}, "fixed_k": {"free": {5: {"none": 1}}}}


def test_model_bias_bound_helper():
    inp = _synthetic_inputs("fro", 5)
    q = 2.1e-6
    expect = 2 * inp.sigma_norm * math.sqrt(q * (1 - q)) + q * inp.range_norm
    assert math.isclose(P.model_bias_bound(inp, q), expect, rel_tol=1e-12)
    with pytest.raises(ValueError):
        P.model_bias_bound(inp, 1.5)


def test_select_for_property_rules():
    base = ftqpe.sf_encoding(_system("h4")["active"], 7, reference=ExactReference.build(_system("h4")["active"]))

    def enc(lam, de_fact):
        return dataclasses.replace(base, lam=lam, de_fact=de_fact)

    params = FTParameters()
    encs = [enc(5.0, 1e-4), enc(4.0, 1e-3), enc(6.0, 2e-4), enc(4.5, 3e-4)]
    # tier 1: |de_fact| <= 0.5 mHa and b <= eps/2, the cheapest QPE (lambda 4.5)
    q = P.select_for_property(encs, [1e-4, 1e-4, 1e-4, 4e-4], 1e-3, params=params)
    assert q.encoding.lam == 4.5
    # tier 1 empty: the cheapest with b <= eps/2 regardless of F9 (lambda 4.0)
    q = P.select_for_property(encs, [9e-4, 1e-4, 9e-4, 9e-4], 1e-3, params=params)
    assert q.encoding.lam == 4.0
    with pytest.raises(ValueError):
        P.select_for_property(encs, [9e-4] * 4, 1e-3, params=params)
    with pytest.raises(ValueError):
        P.select_for_property(encs, [1e-4] * 3, 1e-3, params=params)


def test_compiled_iterates_equal_replay():
    s = _system("h4")
    rec = _record("h4", "ls", 6)
    comp = P.compiled_iterates(s["active"], rec, slices=1)
    ref = replay_elementwise(s["active"], rec.directions, rec.eps, irreps=rec.irreps)
    assert len(comp) == rec.n_steps + 1
    assert all(np.array_equal(x, y) for x, y in zip(comp, ref))
    comp2 = P.compiled_iterates(s["active"], rec, slices=2, sliced_steps=(0,))
    d = list(rec.directions)
    e = list(rec.eps)
    ref2 = replay_elementwise(s["active"], [d[0], d[0]] + d[1:], [e[0] / 2, e[0] / 2] + e[1:], irreps=rec.irreps)
    assert np.array_equal(comp2[0], ref2[0]) and np.array_equal(comp2[1], ref2[2])
    assert np.array_equal(comp2[-1], ref2[-1])
    c1, q1 = P.sliced_prep_cost(rec, slices=1)
    pc = prep_cost(rec.directions, rec.eps, irreps=rec.irreps)
    assert np.array_equal(c1, pc.cumulative) and np.array_equal(q1, pc.logical_qubits)


# ------------------------------------------------------------------ slow
@pytest.mark.slow
def test_matchgate_variances_match_snapshots():
    s = _system("h4")
    a, psi = s["active"], s["psi"]
    mp = QubitMapping(a.norb)
    st = QubitState.from_ci(a.norb, a.nelec, psi, mp)
    for kind in ("rdm1", "rdm2", "occupations", "cumulant", "dipole"):
        ops = s["targets"][kind].operators(mp)[:6]
        ex = schemes.matchgate_shadow_variance(ops, st, method="exact")
        mc = schemes.matchgate_shadow_variance(ops, st, method="mc", n_snapshots=20000, seed=11)
        z = (np.asarray(mc.per_target) - np.asarray(ex.per_target)) / np.asarray(mc.per_target_stderr)
        assert np.all(np.abs(z) <= 4.0), (kind, z)
        assert np.allclose(_inputs("h4", kind).v_shadow[:len(ops)], ex.per_target)


@pytest.mark.slow
@pytest.mark.parametrize("name", ["h4", "n2s"])
def test_model_bias_bound_on_scanned_encodings(name):
    s = _system(name)
    a = s["active"]
    ref = ExactReference.build(a)
    psi = normalize(np.asarray(ref.ci))              # the reference of the scan's infidelities
    encs = ftqpe.scan_encodings(a, reference=ref)
    kinds = [k for k in ("rdm1", "rdm2", "occupations", "dipole") if k in s["targets"]]
    spreads = {k: (s["targets"][k].spreads(a, psi), s["targets"][k].ranges(a, psi)) for k in kinds}
    worst = math.inf
    for enc in encs:
        _, psi_m = model_floors(ref, enc.eri_model, return_state=True)
        psi_m = normalize(np.asarray(psi_m))
        q = 1.0 - float(np.vdot(psi.ravel(), psi_m.ravel()) ** 2)
        assert math.isclose(q, enc.infidelity, rel_tol=1e-9, abs_tol=1e-12)
        for k in kinds:
            t = s["targets"][k]
            b = P._norm(t, t.expectations(a, psi_m) - t.expectations(a, psi))
            bound = P.bias_bound(t, a, psi, q=q, sigma0=spreads[k][0], ranges=spreads[k][1])
            if b > 1e-10:
                worst = min(worst, bound / b)
            assert b <= bound * (1 + 1e-9) + 1e-12, (enc.kind, enc.params, k)
    assert worst >= 1.0


@pytest.mark.slow
@pytest.mark.parametrize("name, qs", [("h4", (0.1, 0.5)), ("lih", (0.5,))])
def test_cumulant_norm_bound_adversarial(name, qs):
    """chi in psi_0's sector (the excited eigenstates of H with HF overlap) chosen to maximize the cumulant norm's
    error at fixed q: the linearized F47 bound fails (it is first order), B_cum + 3/2 B_1^2 holds."""
    from scipy.optimize import minimize

    from thc_engine_cqe.fci_ops import Hamiltonian

    s = _system(name)
    a, t = s["active"], s["targets"]
    space = FCISpace.of(a)
    n = space.size
    ham = Hamiltonian(a)
    hmat = np.column_stack([np.asarray(ham.apply(col.reshape(space.shape))).ravel() for col in np.eye(n)])
    w, v = np.linalg.eigh(0.5 * (hmat + hmat.T))
    reach = np.flatnonzero(np.abs(v.T @ np.asarray(space.hf()).ravel()) > 1e-8)
    psi0 = v[:, reach[0]]
    assert abs(abs(float(psi0 @ s["psi"].ravel())) - 1.0) <= 1e-10
    psi0 = psi0 * np.sign(float(psi0 @ s["psi"].ravel()))
    ref = psi0.reshape(space.shape)
    basis = v[:, reach[1:]]
    tn = t["cumulant_norm"]
    parts = (_f47_inputs(t["cumulant"], a, ref), _f47_inputs(t["rdm1"], a, ref))
    sig, rr = tn.spreads(a, ref), tn.ranges(a, ref)
    y_lin = basis.T @ np.asarray(tn._apply(space, 0, ref)).ravel()            # the first-order direction
    rng = np.random.default_rng(2026)
    for q in qs:
        def err(y):
            chi = basis @ y
            ci = (math.sqrt(1.0 - q) * psi0 + math.sqrt(q) * chi / np.linalg.norm(chi)).reshape(space.shape)
            return P.property_error(tn, a, ci, ref)

        worst = 0.0
        for y0 in [y_lin, -y_lin] + [rng.standard_normal(basis.shape[1]) for _ in range(2)]:
            res = minimize(lambda y: -err(y), y0, method="BFGS", options={"maxiter": 300, "gtol": 1e-10})
            worst = max(worst, -res.fun)
        lin = P.bias_bound(tn, a, ref, q=q, sigma0=sig, ranges=rr)
        full = P.bias_bound(tn, a, ref, q=q, cumulant_parts=parts)
        assert lin < worst <= full, (q, lin, worst, full)
    if name == "h4":
        assert abs(worst - 3.648) <= 0.01 and abs(lin - 3.431) <= 0.01            # [build check] q = 0.5


@pytest.mark.slow
def test_property_selection_on_n2s():
    s = _system("n2s")
    a, psi = s["active"], s["psi"]
    ref = ExactReference.build(a)
    encs = ftqpe.scan_encodings(a, reference=ref)
    t = s["targets"]["occupations"]
    errs = P.model_property_errors(a, encs, t, psi, reference=ref)
    sel = ftqpe.select_qpe(encs)
    assert sel.encoding.kind == "sf" and sel.encoding.params["L"] == 17
    i_sel = [i for i, e in enumerate(encs) if e is sel.encoding][0]
    assert errs[i_sel] > 0.5e-3                                                  # [build check] 9.7e-4
    q = P.select_for_property(encs, errs, 1e-3)
    assert q.encoding.kind == "sf" and q.encoding.params["L"] == 18             # [build check]
    assert 1.0 < q.toffolis / sel.toffolis < 1.05                                # +2.6% [build check]


@pytest.mark.slow
@pytest.mark.parametrize("name", ["h4", "lih", "n2s"])
def test_filter_leakage_bound_on_exact_spectra(name):
    """The constructed step filter on H' applied to HF: success, leaked weight and property errors within F50."""
    s = _system(name)
    e = _encoding(name)
    ms = e["spectrum"]
    a = s["active"]
    hf = FCISpace.of(a).hf()
    cj = ms.vectors.T @ hf.ravel()
    p = float(cj[0] ** 2)
    psi_m = ms.vectors[:, 0].reshape(ms.shape)
    u = e["u"] if e["u"] < e["e1lb"] - 2 * DE else e["e1lb"] - amplify.W_L
    theta = np.arccos(np.clip((ms.energies - e["c"]) / ms.lam, -1, 1))
    th_u = math.acos((u - e["c"]) / ms.lam)
    th_1 = math.acos((e["e1lb"] - e["c"]) / ms.lam)
    kinds = [k for k in ("occupations", "rdm1", "dipole") if k in s["targets"]]
    for eps_f in (1e-2, 1e-3):
        d, coef = amplify.step_filter_polynomial(th_u, th_1, eps_f)
        assert d <= amplify.filter_degree(th_u, th_1, eps_f)
        g = np.cos(np.outer(theta, np.arange(d + 1))) @ coef
        assert np.all(np.abs(g[theta <= th_1]) <= eps_f * (1 + 1e-6))
        amp = g * cj
        success = float(np.sum(amp ** 2))
        assert p * (1 - eps_f) ** 2 <= success <= p + eps_f ** 2 * (1 - p)
        w = float(np.sum(amp[1:] ** 2)) / success
        w_bound = eps_f ** 2 * (1 - p) / (p * (1 - eps_f) ** 2)
        assert w <= w_bound
        phi = (ms.vectors @ amp).reshape(ms.shape)
        for k in kinds:
            t = s["targets"][k]
            sig, rng = t.spreads(a, psi_m), t.ranges(a, psi_m)
            b = P._norm(t, t.expectations(a, phi) - t.expectations(a, psi_m))
            assert b <= P.bias_bound(t, a, psi_m, q=w_bound, sigma0=sig, ranges=rng), (k, eps_f)
            # adversarial stop-band signs stay within the bound
            for e_idx in range(min(t.n_elements, 3)):
                space = FCISpace.of(a)
                o_psi = t._apply(space, e_idx, psi_m)
                o0j = ms.vectors.T @ o_psi.ravel()
                gg = np.where(theta <= th_1, eps_f * np.sign(o0j * cj), 1.0)
                gg[0] = 1.0
                am = gg * cj
                ph = (ms.vectors @ am) / math.sqrt(float(np.sum(am ** 2)))
                ph = ph.reshape(ms.shape)
                w_adv = float(np.sum(am[1:] ** 2)) / float(np.sum(am ** 2))
                dev = abs(t.expectations(a, ph)[e_idx] - t.expectations(a, psi_m)[e_idx])
                assert dev <= 2 * sig[e_idx] * math.sqrt(w_adv) + w_adv * rng[e_idx]


@pytest.mark.slow
def test_finite_field_bias_on_lih():
    import dataclasses as dc

    s = _system("lih")
    a, t = s["active"], s["targets"]["dipole"]
    mu0 = P.property_values(t, a, s["psi"])[0]
    f = t.one_body[0]

    def energy(h):
        b = dc.replace(a, h1=a.h1 + h * f, ecore=a.ecore + h * t.constant)
        return float(np.atleast_1d(b.fci(nroots=1, conv_tol=1e-13)[0])[0])

    h = 1e-3
    central = (energy(h) - energy(-h)) / (2 * h) - mu0
    assert abs(central - 9.69e-5) <= 0.01 * 9.69e-5                              # [build check] 9.685e-5
    ff = P.finite_field_derivatives(a, t)
    assert math.isclose(central, ff["d3"] * h * h / 6, rel_tol=0.01)
    assert abs(ff["d1"] - mu0) <= 1e-5
    # the five-point bias falls as h^4: E^(5) h^4/30
    five = {}
    for hh in (4e-3, 8e-3):
        e2p, ep, em, e2m = energy(2 * hh), energy(hh), energy(-hh), energy(-2 * hh)
        five[hh] = (-e2p + 8 * ep - 8 * em + e2m) / (12 * hh) - mu0
    assert 13.0 < five[8e-3] / five[4e-3] < 19.0
    assert math.isclose(five[4e-3], -ff["d5"] * 4e-3 ** 4 / 30, rel_tol=0.15)
    # at the steps of the LiH HF-QPE rows (the model bias of SF L = 16 added linearly) F51's leading order
    # underestimates the stencil bias; the refined step meets the allocation (FiniteFieldStencil, same model)
    e = _encoding("lih")
    b_m = _lih_model_bias()
    stencil = P.FiniteFieldStencil(a, t)
    assert abs(stencil.expectation - mu0) <= 1e-8

    def actual(hh, st):
        if st == "central":
            return (energy(hh) - energy(-hh)) / (2 * hh) - mu0
        return (-energy(2 * hh) + 8 * energy(hh) - 8 * energy(-hh) + energy(-2 * hh)) / (12 * hh) - mu0

    for eps, st, low in ((1e-3, "five_point", 1.08), (4e-3, "central", 1.005)):       # [build check] 1.105, 1.011
        lead = P.hellmann_feynman_cost(e["qpe"], eps, d3=ff["d3"], d5=ff["d5"], stencil=st, fixed_bias=b_m)
        fit = P.hellmann_feynman_cost(e["qpe"], eps, d3=ff["d3"], d5=ff["d5"], stencil=st, fixed_bias=b_m,
                                      stencil_bias=stencil)
        b = lead["stencil_bias"]
        assert abs(actual(lead["h"], st)) > low * b
        assert fit["h"] < lead["h"] and 0.99 * b <= abs(actual(fit["h"], st)) <= b + 1e-8
        assert abs(stencil.bias(fit["h"], st) - actual(fit["h"], st)) <= 1e-8
        assert math.isclose(fit["stencil_bias_checked"], stencil.bias(fit["h"], st), rel_tol=1e-12)


@pytest.mark.slow
def test_compiled_floors():
    for name, k, ref in (("h4", 10, 1.8e-3), ("lih", 14, 2.9e-4)):                # [build check]
        s = _system(name)
        rec = _record(name, "ls", k)
        comp = P.compiled_iterates(s["active"], rec, slices=1)
        dist = float(np.linalg.norm(normalize(comp[k]) - rec.states[k]))
        assert abs(dist - ref) <= 0.1 * ref, (name, dist)
    s = _system("h4")
    rec = _record("h4", "ls", 10)
    comp8 = P.compiled_iterates(s["active"], rec, slices=8, sliced_steps=(0, 1))
    assert float(np.linalg.norm(normalize(comp8[10]) - rec.states[10])) < 1.4e-3           # [build check] 1.31e-3
    c1, _ = P.sliced_prep_cost(rec, slices=1)
    c8, _ = P.sliced_prep_cost(rec, slices=8, sliced_steps=(0, 1))
    assert 1.6 < c8[10] / c1[10] < 1.9                                              # [build check] 1.74

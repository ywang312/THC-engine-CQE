"""Tests of the M3 state-preparation costs (``thc_engine_cqe.ft.stateprep``, F3-F7).

References independent of the module: the Appendix A table of docs/m3-cost-model.md (recomputed
for the design by a script that pair-symmetrizes, masks and expands with its own tolerance and
fixed point [design]), a literal re-implementation of F3 and F6 written in this file (the expansion
of eps * mask(d) with eps inside, bits by enumeration, the fixed point by search), the rank of the
RHF residual W = 4(ia|jb) - 2(ib|ja) from the ERIs (M1), and the count formulas of F3-F5 typed in
from docs/m3-cost-model.md.
"""

import math

import numpy as np
import pytest

from thc_engine_cqe.cqe import run_cqe
from thc_engine_cqe.diagnostics.cost import Excitation, spin_orbital_excitations
from thc_engine_cqe.factor.generator import factorize_generator
from thc_engine_cqe.ft import stateprep as sp
from thc_engine_cqe.ft import symmetry as sym
from thc_engine_cqe.ft.primitives import FTParameters

# Appendix A, L-BFGS trajectories: (E - E_FCI after the step in mHa, overlap after the step,
# n_double, n_single, b, C_step, cumulative C(Psi_{k+1})).
APPENDIX_A = {
    "h4": [(21.504, 0.9802, 10, 0, 15, 160, 160), (3.890, 0.9960, 40, 24, 18, 1192, 1352),
           (1.203, 0.9981, 38, 24, 18, 1154, 2506)],
    "h6a": [(37.956, 0.9576, 51, 0, 18, 969, 969), (12.014, 0.9850, 273, 120, 21, 8526, 9495),
            (5.236, 0.9907, 271, 120, 21, 8482, 17977), (2.991, 0.9950, 271, 120, 21, 8482, 26459),
            (1.877, 0.9974, 273, 120, 21, 8526, 34985), (1.111, 0.9987, 273, 116, 21, 8442, 43427)],
    "n2": [(35.209, 0.9635, 27, 0, 17, 486, 486), (14.678, 0.9813, 55, 0, 18, 1045, 1531),
           (1.147, 0.9994, 55, 0, 18, 1045, 2576)],
}

_TRAJECTORIES: dict = {}


def trajectory(name: str) -> dict:
    """First 12 exact L-BFGS steps on the symmetry-adapted build (cached; a shorter run is a prefix)."""
    if name not in _TRAJECTORIES:
        act = sym.benchmark_system(name)
        e0, ci0 = act.fci()
        states = []
        res = run_cqe(act, max_iter=12, e_exact=e0, keep_generators=True, optimizer="lbfgs",
                      callback=lambda st, ci: states.append(ci.copy()))
        steps = [s for s in res.steps if s.eps != 0.0]
        _TRAJECTORIES[name] = {"active": act, "irreps": sym.orbital_irreps(act), "e0": e0, "ci0": ci0,
                               "steps": steps, "states": states,
                               "d": [s.generator for s in steps], "eps": [s.eps for s in steps]}
    return _TRAJECTORIES[name]


def _bits(n_add: int, delta: float = 1e-3, b_min: int = 4) -> int:
    """F2 by enumeration: the smallest b >= b_min with n_add pi / 2^b <= delta."""
    return next(b for b in range(b_min, 200) if n_add * math.pi / 2.0 ** b <= delta)


def _irrep_xor(exc: Excitation, irreps: np.ndarray, norb: int) -> int:
    x = 0
    for mode in exc.creators + exc.annihilators:
        x ^= int(irreps[mode % norb])
    return x


def literal_f3(d: np.ndarray, eps: float, irreps=None, delta: float = 1e-3) -> dict:
    """F3 as written: expand eps * mask(d) at tol 1e-12, theta = |amplitude|, and take the largest
    b <= F2(#all) with b = F2(#{theta >= pi / 2^b}) (the limit of the iteration of F3)."""
    g = d if irreps is None else sym.mask_pair_tensor(d, irreps)
    exc = spin_orbital_excitations(eps * g, tol=1e-12)
    theta = np.array([abs(e.amplitude) for e in exc])
    b = next(b for b in range(_bits(len(exc), delta), 3, -1)
             if _bits(int(np.sum(theta >= math.pi / 2.0 ** b)), delta) == b)
    kept = [e for e, t in zip(exc, theta) if t >= math.pi / 2.0 ** b]
    nd = sum(e.kind == "double" for e in kept)
    ns = sum(e.kind == "single" for e in kept)
    ng = sum(e.kind == "diagonal" for e in kept)
    return {"nd": nd, "ns": ns, "ng": ng, "b": b, "toffolis": len(kept) * (b - 2) + 3 * nd + 2 * ns + ng,
            "kept": kept, "theta": theta}


# ------------------------------------------------------------------------- regressions
@pytest.mark.parametrize("name", sorted(APPENDIX_A))
def test_appendix_a_elementwise_regressions(name):
    """Exact L-BFGS steps on the symmetry-adapted builds reproduce Appendix A: the trajectory (E - E_FCI
    to 1e-3 mHa, overlaps to 1e-4), the masked elementwise counts n_d / n_s, b, C_step and C(Psi_k).
    The table was computed for the design with its own expansion and fixed point [design]; the
    rounding bound stays within delta_step, and every step carries only the product-formula flag (it
    compiles more than one excitation)."""
    ref = APPENDIX_A[name]
    tr = trajectory(name)
    prep = sp.prep_cost(tr["d"][:len(ref)], tr["eps"][:len(ref)], irreps=tr["irreps"])
    assert prep.cumulative[0] == 0 and len(prep.steps) == len(ref)
    for k, (err, p, nd, ns, b, c, cum) in enumerate(ref):
        step = tr["steps"][k]
        assert abs(1e3 * (step.energy_after - tr["e0"]) - err) < 1e-3
        assert abs(abs(np.vdot(tr["ci0"].ravel(), tr["states"][k].ravel())) ** 2 - p) < 1e-4
        s = prep.steps[k]
        assert (s.n_double, s.n_single, s.n_diagonal, s.bits, s.toffolis) == (nd, ns, 0, b, c)
        assert prep.cumulative[k + 1] == cum
        assert s.logical_qubits == 2 * tr["active"].norb + 3 * b + 8
        assert s.flags == ("product-formula error not in budget",) and s.rounding_bound <= 1e-3
    if name == "h6a":
        assert prep.logical_qubits[6] == 83          # the H6 Psi_6 shot of Appendix A (83 qubits, 43,427)


@pytest.mark.parametrize("name", ["h4", "h6a", "n2", "lih"])
def test_elementwise_cost_equals_literal_f3_along_trajectories(name):
    """Over 12 exact L-BFGS steps, with and without the mask, the module's counts equal a literal
    implementation of F3 (eps inside the expansion, bits by enumeration, the fixed point by search);
    the kept excitations are those of the literal version, with the same amplitudes to 1e-12, in
    decreasing angle, all symmetry-allowed when masked, and the rounding bound is A pi / 2^b plus the
    dropped angles of the literal expansion."""
    tr = trajectory(name)
    n = tr["active"].norb
    for d, eps in zip(tr["d"], tr["eps"]):
        for irreps in (tr["irreps"], None):
            ref = literal_f3(d, eps, irreps)
            s = sp.elementwise_step_cost(d, eps, irreps=irreps)
            assert (s.n_double, s.n_single, s.n_diagonal, s.bits, s.toffolis) == \
                (ref["nd"], ref["ns"], ref["ng"], ref["b"], ref["toffolis"])
            kept, b = sp.elementwise_excitations(d, eps, irreps=irreps)
            mine = {(e.kind, e.creators, e.annihilators): e.amplitude for e in kept}
            theirs = {(e.kind, e.creators, e.annihilators): e.amplitude for e in ref["kept"]}
            assert b == ref["b"] and mine.keys() == theirs.keys()
            assert all(abs(mine[key] - theirs[key]) <= 1e-12 * abs(theirs[key]) for key in mine)
            assert all(abs(a.amplitude) >= abs(c.amplitude) for a, c in zip(kept, kept[1:]))
            dropped = ref["theta"][len(ref["kept"]):]
            assert s.n_dropped == dropped.size
            assert s.rounding_bound == pytest.approx(len(kept) * math.pi / 2.0 ** b + dropped.sum(), rel=1e-12)
            if irreps is not None:
                assert all(_irrep_xor(e, irreps, n) == 0 for e in kept)


def test_masking_keeps_h6_step1_and_removes_late_n2_roundoff():
    """The mask leaves the H6 step-1 count unchanged (no forbidden excitation survives the expansion
    there) and never costs more than no mask; on the late N2 L-BFGS steps, whose forbidden entries are
    amplified round-off (docs/m2-results.md, section 2), it removes excitations that the unmasked
    expansion would compile (the unmasked count at some step k >= 8 is higher, with n_masked > 0). The
    masked cost equals the cost of the explicitly masked direction without irreps."""
    tr = trajectory("h6a")
    d, eps, ir = tr["d"][1], tr["eps"][1], tr["irreps"]
    masked, plain = sp.elementwise_step_cost(d, eps, irreps=ir), sp.elementwise_step_cost(d, eps)
    assert (masked.n_double, masked.n_single, masked.toffolis) == (273, 120, 8526)
    assert (plain.n_double, plain.n_single, plain.toffolis) == (273, 120, 8526) and masked.n_masked == 0
    tr = trajectory("n2")
    ir = tr["irreps"]
    gains = []
    for k, (d, eps) in enumerate(zip(tr["d"], tr["eps"])):
        masked, plain = sp.elementwise_step_cost(d, eps, irreps=ir), sp.elementwise_step_cost(d, eps)
        explicit = sp.elementwise_step_cost(sym.mask_pair_tensor(d, ir), eps)
        assert (masked.toffolis, masked.bits) == (explicit.toffolis, explicit.bits)
        assert plain.toffolis >= masked.toffolis
        if k >= 8:
            gains.append((plain.toffolis - masked.toffolis, masked.n_masked))
    assert any(gain > 0 and n_masked > 0 for gain, n_masked in gains)


def test_prep_cost_and_candidates_mask_the_late_n2_steps():
    """On the 12-step N2 L-BFGS trajectory, whose late directions carry amplified round-off in the
    forbidden entries (docs/m2-results.md, section 2): prep_cost with and without irreps is the running
    sum of the masked and of the unmasked step costs; the masked cumulative is never higher, and lower
    after the first step whose forbidden excitations survive the rounding (k = 9; C(Psi_12) = 11082
    against 12854 Toffolis when this test was written, one thread). candidate_step_toffolis with irreps
    equals the masked step cost of every candidate and is below the unmasked one for some candidate of
    each of those steps. Q(Psi_k) is the running maximum of the step qubits, which differs from the last
    step's at step 10 (b = 17: 71 qubits, after 74). Without the mask (n_masked = 0) and at eps = 0
    nothing is counted as masked or dropped."""
    tr = trajectory("n2")
    ir, d, e = tr["irreps"], tr["d"], tr["eps"]
    masked, plain = sp.prep_cost(d, e, irreps=ir), sp.prep_cost(d, e)
    m_steps = [sp.elementwise_step_cost(x, y, irreps=ir, step=k) for k, (x, y) in enumerate(zip(d, e))]
    p_steps = [sp.elementwise_step_cost(x, y, step=k) for k, (x, y) in enumerate(zip(d, e))]
    assert masked.steps == tuple(m_steps) and plain.steps == tuple(p_steps)
    assert list(masked.cumulative) == [0] + list(np.cumsum([s.toffolis for s in m_steps]))
    assert list(plain.cumulative) == [0] + list(np.cumsum([s.toffolis for s in p_steps]))
    assert np.all(masked.cumulative <= plain.cumulative)
    first = next(k for k, (m, p) in enumerate(zip(m_steps, p_steps)) if p.toffolis > m.toffolis)
    assert 8 <= first < len(d) - 1 and m_steps[first].n_masked > 0 == p_steps[first].n_masked
    assert np.all(masked.cumulative[first + 1:] < plain.cumulative[first + 1:])
    cands = FTParameters().eps_candidates
    for k in range(first, len(d)):
        cm, cp = (sp.candidate_step_toffolis(d[k], cands, irreps=ir), sp.candidate_step_toffolis(d[k], cands))
        assert list(cm) == [sp.elementwise_step_cost(d[k], c, irreps=ir).toffolis for c in cands]
        assert list(cp) == [sp.elementwise_step_cost(d[k], c).toffolis for c in cands]
        assert np.all(cm <= cp) and np.any(cm < cp)
    q = [2 * tr["active"].norb] + [s.logical_qubits for s in m_steps]
    assert list(masked.logical_qubits) == [max(q[:k + 1]) for k in range(len(q))]
    assert any(s.logical_qubits < masked.logical_qubits[k] for k, s in enumerate(m_steps))
    zero = sp.elementwise_step_cost(d[first], 0.0, irreps=ir)
    assert (zero.toffolis, zero.n_dropped, zero.n_masked, zero.flags) == (0, 0, 0, ())
    assert zero.logical_qubits == 2 * tr["active"].norb


def test_flags_mark_the_product_and_rounding_caveats():
    """"product-formula error not in budget" is set iff two or more excitations are kept (one exponential
    is exact: the pair excitation e_{20,20} - e_{02,02}, one spin-orbital double by
    spin_orbital_excitations, carries no flag), and "rounding bound above delta_step" iff A pi / 2^b plus
    the dropped angles exceeds delta_step. On the H4 step-1 direction the second fires at eps = 0.004,
    where the fixed point of F3 drops all 64 excitations: the step compiles to the identity, and the
    bound is the sum of their angles eps sum |t_x| (from the expansion of the masked direction here)."""
    pair = np.zeros((4,) * 4)
    pair[2, 0, 2, 0], pair[0, 2, 0, 2] = 0.3, -0.3
    assert len(spin_orbital_excitations(pair, tol=1e-12)) == 1
    one = sp.elementwise_step_cost(pair, 0.5)
    assert (one.n_double, one.toffolis, one.flags) == (1, 13, ())
    tr = trajectory("h4")
    d, ir = tr["d"][1], tr["irreps"]
    seen = set()
    for eps in np.geomspace(1e-3, 1.0, 31):
        s = sp.elementwise_step_cost(d, float(eps), irreps=ir)
        n_keep = s.n_double + s.n_single + s.n_diagonal
        expected = ((("product-formula error not in budget",) if n_keep >= 2 else ())
                    + (("rounding bound above delta_step",) if s.rounding_bound > 1e-3 else ()))
        assert s.flags == expected
        seen.update(s.flags)
    assert seen == {"product-formula error not in budget", "rounding bound above delta_step"}
    collapsed = sp.elementwise_step_cost(d, 0.004, irreps=ir)
    expansion = spin_orbital_excitations(sym.mask_pair_tensor(d, ir), tol=1e-12)
    angles = 0.004 * sum(abs(x.amplitude) for x in expansion)
    assert (collapsed.toffolis, collapsed.n_dropped) == (0, len(expansion)) == (0, 64)
    assert collapsed.flags == ("rounding bound above delta_step",)
    assert collapsed.rounding_bound == pytest.approx(angles, rel=1e-12) and angles > 1e-3


def test_mask_removes_injected_forbidden_components():
    """Deterministic version of the late-N2 effect: a forbidden component (pair-symmetric,
    anti-Hermitian, 1e-3 of max|d|) added to the H6 step-1 direction raises the unmasked cost, while
    the masked cost is that of the clean direction and n_masked counts the forbidden excitations of
    the perturbed expansion (checked with the irrep product computed here)."""
    tr = trajectory("h6a")
    d, eps, ir = tr["d"][1], tr["eps"][1], tr["irreps"]
    n = d.shape[0]
    rng = np.random.default_rng(3)
    x = rng.standard_normal(d.shape)
    x = 0.5 * (x + x.transpose(2, 3, 0, 1))
    x = 0.5 * (x - x.transpose(1, 0, 3, 2))
    x = np.where(sym.pair_mask(ir), 0.0, x)
    x *= 1e-3 * np.abs(d).max() / np.abs(x).max()
    clean = sp.elementwise_step_cost(d, eps, irreps=ir)
    noisy_masked = sp.elementwise_step_cost(d + x, eps, irreps=ir)
    noisy_plain = sp.elementwise_step_cost(d + x, eps)
    assert (noisy_masked.toffolis, noisy_masked.bits, noisy_masked.n_double, noisy_masked.n_single) == \
        (clean.toffolis, clean.bits, clean.n_double, clean.n_single)
    assert noisy_plain.toffolis > clean.toffolis
    forbidden = [e for e in spin_orbital_excitations(d + x, tol=1e-12) if _irrep_xor(e, ir, n) != 0]
    assert noisy_masked.n_masked == len(forbidden) > 0


def test_complex_directions_raise_and_real_complex_dtype_is_accepted():
    """F3 excludes complex directions (ValueError); a complex array with zero imaginary part is the
    real direction (same cost)."""
    tr = trajectory("h4")
    d, eps = tr["d"][0], tr["eps"][0]
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sp.elementwise_step_cost(d + 1e-3j * rng.standard_normal(d.shape), eps)
    with pytest.raises(ValueError):
        sp.candidate_step_toffolis(d.astype(complex) + 1e-9j, [0.1])
    assert sp.elementwise_step_cost(d.astype(complex), eps) == sp.elementwise_step_cost(d, eps)
    with pytest.raises(ValueError):
        sp.elementwise_step_cost(d[0], eps)
    with pytest.raises(ValueError):
        sp.elementwise_step_cost(d, float("nan"))
    for bad in (0.1 + 0j, np.complex128(0.1), "0.1", None):              # ValueError, not TypeError
        with pytest.raises(ValueError):
            sp.elementwise_step_cost(d, bad)
    with pytest.raises(ValueError):
        sp.leaf_step_cost(d, 0.1 + 0.2j)
    with pytest.raises(ValueError):
        sp.candidate_step_toffolis(d, [0.1, 0.2j])


# ------------------------------------------------------------------------------- leaves
def literal_f6(d: np.ndarray, delta: float = 1e-3) -> dict:
    """F6 as written, from factorize_generator(d, rank_tol=1e-10) and the eigenvalues of each Q_k."""
    n = d.shape[0]
    fac = factorize_generator(d, rank_tol=1e-10)
    na = []
    for q in fac.Q:
        mu = np.abs(np.linalg.eigvalsh(q))
        na.append(int(np.sum(mu > 1e-6 * mu.max())))
    L = fac.n_leaves
    a_leaf = L * n * (n - 1) + sum(x * (x + 1) // 2 for x in na)
    and_leaf = 7 * L * n * (n - 1) // 2 + sum(x * (2 * x - 1) for x in na)
    b = _bits(a_leaf, delta)
    return {"L": L, "n_active": tuple(na), "b": b, "toffolis": a_leaf * (b - 2) + and_leaf}


@pytest.mark.parametrize("name", ["h4", "lih", "n2"])
def test_leaf_step_cost_matches_f6_and_the_rhf_rank(name):
    """leaf_step_cost equals F6 evaluated here from the generator leaves, for the RHF step and two
    later L-BFGS steps. The RHF step has K_1 = 2 rank W leaves, W[(ia),(jb)] = 4(ia|jb) - 2(ib|ja)
    from the ERIs (M1; for N2 the exchange term raises the rank above 2 r_ov). Every leaf step is
    flagged complex and Trotter-unbudgeted, and Q = 2N + 3b + 8."""
    tr = trajectory(name)
    act = tr["active"]
    n, no = act.norb, act.nelec[0]
    for d, eps in zip(tr["d"][:3], tr["eps"][:3]):
        ref = literal_f6(d)
        s = sp.leaf_step_cost(d, eps)
        assert (s.n_leaves, s.n_active, s.bits, s.toffolis) == (ref["L"], ref["n_active"], ref["b"],
                                                                  ref["toffolis"])
        assert s.method == "leaves" and s.flags == ("complex state", "order-1 Trotter error not in budget")
        assert s.logical_qubits == 2 * n + 3 * s.bits + 8
        # The first steps carry no forbidden component above round-off, so the mask changes nothing.
        assert sp.leaf_step_cost(d, eps, irreps=tr["irreps"]).toffolis == s.toffolis
    blk = act.eri[:no, no:, :no, no:]
    w = 4.0 * blk.reshape(no * (n - no), -1) - 2.0 * np.einsum("ibja->iajb", blk).reshape(no * (n - no), -1)
    sv = np.linalg.svd(w, compute_uv=False)
    rank_w = int(np.sum(sv > 1e-8 * sv[0]))
    assert sp.leaf_step_cost(tr["d"][0], tr["eps"][0]).n_leaves == 2 * rank_w
    assert {"h4": 8, "lih": 16, "n2": 18}[name] == 2 * rank_w                  # M1 results, section 3


def test_leaf_step_cost_options():
    """n_leaves keeps the top leaves (F6 with L = n_leaves); eps = 0 compiles nothing; a complex
    direction is accepted (leaf circuits are complex anyway)."""
    tr = trajectory("h4")
    d, eps = tr["d"][1], tr["eps"][1]
    full = sp.leaf_step_cost(d, eps)
    four = sp.leaf_step_cost(d, eps, n_leaves=4)
    assert four.n_leaves == 4 and four.n_active == full.n_active[:4] and four.toffolis < full.toffolis
    assert sp.leaf_step_cost(d, eps, n_leaves=100).toffolis == full.toffolis
    empty = sp.leaf_step_cost(d, 0.0)
    assert (empty.n_leaves, empty.toffolis, empty.logical_qubits, empty.flags) == (0, 0, 8, ())
    rng = np.random.default_rng(1)
    z = rng.standard_normal(d.shape) + 1j * rng.standard_normal(d.shape)
    z = 0.5 * (z + z.transpose(2, 3, 0, 1))
    z = 0.5 * (z - np.conj(z.transpose(1, 0, 3, 2)))
    assert sp.leaf_step_cost(z, 0.1).n_leaves > 0


# ------------------------------------------------------------------------- cumulative
def test_prep_cost_cumulative_sums_and_qubits():
    """C(Psi_0) = 0 and C(Psi_k) is the running sum of the step costs; Q(Psi_k) is the running maximum of
    the step qubits, 2N at k = 0, also when a later step needs fewer qubits (the H6 steps 1 and 0 in that
    order: 83, then 74). A leaf circuit adds its final complex rotation, at its own budget
    b_f = F2(N(N-1)) (555 Toffolis at N = 6, b_f = 17), to every k >= 1. Steps with eps = 0 cost nothing
    and count nothing as dropped."""
    tr = trajectory("h6a")
    d, eps = tr["d"][:6], tr["eps"][:6]
    for method in sp.METHODS:
        prep = sp.prep_cost(d, eps, method=method, irreps=tr["irreps"])
        tof = [s.toffolis for s in prep.steps]
        extra = prep.final_rotation * (np.arange(len(tof) + 1) > 0)
        np.testing.assert_array_equal(prep.cumulative, np.concatenate([[0], np.cumsum(tof)]) + extra)
        q = [12] + [s.logical_qubits for s in prep.steps]
        assert list(prep.logical_qubits) == [max(q[:k + 1]) for k in range(len(q))]
        assert prep.cumulative[0] == 0 and prep.logical_qubits[0] == 12
        with pytest.raises(ValueError):
            prep.cumulative[1] = 0                                        # read-only
    swapped = sp.prep_cost(d[1::-1], eps[1::-1], irreps=tr["irreps"])
    assert [s.logical_qubits for s in swapped.steps] == [83, 74]
    assert list(swapped.logical_qubits) == [12, 83, 83]
    leaves = sp.prep_cost(d, eps, method="leaves")
    b_f = _bits(30)
    assert b_f == 17 and leaves.final_rotation == 30 * (b_f - 2) + 7 * 15 == 555
    zero = sp.prep_cost(tr["d"][:2], [0.0, 0.0], method="leaves")
    assert list(zero.cumulative) == [0, 0, 0] and list(zero.logical_qubits) == [12, 12, 12]
    zero = sp.prep_cost(tr["d"][:2], [0.0, tr["eps"][1]], irreps=tr["irreps"])
    assert zero.steps[0].toffolis == 0 and zero.cumulative[2] == sp.elementwise_step_cost(
        tr["d"][1], tr["eps"][1], irreps=tr["irreps"]).toffolis
    assert (zero.steps[0].n_dropped, zero.steps[0].n_masked, zero.steps[0].flags) == (0, 0, ())
    with pytest.raises(ValueError):
        sp.prep_cost(d, eps, method="qsp")
    with pytest.raises(ValueError):
        sp.prep_cost(d, eps[:-1])
    with pytest.raises(ValueError):
        sp.prep_cost([], [])


def test_candidate_step_toffolis_matches_step_costs_and_is_monotone():
    """The line-search candidates 0.00625 * 2^j cost elementwise_step_cost(d, eps_c) each, and the cost
    is non-decreasing in |eps|: a longer step keeps every excitation a shorter one keeps, at no fewer
    bits (the fixed point of F3 is monotone in the angles)."""
    tr = trajectory("h6a")
    d, ir = tr["d"][1], tr["irreps"]
    cands = FTParameters().eps_candidates
    costs = sp.candidate_step_toffolis(d, cands, irreps=ir)
    assert costs.dtype == np.int64 and costs.shape == (len(cands),)
    assert list(costs) == [sp.elementwise_step_cost(d, e, irreps=ir).toffolis for e in cands]
    assert np.all(np.diff(costs) >= 0) and costs[0] < costs[-1]
    assert list(sp.candidate_step_toffolis(d, [-e for e in cands], irreps=ir)) == list(costs)


def test_candidate_step_costs_give_qubits_and_flags():
    """candidate_step_costs equals elementwise_step_cost for every candidate (the line search needs its
    qubits and flags, not only the Toffolis). Against literal F3 on the unmasked H6 L-BFGS step 3: the
    kept set and b give Q = 2N + 3b + 8 (2N when nothing is kept), and the flag "rounding bound above
    delta_step" is set exactly where n_kept pi / 2^b plus the dropped angles exceed delta_step. The
    smallest candidate there keeps nothing and is flagged (observed bound 1.73e-3)."""
    tr = trajectory("h6a")
    d, ir = tr["d"][3], tr["irreps"]
    cands = FTParameters().eps_candidates
    costs = sp.candidate_step_costs(d, cands, irreps=ir)
    assert costs == tuple(sp.elementwise_step_cost(d, e, irreps=ir) for e in cands)
    assert [c.toffolis for c in costs] == list(sp.candidate_step_toffolis(d, cands, irreps=ir))
    n = d.shape[0]
    for e, c in zip(cands, costs):
        ref = literal_f3(d, e, ir)
        kept = len(ref["kept"])
        assert c.logical_qubits == (2 * n + 3 * ref["b"] + 8 if kept else 2 * n)
        theta = np.sort(ref["theta"])[::-1]
        bound = kept * math.pi / 2.0 ** ref["b"] + float(np.sum(theta[kept:]))
        assert ("rounding bound above delta_step" in c.flags) == (bound > 1e-3)
    assert costs[0].toffolis == 0 and "rounding bound above delta_step" in costs[0].flags
    assert costs[0].rounding_bound == pytest.approx(1.73e-3, abs=5e-6)


def test_candidate_step_costs_use_the_given_budget():
    """At delta_step = 1e-4 (not the default 1e-3) every line-search candidate follows literal F3 at that budget
    (bits by enumeration, the fixed point by search): the same kept doubles and singles and Toffolis. The tighter
    budget raises the bits of the largest candidate by about log2(10) and its cost (F2)."""
    tr = trajectory("h4")
    d, ir = tr["d"][1], tr["irreps"]
    tight = FTParameters(delta_step=1e-4)
    cands = tight.eps_candidates
    got = sp.candidate_step_costs(d, cands, irreps=ir, params=tight)
    for e, c in zip(cands, got):
        ref = literal_f3(d, e, ir, delta=1e-4)
        assert (c.n_double, c.n_single, c.toffolis) == (ref["nd"], ref["ns"], ref["toffolis"])
        if ref["kept"]:
            assert c.bits == ref["b"]
    loose = sp.candidate_step_costs(d, cands, irreps=ir)
    assert got[-1].bits - loose[-1].bits in (3, 4) and got[-1].toffolis > loose[-1].toffolis


# ------------------------------------------------------------------------------- counts
def test_primitive_count_rules():
    """The per-kind rules of F3 (double 1 + 3, single 1 + 2, diagonal 1 + 1 additions + ANDs; 8 / 4 / 3
    Pauli rotations), F4 (real g + 6g, complex 2g + 7g, g = N(N-1)/2) and F5
    (n(n-1)/2 + n additions, 2n(n-1) + n ANDs), and toffolis(b) = n_add (b - 2) + n_and."""
    excs = [Excitation("double", (5, 6), (1, 2), 0.1), Excitation("single", (2, 6), (2, 3), 0.2),
            Excitation("double", (4, 7), (0, 3), -0.1), Excitation("diagonal", (1, 5), (1, 5), 0.3j)]
    c = sp.excitation_counts(excs)
    assert (c.n_add, c.n_and, c.n_pauli_rot) == (4, 3 + 2 + 3 + 1, 8 + 4 + 8 + 3)
    assert c.toffolis(10) == 4 * 8 + 9 and sp.excitation_counts([]) == sp.PrimitiveCounts()
    assert sp.PrimitiveCounts(0, 5).toffolis(0) == 5
    with pytest.raises(ValueError):
        sp.PrimitiveCounts(1, 0).toffolis(2)
    for n in range(1, 9):
        g = n * (n - 1) // 2
        assert sp.orbital_rotation_counts(n, complex_=False) == sp.PrimitiveCounts(g, 6 * g, 4 * g)
        assert sp.orbital_rotation_counts(n, complex_=True) == sp.PrimitiveCounts(2 * g, 7 * g, 6 * g)
        assert sp.diagonal_layer_counts(n) == sp.PrimitiveCounts(n * (n - 1) // 2 + n, 2 * n * (n - 1) + n,
                                                                 n * (2 * n + 1))
    assert sp.PrimitiveCounts(1, 2, 3) + sp.PrimitiveCounts(4, 5, 6) == sp.PrimitiveCounts(5, 7, 9)
    with pytest.raises(ValueError):
        sp.diagonal_layer_counts(-1)


def test_readout_rotation_toffolis():
    """C_rot = g (b_r - 2) + 6 g with b_r = F2(g, delta_readout = 1e-5): 405 at N = 6 [design],
    and the formula with b_r by enumeration for N = 2..8 and another budget."""
    assert sp.readout_rotation_toffolis(6) == 405
    for n in range(2, 9):
        g = n * (n - 1) // 2
        assert sp.readout_rotation_toffolis(n) == g * (_bits(g, 1e-5) - 2) + 6 * g
        p = FTParameters(delta_readout=1e-3)
        assert sp.readout_rotation_toffolis(n, params=p) == g * (_bits(g, 1e-3) - 2) + 6 * g
    assert sp.readout_rotation_toffolis(1) == 0


def test_readout_rotation_qubits():
    """Q_rot = 2N + 3 b_r + 8 with b_r = F2(g, delta_readout) by enumeration: 89 at N = 6 (b_r = 23), the
    formula for N = 2..8 and another budget, and 2N for N = 1 (no rotation)."""
    assert sp.readout_rotation_qubits(6) == 89
    for n in range(2, 9):
        g = n * (n - 1) // 2
        assert sp.readout_rotation_qubits(n) == 2 * n + 3 * _bits(g, 1e-5) + 8
        p = FTParameters(delta_readout=1e-3)
        assert sp.readout_rotation_qubits(n, params=p) == 2 * n + 3 * _bits(g, 1e-3) + 8
    assert sp.readout_rotation_qubits(1) == 2
    with pytest.raises(ValueError):
        sp.readout_rotation_qubits(-1)

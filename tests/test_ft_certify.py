"""Tests of ft.certify: sector levels, truncated-CI bounds, contaminant weights and energy certificates (F41, F42).

References: dense solves of P H P (the sector Hamiltonian of ``fci_ops`` built column by column),
PySCF FCI with and without the singlet constraint and the irreps of its roots, the singlet
projector applied with ``pyscf.fci.spin_op.contract_ss``, ``ft.residual.energy_sampling`` and
``energy_ae`` and the closed forms of F42, and reference numbers of the M4 cost model [build
check] with their tolerances. Systems (``ft.symmetry.benchmark_system``): M3's h4, n2s
(= n2_r2.0) and h6s (= h6_r3.0), n2_r2.4 and n2_r2.8, with the selected SF encodings pinned as
in tests/test_ft_acceptance.py; "the exact H" is the model spectrum of H itself (the selected
encoding with the exact integrals as its model).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from pyscf.fci import spin_op

from thc_engine_cqe.diagnostics.thc_floors import ExactReference
from thc_engine_cqe.fci_ops import Hamiltonian, normalize
from thc_engine_cqe.ft import acceptance as acc
from thc_engine_cqe.ft import certify
from thc_engine_cqe.ft import qpe as ftqpe
from thc_engine_cqe.ft import stateprep, symmetry
from thc_engine_cqe.ft.circuits import replay_elementwise
from thc_engine_cqe.ft.residual import PrepParams, energy_ae, energy_sampling
from thc_engine_cqe.ft.trajectory import exact_trajectory

DE = 1e-3
# SF ranks: the selected encodings (F9) [build check]; h6s (selected: THC) takes the full rank 18, i.e. H itself
_RANK = {"h4": 7, "n2s": 17, "n2_r2.4": 16, "n2_r2.8": 13, "h6s": 18, "n2": 20}
_CACHE: dict = {}


def _system(name: str) -> dict:
    """active, irreps, the selected SF QPE (h6s: the full-rank SF encoding) and its model H'."""
    if name not in _CACHE:
        a = symmetry.benchmark_system(name)
        ir = symmetry.orbital_irreps(a)
        enc = ftqpe.sf_encoding(a, _RANK[name], reference=ExactReference.build(a))
        q = ftqpe.qpe_cost(enc, DE)
        _CACHE[name] = {"active": a, "irreps": ir, "qpe": q, "model": enc.model(a)}
    return _CACHE[name]


def _spectrum(name: str, exact: bool) -> acc.ModelSpectrum:
    """The model spectrum of H' (selected encoding) or of the exact H."""
    key = ("spec", name, exact)
    if key not in _CACHE:
        s = _system(name)
        q = s["qpe"]
        if exact:
            q = dataclasses.replace(q, encoding=dataclasses.replace(q.encoding, eri_model=np.array(s["active"].eri)))
        _CACHE[key] = acc.model_spectrum(s["active"], q, s["irreps"])
    return _CACHE[key]


def _dense(active) -> np.ndarray:
    """Dense sector Hamiltonian (ecore included), column by column from fci_ops."""
    ham = Hamiltonian(active)
    sp = ham.space
    h = np.column_stack([ham.apply(np.eye(sp.size)[i].reshape(sp.shape)).ravel() for i in range(sp.size)])
    return 0.5 * (h + h.T) + active.ecore * np.eye(sp.size)


def _dense_truncated(active, level: int) -> tuple[float, np.ndarray]:
    """Lowest eigenpair of P H P by a dense solve; the eigenvector embedded in the full CI array."""
    h = _dense(active)
    sel = certify.excitation_levels(active.norb, active.nelec).ravel() <= level
    e, v = np.linalg.eigh(h[np.ix_(sel, sel)])
    x = np.zeros(h.shape[0])
    x[sel] = v[:, 0]
    return float(e[0]), x


def _s2(active, ci) -> float:
    shape = Hamiltonian(active).space.shape
    return float(spin_op.spin_square0(np.asarray(ci).reshape(shape), active.norb, active.nelec)[0])


# ------------------------------------------------------------------ truncated CI (F42)
def test_excitation_levels():
    """Levels count the electrons outside the HF occupation; HF is address 0; the counts match binomials."""
    lev = certify.excitation_levels(6, (3, 3))
    assert lev.shape == (20, 20) and lev[0, 0] == 0
    counts = np.bincount(lev.ravel())
    # level l: sum_{a + b = l} C(3, a)^2 C(3, b)^2 (a, b electrons promoted per spin)
    ref = [sum(math.comb(3, x) ** 2 * math.comb(3, l - x) ** 2 for x in range(max(0, l - 3), min(3, l) + 1))
           for l in range(7)]
    assert counts.tolist() == ref
    assert int(np.sum(lev <= 2)) == 118 and int(np.sum(lev <= 3)) == 282
    assert certify.excitation_levels(8, (5, 5)).shape == (56, 56)
    with pytest.raises(ValueError):
        certify.excitation_levels(4, (5, 1))


@pytest.mark.parametrize("name", ["h4", "n2s", "h6s", "n2_r2.4"])
def test_lanczos_equals_dense_solve(name):
    """The projected Lanczos equals a dense solve of P H P at N <= 6 to 1e-9 Ha (levels 0..3, H and H')."""
    s = _system(name)
    for active in (s["active"], s["model"]):
        for level in (0, 1, 2, 3):
            ref, _ = _dense_truncated(active, level)
            assert abs(certify.truncated_ci_energy(active, level) - ref) <= 1e-9
        assert abs(certify.truncated_ci_energy(active, 0) - _dense(active)[0, 0]) <= 1e-12   # E_HF


def test_truncated_ci_is_variational_before_convergence():
    """A loose residual tolerance returns a Rayleigh quotient of the P space: never below the converged value.

    (Within tol of some eigenvalue of P H P, not necessarily the lowest: at tol = 1e-2 the n2_r2.4 run stops
    38 mHa above the CISD minimum.)
    """
    a = _system("n2_r2.4")["active"]
    ref = certify.truncated_ci_energy(a, 2)
    values = [certify.truncated_ci_energy(a, 2, tol=tol) for tol in (1e-1, 1e-2, 1e-4, 1e-8)]
    assert all(e >= ref - 1e-12 for e in values)
    assert abs(values[-1] - ref) < 1e-10


def test_cisd_values():
    """CISD(H) - E_0 = 133.608 / 224.810 / 292.418 mHa on n2_r2.0 / 2.4 / 2.8, CISD(H') - E_0' = 225.686 mHa on
    n2_r2.4 (1e-3 mHa) [build check]; E_FCI <= E_CISDT <= E_CISD <= E_HF.

    The CISD minimum of stretched N2 is a quintet and the CISDT minimum the septet (S^2 = 6, 12): the level
    projector commutes with S^2, and the spin-adapted singlet CISD is higher (on n2_r2.4 above E_1').
    """
    refs = {"n2s": 133.608, "n2_r2.4": 224.810, "n2_r2.8": 292.418}
    for name, ref in refs.items():
        a = _system(name)["active"]
        e_fci = float(a.fci(spin_square=None)[0])
        assert abs(e_fci - float(a.fci()[0])) < 1e-8            # the sector minimum is the singlet ground state
        e_hf = float(_dense(a)[0, 0])
        cisd = certify.truncated_ci_energy(a, 2)
        cisdt = certify.truncated_ci_energy(a, 3)
        assert abs(1e3 * (cisd - e_fci) - ref) < 1e-3
        assert e_fci <= cisdt <= cisd <= e_hf
        _, x2 = _dense_truncated(a, 2)
        _, x3 = _dense_truncated(a, 3)
        assert abs(_s2(a, x2) - 6.0) < 1e-8 and abs(_s2(a, x3) - 12.0) < 1e-8
    # H' of n2_r2.4 (SF L = 16): CISD(H') - E_0' = 225.686, the singlet CISD 263.969 above E_1' - E_0' = 230.863
    spec = _spectrum("n2_r2.4", exact=False)
    model = _system("n2_r2.4")["model"]
    assert abs(1e3 * (certify.truncated_ci_energy(model, 2) - spec.e0) - 225.686) < 1e-3
    h = _dense(model)
    sel = certify.excitation_levels(model.norb, model.nelec).ravel() <= 2
    e, v = np.linalg.eigh(h[np.ix_(sel, sel)])
    singlet = []
    for t in range(e.size):
        x = np.zeros(h.shape[0])
        x[sel] = v[:, t]
        if _s2(model, x) < 1e-6:
            singlet.append(float(e[t]))
    assert abs(1e3 * (singlet[0] - spec.e0) - 263.969) < 2e-3
    assert singlet[0] > spec.e1_singlet > certify.truncated_ci_energy(model, 2)


def _lowest_singlet(active, level: int) -> float:
    """Lowest eigenvalue of P H P with a singlet eigenvector (dense)."""
    h = _dense(active)
    sel = certify.excitation_levels(active.norb, active.nelec).ravel() <= level
    e, v = np.linalg.eigh(h[np.ix_(sel, sel)])
    for t in range(e.size):
        x = np.zeros(h.shape[0])
        x[sel] = v[:, t]
        if _s2(active, x) < 1e-6:
            return float(e[t])
    raise AssertionError("no singlet")


def test_truncated_ci_table():
    """The table of the module docstring (mHa above E_0' of H', and of H; 2e-3 mHa) [build check]: CISD, the
    spin-adapted singlet CISD and CISDT of stretched N2 (quintet and septet minima), and h6s (singlets)."""
    table = {"n2s": ((134.215, 133.608), (159.641, 159.610), (54.752, 54.135)),
             "n2_r2.4": ((225.686, 224.810), (263.969, 263.394), (9.241, 7.494)),
             "n2_r2.8": ((293.495, 292.418), (331.066, 330.576), (2.455, 0.535))}
    for name, (cisd, singlet, cisdt) in table.items():
        s = _system(name)
        for i, active in enumerate((s["model"], s["active"])):
            e0 = float(np.linalg.eigvalsh(_dense(active))[0])
            assert abs(1e3 * (certify.truncated_ci_energy(active, 2) - e0) - cisd[i]) < 2e-3
            assert abs(1e3 * (_lowest_singlet(active, 2) - e0) - singlet[i]) < 2e-3
            assert abs(1e3 * (certify.truncated_ci_energy(active, 3) - e0) - cisdt[i]) < 2e-3
    a = _system("h6s")["active"]
    e0 = float(a.fci()[0])
    assert abs(1e3 * (certify.truncated_ci_energy(a, 2) - e0) - 54.506) < 2e-3
    assert abs(1e3 * (certify.truncated_ci_energy(a, 3) - e0) - 51.399) < 2e-3
    for level in (2, 3):
        _, x = _dense_truncated(a, level)
        assert _s2(a, x) < 1e-8


@pytest.mark.slow
def test_cisd_at_n10():
    """h10_r3.0 (63,504 determinants, 876 at level <= 2): E_FCI <= E_CISDT <= E_CISD <= E_HF by Lanczos."""
    a = symmetry.benchmark_system("h10_r3.0")
    cisd = certify.truncated_ci_energy(a, 2)
    cisdt = certify.truncated_ci_energy(a, 3)
    e_hf = Hamiltonian(a).energy(Hamiltonian(a).space.hf())
    e_fci = float(a.fci()[0])
    assert e_fci <= cisdt <= cisd <= e_hf
    assert int(np.sum(certify.excitation_levels(a.norb, a.nelec) <= 2)) == 876


def test_truncated_ci_validation():
    a = _system("h4")["active"]
    for bad in (-1, 2.0, True):
        with pytest.raises(ValueError):
            certify.truncated_ci_energy(a, bad)
    with pytest.raises(ValueError):
        certify.truncated_ci_energy(a, 2, tol=0.0)
    with pytest.raises(ValueError):
        certify.truncated_ci_energy("h4", 2)


# ------------------------------------------------------------------ sector levels (F41)
def _irrep_of(active, irreps, ci) -> int:
    return symmetry.ci_irrep(ci, active.norb, active.nelec, irreps)[0]


@pytest.mark.parametrize("name,refs", [("n2s", (191.7, 20.9, 6.0)), ("h6s", (104.7, 88.6, 2.0))])
def test_sector_levels(name, refs):
    """Exact H: n2s same-irrep singlet 191.7 mHa and lowest same-irrep level of any spin 20.9 mHa (a quintet);
    h6s 104.7 / 88.6 mHa (a triplet) [design, to 0.05 mHa]; every level equals PySCF FCI to 1e-8 Ha.

    PySCF: the singlet roots (spin_square = 0) and the roots without the spin constraint, each
    with the irrep of its CI vector.
    """
    s = _system(name)
    a, ir = s["active"], s["irreps"]
    spec = _spectrum(name, exact=True)
    lv = certify.sector_levels(spec)
    e1, any_spin, s2_ref = refs
    assert abs(1e3 * lv["gap_e1_singlet"] - e1) < 0.05
    assert abs(1e3 * lv["gap_e1_any_spin"] - any_spin) < 0.05
    assert abs(lv["s2_e1_any_spin"] - s2_ref) < 1e-6 and lv["s2_e1_singlet"] < 1e-6
    assert lv["irrep_e1_singlet"] == lv["irrep_e1_any_spin"] == lv["irrep_e0"]
    es, cis = a.fci(nroots=12, spin_square=0.0)
    g0 = _irrep_of(a, ir, cis[0])
    assert abs(es[0] - lv["e0"]) < 1e-8 and g0 == lv["irrep_e0"]
    e1_ref = next(float(es[j]) for j in range(1, 12) if _irrep_of(a, ir, cis[j]) == g0)
    assert abs(e1_ref - lv["e1_singlet"]) < 1e-8
    assert abs(float(es[1]) - lv["e1_any_irrep"]) < 1e-8
    ea, cia = a.fci(nroots=12, spin_square=None)
    any_ref = next(float(ea[j]) for j in range(1, 12) if _irrep_of(a, ir, cia[j]) == g0)
    assert abs(any_ref - lv["e1_any_spin"]) < 1e-8
    assert abs(float(ea[1]) - lv["e1_any"]) < 1e-8
    assert lv["e1_any"] <= min(lv["e1_any_spin"], lv["e1_any_irrep"]) <= lv["e1_singlet"]


def test_sector_levels_any_irrep_n2():
    """Exact H of n2 (1.098 A), where the lowest excited singlet lies in another irrep: gap_e1_any_irrep 411.7 mHa
    and gap_e1_singlet 711.5 mHa (M3's 0.4117 / 0.7115 Ha [design], to 0.05 mHa); e1_any_irrep equals PySCF FCI
    singlet root 1 (1e-8 Ha), whose irrep differs from the ground state's (a degenerate pair: either member's irrep)."""
    s = _system("n2")
    a, ir = s["active"], s["irreps"]
    spec = _spectrum("n2", exact=True)
    lv = certify.sector_levels(spec)
    assert abs(1e3 * lv["gap_e1_any_irrep"] - 411.7) < 0.05
    assert abs(1e3 * lv["gap_e1_singlet"] - 711.5) < 0.05
    es, cis = a.fci(nroots=3, spin_square=0.0, conv_tol=1e-14)
    assert abs(float(es[0]) - lv["e0"]) < 1e-8 and abs(float(es[1]) - lv["e1_any_irrep"]) < 1e-8
    assert _irrep_of(a, ir, cis[1]) != _irrep_of(a, ir, cis[0]) == lv["irrep_e0"]
    assert lv["irrep_e1_any_irrep"] != lv["irrep_e0"] == lv["irrep_e1_singlet"]
    assert spec.e1_any_irrep == lv["e1_any_irrep"] < spec.e1_singlet - 0.2


def test_sector_levels_on_model_and_irrep_weights():
    """H' of n2s (SF L = 17): 191.907 mHa (E_1' - E_0') and a 21.395 mHa quintet; stretched N2 quintets at
    4.346 / 1.210 mHa (n2_r2.4 / n2_r2.8) [build check, 1e-3 mHa]; irrep weights from det_irreps; the shift
    E_1(H') - E_1(H) (0.01 mHa).

    The SF encodings of n2s and n2_r2.8 conserve D2h (impurity <= 2e-11); n2_r2.4's SF L = 16 mixes
    near-degenerate pairs of different irreps (impurity 5.5e-3, the ground state's 2.3e-8 [build check]), while
    HF's weight on eigenvectors of another dominant irrep stays below 1e-15 and the same-irrep and any-irrep
    singlet E_1 coincide.
    """
    refs = {"n2s": (191.907, 21.395), "n2_r2.4": (230.863, 4.346), "n2_r2.8": (244.441, 1.210)}
    for name, (e1, quintet) in refs.items():
        s = _system(name)
        spec = _spectrum(name, exact=False)
        det_ir = symmetry.determinant_irreps(s["active"].norb, s["active"].nelec, s["irreps"])
        lv = certify.sector_levels(spec, det_irreps=det_ir)
        assert abs(1e3 * lv["gap_e1_singlet"] - e1) < 1e-3
        assert abs(1e3 * lv["gap_e1_any_spin"] - quintet) < 1e-3 and abs(lv["s2_e1_any_spin"] - 6.0) < 1e-6
        assert lv["e1_any_irrep"] == lv["e1_singlet"]
        hf = np.zeros(spec.shape)
        hf[0, 0] = 1.0
        assert float(spec.weights(hf)[spec.irreps != spec.irreps[0]].sum()) < 1e-15
        if name == "n2_r2.4":
            assert 1e-3 < lv["irrep_impurity"] < 1e-2 and 1e-9 < 1.0 - lv["irrep_weight_e0"] < 1e-7
        else:
            assert lv["irrep_impurity"] < 1e-10 and 1.0 - lv["irrep_weight_e0"] < 1e-12
        # E_1(H') - E_1(H): +0.75, +0.56, -0.06 mHa (F41's 1-mHa assumption without the model spectrum)
        shift = 1e3 * (lv["e1_singlet"] - certify.sector_levels(_spectrum(name, exact=True))["e1_singlet"])
        assert abs(shift - {"n2s": -0.06, "n2_r2.4": 0.56, "n2_r2.8": 0.75}[name]) < 0.01
    with pytest.raises(ValueError):
        certify.sector_levels(spec, det_irreps=np.zeros(3, dtype=int))
    with pytest.raises(ValueError):
        certify.sector_levels("spectrum")


def test_e1_lower_bound():
    assert certify.e1_lower_bound(-1.0, model_spectrum=True) == pytest.approx(-1.0001, abs=1e-15)
    assert certify.e1_lower_bound(-1.0, model_spectrum=False) == pytest.approx(-1.001, abs=1e-15)
    with pytest.raises(ValueError):
        certify.e1_lower_bound(math.nan, model_spectrum=True)


# ------------------------------------------------------------------ contaminant weights (F41)
def _singlet_weight(active, ci) -> float:
    """<Psi|P_0|Psi>/<Psi|Psi>, P_0 = prod_{S'=1..3} (S^2 - S'(S'+1))/(-S'(S'+1)) by contract_ss (6 electrons)."""
    shape = Hamiltonian(active).space.shape
    x = np.asarray(ci, dtype=float).reshape(shape)
    y = x.copy()
    for s in (1, 2, 3):
        y = (np.asarray(spin_op.contract_ss(y, active.norb, active.nelec)) - s * (s + 1) * y) / (-s * (s + 1))
    return float(np.vdot(x, y) / np.vdot(x, x))


def test_contaminant_weight_compiled_n2s():
    """The compiled spin-orbital Psi~_1 of n2s (classical first step) has q_c/p_0 ~ 1.2e-4 below
    theta = CISD(H') + dE (20%) [design; build check 1.193e-4].

    Exact states carry no contaminant (HF and the exact Psi_1 to 1e-14); at theta above the spectrum q_c is
    the non-singlet weight 1 - <P_0>, P_0 the singlet projector applied by contract_ss (1e-10).
    """
    s = _system("n2s")
    a, ir = s["active"], s["irreps"]
    spec = _spectrum("n2s", exact=False)
    rec = exact_trajectory(a, optimizer="lbfgs", metric="spin_orbital", max_iter=1, irreps=ir,
                           first_step="second_order")
    compiled = replay_elementwise(a, rec.directions[:1], rec.eps[:1], irreps=rec.irreps)[1]
    theta = certify.truncated_ci_energy(s["model"], 2) + DE
    w = spec.weights(compiled)
    q_c = certify.contaminant_weight(spec, compiled, theta)
    assert q_c / w[0] == pytest.approx(1.2e-4, rel=0.2)
    top = float(spec.energies[-1]) + 1.0
    assert abs(certify.contaminant_weight(spec, compiled, top) - (1.0 - _singlet_weight(a, compiled))) < 1e-10
    assert certify.contaminant_weight(spec, compiled, spec.e0) == 0.0        # nothing below E_0'
    for state in (rec.states[0], rec.states[1]):
        assert certify.contaminant_weight(spec, state, top) < 1e-14
    # the identification error budget of F41 with the singlet E1lb
    assert q_c / w[0] < 1e-3
    with pytest.raises(ValueError):
        certify.contaminant_weight(spec, compiled, math.inf)
    with pytest.raises(ValueError):
        certify.contaminant_weight(spec, compiled, theta, s2_tol=-1.0)


# ------------------------------------------------------------------ energy certificates (F42)
def _h4_inputs() -> dict:
    key = "h4_certs"
    if key not in _CACHE:
        s = _system("h4")
        a, ir = s["active"], s["irreps"]
        rec = exact_trajectory(a, optimizer="lbfgs", metric="spin_free", max_iter=4, irreps=ir,
                               first_step="second_order")
        prep = stateprep.prep_cost(rec.directions, rec.eps, irreps=ir)
        _CACHE[key] = {"record": rec, "prep": prep, "qpe": s["qpe"], "active": a}
    return _CACHE[key]


def test_certificate_costs_follow_f42():
    """Sampled: S = (2z)^2 V_E/m^2 shots of C(Psi_j) + C_rot (= energy_sampling at sigma = m/(2z)); AE:
    N = 4 pi kappa lambda z/m applications of C(Psi_j) + C_W + Q_A/2 (= energy_ae); U = E_j + m/2,
    m = E1lb - 2 dE - E_j; only admissible U (<= E1lb - 2 dE); loop and oracle sources cost nothing.
    """
    d = _h4_inputs()
    rec, prep, q = d["record"], d["prep"], d["qpe"]
    enc = q.encoding
    e1lb = certify.e1_lower_bound(rec.e1, model_spectrum=False)
    thr = e1lb - 2 * DE
    rng = np.random.default_rng(5)
    var = rng.uniform(0.5, 2.0, len(rec.energies))
    var[2] = math.nan
    c_rot, z, kappa = 1234, 3.0, 1.0
    classical = {"cisd": certify.truncated_ci_energy(d["active"], 2), "too_high": thr + 1e-9}
    loop = {2: (float(rec.energies[2]), 1.6e-3), 3: (thr, 1.6e-3)}
    certs = certify.energy_certificates(rec, prep, e1_lower=e1lb, de=DE, classical=classical,
                                        energy_variances=var, readout_toffolis=c_rot, walk=enc,
                                        loop_energies=loop, z=z, kappa=kappa, include_oracle=True)
    by = {(c.source, c.j): c for c in certs}
    assert ("classical", None) in by and by[("classical", None)].label == "cisd"
    assert sum(c.source == "classical" for c in certs) == 1                 # "too_high" is not admissible
    walk = enc.walk_params()
    n = len(rec.energies)
    for j in range(n):
        m = thr - float(rec.energies[j])
        if m <= 0:
            assert ("sampled", j) not in by and ("ae", j) not in by
            continue
        sigma = m / (2 * z)
        assert by[("ae", j)].sigma == pytest.approx(sigma, rel=1e-14)       # the realized bound's spread
        if j != 2:
            c = by[("sampled", j)]
            ref = energy_sampling(var[j], prep.cumulative[j] + c_rot, int(prep.logical_qubits[j]), eps=sigma)
            assert c.sigma == pytest.approx(sigma, rel=1e-14)
            assert c.upper == pytest.approx(rec.energies[j] + m / 2, abs=1e-15)
            assert c.shots == pytest.approx((2 * z) ** 2 * var[j] / m ** 2, rel=1e-13)
            assert c.toffolis == pytest.approx(c.shots * (prep.cumulative[j] + c_rot), rel=1e-13)
            assert c.shots == ref.n_prep and c.toffolis == ref.toffolis and c.loop_iterate == j
        else:
            assert ("sampled", 2) not in by                                 # nan variance: skipped
        c = by[("ae", j)]
        q_a = int(prep.logical_qubits[j]) + walk.logical_qubits - 2 * rec.norb + 1
        n_app = 4 * math.pi * kappa * walk.lam * z / m
        ref = energy_ae(walk, PrepParams(prep.cumulative[j], int(prep.logical_qubits[j])), eps=sigma,
                        kappa=kappa, norb=rec.norb)
        assert c.applications == pytest.approx(n_app, rel=1e-13)
        assert c.toffolis == pytest.approx(n_app * (prep.cumulative[j] + walk.toffoli_per_step + q_a / 2), rel=1e-13)
        assert c.toffolis == ref.toffolis and c.applications == ref.n_prep
    lc = by[("loop_energy", 2)]
    assert lc.upper == pytest.approx(rec.energies[2] + z * 1.6e-3) and lc.toffolis == 0 and lc.loop_iterate == 3
    assert ("loop_energy", 3) not in by                                     # thr + z eps > thr
    oc = by[("oracle", None)]
    assert oc.upper == rec.e0 and oc.loop_iterate == 0
    assert lc.sigma is None and oc.sigma is None and by[("classical", None)].sigma is None
    # plain sequences with prep_qubits give the same certificates; AE without qubits is refused
    plain = certify.energy_certificates(rec, prep.cumulative, e1_lower=e1lb, de=DE, classical=classical,
                                        energy_variances=var, readout_toffolis=c_rot, walk=enc,
                                        loop_energies=loop, include_oracle=True,
                                        prep_qubits=prep.logical_qubits)
    assert plain == certs
    with pytest.raises(ValueError):
        certify.energy_certificates(rec, prep.cumulative, e1_lower=e1lb, de=DE, walk=enc)
    assert certify.energy_certificates(rec, prep, e1_lower=rec.e0, de=DE, classical=classical,
                                       energy_variances=var, walk=enc) == []


def test_energy_certificate_validation():
    ok = certify.EnergyCertificate("sampled", -1.0, 2, 10.0, 5.0, 0.0, "x")
    assert ok.loop_iterate == 2
    assert certify.EnergyCertificate("classical", -1.0, None, 0.0, 0.0, 0.0, "cisd").loop_iterate == 0
    bad = [("nope", -1.0, None, 0, 0, 0, ""), ("classical", -1.0, 0, 0, 0, 0, ""),
           ("classical", -1.0, None, 1.0, 0, 0, ""), ("sampled", -1.0, None, 1.0, 1.0, 0, ""),
           ("sampled", -1.0, 1, 1.0, 0.0, 0, ""), ("ae", -1.0, 1, 1.0, 1.0, 0.0, ""),
           ("ae", math.nan, 1, 1.0, 0.0, 2.0, ""), ("loop_energy", -1.0, -1, 0, 0, 0, ""),
           ("sampled", -1.0, 1, -1.0, 1.0, 0, ""), ("oracle", -1.0, None, 0, 0, 0, 3)]
    for args in bad:
        with pytest.raises(ValueError):
            certify.EnergyCertificate(*args)
    assert certify.EnergyCertificate("ae", -1.0, 1, 1.0, 0.0, 2.0, "x", sigma=2e-3).sigma == 2e-3
    for source, j, cost in (("classical", None, (0, 0, 0)), ("loop_energy", 1, (0, 0, 0)), ("oracle", None, (0, 0, 0)),
                            ("sampled", 1, (1.0, 1.0, 0.0))):
        for sigma in ((1e-3,) if source != "sampled" else (0.0, -1e-3, math.nan, math.inf)):
            with pytest.raises(ValueError):
                certify.EnergyCertificate(source, -1.0, j, *cost, "x", sigma=sigma)

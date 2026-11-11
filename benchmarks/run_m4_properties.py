"""M4 properties: the CQE iterate (route A) against QPE-projected copies (route B) and Hellmann-Feynman QPE (U10, U11).

Units: Toffolis (in the tables / C_QPE of the system's selected encoding, F9), Hartree; properties in their own
units (electrons; the dipole in atomic units). Per system of ``--systems`` (F20 names; default the property set
h4_r1.8, h4_r3.0, h4_r3.5, h4_r4.0, h6_r1.8, h6a, h6_r3.0, h6_r3.5, h6_r4.0, lih, n2_r1.098, n2_r2.0,
n2_r2.4):

Inputs
    ``ft.closedloop.system_inputs`` (the 12-root spectrum with psi_0, the selected QPE, the allowed spin-free
    elements with M3's per-shot variances at iterate 1 of the exact spin-free record and the LCU
    normalizations); the targets of F46 at psi_0 (``ft.properties``: natural occupations, 1-RDM, 2-RDM,
    cumulant, cumulant norm, and the dipole on LiH, the polar all-active build of the set) with the
    accuracies of ``ft.properties.ACCURACIES``; their estimator inputs at psi_0 (``property_inputs``: exact
    matchgate variances, Pauli 1-norms and term counts, spreads sigma_0 and ranges r_O; F48) and the
    shadow/AE crossover B* (``crossover_accuracy``).
Records (F49)
    r1_so: the unit-step rule R1 with the spin-orbital metric and the classical first step
    (``steprules.run_step_rule``, 60 steps, ``ft.steprules.to_trajectory_record``), the reference record;
    ls_so: M3's exact-line-search record with the same metric and first step (``exact_trajectory``), the
    comparison; compiled_s1: the R1 steps compiled elementwise from HF without slicing
    (``ft.properties.compiled_iterates``, F3); compiled_sliced: the same with F25's slice rule on steps 0-2
    (the smallest s in {1, 2, 4, 8, 16} with ||Psi~^(s) - exp(eps D) Psi~|| <= 0.05 from the current compiled
    state, s = 16 with the flag "slice_cap" if none; ``ft.circuits.compiled_step``, the cost s C_step(d, eps/s)
    in C(Psi~_k)). Preparation costs C(Psi_k), Q(Psi_k) by ``ft.stateprep.prep_cost``. Loop costs
    T_res(j) = min(direct_mg, ae_x) at kappa = 1 and r = 1 ("meas_r1") or 0.3 ("meas_r03") at Psi_j with the
    exact residual of Psi_j and the system's variances and normalizations (M3's F10, F11), T_res(0) = 0
    (the classical first step); the free loop costs nothing.
Route A (F49), route B (F50), Hellmann-Feynman QPE (F51), comparison (F53)
    ``route_a_rows`` on r1_so (and ls_so for comparison) at the fixed budgets k = 2, 5, 10, 20, 60 with the
    exact property errors of the iterates (a budget past the record is reported and "not_usable" in a
    measured loop, and in any loop past a record that did not converge); the compiled ablation
    "route_a_compiled" (N <= 6): the same on compiled_sliced, with the errors of Psi~_k, their sliced
    C(Psi~_k) and the loop costs T_res(j) at Psi~_j; ``route_b_rows`` from the r1_so starts on the
    property-aware QPE of F52 for the target and accuracy (below; F9's selection where F52 picks it): the
    pass edge at
    U = CISD(H') (``ft.certify.truncated_ci_energy``), a start's AE energy certificate, or E1lb - w_L,
    E1lb = E_1' - 0.1 mHa from the model spectrum of that H' (``ft.acceptance.model_spectrum``; the any-irrep
    singlet E_1' for THC encodings and where the eigenvectors of H' leave their irrep by more than 1e-6,
    F41), E_0' for the termination of the E1lb edge, the exact model bias b_fact of that encoding
    (psi_0' by ``diagnostics.thc_floors.model_floors``), and for the dipole Hellmann-Feynman QPE with E''' and
    E^(5) from finite-field FCI (``finite_field_derivatives``, 41 fields on [-0.02, 0.02] a.u.), b_fact added
    linearly to the stencil bias and the field step checked against the finite-field energies
    (``FiniteFieldStencil``; "hf_steps" in the JSON: per stencil b, F51's leading-order and the refined h
    with the finite-field stencil bias at each). Rows on another encoding than F9's carry the flags
    "property_aware_encoding" and "encoding:<kind><params>"
    (F9's rows are kept as "route_b_f9" in the JSON); where no scanned encoding has b_fact <= eps/2 route B
    runs on F9's ("no_property_aware_encoding"). Costs stay in units of F9's C_QPE. The best selectable row
    per loop and estimator (``best_property_row``; AE at kappa = 1 is the reference), with the surface-code layout
    of its longest coherent run (``run_layout``: p_fail_ae, a shadow snapshot at p_fail_shot; rows without one carry
    the flag "no_layout" and stay selectable, F53 comparing Toffolis). Sensitivity (never decisive): the best rows
    with route B and HF-QPE charging F50's F47 bound at q = 1 - F of the same encoding instead of b_fact
    ("model_bias_sensitivity" in the JSON).
U11 (bias floors)
    Per target and record: the smallest property error over k <= 60, its k, the energy error there, and
    k_eps (the first iterate after which the error stays <= eps) per accuracy; F47's bound
    (``bias_bound``, q form with q = 1 - p_k and energy form with d = E_k - E_0 and Delta_1 = E_1 - E_0)
    against the exact error on every iterate of r1_so and ls_so with error > 1e-9 (the smallest
    bound/error); for the cumulant norm the linearized bound is first order, and its all-orders form
    (``bias_bound(..., cumulant_parts=...)``) is checked alongside.
F52 (property-aware selection)
    Every scanned encoding (``ft.qpe.scan_encodings``) with its psi_0', b_fact(P) per target, and
    ``select_for_property`` at each accuracy against the selected (F9) encoding.

Outputs: <out>/properties_<system>.json (strict JSON: nan as null, +-inf as "inf" / "-inf", via
``ft.scaling.to_json``) and <out>/properties_summary.md (U10, U11, F52, LiH finite field). Expensive stages
(system inputs, model spectrum, CISD, records, loop costs, estimator inputs, compiled iterates, the scan
and its psi_0', the finite field) are cached per system in <out>/cache (``ft.scaling.Cache``;
``--no-cache`` recomputes and overwrites). One thread per process (the thread variables are set before
numpy is imported).

Usage:
    python benchmarks/run_m4_properties.py --quick
    python benchmarks/run_m4_properties.py --systems lih,n2_r2.0 --out results/m4
"""

from __future__ import annotations

import os

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
for _var in THREAD_VARS:                         # one thread per process, before numpy is imported
    os.environ[_var] = "1"

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from thc_engine_cqe import steprules  # noqa: E402
from thc_engine_cqe.diagnostics.thc_floors import ExactReference, model_floors  # noqa: E402
from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian, normalize  # noqa: E402
from thc_engine_cqe.ft import acceptance as acc  # noqa: E402
from thc_engine_cqe.ft import amplify, certify, closedloop, scaling  # noqa: E402
from thc_engine_cqe.ft import properties as P  # noqa: E402
from thc_engine_cqe.ft import steprules as ft_steprules  # noqa: E402
from thc_engine_cqe.ft.circuits import compiled_step  # noqa: E402
from thc_engine_cqe.ft.primitives import FTParameters  # noqa: E402
from thc_engine_cqe.ft.qpe import qpe_cost, scan_encodings  # noqa: E402
from thc_engine_cqe.ft.residual import PrepParams, ResidualProblem, ae_x_lcu, sampling_direct  # noqa: E402
from thc_engine_cqe.ft.stateprep import elementwise_step_cost, prep_cost  # noqa: E402
from thc_engine_cqe.ft.symmetry import BENCHMARK_SYSTEMS, benchmark_system, determinant_irreps  # noqa: E402
from thc_engine_cqe.ft.trajectory import exact_trajectory, propagate  # noqa: E402
from thc_engine_cqe.measure import routes  # noqa: E402
from thc_engine_cqe.residual import rdm12  # noqa: E402

DEFAULT_SYSTEMS = scaling.PROPERTIES
QUICK_SYSTEMS = ("h4_r1.8", "lih", "n2_r2.0")
DIPOLE_SYSTEMS = ("lih",)       # the polar all-active build of the set (the chains' dipoles vanish by symmetry)
N_STEPS = 60
BUDGETS = (2, 5, 10, 20, 60)
MEASURED_LOOPS = {"meas_r1": 1.0, "meas_r03": 0.3}
LOOPS = ("free",) + tuple(MEASURED_LOOPS)
SLICED_STEPS = (0, 1, 2)
SLICE_CHOICES = (1, 2, 4, 8, 16)
ETA_SLICE = 0.05                # F25's state-error threshold of a sliced step
FIELD_MAX, FIELD_POINTS, FIELD_DEGREE = 0.02, 41, 8
IMPURITY_TOL = 1e-6
TINY_ERROR = 1e-9               # rows of the F47 check: property error above this
RECORD_LABELS = ("r1_so", "ls_so", "compiled_s1", "compiled_sliced")


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far (MB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _params_key(params: FTParameters) -> dict:
    return {"params": dataclasses.asdict(params)}


# ------------------------------------------------------------------ the QPE side
def qpe_side(name: str, inputs, cache, params: FTParameters, *, qpe=None) -> dict:
    """Model spectrum (E_0', E1lb, c) and CISD(H') of an encoding (default the selected one; module docstring)."""
    active, irreps = inputs.active, inputs.irreps
    q = inputs.qpe if qpe is None else qpe
    enc = q.encoding
    key = {**_params_key(params), "kind": enc.kind, "enc_params": scaling.to_json(enc.params)}
    spec = cache.get(name, "model_spectrum", lambda: acc.model_spectrum(active, q, irreps), key=key)
    model = enc.model(active)
    u = cache.get(name, "cisd", lambda: certify.truncated_ci_energy(model, 2), key=key)
    levels = certify.sector_levels(spec, det_irreps=determinant_irreps(active.norb, tuple(active.nelec), irreps))
    broken = enc.kind == "thc" or levels["irrep_impurity"] > IMPURITY_TOL
    e1_ref = min(spec.e1_singlet, spec.e1_any_irrep) if broken else spec.e1_singlet
    e1lb = certify.e1_lower_bound(e1_ref, model_spectrum=True)
    return {"spec": spec, "enc": enc, "qpe": q, "c": float(spec.c), "u": float(u), "e1lb": float(e1lb),
            "e0_model": float(spec.e0), "levels": levels, "e1lb_level": "e1_any_irrep" if broken else "e1_singlet",
            "flags": ["point_group_broken"] if broken and enc.kind != "thc" else []}


# ------------------------------------------------------------------ records and costs
def build_records(name: str, inputs, cache) -> dict:
    """r1_so and ls_so (module docstring): {label: TrajectoryRecord}."""
    active, irreps, spec = inputs.active, inputs.irreps, inputs.spectrum

    def compute():
        res = steprules.run_step_rule(active, steprules.StepRule("unit"), max_iter=N_STEPS, metric="spin_orbital",
                                      irreps=irreps, keep_states=True)
        r1 = ft_steprules.to_trajectory_record(res, system=name, active=active, irreps=irreps, spectrum=spec)
        ls = exact_trajectory(active, optimizer="lbfgs", metric="spin_orbital", max_iter=N_STEPS, system=name,
                              irreps=irreps, n_roots=spec.n_roots, first_step="second_order", roots=spec.roots)
        return {"r1_so": r1, "ls_so": ls}

    return cache.get(name, "properties_records", compute, key={"n_steps": N_STEPS})


def residual_costs(name: str, inputs, records: dict, preps: dict, cache, params: FTParameters) -> dict:
    """{label: {loop: [T_res(j), j = 0..K-1]}} (module docstring), Toffolis."""
    def compute():
        out = {}
        el, m = inputs.elements, inputs.m
        var, nrm = inputs.variances, inputs.normalizations
        for label, rec in records.items():
            pc = preps[label]
            out[label] = {lp: [0.0] for lp in MEASURED_LOOPS}
            out[label]["route"] = {lp: ["classical"] for lp in MEASURED_LOOPS}
            for j in range(1, rec.n_steps):
                a = routes.exact_residual(inputs.active, rec.states[j], el, "spin_free")
                prob = ResidualProblem(kind="spin_free", elements=el, m=m, A=a, v_direct=var["v_direct"],
                                       w_aux=var["w_aux"], lam_q=nrm["lam_q"], alpha_x=nrm["alpha_x"],
                                       n_terms_x=nrm["n_terms_x"], aux_bias=None, norb=inputs.active.norb)
                pp = PrepParams(float(pc.cumulative[j]), int(pc.logical_qubits[j]))
                for lp, r in MEASURED_LOOPS.items():
                    pr = dataclasses.replace(params, r=r)
                    d = sampling_direct(prob, pp, params=pr)
                    x = ae_x_lcu(prob, pp, kappa=1.0, params=pr)
                    best = d if d.toffolis <= x.toffolis else x
                    out[label][lp].append(float(best.toffolis))
                    out[label]["route"][lp].append(best.route)
        return out

    return cache.get(name, "properties_residual_costs", compute,
                     key={"n_steps": N_STEPS, "loops": MEASURED_LOOPS, **_params_key(params)})


def compiled_residual_costs(name: str, inputs, compiled: dict, cache, params: FTParameters) -> dict:
    """{loop: [T_res(j), j = 0..K-1]} along the F25-sliced compiled record: the exact residual of Psi~_j and its
    sliced C(Psi~_j), Q(Psi~_j), otherwise as :func:`residual_costs` (T_res(0) = 0)."""
    def compute():
        el, m = inputs.elements, inputs.m
        var, nrm = inputs.variances, inputs.normalizations
        states, cum, qub = compiled["states"], compiled["cumulative"], compiled["qubits"]
        out = {lp: [0.0] for lp in MEASURED_LOOPS}
        for j in range(1, len(states) - 1):
            a = routes.exact_residual(inputs.active, states[j], el, "spin_free")
            prob = ResidualProblem(kind="spin_free", elements=el, m=m, A=a, v_direct=var["v_direct"],
                                   w_aux=var["w_aux"], lam_q=nrm["lam_q"], alpha_x=nrm["alpha_x"],
                                   n_terms_x=nrm["n_terms_x"], aux_bias=None, norb=inputs.active.norb)
            pp = PrepParams(float(cum[j]), int(qub[j]))
            for lp, r in MEASURED_LOOPS.items():
                pr = dataclasses.replace(params, r=r)
                d = sampling_direct(prob, pp, params=pr)
                x = ae_x_lcu(prob, pp, kappa=1.0, params=pr)
                out[lp].append(float(min(d.toffolis, x.toffolis)))
        return out

    return cache.get(name, "properties_residual_costs_compiled", compute,
                     key={"n_steps": N_STEPS, "loops": MEASURED_LOOPS, "sliced": list(SLICED_STEPS),
                          "choices": list(SLICE_CHOICES), "eta": ETA_SLICE, **_params_key(params)})


def compiled_records(name: str, inputs, rec, cache, params: FTParameters) -> dict:
    """compiled_s1 and compiled_sliced (module docstring): states, C(Psi~_k), Q(Psi~_k), slices and flags."""
    active, irreps = inputs.active, inputs.irreps

    def compute():
        s1 = P.compiled_iterates(active, rec, slices=1, sliced_steps=None, params=params)
        ham = Hamiltonian(active)
        states = [normalize(np.asarray(s1[0]))]
        slices, errors, flags = [], [], []
        pc1 = prep_cost(rec.directions, rec.eps, irreps=irreps, params=params)
        cum, qub = [0.0], [int(pc1.logical_qubits[0])]                 # Q(HF) = 2N (F7)
        for j, (d, e) in enumerate(zip(rec.directions, rec.eps)):
            psi = states[-1]
            chosen, err, out = 1, None, None
            if j in SLICED_STEPS:
                target = propagate(ham, d, psi, float(e))
                for s in SLICE_CHOICES:
                    trial = psi
                    for _ in range(s):
                        trial = compiled_step(active, trial, d, float(e) / s, irreps=irreps, params=params)
                    err = float(np.linalg.norm(np.asarray(trial).ravel() - np.asarray(target).ravel()))
                    chosen, out = s, trial
                    if err <= ETA_SLICE:
                        break
                else:
                    flags.append(f"slice_cap:step{j}")
            else:
                out = compiled_step(active, psi, d, float(e), irreps=irreps, params=params)
            step = elementwise_step_cost(d, float(e) / chosen, irreps=irreps, params=params, step=j)
            cum.append(cum[-1] + chosen * float(step.toffolis))
            qub.append(max(qub[-1], int(step.logical_qubits)))
            states.append(normalize(np.asarray(out)))
            slices.append(chosen)
            errors.append(err)
        return {"compiled_s1": {"states": [normalize(np.asarray(s)) for s in s1],
                                "cumulative": np.asarray(pc1.cumulative, float),
                                "qubits": np.asarray(pc1.logical_qubits, int), "slices": [1] * rec.n_steps,
                                "flags": []},
                "compiled_sliced": {"states": states, "cumulative": np.asarray(cum, float),
                                    "qubits": np.asarray(qub, int), "slices": slices, "slice_errors": errors,
                                    "flags": flags}}

    return cache.get(name, "properties_compiled", compute, key={"n_steps": N_STEPS, "sliced": list(SLICED_STEPS),
                                                                 "choices": list(SLICE_CHOICES), "eta": ETA_SLICE,
                                                                 **_params_key(params)})


# ------------------------------------------------------------------ targets, inputs, F52
def build_targets(name: str, active, irreps, psi0) -> dict:
    dm1, dm2 = rdm12(FCISpace.of(active), psi0)
    out = {"occupations": P.occupation_target(active, dm1), "rdm1": P.rdm1_target(active, irreps),
           "rdm2": P.rdm2_target(active, irreps), "cumulant": P.cumulant_target(active, irreps, dm1),
           "cumulant_norm": P.cumulant_norm_target(active, irreps, dm1, dm2)}
    if name in DIPOLE_SYSTEMS:
        out["dipole"] = P.dipole_target(active)
    return out


def estimator_inputs(name: str, active, psi0, targets: dict, cache) -> dict:
    return cache.get(name, "properties_inputs",
                     lambda: {k: P.property_inputs(active, psi0, t) for k, t in targets.items()},
                     key={"targets": sorted(targets)})


def scan_states(name: str, active, cache, params: FTParameters) -> list:
    """[(BlockEncoding, psi_0')] for every scanned encoding (F52)."""
    def compute():
        ref = ExactReference.build(active)
        encs = scan_encodings(active, reference=ref, params=params)
        return [(enc, model_floors(ref, enc.eri_model, return_state=True)[1]) for enc in encs]

    return cache.get(name, "properties_scan", compute, key=_params_key(params))


def _same_encoding(a, b) -> bool:
    return (a.kind == b.kind and scaling.to_json(a.params) == scaling.to_json(b.params)
            and abs(a.lam - b.lam) <= 1e-12 * abs(b.lam) and int(a.toffoli_per_step) == int(b.toffoli_per_step))


# ------------------------------------------------------------------ U11 helpers
def floors(errors: np.ndarray, energies: np.ndarray, e0: float, accuracies) -> dict:
    """Minimum error, its k and energy error, and k_eps (the first k after which the error stays <= eps)."""
    err = np.asarray(errors, dtype=float)
    k_min = int(np.argmin(err))
    k_eps = {}
    for eps in accuracies:
        above = np.flatnonzero(err > eps)
        k = 0 if above.size == 0 else int(above[-1]) + 1
        k_eps[str(eps)] = k if k < err.size else None
    return {"min": float(err[k_min]), "k_min": k_min, "e_err_at_min": float(energies[k_min] - e0),
            "last": float(err[-1]), "k_eps": k_eps}


def f47_check(target, active, psi0, inp, rec, errors, gap: float, *, parts=None) -> dict:
    """The smallest bound/|bias| of both F47 forms over the iterates with |bias| > TINY_ERROR.

    For the cumulant norm (``parts``: the cumulant and 1-RDM estimator inputs) also the all-orders form
    B_cum + 3/2 B_1^2 of ``bias_bound(..., cumulant_parts=parts)``; the linearized form is first order there.
    """
    rq, re_, aq, ae = [], [], [], []
    for k, b in enumerate(errors):
        if b <= TINY_ERROR:
            continue
        q = 1.0 - min(float(rec.overlaps[k]), 1.0)
        bq = P.bias_bound(target, active, psi0, q=q, sigma0=inp.sigma0, ranges=inp.r0)
        d = max(float(rec.energies[k] - rec.e0), 0.0)
        be = P.bias_bound(target, active, psi0, energy_error=d, gap=gap, sigma0=inp.sigma0, ranges=inp.r0)
        rq.append(bq / b)
        re_.append(be / b)
        if parts is not None:
            aq.append(P.bias_bound(target, active, psi0, q=q, cumulant_parts=parts) / b)
            ae.append(P.bias_bound(target, active, psi0, energy_error=d, gap=gap, cumulant_parts=parts) / b)
    out = {"n_rows": len(rq), "q_form_min_ratio": min(rq) if rq else None,
           "energy_form_min_ratio": min(re_) if re_ else None}
    if parts is not None:
        out.update(order="first", q_form_min_ratio_all_orders=min(aq) if aq else None,
                   energy_form_min_ratio_all_orders=min(ae) if ae else None)
    return out


def hf_steps(qpe, eps: float, derivs: dict, model_bias: float, stencil, rows) -> dict:
    """The field steps of F51 at ``eps`` with the model bias added linearly, per stencil: the allocation b, F51's
    leading-order h and the refined h with the signed stencil bias of ``stencil`` at both (None without it),
    whether the stencil has no admissible step (h nan), and whether an HF-QPE row of ``rows`` uses it."""
    used = {f[len("stencil:"):] for r in rows if r.route == "HF-QPE" for f in r.flags if f.startswith("stencil:")}
    out = {}
    for st in P.STENCILS:
        if st == "five_point" and derivs.get("d5") is None:
            continue
        res = P.hellmann_feynman_cost(qpe, eps, d3=derivs["d3"], d5=derivs.get("d5"), stencil=st,
                                      fixed_bias=model_bias, stencil_bias=stencil)
        lead = float(stencil(res["h_leading"], st)) if stencil is not None and res["feasible"] else None
        out[st] = {"used": st in used, "feasible": res["feasible"], "no_field_step": res["no_field_step"],
                   "fixed_bias": res["fixed_bias"], "stencil_bias": res["stencil_bias"], "bias": res["bias"],
                   "h_leading": res["h_leading"], "h": res["h"], "stencil_bias_at_leading": lead,
                   "stencil_bias_at_h": res["stencil_bias_checked"]}
    return out


def _row_json(row: P.PropertyRow, c_qpe: float) -> dict:
    out = scaling.to_json(row)
    out["toffolis_over_qpe"] = scaling.to_json(row.toffolis / c_qpe)
    out["selectable"] = bool(row.selectable)
    return out


def _best(rows, loop: str, estimator: str, c_qpe: float, params: FTParameters = FTParameters()) -> dict | None:
    """The best selectable row (``best_property_row``) with the layout of its longest coherent run (``run_layout``;
    None: no layout)."""
    try:
        b = P.best_property_row(rows, loop=loop, estimator=estimator)
    except ValueError:
        return None
    lay = P.run_layout(b, params=params)
    return {"route": b.route, "estimator": b.estimator, "k": b.k, "toffolis_over_qpe": b.toffolis / c_qpe,
            "bias": b.bias, "flags": list(b.flags), "longest_run": b.longest_run, "logical_qubits": b.logical_qubits,
            "layout": None if lay is None else {"physical_qubits": lay.physical_qubits, "distance": lay.distance,
                                                "factory": lay.factory, "seconds": lay.seconds}}


# ------------------------------------------------------------------ one system
def run_one(name: str, cache, params: FTParameters) -> dict:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    def lap(label: str, t: float) -> float:
        now = time.perf_counter()
        timings[label] = timings.get(label, 0.0) + now - t
        return now

    t = time.perf_counter()
    inputs = closedloop.system_inputs(name, cache=cache, params=params)
    active, irreps, sp = inputs.active, inputs.irreps, inputs.spectrum
    psi0 = normalize(np.asarray(sp.psi0))
    c_qpe = float(inputs.qpe.toffolis)
    t = lap("system_inputs", t)
    side = qpe_side(name, inputs, cache, params)
    t = lap("qpe_side", t)
    records = build_records(name, inputs, cache)
    preps = {lab: prep_cost(rec.directions, rec.eps, irreps=irreps, params=params) for lab, rec in records.items()}
    t = lap("records", t)
    tres = residual_costs(name, inputs, records, preps, cache, params)
    t = lap("residual_costs", t)
    comp = compiled_records(name, inputs, records["r1_so"], cache, params) if active.norb <= 6 else {}
    tres_c = (compiled_residual_costs(name, inputs, comp["compiled_sliced"], cache, params)
              if "compiled_sliced" in comp else None)
    t = lap("compiled", t)
    targets = build_targets(name, active, irreps, psi0)
    inps = estimator_inputs(name, active, psi0, targets, cache)
    t = lap("estimator_inputs", t)
    scan = scan_states(name, active, cache, params)
    t = lap("scan", t)
    ff = None
    if "dipole" in targets:
        ff = cache.get(name, "properties_finite_field",
                       lambda: P.finite_field_derivatives(active, targets["dipole"], h_max=FIELD_MAX,
                                                          n_points=FIELD_POINTS, degree=FIELD_DEGREE),
                       key={"h_max": FIELD_MAX, "n_points": FIELD_POINTS, "degree": FIELD_DEGREE})
    stencil = P.FiniteFieldStencil(active, targets["dipole"]) if "dipole" in targets else None
    t = lap("finite_field", t)
    gap = float(sp.e1 - sp.e0)
    sel_index = next((i for i, (enc, _) in enumerate(scan) if _same_encoding(enc, side["enc"])), None)
    record_states = {"r1_so": records["r1_so"].states, "ls_so": records["ls_so"].states}
    record_states.update({lab: v["states"] for lab, v in comp.items()})
    energies = {"r1_so": records["r1_so"].energies, "ls_so": records["ls_so"].energies}
    ham = Hamiltonian(active)
    for lab, v in comp.items():
        energies[lab] = np.array([float(np.vdot(s.ravel(), ham.apply(s).ravel()).real) + active.ecore
                                  for s in v["states"]])
    rec_r1, prep_r1 = records["r1_so"], preps["r1_so"]
    sides: dict = {}                             # QPE sides of property-aware encodings (F52)
    out_targets = {}
    for kind, target in targets.items():
        inp = inps[kind]
        accs = P.ACCURACIES[kind]
        errs = {lab: np.array([P.property_error(target, active, s, psi0) for s in states])
                for lab, states in record_states.items()}
        b_scan = np.array([P.property_error(target, active, psi_m, psi0) for _, psi_m in scan])
        if sel_index is not None:
            b_fact = float(b_scan[sel_index])
        else:
            _, psi_m = model_floors(ExactReference.build(active), side["enc"].eri_model, return_state=True)
            b_fact = float(P.property_error(target, active, psi_m, psi0))
        block = {"norm": target.norm, "n_elements": int(target.n_elements), "accuracies": list(accs),
                 "inputs": {"variance_m": float(np.sum(target.m * inp.v_shadow)) if target.norm == "fro" else
                            float(np.max(inp.v_shadow)), "sigma_norm": inp.sigma_norm, "range_norm": inp.range_norm,
                            "method": inp.method, "crossover": P.crossover_accuracy(inp)},
                 "model_bias": b_fact, "model_bias_scan": [{"kind": enc.kind, "params": enc.params,
                                                            "c_qpe": float(qpe_cost(enc, params.de_qpe).toffolis),
                                                            "de_fact": enc.de_fact, "infidelity": enc.infidelity,
                                                            "b_fact": float(b)} for (enc, _), b in zip(scan, b_scan)],
                 "errors": {lab: e for lab, e in errs.items()},
                 "floors": {lab: floors(e, energies[lab], float(sp.e0), accs) for lab, e in errs.items()},
                 "f47_check": {lab: f47_check(target, active, psi0, inp, records[lab], errs[lab], gap,
                                              parts=(inps["cumulant"], inps["rdm1"]) if kind == "cumulant_norm"
                                              else None)
                               for lab in ("r1_so", "ls_so")},
                 "selection": {}, "rows": {}, "best": {}, "best_ls": {}, "model_bias_sensitivity": {}}
        for eps in accs:
            side_b, b_route, tag = side, b_fact, ()
            try:
                sel = P.select_for_property([enc for enc, _ in scan], b_scan, eps, params=params)
                j = next(i for i, (enc, _) in enumerate(scan) if _same_encoding(enc, sel.encoding))
                changed = not _same_encoding(sel.encoding, side["enc"])
                block["selection"][str(eps)] = {"kind": sel.encoding.kind, "params": sel.encoding.params,
                                                "c_qpe": float(sel.toffolis), "ratio": float(sel.toffolis) / c_qpe,
                                                "b_fact": float(b_scan[j]), "changed": changed}
                if changed:                      # route B and HF-QPE on the property-aware QPE (F52)
                    ek = (sel.encoding.kind, json.dumps(scaling.to_json(sel.encoding.params), sort_keys=True))
                    if ek not in sides:
                        sides[ek] = qpe_side(name, inputs, cache, params, qpe=sel)
                    side_b, b_route = sides[ek], float(b_scan[j])
                    tag = ("property_aware_encoding", f"encoding:{sel.encoding.kind}"
                           + json.dumps(scaling.to_json(sel.encoding.params), sort_keys=True))
            except ValueError as exc:
                block["selection"][str(eps)] = {"error": str(exc)}
                tag = ("no_property_aware_encoding",)
            ra = P.route_a_rows(rec_r1, prep_r1, {lp: tres["r1_so"][lp] for lp in MEASURED_LOOPS}, errs["r1_so"],
                                inp, eps, loops=LOOPS, budgets=BUDGETS, system=name, params=params)
            ra_ls = P.route_a_rows(records["ls_so"], preps["ls_so"], {lp: tres["ls_so"][lp] for lp in MEASURED_LOOPS},
                                   errs["ls_so"], inp, eps, loops=LOOPS, budgets=BUDGETS, system=name, params=params)
            # the compiled ablation (F49): Psi~_k of the F25-sliced replay, its sliced C(Psi~_k) and loop costs
            ra_c = None if tres_c is None else P.route_a_rows(
                rec_r1, (comp["compiled_sliced"]["cumulative"], comp["compiled_sliced"]["qubits"]), tres_c,
                errs["compiled_sliced"], inp, eps, loops=LOOPS, budgets=BUDGETS, system=name, params=params)
            derivs = None if (ff is None or kind != "dipole") else {"d3": ff["d3"], "d5": ff["d5"]}
            def b_rows(sd, bias):
                return P.route_b_rows(rec_r1, prep_r1, {lp: tres["r1_so"][lp] for lp in MEASURED_LOOPS}, inp, eps,
                                      sd["enc"], sd["c"], u_classical=sd["u"], e1_lower=sd["e1lb"], loops=LOOPS,
                                      system=name, e0_model=sd["e0_model"], model_bias=bias, hf_derivatives=derivs,
                                      qpe=sd["qpe"], params=params, hf_stencil_bias=stencil)
            rb = [dataclasses.replace(r, flags=tuple(r.flags) + tag) for r in b_rows(side_b, b_route)]
            rb_f9 = rb if side_b is side else b_rows(side, b_fact)
            rb_f47 = b_rows(side_b, None)                # F50's model-bias bound at q = 1 - F (sensitivity)
            key = str(eps)
            block["rows"][key] = {"route_a": [_row_json(r, c_qpe) for r in ra],
                                  "route_a_ls": [_row_json(r, c_qpe) for r in ra_ls],
                                  "route_a_compiled": None if ra_c is None else [_row_json(r, c_qpe) for r in ra_c],
                                  "route_b": [_row_json(r, c_qpe) for r in rb],
                                  "route_b_f9": None if rb_f9 is rb else [_row_json(r, c_qpe) for r in rb_f9]}
            block["best"][key] = {lp: {est: _best(ra + rb, lp, est, c_qpe, params) for est in ("ae", "shadow")}
                                  for lp in LOOPS}
            block["best_ls"][key] = {lp: {est: _best(ra_ls + rb, lp, est, c_qpe, params) for est in ("ae", "shadow")}
                                     for lp in LOOPS}
            if derivs is not None:                       # the HF-QPE rows' field steps (F51)
                block.setdefault("hf_steps", {})[key] = hf_steps(side_b["qpe"], eps, derivs, b_route, stencil, rb)
            block["model_bias_sensitivity"][key] = {
                "label": ("route B and HF-QPE with F47's model-bias bound at q = 1 - F instead of b_fact "
                          "(never decisive)"),
                "model_bias": P.model_bias_bound(inp, float(side_b["enc"].infidelity)), "b_fact": b_route,
                "best": {lp: {est: _best(ra + rb_f47, lp, est, c_qpe, params) for est in ("ae", "shadow")}
                         for lp in LOOPS}}
        out_targets[kind] = block
    t = lap("routes", t)
    rec_json = {}
    for lab, rec in records.items():
        rec_json[lab] = {"n_steps": rec.n_steps, "first_step": rec.first_step, "metric": rec.metric,
                         "e_err": np.asarray(rec.energies) - float(sp.e0), "overlaps": rec.overlaps, "eps": rec.eps,
                         "prep_cumulative": preps[lab].cumulative, "prep_qubits": preps[lab].logical_qubits,
                         "t_res_over_qpe": {lp: [x / c_qpe for x in tres[lab][lp]] for lp in MEASURED_LOOPS},
                         "t_res_route": tres[lab]["route"], "converged": rec.converged, "flags": list(rec.flags)}
    for lab, v in comp.items():
        p = [abs(float(np.vdot(psi0.ravel(), s.ravel()))) ** 2 for s in v["states"]]
        rec_json[lab] = {"e_err": energies[lab] - float(sp.e0), "overlaps": p, "prep_cumulative": v["cumulative"],
                         "prep_qubits": v["qubits"], "slices": v["slices"], "slice_errors": v.get("slice_errors"),
                         "flags": v["flags"]}
        if lab == "compiled_sliced" and tres_c is not None:
            rec_json[lab]["t_res_over_qpe"] = {lp: [x / c_qpe for x in tres_c[lp]] for lp in MEASURED_LOOPS}
    q = side["qpe"]
    result = {"system": name, "norb": active.norb, "nelec": list(active.nelec), "dim": sp.dim,
              "spectrum": {"e0": sp.e0, "e1": sp.e1, "e1_any": sp.e1_any, "e_hf": sp.e_hf, "p_hf": sp.p_hf,
                           "gap": gap, "flags": list(sp.flags)},
              "qpe": {"kind": q.encoding.kind, "params": q.encoding.params, "lam": q.encoding.lam,
                      "c_w": int(q.encoding.toffoli_per_step), "n_iter": int(q.iterations), "c_qpe": c_qpe,
                      "logical_qubits": int(q.logical_qubits), "de_fact": q.encoding.de_fact,
                      "infidelity": q.encoding.infidelity},
              "model": {"e0_model": side["e0_model"], "e0_model_minus_e0": side["e0_model"] - float(sp.e0),
                        "e1lb": side["e1lb"], "e1lb_level": side["e1lb_level"],
                        "e1lb_minus_e0_model": side["e1lb"] - side["e0_model"], "u_cisd": side["u"],
                        "u_cisd_minus_e0_model": side["u"] - side["e0_model"], "walk_offset": side["c"],
                        "sin_theta0": float(side["spec"].sin_theta0),
                        "w_l_terminates": side["e0_model"] <= side["e1lb"] - amplify.W_L,
                        "irrep_impurity": side["levels"]["irrep_impurity"], "flags": side["flags"]},
              "variance_state": inputs.variance_state, "records": rec_json, "targets": out_targets,
              "finite_field": None if ff is None else {k: ff[k] for k in ("d1", "d2", "d3", "d4", "d5")} | {
                  "mu_fci": float(P.property_values(targets["dipole"], active, psi0)[0]),
                  "mu_hf": float(P.property_values(targets["dipole"], active, rec_r1.states[0])[0]),
                  "fields": ff["fields"], "energies": ff["energies"]},
              "timings": timings, "runtime_s": time.perf_counter() - t0, "peak_rss_mb": peak_rss_mb()}
    return result


# ------------------------------------------------------------------ markdown
def _f(x, spec: str = ".3g") -> str:
    if x is None:
        return "-"
    x = float(x)
    if math.isnan(x):
        return "-"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    text = format(x, spec)
    if "e" in text:
        mant, exp = text.split("e")
        text = f"{mant}e{int(exp)}"
    return text


def _table(header, rows) -> list[str]:
    return (["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows] + [""])


def _cell(b: dict | None) -> str:
    if b is None:
        return "-"
    return f"{b['route']} [{b['k']}] {_f(b['toffolis_over_qpe'])}"


def _num(x) -> float:
    """A strict-JSON number back to a float (null -> nan, "inf"/"-inf" -> +-inf)."""
    if x is None:
        return math.nan
    if isinstance(x, str):
        return {"inf": math.inf, "-inf": -math.inf}[x]
    return float(x)


def _a_marks(row: dict, eps: float) -> str:
    """U10a's marks of a route-A row: "x" bias >= eps; "e" a budget past the record that is not costed (a
    measured loop past the record, whose C_loop(K) only bounds the loop's cost, or a free loop past a record that
    did not converge)."""
    marks = []
    if _num(row["bias"]) >= eps:
        marks.append("x")
    if "record_ended" in row["flags"] and "not_usable" in row["flags"] and (row["loop"] != "free"
                                                                           or _num(row["bias"]) < eps):
        marks.append("e")
    return (" " + " ".join(marks)) if marks else ""


def _layout_cell(b: dict | None) -> str:
    """The layout of a best row's longest coherent run: "d<distance>", "none" (no layout) or "-" (no best row)."""
    if b is None:
        return "-"
    if "layout" not in b:                        # a row without the layout entry: the flag of the row builders
        return "none" if "no_layout" in b.get("flags", ()) else "-"
    return "none" if b["layout"] is None else f"d{b['layout']['distance']}"


def _best_a(rows: list | None, loop: str, c_qpe: float) -> str:
    """The cheapest selectable route-A row (AE) of a JSON row list: "cost [k]" in C_QPE, "x" if none is usable."""
    if rows is None:
        return "-"
    b = P.best_at_budget([P.row_from_json(x) for x in rows if x["route"] == "A"], loop=loop)
    return "x" if b is None else f"{_f(b.toffolis / c_qpe)} [{b.k}]"


def markdown(results: list[dict], command: str, script_s: float) -> str:
    lines = ["# M4 properties: route A against route B and Hellmann-Feynman QPE (U10, U11)", "",
             f"Generated by `{command}` ({script_s:.0f} s, one thread). Costs in units of the system's selected "
             "C_QPE; AE at kappa = 1 (the reference estimator) unless noted; route A on the R1 spin-orbital record "
             "with the classical first step (60 steps); measured loops at r = 1 and 0.3 (T_res = min(direct_mg, "
             "ae_x)); route B with the exact model bias of H' (b_fact) and the leakage share 0.1.", ""]
    rows = []
    for r in results:
        q, m = r["qpe"], r["model"]
        rows.append([r["system"], r["norb"], _f(r["spectrum"]["p_hf"], ".4f"), _f(r["spectrum"]["gap"] * 1e3, ".1f"),
                     f"{q['kind']} {json.dumps(q['params'])}", _f(q["lam"], ".4g"), f"{q['c_qpe']:.4g}",
                     _f(m["e1lb_minus_e0_model"] * 1e3, ".2f"), _f(m["u_cisd_minus_e0_model"] * 1e3, ".2f"),
                     "yes" if m["w_l_terminates"] else "no", r["variance_state"]])
    lines += ["## Systems", ""]
    lines += _table(["system", "N", "p_HF", "Delta_1 (mHa)", "encoding", "lambda", "C_QPE", "E1lb - E0' (mHa)",
                     "CISD - E0' (mHa)", "L forms", "variances at"], rows)
    # U10a: route A at the fixed budgets
    rows = []
    for r in results:
        for kind, blk in r["targets"].items():
            for eps in blk["accuracies"]:
                ra = blk["rows"][str(eps)]["route_a"]
                for lp in LOOPS:
                    cells = []
                    for b in BUDGETS:
                        row = next(x for x in ra if x["loop"] == lp and x["estimator"] == "ae" and x["route"] == "A"
                                   and x["k"] == b)
                        cells.append(_f(row["toffolis_over_qpe"]) + _a_marks(row, eps))
                    orc = next(x for x in ra if x["loop"] == lp and x["estimator"] == "ae" and x["route"] == "A-oracle")
                    rows.append([r["system"], kind, _f(eps, "g"), lp] + cells
                                + [f"{_f(orc['toffolis_over_qpe'])} [{orc['k']}]"])
    lines += ["## U10a: route A at fixed budgets (AE, C_QPE)", "",
              "T_A(k) = C_loop(k) + T_AE at the exact bias of Psi_k; 'x': not usable (bias >= eps); 'e': a budget past "
              "the record that is not costed (k > K in a measured loop, where C_loop(K) only bounds the pre-declared "
              "loop's cost, or past a record that did not converge); both reported, never selected; A-oracle: the "
              "minimum over k with the exact bias [k] (reported only).",
              ""]
    lines += _table(["system", "target", "eps", "loop"] + [f"k = {b}" for b in BUDGETS] + ["A-oracle [k]"], rows)
    # U10b: comparison
    rows = []
    for r in results:
        for kind, blk in r["targets"].items():
            fl = blk["floors"]
            for eps in blk["accuracies"]:
                key = str(eps)
                rb = blk["rows"][key]["route_b"]

                def pick(route, lp, est="ae"):
                    x = [y for y in rb if y["route"] == route and y["loop"] == lp and y["estimator"] == est]
                    return x[0] if x else None
                b0 = pick("B0", "free")
                bk_free, bk_meas = pick("Bk", "free"), pick("Bk", "meas_r1")
                hfq = pick("HF-QPE", "free", "qpe")
                best = blk["best"][key]
                sel = blk["selection"].get(key, {})
                if "error" in sel:
                    enc_b = "F9 (none <= eps/2)"
                elif sel.get("changed"):
                    enc_b = f"{sel['kind']} {json.dumps(sel['params'])} ({_f(sel['ratio'], '.3f')})"
                else:
                    enc_b = "F9"
                comp_rows = blk["rows"][key].get("route_a_compiled")
                rows.append([r["system"], kind, _f(eps, "g"),
                             " / ".join(_f(fl[lab]["min"], ".1e") if lab in fl else "-" for lab in
                                        ("r1_so", "compiled_s1", "compiled_sliced")), enc_b,
                             "-" if b0 is None else f"{_f(b0['toffolis_over_qpe'])} [{b0['k']}]",
                             "-" if bk_free is None else f"{_f(bk_free['toffolis_over_qpe'])} [{bk_free['k']}]",
                             "-" if bk_meas is None else f"{_f(bk_meas['toffolis_over_qpe'])} [{bk_meas['k']}]",
                             "-" if hfq is None else _f(hfq["toffolis_over_qpe"]),
                             _cell(best["free"]["ae"]), _cell(best["meas_r1"]["ae"]), _cell(best["meas_r03"]["ae"]),
                             " / ".join(_layout_cell(best[lp]["ae"]) for lp in LOOPS),
                             _cell(best["free"]["shadow"]),
                             " / ".join(_best_a(comp_rows, lp, r["qpe"]["c_qpe"]) for lp in ("free", "meas_r1")),
                             _f(blk["inputs"]["crossover"])])
    lines += ["## U10b: routes compared (C_QPE)", "",
              "Route-A minimum bias over k <= 60 on R1 exact / compiled (one slice) / compiled (F25 slices on steps "
              "0-2); route B's QPE (F9's selection, or F52's property-aware encoding with its C_QPE ratio); B0 "
              "(start HF or Psi_1, free loop) and Bk (best start with its loop) with AE on coherent copies [start]; "
              "HF-QPE (scalars, free loop; model bias added linearly, field step checked against finite-field "
              "energies); the best selectable row per loop (route [k] cost; AE, and shadows with a free loop) and the "
              "surface-code layout of its longest coherent run at p_fail_ae (code distance d; 'none': no factory and "
              "distance keep the run within the budget, flag no_layout, F53 still compares Toffolis); the "
              "compiled ablation: the cheapest usable route-A row (AE) on the F25-sliced compiled iterates with their "
              "errors and sliced C(Psi~_k), free / r = 1 loop (C_loop(k) from T_res(j) at Psi~_j, j < k, not the "
              "exact record's loop costs; 'x': none usable); B* the shadow/AE crossover (AE is cheaper below it).",
              ""]
    lines += _table(["system", "target", "eps", "A min bias (R1 / s=1 / sliced)", "route-B QPE", "B0 AE [start]",
                     "Bk AE free",
                     "Bk AE meas r=1", "HF-QPE", "best free (AE)", "best meas r=1 (AE)", "best meas r=0.3 (AE)",
                     "layout of the best AE run (free / r=1 / r=0.3)", "best free (shadows)", "A compiled (free / r=1)",
                     "B*"], rows)
    # U11
    rows = []
    for r in results:
        for kind, blk in r["targets"].items():
            fl = blk["floors"]
            cells = []
            for lab in RECORD_LABELS:
                if lab not in fl:
                    cells.append("-")
                    continue
                f = fl[lab]
                keps = ", ".join(f"{e:g}: {f['k_eps'][str(e)] if f['k_eps'][str(e)] is not None else '>60'}"
                                 for e in blk["accuracies"])
                cells.append(f"{_f(f['min'], '.1e')} [{f['k_min']}] ({keps})")
            ck = blk["f47_check"]
            cell = " / ".join(_f(ck[lab]["q_form_min_ratio"], ".3g") + ", "
                              + _f(ck[lab]["energy_form_min_ratio"], ".3g") for lab in ("r1_so", "ls_so"))
            if ck["r1_so"].get("order") == "first":
                cell += (" (first order; all orders " + " / ".join(
                    _f(ck[lab].get("q_form_min_ratio_all_orders"), ".3g") + ", "
                    + _f(ck[lab].get("energy_form_min_ratio_all_orders"), ".3g") for lab in ("r1_so", "ls_so")) + ")")
            cells.append(cell)
            rows.append([r["system"], kind] + cells)
    lines += ["## U11: bias floors", "",
              "Minimum property error over k <= 60 [its k] and k_eps (first k after which the error stays <= eps) "
              "per record; F47: smallest bound/|bias| of the q form and the energy form on R1 / exact line search "
              "(>= 1: the bound holds on every iterate); for the cumulant norm F47 on the linearized operator is first "
              "order, and 'all orders' is B_cum + 3/2 B_1^2 of the cumulant tensor and 1-RDM bounds.", ""]
    lines += _table(["system", "target", "R1 exact", "M3 exact line search", "compiled s = 1", "compiled sliced",
                     "F47 min ratio (q, energy): R1 / LS"], rows)
    rows = []
    for r in results:
        for lab in ("compiled_sliced",):
            rj = r["records"].get(lab)
            if rj:
                rows.append([r["system"], " ".join(map(str, rj["slices"][:3])),
                             " ".join(_f(x, ".3g") for x in (rj.get("slice_errors") or [])[:3]),
                             ", ".join(rj["flags"]) or "-",
                             _f(rj["prep_cumulative"][-1] / max(r["records"]["r1_so"]["prep_cumulative"][-1], 1.0),
                                ".3g")])
    if rows:
        lines += ["F25 slices of the compiled record (steps 0-2), their state errors, and C(Psi~_K)/C(Psi_K).", ""]
        lines += _table(["system", "s (steps 0-2)", "error", "flags", "C ratio"], rows)
    # F52
    rows = []
    for r in results:
        for kind, blk in r["targets"].items():
            for eps, sel in blk["selection"].items():
                if "error" in sel:
                    rows.append([r["system"], kind, eps, _f(blk["model_bias"], ".2e"), "-", "-", "-", sel["error"]])
                    continue
                rows.append([r["system"], kind, eps, _f(blk["model_bias"], ".2e"),
                             f"{sel['kind']} {json.dumps(sel['params'])}", _f(sel["ratio"], ".4f"),
                             _f(sel["b_fact"], ".2e"), "changed" if sel["changed"] else "-"])
    lines += ["## F52: property-aware encoding selection", "",
              "b_fact of the selected (F9) encoding; the cheapest scanned encoding with b_fact <= eps/2 "
              "(|de_fact| <= 0.5 mHa first) and its C_QPE relative to the selected one.", ""]
    lines += _table(["system", "target", "eps", "b_fact (F9)", "property-aware", "C_QPE ratio", "b_fact", "note"],
                    rows)
    for r in results:
        ff = r.get("finite_field")
        if ff:
            lines += [f"## Finite field ({r['system']} dipole, {FIELD_POINTS} fields on [-{FIELD_MAX}, {FIELD_MAX}] "
                      f"a.u., degree {FIELD_DEGREE})", "",
                      f"dE/dh = {ff['d1']:.6f} (<mu> at psi_0 {ff['mu_fci']:.6f}, at HF {ff['mu_hf']:.6f}); "
                      f"E'' = {ff['d2']:.4g}, E''' = {ff['d3']:.4g}, E^(4) = {ff['d4']:.4g}, E^(5) = {ff['d5']:.4g} "
                      "a.u.", ""]
            steps = ((r.get("targets") or {}).get("dipole") or {}).get("hf_steps") or {}
            for key, per in steps.items():
                for st, v in per.items():
                    if not (v["used"] and v["feasible"]):
                        continue
                    text = (f"HF-QPE field step at eps = {key} a.u. ({st.replace('_', '-')} stencil, b_fact "
                            f"{_f(v['fixed_bias'], '.3e')} a.u. added linearly): b = {_f(v['stencil_bias'], '.3e')} "
                            f"a.u.; F51's leading-order h = {_f(v['h_leading'], '.3e')} a.u.")
                    if v["stencil_bias_at_leading"] is not None:
                        at_lead = abs(v["stencil_bias_at_leading"]) / v["stencil_bias"]
                        at_h = abs(v["stencil_bias_at_h"]) / v["stencil_bias"]
                        text += (f" (finite-field stencil bias {_f(at_lead, '.4g')} b), refined h = "
                                 f"{_f(v['h'], '.3e')} a.u. ({_f(at_h, '.4g')} b)")
                    lines += [text + ".", ""]
    rows = [[r["system"], _f(r["runtime_s"], ".1f")] + [_f(r["timings"].get(k, 0.0), ".1f") for k in
            ("system_inputs", "qpe_side", "records", "residual_costs", "compiled", "estimator_inputs", "scan",
             "finite_field", "routes")] + [_f(r["peak_rss_mb"], ".0f")] for r in results]
    lines += ["## Runtime (s, one thread)", ""]
    lines += _table(["system", "total", "inputs", "QPE side", "records", "loop costs", "compiled", "estimator inputs",
                     "scan", "finite field", "routes", "peak RSS (MB)"], rows)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default=None, help="comma-separated F20 names (default: the property set)")
    ap.add_argument("--quick", action="store_true", help=f"systems {','.join(QUICK_SYSTEMS)} (unless --systems)")
    ap.add_argument("--out", default="results/m4", help="output directory (cache in <out>/cache)")
    ap.add_argument("--no-cache", action="store_true", help="recompute and overwrite cached stages")
    args = ap.parse_args(argv)
    t_script = time.perf_counter()
    if args.systems:
        names = [s.strip() for s in args.systems.split(",") if s.strip()]
    else:
        names = list(QUICK_SYSTEMS if args.quick else DEFAULT_SYSTEMS)
    for n in names:
        if n not in BENCHMARK_SYSTEMS:
            benchmark_system(n)                  # ValueError for an unknown name
    os.makedirs(args.out, exist_ok=True)
    cache = scaling.Cache(os.path.join(args.out, "cache"), refresh=args.no_cache)
    params = FTParameters()
    results = []
    for name in names:
        print(f"[{name}] start", flush=True)
        res = run_one(name, cache, params)
        with open(os.path.join(args.out, f"properties_{name}.json"), "w") as fh:
            json.dump(scaling.to_json(res), fh, allow_nan=False)
        results.append(res)
        best = {kind: _cell(blk["best"][str(blk["accuracies"][-1])]["meas_r1"]["ae"])
                for kind, blk in res["targets"].items()}
        print(f"[{name}] {res['runtime_s']:.1f} s; C_QPE {res['qpe']['c_qpe']:.4g} ({res['qpe']['kind']}); best with "
              f"the r = 1 loop at the tighter eps: {best}; peak RSS {res['peak_rss_mb']:.0f} MB", flush=True)
    command = ("benchmarks/run_m4_properties.py" + (" --quick" if args.quick else "")
               + (f" --systems {','.join(names)}" if args.systems else "") + f" --out {os.path.relpath(args.out)}"
               + (" --no-cache" if args.no_cache else ""))
    script_s = time.perf_counter() - t_script
    with open(os.path.join(args.out, "properties_summary.md"), "w") as fh:
        fh.write(markdown(results, command, script_s))
    print(f"wrote {len(results)} JSON files and properties_summary.md to {args.out} ({script_s:.1f} s; cache hits "
          f"{cache.hits}, misses {cache.misses})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

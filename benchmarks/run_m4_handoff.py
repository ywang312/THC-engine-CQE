"""The M4 hand-off: rule B on the model spectrum of H' (U8) and the QPE-side schemes on exact records (U9, part 1).

Units: Hartree, Toffolis; "/ C_QPE" columns divide by the system's selected C_QPE (F9). dE = 1 mHa.

Per system of ``--systems`` (F20 names; M3's names resolve to M3's builds):

QPE side and certificates (F39-F42)
    The selected encoding and the system inputs come from ``ft.closedloop.system_inputs`` (the 12-root
    spectrum, ``ft.qpe.scan_encodings`` + ``select_qpe``, the allowed spin-free elements with their
    per-shot variances and LCU normalizations, the DF split of the energy). The model spectrum of H'
    (``ft.acceptance.model_spectrum``, dense, N <= 8) gives E_0', the sector levels
    (``ft.certify.sector_levels``), the walk offset c, sin(theta_0), M = 2 n_iter (and 4 n_iter for the
    Kaiser window) and the grid g = 2 pi lambda sin(theta_0)/M (F39). E1lb = E_1' - 0.1 mHa with E_1' the
    lowest excited singlet of the ground state's irrep, or of any irrep for THC encodings and wherever
    the eigenvectors of H' leave their irrep by more than 1e-6 (F41; flag "point_group_broken").
    U = CISD(H') (``ft.certify.truncated_ci_energy``); CISDT(H') is reported and not used (F42).

Records (noise-free emulation)
    exact_so, exact_sf: ``exact_trajectory`` (L-BFGS with the spin-orbital and spin-free metric, the
    classical first step of F16, 6 steps); r1_so, r1_sf: the unit-step rule R1
    (``steprules.run_step_rule``, the same metrics, 6 steps, ``ft.steprules.to_trajectory_record``);
    compiled_so: the R1 spin-orbital steps 0-2 compiled elementwise from HF (``ft.circuits.replay_elementwise``,
    one slice, F3). Preparation costs C(Psi_k) and Q(Psi_k) by ``ft.stateprep.prep_cost``. Loop costs
    C_iter(k), k <= 2 (so C_loop(k) for k <= 3): F30 (``ft.steprules.step_rule_iteration_costs``) at
    r = 0.3, 1, 3 for the R1 records (the compiled record takes those of r1_so, flag
    "exact_state_loop_costs"), and M3's F14 (``ft.pipeline.iteration_costs``, the 11-candidate line
    search, r = 0.3) for the exact records.

U8 (rule B, F40-F42)
    Per state HF, Psi_1 (both metrics), Psi_2 and Psi_3 of every record and the compiled Psi~_1..Psi~_3:
    rule B's choice by ``ft.handoff.rule_b_table`` (loop "classical_first_step"; R1 loop costs at r = 1)
    with p the weight on psi_0' of H' (as U9 and F43's table; M3's T6 below keeps p on H): the form
    (B_U / B_L of ``ft.acceptance.acceptance_bracket``, and B_U with more runs where B_L does not
    terminate), the certificate (CISD(H') or a dedicated sampled or AE energy of an iterate,
    ``ft.certify.energy_certificates`` with z = 3; a dedicated energy costed by the expectation of its
    verification runs over the realized bound, the planned U, theta and n_v reported with the expected
    n_v and certificate cost), the margin E1lb - U - 2 dE, D = (E1lb - theta)/g, n_v, the certificate
    cost. The identification error of the accepted state from the exact windowed laws on the full
    (n_alpha, n_beta) sector of H' (``ft.acceptance.acceptance_stats``: sine at M = 2 n_iter, Kaiser(3) at
    4 n_iter; one run and n_v runs) with its kernel term and the logical-failure term at the F40 budgets
    (p_run = eps_fail p_min / f for an unverified run, eps_fail / (2 f) for the last of n_v >= 2 runs,
    f = 1 - theta_0/pi, so that f p_run = 1e-6 and 5e-4); the design-floor bound
    [(1 - p_min)/p_min] tau_W(D)^{n_v} and the state's bound [(1 - p_0)/p_0] [tau_W(D)/K_0]^{n_v}
    (``ft.acceptance.identification_bound``), whose premise is that every populated excited level lies
    at or above E1lb; for compiled states the contaminant weight q_c below theta
    (``ft.certify.contaminant_weight``) and F41's budget q_c/(p_0 K_0^{n_v}) (the kernel term already
    holds those levels exactly, since the weights cover every spin). Where a state puts more than 1e-12
    on excited levels below E1lb the design-floor bound does not apply ("n/a (q_c)" where F41's
    contaminants below theta carry that weight, else "n/a (below E1lb)") and the state's bound is the
    kernel bound plus the weight of those levels over p_0 K_0^{n_v} (accepted with K_j <= 1:
    q_c/(p_0 K_0^{n_v}) where they are the contaminants below theta); M3's rule-E verdict
    (E_1 - E_k > 3 dE with E_1 of H).

U9 part 1 (F43, F44)
    Per record, state and r: T under every scheme of F43 by ``ft.amplify.state_scheme_costs`` with p the
    weight of the state on psi_0' (H', the QPE's Hamiltonian; M3's T6 uses the overlap with psi_0 of H)
    and C_loop(k): rep under M3's rule E (C_loop + (C(Psi_k) + C_QPE)/p, no verification run) and under
    F40; rep, aa, bbht and fp accepted by rule B's choice for the row (its expected n_v - 1 verification
    runs, its certificate's expected cost and the loop to the certified iterate, the same option for the
    four schemes since it enters additively), flagged with its bracket's flags; fp with M3's
    Eckart-Temple p_lb (for compiled states with the lowest populated excited level of H' of any spin,
    less F41's 0.1 mHa, in place of E_1; None where the state's energy lies above it); lt, lt_bbht, 2s
    with the cheapest certified U_k among CISD(H') and the dedicated energies of Psi_k (their cost added,
    one call of ``state_scheme_costs`` per certificate), filter degree from ``ft.amplify.filter_cost``
    (eps_f = 1e-2) and the coarse QPE from ``coarse_qpe_toffolis`` (eps_p = 1e-4), their final QPE B_U at
    D_U >= 1.3 (``ft.amplify.final_qpe_form``, else B_L, flagged); lt_bbht_L and 2s_L with the pass edge
    E1lb - w_L, w_L = 100 mHa, only where E_0' <= E1lb - w_L (else "not_terminating_L"), flagged
    "termination_not_certified" unless CISD(H') <= E1lb - w_L; without a certified U the lower bounds
    "<scheme>@oracle_U" at U = E_0' + 2 dE. The best realizable scheme s* is
    ``ft.amplify.best_realizable``, the row's flags are those of s*, and every scheme's flags are kept;
    the sensitivity keys lt_rep and lt_rep_L (the step filter followed by repetition) and the optimum
    over the set extended by them (``ft.amplify.best_extended``) are reported, never s*. Compiled rows
    carry "contaminant_below_theta" and q_c/(p_0 K_0^n) per scheme: at rule B's threshold (n = n_v) for
    rep, aa, bbht and fp, at the final QPE's threshold (n = 1) for the filtered schemes. Savings of the
    first and second measured iteration per scheme: T_s(k) - T_s(k + 1), k = 1, 2 (C_QPE) and relative
    to T_s(k), and at s* and at the extended optimum (``ft.amplify.iteration_verdict``,
    ``extended_verdict``). U9b: the break-even p_0* of F44 (``ft.amplify.break_even_overlap``) for C_it in
    {0.1, 1, 10, 100, 1000} C_QPE, for a perfect state (p_1 = 1, C(Psi_1) = 0) and for the R1
    spin-orbital Psi_2.

M3's T6 under rule B (M3's systems h4, h6a, h6s, h8, lih, n2, n2s)
    With ``--m3-results DIR`` the records, preparation costs, iteration costs and hand-off rows of M3
    come from DIR/ft_<system>.json; the JSON omits CI vectors and model integrals, so the selected
    encoding is rebuilt from its kind and parameters (``ft.qpe.sf_encoding`` / ``df_encoding`` /
    ``thc_encoding``) and the L-BFGS records with their states by ``exact_trajectory`` (checked against
    the JSON: energies, overlaps, C(Psi_k), the T_E rows and the encoding's lambda and C_QPE; a mismatch
    is flagged). Without the flag M3's pipeline is rerun (``ft.pipeline.run_system``, flag
    "m3_recomputed"). Per family (lbfgs, lbfgs_so) and loop (measured, classical_first_step, free):
    M3's certified rule-E optimum, M3's oracle optimum and rule B's optimum
    (``ft.handoff.rule_b_table``, ``optimal_rule_b``; certificates CISD(H') and the dedicated energies of
    every iterate; p on H as M3).

Outputs: <out>/handoff_<system>.json (strict JSON: nan as null, +-inf as "inf" / "-inf", via
``ft.scaling.to_json``) and <out>/handoff_summary.md. Expensive stages (system inputs, model spectrum,
truncated CI, records, loop costs, compiled states, M3's records) are cached per system in
<out>/cache (``ft.scaling.Cache``; ``--no-cache`` recomputes and overwrites). One thread per process
(the thread variables are set before numpy is imported).

Usage:
    python benchmarks/run_m4_handoff.py --quick --m3-results /home/user/thc-engine-cqe/results
    python benchmarks/run_m4_handoff.py --systems n2_r2.4,n2_r2.8 --out results/m4
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
from pyscf.fci import spin_op  # noqa: E402

from thc_engine_cqe import steprules  # noqa: E402
from thc_engine_cqe.diagnostics.thc_floors import ExactReference  # noqa: E402
from thc_engine_cqe.fci_ops import Hamiltonian, inner, normalize  # noqa: E402
from thc_engine_cqe.ft import acceptance as acc  # noqa: E402
from thc_engine_cqe.ft import amplify, certify, closedloop, handoff, pipeline, scaling  # noqa: E402
from thc_engine_cqe.ft import qpe as ftqpe  # noqa: E402
from thc_engine_cqe.ft import residual_inputs as ri  # noqa: E402
from thc_engine_cqe.ft import steprules as ft_steprules  # noqa: E402
from thc_engine_cqe.ft.circuits import replay_elementwise  # noqa: E402
from thc_engine_cqe.ft.primitives import FTParameters  # noqa: E402
from thc_engine_cqe.ft.stateprep import prep_cost  # noqa: E402
from thc_engine_cqe.ft.symmetry import BENCHMARK_SYSTEMS, benchmark_system, determinant_irreps  # noqa: E402
from thc_engine_cqe.ft.trajectory import exact_trajectory  # noqa: E402
from thc_engine_cqe.residual import compute_residual  # noqa: E402

M3_SYSTEMS = ("h4", "h6a", "h6s", "h8", "lih", "n2", "n2s")
STATE_SYSTEMS = scaling.OVERLAP_AXIS + ("h4_r3.5", "h4_r4.0", "h6_r3.5", "h6_r4.0", "h8_r3.0", "h8_r4.0",
                                        "n2v_r2.0", "n2v_r2.4", "n2v_r2.8")
DEFAULT_SYSTEMS = M3_SYSTEMS + STATE_SYSTEMS
QUICK_SYSTEMS = ("h4", "n2s", "n2_r2.4", "n2_r2.8")
N_STEPS = 6                     # steps of the exact and R1 records
K_STATES = 3                    # hand-off states Psi_0..Psi_3 (C_loop(3) needs C_iter(0..2))
R_VALUES = (0.3, 1.0, 3.0)      # residual accuracies of the R1 loop costs (F30)
R_RULE_B = 1.0                  # the R1 loop costs behind U8's rule-B choice
C_IT = (0.1, 1.0, 10.0, 100.0, 1000.0)   # U9b iteration costs (C_QPE)
EPS_F = 1e-2                    # step-filter error (F43)
EPS_P = 1e-4                    # coarse-QPE leakage (F43)
KAISER_ALPHA = 3.0
D_L = 5                         # B_L's top margin in grid units (F40)
IMPURITY_TOL = 1e-6             # eigenvector weight outside the dominant irrep that counts as broken symmetry
PREMISE_TOL = 1e-12             # weight on excited levels below E1lb (U8's bounds) or below theta (q_c) that counts
POPULATED_TOL = 1e-14           # weight of a level that counts as populated (fp's bound on compiled states)
TAIL_DMAX = 1000.0              # grid units: tau_W is non-increasing, so its value here bounds it beyond
M3_FAMILIES = ("lbfgs", "lbfgs_so")
METRICS = {"so": "spin_orbital", "sf": "spin_free"}
STATE_NAMES = ("HF", "Psi1", "Psi2", "Psi3")


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far (MB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _finite_or_none(x):
    return None if x is None or not math.isfinite(float(x)) else float(x)


def _qpe_key(params: FTParameters) -> dict:
    return {"params": dataclasses.asdict(params)}


# ------------------------------------------------------------------ the QPE side and certificates
def qpe_side(name: str, inputs, cache, params: FTParameters, *, qpe=None, stage_suffix: str = "") -> dict:
    """Model spectrum, E1lb, grid, sector levels, CISD/CISDT and the filter costs of one encoding (module docstring)."""
    active, irreps = inputs.active, inputs.irreps
    q = inputs.qpe if qpe is None else qpe
    enc = q.encoding
    key = {**_qpe_key(params), "kind": enc.kind, "enc_params": scaling.to_json(enc.params)}
    spec = cache.get(name, "model_spectrum" + stage_suffix, lambda: acc.model_spectrum(active, q, irreps), key=key)
    model = enc.model(active)
    u2 = cache.get(name, "cisd" + stage_suffix, lambda: certify.truncated_ci_energy(model, 2), key=key)
    u3 = cache.get(name, "cisdt" + stage_suffix, lambda: certify.truncated_ci_energy(model, 3), key=key)
    levels = certify.sector_levels(spec, det_irreps=determinant_irreps(active.norb, tuple(active.nelec), irreps))
    flags = []
    broken = enc.kind == "thc" or levels["irrep_impurity"] > IMPURITY_TOL
    if broken and enc.kind != "thc":
        flags.append("point_group_broken")
    e1_ref = min(spec.e1_singlet, spec.e1_any_irrep) if broken else spec.e1_singlet
    e1lb = certify.e1_lower_bound(e1_ref, model_spectrum=True)
    de = float(q.de_qpe)
    m2, m4 = acc.qpe_points(q), acc.qpe_points(q, 4)
    grid = spec.grid(m2)
    c = float(spec.c)
    side = {"qpe": q, "enc": enc, "spec": spec, "model": model, "u": float(u2), "u_cisdt": float(u3),
            "e1lb": float(e1lb), "e1_ref": float(e1_ref), "e1_ref_level": "e1_any_irrep" if broken else "e1_singlet",
            "levels": levels, "m2": m2, "m4": m4, "grid": float(grid), "c": c, "de": de, "flags": flags}
    # filter costs (F43): U = CISD(H') where it certifies, the E1lb-referenced forms, the oracle U = E0' + 2 dE
    certified_u = u2 <= e1lb - 2.0 * de
    side["u_certified"] = bool(certified_u)
    side["filters"] = {"u": _filters(enc, c, u2, e1lb, params) if certified_u else None,
                       "w_l": (_filters(enc, c, e1lb - amplify.W_L, e1lb, params)
                               if spec.e0 <= e1lb - amplify.W_L else None),
                       "oracle_u": (_filters(enc, c, spec.e0 + 2.0 * de, e1lb, params)
                                    if not certified_u and spec.e0 + 2.0 * de < e1lb else None)}
    side["w_l_certified"] = bool(certified_u and u2 <= e1lb - amplify.W_L)
    # B_L without U (F40): its termination, and the final QPE of the filtered schemes with U = CISD(H') (F43)
    side["theta_l"] = float(e1lb - D_L * grid)
    side["b_l_terminates"] = bool(spec.e0 + de <= side["theta_l"])
    side["u_final"] = amplify.final_qpe_form(u2, e1lb, de, grid=grid) if certified_u else None
    side["filter_cache"] = {}
    return side


def _filters(enc, c: float, u: float, e1lb: float, params: FTParameters) -> dict:
    fc = amplify.filter_cost(enc, c, u, e1lb, eps=EPS_F, params=params)
    cc = amplify.coarse_qpe_toffolis(enc, c, u, e1lb, eps_p=EPS_P)
    return {"u": float(u), "degree": int(fc.degree), "c_filt": float(fc.toffolis), "c_coarse": float(cc),
            "filter_qubits": int(fc.logical_qubits)}


def _filters_at(side: dict, u: float, params: FTParameters) -> dict:
    key = round(float(u), 12)
    if key not in side["filter_cache"]:
        side["filter_cache"][key] = _filters(side["enc"], side["c"], u, side["e1lb"], params)
    return side["filter_cache"][key]


def side_summary(side: dict, inputs) -> dict:
    """The JSON block of the QPE side."""
    spec, q, enc, lv = side["spec"], side["qpe"], side["enc"], side["levels"]
    e0m, de, g = float(spec.e0), side["de"], side["grid"]
    margin = side["e1lb"] - side["u"] - 2.0 * de
    d_u = (side["e1lb"] - side["u"] - de) / g
    try:
        nv_u = acc.verification_runs(d_u) if d_u > 0 else None
    except ValueError:
        nv_u = None
    out = {"kind": enc.kind, "params": enc.params, "lam": enc.lam, "c_w": int(enc.toffoli_per_step),
           "n_iter": int(q.iterations), "c_qpe": float(q.toffolis), "logical_qubits": int(q.logical_qubits),
           "de_fact": enc.de_fact, "infidelity": enc.infidelity, "walk_offset": side["c"],
           "sin_theta0": float(spec.sin_theta0), "theta0": float(spec.theta0), "m_2n": side["m2"], "m_4n": side["m4"],
           "grid": g, "e0_model": e0m, "e0_model_minus_e0": e0m - float(inputs.spectrum.e0),
           "e1_singlet": float(spec.e1_singlet), "e1_any_spin": float(spec.e1_any_spin),
           "e1_any_irrep": float(spec.e1_any_irrep), "e1lb": side["e1lb"], "e1lb_level": side["e1_ref_level"],
           "e1lb_minus_e0_model": side["e1lb"] - e0m,
           "e1_model_minus_e1": float(spec.e1_singlet) - float(inputs.spectrum.e1),
           "levels": {k: v for k, v in lv.items()},
           "u_cisd": side["u"], "u_cisd_minus_e0_model": side["u"] - e0m, "u_cisdt": side["u_cisdt"],
           "u_cisdt_minus_e0_model": side["u_cisdt"] - e0m, "u_certified": side["u_certified"],
           "margin_cisd": margin, "d_u": d_u, "n_v_u": nv_u, "n_v_l": acc.verification_runs(D_L),
           "tau_d_l": acc.tail_mass(float(D_L), window="sine"),
           "w_l": amplify.W_L, "w_l_terminates": side["filters"]["w_l"] is not None,
           "w_l_certified": side["w_l_certified"], "theta_l_minus_e0_model": side["theta_l"] - e0m,
           "b_l_terminates": side["b_l_terminates"], "u_final": side["u_final"], "filters": side["filters"],
           "flags": side["flags"]}
    c_qpe = float(q.toffolis)
    for key, val in side["filters"].items():
        if val is not None:
            out[f"c_filt_{key}_over_qpe"] = val["c_filt"] / c_qpe
            out[f"c_coarse_{key}_over_qpe"] = val["c_coarse"] / c_qpe
    return out


# ------------------------------------------------------------------ records
def build_records(name: str, inputs, cache, params: FTParameters) -> dict:
    """exact_so, exact_sf, r1_so, r1_sf (module docstring): {label: (TrajectoryRecord, StepRuleResult | None)}."""
    active, irreps, spec = inputs.active, inputs.irreps, inputs.spectrum

    def compute():
        out = {}
        for tag, metric in METRICS.items():
            out[f"exact_{tag}"] = (exact_trajectory(active, optimizer="lbfgs", metric=metric, max_iter=N_STEPS,
                                                    system=name, irreps=irreps, n_roots=spec.n_roots,
                                                    first_step="second_order", roots=spec.roots), None)
        for tag, metric in METRICS.items():
            res = steprules.run_step_rule(active, steprules.StepRule("unit"), max_iter=N_STEPS, metric=metric,
                                          irreps=irreps, keep_states=True)
            rec = ft_steprules.to_trajectory_record(res, system=name, active=active, irreps=irreps, spectrum=spec)
            out[f"r1_{tag}"] = (rec, res)
        return out

    return cache.get(name, "handoff_records", compute, key={"n_steps": N_STEPS, "metrics": list(METRICS.values())})


def loop_costs(name: str, inputs, records: dict, preps: dict, cache, params: FTParameters) -> dict:
    """{(label, r): [C_iter(k), k = 0..2]} with the residual's route and cost (module docstring)."""
    def summary(costs):
        return [{"k": c.k, "toffolis": float(c.toffolis), "residual_toffolis": float(c.residual_toffolis),
                 "best": c.best,
                 "line_search_toffolis": None if c.line_search is None else float(c.line_search.toffolis),
                 "residual_norm_m": float(c.residual_norm_m)} for c in costs]

    def compute():
        out = {}
        enc = inputs.qpe.encoding
        for label, (rec, res) in records.items():
            k_max = min(K_STATES - 1, rec.n_steps - 1)
            if res is not None:
                for r in R_VALUES:
                    out[label, r] = summary(ft_steprules.step_rule_iteration_costs(
                        inputs.active, rec, res, preps[label], enc, elements=inputs.elements,
                        variances=inputs.variances, normalizations=inputs.normalizations,
                        energy_variance=inputs.energy_split, readout_toffolis=inputs.readout_toffolis, r=r,
                        k_max=k_max, params=params))
            else:
                out[label, params.r] = summary(pipeline.iteration_costs(
                    inputs.active, rec, preps[label], enc, elements=inputs.elements, variances=inputs.variances,
                    normalizations=inputs.normalizations, energy_variance=inputs.energy_split,
                    readout_toffolis=inputs.readout_toffolis, classical_first_step=True, k_max=k_max,
                    params=params))
        return out

    return cache.get(name, "handoff_costs", compute, key={"r_values": list(R_VALUES), "k_states": K_STATES,
                                                           "params": dataclasses.asdict(params)})


def compiled_record(name: str, inputs, rec_so, cache, params: FTParameters):
    """The R1 spin-orbital record with Psi~_0..Psi~_3 compiled from HF (energies, variances, overlaps replaced)."""
    def compute():
        k = min(K_STATES, rec_so.n_steps)
        states = replay_elementwise(inputs.active, rec_so.directions[:k], rec_so.eps[:k], irreps=inputs.irreps,
                                    params=params)
        return [normalize(np.asarray(s)) for s in states]

    states = cache.get(name, "handoff_compiled", compute, key={"k_states": K_STATES,
                                                                "params": dataclasses.asdict(params)})
    ham = Hamiltonian(inputs.active)
    rows = [compute_residual(inputs.active, s, ham) for s in states]
    psi0 = inputs.spectrum.psi0
    k = len(states) - 1
    return dataclasses.replace(
        rec_so, energies=np.array([r.energy for r in rows]), variances=np.array([r.variance for r in rows]),
        residual_norms=np.array([r.residual_norm for r in rows]),
        overlaps=np.array([abs(complex(inner(psi0, s))) ** 2 for s in states]), eps=np.asarray(rec_so.eps[:k]),
        directions=list(rec_so.directions[:k]), states=states,
        flags=tuple(rec_so.flags) + ("compiled_elementwise", "exact_state_loop_costs"))


# ------------------------------------------------------------------ U8: rule B per state
def certificates_for(rec, prep, inputs, side) -> list:
    """CISD(H') and the dedicated sampled and AE energies of Psi_0..Psi_3 (F42)."""
    n = min(len(rec.states), K_STATES + 1)
    var = np.full(len(rec.states), np.nan)
    for j in range(n):
        var[j] = ri.energy_variances(inputs.active, rec.states[j])[inputs.energy_split]
    return certify.energy_certificates(rec, prep, e1_lower=side["e1lb"], de=side["de"], classical={"cisd": side["u"]},
                                       energy_variances=var, readout_toffolis=inputs.readout_toffolis,
                                       walk=side["enc"], e0_model=float(side["spec"].e0))


def _stats(spec, w, theta, m, window, n_runs, p_run) -> acc.AcceptanceStats:
    return acc.acceptance_stats(spec, w, theta, n_points=m, window=window, alpha=KAISER_ALPHA, de=1e-3,
                                n_runs=n_runs, p_run=p_run)


def _stats_json(st: acc.AcceptanceStats) -> dict:
    return {"eps_id": st.identification_error, "kernel": st.identification_error - st.logical,
            "logical": st.logical, "accept": st.accept, "ground": st.ground, "k0": st.k0,
            "energy_error": st.energy_error}


def u8_row(state: str, label: str, k: int, ci, rb: handoff.RuleBRow, side: dict, inputs, *, compiled: bool,
           p_h: float) -> dict:
    """One U8 row (module docstring); ``rb`` costed with p on H', ``p_h`` the overlap with psi_0 of H."""
    spec, g, e1lb, de = side["spec"], side["grid"], side["e1lb"], side["de"]
    c_qpe = float(side["qpe"].toffolis)
    w = spec.weights(ci)
    p0 = float(w[0])
    f = 1.0 - spec.theta0 / math.pi
    n_v, theta = rb.n_runs, rb.theta
    distance = (e1lb - theta) / g
    p_run_1 = acc.failure_budget("unverified", f=f)
    p_run_n = acc.failure_budget("unverified" if n_v == 1 else "retested", f=f)
    laws = {}
    for law, (m, window) in (("sine_2n", (side["m2"], "sine")), ("kaiser_4n", (side["m4"], "kaiser"))):
        one = _stats(spec, w, theta, m, window, 1, p_run_1)
        nv = one if n_v == 1 else _stats(spec, w, theta, m, window, n_v, p_run_n)
        laws[law] = {"one_run": _stats_json(one), "n_v_runs": _stats_json(nv)}
    tau = acc.tail_mass(min(distance, TAIL_DMAX), window="sine") if distance > 0 else 1.0
    k0 = laws["sine_2n"]["n_v_runs"]["k0"]
    # the forms side by side (kernel terms only): B_U at U = CISD(H') where it certifies (sine and rect at
    # 2 n_iter, Kaiser(3) at 4 n_iter, one run), B_L at E1lb - D_L g (sine, one and two runs), and the literal
    # E1-referenced threshold E_1' - dE (sine, one run), which F40 rejects
    forms = {}
    if side["u_certified"]:
        th = side["u"] + de
        forms["b_u_cisd"] = {"theta_minus_e0_model": th - float(spec.e0),
                             "sine_2n": _stats(spec, w, th, side["m2"], "sine", 1, 0.0).identification_error,
                             "kaiser_4n": _stats(spec, w, th, side["m4"], "kaiser", 1, 0.0).identification_error,
                             "rect_2n": _stats(spec, w, th, side["m2"], "rect", 1, 0.0).identification_error}
    th = e1lb - D_L * g
    forms["b_l"] = {"theta_minus_e0_model": th - float(spec.e0),
                    "sine_2n_one_run": _stats(spec, w, th, side["m2"], "sine", 1, 0.0).identification_error,
                    "sine_2n_two_runs": _stats(spec, w, th, side["m2"], "sine", 2, 0.0).identification_error}
    th = float(spec.e1_singlet) - de
    forms["literal"] = {"theta_minus_e0_model": th - float(spec.e0),
                        "sine_2n": _stats(spec, w, th, side["m2"], "sine", 1, 0.0).identification_error}
    cert = rb.certificate
    # the bounds' premise: every populated excited level at or above E1lb; populated levels below it (the
    # contaminants below theta of F41, q_c, and any other) are accepted with K_j <= 1, so the state's bound adds
    # their weight over p_0 K_0^n_v (q_c/(p_0 K_0^n_v) where they are the contaminants below theta)
    q_c = certify.contaminant_weight(spec, ci, theta)
    below_e1lb = float(np.sum(w[1:][spec.energies[1:] < e1lb]))
    premise = below_e1lb <= PREMISE_TOL
    if p0 > 0 and k0 > 0:
        bound_state = acc.identification_bound(p0, tau, k0, n_v) + (0.0 if premise else below_e1lb / (p0 * k0 ** n_v))
    else:
        bound_state = None
    flags = list(rb.flags) + (["exact_state_loop_costs"] if compiled else [])
    row = {"state": state, "record": label, "k": k, "compiled": compiled, "p": p_h, "p_model": p0,
           "e_err": float(rb.e_err), "rule_e_certified": handoff.rule_e_certifiable(
               float(inputs.spectrum.e0) + float(rb.e_err), float(inputs.spectrum.e1), de),
           "form": rb.form, "theta": theta, "theta_minus_e0_model": theta - float(spec.e0), "distance": distance,
           "n_v": n_v, "n_v_expected": rb.n_runs_expected, "certified": rb.certified,
           "terminates": math.isfinite(rb.cost), "flags": list(dict.fromkeys(flags)),
           "u_source": None if cert is None else cert.source, "u_label": None if cert is None else cert.label,
           "u": None if cert is None else cert.upper, "u_iterate": None if cert is None else cert.loop_iterate,
           "u_sigma": None if cert is None else cert.sigma,
           "margin": None if cert is None else e1lb - cert.upper - 2.0 * de,
           "certificate_toffolis": 0.0 if cert is None else cert.toffolis,
           "certificate_over_qpe": 0.0 if cert is None else cert.toffolis / c_qpe,
           "certificate_expected_over_qpe": rb.certificate_cost / c_qpe,
           "cost_over_qpe": rb.cost / c_qpe, "cost_oracle_e_over_qpe": rb.cost_oracle_e / c_qpe,
           "laws": laws, "forms": forms, "f": f, "p_run_unverified": p_run_1, "p_run_n_v": p_run_n,
           "tau": tau, "excited_below_e1lb": below_e1lb, "bound_premise": premise,
           "bound_p_min": (1.0 - acc.P_MIN) / acc.P_MIN * tau ** n_v if premise else None,
           "bound_state": bound_state}
    if compiled:
        row["contaminant"] = q_c
        row["contaminant_budget"] = q_c / (p0 * k0 ** n_v) if p0 * k0 > 0 else None
        row["s2"] = float(spin_op.spin_square0(np.asarray(ci), inputs.active.norb, tuple(inputs.active.nelec))[0])
    return row


def bound_cell(u8: dict) -> str:
    """U8's 'bound' cell: the design-floor bound, or why its premise fails ("n/a (q_c)": F41's contaminants below
    theta; "n/a (below E1lb)": populated excited levels below E1lb, none of them a contaminant below theta)."""
    if u8["bound_p_min"] is not None:
        return _f(u8["bound_p_min"], ".2e")
    return "n/a (q_c)" if (u8.get("contaminant") or 0.0) > PREMISE_TOL else "n/a (below E1lb)"


# ------------------------------------------------------------------ U9: schemes per state
def _lowest_populated_level(spec, w: np.ndarray) -> float | None:
    """The lowest excited level of H' (index >= 1) with weight above POPULATED_TOL, any spin and irrep (or None)."""
    idx = np.flatnonzero(w[1:] > POPULATED_TOL) + 1
    return float(spec.energies[idx[0]]) if idx.size else None


def scheme_row(state: str, label: str, k: int, ci, rb: handoff.RuleBRow, c_loops: list[float], rec, certs,
               side: dict, params: FTParameters, *, p_h: float, compiled: bool = False) -> dict:
    """T_s(k) under every scheme of F43 with its flags (module docstring), in Toffolis; p on H', ``p_h`` on H."""
    spec, e1lb, de, grid = side["spec"], side["e1lb"], side["de"], side["grid"]
    c_qpe = float(side["qpe"].toffolis)
    w = spec.weights(ci)
    p = min(float(w[0]), 1.0)
    c_prep, c_loop = float(rb.c_prep), float(rb.c_loop)
    if compiled:
        # compiled states populate levels of any spin below E_1 (F41): the bound refers to the lowest populated one
        e1_pop = _lowest_populated_level(spec, w)
        p_lb = (math.nan if e1_pop is None else handoff.overlap_lower_bound(
            float(rec.energies[k]), float(rec.variances[k]), certify.e1_lower_bound(e1_pop, model_spectrum=True)))
    else:
        p_lb = handoff.overlap_lower_bound(float(rec.energies[k]), float(rec.variances[k]), float(rec.e1))
    cert = rb.certificate                        # rule B's option: the loop to the certified iterate and C_U
    terminates = math.isfinite(rb.cost)
    j_u = 0 if cert is None else cert.loop_iterate
    rule_b = dict(n_verify=rb.n_runs_expected, rule_b_terminates=terminates, rule_b_flags=rb.bracket_flags,
                  rule_b_extra=(c_loops[max(k, j_u)] - c_loops[k]) + rb.certificate_cost if terminates else 0.0)
    common = dict(filt_wl=side["filters"]["w_l"], e0_model=float(spec.e0), e1_lower=e1lb,
                  w_l_certified=side["w_l_certified"], b_l_terminates=side["b_l_terminates"],
                  p_lb=_finite_or_none(p_lb))
    # U-referenced filters: CISD(H') or a dedicated energy of Psi_k, one call per certificate, its cost added
    options = [c for c in certs if c.source == "classical" or (c.source in ("sampled", "ae") and c.j == k)]
    base = amplify.state_scheme_costs(p, c_prep, c_loop, c_qpe, filt_oracle_u=None if options else
                                      side["filters"]["oracle_u"], **rule_b, **common)
    costs, flags = dict(base["costs"]), {s: list(v) for s, v in base["flags"].items()}
    sources = {s: "none" if cert is None else cert.label for s in ("rep", "aa", "bbht", "fp") if costs[s] is not None}
    u_of = {}
    for opt in options:
        u_form = amplify.final_qpe_form(opt.upper, e1lb, de, grid=grid)
        cs = amplify.state_scheme_costs(p, c_prep, c_loop, c_qpe, filt_u=_filters_at(side, opt.upper, params),
                                        u_extra=opt.toffolis, u_final=u_form, **rule_b, **common)
        for s in amplify.ORACLE_U_SCHEMES + ("lt_rep",):
            v = cs["costs"][s]
            if v is not None and (costs[s] is None or v < costs[s]):
                costs[s], flags[s], sources[s], u_of[s] = v, list(cs["flags"][s]), opt.label, (opt.upper, u_form)
            elif v is None and costs[s] is None:
                flags[s] = list(cs["flags"][s])         # a certified U whose final QPE does not terminate
    row_flags = list(rb.flags)
    if side["filters"]["w_l"] is None:
        row_flags.append("not_terminating_L")
    elif not side["w_l_certified"]:
        row_flags.append("termination_not_certified")
    if any(s.endswith("@oracle_U") for s in costs):
        row_flags.append("oracle_U")
    contaminant = None
    if compiled:
        # F41: contaminants below each scheme's acceptance threshold, budgeted as q_c/(p_0 K_0^n)
        unit = np.zeros(spec.energies.size)
        unit[0] = 1.0
        k0_at: dict = {}

        def term(theta: float, n_runs: int) -> tuple[float, float]:
            if theta not in k0_at:
                k0_at[theta] = (certify.contaminant_weight(spec, ci, theta),
                                acc.acceptance_stats(spec, unit, theta, n_points=side["m2"]).k0)
            q_c, k0 = k0_at[theta]
            return q_c, (q_c / (p * k0 ** n_runs) if p * k0 > 0 else math.inf)

        theta_l = side["u"] + de if side["u_final"] == "B_U" else side["theta_l"]
        thetas = {s: (rb.theta, rb.n_runs) for s in ("rep", "aa", "bbht", "fp")}
        thetas.update((s, (theta_l, 1)) for s in ("lt_bbht_L", "2s_L", "lt_rep_L"))
        thetas.update((s, (u + de if form == "B_U" else side["theta_l"], 1)) for s, (u, form) in u_of.items())
        contaminant = {"terms": {}, "q_c": {}}
        for s, (theta, n_runs) in thetas.items():
            if costs.get(s) is None:
                continue
            q_c, budget = term(float(theta), n_runs)
            contaminant["q_c"][s], contaminant["terms"][s] = q_c, budget
            if q_c > PREMISE_TOL:
                flags[s] = list(dict.fromkeys(flags[s] + ["contaminant_below_theta"]))
        if any(v > PREMISE_TOL for v in contaminant["q_c"].values()):
            row_flags.append("contaminant_below_theta")
        for s in flags:
            flags[s] = list(dict.fromkeys(flags[s] + ["exact_state_loop_costs"]))
        row_flags.append("exact_state_loop_costs")
    try:
        best = amplify.best_realizable(costs)
    except ValueError:
        best = None
    try:
        best_ext = amplify.best_extended(costs)
    except ValueError:
        best_ext = None
    rep_rule_e = c_loop + (c_prep + c_qpe) / p
    row = {"state": state, "record": label, "k": k, "p_model": p, "p": p_h, "c_prep": c_prep,
           "c_loop": c_loop, "c_prep_over_qpe": c_prep / c_qpe, "c_loop_over_qpe": c_loop / c_qpe,
           "p_lb": _finite_or_none(p_lb), "n_v": rb.n_runs, "n_v_expected": rb.n_runs_expected, "form": rb.form,
           "costs": costs, "costs_over_qpe": {s: None if v is None else v / c_qpe for s, v in costs.items()},
           "rep_rule_e_over_qpe": rep_rule_e / c_qpe, "sources": sources, "scheme_flags": flags,
           "best_realizable": best, "flags": [] if best is None else list(flags[best]),
           "best_extended": best_ext, "flags_extended": [] if best_ext is None else list(flags[best_ext]),
           "row_flags": list(dict.fromkeys(row_flags))}
    if contaminant is not None:
        row["contaminant"] = contaminant
    return row


def savings(rows: list[dict]) -> list[dict]:
    """T_s(k) - T_s(k + 1) per scheme for k = 1, 2 (C_QPE, relative to T_s(k)), at s* and at the extended optimum."""
    by_k = {r["k"]: r for r in rows}
    out = []
    for k in (1, 2):
        if k not in by_k or k + 1 not in by_k:
            continue
        a, b = by_k[k]["costs_over_qpe"], by_k[k + 1]["costs_over_qpe"]
        sav = {}
        for s in list(amplify.SCHEMES) + list(amplify.SENSITIVITY_SCHEMES) + ["rep_rule_e"]:
            ta = by_k[k]["rep_rule_e_over_qpe"] if s == "rep_rule_e" else a.get(s)
            tb = by_k[k + 1]["rep_rule_e_over_qpe"] if s == "rep_rule_e" else b.get(s)
            if ta is None or tb is None:
                sav[s] = None
            else:
                sav[s] = {"saving_over_qpe": ta - tb, "relative": (ta - tb) / ta}
        ver, ext = amplify.iteration_verdict(a, b), amplify.extended_verdict(a, b)
        out.append({"k": k, "savings": sav, "best_k": ver["best_k"], "best_k1": ver["best_k1"],
                    "saving_best_over_qpe": ver["saving_best"], "relative_best": ver["relative_best"],
                    "pays_best": ver["pays_best"],
                    "flags_best_k": by_k[k]["scheme_flags"].get(ver["best_k"], []) if ver["best_k"] else [],
                    "flags_best_k1": by_k[k + 1]["scheme_flags"].get(ver["best_k1"], []) if ver["best_k1"] else [],
                    "extended": {"best_k": ext["best_k"], "best_k1": ext["best_k1"],
                                 "saving_over_qpe": ext["saving_best"], "relative": ext["relative_best"],
                                 "pays": ext["pays_best"]}})
    return out


def break_even_table(side: dict, p1: float, c1_over_qpe: float) -> dict:
    """U9b: p_0* per scheme and C_it (C_QPE units) for (p_1, C(Psi_1)/C_QPE)."""
    c_qpe = float(side["qpe"].toffolis)
    fu, fl = side["filters"]["u"], side["filters"]["w_l"]
    out = {}
    for s in amplify.SCHEMES:
        cf = cc = None
        if s in ("lt", "lt_bbht", "2s"):
            if fu is None:
                continue
            cf, cc = fu["c_filt"] / c_qpe, fu["c_coarse"] / c_qpe
        elif s in ("lt_bbht_L", "2s_L"):
            if fl is None:
                continue
            cf, cc = fl["c_filt"] / c_qpe, fl["c_coarse"] / c_qpe
        out[s] = {str(c_it): amplify.break_even_overlap(s, c_it, p1, c_qpe=1.0, c_prep1=c1_over_qpe, c_filt=cf,
                                                        c_coarse=cc) for c_it in C_IT}
    return out


def state_path(name: str, inputs, side: dict, cache, params: FTParameters, timings: dict) -> dict:
    """Records, U8 rows, U9 rows, savings and U9b of one system."""
    t = time.perf_counter()
    records = build_records(name, inputs, cache, params)
    preps = {label: prep_cost(rec.directions, rec.eps, irreps=inputs.irreps, params=params)
             for label, (rec, _) in records.items()}
    timings["records"] = time.perf_counter() - t
    t = time.perf_counter()
    costs = loop_costs(name, inputs, records, preps, cache, params)
    timings["loop_costs"] = time.perf_counter() - t
    t = time.perf_counter()
    compiled = None
    if inputs.active.norb <= 8:
        compiled = compiled_record(name, inputs, records["r1_so"][0], cache, params)
    timings["compiled"] = time.perf_counter() - t
    t = time.perf_counter()
    c_qpe = float(side["qpe"].toffolis)
    out_records, u8, u9, sav = {}, [], {}, {}
    certs_by = {}
    for label, (rec, _) in list(records.items()) + ([("compiled_so", (compiled, None))] if compiled else []):
        prep = preps["r1_so"] if label == "compiled_so" else preps[label]
        certs = certificates_for(rec, prep, inputs, side)
        certs_by[label] = certs
        p_all = np.array([float(side["spec"].weights(s)[0]) for s in rec.states])
        p_model = [float(x) for x in p_all[:K_STATES + 1]]
        # rule B on p of H' (U8 and the option of U9's rep, aa, bbht, fp); M3's T6 keeps p on H
        rec_model = dataclasses.replace(rec, overlaps=p_all)
        cost_label = "r1_so" if label == "compiled_so" else label
        r_list = R_VALUES if records.get(cost_label, (None, None))[1] is not None else (params.r,)
        out_records[label] = {
            "first_step": rec.first_step, "metric": rec.metric, "n_steps": rec.n_steps,
            "energies": rec.energies, "e_err": np.asarray(rec.energies) - float(rec.e0), "overlaps": rec.overlaps,
            "overlaps_model": p_model, "eps": rec.eps, "prep_cumulative": prep.cumulative[:len(rec.states)],
            "prep_qubits": prep.logical_qubits[:len(rec.states)], "flags": list(rec.flags),
            "loop_costs": {str(r): costs[cost_label, r] for r in r_list},
            "certificates": certs}
        compiled_row = label == "compiled_so"
        for r in r_list:
            iters = [c["toffolis"] for c in costs[cost_label, r]]
            rows = handoff.rule_b_table(rec_model, prep.cumulative, iters, side["qpe"], certificates=certs,
                                        loop="classical_first_step", e1_lower=side["e1lb"], grid=side["grid"],
                                        e0_model=float(side["spec"].e0), params=params)
            c_loops = [rb.c_loop for rb in rows]
            rows = [rb for rb in rows if rb.k <= K_STATES]
            u9_rows = []
            for rb in rows:
                u9_rows.append(scheme_row(STATE_NAMES[rb.k], label, rb.k, rec.states[rb.k], rb, c_loops, rec, certs,
                                          side, params, p_h=float(rec.overlaps[rb.k]), compiled=compiled_row))
            u9[f"{label}@r={r}"] = u9_rows
            sav[f"{label}@r={r}"] = savings(u9_rows)
            if r == (R_RULE_B if len(r_list) > 1 else params.r):
                for rb in rows:
                    if rb.k == 0 and label != "r1_so":
                        continue                                # HF once (r1_so)
                    if rb.k == 1 and label.startswith("exact"):
                        continue                                # the classical Psi_1 is the R1 record's
                    u8.append(u8_row(STATE_NAMES[rb.k], label, rb.k, rec.states[rb.k], rb, side, inputs,
                                     compiled=compiled_row, p_h=float(rec.overlaps[rb.k])))
    timings["u8_u9"] = time.perf_counter() - t
    r1 = u9.get(f"r1_so@r={R_RULE_B}", [])
    psi2 = next((r for r in r1 if r["k"] == 2), None)
    u9b = {"perfect": break_even_table(side, 1.0, 0.0)}
    if psi2 is not None:
        u9b["r1_so_psi2"] = {"p1": psi2["p_model"], "c1_over_qpe": psi2["c_prep_over_qpe"],
                             "p0_star": break_even_table(side, psi2["p_model"], psi2["c_prep_over_qpe"])}
    return {"records": out_records, "u8": u8, "u9": u9, "savings": sav, "u9b": u9b, "c_qpe": c_qpe}


# ------------------------------------------------------------------ M3's T6 under rule B
def _num(x) -> float:
    if isinstance(x, str):
        return {"inf": math.inf, "-inf": -math.inf}[x]
    return math.nan if x is None else float(x)


def _rebuild_encoding(active, enc_json: dict, params: FTParameters):
    ref = ExactReference.build(active)
    kind, par = enc_json["kind"], enc_json["params"]
    if kind == "sf":
        return ftqpe.sf_encoding(active, int(par["L"]), reference=ref, params=params)
    if kind == "df":
        return ftqpe.df_encoding(active, float(par["threshold"]), reference=ref, params=params)
    if kind == "thc":
        return ftqpe.thc_encoding(active, int(par["M"]), penalty=par["penalty"], reference=ref, params=params)
    raise ValueError(f"unknown encoding kind {kind!r}")


def m3_inputs(name: str, inputs, m3_dir: str | None, cache, params: FTParameters) -> dict:
    """M3's records (with states), preparation and iteration costs, hand-off rows and optima (module docstring)."""
    active, irreps = inputs.active, inputs.irreps
    flags: list[str] = []
    if m3_dir is None:
        def compute():
            rep = pipeline.run_system(name, params=params)
            out = {"qpe": rep.qpe["overall"], "records": {}, "prep": {}, "iterations": {}, "rows": {}, "optima": {},
                   "energy_split": rep.settings["energy_split"], "readout_toffolis": rep.settings["readout_toffolis"]}
            for fam in M3_FAMILIES:
                if fam not in rep.trajectories:
                    continue
                out["records"][fam] = dict(rep.trajectories[fam])
                out["prep"][fam] = {f: rep.prep[fam][f]["elementwise"] for f in rep.prep[fam]}
                out["iterations"][fam] = {lp: [c.toffolis for c in rep.iterations[fam][lp]]
                                          for lp in rep.iterations[fam]}
                out["rows"][fam] = {lp: [r.cost_expected for r in rep.handoff[fam][lp]["same_irrep"]]
                                    for lp in handoff.LOOPS}
                out["optima"][fam] = {lp: {rule: _optimum(rep.handoff[fam][lp]["same_irrep"], rule, rep.qpe["overall"])
                                           for rule in ("expected", "expected_oracle")} for lp in handoff.LOOPS}
            return out

        data = cache.get(name, "m3_report", compute, key={"params": dataclasses.asdict(params)})
        flags.append("m3_recomputed")
        data["flags"] = flags
        return data
    with open(os.path.join(m3_dir, f"ft_{name}.json")) as fh:
        d = json.load(fh)
    enc_json = d["qpe"]["overall"]["encoding"]
    sel = inputs.qpe
    same = (sel.encoding.kind == enc_json["kind"] and scaling.to_json(sel.encoding.params) == enc_json["params"]
            and abs(sel.encoding.lam - enc_json["lam"]) <= 1e-9 * abs(enc_json["lam"])
            and int(sel.toffolis) == int(d["qpe"]["overall"]["toffolis"]))
    if same:
        q = sel
    else:
        enc = cache.get(name, "m3_encoding", lambda: _rebuild_encoding(active, enc_json, params),
                        key={"encoding": enc_json})
        q = ftqpe.qpe_cost(enc, float(d["qpe"]["overall"]["de_qpe"]))
        flags.append("m3_encoding_differs_from_selection")
        if abs(enc.lam - enc_json["lam"]) > 1e-9 * abs(enc_json["lam"]) or int(q.toffolis) != int(
                d["qpe"]["overall"]["toffolis"]):
            flags.append("m3_encoding_mismatch")

    def compute_records():
        out = {}
        for fam in M3_FAMILIES:
            if fam not in d["trajectories"]:
                continue
            opt, metric, _ = pipeline.FAMILIES[fam]
            out[fam] = {}
            for first in pipeline.RECORDS:
                n = len(d["trajectories"][fam][first]["eps"])
                out[fam][first] = exact_trajectory(active, optimizer=opt, metric=metric, max_iter=max(n, 1),
                                                   system=name, irreps=irreps, n_roots=inputs.spectrum.n_roots,
                                                   first_step=first, roots=inputs.spectrum.roots)
        return out

    records = cache.get(name, "m3_records", compute_records, key={"families": list(M3_FAMILIES)})
    out = {"qpe": q, "records": records, "prep": {}, "iterations": {}, "rows": {}, "optima": {},
           "energy_split": d["settings"]["energy_split"], "readout_toffolis": int(d["settings"]["readout_toffolis"])}
    worst = {"energies": 0.0, "overlaps": 0.0, "prep": 0.0}
    for fam, recs in records.items():
        out["prep"][fam] = {}
        for first, rec in recs.items():
            js = d["trajectories"][fam][first]
            worst["energies"] = max(worst["energies"], float(np.max(np.abs(np.asarray(js["energies"]) - rec.energies))))
            worst["overlaps"] = max(worst["overlaps"], float(np.max(np.abs(np.asarray(js["overlaps"]) - rec.overlaps))))
            pc = prep_cost(rec.directions, rec.eps, irreps=irreps, params=params)
            pj = np.asarray(d["prep"][fam][first]["elementwise"]["cumulative"], dtype=float)
            worst["prep"] = max(worst["prep"], float(np.max(np.abs(pj - pc.cumulative))))
            out["prep"][fam][first] = pc
        out["iterations"][fam] = {lp: [_num(it["toffolis"]) for it in d["iterations"][fam][lp]]
                                  for lp in ("measured", "classical_first_step")}
        out["rows"][fam] = {lp: [_num(r["cost_expected"]) for r in d["handoff"][fam][lp]["same_irrep"]]
                            for lp in handoff.LOOPS}
        out["optima"][fam] = {lp: {rule: d["summary"]["families"][fam]["handoff"][lp][rule]
                                   for rule in ("expected", "expected_oracle")} for lp in handoff.LOOPS}
    out["reproduction"] = worst
    if worst["energies"] > 1e-9 or worst["overlaps"] > 1e-9 or worst["prep"] > 0.0:
        flags.append("m3_records_differ")
    out["flags"] = flags
    return out


def _same_qpe(a, b) -> bool:
    """Two QPECosts of the same encoding (kind, parameters, lambda, C_W and C_QPE)."""
    ea, eb = a.encoding, b.encoding
    return (ea.kind == eb.kind and scaling.to_json(ea.params) == scaling.to_json(eb.params)
            and abs(ea.lam - eb.lam) <= 1e-12 * abs(eb.lam) and int(ea.toffoli_per_step) == int(eb.toffoli_per_step)
            and int(a.toffolis) == int(b.toffolis))


def _optimum(rows, rule: str, qpe) -> dict | None:
    try:
        best = handoff.optimal_handoff(rows, rule)
    except ValueError:
        return None
    return {"k": best.k, "cost_over_qpe": best.cost_expected / float(qpe.toffolis)}


def m3_t6(name: str, inputs, side_m3: dict, m3: dict, params: FTParameters) -> dict:
    """M3's T6 under rule B per family and loop (module docstring)."""
    q = m3["qpe"]
    c_qpe = float(q.toffolis)
    out = {"flags": list(m3["flags"]), "reproduction": m3.get("reproduction"), "families": {}}
    max_rows_diff = 0.0
    for fam, recs in m3["records"].items():
        out["families"][fam] = {}
        certs_cache = {}
        for lp in handoff.LOOPS:
            first = "second_order" if lp == "classical_first_step" else "exact"
            rec, pc = recs[first], m3["prep"][fam][first]
            iters = None if lp == "free" else m3["iterations"][fam][lp]
            if first not in certs_cache:
                var = np.array([ri.energy_variances(inputs.active, s)[m3["energy_split"]] for s in rec.states])
                certs_cache[first] = certify.energy_certificates(
                    rec, pc, e1_lower=side_m3["e1lb"], de=side_m3["de"], classical={"cisd": side_m3["u"]},
                    energy_variances=var, readout_toffolis=m3["readout_toffolis"], walk=side_m3["enc"],
                    e0_model=float(side_m3["spec"].e0))
            rows = handoff.rule_b_table(rec, pc.cumulative, iters, q, certificates=certs_cache[first], loop=lp,
                                        e1_lower=side_m3["e1lb"], grid=side_m3["grid"],
                                        e0_model=float(side_m3["spec"].e0), params=params)
            ref = m3["rows"][fam][lp]
            n_cmp = min(len(ref), len(rows))
            if n_cmp:
                max_rows_diff = max(max_rows_diff, max(abs(rows[i].cost_oracle_e - ref[i]) / ref[i]
                                                       for i in range(n_cmp) if math.isfinite(ref[i])))
            try:
                best = handoff.optimal_rule_b(rows)
                rb = {"k": best.k, "cost_over_qpe": best.cost / c_qpe, "form": best.form, "n_v": best.n_runs,
                      "n_v_expected": best.n_runs_expected, "certified": best.certified,
                      "u_label": None if best.certificate is None else best.certificate.label,
                      "certificate_over_qpe": 0.0 if best.certificate is None else best.certificate.toffolis / c_qpe,
                      "certificate_expected_over_qpe": best.certificate_cost / c_qpe,
                      "theta_minus_e0_model": best.theta - float(side_m3["spec"].e0),
                      "distance": (side_m3["e1lb"] - best.theta) / side_m3["grid"], "flags": list(best.flags)}
            except ValueError:
                rb = None
            out["families"][fam][lp] = {"rows_to_k": rows[-1].k, "m3_certified": m3["optima"][fam][lp]["expected"],
                                        "m3_oracle": m3["optima"][fam][lp]["expected_oracle"], "rule_b": rb,
                                        "rows": [_rule_b_json(r, c_qpe, side_m3) for r in rows]}
    out["rows_max_relative_difference"] = max_rows_diff
    if max_rows_diff > 1e-9:
        out["flags"].append("m3_rows_differ")
    return out


def _rule_b_json(rb: handoff.RuleBRow, c_qpe: float, side: dict) -> dict:
    cert = rb.certificate
    return {"k": rb.k, "p": rb.p, "e_err": rb.e_err, "form": rb.form, "certified": rb.certified,
            "n_v": rb.n_runs, "n_v_expected": rb.n_runs_expected,
            "theta_minus_e0_model": rb.theta - float(side["spec"].e0),
            "u_label": None if cert is None else cert.label, "u_iterate": None if cert is None else cert.loop_iterate,
            "certificate_over_qpe": 0.0 if cert is None else cert.toffolis / c_qpe,
            "certificate_expected_over_qpe": rb.certificate_cost / c_qpe,
            "cost_over_qpe": rb.cost / c_qpe, "cost_oracle_e_over_qpe": rb.cost_oracle_e / c_qpe,
            "c_loop_over_qpe": rb.c_loop / c_qpe, "flags": list(rb.flags)}


# ------------------------------------------------------------------ one system
def run_one(name: str, cache, params: FTParameters, m3_dir: str | None) -> dict:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    t = time.perf_counter()
    inputs = closedloop.system_inputs(name, cache=cache, params=params)
    timings["system_inputs"] = time.perf_counter() - t
    t = time.perf_counter()
    side = qpe_side(name, inputs, cache, params)
    timings["qpe_side"] = time.perf_counter() - t
    sp = inputs.spectrum
    result = {"system": name, "norb": inputs.active.norb, "nelec": list(inputs.active.nelec), "dim": sp.dim,
              "spectrum": {"e0": sp.e0, "e1": sp.e1, "e1_any": sp.e1_any, "e_hf": sp.e_hf, "p_hf": sp.p_hf,
                           "gap": sp.e1 - sp.e0, "flags": list(sp.flags)},
              "variance_state": inputs.variance_state, "energy_split": inputs.energy_split,
              "qpe_side": side_summary(side, inputs)}
    result.update(state_path(name, inputs, side, cache, params, timings))
    result["m3_t6"] = None
    if name in M3_SYSTEMS:
        t = time.perf_counter()
        m3 = m3_inputs(name, inputs, m3_dir, cache, params)
        side_m3 = side if _same_qpe(m3["qpe"], inputs.qpe) else qpe_side(name, inputs, cache, params, qpe=m3["qpe"],
                                                                  stage_suffix="_m3")
        result["m3_t6"] = m3_t6(name, inputs, side_m3, m3, params)
        timings["m3_t6"] = time.perf_counter() - t
    result["timings"] = timings
    result["runtime_s"] = time.perf_counter() - t0
    result["peak_rss_mb"] = peak_rss_mb()
    return result


# ------------------------------------------------------------------ markdown
def _f(x, spec: str = ".4g") -> str:
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


def markdown(results: list[dict], command: str, script_s: float) -> str:
    lines = ["# M4 hand-off: rule B (U8) and QPE-side schemes on exact records (U9, part 1)", "",
             f"Generated by `{command}` ({script_s:.0f} s, one thread). Costs in units of the system's selected "
             "C_QPE; energies in mHa above the ground state of H' (E_0') unless noted. dE = 1 mHa; E1lb = E_1' - "
             "0.1 mHa; U = CISD(H'); g = 2 dE sin(theta_0) at M = 2 n_iter; n_v of F40 at p_min = 1e-3, "
             "eps_id = 1e-3.", ""]
    # QPE side
    rows = []
    for r in results:
        q = r["qpe_side"]
        rows.append([r["system"], q["kind"], json.dumps(q["params"]).replace("|", "/"), _f(q["lam"], ".5g"), q["c_w"],
                     f"{q['n_iter']:,}", f"{q['c_qpe']:.4g}", _f(q["sin_theta0"], ".4f"), _f(q["grid"] * 1e3, ".3f"),
                     _f(q["e0_model_minus_e0"] * 1e3, ".3f"),
                     _f((q["e1_singlet"] - q["e0_model"]) * 1e3, ".3f"), _f(q["e1lb_minus_e0_model"] * 1e3, ".3f"),
                     q["e1lb_level"], _f(q["levels"]["gap_e1_any_spin"] * 1e3, ".3f"),
                     _f(q["u_cisd_minus_e0_model"] * 1e3, ".3f"), _f(q["u_cisdt_minus_e0_model"] * 1e3, ".3f"),
                     _f(q["margin_cisd"] * 1e3, ".3f") if q["u_certified"] else "no U",
                     _f(q["d_u"], ".3g") if q["u_certified"] else "-",
                     "-" if not q["u_certified"] or q["n_v_u"] is None else q["n_v_u"],
                     q["n_v_l"], "yes" if q["w_l_terminates"] else "no",
                     ", ".join(q["flags"] + r["spectrum"]["flags"]) or "-"])
    lines += ["## QPE side and certificates (F39-F42)", "",
              "E_0' - E_0 and E_1' - E_1 compare H' with H; 'any spin' is the lowest excited level of the ground "
              "irrep of any S (the sector of compiled states, F41); the margin is E1lb - U - 2 dE, D_U = "
              "(E1lb - U - dE)/g; n_v(B_L) at D_L = 5; 'L forms' terminate iff E_0' <= E1lb - w_L.", ""]
    lines += _table(["system", "kind", "params", "lambda", "C_W", "n_iter", "C_QPE", "sin theta0", "g (mHa)",
                     "E0' - E0", "E1' - E0'", "E1lb - E0'", "E1lb level", "any spin", "CISD - E0'", "CISDT - E0'",
                     "margin", "D_U", "n_v(B_U)", "n_v(B_L)", "L forms", "flags"], rows)
    # U8
    rows = []
    for r in results:
        for u in r["u8"]:
            sine, kai = u["laws"]["sine_2n"], u["laws"]["kaiser_4n"]
            n_v = str(u["n_v"])
            if abs(u["n_v_expected"] - u["n_v"]) > 1e-12:
                n_v += f" (E {_f(u['n_v_expected'], '.3f')})"
            cert = _f(u["certificate_over_qpe"], ".3g")
            if abs(u["certificate_expected_over_qpe"] - u["certificate_over_qpe"]) > 1e-12:
                cert += f" (E {_f(u['certificate_expected_over_qpe'], '.3g')})"
            rows.append([r["system"], u["state"], u["record"], _f(u["p_model"], ".4f"), _f(u["e_err"] * 1e3, ".2f"),
                         "yes" if u["rule_e_certified"] else "no", u["form"], u["u_label"] or "-",
                         _f(None if u["margin"] is None else u["margin"] * 1e3, ".2f"), _f(u["distance"], ".3g"),
                         n_v, cert, _f(u["cost_over_qpe"], ".4g"),
                         _f(sine["one_run"]["kernel"], ".2e"), _f(sine["n_v_runs"]["kernel"], ".2e"),
                         _f(kai["one_run"]["kernel"], ".2e"), _f(kai["n_v_runs"]["kernel"], ".2e"),
                         _f(sine["n_v_runs"]["logical"], ".1e"), _f(sine["n_v_runs"]["eps_id"], ".2e"),
                         _f(u.get("contaminant"), ".2e"), bound_cell(u),
                         _f(u["bound_state"], ".2e"), "yes" if u["terminates"] else "no", ", ".join(u["flags"]) or "-"])
    lines += ["## U8: rule B per state", "",
              "p on psi_0' (H'), the overlap of T_B (as U9); E - E0 of H (mHa); 'rule E' is M3's certification "
              "(E_1 - E_k > 3 dE). n_v at the planned U, '(E x)' the expectation over the realized bound of a "
              "dedicated energy, as 'cert' (its Toffolis / C_QPE) and T_B (C_QPE, verification runs and certificate "
              "included). eps_id kernel terms of the accepted state: sine window at M = 2 n_iter and Kaiser(3) at "
              "4 n_iter, one run and n_v runs at rule B's threshold; 'logical' the failure term at the F40 budgets "
              "(n_v runs); 'total' the sine n_v-run kernel plus logical (plus q_c, which the full-sector kernel holds, "
              "for compiled states); 'bound' [(1 - p_min)/p_min] tau_W(D)^n_v, whose premise (every populated "
              "excited level at or above E1lb) fails for states with more than 1e-12 below E1lb ('n/a (q_c)' where "
              "F41's contaminants below theta carry it, else 'n/a (below E1lb)'); 'state bound' "
              "[(1 - p_0)/p_0] [tau_W(D)/K_0]^n_v, for those states plus the weight of their populated excited levels "
              "below E1lb over p_0 K_0^n_v (q_c/(p_0 K_0^n_v) where all lie below theta).", ""]
    lines += _table(["system", "state", "record", "p", "E - E0", "rule E", "form", "U", "margin", "D", "n_v", "cert",
                     "T_B", "sine 1 run", "sine n_v", "Kaiser 1 run", "Kaiser n_v", "logical", "total", "q_c", "bound",
                     "state bound", "terminates", "flags"], rows)
    rows = []
    for r in results:
        for u in r["u8"]:
            fm = u["forms"]
            bu = fm.get("b_u_cisd")
            rows.append([r["system"], u["state"], u["record"], _f(u["p_model"], ".4f")]
                        + (["no U"] * 3 if bu is None
                           else [_f(bu[k], ".2e") for k in ("sine_2n", "kaiser_4n", "rect_2n")])
                        + [_f(fm["b_l"]["sine_2n_one_run"], ".2e"), _f(fm["b_l"]["sine_2n_two_runs"], ".2e"),
                           _f(fm["literal"]["sine_2n"], ".3g")])
    lines += ["Identification error (kernel term) of the forms side by side: B_U at theta = CISD(H') + dE (sine and "
              "rect at 2 n_iter, Kaiser(3) at 4 n_iter), B_L at E1lb - 5 g (sine 2 n_iter, one and two runs) and the "
              "literal theta = E_1' - dE (sine 2 n_iter); spectrum-weighted for the state, not bounds.", ""]
    lines += _table(["system", "state", "record", "p", "B_U sine", "B_U Kaiser(3) 4n", "B_U rect", "B_L 1 run",
                     "B_L 2 runs", "literal"], rows)
    # M3 T6
    rows = []
    for r in results:
        t6 = r.get("m3_t6")
        if not t6:
            continue
        for fam, loops in t6["families"].items():
            for lp, v in loops.items():
                mc, mo, rb = v["m3_certified"], v["m3_oracle"], v["rule_b"]
                rows.append([r["system"], fam, lp, v["rows_to_k"],
                             "-" if mc is None else f"{mc['k']} ({_f(mc['cost_over_qpe'])})",
                             "-" if mo is None else f"{mo['k']} ({_f(mo['cost_over_qpe'])})",
                             "-" if rb is None else f"{rb['k']} ({_f(rb['cost_over_qpe'])})",
                             "-" if rb is None else rb["form"], "-" if rb is None else rb["n_v"],
                             "-" if rb is None else (rb["u_label"] or "none"),
                             "-" if rb is None else _f(rb["distance"], ".3g"),
                             ", ".join(t6["flags"]) or "-"])
    if rows:
        lines += ["## U8: M3's T6 under rule B", "",
                  "k_opt (T / C_QPE) of M3's certified rule E, M3's oracle T_E and rule B (certificates CISD(H') "
                  "and dedicated energies; verification runs included); p on psi_0 of H as M3.", ""]
        lines += _table(["system", "family", "loop", "rows to k", "M3 certified", "M3 oracle", "rule B", "form",
                         "n_v", "U", "D", "flags"], rows)
    # U9
    show = amplify.SCHEMES
    sens = amplify.SENSITIVITY_SCHEMES
    rows = []
    for r in results:
        for key, u9 in r["u9"].items():
            label, rr = key.split("@r=")
            if label.startswith("r1") and float(rr) != R_RULE_B:
                continue
            for row in u9:
                cq = row["costs_over_qpe"]
                best, ext = row["best_realizable"], row["best_extended"]
                rows.append([r["system"], label, rr, row["state"], _f(row["p_model"], ".4f"),
                             _f(row["c_loop_over_qpe"], ".3g"), _f(row["rep_rule_e_over_qpe"], ".4g")]
                            + [_f(cq.get(s), ".4g") + ("" if cq.get(s) is not None or f"{s}@oracle_U" not in cq
                                                       else f" ({_f(cq[f'{s}@oracle_U'], '.4g')} oracle U)")
                               for s in show]
                            + [_f(cq.get(s), ".4g") for s in sens]
                            + [best or "-", ", ".join(row["flags"]) or "-",
                               "-" if ext is None or ext == best else f"{ext} ({_f(cq.get(ext), '.4g')})"])
    lines += ["## U9 (part 1): QPE-side schemes on exact records", "",
              "T_s / C_QPE with p on psi_0' and C_loop at r = 1 (R1, F30) or M3's r = 0.3 (exact records, F14). "
              "'rep E' is M3's rule E (oracle T_E, no verification run); rep, aa, bbht, fp with rule B's "
              "acceptance (expected n_v - 1 verification runs, certificate); lt, lt_bbht, 2s with a certified U; the "
              "L forms with w_L = 100 mHa; '(x oracle U)' the lower bound at U = E_0' + 2 dE where no U is "
              "certified. 'best' is s* over the realizable set of record, 'flags of s*' its termination and "
              "certificate flags; lt_rep and lt_rep_L (the step filter followed by repetition) are sensitivity "
              "columns, never s*, and 'extended' gives the optimum over the set extended by them where it differs "
              "from s*.", ""]
    lines += _table(["system", "record", "r", "state", "p", "C_loop", "rep E"] + list(show) + list(sens)
                    + ["best", "flags of s*", "extended"], rows)
    rows = []
    for r in results:
        q = r["qpe_side"]
        for key in ("u", "w_l", "oracle_u"):
            f = q["filters"][key]
            if f is not None:
                rows.append([r["system"], key, _f((f["u"] - q["e0_model"]) * 1e3, ".3f"), f["degree"],
                             _f(f["c_filt"] / q["c_qpe"], ".4g"), _f(f["c_coarse"] / q["c_qpe"], ".4g")])
    lines += ["Filter and coarse-QPE costs (F43): pass edge U - E0' (mHa), stop edge E1lb.", ""]
    lines += _table(["system", "pass edge", "U - E0'", "degree", "c_filt", "c_c"], rows)
    rows = []
    for r in results:
        for key, sv in r["savings"].items():
            label, rr = key.split("@r=")
            for s in sv:
                cells = []
                for sch in ("rep_rule_e",) + show + sens:
                    v = s["savings"].get(sch)
                    cells.append("-" if v is None
                                 else f"{_f(v['saving_over_qpe'], '.3g')} ({100 * v['relative']:.1f}%)")
                for blk, (bk, bk1, sv_b, rel) in (
                        ("best", (s["best_k"], s["best_k1"], s["saving_best_over_qpe"], s["relative_best"])),
                        ("ext", (s["extended"]["best_k"], s["extended"]["best_k1"], s["extended"]["saving_over_qpe"],
                                 s["extended"]["relative"]))):
                    cells.append("-" if sv_b is None else
                                 f"{bk} -> {bk1}: {_f(sv_b, '.3g')} ({_f(None if rel is None else 100 * rel, '.3g')}%)")
                rows.append([r["system"], label, rr, f"Psi{s['k']} -> Psi{s['k'] + 1}"] + cells)
    lines += ["Savings of a measured iteration T_s(k) - T_s(k + 1) (C_QPE, and % of T_s(k)); > 0: it pays. 'at s*' "
              "compares the best realizable hand-offs before and after the iteration (the verdict of record), "
              "'extended' the optima with lt_rep and lt_rep_L admitted (a sensitivity).", ""]
    lines += _table(["system", "record", "r", "iteration", "rep E"] + list(show) + list(sens) + ["at s*", "extended"],
                    rows)
    rows = []
    for r in results:
        for ref_key, block in r["u9b"].items():
            tab = block if ref_key == "perfect" else block["p0_star"]
            for s, vals in tab.items():
                rows.append([r["system"], "perfect (p1 = 1)" if ref_key == "perfect" else
                             f"R1 so Psi2 (p1 = {block['p1']:.4f})", s] + [_f(vals[str(c)], ".3g") for c in C_IT])
    lines += ["## U9b: break-even overlap p0* (F44)", "",
              "The p_0 below which an iteration of cost C_it that yields (p_1, C(Psi_1)) pays, per scheme.", ""]
    lines += _table(["system", "state after", "scheme"] + [f"C_it = {c:g}" for c in C_IT], rows)
    rows = [[r["system"], _f(r["runtime_s"], ".1f")] + [_f(r["timings"].get(k, 0.0), ".1f") for k in
            ("system_inputs", "qpe_side", "records", "loop_costs", "compiled", "u8_u9", "m3_t6")]
            + [_f(r["peak_rss_mb"], ".0f")] for r in results]
    lines += ["## Runtime (s, one thread)", ""]
    lines += _table(["system", "total", "inputs", "QPE side", "records", "loop costs", "compiled", "U8/U9", "M3 T6",
                     "peak RSS (MB)"], rows)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default=None, help="comma-separated F20 names (default: M3's systems, the overlap "
                                                    "axis, h4/h6_r3.5/4.0, h8_r3.0/4.0, n2v_r2.0/2.4/2.8)")
    ap.add_argument("--quick", action="store_true", help=f"systems {','.join(QUICK_SYSTEMS)} (unless --systems)")
    ap.add_argument("--out", default="results/m4", help="output directory (cache in <out>/cache)")
    ap.add_argument("--no-cache", action="store_true", help="recompute and overwrite cached stages")
    ap.add_argument("--m3-results", default=None,
                    help="directory with M3's ft_<system>.json (default: rerun ft.pipeline.run_system)")
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
        res = run_one(name, cache, params, args.m3_results)
        res["command_flags"] = {"quick": args.quick, "m3_results": args.m3_results, "no_cache": args.no_cache}
        with open(os.path.join(args.out, f"handoff_{name}.json"), "w") as fh:
            json.dump(scaling.to_json(res), fh, allow_nan=False)
        results.append(res)
        q = res["qpe_side"]
        best = [(row["state"], row["best_realizable"]) for row in res["u9"].get(f"r1_so@r={R_RULE_B}", [])]
        print(f"[{name}] {res['runtime_s']:.1f} s; C_QPE {q['c_qpe']:.4g} ({q['kind']}); E1lb - E0' "
              f"{q['e1lb_minus_e0_model'] * 1e3:.2f} mHa; U certified {q['u_certified']} (margin "
              f"{q['margin_cisd'] * 1e3:.2f} mHa, D_U {q['d_u']:.3g}); best realizable {best}; "
              f"peak RSS {res['peak_rss_mb']:.0f} MB", flush=True)
    command = ("benchmarks/run_m4_handoff.py" + (" --quick" if args.quick else "")
               + (f" --systems {','.join(names)}" if args.systems else "") + f" --out {os.path.relpath(args.out)}"
               + (" --no-cache" if args.no_cache else "")
               + (f" --m3-results {args.m3_results}" if args.m3_results else ""))
    script_s = time.perf_counter() - t_script
    with open(os.path.join(args.out, "handoff_summary.md"), "w") as fh:
        fh.write(markdown(results, command, script_s))
    print(f"wrote {len(results)} JSON files and handoff_summary.md to {args.out} ({script_s:.1f} s; cache hits "
          f"{cache.hits}, misses {cache.misses})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

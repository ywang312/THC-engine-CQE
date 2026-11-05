"""Measurement cost of the CQE energy and ACSE residual along exact CQE trajectories (M2, workflow B).

For every system the exact steepest-descent CQE (``cqe.run_cqe`` defaults: SD with the exact line
search) runs from HF. The states are the iterates ``--states`` (default 0, 2, 4, 5, 7; iterate k is the
state after k steps, iterate 0 the HF determinant) and the FCI ground state ("fci"). For each state the
script records E - E_FCI, ||A|| and sigma^2 = <H^2> - <H>^2 (the variance of a measurement of H in its
own eigenbasis, a lower bound for every scheme), and computes exact per-shot variances with
``measure.schemes``, through the routes of ``measure.routes`` for the residual.

Energy (every system, including the energy-only systems h6b and h8):
  QWC sorted insertion in both qubit orders (number of groups; optimal and uniform allocation), DF
  basis rotation (``schemes.df_variance``) with four splits: leaves from pivoted Cholesky (the
  default) or from the eigendecomposition of the ERI supermatrix ("eigh"), each without and with the
  shift c_l = tr L_l, Pauli shadows (both orders), matchgate shadows (order independent), and shots
  for +-1.6 mHa, S = V / 0.0016^2.

Residual (default systems h4, h6a, h6, lih, n2), on both element sets (spin-orbital: K = 840
independent elements of 2A^{ij}_{kl} for N = 6; spin-free: 315 independent elements of A[p,q,r,s]):
  * aux route A_hat_e = -<Q_e>_Lambda / delta (``routes.aux_residual_variance``) for every delta in
    ``--deltas`` and the methods "exact", "df-trotter" and "df-fragments" (untruncated Cholesky DF,
    L leaves, L + 1 states), with QWC and Pauli shadows in both qubit orders and matchgate shadows;
  * direct route A_e = <X_e>_Psi (``routes.direct_residual_variance``) with matchgate and Pauli shadows;
  * the relative bias of the aux route, ||A_hat(inf shots) - A|| / ||A||;
  * shots for a 30% relative Frobenius error of A (``routes.shots_for_relative_error``), variance-only
    and bias-aware (inf = unreachable, ||bias|| >= 0.3 ||A||).
At the FCI state A = 0, so only variances are reported there.

Conventions
-----------
* v_e is the per-shot (per-snapshot) variance of the estimator of element e under the scheme's
  allocation that minimizes sum_e v_e; "total" is sum_e v_e over the element set, so S shots give
  sum_e Var = total / S. Aux-route variances include the 1/delta^2 of the post-processing; delta^2 v_e
  is the per-shot variance of Q_e itself (of Im 2D^{ij}_{kl} for the spin-orbital set, with
  2D^{ij}_{kl} = <a+_i a+_j a_l a_k>, no factor 1/2).
* Norms are those of the full tensor: ``weights = targets.element_multiplicities`` (8 entries of the
  (2N)^4 tensor per spin-orbital element, 2 or 4 entries of the N^4 pair-order tensor per spin-free
  element). For the spin-orbital set this equals the element-set norm of the earlier claims
  (K = 840). S_var = sum m_e v_e / (0.09 sum m_e A_e^2); S_bias = sum m_e v_e / (0.09 sum m_e A_e^2 -
  sum m_e b_e^2) with b = A_hat(inf) - A.
* DF energy variances use the exact Hamiltonian's factorization (untruncated); their truncation bias
  <H_DF> - <H> is round-off (recorded).

Reductions (to keep the full run within about 60 minutes on one thread)
-----------------------------------------------------------------------
* Oracle cache. Every scheme call builds a dense table of <P(x, z)> for its state (0.4-0.8 s for an
  N = 6 state against 4-16 ms for the pair sums; ``measure.routes`` docstring). While a system runs,
  the private factory ``schemes._Expectations`` is wrapped by a content-keyed LRU cache (qubit count,
  basis and amplitudes, hashed), so the 3 schemes and 2 element sets that measure one state share one
  table. The table is a deterministic function of the state; ``--verify-cache`` (run by ``--quick``)
  recomputes one state's aux and energy variances without the cache and requires bitwise equality.
  ``--no-cache`` disables it.
* df-fragments runs in the blocked order only, except at the claims states (below), and not at FCI.
  Its total is (L + 1)^2 times the "exact" route's to O(delta^2) (tested on H4; reported here as
  the ratio ``frag/(L+1)^2``).
* Direct route with matchgate shadows at N = 6 (1.6e6 Pauli terms): Rao-Blackwellized estimate
  (``schemes``, method "rb") with ``--rb-settings`` random matchings (default 5000), and
  ``--rb-settings-claims`` (default 20000) at the claims states; standard errors are reported. H4
  uses the exact pair sums.
* Direct route with Pauli shadows at N = 6: exact pair sums (89-198 s per call, observed) only at the
  claims states (spin-orbital at both, spin-free at iterate 2) with ``--pauli-direct claims``
  (default); ``all`` computes them at every state, ``none`` nowhere. H4 always uses the exact sums.
The claims states are the iterate 2 of h6a and the h6a iterate closest to 6.6 mHa above FCI (the
earlier analysis's state). ``--rb-settings``, ``--pauli-direct all`` and ``--no-cache`` lift the
reductions.

Claims
------
The comparison with the earlier, unverified H6 analysis (spec.md "Goal") is in the summary: for each
claim our value at h6a iterate 2 and at the state closest to 6.6 mHa above FCI (and for the 1.5 A
energy claims, h6b at FCI), the ratio ours / claimed, the verdict (reproduced if the ratio at the
6.6 mHa state is within 30%, or our value lies within 30% of a claimed range) and the likely reason
for a discrepancy.

Threads: ``--threads n`` pins PySCF's OpenMP threads and, through OMP_NUM_THREADS / OPENBLAS_NUM_THREADS
/ MKL_NUM_THREADS, numpy's BLAS threads (effective only if numpy is not loaded yet, as when this file
runs as a script; ``settings["threads_env"]`` records what was in effect).

Outputs: <out>/measurement_<sys>.json and <out>/measurement_<sys>.md per system,
<out>/measurement_summary.md and <out>/measurement_summary.json (claims, timings).

Usage:
    python benchmarks/run_measurement.py --threads 1 --out results/
    python benchmarks/run_measurement.py --quick --threads 1 --out /tmp/quick
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import zlib
from collections import OrderedDict

# numpy (and everything that imports it) is imported inside the functions, after --threads has
# set the BLAS thread variables; see the module docstring.

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
DEFAULT_SYSTEMS = "h4,h6a,h6b,h6,lih,n2,h8"
DEFAULT_RESIDUAL = "h4,h6a,h6,lih,n2"
DEFAULT_STATES = "0,2,4,5,7,fci"
DEFAULT_DELTAS = "0.01,0.03,0.1,0.3,0.6"
QUICK = {"systems": "h4", "residual_systems": "h4", "states": "0,2,5,fci", "deltas": "0.1,0.6"}
ENERGY_PRECISION = 1.6e-3      # Ha, the target of one line-search energy
REL_ERROR = 0.3                # relative Frobenius error of the residual
CLAIM_ERROR = 6.6e-3           # Ha above FCI: the state of the earlier analysis
CLAIMS_SYSTEM = "h6a"          # H6 at 1.0 A
CLAIMS_SYSTEM_STRETCHED = "h6b"  # H6 at 1.5 A (energy claims)
ORDERS = ("blocked", "interleaved")
KINDS = ("spin_orbital", "spin_free")
METHODS = ("exact", "df-trotter", "df-fragments")
# scheme, qubit orders it is evaluated in (matchgate shadows do not depend on the order)
AUX_SCHEMES = (("qwc", ORDERS), ("pauli_shadow", ORDERS), ("matchgate_shadow", ("blocked",)))
DF_VARIANTS = (("cholesky", {}, False), ("eigh", {"first": "eigh"}, False),
               ("cholesky_shift", {}, True), ("eigh_shift", {"first": "eigh"}, True))
SMALL_PAIRS = 2e7              # direct targets with at most this many term pairs use exact sums everywhere


def build_system(name: str):
    from thc_engine_cqe import systems

    table = {
        "h4": lambda: systems.hydrogen_chain(4, 1.8, "bohr"),
        "h6a": lambda: systems.hydrogen_chain(6, 1.0, "angstrom"),
        "h6b": lambda: systems.hydrogen_chain(6, 1.5, "angstrom"),
        "h6": lambda: systems.hydrogen_chain(6, 1.8, "bohr"),
        "lih": lambda: systems.lih(),
        "n2": lambda: systems.n2(1.098),
        "h8": lambda: systems.hydrogen_chain(8, 1.8, "bohr"),
    }
    if name not in table:
        raise SystemExit(f"unknown system {name!r}; choose from {sorted(table)}")
    return table[name]()


def pin_threads(n: int | None) -> dict:
    """Pin PySCF's OpenMP and numpy's BLAS threads (the BLAS pin needs numpy not yet loaded)."""
    blas_pinned = None
    if n is not None:
        blas_pinned = "numpy" not in sys.modules
        if blas_pinned:
            for var in THREAD_VARS:
                os.environ[var] = str(n)
    from pyscf import lib

    if n is not None:
        lib.num_threads(n)
    return {"threads": n, "blas_pinned": blas_pinned, "pyscf_threads": lib.num_threads(),
            "cpu_count": os.cpu_count(), **{var: os.environ.get(var) for var in THREAD_VARS}}


# =================================================================== oracle cache
class OracleCache:
    """Content-keyed LRU cache of the expectation tables of ``measure.schemes`` (module docstring).

    While active, ``schemes._Expectations(state)`` returns the cached oracle of an equal state
    (same qubit count, basis and amplitudes). Calls with an explicit ``max_entries`` bypass it.
    A no-op if the private factory does not exist.
    """

    def __init__(self, size: int = 24, enabled: bool = True):
        from thc_engine_cqe.measure import schemes

        self.schemes = schemes
        self.orig = getattr(schemes, "_Expectations", None)
        self.enabled = enabled and self.orig is not None
        self.size = int(size)
        self.data: OrderedDict = OrderedDict()
        self.hits = self.misses = 0
        self.build_seconds = 0.0

    def __call__(self, state, max_entries=None):
        if max_entries is not None:
            return self.orig(state, max_entries)
        h = hashlib.blake2b(digest_size=20)
        h.update(state.basis.tobytes())
        h.update(state.amplitudes.tobytes())
        key = (int(state.n_qubits), h.digest())
        hit = self.data.get(key)
        if hit is not None:
            self.data.move_to_end(key)
            self.hits += 1
            return hit
        self.misses += 1
        t = time.perf_counter()
        oracle = self.orig(state)
        self.build_seconds += time.perf_counter() - t
        self.data[key] = oracle
        while len(self.data) > self.size:
            self.data.popitem(last=False)
        return oracle

    def __enter__(self):
        if self.enabled:
            self.schemes._Expectations = self
        return self

    def __exit__(self, *exc):
        if self.enabled:
            self.schemes._Expectations = self.orig
        self.data.clear()
        return False

    def stats(self) -> dict:
        return {"enabled": self.enabled, "hits": self.hits, "misses": self.misses,
                "table_build_seconds": self.build_seconds}


# ======================================================================== helpers
def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _names(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _states(text: str) -> list:
    out = []
    for s in _names(text):
        out.append("fci" if s.lower() == "fci" else int(s))
    return out


def _finite(x):
    """JSON-safe float: None for inf/nan/None."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def stats(v) -> dict:
    import numpy as np

    v = np.asarray(v, dtype=float)
    return {"min": float(v.min()), "mean": float(v.mean()), "median": float(np.median(v)),
            "max": float(v.max())}


def wnorm(x, w) -> float:
    import numpy as np

    return float(np.sqrt(np.sum(w * np.abs(np.asarray(x)) ** 2)))


def state_label(it) -> str:
    return "FCI" if it == "fci" else str(it)


def rb_seed(system: str, it, kind: str) -> int:
    return zlib.crc32(f"{system}/{it}/{kind}".encode())


# ===================================================================== trajectory
def trajectory(act, iterates):
    """[(iterate, ci)] for the requested SD iterates (iterate 0 = HF) and ("fci", ci_FCI); E_FCI."""
    import numpy as np

    from thc_engine_cqe.cqe import run_cqe
    from thc_engine_cqe.fci_ops import FCISpace

    e_fci, ci_fci = act.fci()
    space = FCISpace.of(act)
    want = sorted(k for k in iterates if k != "fci")
    cis = {0: space.hf()}
    if want and want[-1] > 0:
        def callback(step, ci_after):
            cis[len(cis)] = np.array(ci_after, copy=True)

        run_cqe(act, max_iter=want[-1], callback=callback)
    out = []
    for k in iterates:
        if k == "fci" or k in cis:
            c = np.asarray(ci_fci if k == "fci" else cis[k], dtype=float).reshape(space.shape)
            out.append((k, c / np.linalg.norm(c)))
        else:
            raise SystemExit(f"iterate {k} was not reached (the CQE stopped after {len(cis) - 1} steps)")
    return float(e_fci), out


def state_info(act, ci, e_fci) -> dict:
    import numpy as np

    from thc_engine_cqe.fci_ops import Hamiltonian
    from thc_engine_cqe.measure import routes, targets
    from thc_engine_cqe.residual import compute_residual

    rd = compute_residual(act, ci)
    e, var = Hamiltonian(act).energy_and_variance(ci)
    out = {"e": e, "e_err": e - e_fci, "sigma2": var, "norm_A_full_spin_free": float(rd.residual_norm)}
    for kind in KINDS:
        el = targets.element_set(act.norb, kind)
        a = routes.exact_residual(act, ci, el, kind)
        w = targets.element_multiplicities(el, kind, act.norb)
        out[f"norm_A_elements_{kind}"] = float(np.linalg.norm(a))
        out[f"norm_A_full_{kind}"] = wnorm(a, w)
    return out


# ========================================================================= energy
def energy_part(act, ci, ctx) -> dict:
    """Per-shot energy variances of every scheme on one state (module docstring)."""
    from thc_engine_cqe.measure import schemes
    from thc_engine_cqe.measure.pauli import QubitState

    t0 = time.perf_counter()
    out = {"qwc": {}, "pauli_shadow": {}, "df": {}}
    for order in ORDERS:
        m = ctx["mappings"][order]
        h = ctx["hamiltonian"][order]
        psi = QubitState.from_ci(act.norb, act.nelec, ci, m)
        r = schemes.qwc_variance(h, psi, grouping=ctx["energy_grouping"][order])
        out["qwc"][order] = {"variance": r.total, "groups": r.n_settings, "uniform": r.uniform_total}
        r = schemes.pauli_shadow_variance(h, psi)
        out["pauli_shadow"][order] = {"variance": r.total, "method": r.method}
        if order == "blocked":
            r = schemes.matchgate_shadow_variance(h, psi)
            out["matchgate_shadow"] = {"variance": r.total, "method": r.method}
    for name, _, shift in DF_VARIANTS:
        r = schemes.df_variance(act, ci, factorization=ctx["df_energy"][name.replace("_shift", "")], shift=shift)
        out["df"][name] = {"variance": r.total, "settings": r.n_settings, "uniform": r.uniform_total,
                           "bias": r.info["bias"], "n_leaves": r.info["n_leaves"]}
    out["seconds"] = time.perf_counter() - t0
    return out


def energy_shots(v: float) -> float:
    return float(v) / ENERGY_PRECISION ** 2


# ======================================================================= residual
def _entry(res, a, w, d, extra: dict, defined: bool) -> dict:
    """JSON record of one residual VarianceResult: variances, bias, shots (relative quantities only if
    ``defined``, i.e. not at FCI, where A vanishes up to the FCI solver's convergence)."""
    import numpy as np

    from thc_engine_cqe.measure import routes

    per = np.asarray(res.per_target, dtype=float)
    bias = np.asarray(res.expectation, dtype=float) - a
    na = wnorm(a, w)
    defined = bool(defined and na > 0)
    rec = dict(extra)
    rec.update({
        "n_settings": res.n_settings, "total": float(res.total), "per_element": stats(per),
        "total_uniform": _finite(res.uniform_total),
        "per_element_uniform": None if res.per_target_uniform is None else stats(res.per_target_uniform),
        "stderr": _finite(res.stderr), "method": res.method, "n_samples": res.n_samples,
        "rel_bias": wnorm(bias, w) / na if defined else None,
        "shots_var": _finite(routes.shots_for_relative_error(per, a, REL_ERROR, weights=w)) if defined else None,
        "shots_bias": (_finite(routes.shots_for_relative_error(per, a, REL_ERROR, bias=bias, weights=w))
                       if defined else None),
        "unreachable": bool(defined and not math.isfinite(
            routes.shots_for_relative_error(per, a, REL_ERROR, bias=bias, weights=w))),
    })
    if res.per_target_uniform is not None and defined:          # QWC: equal shots per setting
        pu = np.asarray(res.per_target_uniform, dtype=float)
        rec["shots_var_uniform"] = _finite(routes.shots_for_relative_error(pu, a, REL_ERROR, weights=w))
        rec["shots_bias_uniform"] = _finite(routes.shots_for_relative_error(pu, a, REL_ERROR, bias=bias, weights=w))
    if d is not None:
        rec["q_total"] = float(res.total) * d * d                  # sum_e Var(Q_e) per shot
        rec["q_per_element"] = stats(per * d * d)
        if res.per_target_uniform is not None:
            rec["q_per_element_uniform"] = stats(np.asarray(res.per_target_uniform) * d * d)
    if res.per_target_stderr is not None and res.method == "rb":
        rse = np.asarray(res.per_target_stderr, dtype=float) / np.maximum(per, 1e-300)
        rec["per_element_rel_stderr_max"] = _finite(np.max(rse))
        rec["n_covering_min"] = int(np.min(res.info.get("n_covering", [0])))
    return rec


def residual_part(act, ci, it, ctx, plan) -> dict:
    """Aux and direct routes on both element sets for one state (module docstring)."""
    import numpy as np

    from thc_engine_cqe.measure import routes, targets

    out = {k: {"aux": [], "direct": []} for k in KINDS}
    timing = {"aux": 0.0, "direct_matchgate": 0.0, "direct_pauli": 0.0}
    exact = {}
    for kind in KINDS:
        el = ctx["elements"][kind]
        a = routes.exact_residual(act, ci, el, kind)
        w = ctx["weights"][kind]
        exact[kind] = (el, a, w)
        out[kind].update({"K": int(len(el)), "norm_A_elements": float(np.linalg.norm(a)),
                          "norm_A_full": wnorm(a, w), "qwc_groups": ctx["aux_groups"][kind]})
    # ---- aux route: delta -> method -> order -> kind -> scheme (cache-friendly order)
    t_aux = time.perf_counter()
    for d in ctx["deltas"]:
        for method in METHODS:
            orders = plan["fragment_orders"] if method == "df-fragments" else ORDERS
            if not orders:
                continue
            t = time.perf_counter()
            states = routes.aux_states(act, ci, d, method, df=ctx["df"])
            t_states = time.perf_counter() - t
            for order in orders:
                m = ctx["mappings"][order]
                for kind in KINDS:
                    el, a, w = exact[kind]
                    for scheme, scheme_orders in AUX_SCHEMES:
                        if order not in scheme_orders:
                            continue
                        t = time.perf_counter()
                        res = routes.aux_residual_variance(act, ci, d, method, el, kind, scheme, mapping=m,
                                                           df=ctx["df"], states=states)
                        rec = _entry(res, a, w, d, {"route": "aux", "route_method": method, "delta": d,
                                                    "scheme": scheme, "order": order,
                                                    "n_states": res.info["n_states"],
                                                    "n_leaves": res.info["n_leaves"]}, it != "fci")
                        rec["seconds"] = time.perf_counter() - t
                        rec["aux_states_seconds"] = t_states
                        out[kind]["aux"].append(rec)
    timing["aux"] = time.perf_counter() - t_aux
    # ---- direct route
    m = ctx["mappings"]["blocked"]
    for kind in KINDS:
        el, a, w = exact[kind]
        xs = ctx["direct_targets"][kind]
        for scheme in ("matchgate_shadow", "pauli_shadow"):
            small = ctx["direct_pairs"][kind] <= SMALL_PAIRS
            opts = {}
            if scheme == "pauli_shadow" and not small and kind not in plan["pauli_kinds"]:
                continue
            if scheme == "matchgate_shadow" and not small:
                opts = {"n_settings": plan["rb_settings"], "seed": rb_seed(ctx["name"], it, kind)}
            t = time.perf_counter()
            res = routes.direct_residual_variance(act, ci, el, kind, scheme, mapping=m, targets=xs, **opts)
            rec = _entry(res, a, w, None, {"route": "direct", "scheme": scheme, "order": "blocked",
                                           "seed": opts.get("seed")}, it != "fci")
            rec["seconds"] = time.perf_counter() - t
            timing["direct_matchgate" if scheme == "matchgate_shadow" else "direct_pauli"] += rec["seconds"]
            out[kind]["direct"].append(rec)
    out["timing"] = timing
    return out


# ===================================================================== per system
def run_system(name: str, args) -> dict:
    import numpy as np

    from thc_engine_cqe.factor import double_factorize
    from thc_engine_cqe.measure import schemes, targets
    from thc_engine_cqe.measure.pauli import QubitMapping, jw_hamiltonian

    t0 = time.perf_counter()
    act = build_system(name)
    n = act.norb
    e_fci, states = trajectory(act, args.states)
    residual = name in args.residual_systems
    print(f"[{name}] {act.label}: N={n}, nelec={act.nelec}, {len(states)} states, "
          f"residual={'yes' if residual else 'no'}", flush=True)
    ctx = {"name": name, "deltas": args.deltas,
           "mappings": {o: QubitMapping(n, o) for o in ORDERS}}
    ctx["hamiltonian"] = {o: jw_hamiltonian(act, ctx["mappings"][o]) for o in ORDERS}
    ctx["energy_grouping"] = {o: schemes.qwc_grouping(ctx["hamiltonian"][o]) for o in ORDERS}
    ctx["df_energy"] = {"cholesky": double_factorize(act.eri), "eigh": double_factorize(act.eri, first="eigh")}
    ctx["df"] = ctx["df_energy"]["cholesky"]
    out = {"system": name, "label": act.label, "norb": n, "nelec": list(act.nelec), "e_fci": e_fci,
           "n_leaves": int(ctx["df"].n_leaves), "n_leaves_eigh": int(ctx["df_energy"]["eigh"].n_leaves),
           "hamiltonian_terms": {o: int(ctx["hamiltonian"][o].n_terms) for o in ORDERS},
           "energy_qwc_groups": {o: int(ctx["energy_grouping"][o].n_groups) for o in ORDERS},
           "states": [], "timing": {"energy": 0.0, "aux": 0.0, "direct_matchgate": 0.0, "direct_pauli": 0.0}}
    infos = [(it, ci, state_info(act, ci, e_fci)) for it, ci in states]
    # claims states: iterate 2 and the iterate closest to 6.6 mHa above FCI (non-FCI states)
    claims_states = set()
    if name == CLAIMS_SYSTEM:
        non_fci = [(it, inf) for it, _, inf in infos if it != "fci"]
        if any(it == 2 for it, _ in non_fci):
            claims_states.add(2)
        if non_fci:
            claims_states.add(min(non_fci, key=lambda p: abs(p[1]["e_err"] - CLAIM_ERROR))[0])
    out["claims_states"] = sorted(claims_states)
    if residual:
        ctx["elements"] = {k: targets.element_set(n, k) for k in KINDS}
        ctx["weights"] = {k: targets.element_multiplicities(ctx["elements"][k], k, n) for k in KINDS}
        ctx["aux_groups"] = {}
        for kind in KINDS:
            ctx["aux_groups"][kind] = {
                o: schemes.qwc_grouping(targets.aux_targets(ctx["elements"][kind], ctx["mappings"][o], kind)).n_groups
                for o in ORDERS}
        t = time.perf_counter()
        ctx["direct_targets"] = {k: targets.direct_targets(ctx["elements"][k], act, ctx["mappings"]["blocked"], k,
                                                           hamiltonian=ctx["hamiltonian"]["blocked"])
                                 for k in KINDS}
        ctx["direct_pairs"] = {k: float(sum(x.n_terms * (x.n_terms + 1) / 2 for x in ctx["direct_targets"][k]))
                               for k in KINDS}
        out["direct_targets"] = {k: {"terms": int(sum(x.n_terms for x in ctx["direct_targets"][k])),
                                     "pairs": ctx["direct_pairs"][k]} for k in KINDS}
        out["direct_targets_seconds"] = time.perf_counter() - t
    with OracleCache(enabled=not args.no_cache) as cache:
        for it, ci, info in infos:
            ts = time.perf_counter()
            rec = {"iterate": it, **info, "claims_state": it in claims_states}
            rec["energy"] = energy_part(act, ci, ctx)
            out["timing"]["energy"] += rec["energy"]["seconds"]
            if residual:
                claim = it in claims_states
                plan = {"fragment_orders": () if it == "fci" else (ORDERS if claim else ("blocked",)),
                        "rb_settings": args.rb_settings_claims if claim else args.rb_settings,
                        "pauli_kinds": (KINDS if args.pauli_direct == "all"
                                        else () if args.pauli_direct == "none"
                                        else (KINDS if (claim and it == 2) else ("spin_orbital",) if claim else ()))}
                rec["plan"] = {k: list(v) if isinstance(v, tuple) else v for k, v in plan.items()}
                rec["residual"] = residual_part(act, ci, it, ctx, plan)
                for k, v in rec["residual"]["timing"].items():
                    out["timing"][k] += v
            rec["seconds"] = time.perf_counter() - ts
            out["states"].append(rec)
            print(f"[{name}] state {state_label(it):>3}: E-E_FCI={info['e_err'] * 1e3:8.3f} mHa  "
                  f"||A||={info['norm_A_full_spin_free']:.3e}  ({rec['seconds']:.1f} s)", flush=True)
        out["oracle_cache"] = cache.stats()
    out["seconds"] = time.perf_counter() - t0
    return out


# ================================================================= cache check
def verify_cache(args) -> dict:
    """Recompute the aux variances (all schemes, both kinds, delta = 0.1, exact and df-fragments) and the
    energy variances of H4 iterate 2 with and without the cache; require bitwise equality."""
    import numpy as np

    from thc_engine_cqe.measure import routes, schemes
    from thc_engine_cqe.measure.pauli import QubitMapping, QubitState, jw_hamiltonian

    act = build_system("h4")
    _, sts = trajectory(act, [2])
    ci = sts[0][1]
    m = QubitMapping(act.norb)

    def run():
        vals = []
        for method in ("exact", "df-fragments"):
            states = routes.aux_states(act, ci, 0.1, method)
            for kind in KINDS:
                for scheme in ("qwc", "pauli_shadow", "matchgate_shadow"):
                    r = routes.aux_residual_variance(act, ci, 0.1, method, None, kind, scheme, states=states)
                    vals.append(np.concatenate([[r.total], r.per_target, r.expectation]))
        h = jw_hamiltonian(act, m)
        psi = QubitState.from_ci(act.norb, act.nelec, ci, m)
        for scheme in ("qwc", "pauli_shadow", "matchgate_shadow"):
            r = schemes.scheme_variance(h, psi, scheme)
            vals.append(np.array([r.total]))
        return vals

    ref = run()
    with OracleCache() as cache:
        new = run()
        st = cache.stats()
    same = all(np.array_equal(a, b) for a, b in zip(ref, new))
    if not same:
        raise SystemExit("oracle cache changed a result; rerun with --no-cache")
    return {"bitwise_identical": same, "n_results": len(ref), **st}


# ===================================================================== claims
def _state(res: dict, it):
    for s in res["states"]:
        if s["iterate"] == it:
            return s
    return None


def _aux(s: dict, kind: str, method: str, d: float, scheme: str, order: str = "blocked"):
    if s is None or "residual" not in s:
        return None
    for e in s["residual"][kind]["aux"]:
        if (e["route_method"] == method and abs(e["delta"] - d) < 1e-12 and e["scheme"] == scheme
                and e["order"] == order):
            return e
    return None


def _direct(s: dict, kind: str, scheme: str):
    if s is None or "residual" not in s:
        return None
    for e in s["residual"][kind]["direct"]:
        if e["scheme"] == scheme:
            return e
    return None


def _get(fn, s):
    try:
        v = fn(s)
    except (KeyError, TypeError):
        return None
    return v


def _reference_states(res: dict | None) -> dict:
    """States of one system used by the claims table: iterate 2, the iterate closest to 6.6 mHa above FCI,
    the nearest states above and below 6.6 mHa (the bracket; "below" may be FCI) and FCI."""
    keys = ("it2", "closest", "above", "below", "fci")
    if res is None:
        return dict.fromkeys(keys)
    non_fci = [s for s in res["states"] if s["iterate"] != "fci"]
    above = [s for s in non_fci if s["e_err"] >= CLAIM_ERROR]
    below = [s for s in res["states"] if s["e_err"] < CLAIM_ERROR]
    return {"it2": _state(res, 2),
            "closest": min(non_fci, key=lambda s: abs(s["e_err"] - CLAIM_ERROR)) if non_fci else None,
            "above": min(above, key=lambda s: s["e_err"]) if above else None,
            "below": max(below, key=lambda s: s["e_err"]) if below else None,
            "fci": _state(res, "fci")}


def _ev(*path):
    """Energy variance at results[state]["energy"][path...]["variance"]."""
    def f(s):
        x = s["energy"]
        for p in path:
            x = x[p]
        return x["variance"]
    return f


def _aux_field(kind, method, d, scheme, order, *fields):
    def f(s):
        x = _aux(s, kind, method, d, scheme, order)
        for k in fields:
            x = x[k]
        return x
    return f


def _direct_field(kind, scheme, *fields):
    def f(s):
        x = _direct(s, kind, scheme)
        for k in fields:
            x = x[k]
        return x
    return f


def claims(results: dict) -> list[dict]:
    """The comparison with the earlier analysis (module docstring): one row per claim.

    Values at the reference states of h6a (1.0 A; for the 1.5 A energy claims, h6b), the ratio ours /
    claimed at iterate 2 and at the closest state, the verdict (within 30% at the closest state, at FCI
    for the 1.5 A claims; for a claimed range, our value within [0.7 lo, 1.3 hi]), alternatives (other
    variants of our computation) and the likely reason for a discrepancy, built from these numbers.
    """
    ref = {"1.0": _reference_states(results.get(CLAIMS_SYSTEM)),
           "1.5": _reference_states(results.get(CLAIMS_SYSTEM_STRETCHED))}
    rows = []

    def add(cid, what, claimed, fn, *, alts=(), geometry="1.0", rng=None, verdict_at="closest"):
        st = ref[geometry]
        vals = {k: (_get(fn, v) if v is not None else None) for k, v in st.items()}
        row = {"id": cid, "claim": what, "claimed": claimed, "range": rng, "geometry": geometry,
               "values": vals, "verdict_state": verdict_at,
               "states": {k: None if v is None else {"iterate": v["iterate"], "e_err": v["e_err"]}
                          for k, v in st.items()},
               "alternatives": {lab: {k: (_get(f, st[k]) if st[k] is not None else None)
                                      for k in ("it2", "closest", "fci")} for lab, f in alts}}
        if claimed is not None:
            row["ratio"] = {k: (None if vals[k] is None else vals[k] / claimed) for k in ("it2", "closest", "fci")}
        x = vals[verdict_at]
        if x is None:
            row["verdict"] = "not computed"
        elif rng is not None:
            row["verdict"] = "reproduced" if 0.7 * rng[0] <= x <= 1.3 * rng[1] else "not reproduced"
        else:
            row["verdict"] = "reproduced" if 0.7 <= x / claimed <= 1.3 else "not reproduced"
        rows.append(row)
        return row

    # ------------------------------------------------------------ energy
    for geo, (q, dfv, mg) in (("1.0", (28.0, 3.2, 122.0)), ("1.5", (24.7, 1.44, 72.5))):
        at = "closest" if geo == "1.0" else "fci"
        add(f"E-QWC-{geo}", f"energy variance, QWC (Ha^2), {geo} A", q, _ev("qwc", "interleaved"),
            alts=[("QWC blocked", _ev("qwc", "blocked"))], geometry=geo, verdict_at=at)
        add(f"E-DF-{geo}", f"energy variance, DF basis rotation (Ha^2), {geo} A", dfv, _ev("df", "eigh"),
            alts=[("Cholesky leaves", _ev("df", "cholesky")), ("eigh, shifted", _ev("df", "eigh_shift")),
                  ("Cholesky, shifted", _ev("df", "cholesky_shift"))], geometry=geo, verdict_at=at)
        add(f"E-MG-{geo}", f"energy variance, matchgate shadows (Ha^2), {geo} A", mg, _ev("matchgate_shadow"),
            alts=[("Pauli shadows, interleaved", _ev("pauli_shadow", "interleaved")),
                  ("Pauli shadows, blocked", _ev("pauli_shadow", "blocked"))], geometry=geo, verdict_at=at)
    # ------------------------------------------------------------ aux route, Var(Im 2D) per element
    so = "spin_orbital"
    add("Q-MG", "aux route: Var(Im 2D) per element, shadows (mean over the 840 elements)", 5.4,
        _aux_field(so, "exact", 0.1, "matchgate_shadow", "blocked", "q_per_element", "mean"),
        alts=[("min", _aux_field(so, "exact", 0.1, "matchgate_shadow", "blocked", "q_per_element", "min")),
              ("max", _aux_field(so, "exact", 0.1, "matchgate_shadow", "blocked", "q_per_element", "max")),
              ("Pauli shadows blocked", _aux_field(so, "exact", 0.1, "pauli_shadow", "blocked", "q_per_element", "mean")),
              ("Pauli shadows interleaved",
               _aux_field(so, "exact", 0.1, "pauli_shadow", "interleaved", "q_per_element", "mean"))])
    add("Q-QWC", "aux route: Var(Im 2D) per element, QWC tomography with 854 settings", 36.0,
        _aux_field(so, "exact", 0.1, "qwc", "interleaved", "q_per_element_uniform", "mean"),
        alts=[("interleaved, optimal allocation",
               _aux_field(so, "exact", 0.1, "qwc", "interleaved", "q_per_element", "mean")),
              ("blocked, uniform", _aux_field(so, "exact", 0.1, "qwc", "blocked", "q_per_element_uniform", "mean")),
              ("blocked, optimal", _aux_field(so, "exact", 0.1, "qwc", "blocked", "q_per_element", "mean")),
              ("groups interleaved", lambda s: s["residual"][so]["qwc_groups"]["interleaved"]),
              ("groups blocked", lambda s: s["residual"][so]["qwc_groups"]["blocked"])])
    # ------------------------------------------------------------ aux bias
    for d, rng in ((0.1, (0.002, 0.01)), (0.3, (0.02, 0.08)), (0.6, (0.06, 0.29))):
        add(f"bias-{d:g}", f"aux route: relative bias at delta = {d:g} (exp(-i delta H), 840 elements)", None,
            _aux_field(so, "exact", d, "matchgate_shadow", "blocked", "rel_bias"), rng=rng,
            alts=[("315 spin-free", _aux_field("spin_free", "exact", d, "matchgate_shadow", "blocked", "rel_bias")),
                  ("840 df-trotter", _aux_field(so, "df-trotter", d, "matchgate_shadow", "blocked", "rel_bias")),
                  ("840 df-fragments", _aux_field(so, "df-fragments", d, "matchgate_shadow", "blocked", "rel_bias"))])
    # ------------------------------------------------------------ direct route
    row = add("D-MG", "direct route: per-element variance, shadows, 7-41 (mean 12-16)", None,
              _direct_field(so, "matchgate_shadow", "per_element", "mean"), rng=(12.0, 16.0),
              alts=[("min", _direct_field(so, "matchgate_shadow", "per_element", "min")),
                    ("max", _direct_field(so, "matchgate_shadow", "per_element", "max")),
                    ("Pauli shadows, mean", _direct_field(so, "pauli_shadow", "per_element", "mean"))])
    if row["values"]["closest"] is not None:      # mean 12-16 and min / max 7 / 41, each within 30%
        mn, mx = row["alternatives"]["min"]["closest"], row["alternatives"]["max"]["closest"]
        ok = (0.7 * 12 <= row["values"]["closest"] <= 1.3 * 16 and mn is not None and 0.7 * 7 <= mn <= 1.3 * 7
              and mx is not None and 0.7 * 41 <= mx <= 1.3 * 41)
        row["verdict"] = "reproduced" if ok else "not reproduced"
    # ------------------------------------------------------------ shots
    add("S-aux-QWC-0.1", "shots for 30%, aux + QWC, delta = 0.1 (840 elements)", 5.9e9,
        _aux_field(so, "exact", 0.1, "qwc", "interleaved", "shots_var_uniform"),
        alts=[("interleaved, optimal", _aux_field(so, "exact", 0.1, "qwc", "interleaved", "shots_var")),
              ("blocked, uniform", _aux_field(so, "exact", 0.1, "qwc", "blocked", "shots_var_uniform")),
              ("blocked, optimal", _aux_field(so, "exact", 0.1, "qwc", "blocked", "shots_var")),
              ("interleaved, uniform, bias-aware",
               _aux_field(so, "exact", 0.1, "qwc", "interleaved", "shots_bias_uniform"))])
    for d, val in ((0.1, 8.7e8), (0.3, 9.7e7)):
        add(f"S-aux-MG-{d:g}", f"shots for 30%, aux + shadows, delta = {d:g} (840 elements)", val,
            _aux_field(so, "exact", d, "matchgate_shadow", "blocked", "shots_var"),
            alts=[("bias-aware", _aux_field(so, "exact", d, "matchgate_shadow", "blocked", "shots_bias")),
                  ("Pauli shadows blocked", _aux_field(so, "exact", d, "pauli_shadow", "blocked", "shots_var"))])
    add("S-direct-MG", "shots for 30%, direct shadows (840 elements)", 2.6e7,
        _direct_field(so, "matchgate_shadow", "shots_var"),
        alts=[("Pauli shadows", _direct_field(so, "pauli_shadow", "shots_var"))])
    add("S-E-DF", "shots per line-search energy, DF, +-1.6 mHa", 1.2e6,
        lambda s: energy_shots(s["energy"]["df"]["eigh"]["variance"]),
        alts=[("Cholesky leaves", lambda s: energy_shots(s["energy"]["df"]["cholesky"]["variance"])),
              ("eigh, shifted", lambda s: energy_shots(s["energy"]["df"]["eigh_shift"]["variance"]))])
    for row in rows:
        row["reason"] = _reason(row, ref)
    return rows


def _alt(row, label, at="closest"):
    v = row["alternatives"].get(label, {})
    return v.get(at)


def _reason(row: dict, ref: dict) -> str:
    """The likely reason for agreement or discrepancy, from the numbers of the row."""
    cid, v, r = row["id"], row["values"], row.get("ratio", {})
    at = row["verdict_state"]
    st = ref[row["geometry"]]

    def norm_a(key):
        s = st.get(key)
        return None if s is None or "residual" not in s else s["residual"]["spin_orbital"]["norm_A_elements"]

    try:
        if cid.startswith("E-QWC"):
            return (f"QWC depends on the qubit order: the claim matches the interleaved (OpenFermion) order; "
                    f"blocked gives {_g(_alt(row, 'QWC blocked', at))} ({_g(_alt(row, 'QWC blocked', at) / row['claimed'])}x)")
        if cid.startswith("E-DF"):
            return (f"the DF split is not unique: the claim matches leaves from the ERI supermatrix "
                    f"eigendecomposition (OpenFermion's low-rank form), unshifted; pivoted-Cholesky leaves give "
                    f"{_g(_alt(row, 'Cholesky leaves', at) / row['claimed'])}x, shifted leaves (E_pq - delta_pq) "
                    f"{_g(_alt(row, 'eigh, shifted', at) / row['claimed'])}x")
        if cid.startswith("E-MG"):
            return ("state-insensitive (iterate 2 to FCI within "
                    f"{_pct(abs(v['it2'] / v['fci'] - 1))}); Pauli shadows are "
                    f"{_g(_alt(row, 'Pauli shadows, interleaved', at) / v[at])}x larger (interleaved)")
        if cid == "Q-MG":
            return (f"delta^2 v_e at delta = 0.1; nearly state-independent (per element {_g(_alt(row, 'min'))}-"
                    f"{_g(_alt(row, 'max'))}); a 2D with a factor 1/2 would give 1/4 of it")
        if cid == "Q-QWC":
            return (f"matches uniform allocation over the QWC groups (equal shots per setting), interleaved order "
                    f"with {_g(_alt(row, 'groups interleaved'))} groups against the claimed 854; the optimal "
                    f"allocation gives {_g(_alt(row, 'interleaved, optimal allocation'))} "
                    f"({_g(_alt(row, 'interleaved, optimal allocation') / row['claimed'])}x)")
        if cid.startswith("bias"):
            return (f"exact exp(-i delta H); at iterate 2 / closest: spin-free {_pct(_alt(row, '315 spin-free', 'it2'))} / "
                    f"{_pct(_alt(row, '315 spin-free'))}; DF-Trotter {_pct(_alt(row, '840 df-trotter', 'it2'))} / "
                    f"{_pct(_alt(row, '840 df-trotter'))} (does not vanish at FCI)")
        if cid == "D-MG":
            return (f"matchgate shadows (Rao-Blackwellized); min / max {_g(_alt(row, 'min'))} / {_g(_alt(row, 'max'))} "
                    f"at the closest state, {_g(_alt(row, 'min', 'it2'))} / {_g(_alt(row, 'max', 'it2'))} at iterate 2")
        if cid.startswith("S-") and cid != "S-E-DF":
            na2, nac = norm_a("it2"), norm_a("closest")
            txt = (f"S ~ 1/||A||^2: ||A|| (840) = {_g(na2, 4)} at iterate 2, {_g(nac, 4)} at the closest state, so "
                   f"the claims' state (6.6 mHa) matches our iterate {st['closest']['iterate']}, not our iterate 2")
            if cid == "S-aux-QWC-0.1":
                txt += (f"; uniform allocation over the groups (as for Var(Im 2D) = 36); the optimal allocation needs "
                        f"{_g(_alt(row, 'interleaved, optimal') / v['closest'])}x of it")
            elif cid.startswith("S-aux-MG"):
                txt += "; variance only (the bias-aware count is in the alternatives)"
            return txt
        if cid == "S-E-DF":
            return "S = V_DF / 0.0016^2 with eigh leaves (row E-DF-1.0)"
    except (TypeError, ZeroDivisionError, KeyError):
        return "–"
    return "–"


# =================================================================== markdown
def _g(x, digits=3) -> str:
    if x is None:
        return "–"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    x = float(x)
    if not math.isfinite(x):
        return "inf"
    if x == 0:
        return "0"
    if 1e-2 <= abs(x) < 1e4:
        dec = max(0, digits - 1 - int(math.floor(math.log10(abs(x)))))
        return f"{x:.{dec}f}"
    return f"{x:.{max(digits - 1, 1)}e}".replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.3g}%"


def _shots(e: dict | None, key: str) -> str:
    if e is None:
        return "–"
    if key == "shots_bias" and e.get("unreachable"):
        return "unreach."
    return _g(e.get(key))


def md_states(results: dict) -> list[str]:
    lines = ["## States", "",
             "Exact SD CQE iterates (iterate k = after k steps; 0 = HF) and FCI. ||A||: full spin-free "
             "pair-order tensor (the CQE's residual norm) and the norms over the independent element sets "
             "(840/315 for N = 6), which the earlier claims use for the spin-orbital set. sigma^2 = <H^2> - "
             "<H>^2.", "",
             "| system | iterate | E - E_FCI (mHa) | norm A (full) | norm A (spin-orbital elements) | "
             "norm A (spin-free elements) | sigma^2 (Ha^2) |", "|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        for s in res["states"]:
            lines.append(f"| {name} | {state_label(s['iterate'])} | {s['e_err'] * 1e3:.3f} | "
                         f"{_g(s['norm_A_full_spin_free'], 4)} | {_g(s['norm_A_elements_spin_orbital'], 4)} | "
                         f"{_g(s['norm_A_elements_spin_free'], 4)} | {_g(s['sigma2'], 4)} |")
    return lines + [""]


def md_energy(results: dict) -> list[str]:
    lines = ["## Energy: per-shot variance (Ha^2)", "",
             "QWC: sorted insertion, optimal allocation (groups in parentheses); DF: basis-rotation grouping "
             "with L + 1 settings, leaves from pivoted Cholesky or the supermatrix eigendecomposition (eigh), "
             "without and with the shift c_l = tr L_l; shadows: exact per-snapshot variance. sigma^2 is the "
             "variance of a projective measurement of H.", "",
             "| system | iterate | QWC blocked | QWC interleaved | DF Cholesky (L+1) | DF eigh (L+1) | "
             "DF Cholesky shifted | DF eigh shifted | Pauli blocked | Pauli interleaved | matchgate | sigma^2 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        for s in res["states"]:
            en = s["energy"]
            q, df = en["qwc"], en["df"]
            lines.append(
                f"| {name} | {state_label(s['iterate'])} | {_g(q['blocked']['variance'], 4)} ({q['blocked']['groups']}) | "
                f"{_g(q['interleaved']['variance'], 4)} ({q['interleaved']['groups']}) | "
                f"{_g(df['cholesky']['variance'], 4)} ({df['cholesky']['settings']}) | "
                f"{_g(df['eigh']['variance'], 4)} ({df['eigh']['settings']}) | "
                f"{_g(df['cholesky_shift']['variance'], 4)} | {_g(df['eigh_shift']['variance'], 4)} | "
                f"{_g(en['pauli_shadow']['blocked']['variance'], 4)} | "
                f"{_g(en['pauli_shadow']['interleaved']['variance'], 4)} | "
                f"{_g(en['matchgate_shadow']['variance'], 4)} | {_g(s['sigma2'], 4)} |")
    lines += ["", "### Energy: shots for +-1.6 mHa (S = V / 0.0016^2)", "",
              "| system | iterate | QWC blocked | QWC interleaved | DF Cholesky | DF eigh | DF eigh shifted | "
              "Pauli blocked | matchgate | sigma^2 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        for s in res["states"]:
            en = s["energy"]
            vals = [en["qwc"]["blocked"]["variance"], en["qwc"]["interleaved"]["variance"],
                    en["df"]["cholesky"]["variance"], en["df"]["eigh"]["variance"], en["df"]["eigh_shift"]["variance"],
                    en["pauli_shadow"]["blocked"]["variance"], en["matchgate_shadow"]["variance"], s["sigma2"]]
            lines.append(f"| {name} | {state_label(s['iterate'])} | "
                         + " | ".join(_g(energy_shots(v)) for v in vals) + " |")
    return lines + [""]


def _res_states(results: dict):
    for name, res in results.items():
        for s in res["states"]:
            if "residual" in s:
                yield name, res, s


def md_bias(results: dict, deltas) -> list[str]:
    lines = ["## Aux route: relative bias ||A_hat(inf shots) - A|| / ||A|| (full tensor)", "",
             "Cells: spin-orbital / spin-free. Untruncated Cholesky DF (L leaves). Not defined at FCI (A = 0).", "",
             "| system | iterate | method | " + " | ".join(f"delta = {d:g}" for d in deltas) + " |",
             "|---|---|---|" + "---|" * len(deltas)]
    for name, res, s in _res_states(results):
        if s["iterate"] == "fci":
            continue
        for method in METHODS:
            cells = []
            for d in deltas:
                so = _aux(s, "spin_orbital", method, d, "matchgate_shadow")
                sf = _aux(s, "spin_free", method, d, "matchgate_shadow")
                cells.append(f"{_pct(so and so['rel_bias'])} / {_pct(sf and sf['rel_bias'])}")
            lines.append(f"| {name} | {state_label(s['iterate'])} | {method} | " + " | ".join(cells) + " |")
    return lines + [""]


def _ratio_range(s, kind, method, deltas, reference_scale) -> str:
    vals = []
    for d in deltas:
        for scheme, orders in AUX_SCHEMES:
            for order in orders:
                e = _aux(s, kind, method, d, scheme, order)
                x = _aux(s, kind, "exact", d, scheme, order)
                if e is not None and x is not None and x["total"] > 0:
                    vals.append(e["total"] / (reference_scale(e) * x["total"]))
    if not vals:
        return "–"
    return f"{min(vals):.4f}-{max(vals):.4f}"


def md_residual_variance(results: dict, deltas) -> list[str]:
    dref = 0.1 if any(abs(d - 0.1) < 1e-12 for d in deltas) else deltas[0]
    lines = ["## Residual: per-shot variances", "",
             f"Aux route, \"exact\" method at delta = {dref:g}: delta^2 sum_e v_e = sum_e Var(Q_e) per shot "
             "(delta-independent up to O(delta^2)), and the per-element mean of delta^2 v_e (matchgate). QWC "
             "settings in parentheses. trotter/exact: total(df-trotter) / total(exact); frag/(L+1)^2: "
             "total(df-fragments) / ((L + 1)^2 total(exact)); both as the range over delta, schemes and orders. "
             "Direct route: sum_e v_e and the per-element min / mean / max (matchgate: Rao-Blackwellized with "
             "the settings in the per-system files, H4 exact; Pauli: exact, '–' where not computed).", "",
             "| system | iterate | kind | K | QWC blocked | QWC interleaved | Pauli blocked | Pauli interleaved | "
             "matchgate | MG per element | trotter/exact | frag/(L+1)^2 (L) | direct MG | direct MG min/mean/max | "
             "direct Pauli |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, res, s in _res_states(results):
        for kind in KINDS:
            r = s["residual"][kind]
            cell = []
            for scheme, order in (("qwc", "blocked"), ("qwc", "interleaved"), ("pauli_shadow", "blocked"),
                                  ("pauli_shadow", "interleaved"), ("matchgate_shadow", "blocked")):
                e = _aux(s, kind, "exact", dref, scheme, order)
                txt = _g(e and e["q_total"])
                if scheme == "qwc" and e is not None:
                    txt += f" ({e['n_settings']})"
                cell.append(txt)
            mg = _aux(s, kind, "exact", dref, "matchgate_shadow")
            cell.append(_g(mg and mg["q_per_element"]["mean"]))
            cell.append(_ratio_range(s, kind, "df-trotter", deltas, lambda e: 1.0))
            fr = _ratio_range(s, kind, "df-fragments", deltas, lambda e: (e["n_leaves"] + 1) ** 2)
            cell.append(f"{fr} ({res['n_leaves']})" if fr != "–" else "–")
            dm = _direct(s, kind, "matchgate_shadow")
            dp = _direct(s, kind, "pauli_shadow")
            cell.append(_g(dm and dm["total"]))
            cell.append("–" if dm is None else
                        f"{_g(dm['per_element']['min'])} / {_g(dm['per_element']['mean'])} / {_g(dm['per_element']['max'])}")
            cell.append(_g(dp and dp["total"]))
            lines.append(f"| {name} | {state_label(s['iterate'])} | {kind} | {r['K']} | " + " | ".join(cell) + " |")
    return lines + [""]


def _best(s, kind, method, scheme, order, deltas):
    best = None
    for d in deltas:
        e = _aux(s, kind, method, d, scheme, order)
        if e is None or e.get("shots_bias") is None:
            continue
        if best is None or e["shots_bias"] < best["shots_bias"]:
            best = e
    return best


def md_shots(results: dict, deltas) -> list[str]:
    lines = ["## Residual: shots for a 30% relative Frobenius error (full tensor)", "",
             "Bias-aware shots at the best delta of the grid (delta in brackets; 'unreach.' if ||bias|| >= "
             "0.3 ||A|| at every delta); the direct route is unbiased. Variance-only shots at every delta are "
             "in the per-system files. Not defined at FCI.", "",
             "| system | iterate | kind | aux exact QWC blocked | aux exact QWC interleaved | aux exact Pauli "
             "blocked | aux exact MG | df-trotter MG | df-fragments MG | direct MG | direct Pauli |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, res, s in _res_states(results):
        if s["iterate"] == "fci":
            continue
        for kind in KINDS:
            cells = []
            for method, scheme, order in (("exact", "qwc", "blocked"), ("exact", "qwc", "interleaved"),
                                          ("exact", "pauli_shadow", "blocked"), ("exact", "matchgate_shadow", "blocked"),
                                          ("df-trotter", "matchgate_shadow", "blocked"),
                                          ("df-fragments", "matchgate_shadow", "blocked")):
                b = _best(s, kind, method, scheme, order, deltas)
                cells.append("unreach." if b is None else f"{_g(b['shots_bias'])} [{b['delta']:g}]")
            for scheme in ("matchgate_shadow", "pauli_shadow"):
                e = _direct(s, kind, scheme)
                cells.append(_g(e and e["shots_var"]))
            lines.append(f"| {name} | {state_label(s['iterate'])} | {kind} | " + " | ".join(cells) + " |")
    return lines + [""]


def md_claims(rows: list[dict]) -> list[str]:
    lines = ["## Comparison with the earlier, unverified H6 claims", "",
             f"Ours for H6/STO-3G at 1.0 A (h6a): iterate 2, the iterate closest to {CLAIM_ERROR * 1e3:.1f} mHa above "
             "FCI (the claims' state), the two states bracketing it and FCI (iterate and E - E_FCI in mHa in each "
             "cell). The 1.5 A energy claims use h6b, with the verdict at FCI (their FCI-like states). Ratio = ours "
             "/ claimed at iterate 2 and at the closest state. Verdict: reproduced if the ratio at the verdict state "
             "is within 30% (for a claimed range: our value within [0.7 lo, 1.3 hi]). Alternatives (iterate 2 / "
             "closest / FCI): other variants of our computation.", "",
             "| claim | claimed | ours, iterate 2 | ours, closest | bracket (above / below) | ours, FCI | ratio (it. 2 / "
             "closest) | verdict | alternatives | likely reason |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        pct = r["id"].startswith("bias")
        fmt = _pct if pct else _g
        if r["range"] is not None:
            claimed = f"{fmt(r['range'][0])}-{fmt(r['range'][1])}"
        else:
            claimed = _g(r["claimed"])

        def cell(key):
            v, st = r["values"].get(key), r["states"].get(key)
            if v is None or st is None:
                return "–"
            where = "FCI" if st["iterate"] == "fci" else f"it. {st['iterate']}, {st['e_err'] * 1e3:.2f}"
            return f"{fmt(v)} ({where})"

        ratio = "–"
        if r.get("ratio") is not None and (r["ratio"]["it2"] is not None or r["ratio"]["closest"] is not None):
            ratio = f"{_g(r['ratio']['it2'])} / {_g(r['ratio']['closest'])}"
        alts = "; ".join(f"{k}: {fmt(v['it2']) if not k.startswith('groups') else _g(v['it2'])} / "
                         f"{fmt(v['closest']) if not k.startswith('groups') else _g(v['closest'])} / "
                         f"{fmt(v['fci']) if not k.startswith('groups') else _g(v['fci'])}"
                         for k, v in r["alternatives"].items())
        verdict = r["verdict"] + (" (at FCI)" if r["verdict_state"] == "fci" else "")
        bracket = f"{cell('above')} / {cell('below')}"
        lines.append(f"| {r['claim']} | {claimed} | {cell('it2')} | {cell('closest')} | {bracket} | {cell('fci')} | "
                     f"{ratio} | {verdict} | {alts or '–'} | {r['reason']} |")
    return lines + [""]


def md_timing(results: dict, total: float, cache_check) -> list[str]:
    lines = ["## Runtime (one thread)", "",
             "| system | energy (s) | aux route (s) | direct matchgate (s) | direct Pauli (s) | total (s) | "
             "oracle tables built / reused |", "|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        t = res["timing"]
        c = res.get("oracle_cache", {})
        lines.append(f"| {name} | {t['energy']:.1f} | {t['aux']:.1f} | {t['direct_matchgate']:.1f} | "
                     f"{t['direct_pauli']:.1f} | {res['seconds']:.1f} | {c.get('misses', '–')} / {c.get('hits', '–')} |")
    lines.append(f"| all | | | | | {total:.1f} | |")
    if cache_check is not None:
        lines += ["", f"Oracle cache check (H4 iterate 2, {cache_check['n_results']} results): bitwise identical = "
                      f"{cache_check['bitwise_identical']}."]
    return lines + [""]


def md_system(res: dict, deltas) -> str:
    name = res["system"]
    lines = [f"# Measurement cost: {name} ({res['label']})", "",
             f"N = {res['norb']}, nelec = {tuple(res['nelec'])}, E_FCI = {res['e_fci']:.10f} Ha, DF leaves L = "
             f"{res['n_leaves']} (Cholesky) / {res['n_leaves_eigh']} (eigh); H has {res['hamiltonian_terms']['blocked']} "
             f"Pauli terms, {res['energy_qwc_groups']['blocked']} / {res['energy_qwc_groups']['interleaved']} QWC groups "
             "(blocked / interleaved).", ""]
    if "direct_targets" in res:
        dt = res["direct_targets"]
        lines += [f"Direct-route targets X_e: {dt['spin_orbital']['terms']} / {dt['spin_free']['terms']} Pauli terms "
                  f"(spin-orbital / spin-free), {_g(dt['spin_orbital']['pairs'])} / {_g(dt['spin_free']['pairs'])} "
                  "term pairs.", ""]
    for s in res["states"]:
        lines.append(f"## Iterate {state_label(s['iterate'])}: E - E_FCI = {s['e_err'] * 1e3:.3f} mHa, "
                     f"norm A (full) = {_g(s['norm_A_full_spin_free'], 4)}, sigma^2 = {_g(s['sigma2'], 4)}")
        lines.append("")
        if "residual" not in s:
            continue
        if s.get("plan"):
            p = s["plan"]
            lines += [f"Plan: df-fragments orders {p['fragment_orders'] or 'none'}; matchgate RB settings "
                      f"{p['rb_settings']}; Pauli direct kinds {p['pauli_kinds'] or 'none'} (always for small targets).", ""]
        for kind in KINDS:
            r = s["residual"][kind]
            lines += [f"### {kind} (K = {r['K']}, norm A full = {_g(r['norm_A_full'], 4)}, elements = "
                      f"{_g(r['norm_A_elements'], 4)})", "",
                      "| route | delta | rel. bias | scheme | order | settings | sum_e v_e | v_e min / mean / max | "
                      "delta^2 v_e mean | S (variance only) | S (bias-aware) |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
            for e in r["aux"]:
                pe = e["per_element"]
                lines.append(f"| aux {e['route_method']} | {e['delta']:g} | {_pct(e['rel_bias'])} | {e['scheme']} | "
                             f"{e['order']} | {_g(e['n_settings'])} | {_g(e['total'])} | {_g(pe['min'])} / "
                             f"{_g(pe['mean'])} / {_g(pe['max'])} | {_g(e['q_per_element']['mean'])} | "
                             f"{_shots(e, 'shots_var')} | {_shots(e, 'shots_bias')} |")
            for e in r["direct"]:
                pe = e["per_element"]
                se = "" if e.get("stderr") is None else f" +- {_g(e['stderr'], 2)}"
                meth = e["method"] + ("" if e["method"] != "rb" else f" ({e['n_samples']} settings, seed {e['seed']})")
                lines.append(f"| direct | – | 0 | {e['scheme']} | {e['order']} | {meth} | {_g(e['total'])}{se} | "
                             f"{_g(pe['min'])} / {_g(pe['mean'])} / {_g(pe['max'])} | – | {_shots(e, 'shots_var')} | "
                             f"{_shots(e, 'shots_var')} |")
            lines.append("")
    return "\n".join(lines) + "\n"


def markdown_summary(results: dict, rows: list[dict], args, total: float, cache_check) -> str:
    lines = ["# Measurement cost of the CQE energy and residual (M2, workflow B)", "",
             f"Exact per-shot variances along exact SD CQE trajectories (`benchmarks/run_measurement.py`). Systems "
             f"{', '.join(results)}; states {', '.join(state_label(s) for s in args.states)}; deltas "
             f"{', '.join(f'{d:g}' for d in args.deltas)}; residual systems {', '.join(args.residual_systems)}. "
             "Conventions and reductions: the script's module docstring. Per-system detail: measurement_<system>.md.", ""]
    lines += md_claims(rows)
    lines += md_states(results)
    lines += md_energy(results)
    if any(True for _ in _res_states(results)):
        lines += md_bias(results, args.deltas)
        lines += md_residual_variance(results, args.deltas)
        lines += md_shots(results, args.deltas)
    lines += md_timing(results, total, cache_check)
    return "\n".join(lines) + "\n"


# ======================================================================= main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default=DEFAULT_SYSTEMS, help="comma-separated subset of h4,h6a,h6b,h6,lih,n2,h8")
    ap.add_argument("--residual-systems", default=DEFAULT_RESIDUAL,
                    help="systems that also get the residual tables (the others: energy only)")
    ap.add_argument("--states", default=DEFAULT_STATES, help="SD iterates and/or 'fci' (comma-separated)")
    ap.add_argument("--deltas", default=DEFAULT_DELTAS, help="aux-route step sizes (comma-separated)")
    ap.add_argument("--rb-settings", type=int, default=5000,
                    help="random matchings of the Rao-Blackwellized direct-route matchgate estimate (N >= 6)")
    ap.add_argument("--rb-settings-claims", type=int, default=20000, help="the same at the claims states")
    ap.add_argument("--pauli-direct", default="claims", choices=("claims", "all", "none"),
                    help="where the exact direct-route Pauli-shadow sums run for N >= 6")
    ap.add_argument("--quick", action="store_true", help="H4 only, states 0,2,5,fci, deltas 0.1,0.6 (< 2 min)")
    ap.add_argument("--no-cache", action="store_true", help="disable the oracle cache")
    ap.add_argument("--verify-cache", action="store_true", help="check the cache for bitwise identity first")
    ap.add_argument("--threads", type=int, default=None, help="PySCF OpenMP and numpy BLAS threads")
    ap.add_argument("--out", default="results/", help="output directory")
    args = ap.parse_args(argv)
    if args.quick:
        for k, v in QUICK.items():
            setattr(args, k, v)
        args.verify_cache = not args.no_cache
    args.systems = _names(args.systems)
    args.residual_systems = [s for s in _names(args.residual_systems) if s in args.systems]
    args.states = _states(args.states)
    args.deltas = _floats(args.deltas)
    args.threads_env = pin_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.perf_counter()
    cache_check = verify_cache(args) if args.verify_cache else None
    if cache_check is not None:
        print(f"oracle cache check: {cache_check}", flush=True)
    results = {}
    settings = {k: v for k, v in vars(args).items()}
    for name in args.systems:
        res = run_system(name, args)
        res["settings"] = settings
        results[name] = res
        with open(os.path.join(args.out, f"measurement_{name}.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=_json_default)
        with open(os.path.join(args.out, f"measurement_{name}.md"), "w") as fh:
            fh.write(md_system(res, args.deltas))
        t = res["timing"]
        print(f"[{name}] done in {res['seconds']:.1f} s (energy {t['energy']:.1f}, aux {t['aux']:.1f}, direct MG "
              f"{t['direct_matchgate']:.1f}, direct Pauli {t['direct_pauli']:.1f}; cache {res['oracle_cache']})",
              flush=True)
    rows = claims(results)
    total = time.perf_counter() - t0
    with open(os.path.join(args.out, "measurement_summary.md"), "w") as fh:
        fh.write(markdown_summary(results, rows, args, total, cache_check))
    summary = {"settings": settings, "claims": rows, "seconds": total, "cache_check": cache_check,
               "timing": {k: v["timing"] | {"total": v["seconds"]} for k, v in results.items()}}
    with open(os.path.join(args.out, "measurement_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=_json_default)
    print(f"wrote {os.path.join(args.out, 'measurement_summary.md')} ({total:.1f} s)")
    return 0


def _json_default(x):
    try:
        import numpy as np

        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return _finite(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
    except ImportError:
        pass
    if isinstance(x, (set, tuple)):
        return list(x)
    return str(x)


if __name__ == "__main__":
    sys.exit(main())

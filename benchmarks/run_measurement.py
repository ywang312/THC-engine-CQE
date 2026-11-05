"""Measurement cost of the CQE energy and ACSE residual along exact CQE trajectories (M2, workflow B).

For every system the exact steepest-descent CQE (``cqe.run_cqe`` defaults: SD with the exact line
search, generator g = conj(A); "M1 SD") runs from HF. The states are the iterates ``--states`` (default
0, 2, 4, 5, 7; iterate k is the state after k steps, iterate 0 the HF determinant) and the FCI ground
state ("fci"); for the claims systems h6a and h6b the two consecutive iterates whose E - E_FCI brackets
6.6 mHa are added if they are not requested (observed: h6a 4 and 5, requested by default; h6b 10 and 11;
``--no-bracket``), and so is iterate 2 of SD with Mazziotti's spin-orbital ACSE generator ("SO-ACSE 2",
JSON key "so2"; ``--so-iterates``), the state of the earlier claims (section "Claims"). For each state the
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
    and bias-aware (inf = unreachable, ||bias|| >= 0.3 ||A||);
  * per-element statistics over the elements with |A_e| > t (t = 1e-4, 3e-4, 1e-3, 3e-3; ``subsets``
    in the JSON): their number, their share of ||A||^2 and S_ext = (sum_e m_e) mean_t / (0.09 ||A||^2),
    the shots if every element had the subset mean (the earlier direct-route claims match t = 1e-3).
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
* Oracle cache. Every scheme call builds a dense table of <P(x, z)> for its state (0.2-0.8 s for an
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
  ``--rb-settings-claims`` (default 20000) at the claims states; the standard error of the total and
  the largest per-element relative standard error are reported (the min and max over elements of
  noisy estimates are biased outward). At the claims states the spin-orbital set also gets the exact
  pair sums (``max_pairs=inf``; 8.8e8 commuting pairs, 157-168 s per state, observed;
  ``--direct-mg-exact none`` skips them), which the claims table uses, and the RB estimate there is
  compared with them (z-scores in the per-system files). H4 uses the exact pair sums.
* Direct route with Pauli shadows at N = 6: exact pair sums (61-263 s per call, observed) only at the
  claims states (spin-orbital at all three, spin-free at M1 SD iterate 2) with ``--pauli-direct
  claims`` (default); ``all`` computes them at every state, ``none`` nowhere. H4 always uses the exact
  sums.
The claims states (h6a) are SO-ACSE 2 (the claims' state), M1 SD iterate 2 and the M1 SD iterate closest
to 6.6 mHa above FCI (spec_B.md's comparison states). ``--rb-settings``, ``--pauli-direct all`` and
``--no-cache`` lift the reductions.

QWC tie orders (``--tie-orders n``, default 8; 0 skips them)
--------------------------------------------------------------
Sorted insertion orders the strings by decreasing weight and breaks round-off ties of the weights in the
canonical (x, z) order (``measure.schemes``); the grouping, and with it the QWC variance, depends on the
order inside the tie classes, which the algorithm leaves open. For every state of h6a and h6b (energy, both
qubit orders) and at the h6a claims states (aux route, "exact" at delta = 0.1, spin-orbital Q_e, both
orders) the QWC values are recomputed for n random orders inside the tie classes of ``measure.schemes``
(seeded, the same n groupings at every state); the summary reports the canonical value and the range.
With the canonical order, ``qwc_tie_groupings`` reproduces ``schemes.qwc_grouping`` (checked at every run).

Claims
------
The comparison with the earlier, unverified H6 analysis (spec.md "Goal") is in the summary. The claims'
state, "CQE iteration 2, about 6.6 mHa above FCI", is iteration 2 of SD with the exact line search and
Mazziotti's spin-orbital ACSE generator S = sum_{ijkl} ²A^{ij}_{kl} a+_i a+_j a_l a_k, with the
spin-orbital residual ²A^{ij}_{kl} = <[a+_i a+_j a_l a_k, H]> (written 2A^{ij}_{kl} elsewhere in this
file; sum over all spin orbitals; the restricted sum i < j, k < l is S/4, the same direction), not an
iterate of the M1 SD, whose generator is the spin-free g = conj(A). For a singlet,
S = 2 sum_pqrs c[p,q,r,s] e_{pq,rs} with the alpha-beta block c[p,q,r,s] = ²A^{(p alpha)(r beta)}_{(q alpha)(s beta)},
and the spin-free residual is A = 4c - 2c' with c'[p,q,r,s] = c[p,s,r,q], so c = (2 A + A')/6: for real
states ``run_cqe(..., compress=so_acse_generator)`` runs this trajectory (the exact line search makes the
overall factor 2 irrelevant). Observed on H4 (Fock-space oracle, scratch check of this workflow): S equals 2 sum c e to
4e-15 relative, and SD along S with its own exact line search gives the energies of the compressed
run_cqe to 3e-7 mHa. Evidence that SO-ACSE 2 is the claims' state (observed, printed in the summary's
"Trajectories" table): for h6a it is 6.558 mHa above FCI with ||A||_840 = 0.0760, and the three aux-route
shot counts claimed imply ||A||_840 = 0.0755-0.0761 through S = 840 v / (0.09 ||A||^2) with the claimed
per-element variances v; M1 SD iterate 2 is 19.57 mHa above FCI (||A||_840 = 0.133), CG and L-BFGS
iterate 2 12.01 mHa. At 1.5 A (h6b) the same iterate reproduces the three energy claims within 0.3%.

For each claim the summary gives our value at the claims' state (SO-ACSE 2 of h6a; of h6b for the 1.5 A
energy claims), at M1 SD iterate 2, at the M1 SD iterate closest to 6.6 mHa above FCI and at FCI (the
comparison states of spec_B.md), the ratio ours / claimed and the verdict at each (reproduced if the
ratio is within 30%, or our value lies within [0.7 lo, 1.3 hi] of a claimed range; for the direct-route
distribution claim "7-41 (mean 12-16)": min and max within 30% and the mean inside the claimed range of
the mean, "partly reproduced" if only some of these hold), a primary verdict (at the claims' state;
without SO-ACSE states, at the closest state, or at FCI for the 1.5 A claims), alternatives (other
variants of our computation, at the primary state) and the likely reason for a discrepancy.
``--summary-only`` rebuilds the markdown files and the summary from the JSON files in ``--out`` without
computing.

Caveats (section of the summary; ORIENTATION_NOTES, TRAJECTORY_NOTE): N2's active space has two
degenerate pi pairs whose relative orientation RHF leaves to round-off; QWC, Pauli-shadow and
Cholesky-DF numbers (and the DF-route biases and every per-element min and max) depend on it and are
marked † in the summary tables; matchgate, DF-eigh, E, sigma^2 and ||A|| do not. LiH's single degenerate
pair is a symmetry. The SD iterates themselves are reproducible to about 1e-9 Ha (observed; line-search
round-off).

Threads: ``--threads n`` pins PySCF's OpenMP threads and, through OMP_NUM_THREADS / OPENBLAS_NUM_THREADS
/ MKL_NUM_THREADS, numpy's BLAS threads (effective only if numpy is not loaded yet, as when this file
runs as a script; ``settings["threads_env"]`` records what was in effect).

Outputs: <out>/measurement_<sys>.json and <out>/measurement_<sys>.md per system,
<out>/measurement_summary.md and <out>/measurement_summary.json (claims, timings).

Usage:
    python benchmarks/run_measurement.py --threads 1 --out results/
    python benchmarks/run_measurement.py --quick --threads 1 --out /tmp/quick
    python benchmarks/run_measurement.py --systems h6a,h6b --threads 1 --out results/   # the claims systems only
    python benchmarks/run_measurement.py --summary-only --out results/
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
BRACKET_MAX_ITER = 60          # SD steps searched for the iterates bracketing 6.6 mHa (claims systems)
CLAIM_SO_ITERATE = 2           # the claims' "CQE iteration 2": SD with the spin-orbital ACSE generator
CLAIM_SO_KEY = f"so{CLAIM_SO_ITERATE}"
SUBSET_THRESHOLDS = (1e-4, 3e-4, 1e-3, 3e-3)   # per-element statistics over |A_e| > t (``subsets``)
SUBSET_REF = 1e-3              # the subset that reproduces the direct-route claims (row D-MG)
TRAJ_COMPARE_ITERS = 3         # iterations of the trajectory comparison (claims systems)
ORIENTATION_DEPENDENT = ("n2",)  # systems whose numbers marked † depend on the orientation of degenerate MOs
# The earlier analysis's claimed shot counts with their claimed per-element variances (Var(A_hat_e) per shot):
# ||A||_840 implied by S = 840 v / (0.09 ||A||^2) (summary, "Trajectories").
CLAIMED_SHOTS = (("aux + QWC, delta = 0.1", 5.9e9, 36.0 / 0.1 ** 2),
                 ("aux + shadows, delta = 0.1", 8.7e8, 5.4 / 0.1 ** 2),
                 ("aux + shadows, delta = 0.3", 9.7e7, 5.4 / 0.3 ** 2),
                 ("direct shadows (mean 12)", 2.6e7, 12.0),
                 ("direct shadows (mean 16)", 2.6e7, 16.0))
CLAIMS_K = 840                 # spin-orbital elements of the claims (N = 6)
TIE_DELTA = 0.1                # delta of the aux-route QWC tie-order spread (claims states)
# Caveats printed in the markdown (section "Caveats" of the summary and the per-system files). The numbers
# are observed with rotations of the degenerate active orbitals of systems.n2 / systems.lih at FCI.
ORIENTATION_NOTES = {
    "n2": ("Orbital orientation (n2): the CAS(6,6) active space has two degenerate pairs, pi_u (active orbitals "
           "0, 1) and pi_g (3, 4). `systems.n2` runs RHF without symmetry, so the orientation of each pair is set "
           "by round-off (observed: it differed between two full runs of this script, and between fresh processes "
           "with different BLAS thread counts). The relative orientation of the two pairs is not a symmetry of the "
           "qubit Hamiltonian, and these numbers depend on it: the QWC (with its group counts) and Pauli-shadow "
           "energy variances, DF with pivoted-Cholesky leaves, the df-trotter and df-fragments biases (their "
           "fragments are the Cholesky leaves) and every per-element minimum or maximum; the aux-route QWC and "
           "Pauli-shadow totals only weakly (at most 1.1e-4 relative between the two runs). Independent of it: "
           "E, sigma^2, ||A||, the "
           "matchgate totals and per-element means (energy and both routes), DF with eigh leaves and the bias of "
           "the exact aux route. Observed at FCI over the relative angle of the pairs (7.5-degree steps over the "
           "180-degree period): QWC blocked 3.91-8.45 and interleaved 3.00-6.40 Ha^2, Pauli shadows blocked "
           "462-903 and interleaved 335-1467 Ha^2, Cholesky DF 1.033-1.037 Ha^2; matchgate 224.18 and DF eigh "
           "0.928 Ha^2 unchanged. With both pairs aligned to the x and y axes the in-pair order still matters: "
           "3.91 / 4.00 / 903 / 335 (QWC blocked / interleaved, Pauli blocked / interleaved; the orbitals of RHF "
           "with symmetry=True, pi_u ordered y, x and pi_g x, y) or 4.12 / 3.00 / 462 / 1467 (the same order in "
           "both pairs)."),
    "lih": ("Orbital orientation (lih): the active space has one degenerate pair (pi, active orbitals 3, 4) in an "
            "arbitrary orientation, but rotating it is a symmetry of H (axial symmetry), so no number here depends "
            "on it (observed: QWC, Pauli-shadow, matchgate and DF energy variances at FCI unchanged to the printed "
            "5 digits under rotations of the pair)."),
}
TRAJECTORY_NOTE = ("Reproducibility of the states: the SD iterates are reproducible to about 1e-9 Ha in E and 5e-6 "
                   "relative in ||A|| (observed for h4, lih and n2 between runs and containers; FCI and iterate 0 agree "
                   "to round-off). The exact line search locates each minimum from energy values, so its step is fixed "
                   "only to about the square root of the round-off, and round-off differences (BLAS kernels, memory "
                   "alignment) move the trajectory at that level. For h4 and lih the derived numbers changed by at "
                   "most 3e-5 relative (observed), below the printed precision except for the rounding of a last "
                   "digit, with one exception: per-element minima of QWC variances on df-trotter states (up to 1.6%, "
                   "lih).")


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


def is_so(it) -> bool:
    """A state of the spin-orbital-ACSE-generator trajectory (JSON key "so<k>")."""
    return isinstance(it, str) and it.startswith("so")


def is_sd(it) -> bool:
    """An M1 SD iterate (an int; JSON keeps ints as ints)."""
    return isinstance(it, int) and not isinstance(it, bool)


def state_label(it) -> str:
    if it == "fci":
        return "FCI"
    if is_so(it):
        return f"SO-ACSE {it[2:]}"
    return str(it)


def state_order(it) -> tuple:
    """Sort key: M1 SD iterates, then SO-ACSE iterates, then FCI."""
    if is_sd(it):
        return (0, it)
    if is_so(it):
        return (1, int(it[2:]))
    return (2, 0)


def rb_seed(system: str, it, kind: str) -> int:
    return zlib.crc32(f"{system}/{it}/{kind}".encode())


def dagger(name: str) -> str:
    """'†' for systems whose marked numbers depend on the orientation of degenerate MOs (Caveats)."""
    return "†" if name in ORIENTATION_DEPENDENT else ""


def so_acse_generator(g, rd, it):
    """Mazziotti's spin-orbital ACSE generator as a ``run_cqe`` compress callback (module docstring, "Claims").

    For a singlet and real states, g = conj(A) = A (spin-free) and the generator
    S = sum_{ijkl} ²A^{ij}_{kl} a+_i a+_j a_l a_k equals 2 sum_pqrs c[p,q,r,s] e_{pq,rs} with
    c = (2 g + g.transpose(0, 3, 2, 1)) / 6, the alpha-beta block of ²A; the overall factor 2 is
    irrelevant with the exact line search.
    """
    return (2 * g + g.transpose(0, 3, 2, 1)) / 6


def qwc_tie_groupings(targets, n: int, seed: int) -> list:
    """Sorted-insertion QWC groupings with random orders inside the round-off tie classes of the weights.

    Returns [canonical, random_1, ..., random_n]: the strings of ``schemes``'s union basis, ordered by
    decreasing weight sum_e |W_ej| in the tie classes of ``schemes.qwc_grouping`` (consecutive sorted
    weights within ``schemes._QWC_TIE_RTOL`` max w), and inside each class in canonical (index) order
    (element 0, which must equal ``schemes.qwc_grouping``; checked) or in a random order (seeded); then the
    first-fit insertion of ``schemes``. The groupings depend only on the targets, not on the state.
    """
    import numpy as np

    from thc_engine_cqe.measure import schemes

    tg = schemes._prepare(targets)
    weight = np.asarray(abs(tg.W).sum(axis=0)).ravel()
    cand = np.flatnonzero(~((tg.xs == 0) & (tg.zs == 0)))
    w = weight[cand]
    srt = np.argsort(-w, kind="stable")
    ws = w[srt]
    new_class = np.ones(ws.size, dtype=bool)
    if ws.size:
        new_class[1:] = (ws[:-1] - ws[1:]) > schemes._QWC_TIE_RTOL * float(ws[0])
    cls = np.empty(ws.size, dtype=np.int64)
    cls[srt] = np.cumsum(new_class)
    rng = np.random.default_rng(seed)

    def first_fit(order):
        group = np.full(tg.B, -1, dtype=np.int64)
        gx = np.zeros(max(cand.size, 1), dtype=np.uint64)
        gz = np.zeros(max(cand.size, 1), dtype=np.uint64)
        ng = 0
        for j in order:
            x, z = tg.xs[j], tg.zs[j]
            s = x | z
            g = -1
            if ng:
                hx, hz = gx[:ng], gz[:ng]
                ok = (((hx ^ x) | (hz ^ z)) & ((hx | hz) & s)) == 0
                if ok.any():
                    g = int(np.argmax(ok))
            if g < 0:
                g, ng = ng, ng + 1
            gx[g] |= x
            gz[g] |= z
            group[j] = g
        return schemes.QWCGrouping(tg.n, tg.xs, tg.zs, group, gx[:ng].copy(), gz[:ng].copy(), order)

    out = [first_fit(cand[np.lexsort((cand, cls))])]
    ref = schemes.qwc_grouping(targets)
    if not (np.array_equal(out[0].group, ref.group) and np.array_equal(out[0].order, ref.order)):
        raise SystemExit("qwc_tie_groupings: the canonical order does not reproduce schemes.qwc_grouping")
    for _ in range(int(n)):
        out.append(first_fit(cand[np.lexsort((rng.permutation(cand.size), cls))]))
    return out


def tie_summary(values) -> dict | None:
    """{"canonical", "min", "max", "n"} of [canonical, random_1, ...] (None without random orders)."""
    if values is None or len(values) < 2:
        return None
    rnd = values[1:]
    return {"canonical": values[0], "min": min(rnd), "max": max(rnd), "n": len(rnd)}


# ===================================================================== trajectory
def trajectory(act, iterates, bracket: float | None = None, so_iterates=()):
    """SD iterates of the exact CQE from HF (iterate k = after k steps) and the FCI state.

    Returns (E_FCI, [(iterate, ci)], extra): the requested ``iterates`` (ints and/or "fci"), plus, with
    ``bracket`` (Ha), the two consecutive iterates whose E - E_FCI bracket it (the last at or above it
    and the first below it; SD with the exact line search lowers the energy at every step), searched
    up to BRACKET_MAX_ITER steps, plus the iterates ``so_iterates`` (k >= 1) of SD with the spin-orbital
    ACSE generator (``so_acse_generator``), keyed "so<k>". ``extra`` lists the bracketing iterates that
    were not requested; the states are returned in ``state_order``: SD iterates, SO-ACSE iterates, FCI.
    """
    import numpy as np

    from thc_engine_cqe.cqe import run_cqe
    from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian

    e_fci, ci_fci = act.fci()
    space = FCISpace.of(act)
    so_want = sorted({int(k) for k in so_iterates})
    if so_want and so_want[0] < 1:
        raise SystemExit("--so-iterates must be >= 1 (iterate 0 is the HF state of every trajectory)")
    so_cis = {}
    if so_want:
        def so_callback(step, ci_after):
            so_cis[len(so_cis) + 1] = np.array(ci_after, copy=True)

        run_cqe(act, max_iter=so_want[-1], compress=so_acse_generator, callback=so_callback)
    want = sorted(k for k in iterates if k != "fci")
    n_iter = max(want[-1] if want else 0, BRACKET_MAX_ITER if bracket is not None else 0)
    cis = {0: space.hf()}
    if n_iter > 0:
        def callback(step, ci_after):
            cis[len(cis)] = np.array(ci_after, copy=True)

        run_cqe(act, max_iter=n_iter, callback=callback)
    extra = []
    if bracket is not None:
        ham = Hamiltonian(act)
        err = {k: ham.energy(np.asarray(c).reshape(space.shape) / np.linalg.norm(c)) - e_fci for k, c in cis.items()}
        above = [k for k in cis if err[k] >= bracket]
        if above and max(above) + 1 in cis:
            extra = [k for k in (max(above), max(above) + 1) if k not in want]
        else:
            print(f"warning: no SD iterate pair brackets E - E_FCI = {bracket * 1e3:.2f} mHa within "
                  f"{len(cis) - 1} steps", flush=True)
    out = []
    for k in sorted(set(want) | set(extra)) + (["fci"] if "fci" in iterates else []):
        if k == "fci" or k in cis:
            c = np.asarray(ci_fci if k == "fci" else cis[k], dtype=float).reshape(space.shape)
            out.append((k, c / np.linalg.norm(c)))
        else:
            raise SystemExit(f"iterate {k} was not reached (the CQE stopped after {len(cis) - 1} steps)")
    for k in so_want:
        if k not in so_cis:
            raise SystemExit(f"SO-ACSE iterate {k} was not reached (the CQE stopped after {len(so_cis)} steps)")
        c = np.asarray(so_cis[k], dtype=float).reshape(space.shape)
        out.append((f"so{k}", c / np.linalg.norm(c)))
    out.sort(key=lambda p: state_order(p[0]))
    return float(e_fci), out, extra


def trajectory_comparison(act, e_fci: float, n_iter: int = TRAJ_COMPARE_ITERS) -> list[dict]:
    """E - E_FCI and ||A|| after k = 1..n_iter steps from HF (exact line search) for the M1 SD, SD with the
    spin-orbital ACSE generator, CG and L-BFGS (module docstring, "Claims")."""
    import numpy as np

    from thc_engine_cqe.cqe import run_cqe
    from thc_engine_cqe.fci_ops import FCISpace, Hamiltonian
    from thc_engine_cqe.measure import routes, targets
    from thc_engine_cqe.residual import compute_residual

    space = FCISpace.of(act)
    ham = Hamiltonian(act)
    el = targets.element_set(act.norb, "spin_orbital")
    rows = []
    for key, name, kw in (("sd", "M1 SD (generator conj(A))", {}),
                          ("so", "SD, spin-orbital ACSE generator", {"compress": so_acse_generator}),
                          ("cg", "CG (Polak-Ribiere+)", {"optimizer": "cg"}),
                          ("lbfgs", "L-BFGS", {"optimizer": "lbfgs"})):
        cis = []
        run_cqe(act, max_iter=n_iter, callback=lambda st, c: cis.append(np.array(c, copy=True)), **kw)
        its = []
        for k, c in enumerate(cis, start=1):
            c = np.asarray(c, dtype=float).reshape(space.shape)
            c = c / np.linalg.norm(c)
            its.append({"iterate": k, "e_err": float(ham.energy(c) - e_fci),
                        "norm_A_elements_spin_orbital": float(np.linalg.norm(
                            routes.exact_residual(act, c, el, "spin_orbital"))),
                        "norm_A_full_spin_free": float(compute_residual(act, c, ham).residual_norm)})
        rows.append({"key": key, "trajectory": name, "iterates": its})
    return rows


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
def energy_part(act, ci, ctx, ties: bool = False) -> dict:
    """Per-shot energy variances of every scheme on one state (module docstring); with ``ties``, also the
    QWC variances of the random tie orders (``ctx["energy_tie_groupings"]``)."""
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
        if ties and ctx.get("energy_tie_groupings"):
            rs = [schemes.qwc_variance(h, psi, grouping=g) for g in ctx["energy_tie_groupings"][order]]
            out["qwc"][order]["tie_orders"] = {"variance": [x.total for x in rs], "groups": [x.n_settings for x in rs],
                                               "uniform": [x.uniform_total for x in rs]}
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
    if defined:
        rec["subsets"] = subset_stats(per, a, w, d, res.per_target_uniform)
    return rec


def subset_stats(per, a, w, d, per_uniform=None) -> list[dict]:
    """Per-element statistics over the elements with |A_e| > t for t in SUBSET_THRESHOLDS (module docstring):
    their number, share of ||A||^2 (full-tensor weights), stats of v_e (and of delta^2 v_e, also under
    uniform allocation for QWC) and S_ext = sum_e m_e mean_t(v) / (0.09 sum_e m_e A_e^2)."""
    import numpy as np

    per = np.asarray(per, dtype=float)
    a = np.asarray(a, dtype=float)
    w = np.asarray(w, dtype=float)
    a2 = w * a * a
    out = []
    for t in SUBSET_THRESHOLDS:
        mask = np.abs(a) > t
        if not mask.any():
            continue
        mean = float(per[mask].mean())
        rec = {"threshold": t, "n": int(mask.sum()), "norm2_fraction": float(a2[mask].sum() / a2.sum()),
               "per_element": stats(per[mask]),
               "shots_extrapolated": float(w.sum() * mean / (REL_ERROR ** 2 * a2.sum()))}
        if d is not None:
            rec["q_per_element"] = stats(per[mask] * d * d)
            if per_uniform is not None:
                rec["q_per_element_uniform"] = stats(np.asarray(per_uniform, dtype=float)[mask] * d * d)
        out.append(rec)
    return out


def aux_tie_orders(act, ci, d, states, exact_so, ctx) -> dict:
    """QWC on the aux route ("exact", delta ``d``, spin-orbital Q_e) for the canonical and the random tie
    orders (``ctx["aux_tie_groupings"]``), both qubit orders: groups, delta^2 mean v_e (uniform and optimal
    allocation) and the variance-only shots (uniform and optimal)."""
    import numpy as np

    from thc_engine_cqe.measure import routes

    el, a, w = exact_so
    out = {}
    for order in ORDERS:
        rec = {"groups": [], "q_mean_uniform": [], "q_mean": [], "shots_var_uniform": [], "shots_var": []}
        for g in ctx["aux_tie_groupings"][order]:
            r = routes.aux_residual_variance(act, ci, d, "exact", el, "spin_orbital", "qwc",
                                             mapping=ctx["mappings"][order], states=states, grouping=g)
            pu = np.asarray(r.per_target_uniform, dtype=float)
            po = np.asarray(r.per_target, dtype=float)
            rec["groups"].append(int(r.n_settings))
            rec["q_mean_uniform"].append(float(pu.mean() * d * d))
            rec["q_mean"].append(float(po.mean() * d * d))
            rec["shots_var_uniform"].append(_finite(routes.shots_for_relative_error(pu, a, REL_ERROR, weights=w)))
            rec["shots_var"].append(_finite(routes.shots_for_relative_error(po, a, REL_ERROR, weights=w)))
        out[order] = rec
    out["delta"] = d
    return out


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
            if plan.get("tie_aux") and method == "exact" and abs(d - TIE_DELTA) < 1e-12 and ctx.get("aux_tie_groupings"):
                out["spin_orbital"]["qwc_tie_orders"] = aux_tie_orders(act, ci, d, states, exact["spin_orbital"], ctx)
    timing["aux"] = time.perf_counter() - t_aux
    # ---- direct route
    m = ctx["mappings"]["blocked"]
    for kind in KINDS:
        el, a, w = exact[kind]
        xs = ctx["direct_targets"][kind]
        small = ctx["direct_pairs"][kind] <= SMALL_PAIRS
        runs = []                                   # (scheme, scheme options)
        if small:
            runs.append(("matchgate_shadow", {}))
        else:
            if kind in plan["mg_exact_kinds"]:      # exact pair sums (8.8e8 commuting pairs at H6, observed)
                runs.append(("matchgate_shadow", {"max_pairs": np.inf}))
            runs.append(("matchgate_shadow", {"scheme_method": "rb", "n_settings": plan["rb_settings"],
                                              "seed": rb_seed(ctx["name"], it, kind)}))
        if small or kind in plan["pauli_kinds"]:
            runs.append(("pauli_shadow", {}))
        exact_mg = None
        for scheme, opts in runs:
            t = time.perf_counter()
            res = routes.direct_residual_variance(act, ci, el, kind, scheme, mapping=m, targets=xs, **opts)
            rec = _entry(res, a, w, None, {"route": "direct", "scheme": scheme, "order": "blocked",
                                           "seed": opts.get("seed")}, it != "fci")
            rec["seconds"] = time.perf_counter() - t
            if scheme == "matchgate_shadow" and res.method == "exact":
                exact_mg = res
            elif scheme == "matchgate_shadow" and exact_mg is not None:      # RB against the exact sums
                pz = (np.asarray(res.per_target) - exact_mg.per_target) / np.asarray(res.per_target_stderr)
                rec["vs_exact"] = {"total_exact": float(exact_mg.total),
                                   "z_total": float((res.total - exact_mg.total) / res.stderr),
                                   "per_element_z_sd": float(np.std(pz)), "per_element_z_max": float(np.abs(pz).max()),
                                   "per_element_rel_dev_max": float(np.max(np.abs(res.per_target / exact_mg.per_target - 1)))}
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
    claims_system = name in (CLAIMS_SYSTEM, CLAIMS_SYSTEM_STRETCHED)
    so_iterates = args.so_iterates if claims_system else ()
    e_fci, states, extra = trajectory(act, args.states, CLAIM_ERROR if claims_system and not args.no_bracket else None,
                                      so_iterates=so_iterates)
    residual = name in args.residual_systems
    print(f"[{name}] {act.label}: N={n}, nelec={act.nelec}, {len(states)} states"
          f"{f' (added the iterates {extra} bracketing 6.6 mHa)' if extra else ''}"
          f"{f' (added SO-ACSE iterates {list(so_iterates)})' if so_iterates else ''}, "
          f"residual={'yes' if residual else 'no'}", flush=True)
    ctx = {"name": name, "deltas": args.deltas,
           "mappings": {o: QubitMapping(n, o) for o in ORDERS}}
    ctx["hamiltonian"] = {o: jw_hamiltonian(act, ctx["mappings"][o]) for o in ORDERS}
    ctx["energy_grouping"] = {o: schemes.qwc_grouping(ctx["hamiltonian"][o]) for o in ORDERS}
    ctx["df_energy"] = {"cholesky": double_factorize(act.eri), "eigh": double_factorize(act.eri, first="eigh")}
    ctx["df"] = ctx["df_energy"]["cholesky"]
    ties = claims_system and args.tie_orders > 0
    if ties:                       # canonical + random tie orders, the same groupings at every state
        ctx["energy_tie_groupings"] = {o: qwc_tie_groupings(ctx["hamiltonian"][o], args.tie_orders,
                                                            zlib.crc32(f"{name}/energy/{o}".encode()))
                                       for o in ORDERS}
    out = {"system": name, "label": act.label, "norb": n, "nelec": list(act.nelec), "e_fci": e_fci,
           "n_leaves": int(ctx["df"].n_leaves), "n_leaves_eigh": int(ctx["df_energy"]["eigh"].n_leaves),
           "hamiltonian_terms": {o: int(ctx["hamiltonian"][o].n_terms) for o in ORDERS},
           "energy_qwc_groups": {o: int(ctx["energy_grouping"][o].n_groups) for o in ORDERS},
           "bracket_iterates_added": extra, "so_iterates_added": [f"so{k}" for k in sorted(set(so_iterates))],
           "tie_orders": args.tie_orders if ties else 0, "states": [],
           "timing": {"energy": 0.0, "aux": 0.0, "direct_matchgate": 0.0, "direct_pauli": 0.0}}
    if claims_system:
        t = time.perf_counter()
        out["trajectories"] = trajectory_comparison(act, e_fci)
        out["trajectories_seconds"] = time.perf_counter() - t
    infos = [(it, ci, state_info(act, ci, e_fci)) for it, ci in states]
    # claims states (h6a): SO-ACSE 2, M1 SD iterate 2 and the M1 SD iterate closest to 6.6 mHa above FCI
    claims_states = set()
    if name == CLAIMS_SYSTEM:
        sd = [(it, inf) for it, _, inf in infos if is_sd(it)]
        if any(it == 2 for it, _ in sd):
            claims_states.add(2)
        if sd:
            claims_states.add(min(sd, key=lambda p: abs(p[1]["e_err"] - CLAIM_ERROR))[0])
        if any(it == CLAIM_SO_KEY for it, _, _ in infos):
            claims_states.add(CLAIM_SO_KEY)
    out["claims_states"] = sorted(claims_states, key=state_order)
    if residual:
        ctx["elements"] = {k: targets.element_set(n, k) for k in KINDS}
        ctx["weights"] = {k: targets.element_multiplicities(ctx["elements"][k], k, n) for k in KINDS}
        ctx["aux_groups"] = {}
        for kind in KINDS:
            ctx["aux_groups"][kind] = {
                o: schemes.qwc_grouping(targets.aux_targets(ctx["elements"][kind], ctx["mappings"][o], kind)).n_groups
                for o in ORDERS}
        if ties and claims_states:
            so = "spin_orbital"
            ctx["aux_tie_groupings"] = {
                o: qwc_tie_groupings(targets.aux_targets(ctx["elements"][so], ctx["mappings"][o], so), args.tie_orders,
                                     zlib.crc32(f"{name}/aux/{so}/{o}".encode())) for o in ORDERS}
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
            rec["energy"] = energy_part(act, ci, ctx, ties=ties)
            out["timing"]["energy"] += rec["energy"]["seconds"]
            if residual:
                claim = it in claims_states
                plan = {"fragment_orders": () if it == "fci" else (ORDERS if claim else ("blocked",)),
                        "rb_settings": args.rb_settings_claims if claim else args.rb_settings,
                        "pauli_kinds": (KINDS if args.pauli_direct == "all"
                                        else () if args.pauli_direct == "none"
                                        else (KINDS if (claim and it == 2) else ("spin_orbital",) if claim else ())),
                        "mg_exact_kinds": ("spin_orbital",) if (claim and args.direct_mg_exact == "claims") else (),
                        "tie_aux": bool(claim and ctx.get("aux_tie_groupings"))}
                rec["plan"] = {k: list(v) if isinstance(v, tuple) else v for k, v in plan.items()}
                rec["residual"] = residual_part(act, ci, it, ctx, plan)
                for k, v in rec["residual"]["timing"].items():
                    out["timing"][k] += v
            rec["seconds"] = time.perf_counter() - ts
            out["states"].append(rec)
            print(f"[{name}] state {state_label(it):>9}: E-E_FCI={info['e_err'] * 1e3:8.3f} mHa  "
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
    _, sts, _ = trajectory(act, [2])
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


def _direct(s: dict, kind: str, scheme: str, method: str | None = None):
    """The direct-route entry of a scheme (the exact one if there are several, unless ``method`` is given)."""
    if s is None or "residual" not in s:
        return None
    hits = [e for e in s["residual"][kind]["direct"]
            if e["scheme"] == scheme and (method is None or e["method"] == method)]
    exact = [e for e in hits if e["method"] == "exact"]
    return (exact or hits or [None])[0]


def _get(fn, s):
    try:
        v = fn(s)
    except (KeyError, TypeError, IndexError, ValueError, ZeroDivisionError):
        return None
    return v


def _reference_states(res: dict | None) -> dict:
    """States of one system used by the claims table: the claims' state SO-ACSE 2 ("so2"), M1 SD iterate 2,
    the M1 SD iterate closest to 6.6 mHa above FCI, the nearest M1 SD states above and below 6.6 mHa (the
    bracket; "below" may be FCI; JSON only) and FCI."""
    keys = ("so2", "it2", "closest", "above", "below", "fci")
    if res is None:
        return dict.fromkeys(keys)
    sd = [s for s in res["states"] if is_sd(s["iterate"])]
    fci = _state(res, "fci")
    above = [s for s in sd if s["e_err"] >= CLAIM_ERROR]
    below = [s for s in sd + ([fci] if fci is not None else []) if s["e_err"] < CLAIM_ERROR]
    return {"so2": _state(res, CLAIM_SO_KEY), "it2": _state(res, 2),
            "closest": min(sd, key=lambda s: abs(s["e_err"] - CLAIM_ERROR)) if sd else None,
            "above": min(above, key=lambda s: s["e_err"]) if above else None,
            "below": max(below, key=lambda s: s["e_err"]) if below else None,
            "fci": fci}


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


def _subset(entry: dict | None, t: float = SUBSET_REF) -> dict | None:
    """The ``subsets`` record of a residual entry for |A_e| > t."""
    if entry is None:
        return None
    for sub in entry.get("subsets") or []:
        if abs(sub["threshold"] - t) <= 1e-12 * t:
            return sub
    return None


def _subset_field(getter, *fields, t: float = SUBSET_REF):
    """fn(state) -> subset record field of the residual entry ``getter(state)``."""
    def f(s):
        x = _subset(getter(s), t)
        for k in fields:
            x = x[k]
        return x
    return f


def _aux_entry(kind, method, d, scheme, order):
    return lambda s: _aux(s, kind, method, d, scheme, order)


def _direct_entry(kind, scheme):
    return lambda s: _direct(s, kind, scheme)


def implied_norm(shots: float, v: float) -> float:
    """||A||_840 implied by a claimed shot count S and per-element variance v: S = 840 v / (0.09 ||A||^2)."""
    return math.sqrt(CLAIMS_K * v / (REL_ERROR ** 2 * shots))


VERDICT_STATES = ("so2", "it2", "closest", "fci")
STATE_TITLES = {"so2": "SO-ACSE it. 2", "it2": "M1 it. 2", "closest": "M1 closest", "fci": "FCI"}


def _primary(row: dict) -> None:
    """``row["verdict"]`` from the verdict at the row's verdict state ("partly reproduced" when a row with
    sub-checks, D-MG, passes some but not all of them)."""
    k = row["verdict_state"]
    ok = row["verdicts"][k]
    chk = (row.get("checks") or {}).get(k)
    if ok is False and chk and any(chk.values()):
        row["verdict"] = "partly reproduced"
    else:
        row["verdict"] = "not computed" if ok is None else "reproduced" if ok else "not reproduced"


def _within(x, claimed, rng) -> bool | None:
    """Reproduced within 30%: ours / claimed in [0.7, 1.3], or ours in [0.7 lo, 1.3 hi] for a claimed range."""
    if x is None:
        return None
    if rng is not None:
        return bool(0.7 * rng[0] <= x <= 1.3 * rng[1])
    return bool(0.7 <= x / claimed <= 1.3)


def _dmg_checks(mean, mn, mx) -> dict | None:
    """Row D-MG, claimed 7-41 (mean 12-16): min and max within 30%, mean inside the claimed range of the mean."""
    if None in (mean, mn, mx):
        return None
    return {"min": bool(_within(mn, 7.0, None)), "max": bool(_within(mx, 41.0, None)),
            "mean": bool(12.0 <= mean <= 16.0)}


def claims(results: dict) -> list[dict]:
    """The comparison with the earlier analysis (module docstring, "Claims"): one row per claim.

    Values at the reference states of h6a (1.0 A; for the 1.5 A energy claims, h6b): the claims' state
    SO-ACSE 2, M1 SD iterate 2, the M1 SD iterate closest to 6.6 mHa above FCI and FCI (and the bracket
    of 6.6 mHa in ``states``); the ratio ours / claimed and the verdict at each; the primary verdict
    (``verdict``) at SO-ACSE 2 (without it, at the closest state, or at FCI for the 1.5 A rows);
    alternatives (other variants of our computation), the QWC tie-order spreads (``ties``) and the likely
    reason for a discrepancy, built from these numbers.
    """
    ref = {"1.0": _reference_states(results.get(CLAIMS_SYSTEM)),
           "1.5": _reference_states(results.get(CLAIMS_SYSTEM_STRETCHED))}
    rows = []

    def add(cid, what, claimed, fn, *, alts=(), geometry="1.0", rng=None, ties=None):
        st = ref[geometry]
        verdict_at = "so2" if st["so2"] is not None else ("closest" if geometry == "1.0" else "fci")
        vals = {k: (_get(fn, v) if v is not None else None) for k, v in st.items()}
        row = {"id": cid, "claim": what, "claimed": claimed, "range": rng, "geometry": geometry,
               "values": vals, "verdict_state": verdict_at,
               "states": {k: None if v is None else {"iterate": v["iterate"], "e_err": v["e_err"]}
                          for k, v in st.items()},
               "alternatives": {lab: {k: (_get(f, st[k]) if st[k] is not None else None)
                                      for k in VERDICT_STATES} for lab, f in alts}}
        if ties is not None:
            row["ties"] = {lab: {k: (tie_summary(_get(f, st[k])) if st[k] is not None else None)
                                 for k in VERDICT_STATES} for lab, f in ties}
        if claimed is not None:
            row["ratio"] = {k: (None if vals[k] is None else vals[k] / claimed) for k in VERDICT_STATES}
        row["verdicts"] = {k: _within(vals[k], claimed, rng) for k in VERDICT_STATES}
        _primary(row)
        rows.append(row)
        return row

    def e_ties(order, field="variance"):
        return lambda s: s["energy"]["qwc"][order]["tie_orders"][field]

    def a_ties(order, field):
        return lambda s: s["residual"]["spin_orbital"]["qwc_tie_orders"][order][field]

    # ------------------------------------------------------------ energy
    for geo, (q, dfv, mg) in (("1.0", (28.0, 3.2, 122.0)), ("1.5", (24.7, 1.44, 72.5))):
        add(f"E-QWC-{geo}", f"energy variance, QWC (Ha^2), {geo} A", q, _ev("qwc", "interleaved"),
            alts=[("QWC blocked", _ev("qwc", "blocked"))], geometry=geo,
            ties=[("interleaved", e_ties("interleaved")), ("interleaved groups", e_ties("interleaved", "groups")),
                  ("blocked", e_ties("blocked"))])
        add(f"E-DF-{geo}", f"energy variance, DF basis rotation (Ha^2), {geo} A", dfv, _ev("df", "eigh"),
            alts=[("Cholesky leaves", _ev("df", "cholesky")), ("eigh, shifted", _ev("df", "eigh_shift")),
                  ("Cholesky, shifted", _ev("df", "cholesky_shift"))], geometry=geo)
        add(f"E-MG-{geo}", f"energy variance, matchgate shadows (Ha^2), {geo} A", mg, _ev("matchgate_shadow"),
            alts=[("Pauli shadows, interleaved", _ev("pauli_shadow", "interleaved")),
                  ("Pauli shadows, blocked", _ev("pauli_shadow", "blocked"))], geometry=geo)
    # ------------------------------------------------------------ aux route, Var(Im 2D) per element
    so = "spin_orbital"
    mg01 = _aux_entry(so, "exact", 0.1, "matchgate_shadow", "blocked")
    qi01 = _aux_entry(so, "exact", 0.1, "qwc", "interleaved")
    add("Q-MG", "aux route: Var(Im 2D) per element, shadows (mean over the 840 elements)", 5.4,
        _aux_field(so, "exact", 0.1, "matchgate_shadow", "blocked", "q_per_element", "mean"),
        alts=[("min", _aux_field(so, "exact", 0.1, "matchgate_shadow", "blocked", "q_per_element", "min")),
              ("max", _aux_field(so, "exact", 0.1, "matchgate_shadow", "blocked", "q_per_element", "max")),
              (f"mean over |A_e| > {SUBSET_REF:g}", _subset_field(mg01, "q_per_element", "mean")),
              ("Pauli shadows blocked", _aux_field(so, "exact", 0.1, "pauli_shadow", "blocked", "q_per_element", "mean")),
              ("Pauli shadows interleaved",
               _aux_field(so, "exact", 0.1, "pauli_shadow", "interleaved", "q_per_element", "mean"))])
    add("Q-QWC", "aux route: Var(Im 2D) per element, QWC tomography with 854 settings", 36.0,
        _aux_field(so, "exact", 0.1, "qwc", "interleaved", "q_per_element_uniform", "mean"),
        alts=[("interleaved, optimal allocation",
               _aux_field(so, "exact", 0.1, "qwc", "interleaved", "q_per_element", "mean")),
              ("blocked, uniform", _aux_field(so, "exact", 0.1, "qwc", "blocked", "q_per_element_uniform", "mean")),
              ("blocked, optimal", _aux_field(so, "exact", 0.1, "qwc", "blocked", "q_per_element", "mean")),
              (f"interleaved, uniform, mean over |A_e| > {SUBSET_REF:g}",
               _subset_field(qi01, "q_per_element_uniform", "mean")),
              ("groups interleaved", lambda s: s["residual"][so]["qwc_groups"]["interleaved"]),
              ("groups blocked", lambda s: s["residual"][so]["qwc_groups"]["blocked"])],
        ties=[("interleaved", a_ties("interleaved", "q_mean_uniform")),
              ("interleaved groups", a_ties("interleaved", "groups")),
              ("blocked", a_ties("blocked", "q_mean_uniform")), ("blocked groups", a_ties("blocked", "groups"))])
    # ------------------------------------------------------------ aux bias
    for d, rng in ((0.1, (0.002, 0.01)), (0.3, (0.02, 0.08)), (0.6, (0.06, 0.29))):
        add(f"bias-{d:g}", f"aux route: relative bias at delta = {d:g} (exp(-i delta H), 840 elements)", None,
            _aux_field(so, "exact", d, "matchgate_shadow", "blocked", "rel_bias"), rng=rng,
            alts=[("315 spin-free", _aux_field("spin_free", "exact", d, "matchgate_shadow", "blocked", "rel_bias")),
                  ("840 df-trotter", _aux_field(so, "df-trotter", d, "matchgate_shadow", "blocked", "rel_bias")),
                  ("840 df-fragments", _aux_field(so, "df-fragments", d, "matchgate_shadow", "blocked", "rel_bias"))])
    # ------------------------------------------------------------ direct route
    dmg = _direct_entry(so, "matchgate_shadow")
    sub = f"over |A_e| > {SUBSET_REF:g}"
    row = add("D-MG", "direct route: per-element variance, shadows, 7-41 (mean 12-16)", None,
              _direct_field(so, "matchgate_shadow", "per_element", "mean"), rng=(12.0, 16.0),
              alts=[("min", _direct_field(so, "matchgate_shadow", "per_element", "min")),
                    ("max", _direct_field(so, "matchgate_shadow", "per_element", "max")),
                    (f"{sub}: min", _subset_field(dmg, "per_element", "min")),
                    (f"{sub}: mean", _subset_field(dmg, "per_element", "mean")),
                    (f"{sub}: max", _subset_field(dmg, "per_element", "max")),
                    (f"{sub}: elements", _subset_field(dmg, "n")),
                    (f"{sub}: share of ||A||^2", _subset_field(dmg, "norm2_fraction")),
                    ("Pauli shadows, mean", _direct_field(so, "pauli_shadow", "per_element", "mean"))])
    row["checks"], row["checks_subset"] = {}, {}
    for k in VERDICT_STATES:                      # min / max within 30%, mean inside 12-16 (module docstring)
        alt = row["alternatives"]
        row["checks"][k] = _dmg_checks(row["values"][k], alt["min"][k], alt["max"][k])
        row["checks_subset"][k] = _dmg_checks(alt[f"{sub}: mean"][k], alt[f"{sub}: min"][k], alt[f"{sub}: max"][k])
        row["verdicts"][k] = None if row["checks"][k] is None else all(row["checks"][k].values())
    _primary(row)
    # ------------------------------------------------------------ shots
    add("S-aux-QWC-0.1", "shots for 30%, aux + QWC, delta = 0.1 (840 elements)", 5.9e9,
        _aux_field(so, "exact", 0.1, "qwc", "interleaved", "shots_var_uniform"),
        alts=[("interleaved, optimal", _aux_field(so, "exact", 0.1, "qwc", "interleaved", "shots_var")),
              ("blocked, uniform", _aux_field(so, "exact", 0.1, "qwc", "blocked", "shots_var_uniform")),
              ("blocked, optimal", _aux_field(so, "exact", 0.1, "qwc", "blocked", "shots_var")),
              ("interleaved, uniform, bias-aware",
               _aux_field(so, "exact", 0.1, "qwc", "interleaved", "shots_bias_uniform"))],
        ties=[("interleaved", a_ties("interleaved", "shots_var_uniform"))])
    for d, val in ((0.1, 8.7e8), (0.3, 9.7e7)):
        add(f"S-aux-MG-{d:g}", f"shots for 30%, aux + shadows, delta = {d:g} (840 elements)", val,
            _aux_field(so, "exact", d, "matchgate_shadow", "blocked", "shots_var"),
            alts=[("bias-aware", _aux_field(so, "exact", d, "matchgate_shadow", "blocked", "shots_bias")),
                  ("Pauli shadows blocked", _aux_field(so, "exact", d, "pauli_shadow", "blocked", "shots_var"))])
    add("S-direct-MG", "shots for 30%, direct shadows (840 elements)", 2.6e7,
        _direct_field(so, "matchgate_shadow", "shots_var"),
        alts=[(f"K x mean {sub} / (0.09 ||A||^2)", _subset_field(dmg, "shots_extrapolated")),
              ("Pauli shadows", _direct_field(so, "pauli_shadow", "shots_var"))])
    add("S-E-DF", "shots per line-search energy, DF, +-1.6 mHa", 1.2e6,
        lambda s: energy_shots(s["energy"]["df"]["eigh"]["variance"]),
        alts=[("Cholesky leaves", lambda s: energy_shots(s["energy"]["df"]["cholesky"]["variance"])),
              ("eigh, shifted", lambda s: energy_shots(s["energy"]["df"]["eigh_shift"]["variance"]))])
    for row in rows:
        row["reason"] = _reason(row, ref)
    return rows


def _alt(row, label, at):
    v = row["alternatives"].get(label, {})
    return v.get(at)


def _tie(row, label, at):
    return (row.get("ties") or {}).get(label, {}).get(at)


IMPLIED = {"S-aux-QWC-0.1": (5.9e9, 36.0 / 0.1 ** 2, "36 per element at delta = 0.1"),
           "S-aux-MG-0.1": (8.7e8, 5.4 / 0.1 ** 2, "5.4 per element at delta = 0.1"),
           "S-aux-MG-0.3": (9.7e7, 5.4 / 0.3 ** 2, "5.4 per element at delta = 0.3")}


def _reason(row: dict, ref: dict) -> str:
    """The likely reason for agreement or discrepancy, from the numbers of the row."""
    cid, v = row["id"], row["values"]
    at = row["verdict_state"]
    where = STATE_TITLES[at]
    st = ref[row["geometry"]]
    claimed = row["claimed"]
    trend_lab = " / ".join(STATE_TITLES[k].replace(" it. ", " ") for k in VERDICT_STATES)

    def norm_a(key):
        s = st.get(key)
        return None if s is None or "residual" not in s else s["residual"]["spin_orbital"]["norm_A_elements"]

    def trend(label=None, fmt=None):
        """Values at SO-ACSE 2 / M1 2 / M1 closest / FCI (of the row, or of an alternative)."""
        fmt = fmt or _g
        src = v if label is None else row["alternatives"][label]
        return " / ".join(fmt(src.get(k)) for k in VERDICT_STATES)

    def state_desc(key):
        s = st.get(key)
        if s is None:
            return "–"
        return f"{STATE_TITLES[key]}{'' if key in ('so2', 'it2') else ' (it. ' + state_label(s['iterate']) + ')'}, " \
               f"{_mha(s['e_err'], 2)} mHa"

    def ties_text(label, glabel=None, fmt=None, what=""):
        fmt = fmt or _g
        t = _tie(row, label, at)
        if not t:
            return ""
        txt = f"canonical tie order {fmt(t['canonical'])}"
        tg = _tie(row, glabel, at) if glabel else None
        if tg:
            txt += f" ({tg['canonical']} groups)"
        txt += f", {t['n']} random orders inside the round-off tie classes {fmt(t['min'])}-{fmt(t['max'])}"
        if tg:
            txt += f" ({tg['min']}-{tg['max']} groups)"
        if claimed is not None and min(t["canonical"], t["min"]) <= claimed <= max(t["canonical"], t["max"]):
            txt += f", a spread that contains the claimed {fmt(claimed)}"
        return txt + what

    try:
        if cid.startswith("E-QWC"):
            b = _alt(row, "QWC blocked", at)
            tt = ties_text("interleaved", "interleaved groups")
            return (f"the claim matches the interleaved (OpenFermion) order with the optimal allocation; blocked gives "
                    f"{_g(b)} ({_g(b / claimed)}x); sorted insertion leaves the order inside ties of the weights "
                    f"open{': ' + tt if tt else ''}; weakly state-dependent ({trend_lab}: {trend()})")
        if cid.startswith("E-DF"):
            txt = (f"the DF split is not unique: the claim matches leaves from the eigendecomposition of the ERI "
                   f"supermatrix (OpenFermion's low-rank form), unshifted; pivoted-Cholesky leaves give "
                   f"{_g(_alt(row, 'Cholesky leaves', at) / claimed)}x, shifted leaves (E_pq - delta_pq) "
                   f"{_g(_alt(row, 'eigh, shifted', at) / claimed)}x; state-dependent ({trend_lab}: {trend()})")
            if at != "fci" and v.get("fci") is not None and abs(v["fci"] / claimed - 1) > 0.05:
                txt += (f", so the {_signed(v['fci'] / claimed - 1)} at FCI is a state effect: at the claims' "
                        f"state the ratio is {_g(v[at] / claimed)}")
            return txt
        if cid.startswith("E-MG"):
            return (f"state-independent ({trend_lab}: {trend()}); Pauli shadows are "
                    f"{_g(_alt(row, 'Pauli shadows, interleaved', at) / v[at])}x larger (interleaved order)")
        if cid == "Q-MG":
            return (f"delta^2 v_e at delta = 0.1 (exact route, matchgate shadows), mean over all 840 elements; "
                    f"state-independent ({trend_lab}: {trend()}; per element {_g(_alt(row, 'min', at))}-"
                    f"{_g(_alt(row, 'max', at))}); over the elements with |A_e| > {SUBSET_REF:g} (row D-MG) the mean is "
                    f"{_g(_alt(row, f'mean over |A_e| > {SUBSET_REF:g}', at))}, so the claim is an all-element mean; a "
                    f"2D with a factor 1/2 would give a quarter of it; Pauli shadows give "
                    f"{_g(_alt(row, 'Pauli shadows blocked', at) / v[at])}x (blocked) and "
                    f"{_g(_alt(row, 'Pauli shadows interleaved', at) / v[at])}x (interleaved) more")
        if cid == "Q-QWC":
            opt = _alt(row, "interleaved, optimal allocation", at)
            n_g = _alt(row, "groups interleaved", at)
            t, tg = _tie(row, "interleaved", at), _tie(row, "interleaved groups", at)
            txt = (f"uniform allocation over the QWC groups (equal shots per setting), interleaved order, mean over all "
                   f"840 elements: the canonical tie order gives {_g(n_g)} groups and {_g(v[at])}")
            if t and tg:
                inside = tg["min"] <= 854 <= tg["max"]
                txt += (f"; {t['n']} random orders inside the round-off tie classes give {tg['min']}-{tg['max']} groups "
                        f"and {_g(t['min'])}-{_g(t['max'])}"
                        + (", a range that contains the claimed 854 settings: the claim is consistent with sorted "
                           "insertion and another tie order" if inside else
                           f", against the claimed 854 settings"))
            sub = _alt(row, f"interleaved, uniform, mean over |A_e| > {SUBSET_REF:g}", at)
            return (txt + f"; the optimal allocation gives {_g(opt)} ({_g(opt / claimed)}x); over |A_e| > "
                    f"{SUBSET_REF:g}: {_g(sub)} (the claim is an all-element mean)")
        if cid.startswith("bias"):
            lo, hi = row["range"]
            x = v[at]
            if x > hi:
                pos = f"{_signed(x / hi - 1)} above the upper end of the claimed range"
            elif x < lo:
                pos = f"{_signed(x / lo - 1)} below the lower end of the claimed range"
            else:
                pos = "at the top of the claimed range" if x >= 0.9 * hi else "inside the claimed range"
            txt = f"exact exp(-i delta H), relative Frobenius bias over the 840 elements: {_pct(x)} at {where}, {pos}"
            if at != "closest" and v.get("closest"):
                txt += (f"; at {state_desc('closest')} it is {_pct(v['closest'])} ({_g(x / v['closest'])}x smaller): "
                        f"at similar E - E_FCI the relative bias depends on the trajectory")
            if at != "it2" and v.get("it2") is not None:
                txt += f"; M1 SD iterate 2: {_pct(v['it2'])}"
            return (txt + f"; 315 spin-free at {where}: {_pct(_alt(row, '315 spin-free', at))}; DF-Trotter "
                    f"{_pct(_alt(row, '840 df-trotter', at))} and DF fragments {_pct(_alt(row, '840 df-fragments', at))} "
                    f"are larger (their bias does not vanish with A)")
        if cid == "D-MG":
            e = _direct(st[at], "spin_orbital", "matchgate_shadow")
            mean, mn, mx = v[at], _alt(row, "min", at), _alt(row, "max", at)
            sub = f"over |A_e| > {SUBSET_REF:g}"
            smn, smean, smx = (_alt(row, f"{sub}: {k}", at) for k in ("min", "mean", "max"))
            n_sub, frac = _alt(row, f"{sub}: elements", at), _alt(row, f"{sub}: share of ||A||^2", at)
            ok_sub = (row.get("checks_subset") or {}).get(at)

            def dev(x, ref, what):
                r = x / ref
                tag = ("agrees" if abs(r - 1) <= 0.1 else "within 30%" if 0.75 <= r <= 1.25
                       else "at the edge of the 30% band" if 0.7 <= r <= 1.3 else "outside the 30% band")
                return f"the {what} {_signed(r - 1)} ({tag})"

            mean_pos = ("inside the claimed range" if 12.0 <= mean <= 16.0 else
                        f"{_signed(mean / 12 - 1)} below the lower end of the claimed range" if mean < 12 else
                        f"{_signed(mean / 16 - 1)} above its upper end")
            txt = (f"matchgate shadows, {'exact pair sums' if e['method'] == 'exact' else 'Rao-Blackwellized'} at "
                   f"{where}, all 840 elements: min / mean / max = {_g(mn)} / {_g(mean)} / {_g(mx)} against the "
                   f"claimed 7 / 12-16 / 41: {dev(mx, 41.0, 'max')}, {dev(mn, 7.0, 'min')}, the mean {mean_pos} "
                   f"(state-independent; mean {trend_lab}: {trend()})")
            if smean is not None:
                means = [(t, _alt_subset_mean(st[at], t)) for t in SUBSET_THRESHOLDS]
                txt += (f"; over the {n_sub} elements with |A_e| > {SUBSET_REF:g} ({_pct(frac)} of ||A||^2): "
                        f"{_g(smn)} / {_g(smean)} / {_g(smx)}, which "
                        + ("reproduces all three" if ok_sub and all(ok_sub.values()) else "agrees only in part")
                        + " (mean over |A_e| > t: " + ", ".join(f"{_g(m)} for t = {t:g}" for t, m in means if m is not None)
                        + "): the claimed statistics are most likely over the elements with non-negligible A_e (a "
                          "reconstruction consistent with row S-direct-MG, not proof)")
            return txt + (f"; Pauli shadows give {_g(_alt(row, 'Pauli shadows, mean', at) / mean)}x larger values"
                          if _alt(row, "Pauli shadows, mean", at) else "")
        if cid.startswith("S-") and cid != "S-E-DF":
            na, na2, nac = norm_a(at), norm_a("it2"), norm_a("closest")
            txt = (f"S = sum_e v_e / (0.09 ||A||^2) ~ 1/||A||^2: ||A||_840 = {_g(na, 4)} at {state_desc(at)}")
            if cid in IMPLIED:
                s_cl, v_cl, what = IMPLIED[cid]
                txt += (f"; the claimed S with the claimed {what} implies ||A||_840 = "
                        f"{_g(implied_norm(s_cl, v_cl), 3)}")
            elif cid == "S-direct-MG":
                txt += (f"; with the claimed mean 12-16 the claimed S implies ||A||_840 = "
                        f"{_g(implied_norm(2.6e7, 12.0), 3)}-{_g(implied_norm(2.6e7, 16.0), 3)}")
            if at != "it2" and na2 is not None and na is not None:
                txt += (f"; at {state_desc('it2')} ||A||_840 = {_g(na2, 4)}, so S is {_g((na2 / na) ** 2)}x smaller "
                        f"there; at {state_desc('closest')} ||A||_840 = {_g(nac, 4)}")
            if cid == "S-aux-QWC-0.1":
                txt += (f"; uniform allocation over the groups (as for Var(Im 2D) = 36); the optimal allocation needs "
                        f"{_g(_alt(row, 'interleaved, optimal', at) / v[at])}x of it")
                t = _tie(row, "interleaved", at)
                if t:
                    txt += f"; {t['n']} random tie orders: {_g(t['min'])}-{_g(t['max'])}"
            elif cid.startswith("S-aux-MG"):
                txt += f"; variance only (bias-aware: {_g(_alt(row, 'bias-aware', at))})"
            elif cid == "S-direct-MG":
                ext = _alt(row, f"K x mean over |A_e| > {SUBSET_REF:g} / (0.09 ||A||^2)", at)
                txt += (f"; the claim is {_g(claimed / v[at])}x the exact all-element cost at {where}"
                        + (f"; K x (mean v_e over |A_e| > {SUBSET_REF:g}) / (0.09 ||A||^2) = {_g(ext)} reproduces it "
                           f"({_g(ext / claimed)}x): the claimed S extrapolates the subset mean of row D-MG to all 840 "
                           f"elements" if ext else ""))
            return txt
        if cid == "S-E-DF":
            return f"S = V_DF / 0.0016^2 with eigh leaves (row E-DF-1.0); {trend_lab}: {trend()}"
    except (TypeError, ZeroDivisionError, KeyError, ValueError):
        return "–"
    return "–"


def _alt_subset_mean(s, t):
    """Mean direct-route matchgate v_e over |A_e| > t at state s (None if absent)."""
    sub = _subset(_direct(s, "spin_orbital", "matchgate_shadow"), t)
    return None if sub is None else sub["per_element"]["mean"]



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


def _mha(x, dec: int = 3) -> str:
    """E - E_FCI (Ha) in mHa; round-off-level values (FCI) print as 0."""
    v = 1e3 * float(x)
    return f"{0.0 if abs(v) < 0.5 * 10.0 ** -dec else v:.{dec}f}"


def _signed(x) -> str:
    return "–" if x is None else f"{100 * x:+.3g}%"


def _shots(e: dict | None, key: str) -> str:
    if e is None:
        return "–"
    if key == "shots_bias" and e.get("unreachable"):
        return "unreach."
    return _g(e.get(key))


ORIENT_FOOTNOTE = ("† depends on the arbitrary relative orientation of the degenerate pi orbitals of N2's active space "
                   "(Caveats): one sample from the ranges given there.")


def _orient_footnote(results: dict) -> list[str]:
    return ["", ORIENT_FOOTNOTE] if any(dagger(n) for n in results) else []


def md_states(results: dict) -> list[str]:
    lines = ["## States", "",
             "Exact CQE iterates (iterate k = after k steps; 0 = HF): M1 SD (generator conj(A)) and, for h6a and "
             "h6b, SO-ACSE 2 (SD with the spin-orbital ACSE generator; the claims' state); and FCI. ||A||: full "
             "spin-free pair-order tensor (the CQE's residual norm) and the norms over the independent element sets "
             "(840/315 for N = 6), which the earlier claims use for the spin-orbital set. sigma^2 = <H^2> - "
             "<H>^2.", "",
             "| system | iterate | E - E_FCI (mHa) | norm A (full) | norm A (spin-orbital elements) | "
             "norm A (spin-free elements) | sigma^2 (Ha^2) |", "|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        for s in res["states"]:
            lines.append(f"| {name} | {state_label(s['iterate'])} | {_mha(s['e_err'])} | "
                         f"{_g(s['norm_A_full_spin_free'], 4)} | {_g(s['norm_A_elements_spin_orbital'], 4)} | "
                         f"{_g(s['norm_A_elements_spin_free'], 4)} | {_g(s['sigma2'], 4)} |")
    return lines + [""]


def md_trajectories(results: dict) -> list[str]:
    """Table of the trajectory comparison (claims systems) and the ||A|| implied by the claimed shot counts."""
    have = [(name, res) for name, res in results.items() if res.get("trajectories")]
    lines = ["## Trajectories: which state is the claims' \"CQE iteration 2, about 6.6 mHa above FCI\"", "",
             "E - E_FCI (mHa) and, in parentheses, ||A||_840 (the 840 spin-orbital elements) after k steps from HF "
             "with the exact line search, for the M1 SD (generator g = conj(A)), SD with Mazziotti's spin-orbital "
             "ACSE generator (module docstring, \"Claims\"), CG and L-BFGS.", ""]
    if have:
        n_it = max(len(t["iterates"]) for _, res in have for t in res["trajectories"])
        lines += ["| system | trajectory | " + " | ".join(f"it. {k}" for k in range(1, n_it + 1)) + " |",
                  "|---|---|" + "---|" * n_it]
        for name, res in have:
            for t in res["trajectories"]:
                cells = [f"{_mha(x['e_err'], 3)} ({_g(x['norm_A_elements_spin_orbital'], 4)})" for x in t["iterates"]]
                cells += ["–"] * (n_it - len(cells))
                lines.append(f"| {name} | {t['trajectory']} | " + " | ".join(cells) + " |")
    else:
        lines.append("(not computed in this run: the claims systems h6a/h6b were not run)")
    lines += ["", "||A||_840 implied by the claimed shot counts with the claimed per-element variances v, "
                  "sqrt(840 v / (0.09 S)): " + "; ".join(f"{lab}: {_g(implied_norm(s, v), 3)}"
                                                         for lab, s, v in CLAIMED_SHOTS) + ".", ""]
    return lines


def md_energy(results: dict) -> list[str]:
    lines = ["## Energy: per-shot variance (Ha^2)", "",
             "QWC: sorted insertion, optimal allocation (groups in parentheses); DF: basis-rotation grouping "
             "with L + 1 settings, leaves from pivoted Cholesky or the supermatrix eigendecomposition (eigh), "
             "without and with the shift c_l = tr L_l; shadows: exact per-snapshot variance. sigma^2 is the "
             "variance of a projective measurement of H. QWC with random tie orders: per-system files (h6a, h6b).", "",
             "| system | iterate | QWC blocked | QWC interleaved | DF Cholesky (L+1) | DF eigh (L+1) | "
             "DF Cholesky shifted | DF eigh shifted | Pauli blocked | Pauli interleaved | matchgate | sigma^2 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        o = dagger(name)
        for s in res["states"]:
            en = s["energy"]
            q, df = en["qwc"], en["df"]
            lines.append(
                f"| {name} | {state_label(s['iterate'])} | {_g(q['blocked']['variance'], 4)} ({q['blocked']['groups']}){o} | "
                f"{_g(q['interleaved']['variance'], 4)} ({q['interleaved']['groups']}){o} | "
                f"{_g(df['cholesky']['variance'], 4)} ({df['cholesky']['settings']}){o} | "
                f"{_g(df['eigh']['variance'], 4)} ({df['eigh']['settings']}) | "
                f"{_g(df['cholesky_shift']['variance'], 4)}{o} | {_g(df['eigh_shift']['variance'], 4)} | "
                f"{_g(en['pauli_shadow']['blocked']['variance'], 4)}{o} | "
                f"{_g(en['pauli_shadow']['interleaved']['variance'], 4)}{o} | "
                f"{_g(en['matchgate_shadow']['variance'], 4)} | {_g(s['sigma2'], 4)} |")
    lines += _orient_footnote(results)
    lines += ["", "### Energy: shots for +-1.6 mHa (S = V / 0.0016^2)", "",
              "| system | iterate | QWC blocked | QWC interleaved | DF Cholesky | DF eigh | DF eigh shifted | "
              "Pauli blocked | matchgate | sigma^2 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, res in results.items():
        o = dagger(name)
        marks = (o, o, o, "", "", o, "", "")
        for s in res["states"]:
            en = s["energy"]
            vals = [en["qwc"]["blocked"]["variance"], en["qwc"]["interleaved"]["variance"],
                    en["df"]["cholesky"]["variance"], en["df"]["eigh"]["variance"], en["df"]["eigh_shift"]["variance"],
                    en["pauli_shadow"]["blocked"]["variance"], en["matchgate_shadow"]["variance"], s["sigma2"]]
            lines.append(f"| {name} | {state_label(s['iterate'])} | "
                         + " | ".join(_g(energy_shots(v)) + mk for v, mk in zip(vals, marks)) + " |")
    lines += _orient_footnote(results)
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
            mk = dagger(name) if method != "exact" else ""
            lines.append(f"| {name} | {state_label(s['iterate'])} | {method}{mk} | " + " | ".join(cells) + " |")
    lines += _orient_footnote({n: None for n, _, _ in _res_states(results)})
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
             "Direct route: sum_e v_e and the per-element min / mean / max (matchgate: exact pair sums for H4 and, "
             "spin-orbital set, at the claims states; elsewhere Rao-Blackwellized over random matchings, with the "
             "standard error of the total, the number of settings and the largest per-element relative standard "
             "error; min and max of noisy per-element estimates are biased outward. Pauli: exact, '–' where not "
             "computed).", "",
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
            o = dagger(name)
            tr = _ratio_range(s, kind, "df-trotter", deltas, lambda e: 1.0)
            cell.append(tr + (o if tr != "–" else ""))
            fr = _ratio_range(s, kind, "df-fragments", deltas, lambda e: (e["n_leaves"] + 1) ** 2)
            cell.append(f"{fr} ({res['n_leaves']}){o}" if fr != "–" else "–")
            dm = _direct(s, kind, "matchgate_shadow")
            dp = _direct(s, kind, "pauli_shadow")
            if dm is None:
                cell += ["–", "–"]
            else:
                rb = dm["method"] == "rb"
                cell.append(f"{_g(dm['total'])}" + (f" +- {_g(dm['stderr'], 2)} (RB {dm['n_samples']})" if rb
                                                     else " (exact)"))
                pe = dm["per_element"]
                cell.append(f"{_g(pe['min'])}{o} / {_g(pe['mean'])} / {_g(pe['max'])}{o}"
                            + (f" (+-{_pct(dm.get('per_element_rel_stderr_max'))})" if rb else ""))
            cell.append(_g(dp and dp["total"]))
            lines.append(f"| {name} | {state_label(s['iterate'])} | {kind} | {r['K']} | " + " | ".join(cell) + " |")
    if any(dagger(n) for n, _, _ in _res_states(results)):
        lines += ["", ORIENT_FOOTNOTE + " The aux-route QWC and Pauli-shadow totals of N2 depend on it only below "
                  "the printed precision (at most 1.1e-4 relative between two orientations, observed)."]
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
                mk = dagger(name) if method != "exact" else ""
                cells.append("unreach." + mk if b is None else f"{_g(b['shots_bias'])} [{b['delta']:g}]{mk}")
            for scheme in ("matchgate_shadow", "pauli_shadow"):
                e = _direct(s, kind, scheme)
                cells.append(_g(e and e["shots_var"]))
            lines.append(f"| {name} | {state_label(s['iterate'])} | {kind} | " + " | ".join(cells) + " |")
    lines += _orient_footnote({n: None for n, _, _ in _res_states(results)})
    return lines + [""]


def _md_cell(text: str) -> str:
    """Text for a markdown table cell: norms ||x|| as the double bar U+2016, other bars escaped."""
    return str(text).replace("||", "\u2016").replace("|", "\\|")


def md_claims(rows: list[dict]) -> list[str]:
    so_err = None
    for r in rows:
        st = r["states"].get("so2")
        if r["geometry"] == "1.0" and st is not None:
            so_err = st["e_err"]
            break
    lines = ["## Comparison with the earlier, unverified H6 claims", "",
             "The claims' state, \"CQE iteration 2, about 6.6 mHa above FCI\", is SO-ACSE 2: iteration 2 of SD with "
             "the exact line search and Mazziotti's spin-orbital ACSE generator (module docstring, \"Claims\"; table "
             "\"Trajectories\" below)"
             + (f", {_mha(so_err, 2)} mHa above FCI for h6a" if so_err is not None else "")
             + ". Ours for H6/STO-3G at 1.0 A (h6a) at SO-ACSE 2, at M1 SD iterate 2 and at the M1 SD iterate "
             f"closest to {CLAIM_ERROR * 1e3:.1f} mHa above FCI (the comparison states of spec_B.md) and at FCI "
             "(iterate and E - E_FCI in mHa in each cell); the 1.5 A energy claims use h6b (H6 at 1.5 A) at the same "
             "kinds of states. Ratio = ours / claimed (SO-ACSE 2 / M1 2 / M1 closest / FCI). Reproduced (yes / no, "
             "same states): ratio within 30%, or for a claimed range our value within [0.7 lo, 1.3 hi]; row D-MG: "
             "min and max within 30% and the mean inside the claimed 12-16 (\"partly reproduced\" if only some "
             "hold). The primary verdict is the one at SO-ACSE 2. Alternatives: other variants of our computation, "
             "at the primary state.", "",
             "| claim | claimed | ours, SO-ACSE 2 (claims' state) | ours, M1 it. 2 | ours, M1 closest | ours, FCI | "
             "ratio | reproduced | verdict | alternatives (primary state) | likely reason |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    yn = {True: "yes", False: "no", None: "–"}
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
            if st["iterate"] == "fci":
                return f"{fmt(v)} (FCI)"
            where = "" if is_so(st["iterate"]) else f"it. {st['iterate']}, "
            return f"{fmt(v)} ({where}{_mha(st['e_err'], 2)})"

        ratio = "–"
        if r.get("ratio") is not None:
            ratio = " / ".join(_g(r["ratio"][k]) for k in VERDICT_STATES)
        at = r["verdict_state"]

        def afmt(label, x):
            if label.startswith("groups") or label.endswith("elements"):
                return _g(x)
            if "share" in label:
                return _pct(x)
            return fmt(x)

        alts = "; ".join(f"{k}: {afmt(k, v.get(at))}" for k, v in r["alternatives"].items())
        verdicts = " / ".join(yn[r["verdicts"][k]] for k in VERDICT_STATES)
        verdict = r["verdict"] + ("" if at == "so2" else f" (at {STATE_TITLES[at]})")
        lines.append(f"| {_md_cell(r['claim'])} | {claimed} | {cell('so2')} | {cell('it2')} | {cell('closest')} | "
                     f"{cell('fci')} | {ratio} | {verdicts} | {verdict} | {_md_cell(alts or '–')} | "
                     f"{_md_cell(r['reason'])} |")
    lines += ["", "Verdicts:", ""]
    prim = ("the claims' state (SO-ACSE 2; primary verdict)" if all(r["verdict_state"] == "so2" for r in rows)
            else "the primary state (SO-ACSE 2 where computed; otherwise as marked in the table)")
    groups = {f"reproduced at {prim}": [r for r in rows if r["verdict"] == "reproduced"],
              f"partly reproduced at {prim}": [r for r in rows if r["verdict"] == "partly reproduced"],
              f"not reproduced at {prim}": [r for r in rows if r["verdict"] == "not reproduced"],
              "not computed": [r for r in rows if r["verdict"] == "not computed"],
              "not reproduced at M1 SD iterate 2 (a different state, not the claims'; see the reasons)":
                  [r for r in rows if r["verdicts"]["it2"] is False]}
    for label, sel in groups.items():
        if sel:
            lines.append(f"* {label}: " + "; ".join(r["claim"] for r in sel) + ".")
    return lines + [""]


def _threads_label(results: dict) -> str:
    env = next((r.get("settings", {}).get("threads_env") for r in results.values()
                if r.get("settings", {}).get("threads_env")), None)
    if not env or env.get("threads") is None:
        return "threads not pinned"
    n = env["threads"]
    return f"{n} thread{'s' if n != 1 else ''}" + ("" if env.get("blas_pinned") else ", BLAS threads not pinned")


def md_timing(results: dict, total: float, cache_check) -> list[str]:
    lines = [f"## Runtime ({_threads_label(results)})", "",
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


def orientation_note(res: dict) -> list[str]:
    """Markdown caveat for a system with degenerate active orbitals (ORIENTATION_NOTES), with this run's
    orientation-dependent energy variances at FCI for n2."""
    name = res["system"]
    if name not in ORIENTATION_NOTES:
        return []
    text = ORIENTATION_NOTES[name]
    f = _state(res, "fci")
    if name == "n2" and f is not None:
        en = f["energy"]
        text += (f" This run, at FCI: QWC blocked / interleaved {_g(en['qwc']['blocked']['variance'], 4)} / "
                 f"{_g(en['qwc']['interleaved']['variance'], 4)}, Pauli shadows blocked / interleaved "
                 f"{_g(en['pauli_shadow']['blocked']['variance'], 4)} / "
                 f"{_g(en['pauli_shadow']['interleaved']['variance'], 4)}, Cholesky DF "
                 f"{_g(en['df']['cholesky']['variance'], 4)} Ha^2.")
    if dagger(name):
        text += " Marked † in the tables."
    return [text, ""]


def md_caveats(results: dict) -> list[str]:
    lines = ["## Caveats", ""]
    for res in results.values():
        lines += orientation_note(res)
    return lines + [TRAJECTORY_NOTE, ""]


def _tie_line(label: str, vals: dict | None, groups: dict | None, fmt=None) -> str | None:
    fmt = fmt or (lambda x: _g(x, 4))
    if not vals:
        return None
    txt = f"{label}: canonical {fmt(vals['canonical'])}"
    if groups:
        txt += f" ({groups['canonical']} groups)"
    txt += f", {vals['n']} random tie orders {fmt(vals['min'])}-{fmt(vals['max'])}"
    if groups:
        txt += f" ({groups['min']}-{groups['max']} groups)"
    return txt


def md_ties(s: dict) -> list[str]:
    """Per-system lines: QWC values with random orders inside the tie classes (module docstring)."""
    parts = []
    for order in ORDERS:
        t = s["energy"]["qwc"][order].get("tie_orders")
        if t:
            parts.append(_tie_line(f"energy QWC {order} (Ha^2)", tie_summary(t["variance"]), tie_summary(t["groups"])))
    t = s.get("residual", {}).get("spin_orbital", {}).get("qwc_tie_orders")
    if t:
        for order in ORDERS:
            parts.append(_tie_line(f"aux exact delta = {t['delta']:g}, spin-orbital, QWC {order}, uniform allocation, "
                                   f"delta^2 mean v_e", tie_summary(t[order]["q_mean_uniform"]),
                                   tie_summary(t[order]["groups"])))
            parts.append(_tie_line(f"the same, optimal allocation", tie_summary(t[order]["q_mean"]), None))
            parts.append(_tie_line(f"the same, shots for 30% (uniform allocation)",
                                   tie_summary(t[order]["shots_var_uniform"]), None, fmt=_g))
    parts = [p for p in parts if p]
    if not parts:
        return []
    return ["QWC tie orders (random orders inside the round-off tie classes of the sorted insertion): "
            + "; ".join(parts) + ".", ""]


def md_system(res: dict, deltas) -> str:
    name = res["system"]
    o = dagger(name)
    so_added = res.get("so_iterates_added") or []
    lines = [f"# Measurement cost: {name} ({res['label']})", "",
             f"N = {res['norb']}, nelec = {tuple(res['nelec'])}, E_FCI = {res['e_fci']:.10f} Ha, DF leaves L = "
             f"{res['n_leaves']} (Cholesky) / {res['n_leaves_eigh']} (eigh); H has {res['hamiltonian_terms']['blocked']} "
             f"Pauli terms, {res['energy_qwc_groups']['blocked']} / {res['energy_qwc_groups']['interleaved']} QWC groups "
             "(blocked / interleaved)."
             + (f" SD iterates {', '.join(map(str, res['bracket_iterates_added']))} were added because they bracket "
                "6.6 mHa above FCI." if res.get("bracket_iterates_added") else "")
             + (f" Added the SO-ACSE iterates {', '.join(state_label(k) for k in so_added)} (SD with the spin-orbital "
                "ACSE generator; SO-ACSE 2 is the claims' state)." if so_added else "")
             + (f" Claims states (SO-ACSE 2, M1 SD iterate 2 and the M1 SD iterate closest to 6.6 mHa above FCI): "
                f"{', '.join(state_label(k) for k in res['claims_states'])}." if res.get("claims_states") else ""), ""]
    lines += orientation_note(res)
    if res.get("trajectories"):
        lines += md_trajectories({name: res})[:-3] + [""]
    if "direct_targets" in res:
        dt = res["direct_targets"]
        lines += [f"Direct-route targets X_e: {dt['spin_orbital']['terms']} / {dt['spin_free']['terms']} Pauli terms "
                  f"(spin-orbital / spin-free), {_g(dt['spin_orbital']['pairs'])} / {_g(dt['spin_free']['pairs'])} "
                  "term pairs.", ""]
    for s in res["states"]:
        lines.append(f"## Iterate {state_label(s['iterate'])}: E - E_FCI = {_mha(s['e_err'])} mHa, "
                     f"norm A (full) = {_g(s['norm_A_full_spin_free'], 4)}, sigma^2 = {_g(s['sigma2'], 4)}")
        lines.append("")
        lines += md_ties(s)
        if "residual" not in s:
            continue
        if s.get("plan"):
            p = s["plan"]
            lines += [f"Plan: df-fragments orders {p['fragment_orders'] or 'none'}; matchgate RB settings "
                      f"{p['rb_settings']}; exact matchgate direct sums for {p.get('mg_exact_kinds') or 'none'}; Pauli "
                      f"direct kinds {p['pauli_kinds'] or 'none'} (small targets: exact sums for both schemes).", ""]
        for kind in KINDS:
            r = s["residual"][kind]
            lines += [f"### {kind} (K = {r['K']}, norm A full = {_g(r['norm_A_full'], 4)}, elements = "
                      f"{_g(r['norm_A_elements'], 4)})", "",
                      "| route | delta | rel. bias | scheme | order | settings | sum_e v_e | v_e min / mean / max | "
                      "delta^2 v_e mean | S (variance only) | S (bias-aware) |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
            for e in r["aux"]:
                pe = e["per_element"]
                ob = o if e["route_method"] != "exact" else ""
                lines.append(f"| aux {e['route_method']} | {e['delta']:g} | {_pct(e['rel_bias'])}{ob} | {e['scheme']} | "
                             f"{e['order']} | {_g(e['n_settings'])} | {_g(e['total'])} | {_g(pe['min'])}{o} / "
                             f"{_g(pe['mean'])} / {_g(pe['max'])}{o} | {_g(e['q_per_element']['mean'])} | "
                             f"{_shots(e, 'shots_var')} | {_shots(e, 'shots_bias')}{ob} |")
            for e in r["direct"]:
                pe = e["per_element"]
                se = "" if e.get("stderr") is None else f" +- {_g(e['stderr'], 2)}"
                meth = e["method"]
                if e["method"] == "rb":
                    meth += (f" ({e['n_samples']} settings, seed {e['seed']}; per-element rel. stderr <= "
                             f"{_pct(e.get('per_element_rel_stderr_max'))}")
                    vx = e.get("vs_exact")
                    if vx:
                        meth += (f"; against the exact sums: z(total) = {vx['z_total']:+.2f}, per-element z SD "
                                 f"{vx['per_element_z_sd']:.2f}, max |z| {vx['per_element_z_max']:.2f}")
                    meth += ")"
                lines.append(f"| direct | – | 0 | {e['scheme']} | {e['order']} | {_md_cell(meth)} | {_g(e['total'])}{se} | "
                             f"{_g(pe['min'])}{o} / {_g(pe['mean'])} / {_g(pe['max'])}{o} | – | "
                             f"{_shots(e, 'shots_var')} | {_shots(e, 'shots_var')} |")
            if o:
                lines += ["", ORIENT_FOOTNOTE]
            subs = [(e, _subset(e)) for e in r["direct"] if _subset(e) is not None]
            if subs:
                lines += [""] + [
                    f"Direct {e['scheme']} ({e['method']}) over the {sb['n']} elements with |A_e| > {SUBSET_REF:g} "
                    f"({_pct(sb['norm2_fraction'])} of ||A||^2): v_e min / mean / max = {_g(sb['per_element']['min'])} / "
                    f"{_g(sb['per_element']['mean'])} / {_g(sb['per_element']['max'])}; S_ext = "
                    f"{_g(sb['shots_extrapolated'])} (thresholds {', '.join(f'{x:g}' for x in SUBSET_THRESHOLDS)}: "
                    "means " + " / ".join(_g(x['per_element']['mean']) for x in e["subsets"]) + ")."
                    for e, sb in subs]
            lines.append("")
    return "\n".join(lines) + "\n"


def markdown_summary(results: dict, rows: list[dict], args, total: float, cache_check) -> str:
    added = "; ".join(f"{name} {', '.join(map(str, res['bracket_iterates_added']))}" for name, res in results.items()
                      if res.get("bracket_iterates_added"))
    so_added = "; ".join(f"{name} {', '.join(state_label(k) for k in res['so_iterates_added'])}"
                         for name, res in results.items() if res.get("so_iterates_added"))
    lines = ["# Measurement cost of the CQE energy and residual (M2, workflow B)", "",
             f"Exact per-shot variances along exact CQE trajectories (`benchmarks/run_measurement.py`). Systems "
             f"{', '.join(results)}; M1 SD states {', '.join(state_label(s) for s in args.states)}"
             + (f" (plus the iterates bracketing 6.6 mHa above FCI: {added})" if added else "")
             + (f"; SO-ACSE states (SD with the spin-orbital ACSE generator): {so_added}" if so_added else "")
             + "; deltas "
             f"{', '.join(f'{d:g}' for d in args.deltas)}; residual systems {', '.join(args.residual_systems)}. "
             "Conventions and reductions: the script's module docstring. Per-system detail: measurement_<system>.md.", ""]
    lines += md_claims(rows)
    lines += md_trajectories(results)
    lines += md_caveats(results)
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
    ap.add_argument("--direct-mg-exact", default="claims", choices=("claims", "none"),
                    help="exact direct-route matchgate sums (spin-orbital set) at the claims states, besides RB")
    ap.add_argument("--no-bracket", action="store_true",
                    help="do not add the SD iterates bracketing 6.6 mHa above FCI for h6a and h6b")
    ap.add_argument("--so-iterates", default=str(CLAIM_SO_ITERATE),
                    help="iterates of SD with the spin-orbital ACSE generator added for h6a and h6b (comma-separated, "
                         f">= 1; 'none' to skip; default {CLAIM_SO_ITERATE}, the claims' state)")
    ap.add_argument("--tie-orders", type=int, default=8,
                    help="random QWC tie orders for h6a and h6b (energy at every state, aux route at the claims "
                         "states); 0 skips them")
    ap.add_argument("--summary-only", action="store_true",
                    help="rebuild the markdown files and the summary from the JSON files in --out (no computation)")
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
    args.so_iterates = [] if args.so_iterates.strip().lower() in ("", "none") else \
        [int(k) for k in _names(args.so_iterates)]
    if args.summary_only:
        return summary_only(args)
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
    write_summary(results, args, time.perf_counter() - t0, cache_check, settings)
    return 0


def write_summary(results: dict, args, total: float, cache_check, settings: dict) -> None:
    """The claims rows, measurement_summary.md and measurement_summary.json."""
    rows = claims(results)
    with open(os.path.join(args.out, "measurement_summary.md"), "w") as fh:
        fh.write(markdown_summary(results, rows, args, total, cache_check))
    summary = {"settings": settings, "claims": rows, "seconds": total, "cache_check": cache_check,
               "timing": {k: v["timing"] | {"total": v["seconds"]} for k, v in results.items()}}
    with open(os.path.join(args.out, "measurement_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=_json_default)
    print(f"wrote {os.path.join(args.out, 'measurement_summary.md')} ({total:.1f} s)")


def summary_only(args) -> int:
    """Rebuild the markdown files and the summary from the measurement_<system>.json files in --out.

    The states, deltas and residual systems are those recorded in the files (the first system's
    settings). The cache check and, if the files are those of that run, the wall time are taken from an
    existing measurement_summary.json; otherwise the runtime total is the sum of the systems' runtimes.
    """
    results = {}
    for name in args.systems:
        path = os.path.join(args.out, f"measurement_{name}.json")
        if not os.path.exists(path):
            print(f"[{name}] skipped: {path} not found", flush=True)
            continue
        with open(path) as fh:
            results[name] = json.load(fh)
    if not results:
        raise SystemExit(f"no measurement_<system>.json files for {args.systems} in {args.out}")
    recorded = [r["settings"] for r in results.values()]
    if any(s["deltas"] != recorded[0]["deltas"] for s in recorded):
        raise SystemExit("the systems were computed with different deltas")
    args.states = recorded[0]["states"]
    args.deltas = [float(d) for d in recorded[0]["deltas"]]
    args.residual_systems = [n for n, r in results.items() if any("residual" in s for s in r["states"])]
    for name, res in results.items():
        with open(os.path.join(args.out, f"measurement_{name}.md"), "w") as fh:
            fh.write(md_system(res, args.deltas))
    cache_check = None
    total = float(sum(r["seconds"] for r in results.values()))
    old = os.path.join(args.out, "measurement_summary.json")
    if os.path.exists(old):
        with open(old) as fh:
            prev = json.load(fh)
        cache_check = prev.get("cache_check")
        timing = prev.get("timing", {})
        if set(timing) == set(results) and all(timing[n].get("total") == r["seconds"] for n, r in results.items()):
            total = float(prev["seconds"])                  # the same run: keep its wall time
    settings = dict(recorded[0], summary_only=True)
    write_summary(results, args, total, cache_check, settings)
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

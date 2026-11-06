"""Qubitized QPE on OpenFermion's DF, THC and SF block encodings, and GQSP time evolution (F8, F9).

Tags and formula numbers are those of :mod:`thc_engine_cqe.ft.primitives`. Energies are in
Hartree, Toffoli counts are integers, logical qubits are OpenFermion's counts.

Block encodings [OF; cited: Babbush et al., PRX 8, 041015 (2018); Berry et al., Quantum 3,
208 (2019); von Burg et al., PRR 3, 033055 (2021); Lee et al., PRX Quantum 2, 030305 (2021)]
-------------------------------------------------------------------------------------------
A :class:`BlockEncoding` is a factorized model H' of the active-space Hamiltonian H together
with OpenFermion's cost of one step of the qubitized walk W = R SELECT, which block-encodes
(H' - c)/lambda:

* ``"df"`` (:func:`df_encoding`): ``factor.double_factorize(eri, first="eigh", tol=0,
  eig_threshold=threshold, threshold_rule="openfermion")``, which reproduces OpenFermion's
  ``df.factorize`` (tests/test_factor_eri.py); L leaves, Xi eigenvectors.
* ``"thc"`` (:func:`thc_encoding`): ``factor.fit_thc_active(active, M, penalty=...)`` (grid
  guess, L-BFGS-B refinement; ``penalty="auto"`` is OpenFermion's lambda-penalty rule).
* ``"sf"`` (:func:`sf_encoding`): the first ``n_vectors`` supermatrix eigenvectors, the
  truncation of OpenFermion's ``sf.factorize`` (``utils.eigendecomp``, decreasing eigenvalue).

lambda follows the repository conventions (docs/conventions.md): T' is built from the
exact ERIs; DF lambda = sum |eig(T')| + 1/4 sum_l (sum_k |mu_lk|)^2 and THC lambda is
OpenFermion's ``thc.compute_lambda`` (``factor``); SF lambda = sum_pq |T'_pq| +
1/4 sum_P (sum_pq |L_P[p,q]|)^2 (:func:`sf_lambda`, OpenFermion's ``sf.compute_lambda``,
tested against it). Every encoding carries its factorization error relative to H,
de_fact = E_0(H') - E_0(H) and the ground-state infidelity 1 - |<Psi_0(H')|Psi_0(H)>|^2
(``diagnostics.thc_floors.model_floors``), and the model integrals (``eri_model``).

The step cost C_W comes from OpenFermion's ``compute_cost`` (``df/compute_cost_df.py``,
``thc/compute_cost_thc.py``, ``sf/compute_cost_sf.py``) in two passes, as in OpenFermion's
tests: stps = 20000, then stps = the first pass's step cost (:func:`openfermion_cost`).
chi and beta are ``FTParameters.chi`` and ``.beta`` (10 and 16); OpenFermion's SF model has
no rotation-bits parameter (``BlockEncoding.beta`` is 0 for SF). C_W is the cost of one
*controlled* walk step: it includes the control of the phase-estimation iteration [OF].

QROM clamp (:func:`openfermion_qr_clamp`) [derived]
---------------------------------------------------
OpenFermion's ``utils.QR(L, M)`` minimizes the QROM cost L/2^k + M(2^k - 1) over integers k
around k* = log2(L/M)/2 and calls ``sys.exit`` when k* < 0, i.e. L < M, which happens for
every small active space here (e.g. the DF output width n beta/2 = 64 exceeds Xi + n/2 = 31
for H4). For L < M the cost is increasing in k >= 0 (its k = 1 value L/2 + M exceeds L), so
the integer optimum is k = 0 with cost ceil(L): plain unary iteration. The clamp replaces
``QR`` by this rule inside the df, thc and sf cost modules for the duration of a ``with``
block, records whether it was used, and restores the previous functions on exit; for
L >= M it calls OpenFermion's ``QR`` itself, so those costs are unchanged (tested).
``BlockEncoding.qr_clamped`` records the flag per encoding.

QPE (F9) [OF]
-------------
Phase estimation to error dE on the walk takes

    n_iter = ceil(pi lambda / (2 dE)),        C_QPE = C_W n_iter         (qpe_cost)

walk steps [cited: Babbush et al. 2018; Lee et al. 2021], a rule for the error scale, not a
confidence interval. The logical qubits are OpenFermion's ``ancilla_cost``: the 2N system
qubits, the ancillas of the block encoding and the phase-estimation control register of
2 ceil(log2(n_iter + 1)) - 1 qubits (DF, THC) or 2 ceil(log2 n_iter) - 1 qubits (SF).
``BlockEncoding.ancilla_walk`` is ``ancilla_cost`` without that register, the logical qubits
of one walk step (``WalkParams.logical_qubits`` of ``ft.residual``), and :func:`qpe_cost`
adds the register for any dE, so ``qpe_cost(enc, dE).logical_qubits`` equals OpenFermion's
``ancilla_cost`` at that dE and ``.toffolis`` its total cost (tested).

Selection rule (:func:`select_qpe`) [M3 design, memo 3]
-------------------------------------------------------
Among the scanned encodings with |de_fact| <= ``params.de_fact`` (0.5 mHa) and infidelity <=
``params.infid_tol`` (1e-4), take the one with the fewest QPE Toffolis at ``params.de_qpe``
(1 mHa), per kind or overall; ties go to fewer logical qubits, then to the scan order. The
total error is then <= 1.5 mHa, inside chemical accuracy. :func:`scan_encodings` scans the
DF thresholds 1e-1 ... 1e-6, the THC ranks of memo 3 with penalties 0 and "auto", and every
SF rank.

Time evolution (F8) [derived; GQSP: cited, Motlagh and Wiebe, PRX Quantum 5, 020368 (2024)]
------------------------------------------------------------------------------------------
:func:`evolution_cost` costs exp(-i t H') by the GQSP Laurent polynomial of the walk
(``primitives.evolution_toffolis``): tau = lambda t, R = R(tau, ``params.eps_evol``), 2R
controlled walk calls and 3(2R + 1)(beta - 2) Toffolis of phase rotations, on Q_walk + 1
logical qubits (the GQSP control qubit). ``ft.walk`` checks the 2R-call Laurent polynomial on
an explicit Pauli-LCU walk.

Observed (symmetry-adapted builds of ``ft.symmetry``, dE = 1 mHa, one thread)
----------------------------------------------------------------------------
:func:`openfermion_cost` reproduces OpenFermion's reference tuples (C_W, total, qubits):
DF Reiher (21753, 10073183463, 3725) and Li (35011, 64410331887, 6405); THC Reiher
(10912, 5250145120, 2142) and Li (16923, 31938980976, 2196); SF Reiher (14184,
94868988984, 3320) and Li (24396, 117714920508, 3628); none needs the clamp. On the M3
systems the clamp is active for every scanned DF encoding of h4, h6a, n2, lih, h6s and n2s
and of h8 down to threshold 3e-4 (L <= 16; not at 1e-4 to 1e-6, L = 19-21), for every SF
encoding with L <= 14 (none with L >= 15), and for THC only at M = 4 (h4). Selected per
kind (rank; lambda in Ha; C_W; total Toffolis; logical qubits):

    system  DF                                   THC                               SF
    h4      3e-3 (L 7, Xi 27); 4.935; 868;       M 10, pen. 0; 5.864; 739;          L 7; 6.580; 384;
            6.73e6; 188                          6.81e6; 180                        3.97e6; 156
    h6a     3e-3 (L 11, Xi 63); 9.726; 1309;     M 15, auto; 9.125; 1035;           L 11; 17.23; 672;
            2.00e7; 233                          1.48e7; 217                        1.82e7; 184
    n2      3e-3 (L 20, Xi 82); 9.423; 1420;     M 21, pen. 0; 22.58; 1141;         L 20; 9.716; 828;
            2.10e7; 235                          4.05e7; 226                        1.26e7; 185
    lih     3e-4 (L 18, Xi 68); 9.259; 1366;     M 21, pen. 0; 68.92; 1141;         L 16; 13.08; 768;
            1.99e7; 235                          1.24e8; 228                        1.58e7; 187
    h6s     1e-3 (L 11, Xi 61); 7.571; 1303;     -                                 L 11; 14.14; 672;
            1.55e7; 233                                                             1.49e7; 184
    n2s     3e-3 (L 17, Xi 74); 7.716; 1378;     -                                 L 17; 7.779; 777;
            1.67e7; 235                                                             9.49e6; 185
    h8      1e-3 (L 15, Xi 117); 16.66; 1743;    -                                 L 15; 36.70; 829;
            4.56e7; 271                                                             4.78e7; 185

(THC not scanned for h6s, n2s, h8 here.) The DF rows of h4, h6a, n2 and lih are Appendix
A's. In OpenFermion's cost models at N <= 8 the SF walk step is 0.44-0.58 of the DF step,
so SF is the cheapest admissible encoding for h4, n2 and lih (and beats DF for h6s and
n2s), THC for h6a, and DF beats SF for h8. The lih and n2 THC fits pass the selection rule
only at M = 21 and with a large lambda (non-convex fits; spec section 11, risk 6). A full
scan (DF, THC with both penalties, SF) takes 2.6-3.2 / 7.2-7.3 / 6.7-7.0 / 4.5 s (h4, h6a,
n2, lih; two runs, one thread on a shared machine), the THC fits 1.7-5.4 s of it.
"""

from __future__ import annotations

import contextlib
import math
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np

from ..diagnostics.thc_floors import ExactReference, auto_ranks, model_floors
from ..factor.df import DoubleFactorization, double_factorize, eigendecomposition_vectors
from ..factor.eri_factor import one_body_tprime
from ..factor.thc import fit_thc_active
from ..integrals import ActiveSpace
from .primitives import FTParameters, evolution_toffolis, jacobi_anger_degree

if TYPE_CHECKING:  # ft.residual (task B) owns WalkParams; imported lazily in walk_params()
    from .residual import WalkParams

__all__ = [
    "KINDS",
    "DF_THRESHOLDS",
    "THC_RANKS",
    "openfermion_qr_clamp",
    "openfermion_cost",
    "BlockEncoding",
    "QPECost",
    "df_encoding",
    "thc_encoding",
    "sf_encoding",
    "scan_encodings",
    "qpe_cost",
    "select_qpe",
    "evolution_cost",
    "sf_lambda",
]

KINDS = ("df", "thc", "sf")

#: DF eigenvalue thresholds of the scan (OpenFermion's rule) [M3 design, memo 3].
DF_THRESHOLDS: tuple[float, ...] = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6)

#: THC ranks of the scan per number of active orbitals [M3 design, memo 3]; other N use
#: ``diagnostics.thc_floors.auto_ranks``.
THC_RANKS: dict[int, tuple[int, ...]] = {4: (4, 6, 8, 10), 6: (6, 9, 12, 15, 18, 21),
                                         8: (8, 12, 16, 20, 24, 32)}

# First-pass stps of OpenFermion's two-pass cost evaluation (its tests).
_STPS_FIRST = 20000
# Supermatrix eigenvalues kept as SF vectors (OpenFermion's sf.factorize uses 1e-16); the scan
# stops at eigenvalues below this fraction of the largest, which are round-off.
_SF_EIG_TOL = 1e-16
_SF_RANK_RTOL = 1e-12


def _cost_modules() -> dict:
    """OpenFermion's df, thc and sf cost modules (imported without the jax-dependent package parts)."""
    try:
        from openfermion.resource_estimates.df import compute_cost_df
        from openfermion.resource_estimates.sf import compute_cost_sf
        from openfermion.resource_estimates.thc import compute_cost_thc
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("ft.qpe needs OpenFermion's resource_estimates "
                          "(pip install -e '.[ft]')") from exc
    return {"df": compute_cost_df, "thc": compute_cost_thc, "sf": compute_cost_sf}


@contextlib.contextmanager
def openfermion_qr_clamp() -> Iterator[dict]:
    """Patch ``QR`` in OpenFermion's df, thc and sf cost modules with the k >= 0 clamp (module docstring).

    Yields a dict whose ``"active"`` entry becomes True when a call had L < M (and was
    clamped to k = 0, cost ceil(L)). Calls with L >= M go to OpenFermion's ``utils.QR``. The
    previous ``QR`` of every module is restored on exit, also after an exception.
    """
    import openfermion.resource_estimates.utils as of_utils

    modules = _cost_modules()
    original = of_utils.QR
    state = {"active": False}

    def qr_clamped(L, M1):
        if L < M1:  # OpenFermion: k = log2(L / M1) / 2 < 0 -> sys.exit
            state["active"] = True
            return 0, int(math.ceil(L))
        return original(L, M1)

    saved = {kind: mod.QR for kind, mod in modules.items()}
    for mod in modules.values():
        mod.QR = qr_clamped
    try:
        yield state
    finally:
        for kind, mod in modules.items():
            mod.QR = saved[kind]


def _positive_int(value: int, name: str) -> int:
    """``value`` as an int; ValueError unless it is an integer >= 1 (bools are rejected)."""
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _positive_finite(value: float, name: str) -> float:
    """``value`` as a float; ValueError unless it is finite and > 0."""
    v = float(value)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return v


def openfermion_cost(kind: str, n_spin_orbitals: int, lam: float, de: float, *, n_leaves: int | None = None,
                     n_eigenvectors: int | None = None, nthc: int | None = None, chi: int = 10,
                     beta: int = 16) -> tuple[int, int, int, bool]:
    """OpenFermion's ``compute_cost`` in two passes under the QR clamp: (C_W, total, ancilla_cost, clamped).

    Args:
        kind: "df" (needs ``n_leaves`` L and ``n_eigenvectors`` Xi), "thc" (needs ``nthc`` M)
            or "sf" (needs ``n_leaves`` L).
        n_spin_orbitals: 2N, an even integer.
        lam: lambda of the encoding (Hartree), positive and finite.
        de: phase-estimation error (Hartree), positive and finite.
        chi, beta: coefficient and rotation bits (``beta`` is not an input of the SF model).

    The first pass uses stps = 20000 and the second the first pass's step cost, as in
    OpenFermion's tests; the step cost does not depend on ``de``, the total and the qubits do.
    Invalid arguments (unknown kind, a missing, non-integral or non-positive rank, an odd
    ``n_spin_orbitals``, ``lam`` or ``de`` not positive and finite) raise ValueError before
    OpenFermion is called. RuntimeWarnings of OpenFermion's numpy code are suppressed;
    ``SystemExit`` and failed assertions inside OpenFermion are raised as RuntimeError.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    n = _positive_int(n_spin_orbitals, "n_spin_orbitals")
    if n % 2:
        raise ValueError(f"n_spin_orbitals must be even (2N), got {n_spin_orbitals!r}")
    lam, de = _positive_finite(lam, "lam"), _positive_finite(de, "de")
    chi, beta = _positive_int(chi, "chi"), _positive_int(beta, "beta")
    mods = _cost_modules()
    if kind == "df":
        if n_leaves is None or n_eigenvectors is None:
            raise ValueError("the DF cost needs n_leaves and n_eigenvectors")
        L, xi = _positive_int(n_leaves, "n_leaves"), _positive_int(n_eigenvectors, "n_eigenvectors")
        def call(stps):
            return mods["df"].compute_cost(n, lam, de, L, xi, chi, beta, int(stps))
    elif kind == "thc":
        if nthc is None:
            raise ValueError("the THC cost needs nthc")
        m = _positive_int(nthc, "nthc")
        def call(stps):
            return mods["thc"].compute_cost(n, lam, de, chi, beta, m, int(stps))
    else:
        if n_leaves is None:
            raise ValueError("the SF cost needs n_leaves")
        L = _positive_int(n_leaves, "n_leaves")
        def call(stps):
            return mods["sf"].compute_cost(n, lam, de, L, chi, int(stps))
    try:
        with openfermion_qr_clamp() as clamp, warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            first = call(_STPS_FIRST)
            step, total, qubits = call(int(first[0]))
    except (SystemExit, AssertionError) as exc:
        raise RuntimeError(f"OpenFermion's {kind} compute_cost failed: {type(exc).__name__}: {exc}") from exc
    return int(step), int(total), int(qubits), bool(clamp["active"])


def _control_register(kind: str, n_iter: int) -> int:
    """Qubits of OpenFermion's phase-estimation control register for n_iter walk steps (its formulas)."""
    if kind == "sf":
        return int(2 * np.ceil(np.log2(n_iter)) - 1)
    return int(2 * np.ceil(np.log2(n_iter + 1)) - 1)


def _iterations(lam: float, de: float) -> int:
    """ceil(pi lambda / (2 dE)), evaluated as in OpenFermion."""
    return int(np.ceil(np.pi * lam / (2 * de)))


@dataclass(frozen=True)
class BlockEncoding:
    """A factorized Hamiltonian H' and OpenFermion's cost of one controlled walk step.

    Attributes:
        kind: "df", "thc" or "sf".
        params: the factorization: df {"threshold", "L", "Lxi"}; thc {"M", "penalty",
            "penalty_value", "refine_success"}; sf {"L"}.
        lam, lam_one_body, lam_two_body: lambda and its one- and two-body parts (Hartree).
        de_fact: E_0(H') - E_0(H) (Hartree).
        infidelity: 1 - |<Psi_0(H')|Psi_0(H)>|^2.
        toffoli_per_step: C_W, Toffolis of one controlled walk step [OF].
        ancilla_qpe: OpenFermion's ancilla_cost at ``de_qpe`` (logical qubits of the QPE).
        ancilla_walk: ``ancilla_qpe`` without the phase-estimation control register.
        qr_clamped: OpenFermion's QR needed the k >= 0 clamp.
        chi, beta: bits passed to OpenFermion (beta = 0 for SF, whose model has none).
        de_qpe: the QPE error ``ancilla_qpe`` refers to (Hartree).
        eri_model: (N, N, N, N) integrals of H' (``active.with_eri(eri_model)``).

    Encodings are hashable: ``params`` and ``eri_model`` do not enter the hash, and
    ``eri_model`` does not enter equality either.
    """

    kind: str
    params: dict = field(hash=False)
    lam: float
    lam_one_body: float
    lam_two_body: float
    de_fact: float
    infidelity: float
    toffoli_per_step: int
    ancilla_qpe: int
    ancilla_walk: int
    qr_clamped: bool
    chi: int
    beta: int
    de_qpe: float = 1e-3
    eri_model: np.ndarray | None = field(default=None, repr=False, compare=False)

    def walk_params(self, fact_bias_m: float = 0.0) -> "WalkParams":
        """The walk for the residual routes of ``ft.residual``: lambda, C_W, Q_walk, kind and the bias.

        ``fact_bias_m`` is ||A[H'] - A[H]||_m at the current state
        (``ft.residual_inputs.factorization_bias``).
        """
        from .residual import WalkParams

        return WalkParams(lam=self.lam, toffoli_per_step=self.toffoli_per_step,
                          logical_qubits=self.ancilla_walk, kind=self.kind, fact_bias_m=fact_bias_m)

    def model(self, active: ActiveSpace) -> ActiveSpace:
        """The model Hamiltonian H' as an ActiveSpace (``active`` with ``eri_model``)."""
        if self.eri_model is None:
            raise ValueError("this encoding carries no model integrals")
        return active.with_eri(self.eri_model, label=f"{active.label} [{self.kind}]")


@dataclass(frozen=True)
class QPECost:
    """One phase estimation to error ``de_qpe`` on ``encoding`` (F9).

    Attributes:
        encoding: the block encoding.
        de_qpe: QPE error (Hartree).
        iterations: walk steps ceil(pi lambda / (2 de_qpe)).
        toffolis: C_QPE = C_W iterations.
        logical_qubits: OpenFermion's ancilla_cost at ``de_qpe``.
    """

    encoding: BlockEncoding
    de_qpe: float
    iterations: int
    toffolis: int
    logical_qubits: int


def _check_reference(active: ActiveSpace, reference: ExactReference) -> None:
    """ValueError unless ``reference`` was built for the integrals of ``active`` (N, electrons, ecore, h1, eri)."""
    ra = reference.active
    if ra is active:
        return
    if (ra.norb != active.norb or tuple(ra.nelec) != tuple(active.nelec) or ra.ecore != active.ecore
            or not np.array_equal(ra.h1, active.h1) or not np.array_equal(ra.eri, active.eri)):
        raise ValueError("reference must be the ExactReference of the same active space")


def _encoding(kind: str, active: ActiveSpace, reference: ExactReference, params: FTParameters,
              eri_model: np.ndarray, lam: tuple[float, float, float], fparams: dict, *,
              n_leaves: int | None = None, n_eigenvectors: int | None = None,
              nthc: int | None = None) -> BlockEncoding:
    floors = model_floors(reference, eri_model)
    step, _, qubits, clamped = openfermion_cost(kind, 2 * active.norb, lam[0], params.de_qpe,
                                                n_leaves=n_leaves, n_eigenvectors=n_eigenvectors,
                                                nthc=nthc, chi=params.chi, beta=params.beta)
    walk = qubits - _control_register(kind, _iterations(lam[0], params.de_qpe))
    return BlockEncoding(kind=kind, params=fparams, lam=float(lam[0]), lam_one_body=float(lam[1]),
                         lam_two_body=float(lam[2]), de_fact=float(floors["de_fci"]),
                         infidelity=float(floors["infidelity"]), toffoli_per_step=step,
                         ancilla_qpe=qubits, ancilla_walk=walk, qr_clamped=clamped, chi=params.chi,
                         beta=0 if kind == "sf" else params.beta, de_qpe=params.de_qpe,
                         eri_model=eri_model)


def _df_factorization(active: ActiveSpace, threshold: float) -> DoubleFactorization:
    """OpenFermion's DF (eigh first factorization, eigenvalue threshold rule) of the active-space ERIs."""
    if not math.isfinite(threshold):
        raise ValueError(f"the DF threshold must be finite, got {threshold!r}")
    return double_factorize(active.eri, first="eigh", tol=0.0, eig_threshold=float(threshold),
                            threshold_rule="openfermion")


def _df_encoding(active: ActiveSpace, df: DoubleFactorization, threshold: float, reference: ExactReference,
                 params: FTParameters) -> BlockEncoding:
    lam = df.lam(active.h1, eri_exact=active.eri)
    fparams = {"threshold": float(threshold), "L": int(df.n_leaves), "Lxi": int(df.n_eigenvectors)}
    return _encoding("df", active, reference, params, df.to_eri(),
                     (lam.lam_total, lam.lam_one_body, lam.lam_two_body), fparams,
                     n_leaves=df.n_leaves, n_eigenvectors=df.n_eigenvectors)


def df_encoding(active: ActiveSpace, threshold: float, *, reference: ExactReference,
                params: FTParameters = FTParameters()) -> BlockEncoding:
    """DF block encoding at OpenFermion's eigenvalue ``threshold`` (module docstring).

    ``reference`` is the ``diagnostics.thc_floors.ExactReference`` of ``active``. Raises
    ValueError if no leaf survives the threshold.
    """
    _check_reference(active, reference)
    df = _df_factorization(active, threshold)
    if df.n_leaves == 0:
        raise ValueError(f"DF threshold {threshold:g} keeps no leaf")
    return _df_encoding(active, df, threshold, reference, params)


def thc_encoding(active: ActiveSpace, nthc: int, *, penalty: float | str, reference: ExactReference,
                 params: FTParameters = FTParameters(), maxiter: int = 5000) -> BlockEncoding:
    """THC block encoding of rank M = ``nthc``: ``fit_thc_active(active, M, penalty=..., maxiter=...)``.

    ``penalty`` is 0.0 (unregularized) or "auto" (OpenFermion's lambda-penalty rule, the
    value used is in ``params["penalty_value"]``). THC fits are non-convex: lambda depends
    on the penalty and the iteration limit (``diagnostics.thc_floors``). Raises ValueError
    unless ``nthc`` is a positive integer.
    """
    _check_reference(active, reference)
    m = _positive_int(nthc, "nthc")
    fac = fit_thc_active(active, m, penalty=penalty, maxiter=int(maxiter))
    lam = fac.lam(active.h1, eri_exact=active.eri)
    fparams = {"M": m, "penalty": penalty, "penalty_value": float(fac.info["penalty"]),
               "refine_success": bool(fac.info.get("refine", {}).get("success", False))}
    return _encoding("thc", active, reference, params, fac.to_eri(),
                     (lam.lam_total, lam.lam_one_body, lam.lam_two_body), fparams, nthc=m)


def _sf_vectors(eri: np.ndarray) -> np.ndarray:
    """OpenFermion's SF vectors (decreasing supermatrix eigenvalue), without round-off eigenvalues."""
    vecs = eigendecomposition_vectors(eri, tol=_SF_EIG_TOL)
    w = np.einsum("Ppq,Ppq->P", vecs, vecs)                 # the eigenvalues
    return vecs[: int(np.sum(w > _SF_RANK_RTOL * w[0]))] if w.size else vecs


def sf_encoding(active: ActiveSpace, n_vectors: int, *, reference: ExactReference,
                params: FTParameters = FTParameters()) -> BlockEncoding:
    """SF block encoding with the ``n_vectors`` leading supermatrix eigenvectors (OpenFermion's SF).

    Raises ValueError unless ``n_vectors`` is an integer from 1 to the rank of the ERI
    supermatrix.
    """
    _check_reference(active, reference)
    vecs = _sf_vectors(active.eri)
    L = _positive_int(n_vectors, "n_vectors")
    if not L <= vecs.shape[0]:
        raise ValueError(f"n_vectors must lie in 1..{vecs.shape[0]} (the supermatrix rank), got {n_vectors}")
    lv = vecs[:L]
    lam = sf_lambda(active.h1, active.eri, lv)
    eri_model = np.einsum("Ppq,Prs->pqrs", lv, lv)
    return _encoding("sf", active, reference, params, eri_model, lam, {"L": L}, n_leaves=L)


def scan_encodings(active: ActiveSpace, *, kinds: Sequence[str] = KINDS,
                   reference: ExactReference | None = None,
                   df_thresholds: Sequence[float] = DF_THRESHOLDS, thc_ranks: Sequence[int] | None = None,
                   thc_penalties: Sequence[float | str] = (0.0, "auto"),
                   sf_ranks: Sequence[int] | None = None,
                   params: FTParameters = FTParameters()) -> list[BlockEncoding]:
    """Block encodings of every kind in ``kinds`` over its scan (module docstring).

    Defaults: ``thc_ranks`` from :data:`THC_RANKS` (``auto_ranks(N)`` for other N), every
    penalty in ``thc_penalties``; ``sf_ranks`` 1 .. the supermatrix rank. DF thresholds that
    keep no leaf are skipped. ``reference`` defaults to ``ExactReference.build(active)``.
    """
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        raise ValueError(f"unknown kinds {bad}; known: {KINDS}")
    ref = ExactReference.build(active) if reference is None else reference
    out: list[BlockEncoding] = []
    if "df" in kinds:
        _check_reference(active, ref)
        for thr in df_thresholds:
            df = _df_factorization(active, thr)
            if df.n_leaves:
                out.append(_df_encoding(active, df, thr, ref, params))
    if "thc" in kinds:
        ranks = thc_ranks if thc_ranks is not None else THC_RANKS.get(active.norb, auto_ranks(active.norb))
        for m in ranks:
            for pen in thc_penalties:
                out.append(thc_encoding(active, m, penalty=pen, reference=ref, params=params))
    if "sf" in kinds:
        ranks = sf_ranks if sf_ranks is not None else range(1, _sf_vectors(active.eri).shape[0] + 1)
        for L in ranks:
            out.append(sf_encoding(active, L, reference=ref, params=params))
    return out


def qpe_cost(enc: BlockEncoding, de_qpe: float = 1e-3) -> QPECost:
    """F9: n_iter = ceil(pi lambda / (2 de_qpe)) walk steps, C_W n_iter Toffolis, OpenFermion's qubits.

    Raises ValueError unless ``de_qpe`` (Hartree) and ``enc.lam`` are positive and finite.
    """
    de = _positive_finite(de_qpe, "de_qpe")
    n_iter = _iterations(_positive_finite(enc.lam, "the encoding's lambda"), de)
    return QPECost(encoding=enc, de_qpe=de, iterations=n_iter,
                   toffolis=enc.toffoli_per_step * n_iter,
                   logical_qubits=enc.ancilla_walk + _control_register(enc.kind, n_iter))


def select_qpe(encodings: Sequence[BlockEncoding], *, kind: str | None = None,
               params: FTParameters = FTParameters()) -> QPECost:
    """The selection rule (module docstring): the cheapest admissible QPE at ``params.de_qpe``.

    ``kind`` restricts the choice to one kind. Raises ValueError if no encoding is admissible.
    """
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be None or one of {KINDS}, got {kind!r}")
    ok = [e for e in encodings
          if (kind is None or e.kind == kind) and abs(e.de_fact) <= params.de_fact
          and e.infidelity <= params.infid_tol]
    if not ok:
        raise ValueError(f"no {kind or 'block'} encoding meets |de_fact| <= {params.de_fact:g} Ha and "
                         f"infidelity <= {params.infid_tol:g}")
    costs = [qpe_cost(e, params.de_qpe) for e in ok]
    return min(costs, key=lambda c: (c.toffolis, c.logical_qubits))


def evolution_cost(enc: BlockEncoding, t: float, *, params: FTParameters = FTParameters()) -> dict:
    """F8: cost of exp(-i t H') by GQSP on the walk of ``enc`` (module docstring).

    Returns {"t", "tau", "degree" (R), "walk_calls" (2R), "toffolis", "logical_qubits",
    "eps_evol"}; t in 1/Hartree, tau = lambda |t|.
    """
    tau = enc.lam * abs(float(t))
    degree = jacobi_anger_degree(tau, params.eps_evol)
    tof = evolution_toffolis(enc.lam, float(t), enc.toffoli_per_step, delta=params.eps_evol, beta=params.beta)
    return {"t": float(t), "tau": float(tau), "degree": int(degree), "walk_calls": 2 * int(degree),
            "toffolis": int(tof), "logical_qubits": int(enc.ancilla_walk + 1), "eps_evol": params.eps_evol}


def sf_lambda(h1: np.ndarray, eri: np.ndarray, vectors: np.ndarray) -> tuple[float, float, float]:
    """OpenFermion's SF lambda (``sf.compute_lambda``): (lambda, lambda_T, lambda_W) in Hartree.

    lambda_T = sum_pq |T'_pq| (elementwise, T' from the exact ``eri``) and
    lambda_W = 1/4 sum_P (sum_pq |L_P[p,q]|)^2 for SF vectors ``vectors`` of shape (n_vec, N, N)
    (the repository's layout; OpenFermion's is (N, N, n_vec)). |L| is the modulus, so complex
    factors are accepted, as in OpenFermion.
    """
    vecs = np.asarray(vectors)
    if vecs.ndim != 3 or vecs.shape[1:] != np.shape(h1):
        raise ValueError(f"vectors must have shape (n_vec, N, N) with N = {np.shape(h1)[0]}, "
                         f"got {vecs.shape}")
    if not np.all(np.isfinite(vecs)):
        raise ValueError("vectors must be finite")
    lam_t = float(np.sum(np.abs(one_body_tprime(h1, eri))))
    lam_w = 0.25 * float(np.sum(np.abs(vecs).sum(axis=(1, 2)) ** 2))
    return lam_t + lam_w, lam_t, lam_w

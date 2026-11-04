"""THC rank versus the quantities the CQE uses: energy errors, residual and variance floors.

A factorized two-body tensor turns the exact active-space Hamiltonian H into a model

    H' = H + V,    V = 1/2 sum_pqrs dV[p,q,r,s] e_{pq,rs},    dV = eri_model - eri,

with the same h1 and ecore (``active.with_eri(eri_model)``; conventions in
docs/conventions.md). :func:`thc_floor_scan` fits THC models of several ranks M
(and, for comparison, double factorizations at several truncation levels) and
records, for each model, the integral error, the block-encoding 1-norm lambda,
and how the model changes the energy, residual and variance that a CQE works with.

States and notation
-------------------
* Psi0, E0: lowest singlet of H (lowest state of the (n_alpha, n_beta) sector when
  n_alpha != n_beta).
* Psi0', E0': lowest state of H' with the same S^2 as Psi0 (see :func:`lowest_state`).
* E_X(Psi) = <Psi|X|Psi> and sigma^2[X](Psi) = <X^2> - <X>^2 for a Hamiltonian X.
* A[X](Psi): ACSE residual A[p,q,r,s] = <Psi|[e_{pq,rs}, X]|Psi> (pair order,
  :func:`~thc_engine_cqe.residual.compute_residual`); ||A|| is its Frobenius norm.

Row keys (energies in Hartree)
------------------------------
================== ======================================= =========================================
key                definition                              meaning
================== ======================================= =========================================
de_fci             E0' - E0                                model energy error, first order in dV
de_pt1             <Psi0|V|Psi0> = E_H'(Psi0) - E_H(Psi0)  first-order (Hellmann-Feynman) estimate
de_exact_at_model  E_H(Psi0') - E0                         exact energy of the model ground state
de_model_at_exact  E_H'(Psi0) - E0' = de_pt1 - de_fci      Psi0 above the model ground state, in H'
res_model_at_exact ||A[H'](Psi0)||                         residual floor: Psi0 seen through H'
var_model_at_exact sigma^2[H'](Psi0)                       variance floor: Psi0 seen through H'
res_exact_at_model ||A[H](Psi0')||                         exact residual of the model ground state
var_exact_at_model sigma^2[H](Psi0')                       exact variance of the model ground state
infidelity         1 - |<Psi0|Psi0'>|^2
res_noise          max(||A[H](Psi0)||, ||A[H'](Psi0')||)   solver noise (both should be exactly 0)
var_noise          max(sigma^2[H](Psi0), sigma^2[H'](Psi0'))
================== ======================================= =========================================

Further keys: ``kind`` ('thc' or 'df'), ``label``, ``M``, ``M_over_N``, ``n_params``,
``eri_max_abs_err`` = max |dV|, ``eri_fro_err`` = ||dV||_F, ``eri_fro_rel_err`` =
||dV||_F / ||eri||_F, ``lambda``/``lambda_one_body``/``lambda_two_body`` (the
factorization's ``lam(h1, eri_exact=eri)``, OpenFermion's convention with T' from the
exact ERIs, see :mod:`~thc_engine_cqe.factor.eri_factor`), ``e_fci`` = E0, ``e_fci_model``
= E0', ``s2_model``, ``spin_shift`` and ``spin_ok`` (see :func:`lowest_state`), ``res_hf``
= ||A[H](HF)|| (residual at the usual CQE starting point), ``details`` (fit or
factorization provenance) and ``seconds``. DF rows add ``n_leaves`` (L),
``n_eigenvectors`` (Xi) and ``eig_threshold``. Energies are Rayleigh quotients of the
stored vectors. For THC rows, ``details`` holds ``guess``, ``refine``, ``seed``, the
LS rank and error, the refinement outcome (``refine_success``, ``refine_nit``,
``refine_message``, ``refine_f``; None without refinement) and the lambda penalty
used by the fit (``penalty``, ``penalty_rule``). :func:`format_markdown` marks THC rows
whose refinement stopped without meeting L-BFGS-B's convergence test.

THC lambda and the fit penalty
------------------------------
By default the THC fits are unregularized (``penalty = 0`` in
:func:`~thc_engine_cqe.factor.thc.fit_thc`), because the floors are meant to measure
the best ERI fit at each rank. OpenFermion's THC pipeline (``thc_via_cp3``) instead
adds a lambda penalty by default. The THC lambda and FT costs of these rows are
therefore not the ones OpenFermion's pipeline would produce at the same rank. Two
effects make the unregularized lambda unreliable:

* lambda depends on the representation and not only on the model. At M = N(N+1)/2
  every generic point set reproduces the ERIs exactly, but the central tensor can
  have large elements of both signs that cancel in V_THC. For LiH (STO-3G) with the
  grid guess this gives lambda > 5 times the full-DF lambda (tested).
* The fit objective has nearly flat valleys along which lambda changes a lot. For
  example, LiH at M = 12 gave lambda = 40, 103 and 141 for maxiter = 5000, 20000 and
  50000, while 1/2 ||dV||_F^2 fell by only 4% (one thread, scipy 1.18; an
  observation, not a test). Lambda of a refinement that stops at ``maxiter``
  therefore changes with the iteration limit and the floating-point environment.
  So do its floors, to a lesser degree: fits of equal Frobenius error can have
  quite different energy errors (benchmarks/run_thc_floors.py gives examples).

``fit_kwargs={'penalty': 'auto'}`` applies OpenFermion's automatic penalty rule. It
lowers lambda (tested for LiH at M = 12), at the price of larger ERI errors and
floors. For LiH and H6 at M = 9 to 18 it gave lambda = 5.9 to 10.9, against a
full-DF lambda of 9.3 to 10.0, with 1/2 ||dV||_F^2 as large as 0.8 at M = 9
(observation, not a test).

Exact relations (tested in tests/test_thc_floors.py)
----------------------------------------------------
1. Variational sandwich (Psi0 and Psi0' minimize E_H and E_H' in the same spin sector):

       <V>_{Psi0'} <= E0' - E0 <= <V>_{Psi0},

   so de_fci <= de_pt1, de_exact_at_model >= 0 and de_model_at_exact >= 0.
2. H Psi0 = E0 Psi0 gives (H' - <H'>_{Psi0}) Psi0 = (V - <V>_{Psi0}) Psi0, hence

       sigma^2[H'](Psi0) = Var_{Psi0}(V),        A[H'](Psi0) = A[V](Psi0),

   where A[V] is the residual of the operator V alone (h1 = 0, ecore = 0). In the same
   way sigma^2[H](Psi0') = Var_{Psi0'}(V) and A[H](Psi0') = -A[V](Psi0'). The floors at
   Psi0 are therefore exactly linear (residual) and quadratic (variance) in dV. The
   floors at Psi0' have these orders to leading order (relation 3).
3. Orders for a scaled error V = t W, t -> 0, with Psi0 non-degenerate: de_fci, de_pt1
   and both residual floors are O(t); de_exact_at_model, de_model_at_exact, both
   variance floors and the infidelity are O(t^2). With w_k = <k|W|Psi0> over the
   excited eigenstates k of H, to leading order

       de_exact_at_model = de_model_at_exact = t^2 sum_k |w_k|^2 / (E_k - E0)   (= -E^(2)),
       infidelity        = t^2 sum_k |w_k|^2 / (E_k - E0)^2,
       var_model_at_exact = t^2 sum_k |w_k|^2   (exact for all t),

   where E^(2) is the second-order Rayleigh-Schroedinger energy. The tests scale the
   error tensor of an actual THC fit and compare with these sums over the exact
   spectrum of H4.

Relations 1 and 2 hold for exact eigenvectors. The FCI vectors carry a small residual
of their own, recorded as ``res_noise`` and ``var_noise``; floors at that level are
numerically zero.

What the floors mean for a CQE (interpretation, not a numerical claim)
---------------------------------------------------------------------
* A CQE whose residual is evaluated with H' (for example, through a THC block
  encoding) has Psi0' as its ground-state fixed point, because A[H'](Psi0') = 0. At
  convergence its exact energy error is de_exact_at_model, which is second order in
  dV. The energy it reports, E_H'(Psi0'), is off by de_fci, which is first order. Its
  exact residual stalls at res_exact_at_model.
* At the exact solution, the residual and variance evaluated with H' equal
  res_model_at_exact and var_model_at_exact. Convergence thresholds below these
  values test the factorization rather than the state.
* The second-order regime requires the perturbation to be small compared with the
  gap. A large ``infidelity`` means that the model ground state is a different
  state, and the first-order/second-order distinction no longer applies.

Fault-tolerant costing (optional)
---------------------------------
If ``openfermion.resource_estimates.thc.compute_cost`` imports, each THC row also gets
the cost model of Lee et al., PRX Quantum 2, 030305 (2021), as implemented in
OpenFermion (see :func:`thc_ft_cost`): ``ft_toffoli_per_step`` (Toffolis per
qubitization walk step, i.e. one block-encoding query plus the reflection), ``ft_toffoli_total`` (phase
estimation to precision dE: step cost times ceil(pi lambda / (2 dE))) and
``ft_logical_qubits``. The settings are stored with the row as ``ft_dE``, ``ft_chi``,
``ft_beta`` and ``ft_stps``. The cost model takes lambda as input. Here lambda is the
row's ``lambda``, the THC lambda of OpenFermion's ``thc.compute_lambda`` evaluated on
this row's fit. With the default unregularized fits the caveats of the previous
section apply. If OpenFermion is missing, these keys are left out.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from ..factor.df import double_factorize
from ..factor.eri_factor import eri_errors
from ..factor.thc import fit_thc
from ..fci_ops import FCISpace, Hamiltonian, inner, normalize
from ..integrals import ActiveSpace
from ..residual import ResidualData, compute_residual

__all__ = [
    "DEFAULT_DF_THRESHOLDS",
    "FT_DEFAULTS",
    "auto_ranks",
    "spin_square",
    "lowest_state",
    "LowestState",
    "ExactReference",
    "model_floors",
    "thc_ft_cost",
    "thc_floor_scan",
    "format_markdown",
    "NOT_CONVERGED_MARK",
]

#: Per-leaf eigenvalue thresholds of the DF comparison rows (OpenFermion's rule).
DEFAULT_DF_THRESHOLDS: tuple[float, ...] = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 1e-4, 1e-5)

#: Default fault-tolerant cost settings (sources in :func:`thc_ft_cost`).
FT_DEFAULTS: dict = {"dE": 1.6e-3, "chi": 10, "beta": 16}

_FCI_CONV_TOL = 1e-14
_SPIN_TOL = 1e-4
#: fit_thc arguments that thc_floor_scan sets itself (not allowed in fit_kwargs).
_SCAN_FIT_ARGS = ("guess", "refine", "seed", "mol", "mo_coeff")


# ----------------------------------------------------------------------- ranks
def auto_ranks(norb: int, max_rank: int | None = None) -> list[int]:
    """The default THC rank sweep: M = ceil(r N) for r in 1, 1.5, 2, 3, 4, 5, ...

    Ranks below ``max_rank`` (default N(N+1)/2, where LS-THC becomes exact) are
    kept, and ``max_rank`` itself is appended. For example, N = 4 gives
    [4, 6, 8, 10] and N = 6 gives [6, 9, 12, 18, 21].
    """
    n = int(norb)
    if n < 1:
        raise ValueError("norb must be positive")
    mmax = n * (n + 1) // 2 if max_rank is None else int(max_rank)
    out: list[int] = []
    ratios = [1.0, 1.5] + [float(k) for k in range(2, mmax + 2)]
    for r in ratios:
        m = int(math.ceil(r * n - 1e-9))
        if m >= mmax:
            break
        if m not in out:
            out.append(m)
    out.append(mmax)
    return out


# ------------------------------------------------------------------ spin and states
def spin_square(ci: np.ndarray, norb: int, nelec) -> float:
    """<S^2> of a real PySCF CI vector (``pyscf.fci.spin_op.spin_square0``)."""
    from pyscf.fci import spin_op

    ci = np.asarray(ci)
    if np.iscomplexobj(ci):
        raise ValueError("spin_square expects a real CI vector")
    return float(spin_op.spin_square0(ci, norb, tuple(nelec))[0])


def _nearest_s2(s2: float) -> float:
    """s (s + 1) for the half-integer s closest to the given <S^2>."""
    s = round(-1.0 + math.sqrt(1.0 + 4.0 * max(s2, 0.0))) / 2.0
    return s * (s + 1.0)


@dataclass
class LowestState:
    """Result of :func:`lowest_state`."""

    ci: np.ndarray            # normalized real CI vector
    energy: float             # Rayleigh quotient E_X(ci), including ecore
    s2: float                 # <S^2> of ci
    spin_shift: float | None  # None: ActiveSpace.fci was used as is; else the penalty used
    spin_ok: bool             # |s2 - target| <= spin_tol (True when there is no target)


def lowest_state(active: ActiveSpace, *, s2_target: float | None = None,
                 conv_tol: float = _FCI_CONV_TOL, shifts: Sequence[float] = (1.0, 5.0),
                 spin_tol: float = _SPIN_TOL) -> LowestState:
    """Lowest FCI state of ``active`` with <S^2> = ``s2_target``, with a spin check.

    The first attempt is ``active.fci(conv_tol=conv_tol)``. When n_alpha == n_beta,
    this targets a singlet through PySCF's ``fix_spin_`` and itself checks <S^2>,
    retrying with larger penalties (PySCF's default, 0.1 S^2, cannot lift a
    state more than 0.1 S(S+1) Hartree below the lowest singlet; a test
    reproduces this with a toy model). If that solve fails or <S^2> still
    misses the target by more than ``spin_tol``, the solve is repeated here
    with PySCF's ``fix_spin_`` and each penalty in ``shifts``. If none
    succeeds, a ``RuntimeWarning`` is issued and the state with <S^2> closest
    to the target is returned.

    ``s2_target=None`` means a singlet when n_alpha == n_beta (the convention of
    ``ActiveSpace.fci``) and no target otherwise.
    """
    na, nb = active.nelec
    if s2_target is None and na == nb:
        s2_target = 0.0
    ham = Hamiltonian(active)
    try:
        _, ci = active.fci(conv_tol=conv_tol)
    except RuntimeError:  # no root with the target spin for the core's shifts
        _, ci = active.fci(conv_tol=conv_tol, spin_square=None)
    ci = normalize(np.asarray(ci).reshape(ham.space.shape))
    s2 = spin_square(ci, active.norb, active.nelec)
    if s2_target is None or abs(s2 - s2_target) <= spin_tol:
        return LowestState(ci, ham.energy(ci), s2, None, True)

    from pyscf import fci

    best = (abs(s2 - s2_target), ci, s2, None)
    for shift in shifts:
        solver = fci.direct_spin1.FCI()
        solver.conv_tol = conv_tol
        solver.max_cycle = 500
        fci.addons.fix_spin_(solver, shift=float(shift), ss=float(s2_target))
        _, c = solver.kernel(active.h1, active.eri, active.norb, active.nelec, ecore=active.ecore)
        c = normalize(np.asarray(c).reshape(ham.space.shape))
        s = spin_square(c, active.norb, active.nelec)
        if abs(s - s2_target) < best[0]:
            best = (abs(s - s2_target), c, s, float(shift))
        if abs(s - s2_target) <= spin_tol:
            return LowestState(c, ham.energy(c), s, float(shift), True)
    _, ci, s2, shift = best
    warnings.warn(f"lowest_state: no state with <S^2> = {s2_target:g} found for {active.label!r} "
                  f"(best <S^2> = {s2:.6f}); floors compare different spin sectors",
                  RuntimeWarning, stacklevel=2)
    return LowestState(ci, ham.energy(ci), s2, shift, False)


@dataclass
class ExactReference:
    """Exact ground state Psi0 of an active space, plus the context every row reports.

    Attributes:
        active: the exact active space.
        ham: its :class:`~thc_engine_cqe.fci_ops.Hamiltonian`.
        state: :class:`LowestState` Psi0 (lowest singlet when n_alpha == n_beta).
        residual: ``compute_residual(active, Psi0)``. Its ``residual_norm`` and
            ``variance`` are the solver noise of Psi0 (exactly 0 for an exact eigenvector).
        e_hf, res_hf: energy and ||A[H]|| of the Hartree-Fock determinant (CQE start).
    """

    active: ActiveSpace
    ham: Hamiltonian
    state: LowestState
    residual: ResidualData
    e_hf: float
    res_hf: float

    @property
    def ci(self) -> np.ndarray:
        return self.state.ci

    @property
    def energy(self) -> float:
        """E_H(Psi0), Rayleigh quotient including ecore."""
        return self.residual.energy

    @classmethod
    def build(cls, active: ActiveSpace, *, fci_conv_tol: float = _FCI_CONV_TOL) -> "ExactReference":
        ham = Hamiltonian(active)
        state = lowest_state(active, conv_tol=fci_conv_tol)
        rd = compute_residual(active, state.ci, ham)
        rd_hf = compute_residual(active, FCISpace.of(active).hf(), ham)
        return cls(active=active, ham=ham, state=state, residual=rd, e_hf=rd_hf.energy,
                   res_hf=rd_hf.residual_norm)


# --------------------------------------------------------------------- floors
def model_floors(ref: ExactReference, eri_model: np.ndarray, *,
                 fci_conv_tol: float = _FCI_CONV_TOL, return_state: bool = False):
    """Energy errors, residual and variance floors of H' = ref.active.with_eri(eri_model).

    Returns a dict with the keys documented in the module docstring (de_fci,
    de_pt1, de_exact_at_model, de_model_at_exact, res_model_at_exact,
    var_model_at_exact, res_exact_at_model, var_exact_at_model, infidelity,
    res_noise, var_noise, e_fci, e_fci_model, s2_model, spin_shift, spin_ok,
    eri_max_abs_err, eri_fro_err, eri_fro_rel_err). With ``return_state=True`` it
    returns ``(row, psi0_model)``, where ``psi0_model`` is the normalized CI vector of Psi0'.

    Raises:
        ValueError: if ``eri_model`` lacks the 8-fold symmetry of real orbitals beyond
            1e-8 * max(1, max |eri_model|). PySCF's spin1 FCI kernels assume this
            symmetry and would silently use a different operator.
    """
    active = ref.active
    eri_model = np.asarray(eri_model, dtype=float)
    model = active.with_eri(eri_model, label=f"{active.label} [model]")
    model.check_symmetry(atol=1e-8 * max(1.0, float(np.max(np.abs(eri_model)))))
    ham_m = Hamiltonian(model)
    rd_m0 = compute_residual(model, ref.ci, ham_m)          # Psi0 seen through H'
    st = lowest_state(model, s2_target=_nearest_s2(ref.state.s2), conv_tol=fci_conv_tol)
    rd_0m = compute_residual(active, st.ci, ref.ham)         # Psi0' seen through H
    rd_mm = compute_residual(model, st.ci, ham_m)            # solver noise of Psi0'
    e0 = ref.energy
    ov = inner(ref.ci, st.ci)
    err = eri_errors(eri_model, active.eri)
    row = {
        "e_fci": float(e0),
        "e_fci_model": float(rd_mm.energy),
        "de_fci": float(rd_mm.energy - e0),
        "de_pt1": float(rd_m0.energy - e0),
        "de_exact_at_model": float(rd_0m.energy - e0),
        "de_model_at_exact": float(rd_m0.energy - rd_mm.energy),
        "res_model_at_exact": float(rd_m0.residual_norm),
        "var_model_at_exact": float(rd_m0.variance),
        "res_exact_at_model": float(rd_0m.residual_norm),
        "var_exact_at_model": float(rd_0m.variance),
        "infidelity": float(max(0.0, 1.0 - abs(ov) ** 2)),
        "res_noise": float(max(ref.residual.residual_norm, rd_mm.residual_norm)),
        "var_noise": float(max(ref.residual.variance, rd_mm.variance)),
        "s2_model": float(st.s2),
        "spin_shift": st.spin_shift,
        "spin_ok": bool(st.spin_ok),
        "eri_max_abs_err": float(err["max_abs"]),
        "eri_fro_err": float(err["fro"]),
        "eri_fro_rel_err": float(err["rel_fro"]),
    }
    return (row, st.ci) if return_state else row


# ------------------------------------------------------------- FT cost model
def thc_ft_cost(norb: int, lam: float, nthc: int, *, dE: float = FT_DEFAULTS["dE"],
                chi: int = FT_DEFAULTS["chi"], beta: int = FT_DEFAULTS["beta"],
                stps0: int = 20000) -> dict | None:
    """OpenFermion's THC fault-tolerant cost (Lee et al., PRX Quantum 2, 030305 (2021)).

    Calls ``openfermion.resource_estimates.thc.compute_cost(n, lam, dE, chi, beta, M,
    stps)`` with n = 2 N spin orbitals and M = ``nthc``. ``stps`` only sets the
    precision of the rotations that prepare the equal superposition. As in
    OpenFermion's tests (thc/compute_cost_thc_test.py), the call is made twice:
    first with stps = 20000, then with stps equal to the step cost of that
    first pass. A test reproduces both reference values of that file.

    Defaults (keyword arguments):
        dE = 1.6e-3 Ha, the phase-estimation error: ``dE_for_qpe`` default of
            OpenFermion's resource_estimates/pbc/thc/compute_thc_resources.py.
            The molecular tests use 1e-3.
        chi = 10, bits per coefficient: OpenFermion's THC tests and its costing table.
        beta = 16, bits per rotation angle: the OpenFermion test that reproduces
            Reiher FeMoco. The Li et al. test and the costing table use 20.

    Returns ``None`` if OpenFermion's cost function cannot be imported. Otherwise
    it returns a dict with ``ft_toffoli_per_step`` (Toffolis per walk step),
    ``ft_toffoli_total`` (step cost times ceil(pi lam / (2 dE)), phase estimation
    to precision dE), ``ft_logical_qubits``, the settings ``ft_dE``, ``ft_chi``,
    ``ft_beta`` and ``ft_stps``, and ``ft_note``.

    The cost model needs the QROM data size d = M (M + 1) / 2 + N to be at least the
    output width m = 2 ceil(log2(M + 1)) + 2 + chi. Otherwise OpenFermion's ``QR()``
    calls ``sys.exit``. This function checks the condition first, returns ``None``
    counts with an explanation in ``ft_note``, and never lets ``SystemExit`` escape.
    """
    try:
        from openfermion.resource_estimates.thc import compute_cost
    except Exception:  # OpenFermion not installed (or broken): skip silently
        return None
    n = 2 * int(norb)
    m_thc = int(nthc)
    out = {"ft_toffoli_per_step": None, "ft_toffoli_total": None, "ft_logical_qubits": None,
           "ft_dE": float(dE), "ft_chi": int(chi), "ft_beta": int(beta), "ft_stps": None,
           "ft_note": None}
    n_m = int(np.ceil(np.log2(m_thc + 1)))
    d = m_thc * (m_thc + 1) // 2 + n // 2
    width = 2 * n_m + 2 + int(chi)
    if d < width:
        out["ft_note"] = (f"not applicable: QROM data size d = {d} < output width m = {width} "
                          f"(OpenFermion's QR() requires d >= m)")
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            first = compute_cost(n, float(lam), float(dE), int(chi), int(beta), m_thc, int(stps0))
            step, total, qubits = compute_cost(n, float(lam), float(dE), int(chi), int(beta),
                                               m_thc, int(first[0]))
    except (Exception, SystemExit) as exc:  # OpenFermion uses sys.exit for invalid input
        out["ft_note"] = f"compute_cost failed: {type(exc).__name__}: {exc}"
        return out
    out.update({"ft_toffoli_per_step": int(step), "ft_toffoli_total": int(total),
                "ft_logical_qubits": int(qubits), "ft_stps": int(first[0])})
    return out


# ----------------------------------------------------------------------- scan
def _resolve_guess(active: ActiveSpace, guess: str) -> str:
    if guess in ("grid", "auto"):
        return "grid" if (active.mol is not None and active.mo_coeff is not None) else "random"
    if guess == "random":
        return "random"
    raise ValueError("guess must be 'grid', 'auto' or 'random'")


def _context(ref: ExactReference) -> dict:
    a = ref.active
    return {"system": a.label, "norb": int(a.norb), "nelec": [int(x) for x in a.nelec],
            "res_hf": float(ref.res_hf)}


def thc_floor_scan(active: ActiveSpace, ranks: Iterable[int], *, guess: str = "grid",
                   refine: bool = True, seed: int = 0, include_df: bool = True,
                   df_thresholds: Sequence[float] = DEFAULT_DF_THRESHOLDS,
                   ft_cost: bool = True, ft_params: dict | None = None,
                   fit_kwargs: dict | None = None, fci_conv_tol: float = _FCI_CONV_TOL,
                   reference: ExactReference | None = None,
                   progress: Callable[[dict], None] | None = None) -> list[dict]:
    """THC (and DF) models of ``active`` versus the CQE floors.

    Args:
        active: exact active space. Its ERIs must be 8-fold symmetric.
        ranks: THC ranks M (see :func:`auto_ranks` for the default sweep).
        guess: ``'grid'`` (or ``'auto'``) uses the ISDF grid guess when
            ``active.mol`` and ``active.mo_coeff`` exist and falls back to
            ``'random'`` otherwise. ``'random'`` always uses the seeded random
            guess. The guess actually used is stored in ``row['details']['guess']``.
        refine: L-BFGS-B refinement after the least-squares fit
            (:func:`~thc_engine_cqe.factor.thc.fit_thc`).
        seed: seed of the random guess.
        include_df: append rows for ``double_factorize(eri, first='eigh', tol=0.0,
            eig_threshold=t, threshold_rule='openfermion')``, which is OpenFermion's
            DF, for each t in ``df_thresholds``. A threshold that gives the same
            factorization (same L and Xi) as a larger one is skipped. The kept
            eigenpairs are nested, so equal (L, Xi) means equal factors. A threshold
            that keeps no leaf at all is skipped too.
        ft_cost: add OpenFermion's THC cost model to the THC rows when available
            (:func:`thc_ft_cost`). ``ft_params`` overrides its keyword arguments.
        fit_kwargs: extra keyword arguments for ``fit_thc``, e.g. ``maxiter`` or
            ``penalty`` (0 by default; ``'auto'`` is OpenFermion's rule, see the
            module docstring). The arguments that this function sets itself
            (``guess``, ``refine``, ``seed``, ``mol``, ``mo_coeff``) raise ValueError.
        fci_conv_tol: PySCF ``conv_tol`` for all FCI solves. The Davidson residual
            tolerance is its square root.
        reference: precomputed :class:`ExactReference` (built if ``None``).
        progress: optional callback, called with each finished row.

    Returns:
        A list of row dicts: THC rows in the order of ``ranks``, then DF rows by
        decreasing threshold. The keys are documented in the module docstring. All
        values are JSON-serializable (str, int, float, bool, None, lists, dicts).
    """
    used_guess = _resolve_guess(active, guess)
    fit_kw = dict(fit_kwargs or {})
    reserved = sorted(set(fit_kw) & set(_SCAN_FIT_ARGS))
    if reserved:
        raise ValueError(f"fit_kwargs must not contain {reserved}: thc_floor_scan sets "
                         f"{list(_SCAN_FIT_ARGS)} itself (use its guess/refine/seed arguments)")
    if reference is not None and reference.active is not active:
        raise ValueError("reference was built for a different ActiveSpace object")
    ref = ExactReference.build(active, fci_conv_tol=fci_conv_tol) if reference is None else reference
    n = active.norb
    ft_kw = dict(FT_DEFAULTS)
    ft_kw.update(ft_params or {})
    rows: list[dict] = []

    for m in ranks:
        m = int(m)
        t0 = time.perf_counter()
        fac = fit_thc(active.eri, m, guess=used_guess, mol=active.mol, mo_coeff=active.mo_coeff,
                      refine=refine, seed=seed, **fit_kw)
        lam = fac.lam(active.h1, active.eri)
        row = {"kind": "thc", "label": f"THC M={m}", **_context(ref),
               "M": m, "M_over_N": m / n, "n_leaves": None, "n_eigenvectors": None,
               "eig_threshold": None, "n_params": int(fac.n_params()),
               "lambda": float(lam.lam_total), "lambda_one_body": float(lam.lam_one_body),
               "lambda_two_body": float(lam.lam_two_body)}
        row.update(model_floors(ref, fac.to_eri(), fci_conv_tol=fci_conv_tol))
        info = fac.info
        rf = info.get("refine", {})
        row["details"] = {"guess": info.get("guess"), "refine": bool(refine), "seed": int(seed),
                          "ls_rank": int(info["ls"]["rank"]),
                          "ls_eri_error_fro": float(info["ls"]["eri_error_fro"]),
                          "refine_success": rf.get("success"), "refine_nit": rf.get("nit"),
                          "refine_message": rf.get("message"), "refine_f": rf.get("f_final"),
                          "penalty": float(info["penalty"]),
                          "penalty_rule": str(info["penalty_rule"])}
        if ft_cost:
            cost = thc_ft_cost(n, lam.lam_total, m, **ft_kw)
            if cost is not None:
                row.update(cost)
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        if progress is not None:
            progress(row)

    if include_df:
        seen: set[tuple[int, int]] = set()
        for thr in sorted((float(t) for t in df_thresholds), reverse=True):
            t0 = time.perf_counter()
            df = double_factorize(active.eri, first="eigh", tol=0.0, eig_threshold=thr,
                                  threshold_rule="openfermion")
            key = (df.n_leaves, df.n_eigenvectors)
            if key in seen or df.n_leaves == 0:
                continue
            seen.add(key)
            lam = df.lam(active.h1, active.eri)
            row = {"kind": "df", "label": f"DF {thr:g} (L={df.n_leaves}, Xi={df.n_eigenvectors})",
                   **_context(ref), "M": None, "M_over_N": None, "n_leaves": int(df.n_leaves),
                   "n_eigenvectors": int(df.n_eigenvectors), "eig_threshold": thr,
                   "n_params": int(df.n_params()), "lambda": float(lam.lam_total),
                   "lambda_one_body": float(lam.lam_one_body),
                   "lambda_two_body": float(lam.lam_two_body)}
            row.update(model_floors(ref, df.to_eri(), fci_conv_tol=fci_conv_tol))
            row["details"] = {"first": "eigh", "first_tol": 0.0, "threshold_rule": "openfermion"}
            row["seconds"] = time.perf_counter() - t0
            rows.append(row)
            if progress is not None:
                progress(row)
    return rows


# ------------------------------------------------------------------- markdown
#: (header, key, format) of the default table columns.
DEFAULT_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("model", "label", "s"),
    ("M/N", "M_over_N", ".3g"),
    ("n_par", "n_params", "d"),
    ("max\\|ΔV\\|", "eri_max_abs_err", ".1e"),
    ("‖ΔV‖/‖V‖", "eri_fro_rel_err", ".1e"),
    ("λ", "lambda", ".2f"),
    ("ΔE_FCI", "de_fci", "+.2e"),
    ("⟨ΔH⟩₀", "de_pt1", "+.2e"),
    ("E_H(Ψ′)−E₀", "de_exact_at_model", ".1e"),
    ("1−F", "infidelity", ".1e"),
    ("‖A′(Ψ₀)‖", "res_model_at_exact", ".1e"),
    ("σ′²(Ψ₀)", "var_model_at_exact", ".1e"),
    ("‖A(Ψ′)‖", "res_exact_at_model", ".1e"),
    ("Tof/step", "ft_toffoli_per_step", "d"),
    ("qubits", "ft_logical_qubits", "d"),
)


def _fmt(value, spec: str) -> str:
    if value is None:
        return "–"
    if spec == "s":
        return str(value)
    if spec == "d":
        return str(int(value))
    return format(float(value), spec)


#: Marker appended to the label of THC rows whose refinement did not converge.
NOT_CONVERGED_MARK = "†"


def _refine_failed(row: dict) -> bool:
    """True for a THC row whose L-BFGS-B refinement ran and did not report convergence."""
    details = row.get("details") or {}
    return row.get("kind") == "thc" and details.get("refine_success") is False


def _penalty_note(rows: Sequence[dict]) -> str | None:
    """Legend text describing the lambda penalty of the THC fits (None if unknown)."""
    descr = []
    for r in rows:
        details = r.get("details") or {}
        if r.get("kind") != "thc" or "penalty" not in details:
            continue
        if not details.get("refine", True):
            d = ("least-squares only (no refinement, so no λ penalty). Their λ depends on "
                 "the point set and can be far above the DF λ")
        elif details.get("penalty_rule") == "auto":
            d = "refined with OpenFermion's automatic λ penalty (residual / λ_Z at the LS start)"
        elif details["penalty"] == 0:
            d = ("refined without a λ penalty, unlike OpenFermion's default pipeline. Their λ "
                 "and FT costs depend on the representation, can be far above the DF λ, and "
                 "change with maxiter if the refinement did not converge")
        else:
            d = f"refined with λ penalty {details['penalty']:g}"
        if d not in descr:
            descr.append(d)
    if not descr:
        return None
    return "λ uses OpenFermion's convention (T′ from the exact ERIs). THC fits: " + "; ".join(descr) + "."


def format_markdown(rows: Sequence[dict], columns: Sequence[tuple[str, str, str]] | None = None) -> str:
    """Compact markdown table of :func:`thc_floor_scan` rows, followed by a short legend.

    ``columns`` is a sequence of (header, key, format spec), where the spec is
    's', 'd' or a float spec such as '.2e'. Columns whose key no row has (for
    example the FT columns without OpenFermion) are dropped. Missing or ``None``
    values are shown as '–'. THC rows whose refinement did not meet L-BFGS-B's
    convergence test get :data:`NOT_CONVERGED_MARK` after their label, and the
    legend lists them with the iteration count and L-BFGS-B's message. The
    legend also states the lambda penalty of the THC fits.
    """
    if not rows:
        return "(no rows)"
    cols = list(DEFAULT_COLUMNS if columns is None else columns)
    cols = [c for c in cols if any(c[1] in r for r in rows)]
    lines = ["| " + " | ".join(h for h, _, _ in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for _, k, spec in cols:
            cell = _fmt(r.get(k), spec)
            if k == "label" and _refine_failed(r):
                cell = f"{cell} {NOT_CONVERGED_MARK}"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    notes = []
    for system in dict.fromkeys(r.get("system", "") for r in rows):
        sub = [r for r in rows if r.get("system", "") == system]
        parts = []
        if sub[0].get("e_fci") is not None:
            parts.append(f"E₀ = {sub[0]['e_fci']:.10f} Ha")
        if sub[0].get("res_hf") is not None:
            parts.append(f"‖A(HF)‖ = {sub[0]['res_hf']:.3g}")
        res_noise = [r["res_noise"] for r in sub if r.get("res_noise") is not None]
        var_noise = [r["var_noise"] for r in sub if r.get("var_noise") is not None]
        if res_noise and var_noise:
            parts.append(f"solver noise ‖A‖ ≤ {max(res_noise):.1e}, σ² ≤ {max(var_noise):.1e}")
        if parts:
            notes.append(f"{system or 'system'}: " + "; ".join(parts) + ".")
    notes.append("ΔV = model − exact ERIs, ΔH = H′ − H. Ψ₀ and Ψ′ are the lowest states of H "
                 "and H′ with the same S² (singlets for closed shells), and F = |⟨Ψ₀|Ψ′⟩|². "
                 "A′ and σ′² use H′; A uses H. ΔE_FCI, ⟨ΔH⟩₀ and the residuals are first "
                 "order in ΔV; E_H(Ψ′)−E₀, 1−F and σ′² are second order.")
    pen = _penalty_note(rows)
    if pen:
        notes.append(pen)
    ft = [r for r in rows if r.get("ft_toffoli_per_step") is not None and "ft_dE" in r]
    if ft:
        notes.append(f"FT cost: OpenFermion thc.compute_cost with dE = {ft[0]['ft_dE']:g} Ha, "
                     f"χ = {ft[0]['ft_chi']}, β = {ft[0]['ft_beta']}, using each row's λ. "
                     f"Tof/step is the Toffoli count per qubitization walk step (one "
                     f"block-encoding query plus the reflection); qubits are logical qubits.")
    failed = [r for r in rows if _refine_failed(r)]
    if failed:
        items = []
        for r in failed:
            d = r["details"]
            msg = str(d.get("refine_message") or "").strip().rstrip(":").strip()
            items.append(f"{r.get('label')} (nit = {d.get('refine_nit')}"
                         + (f", {msg}" if msg else "") + ")")
        notes.append(f"{NOT_CONVERGED_MARK} THC refinement stopped before L-BFGS-B's convergence "
                     f"test was met: " + "; ".join(items) + ". The floors and λ of these rows "
                     "depend on where the optimizer stopped, which can change with maxiter and "
                     "with round-off.")
    bad = [r["label"] for r in rows if not r.get("spin_ok", True)]
    if bad:
        notes.append("WARNING: spin sector mismatch for " + ", ".join(bad) + ".")
    return "\n".join(lines) + "\n\n" + "\n".join(f"- {n}" for n in notes)

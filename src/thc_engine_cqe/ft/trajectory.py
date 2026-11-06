"""Exact CQE trajectories with their states, overlaps and spectrum, and the classical first step (F16).

Tags and formula numbers are those of :mod:`thc_engine_cqe.ft.primitives`. Energies and
variances are in Hartree (variances Ha^2); step sizes eps multiply the unnormalized
direction d, as in ``cqe.run_cqe``.

Trajectories (:func:`exact_trajectory`)
---------------------------------------
``cqe.run_cqe`` from the HF determinant with ``keep_generators=True`` and a callback that
keeps every iterate Psi_1 .. Psi_K; Psi_0 is HF. A :class:`TrajectoryRecord` holds, for
k = 0..K, the energy E_k, the variance sigma_k^2 = <H^2> - <H>^2, ||A_k|| and the overlap
p_k = |<psi_0|Psi_k>|^2 with the exact ground state psi_0, and for k = 0..K-1 the step
eps_k and the direction d_k (``CQEStep.generator``), the input of ``ft.stateprep``.
Quantities of the last iterate come from its terminal ``CQEStep`` when the run stopped on
r_tol, else from ``residual.compute_residual``. ``optimizer``, ``optimizer_options`` and
``metric`` go to ``run_cqe``: the Appendix A regressions use spin-free L-BFGS (memory 10),
and ``metric="spin_orbital"`` gives the standard (Mazziotti) CQE generator as preconditioner.

With orbital irreps (``ft.symmetry.orbital_irreps``) the residual is projected onto the
symmetry-allowed entries before every decision of the driver,
``residual_estimator=lambda rd, ci, it: symmetry.mask_pair_tensor(rd.A, irreps)``
(F2a). The forbidden entries vanish exactly at symmetry-pure states, but exact L-BFGS
amplifies their round-off: without the projection the weight outside the HF irrep reaches
1.8e-7 (n2) and 1.9e-4 (n2s) in 20 steps (observed below; ``ft.symmetry`` reports the same
effect). With it the weight stays exactly 0, since the projected generator maps each irrep
to itself. ``TrajectoryRecord.irrep_impurity`` is max_k (1 - weight of the HF irrep in
Psi_k). The exact residual is still recorded (``residual_norms``).

First step (``first_step``). "exact" (default) is the trajectory above: step 0 takes the
exact line search eps* along d_0, as a measured loop would (F14, "measured"). "second_order"
is the input of the classical-first-step loop (F14, F16): step 0 takes the classical step
eps_2 of :func:`second_order_step` along the run's own d_0 (masked like the run), through
``run_cqe(..., first_eps=eps_2)``, and the later steps search as usual. The optimizer
records the forced step like any other, so CG and L-BFGS keep the step-0 pair or conjugacy
memory, and every row k >= 1 is the exact continuation after the classical step: row 1
(Psi_1, E_1, sigma_1^2, ||A_1||, p_1) and eps_0 = eps_2 are those of the classical step, and
C(Psi_k) of ``ft.stateprep`` compiles the steps that produced the recorded states. A restart
from Psi_1 would drop the step-0 memory and change later steps by more than eps_2 - eps*
does (below).

Spectrum (the E1 of F15) [derived]
----------------------------------
``active.fci(nroots=n_roots, spin_square=0)`` gives the lowest singlet roots; psi_0 is the
first. E1_any is the second root. E1 (same irrep) is the energy of the first root above
psi_0 that has the irrep of psi_0 (``ft.symmetry.ci_irrep``). Degenerate roots of
different irreps can come out of the solver as arbitrary mixtures (a Delta state of a
linear molecule has an Ag and a B1g component in D2h); a group of roots whose successive
energies differ by <= 1e-6 Ha is therefore tested as a subspace: it contains a state of
the irrep g iff the largest eigenvalue of C^T P_g C exceeds 1/2, with C the group's CI
vectors and P_g the projector on the determinants of irrep g (0 or 1 for an exactly
degenerate subspace of symmetry-pure states). E1 is then the lowest energy of that group,
a lower bound on the same-irrep root by at most the group's width. For roots of one irrep
each this is the per-root rule of the design specification (docs/m3-results.md section 11;
tests/test_ft_trajectory.py::test_e1_degenerate_roots_are_tested_as_a_subspace). If none of
the n_roots roots qualifies, E1 is
replaced by the highest root, a lower bound on it (flag "e1_lower_bound"); without irreps
E1 = E1_any (flag "e1_any_irrep"), also a lower bound. Lower bounds keep the certificate of
``ft.handoff`` valid.

Classical first step (F16) [derived]
------------------------------------
At the RHF determinant A_0 is known in closed form, A_ovov = 4(ia|jb) - 2(ib|ja) [M1,
docs/m1-results.md section 3], so the first direction d_0 = M conj(A_0) needs no
measurement. Along
Psi(eps) = exp(eps G) Psi, G = sum d e_{pq,rs} anti-Hermitian,

    E'(0) = <[H, G]> = -Re sum d A        (= -||A||^2 for the spin-free metric)
    E''(0) = <[[H, G], G]> = 2 <phi|H|phi> + 2 Re <H Psi|G phi>,       phi = G Psi,

and the second-order step is eps_2 = -E'(0)/E''(0) (:func:`second_order_step`): a
CID-type contraction, computed classically. It is compared with the exact line search of
``run_cqe`` (one step from the same state with the same metric and mask, i.e. the
trajectory's eps_0) and checked against a Richardson-extrapolated central second
difference of E(eps) (h = 2e-3 and 1e-3, propagation by ``fci_ops.expm_apply`` at tolerance
1e-15). The step is undefined for E''(0) <= 0 (flag "nonpositive_curvature") and
meaningless at a converged state, where d is solver noise: when the predicted lowering
dE_2 = E'(0)^2 / (2 E''(0)) is below 1e-10 Ha (flag "converged_state") eps_2, eps* and their
ratio are nan. [derived] A line search on energies with round-off |E| u (u = 1.1e-16)
locates a minimum of depth dE_2 only to a relative sqrt(|E| u / dE_2), 1.5e-3 (h4) to
1.1e-2 (n2) at 1e-10 Ha. Observed along 20-step h4 and n2 L-BFGS trajectories, run_cqe's
eps* and a bounded scalar minimizer of E(eps) agree to <= 1.1e-3 (relative) where
dE_2 >= 2e-10 Ha, to 0.2-2.3% between 1e-11 and 3e-13 Ha, and not at all at the h4 FCI state
(2.7e-16 Ha: eps_2/eps* 0.49 there before this rule).

Observed (symmetry-adapted builds of ``ft.symmetry``, one thread)
---------------------------------------------------------------
Spin-free L-BFGS (memory 10), 20 steps, projected residual; k* is the first iterate within
1.6 mHa of E_0. tests/test_ft_trajectory.py reproduces Appendix A (errors to 1e-6 Ha,
overlaps to 1e-6, k*) and checks E1 against PySCF's symmetry-adapted FCI to 1e-8 Ha.

    system  p_HF    k*   E_k* - E_0   p_k*     E1 - E0 (Ha)       eps_2/eps*  E(eps_2) - E(eps*)
                         (mHa)                 same / any irrep               (Ha)
    h4      0.9455   3   1.203        0.9981   0.5657 / 0.5633    1.0159      1.0e-5
    h6a     0.9026   6   1.111        0.9987   0.3909 / 0.3909    1.0220      3.0e-5
    n2      0.9257   3   1.147        0.9994   0.7115 / 0.4117    0.9805      3.4e-5
    lih     0.9743   2   1.011        0.9957   0.1332 / 0.1332    1.0192      6.3e-6
    h6s     0.5733  12   1.396        0.9944   0.1047 / 0.1047    1.1371      3.7e-3
    n2s     0.2528   -   (5.71 at k = 20, p_20 = 0.9879)   0.1917 / 0.1917    0.7955      1.2e-2
    h8      0.8864   9   1.363        0.9985   0.3445 / 0.3445    1.0216      3.5e-5

The second-order step is within 3% of the exact line search at equilibrium, not for the
stretched systems (h6s, n2s). E''(0) agrees with the Richardson difference to <= 2.1e-9
(relative) on all seven. The projection leaves the equilibrium trajectories unchanged:
over the first 7 steps the energies agree with unprojected runs [build check] to 1e-11
(h4), 7e-10 (h6a) and 3e-10 Ha (n2), the overlaps to 2e-8. On n2s, where the unprojected
run leaks 1.9e-4 of its weight out of the HF irrep, the projected run ends at 5.71 mHa
after 20 steps against 7.31 mHa. Cost per system (one thread): the 12 FCI roots 0.10 /
1.45 / 1.17 / 0.54 / 2.57 / 1.44 / 5.44 s (h4, h6a, n2, lih, h6s, n2s, h8; ``roots`` lets a
caller with several records of one system solve once), the 20 steps 0.07-0.27 s (h8 3.2 s,
with or without the projection; timings on a shared machine).

``first_step="second_order"`` (L-BFGS as above): E_1 - E_0 = 21.515 / 37.986 / 35.243 /
3.461 / 88.585 / 224.119 / 47.550 mHa and p_1 = 0.98039 / 0.95801 / 0.96357 / 0.99411 /
0.88310 / 0.24327 / 0.94581 (h4, h6a, n2, lih, h6s, n2s, h8), against 21.504 / 37.956 /
35.209 / 3.455 / 84.900 / 212.223 / 47.515 mHa and 0.98023 / 0.95763 / 0.96350 / 0.99401 /
0.87223 / 0.23144 / 0.94542 after the exact step. The continuation has the same k* as the
exact trajectory on all seven systems, and its rows k >= 2 differ from the exact ones by
<= 0.010 mHa and <= 1.4e-4 in p_k at equilibrium (h4, h6a, n2, lih, h8), by <= 0.45 mHa and
1.4e-3 on h6s, and by up to 30 mHa and 0.11 on n2s (with metric "spin_orbital":
<= 0.012 mHa and 3.5e-5, 0.41 mHa and 3.7e-3, 14.5 mHa and 0.034). A plain
L-BFGS restart from Psi_1, which loses the step-0 pair, lies 3.4-7.6 mHa above the
continuation at k = 2 at equilibrium (h4, h6a, n2, h8; 0.24 mHa on lih) and needs one more
step on n2 (k* = 4); on n2s it deviates by up to 23 mHa. For the memoryless "sd" the
continuation and the restart coincide (to the printed 1e-3 mHa).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from ..cqe import CQEResult, run_cqe
from ..fci_ops import Hamiltonian, expm_apply, inner, normalize, two_body
from ..integrals import ActiveSpace
from ..optimizers import apply_metric, direction_slope
from ..residual import ResidualData, compute_residual, descent_generator
from .symmetry import ci_irrep, determinant_irreps, mask_pair_tensor

__all__ = [
    "FIRST_STEPS",
    "TrajectoryRecord",
    "exact_trajectory",
    "first_iterate_below",
    "second_order_step",
]

FIRST_STEPS = ("exact", "second_order")

# Roots closer than this (Ha) are tested together as one degenerate subspace (module docstring).
_DEGENERACY_TOL = 1e-6
# Finite-difference steps of the second derivative (Richardson pair h, h/2) and the Taylor tolerance.
_FD_STEP = 2e-3
_EXPM_TOL = 1e-15
# Predicted lowering E'(0)^2 / (2 E''(0)) (Ha) below which the step is reported as converged.
_MIN_LOWERING = 1e-10


@dataclass
class TrajectoryRecord:
    """An exact CQE trajectory from HF with everything the cost layer needs (module docstring).

    Attributes:
        system, optimizer, metric: labels; ``metric`` is the ``run_cqe`` metric.
        norb, nelec: active orbitals and (n_alpha, n_beta).
        e0: exact ground-state energy (Hartree).
        e1: lowest excited singlet of the irrep of psi_0 (or a lower bound, see ``flags``).
        e1_any: lowest excited singlet of any irrep.
        energies, variances, residual_norms: E_k, <H^2> - <H>^2 and ||A_k|| (spin-free
            Frobenius norm, the exact residual) for k = 0..K.
        overlaps: p_k = |<psi_0|Psi_k>|^2, k = 0..K.
        eps: step sizes, k = 0..K-1.
        directions: d_k (pair-order (N, N, N, N), ``CQEStep.generator``), k = 0..K-1.
        states: normalized CI vectors Psi_0 .. Psi_K.
        irreps: orbital irrep ids, or None (no symmetry mask).
        ground_irrep: irrep id of psi_0 (None without irreps).
        irrep_impurity: max_k (1 - weight of the irrep of Psi_0 in Psi_k) (None without irreps).
        converged: ``CQEResult.converged`` of the run that produced the last row.
        flags: "e1_any_irrep", "e1_lower_bound", "ground_state_not_in_hf_irrep".
        first_step: "exact" (eps_0 from the exact line search) or "second_order" (eps_0 = eps_2
            of F16, the classical first step, and the exact continuation after it; module
            docstring).
    """

    system: str
    optimizer: str
    norb: int
    nelec: tuple[int, int]
    e0: float
    e1: float
    e1_any: float
    energies: np.ndarray
    variances: np.ndarray
    residual_norms: np.ndarray
    overlaps: np.ndarray
    eps: np.ndarray
    directions: list[np.ndarray]
    states: list[np.ndarray]
    irreps: np.ndarray | None
    metric: str = "spin_free"
    ground_irrep: int | None = None
    irrep_impurity: float | None = None
    converged: bool = False
    flags: tuple[str, ...] = ()
    first_step: str = "exact"

    @property
    def n_steps(self) -> int:
        """K, the number of steps taken."""
        return len(self.eps)

    @property
    def errors(self) -> np.ndarray:
        """E_k - E_0 for k = 0..K (Hartree)."""
        return self.energies - self.e0


def _same_irrep_e1(energies: np.ndarray, cis: list[np.ndarray], det_irreps: np.ndarray,
                   irrep: int) -> float | None:
    """Lowest energy of the first root group above root 0 that contains a state of ``irrep``; None if none."""
    proj = (det_irreps == irrep).ravel()
    i, n = 1, len(energies)
    while i < n:
        j = i + 1
        while j < n and energies[j] - energies[j - 1] <= _DEGENERACY_TOL:
            j += 1
        c = np.column_stack([np.asarray(cis[r]).ravel() for r in range(i, j)])
        c = c / np.linalg.norm(c, axis=0)
        gram = c[proj].conj().T @ c[proj]
        if float(np.linalg.eigvalsh(0.5 * (gram + gram.conj().T))[-1]) > 0.5:
            return float(energies[i])
        i = j
    return None


def _f16(active: ActiveSpace, psi: np.ndarray, ham: Hamiltonian, metric: str,
         irreps: np.ndarray | None) -> tuple[ResidualData, np.ndarray, float, float]:
    """F16 at ``psi``: (residual data, d = M conj(A) with A masked by ``irreps`` if given, E'(0), E''(0))."""
    space = ham.space
    rd = compute_residual(active, psi, ham)
    a_dir = rd.A if irreps is None else mask_pair_tensor(rd.A, irreps)
    d = apply_metric(descent_generator(a_dir), metric)
    slope = float(direction_slope(d, rd.A))
    phi = two_body(space, d, psi)
    d2e = 2.0 * float(np.real(inner(phi, ham.apply(phi))))
    d2e += 2.0 * float(np.real(inner(ham.apply(psi), two_body(space, d, phi))))
    return rd, d, slope, d2e


def _step_flags(slope: float, d2e: float) -> tuple[float, list[str]]:
    """Predicted lowering E'(0)^2 / (2 E''(0)) (nan for E''(0) <= 0) and the flags of the module docstring."""
    if not d2e > 0.0:
        return math.nan, ["nonpositive_curvature"]
    lowering = slope ** 2 / (2.0 * d2e)
    return lowering, (["converged_state"] if lowering < _MIN_LOWERING else [])


def _propagate(ham: Hamiltonian, d: np.ndarray, psi: np.ndarray, eps: float) -> np.ndarray:
    """Normalized exp(eps G) psi, G = sum d e_{pq,rs}."""
    return normalize(expm_apply(lambda v: two_body(ham.space, d, v), psi, eps, tol=_EXPM_TOL))


def _mask_estimator(irreps: np.ndarray | None) -> Callable | None:
    """The run_cqe residual estimator that projects A on the symmetry-allowed entries (None without irreps)."""
    if irreps is None:
        return None
    return lambda rd, ci, it: mask_pair_tensor(rd.A, irreps)


def _run(active: ActiveSpace, ham: Hamiltonian, max_iter: int,
         **run_kw) -> tuple[list[np.ndarray], list[tuple[float, float, float]], list[float],
                            list[np.ndarray], CQEResult]:
    """run_cqe from HF: states, rows (E, sigma^2, ||A||) of every state, eps, directions, result."""
    states = [normalize(ham.space.hf())]
    res = run_cqe(active, max_iter=int(max_iter), keep_generators=True,
                  callback=lambda step, ci: states.append(normalize(np.asarray(ci))), **run_kw)
    K = len(states) - 1
    taken = res.steps[:K]
    rows = [(s.energy, s.variance, s.residual_norm) for s in taken]
    if len(res.steps) > K:                      # terminal record of Psi_K (stopped on r_tol)
        last = res.steps[K]
        rows.append((last.energy, last.variance, last.residual_norm))
    else:
        rd = compute_residual(active, states[K], ham)
        rows.append((rd.energy, rd.variance, rd.residual_norm))
    return states, rows, [s.eps for s in taken], [np.asarray(s.generator) for s in taken], res


def exact_trajectory(active: ActiveSpace, *, optimizer: str = "lbfgs", max_iter: int = 20,
                     system: str = "", irreps: np.ndarray | None = None, n_roots: int = 12,
                     metric: str = "spin_free", optimizer_options: Mapping | None = None,
                     mask_residual: bool = True, first_step: str = "exact",
                     roots: tuple[Sequence[float], Sequence[np.ndarray]] | None = None) -> TrajectoryRecord:
    """Exact CQE trajectory from HF with states, overlaps, E1 and E1_any (module docstring).

    Args:
        active: the active space (a symmetry-adapted build for ``irreps``).
        optimizer, max_iter, metric, optimizer_options: passed to ``cqe.run_cqe``; at most
            ``max_iter`` steps in total.
        system: label stored on the record (default ``active.label``).
        irreps: orbital irrep ids; they select the same-irrep E1 and, with
            ``mask_residual``, project the residual onto the allowed entries (F2a).
        n_roots: singlet FCI roots computed for E1 (>= 2).
        mask_residual: apply the symmetry projection when ``irreps`` is given.
        first_step: "exact" or "second_order" (step 0 takes the classical step eps_2 of F16,
            ``run_cqe(..., first_eps=eps_2)``, and the later steps continue from it; module
            docstring). Raises ValueError if eps_2 is undefined at HF.
        roots: the singlet FCI roots ``active.fci(nroots=n_roots, spin_square=0.0)`` (energies,
            CI vectors), computed once by a caller that builds several records of one system
            (``ft.pipeline``); None computes them here. They must number ``n_roots``.
    """
    if int(n_roots) < 2:
        raise ValueError(f"n_roots must be >= 2 (E1 needs an excited root), got {n_roots}")
    if first_step not in FIRST_STEPS:
        raise ValueError(f"first_step must be one of {FIRST_STEPS}, got {first_step!r}")
    norb, nelec = active.norb, active.nelec
    ir = None if irreps is None else np.asarray(irreps, dtype=int)
    if ir is not None and ir.shape != (norb,):
        raise ValueError(f"irreps must have shape ({norb},), got {ir.shape}")
    es, cis = active.fci(nroots=int(n_roots), spin_square=0.0) if roots is None else roots
    es = np.asarray(es, dtype=float)
    if es.shape != (int(n_roots),) or len(cis) != int(n_roots):
        raise ValueError(f"roots must hold n_roots = {n_roots} energies and CI vectors, got {es.size} "
                         f"and {len(cis)}")
    psi0 = normalize(np.asarray(cis[0]))
    flags: list[str] = []
    ground_irrep = None
    if ir is None:
        e1 = float(es[1])
        flags.append("e1_any_irrep")
    else:
        ground_irrep = ci_irrep(psi0, norb, nelec, ir)[0]
        found = _same_irrep_e1(es, cis, determinant_irreps(norb, nelec, ir), ground_irrep)
        if found is None:
            warnings.warn(f"no excited singlet of irrep {ground_irrep} among {n_roots} roots; "
                          "E1 is replaced by the highest root (a lower bound)", RuntimeWarning, stacklevel=2)
            found = float(es[-1])
            flags.append("e1_lower_bound")
        e1 = found

    ham = Hamiltonian(active)
    mask_ir = ir if (ir is not None and mask_residual) else None
    run_kw = dict(optimizer=optimizer, optimizer_options=optimizer_options, metric=metric,
                  e_exact=float(es[0]), residual_estimator=_mask_estimator(mask_ir))
    if first_step == "second_order" and int(max_iter) >= 1:
        _, _, slope, d2e = _f16(active, normalize(ham.space.hf()), ham, metric, mask_ir)
        lowering, bad = _step_flags(slope, d2e)
        if bad:
            raise ValueError(f"the second-order first step is undefined at HF ({bad[0]}: E''(0) = {d2e:.3e} Ha, "
                             f"predicted lowering {lowering:.3e} Ha)")
        run_kw["first_eps"] = -slope / d2e
    states, rows, eps, directions, res = _run(active, ham, max_iter, **run_kw)
    energies, variances, rnorms = (np.array(col, dtype=float) for col in zip(*rows))
    overlaps = np.array([abs(complex(inner(psi0, s))) ** 2 for s in states])
    impurity = None
    if ir is not None:
        hf_irrep = ci_irrep(states[0], norb, nelec, ir)[0]
        if hf_irrep != ground_irrep:
            flags.append("ground_state_not_in_hf_irrep")
        det_ir = determinant_irreps(norb, nelec, ir)
        impurity = max(float(np.sum(np.abs(s[det_ir != hf_irrep]) ** 2)) for s in states)
    return TrajectoryRecord(
        system=system or active.label, optimizer=res.optimizer, norb=norb, nelec=tuple(nelec),
        e0=float(es[0]), e1=float(e1), e1_any=float(es[1]), energies=energies, variances=variances,
        residual_norms=rnorms, overlaps=overlaps, eps=np.array(eps, dtype=float), directions=directions,
        states=states, irreps=ir, metric=res.metric, ground_irrep=ground_irrep, irrep_impurity=impurity,
        converged=bool(res.converged), flags=tuple(flags), first_step=first_step)


def first_iterate_below(record: TrajectoryRecord, threshold: float) -> int | None:
    """The first k with E_k - E_0 <= ``threshold`` (Hartree), or None if the trajectory never gets there."""
    hit = np.flatnonzero(record.errors <= threshold)
    return int(hit[0]) if hit.size else None


def second_order_step(active: ActiveSpace, psi: np.ndarray | None = None, *, metric: str = "spin_free",
                      irreps: np.ndarray | None = None) -> dict:
    """F16: the second-order step eps_2 = -E'(0)/E''(0) along the first CQE direction (module docstring).

    Args:
        active: the active space.
        psi: the state (default the HF determinant, the classical first step).
        metric: the direction is d = M conj(A) with ``optimizers.apply_metric``.
        irreps: orbital irrep ids; A is projected on the symmetry-allowed entries before d
            is formed and in the exact line search, as in :func:`exact_trajectory`.

    Returns a dict with "eps2", "eps_exact" (exact line search of ``run_cqe``), "ratio"
    (eps2 / eps_exact), "d2e" (E''(0) by the commutator formula), "d2e_fd" (Richardson
    finite differences), "slope" (E'(0), with the exact A), "lowering" (the predicted
    E'(0)^2 / (2 E''(0)), Hartree), "energy0", "energy_eps2", "energy_exact", "energy_lost"
    (E(eps2) - E(eps_exact) >= 0), "direction" (d), "state" (normalized exp(eps2 G) Psi),
    "metric" and "flags". With the flag "nonpositive_curvature" (E''(0) <= 0) eps2 and ratio
    are nan; with "converged_state" (predicted lowering below 1e-10 Ha: d is solver noise)
    eps2, eps_exact, d2e_fd, ratio and the energies after the step are nan and "state" is
    None.
    """
    ham = Hamiltonian(active)
    space = ham.space
    psi = normalize(space.hf() if psi is None else np.asarray(psi).reshape(space.shape))
    ir = None if irreps is None else np.asarray(irreps, dtype=int)
    if ir is not None and ir.shape != (active.norb,):
        raise ValueError(f"irreps must have shape ({active.norb},), got {ir.shape}")
    rd, d, slope, d2e = _f16(active, psi, ham, metric, ir)
    lowering, flags = _step_flags(slope, d2e)
    converged = "converged_state" in flags
    eps2 = -slope / d2e if not flags else math.nan
    d2e_fd = eps_exact = e_exact = math.nan
    if not converged:
        def energy_el(eps: float) -> float:
            v = expm_apply(lambda x: two_body(space, d, x), psi, eps, tol=_EXPM_TOL)
            return float(np.real(inner(v, ham.apply(v))) / np.real(inner(v, v)))

        e_el0 = float(np.real(inner(psi, ham.apply(psi))))

        def second_difference(h: float) -> float:
            return (energy_el(h) - 2.0 * e_el0 + energy_el(-h)) / h ** 2

        d2e_fd = (4.0 * second_difference(0.5 * _FD_STEP) - second_difference(_FD_STEP)) / 3.0
        first = run_cqe(active, ci0=psi, max_iter=1, optimizer="sd", metric=metric,
                        residual_estimator=_mask_estimator(ir)).steps[0]
        if first.eps > 0.0:                     # a terminal record (no step) has eps = 0
            eps_exact, e_exact = float(first.eps), float(first.energy_after)
    state = _propagate(ham, d, psi, eps2) if np.isfinite(eps2) else None
    e_eps2 = ham.energy(state) if state is not None else math.nan
    return {"eps2": float(eps2), "eps_exact": eps_exact, "ratio": float(eps2 / eps_exact),
            "d2e": float(d2e), "d2e_fd": float(d2e_fd), "slope": slope, "lowering": float(lowering),
            "energy0": float(rd.energy), "energy_eps2": float(e_eps2), "energy_exact": e_exact,
            "energy_lost": float(e_eps2 - e_exact), "direction": d, "state": state, "metric": metric,
            "flags": tuple(flags)}

"""Rank, sparsity and compressibility of CQE generators along exact trajectories.

The question this module answers for NISQ hardware: after the first CQE step,
can the generator G = sum g[p,q,r,s] e_{pq,rs} (g = conj(A)) be compressed, and
what does each compression cost in CNOTs (:mod:`thc_engine_cqe.diagnostics.cost`)?
Two compressions are compared:

* **Leaves** (double factorization, :mod:`thc_engine_cqe.factor.generator`):
  G = sum_k c_k i Qhat_k^2 + Ohat_F. The leaf spectrum is basis independent.
  For an anti-Hermitian, pair-symmetric gs, the leaf magnitudes |c_k| are exactly
  the singular values of the N^2 x N^2 pair matrix M[(pq), (rs)] = gs[p,q,r,s]:
  Gm = U^T (i M) U with U unitary, and Gm is real symmetric (tested). A real
  orthogonal orbital rotation u maps M to (u x u)^T M (u x u), so the
  spectrum, the rank and every leaf count are invariant (tested).
* **Elements**: symmetry-unique spin-free elements. The orbit of [p,q,r,s] under
  pair exchange (r,s,p,q) and adjoint (q,p,s,r) has at most 4 members, and an
  anti-Hermitian, pair-symmetric g is fixed on each orbit by one value. Keeping
  a set of whole orbits and zeroing the rest is an orthogonal projection (in
  the Frobenius inner product) that preserves both symmetries, so G stays
  anti-Hermitian. Element counts depend on the orbital basis.

Both projections P are orthogonal and g = conj(A), so the slope of the step is
-Re sum P(g) A = -||P(g)||^2. The captured descent rate is therefore the kept
fraction of ||g||^2 (tested through cqe.run_cqe's captured_descent). This is
why every count here is "elements or leaves needed for a fraction f of
||g||^2".

First step from RHF (exact relation)
------------------------------------
Let |Phi> be a closed-shell determinant in canonical RHF orbitals (occupied i,
j; virtual a, b), so that Brillouin's theorem holds. A Wick contraction of
<Phi| e_{ia,jb} H |Phi> gives (tested to 1e-13; about 1e-16 observed)

    A[i,a,j,b] = W[(ia),(jb)] = 4 (ia|jb) - 2 (ib|ja),     A[a,i,b,j] = -W,

and A vanishes on every other block up to the SCF orbital gradient. The tests
require below 1e-8, and 1e-14 to 4e-10 is observed for H4, H6, LiH and N2.
The pair matrix is block diagonal, M = W (+) (-W), so the first generator has
exactly

    K_1 = 2 rank(W)       leaves, with |c| = the singular values of W, each twice.

The ERI Cholesky rank of the ov-ov block, r_ov = rank of V[(ia),(jb)] = (ia|jb),
bounds only the Coulomb part, 4V, which by itself gives exactly 2 r_ov leaves.
The exchange part X[(ia),(jb)] = (ib|ja) is a partial transpose of V and can
raise the rank. What holds in general is

    K_1 = 2 rank(4V - 2X) <= 2 min(n_occ n_vir, r_ov + rank X).

K_1 <= 2 r_ov is NOT a theorem. N2/STO-3G CAS(6,6) has r_ov = 8 and
K_1 = 18 (tests/test_generator_rank.py). For H4 at 1.8 bohr,
r_ov = n_occ n_vir = 4 and K_1 = 8. See :func:`first_step_rank_relation`.

Orbital bases (:func:`orbital_bases`)
-------------------------------------
Every basis u is real orthogonal, phi'_a = sum_p phi_p u[p, a], and

    g'[a,b,c,d] = sum_pqrs g[p,q,r,s] u[p,a] u[q,b] u[r,c] u[s,d]

(the same rule as ActiveSpace.rotated for the ERIs). 'canonical' is the
identity. 'natural' holds the eigenvectors of the spin-summed dm1 of the
state, ordered by decreasing occupation. Within clusters of degenerate
occupations (for example the occupied and virtual blocks of a determinant, or
pi pairs), the basis is aligned to the canonical orbitals by an orthogonal
Procrustes rotation, which makes it unique. It therefore equals the identity
for the HF determinant. The two Boys bases (pyscf.lo.Boys, best of several
starts) differ in what they mix:

* 'localized-ov' localizes the occupied and the virtual orbitals of the HF
  determinant separately (u is block diagonal). The HF determinant and the
  occupied/virtual block structure of the generator are preserved, as in
  local correlation methods, which never mix the two spaces.
* 'localized' localizes the whole active space at once and therefore mixes
  occupied and virtual orbitals. The first RHF generator lives on the (ov|ov)
  and (vo|vo) blocks in canonical and in 'localized-ov' orbitals, but not in
  this basis (tested). For the H4 RHF generator this raises the number of
  nonzero orbits (canonical <= 'localized-ov' < 'localized', tested). That
  ordering of the two Boys bases is not general;
  benchmarks/run_generator_rank.py compares all bases along trajectories.

Compressed CQE (:func:`compressed_cqe`)
---------------------------------------
'elements' and 'leaves' run :func:`thc_engine_cqe.cqe.run_cqe` with a compress
callback, so the step is the exact exp(eps G_kept) with the exact Krylov line
search. 'leaves-trotter' replaces exp(eps G_L) by the product formula of its
leaves of order ``trotter_order``, as in GeneratorFactorization.trotter_apply.
For order 1 the one-body term acts first, then the leaves in order of
decreasing |c|; order 2 is the symmetric Strang product. The step then has its
own line search, cqe.line_search on directly computed energies. The CNOT
estimates of both leaf schemes are those of this product circuit, of the same
``trotter_order`` (cost.cnot_cost_leaves; order 2 costs 2L rotations and
2L - 1 diagonal layers per step instead of L + 1 and L). 'leaves' is the
Trotter-error-free idealization of the same circuit. For 'elements' the
estimate assumes one exponential per spin-orbital excitation, a first-order
product (so only trotter_order = 1 is accepted), while the emulated step is
the exact exponential (no elementwise Trotter error).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Sequence

import numpy as np

from ..cqe import generator_norm_estimate, line_search, run_cqe
from ..factor.generator import (
    GeneratorFactorization,
    anti_hermitian_part,
    factorize_generator,
    pair_symmetrize,
)
from ..fci_ops import FCISpace, Hamiltonian, normalize
from ..integrals import ActiveSpace
from ..residual import compute_residual, descent_generator, rdm12
from . import cost

__all__ = [
    "FRACTIONS",
    "SCHEMES",
    "rotate_pair_tensor",
    "leaf_profile",
    "symmetry_orbits",
    "element_weights",
    "project_elements",
    "element_profile",
    "natural_orbitals",
    "boys_spread",
    "localization_blocks",
    "localized_orbitals",
    "orbital_bases",
    "first_step_rank_relation",
    "rank_study",
    "TrotterStep",
    "CompressedCQEResult",
    "compressed_cqe",
]

FRACTIONS = (0.9, 0.99, 0.999)
SCHEMES = ("elements", "leaves", "leaves-trotter")


def _fkey(f: float) -> str:
    return f"{f:g}"


def _n_for_fraction(weights_sorted: np.ndarray, fraction: float) -> int:
    """Smallest n with sum(w[:n]) / sum(w) >= fraction (1e-12 slack); w sorted descending."""
    tot = float(np.sum(weights_sorted))
    if tot == 0.0 or fraction <= 0.0:
        return 0
    cum = np.cumsum(weights_sorted) / tot
    return int(min(np.searchsorted(cum, fraction - 1e-12, side="left") + 1, len(weights_sorted)))


def rotate_pair_tensor(g: np.ndarray, u: np.ndarray) -> np.ndarray:
    """g'[a,b,c,d] = sum g[p,q,r,s] u[p,a] u[q,b] u[r,c] u[s,d] for a real orthogonal u."""
    u = np.asarray(u)
    n = u.shape[0]
    if u.shape != (n, n) or np.iscomplexobj(u) or not np.allclose(u.T @ u, np.eye(n), atol=1e-10):
        raise ValueError("u must be a real orthogonal (N, N) matrix")
    return np.einsum("pqrs,pa,qb,rc,sd->abcd", np.asarray(g), u, u, u, u, optimize=True)


def _symmetrized(g: np.ndarray) -> np.ndarray:
    """The canonical (pair-symmetric, anti-Hermitian) tensor of the operator of g."""
    return anti_hermitian_part(pair_symmetrize(np.asarray(g)))


# ------------------------------------------------------------------------ leaves
def leaf_profile(g: np.ndarray, fractions: Sequence[float] = FRACTIONS, rank_tol: float = 1e-8,
                 connectivities: Sequence[str] = cost.CONNECTIVITIES,
                 fact: GeneratorFactorization | None = None) -> dict:
    """Numerical leaf rank and the leaves needed for each fraction of ||g||^2.

    Returns a dict with:
        rank: number of leaves with |c_k| > rank_tol * max |c_k| (the numerical
            rank of the pair matrix at that tolerance).
        n_leaves[f]: fewest leaves capturing a fraction f of the weight.
        n_leaves_complete[f]: the same, rounded up to a complete group of
            degenerate |c|. These are real generators for real g, and they are
            what ``compressed_cqe(scheme='leaves')`` keeps.
        cnots[conn][f]: cost.cnot_cost_leaves of one first-order step with the complete count.
        singular_values: |c_k| (descending). norm2: ||gs||_F^2.
    Keys f are strings such as '0.99' (JSON friendly).
    """
    fac = factorize_generator(g, rank_tol=rank_tol) if fact is None else fact
    w = fac.weights()
    out = {
        "rank": int(fac.n_leaves),
        "rank_tol": float(rank_tol),
        "norm2": float(np.sum(w) + fac.discarded_weight),
        "n_leaves": {},
        "n_leaves_complete": {},
        "cnots": {c: {} for c in connectivities},
        "singular_values": [float(x) for x in np.abs(fac.coeffs)],
    }
    for f in fractions:
        n = fac.n_leaves_for(float(f))
        m = fac.complete_group(n)
        out["n_leaves"][_fkey(f)] = int(n)
        out["n_leaves_complete"][_fkey(f)] = int(m)
        for c in connectivities:
            out["cnots"][c][_fkey(f)] = cost.cnot_cost_leaves(fac, m, c)
    return out


# ---------------------------------------------------------------------- elements
@lru_cache(maxsize=16)
def symmetry_orbits(n: int) -> np.ndarray:
    """rep[x] for every flat index x of an (n,n,n,n) tensor: the smallest flat index of its orbit.

    The orbit of [p,q,r,s] is {[p,q,r,s], [r,s,p,q], [q,p,s,r], [s,r,q,p]}
    (pair exchange, adjoint, both).
    """
    idx = np.arange(n ** 4).reshape((n,) * 4)
    rep = np.minimum.reduce([idx, idx.transpose(2, 3, 0, 1), idx.transpose(1, 0, 3, 2),
                             idx.transpose(3, 2, 1, 0)])
    rep = rep.ravel()
    rep.setflags(write=False)
    return rep


def element_weights(g: np.ndarray, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """(reps, weights) of the nonzero symmetry-unique elements, sorted by decreasing weight.

    weight(orbit) = sum_{x in orbit} |gs[x]|^2 with gs the pair-symmetric,
    anti-Hermitian tensor of g, so sum(weights) = ||gs||_F^2. An orbit counts as
    nonzero if weight > tol^2 * max weight. Ties are broken by the
    representative index, which is deterministic.
    """
    gs = _symmetrized(g)
    n = gs.shape[0]
    rep = symmetry_orbits(n)
    w = np.bincount(rep, weights=np.abs(gs.ravel()) ** 2, minlength=n ** 4)
    wmax = float(w.max()) if w.size else 0.0
    if wmax == 0.0:
        return np.zeros(0, dtype=int), np.zeros(0)
    reps = np.nonzero(w > tol * tol * wmax)[0]
    order = np.lexsort((reps, -w[reps]))
    return reps[order], w[reps[order]]


def project_elements(g: np.ndarray, reps: Sequence[int]) -> np.ndarray:
    """gs restricted to the orbits ``reps`` (zero elsewhere): an orthogonal projection."""
    gs = _symmetrized(g)
    rep = symmetry_orbits(gs.shape[0])
    mask = np.isin(rep, np.asarray(reps, dtype=int)).reshape(gs.shape)
    out = np.where(mask, gs, 0)
    return np.ascontiguousarray(out.real) if np.isrealobj(g) else out


def element_profile(g: np.ndarray, u: np.ndarray | None = None,
                    fractions: Sequence[float] = FRACTIONS, tol: float = 1e-10,
                    connectivities: Sequence[str] = cost.CONNECTIVITIES,
                    double_core: int = cost.DOUBLE_CORE_PUBLISHED) -> dict:
    """Elementwise sparsity of g in the orbital basis u (identity if None).

    Returns a dict with:
        n_unique: number of nonzero symmetry-unique spin-free elements.
        n_elements[f]: fewest such elements (largest weights first) capturing a
            fraction f of ||g||^2.
        n_excitations[f]: distinct spin-orbital excitations
            (cost.spin_orbital_excitations) contained in those elements.
        cnots[conn][f]: cost.cnot_cost_elementwise of those excitations.
        n_excitations_total, cnots_total[conn]: the same for the full g.
        norm2: ||gs||_F^2 (basis invariant).
    """
    gg = np.asarray(g) if u is None else rotate_pair_tensor(g, u)
    n = gg.shape[0]
    reps, w = element_weights(gg, tol)
    full_exc = cost.spin_orbital_excitations(_symmetrized(gg), tol=tol)
    out = {
        "n_unique": int(len(reps)),
        "norm2": float(np.sum(w)),
        "n_elements": {},
        "n_excitations": {},
        "cnots": {c: {} for c in connectivities},
        "n_excitations_total": len(full_exc),
        "cnots_total": {c: cost.cnot_cost_elementwise(full_exc, n, c, double_core) for c in connectivities},
    }
    for f in fractions:
        k = _n_for_fraction(w, float(f))
        exc = cost.spin_orbital_excitations(project_elements(gg, reps[:k]), tol=tol) if k else []
        out["n_elements"][_fkey(f)] = int(k)
        out["n_excitations"][_fkey(f)] = len(exc)
        for c in connectivities:
            out["cnots"][c][_fkey(f)] = cost.cnot_cost_elementwise(exc, n, c, double_core)
    return out


# -------------------------------------------------------------------- orbital bases
def _fix_column_signs(u: np.ndarray) -> np.ndarray:
    idx = np.argmax(np.abs(u), axis=0)
    s = np.sign(u[idx, np.arange(u.shape[1])])
    s[s == 0] = 1.0
    return u * s


def natural_orbitals(dm1: np.ndarray, degeneracy_tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """(occupations, u): eigenpairs of the real symmetric part of dm1, by decreasing occupation.

    Inside each cluster of occupations whose successive differences are at most
    ``degeneracy_tol``, the eigenvectors are replaced by the orthonormal basis of
    the cluster subspace closest to the canonical orbitals. This is an orthogonal
    Procrustes rotation onto the canonical unit vectors with the largest weight
    in the subspace. It makes u unique and gives the identity for the HF
    determinant (occupied orbitals first). Each column's largest entry is positive.
    """
    d = np.asarray(dm1)
    d = 0.5 * (d.real + d.real.T)
    occ, vec = np.linalg.eigh(d)
    order = np.argsort(-occ, kind="stable")
    occ, vec = occ[order], vec[:, order]
    u = vec.copy()
    n = len(occ)
    start = 0
    while start < n:
        stop = start + 1
        while stop < n and abs(occ[stop] - occ[stop - 1]) <= degeneracy_tol:
            stop += 1
        if stop - start > 1:
            vc = vec[:, start:stop]
            weight = np.sum(vc ** 2, axis=1)
            sel = np.sort(np.argsort(-weight, kind="stable")[: stop - start])
            a, _, bt = np.linalg.svd(vc.T[:, sel])  # vc^T E_sel = a s bt
            u[:, start:stop] = vc @ (a @ bt)
        start = stop
    return occ, _fix_column_signs(u)


def boys_spread(mol, c: np.ndarray) -> float:
    """Total Boys spread sum_i (<i|r^2|i> - |<i|r|i>|^2) of the orbitals in the columns of c.

    This is the function that Boys localization minimizes (pyscf.lo.Boys.cost_function).
    Its first term is invariant under rotations among the columns, so minimizing
    the spread maximizes sum_i |<i|r|i>|^2.
    """
    c = np.asarray(c)
    r = mol.intor("int1e_r")  # (3, nao, nao)
    r2 = mol.intor("int1e_r2")
    d = np.einsum("pi,xpq,qi->xi", c, r, c)
    return float(np.einsum("pi,pq,qi->", c, r2, c) - np.sum(d ** 2))


def _boys_block(mol, c_blk: np.ndarray, n_random: int, rng: np.random.Generator) -> np.ndarray:
    """Boys orbitals of span(c_blk): the lowest-spread result of several starts.

    pyscf.lo.Boys (a second-order optimizer) can converge to a stationary point
    that is not a minimum. For example, the canonical occupied pair of the H4
    chain has definite inversion parity, so it is the spread *maximum* among the
    rotations of the pair (tested). Observed with PySCF 2.14: Boys returns that
    pair unchanged from both its 'atomic' and its identity start, and for LiH
    the 'atomic' start of the virtual block ends above the canonical spread.
    Starts: pyscf's 'atomic' guess, then ``n_random`` random rotations of c_blk.
    """
    from pyscf import lo

    m = c_blk.shape[1]
    if m < 2:
        return np.array(c_blk, copy=True)
    starts = [("atomic", c_blk)]
    for _ in range(n_random):
        q, rr = np.linalg.qr(rng.standard_normal((m, m)))
        starts.append((None, c_blk @ (q * np.sign(np.diag(rr)))))
    best, best_val = None, np.inf
    for guess, c0 in starts:
        boys = lo.Boys(mol, c0)
        boys.verbose = 0
        boys.init_guess = guess
        c_loc = np.asarray(boys.kernel())
        val = boys_spread(mol, c_loc)
        if best is None or val < best_val - 1e-9 * max(1.0, abs(best_val)):
            best, best_val = c_loc, val  # ties keep the earlier start (deterministic)
    return best


def localization_blocks(active: ActiveSpace, split_ov: bool) -> list[tuple[int, int]]:
    """Orbital index ranges [a, b) that are localized separately.

    Without ``split_ov`` this is the whole active space. With it, the blocks
    follow the occupations of FCISpace.hf(): doubly occupied [0, n_min),
    singly occupied [n_min, n_max) (empty for closed shells) and virtual
    [n_max, N), with n_min, n_max = min, max of (n_alpha, n_beta).
    """
    n = active.norb
    if not split_ov:
        return [(0, n)]
    na, nb = active.nelec
    cuts = sorted({0, min(na, nb), max(na, nb), n})
    return [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def localized_orbitals(active: ActiveSpace, split_ov: bool = False, n_random: int = 8,
                       seed: int = 0) -> np.ndarray | None:
    """Boys-localized active orbitals as an N x N rotation u (None without mol/mo_coeff).

    Without ``split_ov`` ('localized'), Boys runs over the whole active space.
    That mixes occupied and virtual orbitals: the HF determinant is no longer
    a single determinant of the new orbitals, and the occupied/virtual block
    structure of the CQE generator is lost. For example, the first generator
    from RHF lives only on the (ov|ov) and (vo|vo) blocks in canonical
    orbitals but not in this basis (tested), and for H4 it has more nonzero
    elements here than in either other basis (tested).

    With ``split_ov`` ('localized-ov'), each block of
    :func:`localization_blocks` (occupied and virtual for closed shells) is
    localized separately, as in local correlation methods. u is then exactly
    block diagonal, so the HF determinant and the block structure are
    preserved (tested).

    Each block is localized with pyscf.lo.Boys from several starts, keeping
    the lowest total spread (:func:`boys_spread`); see :func:`_boys_block`
    for why one start is not enough. The localized coefficients span the same
    space, so u = C_act^T S C_loc. It is orthogonalized by polar decomposition,
    off-block entries are set to zero, and column signs are fixed (largest
    entry positive). The orbital order is the localizer's; it is not
    optimized for Jordan-Wigner string lengths. The result is deterministic
    (fixed ``seed``) but not unique where the spread has flat directions, for
    example rotations among the pi orbitals of a linear molecule (LiH, N2), and
    element counts are not invariant under such rotations.
    """
    if active.mol is None or active.mo_coeff is None:
        return None
    mol = active.mol
    c_act = np.asarray(active.mo_coeff)
    n = active.norb
    rng = np.random.default_rng(seed)
    blocks = localization_blocks(active, split_ov)
    c_loc = np.hstack([_boys_block(mol, c_act[:, a:b], n_random, rng) for a, b in blocks])
    s = mol.intor_symmetric("int1e_ovlp")
    u = c_act.T @ s @ c_loc
    a, w, bt = np.linalg.svd(u)
    if not np.allclose(w, 1.0, atol=1e-8):
        raise RuntimeError("Boys orbitals do not span the active space")
    u = a @ bt
    mask = np.zeros((n, n), dtype=bool)
    for lo_, hi in blocks:
        mask[lo_:hi, lo_:hi] = True
    if np.abs(u[~mask]).max(initial=0.0) > 1e-8:
        raise RuntimeError("Boys orbitals mix the localization blocks")
    u = np.where(mask, u, 0.0)
    for lo_, hi in blocks:  # re-orthogonalize each block after zeroing the rest
        ab, _, bb = np.linalg.svd(u[lo_:hi, lo_:hi])
        u[lo_:hi, lo_:hi] = ab @ bb
    return _fix_column_signs(u)


def orbital_bases(active: ActiveSpace, ci: np.ndarray | None = None,
                  localized: bool = True) -> dict[str, np.ndarray]:
    """Orbital bases as N x N rotations: 'canonical', 'natural', 'localized-ov', 'localized'.

    'canonical' is the identity, 'natural' the natural orbitals of ``ci``
    (omitted if ``ci`` is None), 'localized-ov' Boys within the occupied and
    within the virtual orbitals of the HF determinant, and 'localized' Boys over
    the whole active space, which mixes occupied and virtual orbitals
    (:func:`localized_orbitals`). The two localized bases are omitted if
    ``active`` has no ``mol`` or ``mo_coeff`` or if ``localized`` is False.
    """
    n = active.norb
    out = {"canonical": np.eye(n)}
    if ci is not None:
        space = FCISpace.of(active)
        dm1, _ = rdm12(space, np.asarray(ci).reshape(space.shape))
        out["natural"] = natural_orbitals(dm1)[1]
    if localized:
        out.update(_localized_bases(active))
    return out


def _localized_bases(active: ActiveSpace) -> dict[str, np.ndarray]:
    """{'localized-ov': split Boys, 'localized': whole-space Boys}, or {} without mol/mo_coeff."""
    out = {}
    for name, split in (("localized-ov", True), ("localized", False)):
        u = localized_orbitals(active, split_ov=split)
        if u is not None:
            out[name] = u
    return out


# ------------------------------------------------------- first step: exact relation
def _pivoted_cholesky_rank(v: np.ndarray, rtol: float) -> int:
    """Number of pivoted-Cholesky vectors of a PSD matrix until max residual diag <= rtol * max diag."""
    v = np.array(v, dtype=float)
    d = np.diag(v).copy()
    dmax = float(d.max()) if d.size else 0.0
    if dmax <= 0.0:
        return 0
    vecs: list[np.ndarray] = []
    while True:
        a = int(np.argmax(d))
        if not d[a] > rtol * dmax or len(vecs) == len(d):
            return len(vecs)
        col = v[:, a] - sum(l * l[a] for l in vecs) if vecs else v[:, a].copy()
        l = col / np.sqrt(d[a])
        vecs.append(l)
        d = d - l ** 2
        d[a] = 0.0


def first_step_rank_relation(active: ActiveSpace, tol: float = 1e-8) -> dict:
    """Check K_1 = 2 rank(4V - 2X) and the bounds of the module docstring for the RHF generator.

    ``active`` must be in canonical RHF orbitals with the occupied ones first,
    so that the reference is FCISpace.hf() and Brillouin holds (all of
    :mod:`thc_engine_cqe.systems`). Ranks use the relative threshold ``tol``
    (singular values for W and X, residual diagonal for the pivoted Cholesky of V).

    Returns K1 (leaves of factorize_generator(conj(A), rank_tol=tol)), rank_W,
    r_ov (Cholesky rank of (ia|jb)), rank_X, n_ov = n_occ n_vir, bound =
    2 min(n_ov, r_ov + rank_X), max_dev = max |A[i,a,j,b] - W|, off_block =
    the largest |A| outside the (ov|ov) and (vo|vo) blocks, and the singular
    values of W.
    """
    if active.nelec[0] != active.nelec[1]:
        raise ValueError("the closed-shell relation needs n_alpha == n_beta")
    space = FCISpace.of(active)
    rd = compute_residual(active, space.hf())
    n, no = active.norb, active.nelec[0]
    nv = n - no
    o, v = slice(0, no), slice(no, n)
    blk = active.eri[o, v, o, v]
    V = blk.reshape(no * nv, no * nv)
    X = np.einsum("ibja->iajb", blk).reshape(no * nv, no * nv)
    W = 4.0 * V - 2.0 * X
    mask = np.zeros(rd.A.shape, dtype=bool)
    mask[o, v, o, v] = True
    mask[v, o, v, o] = True

    def rank(m):
        s = np.linalg.svd(m, compute_uv=False)
        return int(np.sum(s > tol * s[0])) if s.size and s[0] > 0 else 0, s

    rank_w, sw = rank(W)
    rank_x, _ = rank(X)
    r_ov = _pivoted_cholesky_rank(V, tol)
    fac = factorize_generator(descent_generator(rd.A), rank_tol=tol)
    return {
        "K1": int(fac.n_leaves),
        "rank_W": rank_w,
        "r_ov": r_ov,
        "rank_X": rank_x,
        "n_ov": no * nv,
        "bound": 2 * min(no * nv, r_ov + rank_x),
        "max_dev": float(np.abs(rd.A[o, v, o, v].reshape(no * nv, -1) - W).max()),
        "off_block": float(np.abs(rd.A[~mask]).max()) if (~mask).any() else 0.0,
        "singular_values_W": [float(x) for x in sw],
    }


# ---------------------------------------------------------------------- rank study
def rank_study(active: ActiveSpace, max_iter: int = 15, fractions: Sequence[float] = FRACTIONS,
               rank_tol: float = 1e-8, element_tol: float = 1e-10, localized: bool = True,
               connectivities: Sequence[str] = cost.CONNECTIVITIES, e_fci: float | None = None,
               extra_bases: dict[str, np.ndarray] | None = None) -> list[dict]:
    """Leaf and element profiles of every generator along the exact steepest-descent CQE.

    Runs cqe.run_cqe from the HF determinant with keep_generators=True. For
    iteration t (state Psi_t, generator g_t = conj(A_t)) the dict holds:
    iteration, energy, error (E - E_FCI), residual_norm (||A||), eps,
    energy_after, leaf_profile (:func:`leaf_profile`), element_profile[basis]
    (:func:`element_profile`) for 'canonical', 'natural' (of Psi_t),
    'localized-ov' and 'localized' (the Boys bases of :func:`orbital_bases`, if
    available and ``localized``) and any ``extra_bases``, and natural_occupations.
    """
    e_ref = active.fci()[0] if e_fci is None else float(e_fci)
    space = FCISpace.of(active)
    states = [space.hf()]
    res = run_cqe(active, max_iter=max_iter, keep_generators=True, e_exact=e_ref,
                  callback=lambda step, ci: states.append(ci))
    fixed = {"canonical": np.eye(active.norb)}
    if localized:
        fixed.update(_localized_bases(active))
    if extra_bases:
        fixed.update(extra_bases)
    out = []
    for step in res.steps:
        t = step.iteration
        ci = states[t]
        dm1, _ = rdm12(space, ci)
        occ, u_nat = natural_orbitals(dm1)
        bases = {"canonical": fixed["canonical"], "natural": u_nat}
        bases.update({k: v for k, v in fixed.items() if k != "canonical"})
        g = step.generator
        out.append({
            "iteration": int(t),
            "energy": float(step.energy),
            "error": float(step.energy - e_ref),
            "residual_norm": float(step.residual_norm),
            "eps": float(step.eps),
            "energy_after": float(step.energy_after),
            "natural_occupations": [float(x) for x in occ],
            "leaf_profile": leaf_profile(g, fractions, rank_tol, connectivities),
            "element_profile": {name: element_profile(g, None if name == "canonical" else u,
                                                      fractions, element_tol, connectivities)
                                for name, u in bases.items()},
        })
    return out


# -------------------------------------------------------- fast leaf product formula
@lru_cache(maxsize=8)
def _string_excitations(norb: int, nelec: tuple[int, int]):
    """Dense E_pq on alpha and beta strings: (Ea, Eb) with shape (N, N, n_str, n_str)."""
    from pyscf.fci import cistring

    mats = []
    for ne in nelec:
        ns = cistring.num_strings(norb, ne)
        e = np.zeros((norb, norb, ns, ns))
        if ne > 0:
            link = cistring.gen_linkstr_index(range(norb), ne)
            for i0 in range(ns):
                for a, i, i1, sign in link[i0]:
                    e[a, i, i1, i0] += sign  # a+_a a_i |i0> = sign |i1>
        e.setflags(write=False)
        mats.append(e)
    return mats[0], mats[1]


class TrotterStep:
    """Exact emulation of the first- or second-order product of a factorized generator on CI vectors.

    Same operator sequence as GeneratorFactorization.trotter_apply (checked in
    tests): terms X_0 = Ohat_{F_n} and X_j = c_{j-1} i Qhat_{j-1}^2. Order 1
    applies X_0 first, then X_1, ..., X_n; order 2 is the symmetric Strang
    product. It is faster because every one-body operator splits into commuting
    alpha and beta parts, Qhat = Qa (x) 1 + 1 (x) Qb on the
    (alpha string, beta string) CI matrix. Hence
    exp(i t c Qhat^2) C = Wa [exp(i t c (la_i + lb_j)^2) * (Wa^dag C conj(Wb))] Wb^T,
    from the eigendecompositions Qa = Wa diag(la) Wa^dag and likewise for b.
    The cost is O(n_str^3) per factor, with no Taylor series.
    """

    def __init__(self, space: FCISpace, fact: GeneratorFactorization, n_leaves: int | None = None,
                 order: int = 1):
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        self.space, self.fact, self.order = space, fact, order
        n = fact.n_leaves if n_leaves is None else max(0, min(int(n_leaves), fact.n_leaves))
        self.n_leaves = n
        ea, eb = _string_excitations(space.norb, space.nelec)
        terms = []  # (wa, la, wb, lb, kind, c): factor exp(t X) in the joint eigenbasis
        f = fact.one_body_for(n)
        if np.any(f != 0):
            fa = np.einsum("pq,pqij->ij", f, ea)
            fb = np.einsum("pq,pqij->ij", f, eb)
            # F anti-Hermitian: iF Hermitian, exp(t F) = W exp(-i t lam) W^dag
            la, wa = np.linalg.eigh(1j * fa)
            lb, wb = np.linalg.eigh(1j * fb)
            terms.append((wa, la, wb, lb, "one", 0.0))
        for k in range(n):
            q = fact.Q[k]
            qa = np.einsum("pq,pqij->ij", q, ea)
            qb = np.einsum("pq,pqij->ij", q, eb)
            la, wa = np.linalg.eigh(qa)
            lb, wb = np.linalg.eigh(qb)
            terms.append((wa, la, wb, lb, "leaf", float(fact.coeffs[k])))
        self._terms = terms

    @staticmethod
    def _apply_term(term, t: float, c: np.ndarray) -> np.ndarray:
        wa, la, wb, lb, kind, ck = term
        s = la[:, None] + lb[None, :]
        phase = np.exp(-1j * t * s) if kind == "one" else np.exp(1j * t * ck * s * s)
        x = wa.conj().T @ c @ wb.conj()
        return wa @ (phase * x) @ wb.T

    def apply(self, eps: float, ci: np.ndarray) -> np.ndarray:
        c = np.asarray(ci).reshape(self.space.shape).astype(complex)
        terms = self._terms
        if not terms:
            return c
        if self.order == 1:
            for term in terms:
                c = self._apply_term(term, eps, c)
            return c
        for term in terms[:-1]:
            c = self._apply_term(term, 0.5 * eps, c)
        c = self._apply_term(terms[-1], eps, c)
        for term in reversed(terms[:-1]):
            c = self._apply_term(term, 0.5 * eps, c)
        return c


# ------------------------------------------------------------------- compressed CQE
@dataclass
class CompressedCQEResult:
    """Per-iteration records of :func:`compressed_cqe`.

    Each record has iteration, energy (before), energy_after, error_after
    (energy_after - E_FCI), residual_norm, kept, available (elements or leaves),
    captured (kept fraction of ||g||^2, i.e. slope / full slope), eps,
    step_cnots[conn] and cum_cnots[conn]. ``step_cnots`` is the cost of this
    step alone. For leaves this includes the final un-rotation. ``cum_cnots`` is
    the cost of the whole circuit up to and including this step. Leaf schemes
    merge the rotations at step boundaries, so they add the final
    un-rotation once. Steps with eps = 0 cost nothing. ``trotter_order`` is the
    order of the product formula that the CNOT estimates count (and that
    'leaves-trotter' emulates).
    """

    scheme: str
    per_iter: int
    e_fci: float
    e_hf: float
    records: list[dict] = field(default_factory=list)
    ci: np.ndarray | None = None
    basis: str = "canonical"
    trotter_order: int = 1

    @property
    def energies(self) -> np.ndarray:
        """E_HF followed by the energy after every step."""
        return np.array([self.e_hf] + [r["energy_after"] for r in self.records])

    @property
    def errors(self) -> np.ndarray:
        return self.energies - self.e_fci

    def cum_cnots(self, connectivity: str = "all") -> np.ndarray:
        """0 followed by the cumulative CNOT estimate after every step."""
        return np.array([0] + [r["cum_cnots"][connectivity] for r in self.records])

    def to_dict(self) -> dict:
        return {"scheme": self.scheme, "per_iter": self.per_iter, "basis": self.basis,
                "trotter_order": self.trotter_order, "e_fci": self.e_fci, "e_hf": self.e_hf,
                "records": self.records}


def _cumulate(records: list[dict], connectivities: Sequence[str], n: int, n_fixed_rotations: int,
              leaf_like: bool) -> None:
    """Fill step_cnots/cum_cnots.

    cum = sum over moving steps of the per-step cost (for leaf schemes the cost
    without the final un-rotation) + n_fixed_rotations * C_rot once the first
    step has moved.
    """
    rot = {c: cost.orbital_rotation_cnots(n, c) for c in connectivities}
    acc = {c: 0 for c in connectivities}
    taken = False
    for r in records:
        inner = r.pop("_inner", None)
        moved = r["eps"] != 0.0 and r["kept"] > 0
        if not moved:
            r["step_cnots"] = {c: 0 for c in connectivities}
        else:
            for c in connectivities:
                acc[c] += inner[c] if leaf_like else r["step_cnots"][c]
        taken = taken or moved
        r["cum_cnots"] = {c: int(acc[c] + (n_fixed_rotations * rot[c] if taken else 0))
                          for c in connectivities}


def compressed_cqe(active: ActiveSpace, scheme: str, per_iter: int, max_iter: int = 20,
                   e_fci: float | None = None,
                   connectivities: Sequence[str] = cost.CONNECTIVITIES,
                   rank_tol: float = 1e-8, element_tol: float = 1e-10,
                   complete_groups: bool = True, trotter_order: int = 1,
                   double_core: int = cost.DOUBLE_CORE_PUBLISHED,
                   r_tol: float = 1e-7, e_tol: float = 1e-11, n_grid: int = 48,
                   basis: np.ndarray | None = None, basis_label: str | None = None) -> CompressedCQEResult:
    """CQE from the HF determinant with a compressed generator at every iteration.

    Args:
        scheme: 'elements' keeps the ``per_iter`` symmetry-unique spin-free
            elements of largest weight (:func:`project_elements`), selected in
            the orbital basis ``basis`` (default: the active-space orbitals).
            The projection P_u = R_u^T P R_u, with R_u the tensor rotation of
            :func:`rotate_pair_tensor`, is again orthogonal. For a non-identity
            basis the circuit needs U(u)^dag once at the start and U(u) once at
            the end, which adds 2 C_rot to the cumulative estimate. 'leaves' keeps the
            top ``per_iter`` leaves, rounded up to a complete group of degenerate |c|
            if ``complete_groups`` (factor.generator.projected_generator). Both use the
            exact exponential and Krylov line search of cqe.run_cqe.
            'leaves-trotter' keeps the same leaves but steps with the product formula
            of order ``trotter_order`` (:class:`TrotterStep`) and its own
            line search (cqe.line_search on direct energies, eps_max doubled while
            the minimum sits at the boundary).
        per_iter: elements or leaves per iteration (a huge value keeps everything).
        trotter_order: 1 or 2, the order of the leaf product formula. Both leaf
            schemes charge the CNOTs of that circuit (cost.cnot_cost_leaves),
            and 'leaves-trotter' also emulates it. 'elements' accepts only 1,
            because its cost model is a first-order product of excitations.
        rank_tol: leaves with |c| <= rank_tol max|c| are dropped (numerical noise,
            e.g. the 1e-10 Brillouin terms of the RHF generator).
    """
    if scheme not in SCHEMES:
        raise ValueError(f"scheme must be one of {SCHEMES}")
    if per_iter < 1:
        raise ValueError("per_iter must be >= 1")
    if trotter_order not in (1, 2):
        raise ValueError("trotter_order must be 1 or 2")
    if scheme == "elements" and trotter_order != 1:
        raise ValueError("the elementwise cost model is a first-order product; use trotter_order=1")
    u = None
    if basis is not None:
        if scheme != "elements":
            raise ValueError("a basis only matters for scheme='elements' (leaves are basis invariant)")
        u = np.asarray(basis, dtype=float)
        rotate_pair_tensor(np.zeros((active.norb,) * 4), u)  # validates u
        if np.allclose(u, np.eye(active.norb), atol=1e-14):
            u = None
    connectivities = tuple(connectivities)
    for c in connectivities:
        cost._check_connectivity(c)
    e_ref = active.fci()[0] if e_fci is None else float(e_fci)
    ham = Hamiltonian(active)
    space = ham.space
    n = active.norb
    e_hf = ham.energy(space.hf())
    label = basis_label or ("canonical" if u is None else "custom")
    result = CompressedCQEResult(scheme=scheme, per_iter=int(per_iter), e_fci=e_ref, e_hf=e_hf,
                                 basis=label, trotter_order=int(trotter_order))
    info: list[dict] = []

    def choose_leaves(g):
        fac = factorize_generator(g, rank_tol=rank_tol)
        k = min(int(per_iter), fac.n_leaves)
        if complete_groups:
            k = fac.complete_group(k)
        inner = {c: cost.cnot_cost_leaves(fac, k, c, final_rotation=False, order=trotter_order)
                 for c in connectivities}
        step = {c: cost.cnot_cost_leaves(fac, k, c, final_rotation=True, order=trotter_order)
                for c in connectivities}
        rec = {"kept": int(k), "available": int(fac.n_leaves), "captured": fac.captured_fraction(k),
               "step_cnots": step, "_inner": inner}
        return fac, k, rec

    if scheme in ("elements", "leaves"):
        def compress(g, rd, it):
            if scheme == "elements":
                gu = g if u is None else rotate_pair_tensor(g, u)
                reps, w = element_weights(gu, element_tol)
                k = min(int(per_iter), len(reps))
                gk_u = project_elements(gu, reps[:k])
                exc = cost.spin_orbital_excitations(gk_u, tol=element_tol) if k else []
                gk = gk_u if u is None else rotate_pair_tensor(gk_u, u.T)
                info.append({"kept": int(k), "available": int(len(reps)),
                             "captured": float(np.sum(w[:k]) / np.sum(w)) if len(w) else 1.0,
                             "n_excitations": len(exc),
                             "step_cnots": {c: cost.cnot_cost_elementwise(exc, n, c, double_core)
                                            for c in connectivities}})
                return gk
            fac, k, rec = choose_leaves(g)
            info.append(rec)
            return fac.pair_tensor(k)

        # Odd leaf truncations of real generators are complex; run_cqe propagates
        # them exactly from the real HF start (the state then becomes complex).
        res = run_cqe(active, max_iter=max_iter, compress=compress, e_exact=e_ref,
                      r_tol=r_tol, e_tol=e_tol)
        for step, rec in zip(res.steps, info):
            rec.update({"iteration": int(step.iteration), "energy": float(step.energy),
                        "energy_after": float(step.energy_after),
                        "error_after": float(step.energy_after - e_ref),
                        "residual_norm": float(step.residual_norm), "eps": float(step.eps),
                        "slope": float(step.slope), "captured_run": float(step.captured_descent)})
            result.records.append(rec)
        result.ci = res.ci
    else:
        psi = space.hf().astype(complex)
        e_cur = e_hf
        for it in range(max_iter):
            rd = compute_residual(active, psi, ham)
            if rd.residual_norm < r_tol:
                break
            g = descent_generator(rd.A)
            fac, k, rec = choose_leaves(g)
            gk = fac.pair_tensor(k)
            slope = -float(np.real(np.sum(gk * rd.A)))
            rec.update({"iteration": it, "energy": float(rd.energy), "residual_norm": float(rd.residual_norm),
                        "slope": slope, "captured_run": slope / (-rd.residual_norm ** 2)})
            if k == 0 or slope >= 0.0:
                rec.update({"eps": 0.0, "energy_after": float(rd.energy), "error_after": float(rd.energy - e_ref)})
                result.records.append(rec)
                break
            prop = TrotterStep(space, fac, k, order=trotter_order)
            eps_max = 1.5 / max(generator_norm_estimate(space, gk, psi), 1e-12)

            def energy(eps, prop=prop, psi=psi):
                return ham.energy(prop.apply(eps, psi))

            for _ in range(8):
                eps, e_new = line_search(energy, eps_max, n_grid=n_grid)
                if eps < 0.98 * eps_max:
                    break
                eps_max *= 2.0
            psi = normalize(prop.apply(eps, psi))
            e_new = ham.energy(psi)
            rec.update({"eps": float(eps), "energy_after": float(e_new), "error_after": float(e_new - e_ref)})
            result.records.append(rec)
            converged = abs(e_cur - e_new) < e_tol
            e_cur = e_new
            if converged:
                break
        result.ci = psi
    leaf_like = scheme != "elements"
    _cumulate(result.records, connectivities, n, 1 if leaf_like else (0 if u is None else 2), leaf_like)
    return result

"""Measurement targets of the CQE: the energy and the two residual routes as Hermitian Pauli sums.

A *target* is a list of Hermitian :class:`~thc_engine_cqe.measure.pauli.PauliSum`
operators O_e, one per element e, together with the linear post-processing that
maps their expectation values to the quantity of interest. Measurement schemes
estimate the <O_e>; :mod:`thc_engine_cqe.measure.routes` prepares the states and
applies the post-processing. Conventions are those of docs/conventions.md and of
the module docstring of :mod:`thc_engine_cqe.measure.pauli`.

Notation
--------
* A spin orbital is (p, sigma), sigma = 0 (alpha) or 1 (beta). Its *mode index* is
  p + sigma N, independent of the qubit mapping. Orderings of spin orbitals below
  ("i < j") are orderings of mode indices.
* Gamma_e = a+_i a+_j a_l a_k for a spin-orbital element e = (i, j, k, l)
  (Mazziotti order: <Gamma_e> = 2D^{ij}_{kl}), and Gamma_e = e_{pq,rs} =
  sum_{sigma tau} a+_{p sigma} a+_{r tau} a_{s tau} a_{q sigma} for a spin-free
  element e = (p, q, r, s) (pair order).
* Residuals: 2A^{ij}_{kl} = <[a+_i a+_j a_l a_k, H]> and A[p,q,r,s] = <[e_{pq,rs}, H]>
  (``residual.compute_residual``). Hence
  A[p,q,r,s] = sum_{sigma tau} 2A^{(p sigma)(r tau)}_{(q sigma)(s tau)} (:func:`contract_spin`).

Element sets (real states)
--------------------------
The states are real CI vectors (exact CQE trajectories with real orbitals), so
the residual is real and obeys the symmetries used below.

* :func:`spin_orbital_elements`: rows (i, j, k, l) with i < j, k < l,
  S_z(i) + S_z(j) = S_z(k) + S_z(l) and (i, j) < (k, l) lexicographically,
  sorted lexicographically by (i, j, k, l) (mode indices). Every other entry of
  the S_z-conserving 2A follows from 2A^{ji}_{kl} = 2A^{ij}_{lk} = -2A^{ij}_{kl}
  (operator identities) and, for real states, 2A^{kl}_{ij} = -2A^{ij}_{kl}
  ([Gamma^dagger, H] = -[Gamma, H]^dagger and A real), which also makes the
  diagonal (ij) = (kl) vanish. Each row stands for 8 entries of the (2N)^4
  tensor (:func:`spin_orbital_tensor`). There are C(N,2) alpha-alpha, C(N,2)
  beta-beta and N^2 alpha-beta pairs, so K = 2 C(C(N,2), 2) + C(N^2, 2):
  150 for N = 4 and 840 for N = 6 (tested by brute force over orbits).
* :func:`spin_free_elements`: A[p,q,r,s] = A[r,s,p,q] (pair exchange) and
  A[q,p,s,r] = -conj(A[p,q,r,s]) = -A[p,q,r,s] (real states) generate the group
  {1, P, T, PT} with P: (p,q,r,s) -> (r,s,p,q) (sign +1), T: -> (q,p,s,r) (sign -1)
  and PT: -> (s,r,q,p) (sign -1). An orbit is forced to zero when T or PT fixes
  it (p = q and r = s, or p = s and q = r). The representative of every other
  orbit is its lexicographically smallest member; the representatives are sorted
  lexicographically. In the basis of symmetric and antisymmetric orbital pairs,
  A couples only the N(N+1)/2 symmetric to the N(N-1)/2 antisymmetric
  combinations, so there are N^2 (N^2 - 1)/4 independent elements: 60 for N = 4
  and 315 for N = 6 (tested by brute force over orbits). Orbits have 2 members
  for (p,q,p,q), p != q (N(N-1)/2 of them), and 4 otherwise
  (:func:`element_multiplicities`).

Targets and post-processing
---------------------------
* Energy (:func:`energy_target`): H, including ecore as the identity term; E = <H>.
* Auxiliary-state route (:func:`aux_targets`): Q_e = (Gamma_e - Gamma_e^dagger)/(2i),
  Hermitian, with <Q_e> = Im <Gamma_e>; its Pauli coefficients are the imaginary
  parts of those of Gamma_e. With Lambda = exp(-i delta H) Psi,
  <Gamma_e>_Lambda = sum_k (i delta)^k / k! <ad_H^k(Gamma_e)>, ad_H(X) = [H, X].
  For real Psi and H every <ad_H^k(Gamma_e)> is real, so only odd k enter
  Im <Gamma_e>_Lambda and

      A_hat_e = -<Q_e>_Lambda / delta = A_e + (delta^2/6) <[H,[H,[H,Gamma_e]]]> + O(delta^4)

  (the leading coefficient is tested in tests/test_routes.py). Post-processing
  (:func:`aux_estimates`): A_hat_e = -<Q_e>_Lambda / delta for one auxiliary
  state, sum_l -<Q_e>_{Lambda_l} / delta for DF fragments (``routes.aux_states``).
* Direct route (:func:`direct_targets`): X_e = [Gamma_e - Gamma_e^dagger, H]/2
  = i [Q_e, H], Hermitian with <X_e>_Psi = Re A_e, which is A_e for real Psi
  (identity post-processing). X_e has up to three-body terms.

Complex states (not implemented here)
-------------------------------------
For complex Psi, A_e is complex, and the elements forced to zero above (the
diagonal (ij) = (kl) and the T- or PT-fixed orbits) are purely imaginary, so the
element sets are incomplete. The direct route then also needs
Y_e = [Gamma_e + Gamma_e^dagger, H]/(2i), with <Y_e> = Im A_e. The auxiliary
route measures the real and imaginary parts of <Gamma_e> separately, through the
Hermitian pair (Gamma_e + Gamma_e^dagger)/2 and Q_e, on Lambda_delta and on Psi
(or on Lambda_{-delta}): A_e = i (<Gamma_e>_{Lambda_delta} - <Gamma_e>_Psi)/delta + O(delta),
or i (<Gamma_e>_{Lambda_delta} - <Gamma_e>_{Lambda_-delta})/(2 delta) + O(delta^2).
The tensor helpers of this module reject complex values.

Qubit orders
------------
The operators depend on the :class:`~thc_engine_cqe.measure.pauli.QubitMapping`;
their expectation values on a state built with the same mapping do not (tested
for both orders). Groupings such as QWC do depend on the order.

Size and cost (H6/STO-3G at 1.0 A, 12 qubits)
---------------------------------------------
The Q_e have at most 8 Pauli terms (spin-orbital) and 32 (spin-free) (tested).
Observed, one thread: the 919-term H gives 840 spin-orbital X_e with 1,603,800
Pauli terms in total (mean 1909, max 2448) and 315 spin-free X_e with 1,528,224
terms (mean 4852, max 7288); each set builds in 0.4-0.7 s, and both span the
same 36,474 Pauli strings. See tests/test_targets.py.
"""

from __future__ import annotations

import numpy as np

from .pauli import PauliSum, QubitMapping, jw_hamiltonian, jw_rdm2_elements

__all__ = [
    "KINDS",
    "spin_orbital_elements",
    "spin_free_elements",
    "element_set",
    "element_multiplicities",
    "spin_free_tensor",
    "spin_orbital_tensor",
    "contract_spin",
    "gamma_operators",
    "energy_target",
    "aux_targets",
    "direct_targets",
    "aux_estimates",
]

KINDS = ("spin_orbital", "spin_free")


# ============================================================ element sets
def _check_kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return kind


def _check_norb(norb) -> int:
    n = int(norb)
    if n != norb or n < 1 or n > 32:
        raise ValueError(f"norb must be an integer in [1, 32], got {norb!r}")
    return n


def spin_orbital_elements(norb: int) -> np.ndarray:
    """Independent S_z-conserving elements of the spin-orbital residual 2A^{ij}_{kl}.

    Returns an int64 array of shape (K, 4, 2): row e is (i, j, k, l), each a
    spin orbital (p, sigma), the order of ``jw_rdm2_elements`` (operator
    a+_i a+_j a_l a_k). Mode indices m = p + sigma N satisfy m_i < m_j, m_k < m_l
    and (m_i, m_j) < (m_k, m_l); rows are sorted lexicographically by
    (m_i, m_j, m_k, m_l). K = 2 C(C(N,2), 2) + C(N^2, 2), 840 for N = 6 (module
    docstring).
    """
    n = _check_norb(norb)
    nm = 2 * n
    a, b = np.triu_indices(nm, k=1)                     # pairs (a < b), lexicographic
    pair_sz = (a >= n).astype(int) + (b >= n).astype(int)  # number of beta spin orbitals
    lo, hi = np.triu_indices(a.size, k=1)               # pair index lo < hi, lexicographic
    keep = pair_sz[lo] == pair_sz[hi]
    lo, hi = lo[keep], hi[keep]
    modes = np.stack([a[lo], b[lo], a[hi], b[hi]], axis=1).astype(np.int64)
    return np.stack([modes % n, modes // n], axis=-1)


def _spin_free_images(p, q, r, s, n: int):
    """Flat indices of (p,q,r,s) and of its images under P, T and PT (module docstring)."""
    def flat(w, x, y, z):
        return ((w * n + x) * n + y) * n + z
    return flat(p, q, r, s), flat(r, s, p, q), flat(q, p, s, r), flat(s, r, q, p)


def spin_free_elements(norb: int) -> np.ndarray:
    """Representatives (p, q, r, s) of the independent elements of the real pair-order A.

    One row per orbit of {1, P, T, PT} (module docstring) that is not forced to
    zero; the representative is the lexicographically smallest member and the
    rows are sorted lexicographically. Returns an int64 array of shape
    (N^2 (N^2 - 1)/4, 4): 60 rows for N = 4, 315 for N = 6.
    """
    n = _check_norb(norb)
    p, q, r, s = (g.ravel() for g in np.meshgrid(*(np.arange(n),) * 4, indexing="ij"))
    f0, f_p, f_t, f_pt = _spin_free_images(p, q, r, s, n)
    rep = np.minimum(np.minimum(f0, f_p), np.minimum(f_t, f_pt))
    forced = (f0 == f_t) | (f0 == f_pt)
    keep = (f0 == rep) & ~forced
    return np.stack([p[keep], q[keep], r[keep], s[keep]], axis=1).astype(np.int64)


def element_set(norb: int, kind: str = "spin_orbital") -> np.ndarray:
    """:func:`spin_orbital_elements` or :func:`spin_free_elements` by ``kind``."""
    return spin_orbital_elements(norb) if _check_kind(kind) == "spin_orbital" else spin_free_elements(norb)


def _as_elements(elements, kind: str, norb: int) -> np.ndarray:
    """Validated int64 element array: (K, 4, 2) spin orbitals or (K, 4) orbitals."""
    _check_kind(kind)
    el = np.asarray(elements)
    if el.size == 0:
        shape = (0, 4, 2) if kind == "spin_orbital" else (0, 4)
        return np.zeros(shape, dtype=np.int64)
    if el.dtype.kind not in "iu":
        raise TypeError(f"elements must be integers, got dtype {el.dtype}")
    el = el.astype(np.int64)
    if kind == "spin_orbital":
        if el.ndim == 2 and el.shape == (4, 2):
            el = el[None]
        if el.ndim != 3 or el.shape[1:] != (4, 2):
            raise ValueError(f"spin-orbital elements must have shape (K, 4, 2), got {el.shape}")
        if np.any((el[..., 0] < 0) | (el[..., 0] >= norb)) or np.any((el[..., 1] != 0) & (el[..., 1] != 1)):
            raise ValueError(f"spin orbital out of range for norb={norb}")
    else:
        if el.ndim == 1 and el.shape == (4,):
            el = el[None]
        if el.ndim != 2 or el.shape[1] != 4:
            raise ValueError(f"spin-free elements must have shape (K, 4), got {el.shape}")
        if np.any((el < 0) | (el >= norb)):
            raise ValueError(f"orbital index out of range for norb={norb}")
    return el


def _modes(el: np.ndarray, norb: int) -> np.ndarray:
    """(K, 4) mode indices p + sigma N of spin-orbital elements (K, 4, 2)."""
    return el[..., 0] + el[..., 1] * norb


def _so_orbit_keys(modes: np.ndarray, nm: int):
    """Orbit key of each spin-orbital element under the 8 (anti)symmetries, and the zero mask.

    The key encodes the canonical member (i < j, k < l, (ij) < (kl)); an element is
    forced to zero when i = j, k = l or {i, j} = {k, l}.
    """
    i, j, k, l = modes.T
    pa = np.minimum(i, j) * nm + np.maximum(i, j)
    pb = np.minimum(k, l) * nm + np.maximum(k, l)
    key = np.minimum(pa, pb) * nm * nm + np.maximum(pa, pb)
    zero = (i == j) | (k == l) | (pa == pb)
    return key, zero


def _sf_orbit_keys(el: np.ndarray, n: int):
    p, q, r, s = el.T
    f0, f_p, f_t, f_pt = _spin_free_images(p, q, r, s, n)
    key = np.minimum(np.minimum(f0, f_p), np.minimum(f_t, f_pt))
    zero = (f0 == f_t) | (f0 == f_pt)
    return key, zero, (f0, f_p, f_t, f_pt)


def element_multiplicities(elements, kind: str, norb: int) -> np.ndarray:
    """Number of entries of the full tensor that each element stands for (int64).

    Spin-orbital: 8 entries of the (2N)^4 Mazziotti-order tensor; spin-free: the
    orbit size in the pair-order N^4 tensor (2 or 4). So the Frobenius norm of the
    filled tensor is sqrt(sum_e m_e A_e^2). Elements forced to zero raise.
    """
    n = _check_norb(norb)
    el = _as_elements(elements, kind, n)
    if kind == "spin_orbital":
        _, zero = _so_orbit_keys(_modes(el, n), 2 * n)
        if np.any(zero):
            raise ValueError("elements forced to zero (i = j, k = l or {i,j} = {k,l}) have no multiplicity")
        return np.full(el.shape[0], 8, dtype=np.int64)
    _, zero, images = _sf_orbit_keys(el, n)
    if np.any(zero):
        raise ValueError("spin-free elements forced to zero (p = q and r = s, or p = s and q = r)")
    srt = np.sort(np.stack(images, axis=1), axis=1)
    return (1 + np.count_nonzero(np.diff(srt, axis=1), axis=1)).astype(np.int64)


# ============================================================ tensor helpers
def _real_values(values, k: int) -> np.ndarray:
    v = np.asarray(values)
    if v.shape != (k,):
        raise ValueError(f"values must have shape ({k},), got {v.shape}")
    if np.iscomplexobj(v):
        scale = float(np.max(np.abs(v))) if v.size else 0.0
        if v.size and float(np.max(np.abs(v.imag))) > 1e-10 * scale:
            raise ValueError("values must be real (real states); complex residuals are not supported")
        v = v.real
    return np.asarray(v, dtype=float)


def _check_distinct(key: np.ndarray, what: str) -> None:
    if np.unique(key).size != key.size:
        raise ValueError(f"two {what} elements belong to the same symmetry orbit")


def spin_free_tensor(values, elements, norb: int) -> np.ndarray:
    """Pair-order tensor A[p,q,r,s] filled from its values on independent elements.

    ``elements`` (K, 4) are members of distinct orbits that are not forced to zero
    (e.g. :func:`spin_free_elements`); ``values`` (K,) are real. The orbit of
    element x gets A[x] = A[P x] = v and A[T x] = A[PT x] = -v (module docstring);
    entries of orbits not in ``elements`` are zero. Round trip:
    ``spin_free_tensor(A[tuple(el.T)], el, N) == A`` for the residual A of a real state.
    """
    n = _check_norb(norb)
    el = _as_elements(elements, "spin_free", n)
    v = _real_values(values, el.shape[0])
    key, zero, _ = _sf_orbit_keys(el, n)
    if np.any(zero):
        raise ValueError("spin-free elements forced to zero (p = q and r = s, or p = s and q = r)")
    _check_distinct(key, "spin-free")
    out = np.zeros((n,) * 4)
    p, q, r, s = el.T
    out[p, q, r, s] = v
    out[r, s, p, q] = v
    out[q, p, s, r] = -v
    out[s, r, q, p] = -v
    return out


def spin_orbital_tensor(values, elements, norb: int) -> np.ndarray:
    """Mazziotti-order tensor D[m_i, m_j, m_k, m_l] = 2A^{ij}_{kl} (mode indices) from independent values.

    ``elements`` (K, 4, 2) are members of distinct orbits with i != j, k != l and
    {i, j} != {k, l} (e.g. :func:`spin_orbital_elements`); ``values`` (K,) are
    real (complex input must have zero imaginary parts up to round-off).
    Each value v fills its 8 entries: D[i,j,k,l] = D[j,i,l,k] = v,
    D[j,i,k,l] = D[i,j,l,k] = -v, D[k,l,i,j] = D[l,k,j,i] = -v and
    D[l,k,i,j] = D[k,l,j,i] = v. Other entries are zero. Shape (2N,)^4.
    """
    n = _check_norb(norb)
    el = _as_elements(elements, "spin_orbital", n)
    v = _real_values(values, el.shape[0])
    nm = 2 * n
    modes = _modes(el, n)
    key, zero = _so_orbit_keys(modes, nm)
    if np.any(zero):
        raise ValueError("spin-orbital elements forced to zero (i = j, k = l or {i,j} = {k,l})")
    _check_distinct(key, "spin-orbital")
    out = np.zeros((nm,) * 4)
    i, j, k, l = modes.T
    out[i, j, k, l] = v
    out[j, i, l, k] = v
    out[j, i, k, l] = -v
    out[i, j, l, k] = -v
    out[k, l, i, j] = -v
    out[l, k, j, i] = -v
    out[l, k, i, j] = v
    out[k, l, j, i] = v
    return out


def contract_spin(A_so, elements_so, norb: int) -> np.ndarray:
    """Spin-free A[p,q,r,s] = sum_{sigma tau} 2A^{(p sigma)(r tau)}_{(q sigma)(s tau)}.

    ``A_so`` (K,) holds real values on ``elements_so`` (K, 4, 2), which must cover
    every orbit of :func:`spin_orbital_elements` (extra S_z-violating elements are
    allowed and do not contribute). The (sigma, tau) terms outside the
    independent set are reached through the antisymmetries of
    :func:`spin_orbital_tensor`. Returns the real (N,)^4 pair-order tensor.
    """
    n = _check_norb(norb)
    el = _as_elements(elements_so, "spin_orbital", n)
    nm = 2 * n
    have, _ = _so_orbit_keys(_modes(el, n), nm)
    need, _ = _so_orbit_keys(_modes(spin_orbital_elements(n), n), nm)
    missing = np.setdiff1d(need, have)
    if missing.size:
        raise ValueError(f"elements_so misses {missing.size} of the {need.size} independent "
                         f"S_z-conserving elements; the spin contraction needs all of them")
    d = spin_orbital_tensor(A_so, el, n).reshape(2, n, 2, n, 2, n, 2, n)
    # axes (sigma_i, p_i, sigma_j, p_j, sigma_k, p_k, sigma_l, p_l) with i = (p, sigma), j = (r, tau),
    # k = (q, sigma), l = (s, tau): a = sigma, b = tau
    return np.einsum("apbraqbs->pqrs", d)


# ================================================================= targets
def gamma_operators(elements, mapping: QubitMapping, kind: str = "spin_orbital") -> list[PauliSum]:
    """Gamma_e as PauliSums (not Hermitian): a+_i a+_j a_l a_k, or e_{pq,rs} for ``kind="spin_free"``."""
    el = _as_elements(elements, kind, mapping.norb)
    if el.shape[0] == 0:
        return []
    if kind == "spin_orbital":
        return jw_rdm2_elements(mapping, el)
    p, q, r, s = el.T
    rows, owner = [], []
    for sig in (0, 1):
        for tau in (0, 1):
            # a+_{p sig} a+_{r tau} a_{s tau} a_{q sig}; vanishes identically for sig = tau, p = r or q = s
            idx = np.flatnonzero(~((sig == tau) & ((p == r) | (q == s))))
            blk = np.empty((idx.size, 4, 2), dtype=np.int64)
            blk[:, 0] = np.stack([p[idx], np.full(idx.size, sig)], axis=1)
            blk[:, 1] = np.stack([r[idx], np.full(idx.size, tau)], axis=1)
            blk[:, 2] = np.stack([q[idx], np.full(idx.size, sig)], axis=1)
            blk[:, 3] = np.stack([s[idx], np.full(idx.size, tau)], axis=1)
            rows.append(blk)
            owner.append(idx)
    rows_a = np.concatenate(rows)
    owner_a = np.concatenate(owner)
    ops = jw_rdm2_elements(mapping, rows_a)
    order = np.argsort(owner_a, kind="stable")
    bounds = np.searchsorted(owner_a[order], np.arange(el.shape[0] + 1))
    return [PauliSum.sum([ops[t] for t in order[bounds[e]:bounds[e + 1]]]) for e in range(el.shape[0])]


def energy_target(active, mapping: QubitMapping | None = None) -> PauliSum:
    """The qubit Hamiltonian H (``jw_hamiltonian``: real coefficients, ecore as identity); E = <H>."""
    return jw_hamiltonian(active, mapping)


def aux_targets(elements, mapping: QubitMapping, kind: str = "spin_orbital") -> list[PauliSum]:
    """Q_e = (Gamma_e - Gamma_e^dagger)/(2i) for every element: Hermitian, <Q_e> = Im <Gamma_e>.

    The coefficients are the imaginary parts of those of Gamma_e (exactly real).
    Post-processing on the auxiliary state(s) Lambda: A_hat_e = -<Q_e>_Lambda / delta,
    summed over DF fragments (:func:`aux_estimates`).
    """
    return [g.antihermitian_part() * (-1j) for g in gamma_operators(elements, mapping, kind)]


def _sz_violation(h: PauliSum, mapping: QubitMapping) -> float:
    """max |coefficient| of [H, 2 S_z] relative to the largest non-identity coefficient of H.

    2 S_z = N_alpha - N_beta = -1/2 sum_p (Z_{q(p, alpha)} - Z_{q(p, beta)}) with the qubits of ``mapping``.
    """
    n = mapping.norb
    one = np.uint64(1)
    za = [one << np.uint64(mapping.qubit(p, 0)) for p in range(n)]
    zb = [one << np.uint64(mapping.qubit(p, 1)) for p in range(n)]
    sz2 = PauliSum(mapping.n_qubits, np.zeros(2 * n, dtype=np.uint64), np.array(za + zb, dtype=np.uint64),
                   np.r_[np.full(n, -0.5), np.full(n, 0.5)])
    comm = h.commutator(sz2)
    off = (h.xs != 0) | (h.zs != 0)
    scale = float(np.max(np.abs(h.coeffs[off]))) if np.any(off) else 0.0
    if comm.n_terms == 0:
        return 0.0
    return float(np.max(np.abs(comm.coeffs))) / scale


def direct_targets(elements, active, mapping: QubitMapping | None = None, kind: str = "spin_orbital", *,
                   hamiltonian: PauliSum | None = None) -> list[PauliSum]:
    """X_e = [Gamma_e - Gamma_e^dagger, H]/2 = i [Q_e, H]: Hermitian, <X_e>_Psi = A_e for real Psi.

    Built as the commutator of the short Q_e with H; the identity term of H does
    not contribute. The coefficients are exactly real. H is
    ``jw_hamiltonian(active, mapping)`` (``mapping`` defaults to the blocked order).

    ``hamiltonian`` replaces it by a prebuilt or model qubit Hamiltonian, which must
    be built with ``mapping``; ``mapping`` is then required, because a PauliSum does
    not record its qubit order and a Hamiltonian of the other order gives wrong X_e
    without any other sign. As a consistency check, ``hamiltonian`` must conserve
    the S_z of ``mapping`` (||[H, 2 S_z]|| <= 1e-10 max |c_H|, coefficient-wise):
    this rejects a molecular Hamiltonian of the other qubit order, unless it has no
    term that changes that S_z (e.g. a diagonal H). A model Hamiltonian (e.g.
    ``jw_hamiltonian(active.with_eri(eri_model), mapping)``) gives the residual of
    the model.
    """
    if hamiltonian is not None and mapping is None:
        raise ValueError("pass the mapping the hamiltonian was built with (a PauliSum does not "
                         "record its qubit order)")
    norb = np.asarray(active.h1).shape[0]
    mapping = QubitMapping(norb) if mapping is None else mapping
    if hamiltonian is None:
        h = jw_hamiltonian(active, mapping)
    else:
        h = hamiltonian
        if mapping.norb != norb:
            raise ValueError(f"mapping has norb={mapping.norb}, active space {norb}")
        if h.n_qubits != mapping.n_qubits:
            raise ValueError(f"hamiltonian has {h.n_qubits} qubits, mapping {mapping.n_qubits}")
        viol = _sz_violation(h, mapping)
        if viol > 1e-10:
            raise ValueError(f"hamiltonian does not conserve the S_z of the {mapping.order!r} mapping "
                             f"(max |[H, 2 S_z]| / max |c_H| = {viol:.2g}); was it built with another "
                             f"qubit order?")
    out = []
    for q in aux_targets(elements, mapping, kind):
        x = q.commutator(h) * 1j
        if x.n_terms and float(np.max(np.abs(x.coeffs.imag))) > 1e-12 * float(np.max(np.abs(x.coeffs))):
            raise ValueError("H has non-real Pauli coefficients; i[Q, H] is not Hermitian")
        out.append(x.hermitian_part(rtol=0.0))
    return out


def aux_estimates(q_values, delta: float) -> np.ndarray:
    """Aux-route post-processing: A_hat_e = -sum_l <Q_e>_{Lambda_l} / delta.

    ``q_values`` holds <Q_e> on one auxiliary state, shape (K,), or on the DF
    fragment states, shape (n_states, K) (summed over the states).
    """
    delta = float(delta)
    if delta == 0 or not np.isfinite(delta):
        raise ValueError("delta must be finite and nonzero")
    q = np.asarray(q_values)
    if np.iscomplexobj(q):
        q = q.real
    if q.ndim == 2:
        q = q.sum(axis=0)
    elif q.ndim != 1:
        raise ValueError(f"q_values must have shape (K,) or (n_states, K), got {q.shape}")
    return -q / delta

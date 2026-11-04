"""Exact double factorization of anti-Hermitian CQE generators (NISQ compilation).

Conventions follow docs/conventions.md. A pair-order tensor g defines

    G = sum_pqrs g[p,q,r,s] e_{pq,rs},     e_{pq,rs} = E_pq E_rs - delta_qr E_ps,

and the CQE step is Psi <- exp(eps G) Psi with g = conj(A). For anti-Hermitian G,
:func:`factorize_generator` returns leaves for which, exactly as operators on
the whole Fock space (every particle-number sector),

    G = sum_k c_k (i Qhat_k^2) + Ohat_F                                       (1)

    Qhat_k = sum_pq Q_k[p,q] E_pq     Q_k Hermitian, tr(Q_k Q_l) = delta_kl
    c_k real, sorted by decreasing |c_k|
    F = -i sum_k c_k Q_k^2 = -f       f[p,s] = sum_q gs[p,q,q,s], F anti-Hermitian
    Ohat_F = sum_pq F[p,q] E_pq

where gs = pair_symmetrize(g).

Compilation
-----------
Each leaf is an orbital rotation plus diagonal phases. Write
Q_k = V_k diag(mu_k) V_k^dag (:meth:`GeneratorFactorization.leaf_eigh`). Let U(V)
be the orbital rotation with U(V) a+_{q sigma} U(V)^dag = sum_p V[p,q] a+_{p sigma},
that is U(V) = exp(sum_pq kappa[p,q] E_pq) for V = exp(kappa). Then

    Qhat_k = U(V_k) D_k U(V_k)^dag,    D_k = sum_p mu_kp n_p,   n_p = n_{p alpha} + n_{p beta}
    exp(i t c_k Qhat_k^2) = U(V_k) exp(i t c_k D_k^2) U(V_k)^dag

D_k^2 = sum_pq mu_kp mu_kq n_p n_q is diagonal in the occupation-number basis. It
is a diagonal-Coulomb phase, which becomes Z and ZZ rotations after
Jordan-Wigner. The one-body term is a single orbital rotation,
exp(t Ohat_F) = U(exp(t F)). :meth:`GeneratorFactorization.trotter_apply`
emulates the resulting product formula on CI vectors.

Derivation (each step is checked in tests/test_generator_factor.py)
------------------------------------------------------------------
1. Pair symmetry. Because e_{pq,rs} = e_{rs,pq}, the tensors g and
   gs[p,q,r,s] = (g[p,q,r,s] + g[r,s,p,q]) / 2 give the same operator. On the full
   Fock space the kernel of g -> G is exactly the pair-antisymmetric tensors, so
   gs is the canonical representative. The derivation never uses identities that
   hold on only one N_e-electron sector, such as
   sum_r e_{pq,rr} = (N_e - 1) E_pq. Equation (1) therefore holds on all sectors at
   once, which is what a number-conserving circuit needs.
2. Adjoint. From e_{pq,rs}^dag = e_{qp,sr}, G^dag = sum adjoint(g) e with
   adjoint(g)[p,q,r,s] = conj(g[q,p,s,r]). G is anti-Hermitian iff adjoint(gs) = -gs.
3. Normal order. G = sum gs[p,q,r,s] E_pq E_rs - sum_ps f[p,s] E_ps.
4. Hermitian basis (:func:`hermitian_basis`). The matrices b_alpha,
   alpha = 0..N^2-1, are the N diagonal units, then (E_pq + E_qp)/sqrt2 for p < q,
   then i(E_pq - E_qp)/sqrt2 for p < q (E_pq here are matrix units). They are
   orthonormal for <X, Y> = tr(X^dag Y). Each matrix unit expands as
   E_pq = sum_alpha U[(pq), alpha] b_alpha with U[(pq), alpha] = conj(b_alpha[p,q]),
   and U is unitary (N^2 x N^2). Hence
       sum_pqrs m[p,q,r,s] E_pq E_rs = sum_{alpha,beta} (U^T m U)_{alpha beta} Bhat_alpha Bhat_beta,
   where each Bhat_alpha = sum b_alpha[p,q] E_pq is a Hermitian one-body operator.
5. Coupling matrix (:func:`coupling_matrix`). Gm = U^T (i gs) U. Gm is symmetric
   because gs is pair-symmetric. It is real because adjoint(gs) = -gs; using
   conj(b[q,p]) = b[p,q] for Hermitian b,
       conj(Gm_ab) = -i sum b_a[p,q] conj(gs[p,q,r,s]) b_b[r,s]
                   = +i sum b_a[p,q] gs[q,p,s,r] b_b[r,s] = Gm_ab.
   The Hermitian part of gs maps to i times a real matrix, so ||Im Gm||_F is the
   Frobenius norm of that Hermitian part. In particular Gm is real iff G is
   anti-Hermitian.
6. Leaves. Diagonalize Gm = W diag(lam) W^T with W real orthogonal. Each
   Q_k = sum_alpha W[alpha, k] b_alpha is Hermitian, and the Q_k are orthonormal.
   Then i G = sum_k lam_k Qhat_k^2 - i Ohat_f, so c_k = -lam_k and F = -f.
   As pair tensors, gs = sum_k c_k T_k with T_k[p,q,r,s] = i Q_k[p,q] Q_k[r,s].
   The T_k are orthonormal for <X, Y> = sum conj(X) Y, so c_k = <T_k, gs> and
   sum_k c_k^2 = ||gs||_F^2. Normal-ordering each square,
   sum T_k e = i (Qhat_k^2 - (Q_k^2)hat), shows that F = -i sum_k c_k Q_k^2 = -f.

Real generators
---------------
Real states with real orbitals (the usual CQE case) give a real g. In that case
Gm couples the real symmetric basis elements (the first N(N+1)/2) only to the
imaginary antisymmetric ones: Gm = [[0, X], [X^T, 0]] with X real. This code
builds the leaves from the SVD X = sum_j sigma_j u_j v_j^T. They come in exact
conjugate pairs, stored at adjacent positions (2j, 2j+1):

    (c, Q) = (+sigma_j, Q_j)  and  (-sigma_j, conj(Q_j)),     Q_j = (S_j - i A_j) / sqrt2,

where S_j = sum_alpha u_j[alpha] b_alpha runs over the real symmetric b_alpha,
and A_j = sum_{p<q} v_j[(pq)] (E_pq - E_qp)/sqrt2 is real antisymmetric, with
||S_j||_F = ||A_j||_F = 1. Therefore ||Re Q||_F^2 = ||Im Q||_F^2 = 1/2 for every leaf. Each pair is the real anti-Hermitian operator
i sigma (Qhat^2 - conj(Q)hat^2) = sigma (Shat Ahat + Ahat Shat). The spectrum of
Gm is exactly +/- paired, with at least N zero eigenvalues, so there are at most
N(N-1) leaves (instead of N^2). The SVD route keeps the pairing exact even when
sigma_j is degenerate, for example through molecular point-group symmetry.

Truncation (projected generators)
---------------------------------
:meth:`GeneratorFactorization.pair_tensor` (``n``) returns
g_n = sum_{k<n} c_k T_k, the orthogonal projection of gs onto
span{T_0, ..., T_{n-1}}. :func:`projected_generator` returns the same tensor
for a given g. Its operator is the kept squares plus *its own* one-body term:

    sum g_n e = sum_{k<n} c_k i Qhat_k^2 + Ohat_{F_n},    F_n = -i sum_{k<n} c_k Q_k^2

(:meth:`GeneratorFactorization.one_body_for`). Note that F_n differs from F.
Take a residual generator g = conj(A). A is pair-symmetric, so gs = g, and the
slope of E(exp(eps G_n) Psi) at eps = 0 is

    dE/deps = -Re sum_pqrs g_n A = -Re <g, g_n> = -||g_n||_F^2 = -sum_{k<n} c_k^2 <= 0.

Every truncation is therefore a descent direction. The captured descent
slope_n / slope_full = captured_fraction(n) grows monotonically with n, and
n = K recovers g. Using the full F with truncated leaves would add the term
<[H, Ohat_{F - F_n}]>, which this guarantee does not cover.

For real g, an odd n splits a conjugate pair and g_n becomes complex. If n cuts
through a group of degenerate |c_k|, g_n depends on the basis the eigensolver
picks inside that group. :meth:`GeneratorFactorization.complete_group` rounds n
up to the next gap, which avoids both problems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..fci_ops import FCISpace, expm_apply, one_body

__all__ = [
    "adjoint",
    "anti_hermitian_part",
    "pair_symmetrize",
    "hermitian_basis",
    "coupling_matrix",
    "GeneratorFactorization",
    "factorize_generator",
    "projected_generator",
    "projection_compressor",
]


# --------------------------------------------------------------------------- tensors
def _check_pair_tensor(g: np.ndarray) -> int:
    g = np.asarray(g)
    if g.ndim != 4 or len(set(g.shape)) != 1:
        raise ValueError(f"expected an (N, N, N, N) pair-order tensor, got shape {g.shape}")
    return g.shape[0]


def adjoint(g: np.ndarray) -> np.ndarray:
    """Tensor of G^dag: adjoint(g)[p,q,r,s] = conj(g[q,p,s,r]), from e_{pq,rs}^dag = e_{qp,sr}."""
    _check_pair_tensor(g)
    return np.conj(np.asarray(g).transpose(1, 0, 3, 2))


def anti_hermitian_part(g: np.ndarray) -> np.ndarray:
    """(g - adjoint(g)) / 2, the tensor of the operator (G - G^dag) / 2."""
    g = np.asarray(g)
    return 0.5 * (g - adjoint(g))


def pair_symmetrize(g: np.ndarray) -> np.ndarray:
    """(g[p,q,r,s] + g[r,s,p,q]) / 2, the same operator because e_{pq,rs} = e_{rs,pq}."""
    _check_pair_tensor(g)
    g = np.asarray(g)
    return 0.5 * (g + g.transpose(2, 3, 0, 1))


def hermitian_basis(n: int) -> np.ndarray:
    """Orthonormal Hermitian basis b_alpha of the N x N matrices, shape (N^2, N, N).

    Order: the N diagonal units E_pp; then (E_pq + E_qp)/sqrt2 for p < q (row-major);
    then i (E_pq - E_qp)/sqrt2 for p < q. The first N(N+1)/2 elements are real
    symmetric and the remaining N(N-1)/2 are imaginary antisymmetric. They are
    orthonormal for <X, Y> = tr(X^dag Y).
    """
    n = int(n)
    basis = np.zeros((n * n, n, n), dtype=complex)
    a = 0
    for p in range(n):
        basis[a, p, p] = 1.0
        a += 1
    pairs = [(p, q) for p in range(n) for q in range(p + 1, n)]
    r = 1.0 / np.sqrt(2.0)
    for p, q in pairs:
        basis[a, p, q] = basis[a, q, p] = r
        a += 1
    for p, q in pairs:
        basis[a, p, q] = 1j * r
        basis[a, q, p] = -1j * r
        a += 1
    return basis


def coupling_matrix(g: np.ndarray) -> np.ndarray:
    """Gm = U^T (i g) U as a complex (N^2, N^2) matrix, with U[(pq), alpha] = conj(b_alpha[p,q]).

    Gm is linear in g and does *not* pair-symmetrize. It satisfies
    sum_pqrs (i g)[p,q,r,s] E_pq E_rs = sum_{ab} Gm[a,b] Bhat_a Bhat_b. For
    g = pair_symmetrize(g), Gm is real symmetric iff G is anti-Hermitian (see
    the module docstring). Because U and U^T are both unitary,
    ||Gm||_F = ||g||_F.
    """
    n = _check_pair_tensor(g)
    bm = hermitian_basis(n).reshape(n * n, n * n)  # bm[alpha, (pq)] = b_alpha[p, q]
    m = 1j * np.asarray(g).reshape(n * n, n * n)
    ub = np.conj(bm)  # = U^T
    return ub @ m @ ub.T


def _sector_norm(herm_eigs: np.ndarray, nelec: tuple[int, int]) -> float:
    """Spectral norm of sum_p w_p n_p on the (n_alpha, n_beta) sector (w = eigenvalues).

    The eigenvalues of D = sum_p w_p (n_{p alpha} + n_{p beta}) on the sector are
    sums of n_alpha distinct w's plus sums of n_beta distinct w's. The extreme
    values therefore come from the largest and the smallest w's.
    """
    w = np.sort(np.asarray(herm_eigs, dtype=float))
    na, nb = nelec
    m = len(w)
    top = w[m - na:].sum() + w[m - nb:].sum()
    bottom = w[:na].sum() + w[:nb].sum()
    return float(max(abs(top), abs(bottom)))


# ---------------------------------------------------------------- factorization
@dataclass
class GeneratorFactorization:
    """Leaves of G = sum_k c_k (i Qhat_k^2) + Ohat_F (see the module docstring).

    Attributes:
        norb: N, the number of spatial orbitals.
        coeffs: (K,) real c_k, sorted by decreasing |c_k|.
        Q: (K, N, N) complex Hermitian leaf matrices with tr(Q_k Q_l) = delta_kl.
        one_body: (N, N) anti-Hermitian F = -i sum_k c_k Q_k^2. It equals -f(gs)
            up to the discarded weight, so (1) holds for the operator of
            ``pair_tensor()``.
        discarded_weight: sum of the squared eigenvalues of Gm dropped by
            ``rank_tol``, i.e. ||ga - pair_tensor()||_F^2, where ga is the
            anti-Hermitian part of gs (ga = gs for anti-Hermitian input).
        is_real: True when built from a real g. Leaves (2j, 2j+1) are then
            exact conjugate pairs (c, Q), (-c, conj(Q)), and ``pair_tensor(n)``
            and ``one_body_for(n)`` are real for even n.
    """

    norb: int
    coeffs: np.ndarray
    Q: np.ndarray
    one_body: np.ndarray
    discarded_weight: float = 0.0
    is_real: bool = False

    def __post_init__(self) -> None:
        n = int(self.norb)
        self.norb = n
        self.coeffs = np.asarray(self.coeffs, dtype=float).reshape(-1)
        self.Q = np.asarray(self.Q, dtype=complex).reshape(-1, n, n)
        self.one_body = np.asarray(self.one_body)
        if self.Q.shape[0] != self.coeffs.shape[0]:
            raise ValueError("coeffs and Q must have the same number of leaves")
        if self.one_body.shape != (n, n):
            raise ValueError(f"one_body must have shape {(n, n)}")

    # ------------------------------------------------------------ bookkeeping
    @property
    def n_leaves(self) -> int:
        """K, the number of leaves kept."""
        return int(self.coeffs.shape[0])

    def _clip(self, n_leaves: int | None) -> int:
        if n_leaves is None:
            return self.n_leaves
        n = int(n_leaves)
        if n < 0:
            raise ValueError("n_leaves must be non-negative")
        return min(n, self.n_leaves)

    def weights(self) -> np.ndarray:
        """c_k^2. The Q_k have unit norm, so these sum to ||gs||_F^2 minus the discarded weight."""
        return self.coeffs ** 2

    def captured_fraction(self, n: int) -> float:
        """sum_{k<n} c_k^2 / sum_k c_k^2.

        For g = conj(A) this is the captured descent slope_n / slope_full of
        ``pair_tensor(n)``, up to the relative discarded weight. It returns 1.0
        for a zero generator.
        """
        w = self.weights()
        tot = float(w.sum())
        if tot == 0.0:
            return 1.0
        return float(w[:self._clip(n)].sum() / tot)

    def n_leaves_for(self, fraction: float) -> int:
        """Smallest n with captured_fraction(n) >= fraction (to within 1e-12)."""
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("fraction must lie in [0, 1]")
        w = self.weights()
        tot = float(w.sum())
        if tot == 0.0 or fraction == 0.0:
            return 0
        cum = np.cumsum(w) / tot
        i = int(np.searchsorted(cum, fraction - 1e-12, side="left"))
        return min(i + 1, self.n_leaves)

    def complete_group(self, n: int, rtol: float = 1e-8) -> int:
        """Smallest n' >= n that does not split a group of (near-)degenerate |c_k|.

        Leaves k-1 and k belong to the same group when
        |c_k| >= (1 - rtol) |c_{k-1}|. The projection onto the first n' leaves is
        then independent of the eigenbasis chosen inside a degenerate group.
        For real g the conjugate partners have equal |c|, so ``pair_tensor(n')``
        is real.
        """
        a = np.abs(self.coeffs)
        k = self._clip(n)
        while 0 < k < self.n_leaves and a[k] >= (1.0 - rtol) * a[k - 1]:
            k += 1
        return k

    # ---------------------------------------------------------------- leaves
    def leaf_eigh(self, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(mu, V) with Q_k = V diag(mu) V^dag (mu ascending, V unitary).

        This gives Qhat_k = U(V) (sum_p mu_p n_p) U(V)^dag with
        U(V) a+_q U(V)^dag = sum_p V[p,q] a+_p (see the module docstring).
        """
        mu, v = np.linalg.eigh(self.Q[k])
        return mu, v

    def pair_tensor(self, n_leaves: int | None = None) -> np.ndarray:
        """g_n[p,q,r,s] = sum_{k<n} c_k i Q_k[p,q] Q_k[r,s] (n = all leaves by default).

        This is the orthogonal projection of gs onto the first n leaf
        directions. It is returned as a real array when ``is_real`` and n is even.
        """
        n = self._clip(n_leaves)
        c = self.coeffs[:n]
        q = self.Q[:n]
        t = np.einsum("k,kpq,krs->pqrs", 1j * c, q, q, optimize=True)
        if self.is_real and n % 2 == 0:
            t = np.ascontiguousarray(t.real)
        return t

    def one_body_for(self, n_leaves: int | None = None) -> np.ndarray:
        """F_n = -i sum_{k<n} c_k Q_k^2, the one-body term of the operator of ``pair_tensor(n)``."""
        n = self._clip(n_leaves)
        if n == self.n_leaves:
            return self.one_body
        c = self.coeffs[:n]
        q = self.Q[:n]
        f = -1j * np.einsum("k,kpr,krs->ps", c, q, q, optimize=True)
        if self.is_real and n % 2 == 0:
            f = np.ascontiguousarray(f.real)
        return f

    # ------------------------------------------------------------- operators
    def apply(self, space: FCISpace, ci: np.ndarray, n_leaves: int | None = None) -> np.ndarray:
        """sum_{k<n} c_k i Qhat_k (Qhat_k |ci>) + Ohat_{F_n} |ci>, leaf by leaf.

        With the default n (all leaves) this is G |ci> for the factorized g.
        For smaller n it is the operator of ``pair_tensor(n)``.
        """
        self._check_space(space)
        n = self._clip(n_leaves)
        ci = np.asarray(ci).reshape(space.shape)
        out = np.asarray(one_body(space, self.one_body_for(n), ci), dtype=complex)
        for k in range(n):
            qx = one_body(space, self.Q[k], ci)
            out = out + (1j * self.coeffs[k]) * one_body(space, self.Q[k], qx)
        return out

    def trotter_apply(self, space: FCISpace, eps: float, ci: np.ndarray, order: int = 1,
                      n_leaves: int | None = None, n_steps: int = 1) -> np.ndarray:
        """Emulate the compiled product-formula circuit for exp(eps G_n) on ``ci``.

        G_n is the operator of ``pair_tensor(n)`` (n = all leaves by default).
        Its terms are X_0 = Ohat_{F_n} and X_j = c_{j-1} i Qhat_{j-1}^2 for
        j = 1..n, in order of decreasing |c|. With h = eps / n_steps, the
        circuit applies ``n_steps`` times

            order 1:  U_1(h) = exp(h X_n) ... exp(h X_1) exp(h X_0)       (X_0 acts first)
            order 2:  U_2(h) = exp(h/2 X_0) ... exp(h/2 X_{n-1}) exp(h X_n)
                               exp(h/2 X_{n-1}) ... exp(h/2 X_0)           (Strang)

        Each factor is computed exactly to the series tolerance by
        ``fci_ops.expm_apply`` on complex vectors, with the exact sector norm of
        X_j as the substep bound. The only approximation is therefore the
        splitting error: local error O(eps^2) for order 1 and O(eps^3) for
        order 2. Every factor is unitary and spin-free, so the norm and S^2 are
        preserved. Returns a complex CI array.
        """
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        self._check_space(space)
        n = self._clip(n_leaves)
        nelec = space.nelec
        terms: list[tuple[Callable[[np.ndarray], np.ndarray], float]] = []
        f = self.one_body_for(n)
        if np.any(f != 0):
            fnorm = _sector_norm(np.linalg.eigvalsh(0.5j * (f - f.conj().T)), nelec)
            terms.append((lambda x, f=f: one_body(space, f, x), fnorm))
        for k in range(n):
            qk, ck = self.Q[k], float(self.coeffs[k])
            qnorm = _sector_norm(np.linalg.eigvalsh(qk), nelec)

            def leaf(x, qk=qk, ck=ck):
                return (1j * ck) * one_body(space, qk, one_body(space, qk, x))

            terms.append((leaf, abs(ck) * qnorm ** 2))
        psi = np.asarray(ci).reshape(space.shape).astype(complex)
        if not terms:
            return psi
        h = float(eps) / n_steps

        def ex(term, t, v):
            mv, nrm = term
            return expm_apply(mv, v, t, norm_est=nrm)

        for _ in range(n_steps):
            if order == 1:
                for term in terms:
                    psi = ex(term, h, psi)
            else:
                for term in terms[:-1]:
                    psi = ex(term, 0.5 * h, psi)
                psi = ex(terms[-1], h, psi)
                for term in reversed(terms[:-1]):
                    psi = ex(term, 0.5 * h, psi)
        return psi

    def _check_space(self, space: FCISpace) -> None:
        if space.norb != self.norb:
            raise ValueError(f"space has {space.norb} orbitals, factorization has {self.norb}")


def _fix_sign(w: np.ndarray) -> np.ndarray:
    """Flip columns so that the entry of largest magnitude is positive (determinism only)."""
    if w.size == 0:
        return w
    idx = np.argmax(np.abs(w), axis=0)
    s = np.sign(w[idx, np.arange(w.shape[1])])
    s[s == 0] = 1.0
    return w * s


def factorize_generator(g: np.ndarray, rank_tol: float = 1e-12,
                        ah_tol: float = 1e-8) -> GeneratorFactorization:
    """Exact double factorization G = sum_k c_k (i Qhat_k^2) + Ohat_F of an anti-Hermitian generator.

    Args:
        g: (N, N, N, N) pair-order tensor, real or complex. Only its
            pair-symmetric part matters.
        rank_tol: drop eigenvalues of Gm (singular values of X for real g)
            with |lam| <= rank_tol * max|lam|. Their total squared weight is
            stored as ``discarded_weight``.
        ah_tol: raise ValueError if the Hermitian part of gs exceeds
            ah_tol * ||gs||_F. Below that threshold the Hermitian part is
            discarded, i.e. the anti-Hermitian part of g is factorized.

    Returns:
        GeneratorFactorization with leaves sorted by decreasing |c_k|. For
        real g the leaves are exact conjugate pairs at positions (2j, 2j+1).
    """
    g = np.asarray(g)
    n = _check_pair_tensor(g)
    gs = pair_symmetrize(g)
    ga = anti_hermitian_part(gs)
    norm = float(np.linalg.norm(gs))
    herm = float(np.linalg.norm(gs - ga))
    if herm > ah_tol * norm:
        raise ValueError(
            f"generator is not anti-Hermitian: ||Hermitian part|| / ||g|| = {herm / norm:.3e} "
            f"> {ah_tol:g}; pass anti_hermitian_part(g) to factorize its anti-Hermitian part")
    is_real = not np.iscomplexobj(g) or not np.any(np.imag(g))
    if is_real:
        ga = np.ascontiguousarray(np.real(ga))
    zero = np.zeros((n, n), dtype=float if is_real else complex)
    if norm == 0.0:
        return GeneratorFactorization(n, np.zeros(0), np.zeros((0, n, n), complex), zero,
                                      0.0, is_real)

    basis = hermitian_basis(n)
    bmat = basis.reshape(n * n, n, n)
    gm = coupling_matrix(ga).real  # imaginary part = Hermitian part of ga (~0 here)
    gm = 0.5 * (gm + gm.T)

    if is_real:
        ns = n * (n + 1) // 2
        # For real anti-Hermitian g the SS and AA blocks vanish identically.
        diag_blocks = max(np.abs(gm[:ns, :ns]).max(initial=0.0), np.abs(gm[ns:, ns:]).max(initial=0.0))
        if diag_blocks > 1e-10 * np.abs(gm).max():
            raise RuntimeError("internal error: real generator has nonzero SS/AA coupling blocks")
        x = gm[:ns, ns:]
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        smax = float(s[0]) if s.size else 0.0
        keep = s > rank_tol * smax
        discarded = 2.0 * float(np.sum(s[~keep] ** 2))
        u, s, v = u[:, keep], s[keep], vt[keep].T
        uv = _fix_sign(np.vstack([u, v]))
        u, v = uv[:ns], uv[ns:]
        # c = +sigma: W = (u, -v)/sqrt2 is the eigenvector of Gm with lam = -sigma.
        w_plus = np.vstack([u, -v]) / np.sqrt(2.0)
        q_plus = np.einsum("ak,apq->kpq", w_plus, bmat)
        kk = len(s)
        coeffs = np.empty(2 * kk)
        q = np.empty((2 * kk, n, n), dtype=complex)
        coeffs[0::2], coeffs[1::2] = s, -s
        q[0::2], q[1::2] = q_plus, np.conj(q_plus)
    else:
        lam, w = np.linalg.eigh(gm)
        lmax = float(np.max(np.abs(lam)))
        keep = np.abs(lam) > rank_tol * lmax
        discarded = float(np.sum(lam[~keep] ** 2))
        lam, w = lam[keep], _fix_sign(w[:, keep])
        order = np.argsort(-np.abs(lam), kind="stable")
        lam, w = lam[order], w[:, order]
        coeffs = -lam
        q = np.einsum("ak,apq->kpq", w, bmat)

    q = 0.5 * (q + np.conj(q.transpose(0, 2, 1)))  # exact Hermiticity
    f = -1j * np.einsum("k,kpr,krs->ps", coeffs, q, q, optimize=True)
    f = 0.5 * (f - f.conj().T)
    if is_real:
        f = np.ascontiguousarray(f.real)
    return GeneratorFactorization(n, coeffs, q, f, discarded, is_real)


def projected_generator(g: np.ndarray, n_leaves: int, complete_groups: bool = False,
                        rank_tol: float = 1e-12) -> np.ndarray:
    """Orthogonal projection g_L of pair_symmetrize(g) onto its top ``n_leaves`` leaf directions.

    g_L = sum_{k<L} c_k i Q_k (x) Q_k (Frobenius inner product on pair tensors).
    Its operator is sum_{k<L} c_k i Qhat_k^2 + Ohat_{F_L} with
    F_L = -i sum_{k<L} c_k Q_k^2. For g = conj(A) the slope is
    -Re sum(g_L * A) = -||g_L||_F^2 <= 0, and it grows in magnitude
    monotonically with L. ``n_leaves`` at or above K returns gs. If
    ``complete_groups`` is set, L is first rounded up with
    :meth:`GeneratorFactorization.complete_group`; for real g the result is
    then real.
    """
    fac = factorize_generator(g, rank_tol=rank_tol)
    n = fac._clip(n_leaves)
    if complete_groups:
        n = fac.complete_group(n)
    return fac.pair_tensor(n)


def projection_compressor(n_leaves: int | None = None, fraction: float | None = None,
                          complete_groups: bool = True, rank_tol: float = 1e-12):
    """A ``compress(g, residual_data, iteration)`` callback for :func:`thc_engine_cqe.cqe.run_cqe`.

    Each iteration keeps the top ``n_leaves`` leaves, or the fewest leaves with
    captured_fraction >= ``fraction`` (give exactly one of the two), and
    returns the projected generator. With ``complete_groups`` (the default) n
    is rounded up to the next gap in |c_k|. For real states the compressed
    generator is then real.

    The callback records ``{"iteration", "n_leaves", "n_total", "captured"}``
    per call in its ``history`` attribute.

    Note: if complete_groups is False, an odd truncation of a real generator
    splits a conjugate pair and is complex. run_cqe propagates it exactly (the
    Krylov basis is promoted to complex), so from a real start the state
    leaves the real manifold after the first such step.
    """
    if (n_leaves is None) == (fraction is None):
        raise ValueError("give exactly one of n_leaves or fraction")
    history: list[dict] = []

    def compress(g, residual_data=None, iteration=None):
        fac = factorize_generator(g, rank_tol=rank_tol)
        n = fac._clip(n_leaves) if n_leaves is not None else fac.n_leaves_for(float(fraction))
        if complete_groups:
            n = fac.complete_group(n)
        out = fac.pair_tensor(n)
        history.append({"iteration": iteration, "n_leaves": n, "n_total": fac.n_leaves,
                        "captured": fac.captured_fraction(n)})
        return out

    compress.history = history  # type: ignore[attr-defined]
    return compress

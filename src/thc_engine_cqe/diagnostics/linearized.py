"""Linearized CQE at an eigenstate: which contaminations the ACSE residual sees (M2).

Real states and integrals; e_{pq,rs}, A and the pair order are those of
docs/conventions.md. Let Psi_0 be an eigenstate of H (energy E_0) and Psi_k,
k = 1..n, other eigenstates of a sector the CQE can reach (default: all singlets
coupled to Psi_0, see :func:`linearize`). Write a normalized real state as

    Psi = c_0 Psi_0 + sum_k d_k Psi_k.

To first order in d, the ACSE residual A[p,q,r,s] = <Psi|[e_{pq,rs}, H]|Psi> is

    A = c_0 sum_k d_k w_k M_k,       w_k = E_k - E_0,
    M_k[p,q,r,s] = <Psi_0|e_{pq,rs}|Psi_k> - <Psi_0|e_{qp,sr}|Psi_k>,

where M_k is the anti-Hermitian part of the transition 2-RDM. One CQE step
Psi <- exp(eps G) Psi, with coefficients g = conj(A) + eta (eta is the error of an
estimated residual), changes the amplitudes by

    d <- d - (eps c_0 / 2) (c_0 S W d + R eta),
    S_kj = <M_k, M_j>,  W = diag(w),  (R eta)_k = <M_k, eta>,

with <X, Y> summed over all N^4 pair-order entries. The factor 1/2 comes from
<Psi_k|G|Psi_0> = sum g[p,q,r,s] <Psi_0|e_{qp,sr}|Psi_k> = -<g, M_k>/2, which
holds because g[q,p,s,r] = -g[p,q,r,s].

In the coordinates y = W^{1/2} d, in which E - E_0 = sum y^2 + O(d^3), the
noise-free map is symmetric:

    y <- (I - (eps c_0^2 / 2) K) y,   K = W^{1/2} S W^{1/2} = B^T B,
    B = [M_1 ... M_n] W^{1/2}   (N^4 x n).

The eigenvalues lambda_m >= 0 of K are the stiffnesses of the CQE modes. Under
fixed-step steepest descent a mode decays by the factor 1 - eps lambda_m / 2
per step; the map is stable only for eps < 4 / lambda_max.

A mode with lambda_m = 0 (a null vector of B) is a contamination the residual
does not see at first order: ||A|| grows only quadratically along it, and the CQE
can remove it only through higher-order terms. Observed for N2/STO-3G CAS(6,6)
at 1.098 A (tests/test_linearized.py checks the counts and the scaling):

- B has 5 null vectors among the 174 coupled singlets, and 3 of them lie in the
  HF symmetry block (17 states; ``sector="hf_block"``).
- After 60 L-BFGS steps, more than 90% of the remaining E - E_0 (about 1.8e-4 Ha)
  lies in those null modes.

K is block diagonal by spatial symmetry: transition RDMs to states of different
irreps are orthogonal. :meth:`LinearizedCQE.blocks` returns the blocks as the
connected components of the graph |K_ij| > rtol max|K|, and the block holding the
HF determinant's weight is the sector an exact CQE from HF stays in. In floating
point the exact CQE leaks out of it: round-off contamination of other irreps is
amplified by steps beyond the stability limit of their stiff modes (observed along
60 exact L-BFGS steps: weight outside the HF block up to 2e-7 for N2 and H6, 2e-4
for N2 at 2.0 A). A sector defined by a weight threshold along the trajectory
(``sector="references"``) therefore depends on round-off; the HF block does not.

Two predictions for runs with estimated residuals (``noise_floor``,
``bias_offset``):

- **Noise floor.** Fixed-step steepest descent with unbiased Gaussian errors
  eta = F z, where z ~ N(0, I) is drawn fresh at every step and F is an
  (N^4, r) matrix, reaches a stationary distribution of y. Its covariance solves

      C = P C P + Q,   P = I - (eps c_0^2/2) K,
      Q = (eps c_0/2)^2 W^{1/2} (M F)(M F)^T W^{1/2},

  and the mean energy floor is tr C. For isotropic noise of variance sigma^2 per
  orthonormal direction of the residual's symmetry subspace, Q = (eps^2 sigma^2/4) K,
  so every mode with lambda > 0 holds (eps sigma^2/4) / (1 - eps lambda/4)
  (equipartition). Null modes are neither excited nor damped.
- **Bias offset.** A fixed error b of the estimator's mean moves the fixed point
  to y* = -B^+ b (minimum-norm least squares). The energy offset ||B^+ b||^2 does
  not depend on eps, and soft modes amplify it.

The dense diagonalization limits this to CI spaces of a few thousand
determinants (H8/STO-3G: 4900).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..fci_ops import FCISpace, Hamiltonian
from ..integrals import ActiveSpace
from ..residual import trans_rdm12

__all__ = ["LinearizedCQE", "linearize", "dense_eigenstates", "structured_basis"]

# Penalty used to split spin multiplets: H + _S2_SPLIT * S^2 has joint eigenvectors of H and S^2
# unless an accidental degeneracy of the penalized operator mixes two of them.
_S2_SPLIT = 0.3711


def dense_eigenstates(active: ActiveSpace, spin_square: float | None = 0.0,
                      spin_tol: float = 1e-8, max_dim: int = 6000):
    """All eigenstates of H in the CI space (dense), optionally of one <S^2>.

    Returns ``(energies, states, s2)``: energies including ecore (ascending), the
    orthonormal real CI vectors as rows (n, n_det) and their <S^2>. Joint
    eigenvectors of H and S^2 come from diagonalizing H + 0.3711 S^2 (the
    energies are Rayleigh quotients of H itself).
    """
    from pyscf.fci import spin_op

    space = FCISpace.of(active)
    na, nb = space.shape
    dim = na * nb
    if dim > max_dim:
        raise ValueError(f"CI space of dimension {dim} exceeds max_dim={max_dim} (dense diagonalization)")
    ham = Hamiltonian(active)
    hd = np.empty((dim, dim))
    s2 = np.empty((dim, dim))
    for i in range(dim):
        v = np.zeros(dim)
        v[i] = 1.0
        hd[:, i] = np.asarray(ham.apply(v.reshape(na, nb))).ravel()
        s2[:, i] = spin_op.contract_ss(v.reshape(na, nb), active.norb, active.nelec).ravel()
    hd = 0.5 * (hd + hd.T)
    s2 = 0.5 * (s2 + s2.T)
    _, vecs = np.linalg.eigh(hd + _S2_SPLIT * s2)
    e = np.einsum("ik,ij,jk->k", vecs, hd, vecs) + active.ecore
    ss = np.einsum("ik,ij,jk->k", vecs, s2, vecs)
    keep = np.ones(dim, bool) if spin_square is None else np.abs(ss - spin_square) < spin_tol
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(e[idx], kind="stable")]
    return e[idx], np.ascontiguousarray(vecs[:, idx].T), ss[idx]


def structured_basis(norb: int) -> np.ndarray:
    """Orthonormal basis (columns, shape (N^4, N^2(N^2-1)/4)) of the real pair-order
    tensors X with the residual's symmetries X[p,q,r,s] = X[r,s,p,q] and
    X[q,p,s,r] = -X[p,q,r,s]."""
    n = norb
    seen = np.zeros((n,) * 4, bool)
    cols = []
    for p, q, r, s in np.ndindex(n, n, n, n):
        if seen[p, q, r, s]:
            continue
        orbit = {(p, q, r, s): 1.0}
        for idx, sign in (((r, s, p, q), 1.0), ((q, p, s, r), -1.0), ((s, r, q, p), -1.0)):
            if idx in orbit:
                if orbit[idx] != sign:          # the element must equal minus itself
                    orbit = None
                    break
            else:
                orbit[idx] = sign
        for idx in ((p, q, r, s), (r, s, p, q), (q, p, s, r), (s, r, q, p)):
            seen[idx] = True
        if orbit is None:
            continue
        col = np.zeros((n,) * 4)
        for idx, sign in orbit.items():
            col[idx] = sign
        cols.append(col.ravel() / np.sqrt(len(orbit)))
    return np.array(cols).T


@dataclass
class LinearizedCQE:
    """CQE linearized at the eigenstate ``psi0``; see the module docstring.

    ``states`` (rows) are the sector's other eigenstates with energies
    ``energies``; ``coupling`` holds the M_k flattened over the N^4 pair-order
    entries; ``stiffness`` (ascending) and ``modes`` (columns, y coordinates)
    are the eigen-decomposition of K.
    """

    active: ActiveSpace
    e0: float
    psi0: np.ndarray
    energies: np.ndarray
    states: np.ndarray
    coupling: np.ndarray
    stiffness: np.ndarray
    modes: np.ndarray
    sector: str
    info: dict = field(default_factory=dict)

    @property
    def gaps(self) -> np.ndarray:
        return self.energies - self.e0

    @property
    def n_states(self) -> int:
        return self.energies.size

    def kmatrix(self) -> np.ndarray:
        wh = np.sqrt(self.gaps)
        return wh[:, None] * (self.coupling @ self.coupling.T) * wh[None, :]

    def null_modes(self, rtol: float = 1e-10) -> np.ndarray:
        """Mask of the modes with lambda <= rtol * lambda_max (first-order invisible)."""
        lam_max = float(self.stiffness.max()) if self.stiffness.size else 0.0
        return self.stiffness <= rtol * lam_max

    def amplitudes(self, ci: np.ndarray) -> tuple[float, np.ndarray]:
        """(c_0, d) of the normalized real state ``ci``, with the sign fixed by c_0 >= 0."""
        v = np.asarray(ci).ravel()
        if np.iscomplexobj(v):
            if np.abs(v.imag).max() > 1e-12 * np.abs(v).max():
                raise ValueError("linearized CQE analysis is implemented for real states only")
            v = v.real
        v = v / np.linalg.norm(v)
        c0 = float(self.psi0.ravel() @ v)
        d = self.states @ v
        if c0 < 0:
            c0, d = -c0, -d
        return c0, d

    def mode_energies(self, ci: np.ndarray) -> dict:
        """Second-order energy of ``ci`` per mode: y_m^2 with y = U^T W^{1/2} d.

        Returns ``per_mode`` (aligned with ``stiffness``), ``sector`` (their sum)
        and ``outside`` = (E - E_0) - sector: higher orders and states outside
        the sector.
        """
        c0, d = self.amplitudes(ci)
        y = self.modes.T @ (np.sqrt(self.gaps) * d)
        e = Hamiltonian(self.active).energy(np.asarray(ci).reshape(self.psi0.shape).real
                                            / np.linalg.norm(ci))
        per_mode = y**2
        return {"per_mode": per_mode, "sector": float(per_mode.sum()),
                "outside": float(e - self.e0 - per_mode.sum()), "c0": c0}

    def residual_first_order(self, d: np.ndarray, c0: float = 1.0) -> np.ndarray:
        """A = c_0 sum_k d_k w_k M_k as an (N, N, N, N) pair-order tensor."""
        n = self.active.norb
        return (c0 * (np.asarray(d) * self.gaps) @ self.coupling).reshape(n, n, n, n)

    def sd_amplitude_map(self, eps: float, c0: float = 1.0) -> np.ndarray:
        """The linear map of d for one fixed-step SD step: I - (eps c_0^2/2) S W."""
        s = self.coupling @ self.coupling.T
        return np.eye(self.n_states) - 0.5 * eps * c0**2 * s * self.gaps[None, :]

    def noise_floor(self, eps: float, noise_factor: np.ndarray, c0: float = 1.0) -> dict:
        """Stationary mean of E - E_0 for fixed-step SD with Gaussian residual errors.

        ``noise_factor`` F has shape (N^4, r): the error added to the residual
        each step is F z with z ~ N(0, I_r) (for example
        ``sigma * structured_basis(N)`` for isotropic noise). Returns ``total``
        (= tr C), ``per_mode`` (stationary energy per mode), ``undriven`` (mask of
        null modes, which keep their initial energy) and ``stable``
        (eps < 4 / (c_0^2 lambda_max)). The total is ``inf`` if a null mode is
        driven or the map is unstable.
        """
        n4 = self.active.norb**4
        f = np.asarray(noise_factor, dtype=float).reshape(n4, -1)
        r = self.modes.T @ (np.sqrt(self.gaps)[:, None] * (self.coupling @ f))   # (n, r) in modes
        qdiag = (0.5 * eps * c0) ** 2 * np.einsum("ij,ij->i", r, r)
        pm = 1.0 - 0.5 * eps * c0**2 * self.stiffness
        null = self.null_modes()
        denom = 1.0 - pm**2
        per_mode = np.zeros_like(pm)
        live = ~null
        per_mode[live] = qdiag[live] / denom[live]
        scale = max(float(qdiag.max()), 1e-300) if qdiag.size else 1.0
        driven_null = null & (qdiag > 1e-12 * scale)
        stable = bool(np.all(np.abs(pm[live]) < 1.0))
        per_mode[driven_null] = np.inf
        if not stable:
            per_mode[live & (np.abs(pm) >= 1.0)] = np.inf
        return {"total": float(per_mode.sum()), "per_mode": per_mode,
                "undriven": null & ~driven_null, "stable": stable}

    def blocks(self, rtol: float = 1e-8) -> np.ndarray:
        """Block labels of K: the connected components of the graph |K_ij| > rtol max|K|.

        Labels are 0, 1, ... in the order of each block's first state. Each block is a
        spatial-symmetry sector, up to accidental degeneracies (degenerate eigenstates
        of different irreps may come out mixed and join their blocks).
        """
        from scipy.sparse.csgraph import connected_components

        if self.n_states == 0:
            return np.zeros(0, dtype=int)
        k = np.abs(self.kmatrix())
        _, lab = connected_components(k > rtol * k.max(), directed=False)
        _, first = np.unique(lab, return_index=True)
        relabel = np.empty(first.size, dtype=int)
        relabel[np.argsort(np.argsort(first))] = np.arange(first.size)
        return relabel[lab]

    def block_of(self, ci: np.ndarray, rtol: float = 1e-8, weight_rtol: float = 1e-12) -> np.ndarray:
        """Mask of the states in the block(s) of K that hold the weight of ``ci``.

        A block holds the weight if one of its states has d_k^2 > weight_rtol max d^2.
        With the HF determinant this is the HF symmetry block, the sector an exact
        CQE from HF stays in (up to amplified round-off; see the module docstring).
        """
        _, d = self.amplitudes(ci)
        w = d**2
        lab = self.blocks(rtol)
        if w.size == 0 or w.max() == 0.0:
            return np.zeros(self.n_states, bool)
        return np.isin(lab, np.unique(lab[w > weight_rtol * w.max()]))

    def subset(self, mask: np.ndarray, sector: str | None = None) -> "LinearizedCQE":
        """The linearization restricted to the states in ``mask`` (K re-diagonalized)."""
        mask = np.asarray(mask, bool)
        if mask.shape != (self.n_states,):
            raise ValueError(f"mask must have shape ({self.n_states},)")
        coupling = self.coupling[mask]
        energies = self.energies[mask]
        wh = np.sqrt(energies - self.e0)
        lam, u = np.linalg.eigh(wh[:, None] * (coupling @ coupling.T) * wh[None, :])
        info = dict(self.info, n_selected=int(mask.sum()))
        if "coupling_norms" in info:
            info["coupling_norms"] = np.asarray(info["coupling_norms"])[mask]
        return LinearizedCQE(active=self.active, e0=self.e0, psi0=self.psi0, energies=energies,
                             states=self.states[mask], coupling=coupling,
                             stiffness=np.clip(lam, 0.0, None), modes=u,
                             sector=sector or f"{self.sector}[subset]", info=info)

    def bias_offset(self, bias: np.ndarray, rcond: float = 1e-10) -> dict:
        """Energy offset of the fixed point under a residual bias b (N^4 entries):
        y* = -B^+ b, energy ||y*||^2, per mode (aligned with ``stiffness``)."""
        b = np.asarray(bias, dtype=float).ravel()
        bmat = self.coupling.T * np.sqrt(self.gaps)[None, :]       # (N^4, n)
        y = -np.linalg.pinv(bmat, rcond=rcond) @ b
        per_mode = (self.modes.T @ y) ** 2
        return {"energy": float(y @ y), "per_mode": per_mode, "y": y}


def linearize(active: ActiveSpace, *, spin_square: float | None = 0.0, root: int = 0,
              sector: str = "coupled", references=None, coupling_rtol: float = 1e-8,
              weight_tol: float = 1e-20, max_dim: int = 6000) -> LinearizedCQE:
    """Linearize the CQE at eigenstate ``root`` of the ``spin_square`` sector.

    ``sector`` chooses the excited states that enter K:

    - "coupled" (default): every state with ||M_k|| > coupling_rtol * max ||M||.
      This is the set noisy runs reach, because estimated residuals break
      spatial symmetry; spin-free generators keep S^2.
    - "all": every state of the spin sector.
    - "references": states with weight > ``weight_tol`` in at least one of the
      CI vectors ``references``, e.g. an exact CQE trajectory. In exact
      arithmetic such a trajectory stays in the symmetry sector of its start, but
      amplified round-off leaks out of it, so this set depends on ``weight_tol``
      and on round-off (module docstring).
    - "hf_block": the coupled states in the block(s) of K that hold the HF
      determinant's weight (:meth:`LinearizedCQE.block_of`, rtol ``coupling_rtol``),
      i.e. the HF symmetry sector. It does not depend on round-off.
    """
    energies, states, _ = dense_eigenstates(active, spin_square=spin_square, max_dim=max_dim)
    if root >= energies.size - 1:
        raise ValueError("root out of range")
    e0 = float(energies[root])
    others = np.array([k for k in range(energies.size) if k != root])
    if np.min(np.abs(energies[others] - e0)) < 1e-8:
        raise ValueError("the reference eigenstate is degenerate within the sector; the "
                         "linearization assumes a nondegenerate Psi_0")
    space = FCISpace.of(active)
    psi0 = states[root].reshape(space.shape)
    coupling = np.empty((others.size, active.norb**4))
    for i, k in enumerate(others):
        _, dm2 = trans_rdm12(space, psi0, states[k].reshape(space.shape))
        coupling[i] = (dm2 - dm2.transpose(1, 0, 3, 2)).ravel()
    norms = np.linalg.norm(coupling, axis=1)
    if sector in ("coupled", "hf_block"):
        sel = norms > coupling_rtol * norms.max()
    elif sector == "all":
        sel = np.ones(others.size, bool)
    elif sector == "references":
        if references is None:
            raise ValueError('sector="references" needs reference CI vectors')
        refs = [np.asarray(c).ravel() / np.linalg.norm(c) for c in references]
        w = np.max([(states[others] @ v) ** 2 for v in refs], axis=0)
        sel = w > weight_tol
    else:
        raise ValueError(f"unknown sector {sector!r}")
    coupling = coupling[sel]
    e_sel = energies[others[sel]]
    wh = np.sqrt(e_sel - e0)
    kmat = wh[:, None] * (coupling @ coupling.T) * wh[None, :]
    lam, u = np.linalg.eigh(kmat)
    lam = np.clip(lam, 0.0, None)    # K = B^T B is positive semidefinite; clip round-off
    info = {"n_sector_states": int(energies.size - 1), "n_selected": int(sel.sum()),
            "coupling_norms": norms[sel]}
    lin = LinearizedCQE(active=active, e0=e0, psi0=psi0, energies=e_sel,
                        states=states[others[sel]], coupling=coupling, stiffness=lam,
                        modes=u, sector=sector, info=info)
    if sector == "hf_block":
        lin = lin.subset(lin.block_of(space.hf(), rtol=coupling_rtol), sector="hf_block")
    return lin

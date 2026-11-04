"""Least-squares tensor hypercontraction (LS-THC) of the ERIs.

The THC model, in OpenFermion's layout (leaf ``etaPp`` with shape (M, N), central
``MPQ`` with shape (M, M)):

    (pq|rs)_THC = sum_PQ etaPp[P,p] etaPp[P,q] MPQ[P,Q] etaPp[Q,r] etaPp[Q,s].

In matrix form, with the pair-product matrix B[(pq), P] = etaPp[P,p] etaPp[P,q]
of shape (N^2, M), this reads V_THC = B MPQ B^T, where V[(pq),(rs)] = (pq|rs).

Gauge
-----
The model, the objective f below and lambda are all unchanged by the rescaling
eta_P -> c_P eta_P, MPQ_PQ -> MPQ_PQ / (c_P^2 c_Q^2) with c_P != 0.
:func:`fit_thc` fixes this freedom: it rescales the guess rows to unit norm
before the least-squares step, and again after the refinement. Its factors
therefore have |eta_P| = 1, MPQ equals OpenFermion's normalized central tensor
MPQ' = cP MPQ cP, and lambda_Z = 1/2 sum_PQ |MPQ_PQ| (all tested).

The fixed gauge matters for the refinement. V_THC is linear in MPQ and quartic
in eta, so the MPQ block of the Hessian of f scales as |eta|^8, while the eta
block scales as |MPQ|^2 |eta|^6. The raw grid guess has |eta_P| = w_P^(1/4)
|phi(r_P)|, which is much smaller than 1, and its LS central tensor is
correspondingly large. For H4 (STO-3G) at M = 4, for example, the raw leaves have
|eta_P| < 0.3 and max |MPQ| > 1e3, while max |MPQ| < 10 in the unit gauge
(tested). The two Hessian blocks then differ by a factor of roughly
|MPQ|^2 / |eta|^2, about 1e10 in that example (an estimate from this scaling,
not a test). L-BFGS-B, whose initial Hessian guess is a multiple of the
identity, then crawls. In that gauge it was observed to reach the iteration
limit far above a minimum (H4 and LiH), or to stop on its relative-reduction
test before reaching one (H4, H6). Tests check that the unit-gauge refinement
converges quickly for H4 at M = 4 and reaches f < 5e-2 for LiH at M = 6.

Fitting (:func:`fit_thc`)
-------------------------
1. Leaf guess, then rescaled to unit rows (see Gauge).
   * ``'grid'``: ISDF-style point selection. The active orbitals are evaluated
     on a PySCF DFT grid, and the values are scaled by w_g^(1/4). Pair products
     then carry sqrt(w_g), so the row inner products of the pair-product matrix
     are quadrature approximations of the pair-density overlaps
     int rho_pq rho_rs. ``nthc`` points are chosen by column-pivoted QR of that
     matrix in packed form (rows p <= q, off-diagonal rows weighted by sqrt(2)
     to reproduce the full N^2-row norms). :func:`grid_guess` returns
     etaPp[P, p] = w_P^(1/4) phi_p(r_P). After the rescaling the leaf is
     phi_p(r_P) / |phi(r_P)|, so the weights only enter the point selection.
     QR pivots are nested: the first M points for nthc = M' > M are the points
     for nthc = M.
   * ``'random'``: Gaussian etaPp (seeded). Rows are drawn in order, so the
     guess for M points is a prefix of the guess for M' > M points.
   * ``'auto'`` (default): ``'grid'`` if ``mol`` and ``mo_coeff`` are given,
     else ``'random'``.
2. LS central tensor (Parrish et al., J. Chem. Phys. 137, 224106 (2012)):
   MPQ = B^+ V (B^+)^T. This equals S^+ B^T V B S^+ with S = B^T B, but is
   computed from an SVD of B, without forming S, to avoid squaring the
   condition number. Singular values below ``rcond * s_max`` are discarded.
   B MPQ B^T = P V P, where P projects onto the retained left singular
   vectors of B (its column space when nothing is discarded). Rescaling the
   rows of eta rescales the columns of B. When no singular value is discarded,
   this leaves the column space of B, and so the LS model, unchanged (tested).
   Consequences:
   * the fit is exact once B spans the symmetric pair space, which happens for
     M = N(N+1)/2 generic points;
   * ||V - PVP||_F^2 = ||V||_F^2 - ||PVP||_F^2 for every symmetric V, because
     tr(V PVP) = tr(PVP PVP). For nested point sets, with no singular value
     discarded, ||PVP||_F cannot decrease with M, so the error cannot increase
     with M. This needs no positive semidefiniteness;
   * PVP is positive semidefinite whenever V is. LS-THC models of exact ERIs
     are therefore PSD; refined models need not be.
   All three properties are tested.
3. Optional refinement: L-BFGS-B (scipy) over (etaPp, MPQ) jointly on

       f = 1/2 ||V - V_THC||_F^2 + penalty * lambda_Z^2,
       lambda_Z = 1/2 sum_PQ |MPQ'_PQ|,

   where MPQ' is MPQ with the leaf norms absorbed (see :meth:`THCFactorization.lam`).
   This is OpenFermion's ``thc_objective`` / ``thc_objective_regularized``.
   ``penalty='auto'`` applies OpenFermion's default rule
   (``lbfgsb_opt_thc_l2reg`` with ``penalty_param=None``): penalty =
   (1/2 ||V - V_THC||_F^2) / lambda_Z, evaluated at the starting point. Here
   the starting point is the LS model; OpenFermion starts from its CP3 guess.
   The default is penalty = 0, which is an unregularized fit.
   The analytic gradient (:func:`thc_objective_and_grad`) generalizes
   OpenFermion's numpy ``thc_objective_grad`` to tensors without pair
   symmetry. It is checked against finite differences and against
   OpenFermion's formula. The final MPQ is symmetrized. For pair-symmetric V
   this cannot increase f: symmetrizing is the orthogonal projection of V_THC
   onto pair-symmetric tensors, and by the triangle inequality it does not
   increase sum |MPQ'| either. Whether L-BFGS-B met its convergence test is
   recorded in ``info['refine']['success']``. Runs that stop at ``maxiter``
   report ``False``, even when f is still decreasing slowly. Conversely, for
   f < 1 L-BFGS-B's relative-reduction test (``ftol``) bounds the absolute
   decrease of f. A run that approaches an exact fit (f -> 0, possible once
   the model has about as many parameters as V has unique elements) can
   therefore report success while f is still decreasing. With a penalty, f is
   not differentiable where an element of MPQ' vanishes, and L-BFGS-B often
   ends with an abnormal line-search termination (``success=False``) after f
   has decreased. OpenFermion's regularized fit uses the same objective and
   optimizer.

Lambda (:meth:`THCFactorization.lam`) is OpenFermion's ``thc.compute_lambda``
(Lee et al., PRX Quantum 2, 030305 (2021), Sec. II.D):
lambda = sum_k |eig_k(T')| + 1/2 sum_PQ |MPQ'_PQ|, with T' from the exact ERIs
by default (``use_eri_thc_for_t=False``).
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.optimize

from .eri_factor import ERIFactor, LambdaBreakdown, check_eri, lambda_one_body

__all__ = [
    "THCFactorization",
    "thc_eri",
    "ls_central",
    "grid_guess",
    "random_guess",
    "thc_objective_and_grad",
    "fit_thc",
    "fit_thc_active",
]


# ---------------------------------------------------------------- primitives
def _pair_products(etaPp: np.ndarray) -> np.ndarray:
    """C[P, p, q] = etaPp[P, p] etaPp[P, q]."""
    return etaPp[:, :, None] * etaPp[:, None, :]


def thc_eri(etaPp: np.ndarray, MPQ: np.ndarray) -> np.ndarray:
    """(pq|rs)_THC = sum_PQ eta[P,p] eta[P,q] M[P,Q] eta[Q,r] eta[Q,s]."""
    etaPp = np.asarray(etaPp, dtype=float)
    m, n = etaPp.shape
    c = _pair_products(etaPp).reshape(m, n * n)
    return (c.T @ np.asarray(MPQ, dtype=float) @ c).reshape(n, n, n, n)


def _lambda_z(etaPp: np.ndarray, MPQ: np.ndarray) -> tuple[float, np.ndarray]:
    """(lambda_Z, MPQ') with MPQ'_PQ = |eta_P|^2 MPQ_PQ |eta_Q|^2 (OpenFermion's cP M cP)."""
    n2 = np.einsum("Pp,Pp->P", etaPp, etaPp)
    mn = n2[:, None] * MPQ * n2[None, :]
    return 0.5 * float(np.sum(np.abs(mn))), mn


def _row_scales(etaPp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(|eta_P|, c_P) with c_P = |eta_P|, or 1 for a row of zero norm (left unchanged)."""
    nrm = np.linalg.norm(etaPp, axis=1)
    return nrm, np.where(nrm > 0, nrm, 1.0)


def _unit_rows(etaPp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(eta_P / |eta_P|, |eta_P|). Rows of zero norm are left unchanged."""
    nrm, c = _row_scales(etaPp)
    return etaPp / c[:, None], nrm


def _to_unit_gauge(etaPp: np.ndarray, MPQ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rescale to unit-norm leaf rows; MPQ_PQ -> c_P^2 MPQ_PQ c_Q^2 keeps the model.

    The result's MPQ is exactly symmetric whenever the input MPQ is, because
    each element is multiplied by the commutative product c_P^2 c_Q^2.
    """
    _, c = _row_scales(etaPp)
    c2 = c ** 2
    return etaPp / c[:, None], MPQ * np.outer(c2, c2)


def ls_central(eri: np.ndarray, etaPp: np.ndarray, rcond: float = 1e-12) -> tuple[np.ndarray, dict]:
    """Least-squares central tensor MPQ = B^+ V (B^+)^T for a fixed leaf tensor.

    ``rcond`` is relative to the largest singular value of B (N^2 x M). Returns
    ``(MPQ, info)``, where ``info`` has the singular values of B, the number kept
    (``rank``), and the resulting Frobenius error ``eri_error_fro``.
    """
    eri = check_eri(eri)
    etaPp = np.asarray(etaPp, dtype=float)
    m, n = etaPp.shape
    if eri.shape[0] != n:
        raise ValueError(f"etaPp has {n} orbitals, eri {eri.shape[0]}")
    b = _pair_products(etaPp).reshape(m, n * n).T  # (N^2, M)
    u, s, wt = np.linalg.svd(b, full_matrices=False)
    keep = s > rcond * (s[0] if s.size else 0.0)
    binv = (wt[keep].T / s[keep]) @ u[:, keep].T  # (M, N^2)
    v = eri.reshape(n * n, n * n)
    z = binv @ v @ binv.T
    z = 0.5 * (z + z.T)
    err = float(np.linalg.norm(v - b @ z @ b.T))
    return z, {"singular_values": s, "rank": int(keep.sum()), "rcond": rcond,
               "eri_error_fro": err}


def random_guess(norb: int, nthc: int, seed: int = 0) -> np.ndarray:
    """Gaussian etaPp (nthc, norb); the first M rows do not depend on nthc >= M."""
    return np.random.default_rng(seed).standard_normal((nthc, norb))


def grid_guess(mol, mo_coeff: np.ndarray, nthc: int, grid_level: int = 3) -> tuple[np.ndarray, dict]:
    """ISDF-style leaf guess from a PySCF DFT grid (see the module docstring).

    Args:
        mol: ``pyscf.gto.Mole`` the orbitals are expanded in.
        mo_coeff: (n_ao, N) active-orbital coefficients.
        nthc: number of THC points M.
        grid_level: ``pyscf.dft.gen_grid.Grids.level``.

    Returns:
        (etaPp, info) with etaPp[P, p] = w_P^(1/4) phi_p(r_P), shape (M, N).
        ``info`` has the selected ``coords``, ``weights`` and ``grid_indices``,
        plus ``n_grid``. Negative quadrature weights (if any) are clipped to 0.
        Only the first N(N+1)/2 pivots come from the QR criterion. Any
        further points are the next entries of LAPACK's permutation, which
        are distinct grid points but arbitrary ones.
    """
    from pyscf.dft import gen_grid, numint

    mo_coeff = np.asarray(mo_coeff, dtype=float)
    if mo_coeff.ndim != 2 or mo_coeff.shape[0] != mol.nao_nr():
        raise ValueError(f"mo_coeff must have shape (n_ao={mol.nao_nr()}, N), got {mo_coeff.shape}")
    n = mo_coeff.shape[1]
    grids = gen_grid.Grids(mol)
    grids.level = grid_level
    grids.verbose = 0
    grids.build(with_non0tab=False)
    coords = np.asarray(grids.coords)
    w = np.clip(np.asarray(grids.weights), 0.0, None)
    if nthc > coords.shape[0]:
        raise ValueError(f"nthc={nthc} exceeds the number of grid points {coords.shape[0]}")
    phi = numint.eval_ao(mol, coords) @ mo_coeff  # (n_grid, N)
    phiw = phi * (w ** 0.25)[:, None]
    p, q = np.triu_indices(n)
    fac = np.where(p == q, 1.0, np.sqrt(2.0))
    pair = (phiw[:, p] * phiw[:, q] * fac).T  # (npair, n_grid)
    _, _, piv = scipy.linalg.qr(pair, mode="economic", pivoting=True)
    sel = np.asarray(piv[:nthc])
    eta = np.ascontiguousarray(phiw[sel])
    return eta, {"coords": coords[sel], "weights": w[sel], "grid_indices": sel,
                 "n_grid": int(coords.shape[0]), "grid_level": grid_level}


def thc_objective_and_grad(x: np.ndarray, eri: np.ndarray, nthc: int,
                           penalty: float = 0.0) -> tuple[float, np.ndarray]:
    """THC objective and its gradient (OpenFermion's parameter layout).

    x = concat(etaPp.ravel(), MPQ.ravel()), etaPp of shape (M, N), MPQ of shape (M, M).

        f = 1/2 sum_pqrs (V - V_THC)_pqrs^2 + penalty * lambda_Z^2,
        lambda_Z = 1/2 sum_PQ n_P |MPQ_PQ| n_Q,   n_P = sum_p etaPp[P,p]^2.

    With D = V - V_THC, C_P = eta_P eta_P^T, Y1[(pq),G] = sum D[(pq),(rs)] C_V[rs] M[G,V]
    and Y2[(rs),G] = sum D[(pq),(rs)] C_U[pq] M[U,G], Y = Y1 + Y2:

        df/dM_AB   = -sum_pqrs D_pqrs C_A[pq] C_B[rs]
        df/deta_GT = -sum_q Y[T,q,G] eta_Gq - sum_p Y[p,T,G] eta_Gp

    For 8-fold-symmetric V this equals OpenFermion's ``thc_objective_grad``.
    The penalty gradient uses sign(0) = 0 at the kinks of |MPQ|.
    """
    eri = np.asarray(eri, dtype=float)
    n = eri.shape[0]
    m = int(nthc)
    x = np.asarray(x, dtype=float)
    eta = x[: m * n].reshape(m, n)
    z = x[m * n: m * n + m * m].reshape(m, m)
    v = eri.reshape(n * n, n * n)
    c = _pair_products(eta).reshape(m, n * n)
    d = v - c.T @ z @ c
    f = 0.5 * float(np.sum(d * d))
    dc = d @ c.T  # (N^2, M): sum_rs D[(pq),(rs)] C_V[rs]
    ctd = c @ d   # (M, N^2): sum_pq C_U[pq] D[(pq),(rs)]
    g_z = -(ctd @ c.T)
    y = (dc @ z.T + ctd.T @ z).reshape(n, n, m)
    g_eta = -(np.einsum("TqG,Gq->GT", y, eta) + np.einsum("pTG,Gp->GT", y, eta))
    if penalty:
        n2 = np.einsum("Pp,Pp->P", eta, eta)
        az = np.abs(z)
        lam_z = 0.5 * float(n2 @ az @ n2)
        f += penalty * lam_z ** 2
        pref = 2.0 * penalty * lam_z
        g_z = g_z + pref * 0.5 * np.sign(z) * np.outer(n2, n2)
        g_eta = g_eta + pref * eta * (az @ n2 + az.T @ n2)[:, None]
    return f, np.concatenate([g_eta.ravel(), g_z.ravel()])


# ------------------------------------------------------------- factorization
class THCFactorization(ERIFactor):
    """THC factors in OpenFermion's layout: etaPp (M, N), MPQ (M, M).

    Any gauge is accepted (see the module docstring); :func:`fit_thc` returns
    unit-norm leaf rows. ``n_params`` = M N + M (M + 1) / 2 when MPQ is
    symmetric (always, after :func:`fit_thc`), M N + M^2 otherwise. ``info``
    holds fit diagnostics.
    """

    kind = "thc"

    def __init__(self, etaPp: np.ndarray, MPQ: np.ndarray, info: dict | None = None):
        eta = np.asarray(etaPp, dtype=float)
        z = np.asarray(MPQ, dtype=float)
        if eta.ndim != 2 or z.shape != (eta.shape[0], eta.shape[0]):
            raise ValueError(f"need etaPp (M, N) and MPQ (M, M), got {eta.shape}, {z.shape}")
        self.etaPp = eta
        self.MPQ = z
        self.norb = eta.shape[1]
        self.info = dict(info or {})

    @property
    def nthc(self) -> int:
        return self.etaPp.shape[0]

    def to_eri(self) -> np.ndarray:
        return thc_eri(self.etaPp, self.MPQ)

    def n_params(self) -> int:
        m, n = self.etaPp.shape
        sym = np.allclose(self.MPQ, self.MPQ.T, rtol=0.0, atol=1e-14 * max(1.0, np.abs(self.MPQ).max()))
        return m * n + (m * (m + 1) // 2 if sym else m * m)

    def x(self) -> np.ndarray:
        """Flat parameter vector concat(etaPp.ravel(), MPQ.ravel())."""
        return np.concatenate([self.etaPp.ravel(), self.MPQ.ravel()])

    def objective(self, eri: np.ndarray, penalty: float = 0.0) -> float:
        """The fitting objective f of :func:`thc_objective_and_grad` at these factors."""
        return thc_objective_and_grad(self.x(), eri, self.nthc, penalty)[0]

    def normalized(self) -> tuple[np.ndarray, np.ndarray]:
        """(eta_hat, MPQ'): unit-norm leaf rows; MPQ'_PQ = |eta_P|^2 MPQ_PQ |eta_Q|^2."""
        nrm = np.linalg.norm(self.etaPp, axis=1)
        safe = np.where(nrm > 0, nrm, 1.0)
        return self.etaPp / safe[:, None], _lambda_z(self.etaPp, self.MPQ)[1]

    def lam(self, h1: np.ndarray, eri_exact: np.ndarray | None = None, *,
            use_eri_thc_for_t: bool = False) -> LambdaBreakdown:
        """OpenFermion's THC lambda: sum_k |eig_k(T')| + 1/2 sum_PQ |MPQ'_PQ|.

        MPQ' absorbs the leaf norms (OpenFermion's ``cP.dot(MPQ).dot(cP)``
        with cP = diag(etaPp etaPp^T)), which makes lambda invariant under
        rescaling of the leaves. T' is

        * ``eri_exact`` given (default use): h - 1/2 sum_r (pr|rq) + sum_r (rr|pq)
          with the exact ERIs (OpenFermion's ``use_eri_thc_for_t=False``);
        * ``use_eri_thc_for_t=True``: the Coulomb term sum_r (rr|pq) is taken
          from the THC ERIs, as in OpenFermion's option of that name;
        * ``eri_exact=None``: both terms from the THC ERIs.

        ``details`` holds ``lambda_T``, ``lambda_z``, ``nthc``, ``t_eigenvalues``
        and ``t_eri``. If ``eri_exact`` is given it also holds
        ``ls_residual`` = 1/2 ||V - V_THC||_F^2 (OpenFermion's ``res``) and
        ``eri_error_fro`` = ||V - V_THC||_F.
        """
        eri_thc = self.to_eri()
        lam_z, _ = _lambda_z(self.etaPp, self.MPQ)
        details: dict = {"nthc": self.nthc, "lambda_z": lam_z}
        if eri_exact is None:
            lam_t, t = lambda_one_body(h1, eri_thc)
            details["t_eri"] = "factorized"
        else:
            eri_exact = check_eri(eri_exact, "eri_exact")
            if use_eri_thc_for_t:
                lam_t, t = lambda_one_body(h1, eri_exact, eri_coulomb=eri_thc)
                details["t_eri"] = "exact exchange, THC Coulomb"
            else:
                lam_t, t = lambda_one_body(h1, eri_exact)
                details["t_eri"] = "exact"
            deri = eri_exact - eri_thc
            details["ls_residual"] = 0.5 * float(np.sum(deri ** 2))
            details["eri_error_fro"] = float(np.linalg.norm(deri.ravel()))
        details["lambda_T"] = lam_t
        details["t_eigenvalues"] = t
        return LambdaBreakdown(lam_total=lam_t + lam_z, lam_one_body=lam_t,
                               lam_two_body=lam_z, details=details)


def fit_thc(eri: np.ndarray, nthc: int, *, guess: str = "auto", mol=None,
            mo_coeff: np.ndarray | None = None, refine: bool = True,
            penalty: float | str = 0.0, maxiter: int = 5000, seed: int = 0,
            h1: np.ndarray | None = None, rcond: float = 1e-12, grid_level: int = 3,
            lbfgs_options: dict | None = None) -> THCFactorization:
    """Fit a THC model to (pq|rs). The procedure is described in the module docstring.

    Args:
        eri: (N, N, N, N) real ERIs (8-fold symmetric).
        nthc: number of THC points M.
        guess: ``'grid'``, ``'random'`` or ``'auto'``. ``'auto'`` means grid
            when ``mol`` and ``mo_coeff`` are given, random otherwise.
        mol, mo_coeff: PySCF molecule and (n_ao, N) active orbitals (grid guess).
        refine: run L-BFGS-B on (etaPp, MPQ) after the LS fit.
        penalty: weight of lambda_Z^2 in the refinement objective, a float >= 0
            or ``'auto'`` (OpenFermion's rule, module docstring). The default 0
            is an unregularized fit.
        maxiter: L-BFGS-B iteration limit.
        seed: seed for the random guess.
        h1: optional one-body integrals. If given, ``info`` also records the
            total lambda before and after refinement (T' from ``eri``).
        rcond: relative singular-value cutoff of the LS pseudo-inverse, applied
            to the pair-product matrix of the unit-norm leaves.
        grid_level: PySCF grid level of the grid guess.
        lbfgs_options: extra ``scipy.optimize.minimize`` options. The defaults,
            ftol=1e-15 and gtol=1e-10, are tight: the run ends at ``maxiter`` or
            when the relative decrease of f reaches machine precision.

    Returns:
        :class:`THCFactorization` with unit-norm leaf rows (module docstring,
        Gauge). Its ``info`` dict has the keys ``nthc``, ``guess``, ``rcond``,
        ``penalty`` (the value used, a float), ``penalty_rule`` (``'auto'`` or
        ``'fixed'``), ``guess_row_norms`` (the row norms of the guess before
        rescaling), ``seed`` or ``grid`` (selected points), ``ls`` (rank,
        singular values and ERI error of the LS step), ``lambda_z_ls``,
        ``eri_error_fro`` and ``lambda_z`` of the returned factors, and
        ``lambda_ls`` / ``lambda`` when ``h1`` is given. With ``refine`` it also
        has ``refine`` (L-BFGS-B success, message, nit, nfev, f_initial, f_final).
    """
    eri = check_eri(eri)
    n = eri.shape[0]
    nthc = int(nthc)
    if nthc < 1:
        raise ValueError("nthc must be positive")
    if isinstance(penalty, str):
        if penalty != "auto":
            raise ValueError("penalty must be a non-negative float or 'auto'")
        penalty_rule = "auto"
    else:
        penalty = float(penalty)
        if not penalty >= 0.0:
            raise ValueError("penalty must be a non-negative float or 'auto'")
        penalty_rule = "fixed"
    if guess == "auto":
        guess = "grid" if (mol is not None and mo_coeff is not None) else "random"
    info: dict = {"nthc": nthc, "guess": guess, "rcond": rcond}
    if guess == "grid":
        if mol is None or mo_coeff is None:
            raise ValueError("the grid guess needs mol and mo_coeff")
        mo_coeff = np.asarray(mo_coeff, dtype=float)
        if mo_coeff.shape[1] != n:
            raise ValueError(f"mo_coeff has {mo_coeff.shape[1]} orbitals, eri {n}")
        eta, ginfo = grid_guess(mol, mo_coeff, nthc, grid_level=grid_level)
        info["grid"] = ginfo
    elif guess == "random":
        eta = random_guess(n, nthc, seed)
        info["seed"] = seed
    else:
        raise ValueError("guess must be 'auto', 'grid' or 'random'")
    eta, info["guess_row_norms"] = _unit_rows(eta)

    z, lsinfo = ls_central(eri, eta, rcond=rcond)
    info["ls"] = {"rank": lsinfo["rank"], "eri_error_fro": lsinfo["eri_error_fro"],
                  "singular_values": lsinfo["singular_values"]}
    fac = THCFactorization(eta, z)
    if h1 is not None:
        info["lambda_ls"] = fac.lam(h1, eri).lam_total
    info["lambda_z_ls"] = _lambda_z(eta, z)[0]
    if penalty_rule == "auto":  # OpenFermion: residual / lambda_Z at the starting point
        res_ls = fac.objective(eri, 0.0)
        penalty = res_ls / info["lambda_z_ls"] if info["lambda_z_ls"] > 0 else 0.0
    info["penalty"] = float(penalty)
    info["penalty_rule"] = penalty_rule

    if refine:
        x0 = fac.x()
        f0 = thc_objective_and_grad(x0, eri, nthc, penalty)[0]
        opts = {"maxiter": int(maxiter), "ftol": 1e-15, "gtol": 1e-10,
                "maxfun": max(15000, 2 * int(maxiter))}
        opts.update(lbfgs_options or {})
        res = scipy.optimize.minimize(thc_objective_and_grad, x0, args=(eri, nthc, penalty),
                                      jac=True, method="L-BFGS-B", options=opts)
        x = res.x
        z = x[nthc * n:].reshape(nthc, nthc)
        eta, z = _to_unit_gauge(x[: nthc * n].reshape(nthc, n), 0.5 * (z + z.T))
        fac = THCFactorization(eta, z)
        info["refine"] = {"success": bool(res.success), "message": str(res.message),
                          "nit": int(res.nit), "nfev": int(res.nfev), "f_initial": float(f0),
                          "f_final": fac.objective(eri, penalty)}
    info["eri_error_fro"] = float(np.linalg.norm((eri - fac.to_eri()).ravel()))
    info["lambda_z"] = _lambda_z(fac.etaPp, fac.MPQ)[0]
    if h1 is not None:
        info["lambda"] = fac.lam(h1, eri).lam_total
    fac.info = info
    return fac


def fit_thc_active(active, nthc: int, **kwargs) -> THCFactorization:
    """:func:`fit_thc` with eri, mol, mo_coeff and h1 taken from an ActiveSpace."""
    kw = {"mol": active.mol, "mo_coeff": active.mo_coeff, "h1": active.h1}
    kw.update(kwargs)
    return fit_thc(active.eri, nthc, **kw)

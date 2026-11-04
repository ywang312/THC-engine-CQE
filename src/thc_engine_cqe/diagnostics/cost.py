"""CNOT-count ESTIMATES for compiling CQE generators on NISQ hardware.

This module is a cost *model*. It does not transpile circuits for real
hardware. Each count below is one of three things, and each is labeled:

* [verified] An explicit circuit in this module realizes the count.
  tests/test_generator_rank.py simulates that circuit and compares it with the
  exact fermionic unitary from the Jordan-Wigner oracle
  (:mod:`thc_engine_cqe.testing.fock`). For two-qubit gates the count is the
  minimal one, which the test checks with the Shende-Bullock-Markov
  criterion (:func:`two_qubit_cnot_count`).
* [published] Taken from the literature and not reproduced here.
* [estimate] A counting assumption, stated where it is used.

Qubit layout
------------
Jordan-Wigner with the repo's blocked spin order (as in ``testing.fock``):
spin orbital (p, alpha) is qubit p, (p, beta) is qubit N + p, and
a_j = Z_0 ... Z_{j-1} sigma^-_j. Mode indices below are these qubit indices.

Elementwise compilation (:func:`cnot_cost_elementwise`)
-------------------------------------------------------
:func:`spin_orbital_excitations` expands a spin-free generator
G = sum g[p,q,r,s] e_{pq,rs} exactly into normal-ordered spin-orbital terms

    G = sum_X [ t_X tau_X - conj(t_X) tau_X^dag ] + sum_Y T_Y n_{y0} n_{y1},
    tau(C, D) = a+_{c0} a+_{c1} a_{d1} a_{d0}      (c0 < c1, d0 < d1),

and merges every term that is the same operator. The cost assumes one
exponential per excitation, i.e. a first-order product formula.
generator_rank.compressed_cqe emulates the exact exponential of the kept sum,
so the elementwise Trotter error is not part of its energies. Kinds and costs:

    kind       operator                          CNOTs ('all')
    'double'   |C u D| = 4                       13 + 2|S|   (core [published], parity [verified])
    'single'   n_c (a+_x a_y - h.c.), |C n D| = 1    6 + 2|S|  [verified]
    'diagonal' T n n with T imaginary, C = D     2           [verified]

S is the set of qubits on the Jordan-Wigner strings: for sorted modes
m1 < m2 < m3 < m4 of a double, S = (m1, m2) u (m3, m4) (open intervals). For a
single, S is the open interval between x and y minus the control c, because
n_c Z_c = -n_c. The parity of S is computed into one qubit of S with |S| - 1
CNOTs, and uncomputed with the same number. Its sign enters through a CZ on
an excitation qubit before and after the core, since
Z_x (tau - tau^dag) Z_x = -(tau - tau^dag). That adds 2 CNOT-equivalents, so
the overhead is 2|S| for |S| >= 1 and 0 otherwise. Complex amplitudes need only
single-qubit phase gates.

Core counts:

* double: Yordanov et al. [Y20] report a 13-CNOT double qubit excitation
  [published]. The explicit circuit here
  (:func:`excitation_circuit`) uses 3 + 8 + 3 = 14: a three-CNOT basis map,
  a three-fold uniformly controlled R_y with 2^3 CNOTs [MVBS04], and the
  inverse map. With ``double_core=14`` every 'all' count is [verified].
* single (number-conditioned): 1 + 4 + 1 = 6, a doubly controlled R_y
  between two CNOTs [verified]. Two-body spin-free tensors never produce
  unconditioned one-body excitations, because e_{pq,rs} is normal ordered.
* diagonal: exp(i phi n_P n_Q) is a controlled phase, 2 CNOTs [verified].

'linear' (nearest-neighbour line) [estimate]: the k modes of an excitation are
first gathered into a contiguous block with fermionic swaps and afterwards
scattered back. The gathering uses the minimal number n_g of adjacent swaps
that preserve their order (:func:`gather_swaps`). An fSWAP costs 2 CNOTs (it
equals iSWAP up to local phases [verified]), so routing costs 4 n_g CNOTs. A
gathered excitation has no Jordan-Wigner string.

Inside the block, the explicit double circuit needs its two annihilators
adjacent and its two creators adjacent, i.e. the role pattern AACC or CCAA
along the line. Reaching that pattern takes n_o adjacent fSWAPs: n_o = 0 for
AACC/CCAA, 1 for ACAC/CACA (e.g. every opposite-spin double i_a a_a j_b b_b in
the blocked layout), and 2 for ACCA/CAAC. With that pattern, the basis map is
nearest-neighbour when the rotation target is the inner annihilator. The two
multiplexor CNOTs from the outer creator are then next-nearest-neighbour, and
each is bridged by 4 nearest-neighbour CNOTs instead of 1, adding 6. This
derivation is for the explicit construction; it is applied to the published
13-CNOT core as well. A single needs its control at an end of the block, which
costs one more fSWAP pair (+4) when the control lands between x and y:

    'double'   13 + 6 + 4 (n_g + n_o)    'single'   6 + 4 n_g (+ 4)    'diagonal'   2 + 4 n_g

Leaf compilation (:func:`cnot_cost_leaves`)
-------------------------------------------
:mod:`thc_engine_cqe.factor.generator` writes a generator as
G = sum_k c_k i Qhat_k^2 + Ohat_F. Each leaf is
exp(i t c_k Qhat_k^2) = U(V_k) exp(i t c_k D_k^2) U(V_k)^dag (verified there),
an orbital rotation plus a diagonal-Coulomb layer [M21]. A product-formula
step over L leaves applies the diagonal layers listed by :func:`leaf_schedule`:

    order 1:  leaves 0, 1, ..., L-1                                     (L layers)
    order 2:  leaves 0, ..., L-2 (t/2), L-1 (t), L-2, ..., 0 (t/2)   (2L - 1 layers)

Order 2 is the Strang product of GeneratorFactorization.trotter_apply, in
which the middle leaf L-1 acts once for the full time. Between two layers, the
rotations merge into one (U(V_b)^dag U(V_a) is a single orbital rotation). The
one-body factors exp(t Ohat_F) (order 1) or exp(t/2 Ohat_F) at both ends
(order 2) are rotations too, so they merge with the first and the last
rotation. A step therefore has one rotation before each layer and one final
un-rotation:

    order 1:  (L + 1) C_rot  + sum_{k<L} C_diag(k)
    order 2:  2L C_rot       + 2 sum_{k<L-1} C_diag(k) + C_diag(L-1)

Both orders cost the same for L = 1. For L >= 2, order 2 adds L - 1 rotations
and repeats the layers of leaves 0..L-2. tests/test_generator_rank.py builds
this merged circuit for both orders and checks that it reproduces the emulated
product formula (diagnostics.generator_rank.TrotterStep) with exactly these
layer counts [verified]. The per-layer costs C_rot and C_diag are the model
below.

Along a CQE trajectory the final un-rotation of one step merges with the first
rotation of the next. The cumulative cost of T steps is therefore the sum of
the per-step costs without the final un-rotation, plus C_rot once.

* Orbital rotation, both spins (U(V) = U_alpha(V) U_beta(V)):
  C_rot = 2 * [N(N-1)/2 Givens rotations] * 2 CNOTs = 2 N (N - 1). The
  N(N-1)/2 nearest-neighbour Givens rotations (any N x N unitary, plus
  diagonal phases) are from [K18]. The 2 CNOTs per Givens rotation are
  [verified]; they also hold with a complex phase. The blocked layout keeps
  each spin's N modes contiguous, so 'linear' costs the same as 'all'.
* Diagonal layer: D_k^2 = (sum_P mu_P n_P)^2 couples every pair of the
  2 n_a spin orbitals whose eigenvalue mu_p of Q_k is nonzero
  (|mu| > mu_tol max|mu|, default 1e-6, so that eigenvalue noise is not
  counted). That gives P_a = n_a (2 n_a - 1) ZZ terms. The leaves of the RHF
  generator live on the ov block, so their eigenvalues come in pairs +-s and
  n_a <= 2 min(n_occ, n_vir) (tested).

  - 'all': C_diag = 2 P_a, one CNOT-Rz-CNOT per ZZ term [verified that a
    ZZ rotation needs 2]. [estimate] Phase-polynomial synthesis can do better.
  - 'linear': fermionic swap network [K18]. On the full 2N-qubit line it has
    N(2N-1) gates. P_a of them are fSWAP times a ZZ phase, 3 CNOTs each, and
    the rest are bare fSWAPs, 2 CNOTs each [verified]. The network reverses the
    line, which keeps both spin blocks contiguous. Alternatively, the network
    acts on the contiguous run of the 2 n_a active spin orbitals; the
    preceding rotation can place them at the inner ends of the two spin blocks.
    It is then followed by a bare reversal (P_a fSWAPs). The model takes the
    cheaper option:
    C_diag = min(3 P_a + 2 (N(2N-1) - P_a), 5 P_a).

Sources
-------
[Y20]   Y. S. Yordanov, D. R. M. Arvidsson-Shukur, C. H. W. Barnes,
        Phys. Rev. A 102, 062612 (2020).
[K18]   I. D. Kivlichan et al., Phys. Rev. Lett. 120, 110501 (2018).
[M21]   M. Motta et al., npj Quantum Inf. 7, 83 (2021).
[MVBS04] M. Mottonen, J. J. Vartiainen, V. Bergholm, M. M. Salomaa,
        Phys. Rev. Lett. 93, 130502 (2004) (uniformly controlled rotations).
[SBM04] V. V. Shende, S. S. Bullock, I. L. Markov, Phys. Rev. A 69, 062321 (2004).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "CONNECTIVITIES",
    "GIVENS_CNOTS",
    "FSWAP_CNOTS",
    "ZZ_CNOTS",
    "FSWAP_ZZ_CNOTS",
    "DOUBLE_CORE_PUBLISHED",
    "Excitation",
    "spin_orbital_excitations",
    "parity_length",
    "gather_swaps",
    "role_swaps",
    "excitation_cnots",
    "cnot_cost_elementwise",
    "elementwise_breakdown",
    "orbital_rotation_cnots",
    "diagonal_layer_cnots",
    "leaf_active_orbitals",
    "leaf_schedule",
    "cnot_cost_leaves",
    "two_qubit_cnot_count",
    "excitation_circuit",
    "circuit_cnots",
    "circuit_unitary",
]

CONNECTIVITIES = ("all", "linear")

GIVENS_CNOTS = 2        # [verified] phased Givens rotation on adjacent JW qubits
FSWAP_CNOTS = 2         # [verified] fermionic swap = iSWAP up to local phases
ZZ_CNOTS = 2            # [verified] exp(i theta Z Z)
FSWAP_ZZ_CNOTS = 3      # [verified] fSWAP * exp(i theta Z Z), generic theta
DOUBLE_CORE_PUBLISHED = 13   # [published] Yordanov et al. (2020)
_DOUBLE_CORE_EXPLICIT = 14   # [verified] excitation_circuit (3 + 8 + 3)
_SINGLE_CORE = 6             # [verified] excitation_circuit (1 + 4 + 1)
_DIAGONAL_CORE = 2           # [verified] controlled phase
_DOUBLE_LINE_EXTRA = 6       # [estimate] two next-nearest-neighbour CNOTs, bridged (+3 each)
_DOUBLE_ROLE_SWAPS = {"AACC": 0, "CCAA": 0, "ACAC": 1, "CACA": 1, "ACCA": 2, "CAAC": 2}
_SINGLE_LINE_EXTRA = 4       # [estimate] one fSWAP there and back if the control sits in the middle


def _check_connectivity(connectivity: str) -> str:
    if connectivity not in CONNECTIVITIES:
        raise ValueError(f"connectivity must be one of {CONNECTIVITIES}, got {connectivity!r}")
    return connectivity


# --------------------------------------------------------------------------- expansion
@dataclass(frozen=True)
class Excitation:
    """One spin-orbital term of a spin-free generator (qubit indices, blocked JW order).

    For ``kind`` 'double' or 'single' the operator is
    ``amplitude * tau - conj(amplitude) * tau^dag`` with
    tau = a+_{c0} a+_{c1} a_{d1} a_{d0}, where (c0, c1) = ``creators`` and
    (d0, d1) = ``annihilators`` (both sorted). By convention
    ``creators > annihilators`` as tuples. For 'single' the two tuples share
    one mode, the ``control`` c, and tau = +-n_c a+_x a_y. For 'diagonal'
    (creators == annihilators) the operator is ``amplitude * n_{c0} n_{c1}``.
    Its amplitude is purely imaginary because G is anti-Hermitian, so it occurs
    only for complex g.
    """

    kind: str
    creators: tuple[int, int]
    annihilators: tuple[int, int]
    amplitude: complex

    @property
    def modes(self) -> tuple[int, ...]:
        """Sorted distinct modes acted on (4, 3 or 2 of them)."""
        return tuple(sorted(set(self.creators) | set(self.annihilators)))

    @property
    def control(self) -> int | None:
        """The occupation-number control of a 'single' (None otherwise)."""
        if self.kind != "single":
            return None
        return (set(self.creators) & set(self.annihilators)).pop()

    @property
    def particle(self) -> int | None:
        """x of n_c (a+_x a_y - h.c.) for a 'single'."""
        if self.kind != "single":
            return None
        return (set(self.creators) - set(self.annihilators)).pop()

    @property
    def hole(self) -> int | None:
        """y of n_c (a+_x a_y - h.c.) for a 'single'."""
        if self.kind != "single":
            return None
        return (set(self.annihilators) - set(self.creators)).pop()


def spin_orbital_excitations(g: np.ndarray, tol: float = 1e-10,
                             check_anti_hermitian: float = 1e-8) -> list[Excitation]:
    """Exact spin-orbital expansion of G = sum_pqrs g[p,q,r,s] e_{pq,rs}, merged by operator.

    The spin sum e_{pq,rs} = sum_{sigma tau} a+_{p sigma} a+_{r tau} a_{s tau} a_{q sigma}
    is expanded with modes P = p + sigma N. Terms with a repeated creator or a
    repeated annihilator vanish. Each remaining term is brought to the
    normal-ordered form tau(C, D) = a+_{c0} a+_{c1} a_{d1} a_{d0} with sorted
    C and D, with sign (-1)^{[P > R] + [Q > S]}. Terms with the same (C, D)
    are the same operator and their coefficients are added. The normal-ordered
    monomials are linearly independent, so this merging is complete. The
    anti-Hermitian partners tau(C, D) and tau(D, C) = tau(C, D)^dag are then
    combined into one :class:`Excitation`.

    Args:
        g: (N, N, N, N) pair-order tensor of an anti-Hermitian G (real or
            complex). Only the operator matters, so g need not be pair-symmetric.
        tol: drop excitations with |amplitude| <= tol * max |amplitude|.
        check_anti_hermitian: raise ValueError if the merged coefficients
            violate T(D, C) = -conj(T(C, D)) by more than this, relative to the
            largest coefficient.

    Returns:
        Excitations sorted by decreasing |amplitude|, ties broken by
        (kind, creators, annihilators).
    """
    g = np.asarray(g)
    if g.ndim != 4 or len(set(g.shape)) != 1:
        raise ValueError(f"expected an (N, N, N, N) tensor, got shape {g.shape}")
    n = g.shape[0]
    m = 2 * n
    p, q, r, s = (x.ravel() for x in np.indices(g.shape))
    vals = g.ravel()
    nz = vals != 0
    p, q, r, s, vals = p[nz], q[nz], r[nz], s[nz], vals[nz]
    keys_all, coef_all = [], []
    for sig in (0, 1):
        for tau in (0, 1):
            P, Q, R, S = p + sig * n, q + sig * n, r + tau * n, s + tau * n
            ok = (P != R) & (Q != S)
            P, Q, R, S, v = P[ok], Q[ok], R[ok], S[ok], vals[ok]
            sign = np.where(P > R, -1.0, 1.0) * np.where(Q > S, -1.0, 1.0)
            c0, c1 = np.minimum(P, R), np.maximum(P, R)
            d0, d1 = np.minimum(Q, S), np.maximum(Q, S)
            keys_all.append(((c0 * m + c1) * m + d0) * m + d1)
            coef_all.append(sign * v)
    if not keys_all or sum(k.size for k in keys_all) == 0:
        return []
    keys = np.concatenate(keys_all)
    coefs = np.concatenate(coef_all)
    uniq, inv = np.unique(keys, return_inverse=True)
    tot = np.zeros(uniq.size, dtype=np.result_type(coefs, float))
    np.add.at(tot, inv, coefs)
    table = dict(zip(uniq.tolist(), tot.tolist()))
    scale = max((abs(v) for v in table.values()), default=0.0)
    if scale == 0.0:
        return []

    def unpack(key):
        d1 = key % m
        key //= m
        d0 = key % m
        key //= m
        c1 = key % m
        c0 = key // m
        return (c0, c1), (d0, d1)

    out: list[Excitation] = []
    worst = 0.0
    for key, val in table.items():
        cc, dd = unpack(key)
        if cc == dd:
            worst = max(worst, abs(np.real(val)))  # must be purely imaginary
            if abs(val) > tol * scale:
                out.append(Excitation("diagonal", cc, dd, complex(1j * np.imag(val))))
            continue
        partner = table.get(((dd[0] * m + dd[1]) * m + cc[0]) * m + cc[1], 0.0)
        worst = max(worst, abs(partner + np.conj(val)))
        if cc < dd:
            continue  # handled with its partner
        t = 0.5 * (val - np.conj(partner))  # exact for anti-Hermitian G
        if abs(t) <= tol * scale:
            continue
        kind = "double" if not (set(cc) & set(dd)) else "single"
        t = float(np.real(t)) if np.isrealobj(g) else complex(t)
        out.append(Excitation(kind, cc, dd, t))
    if worst > check_anti_hermitian * scale:
        raise ValueError(f"generator is not anti-Hermitian (violation {worst / scale:.2e} relative)")
    out.sort(key=lambda e: (-abs(e.amplitude), e.kind, e.creators, e.annihilators))
    return out


# ----------------------------------------------------------------- elementwise cost
def parity_length(exc: Excitation) -> int:
    """|S|, the number of qubits on the Jordan-Wigner string of the excitation."""
    if exc.kind == "double":
        m1, m2, m3, m4 = exc.modes
        return (m2 - m1 - 1) + (m4 - m3 - 1)
    if exc.kind == "single":
        x, y, c = exc.particle, exc.hole, exc.control
        lo, hi = min(x, y), max(x, y)
        return (hi - lo - 1) - (1 if lo < c < hi else 0)
    return 0


def gather_swaps(modes: Sequence[int]) -> int:
    """Minimal number of adjacent transpositions that make ``modes`` contiguous, order kept.

    With sorted positions m_0 < ... < m_{k-1} and y_t = m_t - t, the block
    starting at c costs sum_t |y_t - c|, which a median of y minimizes.
    """
    ms = np.sort(np.asarray(list(modes), dtype=int))
    if ms.size <= 1:
        return 0
    y = ms - np.arange(ms.size)
    med = y[(ms.size - 1) // 2]
    return int(np.sum(np.abs(y - med)))


def role_swaps(exc: Excitation) -> int:
    """n_o: adjacent fSWAPs that bring a gathered double to the role pattern AACC or CCAA."""
    if exc.kind != "double":
        return 0
    roles = "".join("A" if m in exc.annihilators else "C" for m in exc.modes)
    return _DOUBLE_ROLE_SWAPS[roles]


def excitation_cnots(exc: Excitation, connectivity: str = "all",
                     double_core: int = DOUBLE_CORE_PUBLISHED) -> int:
    """CNOT estimate for exp(t tau - conj(t) tau^dag) (or exp(T n n)); formulas in the module docstring."""
    _check_connectivity(connectivity)
    if connectivity == "all":
        if exc.kind == "double":
            return int(double_core) + 2 * parity_length(exc)
        if exc.kind == "single":
            return _SINGLE_CORE + 2 * parity_length(exc)
        return _DIAGONAL_CORE
    routing = 2 * FSWAP_CNOTS * gather_swaps(exc.modes)
    if exc.kind == "double":
        return int(double_core) + _DOUBLE_LINE_EXTRA + routing + 2 * FSWAP_CNOTS * role_swaps(exc)
    if exc.kind == "single":
        x, y, c = exc.particle, exc.hole, exc.control
        middle = min(x, y) < c < max(x, y)
        return _SINGLE_CORE + routing + (_SINGLE_LINE_EXTRA if middle else 0)
    return _DIAGONAL_CORE + routing


def cnot_cost_elementwise(excitations: Iterable[Excitation], n_spatial: int,
                          connectivity: str = "all",
                          double_core: int = DOUBLE_CORE_PUBLISHED) -> int:
    """Total CNOT estimate of one exponential per excitation (a first-order product).

    ``n_spatial`` (N) only validates that every mode lies on the 2N-qubit register.
    """
    total = 0
    for exc in excitations:
        if max(exc.modes) >= 2 * n_spatial or min(exc.modes) < 0:
            raise ValueError(f"excitation {exc} does not fit on {2 * n_spatial} qubits")
        total += excitation_cnots(exc, connectivity, double_core)
    return int(total)


def elementwise_breakdown(excitations: Iterable[Excitation], n_spatial: int,
                          connectivity: str = "all",
                          double_core: int = DOUBLE_CORE_PUBLISHED) -> dict:
    """{'double'|'single'|'diagonal': {'count', 'cnots'}, 'n_excitations', 'cnots'}."""
    excitations = list(excitations)
    out: dict = {k: {"count": 0, "cnots": 0} for k in ("double", "single", "diagonal")}
    for exc in excitations:
        out[exc.kind]["count"] += 1
        out[exc.kind]["cnots"] += excitation_cnots(exc, connectivity, double_core)
    out["n_excitations"] = len(excitations)
    out["cnots"] = cnot_cost_elementwise(excitations, n_spatial, connectivity, double_core)
    return out


# ------------------------------------------------------------------------ leaf cost
def orbital_rotation_cnots(n_spatial: int, connectivity: str = "all") -> int:
    """C_rot = 2 spins x N(N-1)/2 Givens rotations x 2 CNOTs = 2 N (N - 1) (both connectivities)."""
    _check_connectivity(connectivity)
    n = int(n_spatial)
    return 2 * (n * (n - 1) // 2) * GIVENS_CNOTS


def diagonal_layer_cnots(n_spatial: int, n_active: int, connectivity: str = "all") -> int:
    """C_diag for exp(i t c (sum_P mu_P n_P)^2) with mu nonzero on ``n_active`` spatial orbitals.

    P_a = n_a (2 n_a - 1) ZZ pairs. 'all': 2 P_a. 'linear':
    min(3 P_a + 2 (N(2N-1) - P_a), 5 P_a); see the module docstring.
    """
    _check_connectivity(connectivity)
    n, na = int(n_spatial), int(n_active)
    if not 0 <= na <= n:
        raise ValueError("n_active must lie in [0, n_spatial]")
    pa = na * (2 * na - 1)
    if connectivity == "all":
        return ZZ_CNOTS * pa
    full = n * (2 * n - 1)
    return int(min(FSWAP_ZZ_CNOTS * pa + FSWAP_CNOTS * (full - pa),
                   (FSWAP_ZZ_CNOTS + FSWAP_CNOTS) * pa))


def leaf_active_orbitals(fact, k: int, tol: float = 1e-6) -> int:
    """n_a of leaf k: eigenvalues mu of Q_k with |mu| > tol * max |mu|."""
    mu = np.linalg.eigvalsh(fact.Q[k])
    amax = float(np.max(np.abs(mu))) if mu.size else 0.0
    if amax == 0.0:
        return 0
    return int(np.sum(np.abs(mu) > tol * amax))


def leaf_schedule(n_leaves: int, order: int = 1) -> tuple[tuple[int, float], ...]:
    """Diagonal layers of one product-formula step, in application order: (leaf k, time fraction).

    order 1: (0, 1), (1, 1), ..., (L-1, 1).
    order 2 (Strang): (0, 1/2), ..., (L-2, 1/2), (L-1, 1), (L-2, 1/2), ..., (0, 1/2).
    This is the leaf order of GeneratorFactorization.trotter_apply; the one-body
    term acts before the first layer and, for order 2, also after the last one.
    In the compiled circuit every layer is preceded by one merged orbital
    rotation, and the step ends with the final un-rotation, so a step has
    ``len(schedule) + 1`` rotations (module docstring). L = 0 gives ().
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    L = int(n_leaves)
    if L < 0:
        raise ValueError("n_leaves must be non-negative")
    if order == 1 or L == 0:
        return tuple((k, 1.0) for k in range(L))
    half = tuple((k, 0.5) for k in range(L - 1))
    return half + ((L - 1, 1.0),) + half[::-1]


def cnot_cost_leaves(fact, n_leaves: int | None = None, connectivity: str = "all",
                     final_rotation: bool = True, mu_tol: float = 1e-6, order: int = 1) -> int:
    """CNOT estimate of the order-``order`` product over the top ``n_leaves`` leaves of ``fact``.

    Cost = (len(S) + [final_rotation]) C_rot + sum_{(k, t) in S} C_diag(n_a(k))
    with S = leaf_schedule(L, order). That is

        order 1:  L C_rot + sum_{k<L} C_diag(k) + [final_rotation] C_rot
        order 2:  (2L - 1) C_rot + 2 sum_{k<L-1} C_diag(k) + C_diag(L-1) + [final_rotation] C_rot.

    Adjacent rotations are merged and the one-body term is absorbed into the
    neighbouring rotations (module docstring). With L = 0 the step is the
    identity (0 CNOTs). ``fact`` is a
    :class:`thc_engine_cqe.factor.generator.GeneratorFactorization`.
    """
    _check_connectivity(connectivity)
    L = fact.n_leaves if n_leaves is None else max(0, min(int(n_leaves), fact.n_leaves))
    schedule = leaf_schedule(L, order)
    if L == 0:
        return 0
    n = fact.norb
    rot = orbital_rotation_cnots(n, connectivity)
    layer = [diagonal_layer_cnots(n, leaf_active_orbitals(fact, k, mu_tol), connectivity)
             for k in range(L)]
    n_rot = len(schedule) + (1 if final_rotation else 0)
    return int(n_rot * rot + sum(layer[k] for k, _ in schedule))


# ------------------------------------------------------- verification utilities
_SY2 = np.kron(np.array([[0, -1j], [1j, 0]]), np.array([[0, -1j], [1j, 0]]))


def two_qubit_cnot_count(u: np.ndarray, atol: float = 1e-9) -> int:
    """Minimal number of CNOTs (with free single-qubit gates) for a 4 x 4 unitary [SBM04].

    With u normalized to SU(4) and gamma(u) = u (Y x Y) u^T (Y x Y):
    0 CNOTs iff gamma = +-1, 1 iff tr gamma = 0 and gamma^2 = -1, 2 iff tr gamma
    is real, and 3 otherwise. These conditions do not depend on the fourth root
    chosen for the normalization.
    """
    u = np.asarray(u, dtype=complex)
    if u.shape != (4, 4) or not np.allclose(u.conj().T @ u, np.eye(4), atol=1e-10):
        raise ValueError("expected a 4 x 4 unitary")
    u = u / np.linalg.det(u) ** 0.25
    gam = u @ _SY2 @ u.T @ _SY2
    tr = np.trace(gam)
    eye = np.eye(4)
    if np.allclose(gam, eye, atol=atol) or np.allclose(gam, -eye, atol=atol):
        return 0
    if abs(tr) < atol and np.allclose(gam @ gam, -eye, atol=atol):
        return 1
    if abs(tr.imag) < atol:
        return 2
    return 3


def _multiplexed_ry(target: int, controls: Sequence[int], alphas: np.ndarray) -> list:
    """Uniformly controlled R_y [MVBS04]: R_y(alphas[j]) on ``target`` for control state j.

    Bit i of j is the value of ``controls[i]``. The gates are 2^k R_y rotations
    and 2^k CNOTs in Gray-code order.
    """
    k = len(controls)
    m = 1 << k
    gray = [i ^ (i >> 1) for i in range(m)]
    bits = [int(np.log2(gray[i] ^ gray[(i + 1) % m])) for i in range(m)]
    # sign of the i-th angle for control state j: (-1)^(<j, gray_i>)
    sgn = np.array([[(-1.0) ** bin(j & gray[i]).count("1") for i in range(m)] for j in range(m)])
    thetas = np.linalg.solve(sgn, np.asarray(alphas, dtype=float))
    gates = []
    for i in range(m):
        gates.append(("ry", target, float(thetas[i])))
        gates.append(("cx", controls[bits[i]], target))
    return gates


def _jw_sign(exc: Excitation) -> int:
    """<s1| tau |s2> with every other mode empty: s1 has C occupied, s2 has D occupied."""
    def apply(ops, occ):
        sign = 1
        occ = set(occ)
        for kind, j in reversed(ops):  # rightmost operator acts first
            below = sum(1 for x in occ if x < j)
            if kind == "a":
                if j not in occ:
                    return 0, occ
                occ.remove(j)
            else:
                if j in occ:
                    return 0, occ
                occ.add(j)
            sign *= (-1) ** below
        return sign, occ

    (c0, c1), (d0, d1) = exc.creators, exc.annihilators
    ops = [("c", c0), ("c", c1), ("a", d1), ("a", d0)]
    sign, _ = apply(ops, {d0, d1})  # s2: the annihilated modes D occupied (for a single, D = {y, c})
    return int(sign)


def excitation_circuit(exc: Excitation) -> list:
    """Explicit CNOT + single-qubit circuit for exp(t tau - conj(t) tau^dag) (or exp(T n n)).

    Gates are tuples: ('cx', control, target), ('cz', a, b), ('ry', q, theta)
    = exp(-i theta Y / 2), and ('p', q, phi) = diag(1, e^{i phi}). The CNOT
    count is 14 + 2|S| for a double, 6 + 2|S| for a single and 2 for a
    diagonal. tests/test_generator_rank.py checks the unitary against the
    Jordan-Wigner oracle.
    """
    t = complex(exc.amplitude)
    if exc.kind == "diagonal":
        a, b = exc.creators
        phi = float(np.imag(t))  # exp(i phi n_a n_b)
        return [("p", a, phi / 2), ("p", b, phi / 2), ("cx", a, b), ("p", b, -phi / 2), ("cx", a, b)]
    mag, beta = abs(t), float(np.angle(t))
    sigma = _jw_sign(exc)
    phi = -2.0 * mag * sigma  # R_y angle; see the derivation in the module docstring
    if exc.kind == "double":
        (a, b), (c, d) = exc.creators, exc.annihilators
        # (a,b,c,d) = (1,1,0,0) and (0,0,1,1) -> (0,1,0,0) and (0,1,0,1)
        basis_map = [("cx", d, c), ("cx", b, a), ("cx", d, b)]
        alphas = np.zeros(8)
        alphas[0b010] = phi  # controls (a, b, c) = (0, 1, 0)
        core = basis_map + _multiplexed_ry(d, [a, b, c], alphas) + basis_map[::-1]
        phase_qubit, flip_qubit = a, d
        lo1, hi1, lo2, hi2 = exc.modes
        string = list(range(lo1 + 1, hi1)) + list(range(lo2 + 1, hi2))
    else:
        x, y, ctl = exc.particle, exc.hole, exc.control
        # (x, y) = (1, 0) and (0, 1) -> (1, 0) and (1, 1); rotate y if x = 1 and ctl = 1
        basis_map = [("cx", y, x)]
        alphas = np.zeros(4)
        alphas[0b11] = phi
        core = basis_map + _multiplexed_ry(y, [x, ctl], alphas) + basis_map
        phase_qubit, flip_qubit = x, y
        string = [j for j in range(min(x, y) + 1, max(x, y)) if j != ctl]
    gates: list = []
    if beta != 0.0:
        gates.append(("p", phase_qubit, -beta))
    if string:
        pq = string[0]
        ladder = [("cx", j, pq) for j in string[1:]]
        gates += ladder + [("cz", pq, flip_qubit)] + core + [("cz", pq, flip_qubit)] + ladder[::-1]
    else:
        gates += core
    if beta != 0.0:
        gates.append(("p", phase_qubit, beta))
    return gates


def circuit_cnots(gates: Iterable[tuple]) -> int:
    """Number of two-qubit gates (CNOT and CZ, which are CNOT-equivalent)."""
    return sum(1 for gt in gates if gt[0] in ("cx", "cz"))


def circuit_unitary(gates: Iterable[tuple], n_qubits: int) -> np.ndarray:
    """Dense unitary of a gate list; basis index bit k = qubit k (as in testing.fock)."""
    dim = 1 << n_qubits
    idx = np.arange(dim)
    u = np.eye(dim, dtype=complex)
    for gt in gates:
        kind = gt[0]
        if kind == "cx":
            c, tq = gt[1], gt[2]
            perm = idx ^ (((idx >> c) & 1) << tq)
            u = u[perm]
        elif kind == "cz":
            a, b = gt[1], gt[2]
            u = u * np.where(((idx >> a) & 1) & ((idx >> b) & 1), -1.0, 1.0)[:, None]
        elif kind == "p":
            q, ph = gt[1], gt[2]
            u = u * np.where((idx >> q) & 1, np.exp(1j * ph), 1.0)[:, None]
        elif kind == "ry":
            q, th = gt[1], gt[2]
            co, si = np.cos(th / 2), np.sin(th / 2)
            lo = idx[((idx >> q) & 1) == 0]
            hi = lo | (1 << q)
            a0, a1 = u[lo].copy(), u[hi].copy()
            u[lo] = co * a0 - si * a1
            u[hi] = si * a0 + co * a1
        else:
            raise ValueError(f"unknown gate {gt!r}")
    return u

# Conventions

The operator and RDM conventions on this page are checked by
`tests/test_conventions.py` against an independent Jordan–Wigner Fock-space
implementation (`thc_engine_cqe.testing.fock`); the factorization, optimizer and
qubit sections are checked by the tests of those modules (`test_factor_*.py`,
`test_generator_factor.py`, `test_optimizers.py`, `test_pauli.py`). If your
in-house code uses different conventions, convert at the boundary using the maps
below.

## Orbitals and integrals

- N real, orthonormal spatial orbitals; indices p, q, r, s ∈ {0, …, N−1}.
- `h1[p, q] = ⟨p|h|q⟩`, which includes the frozen-core mean field for active spaces.
- `eri[p, q, r, s] = (pq|rs)` uses chemist notation and has 8-fold symmetry.
- `ecore` is the nuclear repulsion plus the frozen-core energy.

## Operators

- One-body: E\_pq = Σ\_σ a†\_{pσ} a\_{qσ}.
- Two-body, normal-ordered and spin-free:

  e\_{pq,rs} = Σ\_{στ} a†\_{pσ} a†\_{rτ} a\_{sτ} a\_{qσ} = E\_pq E\_rs − δ\_qr E\_ps.

  Pair exchange leaves it unchanged (e\_{pq,rs} = e\_{rs,pq}), and its adjoint is e\_{pq,rs}† = e\_{qp,sr}.
- Hamiltonian:

  H = ecore + Σ_pq h_pq E_pq + ½ Σ_pqrs (pq|rs) e_{pq,rs}.

The CI-vector routines in `fci_ops` apply exactly these operators:

| Function | Applies |
|---|---|
| `one_body(space, f, ci)` | Σ f[p,q] E_pq |
| `two_body_ee(space, m, ci)` | Σ m[p,q,r,s] E_pq E_rs |
| `two_body(space, c, ci)` | Σ c[p,q,r,s] e_{pq,rs} |
| `Hamiltonian(active).apply(ci)` | H − ecore |

PySCF's `direct_nosym.contract_2e(m)` computes Σ m[p,q,r,s] E_rs E_pq, the
reverse operator order from what its docstring suggests. `two_body_ee`
compensates for this, and a regression test uses a tensor without pair-exchange
symmetry to catch any change.

## Pair-order tensors

Two-body quantities are stored in **pair order**: index `[p, q, r, s]` labels the
operator e_{pq,rs}. This is PySCF's order:

| Quantity | Definition |
|---|---|
| `dm2[p,q,r,s]` | ⟨Ψ| e_{pq,rs} |Ψ⟩, the spin-summed 2-RDM |
| `dm1[p,q]` | ⟨Ψ| E_qp |Ψ⟩. Note the transpose, which is PySCF's convention. |
| `T[p,q,r,s]` | ⟨Ψ| e_{pq,rs} H |Ψ⟩, the CSE matrix |
| `A[p,q,r,s]` | ⟨Ψ| [e_{pq,rs}, H] |Ψ⟩ = T[p,q,r,s] − conj(T[q,p,s,r]), the ACSE residual |

`trans_rdm12(space, bra, ket)` returns `⟨bra| … |ket⟩`: the first argument is the bra.

**Normalization:** no factor of ½. The trace is Σ_pq dm2[p,p,q,q] = N(N−1). If
your code uses ²D with a 1/2! prefactor (trace N(N−1)/2), multiply by ½.

**Mazziotti index order:** ²D^{ij}\_{kl} = ⟨a†\_i a†\_j a\_l a\_k⟩ in spin orbitals.
Summing over spin, with i,k sharing one spin and j,l the other, gives

  ²D^{ij}_{kl} (spin-summed) = dm2[i, k, j, l].

Convert with `residual.pair_to_mazziotti` and `residual.mazziotti_to_pair`; each
is a `transpose(0, 2, 1, 3)` and is its own inverse.

## The CQE update

Ψ ← exp(ε G) Ψ with G = Σ g[p,q,r,s] e_{pq,rs} and **g = conj(A)**
(`residual.descent_generator`). G is anti-Hermitian, and

  dE/dε |₀ = ⟨[H, G]⟩ = −Σ |A|²,

so ε > 0 lowers the energy. For complex states the conjugate matters, because
the unconjugated sum is generally not a descent direction. The driver
(`cqe.run_cqe`) also accepts a compressed generator: it checks the sign of the
slope −Re Σ g·A and flips it if needed.

Because the generators are spin-free, the exact CQE preserves S² (singlets stay
singlets). The default is **steepest descent with an exact line search**, the M1
reference trajectory; it converges slowly in the tail (it zig-zags).
`run_cqe(..., optimizer="cg" | "lbfgs")` uses nonlinear conjugate-gradient
(Polak–Ribière+ by default) or L-BFGS directions d instead of g, with
Ψ ← exp(ε D)Ψ, D = Σ d·e. Both work in the local coordinates of the current
iterate, with gradient −conj(A) and the real inner product Re Σ conj(u) v
(`thc_engine_cqe.optimizers`; this is the "locally parameterized" quasi-Newton
CQE of Smart, Boyn & Mazziotti, JCTC 18, 5286 (2022)).
`run_cqe(..., metric="spin_orbital")` preconditions any of the three with the
metric M = (2I + T)/3, (TX)[p,q,r,s] = X[p,s,r,q] (`optimizers.apply_metric`);
the default `metric="spin_free"` is M = I, which reproduces the M1 and M2
directions bit for bit. On tensors with the structure of A, M is symmetric
positive definite, with eigenvalue 1 on T = +1 (singlet-coupled pairs) and 1/3 on
T = −1 (triplet-coupled pairs). For a singlet state

  Σ\_{ijkl} conj(²A^{ij}\_{kl}) a†\_i a†\_j a\_l a\_k = Σ\_pqrs [M g]\_pqrs e\_{pq,rs},  ²A^{ij}\_{kl} = ⟨[a†\_i a†\_j a\_l a\_k, H]⟩,

with i, j, k, l running independently over all 2N spin orbitals and no
prefactor: the full-sum convention. So steepest descent with this metric is the
standard spin-orbital CQE of Smart & Mazziotti, PRL 126, 070504 (2021), and fixed
step sizes ε multiply the full-sum generator. A sum over i < j, k < l is a
quarter of it, so in that convention the same step is 4ε. The full-sum
convention is the one of the Mazziotti group's HQCA code (github.com/damazz/HQCA,
commit 5733356: `hqca/acse/_quant_A_acse.py` turns every element of the full
(2N)⁴ residual tensor, measured with the unnormalized 2-RDM, into one term of the
generator, and the Euler step multiplies it by ε; details in the `optimizers`
module docstring). The PRL's own generator equation was not checked. The
identity is tested as an operator identity on the Fock space of H4.
Equivalently, M⁻¹ is the Gram matrix of the e_{pq,rs} in the Euclidean inner
product of spin-orbital amplitudes (a triplet-pair amplitude multiplies three
M_S components), so this is steepest descent in the metric of the spin-orbital
amplitudes.
With the metric, CG builds β from z = M g (preconditioned Polak–Ribière+ or
Fletcher–Reeves) and L-BFGS starts from H₀ = γM, γ = ⟨s, y⟩/⟨y, My⟩. For a
non-singlet state Σ (M g)·e is only the spin average of the spin-orbital
generator, and that generator does not commute with S². The same holds for an
estimated residual: if A_est is the spin contraction
(`measure.targets.contract_spin`) of a spin-orbital estimate ²A_est, then
Σ (M conj(A\_est))·e is the spin average of Σ conj(²A\_est^{ij}\_{kl}) a†\_i a†\_j a\_l a\_k
for any state and any estimation error, not that estimated generator itself. With
`run_cqe(..., eps_candidates=...)` each step instead takes the lowest energy among
a few fixed step sizes, as a hardware run would; such steps can raise the energy.
Candidates may be zero (stay) or negative (step against d, useful when a noisy
direction is uphill for the true energy); the exact line search is one-sided,
on [0, ε_max], because the driver only searches along descent directions.

`residual_estimator` and `energy_estimator` replace the exact residual and the
energies in every decision of the driver (direction, slope checks, line-search
bracket, candidate choice, stopping tests) by estimates, for example from a
finite-shot model. The exact quantities are still recorded next to the estimates.
An estimated residual must have the structure of A (pair symmetry and
A[q,p,s,r] = −conj(A[p,q,r,s]); `optimizers.symmetrize_residual` projects onto it).

## Reduced Hamiltonian and variance

On the N-electron space, E_pq = Σ_r e_{pq,rr}/(N−1), so

  K[p,q,r,s] = ½ (pq|rs) + (h_pq δ_rs + δ_pq h_rs) / (2(N−1))

satisfies E_el = Σ K·dm2. It follows that

  σ² = ⟨H²⟩ − E² = Σ K (T − E_el · dm2)

holds exactly. `residual.variance_from_cse` computes this sum, and
`Hamiltonian.energy_and_variance` gives σ² directly as ‖HΨ‖² − E².

## Factorized integrals and λ (`factor.eri_factor`, `cholesky`, `df`, `thc`)

Every ERI factorization returns a dense model tensor through `to_eri()`; the
model Hamiltonian is `active.with_eri(factor.to_eri())`. THC uses OpenFermion's
layout (`etaPp` (M, N), `MPQ` (M, M)), with (pq|rs) ≈ Σ_PQ η_Pp η_Pq M_PQ η_Qr η_Qs.
`fit_thc` returns unit-norm leaf rows, so `MPQ` is OpenFermion's normalized
central tensor. DF stores leaves L_l = U_l diag(μ_l) U_lᵀ with (pq|rs) ≈ Σ_l L_l[p,q] L_l[r,s].

The block-encoding 1-norms follow OpenFermion's `resource_estimates` (Lee et
al., PRX Quantum 2, 030305 (2021)). They start from the exact identity

  H − ecore = Σ_pq T'_pq E_pq + ½ Σ_pqrs (pq|rs) Ẽ_pq Ẽ_rs − ½ Σ_pr (pp|rr),

  Ẽ_pq = E_pq − δ_pq,  T'_pq = h_pq − ½ Σ_r (pr|rq) + Σ_r (rr|pq).

Here λ_T = Σ_k |eig_k(T')|, λ_DF = λ_T + ¼ Σ_l (Σ_k |μ_lk|)², and THC λ is
OpenFermion's `compute_lambda`. As in OpenFermion, T' uses the *exact* ERIs:
pass them as `eri_exact` to `lam`. Agreement with OpenFermion is tested to
1e-10, and the FeMoco reference values (THC λ = 294, DF λ = 294.8) are reproduced.

## Generator double factorization (`factor.generator`)

For an anti-Hermitian pair-order tensor g, with gs = (g + g[r,s,p,q])/2,

  G = Σ_pqrs g[p,q,r,s] e_{pq,rs} = Σ_k c_k i Q̂_k² + Ô_F

holds as an operator identity on the whole Fock space. Here:

- Q̂_k = Σ Q_k[p,q] E_pq, with each Q_k Hermitian and tr(Q_k Q_l) = δ_kl.
- The c_k are real and sorted by decreasing |c_k|. They are minus the
  eigenvalues of the real symmetric N²×N² matrix `coupling_matrix(gs)`, the |c_k|
  are the singular values of gs reshaped to N²×N², and Σ c_k² = ‖gs‖².
- F = −i Σ_k c_k Q_k², so Ô_F = Σ F[p,q] E_pq is an orbital-rotation generator.

A real g gives exact conjugate pairs (c, Q) and (−c, Q*). Each leaf compiles as
Q̂_k = U(V_k) D_k U(V_k)†, with Q_k = V_k diag(μ_k) V_k† and D_k = Σ_p μ_kp n_p,
so exp(i t c_k Q̂_k²) is an orbital rotation, a diagonal Coulomb phase and the
inverse rotation. Here U(V) a†_q U(V)† = Σ_p V[p,q] a†_p.

The truncation to the first n leaves keeps its own one-body term
F_n = −i Σ_{k<n} c_k Q_k². It is the orthogonal projection g_n of g, so its slope
is −‖g_n‖² and every truncation is a descent direction. Odd truncations of real
generators split a conjugate pair and are complex; `run_cqe` propagates them
exactly, and the state becomes complex.

## Qubit conventions (`measure.pauli`)

The module docstring of `thc_engine_cqe.measure.pauli` is the full reference;
the essentials:

- **Qubit orders.** `QubitMapping(N, "blocked")` (default) puts (p, α) on qubit p
  and (p, β) on qubit N + p, the mode order of the Fock oracle below.
  `"interleaved"` uses 2p + σ, OpenFermion's order.
- **Basis states.** Bit k of the index b is the occupation of qubit k, and
  |b⟩ = (a†₀)^{b₀}(a†₁)^{b₁}⋯|vac⟩ in ascending qubit order. Dense vectors are
  indexed by b (qubit 0 least significant); OpenFermion's sparse matrices use the
  bit-reversed index.
- **Pauli strings.** P(x, z) = i^{|x∧z|} X^x Z^z for bitmasks x, z, so
  P(1,1) = Y on one qubit and every P is Hermitian.
- **Jordan–Wigner.** a\_k = Z\_{<k}(X\_k + iY\_k)/2. Majoranas γ\_{2k} = Z\_{<k}X\_k and
  γ\_{2k+1} = Z\_{<k}Y\_k.
- **CI vectors.** `QubitState.from_ci` maps the PySCF determinant (α string,
  β string) to (Π\_{p∈α↑} a†\_{pα})(Π\_{q∈β↑} a†\_{qβ})|vac⟩. In blocked order that is
  b = α_bits | (β_bits << N) with sign +1; other orders include the reordering
  sign. `pyscf_sign=True` adds PySCF's global sign (−1)^{nα(nα−1)/2 + nβ(nβ−1)/2}.
- **Expectations** are the bare ⟨ψ|P|ψ⟩ by default; pass `normalize=True` for
  ⟨ψ|P|ψ⟩/⟨ψ|ψ⟩.

## Spin-orbital conventions (Fock-space oracle only)

Mode p is (p, α) and mode N+p is (p, β). Basis index i has bit k set when mode
k is occupied, and denotes (a†₀)^{n₀}(a†₁)^{n₁}⋯|vac⟩. A PySCF determinant
(α string, β string) maps to `α_bits | (β_bits << N)`, up to a global sign.

## Scope of M1

- Real orbitals and integrals only (`ActiveSpace` rejects complex input).
  Operators and RDMs do accept complex CI vectors, because generator-DF leaves are complex.
- Spin-free (singlet-adapted) two-body generators.

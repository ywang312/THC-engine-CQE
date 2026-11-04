# Conventions

Every statement on this page is enforced by `tests/test_conventions.py`, which
checks it against an independent Jordan–Wigner Fock-space implementation
(`thc_engine_cqe.testing.fock`). If your in-house code uses different
conventions, convert at the boundary using the maps below.

## Orbitals and integrals

- N real, orthonormal spatial orbitals; indices p, q, r, s ∈ {0, …, N−1}.
- `h1[p, q] = ⟨p|h|q⟩`, which includes the frozen-core mean field for active spaces.
- `eri[p, q, r, s] = (pq|rs)` uses chemist notation and has 8-fold symmetry.
- `ecore` is the nuclear repulsion plus the frozen-core energy.

## Operators

- One-body: E_pq = Σ_σ a†_{pσ} a_{qσ}.
- Two-body, normal-ordered and spin-free:

  e_{pq,rs} = Σ_{στ} a†_{pσ} a†_{rτ} a_{sτ} a_{qσ} = E_pq E_rs − δ_qr E_ps.

  Pair exchange leaves it unchanged (e_{pq,rs} = e_{rs,pq}), and its adjoint is e_{pq,rs}† = e_{qp,sr}.
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

**Mazziotti index order:** ²D^{ij}_{kl} = ⟨a†_i a†_j a_l a_k⟩ in spin orbitals.
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
singlets). M1 uses **steepest descent with an exact line search**. This
converges slowly in the tail (it zig-zags), so it serves as a reference
trajectory, not as a production optimizer.

## Reduced Hamiltonian and variance

On the N-electron space, E_pq = Σ_r e_{pq,rr}/(N−1), so

  K[p,q,r,s] = ½ (pq|rs) + (h_pq δ_rs + δ_pq h_rs) / (2(N−1))

satisfies E_el = Σ K·dm2. It follows that

  σ² = ⟨H²⟩ − E² = Σ K (T − E_el · dm2)

holds exactly. `residual.variance_from_cse` computes this sum, and
`Hamiltonian.energy_and_variance` gives σ² directly as ‖HΨ‖² − E².

## Spin-orbital conventions (Fock-space oracle only)

Mode p is (p, α) and mode N+p is (p, β). Basis index i has bit k set when mode
k is occupied, and denotes (a†₀)^{n₀}(a†₁)^{n₁}⋯|vac⟩. A PySCF determinant
(α string, β string) maps to `α_bits | (β_bits << N)`, up to a global sign.

## Scope of M1

- Real orbitals and integrals only (`ActiveSpace` rejects complex input).
  Operators and RDMs do accept complex CI vectors, because generator-DF leaves are complex.
- Spin-free (singlet-adapted) two-body generators.

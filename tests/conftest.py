import numpy as np
import pytest

from thc_engine_cqe.integrals import ActiveSpace


def random_active_space(norb: int, nelec, seed: int = 0, scale: float = 0.3) -> ActiveSpace:
    """Random real Hamiltonian with the full 8-fold ERI symmetry."""
    rng = np.random.default_rng(seed)
    h1 = rng.standard_normal((norb, norb))
    h1 = 0.5 * (h1 + h1.T)
    # A positive semidefinite ERI tensor: (pq|rs) = sum_L L_pq L_rs with symmetric L.
    nl = norb * (norb + 1) // 2
    L = rng.standard_normal((nl, norb, norb)) * scale
    L = 0.5 * (L + L.transpose(0, 2, 1))
    eri = np.einsum("lpq,lrs->pqrs", L, L)
    return ActiveSpace(h1=h1, eri=eri, ecore=0.7, nelec=nelec, label=f"random N={norb}")


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture(params=[(3, (2, 1)), (4, (2, 2)), (4, (3, 1))], ids=lambda p: f"N{p[0]}-{p[1]}")
def small_random_system(request):
    norb, nelec = request.param
    return random_active_space(norb, nelec, seed=norb * 10 + nelec[0])

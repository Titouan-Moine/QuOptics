import math
from typing import Optional
import numpy as np
from clements_scheme.clements_scheme import T
from PyFock.fock import fock_tensor
import sparse

def contract_circuit(output: tuple[list[tuple[int, int, float, float]], np.ndarray]) -> np.ndarray:
    """
    Contract a linear optics circuit into a single matrix.

    Parameters
    ----------
    output : tuple[list[tuple], np.ndarray]
        Each tuple has the form (m, n, phi, theta), where:
            - m, n: target modes
            - phi, theta: parameters of the beamsplitter
        The second element is the final diagonal matrix Dfinal.

    Returns
    -------
    np.ndarray
        Total circuit transformation
    """
    full_decomposition = output[0]
    Dfinal = output[1]
    N = Dfinal.shape[0]   # nombre de lignes
    U_total = Dfinal.copy()  # start with the final diagonal matrix

    for m, n, phi, theta in full_decomposition:
        Tmn = T(m, n, phi, theta, N)
        U_total = U_total @ Tmn   # left → right (input → output)

    return U_total

def clements_macro(output: tuple[list[tuple[int, int, float, float]], np.ndarray],
                   n_photons: Optional[int]=None) -> tuple[sparse.COO, np.ndarray]:
    """
    Contract a linear optics circuit into a single L×L matrix,
    then compute the Fock state amplitude tensor for n_photons.

    Parameters
    ----------
    output : tuple[list[tuple], np.ndarray]
        Each tuple has the form (m, n, phi, theta), where:
            - m, n: target modes
            - phi, theta: parameters of the beamsplitter
        The second element is the final diagonal matrix Dfinal.
    n_photons : int
        Number of photons

    Returns
    -------
    np.ndarray
        The Fock state amplitude tensor.
    """
    if n_photons is None:
        n_photons = math.ceil(output[1].shape[0] / 10)  # default to number of modes divided by 10
    U_total = contract_circuit(output)
    return fock_tensor(U_total, n_photons, sparse_tensor=True, method='glynn_gray'), U_total

def test_empty_circuit():
    dim = 4
    full_decomposition = []
    Dfinal = np.eye(dim, dtype=complex)

    U = contract_circuit((full_decomposition, Dfinal))

    assert np.allclose(U, np.eye(dim))
def test_single_gate():
    dim = 4
    m, n = 1, 2
    phi = 0.3
    theta = 0.7

    full_decomposition = [(m, n, phi, theta)]
    Dfinal = np.eye(dim, dtype=complex)

    U = contract_circuit((full_decomposition, Dfinal))
    U_expected = T(m, n, phi, theta, dim)

    assert np.allclose(U, U_expected)
def test_two_gates_order():
    dim = 5

    gate1 = (0, 1, 0.2, 0.4)
    gate2 = (2, 3, 0.5, 0.9)

    full_decomposition = [gate1, gate2]
    Dfinal = np.eye(dim, dtype=complex)

    U = contract_circuit((full_decomposition, Dfinal))

    U1 = T(*gate1, dim)
    U2 = T(*gate2, dim)

    U_expected = U2 @ U1

    assert np.allclose(U, U_expected)
def test_with_diagonal_final():
    dim = 4

    full_decomposition = [
        (0, 1, 0.1, 0.3),
        (1, 2, 0.4, 0.8),
        (2, 3, 0.6, 0.2),
    ]

    phases = np.exp(1j * np.array([0.2, 1.1, -0.4, 0.7]))
    Dfinal = np.diag(phases)

    U = contract_circuit((full_decomposition, Dfinal))

    U_expected = np.eye(dim, dtype=complex)
    for g in full_decomposition:
        U_expected = T(*g, dim) @ U_expected

    assert np.allclose(U, U_expected)

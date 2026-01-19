import math
import numpy as np
from clements_scheme.clements_scheme import T
from fock_amplitude import fock_tensor

def contract_circuit(output):
    """
    Contract a linear optics circuit into a single L×L matrix.

    Parameters
    ----------
    circuit : list[dict]
        Each dict has keys:
            - 'matrix': local gate
            - 'modes': target modes
    L : int
        Total number of modes

    Returns
    -------
    np.ndarray
        Total circuit transformation
    """
    full_decomposition=output[0]
    Dfinal=output[1]
    dim = Dfinal.shape[0]   # nombre de lignes
    U_total = np.eye(dim, dtype=complex)

    for element in full_decomposition:
        U_gate= T(element[0],element[1],element[2],element[3],dim)
        U_total = U_gate @ U_total   # left → right (input → output)

    return U_total

def contract_circuit_then_fock(output, n_photons=None):
    """
    Contract a linear optics circuit into a single L×L matrix,
    then compute the Fock state amplitude tensor for n_photons.

    Parameters
    ----------
    circuit : list[dict]
        Each dict has keys:
            - 'matrix': local gate
            - 'modes': target modes
    L : int
        Total number of modes
    n_photons : int
        Number of photons

    Returns
    -------
    np.ndarray
        The Fock state amplitude tensor.
    """
    if n_photons is None:
        n_photons = math.ceil(output[1].shape[0] / 10)  # default to number of modes
    U_total = contract_circuit(output)
    return fock_tensor(U_total, n_photons, method='glynn_gray')

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

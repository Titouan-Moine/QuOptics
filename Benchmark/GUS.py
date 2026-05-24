
import math
from typing import Optional
import numpy as np
import sparse
from clements_scheme.clements_scheme import T
from PyFock.fock import fock_tensor, fock_amplitude

from Benchmark.circuit_types import BSGate, Circuit

def contract_circuit(circuit: tuple[list[BSGate], np.ndarray]) -> np.ndarray:
    """
    Contract a linear optics circuit into a single matrix.

    Parameters
    ----------
    circuit : tuple[list[BSGate], np.ndarray]
        Each BSGate has the form (m, n, phi, theta), where:
            - m, n: target modes
            - phi, theta: parameters of the beamsplitter
        The second element is the final diagonal matrix Dfinal.

    Returns
    -------
    np.ndarray
        Total circuit transformation
    """
    full_decomposition = circuit[0]
    Dfinal = circuit[1]
    N = Dfinal.shape[0]   # nombre de lignes
    U_total = Dfinal.copy()  # start with the final diagonal matrix

    for m, n, phi, theta in full_decomposition:
        Tmn = T(m, n, phi, theta, N)
        U_total = U_total @ Tmn   # left → right (input → output)

    return U_total

def clements_GUS(circuit: tuple[list[BSGate], np.ndarray],
                   n_photons: Optional[int]=None) -> tuple[sparse.COO, np.ndarray]:
    """
    Contract a linear optics circuit into a single L×L matrix,
    then compute the Fock state amplitude tensor for n_photons.

    Parameters
    ----------
    circuit : tuple[list[BSGate], np.ndarray]
        Each BSGate has the form (m, n, phi, theta), where:
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
        n_photons = math.ceil(circuit[1].shape[0] / 5)  # default to number of modes divided by 5
    U_total = contract_circuit(circuit)
    return fock_tensor(U_total, n_photons, sparse_tensor=True, method='glynn_gray'), U_total

def clements_GUS_closed_circuit(circuit: tuple[list[BSGate], np.ndarray],
                                input_state: np.ndarray | list[int],
                                output_state: np.ndarray | list[int],
                                n_photons: Optional[int]=None) -> complex:
    """Contract a linear optics circuit into a single L×L matrix,
    then compute the Fock state amplitude tensor for n_photons, using a closed-circuit approach.

    Parameters
    ----------
    circuit : tuple[list[BSGate], np.ndarray]
        Each BSGate has the form (m, n, phi, theta), where:
            - m, n: target modes
            - phi, theta: parameters of the beamsplitter
        The second element is the final diagonal matrix Dfinal.
    input_state : np.ndarray | list[int]
        The input Fock state.
    output_state : np.ndarray | list[int]
        The output Fock state.
    n_photons : int
        Number of photons

    Returns
    -------
    np.ndarray
        The Fock state amplitude tensor.
    """
    if n_photons is None:
        n_photons = math.ceil(circuit[1].shape[0] / 5)  # default to number of modes divided by 5
    U_total = contract_circuit(circuit)
    return fock_amplitude(U_total, np.array(input_state), np.array(output_state), method='glynn_gray')

def GUS_open_circuit(circuit: Circuit,
                     n_photons: int,
                     n_modes: Optional[int]=None) -> sparse.COO:
    """Contract a linear optics circuit into a single L×L matrix,
    then compute the Fock state amplitude tensor for n_photons, using a closed-circuit approach.

    Args:
        circuit (Circuit): A list of gates representing the linear optics circuit. Each gate can be
            either a beamsplitter (m, n, phi, theta) or a phase shift (m, phi).
        n_modes (Optional[int], optional): The number of modes in the circuit. If not provided, it will
            be inferred from the gates. Defaults to None.
        n_photons (Optional[int], optional): The number of photons. Defaults to None.

    Returns:
        complex: The Fock state amplitude.
    """
    if n_modes is None:
        n_modes = max(g[0] if len(g) == 2 else g[1] for g in circuit) + 1  # n_modes inferred from gates
    U_total = np.eye(n_modes, dtype=complex)
    for gate in circuit:
        if len(gate) == 4:  # Beamsplitter
            m, n, phi, theta = gate
            Tmn = T(m, n, phi, theta, n_modes)
            U_total = Tmn @ U_total
        elif len(gate) == 2:  # Phase shift
            m, phi = gate
            Pm = np.eye(n_modes, dtype=complex)
            Pm[m, m] = np.exp(1j * phi)
            U_total = Pm @ U_total
    return fock_tensor(U_total, n_photons=n_photons, sparse_tensor=True, method='glynn_gray')

def GUS_closed_circuit(circuit: Circuit,
                       input_state: np.ndarray | list[int],
                       output_state: np.ndarray | list[int],
                       n_modes: Optional[int]=None,
                       n_photons: Optional[int]=None) -> complex:
    """Contract a linear optics circuit into a single L×L matrix,
    then compute the Fock state amplitude tensor for n_photons, using a closed-circuit approach.

    Args:
        circuit (Circuit): A list of gates representing the linear optics circuit. Each gate can be
            either a beamsplitter (m, n, phi, theta) or a phase shift (m, phi).
        input_state (np.ndarray | list[int]): The input Fock state, represented as a list of photon
            counts per mode or as a numpy array.
        output_state (np.ndarray | list[int]): The output Fock state, represented as a list of photon
            counts per mode or as a numpy array.
        n_modes (Optional[int], optional): The number of modes in the circuit. If not provided, it will
            be inferred from the gates. Defaults to None.
        n_photons (Optional[int], optional): The number of photons. Defaults to None.

    Returns:
        complex: The Fock state amplitude.
    """
    if n_photons is None:
        n_photons = np.sum(input_state)  # default to total number of photons in the input state
    if n_modes is None:
        n_modes = max(g[0] if len(g) == 2 else g[1] for g in circuit) + 1  # n_modes inferred from gates
    U_total = np.eye(n_modes, dtype=complex)
    for gate in circuit:
        if len(gate) == 4:  # Beamsplitter
            m, n, phi, theta = gate
            Tmn = T(m, n, phi, theta, n_modes)
            U_total = Tmn @ U_total
        elif len(gate) == 2:  # Phase shift
            m, phi = gate
            Pm = np.eye(n_modes, dtype=complex)
            Pm[m, m] = np.exp(1j * phi)
            U_total = Pm @ U_total
    return fock_amplitude(U_total, np.array(input_state), np.array(output_state), method='glynn_gray')

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

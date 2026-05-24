"""Module for benchmarking the FTN implementation of the Fock state amplitude tensor
computation for linear optics circuits. The FTN method constructs the Fock tensor
network for the given circuit and contracts it to compute the amplitude tensor for a
specified number of photons. This module is designed to be used in conjunction with
the GUS implementation for performance comparison."""
from typing import Optional
import warnings
import math
# from collections.abc import Sequence

import numpy as np
import sparse

from Benchmark.circuit_types import BSGate, Circuit
import TeNCo.circuit as tnc

def clements_FTN(circuit: tuple[list[BSGate], np.ndarray],
                   n_photons: Optional[int]=None) -> sparse.COO:
    """
    create the Fock tensor network for a linear optics circuit defined by 'circuit',
    then contract it to compute the Fock state amplitude tensor for n_photons.
    
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
    n_modes = circuit[1].shape[0]
    if n_photons is None:
        n_photons = math.ceil(circuit[1].shape[0] / 5)  # default to number of modes divided by 10

    lattice = tnc.Lattice(n_modes, n_photons,
                          name=f"clements ({n_modes} modes, {n_photons} photons)")
    for i, bs in enumerate(circuit[0][::-1]):
        lattice.append_bs((bs[0], bs[1]), (bs[2], bs[3]), f"BS{bs[0]}{bs[1]}_{i}")
    for mode in range(n_modes):
        lattice.append_ps(mode, np.angle(circuit[1][mode, mode]), f"PS{mode}")
    lattice.contract_all(method='greedy')

    gate_names = list(lattice.gates.keys())
    if len(gate_names) != 1:
        raise ValueError(f"Expected exactly one gate after clements circuit contraction, \
            but found {len(gate_names)} gates.")
    final_gate = lattice.gates[gate_names[0]]
    # final_gate.canonicalize_axes()
    return final_gate.tensor

def clements_FTN_closed_circuit(circuit: tuple[list[BSGate], np.ndarray],
                                input_state: np.ndarray | list[int],
                                output_state: np.ndarray | list[int],) -> complex:
    """
    create the Fock tensor network for a linear optics circuit defined by 'circuit',
    then contract it to compute the Fock state amplitude for a specific input and output Fock state.
    
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
    """
    n_modes = circuit[1].shape[0]
    n_photons = sum(input_state)  # Total number of photons is the sum of the input state occupations

    lattice = tnc.Lattice(n_modes, n_photons,
                          name=f"clements ({n_modes} modes, {n_photons} photons)")

    lattice.append_input_state(input_state, name="input")

    for i, bs in enumerate(circuit[0][::-1]):
        lattice.append_bs((bs[0], bs[1]), (bs[2], bs[3]), f"BS{bs[0]}{bs[1]}_{i}")

    for mode in range(n_modes):
        lattice.append_ps(mode, np.angle(circuit[1][mode, mode]), f"PS{mode}")

    lattice.append_output_state(output_state, name="output")

    lattice.contract_all(method='greedy')

    gate_names = list(lattice.gates.keys())
    if len(gate_names) != 1:
        raise ValueError(f"Expected exactly one gate after clements circuit contraction, \
            but found {len(gate_names)} gates.")
    final_gate = lattice.gates[gate_names[0]]
    # final_gate.canonicalize_axes()
    # print("Final gate tensor for FTN:", final_gate.tensor.data)
    if final_gate.tensor.shape != ():
        raise ValueError(f"Expected the final contracted tensor to be a scalar, but got shape {final_gate.tensor.shape}.")
    if final_gate.tensor.data.size == 0:
        return 0.0  # If the tensor is empty, the amplitude is zero
    return final_gate.tensor.data[0]

def FTN_open_circuit(circuit: Circuit,
                     n_photons: int,
                     n_modes: Optional[int]=None) -> sparse.COO:
    """
    create the Fock tensor network for a linear optics circuit defined by 'circuit',
    then contract it to compute the Fock state amplitude for n_photons.
    
    Args:
        circuit (Circuit): A circuit object containing a list of beamsplitter gates and a final
            diagonal matrix. Each gate is represented as a tuple (m, n, phi, theta) for BS, or
            (m, theta) for PS.
        n_modes (Optional[int], optional): The number of modes in the circuit. If not provided,
            it will be inferred from the gates. Defaults to None.
        n_photons (Optional[int], optional): The number of photons. If not provided, it will be
            inferred from the input state. Defaults to None.

    Returns:
        complex: The computed Fock state amplitude for the given input and output states.
    """
    if n_modes is None:
        n_modes = max(max(g[0], g[1]) for g in circuit if len(g) == 4) + 1
        warnings.warn(f"Number of modes not provided, inferred as {n_modes} from the circuit gates."
                      "This may lead to incorrect results if the gates do not cover all modes.")

    lattice = tnc.Lattice(n_modes, n_photons,
                          name=f"clements ({n_modes} modes, {n_photons} photons)")

    for i, gate in enumerate(circuit):
        if len(gate) == 4:  # Beamsplitter
            m, n, phi, theta = gate
            lattice.append_bs((m, n), (phi, theta), f"BS{m}{n}_{i}")
        elif len(gate) == 2:  # Phase shift
            m, phi = gate
            lattice.append_ps(m, phi, f"PS{m}_{i}")

    lattice.contract_all(method='greedy', kron=True)

    gate_names = list(lattice.gates.keys())
    if len(gate_names) != 1:
        print(lattice.gate_graph)
        for gate_name in gate_names:
            print(f"Gate {gate_name}: {lattice.gates[gate_name].inmodes}, {lattice.gates[gate_name].outmodes}")
        raise ValueError(f"Expected only 1 gate after contraction but found {len(gate_names)}."
                         "This might be a problem with the final kronecker product")
    final_gate = lattice.gates[gate_names[0]]

    return final_gate.tensor

def FTN_closed_circuit(circuit: Circuit,
                   input_state: np.ndarray | list[int],
                   output_state: np.ndarray | list[int],
                   n_modes: Optional[int]=None,
                   n_photons: Optional[int]=None) -> complex:
    """
    create the Fock tensor network for a linear optics circuit defined by 'circuit',
    then contract it to compute the Fock state amplitude for n_photons.
    
    Args:
        circuit (Circuit): A circuit object containing a list of beamsplitter gates and a final
            diagonal matrix. Each gate is represented as a tuple (m, n, phi, theta).
        input_state (np.ndarray | list[int]): The input Fock state.
        output_state (np.ndarray | list[int]): The output Fock state.
        n_modes (Optional[int], optional): The number of modes in the circuit. If not provided,
            it will be inferred from the gates. Defaults to None.
        n_photons (Optional[int], optional): The number of photons. If not provided, it will be
            inferred from the input state. Defaults to None.

    Returns:
        complex: The computed Fock state amplitude for the given input and output states.
    """
    if n_modes is None:
        n_modes = max(max(g[0], g[1]) for g in circuit if len(g) == 4) + 1
        warnings.warn(f"Number of modes not provided, inferred as {n_modes} from the circuit gates."
                      "This may lead to incorrect results if the gates do not cover all modes.")
    if n_photons is None:
        n_photons = sum(input_state)
    lattice = tnc.Lattice(n_modes, n_photons,
                          name=f"clements ({n_modes} modes, {n_photons} photons)")

    lattice.append_input_state(input_state, name="INPUT")

    for i, gate in enumerate(circuit):
        if len(gate) == 4:  # Beamsplitter
            m, n, phi, theta = gate
            lattice.append_bs((m, n), (phi, theta), f"BS{m}{n}_{i}")
        elif len(gate) == 2:  # Phase shift
            m, phi = gate
            lattice.append_ps(m, phi, f"PS{m}_{i}")

    lattice.append_output_state(output_state, name="OUTPUT")

    lattice.contract_all(method='greedy', kron=True)

    gate_names = list(lattice.gates.keys())
    if len(gate_names) != 1:
        print(lattice.gate_graph)
        for gate_name in gate_names:
            print(f"Gate {gate_name}: {lattice.gates[gate_name].inmodes}, {lattice.gates[gate_name].outmodes}")
        raise ValueError(f"Expected only 1 gate after contraction but found {len(gate_names)}."
                         "This might be a problem with the final kronecker product")
    final_gate = lattice.gates[gate_names[0]]
    # final_gate.canonicalize_axes()
    # print("Final gate tensor for FTN:", final_gate.tensor.data)
    if final_gate.tensor.shape != ():
        raise ValueError(f"Expected the final contracted tensor to be a scalar, but got shape {final_gate.tensor.shape}.")
    res = np.array(final_gate.tensor.data)
    # print(res, res.size, res.shape)
    if res.size == 0:
        return 0.0  # If the tensor is empty, the amplitude is zero
    return complex(res)

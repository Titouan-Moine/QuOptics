"""Circuit generation functions for benchmarking."""

import random
import warnings
from typing import Optional
from collections.abc import Sequence
import numpy as np
from Benchmark.circuit_types import Gate, BSGate, Circuit
from TeNCo import circuit as tnc

def rnd_BS_circuit(n_modes: int, n_layers: int) -> list[BSGate]:
    """Generate a random circuit with the specified number of modes and layers,
    comprised of beamsplitters.
    
    Args:
        n_modes (int): Number of modes in the circuit.
        n_layers (int): Number of layers of beamsplitters.
    
    Returns:
        list[BSGate]: A list of tuples representing the
            beamsplitters in the circuit. Each tuple has the form (m, n, phi, theta),
            where m and n are the target modes, and phi and theta are the parameters
            of the beamsplitter.
    """
    circuit = []
    for l in range(n_layers):
        if l % 2 == 0:
            for m in range(0, n_modes-1, 2):
                phi = np.random.uniform(0, 2 * np.pi)
                theta = np.random.uniform(0, np.pi / 2)
                circuit.append((m, m+1, phi, theta))
        else:
            for m in range(1, n_modes-1, 2):
                phi = np.random.uniform(0, 2 * np.pi)
                theta = np.random.uniform(0, np.pi / 2)
                circuit.append((m, m+1, phi, theta))
    return circuit

def rnd_clements_like(n_modes: int, n_layers: int) -> Circuit:
    """Generate a random Clements-like circuit with the specified number of modes and layers.
    
    Args:
        n_modes (int): Number of modes in the circuit.
        n_layers (int): Number of layers of beamsplitters.
    
    Returns:
        Circuit: A list of tuples representing the
            beamsplitters and phase shifts in the circuit. Each beamsplitter tuple has the form
            (m, n, phi, theta), where m and n are the target modes, and phi and theta are the
            parameters of the beamsplitter. Each phase shift tuple has the form (m, phi), where
            m is the target mode and phi is the phase shift.
    """
    circuit = []
    for l in range(n_layers):
        if l % 2 == 0:
            for m in range(0, n_modes-1, 2):
                phi = np.random.uniform(0, 2 * np.pi)
                theta = np.random.uniform(0, np.pi / 2)
                circuit.append((m, m+1, phi, theta))
        else:
            for m in range(1, n_modes-1, 2):
                phi = np.random.uniform(0, 2 * np.pi)
                theta = np.random.uniform(0, np.pi / 2)
                circuit.append((m, m+1, phi, theta))
    # Add random phase shifts to each mode
    for m in range(n_modes):
        phi = np.random.uniform(0, 2 * np.pi)
        circuit.append((m, phi))
    return circuit

def rnd_BSPS_lasagna(n_modes: int, n_layers: int) -> Circuit:
    """Generate a random "lasagna" circuit (alternating layers of beamsplitters and phase shifts)
    with the specified number of modes and layers, comprised of beamsplitters and phase shifts.
    
    Args:
        n_modes (int): Number of modes in the circuit.
        n_layers (int): Number of layers of beamsplitters and phase shifts.
    
    Returns:
        Circuit: A list of tuples representing the
            beamsplitters and phase shifts in the circuit. Each beamsplitter tuple has the form
            (m, n, phi, theta), where m and n are the target modes, and phi and theta are the
            parameters of the beamsplitter. Each phase shift tuple has the form (m, phi), where
            m is the target mode and phi is the phase shift.
    """
    circuit = []
    for l in range(n_layers):
        if l % 2 == 0:
            for m in range(0, n_modes-1, 2):
                phi = np.random.uniform(0, 2 * np.pi)
                theta = np.random.uniform(0, np.pi / 2)
                circuit.append((m, m+1, phi, theta))
        else:
            for m in range(1, n_modes-1, 2):
                phi = np.random.uniform(0, 2 * np.pi)
                theta = np.random.uniform(0, np.pi / 2)
                circuit.append((m, m+1, phi, theta))
    # Add random phase shifts to each mode
        for m in range(n_modes):
            phi = np.random.uniform(0, 2 * np.pi)
            circuit.append((m, phi))
    return circuit

def rnd_BSPS_mixed(n_modes: int,
                   n_layers: int,
                   ps_replacement_ratio: float=0.25) -> Circuit:
    """Generate a random "mixed" circuit (BS and PS both present in each layer)
    with the specified number of modes and layers, comprised of beamsplitters and phase shifts.
    
    Args:
        n_modes (int): Number of modes in the circuit.
        n_layers (int): Number of layers of beamsplitters and phase shifts.
        ps_replacement_ratio (float): The ratio of beamsplitters that are replaced with (two) phase shifters (between 0 and 1).
            Default is 0.25, meaning 25% of beamsplitters are replaced with phase shifters.
    
    Returns:
        Circuit: A list of tuples representing the
            beamsplitters and phase shifts in the circuit. Each beamsplitter tuple has the form
            (m, n, phi, theta), where m and n are the target modes, and phi and theta are the
            parameters of the beamsplitter. Each phase shift tuple has the form (m, phi), where
            m is the target mode and phi is the phase shift.
    """
    circuit = []
    for l in range(n_layers):
        idxs = range(0, n_modes-1, 2) if l % 2 == 0 else range(1, n_modes-1, 2)
        ps_idxs = random.sample(idxs, int(len(idxs) * ps_replacement_ratio))
        for m in idxs:
            if m in ps_idxs:
                phi1 = np.random.uniform(0, 2 * np.pi)
                phi2 = np.random.uniform(0, 2 * np.pi)
                circuit.append((m, phi1))
                circuit.append((m+1, phi2))
            else:
                phi = np.random.uniform(0, 2 * np.pi)
                theta = np.random.uniform(0, np.pi / 2)
                circuit.append((m, m+1, phi, theta))
    return circuit

def rnd_circuit(n_modes: int,
                n_layers: int,
                ps_replacement_ratio: float=0.25,
                circuit_type: str="lasagna") -> Sequence[Gate]:
    """Generate a random circuit with the specified number of modes and layers, comprised of beamsplitters and phase shifts."""
    if circuit_type == "lasagna":
        return rnd_BSPS_lasagna(n_modes, n_layers)
    elif circuit_type == "mixed":
        return rnd_BSPS_mixed(n_modes, n_layers, ps_replacement_ratio)
    elif circuit_type == "clements_like":
        return rnd_clements_like(n_modes, n_layers)
    elif circuit_type == "BS_only":
        return rnd_BS_circuit(n_modes, n_layers)
    else:
        raise ValueError("Invalid circuit type. Choose 'lasagna' or 'mixed'.")

def display_circuit(circuit: Sequence[Gate],
                    method: str="plt",
                    n_modes: Optional[int]=None) -> None:
    """Display the given circuit using TeNCo's Lattice class."""
    if n_modes is None:
        n_modes = max(max(g[0], g[1]) for g in circuit if len(g) == 4) + 1
        warnings.warn(f"Number of modes not provided, inferred as {n_modes} from the circuit gates."
                      "This may lead to incorrect display if the gates do not cover all modes.")
    lattice = tnc.Lattice(n_modes, 1, name="Generated Circuit")
    for i, gate in enumerate(circuit):
        if len(gate) == 4:
            lattice.append_bs((gate[0], gate[1]), (gate[2], gate[3]), f"G{i}")
        else:
            lattice.append_ps(gate[0], gate[1], f"G{i}")
    lattice.display(method=method, label_mode='minimal')

def main():
    n_modes = 10
    n_layers = 6
    circuit1 = rnd_BS_circuit(n_modes, n_layers)
    circuit2 = rnd_BSPS_lasagna(n_modes, n_layers)
    circuit3 = rnd_BSPS_mixed(n_modes, n_layers, ps_replacement_ratio=0.5)

    print("Random BS circuit:")
    display_circuit(circuit1)
    
    print("Random BS+PS lasagna circuit:")
    display_circuit(circuit2)
    
    print("Random BS+PS mixed circuit:")
    display_circuit(circuit3)

if __name__ == "__main__":
    main()

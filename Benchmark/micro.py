"""Module for benchmarking the micro implementation of the Fock state amplitude tensor
computation for linear optics circuits. The micro method constructs the Fock tensor
network for the given circuit and contracts it to compute the amplitude tensor for a
specified number of photons. This module is designed to be used in conjunction with
the macro implementation for performance comparison."""
from typing import Optional
import math
import numpy as np
import sparse
# from clements_scheme.clements_scheme import T
import TeNCo.circuit as tnc

def clements_micro(output: tuple[list[tuple[int, int, float, float]], np.ndarray],
                   n_photons: Optional[int]=None) -> sparse.COO:
    """
    create the Fock tensor network for a linear optics circuit defined by 'output',
    then contract it to compute the Fock state amplitude tensor for n_photons.
    
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
    n_modes = output[1].shape[0]
    if n_photons is None:
        n_photons = math.ceil(output[1].shape[0] / 10)  # default to number of modes divided by 10

    lattice = tnc.Lattice(n_modes, n_photons,
                          name=f"clements ({n_modes} modes, {n_photons} photons)")
    for bs in output[0]:
        lattice.append_bs((bs[0], bs[1]), (bs[2], bs[3]))
    for mode in range(n_modes):
        lattice.append_ps(mode, np.angle(output[1][mode, mode]))
    lattice.contract_all(method='greedy')

    gate_names = list(lattice.gates.keys())
    if len(gate_names) != 1:
        raise ValueError(f"Expected exactly one gate after clements circuit contraction, \
            but found {len(gate_names)} gates.")
    return lattice.gates[gate_names[0]].tensor

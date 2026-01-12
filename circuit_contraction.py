"""
Circuit Contraction Module

This module provides functions for performing circuit contraction on quantum circuits.
"""

import numpy as np
import quimb as qu
from jump_network import quimb_network
from fock_amplitude import fock_amplitude, fock_amplitude_bs, fock_amplitude_multi_ps, fock_tensor


def tensor2matrix(U_tensor, input_inds=None, output_inds=None):
    """Convert a tensor network representation of a unitary to a matrix.

    Parameters
    ----------
    U_tensor : quimb.Tensor
        The tensor network representation of the unitary.
    input_inds : list of str
        The list of input indices.
    output_inds : list of str
        The list of output indices.

    Returns
    -------
    np.ndarray
        The unitary matrix.
    """
    if input_inds is not None and output_inds is not None:
        # Fusionner et convertir en matrice
        U_fused = U_tensor.fuse({'in': input_inds, 'out': output_inds})
        U = U_fused.to_dense(['in', 'out'])
        return U
    # Identifier automatiquement les indices d'entrée et de sortie
    all_inds = list(U_tensor.inds)
    
    # Méthode 1: Si les indices suivent une convention de nommage
    input_inds = sorted([ind for ind in all_inds if ind.startswith('k')])
    output_inds = sorted([ind for ind in all_inds if ind.startswith('b')])
    # Méthode 2: Si pas de convention, diviser en deux moitiés
    if not input_inds or not output_inds:
        n = len(all_inds) // 2
        input_inds = all_inds[:n]
        output_inds = all_inds[n:]
    # Fusionner et convertir en matrice
    U_fused = U_tensor.fuse({'in': input_inds, 'out': output_inds})
    U = U_fused.to_dense(['in', 'out'])
    return U_matrix

def compare_contraction(n_modes, n_gates, n_photons):
    """Compare different contraction methods for a given circuit.

    Parameters
    ----------
    n_modes : int
        The number of modes in the circuit.
    n_gates : int
        The number of gates in the circuit.

    Returns
    -------
    tuple of np.ndarray
        The unitary matrix U
    """
    network, gate_list = quimb_network(n_modes, n_gates, orderlist=True)
    U_tensor = network.contract()
    U = tensor2matrix(U_tensor)

    fock_tensor_1 = fock_tensor(U, n_photons, method='glynn_gray')
    
    fock_list=[]
    for gate in gate_list:
        unitary_gate = tensor2matrix(gate)
        fock_gate = fock_tensor(unitary_gate, n_photons, method='glynn_gray')
        fock_list.append(fock_gate)
    
    fock_tensor_2 = np.linalg.multi_dot(fock_list)

    return results


if __name__ == "__main__":
    n_modes = 6
    n_gates = 12
    n_photons = 3
    fock_tensor_1, fock_tensor_2 = compare_contraction(n_modes, n_gates, n_photons)
    print("Fock amplitude from full contraction:\n", fock_tensor_1)
    print("Fock amplitude from gate-by-gate contraction:\n", fock_tensor_2)
    print("Difference norm:", np.linalg.norm(fock_tensor_1 - fock_tensor_2))

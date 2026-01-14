import numpy as np

def embed_gate(local_matrix, modes, L):
    """
    Embed a local k×k gate into an L×L identity matrix.

    Parameters
    ----------
    local_matrix : np.ndarray
        k×k matrix acting on selected modes
    modes : list[int]
        Indices of modes the gate acts on
    L : int
        Total number of modes

    Returns
    -------
    np.ndarray
        L×L embedded matrix
    """
    U = np.eye(L, dtype=complex)
    modes = list(modes)

    for i, mi in enumerate(modes):
        for j, mj in enumerate(modes):
            U[mi, mj] = local_matrix[i, j]

    return U


def contract_circuit(circuit, L):
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
    U_total = np.eye(L, dtype=complex)

    for gate in circuit:
        U_gate = embed_gate(gate["matrix"], gate["modes"], L)
        U_total = U_gate @ U_total   # left → right (input → output)

    return U_total

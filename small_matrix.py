import numpy as np
from clements_scheme/clements_scheme import T

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

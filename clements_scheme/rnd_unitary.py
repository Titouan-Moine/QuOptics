"""Utility module for generating random unitary matrices.

This module provides functions to generate random unitary matrices using the
QR decomposition method. Random unitary matrices are essential for testing
and validating quantum computing algorithms and optical schemes.
"""

import numpy as np


def random_unitary(n):
    """Generate a random n×n unitary matrix using QR decomposition.
    
    Parameters
    ----------
    n : int
        The dimension of the unitary matrix.
    
    Returns
    -------
    np.ndarray
        An n×n random unitary matrix with complex entries.
    """
    # matrice complexe gaussienne
    Z = (np.random.randn(n, n) + 1j*np.random.randn(n, n)) / np.sqrt(2)

    # décomposition QR
    Q, R = np.linalg.qr(Z)

    # correction des phases
    D = np.diag(np.exp(1j * np.angle(np.diag(R))))
    return Q @ D
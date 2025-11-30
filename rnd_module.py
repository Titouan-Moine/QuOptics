"""Utility module for generating random unitary matrices.

This module provides functions to generate random unitary matrices using the
QR decomposition method. Random unitary matrices are essential for testing
and validating quantum computing algorithms and optical schemes.
"""

import numpy as np


def random_unitary(n):
    """Generate a random n×n unitary matrix using QR decomposition (Mezzadri method).
    
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

def random_fock_uniform(n, N):
    """Generate a random Fock state with a uniform distribution of photons across modes.
    
    Parameters
    ----------
    n : int
        The total number of photons.
    N : int
        The number of modes.
    
    Returns
    -------
    np.ndarray
        An array of length n_modes representing the photon distribution.
    """
    if n < 0 or N <= 0:
        raise ValueError("n_photons must be non-negative and n_modes must be positive.")
    
    # Generate a random distribution using stars and bars method
    bars = np.random.choice(np.arange(1, n+N), size=N-1, replace=False)
    bars.sort()
    bars = np.concatenate(([0], bars, [n+N]))
    #print(bars)
    stars = np.diff(bars) - 1
    return stars

def random_fock_sparse(n, N, k):
    """Generate a random Fock state with a sparse distribution of photons across modes.
    
    Parameters
    ----------
    n : int
        The total number of photons.
    N : int
        The number of modes.
    k : int
        The number of non-empty modes.
    
    Returns
    -------
    np.ndarray
        An array of length n_modes representing the photon distribution.
    """
    if n < 0 or N <= 0 or k <= 0 or k > N:
        raise ValueError("Invalid parameters: ensure n_photons >= 0, n_modes > 0, and 0 < k <= n_modes.")
    
    # Select k unique modes to be non-empty
    non_empty_modes = np.random.choice(N, size=k, replace=False)
    
    # Generate a random distribution for the selected modes
    bars = np.random.choice(np.arange(1, n+k), size=k-1, replace=False)
    bars.sort()
    bars = np.concatenate(([0], bars, [n+k]))
    stars = np.diff(bars) - 1
    
    # Create the full distribution
    distribution = np.zeros(N, dtype=int)
    distribution[non_empty_modes] = stars
    return distribution

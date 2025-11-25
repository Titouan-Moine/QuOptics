"""Permanent module for computing the permanent of a matrix using Ryser's algorithm.

This module implements multiple versions of Ryser's algorithm to compute the permanent
of a repeating sub-matrix, as needed for the bosonic amplitude extraction.

Key Functions :
    - ryser: Compute the permanent of a matrix using Ryser's algorithm
    - bin_gray : Generate Gray codes for binary numbers
    - ryser_gray: Compute the permanent using Gray code optimization
    - ryser_hyperrect: Compute the permanent of a repeating sub-matrix using an
        ameliorated Ryser's algorithm
    - mixed_gray : Generate a sort of Gray code for mixed-radix numbers
    - ryser_hyperrect_mixed: Compute the permanent of a repeating sub-matrix using
        mixed-radix Gray code optimization

"""
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
from clements_scheme.rnd_unitary import random_unitary



def ryser(A):
    """Compute the permanent of a matrix A using Ryser's algorithm.

    Parameters
    ----------
    A : np.ndarray
        A square matrix of shape (n, n).

    Returns
    -------
    complex
        The permanent of the matrix A.
    """
    n = A.shape[0]
    perm = 0.0 + 0.0j

    for S in range(1 << n): # Iterate over all subsets of columns
        row_sum = np.zeros(n, dtype=complex)
        bits = bin(S).count('1')
        sign = (-1) ** (n - bits)

        for j in range(n):
            if S & (1 << j):
                row_sum += A[:, j]

        prod = row_sum.prod()
        perm += sign * prod

    return perm

def bin_gray_rec(n):
    """Recursively generate Gray codes for binary numbers of n bits.

    Parameters
    ----------
    n : int
        The number of bits.
    
    Returns
    -------
    list of int
        A list of Gray codes.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return np.array([0,1], dtype=np.int64)

    l = bin_gray(n-1)
    return np.concatenate((l, np.flip(l) + (1 << (n-1)))).astype(np.int64)

def bin_gray(n):
    """Generate Gray codes for binary numbers of n bits.

    Parameters
    ----------
    n : int
        The number of bits.
    
    Returns
    -------
    list of int
        A list of Gray codes.
    """
    l = np.arange(1 << n, dtype=np.int64)
    return l ^ (l >> 1)

def ryser_gray(A):
    """Compute the permanent using Gray code optimization.

    Parameters
    ----------
    A : np.ndarray
        A square matrix of shape (n, n).

    Returns
    -------
    complex
        The permanent of the matrix A.
    """
    n = A.shape[0]
    perm = 0.0 + 0.0j
    sign = (-1)**n
    row_sum = np.zeros(n, dtype=complex)
    prev_S = 0

    for S in bin_gray(n)[1:]:
        j = int(prev_S ^ S).bit_length() - 1
        if (S >> j) % 2 == 0:
            row_sum -= A[:, j]
        else:
            row_sum += A[:, j]
        prod = row_sum.prod()
        sign = - sign
        prev_S = S

        perm += sign * prod

    return perm



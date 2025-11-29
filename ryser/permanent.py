"""Permanent module for computing the permanent of a matrix using Ryser's algorithm.

This module implements multiple versions of Ryser's algorithm to compute the permanent
of a repeating sub-matrix, as needed for the bosonic amplitude extraction.

Key Functions :
    - ryser: Compute the permanent of a matrix using Ryser's algorithm
    - bin_gray : Generate Gray codes for binary numbers
    - ryser_gray: Compute the permanent using Gray code optimization
    - ryser_hyperrect: Compute the permanent of a repeating sub-matrix using an
        ameliorated Ryser's algorithm
    - gray_mixed : Generate a sort of Gray code for mixed-radix numbers
    - ryser_hyperrect_gray: Compute the permanent of a repeating sub-matrix using
        mixed-radix Gray code optimization

"""
import sys
import os
from math import comb
from itertools import product
import numpy as np

# Add parent directory to path to enable imports from infoq package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rnd_module import random_unitary





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



def ryser_hyperrect(U, vecn, vecm, n=None):
    """Compute the permanent of a repeating sub-matrix using an ameliorated Ryser's algorithm.

    Parameters
    ----------
    U : np.ndarray
        The base matrix of shape (N, N).
    n : np.ndarray
        An array containing the row multiplicities.
    m : np.ndarray
        An array containing the column multiplicities.
    
    Returns
    -------
    complex
        The permanent of the repeating sub-matrix.
        
    """
    if n is None:
        n = np.sum(vecn)
    if np.sum(vecn) != np.sum(vecm) or np.sum(vecm) != n:
        raise ValueError("vecn and vecm must sum to the same amount\
                         (i.e. have the same number of photons)")
    N = U.shape[0]
    # take indices of non-zero coords of vecn and vecm
    nzn_mask = vecn != 0
    nzm_mask = vecm != 0
    nzn_index = np.arange(N)[nzn_mask]
    nzm_index = np.arange(N)[nzm_mask]
    nzn = vecn[nzn_mask]
    nzm = vecm[nzm_mask]
    #base_index = np.searchsorted(nzm_index,)
    binom_coef = {(j,c): comb(nzm[j], c) for j in range(len(nzm)) for c in range(nzm[j]+1)}
    perm = 0

    for c in product(*(range(vecm[j]+1) for j in nzm_index)):
        binom_prod = np.prod([binom_coef[(j, c[j])]
                               for j in range(len(nzm))])
        prod = np.prod([pow(np.sum([c[j]*U[i, nzm_index[j]]
                                    for j in range(len(nzm))]), nzn[i]) for i in range(len(nzn))])
        sign = (-1)**(n-np.sum(c))
        
        perm += sign * binom_prod * prod
    
    return perm

def gray_mixed(radix, prefix=(), reversed=False):
    """Generate a mixed-radix Gray code for given radices

    Parameters
    ----------
    radix : np.ndarray
        An array containing the radices.

    Yields
    ------
    tuple of: (np.ndarray, int, int)
        Contains in order:
        - The next vector in gray code order
        - the index of the change
        - the sign of the change.
    """
    if radix.shape[0] == 0:
        yield np.array(prefix)
    else:
        if reversed:
            for c in range(radix[0], -1, -1):
                if c % 2 == 0:
                    yield from gray_mixed(radix[1:], prefix + (c,), reversed=True)
                else:
                    yield from gray_mixed(radix[1:], prefix + (c,))
        else:
            for c in range(radix[0]+1):
                if c % 2 == 0:
                    yield from gray_mixed(radix[1:], prefix + (c,))
                else:
                    yield from gray_mixed(radix[1:], prefix + (c,), reversed=True)

def index_and_sign(c, prev_c):
    """Determine the index and sign of the change between two mixed-radix gray vectors.

    Parameters
    ----------
    c : np.ndarray
        The current vector.
    prev_c : np.ndarray
        The previous vector.
    
    Returns
    -------
    tuple of: (int, int)
        Contains in order:
        - the index of the change
        - the sign of the change.
    """
    diff_idx = np.nonzero(c != prev_c)[0]
    #print(diff_idx)
    if len(diff_idx) != 1:
        raise ValueError("c and prev_c must differ by exactly one element")
    idx = diff_idx[0]
    sign = int(c[idx] - prev_c[idx])
    return idx, sign


def ryser_hyperrect_gray(U, vecn, vecm, n=None):
    """Compute the permanent of a repeating sub-matrix using an ameliorated Ryser's algorithm
    and mixed-radix gray order.

    Parameters
    ----------
    U : np.ndarray
        The base matrix of shape (N, N).
    n : np.ndarray
        An array containing the row multiplicities.
    m : np.ndarray
        An array containing the column multiplicities.
    
    Returns
    -------
    complex
        The permanent of the repeating sub-matrix.
        
    """
    if n is None:
        n = np.sum(vecn)
    if np.sum(vecn) != np.sum(vecm) or np.sum(vecm) != n:
        raise ValueError("vecn and vecm must sum to the same amount\
                         (i.e. have the same number of photons)")

    N = U.shape[0]
    # take indices of non-zero coords of vecn and vecm
    nzn_mask = vecn != 0
    nzm_mask = vecm != 0
    nzn_index = np.arange(N)[nzn_mask]
    nzm_index = np.arange(N)[nzm_mask]
    nzn = vecn[nzn_mask]
    nzm = vecm[nzm_mask]

    sign = (-1)**n
    prev_c = np.zeros(len(nzm), dtype=int)
    binom_prod = 1
    row_sum = np.zeros(len(nzn), dtype=complex)
    prod = 1
    perm = 0.0 + 0.0j

    first = True
    for c in gray_mixed(nzm):
        if first:
            first = False
        else:
            diff_idx, diff_sign = index_and_sign(c, prev_c)
            changed_c = c[diff_idx]
            if diff_sign > 0:
                binom_prod = binom_prod * (nzm[diff_idx] - changed_c + 1) / changed_c
                row_sum += U[nzn_index, nzm_index[diff_idx]]
            else:
                binom_prod = binom_prod * (changed_c + 1) / (nzm[diff_idx] - changed_c)
                row_sum -= U[nzn_index, nzm_index[diff_idx]]

            #prod = np.prod([pow(row_sum[i], nzn[i]) for i in range(len(nzn))])
            prod = np.prod(np.power(row_sum, nzn))
            sign = - sign
            perm += sign * binom_prod * prod
            prev_c = c
    return perm

def repeat_matrix(U, vecn, vecm):
    """Construct the repeating sub-matrix from base matrix U and multiplicity vectors.

    Parameters
    ----------
    U : np.ndarray
        The base matrix of shape (N, N).
    vecn : np.ndarray
        An array containing the row multiplicities.
    vecm : np.ndarray
        An array containing the column multiplicities.
    
    Returns
    -------
    np.ndarray
        The constructed repeating sub-matrix.
    """
    rows = []
    for i in range(U.shape[0]):
        rows.extend([U[i, :]] * vecn[i])
    repeated_U = np.array(rows)

    cols = []
    for j in range(U.shape[1]):
        cols.extend([repeated_U[:, j]] * vecm[j])
    repeated_U = np.array(cols).T

    return repeated_U


# U = random_unitary(4)
# vecn = np.array([1,3,1,1])
# vecm = np.array([1,2,1,2])
# vecn = np.array([1,3,1,1])
# vecm = np.array([1,2,2,1])
# U = np.array([[1, 0, 0, 0],
#               [1, 0, 0, 0],
#               [1, 0, 0, 0],
#               [1, 0, 0, 0]])
# print(repeat_matrix(U, vecn, vecm))
# print("Permanent (Ryser) :", ryser(repeat_matrix(U, vecn, vecm)))
# print("Permanent (Ryser Gray) :", ryser_gray(repeat_matrix(U, vecn, vecm)))
# print("Permanent (Ryser Hyperrect) :", ryser_hyperrect(U, vecn, vecm))
# print("Permanent (Ryser Hyperrect Gray) :", ryser_hyperrect_gray(U, vecn, vecm))
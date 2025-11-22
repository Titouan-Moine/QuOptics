"""Clements scheme for decomposing unitary matrices into beam splitters and phase shifters.

This module implements the Clements decomposition algorithm, which decomposes any N×N
unitary matrix into a product of elementary beam splitter operations and phase shifts.
This decomposition is fundamental for implementing arbitrary photonic quantum gates
in linear optical quantum computing.

The algorithm works by systematically nullifying elements of the unitary matrix
using sequences of beam splitters and phase shifters, building up the decomposition
through a structured diagonal-by-diagonal elimination process.

Functions
---------
T : Create a beam splitter matrix
inverse_T : Create the inverse of a beam splitter matrix
project_U2 : Project a 2×2 matrix onto U(2)
nullify_row : Calculate parameters to nullify a matrix row element
nullify_column : Calculate parameters to nullify a matrix column element
clements_decomposition : First (preliminary) decomposition of a unitary matrix
clements_invert_left : Modify left decomposition to eliminate inverses
full_clements : Full Clements decomposition of a unitary matrix

References
----------
Clements, W. R., Renema, J. J., & O'Brien, J. L. (2016). "Optimal design for 
universal multiport interferometers." Optica, 3(12), 1460-1465.
"""

import numpy as np
from rnd_unitary import random_unitary

def T(m, n, phi, theta, N):
    """Constructs a beam splitter matrix acting on modes m and n.
    
    Parameters
    ----------
    m : int
        The first mode index (0-indexed).
    n : int
        The second mode index (0-indexed).
    phi : float
        The phase parameter (in radians).
    theta : float
        The mixing angle (in radians).
    N : int
        The total number of modes in the system.
    
    Returns
    -------
    np.ndarray
        An N×N unitary beam splitter matrix with complex entries.
    
    Raises
    ------
    ValueError
        If m >= N or n >= N.
    """
    
    if m >= N:
        raise ValueError("Mode index m must be less than N.")
    if n >= N:
        raise ValueError("Mode index n must be less than N.")
    
    T = np.eye(N, dtype=complex)
    e = np.exp(1j * phi)
    c = np.cos(theta)
    s = np.sin(theta)

    T[m, m] = e * c
    T[n, n] = c
    T[m, n] = -s
    T[n, m] = e * s

    return T

def inverse_T(m, n, phi, theta, N):
    """Constructs the inverse of a beam splitter matrix.
    
    Parameters
    ----------
    m : int
        The first mode index (0-indexed).
    n : int
        The second mode index (0-indexed).
    phi : float
        The phase parameter (in radians).
    theta : float
        The mixing angle (in radians).
    N : int
        The total number of modes in the system.
    
    Returns
    -------
    np.ndarray
        An N×N unitary matrix that is the inverse of T(m, n, phi, theta, N).
    
    Raises
    ------
    ValueError
        If m >= N or n >= N.
    """
    
    if m >= N:
        raise ValueError("Mode index m must be less than N.")
    if n >= N:
        raise ValueError("Mode index n must be less than N.")
    
    T_inv = np.eye(N, dtype=complex)
    e = np.exp(-1j * phi)
    c = np.cos(theta)
    s = np.sin(theta)

    T_inv[m, m] = e * c
    T_inv[n, n] = c
    T_inv[m, n] = e * s
    T_inv[n, m] = -s

    return T_inv

def project_U2(A):
    """Projects a 2×2 matrix onto the unitary group U(2).
    
    Parameters
    ----------
    A : np.ndarray
        A 2×2 complex matrix to be projected.
    
    Returns
    -------
    np.ndarray
        A 2×2 unitary matrix (closest to A in Frobenius norm).
    """
    U, S, Vh = np.linalg.svd(A)
    return U @ Vh

def project_D(D):
    """Projects a diagonal matrix onto the unitary group U(N).
    
    Parameters
    ----------
    D : np.ndarray
        An N×N almost diagonal complex matrix to be projected.
    
    Returns
    -------
    np.ndarray
        An N×N unitary diagonal matrix (closest to D in Frobenius norm).
    """
    N = D.shape[0]
    if D.shape[1] != N:
        raise ValueError("Input matrix must be square.")
    
    diag_mask = np.eye(D.shape[0], dtype=bool)
    if not np.allclose(D[~diag_mask], 0):
        raise ValueError("Input matrix must be (almost) diagonal.")

    D_proj = np.zeros((N, N), dtype=complex)
    for i in range(N):
        phase = np.angle(D[i, i])
        D_proj[i, i] = np.exp(1j * phase)
    return D_proj

def nullify_row(U, i, j, m, n):
    """Calculate beam splitter parameters to nullify a matrix row element.
    
    Parameters
    ----------
    U : np.ndarray
        The unitary matrix to be transformed.
    i : int
        The row index of the element to be nullified.
    j : int
        The column index of the element to be nullified.
    m : int
        The first row index for the beam splitter.
    n : int
        The second row index for the beam splitter.
    
    Returns
    -------
    tuple of (float, float)
        A tuple (phi, theta) containing the phase parameter and mixing angle.
    
    Raises
    ------
    ValueError
        If i is not equal to either m or n.
    """
    
    if i not in (m, n):
        raise ValueError("Row index i must be either m or n.")

    u_m = U[m, j]
    u_n = U[n, j]

    if i == m:
        # nullify row m
        theta = np.arctan2(np.abs(u_m), np.abs(u_n))
        phi = np.angle(u_n) - np.angle(u_m)
    else:
        # nullify row n
        theta = np.arctan2(np.abs(u_n), np.abs(u_m))
        phi = np.pi + np.angle(u_n) - np.angle(u_m)

    return phi, theta

def nullify_column(U, i, j, m, n):
    """Calculate beam splitter parameters to nullify a matrix column element.
    
    Parameters
    ----------
    U : np.ndarray
        The unitary matrix to be transformed.
    i : int
        The row index of the element to be nullified.
    j : int
        The column index of the element to be nullified.
    m : int
        The first column index for the beam splitter.
    n : int
        The second column index for the beam splitter.
    
    Returns
    -------
    tuple of (float, float)
        A tuple (phi, theta) containing the phase parameter and mixing angle.
    
    Raises
    ------
    ValueError
        If j is not equal to either m or n.
    """
    
    if j not in (m, n):
        raise ValueError("Column index j must be either m or n.")

    u_m = U[i, m]
    u_n = U[i, n]

    if j == m:
        # nullify column m
        theta = np.arctan2(np.abs(u_m), np.abs(u_n))
        phi = np.angle(u_m) - np.angle(u_n)
    else:
        # nullify column n
        theta = np.arctan2(np.abs(u_n), np.abs(u_m))
        phi = np.pi + np.angle(u_m) - np.angle(u_n)

    return phi, theta

alpha = np.pi/4
Ur = np.array([[np.cos(alpha), -np.sin(alpha)],
              [np.sin(alpha),  np.cos(alpha)]], dtype=complex)

omega = np.exp(2j * np.pi / 3)
Uf = (1/np.sqrt(3)) * np.array([[1, 1, 1],
                               [1, omega, omega**2],
                               [1, omega**2, omega**4]], dtype=complex)

# print(np.round(project_U2(nullify_row(Ur, 0, 1, 0, 1))@Ur, 10))
# print(np.round(project_U2(nullify_row(Uf, 0, 1, 0, 1))@Uf,10))

def clements_decomposition(U, project=True):
    """Decomposes a unitary matrix into beam splitters and phase shifters.
    
    Implements the Clements decomposition algorithm to decompose an N×N unitary
    matrix into a sequence of elementary beam splitter operations and phase shifts
    (and their inverses).
    
    Parameters
    ----------
    U : np.ndarray
        An N×N unitary matrix to be decomposed (complex entries).
    project : bool, optional
        Whether to project intermediate 2×2 matrices onto U(2) and the final diagonal
        matrix for numerical stability.
        Default is True.
    
    Returns
    -------
    tuple of ((list of tuple, list of tuple), np.ndarray)
        Contains, in order:
        list of tuple
            A list of tuples (m, n, phi, theta) representing the left decomposition elements,
            in the order in which they are multiplied to D (the resulting diagonal matrix)
            in order to give back U.
        list of tuple
            A list of tuples (m, n, phi, theta) representing the right decomposition elements,
            in the same order as the left_decomposition.
        np.ndarray
            The resulting diagonal unitary matrix D after decomposition.
    
    Raises
    ------
    ValueError
        If the input matrix is not square.
    """
    N = U.shape[0]
    if U.shape[1] != N:
        raise ValueError("Input matrix must be square.")

    Ucopy = U.copy()

    # right decomposistion elements, i.e. the ones that multiply U from the right
    right_decomposition = []
    # left decomposition elements, i.e. the ones that multiply U from the left
    # (inverses of beam splitters)
    left_decomposition = []

    for d in range(1, N): # loop over diagonals
        for k in range(d):
            if d % 2 == 1: # odd diagonal (odd in mathematical notations)
                (phi, theta) = nullify_column(Ucopy, N-1-k, d-1-k, d-1-k, d-k)
                Tmn_inv = inverse_T(d-1-k, d-k, phi, theta, N)
                if project:
                    Ucopy = Ucopy @ project_U2(Tmn_inv)
                else:
                    Ucopy = Ucopy @ Tmn_inv
                right_decomposition.append((d-1-k, d-k, phi, theta))
            else: # even diagonal (even in mathematical notations)
                (phi, theta) = nullify_row(Ucopy, N-d+k, k, N-d+k-1, N-d+k)
                Tmn = T(N-d+k-1, N-d+k, phi, theta, N)
                if project:
                    Ucopy = project_U2(Tmn) @ Ucopy
                else:
                    Ucopy = Tmn @ Ucopy
                left_decomposition.append((N-d+k-1, N-d+k, phi, theta))
    decomposition = (left_decomposition[::-1], right_decomposition[::-1])
    # print(np.round(Ucopy, 5))
    # print(decomposition)
    return (decomposition, project_D(Ucopy)) if project else (decomposition, Ucopy)

def clements_invert_left(D, left_decomposition, project=True):
    """Modifies the celements left decomposition in order to get rid of the inverses.
    
    Uses a trick to move all the inverses to the right of the decomposition, c.f. clements et al.
    
    Parameters
    ----------
    D : np.ndarray
        An N×N diagonal unitary matrix, represents U after decomposition.
    left_decomposition : list of tuple
        A list of tuples (m, n, phi, theta) representing the left decomposition elements.
    project : bool, optional
        Whether to project intermediate 2×2 matrices onto U(2) and the final diagonal
        matrix for numerical stability.
        Default is True.
    
    Returns
    -------
    list of tuple
        A list of tuples (m, n, phi, theta) representing the final decomposition elements,
        in the order in which they are multiplied to D'.
    np.ndarray
        The resulting diagonal unitary matrix D' after modification.
    
    Raises
    ------
    ValueError
        If the input matrix is not square.
    """
    Dprime = D.copy()
    N = D.shape[0]
    if project:
        Dprime = project_D(D)

    inverse_ldecomp = []
    for m, n, phi, theta in left_decomposition:
        dm = - np.exp(- 1j * phi) * Dprime[n, n]
        phi_prime = np.pi + np.angle(Dprime[m, m]) - np.angle(Dprime[n, n])
        Dprime[m, m] = dm
        if project:
            Dprime = project_D(Dprime)
        inverse_ldecomp.append((m, n, phi_prime, theta))
    return inverse_ldecomp[::-1], Dprime

def full_clements(U, project=True):
    """Performs the full Clements decomposition of a unitary matrix.
    
    Combines the decomposition and inversion of the left decomposition to yield
    a full sequence of beam splitters and phase shifters that implement the
    original unitary matrix.
    
    Parameters
    ----------
    U : np.ndarray
        An N×N unitary matrix to be decomposed (complex entries).
    project : bool, optional
        Whether to project intermediate 2×2 matrices onto U(2) and the final diagonal
        matrix for numerical stability.
        Default is True.
    
    Returns
    -------
    list of tuple
        A list of tuples (m, n, phi, theta) representing the full decomposition elements.
    
    Raises
    ------
    ValueError
        If the input matrix is not square.
    """
    
    if U.shape[0] != U.shape[1]:
        raise ValueError("Input matrix must be square.")
    if not np.allclose(U.conj().T @ U, np.eye(U.shape[0])):
        raise ValueError("Input matrix must be unitary.")
    
    decomposition, D = clements_decomposition(U, project=project)
    left_decomposition, right_decomposition = decomposition
    inverted_left, Dfinal = clements_invert_left(D, left_decomposition, project=project)
    full_decomposition = inverted_left + right_decomposition
    return full_decomposition, Dfinal



# clements_decomposition(random_unitary(4))
# Ur = random_unitary(4)
# phi, theta = nullify_row(Ur, 1, 0, 1, 2)
# print(np.round(project_U2(T(1, 2, phi, theta, 4)@Ur), 10))
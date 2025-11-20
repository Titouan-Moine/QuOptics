import numpy as np
from rnd_unitary import random_unitary

def T(m, n, phi, theta, N):
    """
    Constructs a beam splitter matrix acting on modes m and n with parameters phi and theta
    in an N-dimensional space.

    Parameters:
    m (int): The first mode index.
    n (int): The second mode index.
    phi (float): The phase parameter.
    theta (float): The mixing angle.
    N (int): The total number of modes.

    Returns:
    np.ndarray: The beam splitter matrix.
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
    """
    Constructs the inverse of a beam splitter matrix acting on modes m and n with parameters phi and theta
    in an N-dimensional space.

    Parameters:
    m (int): The first mode index.
    n (int): The second mode index.
    phi (float): The phase parameter.
    theta (float): The mixing angle.
    N (int): The total number of modes.

    Returns:
    np.ndarray: The inverse beam splitter matrix.
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
    """Projects a 2x2 matrix A onto the unitary group U(2).

    Args:
        A (2x2 complex matrix): input matrix to be projected

    Returns:
        2x2 complex matrix: projected unitary matrix
    """
    U, S, Vh = np.linalg.svd(A)
    return U @ Vh

def nullify_row(U, i, j, m, n):
    """_summary_

    Args:
        U (int matrix): unitary matrix to be transformed
        i (int): row index to be nullified
        j (int): column index to be nullified
        m (int): first row index for the beam splitter
        n (int): second row index for the beam splitter

    Raises:
        ValueError: if i is not m or n

    Returns:
        _type_: np.ndarray: beam splitter matrix that nullifies U[i,j]
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
    """_summary_

    Args:
        U (int matrix): unitary matrix to be transformed
        i (int): row index to be nullified
        j (int): column index to be nullified
        m (int): first column index for the beam splitter
        n (int): second column index for the beam splitter

    Raises:
        ValueError: if j is not m or n
    Returns:
        _type_: np.ndarray: inverse of the beam splitter matrix that nullifies U[i,j]
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

def clements_scheme(U):
    """
    Decomposes a given unitary matrix U into a sequence of beam splitters and phase shifters
    following the Clements scheme.

    Parameters:
    U (np.ndarray): The unitary matrix to be decomposed.

    Returns:
    list: A list of tuples representing the beam splitters and phase shifters.
          Each tuple contains (m, n, phi, theta) where phi and theta are the Euler angles
          and m and n are the mode indices.
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
                #Ucopy = Ucopy @ project_U2(Tmn_inv)
                Ucopy = Ucopy @ Tmn_inv
                left_decomposition.append((d-1-k, d-k, phi, theta))
            else: # even diagonal (even in mathematical notations)
                (phi, theta) = nullify_row(Ucopy, N-d+k, k, N-d+k-1, N-d+k)
                Tmn = T(N-d+k-1, N-d+k, phi, theta, N)
                #Ucopy = project_U2(Tmn) @ Ucopy
                Ucopy = Tmn @ Ucopy
                right_decomposition.append((N-d+k-1, N-d+k, phi, theta))
    decomposition = right_decomposition[::-1] + left_decomposition
    print(np.round(Ucopy, 5))
    print(decomposition)
    return decomposition

clements_scheme(random_unitary(4))
# Ur = random_unitary(4)
# phi, theta = nullify_row(Ur, 1, 0, 1, 2)
# print(np.round(project_U2(T(1, 2, phi, theta, 4)@Ur), 10))
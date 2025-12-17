"""Fock amplitude module. Provides functions to compute the amplitudes of Fock states
after passing through a linear optical network.
"""
import math
import numpy as np
from scipy.special import gammaln
# from sympy.physics.quantum.spin import Rotation
from ryser.permanent import ryser, ryser_gray, ryser_hyperrect, ryser_hyperrect_gray, repeat_matrix
from rnd_module import random_unitary
from clements_scheme.clements_scheme import T

def fock_amplitude_ryser(U, vecn, vecm):
    """Compute the Fock state amplitude using Ryser's algorithm.

    Parameters
    ----------
    U : np.ndarray
        The unitary matrix representing the linear optical network.
    vecn : np.ndarray
        An array containing the input Fock state occupation numbers.
    vecm : np.ndarray
        An array containing the output Fock state occupation numbers.

    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    """
    if vecn.sum() != vecm.sum():
        raise ValueError("The sum of input and output occupation numbers must be equal.")

    log_pref = 0.5 * (np.sum(gammaln(vecn + 1)) + np.sum(gammaln(vecm + 1)))
    pref = np.exp(log_pref)

    return ryser(repeat_matrix(U, vecn, vecm)) / pref

def fock_amplitude_ryser_gray(U, vecn, vecm):
    """Compute the Fock state amplitude using Ryser's algorithm with Gray code optimization.

    Parameters
    ----------
    U : np.ndarray
        The unitary matrix representing the linear optical network.
    vecn : np.ndarray
        An array containing the input Fock state occupation numbers.
    vecm : np.ndarray
        An array containing the output Fock state occupation numbers.

    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    """

    if vecn.sum() != vecm.sum():
        raise ValueError("The sum of input and output occupation numbers must be equal.")

    log_pref = 0.5 * (np.sum(gammaln(vecn + 1)) + np.sum(gammaln(vecm + 1)))
    pref = np.exp(log_pref)

    return ryser_gray(repeat_matrix(U, vecn, vecm)) / pref

def fock_amplitude_ryser_hyperrect(U, vecn, vecm):
    """Compute the Fock state amplitude using hyperrectangular Ryser algorithm.

    Parameters
    ----------
    U : np.ndarray
        The unitary matrix representing the linear optical network.
    vecn : np.ndarray
        An array containing the input Fock state occupation numbers.
    vecm : np.ndarray
        An array containing the output Fock state occupation numbers.

    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    """

    if vecn.sum() != vecm.sum():
        raise ValueError("The sum of input and output occupation numbers must be equal.")

    log_pref = 0.5 * (np.sum(gammaln(vecn + 1)) + np.sum(gammaln(vecm + 1)))
    pref = np.exp(log_pref)

    return ryser_hyperrect(U, vecn, vecm) / pref

def fock_amplitude_ryser_hyperrect_gray(U, vecn, vecm):
    """Compute the Fock state amplitude using hyperrectangular Ryser algorithm with Gray code optimization.

    Parameters
    ----------
    U : np.ndarray
        The unitary matrix representing the linear optical network.
    vecn : np.ndarray
        An array containing the input Fock state occupation numbers.
    vecm : np.ndarray
        An array containing the output Fock state occupation numbers.

    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    """

    if vecn.sum() != vecm.sum():
        raise ValueError("The sum of input and output occupation numbers must be equal.")

    log_pref = 0.5 * (np.sum(gammaln(vecn + 1)) + np.sum(gammaln(vecm + 1)))
    pref = np.exp(log_pref)

    return ryser_hyperrect_gray(U, vecn, vecm) / pref

# def fock_amplitude_two_mode_bs(U, n1, n2, m1, m2, phi=None):
#     """Compute the Fock state amplitude for a two-mode beam splitter.

#     Parameters
#     ----------
#     U : np.ndarray
#         The 2x2 unitary matrix representing the linear optical network.
#     n1 : int
#         The occupation number of the first input mode.
#     n2 : int
#         The occupation number of the second input mode.
#     m1 : int
#         The occupation number of the first output mode.
#     m2 : int
#         The occupation number of the second output mode.
#     phi : float, optional
#         The phase associated with the beam splitter. If None, it will be inferred from U.

#     Returns
#     -------
#     complex
#         The amplitude of the transition from input Fock state (n1, n2) to output Fock state (m1, m2).
#     """
#     if n1 + n2 != m1 + m2:
#         raise ValueError("The total number of photons must be conserved.")
    
#     s = -U[0, 1]
#     c = U[1, 1]
    
#     if phi is None:
#         if np.isclose(U[0, 0], 0):
#             phi = np.angle(U[1,0]) + np.angle(U[0,1])
#         else:
#             phi = np.angle(U[0,0]) - np.angle(U[1,1])

#     k_start = max(0, m1 - n2)
#     coeff = (-1)**(n1 - k_start) * s**(n1 + m1 - 2*k_start) * c**(n2 - m1 + 2*k_start)\
#         / (math.factorial(n1 - k_start) * math.factorial(n2 - m1 + k_start)
#            * math.factorial(k_start) * math.factorial(m1 - k_start))

#     amplitude = coeff
#     for k in range(k_start+1, min(n1, m1)+1):
#         coeff *= - c**2 * (n1 - k + 1) * (m1 - k + 1) / (s**2 * k * (n2 - m1 + k))
#         amplitude += coeff

#     amplitude *= np.exp(1j * m1 * phi)

#     return amplitude

def fock_amplitude_bs(p, q, phi, theta, vecn, vecm):
    """Compute the Fock state amplitude for a beam splitter using the two-mode beam splitter function.

    Parameters
    ----------
    p : int
        The index of the first mode the beam splitter acts on.
    q : int
        The index of the second mode the beam splitter acts on.
    phi : float
        The phase associated with the beam splitter.
    theta : float
        The mixing angle of the beam splitter.
    vecn : np.ndarray
        An array containing the input Fock state occupation numbers.
    vecm : np.ndarray
        An array containing the output Fock state occupation numbers.

    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    """
    if vecn.sum() != vecm.sum():
        raise ValueError("The sum of input and output occupation numbers must be equal.")

    id_indices = [i for i in range(len(vecn)) if i not in (p, q)]
    if (vecn[id_indices] != vecm[id_indices]).any():
        return 0 # no photons can be exchanged in other modes

    s = np.sin(theta)
    c = np.cos(theta)
    n1 = vecn[p]
    n2 = vecn[q]
    m1 = vecm[p]
    m2 = vecm[q]

    k_start = max(0, m1 - n2)
    coeff = (-1)**(n1 - k_start) * s**(n1 + m1 - 2*k_start) * c**(n2 - m1 + 2*k_start)\
        / (math.factorial(n1 - k_start) * math.factorial(n2 - m1 + k_start)
           * math.factorial(k_start) * math.factorial(m1 - k_start))

    amplitude = coeff
    for k in range(k_start+1, min(n1, m1)+1):
        coeff *= - c**2 * (n1 - k + 1) * (m1 - k + 1) / (s**2 * k * (n2 - m1 + k))
        amplitude += coeff

    amplitude *= np.exp(1j * m1 * phi)
    amplitude *= math.sqrt(math.factorial(n1) * math.factorial(n2) *
                       math.factorial(m1) * math.factorial(m2))

    return amplitude

# def fock_amplitude_two_mode_bs_wigner(U, n1, n2, m1, m2):
#     """Compute the Fock state amplitude for a two-mode beam splitter using Wigner d-matrix.

#     Parameters
#     ----------
#     U : np.ndarray
#         The 2x2 unitary matrix representing the linear optical network.
#     n1 : int
#         The occupation number of the first input mode.
#     n2 : int
#         The occupation number of the second input mode.
#     m1 : int
#         The occupation number of the first output mode.
#     m2 : int
#         The occupation number of the second output mode.

#     Returns
#     -------
#     complex
#         The amplitude of the transition from input Fock state (n1, n2) to output Fock state (m1, m2).
#     """
#     if n1 + n2 != m1 + m2:
#         raise ValueError("The total number of photons must be conserved.")

#     theta = 2 * np.arccos(np.abs(U[1,1]))
#     phi = np.angle(U[0,0]) - np.angle(U[1,1])

#     j = (n1 + n2) / 2
#     m = (n1 - n2) / 2
#     mp = (m1 - m2) / 2

#     d = T.wigner_d(j, mp, m, theta)

#     amplitude = d * (np.abs(U[1,1])**(m1 + m2)) * (np.abs(U[0,1])**(n1 + n2))

def fock_amplitude(U, vecn, vecm, method='ryser'):
    """Compute the Fock state amplitude using the specified method.

    Parameters
    ----------
    U : np.ndarray
        The unitary matrix representing the linear optical network.
    vecn : np.ndarray
        An array containing the input Fock state occupation numbers.
    vecm : np.ndarray
        An array containing the output Fock state occupation numbers.
    method : str, optional
        The method to use for computation. Options are 'ryser', 'ryser_gray',
        'ryser_hyperrect', 'ryser_hyperrect_gray'. Default is 'ryser'.
    
    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    """
    if method == 'ryser':
        return fock_amplitude_ryser(U, vecn, vecm)
    elif method == 'ryser_gray':
        return fock_amplitude_ryser_gray(U, vecn, vecm)
    elif method == 'ryser_hyperrect':
        return fock_amplitude_ryser_hyperrect(U, vecn, vecm)
    elif method == 'ryser_hyperrect_gray':
        return fock_amplitude_ryser_hyperrect_gray(U, vecn, vecm)
    else:
        raise ValueError(f"Unknown method: {method},\
                         try 'ryser', 'ryser_gray', 'ryser_hyperrect', or 'ryser_hyperrect_gray'.")

# U = random_unitary(4)
# vecn = np.array([1, 1, 2, 1])
# vecm = np.array([0, 2, 3, 0])
# amplitude = [fock_amplitude(U, vecn, vecm, method=meth) for meth in
#              ['ryser', 'ryser_gray', 'ryser_hyperrect', 'ryser_hyperrect_gray']]
# print("Fock state amplitudes for different methods:", amplitude)

# phi = np.pi / 7
# theta = np.pi / 6
# vecn = np.array([3, 4, 1, 2])
# vecm = np.array([2, 5, 1, 2])
# amplitude_bs = fock_amplitude_bs(0, 1, phi, theta, vecn, vecm)
# amplitude_bs_direct = fock_amplitude_ryser(T(0, 1, phi, theta, 4), vecn, vecm)
# print("Fock state amplitude for beam splitter:", amplitude_bs)
# print("Fock state amplitude for beam splitter (direct):", amplitude_bs_direct)
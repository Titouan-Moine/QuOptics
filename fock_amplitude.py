"""Fock amplitude module. Provides functions to compute the amplitudes of Fock states
after passing through a linear optical network.
"""
import math
import warnings
import numpy as np
from scipy.special import gammaln
# from sympy.physics.quantum.spin import Rotation
from ryser.permanent import ryser, ryser_gray, ryser_hyperrect, ryser_hyperrect_gray, glynn, glynn_gray, repeat_matrix
from rnd_module import random_unitary
from clements_scheme.clements_scheme import T

def enumerate_fock(n, N, indexed=True, check_value=True):
    """Generate all Fock states of N modes with a total of n photons
    recursively in lexicographic order.

    Parameters
    ----------
    n : int
        The number of photons.
    N : int
        The number of modes.
    indexed : bool, optional
        If True, return a dictionary mapping each Fock state to its index. Default is True.
    check_value : bool, optional
        If True, check that n is non-negative and N is positive. Default is True.

    Returns
    -------
    List[np.ndarray]
        A list of numpy arrays, each representing a Fock state.
    """
    if check_value:
        if n < 0 or N < 0:
            raise ValueError("Number of modes must be positive and number of photons must be non-negative.")

    if N == 1:
        return [np.array([n])]

    states = []
    for k in range(n + 1):
        for substate in enumerate_fock(n - k, N - 1, indexed=False):
            #print(substate)
            state = np.concatenate(([k], substate))
            states.append(state)

    return {np.array2string(state): i for i, state in enumerate(states)} if indexed else states

def fock_amplitude_ryser(U, vecn, vecm, check_photons=True):
    """Compute the Fock state amplitude using Ryser's algorithm.

    Parameters
    ----------
    U : np.ndarray
        The unitary matrix representing the linear optical network.
    vecn : np.ndarray
        An array containing the input Fock state occupation numbers.
    vecm : np.ndarray
        An array containing the output Fock state occupation numbers.
    check_photons : bool, optional
        If True, check that the total number of photons is conserved. Default is True.

    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    """
    if check_photons:
        if vecn.sum() != vecm.sum():
            raise ValueError("The sum of input and output occupation numbers must be equal.")

    log_pref = 0.5 * (np.sum(gammaln(vecn + 1)) + np.sum(gammaln(vecm + 1)))
    pref = np.exp(log_pref)

    return ryser(repeat_matrix(U, vecn, vecm)) / pref

def fock_amplitude_ryser_gray(U, vecn, vecm, check_photons=True):
    """Compute the Fock state amplitude using Ryser's algorithm with Gray code optimization.

    Parameters
    ----------
    U : np.ndarray
        The unitary matrix representing the linear optical network.
    vecn : np.ndarray
        An array containing the input Fock state occupation numbers.
    vecm : np.ndarray
        An array containing the output Fock state occupation numbers.
    check_photons : bool, optional
        If True, check that the total number of photons is conserved. Default is True.

    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    """
    if check_photons:
        if vecn.sum() != vecm.sum():
            raise ValueError("The sum of input and output occupation numbers must be equal.")

    log_pref = 0.5 * (np.sum(gammaln(vecn + 1)) + np.sum(gammaln(vecm + 1)))
    pref = np.exp(log_pref)

    return ryser_gray(repeat_matrix(U, vecn, vecm)) / pref

def fock_amplitude_ryser_hyperrect(U, vecn, vecm, check_photons=True):
    """Compute the Fock state amplitude using hyperrectangular Ryser algorithm.

    Parameters
    ----------
    U : np.ndarray
        The unitary matrix representing the linear optical network.
    vecn : np.ndarray
        An array containing the input Fock state occupation numbers.
    vecm : np.ndarray
        An array containing the output Fock state occupation numbers.
    check_photons : bool, optional
        If True, check that the total number of photons is conserved. Default is True.

    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    """
    if check_photons:
        if vecn.sum() != vecm.sum():
            raise ValueError("The sum of input and output occupation numbers must be equal.")

    log_pref = 0.5 * (np.sum(gammaln(vecn + 1)) + np.sum(gammaln(vecm + 1)))
    pref = np.exp(log_pref)

    return ryser_hyperrect(U, vecn, vecm) / pref

def fock_amplitude_ryser_hyperrect_gray(U, vecn, vecm, check_photons=True):
    """Compute the Fock state amplitude using hyperrectangular Ryser algorithm with Gray code optimization.

    Parameters
    ----------
    U : np.ndarray
        The unitary matrix representing the linear optical network.
    vecn : np.ndarray
        An array containing the input Fock state occupation numbers.
    vecm : np.ndarray
        An array containing the output Fock state occupation numbers.
    check_photons : bool, optional
        If True, check that the total number of photons is conserved. Default is True.

    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    """
    if check_photons:
        if vecn.sum() != vecm.sum():
            raise ValueError("The sum of input and output occupation numbers must be equal.")

    log_pref = 0.5 * (np.sum(gammaln(vecn + 1)) + np.sum(gammaln(vecm + 1)))
    pref = np.exp(log_pref)

    return ryser_hyperrect_gray(U, vecn, vecm) / pref

def fock_amplitude_glynn(U, vecn, vecm, check_photons=True):
    """Compute the Fock state amplitude using Glynn's algorithm.

    Parameters
    ----------
    U : np.ndarray
        The unitary matrix representing the linear optical network.
    vecn : np.ndarray
        An array containing the input Fock state occupation numbers.
    vecm : np.ndarray
        An array containing the output Fock state occupation numbers.
    check_photons : bool, optional
        If True, check that the total number of photons is conserved. Default is True.

    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    """
    if check_photons:
        if vecn.sum() != vecm.sum():
            raise ValueError("The sum of input and output occupation numbers must be equal.")

    log_pref = 0.5 * (np.sum(gammaln(vecn + 1)) + np.sum(gammaln(vecm + 1)))
    pref = np.exp(log_pref)

    return glynn(repeat_matrix(U, vecn, vecm)) / pref

def fock_amplitude_glynn_gray(U, vecn, vecm, check_photons=True):
    """Compute the Fock state amplitude using Glynn's algorithm with Gray code optimization.

    Parameters
    ----------
    U : np.ndarray
        The unitary matrix representing the linear optical network.
    vecn : np.ndarray
        An array containing the input Fock state occupation numbers.
    vecm : np.ndarray
        An array containing the output Fock state occupation numbers.
    check_photons : bool, optional
        If True, check that the total number of photons is conserved. Default is True.

    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    """
    if check_photons:
        if vecn.sum() != vecm.sum():
            raise ValueError("The sum of input and output occupation numbers must be equal.")

    log_pref = 0.5 * (np.sum(gammaln(vecn + 1)) + np.sum(gammaln(vecm + 1)))
    pref = np.exp(log_pref)

    return glynn_gray(repeat_matrix(U, vecn, vecm)) / pref

def fock_amplitude_bs(p, q, phi, theta, vecn, vecm, check_photons=True):
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
    check_photons : bool, optional
        If True, check that the total number of photons is conserved. Default is True.

    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    """
    if check_photons:
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

def fock_amplitude_ps(i, phi, vecn, vecm, check_photons=True):
    """Compute the Fock state amplitude for a phase shifter.

    Parameters
    ----------
    i : int
        The index of the mode the phase shifter acts on.
    phi : float
        The phase shift applied to the mode.
    vecn : np.ndarray
        An array containing the input Fock state occupation numbers.
    vecm : np.ndarray
        An array containing the output Fock state occupation numbers.
    check_photons : bool, optional
        If True, check that the total number of photons is conserved. Default is True.

    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    """
    if check_photons:
        if vecn.sum() != vecm.sum():
            warnings.warn("The sum of input and output occupation numbers are not equal,\
                the amplitude will be zero.", UserWarning)
    if (vecn != vecm).any():
        return 0 # no photons can be exchanged in any modes

    return np.exp(1j * vecn[i] * phi)

def fock_amplitude_multi_ps(phi, vecn, vecm, check_modes=True, check_photons=True):
    """Compute the Fock state amplitude for a multi-mode phase shifter.

    Parameters
    ----------
    phi : np.ndarray
        An array containing the phase shifts applied to each mode.
    vecn : np.ndarray
        An array containing the input Fock state occupation numbers.
    vecm : np.ndarray
        An array containing the output Fock state occupation numbers.
    check_modes : bool, optional
        If True, check that the length of phi matches the length of vecn and vecm. Default is True.
    check_photons : bool, optional
        If True, check that the total number of photons is conserved. Default is True.

    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    """
    if check_modes:
        if len(phi) != len(vecn) or len(phi) != len(vecm):
            raise ValueError("The length of phi must match the length of vecn and vecm.")
    if check_photons:
        if vecn.sum() != vecm.sum():
            raise Warning("The sum of input and output occupation numbers are not equal,\
                the amplitude will be zero.")
    if (vecn != vecm).any():
        return 0 # no photons can be exchanged in any modes

    amplitude = np.prod(np.exp(1j * vecn * phi))

    return amplitude

def fock_amplitude(U, vecn, vecm, method='ryser_gray', check=True):
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
        'ryser_hyperrect', 'ryser_hyperrect_gray', 'glynn', 'glynn_gray'. Default is 'ryser'.
    check : bool, optional
        If True, performs all checks of the selected method. Default is True.

    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    
    Remarks
    -------
    This function serves as a dispatcher to select the appropriate algorithm
    for computing the Fock state amplitude based on the specified method.
    Gray version of Ryser and hyperrectangular Ryser are almost always faster.
    """
    if method == 'ryser':
        return fock_amplitude_ryser(U, vecn, vecm, check_photons=check)
    elif method == 'ryser_gray':
        return fock_amplitude_ryser_gray(U, vecn, vecm, check_photons=check)
    elif method == 'ryser_hyperrect':
        return fock_amplitude_ryser_hyperrect(U, vecn, vecm, check_photons=check)
    elif method == 'ryser_hyperrect_gray':
        return fock_amplitude_ryser_hyperrect_gray(U, vecn, vecm, check_photons=check)
    elif method == 'glynn':
        return fock_amplitude_glynn(U, vecn, vecm, check_photons=check)
    elif method == 'glynn_gray':
        return fock_amplitude_glynn_gray(U, vecn, vecm, check_photons=check)
    else:
        raise ValueError(f"Unknown method: {method},\
                         try 'ryser', 'ryser_gray', 'ryser_hyperrect',\
                         'ryser_hyperrect_gray', 'glynn', or 'glynn_gray'.")

def fock_tensor(U, n_photons, method='ryser_gray', check=True):
    """Compute the Fock state amplitude tensor for all possible input and output Fock states
    with a total of n_photons.

    Parameters
    ----------
    U : np.ndarray
        The unitary matrix representing the linear optical network.
    n_photons : int
        The total number of photons.
    method : str, optional
        The method to use for computation. Options are 'ryser', 'ryser_gray',
        'ryser_hyperrect', 'ryser_hyperrect_gray'. Default is 'ryser'.
    check : bool, optional
        If True, performs all checks of the selected method. Default is True.
    
    Returns
    -------
    np.ndarray
        A tensor of shape (M, M) where M is the number of Fock states with n_photons
        in N modes, containing the amplitudes of transitions between all pairs of Fock states.
    """
    N = U.shape[0]
    fock_states = enumerate_fock(n_photons, N, indexed=False, check_value=False)
    nb_states = len(fock_states)
    tensor = np.zeros((nb_states, nb_states), dtype=complex)
    for i in range(nb_states):
        for j in range(nb_states):
            tensor[i, j] = fock_amplitude(U, fock_states[i], fock_states[j], method=method, check=check)
    
    return tensor

# U = np.eye(3)
# U = random_unitary(4)
# U = np.diag([1, 2, 3])
# tensor = fock_tensor(U, 3, method='ryser_gray', check=False)
# print(tensor)

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
"""Fock amplitude module. Provides functions to compute the amplitudes of Fock states
after passing through a linear optical network.
"""
import math
import warnings
import numpy as np
from scipy.special import gammaln
import sparse
import quimb as qb
import quimb.tensor as qtn
# from sympy.physics.quantum.spin import Rotation
from ryser.permanent import ryser, ryser_gray, ryser_hyperrect, ryser_hyperrect_gray, glynn, glynn_gray, repeat_matrix
from rnd_module import random_unitary
from clements_scheme.clements_scheme import T, full_clements

def enumerate_fock(n, N, indexed=False, check_value=True):
    """Generate all Fock states of N modes with a total of n photons
    recursively in lexicographic order.

    Parameters
    ----------
    n : int
        The number of photons.
    N : int
        The number of modes.
    indexed : bool, optional
        If True, return a dictionary mapping each Fock state to its index. Default is False.
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
        return {"n": 0} if indexed else [np.array([n])]

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
    if vecn.sum() == 0:
        return 1.0 + 0.0j  # amplitude for vacuum state

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
    if vecn.sum() == 0:
        return 1.0 + 0.0j  # amplitude for vacuum state

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
    if vecn.sum() == 0:
        return 1.0 + 0.0j  # amplitude for vacuum state

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
    if vecn.sum() == 0:
        return 1.0 + 0.0j  # amplitude for vacuum state

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
    if vecn.sum() == 0:
        return 1.0 + 0.0j  # amplitude for vacuum state

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
    if vecn.sum() == 0:
        return 1.0 + 0.0j  # amplitude for vacuum state

    log_pref = 0.5 * (np.sum(gammaln(vecn + 1)) + np.sum(gammaln(vecm + 1)))
    pref = np.exp(log_pref)

    return glynn_gray(repeat_matrix(U, vecn, vecm)) / pref

def fock_amplitude_bs(phi, theta, invec, outvec, check_photons=True):
    """Compute the Fock state amplitude for a beam splitter using the two-mode beam splitter function.

    Parameters
    ----------
    phi : float
        The phase associated with the beam splitter.
    theta : float
        The mixing angle of the beam splitter.
    invec : np.ndarray
        An array containing the input Fock state occupation numbers for the corresponding modes.
    outvec : np.ndarray
        An array containing the output Fock state occupation numbers for the corresponding modes.
    check_photons : bool, optional
        If True, check that the total number of photons is conserved. Default is True.

    Returns
    -------
    complex
        The amplitude of the transition from input Fock state vecn to output Fock state vecm.
    """
    if check_photons:
        if invec.sum() != outvec.sum():
            warnings.warn("The sum of input and output occupation numbers are not equal,\
                the amplitude will be zero.", UserWarning)

    if invec.sum() != outvec.sum():
        return 0 # no photons can be exchanged in other modes

    s = np.sin(theta)
    c = np.cos(theta)
    
    i = invec[0]
    j = invec[1]
    k = outvec[0]
    l = outvec[1]

    p_start = max(0, k - j)
    term = (-1)**(i - p_start) * s**(i + k - 2*p_start) * c**(j - k + 2*p_start)\
            * math.comb(i, p_start) * math.comb(j, k - p_start)

    amplitude = term
    for p in range(p_start+1, min(i, k)+1):
        term *= - c**2 * (i - p + 1) * (k - p + 1) / (s**2 * p * (j - k + p))
        amplitude += term

    amplitude *= np.exp(1j * k * phi)
    amplitude *= math.sqrt(math.factorial(k) * math.factorial(l) /
                       (math.factorial(i) * math.factorial(j)))

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

def fock_amplitude(U, vecn, vecm, method='ryser_gray', check=False):
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

def fock_tensor(U, n_photons, sparse_tensor=True, method='ryser_gray', check=False):
    """Compute the Fock state amplitude tensor for all possible input and output Fock states
    with a total of n_photons. In order to have a coherent tensor structure, it is filled with 0s
    for irrelevant transitions (when there are more than n_photons photons).

    Parameters
    ----------
    U : np.ndarray
        The unitary matrix representing the linear optical network.
    n_photons : int
        The total number of photons.
    sparse_tensor : bool, optional
        If True, returns a sparse tensor. Default is True.
    method : str, optional
        The method to use for computation. Options are 'ryser', 'ryser_gray', 'glynn', 'glynn_gray',
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
    # fock_states = enumerate_fock(n_photons, N, indexed=False, check_value=False)
    # nb_states = len(fock_states)
    # tensor = np.zeros((nb_states, nb_states), dtype=complex)
    # for i in range(nb_states):
    #     for j in range(nb_states):
    #         tensor[i, j] = fock_amplitude(U, fock_states[i], fock_states[j], method=method, check=check)
    if sparse_tensor:
        coords = []
        data = []

        for n in range(n_photons + 1):
            fock_basis = enumerate_fock(n, N, check_value=check)
            for invec in fock_basis:
                for outvec in fock_basis:
                    amp = fock_amplitude(U, invec, outvec, method=method, check=check)
                    
                    if abs(amp) > 1e-15:  # only store non-zero amplitudes
                        # The index is the concatenation of invec and outvec
                        coords.append(list(invec) + list(outvec))
                        data.append(amp)
        
        # verify if coords is empty to avoid errors in sparse.COO
        if not coords:
            return sparse.COO([], [], shape=((n_photons + 1,) * (2 * N)))
        
        coords_array = np.array(coords).T
        shape = (n_photons + 1,) * (2 * N)
        return sparse.COO(coords_array, data, shape=shape)
    else:
        tensor = np.zeros((n_photons + 1,) * (2 * N), dtype=complex)
        for n in range(n_photons + 1):
            fock_basis = enumerate_fock(n, N, check_value=check)
            for invec in fock_basis:
                for outvec in fock_basis:
                    index_in = tuple(invec)
                    index_out = tuple(outvec)
                    tensor[index_in + index_out] = fock_amplitude(U, invec, outvec, method=method, check=check)
    return tensor

def fock_tensor_bs(phi, theta, n_photons, sparse_tensor=True, check=False):
    """Compute the Fock state amplitude tensor for all possible input and output Fock states
    with a total of n_photons for a beam splitter.

    Parameters
    ----------
    phi : float
        The phase associated with the beam splitter.
    theta : float
        The mixing angle of the beam splitter.
    n_photons : int
        The total number of photons.
    sparse_tensor : bool, optional
        If True, returns a sparse tensor. Default is True.
    check : bool, optional
        If True, performs all checks of the selected method. Default is True.
    
    Returns
    -------
    np.ndarray
        A tensor of shape (M, M) where M is the number of Fock states with n_photons
        in 2 modes, containing the amplitudes of transitions between all pairs of Fock states.
    """
    if sparse_tensor:
        coords = []
        data = []

        for n in range(n_photons + 1):
            fock_basis = enumerate_fock(n, 2, check_value=check)
            for invec in fock_basis:
                for outvec in fock_basis:
                    amp = fock_amplitude_bs(phi, theta, invec, outvec, check_photons=check)
                    
                    if abs(amp) > 1e-15:  # only store non-zero amplitudes
                        # The index is the concatenation of invec and outvec
                        coords.append(list(invec) + list(outvec))
                        data.append(amp)
        
        # verify if coords is empty to avoid errors in sparse.COO
        if not coords:
            return sparse.COO([], [], shape=((n_photons + 1,) * 4))
        
        coords_array = np.array(coords).T
        shape = (n_photons + 1,) * 4
        return sparse.COO(coords_array, data, shape=shape)
    else:
        tensor = np.zeros((n_photons+1,) * 4, dtype=complex)
        for n in range(n_photons + 1):
            fock_basis = enumerate_fock(n, 2, check_value=check)
            for invec in fock_basis:
                for outvec in fock_basis:
                    idx = tuple(invec) + tuple(outvec)
                    tensor[idx] =\
                    fock_amplitude_bs(phi, theta, invec, outvec, check_photons=check)
    return tensor

def fock_tensor_ps(phi, n_photons, sparse_tensor=True, check=False):
    """Compute the Fock state amplitude tensor for all possible input and output Fock states
    with a total of n_photons for a phase shifter.

    Parameters
    ----------
    phi : float
        The phase associated with the phase shifter.
    n_photons : int
        The total number of photons.
    sparse_tensor : bool, optional
        If True, returns a sparse tensor. Default is True.
    check : bool, optional
        If True, performs all checks of the selected method. Default is True.
    
    Returns
    -------
    np.ndarray
        A tensor of shape (n_photons+1, n_photons+1), containing the amplitudes of
        transitions between all single mode of Fock states.
    """
    k_indices = np.arange(n_photons + 1)
    amplitudes = np.exp(1j * k_indices * phi)

    if sparse_tensor:
        coords = np.vstack([k_indices, k_indices])
        return sparse.COO(coords, amplitudes, shape=(n_photons + 1, n_photons + 1))
    else:
        return np.diag(amplitudes)

def fock_tensor_multi_ps(phi_list, n_photons, sparse_tensor=True):
    """
    Generates a rank-2p Fock space tensor for a multi-mode Phase Shifter across p modes.

    The Phase Shifter operator is diagonal in the Fock basis. For an input state
    |n1, n2, ..., np>, it applies a total phase of exp(i * sum(nj * phij)).

    Single PS tensors are generally preferred.

    Parameters
    ----------
    phi_list : list or np.ndarray
        List of phase shift angles [phi_1, phi_2, ..., phi_p] for the p modes.
    n_photons : int
        Maximum total number of photons (truncation level).
    sparse_tensor : bool, optional
        If True, returns a sparse.COO tensor. Default is True.

    Returns
    -------
    sp.COO or np.ndarray
        A tensor of shape (n_photons + 1, ..., n_photons + 1) with 2p indices.
        Indices are ordered as (in_0, ..., in_p-1, out_0, ..., out_p-1).
    """
    p = len(phi_list)
    phi_list = np.array(phi_list)
    
    # Shape of the tensor: (n+1) repeated 2*p times
    # p legs for input and p legs for output
    shape = (n_photons + 1,) * (2 * p)
    
    coords = []
    data = []

    # Iterate through each photon sector from 0 to n_photons
    # to respect the global photon number conservation (sum nj <= n_photons)
    for n in range(n_photons + 1):
        # enumerate_fock(n, p) generates all Fock states with total n photons in p modes
        fock_basis = enumerate_fock(n, p) 
        
        for vec in fock_basis:
            # vec is a tuple/list of occupation numbers: (n_1, n_2, ..., n_p)
            vec_np = np.array(vec)
            
            # The total phase is the dot product of occupations and phase angles
            # For the vacuum (0,0,...), total_phase is 0, so amplitude is exp(0) = 1
            total_phase = np.sum(vec_np * phi_list)
            amplitude = np.exp(1j * total_phase)
            
            # Since the operator is diagonal, input indices == output indices
            # The full index is the concatenation: [in_1, ..., in_p, out_1, ..., out_p]
            full_idx = list(vec) + list(vec)
            coords.append(full_idx)
            data.append(amplitude)

    if sparse_tensor:
        # Construct the sparse tensor in Coordinate (COO) format
        # coords needs to be transposed to shape (2*p, number_of_non_zero_elements)
        return sparse.COO(np.array(coords).T, data, shape=shape)
    else:
        # Construct a standard dense NumPy array
        tensor = np.zeros(shape, dtype=complex)
        for c, d in zip(coords, data):
            tensor[tuple(c)] = d
        return tensor

def clements_to_fock_network(BS_list, D, n_photons, sparse_tensor=True, check=False):
    """Construct the tensor network (with quimb) of a clements scheme in the Fock basis.
    
    Parameters
    ----------
    BS_list : list of tuples
        A list of beam splitter parameters (phi, theta) for each beam splitter in the Clements scheme.
    D : np.ndarray
        A diagonal unitary matrix representing the phase shifts in the Clements scheme.
    n_photons : int
        The total number of photons.
    sparse_tensor : bool, optional
        If True, returns a sparse tensor. Default is True.
    check : bool, optional
        If True, performs all checks of the selected method. Default is True.
    
    Returns
    -------
    np.ndarray
        The constructed tensor network in the Fock basis.
    """
    
    N = D.shape[0]
    tensors = []

    # Beam splitters
    for (mode1, mode2, phi, theta) in BS_list:
        bs_tensor = fock_tensor_bs(phi, theta, n_photons, sparse_tensor=sparse_tensor, check=check)
        bs_qtn = qtn.Tensor(bs_tensor, inds=(f'in_{mode1}', f'in_{mode2}', f'out_{mode1}', f'out_{mode2}'))
        tensors.append(bs_qtn)

    # Phase shifts
    for mode in range(N):
        phi = np.angle(D[mode, mode])
        ps_tensor = fock_tensor_ps(phi, n_photons, sparse_tensor=sparse_tensor, check=check)
        ps_qtn = qtn.Tensor(ps_tensor, inds=(f'in_{mode}', f'out_{mode}'))
        tensors.append(ps_qtn)

    # Create the tensor network
    tn = qtn.TensorNetwork(tensors)

    return tn

def clements_fock_tensor(BS_list, D, n_photons=None, sparse_tensor=True, check=False):
    """Compute the Fock state amplitude tensor for a Clements scheme.

    Parameters
    ----------
    BS_list : list of tuples
        A list of beam splitter parameters (phi, theta) for each beam splitter in the Clements scheme.
    D : np.ndarray
        A diagonal unitary matrix representing the phase shifts in the Clements scheme.
    n_photons : int
        The total number of photons.
    sparse_tensor : bool, optional
        If True, returns a sparse tensor. Default is True.
    check : bool, optional
        If True, performs all checks of the selected method. Default is True.
    
    Returns
    -------
    np.ndarray
        The Fock state amplitude tensor for the Clements scheme.
    """
    N = D.shape[0]
    if n_photons is None:
        n_photons = math.ceil(N / 10)  # default to number of modes

    tn = clements_to_fock_network(BS_list, D, n_photons, sparse_tensor=sparse_tensor, check=check)
    output_inds = [f'in_{mode}' for mode in range(N)] + [f'out_{mode}' for mode in range(N)]
    result = tn.contract(all, output_inds=output_inds, optimize='greedy', backend='sparse')
    return result.data
    

if __name__ == "__main__":
    # tests
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

    phi = 0*np.pi / 7
    theta = np.pi / 4
    vecn = np.array([1, 0])
    vecm = np.array([0, 1])
    amplitude_bs = fock_amplitude_bs(phi, theta, vecn, vecm)
    amplitude_bs_direct = fock_amplitude_ryser(T(0, 1, phi, theta, 2), vecn, vecm)
    print("Fock state amplitude for beam splitter:", amplitude_bs)
    print("Fock state amplitude for beam splitter (direct):", amplitude_bs_direct)
    bs_tensor = fock_tensor_bs(phi, theta, 2, sparse_tensor=False)
    print("Fock state amplitude tensor for beam splitter:\n", bs_tensor)
    bs_tensor_direct = fock_tensor(T(0, 1, phi, theta, 2), 2, sparse_tensor=False)
    print("Fock state amplitude tensor for beam splitter (direct):\n", bs_tensor_direct)
    print("Difference between tensors:", np.max(np.abs(bs_tensor - bs_tensor_direct)))
    print("\n" + "="*60)
    print("COMPARISON TEST: bs_tensor vs bs_tensor_direct")
    print("="*60)
    
    # Tolérance pour la comparaison
    atol = 1e-10  # tolérance absolue
    rtol = 1e-8   # tolérance relative
    
    # Test avec numpy.allclose
    are_close = np.allclose(bs_tensor, bs_tensor_direct, atol=atol, rtol=rtol)
    print(f"\nTensors are close (atol={atol}, rtol={rtol}): {are_close}")
    
    if are_close:
        print("✅ Les deux tenseurs sont presque égaux !")
    else:
        print("❌ Les deux tenseurs diffèrent. Analyse des différences:")
        
        # Calculer les différences
        diff = np.abs(bs_tensor - bs_tensor_direct)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"\n  Différence maximale: {max_diff:.2e}")
        print(f"  Différence moyenne: {mean_diff:.2e}")
        print(f"  Nombre d'éléments différents (>atol): {np.sum(diff > atol)}")
        print(f"  Nombre total d'éléments: {diff.size}")
        
        # Trouver les indices où les tenseurs diffèrent significativement
        diff_indices = np.argwhere(diff > atol)
        
        print(f"\n  Indices où les tenseurs diffèrent (diff > {atol}):")
        print(f"  {'Index':<20} {'bs_tensor':<25} {'bs_tensor_direct':<25} {'Diff':<15}")
        print("  " + "-"*85)
        
        for idx in diff_indices[:20]:  # Limiter à 20 pour la lisibilité
            idx_tuple = tuple(idx)
            val1 = bs_tensor[idx_tuple]
            val2 = bs_tensor_direct[idx_tuple]
            d = diff[idx_tuple]
            print(f"  {str(idx_tuple):<20} {val1:<25} {val2:<25} {d:<15.2e}")
        
        if len(diff_indices) > 20:
            print(f"  ... et {len(diff_indices) - 20} autres différences")
        
        # Analyser par nombre de photons
        print("\n  Analyse par nombre de photons (somme des indices d'entrée):")
        n_photons_max = bs_tensor.shape[0] - 1
        for n in range(n_photons_max + 1):
            # Trouver les indices où sum(invec) = n
            mask = np.zeros_like(diff, dtype=bool)
            for idx in np.ndindex(bs_tensor.shape):
                invec = idx[:2]
                if sum(invec) == n:
                    mask[idx] = True
            
            if np.any(mask):
                max_diff_n = np.max(diff[mask])
                num_diff_n = np.sum(diff[mask] > atol)
                print(f"    n={n} photons: max_diff={max_diff_n:.2e}, nb_diff={num_diff_n}")
    
    # tests for clements scheme quimb network and contraction
    
    N = 4
    U = random_unitary(N)
    BS_list, D = full_clements(U)
    n_photons = 4
    tn = clements_to_fock_network(BS_list, D, n_photons)
    output_inds = [f'in_{mode}' for mode in range(N)] + [f'out_{mode}' for mode in range(N)]
    result = tn.contract(all, output_inds=output_inds, optimize='greedy')
    clements_tensor = clements_fock_tensor(BS_list, D, n_photons)
    print("\n" + "="*60)
    print("COMPARISON TEST: clements_tensor vs result.data")
    print("="*60)
    #are_close = np.allclose(clements_tensor, result.data, atol=1e-10, rtol=1e-8)
    #print(f"\nTensors are close: {are_close}")
    print(clements_tensor)
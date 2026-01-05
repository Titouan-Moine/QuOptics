import quimb as qb
import numpy as np
import math

def beamsplitter(m, n, phi, theta, N):
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

def phaseshifter(m, phi, N):
		"""Constructs a phase shifter matrix acting on mode m.
		
		Parameters
		m : (int)The mode index (0-indexed).
		phi : (float) The phase shift (in radians).
		N : (int)	The total number of modes in the system.
		
		Returns
		(np.ndarray)An N×N unitary phase shifter matrix with complex entries.
		
		Raises
		------
		ValueError
				If m >= N.
		"""
		
		if m >= N:
				raise ValueError("Mode index m must be less than N.")
		
		T = np.eye(N, dtype=complex)
		T[m, m] = np.exp(1j * phi)

		return T

def add_gate(U, gate, list=False):
	"""
	Apply a gate to the unitary U by left multiplication.
	If list is True U is assumed to be a list of gates and the new gate is appended to the list.

	Arguments:
	- U: current unitary (n x n numpy.ndarray)
	- gate: gate to apply (n x n numpy.ndarray)
	- list: whether U is a list of gates (default False)

	Returns: updated unitary (n x n numpy.ndarray)
	"""
	if list:
		U.append(gate)
		return U
	else:
		return gate @ U
def create_jump_random_network(n_modes, n_gate, list=False, jump_size=1, theta_range=(0, math.pi/2), phi_range=(0, 2*math.pi), bs=True, ps=True, seed=None):
	"""
	Create a random jump network unitary on n_modes modes with n_gate connected at most with a jump_size neighbour mode
	beamsplitter gates.

	Arguments:
	- n_modes: number of modes (integer >= 1)
	- n_gate: number of beamsplitter gates to apply (integer >= 0)
	- list: whether to return a list of gates instead of the full unitary (default False)
	- jump_size: maximum distance between modes connected by a beamsplitter gate (integer >= 1)
	- theta_range: tuple (min, max) for uniform sampling of theta angles
	- phi_range: tuple (min, max) for uniform sampling of phi angles
	- bs: whether to include beamsplitter gates (default True)
	- ps: whether to include phase shifter gates (default True)
	- seed: optional random seed for reproducibility

	Returns: n_modes x n_modes numpy.ndarray (complex) unitary.
	"""
	if n_modes < 1:
		raise ValueError("n_modes must be >= 1")
	if jump_size < 1:
		raise ValueError("jump_size must be >= 1")
	if n_gate < 0:
		raise ValueError("n_gate must be >= 0")

	rng = np.random.default_rng(seed)
	if list:
		gates=[]
	else:
		U = np.eye(n_modes, dtype=complex)

	for _ in range(n_gate):
		# randomly select one mode then another mode within jump_size
		i = rng.integers(0, n_modes)
		j = rng.integers(max(0, i - jump_size), min(n_modes, i + jump_size + 1))
		if j == i:
			j = rng.integers(max(0, i - jump_size), min(n_modes, i + jump_size + 1))
		# sample random theta and phi
		theta = rng.uniform(*theta_range)
		phi = rng.uniform(*phi_range)
		if ps & bs:
			# apply at random a phase shifter or beamsplitter
			if rng.random() < 0.5:
				P = phaseshifter(i, phi, n_modes)
				add_gate(U, P, list=list)
			else:
				B = beamsplitter(i, j, phi, theta, n_modes)
				add_gate(U, B, list=list)
		elif bs:
			B = beamsplitter(i, j, phi, theta, n_modes)
			add_gate(U, B, list=list)
		elif ps:
			P = phaseshifter(i, phi, n_modes)
			add_gate(U, P, list=list)
	return U


def quimb_beamsplitter(m, n ,phi, theta, N):
	T = np.eye(2, dtype=complex)
	e = np.exp(1j * phi)
	c = np.cos(theta)
	s = np.sin(theta)

	T[0, 0] = e * c
	T[1, 1] = c
	T[0, 1] = -s
	T[1, 0] = e * s
	bs=qb.Tensor(data=T, inds=(m, n), tags={'BEAMSPLITTER'})
	return bs

def quimb_phaseshifter(m, phi, N):
	T = np.eye(1, dtype=complex)
	T[0, 0] = np.exp(1j * phi)
	ps=qb.Tensor(data=T, inds=(m), tags={'PHASESHIFTER'})
	return ps

def quimb_network(n_modes, n_gate, jump_size=1, theta_range=(0, math.pi/2), phi_range=(0, 2*math.pi), bs=True, ps=True, seed=None):
	"""
	Create a random jump network as a quimb TensorNetwork on n_modes modes with n_gate connected at most with a jump_size neighbour mode
	beamsplitter gates.

	Arguments:
	- n_modes: number of modes (integer >= 1)
	- n_gate: number of beamsplitter gates to apply (integer >= 0)
	- jump_size: maximum distance between modes connected by a beamsplitter gate (integer >= 1)
	- theta_range: tuple (min, max) for uniform sampling of theta angles
	- phi_range: tuple (min, max) for uniform sampling of phi angles
	- bs: whether to include beamsplitter gates (default True)
	- ps: whether to include phase shifter gates (default True)
	- seed: optional random seed for reproducibility

	Returns: quimb TensorNetwork unitary.
	"""
	if n_modes < 1:
		raise ValueError("n_modes must be >= 1")
	if jump_size < 1:
		raise ValueError("jump_size must be >= 1")
	if n_gate < 0:
		raise ValueError("n_gate must be >= 0")

	rng = np.random.default_rng(seed)
	TN = qb.TensorNetwork()

	for _ in range(n_gate):
		# randomly select one mode then another mode within jump_size
		i = rng.integers(0, n_modes)
		j = rng.integers(max(0, i - jump_size), min(n_modes, i + jump_size + 1))
		if j == i:
			j = rng.integers(max(0, i - jump_size), min(n_modes, i + jump_size + 1))
		# sample random theta and phi
		theta = rng.uniform(*theta_range)
		phi = rng.uniform(*phi_range)
		if ps & bs:
			# apply at random a phase shifter or beamsplitter
			if rng.random() < 0.5:
				P = quimb_phaseshifter(i, phi, n_modes)
				TN = TN & P
			else:
				B = quimb_beamsplitter(i, j, phi, theta, n_modes)
				TN = TN & B
		elif bs:
			B = quimb_beamsplitter(i, j, phi, theta, n_modes)
			TN = TN & B
		elif ps:
			P = quimb_phaseshifter(i, phi, n_modes)
			TN = TN & P

	# Create a quimb TensorNetwork from the gates
	return TN

if __name__ == "__main__":
	# quick demo and unitarity check for n=4
	# U = create_jump_random_network(4, 2, jump_size=1, seed=42)
	# print("Unitary U (4x4) from jump beamsplitters:")
	# np.set_printoptions(precision=4, suppress=True)
	# print(U)
	# I = np.eye(4)
	# err = np.max(np.abs(U.conj().T @ U - I))
	# print(f"Unitarity error max |U^†U - I| = {err:e}")

	TN = quimb_network(4, 2, jump_size=1, seed=42)
	TN.draw(color=['Beamsplitter', 'Phaseshifter'], figsize=(4, 4), show_inds='all')


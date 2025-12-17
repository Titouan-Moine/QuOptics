import quimb as qb
import numpy as np
import math

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


def create_jump_random_network(n_modes, n_gate, jump_size=1, theta_range=(0, math.pi/2), phi_range=(0, 2*math.pi), seed=None):
	"""
	Create a random jump network unitary on n_modes modes with n_gate connected at most with a jump_size neighbour mode
	beamsplitter gates.

	Arguments:
	- n_modes: number of modes (integer >= 1)
	- jump_size: maximum distance between modes connected by a beamsplitter gate (integer >= 1)
	- n_gate: number of beamsplitter gates to apply (integer >= 0)
	- theta_range: tuple (min, max) for uniform sampling of theta angles
	- phi_range: tuple (min, max) for uniform sampling of phi angles
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
		B = T(i, j, phi, theta, n_modes)
		U = B @ U
	return U

if __name__ == "__main__":
	# quick demo and unitarity check for n=4
	U = create_jump_random_network(4, 2, jump_size=1, seed=42)
	print("Unitary U (4x4) from jump beamsplitters:")
	np.set_printoptions(precision=4, suppress=True)
	print(U)
	I = np.eye(4)
	err = np.max(np.abs(U.conj().T @ U - I))
	print(f"Unitarity error max |U^†U - I| = {err:e}")


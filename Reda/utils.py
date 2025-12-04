import numpy as np
import math
from collections import defaultdict
import itertools

# ----- Generate Haar random unitary -----
def random_unitary(m, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = (np.random.normal(size=(m,m)) + 1j*np.random.normal(size=(m,m))) / np.sqrt(2)
    Q, R = np.linalg.qr(X)
    d = np.diag(R)
    ph = d / np.abs(d)
    return Q * ph.conjugate()

# ----- Generate all photon-count outcomes S: sum S_i = n -----
def all_outcomes(m, n):
    if m == 1:
        yield (n,)
        return
    for i in range(n+1):
        for rest in all_outcomes(m-1, n-i):
            yield (i,) + rest

# ----- Build A_S by repeating rows -----
def build_AS(A, S):
    rows = []
    for i, s in enumerate(S):
        for _ in range(s):
            rows.append(A[i])
    return np.array(rows) if rows else np.zeros((0, 0), dtype=A.dtype)

import numpy as np
import math
from utils import all_outcomes
from collections import defaultdict

def boson_distribution_polynomial(A):
    # A is m x n (m modes, n photons/columns)
    m, n = A.shape

    # Start with the constant polynomial "1" (exponent tuple of all zeros)
    poly = defaultdict(complex)
    zero_monom = tuple([0]*m)
    poly[zero_monom] = 1.0 + 0j

    # Multiply by each linear form L_k = sum_j A[j,k] * x_j
    for k in range(n):
        linear_terms = [
            (tuple(1 if j == i else 0 for j in range(m)), A[i, k])
            for i in range(m)
        ]

        newpoly = defaultdict(complex)
        for S, coeff in poly.items():
            for mon, c in linear_terms:
                newS = tuple(S[i] + mon[i] for i in range(m))
                newpoly[newS] += coeff * c

        poly = newpoly

    # Convert polynomial coefficients to probabilities (Pr[S] = |a_S|^2 * s1! * ... * sm!)
    outcomes = list(all_outcomes(m, n))
    probs = []

    for S in outcomes:
        a = poly.get(S, 0+0j)
        factorial = np.prod([math.factorial(s) for s in S])
        p = (abs(a) ** 2) * factorial   # ✅ CORRECT
        probs.append(float(p.real))

    probs = np.array(probs, dtype=float)
    total = probs.sum()
    if total == 0:
        # numeric underflow or all-zero coefficients: return uniform or raise
        raise ValueError("All outcome probabilities computed as zero (underflow or wrong A).")
    probs /= total

    return outcomes, probs, poly

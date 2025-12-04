import numpy as np
from utils import all_outcomes, build_AS
from permanent import ryser_permanent
import math

# ----- Permanent-based boson sampling distribution -----
def boson_distribution_permanent(A):
    m, n = A.shape
    outcomes = list(all_outcomes(m, n))
    probs = []

    for S in outcomes:
        AS = build_AS(A, S)
        per = ryser_permanent(AS)
        denom = np.prod([math.factorial(s) for s in S]) if len(S) > 0 else 1
        p = abs(per)**2 / denom
        probs.append(p.real)

    probs = np.array(probs)
    probs /= probs.sum()
    return outcomes, probs

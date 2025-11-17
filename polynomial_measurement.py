import sympy as sp
from math import factorial

def apply_unitary_to_polynomial(p, U, variables):
    """
    Apply an m×m unitary matrix U to a multivariate polynomial p.
    variables = [x1, ..., xm]
    Returns the transformed polynomial U[p].
    """
    m = len(variables)

    # Construct substitution: x_i -> Σ_j U[i,j]*x_j
    subs = {}
    for i in range(m):
        new_expr = sum(U[i][j] * variables[j] for j in range(m))
        subs[variables[i]] = new_expr

    return sp.expand(p.subs(subs))


def measurement_probability(p, s_tuple, variables):
    """
    Computes Pr[S] = |a_s|^2 * (s1! s2! ... sm!)
    where a_s is the coefficient of x1^s1 ... xm^sm.

    p: polynomial
    s_tuple: (s1, ..., sm)
    """
    monomial = 1
    for var, exp in zip(variables, s_tuple):
        monomial *= var**exp

    # use as_coefficients_dict() to get coefficient robustly across sympy versions
    p_expanded = sp.expand(p)
    coeffs_dict = p_expanded.as_coefficients_dict()
    coeff = coeffs_dict.get(monomial, 0)

    n = sum(s_tuple)
    numerator = 1
    for s in s_tuple:
        numerator *= factorial(s)

    prob = sp.Abs(coeff)**2 * numerator
    return sp.simplify(prob)


# ========================
# Example Usage
# ========================

# Variables
x1, x2 = sp.symbols('x1 x2')
variables = [x1, x2]

# Example polynomial p(x1,x2) = x1^2 + 2*x1*x2
p = x1**2 + 2*x1*x2

# Example 2×2 unitary (rotation for simplicity)
U = [
    [sp.sqrt(2)/2, sp.sqrt(2)/2],
    [-sp.sqrt(2)/2, sp.sqrt(2)/2]
]

# Apply U to p
U_p = apply_unitary_to_polynomial(p, U, variables)
print("Transformed polynomial U[p] =", U_p)

# Example measurement outcome S = (2, 0)
prob_S = measurement_probability(U_p, (2, 0), variables)
print("Measurement probability P[S=(2,0)] =", prob_S)

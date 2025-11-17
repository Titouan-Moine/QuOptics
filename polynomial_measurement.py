import sympy as sp
from math import factorial
import matplotlib.pyplot as plt
import numpy as np
import itertools
from collections import defaultdict

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
    Returns a float probability.
    """
    monomial = 1
    for var, exp in zip(variables, s_tuple):
        monomial *= var**exp

    p_expanded = sp.expand(p)
    coeffs_dict = p_expanded.as_coefficients_dict()
    coeff = coeffs_dict.get(monomial, 0)

    numerator = 1
    for s in s_tuple:
        numerator *= factorial(s)

    prob_sym = sp.Abs(coeff)**2 * numerator
    # numeric evaluation (real float)
    prob_val = float(sp.N(prob_sym))
    # guard against tiny negative numerical rounding
    return max(0.0, prob_val)


def J_m_n(variables, n):
    """
    Return the standard initial polynomial J_{m,n} = x1 * x2 * ... * x_n,
    where variables = [x1,...,xm] and n <= m.
    """
    m = len(variables)
    if n > m:
        raise ValueError("n must be <= number of variables m")
    prod = 1
    for v in variables[:n]:
        prod *= v
    return sp.expand(prod)


def _compositions(n, m):
    """Generate all m-tuples of nonnegative integers summing to n."""
    if m == 1:
        yield (n,)
    else:
        for i in range(n + 1):
            for tail in _compositions(n - i, m - 1):
                yield (i,) + tail


def all_measurement_probabilities(p, variables):
    """
    Compute Pr[S] for all S in Phi_{m,N} where N = total degree of p.
    Returns a dict mapping tuple S -> probability (float).
    """
    # determine total photon number N as degree of polynomial (total degree)
    # If p is sum of monomials, use degree as max total degree; for J_{m,n} it's n.
    N = sp.Poly(sp.expand(p), *variables).total_degree()
    m = len(variables)

    probs = {}
    for S in _compositions(N, m):
        probs[S] = measurement_probability(p, S, variables)

    return probs


def plot_probabilities(probs, title="Measurement probabilities", sort_by_prob=False, savepath=None):
    """
    Simple bar chart for probabilities dict mapping S -> prob.
    - sort_by_prob: sort bars by descending probability if True.
    - savepath: optional path to save figure (PNG).
    """
    items = list(probs.items())
    if sort_by_prob:
        items.sort(key=lambda kv: kv[1], reverse=True)
    else:
        items.sort()  # lexicographic on tuples

    labels = [str(s) for s, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(max(6, len(values) * 0.4), 4))
    plt.bar(range(len(values)), values, color="C0")
    plt.xticks(range(len(values)), labels, rotation=45, ha="right")
    plt.ylabel("Pr[S]")
    plt.title(title)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.show()


def is_unitary(U):
    """
    Check if U is a unitary matrix.
    U is a list of lists (2D array-like).
    """
    U_mat = sp.Matrix(U)
    identity = sp.eye(U_mat.rows)
    return U_mat.H * U_mat == identity


def apply_unitary_to_J_m_n_sympy(U, variables, n):
    """
    Direct construction via the product formula:
    U[J_{m,n}] = \prod_{i=0..n-1} ( sum_{j=0..m-1} U[i][j] * x_j )

    - U: m x m array-like (can contain sympy expressions)
    - variables: list [x0,...,x_{m-1}] (sympy symbols)
    - n: integer <= m (number of factors in the product)

    Returns a sympy expression equal to the expanded transformed polynomial.
    This is simple and uses sympy expand; ok for small m,n.
    """
    m = len(variables)
    if n > m:
        raise ValueError("n must be <= m")
    # build linear forms L_i = sum_j U[i][j] * variables[j]
    linear_forms = []
    for i in range(n):
        Li = sum(U[i][j] * variables[j] for j in range(m))
        linear_forms.append(Li)
    # product and expand
    prod_expr = 1
    for Li in linear_forms:
        prod_expr *= Li
    return sp.expand(prod_expr)


def apply_unitary_to_J_m_n_coeffs(U, variables, n):
    """
    Combinatorial expansion specialized for J_{m,n}:
    Expand the product by summing over all choices j_1,...,j_n (each in 0..m-1).
    For each choice the monomial x_{j_1} * ... * x_{j_n} contributes a coefficient
    prod_i U[i][j_i]. Collect contributions by counting multiplicities to build
    exponent tuples S = (s0,...,s_{m-1}).

    Returns a dict mapping exponent tuples S -> coefficient (sympy or complex).
    This is often faster/more memory-friendly than full symbolic expand when you
    only need coefficients / probabilities.
    """
    m = len(variables)
    if n > m:
        # It's allowed combinatorially (multiple photons can occupy same mode),
        # so do not forbid n>m here; user earlier enforced n<=m for J_{m,n},
        # but keep flexible.
        pass

    coeffs = defaultdict(lambda: 0)
    # iterate all selections of output modes for each of the n input creation ops
    for choice in itertools.product(range(m), repeat=n):
        # coefficient = prod_i U[i][choice[i]]
        coef = 1
        for i, j in enumerate(choice):
            coef = coef * U[i][j]
        # exponent tuple = counts of each mode j in choice
        counts = [0] * m
        for j in choice:
            counts[j] += 1
        coeffs[tuple(counts)] += coef

    return dict(coeffs)


def transformed_polynomial_from_coeffs(coeffs, variables):
    """
    Build a sympy polynomial expression from coeff dict mapping exponent tuples -> coeff.
    """
    expr = 0
    for exp_tuple, coef in coeffs.items():
        mon = 1
        for var, e in zip(variables, exp_tuple):
            if e:
                mon *= var**e
        expr += coef * mon
    return sp.expand(expr)


def probabilities_from_J_coeffs(coeffs):
    """
    Given coeffs dict S->a_S, return dict S->Pr[S] = |a_S|^2 * prod(s_i!).
    """
    probs = {}
    for S, a in coeffs.items():
        factorials_prod = 1
        for s in S:
            factorials_prod *= factorial(s)
        probs[S] = float(sp.N(sp.Abs(a)**2 * factorials_prod))
    return probs

# ========================
# Example Usage (updated)
# ========================

# Variables
x1, x2 = sp.symbols('x1 x2')
variables = [x1, x2]

# Initial polynomial J_{m,n} = x1 * x2 (n=2 here)
p = J_m_n(variables, 2)

# Example 2×2 unitary (rotation for simplicity)
U = [
    [sp.sqrt(2)/2, sp.sqrt(2)/2],
    [-sp.sqrt(2)/2, sp.sqrt(2)/2]
]
assert(is_unitary(U))  # Should be True
U_p = apply_unitary_to_polynomial(p, U, variables)
print("Transformed polynomial U[p] =", U_p)

# Compute all measurement probabilities for this U[p]
probs = all_measurement_probabilities(U_p, variables)
print("All probabilities:")
for s, prob in sorted(probs.items()):
    print(f"  {s}: {prob}")

# Plot them
plot_probabilities(probs, title="Pr[S] for transformed J_{m,n}", sort_by_prob=False)

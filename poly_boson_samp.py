import math
from collections import defaultdict
import numpy as np

def eval_multivariate_polynomial(x, coeffs, var_order=None):
    """
    Évalue un polynôme multivariable p(x1,...,xm).

    - x: dict mapping variable names to values OR sequence of values (x1,...,xm)
    - coeffs: dict mapping exponent tuples to coefficients (complex or float)
    - var_order: optional when x is dict
    """
    if isinstance(x, dict):
        if var_order is None:
            var_order = sorted(x.keys())
        vals = tuple(float(x[name]) for name in var_order)
    else:
        vals = tuple(float(v) for v in x)

    n = len(vals)
    total = 0.0 + 0.0j
    for exponents, coef in coeffs.items():
        if len(exponents) != n:
            raise ValueError(f"Exposants {exponents} incompatible avec {n} variables")
        term = complex(coef)
        for v, e in zip(vals, exponents):
            if e:
                term *= v ** e
        total += term
    return total

def _compositions(n, m):
    """Génère toutes les compositions de l'entier n en m parties (m-tuples non négatifs)."""
    if m == 1:
        yield (n,)
    else:
        for i in range(n + 1):
            for tail in _compositions(n - i, m - 1):
                yield (i,) + tail

def _multinomial_coef(n, ks):
    """Coefficient multinomial n! / prod(k_i!)."""
    return math.factorial(n) // math.prod(math.factorial(k) for k in ks)

def apply_unitary_to_polynomial(coeffs, U):
    """
    Applique la transformation x -> U x sur le polynôme p(x) donné par coeffs.

    - coeffs: dict mapping exponent tuples (e1,...,em) -> coefficient (complex)
    - U: m x m unitary (array-like). U[i,j] correspond à u_{i+1,j+1} dans la notation,
         i.e. la i-ème variable transformée contient U[i,j] * x_j.

    Retourne: new_coeffs dict mapping exponent tuples (s1,...,sm) -> coefficient a_S
    tel que p(U x) = sum_S a_S x^S.
    """
    U = np.asarray(U, dtype=complex)
    m = U.shape[0]
    if U.shape[1] != m:
        raise ValueError("U must be square m x m")
    new_coeffs = defaultdict(complex)

    zero_exp = (0,) * m
    for exp, coef in coeffs.items():
        if len(exp) != m:
            raise ValueError(f"Exposants {exp} incompatible avec dimension de U ({m})")
        # start partial polynomial as {zero_exp: coef}
        partial = {zero_exp: complex(coef)}
        # pour chaque variable i, multiplier par (sum_j U[i,j] x_j)^{exp[i]}
        for i, ei in enumerate(exp):
            if ei == 0:
                continue
            # build polynomial for this factor
            factor = {}
            for ks in _compositions(ei, m):
                multin = _multinomial_coef(ei, ks)
                # coefficient = multinomial * prod_j (U[i,j] ** ks[j])
                prod_u = 1+0j
                for j, ksj in enumerate(ks):
                    if ksj:
                        prod_u *= U[i, j] ** ksj
                coeff_term = multin * prod_u
                # exponent tuple for this factor is exactly ks
                factor[ks] = coeff_term
            # convolve partial with factor
            new_partial = {}
            for exp1, c1 in partial.items():
                for exp2, c2 in factor.items():
                    summed = tuple(a + b for a, b in zip(exp1, exp2))
                    new_partial[summed] = new_partial.get(summed, 0+0j) + c1 * c2
            partial = new_partial

        # add partial contributions to global new_coeffs
        for s_exp, s_coef in partial.items():
            new_coeffs[s_exp] += s_coef

    return dict(new_coeffs)

def measurement_probabilities_from_coeffs(transformed_coeffs):
    """
    Calcule les probabilités de mesure pour chaque configuration S à partir des
    coefficients a_S du polynôme transformé p(U x) = sum_S a_S x^S, selon
    Pr[S] = |a_S|^2 * (s1! * s2! * ... * sm!).

    Retourne: dict mapping exponent tuples S -> probability (float)
    """
    probs = {}
    for s_exp, a in transformed_coeffs.items():
        factorials_prod = math.prod(math.factorial(s) for s in s_exp)
        prob = (abs(a) ** 2) * factorials_prod
        probs[s_exp] = float(prob.real)  # should be real non-negative
    return probs

def probabilities_after_unitary(initial_coeffs, U):
    """
    Commodité: applique U au polynôme initial puis retourne les probabilités
    de toutes les issues S.
    """
    transformed = apply_unitary_to_polynomial(initial_coeffs, U)
    return measurement_probabilities_from_coeffs(transformed)

if __name__ == "__main__":
    # Exemple simple :
    # m = 3 modes, état initial x1^2 * x2^1 (deux photons en mode 1, un photon en mode 2)
    initial = {
        (2,1,0): 1.0  # coefficient normalisé à 1 pour l'exemple
    }
    # exemple d'unitaire 3x3 (unitary construit via QR pour l'exemple)
    random_mat = (np.random.randn(3,3) + 1j * np.random.randn(3,3))
    q, r = np.linalg.qr(random_mat)
    # s'assurer que diag(r) phases sont retirées pour obtenir unitaire propre
    lam = np.diag(np.exp(-1j * np.angle(np.diag(r))))
    U = q @ lam

    transformed = apply_unitary_to_polynomial(initial, U)
    probs = measurement_probabilities_from_coeffs(transformed)

    print("Coefficients a_S pour p(U x):")
    for s, a in sorted(transformed.items()):
        print(f"  {s}: {a}")
    print("\nProbabilités de mesure Pr[S]:")
    for s, p in sorted(probs.items()):
        print(f"  {s}: {p}")
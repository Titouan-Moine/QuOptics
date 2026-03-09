import math
import sympy as sp
import polynomial_measurement as pm

def run_test(theta_value=sp.pi/4, tol=1e-12):
    # symbols / problem setup
    x1, x2 = sp.symbols('x1 x2')
    variables = [x1, x2]
    n = 2  # J_{2,2} = x1 * x2

    # rotation matrix R(theta) = [[cos, -sin],[sin, cos]]
    c = sp.cos(theta_value)
    s = sp.sin(theta_value)
    U = [[c, -s],
         [s,  c]]

    # compute coefficients and probabilities
    coeffs = pm.apply_unitary_to_J_m_n_coeffs(U, variables, n)
    probs = pm.probabilities_from_J_coeffs(coeffs)

    p20 = probs.get((2, 0), 0.0)
    p02 = probs.get((0, 2), 0.0)
    p11 = probs.get((1, 1), 0.0)

    print(f"theta = {float(sp.N(theta_value))}")
    print(f"Pr(2,0) = {p20:.12g}")
    print(f"Pr(0,2) = {p02:.12g}")
    print(f"Pr(1,1) = {p11:.12g}")

    assert math.isclose(p20, p02, rel_tol=tol, abs_tol=tol), "Pr(2,0) != Pr(0,2)"
    print("Symmetry test passed: Pr(2,0) == Pr(0,2) within tolerance")

if __name__ == "__main__":
    run_test()
import numpy as np
import math
from collections import defaultdict
from utils import all_outcomes

def boson_distribution_polynomial(A):
    """
    Implémentation stricte de la définition du papier :
    1. Démarre avec J_{m,n} = x1 * x2 * ... * xn
    2. Applique le changement de variables x_i -> sum_j A[i,j] x_j
    3. Extrait les coefficients normalisés
    4. Calcule les probabilités Pr[S] = |a_S|^2 * s1! ... sm!
    
    A : matrice m x n (les n premières colonnes de l’unitaire U)
    """

    m, n = A.shape

    # -------------------------------------------
    # 1. Construire le polynôme J_{m,n}(x) = x1 * ... * xn
    # -------------------------------------------

    # Exposants : J_{m,n} = (1,1,...,1,0,...,0)
    exp = [1]*n + [0]*(m-n)
    J = defaultdict(complex)
    J[tuple(exp)] = 1.0 + 0j

    # -------------------------------------------
    # 2. Appliquer le changement de variables :
    #    x_k  -->  sum_i A[i,k] * x_i := Lk
    # -------------------------------------------

    # On doit appliquer le changement sur CHAQUE variable x_1,...,x_m,
    # mais seules les variables x_1...x_n apparaissent dans J.
    poly = J

    for k in range(n):
        # le remplacement x_k -> L_k
        Lk = [
            (tuple(1 if i == r else 0 for i in range(m)), A[r, k])
            for r in range(m)
        ]

        newpoly = defaultdict(complex)

        for S, coeff in poly.items():
            if S[k] == 0:
                # si x_k n'apparaît pas dans le monôme S
                newpoly[S] += coeff
                continue

            power = S[k]
            # enlever x_k^{power}
            S_without = list(S)
            S_without[k] = 0
            S_without = tuple(S_without)

            # on remplace (x_k)^power par (Lk)^power
            # expansion multinomiale
            from itertools import product

            # toutes les façons de distribuer 'power' copies parmi m modes
            # Lent mais correct pour suivi exact du papier
            for assign in product(range(m), repeat=power):
                newS = list(S_without)
                c = coeff
                for r in assign:
                    newS[r] += 1
                    c *= A[r, k]

                newpoly[tuple(newS)] += c

        poly = newpoly

    # -------------------------------------------
    # 3. Calcul des probabilités
    # -------------------------------------------

    outcomes = list(all_outcomes(m, n))
    probs = []

    for S in outcomes:
        a = poly.get(S, 0+0j)
        factorial = np.prod([math.factorial(s) for s in S])
        p = (abs(a)**2) * factorial
        probs.append(float(p.real))

    probs = np.array(probs, float)
    total = probs.sum()

    if total == 0:
        raise ValueError("Probabilités nulles — problème numérique ou matrice incorrecte.")

    probs /= total

    return outcomes, probs, poly

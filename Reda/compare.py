import time
import numpy as np
from utils import random_unitary
from boson_permanent import boson_distribution_permanent
from boson_polynomial import boson_distribution_polynomial
import pandas as pd
import matplotlib.pyplot as plt

def compare(m, n, seed=0):
    U = random_unitary(m, seed)
    A = U[:, :n]

    # Permanent method
    t0 = time.perf_counter()
    outcomes1, probs1 = boson_distribution_permanent(A)
    t1 = time.perf_counter()

    # Polynomial method
    t2 = time.perf_counter()
    outcomes2, probs2, poly = boson_distribution_polynomial(A)
    t3 = time.perf_counter()

    # correctness check
    maxdiff = np.max(np.abs(probs1 - probs2))

    return {
        "m": m,
        "n": n,
        "num_outcomes": len(outcomes1),
        "time_permanent_s": t1 - t0,
        "time_polynomial_s": t3 - t2,
        "max_prob_diff": float(maxdiff),
    }

if __name__ == "__main__":
    # Plusieurs tailles à tester
    sizes = [(5,2),(15, 6)]

    results = [compare(m, n) for m, n in sizes]

    df = pd.DataFrame(results)
    print(df)

    # Préparer les labels pour l'axe x
    labels = [f"({row.m},{row.n})" for idx, row in df.iterrows()]

    # --- Graphiques ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # 1️⃣ Comparaison des temps d'exécution
    x = np.arange(len(labels))
    width = 0.35
    ax[0].bar(x - width/2, df['time_permanent_s'], width, label='Permanent', color='skyblue')
    ax[0].bar(x + width/2, df['time_polynomial_s'], width, label='Polynomial', color='salmon')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("Temps (s)")
    ax[0].set_title("Comparaison des temps d'exécution")
    ax[0].legend()

    # 2️⃣ Différence maximale des probabilités
    ax[1].bar(x, df['max_prob_diff'], color='lightgreen')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("Différence maximale")
    ax[1].set_title("Différence maximale des probabilités")

    plt.tight_layout()
    plt.show()

    print("\nDONE ✓")

import numpy as np

# ----- Ryser permanent (complex) -----
def ryser_permanent(A):
    n = A.shape[0]
    if n == 0:
        return 1

    cols = [A[:, j] for j in range(n)]
    total = 0 + 0j
    N = 1 << n

    for S in range(1, N):
        bits = S
        parity = (-1) ** (n - bin(S).count("1"))
        row_sums = np.zeros(n, dtype=complex)
        j = 0

        while bits:
            if bits & 1:
                row_sums += cols[j]
            j += 1
            bits >>= 1

        total += parity * np.prod(row_sums)

    return total if (n % 2 == 0) else -total

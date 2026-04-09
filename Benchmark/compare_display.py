"""Benchmarking the execution time of two transformations on random unitary matrices
of varying sizes, and plotting the results for comparison."""

import time
# import numpy as np
# import sparse
import matplotlib.pyplot as plt
import numpy as np
from clements_scheme.clements_scheme import full_clements
from clements_scheme.rnd_unitary import random_unitary
import Benchmark.macro as ma
import Benchmark.micro as mi
from Benchmark.utils import sparse_allclose
# import PyFock.fock as fock
#from fock_amplitude import clements_fock_tensor
# Configuration for benchmarking


def transformationA(circuit):
    return ma.clements_macro(circuit)


def transformationB(circuit):
    return mi.clements_micro(circuit)


def main():
    max_size = 2
    circuit_sizes = list(range(2, max_size + 1))  # Vary the circuit sizes

    times_A = []
    times_B = []
    # Benchmarking loop
    for size in circuit_sizes:

        # Generate a random unitary matrix
        U = random_unitary(size)

        # Apply full_clements decomposition as input to transformations
        to_test = full_clements(U)
        to_test = ([(0, 1, 1, np.pi/4)], np.diag([np.exp(1j), 1]))  # Example: a single beamsplitter between modes 0 and 1 with phi=1 and theta=pi/2
        # to_test = (to_test[0], np.eye(size))  # Set Dfinal to identity for fair comparison
        print("(m, n, phi, theta) for each BS : ", to_test[0])
        print("phase shifts:", np.angle(to_test[1].diagonal()))

        # ===== Transformation A (macro) =====
        start_time = time.time()
        result_A, UA = transformationA(to_test)
        # assert np.allclose(UA, U, atol=1e-10), "The contracted unitary does not match the original unitary!"
        elapsed_time_A = time.time() - start_time
        times_A.append(elapsed_time_A)

        # ===== Transformation B (micro) =====

        start_time = time.time()
        result_B = transformationB(to_test)
        elapsed_time_B = time.time() - start_time
        times_B.append(elapsed_time_B)

        if not sparse_allclose(result_A, result_B, atol=1e-6):
            print(f"Results differ for size {size}!")
            print("Result A (macro):", result_A.data)
            print("Result B (micro):", result_B.data)
            print("angle of result A:", np.angle(result_A.data))
            print("angle of result B:", np.angle(result_B.data))
            print("angle difference:", np.angle(result_A.data) - np.angle(result_B.data))

    # plt.figure(figsize=(9, 6))
    # plt.plot(circuit_sizes, times_A, color='blue', marker='o', linestyle='-', label='produit matriciel puis passage dans fock')
    # plt.plot(circuit_sizes, times_B, color='red', marker='s', linestyle='--', label='passage dans fock puis contraction de tenseurs')

    # plt.xlabel('Circuit size (number of modes)')
    # plt.ylabel('Execution time (s)')
    # plt.title('Benchmark: Transformation execution time vs circuit size')
    # plt.grid(alpha=0.3)
    # plt.legend()

    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()

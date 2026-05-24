"""Benchmarking the execution time of two transformations on random unitary matrices
of varying sizes, and plotting the results for comparison."""

import time
import math
# import numpy as np
# import sparse
import matplotlib.pyplot as plt
# import numpy as np
# from clements_scheme.clements_scheme import full_clements
# from clements_scheme.rnd_unitary import random_unitary
from TeNCo import circuit
from rnd_module import random_fock_uniform
from Benchmark import GUS
from Benchmark import FTN
from Benchmark.utils import sparse_allclose
from Benchmark.circuit_gen import rnd_circuit, display_circuit
# import PyFock.fock as fock
#from fock_amplitude import clements_fock_tensor
# Configuration for benchmarking


def transformationA(circuit):
    return GUS.clements_GUS(circuit)


def transformationB(circuit):
    return FTN.clements_FTN(circuit)


def compare_open_circuits(max_size=10, circuit_type="lasagna", depth_factor=1., photon_function=lambda size: size // 5):
    circuit_sizes = list(range(2, max_size + 1))  # Vary the circuit sizes

    times_A = []
    times_B = []
    # Benchmarking loop
    for size in circuit_sizes:
        valid_results = False
        depth = math.ceil(depth_factor * size)
        n_photons = photon_function(size)
        while not valid_results:
            # Generate a random unitary matrix
            # U = random_unitary(size)
            circuit = rnd_circuit(n_modes=size, n_layers=depth, circuit_type=circuit_type)
            # input_state = random_fock_uniform(1, size)

            # Apply full_clements decomposition as input to transformations
            # to_test = full_clements(U)
            # to_test = ([(0, 1, 1, np.pi/4)], np.diag([np.exp(1j), 1]))
            # to_test = (to_test[0], np.eye(size))  # Set Dfinal to identity for fair comparison
            # print("(m, n, phi, theta) for each BS : ", to_test[0])
            # print("phase shifts:", np.angle(to_test[1].diagonal()))

            # ===== Transformation A (GUS) =====
            # Warm-up call to mitigate any initial overhead in the first call
            _ = GUS.GUS_open_circuit(circuit, n_photons=n_photons, n_modes=size)

            start_time = time.time()
            result_A = GUS.GUS_open_circuit(circuit, n_photons=n_photons, n_modes=size)
            # print("UA (macro):\n", UA)
            # assert np.allclose(UA, U, atol=1e-10), "Contracted unitary doesn't match the original unitary!"
            elapsed_time_A = time.time() - start_time
            times_A.append(elapsed_time_A)

            # ===== Transformation B (FTN) =====
            # Warm-up call to mitigate any initial overhead in the first call
            _ = FTN.FTN_open_circuit(circuit, n_photons, size)

            start_time = time.time()
            result_B = FTN.FTN_open_circuit(circuit, n_photons, size)
            elapsed_time_B = time.time() - start_time
            times_B.append(elapsed_time_B)

            if not sparse_allclose(result_A, result_B, atol=1e-10):
                print(f"Results differ for size {size}!")
                print("Result GUS:\n", result_A.coords, result_A.data)
                print("Result FTN:\n", result_B.coords, result_B.data)
            #     print("angle of result A:", np.angle(result_A.data))
            #     print("angle of result B:", np.angle(result_B.data))
            #     print("angle difference:", np.angle(result_A.data) - np.angle(result_B.data))
                # times_A.pop()  # Remove the last recorded time for this size (invalid results)
                # times_B.pop()
            else:
                valid_results = True
            valid_results = True # for testing purposes, to avoid long times while debugging

    plt.figure(figsize=(9, 6))
    plt.plot(circuit_sizes[:], times_A[:], color='blue', marker='o', linestyle='-',
             label='produit matriciel puis passage dans fock')
    plt.plot(circuit_sizes[:], times_B[:], color='red', marker='s', linestyle='--',
             label='passage dans fock puis contraction de tenseurs')

    plt.xlabel('Circuit size (number of modes)')
    plt.ylabel('Execution time (s)')
    plt.title(f'Benchmark: time vs circuit size ({circuit_type} circuits, depth factor={depth_factor}')
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

def compare_closed_circuits(max_size=10, circuit_type="lasagna", depth_factor=1., photon_function=lambda size: size // 5):
    circuit_sizes = list(range(2, max_size + 1))  # Vary the circuit sizes
    times_GUS = []
    times_FTN = []
    # Benchmarking loop
    # time.sleep(1)  # Short pause before starting the benchmark
    for size in circuit_sizes:
        valid_results = False
        depth = math.ceil(size * depth_factor)
        n_photons = photon_function(size)
        while not valid_results:
            # Generate a random unitary matrix
            # U = random_unitary(size)
            circuit = rnd_circuit(size, depth, circuit_type=circuit_type)
            # display_circuit(circuit, method="txt", n_modes=size)
            input_state = random_fock_uniform(n_photons, size)
            output_state = random_fock_uniform(n_photons, size)

            # Apply full_clements decomposition as input to transformations
            # circuit = full_clements(U)
            #to_test = ([(0, 1, 1, np.pi/4)], np.diag([np.exp(1j), 1]))
            # to_test = (to_test[0], np.eye(size))  # Set Dfinal to identity for fair comparison
            # print("(m, n, phi, theta) for each BS : ", to_test[0])
            # print("phase shifts:", np.angle(to_test[1].diagonal()))

            # ===== Transformation A (GUS) =====
            # Warm-up call to mitigate any initial overhead in the first call
            _ = GUS.GUS_closed_circuit(circuit, input_state, output_state,
                                       n_modes=size, n_photons=n_photons)

            start_time = time.time()
            result_GUS = GUS.GUS_closed_circuit(circuit, input_state, output_state,
                                                n_modes=size, n_photons=n_photons)
            # print("UA (macro):\n", UA)
            # assert np.allclose(UA, U, atol=1e-10), "The contracted unitary does not match the original unitary!"
            elapsed_time_GUS = time.time() - start_time
            times_GUS.append(elapsed_time_GUS)

            # ===== Transformation B (FTN) =====
            # Warm-up call to mitigate any initial overhead in the first call
            _ = FTN.FTN_closed_circuit(circuit, input_state, output_state,
                                       n_modes=size, n_photons=n_photons)

            start_time = time.time()
            result_FTN = FTN.FTN_closed_circuit(circuit, input_state, output_state,
                                                n_modes=size, n_photons=n_photons)
            elapsed_time_FTN = time.time() - start_time
            times_FTN.append(elapsed_time_FTN)

            if abs(result_GUS - result_FTN) > 1e-10:
                print(f"Results differ for size {size}!")
                print("Result GUS:\n", result_GUS)
                print("Result FTN:\n", result_FTN)
            #     print(f"Results differ for size {size}!")
            #     print("Result A (macro):\n", result_A.coords, result_A.data)
            #     print("Result B (micro):\n", result_B.coords, result_B.data)
            #     print("angle of result A:", np.angle(result_A.data))
            #     print("angle of result B:", np.angle(result_B.data))
            #     print("angle difference:", np.angle(result_A.data) - np.angle(result_B.data))
                times_GUS.pop()  # Remove the last recorded time for this size (invalid results)
                times_FTN.pop()
                # pass
            else:
                valid_results = True
            # valid_results = True # for testing purposes, to avoid long times while debugging

    plt.figure(figsize=(9, 6))
    plt.plot(circuit_sizes[:], times_GUS[:], color='blue', marker='o', linestyle='-',
             label='GUS (produit matriciel puis passage dans fock)')
    plt.plot(circuit_sizes[:], times_FTN[:], color='red', marker='s', linestyle='--',
             label='FTN (passage dans fock puis contraction de tenseurs)')

    plt.xlabel('Circuit size (number of modes)')
    plt.ylabel('Execution time (s)')
    plt.title('Benchmark: simulation time vs circuit size')
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

def only_GUS_benchmark(max_size=10, circuit_type="lasagna", depth_factor=1., photon_function=lambda size: math.ceil(size / 5)):
    circuit_sizes = [4] + list(range(2, max_size + 1))  # Vary the circuit sizes
    times_GUS = []
    # Benchmarking loop
    for size in circuit_sizes:
        depth = math.ceil(size * depth_factor)  # Set depth proportional to size for more realistic benchmarking
        n_photons = photon_function(size)  # Determine number of photons based on the provided function
        circuit = rnd_circuit(size, depth, circuit_type=circuit_type)
        input_state = random_fock_uniform(n_photons, size)
        output_state = random_fock_uniform(n_photons, size)

        start_time = time.time()
        result_GUS = GUS.GUS_closed_circuit(circuit, input_state, output_state, n_modes=size, n_photons=n_photons)
        elapsed_time_GUS = time.time() - start_time
        times_GUS.append(elapsed_time_GUS)

    plt.figure(figsize=(9, 6))
    plt.plot(circuit_sizes[1:], times_GUS[1:], color='blue', marker='o', linestyle='-', label='GUS (produit matriciel puis passage dans fock)')

    plt.xlabel('Circuit size (number of modes)')
    plt.ylabel('Execution time (s)')
    plt.title('Benchmark: GUS simulation time vs circuit size')
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()

def photon_capped_linear(size):
    return min(size, 5)

def main():
    compare_open_circuits(max_size=6, depth_factor=0.25, circuit_type="mixed",
                          photon_function=math.ceil)
    # compare_closed_circuits(max_size=10, depth_factor=0.25, circuit_type="mixed",
    #                         photon_function=lambda size: 20)
    # only_GUS_benchmark(max_size=100, depth_factor=1, circuit_type="mixed",
    #                    photon_function=lambda size: size // 5)

if __name__ == "__main__":
    main()

"""Benchmarking the execution time of two transformations on random unitary matrices
of varying sizes, and plotting the results for comparison."""

# from os import times
import multiprocessing as mp
import time
import math
from collections.abc import Sequence
from typing import Optional
import os
import json
from datetime import datetime, timezone
# import numpy as np
# import sparse
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
# import numpy as np
# from clements_scheme.clements_scheme import full_clements
# from clements_scheme.rnd_unitary import random_unitary
# from TeNCo import circuit
from rnd_module import random_fock_uniform
from Benchmark import GUS
from Benchmark import FTN
from Benchmark.utils import sparse_allclose
from Benchmark.circuit_gen import rnd_circuit
# import PyFock.fock as fock
#from fock_amplitude import clements_fock_tensor
# Configuration for benchmarking


def _benchmark_open_worker(queue, circuit, size: int, n_photons: int):
    start_time = time.time()
    result_GUS = GUS.GUS_open_circuit(circuit, n_photons=n_photons, n_modes=size)
    elapsed_time_GUS = time.time() - start_time

    start_time = time.time()
    result_FTN = FTN.FTN_open_circuit(circuit, n_photons=n_photons, n_modes=size)
    elapsed_time_FTN = time.time() - start_time

    queue.put(("ok", elapsed_time_GUS, elapsed_time_FTN, result_GUS, result_FTN))


def _benchmark_closed_worker(queue, circuit, size: int, n_photons: int, input_state, output_state):
    start_time = time.time()
    result_GUS = GUS.GUS_closed_circuit(circuit, input_state, output_state,
                                        n_modes=size, n_photons=n_photons)
    elapsed_time_GUS = time.time() - start_time

    start_time = time.time()
    result_FTN = FTN.FTN_closed_circuit(circuit, input_state, output_state,
                                        n_modes=size, n_photons=n_photons)
    elapsed_time_FTN = time.time() - start_time

    queue.put(("ok", elapsed_time_GUS, elapsed_time_FTN, result_GUS, result_FTN))


def _benchmark_method_worker(queue, method_name: str, circuit, size: int, n_photons: int,
                             input_state=None, output_state=None):
    start_time = time.time()
    if method_name == "GUS_open":
        result = GUS.GUS_open_circuit(circuit, n_photons=n_photons, n_modes=size)
    elif method_name == "FTN_open":
        result = FTN.FTN_open_circuit(circuit, n_photons, size)
    elif method_name == "GUS_closed":
        assert input_state is not None
        assert output_state is not None
        result = GUS.GUS_closed_circuit(circuit, input_state, output_state,
                                        n_modes=size, n_photons=n_photons)
    elif method_name == "FTN_closed":
        assert input_state is not None
        assert output_state is not None
        result = FTN.FTN_closed_circuit(circuit, input_state, output_state,
                                        n_modes=size, n_photons=n_photons)
    else:
        raise ValueError(f"Unknown benchmark method: {method_name}")

    elapsed_time = time.time() - start_time
    queue.put(("ok", elapsed_time, result))


def _run_benchmark_with_timeout(timeout_seconds: int,
                                method_name: str,
                                circuit,
                                size: int,
                                n_photons: int,
                                input_state=None,
                                output_state=None):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_benchmark_method_worker,
                          args=(queue, method_name, circuit, size, n_photons, input_state,
                                output_state))

    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"Benchmark exceeded {timeout_seconds} seconds")

    if process.exitcode not in (0, None):
        raise RuntimeError(f"Benchmark worker failed with exit code {process.exitcode}")

    if queue.empty():
        raise RuntimeError("Benchmark worker exited without returning a result.")

    status, *payload = queue.get()
    if status == "error":
        raise RuntimeError(payload[0])

    return payload


def _serialize_results(times_GUS: dict, times_FTN: dict,
                       sizes: Sequence[int], photon_numbers: Sequence[int],
                       combining: str, is_open: bool, circuit_type: str,
                       depth_factor: float, timeout_seconds: int) -> dict:
    def k_to_str(k):
        return f"{k[0]},{k[1]}"

    return {
        "meta": {
            "sizes": list(sizes),
            "photon_numbers": list(photon_numbers),
            "combining": combining,
            "is_open": is_open,
            "circuit_type": circuit_type,
            "depth_factor": depth_factor,
            "timeout_seconds": timeout_seconds,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        "times_GUS": {k_to_str(k): v for k, v in times_GUS.items()},
        "times_FTN": {k_to_str(k): v for k, v in times_FTN.items()},
    }


def save_benchmark_results(times_GUS: dict, times_FTN: dict,
                           sizes: Sequence[int], photon_numbers: Sequence[int],
                           combining: str, is_open: bool, circuit_type: str,
                           depth_factor: float, timeout_seconds: int,
                           out_dir: Optional[str] = None) -> str:
    base_dir = out_dir or os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(base_dir, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{stamp}.json"
    path = os.path.join(base_dir, filename)
    payload = _serialize_results(times_GUS, times_FTN, sizes, photon_numbers,
                                 combining, is_open, circuit_type, depth_factor,
                                 timeout_seconds)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_benchmark_results(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    def parse_key(s):
        a, b = s.split(",")
        return (int(a), int(b))
    times_GUS = {parse_key(k): v for k, v in data.get("times_GUS", {}).items()}
    times_FTN = {parse_key(k): v for k, v in data.get("times_FTN", {}).items()}
    return {
        "meta": data.get("meta", {}),
        "times_GUS": times_GUS,
        "times_FTN": times_FTN,
    }


def display_results_from_file(path: str, timeout_seconds: int = 60) -> None:
    data = load_benchmark_results(path)
    meta = data["meta"]
    sizes = meta.get("sizes", [])
    photon_numbers = meta.get("photon_numbers", [])
    combining = meta.get("combining", "zip")
    is_open = meta.get("is_open", True)
    depth_factor = meta.get("depth_factor", 1.)
    times_GUS = data["times_GUS"]
    times_FTN = data["times_FTN"]

    if combining == "zip":
        labels = [f"{s}\n{p}" for s, p in zip(sizes, photon_numbers)]
        gus_values = [times_GUS.get((size, photons), float("nan"))
                      for size, photons in zip(sizes, photon_numbers)]
        ftn_values = [times_FTN.get((size, photons), float("nan"))
                      for size, photons in zip(sizes, photon_numbers)]

        plt.figure(figsize=(10, 6))
        plt.plot(labels, gus_values, color='blue', marker='o', linestyle='-',
                 label='GUS')
        plt.plot(labels, ftn_values, color='red', marker='s', linestyle='--',
                 label='FTN')
        plt.xlabel('size / photons')
        plt.ylabel('Execution time (s)')
        # plt.title(f"Benchmark loaded from {os.path.basename(path)}")
        plt.title(f"Benchmark: execution time vs circuit size and photon number\n"
                  f"Combining: {combining}, {"open circuits" if is_open else "closed circuits"},"
                  f" depth factor: {depth_factor}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    # combining == product -> heatmaps
    plt_sizes = list(sizes)
    plt_photons = list(photon_numbers)
    gus_grid = [[times_GUS.get((size, photons), float("nan")) for size in plt_sizes]
                for photons in plt_photons]
    ftn_grid = [[times_FTN.get((size, photons), float("nan")) for size in plt_sizes]
                for photons in plt_photons]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    plots = [
        (axes[0], gus_grid, 'GUS'),
        (axes[1], ftn_grid, 'FTN'),
    ]

    for ax, grid, title in plots:
        image = ax.imshow(grid, origin='lower', aspect='auto', cmap='plasma')#, norm=LogNorm(vmin=1e-4, vmax=timeout_seconds))
        ax.set_title(title)
        ax.set_xticks(range(len(plt_sizes)))
        ax.set_xticklabels(plt_sizes)
        ax.set_yticks(range(len(plt_photons)))
        ax.set_yticklabels(plt_photons)
        ax.set_xlabel('Circuit size (number of modes)')
        ax.set_ylabel('Number of photons')
        fig.colorbar(image, ax=ax, label='Execution time (s)')

    # fig.suptitle(f'Benchmark loaded from {os.path.basename(path)}')
    fig.suptitle(f"Benchmark: execution time vs circuit size and photon number\n"
                 f"Combining: {combining}, {'open circuits' if is_open else 'closed circuits'},"
                 f" depth factor: {depth_factor}")
    fig.tight_layout()
    plt.show()


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
        _ = GUS.GUS_closed_circuit(circuit, input_state, output_state, n_modes=size, n_photons=n_photons)
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

def compare_photons_modes(sizes: Sequence[int],
                          photon_numbers: Sequence[int],
                          combining: str="zip",
                          is_open: bool=True,
                          circuit_type: str="lasagna",
                          depth_factor: float=1.) -> None:
    """Compare the execution time of GUS and FTN transformations for varying numbers of photons and
    modes.
    
    Args:
        sizes (Sequence[int]): A sequence of circuit sizes (number of modes) to benchmark.
        photon_numbers (Sequence[int]): A sequence of photon numbers to benchmark.
        combining (str): The method to combine sizes and photon numbers ('zip' or 'product').
        is_open (bool): Whether to benchmark open circuits (True) or closed circuits (False).
        circuit_type (str): The type of random circuit to generate ('lasagna', 'mixed', etc.).
        depth_factor (float): The factor to determine the depth of the circuit based on its size.
    """
    if combining == "zip":
        benchmarks = zip(sizes, photon_numbers)
    elif combining == "product":
        benchmarks = ((size, n_photons) for size in sizes for n_photons in photon_numbers)
    else:
        raise ValueError("Invalid combining method. Choose 'zip' or 'product'.")
    times_GUS = {}
    times_FTN = {}
    timeout_seconds = 60
    timed_out_pairs_GUS: list[tuple[int, int]] = []
    timed_out_pairs_FTN: list[tuple[int, int]] = []

    def is_dominated_by_timeout(timeout_pairs: list[tuple[int, int]],
                                size: int,
                                n_photons: int) -> bool:
        return any(timeout_size <= size and timeout_photons <= n_photons
                   for timeout_size, timeout_photons in timeout_pairs)

    def record_timeout(times: dict[tuple[int, int], float],
                       timeout_pairs: list[tuple[int, int]],
                       size: int,
                       n_photons: int) -> None:
        times[(size, n_photons)] = timeout_seconds
        timeout_pairs.append((size, n_photons))

    for size, n_photons in benchmarks:
        depth = math.ceil(size * depth_factor)
        circuit = rnd_circuit(size, depth, circuit_type=circuit_type)
        input_state = random_fock_uniform(n_photons, size)
        output_state = random_fock_uniform(n_photons, size)

        gus_dominated = is_dominated_by_timeout(timed_out_pairs_GUS, size, n_photons)
        ftn_dominated = is_dominated_by_timeout(timed_out_pairs_FTN, size, n_photons)

        result_GUS = None
        result_FTN = None

        if gus_dominated:
            print(f"Skipping GUS for size={size}, photons={n_photons} because a smaller or equal"
                  " GUS timeout pair already exists")
            record_timeout(times_GUS, timed_out_pairs_GUS, size, n_photons)
        else:
            try:
                elapsed_time_GUS, result_GUS = _run_benchmark_with_timeout(
                    timeout_seconds, "GUS_open" if is_open else "GUS_closed", circuit, size,
                    n_photons, input_state=input_state, output_state=output_state)
                times_GUS[(size, n_photons)] = elapsed_time_GUS
            except TimeoutError:
                print(f"Timeout after {timeout_seconds}s for GUS at size={size}, photons={n_photons}")
                record_timeout(times_GUS, timed_out_pairs_GUS, size, n_photons)

        if ftn_dominated:
            print(f"Skipping FTN for size={size}, photons={n_photons} because a smaller or equal"
                  " FTN timeout pair already exists")
            record_timeout(times_FTN, timed_out_pairs_FTN, size, n_photons)
        else:
            try:
                elapsed_time_FTN, result_FTN = _run_benchmark_with_timeout(
                    timeout_seconds, "FTN_open" if is_open else "FTN_closed", circuit, size,
                    n_photons, input_state=input_state, output_state=output_state)
                times_FTN[(size, n_photons)] = elapsed_time_FTN
            except TimeoutError:
                print(f"Timeout after {timeout_seconds}s for FTN at size={size}, photons={n_photons}")
                record_timeout(times_FTN, timed_out_pairs_FTN, size, n_photons)

        if result_GUS is None or result_FTN is None:
            continue

        if is_open and not sparse_allclose(result_GUS, result_FTN, atol=1e-9):
            print(f"Results differ for size {size} and {n_photons} photons!")
        elif not is_open and abs(result_GUS - result_FTN) > 1e-9:
            print(f"Results differ for size {size} and {n_photons} photons!")

    if combining == "zip":
        plotted_sizes = list(sizes)
        plotted_photons = list(photon_numbers)
        labels = [f"{size}\n{photons}" for size, photons in zip(plotted_sizes, plotted_photons)]

        gus_values = [times_GUS.get((size, photons), float("nan"))
                      for size, photons in zip(plotted_sizes, plotted_photons)]
        ftn_values = [times_FTN.get((size, photons), float("nan"))
                      for size, photons in zip(plotted_sizes, plotted_photons)]

        # auto-save results
        try:
            saved_path = save_benchmark_results(times_GUS, times_FTN, sizes, photon_numbers,
                                                combining, is_open, circuit_type, depth_factor,
                                                timeout_seconds)
            print(f"Saved benchmark results to {saved_path}")
            try:
                display_results_from_file(saved_path)
            except (FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
                print(f"Warning: failed to display saved benchmark results: {e}")
        except (FileNotFoundError, PermissionError, TypeError) as e:
            print(f"Warning: failed to save benchmark results: {e}")

        plt.figure(figsize=(10, 6))
        plt.plot(labels, gus_values, color='blue', marker='o', linestyle='-',
                 label='GUS (produit matriciel puis passage dans fock)')
        plt.plot(labels, ftn_values, color='red', marker='s', linestyle='--',
                 label='FTN (passage dans fock puis contraction de tenseurs)')
        plt.xlabel('size / photons')
        plt.ylabel('Execution time (s)')
        plt.title(f'Benchmark: time vs size/photons ({circuit_type} circuits,'
                  f' depth factor={depth_factor})')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        return

    plt_sizes = list(sizes)
    plt_photons = list(photon_numbers)
    gus_grid = [[times_GUS.get((size, photons), float("nan")) for size in plt_sizes]
                for photons in plt_photons]
    ftn_grid = [[times_FTN.get((size, photons), float("nan")) for size in plt_sizes]
                for photons in plt_photons]

    # auto-save results
    try:
        saved_path = save_benchmark_results(times_GUS, times_FTN, sizes, photon_numbers,
                                            combining, is_open, circuit_type, depth_factor,
                                            timeout_seconds)
        print(f"Saved benchmark results to {saved_path}")
        try:
            display_results_from_file(saved_path)
        except (FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
            print(f"Warning: failed to display saved benchmark results: {e}")
    except (FileNotFoundError, PermissionError, TypeError) as e:
        print(f"Warning: failed to save benchmark results: {e}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    plots = [
        (axes[0], gus_grid, 'GUS'),
        (axes[1], ftn_grid, 'FTN'),
    ]

    for ax, grid, title in plots:
        image = ax.imshow(grid, origin='lower', aspect='auto', cmap='plasma')
        ax.set_title(title)
        ax.set_xticks(range(len(plt_sizes)))
        ax.set_xticklabels(plt_sizes)
        ax.set_yticks(range(len(plt_photons)))
        ax.set_yticklabels(plt_photons)
        ax.set_xlabel('Circuit size (number of modes)')
        ax.set_ylabel('Number of photons')
        fig.colorbar(image, ax=ax, label='Execution time (s)')

    fig.suptitle(f'Benchmark: simulation time vs size and photons ({circuit_type} circuits,'
                 f' depth factor={depth_factor}), {"open" if is_open else "closed"} circuits')
    fig.tight_layout()
    plt.show()


def photon_capped_linear(size: int) -> int:
    """Example photon number function that grows linearly with size but is capped at a maximum
    value."""
    return min(size, 5)

def main():
    """main function to run the benchmarks. Uncomment the desired benchmark to execute it."""
    # compare_open_circuits(max_size=6, depth_factor=0.25, circuit_type="mixed",
    #                       photon_function=math.ceil)
    # compare_closed_circuits(max_size=10, depth_factor=0.25, circuit_type="mixed",
    #                         photon_function=lambda size: 20)
    # only_GUS_benchmark(max_size=100, depth_factor=1, circuit_type="mixed",
    #                    photon_function=lambda size: size // 5)
    compare_photons_modes(sizes=range(20, 26, 2), photon_numbers=range(23, 27), combining="product",
                          is_open=False, circuit_type="mixed", depth_factor=.25)

if __name__ == "__main__":
    main()
    #display_results_from_file(os.path.join(os.path.dirname(__file__), "results", "benchmark_20260604_085623.json"))

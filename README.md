# QuOptics

**QuOptics** is a research-oriented Python codebase for simulating
multi-photon states propagating through passive linear-optical circuits. It
contains complementary implementations based on Fock-state amplitudes,
matrix permanents, symbolic polynomials, and tensor-network contraction.

The project was developed as part of the Quantum Computing Research Project
with Quandela at CentraleSupélec (2025–2026).

> **Status:** Work in progress. The repository is primarily an experimental
> and benchmarking codebase; APIs and module organization may evolve.

## What the project does

Given a circuit acting on a set of optical modes, QuOptics can:

- represent beam splitters and phase shifters as elementary optical gates;
- generate random Fock states and Haar-random unitary matrices;
- decompose arbitrary unitary matrices using a Clements interferometer;
- compute transition amplitudes between input and output Fock states;
- enumerate Fock states and construct complete output amplitude tensors;
- compute matrix permanents using Ryser- and Glynn-family algorithms;
- contract photonic circuits as tensor networks, including sparse tensors;
- build symbolic bosonic polynomials and derive measurement probabilities;
- visualize circuits, tensor networks, and probability distributions;
- compare direct Fock calculations with tensor-network calculations through
  reproducible benchmark scripts.

For an input occupation vector $n=(n_0,\ldots,n_{m-1})$, an output
occupation vector $s$, and a mode transformation $U$, the central
quantity is the bosonic transition amplitude

$$
\langle s|U|n\rangle =
\frac{\mathrm{Perm}(U_{s,n})}
{\sqrt{\prod_i s_i!\prod_j n_j!}},
$$

where $U_{s,n}$ is the matrix obtained by repeating rows and columns of
$U$ according to the output and input occupations.

## Repository layout

| Path | Purpose |
| --- | --- |
| [`PyFock/`](./PyFock/) | Fock-state enumeration, amplitudes, amplitude tensors, and permanent algorithms. |
| [`clements_scheme/`](./clements_scheme/) | Beam-splitter matrices and Clements decompositions of unitary matrices. |
| [`TeNCo/`](./TeNCo/) | Tensor-network circuit representation, contraction, sparse backends, and visualization. |
| [`Benchmark/`](./Benchmark/) | Random circuit generation and comparisons between the GUS and FTN approaches. |
| [`Reda/`](./Reda/) | Alternative permanent and bosonic-polynomial implementations used for comparison. |
| [`polynomial_measurement.py`](./polynomial_measurement.py) | Symbolic polynomial transformations and measurement-probability utilities. |
| [`rnd_module.py`](./rnd_module.py) | Random unitary and random Fock-state generation. |
| [`misc_and_tests/`](./misc_and_tests/) | Additional tests and exploratory notebooks. |

## Main simulation approaches

### Direct Fock-state amplitudes

[`PyFock/fock.py`](./PyFock/fock.py) provides routines for:

- enumerating all $N$-mode Fock states with a fixed photon number;
- calculating a single input-to-output amplitude;
- constructing the full amplitude tensor for a unitary transformation;
- handling common elementary gates such as beam splitters and phase shifters.

Several permanent backends are available in
[`PyFock/permanent.py`](./PyFock/permanent.py), including standard Ryser,
Gray-code Ryser, hyper-rectangle variants for repeated rows and columns,
Glynn, and Gray-code Glynn algorithms.

### Clements decomposition

[`clements_scheme/clements_scheme.py`](./clements_scheme/clements_scheme.py)
implements the Clements decomposition of an $N\times N$ unitary into
two-mode beam splitters and a final diagonal phase matrix. It also exposes
the elementary beam-splitter matrix, its inverse, and helper routines used
during the elimination procedure.

### Tensor-network contraction

[`TeNCo/circuit.py`](./TeNCo/circuit.py) models a photonic circuit as a
network of tensor gates. A `Lattice` can be populated with beam splitters,
phase shifters, input states, and output states, then contracted using the
available contraction strategies. The implementation supports sparse tensors
and prunes Fock configurations that exceed the configured photon number.

[`TeNCo/draw.py`](./TeNCo/draw.py) provides text and Matplotlib-based
visualizations, including circuit and tensor-network drawings. Generated
figures are optional outputs and are not required to run the simulator.

### Symbolic polynomial measurements

[`polynomial_measurement.py`](./polynomial_measurement.py) represents a
bosonic state as a SymPy multivariate polynomial. It can:

1. construct the initial polynomial $J_{m,n}=x_1\cdots x_n$;
2. apply a mode transformation by substituting the linear forms induced by a
   unitary matrix;
3. extract coefficients for occupation patterns;
4. convert coefficients into measurement probabilities using the appropriate
   factorial factors;
5. plot the resulting distribution.

This is useful for small systems where an explicit symbolic expression is
more informative than a numerical tensor.

## Installation

QuOptics is currently organized as a source repository rather than a
published package and does not yet provide a `pyproject.toml` or
`requirements.txt`. Create a virtual environment, clone the repository, and
install the numerical and symbolic dependencies used by the modules:

```bash
git clone https://github.com/Titouan-Moine/QuOptics.git
cd QuOptics

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install numpy scipy sympy matplotlib sparse
```

Some exploratory or optional modules may additionally use packages such as
`quimb` or `networkx`. Install them only when using the corresponding
module.

Run commands from the repository root so imports such as `PyFock.fock`,
`Benchmark.GUS`, and `TeNCo.circuit` resolve correctly.

## Quick start

### Generate a random interferometer and compute an amplitude

```python
import numpy as np

from rnd_module import random_unitary
from PyFock.fock import fock_amplitude

n_modes = 4
unitary = random_unitary(n_modes)

input_state = np.array([1, 1, 0, 0])
output_state = np.array([0, 1, 1, 0])

amplitude = fock_amplitude(
    unitary,
    input_state,
    output_state,
    method="glynn_gray",
)

print(amplitude)
print("Probability:", abs(amplitude) ** 2)
```

The `method` argument selects one of the permanent-based implementations
available in [`PyFock/fock.py`](./PyFock/fock.py), including
`ryser`, `ryser_gray`, `ryser_hyperrect`, `glynn`, and `glynn_gray`.

### Generate and display a random circuit

```python
from Benchmark.circuit_gen import rnd_circuit, display_circuit

circuit = rnd_circuit(
    n_modes=8,
    n_layers=4,
    circuit_type="lasagna",
)

display_circuit(circuit, method="plt", n_modes=8)
```

The circuit generator supports beam-splitter-only, Clements-like,
alternating (“lasagna”), and mixed beam-splitter/phase-shifter circuits.

## Benchmarks

The scripts in [`Benchmark/`](./Benchmark/) compare two calculation paths:

- **GUS (Global Unitary Synthesis):** contract the optical circuit into a mode transformation matrix,
  then compute Fock amplitudes using permanent-based methods;
- **FTN (Fock Tensor Network):** construct and contract the corresponding Fock tensor network.

The comparison utilities support open-circuit amplitude tensors, closed
circuits with specified input and output states, configurable circuit depth,
photon counts, timeouts, and JSON result files. Saved benchmark results are
stored under [`Benchmark/results/`](./Benchmark/results/). Plotting functions
display figures interactively; image files can be saved separately when
desired.

For example, benchmark helpers can be imported with:

```python
from Benchmark.compare_display import compare_open_circuits

compare_open_circuits(
    max_size=10,
    circuit_type="lasagna",
)
```

Benchmark sizes should be chosen carefully: the Fock-space dimension and the
cost of permanent evaluation grow rapidly with the number of photons and
modes.

## Validation

The repository includes focused tests for the Clements decomposition in
[`clements_scheme/testclements.py`](./clements_scheme/testclements.py), helper
tests in [`Benchmark/GUS.py`](./Benchmark/GUS.py), and additional checks in
[`misc_and_tests/`](./misc_and_tests/).

From the repository root, the standard-library test runner can be used for
the Clements tests:

```bash
python -m unittest clements_scheme.testclements
```

## Limitations and scope

- The project focuses on passive, finite-mode linear optics and does not
  provide a general-purpose quantum-computing SDK.
- Many routines are intended for small or moderate systems; exhaustive Fock
  tensors and symbolic expansions become expensive quickly.
- The public API is not yet versioned, packaged, or guaranteed stable.
- The repository contains experimental implementations and comparison code;
  benchmark results depend on the machine, dependency versions, and chosen
  circuit parameters.

## Authors and contributors

### Primary author and maintainer

**Titouan Moine**

Titouan Moine led the development and current maintenance of QuOptics,
including the permanent-computation methods, the Clements decomposition, the
TeNCo tensor-network framework, and the Benchmark suite.

### Initial contributors

The project was initially developed as a group project with:

- Gaëtan Lechoux
- Yassir Bouarbi
- Aurore Bellessa
- Reda Hamama
- Maroua El Fathi
- Matias Philippe

The project was conducted as part of the Quantum Computing Research Project
with Quandela at CentraleSupélec during 2025–2026.

## Reference

The interferometer decomposition follows the Clements scheme described in:

> W. R. Clements, J. J. Renema, and J. L. O’Brien, “Optimal design for
> universal multiport interferometers,” *Optica* 3, 1460–1465 (2016).

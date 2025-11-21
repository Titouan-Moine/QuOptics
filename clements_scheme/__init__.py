"""Clements scheme for unitary matrix decomposition.

This module implements the Clements decomposition scheme, which decomposes
arbitrary unitary matrices into a sequence of beam splitters and phase shifters.
This is particularly useful for linear optical quantum computing and photonic 
quantum information processing.

The main function `clements_scheme` takes an N×N unitary matrix and returns
a sequence of elementary operations (beam splitters with tunable parameters)
that can be used to implement the unitary transformation in a photonic or
linear optical system.

Key functions:
    - clements_scheme: Decompose a unitary matrix into beam splitters and phase shifters
    - T: Construct a beam splitter matrix
    - inverse_T: Construct the inverse of a beam splitter matrix
    - project_U2: Project a 2×2 matrix onto the unitary group U(2)
    - nullify_row: Calculate beam splitter parameters to nullify a matrix row
    - nullify_column: Calculate beam splitter parameters to nullify a matrix column
"""
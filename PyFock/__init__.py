"""Ryser module for matrix permanents in quantum computing.

This module provides efficient implementations of Ryser's algorithm for
computing the permanent of a matrix, which is a key operation in various
quantum computing applications, including boson sampling.

We also provide an ameliorated version of Ryser's algorithm that reduces computational
complexity for repeated sub-matrices. The worst case complexity remains O(n 2^n), but
many practical cases are significantly faster, with a best case complexity in O(n^2).

Key Functions :
    - 
"""
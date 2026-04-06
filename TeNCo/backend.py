"""Sparse tensor contraction backend.

This module implements a sparse tensor contraction function that uses scipy.sparse
for the actual multiplication, while keeping everything in sparse format to avoid
densification. The main function is `sparse_tensordot_via_scipy`, which takes two
sparse.COO tensors and contracts them along specified axes, returning a new sparse.COO
tensor as the result.
"""

import scipy.sparse as sps
import numpy as np
import sparse
import tracemalloc

def sparse_tensordot_via_scipy(a, b, axes_a, axes_b):
    """Contraction of two sparse.COO tensors via scipy.sparse, 
    without ever densifying.

    Parameters
    ----------
    a, b : sparse.COO
        Tensors to contract.
    axes_a, axes_b : list of int
        Axes to contract (like np.tensordot).
    
    Returns
    -------
    sparse.COO
        Result of the contraction.
    """
    # Identify free (non-contracted) axes
    free_axes_a = [i for i in range(a.ndim) if i not in axes_a]
    free_axes_b = [i for i in range(b.ndim) if i not in axes_b]
    
    # Dimensions
    free_shape_a = tuple(a.shape[i] for i in free_axes_a)
    free_shape_b = tuple(b.shape[i] for i in free_axes_b)
    
    free_size_a = int(np.prod(free_shape_a)) if free_shape_a else 1
    free_size_b = int(np.prod(free_shape_b)) if free_shape_b else 1
    contract_size = int(np.prod([a.shape[i] for i in axes_a])) if axes_a else 1

    # Transpose to align the axes to be contracted
    a_t = a.transpose(free_axes_a + axes_a)
    b_t = b.transpose(axes_b + free_axes_b)
    
    # Reshape into 2D matrices for scipy.sparse multiplication
    a_mat = a_t.reshape((free_size_a, contract_size))
    b_mat = b_t.reshape((contract_size, free_size_b))

    # Convert to scipy.sparse CSR for multiplication
    a_csr = sps.csr_matrix(a_mat.to_scipy_sparse())
    b_csc = sps.csc_matrix(b_mat.to_scipy_sparse())

    # Sparse matrix multiplication (NEVER dense)
    result_scipy = a_csr @ b_csc

    # Reconvert into sparse.COO and reshape
    result_coo = sparse.COO.from_scipy_sparse(result_scipy)
    result = result_coo.reshape(free_shape_a + free_shape_b)
    
    return result


def sparse_tensordot_via_scipy_debug(a, b, axes_a, axes_b):
    """Debug version that traces memory at each step."""
    tracemalloc.start()
    
    free_axes_a = [i for i in range(a.ndim) if i not in axes_a]
    free_axes_b = [i for i in range(b.ndim) if i not in axes_b]
    
    free_shape_a = tuple(a.shape[i] for i in free_axes_a)
    free_shape_b = tuple(b.shape[i] for i in free_axes_b)
    
    free_size_a = int(np.prod(free_shape_a)) if free_shape_a else 1
    free_size_b = int(np.prod(free_shape_b)) if free_shape_b else 1
    contract_size = int(np.prod([a.shape[i] for i in axes_a])) if axes_a else 1
    
    print(f"  a: shape={a.shape}, nnz={a.nnz}")
    print(f"  b: shape={b.shape}, nnz={b.nnz}")
    print(f"  free_size_a={free_size_a}, free_size_b={free_size_b}, contract_size={contract_size}")
    print(f"  Scipy Matrix will be: ({free_size_a} x {contract_size}) @ ({contract_size} x {free_size_b})")
    
    snapshot = tracemalloc.take_snapshot()
    print(f"  Memory before transpose: {tracemalloc.get_traced_memory()[1]/1e6:.1f} MB peak")

    a_t = a.transpose(free_axes_a + axes_a)
    snapshot = tracemalloc.take_snapshot()
    print(f"  Memory after transpose a: {tracemalloc.get_traced_memory()[1]/1e6:.1f} MB peak")

    # THIS IS THE POTENTIAL PROBLEM AREA
    a_mat = a_t.reshape((free_size_a, contract_size))
    snapshot = tracemalloc.take_snapshot()
    print(f"  Memory after reshape a: {tracemalloc.get_traced_memory()[1]/1e6:.1f} MB peak")

    tracemalloc.stop()

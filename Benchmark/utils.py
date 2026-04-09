"""Utils for benchmarking quantum circuits"""

# import numpy as np
import sparse

def sparse_allclose(a, b, atol=1e-8):
    """Check if two sparse tensors are approximately equal."""
    if a.shape != b.shape:
        # print(f"Shape mismatch: {a.shape} vs {b.shape}")
        return False
    diff = a - b
    return sparse.all(sparse.abs(diff) <= atol)


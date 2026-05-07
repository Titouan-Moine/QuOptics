"""
Utility functions for tensor networks.
"""

def find_duplicates(l: list | tuple) -> set:
    """Find duplicate elements in a list and return them as a set."""
    seen = set()
    duplicates = set()
    for item in l:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return duplicates
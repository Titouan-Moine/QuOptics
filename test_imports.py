#!/usr/bin/env python3
"""
Test script to verify all imports work correctly from various execution contexts.
This simulates what happens when running files via VS Code's "Run Python File" button.
"""

import sys
import os
import traceback

def test_clements_imports():
    """Test imports from clements_scheme module"""
    print("Testing clements_scheme imports...")
    try:
        from clements_scheme.clements_scheme import full_clements
        from rnd_module import random_unitary
        U = random_unitary(3)
        result = full_clements(U)
        print("✓ clements_scheme imports: OK\n")
        return True
    except Exception as e:
        print(f"✗ clements_scheme imports FAILED: {e}")
        traceback.print_exc()
        print()
        return False


def test_ryser_imports():
    """Test imports from ryser module"""
    print("Testing ryser imports...")
    try:
        from ryser.permanent import ryser
        from rnd_module import random_unitary
        U = random_unitary(3)
        result = ryser(U)
        print("✓ ryser imports: OK\n")
        return True
    except Exception as e:
        print(f"✗ ryser imports FAILED: {e}")
        traceback.print_exc()
        print()
        return False


def test_infoq_package_imports():
    """Test imports from infoq package"""
    print("Testing infoq package imports...")
    try:
        # When running from within infoq directory, add parent to path
        import sys
        import os
        parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if parent not in sys.path:
            sys.path.insert(0, parent)
        
        from infoq import random_unitary
        U = random_unitary(3)
        print("✓ infoq package imports: OK\n")
        return True
    except Exception as e:
        print(f"✗ infoq package imports FAILED: {e}")
        traceback.print_exc()
        print()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Running comprehensive import tests")
    print("=" * 60)
    print()
    
    all_passed = True
    all_passed &= test_clements_imports()
    all_passed &= test_ryser_imports()
    all_passed &= test_infoq_package_imports()
    
    print("=" * 60)
    if all_passed:
        print("✓ All import tests PASSED!")
        sys.exit(0)
    else:
        print("✗ Some import tests FAILED")
        sys.exit(1)

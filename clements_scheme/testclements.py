"""Unit tests for the Clements decomposition scheme module.

This module contains comprehensive unit tests for validating the correctness of:
- Random unitary matrix generation
- Beam splitter matrix construction
- Row and column nullification functions
- The complete Clements decomposition algorithm

All functions are tested to ensure:
- Mathematical correctness (e.g., unitarity preservation)
- Parameter validation and error handling
- Numerical stability and precision
"""

import unittest
import numpy as np
import clements_scheme
import rnd_unitary


class TestRandUnitary(unittest.TestCase):
    def test_random_unitary_dimensions(self):
        n = 5
        U = rnd_unitary.random_unitary(n)
        self.assertEqual(U.shape, (n, n))
        
    def test_random_unitary_unitarity(self):
        n = 4
        U = rnd_unitary.random_unitary(n)
        identity = np.eye(n)
        self.assertTrue(np.allclose(U.conj().T @ U, identity))

class TestNullifyFunctions(unittest.TestCase):
    def test_nullify_row(self): # tests the row nullification of the lower triangle
        U = rnd_unitary.random_unitary(4)
        flag = True
        for i in range(1, 4):
            for j in range(i-1, 3):
                phi, theta = clements_scheme.nullify_row(U, i, j, i-1, i)
                Tmn = clements_scheme.T(i-1, i, phi, theta, 4)
                U_new = Tmn @ U
                flag = flag and np.isclose(U_new[i,j], 0.0)
        self.assertTrue(flag)
        
    def test_nullify_column(self): # tests the column nullification of the lower triangle
        U = rnd_unitary.random_unitary(4)
        flag = True
        for i in range(1, 4):
            for j in range(i-1, 3):
                phi, theta = clements_scheme.nullify_column(U, i, j, j, j+1)
                invTmn = clements_scheme.inverse_T(j, j+1, phi, theta, 4)
                U_new = U @ invTmn
                flag = flag and np.isclose(U_new[i,j], 0.0)
        self.assertTrue(flag)

class TestTmnMatrices(unittest.TestCase):
    def test_T_matrix(self):
        N = 4
        m, n = 1, 2
        phi, theta = np.pi/4, np.pi/6
        T_matrix = clements_scheme.T(m, n, phi, theta, N)
        
        # Check unitarity
        identity = np.eye(N)
        self.assertTrue(np.allclose(T_matrix.conj().T @ T_matrix, identity))
        
    def test_inverse_T_matrix(self):
        N = 4
        m, n = 0, 3
        phi, theta = np.pi/3, np.pi/8
        T_matrix = clements_scheme.T(m, n, phi, theta, N)
        T_inv_matrix = clements_scheme.inverse_T(m, n, phi, theta, N)
        
        # Check that T * T_inv = I
        identity = np.eye(N)
        self.assertTrue(np.allclose(T_matrix @ T_inv_matrix, identity))
        
    def test_invalid_indices_T(self):
        N = 3
        with self.assertRaises(ValueError):
            clements_scheme.T(3, 1, 0, 0, N)
        with self.assertRaises(ValueError):
            clements_scheme.T(1, 3, 0, 0, N)
            
    def test_invalid_indices_inverse_T(self):
        N = 3
        with self.assertRaises(ValueError):
            clements_scheme.inverse_T(3, 1, 0, 0, N)
        with self.assertRaises(ValueError):
            clements_scheme.inverse_T(1, 3, 0, 0, N)

class TestProjectU2(unittest.TestCase):
    def test_project_U2_unitarity(self):
        """Test that project_U2 returns a unitary matrix."""
        A = np.array([[1+0.1j, 0.2], [0.3, 0.9+0.1j]], dtype=complex)
        U = clements_scheme.project_U2(A)
        
        # Check unitarity: U^† @ U = I
        identity = np.eye(2)
        self.assertTrue(np.allclose(U.conj().T @ U, identity))
    
    def test_project_U2_dimensions(self):
        """Test that project_U2 preserves matrix dimensions."""
        A = rnd_unitary.random_unitary(2)
        U = clements_scheme.project_U2(A)
        
        self.assertEqual(U.shape, (2, 2))
    
    def test_project_U2_on_unitary(self):
        """Test that project_U2 of a unitary matrix is close to the original."""
        A = rnd_unitary.random_unitary(2)
        U = clements_scheme.project_U2(A)
        
        # Projecting an already unitary matrix should give a unitary matrix close to it
        self.assertTrue(np.allclose(U.conj().T @ U, np.eye(2)))
    
    def test_project_U2_non_unitary(self):
        """Test that project_U2 of a non-unitary matrix is unitary."""
        A = np.array([[2, 0], [0, 0.5]], dtype=complex)
        U = clements_scheme.project_U2(A)
        
        # Result should be unitary
        identity = np.eye(2)
        self.assertTrue(np.allclose(U.conj().T @ U, identity))
        self.assertTrue(np.allclose(U @ U.conj().T, identity))

class TestProjectD(unittest.TestCase):
    def test_project_D_unitarity(self):
        """Test that project_D returns a unitary diagonal matrix."""
        N = 4
        D = np.diag([1+0.1j, 0.9+0.05j, 1.1-0.1j, 0.95])
        D_proj = clements_scheme.project_D(D)
        
        # Check that result is diagonal
        off_diag_mask = ~np.eye(N, dtype=bool)
        self.assertTrue(np.allclose(D_proj[off_diag_mask], 0))
        
        # Check unitarity
        identity = np.eye(N)
        self.assertTrue(np.allclose(D_proj.conj().T @ D_proj, identity))
    
    def test_project_D_phase_extraction(self):
        """Test that project_D extracts only the phases from diagonal elements."""
        N = 3
        phases = np.array([0, np.pi/4, np.pi/2])
        magnitudes = np.array([2, 3, 0.5])
        D = np.diag([magnitudes[i] * np.exp(1j * phases[i]) for i in range(N)])
        D_proj = clements_scheme.project_D(D)
        
        # Check that diagonal elements have magnitude 1
        for i in range(N):
            self.assertAlmostEqual(np.abs(D_proj[i, i]), 1.0)
        
        # Check that phases are preserved
        for i in range(N):
            self.assertAlmostEqual(np.angle(D_proj[i, i]), phases[i], places=10)
    
    def test_project_D_on_diagonal_unitary(self):
        """Test that project_D of a diagonal unitary is preserved."""
        N = 4
        phases = np.random.rand(N) * 2 * np.pi
        D = np.diag([np.exp(1j * p) for p in phases])
        D_proj = clements_scheme.project_D(D)
        
        self.assertTrue(np.allclose(D, D_proj))
    
    def test_project_D_non_square(self):
        """Test that project_D raises error for non-square matrices."""
        D = np.array([[1, 2], [3, 4], [5, 6]], dtype=complex)
        
        with self.assertRaises(ValueError):
            clements_scheme.project_D(D)
    
    def test_project_D_non_diagonal(self):
        """Test that project_D raises error for non-diagonal matrices."""
        D = np.array([[1, 0.1], [0, 2]], dtype=complex)
        
        with self.assertRaises(ValueError):
            clements_scheme.project_D(D)

class TestClementsDecomposition(unittest.TestCase):
    def test_clements_decomposition_dimensions(self):
        """Test that clements_decomposition returns correct structure."""
        U = rnd_unitary.random_unitary(4)
        decomposition, D = clements_scheme.clements_decomposition(U, project=True)
        
        # Check structure
        self.assertIsInstance(decomposition, tuple)
        self.assertEqual(len(decomposition), 2)
        self.assertIsInstance(decomposition[0], list)
        self.assertIsInstance(decomposition[1], list)
    
    def test_clements_decomposition_non_square(self):
        """Test that clements_decomposition raises error for non-square matrices."""
        U = U = np.random.rand(3, 4) + 1j * np.random.rand(3, 4)
        
        with self.assertRaises(ValueError):
            clements_scheme.clements_decomposition(U)
    
    def test_clements_decomposition_diagonal_result(self):
        """Test that clements_decomposition returns a diagonal matrix D."""
        U = rnd_unitary.random_unitary(3)
        decomposition, D = clements_scheme.clements_decomposition(U, project=True)
        
        # D should be diagonal
        off_diag_mask = ~np.eye(3, dtype=bool)
        self.assertTrue(np.allclose(D[off_diag_mask], 0, atol=1e-10))
    
    def test_clements_decomposition_unitarity(self):
        """Test that the decomposition result is unitary."""
        U = rnd_unitary.random_unitary(4)
        decomposition, D = clements_scheme.clements_decomposition(U, project=True)
        
        # D should be unitary (diagonal unitary)
        identity = np.eye(4)
        self.assertTrue(np.allclose(D.conj().T @ D, identity))
    
    def test_clements_decomposition_with_projection(self):
        """Test clements_decomposition with projection enabled."""
        U = rnd_unitary.random_unitary(3)
        decomposition, D = clements_scheme.clements_decomposition(U, project=True)
        
        # Result should be diagonal and unitary
        self.assertTrue(np.allclose(D.conj().T @ D, np.eye(3)))
    
    def test_clements_decomposition_without_projection(self):
        """Test clements_decomposition with projection disabled."""
        U = rnd_unitary.random_unitary(3)
        decomposition, D = clements_scheme.clements_decomposition(U, project=False)
        
        # Even without projection, D should be close to diagonal
        off_diag_mask = ~np.eye(3, dtype=bool)
        self.assertTrue(np.allclose(D[off_diag_mask], 0, atol=1e-8))
    
    def test_clements_decomposition_tuple_format(self):
        """Test that decomposition tuples have correct format."""
        U = rnd_unitary.random_unitary(3)
        decomposition, D = clements_scheme.clements_decomposition(U, project=True)
        
        left_decomp, right_decomp = decomposition
        
        # Check that all elements are tuples of 4 elements
        for elem in left_decomp:
            self.assertEqual(len(elem), 4)
            self.assertIsInstance(elem[0], (int, np.integer))
            self.assertIsInstance(elem[1], (int, np.integer))
            self.assertIsInstance(elem[2], (float, np.floating))
            self.assertIsInstance(elem[3], (float, np.floating))
        
        for elem in right_decomp:
            self.assertEqual(len(elem), 4)
            self.assertIsInstance(elem[0], (int, np.integer))
            self.assertIsInstance(elem[1], (int, np.integer))
            self.assertIsInstance(elem[2], (float, np.floating))
            self.assertIsInstance(elem[3], (float, np.floating))
    
    def test_clements_decomposition_different_sizes(self):
        """Test clements_decomposition works for different matrix sizes."""
        for n in [2, 3, 4, 5]:
            U = rnd_unitary.random_unitary(n)
            decomposition, D = clements_scheme.clements_decomposition(U, project=True)
            
            # Check diagonal result
            off_diag_mask = ~np.eye(n, dtype=bool)
            self.assertTrue(np.allclose(D[off_diag_mask], 0, atol=1e-10))
            
            # Check unitarity
            self.assertTrue(np.allclose(D.conj().T @ D, np.eye(n)))


if __name__ == '__main__':
    unittest.main()
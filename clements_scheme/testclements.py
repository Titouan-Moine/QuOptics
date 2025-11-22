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

class TestWrapAngleFunction(unittest.TestCase):
    def test_wrap_angle_within_bounds(self):
        angles = [0, np.pi/2, np.pi, -np.pi/2, -np.pi]
        for angle in angles:
            wrapped = clements_scheme.wrap_angle(angle)
            self.assertTrue(-np.pi <= wrapped <= np.pi)
            self.assertAlmostEqual(wrapped, angle)
    
    def test_wrap_angle_exceeding_bounds(self):
        test_cases = [
            (4 * np.pi, 0),
            (-4 * np.pi, 0),
            (5 * np.pi / 2, np.pi / 2),
            (-5 * np.pi / 2, -np.pi / 2)
        ]
        for angle, expected in test_cases:
            wrapped = clements_scheme.wrap_angle(angle)
            self.assertTrue(-np.pi <= wrapped <= np.pi)
            self.assertAlmostEqual(wrapped, expected)

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
    
    def test_clements_decomposition_correctness(self):
        """Test that clements_decomposition correctly decomposes the unitary matrix."""
        U = rnd_unitary.random_unitary(4)
        decomposition, D = clements_scheme.clements_decomposition(U, project=True)
        left_decomp, right_decomp = decomposition
        
        # Reconstruct U from the decomposition
        N = U.shape[0]
        U_reconstructed = D.copy()
        
        # Apply right decomposition
        for m, n, phi, theta in right_decomp:
            Tmn = clements_scheme.T(m, n, phi, theta, N)
            U_reconstructed = U_reconstructed @ Tmn
        
        # Apply left decomposition
        for m, n, phi, theta in left_decomp:
            invTmn = clements_scheme.inverse_T(m, n, phi, theta, N)
            U_reconstructed = invTmn @ U_reconstructed
        
        # Check that reconstructed U is close to original U
        self.assertTrue(np.allclose(U_reconstructed, U, atol=1e-10))
    
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

class TestClementsInvertLeft(unittest.TestCase):
    def test_clements_invert_left_output_structure(self):
        """Test that clements_invert_left returns correct structure."""
        U = rnd_unitary.random_unitary(4)
        decomposition, D = clements_scheme.clements_decomposition(U, project=True)
        left_decomp, right_decomp = decomposition
        
        inverted_left, D_final = clements_scheme.clements_invert_left(D, left_decomp, project=True)
        
        # Check that inverted_left is a list
        self.assertIsInstance(inverted_left, list)
        
        # Check that D_final is a numpy array
        self.assertIsInstance(D_final, np.ndarray)
        
        # Check dimensions
        self.assertEqual(D_final.shape, D.shape)
    
    def test_clements_invert_left_tuple_format(self):
        """Test that inverted left decomposition tuples have correct format."""
        U = rnd_unitary.random_unitary(3)
        decomposition, D = clements_scheme.clements_decomposition(U, project=True)
        left_decomp, right_decomp = decomposition
        
        inverted_left, D_final = clements_scheme.clements_invert_left(D, left_decomp, project=True)
        
        # Check that all elements are tuples of 4 elements
        for elem in inverted_left:
            self.assertEqual(len(elem), 4)
            self.assertIsInstance(elem[0], (int, np.integer))
            self.assertIsInstance(elem[1], (int, np.integer))
            self.assertIsInstance(elem[2], (float, np.floating))
            self.assertIsInstance(elem[3], (float, np.floating))
    
    def test_clements_invert_left_diagonal_result(self):
        """Test that D_final from clements_invert_left is diagonal."""
        U = rnd_unitary.random_unitary(4)
        decomposition, D = clements_scheme.clements_decomposition(U, project=True)
        left_decomp, right_decomp = decomposition
        
        inverted_left, D_final = clements_scheme.clements_invert_left(D, left_decomp, project=True)
        
        # D_final should be diagonal
        off_diag_mask = ~np.eye(4, dtype=bool)
        self.assertTrue(np.allclose(D_final[off_diag_mask], 0, atol=1e-10))
    
    def test_clements_invert_left_unitarity(self):
        """Test that D_final is unitary."""
        U = rnd_unitary.random_unitary(3)
        decomposition, D = clements_scheme.clements_decomposition(U, project=True)
        left_decomp, right_decomp = decomposition
        
        inverted_left, D_final = clements_scheme.clements_invert_left(D, left_decomp, project=True)
        
        # D_final should be unitary
        identity = np.eye(3)
        self.assertTrue(np.allclose(D_final.conj().T @ D_final, identity))
    
    def test_clements_invert_left_with_projection(self):
        """Test clements_invert_left with projection enabled."""
        U = rnd_unitary.random_unitary(4)
        decomposition, D = clements_scheme.clements_decomposition(U, project=True)
        left_decomp, right_decomp = decomposition

        inverted_left, D_final = clements_scheme.clements_invert_left(D, left_decomp, project=True)

        # Result should be diagonal and unitary
        off_diag_mask = ~np.eye(4, dtype=bool)
        self.assertTrue(np.allclose(D_final[off_diag_mask], 0, atol=1e-10))
        self.assertTrue(np.allclose(D_final.conj().T @ D_final, np.eye(4)))

    def test_clements_invert_left_without_projection(self):
        """Test clements_invert_left with projection disabled."""
        U = rnd_unitary.random_unitary(3)
        decomposition, D = clements_scheme.clements_decomposition(U, project=True)
        left_decomp, right_decomp = decomposition

        inverted_left, D_final = clements_scheme.clements_invert_left(D, left_decomp, project=False)

        # Even without projection, D_final should be close to diagonal
        off_diag_mask = ~np.eye(3, dtype=bool)
        self.assertTrue(np.allclose(D_final[off_diag_mask], 0, atol=1e-8))

    def test_clements_invert_left_preserves_dimension(self):
        """Test that clements_invert_left preserves matrix dimension."""
        for n in [2, 3, 4, 5]:
            U = rnd_unitary.random_unitary(n)
            decomposition, D = clements_scheme.clements_decomposition(U, project=True)
            left_decomp, right_decomp = decomposition

            inverted_left, D_final = clements_scheme.clements_invert_left(D, left_decomp, project=True)

            self.assertEqual(D_final.shape, (n, n))

    def test_clemets_invert_left_correctness(self):
        """test that clements_invert_left correctly inverts the left decomposition."""
        U = rnd_unitary.random_unitary(4)
        decomposition, D = clements_scheme.clements_decomposition(U, project=True)
        left_decomp, right_decomp = decomposition
        inverted_left, D_prime = clements_scheme.clements_invert_left(D, left_decomp, project=True)
        N = U.shape[0]

        for m, n, phi, theta in inverted_left:
            Tmn = clements_scheme.T(m, n, phi, theta, N)
            D_prime = D_prime @ Tmn

        for m, n, phi, theta in left_decomp:
            invTmn = clements_scheme.inverse_T(m, n, phi, theta, N)
            D = invTmn @ D

        self.assertTrue(np.allclose(D_prime, D, atol=1e-10))

class TestFullClements(unittest.TestCase):
    def test_full_clements_output_structure(self):
        """Test that full_clements returns correct structure."""
        U = rnd_unitary.random_unitary(4)
        full_decomp, D_final = clements_scheme.full_clements(U, project=True)
        
        # Check that full_decomp is a list
        self.assertIsInstance(full_decomp, list)
        
        # Check that D_final is a numpy array
        self.assertIsInstance(D_final, np.ndarray)
    
    def test_full_clements_tuple_format(self):
        """Test that full_clements tuples have correct format."""
        U = rnd_unitary.random_unitary(3)
        full_decomp, D_final = clements_scheme.full_clements(U, project=True)
        
        # Check that all elements are tuples of 4 elements
        for elem in full_decomp:
            self.assertEqual(len(elem), 4)
            self.assertIsInstance(elem[0], (int, np.integer))
            self.assertIsInstance(elem[1], (int, np.integer))
            self.assertIsInstance(elem[2], (float, np.floating))
            self.assertIsInstance(elem[3], (float, np.floating))
    
    def test_full_clements_non_square(self):
        """Test that full_clements raises error for non-square matrices."""
        U = np.random.rand(3, 4) + 1j * np.random.rand(3, 4)
        
        with self.assertRaises(ValueError):
            clements_scheme.full_clements(U)
    
    def test_full_clements_non_unitary(self):
        """Test that full_clements raises error for non-unitary matrices."""
        U = np.random.rand(3, 3) + 1j * np.random.rand(3, 3)
        
        with self.assertRaises(ValueError):
            clements_scheme.full_clements(U)
    
    def test_full_clements_diagonal_result(self):
        """Test that D_final from full_clements is diagonal."""
        U = rnd_unitary.random_unitary(4)
        full_decomp, D_final = clements_scheme.full_clements(U, project=True)
        
        # D_final should be diagonal
        off_diag_mask = ~np.eye(4, dtype=bool)
        self.assertTrue(np.allclose(D_final[off_diag_mask], 0, atol=1e-10))
    
    def test_full_clements_unitarity(self):
        """Test that D_final is unitary."""
        U = rnd_unitary.random_unitary(3)
        full_decomp, D_final = clements_scheme.full_clements(U, project=True)
        
        # D_final should be unitary
        identity = np.eye(3)
        self.assertTrue(np.allclose(D_final.conj().T @ D_final, identity))
    
    def test_full_clements_with_projection(self):
        """Test full_clements with projection enabled."""
        U = rnd_unitary.random_unitary(4)
        full_decomp, D_final = clements_scheme.full_clements(U, project=True)
        
        # Result should be diagonal and unitary
        off_diag_mask = ~np.eye(4, dtype=bool)
        self.assertTrue(np.allclose(D_final[off_diag_mask], 0, atol=1e-10))
        self.assertTrue(np.allclose(D_final.conj().T @ D_final, np.eye(4)))
    
    def test_full_clements_without_projection(self):
        """Test full_clements with projection disabled."""
        U = rnd_unitary.random_unitary(3)
        full_decomp, D_final = clements_scheme.full_clements(U, project=False)
        
        # Even without projection, D_final should be close to diagonal
        off_diag_mask = ~np.eye(3, dtype=bool)
        self.assertTrue(np.allclose(D_final[off_diag_mask], 0, atol=1e-8))
    
    def test_full_clements_preserves_dimension(self):
        """Test that full_clements preserves matrix dimension."""
        for n in [2, 3, 4, 5]:
            U = rnd_unitary.random_unitary(n)
            full_decomp, D_final = clements_scheme.full_clements(U, project=True)
            
            self.assertEqual(D_final.shape, (n, n))
    
    def test_full_clements_consistency(self):
        """Test that full_clements gives consistent results for the same input."""
        U = rnd_unitary.random_unitary(3)
        full_decomp1, D_final1 = clements_scheme.full_clements(U, project=True)
        full_decomp2, D_final2 = clements_scheme.full_clements(U, project=True)
        
        # Results should be identical
        self.assertEqual(len(full_decomp1), len(full_decomp2))
        self.assertTrue(np.allclose(D_final1, D_final2))
    
    def test_full_clements_valid_operations(self):
        """Test that full_clements decomposition contains only valid operations."""
        U = rnd_unitary.random_unitary(4)
        full_decomp, D_final = clements_scheme.full_clements(U, project=True)
        
        # All beam splitters should have valid indices
        N = U.shape[0]
        for m, n, phi, theta in full_decomp:
            self.assertGreaterEqual(m, 0)
            self.assertGreaterEqual(n, 0)
            self.assertLess(m, N)
            self.assertLess(n, N)
            # phi and theta should be real numbers
            self.assertIsInstance(phi, (float, np.floating))
            self.assertIsInstance(theta, (float, np.floating))
    
    def test_full_clements_different_sizes(self):
        """Test full_clements works for different matrix sizes."""
        for n in [2, 3, 4, 5]:
            U = rnd_unitary.random_unitary(n)
            full_decomp, D_final = clements_scheme.full_clements(U, project=True)
            
            # Check diagonal result
            off_diag_mask = ~np.eye(n, dtype=bool)
            self.assertTrue(np.allclose(D_final[off_diag_mask], 0, atol=1e-10))
            
            # Check unitarity
            self.assertTrue(np.allclose(D_final.conj().T @ D_final, np.eye(n)))
    
    def test_full_clements_correctness(self):
        """Test that full_clements correctly decomposes the unitary matrix."""
        U = rnd_unitary.random_unitary(4)
        full_decomp, D_final = clements_scheme.full_clements(U, project=True)
        N = U.shape[0]
        
        # Reconstruct U from the decomposition
        U_reconstructed = D_final.copy()
        
        for m, n, phi, theta in full_decomp:
            Tmn = clements_scheme.T(m, n, phi, theta, N)
            U_reconstructed = U_reconstructed @ Tmn
        
        # Check that reconstructed U is close to original U
        self.assertTrue(np.allclose(U_reconstructed, U, atol=1e-10))
    
    def test_full_clements_bs_order(self):
        """Test that full_clements returns beam splitters in correct order."""
        U = rnd_unitary.random_unitary(3)
        full_decomp, D_final = clements_scheme.full_clements(U, project=True)
        
        # Check that beam splitters are ordered correctly
        N = U.shape[0]
        bs_index = 0
        for i in range(2, N, 2):
            for j in range(i):
                m, n, _, _ = full_decomp[bs_index + j]
                self.assertEqual(m + 1, n)
                self.assertEqual(n, N - i + j)
            bs_index += i
        start = N-1 if N % 2 == 0 else N-2
        for i in range(start, 0, -2):
            for j in range(i):
                m, n, _, _ = full_decomp[bs_index + j]
                self.assertEqual(m + 1, n)
                self.assertEqual(m, j)
            bs_index += i

if __name__ == '__main__':
    unittest.main()
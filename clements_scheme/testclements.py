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

class TestClementsScheme(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
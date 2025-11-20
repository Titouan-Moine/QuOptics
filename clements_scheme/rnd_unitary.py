import numpy as np

def random_unitary(n):
    # matrice complexe gaussienne
    Z = (np.random.randn(n, n) + 1j*np.random.randn(n, n)) / np.sqrt(2)

    # décomposition QR
    Q, R = np.linalg.qr(Z)

    # correction des phases
    D = np.diag(np.exp(1j * np.angle(np.diag(R))))
    return Q @ D
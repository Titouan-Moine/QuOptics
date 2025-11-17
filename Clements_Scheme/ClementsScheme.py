import numpy as np


def clements_scheme(U):    """
    Decomposes a unitary matrix U into a sequence of beam splitters and phase shifters
    according to the Clements scheme.

    Parameters:
    U (np.ndarray): A unitary matrix of shape (N, N).

    Returns:
    list: A list of tuples representing the beam splitters and phase shifters.
          Each tuple contains (type, parameters), where type is 'BS' for beam splitter
          or 'PS' for phase shifter, and parameters are the corresponding values.
    """
    
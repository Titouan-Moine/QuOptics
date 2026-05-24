
from typing import TypeAlias
from collections.abc import Sequence

BSGate: TypeAlias = tuple[int, int, float, float]  # (m, n, phi, theta)
PSGate: TypeAlias = tuple[int, float]  # (m, phi)
Gate: TypeAlias = BSGate | PSGate
Circuit: TypeAlias = Sequence[Gate]  # (list of gates, Dfinal)

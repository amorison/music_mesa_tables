from enum import Enum, auto
from numpy.typing import NDArray
import numpy as np


class StateVar(Enum):
    Density = auto()
    Pressure = auto()
    Pgas = auto()
    Temperature = auto()
    DPresDDensEcst = auto()
    DPresDEnerDcst = auto()
    DTempDDensEcst = auto()
    DTempDEnerDcst = auto()
    Entropy = auto()
    DTempDPresScst = auto()
    Gamma1 = auto()
    Gamma = auto()


class CstCompoState:
    def __init__(
        self,
        metallicity: float,
        he_frac: float,
        density: NDArray[np.float64],
        energy: NDArray[np.float64]
    ): ...

    def set_state(
        self,
        density: NDArray[np.float64],
        energy: NDArray[np.float64]
    ): ...

    def compute(self, var: StateVar): ...


class CstMetalState:
    def __init__(
        self,
        metallicity: float,
        he_frac: NDArray[np.float64],
        density: NDArray[np.float64],
        energy: NDArray[np.float64]
    ): ...

    def set_state(
        self,
        he_frac: NDArray[np.float64],
        density: NDArray[np.float64],
        energy: NDArray[np.float64]
    ): ...

    def compute(self, var: StateVar): ...

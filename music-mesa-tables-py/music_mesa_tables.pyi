from enum import Enum, auto
from numpy.typing import NDArray
import numpy as np


def get_mesa_tables_version() -> str: ...


class CstCompoEos:
    def __init__(self, metallicity: float, he_frac: float): ...


class CstMetalEos:
    def __init__(self, metallicity: float): ...


class StateVar(Enum):
    LogDensity = auto()
    LogPressure = auto()
    LogPgas = auto()
    LogTemperature = auto()
    DPresDDensEcst = auto()
    DPresDEnerDcst = auto()
    DTempDDensEcst = auto()
    DTempDEnerDcst = auto()
    LogEntropy = auto()
    DTempDPresScst = auto()
    Gamma1 = auto()
    Gamma = auto()


class CstCompoState:
    def __init__(
        self,
        table: CstCompoEos,
        density: NDArray[np.float64],
        energy: NDArray[np.float64]
    ): ...

    def compute(self, var: StateVar): ...


class CstMetalState:
    def __init__(
        self,
        table: CstMetalEos,
        he_frac: NDArray[np.float64],
        density: NDArray[np.float64],
        energy: NDArray[np.float64]
    ): ...

    def compute(self, var: StateVar): ...


class CstCompoOpacity:
    def __init__(self, state: CstCompoState): ...
    def log_opacity(self) -> NDArray[np.float64]: ...


class CstMetalOpacity:
    def __init__(self, state: CstMetalState): ...
    def log_opacity(self) -> NDArray[np.float64]: ...

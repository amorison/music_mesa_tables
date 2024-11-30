import music_mesa_tables as mmt
import numpy as np


def test_const_compo() -> None:
    eos = mmt.CstCompoEos(metallicity=0.02, he_frac=0.28)

    density = np.array([1.05, 15.7, 134.9])
    internal_energy = np.array([1e12, 1e15, 3e15])
    state = mmt.CstCompoState(eos, density, internal_energy)

    temperature = 10 ** state.compute(mmt.StateVar.LogTemperature)
    expected = [2.21745558e3, 5.00852231e6, 1.48317986e7]
    assert np.allclose(temperature, expected)

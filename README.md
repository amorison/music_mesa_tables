# music-mesa-tables

Fast implementation of the `'mesa'` equation of state as used in the MUSIC
project.

MUSIC is a stellar convection code. See e.g.
[Viallet _et al_ (2016)](https://doi.org/10.1051/0004-6361/201527339),
[Goffrey _et al_ (2017)](https://doi.org/10.1051/0004-6361/201628960).

The so-called `'mesa'` equation of state implemented in MUSIC relies on tables
from the [MESA project](https://docs.mesastar.org) (version 15140).
In the regimes typically simulated in MUSIC, the relevant tables are:

- FreeEOS, from [Irwin](http://freeeos.sourceforge.net/);
- OPAL, from [Rogers and Nayfonov (2002)](https://doi.org/10.1086/341894);
- SCVH, from [Saumon, Chabrier, and van Horn (1995)](https://doi.org/10.1086/192204).

See the [MESA documentation for more details](https://docs.mesastar.org/en/r15140/eos/overview.html).

For use in MUSIC, these tables are reinterpolated in (density, internal energy)
space. The reinterpolated tables are bundled in this package. This package is
intended for post-processing of data from MUSIC simulations. To build stellar
evolution models, refer to the MESA project.

---

This offers a Python API. Here is a simple example:

```python
import music_mesa_tables as mmt
import numpy as np

eos = mmt.CstCompoEos(metallicity=0.02, he_frac=0.28)

density = np.array([1.05, 15.7, 134.9])
internal_energy = np.array([1e12, 1e15, 3e15])
state = mmt.CstCompoState(eos, density, internal_energy)

temperature = 10 ** state.compute(mmt.StateVar.LogTemperature)
# array([2.21745558e+03, 5.00852231e+06, 1.48317986e+07])
```

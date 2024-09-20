# music-mesa-tables

Fast implementation of the `'mesa'` equation of state as used in the MUSIC
project.

MUSIC is a stellar convection code. See e.g.
[Viallet _et al_ (2016)](https://doi.org/10.1051/0004-6361/201527339),
[Goffrey _et al_ (2017)](https://doi.org/10.1051/0004-6361/201628960).

The so-called `'mesa'` equation of state implemented in MUSIC relies on tables
from the [MESA project](https://docs.mesastar.org) (version 15140),
reinterpolated in (density, internal energy) space. In the regimes typically
simulated in MUSIC, those tables come from the work of
[Rogers and Nayfonov (2002)](https://doi.org/10.1086/341894).
The reinterpolated tables are bundled in this package.

This package is intended for post-processing of data from MUSIC simulations.
To build stellar evolution models, refer to the MESA project.

---

This offers a Python API.

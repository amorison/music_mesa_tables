# music-mesa-tables

Fast and convenient interpolation routines for the MESA EoS tables used by the
MUSIC project.

This offers a Python API.  Precompiled wheels for macOS are produced with a
GitHub action (only for `x86_64` for now), and Linux wheels can be produced
with:

```
podman run --rm -v $(pwd):/io ghcr.io/pyo3/maturin build -m music-mesa-tables-py/Cargo.toml
```

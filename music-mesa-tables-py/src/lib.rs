mod eos;
use pyo3::prelude::*;
use pyo3::types::PyString;

/// Get the version number of the MESA tables backend library.
#[pyfunction]
fn get_mesa_tables_version(py: Python<'_>) -> &PyString {
    PyString::new(py, music_mesa_tables::VERSION)
}

/// This exposes interpolation routines of MESA tables.
#[pymodule]
fn music_mesa_tables(_py: Python<'_>, pymod: &PyModule) -> PyResult<()> {
    pymod.add_class::<eos::CstCompoState>()?;
    pymod.add_class::<eos::CstMetalState>()?;
    pymod.add_class::<eos::StateVar>()?;
    pymod.add_function(wrap_pyfunction!(get_mesa_tables_version, pymod)?)?;
    Ok(())
}

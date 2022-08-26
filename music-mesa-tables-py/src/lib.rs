mod eos;
mod eos_tables;
mod opacity;
use pyo3::prelude::*;

/// Get the version number of the MESA tables backend library.
#[pyfunction]
fn get_mesa_tables_version() -> &'static str {
    ::music_mesa_tables::VERSION
}

/// This exposes interpolation routines of MESA tables.
#[pymodule]
fn music_mesa_tables(_py: Python<'_>, pymod: &PyModule) -> PyResult<()> {
    pymod.add_class::<eos_tables::CstCompoEos>()?;
    pymod.add_class::<eos_tables::CstMetalEos>()?;
    pymod.add_class::<eos::CstCompoState>()?;
    pymod.add_class::<eos::CstMetalState>()?;
    pymod.add_class::<eos::StateVar>()?;
    pymod.add_class::<opacity::CstCompoOpacity>()?;
    pymod.add_class::<opacity::CstMetalOpacity>()?;
    pymod.add_function(wrap_pyfunction!(get_mesa_tables_version, pymod)?)?;
    Ok(())
}

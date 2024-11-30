mod eos;
mod eos_tables;
mod opacity;
use pyo3::pymodule;

/// This exposes interpolation routines of MESA tables.
#[pymodule]
mod music_mesa_tables {
    use pyo3::prelude::*;

    /// Get the version number of the MESA tables backend library.
    #[pyfunction]
    fn get_mesa_tables_version() -> &'static str {
        ::music_mesa_tables::VERSION
    }

    #[pymodule_export]
    use crate::{
        eos::{CstCompoState, CstMetalState, StateVar},
        eos_tables::{CstCompoEos, CstMetalEos},
        opacity::{CstCompoOpacity, CstMetalOpacity},
    };
}

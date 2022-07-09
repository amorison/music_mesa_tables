use music_mesa_tables::opacity;
use numpy::{IxDyn, PyArrayDyn};
use pyo3::prelude::*;

use crate::eos::{CstCompoState, CstMetalState};

/// Opacity of a state at constant metallicity and helium fraction.
#[pyclass]
pub struct CstCompoOpacity(opacity::CstCompoOpacity<IxDyn>);

#[pymethods]
impl CstCompoOpacity {
    #[new]
    pub fn new(state: &CstCompoState) -> Self {
        let state = opacity::CstCompoOpacity::new(state.inner_state());
        Self(state)
    }

    /// Compute the opacity for this state.
    pub fn log_opacity<'py>(&self, py: Python<'py>) -> &'py PyArrayDyn<f64> {
        let out = self.0.log_opacity();
        PyArrayDyn::from_owned_array(py, out)
    }
}

/// Opacity of a state at constant metallicity.
#[pyclass]
pub struct CstMetalOpacity(opacity::CstMetalOpacity<IxDyn>);

#[pymethods]
impl CstMetalOpacity {
    #[new]
    pub fn new(state: &CstMetalState) -> Self {
        let state = opacity::CstMetalOpacity::new(state.inner_state());
        Self(state)
    }

    /// Compute the opacity for this state.
    pub fn log_opacity<'py>(&self, py: Python<'py>) -> &'py PyArrayDyn<f64> {
        let out = self.0.log_opacity();
        PyArrayDyn::from_owned_array(py, out)
    }
}

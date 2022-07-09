use std::sync::{Arc, RwLock};

use music_mesa_tables::{eos_tables, state};
use numpy::{IxDyn, PyArrayDyn};
use pyo3::prelude::*;

/// Represent a state variable that can be computed from MESA tables.
#[pyclass]
#[derive(Copy, Clone)]
pub enum StateVar {
    LogDensity,
    LogPressure,
    LogPgas,
    LogTemperature,
    DPresDDensEcst,
    DPresDEnerDcst,
    DTempDDensEcst,
    DTempDEnerDcst,
    LogEntropy,
    DTempDPresScst,
    Gamma1,
    Gamma,
}

impl From<StateVar> for eos_tables::StateVar {
    fn from(var: StateVar) -> Self {
        match var {
            StateVar::LogDensity => eos_tables::StateVar::LogDensity,
            StateVar::LogPressure => eos_tables::StateVar::LogPressure,
            StateVar::LogPgas => eos_tables::StateVar::LogPgas,
            StateVar::LogTemperature => eos_tables::StateVar::LogTemperature,
            StateVar::DPresDDensEcst => eos_tables::StateVar::DPresDDensEcst,
            StateVar::DPresDEnerDcst => eos_tables::StateVar::DPresDEnerDcst,
            StateVar::DTempDDensEcst => eos_tables::StateVar::DTempDDensEcst,
            StateVar::DTempDEnerDcst => eos_tables::StateVar::DTempDEnerDcst,
            StateVar::LogEntropy => eos_tables::StateVar::LogEntropy,
            StateVar::DTempDPresScst => eos_tables::StateVar::DTempDPresScst,
            StateVar::Gamma1 => eos_tables::StateVar::Gamma1,
            StateVar::Gamma => eos_tables::StateVar::Gamma,
        }
    }
}

/// A state at constant metallicity and helium fraction.
#[pyclass]
pub struct CstCompoState(Arc<RwLock<state::CstCompoState<IxDyn>>>);

#[pymethods]
impl CstCompoState {
    #[new]
    pub fn new(
        metallicity: f64,
        he_frac: f64,
        density: &PyArrayDyn<f64>,
        energy: &PyArrayDyn<f64>,
    ) -> Self {
        let density = density.readonly();
        let energy = energy.readonly();
        let state =
            state::CstCompoState::new(metallicity, he_frac, density.as_array(), energy.as_array());
        Self(Arc::new(RwLock::new(state)))
    }

    /// Change the density and internal energy of the state, keeping the same composition.
    /// This is more efficient than recreating a new state with the same composition
    /// since this reuses the tables interpolated at the desired composition.
    pub fn set_state(&mut self, density: &PyArrayDyn<f64>, energy: &PyArrayDyn<f64>) {
        let density = density.readonly();
        let energy = energy.readonly();
        self.0
            .write()
            .unwrap()
            .set_state(density.as_array(), energy.as_array());
    }

    /// Compute the requested [`StateVar`] for this state.
    pub fn compute<'py>(&self, py: Python<'py>, var: StateVar) -> &'py PyArrayDyn<f64> {
        let out = self.0.read().unwrap().compute(var.into());
        PyArrayDyn::from_owned_array(py, out)
    }
}

impl CstCompoState {
    pub(crate) fn inner_state(&self) -> Arc<RwLock<state::CstCompoState<IxDyn>>> {
        self.0.clone()
    }
}

/// A state at constant metallicity.
#[pyclass]
pub struct CstMetalState(Arc<RwLock<state::CstMetalState<IxDyn>>>);

#[pymethods]
impl CstMetalState {
    #[new]
    pub fn new(
        metallicity: f64,
        he_frac: &PyArrayDyn<f64>,
        density: &PyArrayDyn<f64>,
        energy: &PyArrayDyn<f64>,
    ) -> Self {
        let density = density.readonly();
        let energy = energy.readonly();
        let he_frac = he_frac.readonly();
        let state = state::CstMetalState::new(
            metallicity,
            he_frac.as_array(),
            density.as_array(),
            energy.as_array(),
        );
        Self(Arc::new(RwLock::new(state)))
    }

    /// Change the helium fraction, density, and internal energy of the state,
    /// keeping the same metallicity. This is more efficient than recreating a
    /// new state with the same metallicity since this reuses the tables
    /// interpolated at the desired metallicity.
    pub fn set_state(
        &mut self,
        he_frac: &PyArrayDyn<f64>,
        density: &PyArrayDyn<f64>,
        energy: &PyArrayDyn<f64>,
    ) {
        let he_frac = he_frac.readonly();
        let density = density.readonly();
        let energy = energy.readonly();
        self.0.write().unwrap().set_state(
            he_frac.as_array(),
            density.as_array(),
            energy.as_array(),
        );
    }

    /// Compute the requested [`StateVar`] for this state.
    pub fn compute<'py>(&self, py: Python<'py>, var: StateVar) -> &'py PyArrayDyn<f64> {
        let out = self.0.read().unwrap().compute(var.into());
        PyArrayDyn::from_owned_array(py, out)
    }
}

impl CstMetalState {
    pub(crate) fn inner_state(&self) -> Arc<RwLock<state::CstMetalState<IxDyn>>> {
        self.0.clone()
    }
}

use music_mesa_tables::eos_tables;
use pyo3::prelude::*;
use std::sync::Arc;

/// EOS tables at constant metallicity and helium fraction.
#[pyclass(frozen)]
pub struct CstCompoEos(Arc<eos_tables::VolumeEnergyTable>);

#[pymethods]
impl CstCompoEos {
    #[new]
    fn new(metallicity: f64, he_frac: f64) -> Self {
        let inner = eos_tables::AllTables::default()
            .take_at_metallicity(metallicity)
            .expect("metallicity is out of range")
            .take_at_he_frac(he_frac)
            .expect("helium fraction is out of range");
        Self(inner.into())
    }
}

impl CstCompoEos {
    pub(crate) fn inner_table(&self) -> Arc<eos_tables::VolumeEnergyTable> {
        self.0.clone()
    }
}

/// EOS tables at constant metallicity.
#[pyclass(frozen)]
pub struct CstMetalEos(Arc<eos_tables::ConstMetalTables>);

#[pymethods]
impl CstMetalEos {
    #[new]
    fn new(metallicity: f64) -> Self {
        let inner = eos_tables::AllTables::default()
            .take_at_metallicity(metallicity)
            .expect("metallicity is out of range");
        Self(inner.into())
    }
}

impl CstMetalEos {
    pub(crate) fn inner_table(&self) -> Arc<eos_tables::ConstMetalTables> {
        self.0.clone()
    }
}

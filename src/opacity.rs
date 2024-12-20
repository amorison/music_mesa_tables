use std::sync::Arc;

use ndarray::{Array, Dimension, Zip};

use crate::{
    eos_tables::StateVar,
    is_close::IsClose,
    opacity_tables::{AllTables, ConstMetalTables, RTempTable},
    state::{CstCompoState, CstMetalState},
};

pub struct CstCompoOpacity<D: Dimension> {
    state: Arc<CstCompoState<D>>,
    table: RTempTable,
}

impl<D: Dimension> CstCompoOpacity<D> {
    pub fn new(state: Arc<CstCompoState<D>>) -> Self {
        let table = AllTables::default()
            .take_at_metallicity(state.metallicity())
            .expect("metallicity is in range")
            .take_at_h_frac(state.h_frac())
            .expect("He fraction is in range");
        Self { state, table }
    }

    pub fn with_table(table: RTempTable, state: Arc<CstCompoState<D>>) -> Self {
        assert!(table.metallicity().is_close(state.metallicity()));
        assert!(table.h_frac().is_close(state.h_frac()));
        Self { state, table }
    }

    pub fn log_opacity(&self) -> Array<f64, D> {
        let logt = self.state.compute(StateVar::LogTemperature);
        Zip::from(&logt)
            .and(self.state.log_density())
            .map_collect(|&logt, &logd| {
                let logr = logd + 18.0 - 3.0 * logt;
                self.table.at(logt, logr).expect("out of table")
            })
    }
}

pub struct CstMetalOpacity<D: Dimension> {
    state: Arc<CstMetalState<D>>,
    table: ConstMetalTables,
}

impl<D: Dimension> CstMetalOpacity<D> {
    pub fn new(state: Arc<CstMetalState<D>>) -> Self {
        let table = AllTables::default()
            .take_at_metallicity(state.metallicity())
            .expect("metallicity is in range");
        Self { state, table }
    }

    pub fn with_table(table: ConstMetalTables, state: Arc<CstMetalState<D>>) -> Self {
        assert!(table.metallicity().is_close(state.metallicity()));
        Self { state, table }
    }

    pub fn log_opacity(&self) -> Array<f64, D> {
        let logt = self.state.compute(StateVar::LogTemperature);
        Zip::from(&logt)
            .and(self.state.log_density())
            .and(self.state.h_frac())
            .map_collect(|&logt, &logd, &h_frac| {
                let logr = logd + 18.0 - 3.0 * logt;
                self.table.at(h_frac, logt, logr).expect("out of table")
            })
    }
}

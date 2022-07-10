use std::sync::Arc;

use ndarray::{Array, ArrayView, Dimension, Zip};

use crate::eos_tables::{ConstMetalTables, StateVar, VolumeEnergyTable};

pub struct CstCompoState<D: Dimension> {
    log_density: Array<f64, D>,
    log_volume: Array<f64, D>,
    log_energy: Array<f64, D>,
    table: Arc<VolumeEnergyTable>,
}

fn from_de_to_logdve<D: Dimension>(
    density: ArrayView<'_, f64, D>,
    energy: ArrayView<'_, f64, D>,
) -> (Array<f64, D>, Array<f64, D>, Array<f64, D>) {
    let log_density = density.mapv(f64::log10);
    let log_energy = energy.mapv(f64::log10);
    let log_volume = Zip::from(&log_energy)
        .and(&log_density)
        .map_collect(|&loge, &logd| 20.0 + logd - 0.7 * loge);
    (log_density, log_volume, log_energy)
}

impl<D: Dimension> CstCompoState<D> {
    pub fn new(
        table: Arc<VolumeEnergyTable>,
        density: ArrayView<'_, f64, D>,
        energy: ArrayView<'_, f64, D>,
    ) -> Self {
        assert_eq!(density.shape(), energy.shape());
        let (log_density, log_volume, log_energy) = from_de_to_logdve(density, energy);
        Self {
            log_density,
            log_volume,
            log_energy,
            table,
        }
    }

    pub fn compute(&self, var: StateVar) -> Array<f64, D> {
        Zip::from(&self.log_volume)
            .and(&self.log_energy)
            .map_collect(|&logv, &loge| self.table.at(loge, logv, var).expect("out of table"))
    }

    pub fn metallicity(&self) -> f64 {
        self.table.metallicity()
    }

    pub fn he_frac(&self) -> f64 {
        1.0 - self.h_frac() - self.metallicity()
    }

    pub fn h_frac(&self) -> f64 {
        self.table.h_frac()
    }

    pub fn log_density(&self) -> ArrayView<'_, f64, D> {
        self.log_density.view()
    }
}

pub struct CstMetalState<D: Dimension> {
    h_frac: Array<f64, D>,
    log_density: Array<f64, D>,
    log_volume: Array<f64, D>,
    log_energy: Array<f64, D>,
    table: Arc<ConstMetalTables>,
}

impl<D: Dimension> CstMetalState<D> {
    pub fn new(
        table: Arc<ConstMetalTables>,
        he_frac: ArrayView<'_, f64, D>,
        density: ArrayView<'_, f64, D>,
        energy: ArrayView<'_, f64, D>,
    ) -> Self {
        assert_eq!(he_frac.shape(), density.shape());
        assert_eq!(he_frac.shape(), energy.shape());
        let h_frac = he_frac.mapv(|he| 1.0 - he - table.metallicity());
        let (log_density, log_volume, log_energy) = from_de_to_logdve(density, energy);
        Self {
            h_frac,
            log_density,
            log_volume,
            log_energy,
            table,
        }
    }

    pub fn set_state(
        &mut self,
        he_frac: ArrayView<'_, f64, D>,
        density: ArrayView<'_, f64, D>,
        energy: ArrayView<'_, f64, D>,
    ) {
        assert_eq!(he_frac.shape(), density.shape());
        assert_eq!(he_frac.shape(), energy.shape());
        let (log_density, log_volume, log_energy) = from_de_to_logdve(density, energy);
        self.h_frac = he_frac.mapv(|he| 1.0 - he - self.metallicity());
        self.log_density = log_density;
        self.log_volume = log_volume;
        self.log_energy = log_energy;
    }

    pub fn compute(&self, var: StateVar) -> Array<f64, D> {
        Zip::from(&self.log_volume)
            .and(&self.log_energy)
            .and(&self.h_frac)
            .map_collect(|&logv, &loge, &h_frac| {
                self.table
                    .at(h_frac, loge, logv, var)
                    .expect("out of table")
            })
    }

    pub fn metallicity(&self) -> f64 {
        self.table.metallicity()
    }

    pub fn h_frac(&self) -> ArrayView<'_, f64, D> {
        self.h_frac.view()
    }

    pub fn log_density(&self) -> ArrayView<'_, f64, D> {
        self.log_density.view()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, Zip};

    use crate::{
        eos_tables::{AllTables, StateVar},
        is_close::IsClose,
        state::CstMetalState,
    };

    use super::CstCompoState;

    #[test]
    fn constant_compo() {
        let table = AllTables::default()
            .take_at_metallicity(0.02)
            .unwrap()
            .take_at_he_frac(0.42)
            .unwrap();
        let density = arr1(&[3.5, 10.3, 10.5]);
        let energy = arr1(&[5.7e14, 4.5e15, 6.7e16]);
        let state = CstCompoState::new(table.into(), density.view(), energy.view());
        let d_comp = state
            .compute(StateVar::LogDensity)
            .mapv_into(|logd| 10.0_f64.powf(logd));
        assert!(Zip::from(&density)
            .and(&d_comp)
            .all(|&a, &b| (a - b).abs() / a < 5e-2));
    }

    #[test]
    fn constant_metal() {
        let table = AllTables::default().take_at_metallicity(0.02).unwrap();
        let he_frac = arr1(&[0.2, 0.3, 0.4]);
        let mut density = arr1(&[3.5, 10.3, 10.5]);
        let mut energy = arr1(&[5.7e14, 4.5e15, 6.7e16]);
        let mut state =
            CstMetalState::new(table.into(), he_frac.view(), density.view(), energy.view());
        let d_comp = state
            .compute(StateVar::LogDensity)
            .mapv_into(|logd| 10.0_f64.powf(logd));
        assert!(Zip::from(&density)
            .and(&d_comp)
            .all(|&a, &b| (a / b).is_close(1.0)));

        density.fill(10.3);
        energy.fill(4.5e15);
        state.set_state(he_frac.view(), density.view(), energy.view());
        let logt = state.compute(StateVar::LogTemperature);
        assert!(((logt[0] + logt[2]) / 2.0 - logt[1]) / logt[1] < 1e-4);
    }
}

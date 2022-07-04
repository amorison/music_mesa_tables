use ndarray::{Array, ArrayView, Dimension, Zip};

use crate::eos_tables::{ConstMetalTables, StateVar, VolumeEnergyTable};

pub struct CstCompoState<D: Dimension> {
    log_volume: Array<f64, D>,
    log_energy: Array<f64, D>,
    table: VolumeEnergyTable,
}

fn from_de_to_logve<D: Dimension>(
    density: ArrayView<'_, f64, D>,
    energy: ArrayView<'_, f64, D>,
) -> (Array<f64, D>, Array<f64, D>) {
    let log_energy = energy.mapv(f64::log10);
    let log_volume = Zip::from(&log_energy)
        .and(density)
        .map_collect(|&loge, d| 20.0 + d.log10() - 0.7 * loge);
    (log_volume, log_energy)
}

impl<D: Dimension> CstCompoState<D> {
    pub fn new(
        density: ArrayView<'_, f64, D>,
        energy: ArrayView<'_, f64, D>,
        table: VolumeEnergyTable,
    ) -> Self {
        assert_eq!(density.shape(), energy.shape());
        let (log_volume, log_energy) = from_de_to_logve(density, energy);
        Self {
            log_volume,
            log_energy,
            table,
        }
    }

    pub fn set_state(&mut self, density: ArrayView<'_, f64, D>, energy: ArrayView<'_, f64, D>) {
        assert_eq!(density.shape(), energy.shape());
        let (log_volume, log_energy) = from_de_to_logve(density, energy);
        self.log_volume = log_volume;
        self.log_energy = log_energy;
    }

    pub fn compute(&self, var: StateVar) -> Array<f64, D> {
        Zip::from(&self.log_volume)
            .and(&self.log_energy)
            .map_collect(|&logv, &loge| self.table.at(loge, logv, var).expect("out of table"))
    }
}

pub struct CstMetalState<D: Dimension> {
    h_frac: Array<f64, D>,
    metallicity: f64,
    log_volume: Array<f64, D>,
    log_energy: Array<f64, D>,
    table: ConstMetalTables,
}

impl<D: Dimension> CstMetalState<D> {
    pub fn new(
        metallicity: f64,
        he_frac: ArrayView<'_, f64, D>,
        density: ArrayView<'_, f64, D>,
        energy: ArrayView<'_, f64, D>,
        table: ConstMetalTables,
    ) -> Self {
        assert_eq!(he_frac.shape(), density.shape());
        assert_eq!(he_frac.shape(), energy.shape());
        let h_frac = he_frac.mapv(|he| 1.0 - he - metallicity);
        let (log_volume, log_energy) = from_de_to_logve(density, energy);
        Self {
            h_frac,
            metallicity,
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
        let (log_volume, log_energy) = from_de_to_logve(density, energy);
        self.h_frac = he_frac.mapv(|he| 1.0 - he - self.metallicity);
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
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, Zip};

    use crate::{
        eos_tables::{AllTables, StateVar},
        state::CstMetalState,
    };

    use super::CstCompoState;

    #[test]
    fn constant_compo() {
        let eos_table = AllTables::default()
            .take_at_metallicity(0.02)
            .unwrap()
            .take_at_h_frac(0.56)
            .unwrap();
        let density = arr1(&[3.5, 10.3, 10.5]);
        let energy = arr1(&[5.7e14, 4.5e15, 6.7e16]);
        let state = CstCompoState::new(density.view(), energy.view(), eos_table);
        let d_comp = state
            .compute(StateVar::Density)
            .mapv_into(|logd| 10.0_f64.powf(logd));
        assert!(Zip::from(&density)
            .and(&d_comp)
            .all(|&a, &b| (a - b).abs() / a < 5e-2));
    }

    #[test]
    fn constant_metal() {
        let metal = 0.02;
        let eos_table = AllTables::default().take_at_metallicity(metal).unwrap();
        let he_frac = arr1(&[0.2, 0.3, 0.4]);
        let mut density = arr1(&[3.5, 10.3, 10.5]);
        let mut energy = arr1(&[5.7e14, 4.5e15, 6.7e16]);
        let mut state = CstMetalState::new(
            metal,
            he_frac.view(),
            density.view(),
            energy.view(),
            eos_table,
        );
        let d_comp = state
            .compute(StateVar::Density)
            .mapv_into(|logd| 10.0_f64.powf(logd));
        assert!(Zip::from(&density)
            .and(&d_comp)
            .all(|&a, &b| (a - b).abs() / a < 5e-2));

        density.fill(10.3);
        energy.fill(4.5e15);
        state.set_state(he_frac.view(), density.view(), energy.view());
        let logt = state.compute(StateVar::Temperature);
        assert!(((logt[0] + logt[2]) / 2.0 - logt[1]) / logt[1] < 1e-4);
    }
}

use ndarray::{Array3, ArrayView3, Axis};

use crate::{
    index::{IdxLin, Indexable, LinearInterpolable, OutOfBoundsError, Range},
    interp::{cubic_spline_2d, LinearInterpolator},
    is_close::IsClose,
    raw_tables::eos::{AllRawTables, MetalRawTables, RawTableContent, RAW_TABLES},
};

/// State variable labels.
#[derive(Copy, Clone)]
#[repr(usize)]
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

/// The collection of all MESA tables available
pub struct AllTables {
    metallicities: Range,
    tables: Vec<ConstMetalTables>,
}

impl AllTables {
    pub fn take_at_metallicity(
        mut self,
        metallicity: f64,
    ) -> Result<ConstMetalTables, OutOfBoundsError> {
        match self.metallicities.idx_lin(metallicity)? {
            IdxLin::Exact(i) => Ok(self.tables.swap_remove(i)),
            IdxLin::Between(i, j) => {
                let r_tables = self.tables.swap_remove(j);
                let l_tables = self.tables.swap_remove(i);
                let h_fracs = l_tables
                    .h_fracs
                    .subrange_in(r_tables.h_fracs)
                    .expect("Hydrogen fractions should overlap");
                let tables: Vec<_> = h_fracs
                    .into_iter()
                    .map(move |h_frac| {
                        // this is not in-place!
                        let left = l_tables.at_h_frac(h_frac)?;
                        let right = r_tables.at_h_frac(h_frac)?;
                        Ok(left.interp_at_metal(&right, metallicity))
                    })
                    .collect::<Result<_, _>>()?;
                Ok(ConstMetalTables {
                    metallicity,
                    h_fracs,
                    tables,
                })
            }
        }
    }
}

impl From<&AllRawTables> for AllTables {
    fn from(rawtbls: &AllRawTables) -> Self {
        let metallicities = rawtbls.metallicities;
        Self {
            metallicities,
            tables: rawtbls
                .tables
                .iter()
                .zip(metallicities)
                .map(|(t, m)| ConstMetalTables::from_raw(m, t))
                .collect(),
        }
    }
}

impl Default for AllTables {
    fn default() -> Self {
        (&RAW_TABLES).into()
    }
}

/// The collection of MESA tables at a given metallicity
pub struct ConstMetalTables {
    metallicity: f64,
    h_fracs: Range,
    tables: Vec<VolumeEnergyTable>,
}

impl ConstMetalTables {
    fn from_raw(metallicity: f64, raw: &MetalRawTables) -> Self {
        let h_fracs = raw.h_fracs;
        Self {
            metallicity,
            h_fracs,
            tables: raw
                .tables
                .iter()
                .zip(h_fracs)
                .map(|(t, h)| VolumeEnergyTable::from_raw(metallicity, h, t.into()))
                .collect(),
        }
    }

    pub fn take_at_h_frac(mut self, h_frac: f64) -> Result<VolumeEnergyTable, OutOfBoundsError> {
        match self.h_fracs.idx_lin(h_frac)? {
            IdxLin::Exact(i) => Ok(self.tables.swap_remove(i)),
            IdxLin::Between(i, j) => {
                let right = self.tables.swap_remove(j);
                let left = self.tables.swap_remove(i);
                // not in-place!
                Ok(left.interp_at_h_frac(&right, h_frac))
            }
        }
    }

    pub fn at_h_frac(&self, h_frac: f64) -> Result<VolumeEnergyTable, OutOfBoundsError> {
        match self.h_fracs.idx_lin(h_frac)? {
            IdxLin::Exact(i) => Ok(self.tables[i].clone()),
            IdxLin::Between(i, j) => {
                let left = &self.tables[i];
                let right = &self.tables[j];
                Ok(left.interp_at_h_frac(right, h_frac))
            }
        }
    }

    pub fn metallicity(&self) -> f64 {
        self.metallicity
    }

    pub fn at(
        &self,
        h_frac: f64,
        log_energy: f64,
        log_volume: f64,
        var: StateVar,
    ) -> Result<f64, OutOfBoundsError> {
        match self.h_fracs.idx_lin(h_frac)? {
            IdxLin::Exact(i) => self.tables[i].at(log_energy, log_volume, var),
            IdxLin::Between(i, j) => {
                let lin = LinearInterpolator::new(self.h_fracs.at(i), self.h_fracs.at(j), h_frac);
                let loge_stencil = self.tables[i].log_energy().spline_stencil(log_energy)?;
                let logv_stencil = self.tables[i].log_volume().spline_stencil(log_volume)?;

                let mut ltable = self.tables[i].values();
                let mut rtable = self.tables[j].values();

                // take only the elements of the tables that are needed for the
                // spline interpolation.
                loge_stencil.slice_view(Axis(0), &mut ltable);
                let loge_stencil = loge_stencil.slice_view(Axis(0), &mut rtable);
                logv_stencil.slice_view(Axis(1), &mut ltable);
                let logv_stencil = logv_stencil.slice_view(Axis(1), &mut rtable);
                let table = lin.interp(
                    ltable.index_axis(Axis(2), var as usize),
                    rtable.index_axis(Axis(2), var as usize),
                );
                Ok(cubic_spline_2d(loge_stencil, logv_stencil, table.view()))
            }
        }
    }
}

#[derive(Clone)]
/// Represent a MESA table in volume/energy space at a given composition.
pub struct VolumeEnergyTable {
    /// Metallicity
    metallicity: f64,
    /// Hydrogen fraction
    h_frac: f64,
    /// Volume index (in log)
    log_volume: Range,
    /// Energy index (in log)
    log_energy: Range,
    /// Table indexed by energy, volume, and state variables
    values: Array3<f64>,
}

impl VolumeEnergyTable {
    fn from_raw(metallicity: f64, h_frac: f64, raw: RawTableContent) -> Self {
        let RawTableContent {
            log_volume,
            log_energy,
            values,
        } = raw;
        Self {
            metallicity,
            h_frac,
            log_volume,
            log_energy,
            values,
        }
    }

    pub fn metallicity(&self) -> f64 {
        self.metallicity
    }

    pub fn h_frac(&self) -> f64 {
        self.h_frac
    }

    pub fn log_volume(&self) -> Range {
        self.log_volume
    }

    pub fn log_energy(&self) -> Range {
        self.log_energy
    }

    pub fn values(&self) -> ArrayView3<f64> {
        self.values.view()
    }

    pub(crate) fn interp_at_metal(&self, other: &Self, metallicity: f64) -> Self {
        assert!(self.log_volume.is_close(other.log_volume));
        assert!(self.log_energy.is_close(other.log_energy));
        assert!(self.h_frac.is_close(other.h_frac));
        let lin = LinearInterpolator::new(self.metallicity, other.metallicity, metallicity);
        Self {
            metallicity,
            h_frac: self.h_frac,
            values: lin.interp(self.values.view(), other.values.view()),
            log_volume: self.log_volume,
            log_energy: self.log_energy,
        }
    }

    pub(crate) fn interp_at_h_frac(&self, other: &Self, h_frac: f64) -> Self {
        assert!(self.log_volume.is_close(other.log_volume));
        assert!(self.log_energy.is_close(other.log_energy));
        assert!(self.metallicity.is_close(other.metallicity));
        let lin = LinearInterpolator::new(self.h_frac, other.h_frac, h_frac);
        Self {
            metallicity: self.metallicity,
            h_frac,
            values: lin.interp(self.values.view(), other.values.view()),
            log_volume: self.log_volume,
            log_energy: self.log_energy,
        }
    }

    pub fn at(
        &self,
        log_energy: f64,
        log_volume: f64,
        var: StateVar,
    ) -> Result<f64, OutOfBoundsError> {
        Ok(cubic_spline_2d(
            self.log_energy.spline_stencil(log_energy)?,
            self.log_volume.spline_stencil(log_volume)?,
            self.values().index_axis(Axis(2), var as usize),
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::is_close::IsClose;

    use super::{AllTables, StateVar};

    #[test]
    fn read_eos_table() {
        let ve_eos = AllTables::default()
            .take_at_metallicity(0.02)
            .expect("metallicity is in range")
            .take_at_h_frac(0.8)
            .expect("helium fraction is in range");
        assert!(ve_eos.log_volume().first().is_close(0.0));
        assert!(ve_eos.log_volume().last().is_close(14.0));
        assert!(ve_eos.log_energy().first().is_close(10.5));
        assert!(ve_eos.log_energy().last().is_close(17.5));
    }

    #[test]
    fn check_density() {
        let ve_eos = AllTables::default()
            .take_at_metallicity(0.02)
            .expect("metallicity is in range")
            .take_at_h_frac(0.8)
            .expect("hydrogen fraction is in range");
        // these values are voluntarily intricate to not be at a grid point.
        let log_energy = 2.24e15_f64.log10();
        let log_volume = 1.32e8_f64.log10();
        let log_density = ve_eos
            .at(log_energy, log_volume, StateVar::LogDensity)
            .expect("point is on the grid");
        let fit_density = log_volume + 0.7 * log_energy - 20.0;
        assert!(log_density.is_close(fit_density));
    }

    #[test]
    fn interp_compo_consistency() {
        let z_eos = AllTables::default()
            .take_at_metallicity(0.02)
            .expect("metallicity is in range");
        let h_frac = 1.0 - 0.35776 - 0.02;
        let log_energy = 3.6349e+15_f64.log10();
        let log_density = 8.3537_f64.log10();
        let log_vol = 20.0 + log_density - 0.7 * log_energy;

        let logt_direct = z_eos
            .at(h_frac, log_energy, log_vol, StateVar::LogTemperature)
            .expect("requested state in range");

        let ve_eos = z_eos
            .take_at_h_frac(h_frac)
            .expect("hydrogen fraction is in range");
        let logt_full_interp = ve_eos
            .at(log_energy, log_vol, StateVar::LogTemperature)
            .expect("requested state in range");

        assert!(logt_direct.is_close(logt_full_interp))
    }
}

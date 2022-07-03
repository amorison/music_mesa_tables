use std::io::{self, Read};

use ndarray::{s, Array3, ArrayView3, Axis};

use crate::{
    fort_unfmt::read_fort_record,
    index::{IdxLin, Range},
    interp::{cubic_spline_2d, LinearInterpolator},
    is_close::IsClose,
    raw_tables::eos::{AllRawTables, MetalRawTables, RawTable, RAW_TABLES},
};

/// State variable labels, all quantities except Gamma1 and Gamma are logarithmic.
#[derive(Copy, Clone)]
#[repr(usize)]
pub enum StateVar {
    Density,
    Pressure,
    Pgas,
    Temperature,
    DPresDDensEcst,
    DPresDEnerDcst,
    DTempDDensEcst,
    DTempDEnerDcst,
    Entropy,
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
    pub fn take_at_metallicity(mut self, metallicity: f64) -> Result<ConstMetalTables, String> {
        match self.metallicities.idx_lin(metallicity) {
            IdxLin::Exact(i) => Ok(self.tables.swap_remove(i)),
            IdxLin::Between(i, j) => {
                let r_tables = self.tables.swap_remove(j);
                let l_tables = self.tables.swap_remove(i);
                let lin = LinearInterpolator::new(
                    self.metallicities.at(i),
                    self.metallicities.at(j),
                    metallicity,
                );
                let h_fracs = l_tables
                    .h_fracs
                    .subrange_in(r_tables.h_fracs)
                    .ok_or("Hydrogen fractions do not overlap")?;
                let tables: Vec<_> = h_fracs
                    .into_iter()
                    .map(move |h_frac| {
                        // this is not in-place!
                        let left = l_tables.at_h_frac(h_frac)?;
                        let right = r_tables.at_h_frac(h_frac)?;
                        left.interp_with(&right, &lin)
                    })
                    .collect::<Result<_, _>>()?;
                Ok(ConstMetalTables { h_fracs, tables })
            }
            IdxLin::OutOfRange => Err("metallicity out of range".to_owned()),
        }
    }
}

impl From<&AllRawTables> for AllTables {
    fn from(rawtbls: &AllRawTables) -> Self {
        Self {
            metallicities: rawtbls.metallicities,
            tables: rawtbls.tables.iter().map(|t| t.into()).collect(),
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
    h_fracs: Range,
    tables: Vec<VolumeEnergyTable>,
}

impl From<&MetalRawTables> for ConstMetalTables {
    fn from(rawtbls: &MetalRawTables) -> Self {
        Self {
            h_fracs: rawtbls.h_fracs,
            tables: rawtbls.tables.iter().map(|t| t.into()).collect(),
        }
    }
}

impl ConstMetalTables {
    pub fn take_at_h_frac(mut self, h_frac: f64) -> Result<VolumeEnergyTable, String> {
        match self.h_fracs.idx_lin(h_frac) {
            IdxLin::Exact(i) => Ok(self.tables.swap_remove(i)),
            IdxLin::Between(i, j) => {
                let right = self.tables.swap_remove(j);
                let left = self.tables.swap_remove(i);
                let lin = LinearInterpolator::new(self.h_fracs.at(i), self.h_fracs.at(j), h_frac);
                // not in-place! lin should have in-place impl too
                left.interp_with(&right, &lin)
            }
            IdxLin::OutOfRange => Err("Hydrogen fraction out of range".to_owned()),
        }
    }

    pub fn at_h_frac(&self, h_frac: f64) -> Result<VolumeEnergyTable, String> {
        match self.h_fracs.idx_lin(h_frac) {
            IdxLin::Exact(i) => Ok(self.tables[i].clone()),
            IdxLin::Between(i, j) => {
                let left = &self.tables[i];
                let right = &self.tables[j];
                let lin = LinearInterpolator::new(self.h_fracs.at(i), self.h_fracs.at(j), h_frac);
                left.interp_with(&right, &lin)
            }
            IdxLin::OutOfRange => Err("Hydrogen fraction out of range".to_owned()),
        }
    }

    pub fn at(
        &self,
        h_frac: f64,
        log_energy: f64,
        log_volume: f64,
        var: StateVar,
    ) -> Result<f64, &'static str> {
        match self.h_fracs.idx_lin(h_frac) {
            IdxLin::Exact(i) => self.tables[i].at(log_energy, log_volume, var),
            IdxLin::Between(i, j) => {
                let lin = LinearInterpolator::new(self.h_fracs.at(i), self.h_fracs.at(j), h_frac);
                let loges = self.tables[i].log_energy();
                let logvs = self.tables[i].log_volume();
                let table = lin.interp(
                    self.tables[i].values().index_axis(Axis(2), var as usize),
                    self.tables[j].values().index_axis(Axis(2), var as usize),
                );
                cubic_spline_2d(
                    loges.spline_stencil(log_energy),
                    logvs.spline_stencil(log_volume),
                    table.view(),
                )
            }
            IdxLin::OutOfRange => Err("Hydrogen fraction out of range"),
        }
    }
}

impl From<&RawTable> for VolumeEnergyTable {
    fn from(rawtbl: &RawTable) -> Self {
        Self::read_from(rawtbl.0).expect("raw tables are well-formed")
    }
}

#[derive(Clone)]
/// Represent a MESA table in volume/energy space at a given composition.
pub struct VolumeEnergyTable {
    /// Volume index (in log)
    log_volume: Range,
    /// Energy index (in log)
    log_energy: Range,
    /// Table indexed by energy, volume, and state variables
    values: Array3<f64>,
}

impl VolumeEnergyTable {
    fn read_from<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut shape = [0_u32; 3]; // ne, nv, nvars
        read_fort_record(&mut reader, &mut shape)?;
        let shape = shape.map(|e| e as usize);

        let mut log_volume = vec![0.0; shape[1]];
        read_fort_record(&mut reader, &mut log_volume)?;
        let log_volume = Range::from_slice(&log_volume)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut log_energy = vec![0.0; shape[0]];
        read_fort_record(&mut reader, &mut log_energy)?;
        let log_energy = Range::from_slice(&log_energy)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut values = Array3::zeros(shape);
        for i_v in 0..shape[1] {
            for i_e in 0..shape[0] {
                let mut slc = values.slice_mut(s![i_e, i_v, ..]);
                let raw_slc = slc.as_slice_mut().expect("values should be contiguous");
                read_fort_record(&mut reader, raw_slc)?;
            }
        }

        Ok(Self {
            log_volume,
            log_energy,
            values,
        })
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

    pub(crate) fn interp_with(
        &self,
        other: &Self,
        lin: &LinearInterpolator,
    ) -> Result<Self, String> {
        if !self.log_volume.is_close(other.log_volume) {
            Err("log V index don't match".to_owned())
        } else if !self.log_energy.is_close(other.log_energy) {
            Err("log E index don't match".to_owned())
        } else {
            Ok(Self {
                values: lin.interp(self.values.view(), other.values.view()),
                log_volume: self.log_volume,
                log_energy: self.log_energy,
            })
        }
    }

    pub fn at(&self, log_energy: f64, log_volume: f64, var: StateVar) -> Result<f64, &'static str> {
        cubic_spline_2d(
            self.log_energy.spline_stencil(log_energy),
            self.log_volume.spline_stencil(log_volume),
            self.values().index_axis(Axis(2), var as usize),
        )
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
            .at(log_energy, log_volume, StateVar::Density)
            .expect("point is on the grid");
        let fit_density = log_volume + 0.7 * log_energy - 20.0;
        assert!((log_density - fit_density) / fit_density < 1e-2);
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
            .at(h_frac, log_energy, log_vol, StateVar::Temperature)
            .expect("requested state in range");

        let ve_eos = z_eos
            .take_at_h_frac(h_frac)
            .expect("hydrogen fraction is in range");
        let logt_full_interp = ve_eos
            .at(log_energy, log_vol, StateVar::Temperature)
            .expect("requested state in range");

        assert!(logt_direct.is_close(logt_full_interp))
    }
}

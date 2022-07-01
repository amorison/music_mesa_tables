use std::io::{self, Read};

use ndarray::{s, Array3, ArrayView3};

use crate::{
    fort_unfmt::read_fort_record,
    index::{Idx, Range},
    interp::{cubic_spline, LinearInterpolator},
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
        match self.metallicities.find_value(metallicity) {
            Idx::Exact(i) => Ok(self.tables.swap_remove(i)),
            Idx::Between(i, j) => {
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
            Idx::OutOfRange => Err("metallicity out of range".to_owned()),
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
        match self.h_fracs.find_value(h_frac) {
            Idx::Exact(i) => Ok(self.tables.swap_remove(i)),
            Idx::Between(i, j) => {
                let right = self.tables.swap_remove(j);
                let left = self.tables.swap_remove(i);
                let lin = LinearInterpolator::new(self.h_fracs.at(i), self.h_fracs.at(j), h_frac);
                // not in-place! lin should have in-place impl too
                left.interp_with(&right, &lin)
            }
            Idx::OutOfRange => Err("Hydrogen fraction out of range".to_owned()),
        }
    }

    pub fn at_h_frac(&self, h_frac: f64) -> Result<VolumeEnergyTable, String> {
        match self.h_fracs.find_value(h_frac) {
            Idx::Exact(i) => Ok(self.tables[i].clone()),
            Idx::Between(i, j) => {
                let left = &self.tables[i];
                let right = &self.tables[j];
                let lin = LinearInterpolator::new(self.h_fracs.at(i), self.h_fracs.at(j), h_frac);
                left.interp_with(&right, &lin)
            }
            Idx::OutOfRange => Err("Hydrogen fraction out of range".to_owned()),
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

    pub fn at(&self, energy: f64, volume: f64, var: StateVar) -> Result<f64, &'static str> {
        let ivar = var as usize;
        match (
            self.log_energy.find_value(energy.log10()),
            self.log_volume.find_value(volume.log10()),
        ) {
            (Idx::OutOfRange, _) | (_, Idx::OutOfRange) => Err("energy or volume out of range"),
            (Idx::Exact(0), _) | (Idx::Between(0, _), _) => Err("energy in lower values"),
            (_, Idx::Exact(0)) | (_, Idx::Between(0, _)) => Err("temperature in lower values"),
            (Idx::Exact(n), _) | (Idx::Between(_, n), _) if n == self.log_energy.n_values() - 1 => {
                Err("energy in higher values")
            }
            (_, Idx::Exact(n)) | (_, Idx::Between(_, n)) if n == self.log_volume.n_values() - 1 => {
                Err("temperature in higher values")
            }
            (Idx::Exact(i_e), Idx::Exact(i_v)) => Ok(self.values[[i_e, i_v, var as usize]]),
            (Idx::Exact(i_e), Idx::Between(i_v, _)) => Ok(cubic_spline(
                [
                    self.log_volume.at(i_v - 1),
                    self.log_volume.at(i_v),
                    self.log_volume.at(i_v + 1),
                    self.log_volume.at(i_v + 2),
                ],
                [
                    self.values[[i_e, i_v - 1, ivar]],
                    self.values[[i_e, i_v, ivar]],
                    self.values[[i_e, i_v + 1, ivar]],
                    self.values[[i_e, i_v + 2, ivar]],
                ],
                volume.log10(),
            )),
            (Idx::Between(i_e, _), Idx::Exact(i_v)) => Ok(cubic_spline(
                [
                    self.log_energy.at(i_e - 1),
                    self.log_energy.at(i_e),
                    self.log_energy.at(i_e + 1),
                    self.log_energy.at(i_e + 2),
                ],
                [
                    self.values[[i_e - 1, i_v, ivar]],
                    self.values[[i_e, i_v, ivar]],
                    self.values[[i_e + 1, i_v, ivar]],
                    self.values[[i_e + 2, i_v, ivar]],
                ],
                energy.log10(),
            )),
            (Idx::Between(i_e, _), Idx::Between(i_v, _)) => {
                let loge = [
                    self.log_energy.at(i_e - 1),
                    self.log_energy.at(i_e),
                    self.log_energy.at(i_e + 1),
                    self.log_energy.at(i_e + 2),
                ];
                let mut vals = [0.0; 4];
                for (i, iv) in (i_v - 1..=i_v + 2).enumerate() {
                    vals[i] = cubic_spline(
                        loge,
                        [
                            self.values[[i_e - 1, iv, ivar]],
                            self.values[[i_e, iv, ivar]],
                            self.values[[i_e + 1, iv, ivar]],
                            self.values[[i_e + 2, iv, ivar]],
                        ],
                        energy.log10(),
                    );
                }
                Ok(cubic_spline(
                    [
                        self.log_volume.at(i_v - 1),
                        self.log_volume.at(i_v),
                        self.log_volume.at(i_v + 1),
                        self.log_volume.at(i_v + 2),
                    ],
                    vals,
                    volume.log10(),
                ))
            }
        }
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
        let energy = 2.24e15;
        let volume = 1.32e8;
        let log_density = ve_eos
            .at(energy, volume, StateVar::Density)
            .expect("point is on the grid");
        let fit_density = volume.log10() + 0.7 * energy.log10() - 20.0;
        assert!((log_density - fit_density) / fit_density < 1e-2);
    }
}

use std::io::{self, Read};

use ndarray::{s, Array3, ArrayView3};

use crate::{
    fort_unfmt::read_fort_record,
    index::{Idx, Range},
    interp::LinearInterpolator,
    is_close::IsClose,
    raw_tables::eos::{AllRawTables, MetalRawTables, RawTable, RAW_TABLES},
};

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
                let l_tables = self.tables.swap_remove(i);
                let r_tables = self.tables.swap_remove(j);
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
                let left = self.tables.swap_remove(i);
                let right = self.tables.swap_remove(j);
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
}

#[cfg(test)]
mod tests {
    use crate::is_close::IsClose;

    use super::AllTables;

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
    fn interp_metal() {}
}

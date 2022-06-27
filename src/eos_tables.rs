use std::io::{Read, self};

use ndarray::{Array3, s, ArrayView3};

use crate::{fort_unfmt::read_fort_record, index::{Range, Idx}, raw_tables::eos::{AllRawTables, MetalRawTables, RawTable, RAW_TABLES}};

/// The collection of all MESA tables available
pub struct AllTables {
    metallicities: Range,
    tables: Vec<ConstMetalTables>,
}

impl AllTables {
    pub fn take_at_metallicity(mut self, metallicity: f64) -> Result<ConstMetalTables, String> {
        match self.metallicities.find_value(metallicity) {
            Idx::Exact(i) => Ok(self.tables.swap_remove(i)),
            Idx::Between(i, j) => todo!(),
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
    he_fracs: Range,
    tables: Vec<VolumeEnergyTable>,
}

impl From<&MetalRawTables> for ConstMetalTables {
    fn from(rawtbls: &MetalRawTables) -> Self {
        Self {
            he_fracs: rawtbls.he_fracs,
            tables: rawtbls.tables.iter().map(|t| t.into()).collect(),
        }
    }
}

impl ConstMetalTables {
    pub fn take_at_he_frac(mut self, he_frac: f64) -> Result<VolumeEnergyTable, String> {
        match self.he_fracs.find_value(he_frac) {
            Idx::Exact(i) => Ok(self.tables.swap_remove(i)),
            Idx::Between(i, j) => todo!(),
            Idx::OutOfRange => Err("Helium fraction out of range".to_owned()),
        }
    }
}

impl From<&RawTable> for VolumeEnergyTable {
    fn from(rawtbl: &RawTable) -> Self {
        Self::read_from(rawtbl.0).expect("raw tables are well-formed")
    }
}

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
        let mut shape = [0_u32; 3];  // ne, nv, nvars
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
        for i_v in 0..shape[0] {
            for i_e in 0..shape[1] {
                let mut slc = values.slice_mut(s![i_v, i_e, ..]);
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
}

#[cfg(test)]
mod tests {
    use crate::{is_close::IsClose};

    use super::AllTables;

    #[test]
    fn read_eos_table() {
        let ve_eos = AllTables::default()
            .take_at_metallicity(0.02)
            .expect("metallicity is in range")
            .take_at_he_frac(0.18)
            .expect("helium fraction is in range");
        assert!(ve_eos.log_volume().first().is_close(0.0));
        assert!(ve_eos.log_volume().last().is_close(14.0));
        assert!(ve_eos.log_energy().first().is_close(10.5));
        assert!(ve_eos.log_energy().last().is_close(17.5));
    }
}

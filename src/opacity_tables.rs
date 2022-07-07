use std::io::{self, Read};

use ndarray::{s, Array2, Array3, Array4, Axis};

use crate::{
    fort_unfmt::read_fort_record,
    index::{IdxLin, OutOfBoundsError, Range},
    interp::LinearInterpolator,
    raw_tables::opacity::{RawOpacityTable, RAW_TABLES},
};

/// The full opacity table.
pub struct AllTables {
    metallicities: Range,
    h_fracs: Range,
    log_temperature: Range,
    log_r: Range,
    values: Array4<f64>,
}

fn read_range<R: Read>(reader: &mut R, size: usize) -> io::Result<Range> {
    let mut range_vals = vec![0.0; size];
    read_fort_record(reader, &mut range_vals)?;
    Range::from_slice(&range_vals).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

impl AllTables {
    fn read_from<R: Read>(mut reader: R) -> io::Result<Self> {
        let mut shape = [0_u32; 4]; // nz, nx, nt, nr
        read_fort_record(&mut reader, &mut shape)?;
        shape.swap(2, 3); // nr, nt in file header
        let shape = shape.map(|e| e as usize);

        let metallicities = read_range(&mut reader, shape[0])?;
        let h_fracs = read_range(&mut reader, shape[1])?;
        let log_temperature = read_range(&mut reader, shape[2])?;
        let log_r = read_range(&mut reader, shape[3])?;

        let mut values = Array4::zeros(shape);
        for i_z in 0..metallicities.n_values() {
            for i_x in 0..h_fracs.n_values() {
                for i_t in 0..log_temperature.n_values() {
                    let mut slc = values.slice_mut(s![i_z, i_x, i_t, ..]);
                    let raw_slc = slc.as_slice_mut().expect("values should be contiguous");
                    read_fort_record(&mut reader, raw_slc)?;
                }
            }
        }

        Ok(Self {
            metallicities,
            h_fracs,
            log_temperature,
            log_r,
            values,
        })
    }

    pub fn take_at_metallicity(
        self,
        metallicity: f64,
    ) -> Result<ConstMetalTables, OutOfBoundsError> {
        match self.metallicities.idx_lin(metallicity)? {
            IdxLin::Exact(i) => Ok(ConstMetalTables {
                h_fracs: self.h_fracs,
                log_temperature: self.log_temperature,
                log_r: self.log_r,
                values: self.values.index_axis_move(Axis(0), i),
            }),
            IdxLin::Between(i, j) => {
                let lin = LinearInterpolator::new(
                    self.metallicities.at(i),
                    self.metallicities.at(j),
                    metallicity,
                );
                let values = lin.interp(
                    self.values.index_axis(Axis(0), i),
                    self.values.index_axis(Axis(0), j),
                );
                Ok(ConstMetalTables {
                    h_fracs: self.h_fracs,
                    log_temperature: self.log_temperature,
                    log_r: self.log_r,
                    values,
                })
            }
        }
    }
}

impl From<&RawOpacityTable> for AllTables {
    fn from(rawtbl: &RawOpacityTable) -> Self {
        Self::read_from(rawtbl.0).expect("raw tables are well-formed")
    }
}

impl Default for AllTables {
    fn default() -> Self {
        (&RAW_TABLES).into()
    }
}

/// Opacity table at constant metallicity.
pub struct ConstMetalTables {
    h_fracs: Range,
    log_temperature: Range,
    log_r: Range,
    values: Array3<f64>,
}

impl ConstMetalTables {
    pub fn take_at_h_frac(self, h_frac: f64) -> Result<RTempTable, OutOfBoundsError> {
        match self.h_fracs.idx_lin(h_frac)? {
            IdxLin::Exact(i) => Ok(RTempTable {
                log_temperature: self.log_temperature,
                log_r: self.log_r,
                values: self.values.index_axis_move(Axis(0), i),
            }),
            IdxLin::Between(i, j) => {
                let lin = LinearInterpolator::new(self.h_fracs.at(i), self.h_fracs.at(j), h_frac);
                let values = lin.interp(
                    self.values.index_axis(Axis(0), i),
                    self.values.index_axis(Axis(0), j),
                );
                Ok(RTempTable {
                    log_temperature: self.log_temperature,
                    log_r: self.log_r,
                    values,
                })
            }
        }
    }
}

/// Opacity table at constant metallicity and helium fraction.
pub struct RTempTable {
    log_temperature: Range,
    log_r: Range,
    values: Array2<f64>,
}

impl RTempTable {
    pub fn at(&self, log_temperature: f64, log_r: f64) -> Result<f64, OutOfBoundsError> {
        match (
            self.log_temperature.idx_lin(log_temperature)?,
            self.log_r.idx_lin(log_r)?,
        ) {
            (IdxLin::Exact(it), IdxLin::Exact(ir)) => Ok(self.values[[it, ir]]),
            (IdxLin::Between(it, itp), IdxLin::Exact(ir)) => {
                let lin = LinearInterpolator::new(
                    self.log_temperature.at(it),
                    self.log_temperature.at(itp),
                    log_temperature,
                );
                let val = lin.interp(
                    self.values.slice(s![it, ir..ir + 1]),
                    self.values.slice(s![itp, ir..ir + 1]),
                );
                Ok(val[0])
            }
            (IdxLin::Exact(it), IdxLin::Between(ir, irp)) => {
                let lin =
                    LinearInterpolator::new(self.log_r.at(ir), self.log_temperature.at(irp), log_r);
                let val = lin.interp(
                    self.values.slice(s![it..it + 1, ir]),
                    self.values.slice(s![it..it + 1, irp]),
                );
                Ok(val[0])
            }
            (IdxLin::Between(it, itp), IdxLin::Between(ir, irp)) => {
                let lint = LinearInterpolator::new(
                    self.log_temperature.at(it),
                    self.log_temperature.at(itp),
                    log_temperature,
                );
                let linr =
                    LinearInterpolator::new(self.log_r.at(ir), self.log_temperature.at(irp), log_r);
                let at_it = linr.interp(
                    self.values.slice(s![it..it + 1, ir]),
                    self.values.slice(s![it..it + 1, irp]),
                );
                let at_itp = linr.interp(
                    self.values.slice(s![itp..itp + 1, ir]),
                    self.values.slice(s![itp..itp + 1, irp]),
                );
                let val = lint.interp(at_it, at_itp);
                Ok(val[0])
            }
        }
    }
}

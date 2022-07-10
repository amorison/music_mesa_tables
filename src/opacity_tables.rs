use std::io::{self, Read};

use ndarray::{s, Array2, Array3, Array4, ArrayView2, ArrayView3, Axis};

use crate::{
    fort_unfmt::read_fort_record,
    index::{CustomRange, IdxLin, Indexable, LinearInterpolable, OutOfBoundsError, Range},
    interp::{lin_interp_2d, LinearInterpolator, LinearStencil},
    raw_tables::opacity::{RawOpacityTable, RAW_TABLES},
};

/// The full opacity table.
pub struct AllTables {
    metallicities: CustomRange,
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

        let mut z_range = vec![0.0; shape[0]];
        read_fort_record(&mut reader, &mut z_range)?;
        let metallicities = CustomRange::new(z_range).unwrap();

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
                metallicity,
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
                    metallicity,
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
    metallicity: f64,
    h_fracs: Range,
    log_temperature: Range,
    log_r: Range,
    values: Array3<f64>,
}

impl ConstMetalTables {
    pub fn take_at_h_frac(self, h_frac: f64) -> Result<RTempTable, OutOfBoundsError> {
        match self.h_fracs.idx_lin(h_frac)? {
            IdxLin::Exact(i) => Ok(RTempTable {
                metallicity: self.metallicity,
                h_frac,
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
                    metallicity: self.metallicity,
                    h_frac,
                    log_temperature: self.log_temperature,
                    log_r: self.log_r,
                    values,
                })
            }
        }
    }

    pub fn metallicity(&self) -> f64 {
        self.metallicity
    }

    pub fn values(&self) -> ArrayView3<f64> {
        self.values.view()
    }

    pub fn at(
        &self,
        h_frac: f64,
        log_temperature: f64,
        log_r: f64,
    ) -> Result<f64, OutOfBoundsError> {
        let logt_st = self.log_temperature.linear_stencil(log_temperature)?;
        let logr_st = self.log_r.linear_stencil(log_r)?;
        match self.h_fracs.linear_stencil(h_frac)? {
            LinearStencil::Exact { i, .. } => Ok(lin_interp_2d(
                logt_st,
                logr_st,
                self.values().index_axis_move(Axis(0), i),
            )),
            LinearStencil::Between { ileft, iright, lin } => {
                let mut ltable = self.values().index_axis_move(Axis(0), ileft);
                let mut rtable = self.values().index_axis_move(Axis(0), iright);
                logt_st.slice_view(Axis(0), &mut ltable);
                let logt_st = logt_st.slice_view(Axis(0), &mut rtable);
                logr_st.slice_view(Axis(1), &mut ltable);
                let logr_st = logr_st.slice_view(Axis(1), &mut rtable);
                let table = lin.interp(ltable, rtable);
                Ok(lin_interp_2d(logt_st, logr_st, table.view()))
            }
        }
    }
}

/// Opacity table at constant metallicity and helium fraction.
pub struct RTempTable {
    metallicity: f64,
    h_frac: f64,
    log_temperature: Range,
    log_r: Range,
    values: Array2<f64>,
}

impl RTempTable {
    pub fn metallicity(&self) -> f64 {
        self.metallicity
    }

    pub fn h_frac(&self) -> f64 {
        self.h_frac
    }

    pub fn values(&self) -> ArrayView2<f64> {
        self.values.view()
    }

    pub fn at(&self, log_temperature: f64, log_r: f64) -> Result<f64, OutOfBoundsError> {
        Ok(lin_interp_2d(
            self.log_temperature.linear_stencil(log_temperature)?,
            self.log_r.linear_stencil(log_r)?,
            self.values(),
        ))
    }
}

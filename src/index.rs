use crate::{
    interp::{LinearInterpolator, LinearStencil, SplineStencil},
    is_close::IsClose,
};
use thiserror::Error;

#[derive(Copy, Clone)]
pub struct Range {
    first: f64,
    step: f64,
    n_values: usize,
}

pub struct CustomRange(Vec<f64>);

#[derive(Error, Debug)]
pub enum RangeError {
    #[error("range should have at least two elements")]
    FewerThanTwoValues,
    #[error("range should be in stricly increasing order")]
    NotInIncreasingOrder,
    #[error("range should be a linear space")]
    NotLinear,
}

#[derive(Error, Debug)]
#[error("value {value} is out of bounds")]
pub struct OutOfBoundsError {
    value: f64,
}

pub enum IdxLin {
    Exact(usize),
    Between(usize, usize),
}

pub trait Indexable {
    fn get(&self, index: usize) -> Option<f64>;

    fn at(&self, index: usize) -> f64 {
        self.get(index).expect("index is out of range")
    }
}

/// Index can be used for linear interpolation.
pub trait LinearInterpolable: Indexable {
    fn idx_lin(&self, value: f64) -> Result<IdxLin, OutOfBoundsError>;

    fn linear_stencil(&self, value: f64) -> Result<LinearStencil, OutOfBoundsError> {
        match self.idx_lin(value)? {
            IdxLin::Exact(i) => Ok(LinearStencil::Exact { i, value }),
            IdxLin::Between(ileft, iright) => Ok(LinearStencil::Between {
                ileft,
                iright,
                lin: LinearInterpolator::new(self.at(ileft), self.at(iright), value),
            }),
        }
    }
}

pub struct RangeIterator {
    range: Range,
    idx: usize,
}

impl RangeIterator {
    fn new(range: Range) -> Self {
        Self { range, idx: 0 }
    }
}

impl Iterator for RangeIterator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let current_val = self.range.get(self.idx);
        self.idx += 1;
        current_val
    }
}

impl IntoIterator for Range {
    type Item = f64;

    type IntoIter = RangeIterator;

    fn into_iter(self) -> Self::IntoIter {
        RangeIterator::new(self)
    }
}

impl Range {
    pub(crate) const fn new(first: f64, step: f64, n_values: usize) -> Self {
        // FP arithmetic not available in const fn yet, keeping this function
        // in the crate to avoid misuses.
        // assert!(step > 0.0);
        assert!(n_values > 1);
        Self {
            first,
            step,
            n_values,
        }
    }

    pub fn from_slice(slc: &[f64]) -> Result<Self, RangeError> {
        let n_values = slc.len();
        if n_values < 2 {
            return Err(RangeError::FewerThanTwoValues);
        }
        let first = slc[0];
        let step = (slc[n_values - 1] - first) / (n_values - 1) as f64;
        if step <= 0.0 {
            return Err(RangeError::NotInIncreasingOrder);
        }
        let range = Self {
            first,
            step,
            n_values,
        };
        if !range
            .into_iter()
            .enumerate()
            .all(|(i, v)| v.is_close(slc[i]))
        {
            return Err(RangeError::NotLinear);
        }
        Ok(range)
    }

    pub fn contains(&self, value: f64) -> bool {
        let last = self.last();
        (value >= self.first && value <= last) || value.is_close(self.first) || value.is_close(last)
    }

    pub fn subrange_in(&self, other: Range) -> Option<Range> {
        let (ifirst, first) = self
            .into_iter()
            .enumerate()
            .find(|&(_, v)| other.contains(v))?;
        let n_values = 1
            + (ifirst + 1..self.n_values)
                .filter(|&i| other.contains(self.at(i)))
                .count();
        if n_values >= 2 {
            Some(Self {
                first,
                n_values,
                step: self.step,
            })
        } else {
            None
        }
    }

    pub fn first(&self) -> f64 {
        self.first
    }

    pub fn last(&self) -> f64 {
        self.at(self.n_values - 1)
    }

    pub fn step(&self) -> f64 {
        self.step
    }

    pub fn n_values(&self) -> usize {
        self.n_values
    }

    pub fn spline_stencil(&self, value: f64) -> Result<SplineStencil, OutOfBoundsError> {
        let lside = self.at(1);
        let rside = self.at(self.n_values - 2);
        if self.n_values < 4 {
            Err(OutOfBoundsError { value })
        } else if value < lside || value >= rside {
            Err(OutOfBoundsError { value })
        } else {
            let iguess = ((value - self.first) / self.step).floor() as usize;
            Ok(SplineStencil {
                r: iguess - 1..iguess + 3,
                xs: [
                    self.at(iguess - 1),
                    self.at(iguess),
                    self.at(iguess + 1),
                    self.at(iguess + 2),
                ],
                at: value,
            })
        }
    }
}

impl Indexable for Range {
    fn get(&self, index: usize) -> Option<f64> {
        if index >= self.n_values {
            None
        } else {
            Some(index as f64 * self.step + self.first)
        }
    }
}

impl LinearInterpolable for Range {
    fn idx_lin(&self, value: f64) -> Result<IdxLin, OutOfBoundsError> {
        if value.is_close(self.first) {
            Ok(IdxLin::Exact(0))
        } else if value.is_close(self.last()) {
            Ok(IdxLin::Exact(self.n_values - 1))
        } else if value < self.first || value > self.last() {
            Err(OutOfBoundsError { value })
        } else {
            let iguess = ((value - self.first) / self.step).floor() as usize;
            if value.is_close(self.at(iguess)) {
                Ok(IdxLin::Exact(iguess))
            } else if self.get(iguess + 1).map_or(false, |v| v.is_close(value)) {
                Ok(IdxLin::Exact(iguess + 1))
            } else {
                Ok(IdxLin::Between(iguess, iguess + 1))
            }
        }
    }
}

impl CustomRange {
    pub fn new(values: Vec<f64>) -> Result<Self, RangeError> {
        let n_values = values.len();
        if n_values < 2 {
            return Err(RangeError::FewerThanTwoValues);
        }
        if !(1..n_values).all(|i| values[i] > values[i - 1]) {
            return Err(RangeError::NotInIncreasingOrder);
        }
        Ok(Self(values))
    }

    pub fn n_values(&self) -> usize {
        self.0.len()
    }
}

impl Indexable for CustomRange {
    fn get(&self, index: usize) -> Option<f64> {
        self.0.get(index).copied()
    }
}

impl LinearInterpolable for CustomRange {
    fn idx_lin(&self, value: f64) -> Result<IdxLin, OutOfBoundsError> {
        let ilast = self.0.len() - 1;
        if value.is_close(self.0[0]) {
            Ok(IdxLin::Exact(0))
        } else if value.is_close(self.0[ilast]) {
            Ok(IdxLin::Exact(self.0.len() - 1))
        } else if value < self.0[0] || value > self.0[ilast] {
            Err(OutOfBoundsError { value })
        } else {
            // This could be implemented with a dichotomy, but in practice this
            // is only used once on ranges with few elements (metallicity of
            // opacity tables).
            let iguess = self
                .0
                .iter()
                .enumerate()
                .find_map(|(i, &v)| (v > value).then_some(i))
                .unwrap()
                - 1;
            if value.is_close(self.at(iguess)) {
                Ok(IdxLin::Exact(iguess))
            } else if self.get(iguess + 1).map_or(false, |v| v.is_close(value)) {
                Ok(IdxLin::Exact(iguess + 1))
            } else {
                Ok(IdxLin::Between(iguess, iguess + 1))
            }
        }
    }
}

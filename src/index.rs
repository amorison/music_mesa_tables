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

    pub fn at(&self, index: usize) -> f64 {
        self.get(index).expect("index is out of range")
    }

    pub fn get(&self, index: usize) -> Option<f64> {
        if index >= self.n_values {
            None
        } else {
            Some(index as f64 * self.step + self.first)
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

    pub fn idx_lin(&self, value: f64) -> Result<IdxLin, OutOfBoundsError> {
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

    pub fn linear_stencil(&self, value: f64) -> Result<LinearStencil, OutOfBoundsError> {
        match self.idx_lin(value)? {
            IdxLin::Exact(i) => Ok(LinearStencil::Exact { i, value }),
            IdxLin::Between(ileft, iright) => Ok(LinearStencil::Between {
                ileft,
                iright,
                lin: LinearInterpolator::new(self.at(iright), self.at(ileft), value),
            }),
        }
    }

    pub fn spline_stencil(&self, value: f64) -> Result<SplineStencil, OutOfBoundsError> {
        let lside = self.at(1);
        let rside = self.at(self.n_values - 2);
        if self.n_values < 4 {
            Err(OutOfBoundsError { value })
        } else if value.is_close(lside) {
            Ok(SplineStencil::Exact { i: 1, value })
        } else if value.is_close(rside) {
            Ok(SplineStencil::Exact {
                i: self.n_values - 2,
                value,
            })
        } else if value < lside || value > rside {
            Err(OutOfBoundsError { value })
        } else {
            let iguess = ((value - self.first) / self.step).floor() as usize;
            if value.is_close(self.at(iguess)) {
                Ok(SplineStencil::Exact { i: iguess, value })
            } else if self.get(iguess + 1).map_or(false, |v| v.is_close(value)) {
                Ok(SplineStencil::Exact {
                    i: iguess + 1,
                    value,
                })
            } else {
                Ok(SplineStencil::Centered {
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
}

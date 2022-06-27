use crate::is_close::IsClose;

#[derive(Copy, Clone)]
pub struct Range {
    first: f64,
    step: f64,
    n_values: usize,
}

pub enum Idx {
    Exact(usize),
    Between(usize, usize),
    OutOfRange,
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
        Self { first, step, n_values }
    }

    pub fn from_slice(slc: &[f64]) -> Result<Self, &'static str> {
        let n_values = slc.len();
        if n_values < 2 {
            return Err("given slice should have at least two elements");
        }
        let first = slc[0];
        let step = (slc[n_values - 1] - first) / (n_values - 1) as f64;
        if step <= 0.0 {
            return Err("given slice should be in stricly increasing order");
        }
        let range = Self { first, step, n_values};
        if !range.into_iter().enumerate().all(|(i, v)| v.is_close(slc[i])) {
            return Err("given slice should be a linear space");
        }
        Ok(range)
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

    pub fn find_value(&self, value: f64) -> Idx {
        if value.is_close(self.first) {
            Idx::Exact(0)
        } else if value.is_close(self.last()) {
            Idx::Exact(self.n_values - 1)
        } else if value < self.first || value > self.last() {
            Idx::OutOfRange
        } else {
            let iguess = ((value - self.first) / self.step).floor() as usize;
            if value == self.at(iguess) {
                Idx::Exact(iguess)
            } else if self.get(iguess + 1).map_or(false, |v| v == value) {
                Idx::Exact(iguess + 1)
            } else {
                Idx::Between(iguess, iguess + 1)
            }
        }
    }
}

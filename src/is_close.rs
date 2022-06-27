use crate::index::Range;

pub(crate) trait IsClose {
    fn is_close(&self, other: Self) -> bool;
}

impl IsClose for f64 {
    #[inline]
    fn is_close(&self, other: f64) -> bool {
        (self - other).abs() <= 1e-12
    }
}

impl IsClose for Range {
    #[inline]
    fn is_close(&self, other: Range) -> bool {
        self.first().is_close(other.first())
            && self.step().is_close(other.step())
            && self.n_values() == other.n_values()
    }
}

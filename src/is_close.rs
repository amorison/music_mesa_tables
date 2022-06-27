pub trait IsClose {
    fn is_close(&self, other: Self) -> bool;
}

impl IsClose for f64 {
    #[inline]
    fn is_close(&self, other: f64) -> bool {
        (self - other).abs() <= 1e-12
    }
}

use ndarray::{Array, ArrayBase, ArrayView2, Data, Dimension};

use crate::index::{IdxSpline, Range};

pub struct LinearInterpolator {
    left_coef: f64,
}

impl LinearInterpolator {
    pub fn new(left_anchor: f64, right_anchor: f64, at: f64) -> Self {
        assert!(left_anchor < right_anchor);
        assert!(at > left_anchor && at < right_anchor);
        let left_coef = (right_anchor - at) / (right_anchor - left_anchor);
        Self { left_coef }
    }

    pub fn interp<D, S1, S2>(
        &self,
        left: ArrayBase<S1, D>,
        right: ArrayBase<S2, D>,
    ) -> Array<f64, D>
    where
        S1: Data<Elem = f64>,
        S2: Data<Elem = f64>,
        D: Dimension,
    {
        let mut left: Array<f64, D> = left.to_owned();
        left *= self.left_coef;
        left.scaled_add(1.0 - self.left_coef, &right);
        left
    }
}

/// Centered spline interpolator.
///
/// This takes 2 known points on each side on the desired interpolation location.
pub fn cubic_spline(x: [f64; 4], y: [f64; 4], at: f64) -> f64 {
    let dy_dx_left = (y[2] - y[0]) / (x[2] - x[0]);
    let dy_dx_right = (y[3] - y[1]) / (x[3] - x[1]);
    let a = dy_dx_left * (x[2] - x[1]) - (y[2] - y[1]);
    let b = -dy_dx_right * (x[2] - x[1]) - (y[2] - y[1]);
    let t = (at - x[1]) / (x[2] - x[1]);
    (1.0 - t) * y[1] + t * y[2] + t * (1.0 - t) * (a * (1.0 - t) + b * t)
}

pub(crate) fn cubic_spline_2d(
    x: Range,
    y: Range,
    z: ArrayView2<'_, f64>,
    at_x: f64,
    at_y: f64,
) -> Result<f64, &'static str> {
    match (x.idx_spline(at_x), y.idx_spline(at_y)) {
        (IdxSpline::OutOfRange, _) | (_, IdxSpline::OutOfRange) => {
            Err("requested position is out of range")
        }
        (IdxSpline::Exact(i_x), IdxSpline::Exact(i_y)) => Ok(z[[i_x, i_y]]),
        (IdxSpline::Exact(i_x), IdxSpline::Centered(yl2, yl1, yr1, yr2)) => Ok(cubic_spline(
            [y.at(yl2), y.at(yl1), y.at(yr1), y.at(yr2)],
            [z[[i_x, yl2]], z[[i_x, yl1]], z[[i_x, yr1]], z[[i_x, yr2]]],
            at_y,
        )),
        (IdxSpline::Centered(xl2, xl1, xr1, xr2), IdxSpline::Exact(i_y)) => Ok(cubic_spline(
            [x.at(xl2), x.at(xl1), x.at(xr1), x.at(xr2)],
            [z[[xl2, i_y]], z[[xl1, i_y]], z[[xr1, i_y]], z[[xr2, i_y]]],
            at_x,
        )),
        (IdxSpline::Centered(xl2, xl1, xr1, xr2), IdxSpline::Centered(yl2, yl1, yr1, yr2)) => {
            let xs = [x.at(xl2), x.at(xl1), x.at(xr1), x.at(xr2)];
            let mut z_at_ys = [0.0; 4];
            for (i, iy) in (yl2..=yr2).enumerate() {
                z_at_ys[i] = cubic_spline(
                    xs,
                    [z[[xl2, iy]], z[[xl1, iy]], z[[xr1, iy]], z[[xr2, iy]]],
                    at_x,
                );
            }
            Ok(cubic_spline(
                [y.at(yl2), y.at(yl1), y.at(yr1), y.at(yr2)],
                z_at_ys,
                at_y,
            ))
        }
    }
}

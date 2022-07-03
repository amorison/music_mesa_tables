use ndarray::{Array, ArrayBase, ArrayView2, Data, Dimension};

use crate::index::{IdxLin, Range};

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
    match (x.find_value(at_x), y.find_value(at_y)) {
        (IdxLin::OutOfRange, _) | (_, IdxLin::OutOfRange) => {
            Err("requested position is out of range")
        }
        (IdxLin::Exact(0), _) | (IdxLin::Between(0, _), _) => Err("requested x is in lower values"),
        (_, IdxLin::Exact(0)) | (_, IdxLin::Between(0, _)) => Err("requested y is in lower values"),
        (IdxLin::Exact(n), _) | (IdxLin::Between(_, n), _) if n == x.n_values() - 1 => {
            Err("requested x is in higher values")
        }
        (_, IdxLin::Exact(n)) | (_, IdxLin::Between(_, n)) if n == y.n_values() - 1 => {
            Err("requested y is in higher values")
        }
        (IdxLin::Exact(i_x), IdxLin::Exact(i_y)) => Ok(z[[i_x, i_y]]),
        (IdxLin::Exact(i_x), IdxLin::Between(i_y, _)) => Ok(cubic_spline(
            [y.at(i_y - 1), y.at(i_y), y.at(i_y + 1), y.at(i_y + 2)],
            [
                z[[i_x, i_y - 1]],
                z[[i_x, i_y]],
                z[[i_x, i_y + 1]],
                z[[i_x, i_y + 2]],
            ],
            at_y,
        )),
        (IdxLin::Between(i_x, _), IdxLin::Exact(i_y)) => Ok(cubic_spline(
            [x.at(i_x - 1), x.at(i_x), x.at(i_x + 1), x.at(i_x + 2)],
            [
                z[[i_x - 1, i_y]],
                z[[i_x, i_y]],
                z[[i_x + 1, i_y]],
                z[[i_x + 2, i_y]],
            ],
            at_x,
        )),
        (IdxLin::Between(i_x, _), IdxLin::Between(i_y, _)) => {
            let xs = [x.at(i_x - 1), x.at(i_x), x.at(i_x + 1), x.at(i_x + 2)];
            let mut z_at_ys = [0.0; 4];
            for (i, iy) in (i_y - 1..=i_y + 2).enumerate() {
                z_at_ys[i] = cubic_spline(
                    xs,
                    [
                        z[[i_x - 1, iy]],
                        z[[i_x, iy]],
                        z[[i_x + 1, iy]],
                        z[[i_x + 2, iy]],
                    ],
                    at_x,
                );
            }
            Ok(cubic_spline(
                [y.at(i_y - 1), y.at(i_y), y.at(i_y + 1), y.at(i_y + 2)],
                z_at_ys,
                at_y,
            ))
        }
    }
}

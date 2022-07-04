use ndarray::{Array, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Dimension};

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

fn low_level_spline(x: [f64; 4], y: [f64; 4], at: f64) -> f64 {
    let dy_dx_left = (y[2] - y[0]) / (x[2] - x[0]);
    let dy_dx_right = (y[3] - y[1]) / (x[3] - x[1]);
    let a = dy_dx_left * (x[2] - x[1]) - (y[2] - y[1]);
    let b = -dy_dx_right * (x[2] - x[1]) + (y[2] - y[1]);
    let t = (at - x[1]) / (x[2] - x[1]);
    (1.0 - t) * y[1] + t * y[2] + t * (1.0 - t) * (a * (1.0 - t) + b * t)
}

/// Centered cubic spline interpolator.
pub enum SplineStencil {
    Exact {
        i: usize,
        value: f64,
    },
    Centered {
        r: std::ops::Range<usize>,
        xs: [f64; 4],
        at: f64,
    },
}

impl SplineStencil {
    pub fn apply_to(&self, arr: ArrayView1<'_, f64>) -> f64 {
        match self {
            SplineStencil::Exact { i, .. } => arr[*i],
            SplineStencil::Centered { r, xs, at } => {
                let i = r.start;
                let y: [f64; 4] = [arr[i], arr[i + 1], arr[i + 2], arr[i + 3]];
                low_level_spline(*xs, y, *at)
            }
        }
    }
}

pub(crate) fn cubic_spline_2d(
    x_st: SplineStencil,
    y_st: SplineStencil,
    z: ArrayView2<'_, f64>,
) -> f64 {
    match (x_st, y_st) {
        (SplineStencil::Exact { i: i_x, .. }, y_st) => y_st.apply_to(z.index_axis(Axis(0), i_x)),
        (x_st, SplineStencil::Exact { i: i_y, .. }) => x_st.apply_to(z.index_axis(Axis(1), i_y)),
        (
            x_st @ SplineStencil::Centered { .. },
            SplineStencil::Centered {
                r: y_r,
                xs: ys,
                at: at_y,
            },
        ) => {
            let mut z_at_ys = [0.0; 4];
            for (i, iy) in y_r.enumerate() {
                z_at_ys[i] = x_st.apply_to(z.index_axis(Axis(1), iy));
            }
            low_level_spline(ys, z_at_ys, at_y)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::is_close::IsClose;

    use super::low_level_spline;

    fn low_level_spline_analytic<F: Fn(f64) -> f64>(f: F) {
        let xs = [-1., 0., 1., 2.];
        let ys = xs.map(&f);
        assert!((0..=10)
            .map(|i| i as f64 / 10.0)
            .all(|at| { dbg!(low_level_spline(xs, ys, at)).is_close(dbg!(f(at))) }));
    }

    #[test]
    fn low_level_spline_quad_funcs() {
        low_level_spline_analytic(|x| 3.0 * x * x - 2.0 * x + 5.0);
        low_level_spline_analytic(|x| 42.0 * x - 7.0);
    }
}

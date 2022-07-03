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
    OutOfRange,
}

impl SplineStencil {
    fn low_level_spline(x: [f64; 4], y: [f64; 4], at: f64) -> f64 {
        let dy_dx_left = (y[2] - y[0]) / (x[2] - x[0]);
        let dy_dx_right = (y[3] - y[1]) / (x[3] - x[1]);
        let a = dy_dx_left * (x[2] - x[1]) - (y[2] - y[1]);
        let b = -dy_dx_right * (x[2] - x[1]) - (y[2] - y[1]);
        let t = (at - x[1]) / (x[2] - x[1]);
        (1.0 - t) * y[1] + t * y[2] + t * (1.0 - t) * (a * (1.0 - t) + b * t)
    }

    pub fn apply_to(&self, arr: ArrayView1<'_, f64>) -> Result<f64, &'static str> {
        match self {
            SplineStencil::Exact { i, .. } => Ok(arr[*i]),
            SplineStencil::OutOfRange => Err("out of range"),
            SplineStencil::Centered { r, xs, at } => {
                let i = r.start;
                let y: [f64; 4] = [arr[i], arr[i + 1], arr[i + 2], arr[i + 3]];
                Ok(Self::low_level_spline(*xs, y, *at))
            }
        }
    }
}

pub(crate) fn cubic_spline_2d(
    x_st: SplineStencil,
    y_st: SplineStencil,
    z: ArrayView2<'_, f64>,
) -> Result<f64, &'static str> {
    match (x_st, y_st) {
        (SplineStencil::OutOfRange, _) | (_, SplineStencil::OutOfRange) => {
            Err("requested position is out of range")
        }
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
                z_at_ys[i] = x_st.apply_to(z.index_axis(Axis(1), iy))?;
            }
            Ok(SplineStencil::low_level_spline(ys, z_at_ys, at_y))
        }
    }
}

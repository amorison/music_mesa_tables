use ndarray::{Array, ArrayBase, ArrayView, ArrayView1, ArrayView2, Axis, Data, Dimension};

#[derive(Copy, Clone)]
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

    pub fn interp_scalar(&self, left: f64, right: f64) -> f64 {
        left * self.left_coef + right * (1.0 - self.left_coef)
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

pub enum LinearStencil {
    Exact {
        i: usize,
        value: f64,
    },
    Between {
        ileft: usize,
        iright: usize,
        lin: LinearInterpolator,
    },
}

impl LinearStencil {
    pub fn apply_to(&self, arr: ArrayView1<'_, f64>) -> f64 {
        match self {
            LinearStencil::Exact { i, .. } => arr[*i],
            LinearStencil::Between { ileft, iright, lin } => {
                lin.interp_scalar(arr[*ileft], arr[*iright])
            }
        }
    }

    pub(crate) fn slice_view<D: Dimension>(
        &self,
        axis: Axis,
        arr: &mut ArrayView<'_, f64, D>,
    ) -> Self {
        match self {
            LinearStencil::Exact { i, value } => {
                arr.slice_axis_inplace(axis, (*i..*i + 1).into());
                LinearStencil::Exact {
                    i: 0,
                    value: *value,
                }
            }
            LinearStencil::Between { ileft, iright, lin } => {
                arr.slice_axis_inplace(axis, (*ileft..=*iright).into());
                LinearStencil::Between {
                    ileft: 0,
                    iright: 1,
                    lin: *lin,
                }
            }
        }
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
pub struct SplineStencil {
    pub ileft: usize,
    pub xs: [f64; 4],
    pub at: f64,
}

impl SplineStencil {
    pub fn apply_to(&self, arr: ArrayView1<'_, f64>) -> f64 {
        let i = self.ileft;
        let y: [f64; 4] = [arr[i], arr[i + 1], arr[i + 2], arr[i + 3]];
        low_level_spline(self.xs, y, self.at)
    }

    pub(crate) fn slice_view<D: Dimension>(
        &self,
        axis: Axis,
        arr: &mut ArrayView<'_, f64, D>,
    ) -> Self {
        let i0 = self.ileft;
        arr.slice_axis_inplace(axis, (i0..i0 + 4).into());
        Self {
            ileft: 0,
            xs: self.xs,
            at: self.at,
        }
    }
}

pub(crate) fn lin_interp_2d(
    x_st: LinearStencil,
    y_st: LinearStencil,
    z: ArrayView2<'_, f64>,
) -> f64 {
    match (x_st, y_st) {
        (LinearStencil::Exact { i: i_x, .. }, y_st) => y_st.apply_to(z.index_axis(Axis(0), i_x)),
        (x_st, LinearStencil::Exact { i: i_y, .. }) => x_st.apply_to(z.index_axis(Axis(1), i_y)),
        (
            x_st @ LinearStencil::Between { .. },
            LinearStencil::Between {
                ileft: iyl,
                iright: iyr,
                lin: ylin,
            },
        ) => {
            let z_at_iyl = x_st.apply_to(z.index_axis(Axis(1), iyl));
            let z_at_iyr = x_st.apply_to(z.index_axis(Axis(1), iyr));
            ylin.interp_scalar(z_at_iyl, z_at_iyr)
        }
    }
}

pub(crate) fn cubic_spline_2d(
    x_st: SplineStencil,
    y_st: SplineStencil,
    z: ArrayView2<'_, f64>,
) -> f64 {
    let SplineStencil {
        ileft: iy0,
        xs: ys,
        at: at_y,
    } = y_st;
    let z_at_ys = [
        x_st.apply_to(z.index_axis(Axis(1), iy0)),
        x_st.apply_to(z.index_axis(Axis(1), iy0 + 1)),
        x_st.apply_to(z.index_axis(Axis(1), iy0 + 2)),
        x_st.apply_to(z.index_axis(Axis(1), iy0 + 3)),
    ];
    low_level_spline(ys, z_at_ys, at_y)
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

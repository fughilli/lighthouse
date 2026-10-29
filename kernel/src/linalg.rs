//! Small dense row-major matrices for the filters and solvers (sizes ~3..40),
//! plus the SO(3) pieces the error-state models need.

use crate::math::{sqrt, Quat, Vec3};
use alloc::vec;
use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub d: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Mat {
        Mat { rows, cols, d: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Mat {
        let mut m = Mat::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    pub fn diag(v: &[f64]) -> Mat {
        let mut m = Mat::zeros(v.len(), v.len());
        for (i, x) in v.iter().enumerate() {
            m[(i, i)] = *x;
        }
        m
    }

    pub fn t(&self) -> Mat {
        let mut m = Mat::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                m[(c, r)] = self[(r, c)];
            }
        }
        m
    }

    pub fn mul(&self, o: &Mat) -> Mat {
        assert_eq!(self.cols, o.rows);
        let mut m = Mat::zeros(self.rows, o.cols);
        for r in 0..self.rows {
            for k in 0..self.cols {
                let a = self[(r, k)];
                if a == 0.0 {
                    continue;
                }
                for c in 0..o.cols {
                    m.d[r * o.cols + c] += a * o.d[k * o.cols + c];
                }
            }
        }
        m
    }

    pub fn add(&self, o: &Mat) -> Mat {
        let mut m = self.clone();
        for (a, b) in m.d.iter_mut().zip(&o.d) {
            *a += b;
        }
        m
    }

    pub fn sub(&self, o: &Mat) -> Mat {
        let mut m = self.clone();
        for (a, b) in m.d.iter_mut().zip(&o.d) {
            *a -= b;
        }
        m
    }

    pub fn scale(&self, s: f64) -> Mat {
        let mut m = self.clone();
        m.d.iter_mut().for_each(|x| *x *= s);
        m
    }

    /// (A + Aᵀ) / 2, to keep covariances symmetric.
    pub fn symmetrize(&mut self) {
        for r in 0..self.rows {
            for c in r + 1..self.cols {
                let v = 0.5 * (self[(r, c)] + self[(c, r)]);
                self[(r, c)] = v;
                self[(c, r)] = v;
            }
        }
    }

    pub fn block(&self, r0: usize, c0: usize, rows: usize, cols: usize) -> Mat {
        let mut m = Mat::zeros(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                m[(r, c)] = self[(r0 + r, c0 + c)];
            }
        }
        m
    }

    pub fn set_block(&mut self, r0: usize, c0: usize, b: &Mat) {
        for r in 0..b.rows {
            for c in 0..b.cols {
                self[(r0 + r, c0 + c)] = b[(r, c)];
            }
        }
    }

    /// Lower Cholesky factor of a symmetric positive-definite matrix.
    pub fn cholesky(&self) -> Option<Mat> {
        let n = self.rows;
        let mut l = Mat::zeros(n, n);
        for j in 0..n {
            let mut s = self[(j, j)];
            for k in 0..j {
                s -= l[(j, k)] * l[(j, k)];
            }
            if !(s > 0.0) {
                return None;
            }
            let ljj = sqrt(s);
            l[(j, j)] = ljj;
            for i in j + 1..n {
                let mut s = self[(i, j)];
                for k in 0..j {
                    s -= l[(i, k)] * l[(j, k)];
                }
                l[(i, j)] = s / ljj;
            }
        }
        Some(l)
    }

    /// Solves self * X = B for SPD self.
    pub fn solve_spd(&self, b: &Mat) -> Option<Mat> {
        let l = self.cholesky()?;
        let n = self.rows;
        let mut x = b.clone();
        for c in 0..b.cols {
            for i in 0..n {
                let mut s = x[(i, c)];
                for k in 0..i {
                    s -= l[(i, k)] * x[(k, c)];
                }
                x[(i, c)] = s / l[(i, i)];
            }
            for i in (0..n).rev() {
                let mut s = x[(i, c)];
                for k in i + 1..n {
                    s -= l[(k, i)] * x[(k, c)];
                }
                x[(i, c)] = s / l[(i, i)];
            }
        }
        Some(x)
    }

    pub fn inverse_spd(&self) -> Option<Mat> {
        self.solve_spd(&Mat::identity(self.rows))
    }

    pub fn col(v: &[f64]) -> Mat {
        Mat { rows: v.len(), cols: 1, d: v.to_vec() }
    }
}

impl core::ops::Index<(usize, usize)> for Mat {
    type Output = f64;
    fn index(&self, (r, c): (usize, usize)) -> &f64 {
        &self.d[r * self.cols + c]
    }
}

impl core::ops::IndexMut<(usize, usize)> for Mat {
    fn index_mut(&mut self, (r, c): (usize, usize)) -> &mut f64 {
        &mut self.d[r * self.cols + c]
    }
}

/// Rotation matrix of a unit quaternion (rows).
pub fn rot_mat(q: Quat) -> [[f64; 3]; 3] {
    let (w, x, y, z) = (q.w, q.x, q.y, q.z);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

/// [v]ₓ such that [v]ₓ u = v × u.
pub fn skew(v: Vec3) -> [[f64; 3]; 3] {
    [[0.0, -v.z, v.y], [v.z, 0.0, -v.x], [-v.y, v.x, 0.0]]
}

pub fn mat3_vec(m: &[[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}

/// vᵀ M (row vector times matrix).
pub fn vec_mat3(v: Vec3, m: &[[f64; 3]; 3]) -> Vec3 {
    Vec3::new(
        v.x * m[0][0] + v.y * m[1][0] + v.z * m[2][0],
        v.x * m[0][1] + v.y * m[1][1] + v.z * m[2][1],
        v.x * m[0][2] + v.y * m[1][2] + v.z * m[2][2],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spd_solve_and_inverse() {
        let a = Mat { rows: 3, cols: 3, d: alloc::vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0] };
        let inv = a.inverse_spd().unwrap();
        let i = a.mul(&inv);
        for r in 0..3 {
            for c in 0..3 {
                assert!((i[(r, c)] - if r == c { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
        assert!(Mat::diag(&[1.0, -1.0]).cholesky().is_none());
    }

    #[test]
    fn rot_mat_matches_quaternion_rotation() {
        let q = Quat::from_rotvec(Vec3::new(0.4, -1.2, 0.7));
        let v = Vec3::new(0.3, 0.9, -1.4);
        let a = mat3_vec(&rot_mat(q), v);
        let b = q.rotate(v);
        assert!(a.sub(b).norm() < 1e-12);
    }
}

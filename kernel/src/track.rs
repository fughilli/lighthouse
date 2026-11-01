//! One tracker's error-state EKF: IMU propagation, one scalar update per
//! sweep hit at that hit's own time.
//!
//! Nominal state: position and velocity (world), orientation world_from_body,
//! gyro and accelerometer biases (body), and a sweep-angle bias per (station,
//! axis) for up to [`MAX_STATIONS`] stations. Error state (15 + 2·stations):
//! δp, δv, δθ (right perturbation), δb_g, δb_a, δb_angle.
//!
//! The angle biases soak up what calibration and factory corrections leave
//! over (each station/axis is off by its own small constant). Without them a
//! sequential filter is pulled toward each sweep group in turn: a ripple at
//! the 30 Hz sweep cycle. The station poses are not states; their
//! covariance is folded into each hit's noise (so pose covariances include
//! calibration uncertainty).
//!
//! Data association: a sensor can report several pulses in one sweep (direct
//! hit plus reflections); the filter uses the one whose innovation is
//! smallest, if it passes the χ² gate.

use crate::Real;
use crate::linalg::{rot_mat, skew, Mat};
use crate::math::{Pose, Quat, Vec3};
use crate::model::{predict, StationModel, TrackerModel};
use crate::timing::SensorHits;

pub const GRAVITY: Real = 9.80665;
/// Stations whose sweep-angle biases are estimated (per tracker).
pub const MAX_STATIONS: usize = 4;
const N: usize = 15 + 2 * MAX_STATIONS;
const P: usize = 0;
const V: usize = 3;
const TH: usize = 6;
const BG: usize = 9;
const BA: usize = 12;
const BANG: usize = 15;

#[derive(Clone, Copy, Debug)]
pub struct TrackConfig {
    /// Angle noise (rad, 1σ).
    pub sigma_angle: Real,
    /// Accelerometer white noise (m/s², 1σ per sample-second, i.e. m/s²·√s).
    pub accel_noise: Real,
    /// Gyro white noise (rad/s·√s).
    pub gyro_noise: Real,
    pub accel_bias_walk: Real,
    pub gyro_bias_walk: Real,
    /// χ² gate for one hit (1 dof).
    pub gate: Real,
    /// Sweep-angle bias: initial σ and random walk (rad, rad·√s⁻¹).
    pub angle_bias_init: Real,
    pub angle_bias_walk: Real,
}

impl Default for TrackConfig {
    fn default() -> TrackConfig {
        TrackConfig {
            // Per-hit noise is ~0.05-0.4 mrad, but calibration and model errors
            // are correlated across hits; 3 mrad (with the IMU) tracked best on
            // the recorded turntable/slider sessions.
            sigma_angle: 3e-3,
            accel_noise: 0.5,
            gyro_noise: 0.01,
            accel_bias_walk: 1e-3,
            gyro_bias_walk: 1e-4,
            gate: 16.0,
            // Tuned on the recorded sessions + a live still capture: removes the
            // 30 Hz ripple (0.95 -> 0.03 mm) at ~0.1-0.3 mm cost on the turntables.
            angle_bias_init: 1e-3,
            angle_bias_walk: 2e-6,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TrackState {
    pub pos: Vec3,
    pub vel: Vec3,
    pub rot: Quat,
    pub gyro_bias: Vec3,
    pub accel_bias: Vec3,
    /// Per (station slot, axis): `angle_bias[2 * slot + axis]` (rad).
    pub angle_bias: [Real; 2 * MAX_STATIONS],
    pub cov: Mat,
    /// Device time of the state (ticks, unwrapped).
    pub time: i64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UpdateStats {
    pub used: u32,
    pub gated: u32,
    /// Of `used`, how many weren't the longest pulse (a reflection was longer).
    pub not_longest: u32,
    pub sum_sq_norm: Real,
    /// Innovation (rad) of the last hit used.
    pub last_innovation: Real,
}

impl TrackState {
    pub fn new(pose: &Pose, pose_cov: &Mat, time: i64, angle_bias_init: Real) -> TrackState {
        let mut cov = Mat::zeros(N, N);
        for i in 0..3 {
            for j in 0..3 {
                cov[(P + i, P + j)] = pose_cov[(i, j)];
                cov[(TH + i, TH + j)] = pose_cov[(3 + i, 3 + j)];
                cov[(P + i, TH + j)] = pose_cov[(i, 3 + j)];
                cov[(TH + i, P + j)] = pose_cov[(3 + i, j)];
            }
            cov[(V + i, V + i)] = 0.25;
            cov[(BG + i, BG + i)] = 1e-4;
            cov[(BA + i, BA + i)] = 0.05;
        }
        for i in 0..2 * MAX_STATIONS {
            cov[(BANG + i, BANG + i)] = angle_bias_init * angle_bias_init;
        }
        TrackState {
            pos: pose.pos,
            vel: Vec3::ZERO,
            rot: pose.rot,
            gyro_bias: Vec3::ZERO,
            accel_bias: Vec3::ZERO,
            angle_bias: [0.0; 2 * MAX_STATIONS],
            cov,
            time,
        }
    }

    pub fn pose(&self) -> Pose {
        Pose { pos: self.pos, rot: self.rot }
    }

    /// 6x6 (δp, δθ) covariance of the pose.
    pub fn pose_cov(&self) -> Mat {
        let mut c = Mat::zeros(6, 6);
        let idx = [P, P + 1, P + 2, TH, TH + 1, TH + 2];
        for (a, &i) in idx.iter().enumerate() {
            for (b, &j) in idx.iter().enumerate() {
                c[(a, b)] = self.cov[(i, j)];
            }
        }
        c
    }

    /// Propagates to `time` with one IMU sample (accel in g, gyro rad/s,
    /// body frame), held constant over the interval.
    pub fn propagate(&mut self, time: i64, accel_g: Vec3, gyro: Vec3, cfg: &TrackConfig) {
        let dt = (time - self.time) as Real / crate::lh1::TICKS_PER_SECOND;
        if dt <= 0.0 {
            return;
        }
        let dt = dt.min(0.1);
        let f = accel_g.scale(GRAVITY).sub(self.accel_bias);
        let w = gyro.sub(self.gyro_bias);
        let r = rot_mat(self.rot);
        let a = self.rot.rotate(f).add(Vec3::new(0.0, 0.0, -GRAVITY));
        self.pos = self.pos.add(self.vel.scale(dt)).add(a.scale(0.5 * dt * dt));
        self.vel = self.vel.add(a.scale(dt));
        self.rot = self.rot.mul(Quat::from_rotvec(w.scale(dt))).normalized();

        // P ← Φ P Φᵀ + Q, Φ = I + G (G = F dt, phi_entries: 30 entries, all in the δp, δv,
        // δθ rows). Φ P = P + G P changes only those 9 rows, and (Φ P) Φᵀ = A + A Gᵀ only
        // those 9 columns: ~1.4k multiply-adds in place, against ~13k and five 23x23
        // allocations for the dense products. This runs for every sweep hit (the filter
        // follows the IMU to each hit's time) and was most of the tracker's time on the P4.
        let g = phi_entries(&r, f, w, dt);
        let p = &mut self.cov.d;
        let mut old = [0.0; 9 * N];
        old.copy_from_slice(&p[..9 * N]); // rows 0..9 of P
        for &(i, k, v) in &g {
            let (i, k) = (i as usize, k as usize);
            for c in 0..N {
                let pk = if k < 9 { old[k * N + c] } else { p[k * N + c] };
                p[i * N + c] += v * pk;
            }
        }
        for rr in 0..N {
            old[rr * 9..rr * 9 + 9].copy_from_slice(&p[rr * N..rr * N + 9]); // columns 0..9 of Φ P
        }
        for &(i, k, v) in &g {
            let (i, k) = (i as usize, k as usize);
            for rr in 0..N {
                let a = if k < 9 { old[rr * 9 + k] } else { p[rr * N + k] };
                p[rr * N + i] += v * a;
            }
        }
        for i in 0..3 {
            p[(V + i) * N + V + i] += cfg.accel_noise * cfg.accel_noise * dt;
            p[(TH + i) * N + TH + i] += cfg.gyro_noise * cfg.gyro_noise * dt;
            p[(BG + i) * N + BG + i] += cfg.gyro_bias_walk * cfg.gyro_bias_walk * dt;
            p[(BA + i) * N + BA + i] += cfg.accel_bias_walk * cfg.accel_bias_walk * dt;
        }
        for i in 0..2 * MAX_STATIONS {
            p[(BANG + i) * N + BANG + i] += cfg.angle_bias_walk * cfg.angle_bias_walk * dt;
        }
        self.cov.symmetrize();
        self.time = time;
    }

    /// Without IMU: constant velocity to `time`.
    pub fn coast(&mut self, time: i64, cfg: &TrackConfig) {
        let accel = self.rot.conj().rotate(Vec3::new(0.0, 0.0, 1.0)).add(self.accel_bias.scale(1.0 / GRAVITY));
        let gyro = self.gyro_bias;
        self.propagate(time, accel, gyro, cfg);
    }

    /// Updates with one sensor's hit candidates from one sweep. Returns
    /// whether a candidate was used.
    pub fn update_hit(
        &mut self,
        model: &TrackerModel,
        station: &StationModel,
        station_cov: Option<&Mat>,
        station_slot: Option<usize>,
        axis: u8,
        hits: &SensorHits,
        cfg: &TrackConfig,
        stats: &mut UpdateStats,
    ) -> bool {
        let sensor = match model.sensors.get(hits.sensor as usize) {
            Some(s) => *s,
            None => return false,
        };
        let mut j = predict(station, &self.pose(), sensor, axis);
        let bias_idx = station_slot.filter(|s| *s < MAX_STATIONS).map(|s| 2 * s + axis as usize);
        if let Some(b) = bias_idx {
            j.angle += self.angle_bias[b];
        }
        // Visible only in front of the station.
        if j.in_station.z >= 0.0 {
            return false;
        }
        let mut h = [0.0; N];
        for k in 0..3 {
            h[P + k] = j.tracker[k];
            h[TH + k] = j.tracker[3 + k];
        }
        if let Some(b) = bias_idx {
            h[BANG + b] = 1.0;
        }
        // S = H P Hᵀ + σ² + Js Σs Jsᵀ; H has at most 7 non-zeros (δp, δθ, one bias).
        let mut nz = [0usize; 7];
        let mut nn = 0;
        for k in 0..N {
            if h[k] != 0.0 {
                nz[nn] = k;
                nn += 1;
                if nn == nz.len() {
                    break;
                }
            }
        }
        let mut ph = [0.0; N];
        for r in 0..N {
            let row = &self.cov.d[r * N..(r + 1) * N];
            let mut s = 0.0;
            for &c in &nz[..nn] {
                s += row[c] * h[c];
            }
            ph[r] = s;
        }
        let mut s = cfg.sigma_angle * cfg.sigma_angle;
        for r in 0..N {
            s += h[r] * ph[r];
        }
        if let Some(sc) = station_cov {
            for a in 0..6 {
                for b in 0..6 {
                    s += j.station[a] * sc[(a, b)] * j.station[b];
                }
            }
        }
        let best = hits
            .candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.angle - j.angle))
            .min_by(|a, b| crate::math::fabs(a.1).partial_cmp(&crate::math::fabs(b.1)).unwrap());
        let (rank, innov) = match best {
            Some(b) => b,
            None => return false,
        };
        let d2 = innov * innov / s;
        if d2 > cfg.gate {
            stats.gated += 1;
            return false;
        }
        stats.used += 1;
        stats.last_innovation = innov;
        stats.sum_sq_norm += d2;
        if rank != 0 {
            stats.not_longest += 1;
        }
        // K = P Hᵀ / S; x ⊕= K innov; P = (I - K H) P (Joseph not needed for scalar
        // with symmetrize).
        let mut dx = [0.0; N];
        for r in 0..N {
            dx[r] = ph[r] / s * innov;
        }
        for r in 0..N {
            let a = ph[r] / s;
            for (x, pc) in self.cov.d[r * N..(r + 1) * N].iter_mut().zip(&ph) {
                *x -= a * pc;
            }
        }
        self.cov.symmetrize();
        self.inject(&dx);
        true
    }

    /// While the IMU says the tracker is still: velocity is zero and the
    /// accelerometer (bias-corrected) points along world +z.
    pub fn update_still(&mut self, accel_g: Vec3, sigma_vel: Real, sigma_up: Real) {
        for i in 0..3 {
            let mut h = [0.0; N];
            h[V + i] = 1.0;
            let innov = -[self.vel.x, self.vel.y, self.vel.z][i];
            self.scalar_update(&h, innov, sigma_vel * sigma_vel);
        }
        let f = accel_g.scale(GRAVITY).sub(self.accel_bias).normalized();
        let r = rot_mat(self.rot);
        // up = R f; d up / dθ = -R [f]x
        let fx = skew(f);
        for i in 0..2 {
            let up = self.rot.rotate(f);
            let mut h = [0.0; N];
            for c in 0..3 {
                let mut v = 0.0;
                for k in 0..3 {
                    v += r[i][k] * fx[k][c];
                }
                h[TH + c] = -v;
            }
            let innov = -[up.x, up.y][i];
            self.scalar_update(&h, innov, sigma_up * sigma_up);
        }
    }

    fn scalar_update(&mut self, h: &[Real; N], innov: Real, r: Real) {
        let mut ph = [0.0; N];
        for a in 0..N {
            let mut s = 0.0;
            for c in 0..N {
                s += self.cov[(a, c)] * h[c];
            }
            ph[a] = s;
        }
        let mut s = r;
        for a in 0..N {
            s += h[a] * ph[a];
        }
        let mut dx = [0.0; N];
        for a in 0..N {
            dx[a] = ph[a] / s * innov;
        }
        for a in 0..N {
            for c in 0..N {
                self.cov[(a, c)] -= ph[a] * ph[c] / s;
            }
        }
        self.cov.symmetrize();
        self.inject(&dx);
    }

    fn inject(&mut self, dx: &[Real; N]) {
        self.pos = self.pos.add(Vec3::new(dx[P], dx[P + 1], dx[P + 2]));
        self.vel = self.vel.add(Vec3::new(dx[V], dx[V + 1], dx[V + 2]));
        self.rot = self.rot.mul(Quat::from_rotvec(Vec3::new(dx[TH], dx[TH + 1], dx[TH + 2]))).normalized();
        self.gyro_bias = self.gyro_bias.add(Vec3::new(dx[BG], dx[BG + 1], dx[BG + 2]));
        self.accel_bias = self.accel_bias.add(Vec3::new(dx[BA], dx[BA + 1], dx[BA + 2]));
        for i in 0..2 * MAX_STATIONS {
            self.angle_bias[i] += dx[BANG + i];
        }
    }
}

/// The off-identity entries of Φ = I + F dt (row, column, value): the error state's
/// kinematics (δp from δv; δv from δθ and δb_a; δθ from itself and δb_g).
fn phi_entries(r: &[[Real; 3]; 3], f: Vec3, w: Vec3, dt: Real) -> [(u8, u8, Real); 30] {
    let rf = mul3(r, &skew(f));
    let wx = skew(w);
    let mut g = [(0u8, 0u8, 0.0); 30];
    let mut n = 0;
    let mut put = |i: usize, k: usize, v: Real| {
        g[n] = (i as u8, k as u8, v);
        n += 1;
    };
    for i in 0..3 {
        put(P + i, V + i, dt);
        for j in 0..3 {
            put(V + i, TH + j, -rf[i][j] * dt);
            put(V + i, BA + j, -r[i][j] * dt);
            if i != j {
                put(TH + i, TH + j, -wx[i][j] * dt);
            }
        }
        put(TH + i, BG + i, -dt);
    }
    g
}

fn mul3(a: &[[Real; 3]; 3], b: &[[Real; 3]; 3]) -> [[Real; 3]; 3] {
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                m[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lh1::AxisCal;
    use crate::timing::HitCandidate;

    #[test]
    fn propagate_matches_the_dense_product() {
        let cfg = TrackConfig::default();
        let pose = Pose { pos: Vec3::new(0.1, 0.2, 1.0), rot: Quat::from_rotvec(Vec3::new(0.3, -0.5, 1.2)) };
        let mut st = TrackState::new(&pose, &Mat::diag(&[1e-4, 2e-4, 3e-4, 1e-3, 2e-3, 3e-3]), 0, 3e-3);
        // A full, non-trivial covariance (SPD: L Lᵀ plus the diagonal).
        let mut l = Mat::zeros(N, N);
        for r in 0..N {
            for c in 0..=r {
                l[(r, c)] = 0.01 * crate::math::sin((r * 31 + c * 7) as Real);
            }
        }
        st.cov = st.cov.add(&l.mul(&l.t()));
        st.vel = Vec3::new(0.5, -0.2, 0.1);
        st.gyro_bias = Vec3::new(0.01, -0.02, 0.005);
        st.accel_bias = Vec3::new(0.05, 0.0, -0.03);
        let (accel, gyro) = (Vec3::new(0.1, -0.3, 0.95), Vec3::new(1.5, -0.7, 2.2));
        let mut dense = st.clone();
        st.propagate(48_000 * 4, accel, gyro, &cfg);
        // The reference: Φ P Φᵀ + Q with the dense Φ.
        let dt = 4e-3;
        let f = accel.scale(GRAVITY).sub(dense.accel_bias);
        let w = gyro.sub(dense.gyro_bias);
        let r = rot_mat(dense.rot);
        let mut phi = Mat::identity(N);
        for (i, k, v) in phi_entries(&r, f, w, dt) {
            phi[(i as usize, k as usize)] += v;
        }
        let mut q = Mat::zeros(N, N);
        for i in 0..3 {
            q[(V + i, V + i)] = cfg.accel_noise * cfg.accel_noise * dt;
            q[(TH + i, TH + i)] = cfg.gyro_noise * cfg.gyro_noise * dt;
            q[(BG + i, BG + i)] = cfg.gyro_bias_walk * cfg.gyro_bias_walk * dt;
            q[(BA + i, BA + i)] = cfg.accel_bias_walk * cfg.accel_bias_walk * dt;
        }
        for i in 0..2 * MAX_STATIONS {
            q[(BANG + i, BANG + i)] = cfg.angle_bias_walk * cfg.angle_bias_walk * dt;
        }
        dense.cov = phi.mul(&dense.cov).mul(&phi.t()).add(&q);
        let worst = st.cov.d.iter().zip(&dense.cov.d).map(|(a, b)| crate::math::fabs(a - b)).fold(0.0, Real::max);
        let scale = dense.cov.d.iter().map(|x| crate::math::fabs(*x)).fold(0.0, Real::max);
        assert!(worst <= 1e-5 * scale, "max |diff| {} (scale {})", worst, scale);
    }

    #[test]
    fn still_tracker_stays_put_and_gets_confident() {
        let cfg = TrackConfig::default();
        let station = StationModel {
            cal: [AxisCal::default(); 2],
            world_from_station: Pose { pos: Vec3::new(0.0, -2.0, 1.0), rot: Quat::from_rotvec(Vec3::new(crate::consts::FRAC_PI_2, 0.0, 0.0)) },
        };
        let mut model = TrackerModel::default();
        for i in 0..8 {
            let a = i as Real * crate::consts::FRAC_PI_4;
            model.sensors.push(Vec3::new(0.04 * crate::math::cos(a), 0.04 * crate::math::sin(a), 0.01 * (i % 2) as Real));
        }
        let truth = Pose { pos: Vec3::new(0.05, 0.0, 1.0), rot: Quat::IDENTITY };
        let start = Pose { pos: Vec3::new(0.06, 0.01, 0.99), rot: Quat::from_rotvec(Vec3::new(0.01, -0.01, 0.02)) };
        let mut st = TrackState::new(&start, &Mat::diag(&[1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]), 0, 3e-3);
        let mut stats = UpdateStats::default();
        let mut t = 0i64;
        for k in 0..400 {
            // IMU at 1 kHz: still, so accel = +1 g up in body (identity rotation).
            for _ in 0..8 {
                t += 48_000;
                st.propagate(t, Vec3::new(0.0, 0.0, 1.0), Vec3::ZERO, &cfg);
            }
            let axis = (k % 2) as u8;
            for (i, s) in model.sensors.iter().enumerate() {
                let a = predict(&station, &truth, *s, axis).angle;
                let hits = SensorHits {
                    sensor: i as u16 as u8,
                    // A longer reflection 0.1 rad off, then the direct hit.
                    candidates: alloc::vec![
                        HitCandidate { angle: a + 0.1, len: 2000, time: 0 },
                        HitCandidate { angle: a, len: 800, time: 0 },
                    ],
                };
                // One station: no angle bias (it would be indistinguishable from
                // moving the tracker).
                st.update_hit(&model, &station, None, None, axis, &hits, &cfg, &mut stats);
            }
        }
        let err = st.pos.sub(truth.pos).norm();
        assert!(err < 1e-3, "position off by {} m", err);
        assert!(stats.not_longest > 0 && stats.not_longest == stats.used, "{:?}", stats);
        assert!(st.pose_cov()[(0, 0)] < 1e-5);
    }
}

//! Batch solvers: one tracker pose from a set of sweep hits (bootstrap), and
//! the base-station calibration from still "scenes" (station and scene poses
//! together).
//!
//! Both are robust (Huber-weighted) Levenberg–Marquardt over right-perturbed
//! poses, return covariances from the final normal matrix, and use gravity:
//! the tracker's accelerometer while still and each station's OOTX accel
//! vector must point along world +z.

use crate::linalg::{rot_mat, skew, Mat};
use crate::math::{Pose, Quat, Vec3};
use crate::model::{predict, retract, StationModel, TrackerModel};
use alloc::vec;
use alloc::vec::Vec;

/// One sweep-angle observation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obs {
    pub station: usize,
    pub axis: u8,
    pub sensor: u16,
    pub angle: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct SolveConfig {
    /// Angle noise (rad, 1σ).
    pub sigma_angle: f64,
    /// Tracker gravity direction noise while still (rad).
    pub sigma_tracker_up: f64,
    /// Station OOTX gravity direction noise (rad; the vector is 8-bit).
    pub sigma_station_up: f64,
    /// Huber threshold, in σ.
    pub huber: f64,
    pub max_iterations: usize,
}

impl Default for SolveConfig {
    fn default() -> SolveConfig {
        SolveConfig { sigma_angle: 3e-4, sigma_tracker_up: 0.01, sigma_station_up: 0.03, huber: 3.0, max_iterations: 50 }
    }
}

fn huber_weight(r: f64, k: f64) -> f64 {
    let a = libm::fabs(r);
    if a <= k {
        1.0
    } else {
        k / a
    }
}

/// Rows of a weighted least-squares system: Σ w (r + J δ)².
struct Rows {
    n: usize,
    jtj: Mat,
    jtr: Vec<f64>,
    cost: f64,
    count: usize,
}

impl Rows {
    fn new(n: usize) -> Rows {
        Rows { n, jtj: Mat::zeros(n, n), jtr: vec![0.0; n], cost: 0.0, count: 0 }
    }

    /// Adds a residual (already divided by its σ) with sparse Jacobian entries.
    fn add(&mut self, r: f64, j: &[(usize, f64)], huber: Option<f64>) {
        let w = huber.map_or(1.0, |k| huber_weight(r, k));
        for &(a, ja) in j {
            self.jtr[a] += w * ja * r;
            for &(b, jb) in j {
                self.jtj[(a, b)] += w * ja * jb;
            }
        }
        // Huber cost.
        self.cost += match huber {
            Some(k) if libm::fabs(r) > k => k * (2.0 * libm::fabs(r) - k),
            _ => r * r,
        };
        self.count += 1;
    }

    /// LM step: (JᵀWJ + λ diag) δ = -JᵀWr.
    fn step(&self, lambda: f64) -> Option<Vec<f64>> {
        let mut a = self.jtj.clone();
        for i in 0..self.n {
            a[(i, i)] += lambda * (a[(i, i)] + 1e-9);
        }
        let b = Mat::col(&self.jtr.iter().map(|x| -x).collect::<Vec<_>>());
        a.solve_spd(&b).map(|x| x.d)
    }
}

/// Gravity residual: R·a (unit) should be +z; residual = R·a - z (all three
/// components, so upside down is as wrong as it looks). Returns the residual
/// and its d/dθ rows for a right perturbation of R.
fn up_residual(r: Quat, a: Vec3) -> ([f64; 3], [[f64; 3]; 3]) {
    let rm = rot_mat(r);
    let up = r.rotate(a.normalized());
    // d(R exp(δθ) a)/dθ = -R [a]x
    let ax = skew(a.normalized());
    let mut d = [[0.0; 3]; 3];
    for row in 0..3 {
        for c in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += rm[row][k] * ax[k][c];
            }
            d[row][c] = -s;
        }
    }
    ([up.x, up.y, up.z - 1.0], d)
}

// ------------------------------------------------------------------ pose

#[derive(Clone, Debug)]
pub struct PoseSolution {
    pub world_from_body: Pose,
    /// 6x6 over (δp, δθ).
    pub cov: Mat,
    /// RMS angle residual over inliers (rad).
    pub rms: f64,
    pub inliers: usize,
    pub obs: usize,
}

fn pose_rows(
    model: &TrackerModel,
    stations: &[StationModel],
    obs: &[Obs],
    up_body: Option<Vec3>,
    pose: &Pose,
    cfg: &SolveConfig,
) -> Rows {
    let mut rows = Rows::new(6);
    for o in obs {
        let st = &stations[o.station];
        let j = predict(st, pose, model.sensors[o.sensor as usize], o.axis);
        let r = (j.angle - o.angle) / cfg.sigma_angle;
        let jr: Vec<(usize, f64)> = (0..6).map(|k| (k, j.tracker[k] / cfg.sigma_angle)).collect();
        rows.add(r, &jr, Some(cfg.huber));
    }
    if let Some(a) = up_body {
        let (r, d) = up_residual(pose.rot, a);
        for i in 0..3 {
            let jr: Vec<(usize, f64)> = (0..3).map(|k| (3 + k, d[i][k] / cfg.sigma_tracker_up)).collect();
            rows.add(r[i] / cfg.sigma_tracker_up, &jr, None);
        }
    }
    rows
}

/// Refines a tracker pose from `init`.
pub fn refine_pose(
    model: &TrackerModel,
    stations: &[StationModel],
    obs: &[Obs],
    up_body: Option<Vec3>,
    init: Pose,
    cfg: &SolveConfig,
) -> Option<PoseSolution> {
    let mut pose = init;
    let mut lambda = 1e-3;
    let mut rows = pose_rows(model, stations, obs, up_body, &pose, cfg);
    for _ in 0..cfg.max_iterations {
        let d = rows.step(lambda)?;
        let cand = retract(&pose, &d);
        let crow = pose_rows(model, stations, obs, up_body, &cand, cfg);
        if crow.cost < rows.cost {
            let done = d.iter().map(|x| x * x).sum::<f64>() < 1e-16;
            pose = cand;
            rows = crow;
            lambda = (lambda * 0.3).max(1e-9);
            if done {
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e8 {
                break;
            }
        }
    }
    let cov = rows.jtj.inverse_spd()?;
    let mut sum = 0.0;
    let mut inl = 0;
    for o in obs {
        let j = predict(&stations[o.station], &pose, model.sensors[o.sensor as usize], o.axis);
        let r = j.angle - o.angle;
        if libm::fabs(r) <= cfg.huber * cfg.sigma_angle {
            sum += r * r;
            inl += 1;
        }
    }
    Some(PoseSolution {
        world_from_body: pose,
        cov,
        rms: if inl > 0 { libm::sqrt(sum / inl as f64) } else { f64::INFINITY },
        inliers: inl,
        obs: obs.len(),
    })
}

/// Finds a tracker pose without a prior: tries the tracker in front of each
/// station at several ranges and headings (gravity fixes roll/pitch when
/// given) and keeps the best fit.
pub fn bootstrap_pose(
    model: &TrackerModel,
    stations: &[StationModel],
    obs: &[Obs],
    up_body: Option<Vec3>,
    cfg: &SolveConfig,
) -> Option<PoseSolution> {
    let mut best: Option<PoseSolution> = None;
    let level = up_body.map(|a| quat_between(a.normalized(), Vec3::new(0.0, 0.0, 1.0))).unwrap_or(Quat::IDENTITY);
    let mut used = [false; 16];
    for o in obs {
        if o.station < 16 {
            used[o.station] = true;
        }
    }
    for (si, st) in stations.iter().enumerate() {
        if si < 16 && !used[si] {
            continue;
        }
        let ahead = st.world_from_station.rot.rotate(Vec3::new(0.0, 0.0, -1.0));
        for range in [0.3, 0.8, 1.6, 3.0] {
            let pos = st.world_from_station.pos.add(ahead.scale(range));
            for yaw in 0..8 {
                let q = Quat::from_rotvec(Vec3::new(0.0, 0.0, yaw as f64 * core::f64::consts::FRAC_PI_4)).mul(level);
                if let Some(s) = refine_pose(model, stations, obs, up_body, Pose { pos, rot: q }, cfg) {
                    let better = match &best {
                        None => true,
                        Some(b) => (s.inliers, -s.rms) > (b.inliers, -b.rms) && s.rms.is_finite(),
                    };
                    if better {
                        best = Some(s);
                    }
                }
            }
        }
    }
    best
}

/// Do the stations agree on where a still tracker is? Solves its pose from
/// each station alone (both axes, at least `min_sensors` sensors, a clean
/// fit) and returns the largest pairwise difference (position m, rotation
/// rad). None with fewer than two such stations. A station whose stored pose
/// is wrong (it was moved) puts the tracker somewhere else.
pub fn station_disagreement(
    model: &TrackerModel,
    stations: &[StationModel],
    obs: &[Obs],
    up_body: Vec3,
    min_sensors: usize,
    cfg: &SolveConfig,
) -> Option<(f64, f64)> {
    let mut poses: Vec<Pose> = Vec::new();
    for (si, st) in stations.iter().enumerate() {
        let mine: Vec<Obs> = obs.iter().filter(|o| o.station == si).map(|o| Obs { station: 0, ..*o }).collect();
        let mut sensors = 0u64;
        let mut axes = 0u8;
        for o in &mine {
            sensors |= 1 << (o.sensor & 63);
            axes |= 1 << o.axis;
        }
        if axes != 3 || (sensors.count_ones() as usize) < min_sensors {
            continue;
        }
        let Some(sol) = bootstrap_pose(model, core::slice::from_ref(st), &mine, Some(up_body), cfg) else { continue };
        if sol.rms < 2e-3 && sol.inliers >= 8 && sol.inliers * 3 >= sol.obs * 2 {
            poses.push(sol.world_from_body);
        }
    }
    if poses.len() < 2 {
        return None;
    }
    let (mut dp, mut dr) = (0.0f64, 0.0f64);
    for i in 0..poses.len() {
        for j in i + 1..poses.len() {
            dp = dp.max(poses[i].pos.sub(poses[j].pos).norm());
            let w = libm::fabs(poses[i].rot.conj().mul(poses[j].rot).w).min(1.0);
            dr = dr.max(2.0 * libm::acos(w));
        }
    }
    Some((dp, dr))
}

/// Candidate angles of one (station, axis, sensor): several when a sensor saw
/// a reflection as well as the direct hit.
#[derive(Clone, Debug)]
pub struct CandidateObs {
    pub station: usize,
    pub axis: u8,
    pub sensor: u16,
    pub angles: Vec<f64>,
}

/// Bootstrap with data association: start from each key's first candidate,
/// then repeatedly re-pick the candidate nearest the current pose, drop keys
/// with none within `reject` (rad), and refit.
pub fn bootstrap_pose_assoc(
    model: &TrackerModel,
    stations: &[StationModel],
    cands: &[CandidateObs],
    up_body: Option<Vec3>,
    reject: f64,
    cfg: &SolveConfig,
) -> Option<PoseSolution> {
    let first: Vec<Obs> =
        cands.iter().map(|c| Obs { station: c.station, axis: c.axis, sensor: c.sensor, angle: c.angles[0] }).collect();
    let mut sol = bootstrap_pose(model, stations, &first, up_body, cfg)?;
    for _ in 0..5 {
        let mut obs = Vec::new();
        for c in cands {
            let p = predict(&stations[c.station], &sol.world_from_body, model.sensors[c.sensor as usize], c.axis);
            let best = c.angles.iter().copied().min_by(|a, b| libm::fabs(a - p.angle).partial_cmp(&libm::fabs(b - p.angle)).unwrap());
            if let Some(a) = best {
                if libm::fabs(a - p.angle) <= reject {
                    obs.push(Obs { station: c.station, axis: c.axis, sensor: c.sensor, angle: a });
                }
            }
        }
        if obs.len() < 8 {
            return None;
        }
        let next = refine_pose(model, stations, &obs, up_body, sol.world_from_body, cfg)?;
        let moved = next.world_from_body.pos.sub(sol.world_from_body.pos).norm();
        sol = next;
        if moved < 1e-5 {
            break;
        }
    }
    Some(sol)
}

/// Rotation vector of a unit quaternion (log map).
pub fn rotvec(q: Quat) -> Vec3 {
    let q = if q.w < 0.0 { Quat::new(-q.w, -q.x, -q.y, -q.z) } else { q };
    let s = libm::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
    if s < 1e-12 {
        return Vec3::new(2.0 * q.x, 2.0 * q.y, 2.0 * q.z);
    }
    let angle = 2.0 * libm::atan2(s, q.w);
    Vec3::new(q.x, q.y, q.z).scale(angle / s)
}

/// Rotation taking unit vector a onto unit vector b.
pub fn quat_between(a: Vec3, b: Vec3) -> Quat {
    let c = a.cross(b);
    let d = a.dot(b);
    if d < -0.999999 {
        // Opposite: any perpendicular axis.
        let axis = if libm::fabs(a.x) < 0.9 { a.cross(Vec3::new(1.0, 0.0, 0.0)) } else { a.cross(Vec3::new(0.0, 1.0, 0.0)) };
        return Quat::from_rotvec(axis.normalized().scale(core::f64::consts::PI));
    }
    Quat::new(1.0 + d, c.x, c.y, c.z).normalized()
}

/// Finds a station's pose from a still tracker at a known pose (the first
/// calibration, with nothing known about the station): tries the station at
/// several directions and ranges around the tracker, upright and looking at
/// it, and refines the best (Huber LM over the station's 6 dof).
pub fn bootstrap_station(
    model: &TrackerModel,
    cal: [crate::lh1::AxisCal; 2],
    world_from_body: &Pose,
    obs: &[Obs],
    up_station: Option<Vec3>,
    cfg: &SolveConfig,
) -> Option<(StationModel, f64)> {
    let target = world_from_body.pos;
    let mut best: Option<(StationModel, f64, usize)> = None;
    for range in [0.5, 1.0, 2.0, 3.5] {
        for el in [-0.3, 0.3, 0.8] {
            for az in 0..12 {
                let azr = az as f64 * core::f64::consts::PI / 6.0;
                let dir = Vec3::new(libm::cos(el) * libm::cos(azr), libm::cos(el) * libm::sin(azr), libm::sin(el));
                let pos = target.add(dir.scale(range));
                let st = StationModel { cal, world_from_station: look_at(pos, target) };
                if let Some((s, cost, inl)) = refine_station(model, st, world_from_body, obs, up_station, cfg) {
                    if best.as_ref().map_or(true, |b| inl > b.2 || (inl == b.2 && cost < b.1)) {
                        best = Some((s, cost, inl));
                    }
                }
            }
        }
    }
    best.map(|b| (b.0, b.1))
}

/// An upright station (+y up) at `pos` looking at `target` (along its -z).
pub fn look_at(pos: Vec3, target: Vec3) -> Pose {
    let z = pos.sub(target).normalized();
    let mut x = Vec3::new(0.0, 0.0, 1.0).cross(z);
    if x.norm() < 1e-6 {
        x = Vec3::new(1.0, 0.0, 0.0);
    }
    let x = x.normalized();
    let y = z.cross(x);
    Pose { pos, rot: quat_from_columns(x, y, z) }
}

pub fn quat_from_columns(x: Vec3, y: Vec3, z: Vec3) -> Quat {
    let m = [[x.x, y.x, z.x], [x.y, y.y, z.y], [x.z, y.z, z.z]];
    let tr = m[0][0] + m[1][1] + m[2][2];
    let q = if tr > 0.0 {
        let s = libm::sqrt(tr + 1.0) * 2.0;
        Quat::new(0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = libm::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0;
        Quat::new((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    } else if m[1][1] > m[2][2] {
        let s = libm::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0;
        Quat::new((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    } else {
        let s = libm::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0;
        Quat::new((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)
    };
    q.normalized()
}

fn refine_station(
    model: &TrackerModel,
    init: StationModel,
    world_from_body: &Pose,
    obs: &[Obs],
    up_station: Option<Vec3>,
    cfg: &SolveConfig,
) -> Option<(StationModel, f64, usize)> {
    let rows_for = |st: &StationModel| -> Rows {
        let mut rows = Rows::new(6);
        for o in obs {
            let p = predict(st, world_from_body, model.sensors[o.sensor as usize], o.axis);
            let jr: Vec<(usize, f64)> = (0..6).map(|k| (k, p.station[k] / cfg.sigma_angle)).collect();
            rows.add((p.angle - o.angle) / cfg.sigma_angle, &jr, Some(cfg.huber));
        }
        if let Some(a) = up_station {
            let (r, d) = up_residual(st.world_from_station.rot, a);
            for i in 0..3 {
                let jr: Vec<(usize, f64)> = (0..3).map(|k| (3 + k, d[i][k] / cfg.sigma_station_up)).collect();
                rows.add(r[i] / cfg.sigma_station_up, &jr, None);
            }
        }
        rows
    };
    let mut st = init;
    let mut rows = rows_for(&st);
    let mut lambda = 1e-3;
    for _ in 0..cfg.max_iterations {
        let d = rows.step(lambda)?;
        let mut cand = st.clone();
        cand.world_from_station = retract(&st.world_from_station, &d);
        let cr = rows_for(&cand);
        if cr.cost < rows.cost {
            st = cand;
            rows = cr;
            lambda = (lambda * 0.3).max(1e-9);
        } else {
            lambda *= 10.0;
            if lambda > 1e8 {
                break;
            }
        }
    }
    // The tracker must be in front of the station.
    let inl = obs
        .iter()
        .filter(|o| {
            let p = predict(&st, world_from_body, model.sensors[o.sensor as usize], o.axis);
            p.in_station.z < 0.0 && libm::fabs(p.angle - o.angle) < cfg.huber * cfg.sigma_angle
        })
        .count();
    Some((st, rows.cost, inl))
}

// ------------------------------------------------------------------ calibration

/// A stretch where the tracker was still: its observations and gravity.
/// `obs` may hold several candidates for one (station, axis, sensor) — a
/// direct hit and a steady reflection; the fit uses whichever fits, the first
/// listed to start with.
#[derive(Clone, Debug)]
pub struct Scene {
    pub obs: Vec<Obs>,
    /// Accelerometer direction (body frame), i.e. "up" as the tracker sees it.
    pub up_body: Vec3,
    pub world_from_body: Pose,
}

#[derive(Clone, Debug)]
pub struct Calibration {
    pub stations: Vec<StationModel>,
    pub scenes: Vec<Pose>,
    /// Per station: 6x6 covariance over (δp, δθ) in the gauge (reference
    /// station's position and heading fixed).
    pub station_cov: Vec<Mat>,
    /// Per station: RMS angle residual (rad) over inliers.
    pub rms: Vec<f64>,
    /// Gravity disagreement (rad) per station and mean over scenes.
    pub station_up_error: Vec<f64>,
    pub tracker_up_error: f64,
    pub dropped: usize,
}

/// A station pose known beforehand (a saved calibration), as a prior: the
/// fit pays (δp/σp)² + (δθ/σθ)² for moving away from it.
#[derive(Clone, Copy, Debug)]
pub struct StationPrior {
    pub pose: Pose,
    pub sigma_pos: f64,
    pub sigma_rot: f64,
}

/// Jointly refines station poses (all but `reference`, which keeps its
/// position and heading and may only tilt) and scene poses. `priors` (per
/// station, optional) keep a saved calibration from being overwritten by a
/// few new scenes.
pub fn calibrate(
    model: &TrackerModel,
    stations_init: &[StationModel],
    station_up: &[Option<Vec3>],
    priors: &[Option<StationPrior>],
    scenes_init: &[Scene],
    reference: usize,
    cfg: &SolveConfig,
    reject: f64,
) -> Option<Calibration> {
    let ns = stations_init.len();
    let free: Vec<usize> = (0..ns).filter(|&i| i != reference).collect();
    // Parameter layout: [ref tilt (2)] [free stations (6 each)] [scenes (6 each)]
    let n = 2 + 6 * free.len() + 6 * scenes_init.len();
    let st_off = |i: usize| -> Option<usize> { free.iter().position(|&f| f == i).map(|k| 2 + 6 * k) };
    let sc_off = |j: usize| 2 + 6 * free.len() + 6 * j;

    let mut stations = stations_init.to_vec();
    let mut scenes: Vec<Pose> = scenes_init.iter().map(|s| s.world_from_body).collect();
    let key = |o: &Obs| (o.station, o.axis, o.sensor);
    // Active observations: the first candidate per key.
    let first_per_key = |all: &[Obs]| -> Vec<Obs> {
        let mut v: Vec<Obs> = Vec::new();
        for o in all {
            if !v.iter().any(|a| key(a) == key(o)) {
                v.push(*o);
            }
        }
        v
    };
    let mut obs: Vec<Vec<Obs>> = scenes_init.iter().map(|s| first_per_key(&s.obs)).collect();
    let mut dropped = 0;

    let build = |stations: &[StationModel], scenes: &[Pose], obs: &[Vec<Obs>]| -> Rows {
        let mut rows = Rows::new(n);
        for (j, sc) in scenes.iter().enumerate() {
            for o in &obs[j] {
                let st = &stations[o.station];
                let p = predict(st, sc, model.sensors[o.sensor as usize], o.axis);
                let r = (p.angle - o.angle) / cfg.sigma_angle;
                let mut jr: Vec<(usize, f64)> = (0..6).map(|k| (sc_off(j) + k, p.tracker[k] / cfg.sigma_angle)).collect();
                match st_off(o.station) {
                    Some(off) => jr.extend((0..6).map(|k| (off + k, p.station[k] / cfg.sigma_angle))),
                    None => {
                        // Reference: world tilt about x, y (left perturbation).
                        // d/dθ_world = d/dθ_right · Rᵀ.
                        let rt = rot_mat(st.world_from_station.rot);
                        for a in 0..2 {
                            let mut v = 0.0;
                            for k in 0..3 {
                                v += p.station[3 + k] * rt[a][k];
                            }
                            // The reference's position also rotates with the tilt.
                            let pos = st.world_from_station.pos;
                            let e = if a == 0 { Vec3::new(1.0, 0.0, 0.0) } else { Vec3::new(0.0, 1.0, 0.0) };
                            let dp = e.cross(pos);
                            v += p.station[0] * dp.x + p.station[1] * dp.y + p.station[2] * dp.z;
                            jr.push((a, v / cfg.sigma_angle));
                        }
                    }
                }
                rows.add(r, &jr, Some(cfg.huber));
            }
            let (r, d) = up_residual(sc.rot, scenes_init[j].up_body);
            for i in 0..3 {
                let jr: Vec<(usize, f64)> = (0..3).map(|k| (sc_off(j) + 3 + k, d[i][k] / cfg.sigma_tracker_up)).collect();
                rows.add(r[i] / cfg.sigma_tracker_up, &jr, None);
            }
        }
        for (i, st) in stations.iter().enumerate() {
            if let (Some(pr), Some(off)) = (priors.get(i).copied().flatten(), st_off(i)) {
                // Right-perturbation residual: position difference, and the
                // rotation vector of prior⁻¹·R (Jacobian ≈ identity near it).
                let dp = st.world_from_station.pos.sub(pr.pose.pos);
                let dq = pr.pose.rot.conj().mul(st.world_from_station.rot);
                let dth = rotvec(dq);
                for k in 0..3 {
                    rows.add(dp.to_array()[k] / pr.sigma_pos, &[(off + k, 1.0 / pr.sigma_pos)], None);
                    rows.add(dth.to_array()[k] / pr.sigma_rot, &[(off + 3 + k, 1.0 / pr.sigma_rot)], None);
                }
            }
            if let Some(a) = station_up[i] {
                let (r, d) = up_residual(st.world_from_station.rot, a);
                for k in 0..3 {
                    let jr: Vec<(usize, f64)> = match st_off(i) {
                        Some(off) => (0..3).map(|c| (off + 3 + c, d[k][c] / cfg.sigma_station_up)).collect(),
                        None => {
                            let rt = rot_mat(st.world_from_station.rot);
                            (0..2)
                                .map(|a| {
                                    let mut v = 0.0;
                                    for c in 0..3 {
                                        v += d[k][c] * rt[a][c];
                                    }
                                    (a, v / cfg.sigma_station_up)
                                })
                                .collect()
                        }
                    };
                    rows.add(r[k] / cfg.sigma_station_up, &jr, None);
                }
            }
        }
        rows
    };

    let apply = |stations: &[StationModel], scenes: &[Pose], d: &[f64]| -> (Vec<StationModel>, Vec<Pose>) {
        let mut st = stations.to_vec();
        let tilt = Quat::from_rotvec(Vec3::new(d[0], d[1], 0.0));
        for (i, s) in st.iter_mut().enumerate() {
            match st_off(i) {
                Some(off) => s.world_from_station = retract(&s.world_from_station, &d[off..off + 6]),
                None => {
                    s.world_from_station =
                        Pose { pos: tilt.rotate(s.world_from_station.pos), rot: tilt.mul(s.world_from_station.rot).normalized() }
                }
            }
        }
        let sc = scenes.iter().enumerate().map(|(j, p)| retract(p, &d[sc_off(j)..sc_off(j) + 6])).collect();
        (st, sc)
    };

    for _pass in 0..6 {
        let mut lambda = 1e-3;
        let mut rows = build(&stations, &scenes, &obs);
        for _ in 0..cfg.max_iterations {
            let d = rows.step(lambda)?;
            let (cs, cc) = apply(&stations, &scenes, &d);
            let crow = build(&cs, &cc, &obs);
            if crow.cost < rows.cost {
                let small = d.iter().map(|x| x * x).sum::<f64>() < 1e-18;
                stations = cs;
                scenes = cc;
                rows = crow;
                lambda = (lambda * 0.3).max(1e-9);
                if small {
                    break;
                }
            } else {
                lambda *= 10.0;
                if lambda > 1e8 {
                    break;
                }
            }
        }
        // Per key, the candidate that fits best; drop keys the fit can't
        // explain at all; refit until nothing changes.
        let mut changed = 0;
        let mut removed = 0;
        for (j, sc) in scenes.iter().enumerate() {
            let mut next: Vec<Obs> = Vec::new();
            for o in first_per_key(&scenes_init[j].obs) {
                let best = scenes_init[j]
                    .obs
                    .iter()
                    .filter(|c| key(c) == key(&o))
                    .map(|c| {
                        let p = predict(&stations[c.station], sc, model.sensors[c.sensor as usize], c.axis);
                        (*c, libm::fabs(p.angle - c.angle))
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();
                if best.1 <= reject {
                    next.push(best.0);
                } else {
                    removed += 1;
                }
            }
            let same = next.len() == obs[j].len() && next.iter().zip(&obs[j]).all(|(a, b)| a == b);
            if !same {
                changed += 1;
            }
            obs[j] = next;
        }
        dropped = removed;
        if changed == 0 {
            break;
        }
    }

    let rows = build(&stations, &scenes, &obs);
    let cov = rows.jtj.inverse_spd()?;
    let mut station_cov = Vec::new();
    for i in 0..ns {
        station_cov.push(match st_off(i) {
            Some(off) => cov.block(off, off, 6, 6),
            None => {
                let mut c = Mat::zeros(6, 6);
                c.set_block(3, 3, &Mat::diag(&[cov[(0, 0)], cov[(1, 1)], 0.0]));
                c
            }
        });
    }
    let mut rms = vec![0.0; ns];
    let mut cnt = vec![0usize; ns];
    for (j, sc) in scenes.iter().enumerate() {
        for o in &obs[j] {
            let p = predict(&stations[o.station], sc, model.sensors[o.sensor as usize], o.axis);
            rms[o.station] += (p.angle - o.angle) * (p.angle - o.angle);
            cnt[o.station] += 1;
        }
    }
    for i in 0..ns {
        rms[i] = if cnt[i] > 0 { libm::sqrt(rms[i] / cnt[i] as f64) } else { f64::NAN };
    }
    let tilt_of = |q: Quat, a: Vec3| libm::acos(q.rotate(a.normalized()).z.clamp(-1.0, 1.0));
    let station_up_error = (0..ns).map(|i| station_up[i].map_or(f64::NAN, |a| tilt_of(stations[i].world_from_station.rot, a))).collect();
    let tracker_up_error = scenes.iter().zip(scenes_init).map(|(p, s)| tilt_of(p.rot, s.up_body)).sum::<f64>() / scenes.len().max(1) as f64;
    Some(Calibration { stations, scenes, station_cov, rms, station_up_error, tracker_up_error, dropped })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lh1::AxisCal;

    fn model() -> TrackerModel {
        // A puck-ish layout: 16 sensors on a 4 cm ring at two heights.
        let mut sensors = Vec::new();
        let mut normals = Vec::new();
        for i in 0..16 {
            let a = i as f64 * core::f64::consts::PI / 8.0;
            let z = if i % 2 == 0 { 0.0 } else { 0.015 };
            sensors.push(Vec3::new(0.04 * libm::cos(a), 0.04 * libm::sin(a), z));
            normals.push(Vec3::new(libm::cos(a), libm::sin(a), 0.5).normalized());
        }
        TrackerModel { sensors, normals, body_from_head: Pose::IDENTITY }
    }

    fn stations() -> Vec<StationModel> {
        let cal = [AxisCal { phase: 0.02, tilt: -0.004, curve: 0.001, gibphase: 2.7, gibmag: 0.006 }; 2];
        // Two upright stations (station +y up), 2 m high, looking at the origin.
        let look = |pos: Vec3| {
            let fwd = Vec3::ZERO.sub(pos).normalized(); // station -z
            let z = fwd.scale(-1.0);
            let x = Vec3::new(0.0, 0.0, 1.0).cross(z).normalized();
            let y = z.cross(x);
            // Rotation with columns x, y, z.
            let m = [[x.x, y.x, z.x], [x.y, y.y, z.y], [x.z, y.z, z.z]];
            let w = libm::sqrt((1.0 + m[0][0] + m[1][1] + m[2][2]).max(1e-12)) / 2.0;
            let q = Quat::new(w, (m[2][1] - m[1][2]) / (4.0 * w), (m[0][2] - m[2][0]) / (4.0 * w), (m[1][0] - m[0][1]) / (4.0 * w));
            Pose { pos, rot: q.normalized() }
        };
        alloc::vec![
            StationModel { cal, world_from_station: look(Vec3::new(2.0, -1.0, 2.0)) },
            StationModel { cal, world_from_station: look(Vec3::new(-1.5, 2.0, 2.2)) },
        ]
    }

    fn observe(m: &TrackerModel, st: &[StationModel], pose: &Pose) -> Vec<Obs> {
        let mut v = Vec::new();
        for (si, s) in st.iter().enumerate() {
            for (k, sen) in m.sensors.iter().enumerate() {
                if !crate::model::faces(pose, *sen, m.normals[k], s) {
                    continue;
                }
                for axis in 0..2 {
                    let a = predict(s, pose, *sen, axis).angle;
                    v.push(Obs { station: si, axis, sensor: k as u16, angle: a });
                }
            }
        }
        v
    }

    #[test]
    fn bootstrap_finds_the_pose() {
        let m = model();
        let st = stations();
        let truth = Pose { pos: Vec3::new(0.2, -0.1, 0.9), rot: Quat::from_rotvec(Vec3::new(0.0, 0.0, 1.3)) };
        let obs = observe(&m, &st, &truth);
        let up = truth.rot.conj().rotate(Vec3::new(0.0, 0.0, 1.0));
        let sol = bootstrap_pose(&m, &st, &obs, Some(up), &SolveConfig::default()).unwrap();
        assert!(sol.world_from_body.pos.sub(truth.pos).norm() < 1e-6, "{:?}", sol.world_from_body);
        assert!(sol.rms < 1e-8);
    }

    #[test]
    fn a_moved_station_disagrees() {
        let m = model();
        let truth = stations();
        let pose = Pose { pos: Vec3::new(0.3, 0.2, 0.8), rot: Quat::from_rotvec(Vec3::new(0.05, 0.0, 0.7)) };
        let obs = observe(&m, &truth, &pose);
        let up = pose.rot.conj().rotate(Vec3::new(0.0, 0.0, 1.0));
        let cfg = SolveConfig::default();
        let (dp, dr) = station_disagreement(&m, &truth, &obs, up, 4, &cfg).unwrap();
        assert!(dp < 1e-4 && dr < 1e-4, "stored = true: {} m, {} rad", dp, dr);
        // Station 1 was moved 30 cm and turned 5 deg since it was calibrated.
        let mut stored = truth.clone();
        stored[1].world_from_station = retract(&truth[1].world_from_station, &[0.2, -0.15, 0.15, 0.0, 0.087, 0.0]);
        let (dp, dr) = station_disagreement(&m, &stored, &obs, up, 4, &cfg).unwrap();
        assert!(dp > 0.05 || dr > 0.05, "moved: {} m, {} rad", dp, dr);
    }

    #[test]
    fn calibration_recovers_a_displaced_station() {
        let m = model();
        let truth = stations();
        let poses = [
            Pose { pos: Vec3::new(0.0, 0.0, 0.8), rot: Quat::from_rotvec(Vec3::new(0.0, 0.0, 0.2)) },
            Pose { pos: Vec3::new(0.5, 0.3, 0.8), rot: Quat::from_rotvec(Vec3::new(0.0, 0.0, 1.9)) },
            Pose { pos: Vec3::new(-0.4, 0.4, 1.1), rot: Quat::from_rotvec(Vec3::new(0.1, 0.0, 3.0)) },
            Pose { pos: Vec3::new(0.2, -0.5, 0.6), rot: Quat::from_rotvec(Vec3::new(0.0, -0.1, -1.2)) },
        ];
        let scenes: Vec<Scene> = poses
            .iter()
            .map(|p| Scene {
                obs: observe(&m, &truth, p),
                up_body: p.rot.conj().rotate(Vec3::new(0.0, 0.0, 1.0)),
                world_from_body: retract(p, &[0.01, -0.01, 0.02, 0.01, 0.0, -0.02]),
            })
            .collect();
        let mut init = truth.clone();
        init[1].world_from_station = retract(&truth[1].world_from_station, &[0.08, -0.05, 0.04, 0.03, -0.02, 0.05]);
        let up: Vec<Option<Vec3>> = truth.iter().map(|s| Some(s.world_from_station.rot.conj().rotate(Vec3::new(0.0, 0.0, 1.0)))).collect();
        let cal = calibrate(&m, &init, &up, &[None, None], &scenes, 0, &SolveConfig::default(), 2e-3).unwrap();
        let err = cal.stations[1].world_from_station.pos.sub(truth[1].world_from_station.pos).norm();
        assert!(err < 5e-5, "station 1 off by {} m", err);
        assert!(cal.rms[1] < 1e-6, "rms {}", cal.rms[1]);
    }
}

//! Geometry of trackers and stations and the sweep-angle measurement as a
//! function of both poses, with Jacobians.
//!
//! Frames: world (gravity-aligned, z up), station (looks down its -z, +y up
//! when upright), and the tracker body = its IMU frame (as in libsurvive;
//! sensor positions and the accelerometer live there). Poses map child to
//! parent: `world_from_body`, `world_from_station`.
//!
//! Perturbations are on the right for rotation and additive for position:
//! `R ← R·exp(δθ)`, `p ← p + δp`. Jacobians are rows over (δp, δθ).

use crate::lh1::{project, AxisCal};
use crate::linalg::{mat3_vec, rot_mat, skew, vec_mat3};
use crate::math::{Pose, Vec3};
use alloc::vec::Vec;

/// A tracked object's light sensors (body frame) and frames of interest.
#[derive(Clone, Debug, Default)]
pub struct TrackerModel {
    pub sensors: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    /// The device's reporting ("head") frame in the body frame; poses are
    /// published for it (libsurvive's convention).
    pub body_from_head: Pose,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StationModel {
    pub cal: [AxisCal; 2],
    pub world_from_station: Pose,
}

/// One sweep-angle measurement's prediction and gradients.
#[derive(Clone, Copy, Debug)]
pub struct AngleJacobian {
    pub angle: f64,
    /// d angle / d (δp, δθ) of the tracker's world_from_body.
    pub tracker: [f64; 6],
    /// d angle / d (δp, δθ) of the station's world_from_station.
    pub station: [f64; 6],
    /// The sensor in the station frame (for visibility checks).
    pub in_station: Vec3,
}

/// Predicted angle of `sensor` (body frame) seen by `station` on `axis`.
pub fn predict(station: &StationModel, world_from_body: &Pose, sensor: Vec3, axis: u8) -> AngleJacobian {
    let rb = rot_mat(world_from_body.rot);
    let rs = rot_mat(station.world_from_station.rot);
    let w = world_from_body.apply(sensor);
    let x = station.world_from_station.inverse().apply(w);
    let (angle, g) = project(&station.cal, x, axis);
    // g·Rsᵀ: gradient with respect to the world point.
    let gw = mat3_vec(&rs, g);
    // d/dθ_b: -g Rsᵀ Rb [s]x  =  -(gw Rb) [s]x
    let gwrb = vec_mat3(gw, &rb);
    let sx = skew(sensor);
    let dth_b = vec_mat3(gwrb, &sx).scale(-1.0);
    // Station: d/dt_s = -g Rsᵀ, d/dθ_s = g [x]x.
    let dth_s = vec_mat3(g, &skew(x));
    AngleJacobian {
        angle,
        tracker: [gw.x, gw.y, gw.z, dth_b.x, dth_b.y, dth_b.z],
        station: [-gw.x, -gw.y, -gw.z, dth_s.x, dth_s.y, dth_s.z],
        in_station: x,
    }
}

/// Applies a (δp, δθ) correction to a pose.
pub fn retract(p: &Pose, d: &[f64]) -> Pose {
    use crate::math::Quat;
    Pose {
        pos: p.pos.add(Vec3::new(d[0], d[1], d[2])),
        rot: p.rot.mul(Quat::from_rotvec(Vec3::new(d[3], d[4], d[5]))).normalized(),
    }
}

/// A sensor faces the station if its normal points at it (plus margin).
pub fn faces(world_from_body: &Pose, sensor: Vec3, normal: Vec3, station: &StationModel) -> bool {
    let w = world_from_body.apply(sensor);
    let n = world_from_body.rot.rotate(normal);
    let to_station = station.world_from_station.pos.sub(w).normalized();
    n.dot(to_station) > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Quat;

    fn station() -> StationModel {
        StationModel {
            cal: [
                AxisCal { phase: 0.02, tilt: -0.004, curve: 0.001, gibphase: 2.7, gibmag: 0.006 },
                AxisCal { phase: 0.03, tilt: 0.003, curve: -0.0006, gibphase: -1.2, gibmag: 0.002 },
            ],
            world_from_station: Pose {
                pos: Vec3::new(0.9, -0.3, 2.0),
                rot: Quat::from_rotvec(Vec3::new(0.9, 0.3, -0.4)),
            },
        }
    }

    #[test]
    fn jacobians_match_finite_differences() {
        let st = station();
        let body = Pose { pos: Vec3::new(0.1, 0.4, 0.8), rot: Quat::from_rotvec(Vec3::new(-0.3, 0.5, 2.0)) };
        let s = Vec3::new(0.03, -0.02, 0.01);
        for axis in 0..2u8 {
            let j = predict(&st, &body, s, axis);
            let h = 1e-6;
            for k in 0..6 {
                let mut d = [0.0; 6];
                d[k] = h;
                let up = predict(&st, &retract(&body, &d), s, axis).angle;
                d[k] = -h;
                let dn = predict(&st, &retract(&body, &d), s, axis).angle;
                let num = (up - dn) / (2.0 * h);
                assert!((num - j.tracker[k]).abs() < 1e-6, "tracker axis {} k {}: {} vs {}", axis, k, num, j.tracker[k]);

                let mut d = [0.0; 6];
                d[k] = h;
                let mut st2 = st.clone();
                st2.world_from_station = retract(&st.world_from_station, &d);
                let up = predict(&st2, &body, s, axis).angle;
                d[k] = -h;
                st2.world_from_station = retract(&st.world_from_station, &d);
                let dn = predict(&st2, &body, s, axis).angle;
                let num = (up - dn) / (2.0 * h);
                assert!((num - j.station[k]).abs() < 1e-6, "station axis {} k {}: {} vs {}", axis, k, num, j.station[k]);
            }
        }
    }
}

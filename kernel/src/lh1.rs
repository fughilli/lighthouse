//! Lighthouse 1.0 base stations: timing constants, the sync-pulse code, and the
//! sweep-angle measurement model with the factory (OOTX) corrections.
//!
//! Timing (48 MHz device ticks): a station spins its two rotors at 60 Hz and
//! alternates them, so every 8.33 ms cycle (400 000 ticks) holds two sync
//! slots 20 000 ticks apart (the master — mode b, or A with a cable — first)
//! and then one sweep by whichever station's sync said it isn't skipping. A
//! sync's width encodes three bits: skip, OOTX data, and the axis of the coming
//! sweep. The sweep angle is the time from the sweeping station's sync to the
//! hit: `angle = ticks * pi / 400 000 - pi / 2` (0 straight ahead, ±60° FOV).

use crate::math::{asin, atan2, cos, sin, sqrt, Vec3};
use core::f64::consts::{FRAC_PI_2, PI};

/// Device clock.
pub const TICKS_PER_SECOND: f64 = 48_000_000.0;
/// One sync-sync-sweep cycle (half a rotor turn).
pub const CYCLE_TICKS: u32 = 400_000;
/// Spacing of the two sync slots in a cycle.
pub const SLOT_TICKS: u32 = 20_000;
/// Sync flash widths: 62.5 us (code 0) to 135 us (code 7) in 10.4 us steps.
pub const SYNC_MIN_TICKS: u16 = 2_250;
pub const SYNC_MAX_TICKS: u16 = 6_750;

/// The three bits a sync flash's width carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyncCode {
    /// This station won't sweep in this cycle.
    pub skip: bool,
    /// One OOTX bit.
    pub data: bool,
    /// Axis of the sweep that follows (0 = horizontal sweep "x", 1 = "y").
    pub axis: u8,
}

impl SyncCode {
    pub fn bits(self) -> u8 {
        ((self.skip as u8) << 2) | ((self.data as u8) << 1) | self.axis
    }

    pub fn from_bits(b: u8) -> SyncCode {
        SyncCode { skip: b & 4 != 0, data: b & 2 != 0, axis: b & 1 }
    }

    /// Nominal width in ticks: 2 750 + 500 * code.
    pub fn nominal_ticks(self) -> u32 {
        2_750 + 500 * self.bits() as u32
    }

    /// Nearest code to a measured width, and how far off it is (ticks).
    pub fn decode(len: u32) -> (SyncCode, u32) {
        let c = ((len as i64 - 2_750 + 250).div_euclid(500)).clamp(0, 7) as u8;
        let code = SyncCode::from_bits(c);
        (code, (len as i64 - code.nominal_ticks() as i64).unsigned_abs() as u32)
    }
}

/// Per-axis factory calibration broadcast in OOTX ("fcal").
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AxisCal {
    pub phase: f64,
    pub tilt: f64,
    pub curve: f64,
    pub gibphase: f64,
    pub gibmag: f64,
}

pub fn ticks_to_angle(ticks: f64) -> f64 {
    ticks * (PI / CYCLE_TICKS as f64) - FRAC_PI_2
}

/// One axis of libsurvive's model (survive_reproject_axis), with its gradient
/// in (axis_value, other, z) where z = -p_lh.z (the station looks down -z).
fn axis_model(cal: &AxisCal, a: f64, o: f64, z: f64, invert: bool) -> (f64, [f64; 3]) {
    let s = if invert { -1.0 } else { 1.0 };
    let r2 = a * a + z * z;
    let mag = sqrt(r2);
    let ang0 = FRAC_PI_2 - s * atan2(a, z);
    let d_ang0 = [-s * z / r2, 0.0, s * a / r2];

    let u = (cal.tilt * o / mag).clamp(-1.0, 1.0);
    let du = [-cal.tilt * o * a / (mag * r2), cal.tilt / mag, -cal.tilt * o * z / (mag * r2)];
    let k = 1.0 / sqrt((1.0 - u * u).max(1e-12));
    let ang1 = ang0 - cal.phase - asin(u);
    let d_ang1 = [d_ang0[0] - k * du[0], d_ang0[1] - k * du[1], d_ang0[2] - k * du[2]];

    let g = 1.0 + sin(cal.gibphase + ang1) * cal.gibmag;
    let ang2 = ang1 - cos(cal.gibphase + ang1) * cal.gibmag;

    let oz2 = o * o + z * z;
    let c = atan2(o, z);
    let ang = ang2 + cal.curve * c * c - FRAC_PI_2;
    let dc = [0.0, 2.0 * cal.curve * c * z / oz2, -2.0 * cal.curve * c * o / oz2];
    (ang, [d_ang1[0] * g + dc[0], d_ang1[1] * g + dc[1], d_ang1[2] * g + dc[2]])
}

/// Predicted sweep angle of a point given in the station's frame, and its
/// gradient with respect to that point.
pub fn project(cal: &[AxisCal; 2], p: Vec3, axis: u8) -> (f64, Vec3) {
    if axis == 0 {
        let (ang, g) = axis_model(&cal[0], p.x, p.y, -p.z, false);
        (ang, Vec3::new(g[0], g[1], -g[2]))
    } else {
        let (ang, g) = axis_model(&cal[1], p.y, p.x, -p.z, true);
        (ang, Vec3::new(g[1], g[0], -g[2]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cal() -> [AxisCal; 2] {
        [
            AxisCal { phase: 0.0275, tilt: -0.0047, curve: 0.0011, gibphase: 2.74, gibmag: 0.0059 },
            AxisCal { phase: 0.0319, tilt: -0.0069, curve: 0.0010, gibphase: 0.052, gibmag: -0.0085 },
        ]
    }

    #[test]
    fn code_round_trip() {
        for b in 0..8u8 {
            let c = SyncCode::from_bits(b);
            let (d, err) = SyncCode::decode(c.nominal_ticks() + 180);
            assert_eq!(d, c);
            assert_eq!(err, 180);
        }
    }

    #[test]
    fn straight_ahead_is_minus_phase() {
        let c = [AxisCal::default(); 2];
        let (a, _) = project(&c, Vec3::new(0.0, 0.0, -1.0), 0);
        assert!(a.abs() < 1e-12);
        // +x in the station frame is hit earlier on axis 0 (negative angle),
        // matching libsurvive's convention.
        let (a, _) = project(&c, Vec3::new(0.1, 0.0, -1.0), 0);
        assert!((a + libm::atan2(0.1, 1.0)).abs() < 1e-12);
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let c = cal();
        let p = Vec3::new(0.31, -0.22, -1.4);
        for axis in 0..2u8 {
            let (_, g) = project(&c, p, axis);
            let h = 1e-7;
            let num = [
                (project(&c, p.add(Vec3::new(h, 0.0, 0.0)), axis).0 - project(&c, p.sub(Vec3::new(h, 0.0, 0.0)), axis).0) / (2.0 * h),
                (project(&c, p.add(Vec3::new(0.0, h, 0.0)), axis).0 - project(&c, p.sub(Vec3::new(0.0, h, 0.0)), axis).0) / (2.0 * h),
                (project(&c, p.add(Vec3::new(0.0, 0.0, h)), axis).0 - project(&c, p.sub(Vec3::new(0.0, 0.0, h)), axis).0) / (2.0 * h),
            ];
            let an = g.to_array();
            for i in 0..3 {
                assert!((an[i] - num[i]).abs() < 1e-6, "axis {} d{}: {} vs {}", axis, i, an[i], num[i]);
            }
        }
    }
}

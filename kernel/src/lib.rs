//! Lighthouse tracking kernel: Vive base-station light and IMU in, poses of
//! the base stations and trackers in a gravity-aligned world frame (with
//! covariances) and an event stream out.
//!
//! The kernel does no I/O. A transport delivers device reports, the protocol
//! layer turns them into [`Input`]s, and everything from there on happens
//! here:
//!
//!   light pulses ─► [`timing`] ─► sweep hits (station, axis, sensor, angle)
//!                       │
//!                       └─► sync data bits ─► [`ootx`] ─► station calibration
//!
//! Constraints: `no_std` + `alloc`, math through `libm`, so the same code can
//! run on a host or an embedded target. The scalar is [`Real`]: f64 by default,
//! f32 with the `f32` feature (for FPUs without doubles, like the ESP32-P4's).
//! Time stays in integer device ticks, and anything else that needs range
//! rather than fractions stays integer too.
#![cfg_attr(not(test), no_std)]

extern crate alloc;

#[cfg(feature = "f32")]
pub type Real = f32;
#[cfg(not(feature = "f32"))]
pub type Real = f64;

/// Constants of [`Real`].
pub mod consts {
    #[cfg(feature = "f32")]
    pub use core::f32::consts::*;
    #[cfg(not(feature = "f32"))]
    pub use core::f64::consts::*;
}

pub mod kernel;
pub mod lh1;
pub mod linalg;
pub mod math;
pub mod model;
pub mod solve;
pub mod ootx;
pub mod timing;
pub mod track;

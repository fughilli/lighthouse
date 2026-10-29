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
//! Constraints: `no_std` + `alloc`, f64 math through `libm`, so the same code
//! can run on a host or an embedded target.
#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod kernel;
pub mod lh1;
pub mod linalg;
pub mod math;
pub mod model;
pub mod solve;
pub mod ootx;
pub mod timing;
pub mod track;

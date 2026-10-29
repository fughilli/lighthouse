//! Vive / Watchman device protocol: report bytes in, decoded samples out, and
//! the command bytes to send. Pure (no I/O); the transport crate moves bytes.
//!
//! The byte formats follow libsurvive (see docs/lighthouse-protocol.md, which
//! cites the C for every layout); decoding was validated against libsurvive's
//! output on the recorded sessions (tools/lh_protocheck).
//!
//! `no_std` + `alloc` (the config readout inflates with miniz_oxide and parses
//! with serde_json's alloc build), so it runs on a microcontroller host too.
#![cfg_attr(not(test), no_std)]

extern crate alloc;

pub mod commands;
pub mod config;
pub mod devices;
pub mod imu;
pub mod rf;
pub mod usb;

/// One light pulse on one sensor (raw sensor id; map through the config's
/// channel map before use).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Lightcap {
    pub sensor: u8,
    /// Rising edge, device ticks (48 MHz).
    pub timestamp: u32,
    /// Ticks.
    pub length: u16,
}

/// One raw IMU sample (counts; scale with [`imu::ImuCalibration`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawImu {
    pub timecode: u32,
    pub accel: [i16; 3],
    pub gyro: [i16; 3],
}

/// Everything a device's input reports can carry.
#[derive(Clone, Debug, PartialEq)]
pub enum Sample {
    Imu(RawImu),
    Light(Lightcap),
    Battery { percent: u8, charging: bool },
    Buttons { buttons: u32 },
    /// The dongle's tracker went off (RF report 0x26).
    TurnedOff,
    /// A report or sub-packet that couldn't be decoded (kept for diagnostics).
    Malformed(&'static str),
}

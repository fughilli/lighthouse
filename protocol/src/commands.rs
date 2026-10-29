//! Feature reports to send. Command reports are `ff <cmd> <len> <data…>`,
//! padded to 64 bytes for the dongle.

use alloc::vec;
use alloc::vec::Vec;

use crate::devices::DeviceKind;

pub const REPORT_LEN: usize = 64;

fn padded(bytes: &[u8]) -> Vec<u8> {
    let mut v = bytes.to_vec();
    v.resize(REPORT_LEN, 0);
    v
}

/// Startup "magics" per device kind, in the order libsurvive sends them.
pub fn startup(kind: DeviceKind) -> Vec<Vec<u8>> {
    match kind {
        DeviceKind::Dongle => vec![
            // CONFIGURE_RADIO
            padded(&[0xff, 0x96, 0x10, 0xbe, 0x5b, 0x32, 0x54, 0x11, 0xcf, 0x83, 0x75, 0x53, 0x8a, 0x08, 0x6a, 0x53, 0x58, 0xd0, 0xb1]),
            // CHANGE_PROTOCOL: lightcap raw0 (LH1)
            padded(&[0xff, 0x87, 0x06, 0x01, 0x00, 0x00, 0x02, 0x00, 0x00]),
            // CHANGE_PROTOCOL: protocol switch (starts 0x23/0x24 traffic)
            padded(&[0xff, 0x87, 0x03, 0x00, 0x01, 0x00]),
        ],
        DeviceKind::TrackerWired | DeviceKind::Tracker2017Wired => vec![
            // Change mode: raw0 (LH1). On macOS the first one after idle resets the
            // tracker (it re-enumerates); send again after reconnecting.
            vec![0x04, 0x00, 0x00, 0x00, 0x00],
            // Lightcap verbosity 3.
            vec![0x07, 0x03, 0x00, 0x00, 0x00],
        ],
    }
}

/// Put a dongle into pairing mode (~10 s); the tracker must be in pairing mode
/// too (hold its button until the LED blinks blue).
pub fn pair() -> Vec<u8> {
    padded(&[0xff, 0xad, 0x03, 0x01, 0x10, 0x27])
}

/// Haptic pulse: `frequency` Hz, `amplitude` 0..1, `duration` s.
pub fn haptic(frequency: f64, amplitude: f64, duration: f64) -> Vec<u8> {
    let period_us = 1e6 / frequency.max(1.0);
    let lo = (amplitude * period_us / 2.0).max(1.0).min(65535.0) as u16;
    let hi = (period_us - lo as f64).max(0.0).min(65535.0) as u16;
    let repeat = (duration * frequency).max(1.0).min(65535.0) as u16;
    let mut v = vec![0xff, 0x8f, 0x07, 0x00];
    v.extend_from_slice(&hi.to_le_bytes());
    v.extend_from_slice(&lo.to_le_bytes());
    v.extend_from_slice(&repeat.to_le_bytes());
    padded(&v)
}

/// Power the tracker off (as SteamVR does: "off!").
pub fn power_off() -> Vec<u8> {
    vec![0x9f, 0x04, 0x6f, 0x66, 0x66, 0x21]
}

/// Feature report ids read during startup.
pub const GET_CONFIG_START: u8 = 0x10;
pub const GET_CONFIG_CHUNK: u8 = 0x11;
pub const GET_VERSION: u8 = 0x05;
pub const GET_IMU_SCALES: u8 = 0x01;

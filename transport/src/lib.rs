//! Moving bytes to and from lighthouse devices. Everything above this (the
//! protocol decoding, the kernel) is pure; a [`Transport`] is the one place
//! with I/O, so the same daemon runs on live hardware or a recording.
//!
//! Backends:
//! - [`hid::HidTransport`]: live devices through hidapi (one reader thread per
//!   interface; periodic enumeration for hotplug, which also covers a wired
//!   tracker re-enumerating after its first mode command on macOS).
//! - [`replay::ReplayTransport`]: a recorded session (`.rec.gz` with HIDIN /
//!   HIDFEAT lines), answering feature requests from the recorded responses.

pub mod hid;
pub mod replay;

use std::time::Duration;

/// Stable key for a device: USB product id plus serial number.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeviceKey {
    pub pid: u16,
    pub serial: String,
}

#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub key: DeviceKey,
    pub vid: u16,
    /// HID interface numbers present.
    pub interfaces: Vec<i32>,
    /// A short name for logs and recordings (WM0, T20, …).
    pub name: String,
}

#[derive(Clone, Debug)]
pub enum TransportEvent {
    Added(DeviceInfo),
    Removed(DeviceKey),
    Report {
        key: DeviceKey,
        interface: i32,
        /// Host receive time, µs (monotonic for live; recorded for replay).
        host_us: u64,
        data: Vec<u8>,
    },
}

#[derive(Debug)]
pub enum TransportError {
    NoDevice,
    Io(String),
    /// Replay: no recorded response for this request.
    NotRecorded,
}

pub trait Transport {
    /// Waits up to `timeout` for reports or device changes.
    fn poll(&mut self, timeout: Duration) -> Vec<TransportEvent>;
    /// GET_FEATURE `report_id` on interface 0; returns the buffer with the
    /// report id first.
    fn get_feature(&mut self, key: &DeviceKey, report_id: u8, len: usize) -> Result<Vec<u8>, TransportError>;
    /// SET_FEATURE on interface 0 (`data[0]` is the report id).
    fn set_feature(&mut self, key: &DeviceKey, data: &[u8]) -> Result<usize, TransportError>;
    /// True once a finite source (a recording) has nothing more to deliver.
    fn finished(&self) -> bool {
        false
    }
}

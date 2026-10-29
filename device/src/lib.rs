//! One Vive device (a Watchman dongle or a wired tracker) over an abstract HID
//! [`Link`]: the handshake that reads its config and starts its report stream,
//! and a [`Decoder`] that turns its input reports into kernel inputs.
//!
//! The host supplies the link: lhd over hidapi (and its recorder), the ESP32-P4
//! player over the ESP-IDF USB host. Everything here is `no_std` + `alloc`.
#![cfg_attr(not(test), no_std)]

extern crate alloc;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use lh_kernel::kernel::Input;
use lh_kernel::math::Vec3;
use lh_kernel::timing::Pulse;
use lh_protocol::config::{ConfigReader, DeviceConfig, ReadStep};
use lh_protocol::devices::{DeviceType, Interface};
use lh_protocol::imu::ImuCalibration;
use lh_protocol::rf::RfDecoder;
use lh_protocol::usb::{self, UsbImu};
use lh_protocol::{commands, Sample};

#[derive(Debug)]
pub enum LinkError {
    /// The device didn't answer (yet): a dongle NAKs until its tracker replied.
    Nak,
    /// The device is gone.
    Gone,
    /// Replay: no recorded answer. Don't retry.
    NotRecorded,
    Other(String),
}

/// The control side of one device's HID link (interface 0).
pub trait Link {
    /// GET_FEATURE `report_id` (`len` bytes); the reply starts with the report id.
    fn get_feature(&mut self, report_id: u8, len: usize) -> Result<Vec<u8>, LinkError>;
    /// SET_FEATURE; `data` starts with the report id.
    fn set_feature(&mut self, data: &[u8]) -> bool;
    fn delay_ms(&mut self, ms: u32);
}

/// Retry a GET a few times: a dongle NAKs until it has the answer from its
/// tracker (~20 ms).
fn get(link: &mut dyn Link, id: u8) -> Result<Vec<u8>, LinkError> {
    let mut last = Err(LinkError::Nak);
    for _ in 0..5 {
        last = link.get_feature(id, commands::REPORT_LEN);
        match &last {
            Ok(_) | Err(LinkError::NotRecorded) | Err(LinkError::Gone) => return last,
            Err(_) => link.delay_ms(20),
        }
    }
    last
}

/// Reads the device config and starts the report stream. A dongle answers only
/// while its tracker is connected, so the caller retries on error.
pub fn handshake(link: &mut dyn Link, kind: &DeviceType) -> Result<(DeviceConfig, ImuCalibration), String> {
    let mut reader = ConfigReader::default();
    let first = get(link, commands::GET_CONFIG_START).map_err(|e| format!("config start: {:?}", e))?;
    if let ReadStep::Restart(why) = reader.start(&first) {
        return Err(why.into());
    }
    let mut waits = 0;
    let json = loop {
        let chunk = get(link, commands::GET_CONFIG_CHUNK).map_err(|e| format!("config chunk: {:?}", e))?;
        match reader.chunk(&chunk) {
            ReadStep::More => continue,
            ReadStep::Done(j) => break j,
            ReadStep::Restart(why) => return Err(why.into()),
            ReadStep::Short { got, expected } => {
                waits += 1;
                if waits > 200 {
                    return Err(format!("config stuck at {} of {} bytes", got, expected));
                }
                link.delay_ms(5);
            }
        }
    };
    let config = lh_protocol::config::parse(&json)?;
    for m in commands::startup(kind.kind) {
        link.set_feature(&m);
    }
    let _ = get(link, commands::GET_VERSION);
    let mut imu = config.imu;
    if let Ok(s) = get(link, commands::GET_IMU_SCALES) {
        if s.len() >= 3 {
            imu.set_scale_modes(s[1], s[2]);
        }
    }
    Ok((config, imu))
}

/// What one input report carried, for the kernel and the host.
#[derive(Clone, Debug)]
pub enum Output {
    /// Feed to the kernel (with this device's kernel id filled in).
    Kernel(Input),
    Battery { percent: u8, charging: bool },
    Buttons(u32),
    /// The dongle's tracker went off (RF report 0x26).
    TurnedOff,
}

/// Decodes one configured device's input reports.
pub struct Decoder {
    pub kind: &'static DeviceType,
    pub config: DeviceConfig,
    pub imu_cal: ImuCalibration,
    /// Kernel device id the inputs are tagged with.
    pub dev: u32,
    rf: RfDecoder,
    usb_imu: UsbImu,
    samples: Vec<Sample>,
}

impl Decoder {
    pub fn new(kind: &'static DeviceType, config: DeviceConfig, imu_cal: ImuCalibration, dev: u32) -> Decoder {
        Decoder { kind, config, imu_cal, dev, rf: RfDecoder::default(), usb_imu: UsbImu::default(), samples: Vec::new() }
    }

    /// The kernel input that announces this device.
    pub fn added(&self) -> Input {
        Input::DeviceAdded { dev: self.dev, model: self.config.model.clone(), wireless: self.kind.wireless() }
    }

    /// One input report from HID interface `interface`.
    pub fn report(&mut self, interface: i32, data: &[u8], out: &mut Vec<Output>) {
        self.samples.clear();
        match self.kind.interface(interface) {
            Some(Interface::Watchman) => self.rf.decode(data, &mut self.samples),
            Some(Interface::Imu) => self.usb_imu.decode(data, &mut self.samples),
            Some(Interface::Lightcap) => usb::decode_lightcap(data, &mut self.samples),
            Some(Interface::Buttons) => usb::decode_buttons(data, &mut self.samples),
            None => {}
        }
        for s in self.samples.drain(..) {
            match s {
                Sample::Imu(i) => {
                    let (a, g) = self.imu_cal.apply(&i);
                    out.push(Output::Kernel(Input::Imu {
                        dev: self.dev,
                        time: i.timecode,
                        accel_g: Vec3::new(a[0], a[1], a[2]),
                        gyro: Vec3::new(g[0], g[1], g[2]),
                    }));
                }
                Sample::Light(l) => {
                    if let Some(sensor) = self.config.channel_map.get(l.sensor as usize).copied().flatten() {
                        out.push(Output::Kernel(Input::Pulse {
                            dev: self.dev,
                            pulse: Pulse { sensor, start: l.timestamp, len: l.length as u32 },
                        }));
                    }
                }
                Sample::Battery { percent, charging } => out.push(Output::Battery { percent, charging }),
                Sample::Buttons { buttons } => out.push(Output::Buttons(buttons)),
                Sample::TurnedOff => out.push(Output::TurnedOff),
                Sample::Malformed(_) => {}
            }
        }
    }
}

//! Per-device state: handshake (config readout, startup magics, version, IMU
//! scales), then streaming reports into the kernel. A dongle answers the
//! config read only while a tracker is connected to it, so an unconfigured
//! dongle is retried; a configured one that goes quiet counts as its tracker
//! disconnecting.

use crate::record::Recorder;
use lh_device::{Decoder, Link, LinkError, Output};
use lh_kernel::kernel::{Event, Input, Kernel};
use lh_protocol::commands;
use lh_protocol::devices::{self, DeviceKind, DeviceType, Interface};
use lh_transport::{DeviceInfo, Transport, TransportError, TransportEvent};
use serde_json::{json, Value};
use std::time::Duration;

/// A status line for stdout.
pub struct Note(Value);

impl Note {
    pub fn to_json(&self) -> Value {
        self.0.clone()
    }
}

enum State {
    Unconfigured { next_try: Duration },
    Streaming,
}

struct Dev {
    info: DeviceInfo,
    kind: &'static DeviceType,
    state: State,
    /// Set once the handshake succeeded (config, IMU scales, kernel id).
    dec: Option<Decoder>,
    last_report: Duration,
    last_imu_note: f64,
    /// Last button mask reported (the tracker repeats states).
    buttons: Option<u32>,
    last_error: Option<String>,
}

pub struct DeviceManager {
    devs: Vec<Dev>,
    next_kernel_id: u32,
    now: Duration,
}

/// A device's control link through the transport, recording every feature
/// request (a failed GET is recorded as -1, like libsurvive's recordings).
struct HidLink<'a> {
    t: &'a mut dyn Transport,
    dev: &'a Dev,
    rec: &'a mut Recorder,
}

impl Link for HidLink<'_> {
    fn get_feature(&mut self, id: u8, len: usize) -> Result<Vec<u8>, LinkError> {
        match self.t.get_feature(&self.dev.info.key, id, len) {
            Ok(b) => {
                self.rec.feature(&self.dev.info.name, "GET", b.len() as i64, &b);
                Ok(b)
            }
            Err(TransportError::NotRecorded) => Err(LinkError::NotRecorded),
            Err(e) => {
                self.rec.feature(&self.dev.info.name, "GET", -1, &[id]);
                Err(match e {
                    TransportError::NoDevice => LinkError::Gone,
                    TransportError::Io(m) => LinkError::Other(m),
                    TransportError::NotRecorded => LinkError::NotRecorded,
                })
            }
        }
    }

    fn set_feature(&mut self, data: &[u8]) -> bool {
        set(self.t, self.dev, data, self.rec)
    }

    fn delay_ms(&mut self, ms: u32) {
        std::thread::sleep(Duration::from_millis(ms as u64));
    }
}

fn set(t: &mut dyn Transport, dev: &Dev, data: &[u8], rec: &mut Recorder) -> bool {
    let r = t.set_feature(&dev.info.key, data);
    rec.feature(&dev.info.name, "SET", r.as_ref().map(|n| *n as i64).unwrap_or(-1), data);
    r.is_ok()
}

impl DeviceManager {
    pub fn new() -> DeviceManager {
        DeviceManager { devs: Vec::new(), next_kernel_id: 0, now: Duration::ZERO }
    }

    /// Set the clock before handling a batch. A handshake blocks the loop for a
    /// second or two while other dongles' reports queue up; stamped with the
    /// previous tick's time, those would look like their trackers going quiet.
    pub fn set_now(&mut self, now: Duration) {
        self.now = now;
    }

    pub fn serial_of(&self, kernel_id: u32) -> Option<String> {
        self.devs.iter().filter_map(|d| d.dec.as_ref()).find(|x| x.dev == kernel_id).map(|x| x.config.serial.clone())
    }

    pub fn handle(
        &mut self,
        ev: TransportEvent,
        t: &mut dyn Transport,
        kernel: &mut Kernel,
        events: &mut Vec<Event>,
        rec: &mut Recorder,
        notes: &mut Vec<Note>,
    ) {
        match ev {
            TransportEvent::Added(info) => {
                let Some(kind) = devices::lookup(info.vid, info.key.pid) else { return };
                notes.push(Note(json!({"type":"usb","name":info.name,"product":kind.name,"state":"added"})));
                self.devs.push(Dev {
                    info,
                    kind,
                    state: State::Unconfigured { next_try: self.now },
                    dec: None,
                    last_report: self.now,
                    last_imu_note: 0.0,
                    buttons: None,
                    last_error: None,
                });
                // Handshake right away (a recording's responses come first).
                self.tick(t, kernel, events, rec, notes, self.now);
            }
            TransportEvent::Removed(key) => {
                if let Some(i) = self.devs.iter().position(|d| d.info.key == key) {
                    let d = self.devs.remove(i);
                    if let Some(id) = d.dec.as_ref().map(|x| x.dev) {
                        kernel.push(Input::DeviceRemoved { dev: id }, events);
                    }
                    notes.push(Note(json!({"type":"usb","name":d.info.name,"state":"removed"})));
                }
            }
            TransportEvent::Report { key, interface, host_us, data } => {
                let now = self.now;
                let Some(d) = self.devs.iter_mut().find(|d| d.info.key == key) else { return };
                let role = d.kind.interface(interface);
                rec.report(&d.info.name, iface_name(role), host_us, &data);
                d.last_report = now;
                if !matches!(d.state, State::Streaming) {
                    return;
                }
                let Some(dec) = d.dec.as_mut() else { return };
                let mut outs = Vec::new();
                dec.report(interface, &data, &mut outs);
                let serial = dec.config.serial.clone();
                let head_rot = dec.config.model.body_from_head.rot.conj();
                for o in outs {
                    match o {
                        Output::Kernel(input) => {
                            if let Input::Imu { time, accel_g, gyro, .. } = &input {
                                // The viewer shows the accel vector; ~30 Hz is plenty.
                                let ts = *time as f64 / 48e6;
                                if (ts - d.last_imu_note).abs() > 0.033 {
                                    d.last_imu_note = ts;
                                    // In the head frame, like poses.
                                    let ah = head_rot.rotate(*accel_g).to_array();
                                    let gh = head_rot.rotate(*gyro).to_array();
                                    notes.push(Note(json!({"type":"imu","serial":serial,"acc":ah,"gyro":gh})));
                                }
                            }
                            kernel.push(input, events);
                        }
                        Output::Battery { percent, charging } => {
                            notes.push(Note(json!({"type":"battery","serial":serial,"percent":percent,"charging":charging})))
                        }
                        Output::Buttons(buttons) => {
                            if d.buttons != Some(buttons) {
                                d.buttons = Some(buttons);
                                notes.push(Note(json!({"type":"buttons","serial":serial,"buttons":buttons})));
                            }
                        }
                        Output::TurnedOff => d.last_report = Duration::ZERO,
                    }
                }
            }
        }
    }

    /// Retries handshakes and notices wireless trackers going quiet.
    pub fn tick(
        &mut self,
        t: &mut dyn Transport,
        kernel: &mut Kernel,
        events: &mut Vec<Event>,
        rec: &mut Recorder,
        notes: &mut Vec<Note>,
        now: Duration,
    ) {
        self.now = now;
        for i in 0..self.devs.len() {
            let quiet = now.saturating_sub(self.devs[i].last_report);
            match self.devs[i].state {
                State::Unconfigured { next_try } if now >= next_try => match lh_device::handshake(
                    &mut HidLink { t: &mut *t, dev: &self.devs[i], rec: &mut *rec },
                    self.devs[i].kind,
                ) {
                    Ok((config, imu)) => {
                        let id = self.next_kernel_id;
                        self.next_kernel_id += 1;
                        let d = &mut self.devs[i];
                        // Sensors in the head frame (the frame poses are reported in), for viewers.
                        let head_from_body = config.model.body_from_head.inverse();
                        let sensors_head: Vec<[f64; 3]> = config.model.sensors.iter().map(|p| head_from_body.apply(*p).to_array()).collect();
                        let normals_head: Vec<[f64; 3]> =
                            config.model.normals.iter().map(|n| head_from_body.rot.rotate(*n).to_array()).collect();
                        notes.push(Note(json!({"type":"device","name":d.info.name,"serial":config.serial,
                            "model":config.model_number,"sensors":config.model.sensors.len(),"state":"connected",
                            "sensors_head":sensors_head,"normals_head":normals_head})));
                        let dec = Decoder::new(d.kind, config, imu, id);
                        kernel.push(dec.added(), events);
                        d.dec = Some(dec);
                        d.state = State::Streaming;
                        d.last_report = now;
                    }
                    Err(why) => {
                        // Logged when it changes (a dongle without a tracker fails every 0.5 s).
                        let d = &mut self.devs[i];
                        if d.last_error.as_deref() != Some(why.as_str()) {
                            notes.push(Note(json!({"type":"log","msg":format!("{}: handshake: {}", d.info.name, why)})));
                            d.last_error = Some(why);
                        }
                        d.state = State::Unconfigured { next_try: now + Duration::from_millis(500) };
                    }
                },
                State::Streaming if self.devs[i].kind.wireless() && quiet > Duration::from_secs(1) => {
                    let d = &mut self.devs[i];
                    let dec = d.dec.take();
                    if let Some(x) = &dec {
                        kernel.push(Input::DeviceRemoved { dev: x.dev }, events);
                    }
                    let serial = dec.map(|x| x.config.serial).unwrap_or_default();
                    notes.push(Note(json!({"type":"device","name":d.info.name,"serial":serial,"state":"disconnected"})));
                    d.state = State::Unconfigured { next_try: now + Duration::from_millis(500) };
                }
                _ => {}
            }
        }
    }

    pub fn pair_all(&mut self, t: &mut dyn Transport, rec: &mut Recorder, notes: &mut Vec<Note>) {
        for d in &self.devs {
            if d.kind.kind == DeviceKind::Dongle {
                let ok = set(t, d, &commands::pair(), rec);
                notes.push(Note(json!({"type":"pairing","name":d.info.name,"sent":ok,
                    "hint":"hold the tracker's button until its LED blinks blue"})));
            }
        }
    }

    pub fn haptic(&mut self, t: &mut dyn Transport, serial: &str, rec: &mut Recorder) {
        for d in &self.devs {
            if d.dec.as_ref().map_or(false, |x| x.config.serial == serial || serial.is_empty()) {
                set(t, d, &commands::haptic(200.0, 0.5, 0.3), rec);
            }
        }
    }
}

/// libsurvive's interface names, so recordings stay compatible.
fn iface_name(role: Option<Interface>) -> &'static str {
    match role {
        Some(Interface::Watchman) => "IMU/Lightcap/Buttons",
        Some(Interface::Imu) => "IMU",
        Some(Interface::Lightcap) => "Lightcap",
        Some(Interface::Buttons) => "Buttons",
        None => "Unknown",
    }
}

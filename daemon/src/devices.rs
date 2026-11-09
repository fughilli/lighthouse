//! Per-device state: handshake (config readout, startup magics, version, IMU
//! scales), then streaming reports into the kernel. A dongle answers the
//! config read only while a tracker is connected to it, so an unconfigured
//! dongle is retried; a configured one that goes quiet counts as its tracker
//! disconnecting.

use crate::record::Recorder;
use lh_device::clock::HostClock;
use lh_device::{Decoder, Link, LinkError, Output};
use lh_kernel::kernel::{Event, Input, Kernel};
use lh_kernel::Real;
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
    /// --host-clock: the device's clock on the report timestamps, and the last host time
    /// told to the kernel (Input::Clock).
    clock: HostClock,
    clock_us: i64,
    /// --pose-timing: report lateness over the fastest, 0.5 ms bins, [IMU-only, with light].
    late_hist: [[u32; 16]; 2],
}

pub struct DeviceManager {
    devs: Vec<Dev>,
    next_kernel_id: u32,
    /// Kernel device id per tracker serial: a tracker that drops off the radio and comes
    /// back keeps its id, so the kernel can still use the still scenes it collected
    /// before (calibration needs their device's sensor model).
    kernel_ids: Vec<(String, u32)>,
    now: Duration,
    /// --host-clock: map each device's clock onto the report timestamps and tick the kernel
    /// with it (1 ms of host time apart, like the P4), so a sweep goes out once the laser
    /// has passed; each pose then also gets a "pose_timing" note on the host clock.
    pub host_clock: bool,
    /// --pose-timing: the "pose_timing" notes alone (no clock ticks), to compare.
    pub pose_timing: bool,
    /// --stream-sweeps (with --host-clock): the kernel streams sweeps, so the clock it gets
    /// is the mapped time itself, not less the light margin.
    pub stream: bool,
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
        DeviceManager { devs: Vec::new(), next_kernel_id: 0, kernel_ids: Vec::new(), now: Duration::ZERO, host_clock: false, pose_timing: false, stream: false }
    }

    /// Set the clock before handling a batch. A handshake blocks the loop for a
    /// second or two while other dongles' reports queue up; stamped with the
    /// previous tick's time, those would look like their trackers going quiet.
    pub fn set_now(&mut self, now: Duration) {
        self.now = now;
    }

    /// The kernel id for a tracker serial: the one it had before, else a new one. Two
    /// devices streaming at once never share one (an empty serial always gets a new id).
    fn kernel_id(&mut self, serial: &str) -> u32 {
        let in_use = |id: u32, devs: &[Dev]| devs.iter().filter_map(|d| d.dec.as_ref()).any(|x| x.dev == id);
        if !serial.is_empty() {
            if let Some(&(_, id)) = self.kernel_ids.iter().find(|x| x.0 == serial) {
                if !in_use(id, &self.devs) {
                    return id;
                }
            }
        }
        let id = self.next_kernel_id;
        self.next_kernel_id += 1;
        if !serial.is_empty() {
            self.kernel_ids.retain(|x| x.0 != serial);
            self.kernel_ids.push((serial.to_string(), id));
        }
        id
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
                    clock: HostClock::new(),
                    clock_us: 0,
                    late_hist: [[0; 16]; 2],
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
                let host_clock = self.host_clock;
                let stream = self.stream;
                let passed = |c: &HostClock, us: i64| if stream { c.now(us) } else { c.passed(us) };
                let timing = host_clock || self.pose_timing;
                let host_us = host_us as i64;
                if host_clock && d.clock.mapped() {
                    // The P4 ticks every millisecond between reports.
                    while d.clock_us != 0 && host_us - d.clock_us > 1_000 {
                        d.clock_us += 1_000;
                        let start = events.len();
                        if let Some(raw) = passed(&d.clock, d.clock_us) {
                            kernel.push(Input::Clock { dev: dec.dev, time: raw }, events);
                        }
                        pose_timing(&d.clock, &events[start..], d.clock_us, &dec.config.serial, notes);
                    }
                }
                let mut outs = Vec::new();
                dec.report(interface, &data, &mut outs);
                let start = events.len();
                if timing {
                    if let Some((late, light)) = d.clock.observe(&outs, host_us) {
                        d.late_hist[light as usize][((late.max(0) / 500) as usize).min(15)] += 1;
                    }
                }
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
                if host_clock {
                    if let Some(raw) = passed(&d.clock, host_us) {
                        kernel.push(Input::Clock { dev: dec.dev, time: raw }, events);
                    }
                    d.clock_us = host_us;
                }
                if timing {
                    pose_timing(&d.clock, &events[start..], host_us, &serial, notes);
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
                        let id = self.kernel_id(&config.serial);
                        let d = &mut self.devs[i];
                        // Sensors in the head frame (the frame poses are reported in), for viewers.
                        let head_from_body = config.model.body_from_head.inverse();
                        let sensors_head: Vec<[Real; 3]> = config.model.sensors.iter().map(|p| head_from_body.apply(*p).to_array()).collect();
                        let normals_head: Vec<[Real; 3]> =
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

    /// --pose-timing: each device's report lateness histogram, as notes.
    pub fn timing_summary(&self, notes: &mut Vec<Note>) {
        for d in &self.devs {
            let serial = d.dec.as_ref().map(|x| x.config.serial.clone()).unwrap_or_default();
            notes.push(Note(json!({"type":"report_lateness","serial":serial,"bin_us":500,
                "imu":d.late_hist[0],"light":d.late_hist[1]})));
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
/// --host-clock: each new pose's light time on the host clock and the host time it came
/// out at (a report's arrival or a clock tick), for latency analysis.
fn pose_timing(clock: &HostClock, new: &[Event], at_us: i64, serial: &str, notes: &mut Vec<Note>) {
    for e in new {
        if let Event::Pose { time, light, .. } = e {
            notes.push(Note(json!({"type":"pose_timing","serial":serial,"t":*time as f64 / 48e6,
                "light_us":clock.to_host_us(*light as u32),"pose_us":clock.to_host_us(*time as u32),
                "at_us":at_us,"margin_us":clock.margin_us()})));
        }
    }
}

fn iface_name(role: Option<Interface>) -> &'static str {
    match role {
        Some(Interface::Watchman) => "IMU/Lightcap/Buttons",
        Some(Interface::Imu) => "IMU",
        Some(Interface::Lightcap) => "Lightcap",
        Some(Interface::Buttons) => "Buttons",
        None => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tracker_that_reconnects_keeps_its_kernel_id() {
        let mut m = DeviceManager::new();
        let a = m.kernel_id("LHR-AAAA");
        let b = m.kernel_id("LHR-BBBB");
        assert_ne!(a, b);
        // Disconnected (no decoder streaming), then handshaken again.
        assert_eq!(m.kernel_id("LHR-AAAA"), a);
        assert_eq!(m.kernel_id("LHR-BBBB"), b);
        // No serial: always a fresh id.
        let e = m.kernel_id("");
        assert_ne!(m.kernel_id(""), e);
    }
}

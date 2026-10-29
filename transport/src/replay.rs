//! A recorded session as a transport: reports come out in recorded order (as
//! fast as they're read, or paced by their host times), and feature requests
//! are answered with the recorded responses to the same report id, in order.
//!
//! Format: the vive_web / lhd recording text (gzip), lines
//!   `<t> <dev> HIDIN <interface> <host_us> <len> <hex>`
//!   `<t> <dev> HIDFEAT <GET|SET> <host_us> <rtn> <hex>`
//! Interface names (libsurvive's IMU / Lightcap / Buttons / IMU/Lightcap/Buttons)
//! map to numbers 0 / 1 / 2 / 0.

use crate::{DeviceInfo, DeviceKey, Transport, TransportError, TransportEvent};
use flate2::read::GzDecoder;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, Lines};
use std::time::{Duration, Instant};

pub struct ReplayTransport {
    lines: Lines<BufReader<GzDecoder<File>>>,
    /// Device name → key (pid from the name's kind; serial = name).
    devices: HashMap<String, DeviceKey>,
    /// Recorded GET responses per device and report id, in order.
    gets: HashMap<(DeviceKey, u8), VecDeque<Vec<u8>>>,
    pending: VecDeque<TransportEvent>,
    /// Pace reports by their recorded host times (1.0 = real time, 0 = as
    /// fast as possible).
    pub speed: f64,
    start: Option<(u64, Instant)>,
    done: bool,
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).filter_map(|i| u8::from_str_radix(s.get(2 * i..2 * i + 2)?, 16).ok()).collect()
}

fn pid_for(name: &str) -> u16 {
    match &name[..2.min(name.len())] {
        "WM" => 0x2101,
        "KN" => 0x2102,
        "TR" => 0x2022,
        _ => 0x2300,
    }
}

fn interface_number(name: &str) -> i32 {
    match name {
        "Lightcap" => 1,
        "Buttons" => 2,
        _ => 0,
    }
}

impl ReplayTransport {
    pub fn open(path: &str, speed: f64) -> Result<ReplayTransport, TransportError> {
        let f = File::open(path).map_err(|e| TransportError::Io(e.to_string()))?;
        Ok(ReplayTransport {
            lines: BufReader::new(GzDecoder::new(f)).lines(),
            devices: HashMap::new(),
            gets: HashMap::new(),
            pending: VecDeque::new(),
            speed,
            start: None,
            done: false,
        })
    }

    fn key(&mut self, name: &str, out: &mut VecDeque<TransportEvent>) -> DeviceKey {
        if let Some(k) = self.devices.get(name) {
            return k.clone();
        }
        let key = DeviceKey { pid: pid_for(name), serial: name.to_string() };
        self.devices.insert(name.to_string(), key.clone());
        out.push_back(TransportEvent::Added(DeviceInfo { key: key.clone(), vid: 0x28de, interfaces: vec![0], name: name.to_string() }));
        key
    }

    /// Reads ahead until a report is queued (feature responses are filed as
    /// they pass).
    fn fill(&mut self) {
        let mut out = VecDeque::new();
        while out.is_empty() {
            let Some(Ok(line)) = self.lines.next() else {
                self.done = true;
                break;
            };
            let p: Vec<&str> = line.split_whitespace().collect();
            if p.len() != 7 {
                continue;
            }
            match p[2] {
                "HIDFEAT" if p[3] == "GET" => {
                    let rtn: i64 = p[5].parse().unwrap_or(-1);
                    let key = self.key(p[1], &mut out);
                    if rtn > 0 {
                        let data = unhex(p[6]);
                        if let Some(&id) = data.first() {
                            self.gets.entry((key, id)).or_default().push_back(data[..(rtn as usize).min(data.len())].to_vec());
                        }
                    }
                }
                "HIDIN" => {
                    let key = self.key(p[1], &mut out);
                    let host_us: u64 = p[4].parse().unwrap_or(0);
                    out.push_back(TransportEvent::Report { key, interface: interface_number(p[3]), host_us, data: unhex(p[6]) });
                }
                _ => {}
            }
        }
        self.pending.extend(out);
    }
}

impl Transport for ReplayTransport {
    fn poll(&mut self, timeout: Duration) -> Vec<TransportEvent> {
        if self.pending.is_empty() && !self.done {
            self.fill();
        }
        let mut out = Vec::new();
        while let Some(e) = self.pending.front() {
            if let (TransportEvent::Report { host_us, .. }, true) = (e, self.speed > 0.0) {
                let (h0, t0) = *self.start.get_or_insert((*host_us, Instant::now()));
                let due = Duration::from_secs_f64((host_us.saturating_sub(h0)) as f64 / 1e6 / self.speed);
                let now = t0.elapsed();
                if due > now {
                    if out.is_empty() {
                        std::thread::sleep((due - now).min(timeout));
                    }
                    break;
                }
            }
            out.push(self.pending.pop_front().unwrap());
            if self.pending.is_empty() && !self.done && out.len() < 256 {
                self.fill();
            }
        }
        out
    }

    fn get_feature(&mut self, key: &DeviceKey, report_id: u8, _len: usize) -> Result<Vec<u8>, TransportError> {
        // Responses can be recorded after the request's position in the file
        // (they sit at the top): read ahead a little if needed.
        for _ in 0..64 {
            if let Some(q) = self.gets.get_mut(&(key.clone(), report_id)) {
                if let Some(r) = q.pop_front() {
                    return Ok(r);
                }
            }
            if self.done {
                break;
            }
            self.fill();
        }
        Err(TransportError::NotRecorded)
    }

    fn set_feature(&mut self, _key: &DeviceKey, data: &[u8]) -> Result<usize, TransportError> {
        Ok(data.len())
    }

    fn finished(&self) -> bool {
        self.done && self.pending.is_empty()
    }
}

//! Live devices through hidapi.

use crate::{DeviceInfo, DeviceKey, Transport, TransportError, TransportEvent};
use hidapi::{HidApi, HidDevice};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const VALVE_VID: u16 = 0x28de;
/// Product ids we drive (dongles, wired trackers).
const PIDS: &[u16] = &[0x2101, 0x2102, 0x2300, 0x2022];

struct Open {
    /// Interface 0 handle, shared with its reader (feature reports interleave
    /// with reads). macOS opens HID devices exclusively, so there's one handle.
    feature: Arc<Mutex<HidDevice>>,
    /// Set while a feature request waits for the handle: the reader steps
    /// aside (a mutex isn't fair, and the reader would otherwise keep it).
    want: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    readers: Vec<JoinHandle<()>>,
}

pub struct HidTransport {
    api: HidApi,
    open: HashMap<DeviceKey, Open>,
    tx: Sender<TransportEvent>,
    rx: Receiver<TransportEvent>,
    last_scan: Option<Instant>,
    epoch: Instant,
    counters: HashMap<u16, usize>,
}

impl HidTransport {
    pub fn new() -> Result<HidTransport, TransportError> {
        let api = HidApi::new().map_err(|e| TransportError::Io(e.to_string()))?;
        let (tx, rx) = channel();
        Ok(HidTransport { api, open: HashMap::new(), tx, rx, last_scan: None, epoch: Instant::now(), counters: HashMap::new() })
    }

    fn name_for(&mut self, pid: u16) -> String {
        let n = self.counters.entry(pid).or_insert(0);
        let base = match pid {
            0x2101 => "WM",
            0x2102 => "KN",
            0x2300 => "T2",
            0x2022 => "TR",
            _ => "DV",
        };
        let name = format!("{}{}", base, n);
        *n += 1;
        name
    }

    /// Enumerates; opens new devices and reports vanished ones.
    fn scan(&mut self, out: &mut Vec<TransportEvent>) {
        if self.api.refresh_devices().is_err() {
            return;
        }
        let mut seen: BTreeMap<DeviceKey, Vec<(i32, std::ffi::CString)>> = BTreeMap::new();
        for d in self.api.device_list() {
            if d.vendor_id() != VALVE_VID || !PIDS.contains(&d.product_id()) {
                continue;
            }
            let key = DeviceKey { pid: d.product_id(), serial: d.serial_number().unwrap_or("").to_string() };
            seen.entry(key).or_default().push((d.interface_number().max(0), d.path().to_owned()));
        }
        let gone: Vec<DeviceKey> = self.open.keys().filter(|k| !seen.contains_key(*k)).cloned().collect();
        for k in gone {
            self.close(&k);
            out.push(TransportEvent::Removed(k));
        }
        for (key, mut ifaces) in seen {
            if self.open.contains_key(&key) {
                continue;
            }
            ifaces.sort_by_key(|i| i.0);
            let mut handles = Vec::new();
            for (num, path) in &ifaces {
                match self.api.open_path(path) {
                    Ok(dev) => handles.push((*num, dev)),
                    Err(_) => {
                        handles.clear();
                        break;
                    }
                }
            }
            // Busy (another program has it) or mid re-enumeration: next scan.
            if handles.is_empty() {
                continue;
            }
            let stop = Arc::new(AtomicBool::new(false));
            let want = Arc::new(AtomicBool::new(false));
            let mut feature = None;
            let mut readers = Vec::new();
            for (num, dev) in handles {
                let dev = Arc::new(Mutex::new(dev));
                if num == 0 || feature.is_none() {
                    feature = Some(dev.clone());
                }
                readers.push(spawn_reader(dev, num, key.clone(), self.tx.clone(), stop.clone(), want.clone(), self.epoch));
            }
            let name = self.name_for(key.pid);
            out.push(TransportEvent::Added(DeviceInfo {
                key: key.clone(),
                vid: VALVE_VID,
                interfaces: ifaces.iter().map(|i| i.0).collect(),
                name,
            }));
            self.open.insert(key, Open { feature: feature.unwrap(), want, stop, readers });
        }
    }

    fn close(&mut self, key: &DeviceKey) {
        if let Some(o) = self.open.remove(key) {
            o.stop.store(true, Ordering::Relaxed);
            for r in o.readers {
                let _ = r.join();
            }
        }
    }
}

fn spawn_reader(
    dev: Arc<Mutex<HidDevice>>,
    interface: i32,
    key: DeviceKey,
    tx: Sender<TransportEvent>,
    stop: Arc<AtomicBool>,
    want: Arc<AtomicBool>,
    epoch: Instant,
) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut buf = [0u8; 256];
        while !stop.load(Ordering::Relaxed) {
            if want.load(Ordering::Acquire) {
                std::thread::sleep(Duration::from_micros(100));
                continue;
            }
            // Short timeouts so feature requests on the same handle get a turn.
            let n = match dev.lock().unwrap().read_timeout(&mut buf, 1) {
                Ok(n) => n,
                Err(_) => {
                    // Gone (unplugged, or the tracker reset): the next scan
                    // reports it removed.
                    break;
                }
            };
            if n > 0 {
                let host_us = epoch.elapsed().as_micros() as u64;
                if tx.send(TransportEvent::Report { key: key.clone(), interface, host_us, data: buf[..n].to_vec() }).is_err() {
                    break;
                }
            } else {
                std::thread::sleep(Duration::from_micros(200));
            }
        }
    })
}

impl Transport for HidTransport {
    fn poll(&mut self, timeout: Duration) -> Vec<TransportEvent> {
        let mut out = Vec::new();
        if self.last_scan.map_or(true, |t| t.elapsed() > Duration::from_secs(1)) {
            self.last_scan = Some(Instant::now());
            self.scan(&mut out);
        }
        if out.is_empty() {
            if let Ok(e) = self.rx.recv_timeout(timeout) {
                out.push(e);
            }
        }
        while let Ok(e) = self.rx.try_recv() {
            out.push(e);
            if out.len() > 4096 {
                break;
            }
        }
        out
    }

    fn get_feature(&mut self, key: &DeviceKey, report_id: u8, len: usize) -> Result<Vec<u8>, TransportError> {
        let o = self.open.get(key).ok_or(TransportError::NoDevice)?;
        let mut buf = vec![0u8; len.max(2)];
        buf[0] = report_id;
        o.want.store(true, Ordering::Release);
        let r = o.feature.lock().unwrap().get_feature_report(&mut buf);
        o.want.store(false, Ordering::Release);
        let n = r.map_err(|e| TransportError::Io(e.to_string()))?;
        buf.truncate(n);
        Ok(buf)
    }

    fn set_feature(&mut self, key: &DeviceKey, data: &[u8]) -> Result<usize, TransportError> {
        let o = self.open.get(key).ok_or(TransportError::NoDevice)?;
        o.want.store(true, Ordering::Release);
        let r = o.feature.lock().unwrap().send_feature_report(data);
        o.want.store(false, Ordering::Release);
        r.map_err(|e| TransportError::Io(e.to_string()))?;
        Ok(data.len())
    }
}

impl Drop for HidTransport {
    fn drop(&mut self) {
        let keys: Vec<DeviceKey> = self.open.keys().cloned().collect();
        for k in keys {
            self.close(&k);
        }
    }
}

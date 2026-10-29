//! Checks lh_protocol against libsurvive on a recording: decodes every raw
//! HID report (HIDIN lines) and feature-report exchange (HIDFEAT) and compares
//! with what libsurvive decoded from the same bytes (C, i and CONFIG lines).
//!
//!   lh_protocheck captures/vive/X.rec.gz
//!
//! libsurvive drops light before it has detected the base-station generation
//! and while its startup guard is up, so we may decode a few extra pulses at
//! the start; every one libsurvive logged must be among ours, and in order.
use flate2::read::GzDecoder;
use lh_protocol::config::{self, ConfigReader, ReadStep};
use lh_protocol::rf::RfDecoder;
use lh_protocol::usb::{self, UsbImu};
use lh_protocol::{Lightcap, RawImu, Sample};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn hex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap()).collect()
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: lh_protocheck REC.gz");
    let reader = BufReader::new(GzDecoder::new(File::open(&path).unwrap()));
    let mut rf: HashMap<String, RfDecoder> = HashMap::new();
    let mut usb_imu: HashMap<String, UsbImu> = HashMap::new();
    let mut cfg: HashMap<String, ConfigReader> = HashMap::new();
    let mut ours_light: Vec<Lightcap> = Vec::new();
    let mut ours_imu: Vec<RawImu> = Vec::new();
    let mut theirs_light: Vec<Lightcap> = Vec::new();
    let mut theirs_imu: Vec<RawImu> = Vec::new();
    let mut malformed: HashMap<&'static str, usize> = HashMap::new();
    let mut configs_ok = 0;
    let mut configs_bad = 0;
    let mut their_config: HashMap<String, serde_json::Value> = HashMap::new();
    let mut our_config: HashMap<String, Vec<u8>> = HashMap::new();
    let mut reports = 0usize;
    for line in reader.lines() {
        let Ok(line) = line else { continue };
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() < 3 {
            continue;
        }
        match p[2] {
            "HIDIN" if p.len() == 7 => {
                reports += 1;
                let bytes = hex(p[6]);
                let mut out = Vec::new();
                match p[3] {
                    "IMU/Lightcap/Buttons" => rf.entry(p[1].to_string()).or_default().decode(&bytes, &mut out),
                    "IMU" => usb_imu.entry(p[1].to_string()).or_default().decode(&bytes, &mut out),
                    "Lightcap" => usb::decode_lightcap(&bytes, &mut out),
                    "Buttons" => usb::decode_buttons(&bytes, &mut out),
                    _ => {}
                }
                for s in out {
                    match s {
                        Sample::Light(l) => ours_light.push(l),
                        Sample::Imu(i) => ours_imu.push(i),
                        Sample::Malformed(m) => *malformed.entry(m).or_default() += 1,
                        _ => {}
                    }
                }
            }
            "HIDFEAT" if p.len() == 7 && p[3] == "GET" => {
                let rtn: i32 = p[5].parse().unwrap_or(-1);
                if rtn <= 0 {
                    continue;
                }
                let bytes = hex(p[6]);
                let r = cfg.entry(p[1].to_string()).or_default();
                match bytes.first() {
                    Some(0x10) => {
                        r.start(&bytes);
                    }
                    Some(0x11) => match r.chunk(&bytes) {
                        ReadStep::Done(json) => {
                            our_config.insert(p[1].to_string(), json);
                        }
                        ReadStep::Restart(_) | ReadStep::Short { .. } | ReadStep::More => {}
                    },
                    _ => {}
                }
            }
            "C" if p.len() == 6 => {
                if let (Ok(sensor), Ok(timestamp), Ok(length)) = (p[3].parse(), p[4].parse(), p[5].parse()) {
                    theirs_light.push(Lightcap { sensor, timestamp, length });
                }
            }
            "i" if p.len() >= 11 => {
                let n: Vec<f64> = p[5..11].iter().filter_map(|x| x.parse().ok()).collect();
                if let (Ok(tc), 6) = (p[4].parse::<u32>(), n.len()) {
                    theirs_imu.push(RawImu {
                        timecode: tc,
                        accel: [n[0] as i16, n[1] as i16, n[2] as i16],
                        gyro: [n[3] as i16, n[4] as i16, n[5] as i16],
                    });
                }
            }
            "CONFIG" => {
                if let Ok(v) = serde_json::from_str(line.splitn(2, "CONFIG").nth(1).unwrap()) {
                    their_config.insert(p[1].to_string(), v);
                }
            }
            _ => {}
        }
    }
    for (dev, json) in &our_config {
        let ours: serde_json::Value = serde_json::from_slice(json).unwrap_or_default();
        match their_config.get(dev) {
            Some(t) if *t == ours => configs_ok += 1,
            _ => configs_bad += 1,
        }
        if let Ok(c) = config::parse(json) {
            println!("config {}: {} {} ({} sensors, sensor scale applied)", dev, c.model_number, c.serial, c.model.sensors.len());
        }
    }
    // Every libsurvive element must appear in ours, in order (ours may have extras).
    let in_order = |theirs: &[Lightcap], ours: &[Lightcap]| -> (usize, usize) {
        let mut j = 0;
        let mut found = 0;
        for t in theirs {
            let start = j;
            while j < ours.len() && ours[j] != *t {
                j += 1;
            }
            if j < ours.len() {
                found += 1;
                j += 1;
            } else {
                j = start;
            }
        }
        (found, theirs.len())
    };
    let (lf, lt) = in_order(&theirs_light, &ours_light);
    let mut j = 0;
    let mut imu_found = 0;
    for t in &theirs_imu {
        let start = j;
        while j < ours_imu.len() && ours_imu[j] != *t {
            j += 1;
        }
        if j < ours_imu.len() {
            imu_found += 1;
            j += 1;
        } else {
            j = start;
        }
    }
    println!(
        "{}: {} reports | light: libsurvive {} found {} ({} extra of ours) | raw IMU: libsurvive {} found {} ({} extra) | configs identical {}, differing {} | malformed {:?}",
        path,
        reports,
        lt,
        lf,
        ours_light.len() as i64 - lf as i64,
        theirs_imu.len(),
        imu_found,
        ours_imu.len() as i64 - imu_found as i64,
        configs_ok,
        configs_bad,
        malformed
    );
    if lf != lt || imu_found != theirs_imu.len() || configs_bad > 0 {
        std::process::exit(1);
    }
}

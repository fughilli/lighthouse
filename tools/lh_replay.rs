//! Runs the lighthouse kernel over a recorded session and writes its output
//! in libsurvive's recording format (so indoor_localization's swarm/tools/vive_diag.py metrics
//! apply) plus the kernel's events as JSON lines.
//!
//!   lh_replay captures/vive/X.rec.gz --out poses.txt [--events events.jsonl]
//!             [--stations libsurvive-config.json] [--no-auto-calibrate] [--clock]
//!
//! --save-stations FILE: each calibration's stations, in the --stations format (to replay
//! another run on the same stations with --no-auto-calibrate).
//!
//! --clock MS: as a host that maps its clock onto the device's would, tick Input::Clock
//! every 1 ms of device time (the P4's tick) up to the newest input's time less MS (how far light can
//! trail the IMU on the link: ~4 ms over the Watchman radio), decoded recordings only, so
//! sweeps go out as soon as the laser has passed (the P4 does this). --stream: with
//! --clock, send each sweep's hits as they come (TimingConfig::stream_sweeps).
//!
//! With --raw, everything comes from the raw HID bytes in the recording
//! (HIDIN reports, HIDFEAT config readout) through lh_protocol: the whole
//! Rust pipeline, no libsurvive. Without it, libsurvive's decoded lines:
//!   "<t> <dev> CONFIG <json>"   tracker model (sensors, IMU and head frames)
//!   "<t> <dev> C <sensor> <timecode> <length>"   light pulses
//!   "<t> <dev> I <mask> <timecode> ax ay az gx gy gz ..."   calibrated IMU
//! Output times are device seconds from the first sample.
use flate2::read::GzDecoder;
use lh_kernel::kernel::{Event, Input, Kernel, KernelConfig};
use lh_kernel::linalg::Mat;
use lh_kernel::lh1::AxisCal;
use lh_kernel::math::{Pose, Quat, Vec3};
use lh_kernel::model::TrackerModel;
use lh_kernel::timing::Pulse;
use lh_protocol::config::{ConfigReader, ReadStep};
use lh_protocol::imu::ImuCalibration;
use lh_protocol::rf::RfDecoder;
use lh_protocol::Sample;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

fn frame(v: &Value) -> Option<Pose> {
    let arr = |k: &str| -> Option<Vec3> {
        let a = v.get(k)?.as_array()?;
        Some(Vec3::new(a.first()?.as_f64()?, a.get(1)?.as_f64()?, a.get(2)?.as_f64()?))
    };
    let x = arr("plus_x")?.normalized();
    let z = arr("plus_z")?.normalized();
    let y = z.cross(x).normalized();
    let pos = arr("position").unwrap_or(Vec3::ZERO);
    Some(Pose { pos, rot: lh_kernel::solve::quat_from_columns(x, y, z) })
}

/// Tracker model from the device's JSON config, body frame = IMU frame.
pub fn tracker_model(cfg: &Value) -> Option<TrackerModel> {
    let lc = cfg.get("lighthouse_config")?;
    let pts = lc.get("modelPoints")?.as_array()?;
    let nrm = lc.get("modelNormals")?.as_array()?;
    let v3 = |p: &Value| -> Vec3 {
        let a = p.as_array().unwrap();
        Vec3::new(a[0].as_f64().unwrap(), a[1].as_f64().unwrap(), a[2].as_f64().unwrap())
    };
    let trackref_from_imu = frame(cfg.get("imu")?).unwrap_or(Pose::IDENTITY);
    let trackref_from_head = frame(cfg.get("head")?).unwrap_or(Pose::IDENTITY);
    let imu_from_trackref = trackref_from_imu.inverse();
    Some(TrackerModel {
        sensors: pts.iter().map(|p| imu_from_trackref.apply(v3(p))).collect(),
        normals: nrm.iter().map(|n| imu_from_trackref.rot.rotate(v3(n))).collect(),
        body_from_head: imu_from_trackref.compose(&trackref_from_head),
    })
}

/// Station priors from lhd's calibration JSON ({"stations":[{"id","pose",
/// "cov","cal"}]}) or a libsurvive config.json (almost-JSON).
fn station_priors(path: &str) -> Vec<Input> {
    let text = std::fs::read_to_string(path).expect("stations config");
    if let Ok(v) = serde_json::from_str::<Value>(&text) {
        if let Some(st) = v.get("stations").and_then(|s| s.as_array()) {
            return st
                .iter()
                .map(|s| {
                    let f = |x: &Value| x.as_f64().unwrap_or(0.0);
                    let p: Vec<f64> = s["pose"].as_array().unwrap().iter().map(f).collect();
                    let cov: Vec<f64> = s["cov"].as_array().unwrap().iter().map(f).collect();
                    let ax = |a: &Value| {
                        let c: Vec<f64> = a.as_array().unwrap().iter().map(f).collect();
                        AxisCal { phase: c[0], tilt: c[1], curve: c[2], gibphase: c[3], gibmag: c[4] }
                    };
                    let cal = s["cal"].as_array().unwrap();
                    Input::StationPrior {
                        id: s["id"].as_u64().unwrap() as u32,
                        world_from_station: Pose { pos: Vec3::new(p[0], p[1], p[2]), rot: Quat::new(p[3], p[4], p[5], p[6]).normalized() },
                        cov: Mat { rows: 6, cols: 6, d: cov },
                        cal: Some([ax(&cal[0]), ax(&cal[1])]),
                    }
                })
                .collect();
        }
    }
    let fixed = regex_lite_fix(&text);
    let v: Value = serde_json::from_str(&fixed).expect("parse stations config");
    let mut out = Vec::new();
    for (k, lh) in v.as_object().unwrap() {
        if !k.starts_with("lighthouse") {
            continue;
        }
        let f = |key: &str| -> Vec<f64> {
            lh[key].as_array().unwrap().iter().map(|x| x.as_str().unwrap().parse::<f64>().unwrap()).collect()
        };
        let p = f("pose");
        let var = f("variance");
        let cal = [0, 1].map(|a| AxisCal {
            phase: f("fcalphase")[a],
            tilt: f("fcaltilt")[a],
            curve: f("fcalcurve")[a],
            gibphase: f("fcalgibpha")[a],
            gibmag: f("fcalgibmag")[a],
        });
        out.push(Input::StationPrior {
            id: lh["id"].as_str().unwrap().parse().unwrap(),
            world_from_station: Pose { pos: Vec3::new(p[0], p[1], p[2]), rot: Quat::new(p[3], p[4], p[5], p[6]).normalized() },
            cov: Mat::diag(&var),
            cal: Some(cal),
        });
    }
    out
}

/// libsurvive writes `"a":"1"\n"b":...` without commas between items.
fn regex_lite_fix(text: &str) -> String {
    let mut s = String::from("{");
    let lines: Vec<&str> = text.lines().collect();
    for (i, l) in lines.iter().enumerate() {
        s.push_str(l);
        let next = lines.get(i + 1).map(|n| n.trim_start()).unwrap_or("");
        let t = l.trim_end();
        if next.starts_with('"') && (t.ends_with('"') || t.ends_with('}') || t.ends_with(']')) && !t.ends_with('{') {
            s.push(',');
        }
        s.push('\n');
    }
    s.push('}');
    s
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rec = args.get(1).expect("usage: lh_replay REC.gz --out FILE [--events FILE] [--stations CFG]");
    let opt = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let out_path = opt("--out").expect("--out");
    let mut out_w = BufWriter::new(File::create(&out_path).unwrap());
    let mut events = opt("--events").map(|p| BufWriter::new(File::create(p).unwrap()));
    // --innov FILE: per sweep "<line> <dev> <station> <axis> <mean> <rms> <n>" (rad), --raw only.
    let mut innov = opt("--innov").map(|p| BufWriter::new(File::create(p).unwrap()));
    let mut line_no = 0u64;
    let mut cfg = KernelConfig::default();
    if args.iter().any(|a| a == "--no-auto-calibrate") {
        cfg.auto_calibrate = false;
    }
    if args.iter().any(|a| a == "--stream") {
        cfg.timing.stream_sweeps = true;
    }
    if args.iter().any(|a| a == "--no-station-cov") {
        cfg.station_cov_in_filter = false;
    }
    // --set key=value for tuning experiments.
    for (i, a) in args.iter().enumerate() {
        if a != "--set" {
            continue;
        }
        let (k, v) = args[i + 1].split_once('=').expect("--set key=value");
        let v: f64 = v.parse().expect("number");
        match k {
            "sigma_angle" => cfg.track.sigma_angle = v,
            "accel_noise" => cfg.track.accel_noise = v,
            "gyro_noise" => cfg.track.gyro_noise = v,
            "gate" => cfg.track.gate = v,
            "accel_bias_walk" => cfg.track.accel_bias_walk = v,
            "still_updates" => cfg.still_updates = v != 0.0,
            "still_sigma_up" => cfg.still_sigma_up = v,
            "angle_bias_init" => cfg.track.angle_bias_init = v,
            "angle_bias_walk" => cfg.track.angle_bias_walk = v,
            "imu_time_offset" => cfg.imu_time_offset = v,
            _ => panic!("unknown --set {}", k),
        }
    }
    let mut kernel = Kernel::new(cfg);
    let mut ev = Vec::new();
    if let Some(p) = opt("--stations") {
        for i in station_priors(&p) {
            kernel.push(i, &mut ev);
        }
    }
    *SAVE_STATIONS.lock().unwrap() = opt("--save-stations");
    let raw = args.iter().any(|a| a == "--raw");
    let clock_lag: Option<u32> = opt("--clock").map(|v| (v.parse::<f64>().expect("--clock MS") * 48_000.0) as u32);
    let mut clock: HashMap<u32, u32> = HashMap::new();
    let reader = BufReader::new(GzDecoder::new(File::open(rec).unwrap()));
    // --raw state per device name.
    let mut cfg_readers: HashMap<String, ConfigReader> = HashMap::new();
    let mut rf: HashMap<String, RfDecoder> = HashMap::new();
    let mut imu_cal: HashMap<String, ImuCalibration> = HashMap::new();
    let mut chan: HashMap<String, Vec<Option<u8>>> = HashMap::new();
    let mut devs: HashMap<String, u32> = HashMap::new();
    let mut t0: Option<i64> = None;
    let mut clocks: HashMap<u32, (u32, i64)> = HashMap::new();
    let mut unwrap = |dev: u32, t: u32| -> i64 {
        let e = clocks.entry(dev).or_insert((t, t as i64));
        let u = e.1 + t.wrapping_sub(e.0) as i32 as i64;
        if u > e.1 {
            *e = (t, u);
        }
        u
    };
    for line in reader.lines() {
        let Ok(line) = line else { continue };
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() < 3 {
            continue;
        }
        if raw {
            let mut inputs: Vec<Input> = Vec::new();
            match p[2] {
                "HIDFEAT" if p.len() == 7 && p[3] == "GET" && p[5].parse::<i32>().unwrap_or(-1) > 0 => {
                    let b = unhex(p[6]);
                    let r = cfg_readers.entry(p[1].to_string()).or_default();
                    match b.first() {
                        Some(0x10) => {
                            r.start(&b);
                        }
                        Some(0x11) => {
                            if let ReadStep::Done(json) = r.chunk(&b) {
                                if let Ok(c) = lh_protocol::config::parse(&json) {
                                    let scales = imu_cal.get(p[1]).copied();
                                    let mut cal = c.imu;
                                    if let Some(sc) = scales {
                                        cal.acc_per_count = sc.acc_per_count;
                                        cal.gyro_per_count = sc.gyro_per_count;
                                    }
                                    imu_cal.insert(p[1].to_string(), cal);
                                    chan.insert(p[1].to_string(), c.channel_map.clone());
                                    let n = devs.len() as u32;
                                    let id = *devs.entry(p[1].to_string()).or_insert(n);
                                    inputs.push(Input::DeviceAdded { dev: id, model: c.model, wireless: p[1].starts_with("WM") });
                                }
                            }
                        }
                        Some(0x01) if b.len() >= 3 => {
                            imu_cal.entry(p[1].to_string()).or_default().set_scale_modes(b[1], b[2]);
                        }
                        _ => {}
                    }
                }
                "HIDIN" if p.len() == 7 && p[3] == "IMU/Lightcap/Buttons" => {
                    let Some(&dev) = devs.get(p[1]) else { continue };
                    let mut out = Vec::new();
                    rf.entry(p[1].to_string()).or_default().decode(&unhex(p[6]), &mut out);
                    for smp in out {
                        match smp {
                            Sample::Imu(i) => {
                                let (a, g) = imu_cal.get(p[1]).copied().unwrap_or_default().apply(&i);
                                let u = unwrap(dev, i.timecode);
                                let base = *t0.get_or_insert(u);
                                writeln!(out_file(&mut out_w), "{:.6} {} I 0 {} {} {} {} {} {} {} 0 0 0 0", (u - base) as f64 / 48e6, p[1], i.timecode, a[0], a[1], a[2], g[0], g[1], g[2]).unwrap();
                                inputs.push(Input::Imu { dev, time: i.timecode, accel_g: Vec3::new(a[0], a[1], a[2]), gyro: Vec3::new(g[0], g[1], g[2]) });
                            }
                            Sample::Light(l) => {
                                let Some(sensor) = chan.get(p[1]).and_then(|m| m.get(l.sensor as usize).copied().flatten()) else { continue };
                                inputs.push(Input::Pulse { dev, pulse: Pulse { sensor, start: l.timestamp, len: l.length as u32 } });
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
            line_no += 1;
            for input in inputs {
                kernel.push(input, &mut ev);
                for e in ev.drain(..) {
                    emit(&e, &devs, t0.unwrap_or(0), out_file(&mut out_w), events.as_mut());
                }
            }
            if let Some(w) = innov.as_mut() {
                for (name, &id) in &devs {
                    for (st, axis, mean, rms, n) in kernel.take_innovations(id) {
                        writeln!(w, "{} {} {} {} {:e} {:e} {}", line_no, name, st, axis, mean, rms, n).unwrap();
                    }
                }
            }
            continue;
        }
        let out = out_file(&mut out_w);
        let input = match p[2] {
            "CONFIG" => {
                let json = line.splitn(2, "CONFIG").nth(1).unwrap();
                let Ok(v) = serde_json::from_str::<Value>(json) else { continue };
                let Some(model) = tracker_model(&v) else { continue };
                let n = devs.len() as u32;
                let id = *devs.entry(p[1].to_string()).or_insert(n);
                Some(Input::DeviceAdded { dev: id, model, wireless: p[1].starts_with("WM") })
            }
            "C" if p.len() == 6 => {
                let Some(&dev) = devs.get(p[1]) else { continue };
                let (Ok(sensor), Ok(start), Ok(len)) = (p[3].parse(), p[4].parse(), p[5].parse()) else { continue };
                Some(Input::Pulse { dev, pulse: Pulse { sensor, start, len } })
            }
            "I" if p.len() >= 11 => {
                let Some(&dev) = devs.get(p[1]) else { continue };
                let Ok(time) = p[4].parse::<u32>() else { continue };
                let f: Vec<f64> = p[5..11].iter().filter_map(|x| x.parse().ok()).collect();
                if f.len() != 6 {
                    continue;
                }
                let u = unwrap(dev, time);
                let base = *t0.get_or_insert(u);
                writeln!(out, "{:.6} {} I 0 {} {} {} {} {} {} {} 0 0 0 0", (u - base) as f64 / 48e6, p[1], time, f[0], f[1], f[2], f[3], f[4], f[5]).unwrap();
                Some(Input::Imu { dev, time, accel_g: Vec3::new(f[0], f[1], f[2]), gyro: Vec3::new(f[3], f[4], f[5]) })
            }
            _ => None,
        };
        let Some(input) = input else { continue };
        if let Some(lag) = clock_lag {
            let at = match &input {
                Input::Pulse { dev, pulse } => Some((*dev, pulse.start.wrapping_add(pulse.len).wrapping_sub(lag))),
                Input::Imu { dev, time, .. } => Some((*dev, time.wrapping_sub(lag))),
                _ => None,
            };
            if let Some((dev, t)) = at {
                let last = clock.entry(dev).or_insert(t);
                while (t.wrapping_sub(*last) as i32) > 48_000 {
                    *last = last.wrapping_add(48_000);
                    kernel.push(Input::Clock { dev, time: *last }, &mut ev);
                    for e in ev.drain(..) {
                        emit(&e, &devs, t0.unwrap_or(0), out_file(&mut out_w), events.as_mut());
                    }
                }
            }
        }
        kernel.push(input, &mut ev);
        for e in ev.drain(..) {
            emit(&e, &devs, t0.unwrap_or(0), out_file(&mut out_w), events.as_mut());
        }
    }
    for (name, &id) in &devs {
        if let Some(ts) = kernel.timing_stats(id) {
            eprintln!("timing {}: {} sweeps, {} early, {} late sweep pulses", name, ts.sweeps, ts.early_sweeps, ts.late_sweep_pulses);
        }
    }
    // Innovation statistics per (station, axis) since the last report.
    for (name, &id) in &devs {
        for (st, axis, mean, rms, n) in kernel.take_innovations(id) {
            eprintln!("innovations {} station {} axis {}: mean {:+.0} urad, rms {:.0} urad, n {}", name, st, axis, mean * 1e6, rms * 1e6, n);
        }
    }
    out_w.flush().unwrap();
    if let Some(e) = events.as_mut() {
        e.flush().unwrap();
    }
}

fn out_file(w: &mut BufWriter<File>) -> &mut BufWriter<File> {
    w
}

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap_or(0)).collect()
}

/// --save-stations FILE.
static SAVE_STATIONS: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

fn emit(e: &Event, devs: &HashMap<String, u32>, t0: i64, out: &mut impl Write, events: Option<&mut BufWriter<File>>) {
    if let (Event::Calibrated { stations, .. }, Some(path)) = (e, SAVE_STATIONS.lock().unwrap().as_ref()) {
        let st: Vec<Value> = stations
            .iter()
            .map(|s| {
                let p = &s.world_from_station;
                let ax = |c: &AxisCal| serde_json::json!([c.phase, c.tilt, c.curve, c.gibphase, c.gibmag]);
                serde_json::json!({"id": s.id, "pose": [p.pos.x, p.pos.y, p.pos.z, p.rot.w, p.rot.x, p.rot.y, p.rot.z],
                    "cov": s.cov.d, "cal": [ax(&s.cal[0]), ax(&s.cal[1])]})
            })
            .collect();
        std::fs::write(path, serde_json::json!({ "stations": st }).to_string()).unwrap();
    }
    let name = |d: u32| devs.iter().find(|(_, v)| **v == d).map(|(k, _)| k.clone()).unwrap_or_default();
    if let Event::Pose { dev, time, used, gated, world_from_body: b, .. } = e {
        writeln!(out, "{:.6} {} USEDGATED {} {}", (*time - t0) as f64 / 48e6, name(*dev), used, gated).unwrap();
        writeln!(
            out,
            "{:.6} {} POSEBODY {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            (*time - t0) as f64 / 48e6,
            name(*dev),
            b.pos.x, b.pos.y, b.pos.z, b.rot.w, b.rot.x, b.rot.y, b.rot.z
        )
        .unwrap();
    }
    if let Event::Pose { dev, time, light, world_from_head: p, velocity: v, .. } = e {
        // The head position at the pose's newest light hit (back along the velocity), to
        // compare runs whose poses were propagated to different IMU times.
        let back = (*time - *light) as f64 / 48e6;
        writeln!(
            out,
            "{:.6} {} LIGHTPOSE {:.6} {:.6} {:.6}",
            (*light - t0) as f64 / 48e6,
            name(*dev),
            p.pos.x - v.x * back,
            p.pos.y - v.y * back,
            p.pos.z - v.z * back
        )
        .unwrap();
        writeln!(
            out,
            "{:.6} {} POSE {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
            (*time - t0) as f64 / 48e6,
            name(*dev),
            p.pos.x, p.pos.y, p.pos.z, p.rot.w, p.rot.x, p.rot.y, p.rot.z
        )
        .unwrap();
    }
    if let Some(w) = events {
        let line = match e {
            Event::Pose { .. } => return,
            Event::Calibrated { reason, stations, scenes, dropped, tracker_up_error } => {
                let st: Vec<String> = stations
                    .iter()
                    .map(|s| {
                        let sd: Vec<String> = (0..6).map(|i| format!("{:.2e}", s.cov[(i, i)].max(0.0).sqrt())).collect();
                        format!(
                            "{{\"id\":{},\"pos\":[{:.4},{:.4},{:.4}],\"rot\":[{:.5},{:.5},{:.5},{:.5}],\"sd\":[{}],\"rms_urad\":{:.0},\"up_err_deg\":{:.2}}}",
                            s.id,
                            s.world_from_station.pos.x, s.world_from_station.pos.y, s.world_from_station.pos.z,
                            s.world_from_station.rot.w, s.world_from_station.rot.x, s.world_from_station.rot.y, s.world_from_station.rot.z,
                            sd.join(","), s.rms * 1e6, s.up_error.to_degrees()
                        )
                    })
                    .collect();
                format!(
                    "{{\"event\":\"calibrated\",\"reason\":\"{:?}\",\"scenes\":{},\"dropped\":{},\"tracker_up_err_deg\":{:.2},\"stations\":[{}]}}",
                    reason, scenes, dropped, tracker_up_error.to_degrees(), st.join(",")
                )
            }
            other => format!("{{\"event\":\"{:?}\"}}", other).replace('"', "'").replacen('\'', "\"", 2),
        };
        writeln!(w, "{}", line).unwrap();
    }
}

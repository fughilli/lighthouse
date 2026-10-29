//! lhd: the lighthouse tracking daemon.
//!
//!   lhd [--replay REC.gz [--speed 1.0]] [--calibration FILE.json] [--record FILE.rec.gz]
//!
//! Transport (live hidapi, or a recording) → lh_protocol (device handshake,
//! report decoding) → lh_kernel → JSON lines on stdout. Commands come in as
//! JSON lines on stdin:
//!   {"cmd":"pair"}                 put every dongle in pairing mode (~10 s)
//!   {"cmd":"recalibrate"}          solve the station calibration again
//!   {"cmd":"record","path":"…"}    start recording raw traffic (.rec.gz)
//!   {"cmd":"stop_record"}
//!   {"cmd":"haptic","serial":"…"}  buzz a tracker
//!
//! Output events (one JSON object per line, "type" first): device, timing,
//! station, still, calibrated, tracking, pose, imu (decimated), battery, log.
//! Poses are the tracker's head frame in the world (z up, metres),
//! quaternions w-first, covariance 6x6 row-major over (position, rotation)
//! in world axes.

mod devices;
mod record;
mod store;

use devices::{DeviceManager, Note};
use lh_kernel::kernel::{Event, Kernel, KernelConfig};
use lh_transport::hid::HidTransport;
use lh_transport::replay::ReplayTransport;
use lh_transport::Transport;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let opt = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let calib_path = opt("--calibration");
    let speed: f64 = opt("--speed").and_then(|s| s.parse().ok()).unwrap_or(1.0);

    let mut transport: Box<dyn Transport> = match opt("--replay") {
        Some(p) => Box::new(ReplayTransport::open(&p, speed).expect("open replay")),
        None => Box::new(HidTransport::new().expect("hidapi")),
    };
    let replay = opt("--replay").is_some();

    let mut kernel = Kernel::new(KernelConfig::default());
    let mut events: Vec<Event> = Vec::new();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    if let Some(p) = &calib_path {
        match store::load(p) {
            Ok(inputs) => {
                let n = inputs.len();
                for i in inputs {
                    kernel.push(i, &mut events);
                }
                emit(&mut out, json!({"type":"log","msg":format!("loaded {} stations from {}", n, p)}));
                if let Some(v) = store::loaded_event(p) {
                    emit(&mut out, v);
                }
            }
            Err(e) => emit(&mut out, json!({"type":"log","msg":format!("no calibration loaded ({}): {}", p, e)})),
        }
    }

    let mut recorder = record::Recorder::default();
    if let Some(p) = opt("--record") {
        if let Err(e) = recorder.start(&p) {
            emit(&mut out, json!({"type":"log","msg":format!("record: {}", e)}));
        }
    }

    // Commands on stdin.
    let (cmd_tx, cmd_rx) = channel::<Value>();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            if let Ok(v) = serde_json::from_str::<Value>(&line) {
                if cmd_tx.send(v).is_err() {
                    break;
                }
            }
        }
    });

    let mut mgr = DeviceManager::new();
    let mut notes: Vec<Note> = Vec::new();
    let start = Instant::now();
    loop {
        while let Ok(cmd) = cmd_rx.try_recv() {
            match cmd["cmd"].as_str().unwrap_or("") {
                "pair" => mgr.pair_all(transport.as_mut(), &mut recorder, &mut notes),
                "recalibrate" => kernel.push(lh_kernel::kernel::Input::Recalibrate, &mut events),
                "record" => {
                    let path = cmd["path"].as_str().unwrap_or("lhd.rec.gz").to_string();
                    let r = recorder.start(&path);
                    emit(&mut out, json!({"type":"log","msg":format!("record {}: {:?}", path, r.err())}));
                }
                "stop_record" => {
                    let r = recorder.stop();
                    emit(&mut out, json!({"type":"log","msg":format!("recording stopped: {:?}", r)}));
                }
                "haptic" => mgr.haptic(transport.as_mut(), cmd["serial"].as_str().unwrap_or(""), &mut recorder),
                other => emit(&mut out, json!({"type":"log","msg":format!("unknown command {:?}", other)})),
            }
        }

        let batch = transport.poll(Duration::from_millis(5));
        let idle = batch.is_empty();
        mgr.set_now(start.elapsed());
        for ev in batch {
            mgr.handle(ev, transport.as_mut(), &mut kernel, &mut events, &mut recorder, &mut notes);
        }
        mgr.tick(transport.as_mut(), &mut kernel, &mut events, &mut recorder, &mut notes, start.elapsed());

        for n in notes.drain(..) {
            emit(&mut out, n.to_json());
        }
        for e in events.drain(..) {
            if let Event::Calibrated { .. } = &e {
                if let Some(p) = &calib_path {
                    if let Err(err) = store::save(p, &e) {
                        emit(&mut out, json!({"type":"log","msg":format!("saving calibration: {}", err)}));
                    }
                }
            }
            if let Some(v) = event_json(&e, &mgr) {
                emit(&mut out, v);
            }
        }
        out.flush().ok();
        if replay && idle && transport.finished() {
            break;
        }
    }
    recorder.stop().ok();
}

fn emit(out: &mut impl Write, v: Value) {
    writeln!(out, "{}", v).ok();
}

fn pose_json(p: &lh_kernel::math::Pose) -> (Value, Value) {
    (json!([p.pos.x, p.pos.y, p.pos.z]), json!([p.rot.w, p.rot.x, p.rot.y, p.rot.z]))
}

fn event_json(e: &Event, mgr: &DeviceManager) -> Option<Value> {
    let serial = |d: u32| mgr.serial_of(d).unwrap_or_default();
    Some(match e {
        Event::DeviceAdded { dev } => json!({"type":"device","serial":serial(*dev),"state":"tracking-ready"}),
        Event::DeviceRemoved { dev } => json!({"type":"device","serial":serial(*dev),"state":"removed"}),
        Event::TimingLocked { dev } => json!({"type":"timing","serial":serial(*dev),"locked":true}),
        Event::TimingLost { dev, reason } => json!({"type":"timing","serial":serial(*dev),"locked":false,"reason":format!("{:?}",reason)}),
        Event::StationSeen { dev, slot, info } => json!({
            "type":"station","serial":serial(*dev),"slot":slot,"id":info.id,"mode":info.mode,
            "accel":info.accel_dir,"fw":info.fw_version,"unlock_count":info.unlock_count
        }),
        Event::Still { dev, still } => json!({"type":"still","serial":serial(*dev),"still":still}),
        Event::Calibrated { reason, stations, scenes, dropped, tracker_up_error } => {
            let st: Vec<Value> = stations
                .iter()
                .map(|s| {
                    let (p, q) = pose_json(&s.world_from_station);
                    json!({"id":s.id,"mode":s.mode,"p":p,"q":q,"cov":s.cov.d,"rms":s.rms,"up_error":s.up_error})
                })
                .collect();
            json!({"type":"calibrated","reason":format!("{:?}",reason),"scenes":scenes,"dropped":dropped,
                   "tracker_up_error":tracker_up_error,"stations":st})
        }
        Event::StationsDisagree { dev, pos, rot, count } => json!({"type":"stations_disagree","serial":serial(*dev),
            "pos_m":pos,"rot_deg":rot.to_degrees(),"count":count}),
        Event::TrackingStarted { dev, rms } => json!({"type":"tracking","serial":serial(*dev),"state":"started","rms":rms}),
        Event::TrackingLost { dev, reason } => json!({"type":"tracking","serial":serial(*dev),"state":"lost","reason":format!("{:?}",reason)}),
        Event::Pose { dev, time, world_from_head, velocity, cov, used, gated, .. } => {
            let (p, q) = pose_json(world_from_head);
            json!({"type":"pose","serial":serial(*dev),"t":*time as f64 / 48e6,"p":p,"q":q,
                   "v":[velocity.x,velocity.y,velocity.z],"cov":cov.d,"used":used,"gated":gated})
        }
    })
}

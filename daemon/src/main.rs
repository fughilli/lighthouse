//! lhd: the lighthouse tracking daemon.
//!
//!   lhd [--replay REC.gz [--speed 1.0]] [--calibration FILE.json [--load-calibration]] [--record FILE.rec.gz]
//!       [--host-clock [--stream-sweeps]] [--pose-timing]
//!   lhd --help | --version       (exit at once; no device is opened)
//!
//! --host-clock maps each device's clock onto the report timestamps and tells the kernel
//! which device time has passed (Input::Clock, every ms), so a sweep goes out once the laser
//! has passed instead of at the next cycle's sync. With it or --pose-timing alone, each pose
//! also gets a "pose_timing" line: its light and pose times on the host clock, and when it
//! came out. --stream-sweeps sends each sweep's hits as they come (TimingConfig::stream_sweeps).
//!
//! Stations are solved afresh at every start (FILE gets each new calibration); with
//! --load-calibration FILE's stations are the starting point instead. A station no tracker
//! has seen for a while is dropped ("station_dropped"), and one tilted in place is levelled
//! again from its OOTX gravity ("calibrated", reason "StationTilted").
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
//! station, still, calibrated (its "pending": stations seen but not yet placed),
//! calibration_skipped (a calibration that changed nothing, and why), tracking, pose,
//! imu (decimated), battery, log.
//! Poses are the tracker's head frame in the world (z up, metres),
//! quaternions w-first, covariance 6x6 row-major over (position, rotation)
//! in world axes.

mod cli;
mod devices;
mod record;
mod store;

use cli::{Command, Options};
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
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = start(&args, &mut std::io::stdout(), &mut std::io::stderr(), open_transport);
    std::process::exit(code);
}

/// The live receivers (hidapi) or a recording.
fn open_transport(o: &Options) -> Result<Box<dyn Transport>, String> {
    match &o.replay {
        Some(p) => ReplayTransport::open(p, o.speed).map(|t| Box::new(t) as Box<dyn Transport>).map_err(|e| format!("opening replay {}: {:?}", p, e)),
        None => HidTransport::new().map(|t| Box::new(t) as Box<dyn Transport>).map_err(|e| format!("opening the USB receivers (hidapi): {:?}", e)),
    }
}

/// Parses the arguments; `--help`, `--version` and argument errors are answered here,
/// before `open` (the transport: the USB receivers) is called. Otherwise tracks until a
/// replay ends. Returns the exit code.
fn start(args: &[String], out: &mut dyn Write, err: &mut dyn Write, open: impl FnOnce(&Options) -> Result<Box<dyn Transport>, String>) -> i32 {
    let opts = match cli::parse(args) {
        Ok(Command::Help) => {
            write!(out, "{}", cli::USAGE).ok();
            return 0;
        }
        Ok(Command::Version) => {
            writeln!(out, "lhd {}", option_env!("CARGO_PKG_VERSION").unwrap_or("dev")).ok();
            return 0;
        }
        Ok(Command::Run(o)) => o,
        Err(e) => {
            writeln!(err, "lhd: {}\n(lhd --help for usage)", e).ok();
            return 2;
        }
    };
    match open(&opts) {
        Ok(t) => {
            track(&opts, t);
            0
        }
        Err(e) => {
            writeln!(err, "lhd: {}", e).ok();
            1
        }
    }
}

fn track(opts: &Options, mut transport: Box<dyn Transport>) {
    let calib_path = opts.calibration.clone();
    let replay = opts.replay.is_some();

    let stream = opts.stream_sweeps;
    let mut cfg = KernelConfig::default();
    cfg.timing.stream_sweeps = stream;
    let mut kernel = Kernel::new(cfg);
    let mut events: Vec<Event> = Vec::new();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());

    if let (Some(p), false) = (&calib_path, opts.load_calibration) {
        emit(&mut out, json!({"type":"log","msg":format!("solving the stations afresh (saving to {}; --load-calibration starts from it)", p)}));
    }
    if let (Some(p), true) = (&calib_path, opts.load_calibration) {
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
    if let Some(p) = &opts.record {
        if let Err(e) = recorder.start(p) {
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
    mgr.host_clock = opts.host_clock;
    mgr.pose_timing = opts.pose_timing;
    mgr.stream = stream;
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
            if mgr.pose_timing || mgr.host_clock {
                mgr.timing_summary(&mut notes);
                for n in notes.drain(..) {
                    emit(&mut out, n.to_json());
                }
            }
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
        Event::Calibrated { reason, stations, scenes, dropped, tracker_up_error, pending } => {
            let st: Vec<Value> = stations
                .iter()
                .map(|s| {
                    let (p, q) = pose_json(&s.world_from_station);
                    json!({"id":s.id,"mode":s.mode,"p":p,"q":q,"cov":s.cov.d,"rms":s.rms,"up_error":s.up_error})
                })
                .collect();
            json!({"type":"calibrated","reason":format!("{:?}",reason),"scenes":scenes,"dropped":dropped,
                   "tracker_up_error":tracker_up_error,"stations":st,"pending":pending})
        }
        Event::CalibrationSkipped { reason, why } => {
            json!({"type":"calibration_skipped","reason":format!("{:?}",reason),"why":why})
        }
        Event::StationsDisagree { dev, pos, rot, count } => json!({"type":"stations_disagree","serial":serial(*dev),
            "pos_m":pos,"rot_deg":rot.to_degrees(),"count":count}),
        Event::TrackingStarted { dev, rms } => json!({"type":"tracking","serial":serial(*dev),"state":"started","rms":rms}),
        Event::TrackingLost { dev, reason } => json!({"type":"tracking","serial":serial(*dev),"state":"lost","reason":format!("{:?}",reason)}),
        Event::StationDropped { id, why } => json!({"type":"station_dropped","id":id,"why":format!("{:?}",why)}),
        Event::Pose { dev, time, world_from_head, velocity, cov, used, gated, .. } => {
            let (p, q) = pose_json(world_from_head);
            json!({"type":"pose","serial":serial(*dev),"t":*time as f64 / 48e6,"p":p,"q":q,
                   "v":[velocity.x,velocity.y,velocity.z],"cov":cov.d,"used":used,"gated":gated})
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// Runs `start` with a transport factory that only records being called (and fails, so
    /// a run stops right there).
    fn start_spy(args: &[&str]) -> (i32, bool, String, String) {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let opened = Cell::new(false);
        let (mut out, mut err) = (Vec::new(), Vec::new());
        let code = start(&args, &mut out, &mut err, |_| {
            opened.set(true);
            Err("spy: no device".into())
        });
        (code, opened.get(), String::from_utf8(out).unwrap(), String::from_utf8(err).unwrap())
    }

    #[test]
    fn help_and_version_open_no_device() {
        for a in [&["--help"][..], &["-h"], &["--calibration", "c.json", "--help"], &["--version"], &["-V"]] {
            let (code, opened, out, _) = start_spy(a);
            assert_eq!(code, 0, "{:?}", a);
            assert!(!opened, "{:?} opened the transport", a);
            assert!(out.starts_with("usage: lhd") || out.starts_with("lhd "), "{:?}: {}", a, out);
        }
    }

    #[test]
    fn argument_errors_open_no_device() {
        for a in [&["--bogus"][..], &["--calibration"], &["--speed", "2"], &["--replay", "r.gz", "--speed", "x"]] {
            let (code, opened, _, err) = start_spy(a);
            assert_eq!(code, 2, "{:?}", a);
            assert!(!opened, "{:?} opened the transport", a);
            assert!(err.starts_with("lhd: "), "{:?}: {}", a, err);
        }
    }

    #[test]
    fn a_run_opens_the_transport_and_reports_failure() {
        let (code, opened, _, err) = start_spy(&["--calibration", "c.json"]);
        assert!(opened);
        assert_eq!(code, 1);
        assert!(err.contains("spy: no device"), "{}", err);
    }
}

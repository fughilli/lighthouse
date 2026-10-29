//! Runs the kernel's timing stage over light pulses read from stdin and prints
//! what it finds, for comparison with libsurvive on the same recording:
//!
//!   zcat captures/vive/X.rec.gz | bazel-bin/tools/lh_angles [--wireless] > angles.txt
//!
//! Input: libsurvive recording lines "<t> <dev> C <sensor> <timecode> <length>"
//! (other lines are ignored). Output, one line per event:
//!   SWEEP <slot> <axis> <sync_time> then per sensor: H <sensor> <n> <angle,len,start>...
//!   STATION <slot> <id> <mode> ...    LOCKED / LOST / OOTXERR
use lh_kernel::timing::{Pulse, Timing, TimingConfig, TimingEvent};
use std::io::{self, BufRead, Write};

fn main() {
    let wireless = std::env::args().any(|a| a == "--wireless");
    let mut timing = Timing::new(TimingConfig { wireless, ..TimingConfig::default() });
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut w = io::BufWriter::new(stdout.lock());
    let mut out = Vec::new();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };
        let p: Vec<&str> = line.split_whitespace().collect();
        if p.len() != 6 || p[2] != "C" {
            continue;
        }
        let (Ok(sensor), Ok(start), Ok(len)) = (p[3].parse::<u8>(), p[4].parse::<u32>(), p[5].parse::<u32>()) else {
            continue;
        };
        timing.push(Pulse { sensor, start, len }, &mut out);
        emit(&mut w, &mut out);
    }
    timing.flush(&mut out);
    emit(&mut w, &mut out);
    writeln!(w, "STATS {:?}", timing.stats).unwrap();
}

fn emit(w: &mut impl Write, out: &mut Vec<TimingEvent>) {
    for e in out.drain(..) {
        match e {
            TimingEvent::Sweep(s) => {
                write!(w, "SWEEP {} {} {}", s.slot, s.axis, s.sync_time).unwrap();
                for h in &s.hits {
                    write!(w, " H {} {}", h.sensor, h.candidates.len()).unwrap();
                    for c in &h.candidates {
                        write!(w, " {:.6},{},{}", c.angle, c.len, c.time.wrapping_sub(c.len / 2)).unwrap();
                    }
                }
                writeln!(w).unwrap();
            }
            TimingEvent::Station { slot, info } => writeln!(
                w,
                "STATION {} {} mode {} accel {:?} cal {:?}",
                slot, info.id, info.mode, info.accel_dir, info.cal
            )
            .unwrap(),
            TimingEvent::Locked { phase, inliers } => writeln!(w, "LOCKED {} {}", phase, inliers).unwrap(),
            TimingEvent::LockLost { reason } => writeln!(w, "LOST {:?}", reason).unwrap(),
            TimingEvent::OotxError { slot } => writeln!(w, "OOTXERR {}", slot).unwrap(),
        }
    }
}

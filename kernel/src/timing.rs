//! Light pulses to sweep hits: finds and tracks the base stations' 8.33 ms
//! cycle in one tracker's pulse stream, reads the sync codes and OOTX bits,
//! and turns hits in the sweep window into angles.
//!
//! Lighthouse 1.0 cycle (see [`crate::lh1`]): sync slot 0 at the cycle start,
//! slot 1 [`SLOT_TICKS`] later, then one station sweeps — the one whose sync
//! code says it isn't skipping. Each slot belongs to one station (slot 0 the
//! master), so the only thing to lock onto is the cycle phase:
//!
//! - **Sync candidates** are flashes of sync width that several sensors see at
//!   once (`min_sync_sensors`; 1 on wireless links, which report a flash on
//!   about one sensor). A lone pulse of sync width elsewhere — a reflection, a
//!   long sweep hit close to a station — is never one.
//! - **Lock** needs `lock_inliers` of the last [`HISTORY`] candidates to sit
//!   on one cycle phase with a valid code; strays don't block it.
//! - **Tracking**: each cycle's syncs are judged as a whole when the cycle
//!   ends; the phase and period follow them (a slow PLL). Width errors,
//!   coincident flashes inside the sweep window, cycles without any sync and
//!   OOTX framing errors cost confidence; at zero the lock is dropped.
//! - **Width code**: sync widths sit on a 500-tick grid, but where depends on
//!   the device and link (wired trackers report libsurvive's 2 750 + 500·code,
//!   the wireless link ~300 ticks longer). The grid offset is learned from the
//!   widths themselves; which whole code step is right comes from the b/c
//!   structure (both syncs in a cycle name the same axis, exactly one skips;
//!   a lone station never skips and alternates axes). Sweeps and OOTX wait for
//!   that, ~8 cycles.
//! - **Sweep hits**: every pulse a sensor sees in the sweep window is kept as
//!   a candidate (a reflection can be longer than the direct hit), longest
//!   first; choosing among them needs a pose and is the solver's job.
//!
//! Times are device ticks (48 MHz, u32 wrapping), unwrapped internally.

use crate::lh1::{ticks_to_angle, SyncCode, CYCLE_TICKS, SLOT_TICKS, SYNC_MAX_TICKS, SYNC_MIN_TICKS};
use crate::ootx::{OotxDecoder, OotxEvent, StationInfo};
use alloc::vec::Vec;

/// Sync candidates kept for finding the lock.
pub const HISTORY: usize = 48;
/// Pulses on different sensors from one flash start within this (ticks).
const COINCIDENCE: i64 = 1_000;
/// Out-of-order tolerance of the pulse stream (ticks); pulses are held this
/// long and released in start order.
const REORDER_HOLD: i64 = 6_000;
/// A sync's start may sit this far from its slot (ticks).
const SLOT_TOLERANCE: i64 = 500;
/// Largest acceptable sync width error (ticks) before it counts as wrong.
const CODE_TOLERANCE: u32 = 800;
/// Sweep window within the cycle (ticks from slot 0).
const SWEEP_START: i64 = 40_000;
const SWEEP_END: i64 = CYCLE_TICKS as i64 - 3_000;

#[derive(Clone, Copy, Debug)]
pub struct TimingConfig {
    /// 0 = auto from `wireless`: 1 wireless, 3 wired.
    pub min_sync_sensors: usize,
    pub wireless: bool,
    pub lock_inliers: usize,
    /// Longest sweep hit kept (ticks); hits get long close to a station.
    pub sweep_max_ticks: u32,
    /// OOTX framing errors without a good packet that drop the lock (0 = off).
    pub ootx_max_errors: u32,
    /// Cycles with no usable sync at all before the lock is dropped.
    pub max_silent_cycles: u32,
}

impl Default for TimingConfig {
    fn default() -> TimingConfig {
        TimingConfig {
            min_sync_sensors: 0,
            wireless: false,
            lock_inliers: 24,
            sweep_max_ticks: 48_000,
            ootx_max_errors: 4,
            max_silent_cycles: 120,
        }
    }
}

impl TimingConfig {
    fn min_sensors(&self) -> usize {
        match self.min_sync_sensors {
            0 if self.wireless => 1,
            0 => 3,
            n => n,
        }
    }
}

/// One light pulse on one sensor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pulse {
    pub sensor: u8,
    /// Start, device ticks.
    pub start: u32,
    /// Width, ticks.
    pub len: u32,
}

/// A candidate hit of a sweep on one sensor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HitCandidate {
    /// Sweep angle (rad) of the pulse's center.
    pub angle: f64,
    pub len: u32,
    /// Device time of the pulse center (ticks, wrapping).
    pub time: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SensorHits {
    pub sensor: u8,
    /// Longest first.
    pub candidates: Vec<HitCandidate>,
}

/// One station's sweep of one axis across the tracker.
#[derive(Clone, Debug, PartialEq)]
pub struct Sweep {
    /// Sync slot of the sweeping station (0 = master).
    pub slot: u8,
    pub axis: u8,
    /// Device time of that station's sync (ticks, wrapping).
    pub sync_time: u32,
    pub hits: Vec<SensorHits>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum TimingEvent {
    Locked { phase: u32, inliers: usize },
    LockLost { reason: LockLoss },
    Sweep(Sweep),
    /// A station's OOTX packet, decoded from the data bits of its slot.
    Station { slot: u8, info: StationInfo },
    OotxError { slot: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LockLoss {
    Confidence,
    Silence,
    Ootx,
}

/// Learned mapping from sync width to code: code = round((len - 2750 -
/// fine) / 500) + shift.
#[derive(Clone, Copy, Debug, Default)]
struct WidthCal {
    learned: u32,
    sin_sum: f64,
    cos_sum: f64,
    votes: [u32; 3],
    cycles: u32,
    shift: Option<i32>,
}

impl WidthCal {
    fn learn_fine(&mut self, len: u32) {
        let x = (len as f64 - 2_750.0) / 500.0;
        let r = (x - libm::floor(x)) * 2.0 * core::f64::consts::PI;
        self.learned += 1;
        self.sin_sum = self.sin_sum * 0.98 + libm::sin(r);
        self.cos_sum = self.cos_sum * 0.98 + libm::cos(r);
    }

    fn fine(&self) -> f64 {
        if self.sin_sum == 0.0 && self.cos_sum == 0.0 {
            return 0.0;
        }
        libm::atan2(self.sin_sum, self.cos_sum) / (2.0 * core::f64::consts::PI) * 500.0
    }

    fn raw_code(&self, len: u32) -> (i32, u32) {
        let x = (len as f64 - 2_750.0 - self.fine()) / 500.0;
        let k = libm::round(x);
        (k as i32, (libm::fabs(x - k) * 500.0) as u32)
    }

    /// The code for a width, once the step is known.
    fn decode(&self, len: u32) -> Option<(SyncCode, u32)> {
        let (k, err) = self.raw_code(len);
        let c = k + self.shift?;
        (0..8).contains(&c).then(|| (SyncCode::from_bits(c as u8), err))
    }

    /// One cycle's evidence for each whole-step shift (-1, 0, +1). Under the
    /// right shift: both syncs of a cycle name the same axis and exactly one
    /// skips; per station, the skip bit toggles exactly when the axis goes
    /// from 1 back to 0 (b/c sequence 4,5,0,1); a lone station never skips.
    fn vote(&mut self, raw: [Option<i32>; 2], prev_raw: [Option<i32>; 2]) {
        if self.shift.is_some() {
            return;
        }
        let both = raw[0].is_some() && raw[1].is_some();
        for (i, shift) in (-1..=1).enumerate() {
            let code = |k: Option<i32>| k.map(|k| k + shift).filter(|c| (0..8).contains(c)).map(|c| SyncCode::from_bits(c as u8));
            let mut ok = true;
            match (code(raw[0]), code(raw[1])) {
                (Some(a), Some(b)) => ok &= a.axis == b.axis && a.skip != b.skip,
                (Some(a), None) | (None, Some(a)) => ok &= !a.skip,
                (None, None) => ok = false,
            }
            for slot in 0..2 {
                if raw[slot].is_some() && code(raw[slot]).is_none() {
                    ok = false;
                }
                if let (Some(a), Some(p)) = (code(raw[slot]), code(prev_raw[slot])) {
                    if both {
                        let toggled = a.skip != p.skip;
                        ok &= if p.axis == 1 && a.axis == 0 { toggled } else { !toggled };
                    }
                }
            }
            if ok {
                self.votes[i] += 1;
            }
        }
        self.cycles += 1;
        if self.cycles >= 8 {
            let best = (0..3).max_by_key(|&i| self.votes[i]).unwrap();
            let second = (0..3).filter(|&i| i != best).map(|i| self.votes[i]).max().unwrap();
            if self.votes[best] >= 6 && self.votes[best] >= 2 * second.max(1) {
                self.shift = Some(best as i32 - 1);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Candidate {
    start: i64,
    len: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TimingStats {
    pub pulses: u64,
    pub sync_candidates: u64,
    pub lone_sync_rejected: u64,
    pub cycles: u64,
    pub syncs_ok: u64,
    pub syncs_bad_code: u64,
    pub syncs_missing: u64,
    pub misplaced_flashes: u64,
    pub sweeps: u64,
    pub hits: u64,
    pub locks: u64,
    pub lock_losses: u64,
}

/// Lock state: cycle `index` starts at `origin` (unwrapped ticks).
#[derive(Clone, Copy, Debug)]
struct Lock {
    origin: f64,
    period: f64,
    confidence: i32,
    silent: u32,
    /// Last code per slot, to predict the next (tie-breaks).
    last_code: [Option<SyncCode>; 2],
}

pub struct Timing {
    cfg: TimingConfig,
    held: Vec<(i64, Pulse)>,
    last_raw: Option<u32>,
    /// Unwrapped time of `last_raw`.
    unwrap_hi: i64,
    // Unlocked: flash being collected, and finished candidates.
    flash: Vec<(i64, u32, u8)>,
    history: Vec<Candidate>,
    lock: Option<Lock>,
    // Locked: pulses of the cycle being collected.
    cycle_sync: [Vec<(i64, u32, u8)>; 2],
    cycle_sweep: Vec<(i64, u32, u8)>,
    ootx: [OotxDecoder; 2],
    ootx_errors: [u32; 2],
    width: WidthCal,
    prev_raw: [Option<i32>; 2],
    pub stats: TimingStats,
}

impl Timing {
    pub fn new(cfg: TimingConfig) -> Timing {
        Timing {
            cfg,
            held: Vec::new(),
            last_raw: None,
            unwrap_hi: 0,
            flash: Vec::new(),
            history: Vec::new(),
            lock: None,
            cycle_sync: [Vec::new(), Vec::new()],
            cycle_sweep: Vec::new(),
            ootx: [OotxDecoder::new(), OotxDecoder::new()],
            ootx_errors: [0, 0],
            width: WidthCal::default(),
            prev_raw: [None, None],
            stats: TimingStats::default(),
        }
    }

    pub fn is_locked(&self) -> bool {
        self.lock.is_some()
    }

    /// Device ticks to a monotonic i64 timeline: each pulse is placed relative
    /// to the newest one seen (slightly late pulses land slightly earlier).
    fn unwrap(&mut self, t: u32) -> i64 {
        let u = match self.last_raw {
            None => t as i64,
            Some(last) => self.unwrap_hi + t.wrapping_sub(last) as i32 as i64,
        };
        if self.last_raw.is_none() || u > self.unwrap_hi {
            self.last_raw = Some(t);
            self.unwrap_hi = u;
        }
        u
    }

    /// Feeds one pulse; events go to `out`.
    pub fn push(&mut self, p: Pulse, out: &mut Vec<TimingEvent>) {
        self.stats.pulses += 1;
        let t = self.unwrap(p.start);
        let at = self.held.partition_point(|(ht, _)| *ht <= t);
        self.held.insert(at, (t, p));
        let newest = self.held.last().map(|h| h.0).unwrap_or(t);
        let ready = self.held.partition_point(|(ht, _)| *ht < newest - REORDER_HOLD);
        let released: Vec<(i64, Pulse)> = self.held.drain(..ready).collect();
        for (t, p) in released {
            self.process(t, p, out);
        }
    }

    /// Releases everything held (end of stream).
    pub fn flush(&mut self, out: &mut Vec<TimingEvent>) {
        let released: Vec<(i64, Pulse)> = self.held.drain(..).collect();
        for (t, p) in released {
            self.process(t, p, out);
        }
        if self.lock.is_some() {
            self.end_cycle(out);
        }
    }

    fn is_sync_width(&self, len: u32) -> bool {
        len >= SYNC_MIN_TICKS as u32 && len <= SYNC_MAX_TICKS as u32
    }

    fn process(&mut self, t: i64, p: Pulse, out: &mut Vec<TimingEvent>) {
        match self.lock {
            None => self.process_unlocked(t, p, out),
            Some(lock) => {
                // Close finished cycles first (possibly several if light was lost).
                let mut lock = lock;
                while t >= lock.origin as i64 + lock.period as i64 - 3_000 {
                    self.end_cycle(out);
                    match self.lock {
                        Some(l) => lock = l,
                        None => return self.process_unlocked(t, p, out),
                    }
                }
                let phase = t - lock.origin as i64;
                let slot1 = SLOT_TICKS as i64;
                if phase < slot1 - 3_000 {
                    self.cycle_sync[0].push((t, p.len, p.sensor));
                } else if phase < SWEEP_START - 3_000 {
                    self.cycle_sync[1].push((t, p.len, p.sensor));
                } else if phase < SWEEP_END {
                    self.cycle_sweep.push((t, p.len, p.sensor));
                }
            }
        }
    }

    // ------------------------------------------------------------- unlocked

    fn process_unlocked(&mut self, t: i64, p: Pulse, out: &mut Vec<TimingEvent>) {
        if !self.is_sync_width(p.len) {
            return;
        }
        if let Some(first) = self.flash.first() {
            if t - first.0 > COINCIDENCE {
                self.finish_flash(out);
            }
        }
        self.flash.push((t, p.len, p.sensor));
    }

    fn finish_flash(&mut self, out: &mut Vec<TimingEvent>) {
        let flash = core::mem::take(&mut self.flash);
        let (sensors, start, len) = group_stats(&flash);
        if sensors < self.cfg.min_sensors() {
            self.stats.lone_sync_rejected += 1;
            return;
        }
        self.stats.sync_candidates += 1;
        if self.history.len() == HISTORY {
            self.history.remove(0);
        }
        self.history.push(Candidate { start, len });
        self.try_lock(out);
    }

    /// Tries the newest candidate as slot 0 or slot 1 of a cycle; locks when
    /// enough of the history fits that phase with a plausible code.
    fn try_lock(&mut self, out: &mut Vec<TimingEvent>) {
        let newest = *self.history.last().unwrap();
        let mut best: Option<(usize, i64)> = None;
        for slot in 0..2i64 {
            let origin = newest.start - slot * SLOT_TICKS as i64;
            let inliers = self
                .history
                .iter()
                .filter(|c| {
                    let ph = (c.start - origin).rem_euclid(CYCLE_TICKS as i64);
                    let near0 = ph.min(CYCLE_TICKS as i64 - ph);
                    let near1 = (ph - SLOT_TICKS as i64).abs();
                    let (_, err) = SyncCode::decode(c.len);
                    (near0 < SLOT_TOLERANCE || near1 < SLOT_TOLERANCE) && err < CODE_TOLERANCE
                })
                .count();
            if best.map_or(true, |b| inliers > b.0) {
                best = Some((inliers, origin));
            }
        }
        let (inliers, origin) = best.unwrap();
        if inliers >= self.cfg.lock_inliers && inliers * 2 >= self.history.len() {
            // Seed the width grid from the candidates that made the lock.
            let mut width = WidthCal::default();
            for c in &self.history {
                width.learn_fine(c.len);
            }
            self.width = width;
            // Start collecting at the cycle containing the newest candidate.
            self.lock = Some(Lock {
                origin: origin as f64,
                period: CYCLE_TICKS as f64,
                confidence: 20,
                silent: 0,
                last_code: [None, None],
            });
            self.stats.locks += 1;
            self.history.clear();
            self.flash.clear();
            self.cycle_sync = [Vec::new(), Vec::new()];
            self.cycle_sweep.clear();
            out.push(TimingEvent::Locked { phase: origin as u32, inliers });
        }
    }

    // ------------------------------------------------------------- locked

    fn drop_lock(&mut self, reason: LockLoss, out: &mut Vec<TimingEvent>) {
        self.lock = None;
        self.stats.lock_losses += 1;
        self.cycle_sync = [Vec::new(), Vec::new()];
        self.cycle_sweep.clear();
        self.ootx = [OotxDecoder::new(), OotxDecoder::new()];
        self.ootx_errors = [0, 0];
        self.width = WidthCal::default();
        self.prev_raw = [None, None];
        out.push(TimingEvent::LockLost { reason });
    }

    fn end_cycle(&mut self, out: &mut Vec<TimingEvent>) {
        let mut lock = match self.lock {
            Some(l) => l,
            None => return,
        };
        self.stats.cycles += 1;
        let min_sensors = self.cfg.min_sensors();
        let mut syncs: [Option<(i64, SyncCode)>; 2] = [None, None];
        let mut raw: [Option<i32>; 2] = [None, None];
        let mut phase_err = 0.0;
        let mut n_err = 0;
        for slot in 0..2 {
            let pulses = core::mem::take(&mut self.cycle_sync[slot]);
            let expect = lock.last_code[slot].map(next_code);
            let sync_like: Vec<(i64, u32, u8)> =
                pulses.into_iter().filter(|p| self.is_sync_width_ext(p.1)).collect();
            if let Some((sensors, start, len)) = best_group(&sync_like, expect.zip(self.width.shift).map(|(e, sh)| (e, self.width.fine() - 500.0 * sh as f64))) {
                if sensors < min_sensors {
                    continue;
                }
                let (k, err) = self.width.raw_code(len);
                // Tight grid check once the grid offset is known.
                if self.width.learned >= 16 && err > 150 {
                    // Off the width grid: not a sync (or a saturated one).
                    self.stats.syncs_bad_code += 1;
                    lock.confidence -= 3;
                    continue;
                }
                self.width.learn_fine(len);
                self.stats.syncs_ok += 1;
                lock.confidence = (lock.confidence + 1).min(100);
                let slot_start = lock.origin + (slot as f64) * SLOT_TICKS as f64;
                phase_err += start as f64 - slot_start;
                n_err += 1;
                raw[slot] = Some(k);
                if let Some((code, _)) = self.width.decode(len) {
                    syncs[slot] = Some((start, code));
                    lock.last_code[slot] = Some(code);
                }
            }
        }
        self.width.vote(raw, self.prev_raw);
        self.prev_raw = raw;
        if n_err == 0 {
            self.stats.syncs_missing += 1;
            lock.silent += 1;
        } else {
            lock.silent = 0;
        }

        // Sweep: whichever station isn't skipping; its axis from its code.
        let sweep_pulses = core::mem::take(&mut self.cycle_sweep);
        if min_sensors > 1 && coincident_sync_flash(&sweep_pulses, min_sensors) {
            self.stats.misplaced_flashes += 1;
            lock.confidence -= 1;
        }
        let sweeper = match (syncs[0], syncs[1]) {
            (Some((t0, c0)), Some((t1, c1))) => match (c0.skip, c1.skip) {
                (false, true) => Some((0u8, t0, c0.axis)),
                (true, false) => Some((1u8, t1, c1.axis)),
                _ => None,
            },
            (Some((t0, c0)), None) if !c0.skip => Some((0, t0, c0.axis)),
            (None, Some((t1, c1))) if !c1.skip => Some((1, t1, c1.axis)),
            _ => None,
        };
        if let Some((slot, sync_t, axis)) = sweeper {
            let sweep = self.build_sweep(slot, sync_t, axis, &sweep_pulses);
            if !sweep.hits.is_empty() {
                self.stats.sweeps += 1;
                self.stats.hits += sweep.hits.len() as u64;
                out.push(TimingEvent::Sweep(sweep));
            }
        }

        // OOTX bits per slot.
        for slot in 0..2u8 {
            if let Some((_, code)) = syncs[slot as usize] {
                match self.ootx[slot as usize].push(code.data) {
                    Some(OotxEvent::Packet(info)) => {
                        self.ootx_errors[slot as usize] = 0;
                        out.push(TimingEvent::Station { slot, info });
                    }
                    Some(OotxEvent::FramingError) => {
                        self.ootx_errors[slot as usize] += 1;
                        out.push(TimingEvent::OotxError { slot });
                    }
                    None => {}
                }
            }
        }

        // Follow the phase (and slowly the period).
        lock.origin += lock.period;
        if n_err > 0 {
            let e = phase_err / n_err as f64;
            lock.origin += 0.5 * e;
            lock.period += 0.002 * e;
        }
        self.lock = Some(lock);

        let ootx_bad = self.cfg.ootx_max_errors > 0
            && self.ootx_errors.iter().any(|&e| e >= self.cfg.ootx_max_errors);
        if lock.confidence <= 0 {
            self.drop_lock(LockLoss::Confidence, out);
        } else if lock.silent >= self.cfg.max_silent_cycles {
            self.drop_lock(LockLoss::Silence, out);
        } else if ootx_bad {
            self.drop_lock(LockLoss::Ootx, out);
        }
    }

    /// Close to a station a sync flash saturates and stretches; while locked,
    /// widths a little past the code range still count (the code check follows).
    fn is_sync_width_ext(&self, len: u32) -> bool {
        len >= SYNC_MIN_TICKS as u32 && len <= SYNC_MAX_TICKS as u32 + 1_000
    }

    fn build_sweep(&self, slot: u8, sync_t: i64, axis: u8, pulses: &[(i64, u32, u8)]) -> Sweep {
        let mut hits: Vec<SensorHits> = Vec::new();
        for &(t, len, sensor) in pulses {
            if len >= self.cfg.sweep_max_ticks {
                continue;
            }
            let center = t + (len / 2) as i64;
            let ticks = (center - sync_t) as f64;
            let cand = HitCandidate { angle: ticks_to_angle(ticks), len, time: center as u32 };
            match hits.iter_mut().find(|h| h.sensor == sensor) {
                Some(h) => h.candidates.push(cand),
                None => hits.push(SensorHits { sensor, candidates: alloc::vec![cand] }),
            }
        }
        for h in hits.iter_mut() {
            h.candidates.sort_by(|a, b| b.len.cmp(&a.len));
        }
        hits.sort_by_key(|h| h.sensor);
        Sweep { slot, axis, sync_time: sync_t as u32, hits }
    }
}

/// b/c sequence per slot: the axis alternates every cycle and the skip bit
/// every two; the data bit is unknown.
fn next_code(c: SyncCode) -> SyncCode {
    let skip = if c.axis == 1 { !c.skip } else { c.skip };
    SyncCode { skip, data: false, axis: c.axis ^ 1 }
}

/// (distinct sensors, median start, median width) of a set of pulses.
fn group_stats(pulses: &[(i64, u32, u8)]) -> (usize, i64, u32) {
    let mut seen = 0u64;
    let mut n = 0;
    for p in pulses {
        let bit = 1u64 << (p.2 & 63);
        if seen & bit == 0 {
            seen |= bit;
            n += 1;
        }
    }
    let mut starts: Vec<i64> = pulses.iter().map(|p| p.0).collect();
    let mut lens: Vec<u32> = pulses.iter().map(|p| p.1).collect();
    starts.sort_unstable();
    lens.sort_unstable();
    (n, starts[starts.len() / 2], lens[lens.len() / 2])
}

/// The largest coincident group among `pulses`; ties go to the one whose width
/// best fits `expect` (ignoring the data bit).
fn best_group(pulses: &[(i64, u32, u8)], expect: Option<(SyncCode, f64)>) -> Option<(usize, i64, u32)> {
    let mut best: Option<((usize, i64, u32), u32)> = None;
    for anchor in pulses {
        let group: Vec<(i64, u32, u8)> =
            pulses.iter().copied().filter(|p| p.0 >= anchor.0 && p.0 - anchor.0 <= COINCIDENCE).collect();
        let g = group_stats(&group);
        let fit = match expect {
            Some((e, fine)) => {
                let off = |c: SyncCode| libm::fabs(g.2 as f64 - c.nominal_ticks() as f64 - fine) as u32;
                off(e).min(off(SyncCode { data: true, ..e }))
            }
            None => 0,
        };
        let better = match best {
            None => true,
            Some((b, bfit)) => g.0 > b.0 || (g.0 == b.0 && fit < bfit),
        };
        if better {
            best = Some((g, fit));
        }
    }
    best.map(|b| b.0)
}

/// Several sensors flashing at once with sync width: a sync where a sweep
/// should be, so the lock is off.
fn coincident_sync_flash(pulses: &[(i64, u32, u8)], min_sensors: usize) -> bool {
    let sync_like: Vec<(i64, u32, u8)> = pulses
        .iter()
        .copied()
        .filter(|p| p.1 >= SYNC_MIN_TICKS as u32 && p.1 <= SYNC_MAX_TICKS as u32)
        .collect();
    best_group(&sync_like, None).map_or(false, |g| g.0 >= min_sensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lh1::SyncCode;

    /// A synthetic b/c stream: two stations, 4-cycle code sequence, `n`
    /// sensors seeing each sync, sweeps hitting `hits` sensors at given angles.
    fn stream(cycles: u32, sync_sensors: u8, strays: bool) -> Vec<Pulse> {
        let mut v = Vec::new();
        let base: u32 = 4_294_000_000; // wraps during the stream
        let seq0 = [4u8, 5, 0, 1];
        let seq1 = [0u8, 1, 4, 5];
        for k in 0..cycles {
            let t0 = base.wrapping_add(k * CYCLE_TICKS);
            for (slot, code) in [(0u32, seq0[(k % 4) as usize]), (1, seq1[(k % 4) as usize])] {
                let c = SyncCode::from_bits(code);
                for s in 0..sync_sensors {
                    v.push(Pulse { sensor: s, start: t0.wrapping_add(slot * SLOT_TICKS + s as u32 * 3), len: c.nominal_ticks() + 40 });
                }
            }
            // Sweeper: slot with skip = 0.
            let c0 = SyncCode::from_bits(seq0[(k % 4) as usize]);
            let slot = if c0.skip { 1 } else { 0 };
            let sync = t0.wrapping_add(slot * SLOT_TICKS);
            for s in 0..6u8 {
                let ticks = 200_000 + s as u32 * 1_000; // angle ~ 0 + s * 7.85 mrad
                v.push(Pulse { sensor: s, start: sync.wrapping_add(ticks - 400), len: 800 });
            }
            if strays {
                // A long reflection on sensor 2 and a sync-width stray on sensor 9.
                v.push(Pulse { sensor: 2, start: sync.wrapping_add(180_000), len: 2_000 });
                v.push(Pulse { sensor: 9, start: t0.wrapping_add(150_000), len: 4_000 });
            }
        }
        v
    }

    fn run(pulses: &[Pulse], cfg: TimingConfig) -> (Vec<TimingEvent>, Timing) {
        let mut t = Timing::new(cfg);
        let mut out = Vec::new();
        for p in pulses {
            t.push(*p, &mut out);
        }
        t.flush(&mut out);
        (out, t)
    }

    #[test]
    fn locks_and_measures_angles() {
        let (out, t) = run(&stream(200, 5, false), TimingConfig::default());
        assert_eq!(t.stats.locks, 1, "{:?}", t.stats);
        assert_eq!(t.stats.lock_losses, 0);
        let sweeps: Vec<&Sweep> =
            out.iter().filter_map(|e| if let TimingEvent::Sweep(s) = e { Some(s) } else { None }).collect();
        assert!(sweeps.len() > 150);
        for s in &sweeps {
            for h in &s.hits {
                let want = ticks_to_angle(200_000.0 + h.sensor as f64 * 1_000.0);
                assert!((h.candidates[0].angle - want).abs() < 1e-3, "{} vs {}", h.candidates[0].angle, want);
            }
        }
        // Both stations and both axes show up.
        for slot in 0..2 {
            for axis in 0..2 {
                assert!(sweeps.iter().any(|s| s.slot == slot && s.axis == axis));
            }
        }
    }

    #[test]
    fn strays_dont_block_lock_and_direct_hits_survive() {
        let (out, t) = run(&stream(200, 5, true), TimingConfig::default());
        assert_eq!(t.stats.locks, 1);
        assert_eq!(t.stats.lock_losses, 0);
        let mut saw_reflection = false;
        for e in &out {
            if let TimingEvent::Sweep(s) = e {
                for h in &s.hits {
                    if h.sensor == 2 {
                        // The reflection is longer, so it's listed first — but the
                        // direct hit is still there for the solver.
                        assert_eq!(h.candidates.len(), 2);
                        saw_reflection = true;
                    }
                }
            }
        }
        assert!(saw_reflection);
    }

    #[test]
    fn wireless_single_sensor_syncs() {
        let cfg = TimingConfig { wireless: true, ..TimingConfig::default() };
        let (_, t) = run(&stream(200, 1, true), cfg);
        assert_eq!(t.stats.locks, 1, "{:?}", t.stats);
        assert!(t.stats.sweeps > 150);
        // Wired rules reject the same stream (one sensor per sync).
        let (_, t) = run(&stream(200, 1, false), TimingConfig::default());
        assert_eq!(t.stats.locks, 0);
    }
}

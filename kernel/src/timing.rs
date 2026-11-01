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

use crate::Real;
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
    /// OOTX framing errors in a row on a slot (0 = off) that drop the lock if neither slot
    /// gave a good packet under it, else restart only that slot's decoder.
    pub ootx_max_errors: u32,
    /// Cycles with no usable sync at all before the lock is dropped.
    pub max_silent_cycles: u32,
    /// With a host clock ([`Timing::clock`]): a sweep goes out once its window has had no
    /// pulse for this long (ticks) instead of at the next cycle's sync, up to ~7 ms later.
    /// The laser crosses a tracker in well under a millisecond.
    pub sweep_quiet_ticks: u32,
    /// With a host clock: send each sweep in parts, the hits that came in since the last
    /// clock tick (the host ticks after every report), not waiting for the sweep to end. A
    /// wired tracker batches its light by up to ~3 ms, so waiting for quiet costs that much
    /// again; the filter takes hits one at a time anyway.
    pub stream_sweeps: bool,
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
            sweep_quiet_ticks: 48_000,
            stream_sweeps: false,
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
    pub angle: Real,
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
    /// 0: the sweep, or its first part when it goes out in parts (`stream_sweeps`, or hits
    /// that came after an early sweep); 1, 2, ...: more hits of the same sweep (same slot,
    /// axis and sync_time). A sensor can appear in more than one part.
    pub part: u8,
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
    sin_sum: Real,
    cos_sum: Real,
    votes: [u32; 3],
    cycles: u32,
    shift: Option<i32>,
}

impl WidthCal {
    fn learn_fine(&mut self, len: u32) {
        let x = (len as Real - 2_750.0) / 500.0;
        let r = (x - crate::math::floor(x)) * 2.0 * crate::consts::PI;
        self.learned += 1;
        self.sin_sum = self.sin_sum * 0.98 + crate::math::sin(r);
        self.cos_sum = self.cos_sum * 0.98 + crate::math::cos(r);
    }

    fn fine(&self) -> Real {
        if self.sin_sum == 0.0 && self.cos_sum == 0.0 {
            return 0.0;
        }
        crate::math::atan2(self.sin_sum, self.cos_sum) / (2.0 * crate::consts::PI) * 500.0
    }

    fn raw_code(&self, len: u32) -> (i32, u32) {
        let x = (len as Real - 2_750.0 - self.fine()) / 500.0;
        let k = crate::math::round(x);
        (k as i32, (crate::math::fabs(x - k) * 500.0) as u32)
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
    /// Of `sweeps`, sent early on the host clock (`Timing::clock`).
    pub early_sweeps: u64,
    /// Sweep-window pulses that arrived after their sweep went out early (not streaming):
    /// they go out as a later part.
    pub late_sweep_pulses: u64,
    /// Sweep parts sent (streaming: about one per light report in the sweep).
    pub sweep_parts: u64,
    pub hits: u64,
    pub locks: u64,
    pub lock_losses: u64,
    /// A proven slot's OOTX decoder restarted after framing errors (the lock kept).
    pub ootx_restarts: u64,
}

/// Lock state: the current cycle starts at `origin + origin_frac` (unwrapped
/// ticks). The whole ticks stay integer: as a float they would outgrow f32's
/// 24-bit mantissa within a second of device time.
#[derive(Clone, Copy, Debug)]
struct Lock {
    origin: i64,
    origin_frac: Real,
    period: Real,
    confidence: i32,
    silent: u32,
    /// Last code per slot, to predict the next (tie-breaks).
    last_code: [Option<SyncCode>; 2],
}

impl Lock {
    /// Moves the cycle start by `ticks`, carrying whole ticks into `origin`.
    fn advance(&mut self, ticks: Real) {
        let x = self.origin_frac + ticks;
        let whole = crate::math::floor(x);
        self.origin += whole as i64;
        self.origin_frac = x - whole;
    }
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
    /// Of `cycle_sweep`, the pulses already sent (`clock`), in how many parts, and as which
    /// station's sweep: later pulses go out as further parts of the same sweep.
    sweep_sent: usize,
    sweep_parts: u8,
    sweeper: Option<(u8, i64, u8)>,
    ootx: [OotxDecoder; 2],
    ootx_errors: [u32; 2],
    /// Good OOTX packets per slot under the current lock: one on either proves the lock.
    ootx_good: [u32; 2],
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
            sweep_sent: 0,
            sweep_parts: 0,
            sweeper: None,
            ootx: [OotxDecoder::new(), OotxDecoder::new()],
            ootx_errors: [0, 0],
            ootx_good: [0, 0],
            width: WidthCal::default(),
            prev_raw: [None, None],
            stats: TimingStats::default(),
        }
    }

    /// Switch `stream_sweeps` (takes effect from the next clock tick).
    pub fn set_stream_sweeps(&mut self, on: bool) {
        self.cfg.stream_sweeps = on;
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

    /// The host's word for device time `now` (ticks, wrapping): every pulse from before
    /// `now` has been pushed (the host maps its own clock to the device's). Releases held
    /// pulses up to it, then sends the current sweep's hits so far: all of them once the
    /// window has been quiet for `sweep_quiet_ticks`, or with `stream_sweeps` whatever came
    /// in since the last tick. Optional: without it, sweeps go out at the cycle's end.
    pub fn clock(&mut self, now: u32, out: &mut Vec<TimingEvent>) {
        let Some(last) = self.last_raw else { return };
        let u = self.unwrap_hi + now.wrapping_sub(last) as i32 as i64;
        // A host clock that disagrees with the pulse stream by more than two cycles is wrong
        // (not yet mapped, or the device restarted): ignore it.
        if (u - self.unwrap_hi).abs() > 2 * CYCLE_TICKS as i64 {
            return;
        }
        // Held pulses that old are in order by now (the quiet test below needs them
        // processed).
        let stream = self.cfg.stream_sweeps;
        let margin = if stream { REORDER_HOLD } else { REORDER_HOLD.max(self.cfg.sweep_quiet_ticks as i64) };
        let ready = self.held.partition_point(|(ht, _)| *ht < u - margin);
        if ready > 0 {
            let released: Vec<(i64, Pulse)> = self.held.drain(..ready).collect();
            for (t, p) in released {
                self.process(t, p, out);
            }
        }
        let Some(lock) = self.lock else { return };
        if self.cycle_sweep.len() == self.sweep_sent || u >= lock.origin + lock.period as i64 - 3_000 {
            return;
        }
        if !stream {
            if self.sweep_sent > 0 || !self.held.is_empty() {
                return;
            }
            let quiet_since = self.cycle_sweep.iter().map(|p| p.0 + p.1 as i64).max().unwrap_or(0);
            if u < quiet_since + self.cfg.sweep_quiet_ticks as i64 {
                return;
            }
        }
        if self.sweeper.is_none() {
            self.sweeper = self.sweeper_peek();
        }
        if let Some(sw) = self.sweeper {
            if self.send_sweep(sw, out) {
                self.stats.early_sweeps += (self.sweep_parts == 1) as u64;
            }
        }
    }

    /// Sends the cycle's sweep pulses not sent yet: the sweep, or its next part. True if
    /// it had hits.
    fn send_sweep(&mut self, (slot, sync_t, axis): (u8, i64, u8), out: &mut Vec<TimingEvent>) -> bool {
        let sweep = self.build_sweep(slot, sync_t, axis, self.sweep_parts, &self.cycle_sweep[self.sweep_sent..]);
        self.sweep_sent = self.cycle_sweep.len();
        if sweep.hits.is_empty() {
            return false;
        }
        self.stats.sweeps += (self.sweep_parts == 0) as u64;
        self.stats.sweep_parts += 1;
        self.stats.hits += sweep.hits.len() as u64;
        self.sweep_parts = self.sweep_parts.saturating_add(1);
        out.push(TimingEvent::Sweep(sweep));
        true
    }

    /// The current cycle's sweeping station from its syncs so far, as `end_cycle` will
    /// judge them, without its side effects (width learning, confidence, stats).
    fn sweeper_peek(&self) -> Option<(u8, i64, u8)> {
        let lock = self.lock?;
        let min_sensors = self.cfg.min_sensors();
        let mut syncs: [Option<(i64, SyncCode)>; 2] = [None, None];
        for slot in 0..2 {
            let sync_like: Vec<(i64, u32, u8)> =
                self.cycle_sync[slot].iter().copied().filter(|p| self.is_sync_width_ext(p.1)).collect();
            let expect = lock.last_code[slot].map(next_code);
            let Some((sensors, start, len)) =
                best_group(&sync_like, expect.zip(self.width.shift).map(|(e, sh)| (e, self.width.fine() - 500.0 * sh as Real)))
            else {
                continue;
            };
            if sensors < min_sensors {
                continue;
            }
            let (_, err) = self.width.raw_code(len);
            if self.width.learned >= 16 && err > 150 {
                continue;
            }
            if let Some((code, _)) = self.width.decode(len) {
                syncs[slot] = Some((start, code));
            }
        }
        sweeper_of(syncs)
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
                while t >= lock.origin + lock.period as i64 - 3_000 {
                    self.end_cycle(out);
                    match self.lock {
                        Some(l) => lock = l,
                        None => return self.process_unlocked(t, p, out),
                    }
                }
                let phase = t - lock.origin;
                let slot1 = SLOT_TICKS as i64;
                if phase < slot1 - 3_000 {
                    self.cycle_sync[0].push((t, p.len, p.sensor));
                } else if phase < SWEEP_START - 3_000 {
                    self.cycle_sync[1].push((t, p.len, p.sensor));
                } else if phase < SWEEP_END {
                    if self.sweep_sent > 0 && !self.cfg.stream_sweeps {
                        // Came after the early sweep (the link delayed it past the quiet
                        // window): a later part. Counted to tune sweep_quiet_ticks.
                        self.stats.late_sweep_pulses += 1;
                    }
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
                origin,
                origin_frac: 0.0,
                period: CYCLE_TICKS as Real,
                confidence: 20,
                silent: 0,
                last_code: [None, None],
            });
            self.stats.locks += 1;
            self.history.clear();
            self.flash.clear();
            self.cycle_sync = [Vec::new(), Vec::new()];
            self.cycle_sweep.clear();
            self.sweep_sent = 0;
            self.sweep_parts = 0;
            self.sweeper = None;
            out.push(TimingEvent::Locked { phase: origin as u32, inliers });
        }
    }

    // ------------------------------------------------------------- locked

    fn drop_lock(&mut self, reason: LockLoss, out: &mut Vec<TimingEvent>) {
        self.lock = None;
        self.stats.lock_losses += 1;
        self.cycle_sync = [Vec::new(), Vec::new()];
        self.cycle_sweep.clear();
        self.sweep_sent = 0;
        self.sweep_parts = 0;
        self.sweeper = None;
        self.ootx = [OotxDecoder::new(), OotxDecoder::new()];
        self.ootx_errors = [0, 0];
        self.ootx_good = [0, 0];
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
            if let Some((sensors, start, len)) = best_group(&sync_like, expect.zip(self.width.shift).map(|(e, sh)| (e, self.width.fine() - 500.0 * sh as Real))) {
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
                let slot_start = lock.origin + slot as i64 * SLOT_TICKS as i64;
                phase_err += (start - slot_start) as Real - lock.origin_frac;
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

        // Sweep: whichever station isn't skipping; its axis from its code. What `clock`
        // already sent went as that station's; the rest goes out now, as its next part.
        if min_sensors > 1 && coincident_sync_flash(&self.cycle_sweep, min_sensors) {
            self.stats.misplaced_flashes += 1;
            lock.confidence -= 1;
        }
        if let Some(sw) = self.sweeper.or(sweeper_of(syncs)) {
            self.send_sweep(sw, out);
        }
        self.cycle_sweep.clear();
        self.sweep_sent = 0;
        self.sweep_parts = 0;
        self.sweeper = None;

        // OOTX bits per slot.
        for slot in 0..2u8 {
            if let Some((_, code)) = syncs[slot as usize] {
                match self.ootx[slot as usize].push(code.data) {
                    Some(OotxEvent::Packet(info)) => {
                        self.ootx_errors[slot as usize] = 0;
                        self.ootx_good[slot as usize] += 1;
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
        let mut step = lock.period;
        if n_err > 0 {
            let e = phase_err / n_err as Real;
            step += 0.5 * e;
            lock.period += 0.002 * e;
        }
        lock.advance(step);
        self.lock = Some(lock);

        // Framing errors in a row on a slot. Before any good packet under this lock they
        // say the lock itself is wrong (phase or widths): drop it. Once either slot has
        // given one, the phase and widths (shared by both slots) are proven, and errors are
        // a slot's data bits getting garbled (missed syncs at a grazing angle, a partial
        // occlusion): its sweeps are still good, so only its decoder restarts.
        let proven = self.ootx_good.iter().any(|&g| g > 0);
        let mut ootx_bad = false;
        if self.cfg.ootx_max_errors > 0 {
            for slot in 0..2 {
                if self.ootx_errors[slot] >= self.cfg.ootx_max_errors {
                    if proven {
                        self.ootx[slot] = OotxDecoder::new();
                        self.ootx_errors[slot] = 0;
                        self.stats.ootx_restarts += 1;
                    } else {
                        ootx_bad = true;
                    }
                }
            }
        }
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

    fn build_sweep(&self, slot: u8, sync_t: i64, axis: u8, part: u8, pulses: &[(i64, u32, u8)]) -> Sweep {
        let mut hits: Vec<SensorHits> = Vec::new();
        for &(t, len, sensor) in pulses {
            if len >= self.cfg.sweep_max_ticks {
                continue;
            }
            let center = t + (len / 2) as i64;
            let ticks = (center - sync_t) as Real;
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
        Sweep { slot, axis, sync_time: sync_t as u32, part, hits }
    }
}

/// Whichever station isn't skipping sweeps; its axis from its code: (slot, sync, axis).
fn sweeper_of(syncs: [Option<(i64, SyncCode)>; 2]) -> Option<(u8, i64, u8)> {
    match (syncs[0], syncs[1]) {
        (Some((t0, c0)), Some((t1, c1))) => match (c0.skip, c1.skip) {
            (false, true) => Some((0u8, t0, c0.axis)),
            (true, false) => Some((1u8, t1, c1.axis)),
            _ => None,
        },
        (Some((t0, c0)), None) if !c0.skip => Some((0, t0, c0.axis)),
        (None, Some((t1, c1))) if !c1.skip => Some((1, t1, c1.axis)),
        _ => None,
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
fn best_group(pulses: &[(i64, u32, u8)], expect: Option<(SyncCode, Real)>) -> Option<(usize, i64, u32)> {
    let mut best: Option<((usize, i64, u32), u32)> = None;
    for anchor in pulses {
        let group: Vec<(i64, u32, u8)> =
            pulses.iter().copied().filter(|p| p.0 >= anchor.0 && p.0 - anchor.0 <= COINCIDENCE).collect();
        let g = group_stats(&group);
        let fit = match expect {
            Some((e, fine)) => {
                let off = |c: SyncCode| crate::math::fabs(g.2 as Real - c.nominal_ticks() as Real - fine) as u32;
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
        stream_data(cycles, sync_sensors, strays, &[Vec::new(), Vec::new()])
    }

    /// stream(), with OOTX data bits per slot (one per cycle; false past the end).
    fn stream_data(cycles: u32, sync_sensors: u8, strays: bool, data: &[Vec<bool>; 2]) -> Vec<Pulse> {
        let mut v = Vec::new();
        let base: u32 = 4_294_000_000; // wraps during the stream
        let seq0 = [4u8, 5, 0, 1];
        let seq1 = [0u8, 1, 4, 5];
        for k in 0..cycles {
            let t0 = base.wrapping_add(k * CYCLE_TICKS);
            for (slot, code) in [(0u32, seq0[(k % 4) as usize]), (1, seq1[(k % 4) as usize])] {
                let bit = data[slot as usize].get(k as usize).copied().unwrap_or(false);
                let c = SyncCode::from_bits(code | (bit as u8) << 1);
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

    /// OOTX bits for one packet from station `id` (the framing ootx.rs decodes).
    fn ootx_frame(id: u32) -> Vec<bool> {
        let mut payload = [0u8; 33];
        payload[2..6].copy_from_slice(&id.to_le_bytes());
        payload[21] = 127;
        payload[31] = 1;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes.push(0);
        bytes.extend_from_slice(&crate::ootx::crc32(&payload).to_le_bytes());
        let mut bits = alloc::vec![false; 17];
        bits.push(true);
        for w in bytes.chunks(2) {
            let word = ((w[0] as u16) << 8) | w[1] as u16;
            for i in (0..16).rev() {
                bits.push(word >> i & 1 != 0);
            }
            bits.push(true);
        }
        bits
    }

    /// A garbled frame: a preamble, then a length word of all ones (a framing error).
    fn ootx_garbled() -> Vec<bool> {
        let mut bits = alloc::vec![false; 17];
        bits.push(true);
        bits.extend(core::iter::repeat(true).take(17));
        bits
    }

    fn stations_and_losses(out: &[TimingEvent]) -> (usize, usize) {
        let st = out.iter().filter(|e| matches!(e, TimingEvent::Station { slot: 0, .. })).count();
        let lost = out.iter().filter(|e| matches!(e, TimingEvent::LockLost { reason: LockLoss::Ootx })).count();
        (st, lost)
    }

    #[test]
    fn host_clock_sends_each_sweep_when_the_laser_has_passed() {
        let pulses = stream(300, 5, true);
        let (plain, _) = run(&pulses, TimingConfig::default());
        // The same stream with the host's clock ticking every 0.25 ms between pulses.
        let mut t = Timing::new(TimingConfig::default());
        let mut out = Vec::new();
        let mut sent_at = Vec::new(); // (sweep sync time, device time it went out)
        let mut clock = pulses[0].start;
        for p in &pulses {
            while (p.start.wrapping_sub(clock) as i32) > 12_000 {
                clock = clock.wrapping_add(12_000);
                let n = out.len();
                t.clock(clock, &mut out);
                for e in &out[n..] {
                    if let TimingEvent::Sweep(s) = e {
                        sent_at.push((s.sync_time, clock));
                    }
                }
            }
            t.push(*p, &mut out);
        }
        t.flush(&mut out);
        let sweeps = |v: &[TimingEvent]| {
            v.iter().filter_map(|e| if let TimingEvent::Sweep(s) = e { Some(s.clone()) } else { None }).collect::<Vec<_>>()
        };
        let (a, b) = (sweeps(&plain), sweeps(&out));
        assert!(a.len() > 250);
        assert_eq!(a, b, "the same sweeps, early or not");
        assert!(t.stats.early_sweeps as usize >= a.len() - 2, "{} of {} early", t.stats.early_sweeps, a.len());
        assert_eq!(t.stats.late_sweep_pulses, 0);
        // Hits end ~205k ticks after the sync: out within quiet + a clock tick of that, not
        // at the next cycle (~400k).
        assert!(!sent_at.is_empty());
        for (sync, at) in sent_at {
            let after = at.wrapping_sub(sync);
            assert!(after < 205_000 + 48_000 + 12_000 + 6_000, "sent {} ticks after its sync", after);
        }
    }

    #[test]
    fn streamed_sweeps_send_each_hit_at_the_next_tick() {
        let pulses = stream(300, 5, true);
        let (plain, _) = run(&pulses, TimingConfig::default());
        let mut t = Timing::new(TimingConfig { stream_sweeps: true, ..TimingConfig::default() });
        let mut out = Vec::new();
        let mut clock = pulses[0].start;
        // When each pulse was pushed (the clock tick before it), by (sensor, start).
        let mut pushed: std::collections::HashMap<(u8, u32), u32> = Default::default();
        let mut worst = 0u32; // device ticks from a hit's push to its part going out
        for p in &pulses {
            while (p.start.wrapping_sub(clock) as i32) > 12_000 {
                clock = clock.wrapping_add(12_000);
                let n = out.len();
                t.clock(clock, &mut out);
                for e in &out[n..] {
                    if let TimingEvent::Sweep(s) = e {
                        for h in &s.hits {
                            for c in &h.candidates {
                                let at = pushed[&(h.sensor, c.time.wrapping_sub(c.len / 2))];
                                worst = worst.max(clock.wrapping_sub(at));
                            }
                        }
                    }
                }
            }
            pushed.insert((p.sensor, p.start), clock);
            t.push(*p, &mut out);
        }
        t.flush(&mut out);
        // Per sweep (sync time): sensor -> its candidates, over all parts.
        let merged = |v: &[TimingEvent]| {
            let mut m: alloc::collections::BTreeMap<(u32, u8), Vec<(u32, u32)>> = Default::default();
            let mut parts = 0;
            for e in v {
                if let TimingEvent::Sweep(s) = e {
                    parts += 1;
                    for h in &s.hits {
                        let c = m.entry((s.sync_time, h.sensor)).or_default();
                        c.extend(h.candidates.iter().map(|x| (x.time, x.len)));
                        c.sort();
                    }
                }
            }
            (m, parts)
        };
        let ((a, na), (b, nb)) = (merged(&plain), merged(&out));
        assert_eq!(a, b, "the same hits, in parts or not");
        assert!(nb > na, "{} parts for {} sweeps", nb, na);
        assert_eq!(t.stats.sweeps as usize, na);
        // Out at the first or second tick after it came in (the reorder hold).
        assert!(worst <= 2 * 12_000, "a hit went out {} ticks after it came in", worst);
    }

    #[test]
    fn garbled_ootx_on_a_proven_slot_keeps_the_lock() {
        // A good packet, then garbled data (missed syncs at a grazing angle), then good again
        // (after 100 cycles of nothing while the lock comes up).
        let mut bits = alloc::vec![false; 100];
        bits.extend(ootx_frame(1416427640));
        for _ in 0..8 {
            bits.extend(ootx_garbled());
        }
        bits.extend(ootx_frame(1416427640));
        let n = bits.len() as u32 + 200;
        let (out, t) = run(&stream_data(n, 5, false, &[bits, Vec::new()]), TimingConfig::default());
        let (stations, lost) = stations_and_losses(&out);
        assert_eq!(lost, 0, "{:?}", t.stats);
        assert_eq!(t.stats.lock_losses, 0);
        assert!(t.stats.ootx_restarts >= 1, "{:?}", t.stats);
        assert_eq!(stations, 2, "the packet after the garbling decodes");
    }

    #[test]
    fn garbled_ootx_on_one_slot_is_proven_by_the_other() {
        // Slot 1 only ever garbles; slot 0's packet proves the lock, so slot 1 restarts.
        let mut b0 = alloc::vec![false; 100];
        b0.extend(ootx_frame(699579721));
        let mut b1 = alloc::vec![false; 100 + b0.len()];
        for _ in 0..8 {
            b1.extend(ootx_garbled());
        }
        let n = b1.len() as u32 + 50;
        let (out, t) = run(&stream_data(n, 5, false, &[b0, b1]), TimingConfig::default());
        let (_, lost) = stations_and_losses(&out);
        assert_eq!(lost, 0, "{:?}", t.stats);
        assert!(t.stats.ootx_restarts >= 1, "{:?}", t.stats);
    }

    #[test]
    fn garbled_ootx_on_an_unproven_slot_drops_the_lock() {
        // No good packet under this lock: the data bits being wrong says the lock is.
        let mut bits = alloc::vec![false; 100];
        for _ in 0..8 {
            bits.extend(ootx_garbled());
        }
        let n = bits.len() as u32 + 50;
        let (out, t) = run(&stream_data(n, 5, false, &[bits, Vec::new()]), TimingConfig::default());
        let (_, lost) = stations_and_losses(&out);
        assert!(lost >= 1, "{:?}", t.stats);
        assert_eq!(t.stats.ootx_restarts, 0);
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
                let want = ticks_to_angle(200_000.0 + h.sensor as Real * 1_000.0);
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

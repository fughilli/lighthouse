//! The device's clock (48 MHz ticks, u32 wrapping) on the host's, from when reports
//! arrive, so a host can (1) put its pose times on its own clock and (2) tell the kernel
//! which device time has certainly passed ([`lh_kernel::kernel::Input::Clock`]): then a
//! sweep goes out once the laser has passed, not at the next cycle's sync (up to ~7 ms).
//!
//! The offset is the smallest (arrival − device time) over the last 1-2 s: the link's
//! fastest delivery. Its true value (USB polling, the tracker's own buffering) can't be
//! seen from the host, so it counts as zero: mapped times are late by that constant, and a
//! report's lateness is "above the fastest". Two 1 s windows follow the crystals' drift
//! (tens of ppm). Light can trail that by a few ms (a tracker batching pulses; the Watchman
//! radio, ~4 ms behind its IMU), so the clock handed to the kernel lags by a margin: the
//! recent worst lateness of reports with light, decaying over a few seconds.
use crate::Output;
use lh_kernel::kernel::Input;

/// Ticks per µs.
const TICKS_PER_US: i64 = 48;
/// Kept on top of the measured light lateness (µs).
const MARGIN_FLOOR_US: i64 = 250;
/// The margin decays toward each new lateness by 1/this per light report (~1 kHz: ~4 s).
const MARGIN_DECAY: i64 = 4096;
/// A stall (a USB hiccup) mustn't hold the margin up for long: past this a sweep goes out
/// at its cycle's end anyway.
const MARGIN_MAX_US: i64 = 6_000;

pub struct HostClock {
    last_raw: Option<u32>,
    /// Unwrapped ticks of `last_raw` (the newest device time seen).
    hi: i64,
    /// Per window: min over reports of (arrival in ticks − device ticks).
    cur: i64,
    prev: i64,
    win_start_us: i64,
    /// Light lateness margin (µs), a decaying peak.
    margin_us: i64,
}

impl Default for HostClock {
    fn default() -> HostClock {
        HostClock::new()
    }
}

impl HostClock {
    const NONE: i64 = i64::MAX;

    pub fn new() -> HostClock {
        HostClock { last_raw: None, hi: 0, cur: Self::NONE, prev: Self::NONE, win_start_us: i64::MIN / 2, margin_us: 0 }
    }

    /// `raw` on the unwrapped timeline, relative to the newest device time seen.
    fn place(&self, raw: u32) -> Option<i64> {
        self.last_raw.map(|last| self.hi + raw.wrapping_sub(last) as i32 as i64)
    }

    fn offset(&self) -> Option<i64> {
        let o = self.cur.min(self.prev);
        (o != Self::NONE).then_some(o)
    }

    pub fn mapped(&self) -> bool {
        self.offset().is_some()
    }

    /// A report's decoded outputs, which arrived at `host_us`: its newest device time (a
    /// pulse is reported once it ends) maps the clocks. Returns how much later than the
    /// fastest delivery it came (µs) and whether it carried light, once mapped.
    pub fn observe(&mut self, outs: &[Output], host_us: i64) -> Option<(i64, bool)> {
        let mut newest: Option<u32> = None;
        let mut light = false;
        for o in outs {
            let raw = match o {
                Output::Kernel(Input::Imu { time, .. }) => *time,
                Output::Kernel(Input::Pulse { pulse, .. }) => {
                    light = true;
                    pulse.start.wrapping_add(pulse.len)
                }
                _ => continue,
            };
            if newest.map_or(true, |n| (raw.wrapping_sub(n) as i32) > 0) {
                newest = Some(raw);
            }
        }
        let raw = newest?;
        let t = self.place(raw).unwrap_or(raw as i64);
        if self.last_raw.is_none() || t > self.hi {
            self.last_raw = Some(raw);
            self.hi = t;
        }
        if host_us - self.win_start_us >= 1_000_000 {
            self.prev = self.cur;
            self.cur = Self::NONE;
            self.win_start_us = host_us;
        }
        let d = host_us * TICKS_PER_US - t;
        self.cur = self.cur.min(d);
        let late = (d - self.offset()?) / TICKS_PER_US;
        if light {
            if late > self.margin_us {
                self.margin_us = late.min(MARGIN_MAX_US);
            } else {
                self.margin_us -= (self.margin_us - late) / MARGIN_DECAY;
            }
        }
        Some((late, light))
    }

    /// Device time `raw` on the host's clock (µs).
    pub fn to_host_us(&self, raw: u32) -> Option<i64> {
        Some((self.place(raw)? + self.offset()?) / TICKS_PER_US)
    }

    /// The device time (wrapping ticks) the kernel may take as passed at host time
    /// `host_us`: mapped, less the light margin.
    pub fn passed(&self, host_us: i64) -> Option<u32> {
        let us = host_us - self.margin_us - MARGIN_FLOOR_US;
        Some((us * TICKS_PER_US - self.offset()?) as u32)
    }

    /// The device time (wrapping ticks) host time `host_us` maps to, no margin: for a
    /// kernel streaming sweeps (TimingConfig::stream_sweeps), which only needs to know a
    /// report has been pushed.
    pub fn now(&self, host_us: i64) -> Option<u32> {
        Some((host_us * TICKS_PER_US - self.offset()?) as u32)
    }

    /// The current light margin (µs, without the floor).
    pub fn margin_us(&self) -> i64 {
        self.margin_us
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lh_kernel::timing::Pulse;

    fn pulse(end: u32) -> Output {
        Output::Kernel(Input::Pulse { dev: 0, pulse: Pulse { sensor: 0, start: end.wrapping_sub(480), len: 480 } })
    }

    #[test]
    fn maps_by_the_fastest_report_and_keeps_a_light_margin() {
        let mut c = HostClock::new();
        // Device clock starts near the wrap; reports arrive 300 µs after their newest pulse,
        // every 20th one 2 ms late.
        let base: u32 = u32::MAX - 48_000 * 50;
        let host0: i64 = 5_000_000;
        for k in 0..3000i64 {
            let dev = base.wrapping_add((k * 48_000) as u32);
            let late = if k % 20 == 0 { 2_000 } else { 0 };
            let r = c.observe(&[pulse(dev)], host0 + k * 1000 + 300 + late);
            if k > 0 {
                let (l, light) = r.unwrap();
                assert!(light);
                assert_eq!(l, late, "report {}", k);
            }
        }
        // Device time k ms is host time host0 + k ms + the fastest delivery (300 µs).
        let raw = base.wrapping_add(2_990 * 48_000);
        assert_eq!(c.to_host_us(raw), Some(host0 + 2_990_000 + 300));
        // Passed: the margin is the recent worst lateness (2 ms), plus the floor.
        assert!(c.margin_us() > 1_900 && c.margin_us() <= 2_000, "{}", c.margin_us());
        let now = host0 + 2_990_000 + 300;
        let passed = c.passed(now).unwrap();
        let behind = raw.wrapping_sub(passed) as i32 as i64 / TICKS_PER_US;
        assert_eq!(behind, c.margin_us() + MARGIN_FLOOR_US);
    }
}

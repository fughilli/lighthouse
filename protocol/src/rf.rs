//! Watchman RF dongle input: reports 0x23 (one 29-byte sub-packet) and 0x24
//! (two), plus 0x26 (tracker off). docs/lighthouse-protocol.md §4.2, §5.3–5.6.
//!
//! Sub-packet: `[t_hi] [size] [t_lo] [flags] blocks…`, where time16 = t_hi<<8 |
//! t_lo is device clock bits 31..16 and the payload is bytes 3 .. 3+size-1.
//! Blocks, in order, as flagged: IMU (0x80, 13 B), metadata (0x40), input
//! (0x20), light (0x10, the rest). Bytes past the payload are stale.

use alloc::vec;
use alloc::vec::Vec;

use crate::{Lightcap, RawImu, Sample};

#[derive(Clone, Debug, Default)]
pub struct RfDecoder {
    /// Timecode of the last IMU sample: the epoch reference for light.
    last_imu: Option<u32>,
}

impl RfDecoder {
    pub fn decode(&mut self, report: &[u8], out: &mut Vec<Sample>) {
        match report.first() {
            Some(0x23) if report.len() >= 30 => self.subpacket(&report[1..30], out),
            Some(0x24) if report.len() >= 59 => {
                self.subpacket(&report[1..30], out);
                self.subpacket(&report[30..59], out);
            }
            Some(0x26) => out.push(Sample::TurnedOff),
            _ => out.push(Sample::Malformed("rf report")),
        }
    }

    fn subpacket(&mut self, sp: &[u8], out: &mut Vec<Sample>) {
        let time16 = ((sp[0] as u32) << 8) | sp[2] as u32;
        let size = sp[1].wrapping_sub(1) as usize;
        let pend = (3 + size).min(sp.len());
        if pend <= 3 {
            return;
        }
        let mut p = 3;
        let flags = sp[p];
        p += 1;
        if flags == 0xe2 {
            return; // heartbeat
        }
        if flags & 0x80 != 0 {
            if pend < p + 13 {
                out.push(Sample::Malformed("rf imu block short"));
                return;
            }
            let i16_at = |i: usize| i16::from_le_bytes([sp[i], sp[i + 1]]);
            let tc = (time16 << 16) | ((sp[p] as u32) << 8);
            out.push(Sample::Imu(RawImu {
                timecode: tc,
                accel: [i16_at(p + 1), i16_at(p + 3), i16_at(p + 5)],
                gyro: [i16_at(p + 7), i16_at(p + 9), i16_at(p + 11)],
            }));
            self.last_imu = Some(tc);
            p += 13;
        }
        if flags & 0x40 != 0 {
            let Some(&marker) = sp.get(p) else { return };
            p += 1;
            match marker {
                0x80 => {
                    let b = sp[p];
                    p += 1;
                    if b != 0 {
                        out.push(Sample::Battery { percent: b & 0x7f, charging: b & 0x80 != 0 });
                    }
                }
                0x20 => p += 1,
                0x30 => p += 2,
                _ => {
                    out.push(Sample::Malformed("rf metadata marker"));
                    return;
                }
            }
        }
        if flags & 0x20 != 0 {
            match input_block(sp, p, pend, out) {
                Some(np) => p = np,
                None => return,
            }
        }
        if flags & 0x10 != 0 {
            match read_light(time16, sp, p, pend, self.last_imu) {
                Ok(v) => out.extend(v.into_iter().filter(|l| l.length != 0).map(Sample::Light)),
                Err(e) => out.push(Sample::Malformed(e)),
            }
        }
    }
}

/// RF input block (buttons; Knuckles extras). Returns the position after it.
fn input_block(sp: &[u8], mut p: usize, pend: usize, out: &mut Vec<Sample>) -> Option<usize> {
    let f = *sp.get(p)?;
    p += 1;
    if f & 7 != 0 {
        if f & 1 != 0 {
            out.push(Sample::Buttons { buttons: *sp.get(p)? as u32 });
            p += 1;
        }
        if f & 4 != 0 {
            p += 1;
        }
        if f & 2 != 0 {
            p += 4;
        }
    } else {
        match sp.get(p)? {
            0xa1 => p += 1 + 1 + 6,
            _ => {
                out.push(Sample::Malformed("rf input type"));
                return None;
            }
        }
    }
    (p <= pend).then_some(p)
}

/// The light block `[sensor bytes][deltas, reverse VLQ][u24 end time]`
/// (libsurvive's read_light_data). Pulses come out in ascending start time.
pub fn read_light(time16: u32, buf: &[u8], p0: usize, pend: usize, last_imu: Option<u32>) -> Result<Vec<Lightcap>, &'static str> {
    if pend <= p0 + 3 {
        return Ok(Vec::new());
    }
    let ids = p0;
    let mut ev = pend - 4;
    let mut t: u32 = ((time16 >> 8) << 24) | (buf[ev + 3] as u32) << 16 | (buf[ev + 2] as u32) << 8 | buf[ev + 1] as u32;
    // The header's top byte can be one epoch off near a 2^24 rollover: take
    // the epoch nearest the last IMU time.
    if let Some(r) = last_imu {
        if t > r && t - r > 1 << 23 {
            t = t.wrapping_sub(1 << 24);
        } else if r > t && r - t > 1 << 23 {
            t = t.wrapping_add(1 << 24);
        }
    }
    // Edge times, newest first.
    let mut times: Vec<u32> = vec![t];
    'outer: while ids + (times.len() - 1) / 2 < ev {
        let mut d: u32 = 0;
        let start = ev;
        loop {
            let b = buf[ev];
            d = (d << 7) | (b & 0x7f) as u32;
            ev -= 1;
            if b & 0x80 != 0 {
                break;
            }
            if ids + (times.len() - 1) / 2 > ev {
                ev = start;
                break 'outer;
            }
        }
        t = t.wrapping_sub(d);
        if times.len() >= 16 {
            return Err("rf light: too many edges");
        }
        times.push(t);
    }
    let mut ti = times.len() - 1;
    if ti % 2 == 0 && ti > 0 {
        // An odd number of edges: give the last delta back.
        ti -= 1;
        times.pop();
        loop {
            ev += 1;
            if buf[ev] & 0x80 != 0 {
                break;
            }
        }
    }
    let mut ids = ids;
    if ev + 1 - ids > ti / 2 + 1 {
        ids = ev - ti / 2;
    }
    let count = (ti + 1) / 2;
    let sensors: Vec<(u8, usize)> = (0..=ti / 2).map(|i| ((buf[ids + i] >> 3) & 0x1f, (buf[ids + i] & 7) as usize)).collect();

    // Pulse i ends at the newest unclaimed edge and starts `edges + 1` edges
    // further back (edges = other edges inside it).
    let mut edge: Vec<Option<u32>> = times.iter().map(|t| Some(*t)).collect();
    edge.resize(16, None);
    let mut order: [Option<usize>; 16] = [None; 16];
    let mut pulses: Vec<Lightcap> = Vec::with_capacity(count);
    let mut k: isize = -1;
    for i in 0..count {
        loop {
            k += 1;
            if k as usize >= 16 {
                return Err("rf light: no end edge");
            }
            if edge[k as usize].is_some() {
                break;
            }
        }
        let s = k as usize + sensors[i].1 + 1;
        if s >= 16 || edge[s].is_none() {
            return Err("rf light: start edge out of range");
        }
        if order[s].is_some() {
            return Err("rf light: start edge claimed twice");
        }
        order[s] = Some(i);
        let (end, start) = (edge[k as usize].unwrap(), edge[s].unwrap());
        pulses.push(Lightcap { sensor: sensors[i].0, timestamp: start, length: end.wrapping_sub(start) as u16 });
        edge[s] = None;
    }
    // Newest start first, then reversed: ascending start time.
    let mut out: Vec<Lightcap> = order.iter().filter_map(|o| o.map(|i| pulses[i])).take(10).collect();
    out.reverse();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        s.split_whitespace().map(|b| u8::from_str_radix(b, 16).unwrap()).collect()
    }

    #[test]
    fn worked_example_from_the_spec() {
        // docs/lighthouse-protocol.md §5.5, capture 131201.
        let mut r = hex("23 80 18 80 10 42 2a 32 98 87 18 93 0a 0d b4 0b 9c 02 b6 02 a7 01 97 02 31 0a 85 ce c8");
        r.resize(30, 0); // the spec trims the stale tail
        let mut d = RfDecoder::default();
        let mut out = Vec::new();
        d.decode(&r, &mut out);
        let got: Vec<(u8, u32, u16)> =
            out.iter().filter_map(|s| if let Sample::Light(l) = s { Some((l.sensor, l.timestamp, l.length)) } else { None }).collect();
        assert_eq!(got, vec![(19, 2155982675, 3079), (6, 2156200045, 2054), (5, 2156201505, 761), (8, 2156201789, 756)]);
    }
}

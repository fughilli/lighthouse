//! Wired tracker input reports (docs/lighthouse-protocol.md §4.1, §5.1, §6.1).

use alloc::vec::Vec;

use crate::{Lightcap, RawImu, Sample};

/// IMU interface, report 0x20: a ring of three samples
/// {i16 accel[3], i16 gyro[3], u32 timecode, u8 seq}; each report re-sends
/// recent ones. New samples are emitted once each, in sequence order (not in
/// slot order: libsurvive's slot-order rule could drop samples if the device
/// fills the ring newest first).
#[derive(Clone, Debug, Default)]
pub struct UsbImu {
    last_seq: Option<u8>,
}

impl UsbImu {
    pub fn decode(&mut self, report: &[u8], out: &mut Vec<Sample>) {
        if report.len() < 52 || report[0] != 0x20 {
            out.push(Sample::Malformed("usb imu report"));
            return;
        }
        let mut slots: Vec<(u8, RawImu)> = (0..3)
            .map(|k| {
                let o = 1 + 17 * k;
                let i16_at = |i: usize| i16::from_le_bytes([report[i], report[i + 1]]);
                let s = RawImu {
                    accel: [i16_at(o), i16_at(o + 2), i16_at(o + 4)],
                    gyro: [i16_at(o + 6), i16_at(o + 8), i16_at(o + 10)],
                    timecode: u32::from_le_bytes([report[o + 12], report[o + 13], report[o + 14], report[o + 15]]),
                };
                (report[o + 16], s)
            })
            .collect();
        let base = self.last_seq.unwrap_or_else(|| slots.iter().map(|s| s.0).min().unwrap().wrapping_sub(1));
        slots.retain(|s| (s.0.wrapping_sub(base) as i8) > 0);
        slots.sort_by_key(|s| s.0.wrapping_sub(base) as i8);
        for (seq, s) in slots {
            self.last_seq = Some(seq);
            out.push(Sample::Imu(s));
        }
    }
}

/// Lightcap interface, report 0x25 (LH1): up to 9 × {u8 sensor, u16 length,
/// u32 timestamp}; sensor 0xfe/0xff = empty.
pub fn decode_lightcap(report: &[u8], out: &mut Vec<Sample>) {
    if report.is_empty() || report[0] != 0x25 {
        return;
    }
    for k in 0..9 {
        let o = 1 + 7 * k;
        if o + 7 > report.len() {
            break;
        }
        let sensor = report[o];
        if sensor > 0xfd {
            continue;
        }
        let length = u16::from_le_bytes([report[o + 1], report[o + 2]]);
        let timestamp = u32::from_le_bytes([report[o + 3], report[o + 4], report[o + 5], report[o + 6]]);
        if length > 0 {
            out.push(Sample::Light(Lightcap { sensor, timestamp, length }));
        }
    }
}

/// Buttons interface, report 0x01: user input (0x0100) and battery (0x0400).
pub fn decode_buttons(report: &[u8], out: &mut Vec<Sample>) {
    if report.len() < 16 || report[0] != 0x01 {
        return;
    }
    match u16::from_le_bytes([report[1], report[2]]) {
        0x0100 if report.len() >= 12 => {
            out.push(Sample::Buttons { buttons: u32::from_le_bytes([report[8], report[9], report[10], report[11]]) })
        }
        0x0400 => {
            let b = report[14];
            out.push(Sample::Battery { percent: b & 0x7f, charging: b & 0x80 != 0 });
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn imu_report(slots: [(u8, u32, i16); 3]) -> Vec<u8> {
        let mut r = vec![0x20];
        for (seq, tc, ax) in slots {
            let mut s = Vec::new();
            for v in [ax, 0, 1000, 1, 2, 3] {
                s.extend_from_slice(&v.to_le_bytes());
            }
            s.extend_from_slice(&tc.to_le_bytes());
            s.push(seq);
            r.extend_from_slice(&s);
        }
        r
    }

    #[test]
    fn imu_ring_emits_each_sample_once_in_order() {
        let mut d = UsbImu::default();
        let mut out = Vec::new();
        // Newest first in the ring: 3, 2, 1, then 5, 4, 3.
        d.decode(&imu_report([(3, 300, 3), (2, 200, 2), (1, 100, 1)]), &mut out);
        d.decode(&imu_report([(5, 500, 5), (4, 400, 4), (3, 300, 3)]), &mut out);
        let tcs: Vec<u32> = out.iter().map(|s| if let Sample::Imu(i) = s { i.timecode } else { 0 }).collect();
        assert_eq!(tcs, vec![100, 200, 300, 400, 500]);
    }

    #[test]
    fn lightcap_layout() {
        let mut r = vec![0x25];
        r.extend_from_slice(&[13, 0xe8, 0x03, 0x44, 0x33, 0x22, 0x11]);
        r.extend_from_slice(&[0xff; 7 * 8]);
        let mut out = Vec::new();
        decode_lightcap(&r, &mut out);
        assert_eq!(out, vec![Sample::Light(Lightcap { sensor: 13, length: 1000, timestamp: 0x11223344 })]);
    }
}

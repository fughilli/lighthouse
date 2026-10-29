//! OOTX: the slow data channel a Lighthouse 1.0 station sends one bit per sync
//! flash, carrying its id, mode and factory calibration.
//!
//! Framing: a preamble of 17 zeros and a 1; then 16-bit words, MSB first, each
//! followed by a sync bit that must be 1. The first word is the payload length
//! in bytes (little-endian: its two bytes as they arrive); the payload follows,
//! padded to an even length, then a CRC-32 (zlib polynomial) of the payload.

use crate::lh1::AxisCal;
use alloc::vec::Vec;

/// What a station tells us about itself (libsurvive's lighthouse_info_v6).
#[derive(Clone, Debug, PartialEq)]
pub struct StationInfo {
    pub fw_version: u16,
    pub id: u32,
    pub cal: [AxisCal; 2],
    pub unlock_count: u8,
    pub hw_version: u8,
    /// Gravity as the station's accelerometer sees it, in the station frame
    /// (up is +y for an upright station). Coarse: 8 bits per axis.
    pub accel_dir: [i8; 3],
    /// 0 = A, 1 = B, 2 = C.
    pub mode: u8,
    pub faults: u8,
}

#[derive(Debug, PartialEq)]
pub enum OotxEvent {
    Packet(StationInfo),
    /// A sync bit was 0 inside a frame, or the CRC didn't match: the bit
    /// stream is wrong (misread syncs, or a wrong timing lock).
    FramingError,
}

#[derive(Clone, Debug, Default)]
pub struct OotxDecoder {
    zeros: u32,
    in_frame: bool,
    bits_in_word: u8,
    word: u16,
    bytes: Vec<u8>,
    expected: Option<usize>,
}

impl OotxDecoder {
    pub fn new() -> OotxDecoder {
        OotxDecoder::default()
    }

    pub fn push(&mut self, bit: bool) -> Option<OotxEvent> {
        let ev = self.push_framed(bit);
        // A preamble can start at any time, even inside a frame.
        if bit {
            if self.zeros >= 17 {
                self.start_frame();
                self.zeros = 0;
                return ev;
            }
            self.zeros = 0;
        } else {
            self.zeros += 1;
        }
        ev
    }

    fn start_frame(&mut self) {
        self.in_frame = true;
        self.bits_in_word = 0;
        self.word = 0;
        self.bytes.clear();
        self.expected = None;
    }

    fn push_framed(&mut self, bit: bool) -> Option<OotxEvent> {
        if !self.in_frame {
            return None;
        }
        if self.bits_in_word == 16 {
            // Sync bit after each word.
            self.bits_in_word = 0;
            if !bit {
                self.in_frame = false;
                // 17 zeros in a row is the next preamble, not an error.
                return if self.zeros >= 16 { None } else { Some(OotxEvent::FramingError) };
            }
            return None;
        }
        self.word = (self.word << 1) | bit as u16;
        self.bits_in_word += 1;
        if self.bits_in_word < 16 {
            return None;
        }
        self.bytes.push((self.word >> 8) as u8);
        self.bytes.push(self.word as u8);
        self.word = 0;
        if self.expected.is_none() && self.bytes.len() >= 2 {
            let len = u16::from_le_bytes([self.bytes[0], self.bytes[1]]) as usize;
            if len > 64 {
                self.in_frame = false;
                return Some(OotxEvent::FramingError);
            }
            self.expected = Some(2 + len + (len & 1) + 4);
        }
        match self.expected {
            Some(n) if self.bytes.len() >= n => {
                self.in_frame = false;
                let len = u16::from_le_bytes([self.bytes[0], self.bytes[1]]) as usize;
                let payload = &self.bytes[2..2 + len];
                let at = 2 + len + (len & 1);
                let crc = u32::from_le_bytes([self.bytes[at], self.bytes[at + 1], self.bytes[at + 2], self.bytes[at + 3]]);
                if crc32(payload) != crc {
                    return Some(OotxEvent::FramingError);
                }
                Some(match parse_v6(payload) {
                    Some(info) => OotxEvent::Packet(info),
                    None => OotxEvent::FramingError,
                })
            }
            _ => None,
        }
    }
}

pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xedb8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

/// IEEE half precision to f64.
pub fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f64;
    match exp {
        0 => sign * frac * libm::pow(2.0, -24.0),
        31 => sign * f64::INFINITY,
        e => sign * (1.0 + frac / 1024.0) * libm::pow(2.0, (e - 15) as f64),
    }
}

fn parse_v6(p: &[u8]) -> Option<StationInfo> {
    if p.len() < 33 {
        return None;
    }
    let u16_at = |i: usize| u16::from_le_bytes([p[i], p[i + 1]]);
    let f = |i: usize| f16_to_f64(u16_at(i));
    Some(StationInfo {
        fw_version: u16_at(0),
        id: u32::from_le_bytes([p[2], p[3], p[4], p[5]]),
        cal: [
            AxisCal { phase: f(6), tilt: f(10), curve: f(16), gibphase: f(23), gibmag: f(27) },
            AxisCal { phase: f(8), tilt: f(12), curve: f(18), gibphase: f(25), gibmag: f(29) },
        ],
        unlock_count: p[14],
        hw_version: p[15],
        accel_dir: [p[20] as i8, p[21] as i8, p[22] as i8],
        mode: p[31],
        faults: p[32],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(payload: &[u8]) -> Vec<bool> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(payload.len() as u16).to_le_bytes());
        bytes.extend_from_slice(payload);
        if payload.len() & 1 == 1 {
            bytes.push(0);
        }
        bytes.extend_from_slice(&crc32(payload).to_le_bytes());
        let mut bits = Vec::new();
        for _ in 0..17 {
            bits.push(false);
        }
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

    #[test]
    fn crc_matches_zlib() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn half_floats() {
        assert_eq!(f16_to_f64(0x3c00), 1.0);
        assert_eq!(f16_to_f64(0xc000), -2.0);
        assert!((f16_to_f64(0x2e66) - 0.0999755859375).abs() < 1e-12);
    }

    #[test]
    fn decodes_a_packet_after_noise() {
        let mut payload = [0u8; 33];
        payload[2..6].copy_from_slice(&1416427640u32.to_le_bytes());
        payload[6..8].copy_from_slice(&0x2700u16.to_le_bytes()); // phase0 ~ 0.0273
        payload[20] = 1;
        payload[21] = 127;
        payload[31] = 2;
        let mut d = OotxDecoder::new();
        let mut got = None;
        let noise = [true, false, true, true, false];
        for &b in noise.iter().chain(frame(&payload).iter()) {
            if let Some(e) = d.push(b) {
                got = Some(e);
            }
        }
        match got {
            Some(OotxEvent::Packet(info)) => {
                assert_eq!(info.id, 1416427640);
                assert_eq!(info.accel_dir, [1, 127, 0]);
                assert_eq!(info.mode, 2);
                assert!((info.cal[0].phase - 0.02734375).abs() < 1e-9);
            }
            other => panic!("{:?}", other),
        }
    }

    #[test]
    fn flags_a_broken_sync_bit() {
        let payload = [7u8; 33];
        let mut bits = frame(&payload);
        bits[18 + 16] = false; // first word's sync bit
        let mut d = OotxDecoder::new();
        let evs: Vec<_> = bits.into_iter().filter_map(|b| d.push(b)).collect();
        assert_eq!(evs, alloc::vec![OotxEvent::FramingError]);
    }
}

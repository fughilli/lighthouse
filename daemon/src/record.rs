//! Raw traffic recording, in the format vive_web's recorder uses (so
//! lh_replay, lh_protocheck and the replay transport read it):
//!   `<t> <dev> HIDIN <interface> <host_us> <len> <hex>`
//!   `<t> <dev> HIDFEAT <GET|SET> <host_us> <rtn> <hex>`
//! Feature exchanges happen at device startup, usually before a recording
//! starts; all of them are kept and written at the top of each recording.

use flate2::write::GzEncoder;
use flate2::Compression;
use std::fs::File;
use std::io::Write;
use std::time::Instant;

pub struct Recorder {
    file: Option<GzEncoder<File>>,
    start: Instant,
    header: Vec<String>,
    lines: u64,
}

impl Default for Recorder {
    fn default() -> Recorder {
        Recorder { file: None, start: Instant::now(), header: Vec::new(), lines: 0 }
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{:02x}", x)).collect()
}

impl Recorder {
    pub fn start(&mut self, path: &str) -> std::io::Result<()> {
        self.stop()?;
        let mut f = GzEncoder::new(File::create(path)?, Compression::new(3));
        for l in &self.header {
            writeln!(f, "{}", l)?;
        }
        self.file = Some(f);
        self.lines = 0;
        Ok(())
    }

    pub fn stop(&mut self) -> std::io::Result<u64> {
        if let Some(f) = self.file.take() {
            f.finish()?;
        }
        Ok(self.lines)
    }

    fn t(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn host_us(&self) -> u64 {
        self.start.elapsed().as_micros() as u64
    }

    pub fn report(&mut self, dev: &str, iface: &str, host_us: u64, data: &[u8]) {
        if let Some(f) = self.file.as_mut() {
            let line = format!("{:.6} {} HIDIN {} {} {} {}", self.start.elapsed().as_secs_f64(), dev, iface, host_us, data.len(), hex(data));
            if writeln!(f, "{}", line).is_ok() {
                self.lines += 1;
            }
        }
    }

    pub fn feature(&mut self, dev: &str, kind: &str, rtn: i64, data: &[u8]) {
        let line = format!("{:.6} {} HIDFEAT {} {} {} {}", self.t(), dev, kind, self.host_us(), rtn, hex(data));
        if self.header.len() < 20_000 {
            self.header.push(line.clone());
        }
        if let Some(f) = self.file.as_mut() {
            if writeln!(f, "{}", line).is_ok() {
                self.lines += 1;
            }
        }
    }
}

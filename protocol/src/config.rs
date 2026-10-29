//! Device config readout and parsing (docs/lighthouse-protocol.md §3).
//!
//! Readout: GET 0x10 answers `10 LL LL` (expected length, big-endian), then
//! repeated GET 0x11 answers `11 n data[n]` until n == 0. The data is a zlib
//! stream of JSON (the announced length excludes the 2-byte zlib header).

use crate::imu::ImuCalibration;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use lh_kernel::math::{Pose, Vec3};
use lh_kernel::model::TrackerModel;
use serde_json::Value;

/// Collects the config chunks.
#[derive(Clone, Debug, Default)]
pub struct ConfigReader {
    expected: Option<usize>,
    data: Vec<u8>,
}

pub enum ReadStep {
    /// Send GET 0x11 next.
    More,
    /// Complete: the inflated JSON.
    Done(Vec<u8>),
    /// Start again with GET 0x10.
    Restart(&'static str),
    /// An empty chunk before the announced length: not ready yet, send the
    /// same GET 0x11 again shortly.
    Short { got: usize, expected: usize },
}

impl ConfigReader {
    /// Reply to GET 0x10.
    pub fn start(&mut self, reply: &[u8]) -> ReadStep {
        self.data.clear();
        if reply.len() < 3 || reply[0] != 0x10 {
            return ReadStep::Restart("bad 0x10 reply");
        }
        self.expected = Some(((reply[1] as usize) << 8) | reply[2] as usize);
        ReadStep::More
    }

    /// Reply to GET 0x11.
    pub fn chunk(&mut self, reply: &[u8]) -> ReadStep {
        if reply.len() < 2 || reply[0] != 0x11 {
            return ReadStep::Restart("bad 0x11 reply");
        }
        let n = reply[1] as usize;
        if n > 0 {
            if reply.len() < 2 + n {
                return ReadStep::Restart("short 0x11 reply");
            }
            self.data.extend_from_slice(&reply[2..2 + n]);
            return ReadStep::More;
        }
        // Before the announced length an empty chunk means "not yet" (a dongle
        // is still fetching from its tracker over the radio): ask again.
        if self.data.len() < self.expected.unwrap_or(0) {
            return ReadStep::Short { got: self.data.len(), expected: self.expected.unwrap_or(0) };
        }
        match miniz_oxide::inflate::decompress_to_vec_zlib(&self.data) {
            Ok(json) => ReadStep::Done(json),
            Err(_) => ReadStep::Restart("config doesn't inflate"),
        }
    }
}

/// What the kernel and the IMU scaling need from a device's JSON config.
#[derive(Clone, Debug)]
pub struct DeviceConfig {
    pub serial: String,
    pub model_number: String,
    pub device_class: String,
    /// Raw sensor id → model index (identity when absent).
    pub channel_map: Vec<Option<u8>>,
    pub model: TrackerModel,
    pub imu: ImuCalibration,
}

fn vec3(v: &Value) -> Option<Vec3> {
    let a = v.as_array()?;
    Some(Vec3::new(a.first()?.as_f64()?, a.get(1)?.as_f64()?, a.get(2)?.as_f64()?))
}

fn arr3(v: Option<&Value>, default: [f64; 3]) -> [f64; 3] {
    v.and_then(vec3).map(|v| v.to_array()).unwrap_or(default)
}

/// libsurvive's solve_vive_pose: columns [plus_x, plus_z × plus_x, plus_z],
/// translation `position`; the pose maps the frame into trackref.
fn frame(v: Option<&Value>) -> Pose {
    let Some(v) = v else { return Pose::IDENTITY };
    let (Some(x), Some(z)) = (v.get("plus_x").and_then(vec3), v.get("plus_z").and_then(vec3)) else { return Pose::IDENTITY };
    let x = x.normalized();
    let z = z.normalized();
    let y = z.cross(x).normalized();
    let pos = v.get("position").and_then(vec3).unwrap_or(Vec3::ZERO);
    Pose { pos, rot: lh_kernel::solve::quat_from_columns(x, y, z) }
}

/// Sensor-position scale per model (libsurvive: survive_default_devices.c).
fn sensor_scale(model_number: &str) -> f64 {
    match model_number {
        "VIVE Tracker 3.0 MV" => 1.0034,
        "Vive Tracker MV" | "Vive. Tracker MV" => 1.00585,
        _ => 1.0,
    }
}

pub fn parse(json: &[u8]) -> Result<DeviceConfig, String> {
    let v: Value = serde_json::from_slice(json).map_err(|e| e.to_string())?;
    let s = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
    let model_number = s("model_number");
    let lc = v.get("lighthouse_config").ok_or("no lighthouse_config")?;
    let pts = lc.get("modelPoints").and_then(|x| x.as_array()).ok_or("no modelPoints")?;
    let nrm = lc.get("modelNormals").and_then(|x| x.as_array()).ok_or("no modelNormals")?;
    let scale = sensor_scale(&model_number);
    // Body frame = IMU frame (accelerometer, gyro and solver agree on it).
    let trackref_from_imu = frame(v.get("imu"));
    let trackref_from_head = frame(v.get("head"));
    let imu_from_trackref = trackref_from_imu.inverse();
    let model = TrackerModel {
        sensors: pts.iter().filter_map(vec3).map(|p| imu_from_trackref.apply(p.scale(scale))).collect(),
        normals: nrm.iter().filter_map(vec3).map(|n| imu_from_trackref.rot.rotate(n)).collect(),
        body_from_head: imu_from_trackref.compose(&trackref_from_head),
    };
    let mut channel_map = vec![None; 32];
    match lc.get("channelMap").and_then(|x| x.as_array()) {
        Some(m) => {
            for (index, raw) in m.iter().enumerate() {
                if let Some(r) = raw.as_u64() {
                    if (r as usize) < channel_map.len() {
                        channel_map[r as usize] = Some(index as u8);
                    }
                }
            }
        }
        None => {
            for i in 0..model.sensors.len().min(32) {
                channel_map[i] = Some(i as u8);
            }
        }
    }
    let imu_v = v.get("imu");
    let mut imu = ImuCalibration::default();
    let bias = arr3(imu_v.and_then(|i| i.get("acc_bias")), [0.0; 3]);
    imu.acc_bias = [bias[0] / 1000.0, bias[1] / 1000.0, bias[2] / 1000.0];
    imu.acc_scale = arr3(imu_v.and_then(|i| i.get("acc_scale")), [1.0; 3]);
    imu.gyro_bias = arr3(imu_v.and_then(|i| i.get("gyro_bias")), [0.0; 3]);
    imu.gyro_scale = arr3(imu_v.and_then(|i| i.get("gyro_scale")), [1.0; 3]);
    Ok(DeviceConfig {
        serial: s("device_serial_number"),
        model_number,
        device_class: s("device_class"),
        channel_map,
        model,
        imu,
    })
}

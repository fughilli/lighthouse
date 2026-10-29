//! The station calibration on disk (JSON), loaded as kernel priors at
//! startup and rewritten after every calibration:
//!   {"version":1,"stations":[{"id":…,"mode":…,"pose":[x,y,z,qw,qx,qy,qz],
//!     "cov":[36 numbers],"cal":[[phase,tilt,curve,gibphase,gibmag],[…]],
//!     "rms":…,"up_error":…}]}

use lh_kernel::kernel::{Event, Input};
use lh_kernel::lh1::AxisCal;
use lh_kernel::linalg::Mat;
use lh_kernel::math::{Pose, Quat, Vec3};
use serde_json::{json, Value};

pub fn load(path: &str) -> Result<Vec<Input>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for s in v["stations"].as_array().ok_or("no stations")? {
        let f = |x: &Value| x.as_f64().unwrap_or(0.0);
        let p: Vec<f64> = s["pose"].as_array().ok_or("pose")?.iter().map(f).collect();
        let cov: Vec<f64> = s["cov"].as_array().ok_or("cov")?.iter().map(f).collect();
        let cal_v = s["cal"].as_array().ok_or("cal")?;
        let axis = |a: &Value| {
            let c: Vec<f64> = a.as_array().map(|x| x.iter().map(f).collect()).unwrap_or_default();
            AxisCal { phase: c[0], tilt: c[1], curve: c[2], gibphase: c[3], gibmag: c[4] }
        };
        out.push(Input::StationPrior {
            id: s["id"].as_u64().ok_or("id")? as u32,
            world_from_station: Pose { pos: Vec3::new(p[0], p[1], p[2]), rot: Quat::new(p[3], p[4], p[5], p[6]).normalized() },
            cov: Mat { rows: 6, cols: 6, d: cov },
            cal: Some([axis(&cal_v[0]), axis(&cal_v[1])]),
        });
    }
    Ok(out)
}

/// The saved stations as a "calibrated" event (reason "Loaded"), so viewers
/// show them before any new calibration.
pub fn loaded_event(path: &str) -> Option<Value> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    let st: Vec<Value> = v["stations"]
        .as_array()?
        .iter()
        .map(|s| {
            let p = s["pose"].as_array().cloned().unwrap_or_default();
            json!({"id":s["id"],"mode":s["mode"],"p":p[..3.min(p.len())],"q":p[3.min(p.len())..],
                   "cov":s["cov"],"rms":s["rms"],"up_error":s["up_error"]})
        })
        .collect();
    Some(json!({"type":"calibrated","reason":"Loaded","stations":st}))
}

pub fn save(path: &str, e: &Event) -> Result<(), String> {
    let Event::Calibrated { stations, .. } = e else { return Ok(()) };
    let st: Vec<Value> = stations
        .iter()
        .map(|s| {
            let p = &s.world_from_station;
            let cal: Vec<Value> = s.cal.iter().map(|c| json!([c.phase, c.tilt, c.curve, c.gibphase, c.gibmag])).collect();
            json!({"id":s.id,"mode":s.mode,"pose":[p.pos.x,p.pos.y,p.pos.z,p.rot.w,p.rot.x,p.rot.y,p.rot.z],
                   "cov":s.cov.d,"cal":cal,"rms":s.rms,"up_error":s.up_error})
        })
        .collect();
    let text = serde_json::to_string_pretty(&json!({"version":1,"stations":st})).map_err(|e| e.to_string())?;
    let tmp = format!("{}.tmp", path);
    std::fs::write(&tmp, text).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, path).map_err(|e| e.to_string())
}

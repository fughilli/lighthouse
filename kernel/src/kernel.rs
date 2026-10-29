//! The kernel: devices' light and IMU in, events out.
//!
//! Per device: [`Timing`] turns pulses into sweeps and decodes each slot's
//! OOTX (which names the station: slot → station id); an IMU stillness
//! detector cuts still stretches into calibration scenes; a [`TrackState`]
//! EKF tracks the device once the stations it sees are calibrated.
//!
//! Stations are global, keyed by OOTX id. Their poses come from a saved
//! calibration ([`Input::StationPrior`]) or are solved here from still scenes
//! — automatically when a station without a pose shows up, and again as new
//! scenes add information. A loaded calibration is checked against every
//! still scene that sees two or more of its stations: if the stations put
//! the tracker in different places a few scenes running, they were moved, so
//! the calibration is dropped and solved again from scratch (once: one solved
//! here is refined by more scenes instead).
//!
//! World frame: z up (gravity, from the trackers' accelerometers and the
//! stations' OOTX accel vectors). Without a prior, the first calibration sets
//! the origin at the tracker's IMU in its first still scene and the x axis
//! along that tracker's body x (levelled). A loaded calibration keeps its frame.

use crate::lh1::AxisCal;
use crate::linalg::{rot_mat, skew, Mat};
use crate::math::{Pose, Quat, Vec3};
use crate::model::{StationModel, TrackerModel};
use crate::ootx::StationInfo;
use crate::solve::{
    bootstrap_pose, bootstrap_pose_assoc, bootstrap_station, calibrate, quat_between, station_disagreement, CandidateObs, Obs, Scene, SolveConfig,
    StationPrior,
};
use crate::timing::{LockLoss, Pulse, SensorHits, Sweep, Timing, TimingConfig, TimingEvent};
use crate::track::{TrackConfig, TrackState, UpdateStats};
use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

pub type DeviceId = u32;

#[derive(Clone, Debug)]
pub enum Input {
    /// A tracker appeared (its config read); `wireless` selects the timing
    /// rules for radio links.
    DeviceAdded { dev: DeviceId, model: TrackerModel, wireless: bool },
    DeviceRemoved { dev: DeviceId },
    Pulse { dev: DeviceId, pulse: Pulse },
    /// Calibrated IMU sample in the body frame: accel in g, gyro in rad/s.
    Imu { dev: DeviceId, time: u32, accel_g: Vec3, gyro: Vec3 },
    /// A saved station calibration (pose in the world frame, 6x6 covariance
    /// over (δp, δθ)); `cal` if known (else from OOTX).
    StationPrior { id: u32, world_from_station: Pose, cov: Mat, cal: Option<[AxisCal; 2]> },
    /// Solve the station calibration again from the scenes collected.
    Recalibrate,
}

#[derive(Clone, Debug)]
pub struct StationReport {
    pub id: u32,
    pub mode: Option<u8>,
    pub world_from_station: Pose,
    pub cov: Mat,
    /// RMS angle residual of its hits in the fit (rad).
    pub rms: f64,
    /// Angle between its OOTX gravity vector and world z (rad).
    pub up_error: f64,
    pub cal: [AxisCal; 2],
}

#[derive(Clone, Debug)]
pub enum Event {
    DeviceAdded { dev: DeviceId },
    DeviceRemoved { dev: DeviceId },
    TimingLocked { dev: DeviceId },
    TimingLost { dev: DeviceId, reason: LockLoss },
    /// First OOTX packet from a station (or a changed one).
    StationSeen { dev: DeviceId, slot: u8, info: StationInfo },
    Still { dev: DeviceId, still: bool },
    /// New station calibration; `reason` says why it ran.
    Calibrated { reason: CalibrationReason, stations: Vec<StationReport>, scenes: usize, dropped: usize, tracker_up_error: f64 },
    /// A still scene where the calibrated stations put the tracker in
    /// different places (largest difference, m and rad); `count` in a row.
    StationsDisagree { dev: DeviceId, pos: f64, rot: f64, count: u32 },
    TrackingStarted { dev: DeviceId, rms: f64 },
    TrackingLost { dev: DeviceId, reason: TrackLoss },
    /// The device's pose ("head" frame) in the world. `cov` is 6x6 over
    /// (position, rotation), both in world axes; `time` device ticks
    /// (unwrapped).
    Pose { dev: DeviceId, time: i64, world_from_head: Pose, world_from_body: Pose, velocity: Vec3, cov: Mat, used: u32, gated: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationReason {
    FirstStations,
    NewStation,
    MoreScenes,
    Requested,
    /// The stations disagreed: the old calibration was dropped.
    StationsMoved,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackLoss {
    Gated,
    NoLight,
    Diverged,
    /// The station calibration was dropped (see [`CalibrationReason::StationsMoved`]).
    StationsMoved,
}

#[derive(Clone, Copy, Debug)]
pub struct KernelConfig {
    pub timing: TimingConfig,
    pub solve: SolveConfig,
    pub track: TrackConfig,
    /// Still: mean gyro norm below (rad/s) and accel norm spread below (g),
    /// over `still_window` seconds.
    pub still_gyro: f64,
    pub still_accel: f64,
    pub still_window: f64,
    /// A still stretch becomes a scene after this long (s); one scene is cut
    /// at most this long.
    pub scene_min: f64,
    pub scene_max: f64,
    pub max_scenes: usize,
    /// Calibration: per station, sensors needed in a scene to use it there.
    pub scene_min_sensors: usize,
    /// Calibration: residual beyond which an observation is dropped (rad).
    pub calib_reject: f64,
    pub auto_calibrate: bool,
    /// Fold each station's calibration covariance into its hits' noise. Off:
    /// the covariance is dominated by the shared world tilt (a gauge freedom),
    /// and as independent per-hit noise it starves the filter. Calibration
    /// uncertainty is reported in [`Event::Calibrated`] instead.
    pub station_cov_in_filter: bool,
    /// While still: zero-velocity and gravity-direction updates, their σ
    /// (m/s, rad).
    pub still_updates: bool,
    pub still_sigma_vel: f64,
    pub still_sigma_up: f64,
    /// Tracking is dropped after this many gated hits in a row.
    pub lost_after_gated: u32,
    /// ... or this long without a used hit (s).
    pub lost_after_dark: f64,
    /// The stations count as moved when single-station poses of a still
    /// tracker differ by more than these (m, rad) in `moved_after` scenes in
    /// a row. A good calibration agrees to ~1 cm.
    pub moved_pos: f64,
    pub moved_rot: f64,
    pub moved_after: u32,
    /// Added to IMU sample times (s): IMU vs light latency.
    pub imu_time_offset: f64,
}

impl Default for KernelConfig {
    fn default() -> KernelConfig {
        KernelConfig {
            timing: TimingConfig::default(),
            solve: SolveConfig::default(),
            track: TrackConfig::default(),
            still_gyro: 0.02,
            still_accel: 0.004,
            still_window: 0.3,
            scene_min: 1.0,
            scene_max: 4.0,
            max_scenes: 16,
            scene_min_sensors: 4,
            calib_reject: 3e-3,
            auto_calibrate: true,
            station_cov_in_filter: false,
            still_updates: true,
            still_sigma_vel: 0.01,
            still_sigma_up: 0.02,
            lost_after_gated: 60,
            lost_after_dark: 0.5,
            moved_pos: 0.05,
            moved_rot: 3f64.to_radians(),
            moved_after: 3,
            imu_time_offset: 0.0,
        }
    }
}

const TPS: f64 = crate::lh1::TICKS_PER_SECOND;

/// Device ticks (u32, wrapping) to a monotonic timeline.
#[derive(Clone, Copy, Debug, Default)]
struct Clock {
    last: Option<u32>,
    hi: i64,
}

impl Clock {
    fn unwrap(&mut self, t: u32) -> i64 {
        let u = match self.last {
            None => t as i64,
            Some(l) => self.hi + t.wrapping_sub(l) as i32 as i64,
        };
        if self.last.is_none() || u > self.hi {
            self.last = Some(t);
            self.hi = u;
        }
        u
    }
}

struct StationEntry {
    id: u32,
    /// From a saved calibration: refinements are pulled toward it.
    prior: Option<StationPrior>,
    info: Option<StationInfo>,
    cal: Option<[AxisCal; 2]>,
    model: Option<StationModel>,
    cov: Option<Mat>,
}

#[derive(Default)]
struct StillDetector {
    gyro: VecDeque<(i64, f64)>,
    acc: VecDeque<(i64, f64)>,
    still: bool,
    since: i64,
}

struct Cluster {
    values: Vec<f64>,
}

impl Cluster {
    fn median(&self) -> f64 {
        let mut v = self.values.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }
}

/// Collects a still stretch: per (station, axis, sensor), clusters of hit
/// angles (direct hit and steady reflections stay apart).
struct SceneBuilder {
    start: i64,
    clusters: Vec<((u32, u8, u8), Vec<Cluster>)>,
    acc_sum: Vec3,
    acc_n: usize,
}

impl SceneBuilder {
    fn add(&mut self, station: u32, axis: u8, hits: &[SensorHits]) {
        for h in hits {
            let key = (station, axis, h.sensor);
            let idx = match self.clusters.iter().position(|c| c.0 == key) {
                Some(i) => i,
                None => {
                    self.clusters.push((key, Vec::new()));
                    self.clusters.len() - 1
                }
            };
            let cl = &mut self.clusters[idx].1;
            for c in &h.candidates {
                match cl.iter_mut().find(|k| libm::fabs(k.values[k.values.len() - 1] - c.angle) < 2e-3) {
                    Some(k) => {
                        if k.values.len() < 1024 {
                            k.values.push(c.angle)
                        }
                    }
                    None => {
                        if cl.len() < 4 {
                            cl.push(Cluster { values: vec![c.angle] })
                        }
                    }
                }
            }
        }
    }
}

/// A finished still scene, stations by id.
#[derive(Clone)]
struct SceneRecord {
    dev: DeviceId,
    up_body: Vec3,
    /// (station id, axis, sensor, angle, count), several per key possible.
    obs: Vec<(u32, u8, u8, f64, usize)>,
    pose: Option<Pose>,
}

struct Device {
    id: DeviceId,
    model: TrackerModel,
    timing: Timing,
    slot_station: [Option<u32>; 2],
    clock: Clock,
    imu: VecDeque<(i64, Vec3, Vec3)>,
    last_imu: Option<(i64, Vec3, Vec3)>,
    still: StillDetector,
    scene: Option<SceneBuilder>,
    track: Option<TrackState>,
    recent: VecDeque<(u32, u8, Vec<SensorHits>)>,
    gated_run: u32,
    last_used: i64,
    window: UpdateStats,
    /// Diagnostics: innovation sum, sum of squares and count per (station, axis).
    innovations: Vec<((u32, u8), f64, f64, u32)>,
    /// Station id → index of its angle-bias states in the tracker filter.
    bias_slots: Vec<u32>,
}

pub struct Kernel {
    pub cfg: KernelConfig,
    devices: Vec<Device>,
    stations: Vec<StationEntry>,
    scenes: Vec<SceneRecord>,
    reference: Option<u32>,
    world_fixed: bool,
    /// Still scenes in a row where the stations disagreed.
    disagreeing: u32,
    /// The calibration was dropped; the next one reports StationsMoved.
    moved: bool,
    /// A calibration was solved from scratch in this session: disagreement
    /// then means it needs more scenes (which refine it), not a reset.
    solved_here: bool,
}

impl Kernel {
    pub fn new(cfg: KernelConfig) -> Kernel {
        Kernel {
            cfg,
            devices: Vec::new(),
            stations: Vec::new(),
            scenes: Vec::new(),
            reference: None,
            world_fixed: false,
            disagreeing: 0,
            moved: false,
            solved_here: false,
        }
    }

    pub fn push(&mut self, input: Input, out: &mut Vec<Event>) {
        match input {
            Input::DeviceAdded { dev, model, wireless } => {
                self.devices.retain(|d| d.id != dev);
                let timing = Timing::new(TimingConfig { wireless, ..self.cfg.timing });
                self.devices.push(Device {
                    id: dev,
                    model,
                    timing,
                    slot_station: [None, None],
                    clock: Clock::default(),
                    imu: VecDeque::new(),
                    last_imu: None,
                    still: StillDetector::default(),
                    scene: None,
                    track: None,
                    recent: VecDeque::new(),
                    gated_run: 0,
                    last_used: 0,
                    window: UpdateStats::default(),
                    innovations: Vec::new(),
                    bias_slots: Vec::new(),
                });
                out.push(Event::DeviceAdded { dev });
            }
            Input::DeviceRemoved { dev } => {
                self.devices.retain(|d| d.id != dev);
                out.push(Event::DeviceRemoved { dev });
            }
            Input::Imu { dev, time, accel_g, gyro } => self.imu(dev, time, accel_g, gyro, out),
            Input::Pulse { dev, pulse } => {
                let Some(di) = self.devices.iter().position(|d| d.id == dev) else { return };
                let mut tev = Vec::new();
                self.devices[di].timing.push(pulse, &mut tev);
                for e in tev {
                    self.timing_event(di, e, out);
                }
            }
            Input::StationPrior { id, world_from_station, cov, cal } => {
                let si = self.station_index(id);
                let st = &mut self.stations[si];
                if let Some(c) = cal {
                    st.cal = Some(c);
                }
                let cal = st.cal.unwrap_or([AxisCal::default(); 2]);
                st.model = Some(StationModel { cal, world_from_station });
                // Prior strength from the saved covariance, floored: a saved
                // calibration is trusted to ~mm / ~mrad at best, and its gauge
                // directions (zero variance) aren't certainty.
                let sd = |i: usize| libm::sqrt(cov[(i, i)].max(0.0));
                let sp = sd(0).max(sd(1)).max(sd(2)).max(2e-3);
                let sr = sd(3).max(sd(4)).max(sd(5)).max(2e-3);
                st.prior = Some(StationPrior { pose: world_from_station, sigma_pos: sp, sigma_rot: sr });
                st.cov = Some(cov);
                self.world_fixed = true;
                if self.reference.is_none() {
                    self.reference = Some(id);
                }
            }
            Input::Recalibrate => self.calibrate(CalibrationReason::Requested, out),
        }
    }

    /// Diagnostics: per (station id, axis) of a device, the mean and RMS
    /// innovation (rad) of the hits used, and their count; then resets.
    pub fn take_innovations(&mut self, dev: DeviceId) -> Vec<(u32, u8, f64, f64, u32)> {
        let Some(d) = self.devices.iter_mut().find(|d| d.id == dev) else { return Vec::new() };
        let v = d
            .innovations
            .iter()
            .map(|e| (e.0 .0, e.0 .1, e.1 / e.3 as f64, libm::sqrt(e.2 / e.3 as f64), e.3))
            .collect();
        d.innovations.clear();
        v
    }

    fn station_index(&mut self, id: u32) -> usize {
        match self.stations.iter().position(|s| s.id == id) {
            Some(i) => i,
            None => {
                self.stations.push(StationEntry { id, prior: None, info: None, cal: None, model: None, cov: None });
                self.stations.len() - 1
            }
        }
    }

    // ------------------------------------------------------------ IMU / still

    fn imu(&mut self, dev: DeviceId, time: u32, accel_g: Vec3, gyro: Vec3, out: &mut Vec<Event>) {
        let cfg = self.cfg;
        let Some(di) = self.devices.iter().position(|d| d.id == dev) else { return };
        let d = &mut self.devices[di];
        let t = d.clock.unwrap(time) + (cfg.imu_time_offset * TPS) as i64;
        d.imu.push_back((t, accel_g, gyro));
        d.last_imu = Some((t, accel_g, gyro));
        while d.imu.len() > 2000 {
            d.imu.pop_front();
        }
        // Stillness over a sliding window.
        let w = (cfg.still_window * TPS) as i64;
        d.still.gyro.push_back((t, gyro.norm()));
        d.still.acc.push_back((t, accel_g.norm()));
        while d.still.gyro.front().map_or(false, |x| x.0 < t - w) {
            d.still.gyro.pop_front();
        }
        while d.still.acc.front().map_or(false, |x| x.0 < t - w) {
            d.still.acc.pop_front();
        }
        let n = d.still.gyro.len().max(1) as f64;
        let g = d.still.gyro.iter().map(|x| x.1).sum::<f64>() / n;
        let am = d.still.acc.iter().map(|x| x.1).sum::<f64>() / n;
        let asd = libm::sqrt(d.still.acc.iter().map(|x| (x.1 - am) * (x.1 - am)).sum::<f64>() / n);
        let full = d.still.gyro.front().map_or(false, |x| x.0 <= t - w * 9 / 10);
        let still = full && g < cfg.still_gyro && asd < cfg.still_accel;
        if still != d.still.still {
            d.still.still = still;
            d.still.since = t;
            out.push(Event::Still { dev, still });
            if !still {
                self.finish_scene(di, out);
                return;
            }
        }
        let d = &mut self.devices[di];
        if still && cfg.still_updates {
            if let Some(track) = d.track.as_mut() {
                if t > track.time {
                    // Bring the filter up to this sample first (it may lag the light).
                    advance(track, &mut d.imu, t, d.last_imu, &cfg.track);
                }
                track.update_still(accel_g, cfg.still_sigma_vel, cfg.still_sigma_up);
            }
        }
        if still {
            if let Some(sb) = d.scene.as_mut() {
                sb.acc_sum = sb.acc_sum.add(accel_g);
                sb.acc_n += 1;
            }
            let dur = (t - d.still.since) as f64 / TPS;
            if d.scene.is_none() && dur >= 0.3 {
                d.scene = Some(SceneBuilder { start: t, clusters: Vec::new(), acc_sum: Vec3::ZERO, acc_n: 0 });
            }
            if d.scene.as_ref().map_or(false, |s| (t - s.start) as f64 / TPS >= cfg.scene_max) {
                self.finish_scene(di, out);
                // Keep collecting the next scene only if this stays still a while.
                let d = &mut self.devices[di];
                d.still.since = t;
            }
        }
    }

    fn finish_scene(&mut self, di: usize, out: &mut Vec<Event>) {
        let d = &mut self.devices[di];
        let Some(sb) = d.scene.take() else { return };
        let dur = d.last_imu.map_or(0, |x| x.0 - sb.start) as f64 / TPS;
        if dur < self.cfg.scene_min || sb.acc_n == 0 {
            return;
        }
        let mut obs = Vec::new();
        for ((st, axis, sensor), cl) in &sb.clusters {
            let top = cl.iter().map(|c| c.values.len()).max().unwrap_or(0);
            let mut ranked: Vec<&Cluster> = cl.iter().filter(|c| c.values.len() >= 8 && c.values.len() * 3 >= top).collect();
            ranked.sort_by(|a, b| b.values.len().cmp(&a.values.len()));
            for c in ranked {
                obs.push((*st, *axis, *sensor, c.median(), c.values.len()));
            }
        }
        let up = sb.acc_sum.scale(1.0 / sb.acc_n as f64);
        let pose = d.track.as_ref().map(|t| t.pose());
        let mut rec = SceneRecord { dev: d.id, up_body: up.normalized(), obs, pose };
        // Only scenes that see some station well.
        let min = self.cfg.scene_min_sensors;
        let usable = rec.obs.iter().map(|o| o.0).any(|st| sensors_seen(&rec, st) >= min);
        if !usable {
            return;
        }
        self.check_stations(&rec, out);
        if self.moved {
            rec.pose = None; // tracked with the dropped calibration
        }
        // The same place as a scene we have (the tracker kept still, or came
        // back): refresh that one; it adds no new information, unless the
        // calibration was dropped and still has to be solved again.
        if let Some(j) = self.scenes.iter().position(|old| same_place(old, &rec)) {
            let pose = self.scenes[j].pose;
            self.scenes[j] = SceneRecord { pose: pose.or(rec.pose), ..rec };
            if self.moved && self.cfg.auto_calibrate {
                self.calibrate(CalibrationReason::StationsMoved, out);
            }
            return;
        }
        if self.scenes.len() >= self.cfg.max_scenes {
            self.scenes.remove(0);
        }
        self.scenes.push(rec);
        if self.cfg.auto_calibrate {
            let needs = self.scenes.last().unwrap().obs.iter().any(|o| {
                let s = self.stations.iter().find(|s| s.id == o.0);
                s.map_or(false, |s| s.model.is_none() && s.info.is_some())
            });
            let reason = if self.moved {
                CalibrationReason::StationsMoved
            } else if !self.stations.iter().any(|s| s.model.is_some()) {
                CalibrationReason::FirstStations
            } else if needs {
                CalibrationReason::NewStation
            } else {
                CalibrationReason::MoreScenes
            };
            if reason != CalibrationReason::MoreScenes || !self.world_fixed || self.scenes.len() >= 2 {
                self.calibrate(reason, out);
            }
        }
    }

    // ------------------------------------------------------------ light

    fn timing_event(&mut self, di: usize, e: TimingEvent, out: &mut Vec<Event>) {
        let dev = self.devices[di].id;
        match e {
            TimingEvent::Locked { .. } => out.push(Event::TimingLocked { dev }),
            TimingEvent::LockLost { reason } => {
                self.devices[di].slot_station = [None, None];
                out.push(Event::TimingLost { dev, reason });
            }
            TimingEvent::Station { slot, info } => {
                let changed = self.devices[di].slot_station[slot as usize] != Some(info.id);
                self.devices[di].slot_station[slot as usize] = Some(info.id);
                let si = self.station_index(info.id);
                let st = &mut self.stations[si];
                let new_info = st.info.as_ref() != Some(&info);
                st.cal = Some(info.cal);
                if let Some(m) = st.model.as_mut() {
                    m.cal = info.cal;
                }
                st.info = Some(info.clone());
                if changed || new_info {
                    out.push(Event::StationSeen { dev, slot, info });
                }
            }
            TimingEvent::OotxError { .. } => {}
            TimingEvent::Sweep(s) => self.sweep(di, s, out),
        }
    }

    fn sweep(&mut self, di: usize, s: Sweep, out: &mut Vec<Event>) {
        let Some(sid) = self.devices[di].slot_station[s.slot as usize] else { return };
        if let Some(sb) = self.devices[di].scene.as_mut() {
            sb.add(sid, s.axis, &s.hits);
        }
        let Some(si) = self.stations.iter().position(|x| x.id == sid) else { return };
        let Some(station) = self.stations[si].model.clone() else { return };
        let station_cov = self.stations[si].cov.clone();
        let cfg = self.cfg;
        let d = &mut self.devices[di];

        if d.track.is_none() {
            d.recent.push_back((sid, s.axis, s.hits.clone()));
            while d.recent.len() > 8 {
                d.recent.pop_front();
            }
            self.try_start_tracking(di, out);
            return;
        }

        // Angle-bias states only with two or more calibrated stations: with one,
        // a bias can't be told apart from the tracker moving.
        let calibrated = self.stations.iter().filter(|s| s.model.is_some()).count();
        let d = &mut self.devices[di];
        let station_slot = if calibrated < 2 {
            None
        } else {
            match d.bias_slots.iter().position(|x| *x == sid) {
                Some(i) => Some(i),
                None if d.bias_slots.len() < crate::track::MAX_STATIONS => {
                    d.bias_slots.push(sid);
                    Some(d.bias_slots.len() - 1)
                }
                None => None,
            }
        };
        // Hits in time order; the filter follows the IMU up to each.
        let mut hits: Vec<(i64, SensorHits)> = Vec::new();
        for h in &s.hits {
            let t = d.clock.unwrap(h.candidates[0].time);
            hits.push((t, h.clone()));
        }
        hits.sort_by_key(|h| h.0);
        let mut stats = UpdateStats::default();
        {
            let track = d.track.as_mut().unwrap();
            for (t, h) in &hits {
                advance(track, &mut d.imu, *t, d.last_imu, &cfg.track);
                let before = stats.used;
                track.update_hit(
                    &d.model,
                    &station,
                    if cfg.station_cov_in_filter { station_cov.as_ref() } else { None },
                    station_slot,
                    s.axis,
                    h,
                    &cfg.track,
                    &mut stats,
                );
                if stats.used > before {
                    let r = stats.last_innovation;
                    match d.innovations.iter_mut().find(|e| e.0 == (sid, s.axis)) {
                        Some(e) => {
                            e.1 += r;
                            e.2 += r * r;
                            e.3 += 1;
                        }
                        None => d.innovations.push(((sid, s.axis), r, r * r, 1)),
                    }
                }
            }
        }
        d.window.used += stats.used;
        d.window.gated += stats.gated;
        if stats.used > 0 {
            d.gated_run = 0;
            d.last_used = hits.last().map_or(d.last_used, |h| h.0);
        } else {
            d.gated_run += stats.gated;
        }
        let track = d.track.as_ref().unwrap();
        let now = hits.last().map_or(track.time, |h| h.0);
        let dark = (now - d.last_used) as f64 / TPS;
        let trace: f64 = (0..3).map(|i| track.cov[(i, i)]).sum();
        let loss = if d.gated_run >= cfg.lost_after_gated {
            Some(TrackLoss::Gated)
        } else if dark > cfg.lost_after_dark {
            Some(TrackLoss::NoLight)
        } else if !(trace < 1.0) {
            Some(TrackLoss::Diverged)
        } else {
            None
        };
        if let Some(reason) = loss {
            d.track = None;
            d.recent.clear();
            out.push(Event::TrackingLost { dev: d.id, reason });
            return;
        }
        // Publish at the newest IMU time (a propagated copy; the filter stays
        // at the light).
        let mut pubst = track.clone();
        if let Some(last) = d.last_imu {
            let mut q: VecDeque<(i64, Vec3, Vec3)> = d.imu.iter().copied().filter(|x| x.0 > pubst.time).collect();
            advance(&mut pubst, &mut q, last.0, d.last_imu, &cfg.track);
        }
        let (pose, cov) = head_pose(&pubst, &d.model.body_from_head);
        out.push(Event::Pose {
            dev: d.id,
            time: pubst.time,
            world_from_head: pose,
            world_from_body: pubst.pose(),
            velocity: pubst.vel,
            cov,
            used: d.window.used,
            gated: d.window.gated,
        });
        d.window = UpdateStats::default();
    }

    fn try_start_tracking(&mut self, di: usize, out: &mut Vec<Event>) {
        let cfg = self.cfg;
        // Stations involved, in solver order.
        let ids: Vec<u32> = {
            let d = &self.devices[di];
            let mut v: Vec<u32> = Vec::new();
            for r in &d.recent {
                if !v.contains(&r.0) {
                    v.push(r.0);
                }
            }
            v
        };
        let models: Vec<StationModel> = ids
            .iter()
            .filter_map(|id| self.stations.iter().find(|s| s.id == *id).and_then(|s| s.model.clone()))
            .collect();
        if models.len() != ids.len() {
            return;
        }
        let d = &mut self.devices[di];
        // Both axes of some station needed.
        let axes = |id: u32| d.recent.iter().filter(|r| r.0 == id).map(|r| r.1).fold(0u8, |m, a| m | (1 << a));
        if !ids.iter().any(|&id| axes(id) == 3) {
            return;
        }
        let mut cands: Vec<CandidateObs> = Vec::new();
        for (sid, axis, hits) in &d.recent {
            let si = ids.iter().position(|x| x == sid).unwrap();
            for h in hits {
                let angles: Vec<f64> = h.candidates.iter().map(|c| c.angle).collect();
                match cands.iter_mut().find(|c| c.station == si && c.axis == *axis && c.sensor == h.sensor as u16) {
                    Some(c) => {
                        // Same key in a later sweep: keep the newest.
                        c.angles = angles;
                    }
                    None => cands.push(CandidateObs { station: si, axis: *axis, sensor: h.sensor as u16, angles }),
                }
            }
        }
        if cands.len() < 10 {
            return;
        }
        let up = d.last_imu.map(|x| x.1);
        let Some(sol) = bootstrap_pose_assoc(&d.model, &models, &cands, up, 3e-3, &cfg.solve) else { return };
        if !(sol.rms < 2e-3) || sol.inliers < 8 || sol.inliers * 3 < sol.obs * 2 {
            return;
        }
        let t = d.last_imu.map_or(0, |x| x.0);
        d.track = Some(TrackState::new(&sol.world_from_body, &sol.cov.scale(4.0), t, cfg.track.angle_bias_init));
        d.last_used = t;
        d.gated_run = 0;
        d.recent.clear();
        out.push(Event::TrackingStarted { dev: d.id, rms: sol.rms });
    }

    // ------------------------------------------------------------ calibration

    /// Checks a still scene against the calibrated stations it sees; after
    /// `moved_after` disagreeing scenes in a row, drops the calibration.
    fn check_stations(&mut self, rec: &SceneRecord, out: &mut Vec<Event>) {
        let cfg = self.cfg;
        let (ids, known): (Vec<u32>, Vec<StationModel>) = self
            .stations
            .iter()
            .filter(|s| sensors_seen(rec, s.id) >= cfg.scene_min_sensors)
            .filter_map(|s| s.model.clone().map(|m| (s.id, m)))
            .unzip();
        if ids.len() < 2 {
            return;
        }
        let Some(model) = self.devices.iter().find(|d| d.id == rec.dev).map(|d| d.model.clone()) else { return };
        let mut obs = Vec::new();
        for (k, id) in ids.iter().enumerate() {
            for (axis, sensor, angle) in first_obs(rec, *id) {
                obs.push(Obs { station: k, axis, sensor: sensor as u16, angle });
            }
        }
        let Some((pos, rot)) = station_disagreement(&model, &known, &obs, rec.up_body, cfg.scene_min_sensors, &cfg.solve) else { return };
        if pos <= cfg.moved_pos && rot <= cfg.moved_rot {
            self.disagreeing = 0;
            return;
        }
        self.disagreeing += 1;
        out.push(Event::StationsDisagree { dev: rec.dev, pos, rot, count: self.disagreeing });
        if self.disagreeing < cfg.moved_after || self.solved_here {
            return;
        }
        // Forget every station pose (and the world frame they defined) and the
        // scene poses tracked with them; the scenes' angles stay, and the
        // caller's calibration starts over from them.
        for st in &mut self.stations {
            st.model = None;
            st.prior = None;
            st.cov = None;
        }
        for sc in &mut self.scenes {
            sc.pose = None;
        }
        for d in &mut self.devices {
            if d.track.take().is_some() {
                d.recent.clear();
                out.push(Event::TrackingLost { dev: d.id, reason: TrackLoss::StationsMoved });
            }
        }
        self.reference = None;
        self.world_fixed = false;
        self.disagreeing = 0;
        self.moved = true;
    }

    fn calibrate(&mut self, reason: CalibrationReason, out: &mut Vec<Event>) {
        if self.scenes.is_empty() {
            return;
        }
        let cfg = self.cfg;
        // Stations with known factory calibration that scenes see well enough.
        let ids: Vec<u32> = self
            .stations
            .iter()
            .filter(|s| s.cal.is_some())
            .map(|s| s.id)
            .filter(|id| self.scenes.iter().any(|sc| sensors_seen(sc, *id) >= cfg.scene_min_sensors))
            .collect();
        if ids.is_empty() {
            return;
        }
        let reference = match self.reference {
            Some(r) if ids.contains(&r) => r,
            _ => {
                // The station seen with most sensors in the first scene.
                let sc = &self.scenes[0];
                *ids.iter().max_by_key(|id| sensors_seen(sc, **id)).unwrap()
            }
        };
        let ref_idx = ids.iter().position(|x| *x == reference).unwrap();

        // Scene poses: tracked pose if there was one, else solved below.
        let mut scene_poses: Vec<Option<Pose>> = self.scenes.iter().map(|s| s.pose).collect();
        let mut models: Vec<Option<StationModel>> =
            ids.iter().map(|id| self.stations.iter().find(|s| s.id == *id).unwrap().model.clone()).collect();
        let cal_of = |k: &Kernel, id: u32| k.stations.iter().find(|s| s.id == id).unwrap().cal.unwrap();
        let up_of = |k: &Kernel, id: u32| -> Option<Vec3> {
            k.stations.iter().find(|s| s.id == id).unwrap().info.as_ref().map(|i| {
                Vec3::new(i.accel_dir[0] as f64, i.accel_dir[1] as f64, i.accel_dir[2] as f64).normalized()
            })
        };

        // Bootstrap: nothing known at all. The first usable scene defines the
        // world (tracker at the origin, levelled, body x as world x).
        let from_scratch = models.iter().all(|m| m.is_none());
        if from_scratch {
            let first = self.scenes.iter().position(|sc| sensors_seen(sc, reference) >= cfg.scene_min_sensors).unwrap();
            let sc = &self.scenes[first];
            let level = quat_between(sc.up_body, Vec3::new(0.0, 0.0, 1.0));
            // Keep the body x heading: remove yaw introduced by levelling.
            let x = level.rotate(Vec3::new(1.0, 0.0, 0.0));
            let yaw = libm::atan2(x.y, x.x);
            let rot = Quat::from_rotvec(Vec3::new(0.0, 0.0, -yaw)).mul(level).normalized();
            scene_poses[first] = Some(Pose { pos: Vec3::ZERO, rot });
            self.world_fixed = true;
        }
        // Stations without a pose: solve each from a scene with a known pose.
        for (k, id) in ids.iter().enumerate() {
            if models[k].is_some() {
                continue;
            }
            let cand = self.scenes.iter().enumerate().filter(|(j, _)| scene_poses[*j].is_some()).max_by_key(|(_, sc)| sensors_seen(sc, *id));
            let Some((j, sc)) = cand else { continue };
            if sensors_seen(sc, *id) < cfg.scene_min_sensors {
                continue;
            }
            let obs: Vec<Obs> = first_obs(sc, *id).into_iter().map(|(axis, sensor, angle)| Obs { station: 0, axis, sensor: sensor as u16, angle }).collect();
            let model = &self.devices.iter().find(|d| d.id == sc.dev).map(|d| d.model.clone());
            let Some(model) = model else { continue };
            if let Some((st, _)) = bootstrap_station(model, cal_of(self, *id), &scene_poses[j].unwrap(), &obs, up_of(self, *id), &cfg.solve) {
                models[k] = Some(st);
            }
        }
        let known: Vec<usize> = (0..ids.len()).filter(|&k| models[k].is_some()).collect();
        if !known.contains(&ref_idx) {
            return;
        }
        // Scenes without a pose: bootstrap against the known stations.
        let km: Vec<StationModel> = known.iter().map(|&k| models[k].clone().unwrap()).collect();
        for j in 0..self.scenes.len() {
            if scene_poses[j].is_some() {
                continue;
            }
            let sc = &self.scenes[j];
            let Some(model) = self.devices.iter().find(|d| d.id == sc.dev).map(|d| d.model.clone()) else { continue };
            let mut obs = Vec::new();
            for (ki, &k) in known.iter().enumerate() {
                for (axis, sensor, angle) in first_obs(sc, ids[k]) {
                    obs.push(Obs { station: ki, axis, sensor: sensor as u16, angle });
                }
            }
            if let Some(sol) = bootstrap_pose(&model, &km, &obs, Some(sc.up_body), &cfg.solve) {
                if sol.rms < 3e-3 {
                    scene_poses[j] = Some(sol.world_from_body);
                }
            }
        }
        // Joint fit over the known stations and posed scenes (one tracker
        // model per scene's device; scenes from other devices skipped for now).
        let dev0 = self.scenes.iter().position(|_| true).map(|j| self.scenes[j].dev).unwrap();
        let Some(model) = self.devices.iter().find(|d| d.id == dev0).map(|d| d.model.clone()) else { return };
        let mut scenes = Vec::new();
        for (j, sc) in self.scenes.iter().enumerate() {
            let (Some(p), true) = (scene_poses[j], sc.dev == dev0) else { continue };
            let mut obs = Vec::new();
            for (ki, &k) in known.iter().enumerate() {
                for o in sc.obs.iter().filter(|o| o.0 == ids[k]) {
                    obs.push(Obs { station: ki, axis: o.1, sensor: o.2 as u16, angle: o.3 });
                }
            }
            scenes.push(Scene { obs, up_body: sc.up_body, world_from_body: p });
        }
        if scenes.is_empty() {
            return;
        }
        let ref_in_known = known.iter().position(|&k| k == ref_idx).unwrap();
        let ups: Vec<Option<Vec3>> = known.iter().map(|&k| up_of(self, ids[k])).collect();
        let priors: Vec<Option<StationPrior>> =
            known.iter().map(|&k| self.stations.iter().find(|s| s.id == ids[k]).and_then(|s| s.prior)).collect();
        let Some(cal) = calibrate(&model, &km, &ups, &priors, &scenes, ref_in_known, &cfg.solve, cfg.calib_reject) else { return };
        // Quality gate: residuals a few σ at most.
        if cal.rms.iter().any(|r| !(*r < 3e-3)) {
            return;
        }
        let mut reports = Vec::new();
        for (ki, &k) in known.iter().enumerate() {
            let id = ids[k];
            let si = self.station_index(id);
            let st = &mut self.stations[si];
            st.model = Some(cal.stations[ki].clone());
            st.cov = Some(cal.station_cov[ki].clone());
            reports.push(StationReport {
                id,
                mode: st.info.as_ref().map(|i| i.mode),
                world_from_station: cal.stations[ki].world_from_station,
                cov: cal.station_cov[ki].clone(),
                rms: cal.rms[ki],
                up_error: cal.station_up_error[ki],
                cal: cal.stations[ki].cal,
            });
        }
        self.reference = Some(reference);
        self.moved = false;
        self.solved_here |= from_scratch;
        // Scene poses refined by the fit (for the next round).
        let mut it = cal.scenes.iter();
        for (j, sc) in self.scenes.iter_mut().enumerate() {
            if scene_poses[j].is_some() && sc.dev == dev0 {
                sc.pose = it.next().copied();
            }
        }
        out.push(Event::Calibrated {
            reason,
            stations: reports,
            scenes: scenes.len(),
            dropped: cal.dropped,
            tracker_up_error: cal.tracker_up_error,
        });
    }
}

/// Two scenes of one device with nearly the same angles on the sensors they
/// share: the tracker was in the same place.
fn same_place(a: &SceneRecord, b: &SceneRecord) -> bool {
    if a.dev != b.dev {
        return false;
    }
    let mut d: Vec<f64> = Vec::new();
    for o in &b.obs {
        if let Some(p) = a.obs.iter().find(|p| p.0 == o.0 && p.1 == o.1 && p.2 == o.2) {
            d.push(libm::fabs(p.3 - o.3));
        }
    }
    if d.len() < 6 {
        return false;
    }
    d.sort_by(|x, y| x.partial_cmp(y).unwrap());
    d[d.len() / 2] < 1e-3
}

fn sensors_seen(sc: &SceneRecord, station: u32) -> usize {
    let mut seen = 0u64;
    for o in &sc.obs {
        if o.0 == station {
            seen |= 1 << (o.2 & 63);
        }
    }
    seen.count_ones() as usize
}

/// The most-hit candidate per (axis, sensor) of one station in a scene.
fn first_obs(sc: &SceneRecord, station: u32) -> Vec<(u8, u8, f64)> {
    let mut v: Vec<(u8, u8, f64)> = Vec::new();
    for o in &sc.obs {
        if o.0 == station && !v.iter().any(|x| x.0 == o.1 && x.1 == o.2) {
            v.push((o.1, o.2, o.3));
        }
    }
    v
}

/// Propagates `track` through the queued IMU samples up to `t`, then to `t`.
fn advance(track: &mut TrackState, imu: &mut VecDeque<(i64, Vec3, Vec3)>, t: i64, last: Option<(i64, Vec3, Vec3)>, cfg: &TrackConfig) {
    let mut held = last;
    while let Some(&s) = imu.front() {
        if s.0 > t {
            break;
        }
        if s.0 > track.time {
            track.propagate(s.0, s.1, s.2, cfg);
        }
        held = Some(s);
        imu.pop_front();
    }
    if t > track.time {
        match held.or(imu.front().copied()) {
            Some(s) => track.propagate(t, s.1, s.2, cfg),
            None => track.coast(t, cfg),
        }
    }
}

/// The head pose and its covariance (position, rotation; world axes).
fn head_pose(st: &TrackState, body_from_head: &Pose) -> (Pose, Mat) {
    let wb = st.pose();
    let head = wb.compose(body_from_head);
    // δp_head = δp - R [t]x δθ;  δθ_world = R δθ.
    let r = rot_mat(wb.rot);
    let tx = skew(body_from_head.pos);
    let mut j = Mat::zeros(6, 6);
    for i in 0..3 {
        j[(i, i)] = 1.0;
        for c in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += r[i][k] * tx[k][c];
            }
            j[(i, 3 + c)] = -s;
            j[(3 + i, 3 + c)] = r[i][c];
        }
    }
    let cov = j.mul(&st.pose_cov()).mul(&j.t());
    (head, cov)
}

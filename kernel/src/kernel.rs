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
//!
//! Slow solves (a pose bootstrap to start tracking, a station calibration, a scene's
//! station check: up to seconds on a microcontroller) run inside [`Kernel::push`] by
//! default. With [`KernelConfig::defer_solves`] they come out as [`Event::Solve`] instead:
//! the host runs [`run_job`] wherever it likes (another thread, another core) and pushes
//! the result back as [`Input::Solved`], while light and IMU keep coming in. A pose that
//! comes back starts the track at the time it was solved for, and the sweeps that came
//! meanwhile go through it, so the track doesn't start out in the dark. A result the world
//! changed under (the stations were placed again meanwhile) is dropped.

use crate::Real;
use crate::lh1::AxisCal;
use crate::linalg::{rot_mat, skew, Mat};
use crate::math::{Pose, Quat, Vec3};
use crate::model::{StationModel, TrackerModel};
use crate::ootx::StationInfo;
use crate::solve::{
    bootstrap_pose, bootstrap_pose_assoc, bootstrap_station, calibrate, quat_between, station_disagreement, Calibration, CandidateObs, Obs,
    PoseSolution, Scene, SolveConfig, StationPrior,
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
    /// The host's word that the device's clock has passed `time` (ticks, wrapping) and
    /// every pulse before it was pushed: a host that maps its clock to the device's sends
    /// this between reports, and a sweep then goes out as soon as the laser has passed
    /// (TimingConfig::sweep_quiet_ticks) instead of at the next cycle's sync.
    Clock { dev: DeviceId, time: u32 },
    /// The result of a job handed out as [`Event::Solve`] ([`run_job`]).
    Solved(SolveResult),
}

#[derive(Clone, Debug)]
pub struct StationReport {
    pub id: u32,
    pub mode: Option<u8>,
    pub world_from_station: Pose,
    pub cov: Mat,
    /// RMS angle residual of its hits in the fit (rad).
    pub rms: Real,
    /// Angle between its OOTX gravity vector and world z (rad).
    pub up_error: Real,
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
    /// New station calibration; `reason` says why it ran. `pending`: stations named by
    /// OOTX and seen well in a still scene that this calibration still couldn't place (no
    /// still scene of a present device sees them together with a placed station).
    Calibrated {
        reason: CalibrationReason,
        stations: Vec<StationReport>,
        scenes: usize,
        dropped: usize,
        tracker_up_error: Real,
        pending: Vec<u32>,
    },
    /// A calibration (other than a routine MoreScenes refinement) found nothing to solve,
    /// or its fit failed the quality gate; `why` says which. The stations stay as they were.
    CalibrationSkipped { reason: CalibrationReason, why: &'static str },
    /// A still scene where the calibrated stations put the tracker in
    /// different places (largest difference, m and rad); `count` in a row.
    StationsDisagree { dev: DeviceId, pos: Real, rot: Real, count: u32 },
    TrackingStarted { dev: DeviceId, rms: Real },
    TrackingLost { dev: DeviceId, reason: TrackLoss },
    /// The device's pose ("head" frame) in the world. `cov` is 6x6 over
    /// (position, rotation), both in world axes; `time` device ticks
    /// (unwrapped); `light` the device time of the newest hit in it (the pose is
    /// propagated past it to the newest IMU sample, `time`). `speculative`: no hit has been
    /// used for `coast_after` (KernelConfig): the pose is the IMU's dead reckoning since
    /// the light was lost, published every `coast_interval` until light confirms it again
    /// or its covariance passes `lost_sigma_pos` / `lost_sigma_rot` (then TrackingLost).
    Pose {
        dev: DeviceId,
        time: i64,
        light: i64,
        world_from_head: Pose,
        world_from_body: Pose,
        velocity: Vec3,
        cov: Mat,
        used: u32,
        gated: u32,
        speculative: bool,
    },
    /// A slow solve to run off the input path (KernelConfig::defer_solves): [`run_job`] it
    /// anywhere and push the result back as [`Input::Solved`]. At most one per kind (per
    /// device for a pose bootstrap) is out at a time.
    Solve(SolveJob),
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
    /// Dead reckoning in the dark grew past `lost_sigma_pos` / `lost_sigma_rot`.
    Drifted,
    /// The light that came back after coasting disagreed with the dead-reckoned state
    /// (most hits gated): a fresh pose solve follows.
    Inconsistent,
}

#[derive(Clone, Copy, Debug)]
pub struct KernelConfig {
    pub timing: TimingConfig,
    pub solve: SolveConfig,
    pub track: TrackConfig,
    /// Still: mean gyro norm below (rad/s) and accel norm spread below (g),
    /// over `still_window` seconds.
    pub still_gyro: Real,
    pub still_accel: Real,
    pub still_window: Real,
    /// A scene (still stretch for calibration) ends only once the accelerometer spread
    /// passes `still_accel` times this. A tracker at rest can read right at `still_accel`
    /// (a Vive Tracker 3.0 on the P4: 0.0040 g); without the margin its scenes restarted a
    /// few times a second and never reached `scene_min`, so nothing was calibrated. The
    /// filter's still updates keep the strict thresholds (a slow carry isn't still).
    pub still_exit: Real,
    /// A still stretch becomes a scene after this long (s); one scene is cut
    /// at most this long.
    pub scene_min: Real,
    pub scene_max: Real,
    pub max_scenes: usize,
    /// Calibration: per station, sensors needed in a scene to use it there.
    pub scene_min_sensors: usize,
    /// Calibration: residual beyond which an observation is dropped (rad).
    pub calib_reject: Real,
    pub auto_calibrate: bool,
    /// Fold each station's calibration covariance into its hits' noise. Off:
    /// the covariance is dominated by the shared world tilt (a gauge freedom),
    /// and as independent per-hit noise it starves the filter. Calibration
    /// uncertainty is reported in [`Event::Calibrated`] instead.
    pub station_cov_in_filter: bool,
    /// While still: zero-velocity and gravity-direction updates, their σ
    /// (m/s, rad).
    pub still_updates: bool,
    pub still_sigma_vel: Real,
    pub still_sigma_up: Real,
    /// Tracking is dropped after this many gated hits in a row.
    pub lost_after_gated: u32,
    /// Coasting: no hit used for this long (s), and the tracker is dead reckoned on the IMU
    /// (the filter's own propagation, at `TrackConfig::coast_*_noise`), its pose published
    /// as speculative every `coast_interval` (s).
    pub coast_after: Real,
    pub coast_interval: Real,
    /// On entering coasting, the velocity's standard deviation is capped at this (m/s per
    /// axis; TrackState::cap_velocity_sigma). The tracking process noise (`accel_noise`, a
    /// model-error budget) leaves it at a multiple of the velocity's real error at a lock
    /// (~1-2 cm/s simulated mid-carry), which alone would cross `lost_sigma_pos` within
    /// ~0.1 s of coasting.
    pub coast_vel_sigma: Real,
    /// Coasting ends in TrackingLost (Drifted) once the position uncertainty, sqrt of the
    /// covariance trace (m), or the orientation's (rad) passes these: the primary rule.
    pub lost_sigma_pos: Real,
    pub lost_sigma_rot: Real,
    /// The backstop: at most this long without a used hit (s), whatever the covariance
    /// says (TrackingLost NoLight).
    pub lost_after_dark: Real,
    /// Light back after coasting: of the first this many hits, at least half must pass the
    /// gate against the dead-reckoned state, or it was wrong (TrackingLost Inconsistent).
    pub coast_check_hits: u32,
    /// The stations count as moved when single-station poses of a still
    /// tracker differ by more than these (m, rad) in `moved_after` scenes in
    /// a row. A good calibration agrees to ~1 cm.
    pub moved_pos: Real,
    pub moved_rot: Real,
    pub moved_after: u32,
    /// Check still scenes against the calibrated stations at all (a pose solve per
    /// scene). Off where another host owns the stations (the P4 adopting lhd's).
    pub check_stations: bool,
    /// While not tracking, try to start (a pose bootstrap) at most this often (s, IMU
    /// time); 0: on every sweep. A slow host (the P4) can't bootstrap a few hundred
    /// times a second and falls behind its input.
    pub start_interval: Real,
    /// Added to IMU sample times (s): IMU vs light latency.
    pub imu_time_offset: Real,
    /// Hand slow solves out as [`Event::Solve`] instead of running them in `push` (see the
    /// module doc). For a host that can't stop taking input for a second or more.
    pub defer_solves: bool,
    /// Deferred: sweeps kept for a pose bootstrap that's out (to run through the track it
    /// starts). Past this many the bootstrap is given up (its result would be too old for
    /// the IMU history to bring forward) and a fresh one follows.
    pub max_backlog: usize,
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
            still_exit: 1.5,
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
            coast_after: 0.05,
            coast_interval: 0.008,
            coast_vel_sigma: 0.03,
            lost_sigma_pos: 0.02,
            lost_sigma_rot: 0.05,
            lost_after_dark: 1.0,
            coast_check_hits: 8,
            moved_pos: 0.05,
            moved_rot: (3.0 as Real).to_radians(),
            moved_after: 3,
            check_stations: true,
            start_interval: 0.0,
            imu_time_offset: 0.0,
            defer_solves: false,
            max_backlog: 900,
        }
    }
}

const TPS: Real = crate::lh1::TICKS_PER_SECOND;

/// A slow solve, with everything it needs (owned): see [`Event::Solve`].
#[derive(Clone, Debug)]
pub struct SolveJob {
    pub id: u32,
    work: Work,
}

impl SolveJob {
    /// "start", "calibrate" or "check" (logging).
    pub fn kind(&self) -> &'static str {
        match self.work {
            Work::Start(_) => "start",
            Work::Calibrate(_) => "calibrate",
            Work::Check(_) => "check",
        }
    }
}

/// A job's result, for [`Input::Solved`]. Carries the job's inputs back.
#[derive(Clone, Debug)]
pub struct SolveResult {
    pub id: u32,
    work: Work,
    done: Done,
}

#[derive(Clone, Debug)]
enum Work {
    Start(StartWork),
    Calibrate(CalibWork),
    Check(CheckWork),
}

#[derive(Clone, Debug)]
enum Done {
    Start(Option<PoseSolution>),
    Calibrate(Result<CalibDone, &'static str>),
    Check(Option<(Real, Real)>),
}

/// A pose bootstrap to start tracking a device.
#[derive(Clone, Debug)]
struct StartWork {
    dev: DeviceId,
    model: TrackerModel,
    models: Vec<StationModel>,
    cands: Vec<CandidateObs>,
    up: Option<Vec3>,
    solve: SolveConfig,
    /// The device time the pose is for (its newest IMU sample then).
    t: i64,
    stations_gen: u64,
}

/// A station calibration: the scenes and stations as they were when it was asked for.
#[derive(Clone, Debug)]
struct CalibWork {
    reason: CalibrationReason,
    ids: Vec<u32>,
    ref_idx: usize,
    reference: u32,
    from_scratch: bool,
    scenes: Vec<SceneRecord>,
    /// Per scene: its device's tracker model (None: the device is gone).
    scene_models: Vec<Option<TrackerModel>>,
    scene_poses: Vec<Option<Pose>>,
    models: Vec<Option<StationModel>>,
    cals: Vec<[AxisCal; 2]>,
    ups: Vec<Option<Vec3>>,
    priors: Vec<Option<StationPrior>>,
    min_sensors: usize,
    solve: SolveConfig,
    reject: Real,
    stations_gen: u64,
}

#[derive(Clone, Debug)]
struct CalibDone {
    known: Vec<usize>,
    cal: Calibration,
    scene_poses: Vec<Option<Pose>>,
    dev0: DeviceId,
    scenes: usize,
}

/// A still scene checked against the calibrated stations it sees.
#[derive(Clone, Debug)]
struct CheckWork {
    dev: DeviceId,
    scene: u64,
    model: TrackerModel,
    known: Vec<StationModel>,
    obs: Vec<Obs>,
    up: Vec3,
    min_sensors: usize,
    solve: SolveConfig,
    stations_gen: u64,
}

/// Runs a job from [`Event::Solve`]: pure, touches no kernel, so any thread can.
pub fn run_job(job: SolveJob) -> SolveResult {
    let done = match &job.work {
        Work::Start(w) => Done::Start(bootstrap_pose_assoc(&w.model, &w.models, &w.cands, w.up, 3e-3, &w.solve)),
        Work::Calibrate(w) => Done::Calibrate(run_calibration(w)),
        Work::Check(w) => Done::Check(station_disagreement(&w.model, &w.known, &w.obs, w.up, w.min_sensors, &w.solve)),
    };
    SolveResult { id: job.id, work: job.work, done }
}

/// Poses the scenes without one against the stations placed so far (`models`, by index
/// into `ids`).
fn pose_scenes(w: &CalibWork, models: &[Option<StationModel>], scene_poses: &mut [Option<Pose>]) {
    let known: Vec<usize> = (0..w.ids.len()).filter(|&k| models[k].is_some()).collect();
    if known.is_empty() {
        return;
    }
    let km: Vec<StationModel> = known.iter().map(|&k| models[k].clone().unwrap()).collect();
    for j in 0..w.scenes.len() {
        if scene_poses[j].is_some() {
            continue;
        }
        let sc = &w.scenes[j];
        let Some(model) = w.scene_models[j].as_ref() else { continue };
        let mut obs = Vec::new();
        for (ki, &k) in known.iter().enumerate() {
            for (axis, sensor, angle) in first_obs(sc, w.ids[k]) {
                obs.push(Obs { station: ki, axis, sensor: sensor as u16, angle });
            }
        }
        if let Some(sol) = bootstrap_pose(model, &km, &obs, Some(sc.up_body), &w.solve) {
            if sol.rms < 3e-3 {
                scene_poses[j] = Some(sol.world_from_body);
            }
        }
    }
}

/// The solving part of a station calibration (Kernel::plan_calibration has the rest).
fn run_calibration(w: &CalibWork) -> Result<CalibDone, &'static str> {
    let mut scene_poses = w.scene_poses.clone();
    let mut models = w.models.clone();
    if !w.from_scratch {
        // Scenes collected before tracking started have no pose yet: place them against
        // the stations already known first, so a station seen only in such scenes can be
        // solved from them right away.
        pose_scenes(w, &models, &mut scene_poses);
    }
    // Stations without a pose: solve each from a scene with a known pose.
    for (k, id) in w.ids.iter().enumerate() {
        if models[k].is_some() {
            continue;
        }
        let cand = w
            .scenes
            .iter()
            .enumerate()
            .filter(|(j, _)| scene_poses[*j].is_some() && w.scene_models[*j].is_some())
            .max_by_key(|(_, sc)| sensors_seen(sc, *id));
        let Some((j, sc)) = cand else { continue };
        if sensors_seen(sc, *id) < w.min_sensors {
            continue;
        }
        let obs: Vec<Obs> = first_obs(sc, *id).into_iter().map(|(axis, sensor, angle)| Obs { station: 0, axis, sensor: sensor as u16, angle }).collect();
        let model = w.scene_models[j].as_ref().unwrap();
        if let Some((st, _)) = bootstrap_station(model, w.cals[k], &scene_poses[j].unwrap(), &obs, w.ups[k], &w.solve) {
            models[k] = Some(st);
        }
    }
    let known: Vec<usize> = (0..w.ids.len()).filter(|&k| models[k].is_some()).collect();
    if !known.contains(&w.ref_idx) {
        return Err("the reference station could not be placed");
    }
    // Scenes without a pose: bootstrap against the known stations.
    let km: Vec<StationModel> = known.iter().map(|&k| models[k].clone().unwrap()).collect();
    pose_scenes(w, &models, &mut scene_poses);
    // Joint fit over the known stations and posed scenes (one tracker model per scene's
    // device; scenes from other devices skipped for now). The device with the most posed
    // scenes: the first scene's may be gone (a tracker that dropped off the radio and
    // came back as a new device), and its scenes with it.
    let mut devs: Vec<(DeviceId, usize)> = Vec::new();
    for (j, sc) in w.scenes.iter().enumerate() {
        if scene_poses[j].is_none() {
            continue;
        }
        match devs.iter_mut().find(|x| x.0 == sc.dev) {
            Some(x) => x.1 += 1,
            None => devs.push((sc.dev, 1)),
        }
    }
    let Some(dev0) = devs.iter().max_by_key(|x| x.1).map(|x| x.0) else { return Err("no still scene could be placed") };
    let model = (0..w.scenes.len()).find(|&j| w.scenes[j].dev == dev0).and_then(|j| w.scene_models[j].clone());
    let Some(model) = model else { return Err("no still scene could be placed") };
    let mut scenes = Vec::new();
    for (j, sc) in w.scenes.iter().enumerate() {
        let (Some(p), true) = (scene_poses[j], sc.dev == dev0) else { continue };
        let mut obs = Vec::new();
        for (ki, &k) in known.iter().enumerate() {
            for o in sc.obs.iter().filter(|o| o.0 == w.ids[k]) {
                obs.push(Obs { station: ki, axis: o.1, sensor: o.2 as u16, angle: o.3 });
            }
        }
        scenes.push(Scene { obs, up_body: sc.up_body, world_from_body: p });
    }
    if scenes.is_empty() {
        return Err("no still scene could be placed");
    }
    let ref_in_known = known.iter().position(|&k| k == w.ref_idx).unwrap();
    let ups: Vec<Option<Vec3>> = known.iter().map(|&k| w.ups[k]).collect();
    let priors: Vec<Option<StationPrior>> = known.iter().map(|&k| w.priors[k]).collect();
    let Some(cal) = calibrate(&model, &km, &ups, &priors, &scenes, ref_in_known, &w.solve, w.reject) else {
        return Err("the station fit failed");
    };
    // Quality gate: residuals a few σ at most.
    if cal.rms.iter().any(|r| !(*r < 3e-3)) {
        return Err("the station fit's residuals are too large");
    }
    Ok(CalibDone { known, cal, scene_poses, dev0, scenes: scenes.len() })
}

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
    gyro: VecDeque<(i64, Real)>,
    acc: VecDeque<(i64, Real)>,
    still: bool,
    since: i64,
    /// Still for collecting a scene: as `still`, but once still it takes `still_exit` times
    /// the accelerometer threshold to end (see KernelConfig::still_exit).
    scene_still: bool,
    scene_since: i64,
}

struct Cluster {
    values: Vec<Real>,
}

impl Cluster {
    fn median(&self) -> Real {
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
                match cl.iter_mut().find(|k| crate::math::fabs(k.values[k.values.len() - 1] - c.angle) < 2e-3) {
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
#[derive(Clone, Debug)]
struct SceneRecord {
    /// Kept when the scene is refreshed (the same place again).
    id: u64,
    dev: DeviceId,
    up_body: Vec3,
    /// (station id, axis, sensor, angle, count), several per key possible.
    obs: Vec<(u32, u8, u8, Real, usize)>,
    pose: Option<Pose>,
}

struct Device {
    id: DeviceId,
    model: TrackerModel,
    timing: Timing,
    slot_station: [Option<u32>; 2],
    /// The stations the slots had before the sweep-timing lock was lost. A re-lock puts
    /// them back once sweeps have come from both slots (which pins the cycle phase, so slot
    /// 0 is the master again), instead of waiting ~3-6 s for each station's next OOTX
    /// packet; the OOTX that follows confirms or corrects them.
    prev_slot_station: [Option<u32>; 2],
    /// Since the re-lock, sweeps seen per slot (bit per slot), while `prev_slot_station`
    /// waits to be put back.
    relock_slots: u8,
    clock: Clock,
    imu: VecDeque<(i64, Vec3, Vec3)>,
    last_imu: Option<(i64, Vec3, Vec3)>,
    still: StillDetector,
    scene: Option<SceneBuilder>,
    track: Option<TrackState>,
    recent: VecDeque<(u32, u8, Vec<SensorHits>)>,
    gated_run: u32,
    /// Dead reckoning since the light was lost (KernelConfig::coast_after); its last
    /// speculative pose (IMU time), and the hits used / gated since light came back.
    coasting: bool,
    last_coast_pub: i64,
    coast_used: u32,
    coast_gated: u32,
    /// IMU time of the last start attempt (cfg.start_interval).
    last_start: Option<i64>,
    last_used: i64,
    window: UpdateStats,
    /// Diagnostics: innovation sum, sum of squares and count per (station, axis).
    innovations: Vec<((u32, u8), Real, Real, u32)>,
    /// Station id → index of its angle-bias states in the tracker filter.
    bias_slots: Vec<u32>,
    /// A pose bootstrap out (defer_solves): its job id, and the sweeps since (station,
    /// axis, part, hits at unwrapped times) to run through the track it starts.
    start_pending: Option<u32>,
    backlog: Vec<(u32, u8, u8, Vec<(i64, SensorHits)>)>,
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
    /// Bumped whenever a station's pose changes: a solve made before is for another world.
    stations_gen: u64,
    next_job: u32,
    next_scene: u64,
    /// The calibration / station check out (defer_solves), and a calibration asked for
    /// while one was out (run when it's back).
    cal_pending: Option<u32>,
    cal_again: Option<CalibrationReason>,
    check_pending: Option<u32>,
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
            stations_gen: 0,
            next_job: 0,
            next_scene: 0,
            cal_pending: None,
            cal_again: None,
            check_pending: None,
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
                    prev_slot_station: [None, None],
                    relock_slots: 0,
                    clock: Clock::default(),
                    imu: VecDeque::new(),
                    last_imu: None,
                    still: StillDetector::default(),
                    scene: None,
                    track: None,
                    recent: VecDeque::new(),
                    gated_run: 0,
                    coasting: false,
                    last_coast_pub: 0,
                    coast_used: 0,
                    coast_gated: 0,
                    last_start: None,
                    last_used: 0,
                    window: UpdateStats::default(),
                    innovations: Vec::new(),
                    bias_slots: Vec::new(),
                    start_pending: None,
                    backlog: Vec::new(),
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
            Input::Clock { dev, time } => {
                let Some(di) = self.devices.iter().position(|d| d.id == dev) else { return };
                let mut tev = Vec::new();
                self.devices[di].timing.clock(time, &mut tev);
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
                let sd = |i: usize| crate::math::sqrt(cov[(i, i)].max(0.0));
                let sp = sd(0).max(sd(1)).max(sd(2)).max(2e-3);
                let sr = sd(3).max(sd(4)).max(sd(5)).max(2e-3);
                st.prior = Some(StationPrior { pose: world_from_station, sigma_pos: sp, sigma_rot: sr });
                st.cov = Some(cov);
                self.stations_gen += 1;
                self.world_fixed = true;
                if self.reference.is_none() {
                    self.reference = Some(id);
                }
            }
            Input::Recalibrate => self.calibrate(CalibrationReason::Requested, out),
            Input::Solved(r) => self.apply(r, true, out),
        }
    }

    /// Switch TimingConfig::stream_sweeps, for every device now and later.
    pub fn set_stream_sweeps(&mut self, on: bool) {
        self.cfg.timing.stream_sweeps = on;
        for d in &mut self.devices {
            d.timing.set_stream_sweeps(on);
        }
    }

    /// Diagnostics: a device's light timing counters.
    pub fn timing_stats(&self, dev: DeviceId) -> Option<crate::timing::TimingStats> {
        self.devices.iter().find(|d| d.id == dev).map(|d| d.timing.stats)
    }

    /// Diagnostics: per (station id, axis) of a device, the mean and RMS
    /// innovation (rad) of the hits used, and their count; then resets.
    pub fn take_innovations(&mut self, dev: DeviceId) -> Vec<(u32, u8, Real, Real, u32)> {
        let Some(d) = self.devices.iter_mut().find(|d| d.id == dev) else { return Vec::new() };
        let v = d
            .innovations
            .iter()
            .map(|e| (e.0 .0, e.0 .1, e.1 / e.3 as Real, crate::math::sqrt(e.2 / e.3 as Real), e.3))
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
        let n = d.still.gyro.len().max(1) as Real;
        let g = d.still.gyro.iter().map(|x| x.1).sum::<Real>() / n;
        let am = d.still.acc.iter().map(|x| x.1).sum::<Real>() / n;
        let asd = crate::math::sqrt(d.still.acc.iter().map(|x| (x.1 - am) * (x.1 - am)).sum::<Real>() / n);
        let full = d.still.gyro.front().map_or(false, |x| x.0 <= t - w * 9 / 10);
        let still = full && g < cfg.still_gyro && asd < cfg.still_accel;
        let k = if d.still.scene_still { cfg.still_exit } else { 1.0 };
        let scene_still = full && g < cfg.still_gyro && asd < cfg.still_accel * k;
        let mut scene_ended = false;
        if scene_still != d.still.scene_still {
            d.still.scene_still = scene_still;
            d.still.scene_since = t;
            scene_ended = !scene_still;
        }
        if still != d.still.still {
            d.still.still = still;
            d.still.since = t;
            out.push(Event::Still { dev, still });
            if !still {
                if scene_ended {
                    self.finish_scene(di, out);
                }
                return;
            }
        }
        if scene_ended {
            self.finish_scene(di, out);
        }
        let d = &mut self.devices[di];
        let d = &mut self.devices[di];
        // In the dark (no hit used for coast_after): dead reckon on the IMU, judged on the
        // IMU clock (no sweep comes while both stations are blocked).
        let dark = (t - d.last_used) as Real / TPS;
        let coast = d.track.is_some() && dark > cfg.coast_after;
        let tc = if coast { cfg.track.coasting() } else { cfg.track };
        if coast && !d.coasting {
            d.coasting = true;
            if let Some(track) = d.track.as_mut() {
                track.cap_velocity_sigma(cfg.coast_vel_sigma);
            }
            d.coast_used = 0;
            d.coast_gated = 0;
            d.last_coast_pub = 0;
        }
        if still && cfg.still_updates {
            if let Some(track) = d.track.as_mut() {
                if t > track.time {
                    // Bring the filter up to this sample first (it may lag the light).
                    advance(track, &mut d.imu, t, d.last_imu, &tc);
                }
                track.update_still(accel_g, cfg.still_sigma_vel, cfg.still_sigma_up);
            }
        }
        if coast {
            let track = d.track.as_mut().unwrap();
            if t > track.time {
                advance(track, &mut d.imu, t, d.last_imu, &tc);
            }
            let (sp, sr) = sigmas(track);
            let loss = if dark > cfg.lost_after_dark {
                Some(TrackLoss::NoLight)
            } else if !(sp <= cfg.lost_sigma_pos && sr <= cfg.lost_sigma_rot) {
                Some(TrackLoss::Drifted)
            } else {
                None
            };
            if let Some(reason) = loss {
                d.track = None;
                d.coasting = false;
                d.recent.clear();
                out.push(Event::TrackingLost { dev, reason });
            } else if (t - d.last_coast_pub) as Real >= cfg.coast_interval * TPS {
                d.last_coast_pub = t;
                let track = d.track.as_ref().unwrap();
                let (pose, cov) = head_pose(track, &d.model.body_from_head);
                out.push(Event::Pose {
                    dev,
                    time: track.time,
                    light: d.last_used,
                    world_from_head: pose,
                    world_from_body: track.pose(),
                    velocity: track.vel,
                    cov,
                    used: 0,
                    gated: d.window.gated,
                    speculative: true,
                });
                d.window = UpdateStats::default();
            }
        }
        if scene_still {
            if let Some(sb) = d.scene.as_mut() {
                sb.acc_sum = sb.acc_sum.add(accel_g);
                sb.acc_n += 1;
            }
            let dur = (t - d.still.scene_since) as Real / TPS;
            if d.scene.is_none() && dur >= 0.3 {
                d.scene = Some(SceneBuilder { start: t, clusters: Vec::new(), acc_sum: Vec3::ZERO, acc_n: 0 });
            }
            if d.scene.as_ref().map_or(false, |s| (t - s.start) as Real / TPS >= cfg.scene_max) {
                self.finish_scene(di, out);
                // Keep collecting the next scene only if this stays still a while.
                let d = &mut self.devices[di];
                d.still.scene_since = t;
            }
        }
    }

    fn finish_scene(&mut self, di: usize, out: &mut Vec<Event>) {
        let d = &mut self.devices[di];
        let Some(sb) = d.scene.take() else { return };
        let dur = d.last_imu.map_or(0, |x| x.0 - sb.start) as Real / TPS;
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
        let up = sb.acc_sum.scale(1.0 / sb.acc_n as Real);
        let pose = d.track.as_ref().map(|t| t.pose());
        let mut rec = SceneRecord { id: self.next_scene, dev: d.id, up_body: up.normalized(), obs, pose };
        self.next_scene += 1;
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
        //
        // Unless it sees a station the old one didn't: hits of a station only count once its
        // OOTX has named it, so a tracker left in one place sees a station that came up
        // later only in its later scenes there. Those must reach the calibration, or the
        // new station waits for the tracker to be moved (and an explicit recalibration
        // never sees it either).
        if let Some(j) = self.scenes.iter().position(|old| same_place(old, &rec)) {
            let old = &self.scenes[j];
            let newly_seen = rec.obs.iter().any(|o| sensors_seen(&rec, o.0) >= min && sensors_seen(old, o.0) < min);
            let pose = old.pose;
            let id = old.id;
            self.scenes[j] = SceneRecord { id, pose: pose.or(rec.pose), ..rec };
            if self.cfg.auto_calibrate && (self.moved || newly_seen) {
                let reason = self.auto_reason(j);
                self.calibrate(reason, out);
            }
            return;
        }
        if self.scenes.len() >= self.cfg.max_scenes {
            self.scenes.remove(0);
        }
        self.scenes.push(rec);
        if self.cfg.auto_calibrate {
            let reason = self.auto_reason(self.scenes.len() - 1);
            if reason != CalibrationReason::MoreScenes || !self.world_fixed || self.scenes.len() >= 2 {
                self.calibrate(reason, out);
            }
        }
    }

    /// Why scene `j` (new or refreshed) calls for a calibration.
    fn auto_reason(&self, j: usize) -> CalibrationReason {
        let needs = self.scenes[j].obs.iter().any(|o| {
            let s = self.stations.iter().find(|s| s.id == o.0);
            s.map_or(false, |s| s.model.is_none() && s.info.is_some())
        });
        if self.moved {
            CalibrationReason::StationsMoved
        } else if !self.stations.iter().any(|s| s.model.is_some()) {
            CalibrationReason::FirstStations
        } else if needs {
            CalibrationReason::NewStation
        } else {
            CalibrationReason::MoreScenes
        }
    }

    // ------------------------------------------------------------ light

    fn timing_event(&mut self, di: usize, e: TimingEvent, out: &mut Vec<Event>) {
        let dev = self.devices[di].id;
        match e {
            TimingEvent::Locked { .. } => {
                self.devices[di].relock_slots = 0;
                out.push(Event::TimingLocked { dev });
            }
            TimingEvent::LockLost { reason } => {
                let d = &mut self.devices[di];
                if reason == LockLoss::SlotPhase {
                    // What this lock called slot 0 was the slave's slot.
                    if let Some(s) = d.slot_station[0] {
                        d.prev_slot_station[1] = Some(s);
                        if d.prev_slot_station[0] == Some(s) {
                            d.prev_slot_station[0] = None;
                        }
                    }
                } else {
                    for slot in 0..2 {
                        if d.slot_station[slot].is_some() {
                            d.prev_slot_station[slot] = d.slot_station[slot];
                        }
                    }
                }
                d.slot_station = [None, None];
                d.relock_slots = 0;
                out.push(Event::TimingLost { dev, reason });
            }
            TimingEvent::Station { slot, info } => {
                let d = &mut self.devices[di];
                let was = d.slot_station[slot as usize];
                let changed = was != Some(info.id);
                d.slot_station[slot as usize] = Some(info.id);
                // A slot put back after a re-lock turned out to be another station: what
                // was tracked with it is wrong.
                if was.is_some() && changed && d.track.take().is_some() {
                    d.recent.clear();
                    out.push(Event::TrackingLost { dev, reason: TrackLoss::StationsMoved });
                }
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
        {
            let d = &mut self.devices[di];
            if d.relock_slots != 3 && d.slot_station.contains(&None) && d.prev_slot_station != [None, None] {
                d.relock_slots |= 1 << (s.slot & 1);
                if d.relock_slots == 3 {
                    for slot in 0..2 {
                        if d.slot_station[slot].is_none() {
                            d.slot_station[slot] = d.prev_slot_station[slot];
                        }
                    }
                }
            }
        }
        let Some(sid) = self.devices[di].slot_station[s.slot as usize] else { return };
        if let Some(sb) = self.devices[di].scene.as_mut() {
            sb.add(sid, s.axis, &s.hits);
        }
        // Only light from a placed station is used.
        if !self.stations.iter().any(|x| x.id == sid && x.model.is_some()) {
            return;
        }
        let cfg = self.cfg;
        let d = &mut self.devices[di];

        if d.track.is_none() {
            // A later part of the sweep just before (streaming) joins it.
            match d.recent.back_mut() {
                Some(r) if s.part > 0 && r.0 == sid && r.1 == s.axis => {
                    for h in &s.hits {
                        match r.2.iter_mut().find(|x| x.sensor == h.sensor) {
                            Some(x) => x.candidates.extend_from_slice(&h.candidates),
                            None => r.2.push(h.clone()),
                        }
                    }
                }
                _ => d.recent.push_back((sid, s.axis, s.hits.clone())),
            }
            while d.recent.len() > 8 {
                d.recent.pop_front();
            }
            if d.start_pending.is_some() {
                // A bootstrap is out: keep the light for the track it may start.
                if d.backlog.len() >= cfg.max_backlog {
                    d.start_pending = None; // too old by the time it's back
                    d.backlog.clear();
                } else {
                    let hits = unwrap_hits(&mut d.clock, &s.hits);
                    d.backlog.push((sid, s.axis, s.part, hits));
                }
            }
            self.try_start_tracking(di, out);
            return;
        }
        let hits = unwrap_hits(&mut self.devices[di].clock, &s.hits);
        self.track_sweep(di, sid, s.axis, s.part, hits, true, out);
    }

    /// A sweep's hits (unwrapped times, in time order) through a device's track: `publish`
    /// its pose after.
    fn track_sweep(&mut self, di: usize, sid: u32, axis: u8, part: u8, hits: Vec<(i64, SensorHits)>, publish: bool, out: &mut Vec<Event>) {
        let Some(si) = self.stations.iter().position(|x| x.id == sid) else { return };
        let Some(station) = self.stations[si].model.clone() else { return };
        let station_cov = self.stations[si].cov.clone();
        let cfg = self.cfg;

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
        // The filter follows the IMU up to each hit.
        let mut stats = UpdateStats::default();
        {
            let track = d.track.as_mut().unwrap();
            if part == 0 {
                track.begin_sweep(cfg.track.sigma_sweep);
            }
            for (t, h) in &hits {
                advance(track, &mut d.imu, *t, d.last_imu, &cfg.track);
                let before = stats.used;
                track.update_hit(
                    &d.model,
                    &station,
                    if cfg.station_cov_in_filter { station_cov.as_ref() } else { None },
                    station_slot,
                    axis,
                    h,
                    &cfg.track,
                    &mut stats,
                );
                if stats.used > before {
                    let r = stats.last_innovation;
                    match d.innovations.iter_mut().find(|e| e.0 == (sid, axis)) {
                        Some(e) => {
                            e.1 += r;
                            e.2 += r * r;
                            e.3 += 1;
                        }
                        None => d.innovations.push(((sid, axis), r, r * r, 1)),
                    }
                }
            }
        }
        d.window.used += stats.used;
        d.window.gated += stats.gated;
        // Light back after coasting: fused into the dead-reckoned track if it agrees (most
        // of the first hits pass the gate), else that state was wrong and a fresh solve
        // takes over.
        let mut inconsistent = false;
        if d.coasting && stats.used + stats.gated > 0 {
            d.coast_used += stats.used;
            d.coast_gated += stats.gated;
            if d.coast_used + d.coast_gated >= cfg.coast_check_hits {
                inconsistent = d.coast_used * 2 < d.coast_used + d.coast_gated;
                d.coasting = false;
            }
        }
        if stats.used > 0 {
            d.gated_run = 0;
            d.last_used = hits.last().map_or(d.last_used, |h| h.0);
        } else {
            d.gated_run += stats.gated;
        }
        let track = d.track.as_ref().unwrap();
        let now = hits.last().map_or(track.time, |h| h.0);
        let dark = (now - d.last_used) as Real / TPS;
        let trace: Real = (0..3).map(|i| track.cov[(i, i)]).sum();
        let loss = if inconsistent {
            Some(TrackLoss::Inconsistent)
        } else if d.gated_run >= cfg.lost_after_gated {
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
            d.coasting = false;
            d.recent.clear();
            out.push(Event::TrackingLost { dev: d.id, reason });
            return;
        }
        if !publish {
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
            light: now,
            world_from_head: pose,
            world_from_body: pubst.pose(),
            velocity: pubst.vel,
            cov,
            used: d.window.used,
            gated: d.window.gated,
            speculative: d.coasting || dark > cfg.coast_after,
        });
        d.window = UpdateStats::default();
    }

    fn try_start_tracking(&mut self, di: usize, out: &mut Vec<Event>) {
        if self.devices[di].start_pending.is_some() {
            return;
        }
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
                let angles: Vec<Real> = h.candidates.iter().map(|c| c.angle).collect();
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
        if cfg.start_interval > 0.0 {
            let now = d.last_imu.map_or(0, |x| x.0);
            if d.last_start.is_some_and(|t| now >= t && ((now - t) as Real) < cfg.start_interval * TPS) {
                return;
            }
            d.last_start = Some(now);
        }
        // The accelerometer is "up" only while it reads gravity alone: mid-motion it adds
        // the tracker's own acceleration, and a bootstrap held to that tilts the pose (a
        // fresh lock ~2 cm off while a swing came to rest).
        let up = d.last_imu.filter(|x| crate::math::fabs(x.1.norm() - 1.0) < 0.05 && x.2.norm() < 0.5).map(|x| x.1);
        let t = d.last_imu.map_or(0, |x| x.0);
        let work = StartWork { dev: d.id, model: d.model.clone(), models, cands, up, solve: cfg.solve, t, stations_gen: self.stations_gen };
        self.dispatch(Work::Start(work), out);
    }

    /// A pose bootstrap is back: start the track at the time it was solved for, then run
    /// the light that came meanwhile through it.
    fn apply_start(&mut self, id: u32, w: StartWork, sol: Option<PoseSolution>, out: &mut Vec<Event>) {
        let cfg = self.cfg;
        let Some(di) = self.devices.iter().position(|d| d.id == w.dev) else { return };
        let d = &mut self.devices[di];
        if d.start_pending != Some(id) {
            return; // given up, or the device came back as a new one
        }
        d.start_pending = None;
        let backlog = core::mem::take(&mut d.backlog);
        let Some(sol) = sol else { return };
        if w.stations_gen != self.stations_gen || d.track.is_some() {
            return; // solved against stations that have moved since
        }
        if !(sol.rms < 2e-3) || sol.inliers < 8 || sol.inliers * 3 < sol.obs * 2 {
            return;
        }
        d.track = Some(TrackState::new(&sol.world_from_body, &sol.cov.scale(4.0), w.t, cfg.track.angle_bias_init));
        d.last_used = w.t;
        d.gated_run = 0;
        d.coasting = false;
        d.recent.clear();
        out.push(Event::TrackingStarted { dev: d.id, rms: sol.rms });
        let n = backlog.len();
        for (i, (sid, axis, part, hits)) in backlog.into_iter().enumerate() {
            if self.devices[di].track.is_none() {
                break;
            }
            self.track_sweep(di, sid, axis, part, hits, i + 1 == n, out);
        }
    }

    /// Runs a slow solve here, or hands it out (defer_solves).
    fn dispatch(&mut self, work: Work, out: &mut Vec<Event>) {
        let id = self.next_job;
        self.next_job = self.next_job.wrapping_add(1);
        match &work {
            Work::Start(w) => {
                if let Some(d) = self.devices.iter_mut().find(|d| d.id == w.dev) {
                    d.start_pending = Some(id);
                    d.backlog.clear();
                }
            }
            Work::Calibrate(_) => self.cal_pending = Some(id),
            Work::Check(_) => self.check_pending = Some(id),
        }
        let job = SolveJob { id, work };
        if self.cfg.defer_solves {
            out.push(Event::Solve(job));
        } else {
            self.apply(run_job(job), false, out);
        }
    }

    /// A solve's result (`deferred`: it came back as Input::Solved).
    fn apply(&mut self, r: SolveResult, deferred: bool, out: &mut Vec<Event>) {
        match (r.work, r.done) {
            (Work::Start(w), Done::Start(sol)) => self.apply_start(r.id, w, sol, out),
            (Work::Calibrate(w), Done::Calibrate(res)) => self.apply_calibration(r.id, w, res, out),
            (Work::Check(w), Done::Check(d)) => {
                let scene = w.scene;
                if self.apply_check(r.id, w, d, out) && deferred && self.cfg.auto_calibrate {
                    // The calibration was dropped after finish_scene ran (inline, it goes on
                    // to solve it again from this scene): solve it again now.
                    let reason = match self.scenes.iter().position(|sc| sc.id == scene) {
                        Some(j) => self.auto_reason(j),
                        None => CalibrationReason::StationsMoved,
                    };
                    self.calibrate(reason, out);
                }
            }
            _ => {}
        }
    }

    // ------------------------------------------------------------ calibration

    /// Checks a still scene against the calibrated stations it sees; after
    /// `moved_after` disagreeing scenes in a row, drops the calibration.
    fn check_stations(&mut self, rec: &SceneRecord, out: &mut Vec<Event>) {
        let cfg = self.cfg;
        if !cfg.check_stations {
            return;
        }
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
        if self.check_pending.is_some() {
            return; // one out already; the next scene is checked
        }
        let work = CheckWork {
            dev: rec.dev,
            scene: rec.id,
            model,
            known,
            obs,
            up: rec.up_body,
            min_sensors: cfg.scene_min_sensors,
            solve: cfg.solve,
            stations_gen: self.stations_gen,
        };
        self.dispatch(Work::Check(work), out);
    }

    /// A station check is back: true if it dropped the calibration.
    fn apply_check(&mut self, id: u32, w: CheckWork, d: Option<(Real, Real)>, out: &mut Vec<Event>) -> bool {
        let cfg = self.cfg;
        if self.check_pending != Some(id) {
            return false;
        }
        self.check_pending = None;
        if w.stations_gen != self.stations_gen {
            return false;
        }
        let Some((pos, rot)) = d else { return false };
        if pos <= cfg.moved_pos && rot <= cfg.moved_rot {
            self.disagreeing = 0;
            return false;
        }
        self.disagreeing += 1;
        out.push(Event::StationsDisagree { dev: w.dev, pos, rot, count: self.disagreeing });
        if self.disagreeing < cfg.moved_after || self.solved_here {
            return false;
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
        self.stations_gen += 1;
        self.reference = None;
        self.world_fixed = false;
        self.disagreeing = 0;
        self.moved = true;
        true
    }

    fn calibrate(&mut self, reason: CalibrationReason, out: &mut Vec<Event>) {
        if self.cal_pending.is_some() {
            // One is out: this one runs when it's back (a routine refinement gives way to
            // any other reason).
            self.cal_again = match self.cal_again {
                Some(r) if r != CalibrationReason::MoreScenes => Some(r),
                _ => Some(reason),
            };
            return;
        }
        match self.plan_calibration(reason) {
            Ok(work) => self.dispatch(Work::Calibrate(work), out),
            Err(why) => {
                if reason != CalibrationReason::MoreScenes {
                    out.push(Event::CalibrationSkipped { reason, why });
                }
            }
        }
    }

    /// Scenes usable for calibration: their device is still here (its tracker model is
    /// needed to use their angles).
    fn scene_usable(&self, sc: &SceneRecord) -> bool {
        self.devices.iter().any(|d| d.id == sc.dev)
    }

    /// What a calibration needs, as things are now (run_calibration solves it).
    fn plan_calibration(&mut self, reason: CalibrationReason) -> Result<CalibWork, &'static str> {
        if !self.scenes.iter().any(|sc| self.scene_usable(sc)) {
            return Err("no still scenes from a connected tracker");
        }
        let cfg = self.cfg;
        // Stations with known factory calibration that scenes see well enough.
        let ids: Vec<u32> = self
            .stations
            .iter()
            .filter(|s| s.cal.is_some())
            .map(|s| s.id)
            .filter(|id| self.scenes.iter().any(|sc| self.scene_usable(sc) && sensors_seen(sc, *id) >= cfg.scene_min_sensors))
            .collect();
        if ids.is_empty() {
            return Err("no station with a decoded OOTX is seen well in a still scene");
        }
        let reference = match self.reference {
            Some(r) if ids.contains(&r) => r,
            _ => {
                // The station seen with most sensors in the first usable scene.
                let sc = self.scenes.iter().find(|sc| self.scene_usable(sc)).unwrap();
                *ids.iter().max_by_key(|id| sensors_seen(sc, **id)).unwrap()
            }
        };
        let ref_idx = ids.iter().position(|x| *x == reference).unwrap();

        // Scene poses: tracked pose if there was one, else solved by run_calibration.
        let mut scene_poses: Vec<Option<Pose>> = self.scenes.iter().map(|s| if self.scene_usable(s) { s.pose } else { None }).collect();
        let models: Vec<Option<StationModel>> =
            ids.iter().map(|id| self.stations.iter().find(|s| s.id == *id).unwrap().model.clone()).collect();
        let entry = |id: u32| self.stations.iter().find(|s| s.id == id).unwrap();
        let cals: Vec<[AxisCal; 2]> = ids.iter().map(|id| entry(*id).cal.unwrap()).collect();
        let ups: Vec<Option<Vec3>> = ids
            .iter()
            .map(|id| {
                entry(*id).info.as_ref().map(|i| Vec3::new(i.accel_dir[0] as Real, i.accel_dir[1] as Real, i.accel_dir[2] as Real).normalized())
            })
            .collect();
        let priors: Vec<Option<StationPrior>> = ids.iter().map(|id| entry(*id).prior).collect();

        // Bootstrap: nothing known at all. The first usable scene defines the
        // world (tracker at the origin, levelled, body x as world x).
        let from_scratch = models.iter().all(|m| m.is_none());
        if from_scratch {
            let first = self.scenes.iter().position(|sc| self.scene_usable(sc) && sensors_seen(sc, reference) >= cfg.scene_min_sensors).unwrap();
            let sc = &self.scenes[first];
            let level = quat_between(sc.up_body, Vec3::new(0.0, 0.0, 1.0));
            // Keep the body x heading: remove yaw introduced by levelling.
            let x = level.rotate(Vec3::new(1.0, 0.0, 0.0));
            let yaw = crate::math::atan2(x.y, x.x);
            let rot = Quat::from_rotvec(Vec3::new(0.0, 0.0, -yaw)).mul(level).normalized();
            scene_poses[first] = Some(Pose { pos: Vec3::ZERO, rot });
            self.world_fixed = true;
        }
        let scene_models: Vec<Option<TrackerModel>> =
            self.scenes.iter().map(|sc| self.devices.iter().find(|d| d.id == sc.dev).map(|d| d.model.clone())).collect();
        Ok(CalibWork {
            reason,
            ids,
            ref_idx,
            reference,
            from_scratch,
            scenes: self.scenes.clone(),
            scene_models,
            scene_poses,
            models,
            cals,
            ups,
            priors,
            min_sensors: cfg.scene_min_sensors,
            solve: cfg.solve,
            reject: cfg.calib_reject,
            stations_gen: self.stations_gen,
        })
    }

    /// A calibration is back: the stations (and the scene poses refined with them) as it
    /// solved them, unless they were placed otherwise meanwhile (then it runs again).
    fn apply_calibration(&mut self, id: u32, w: CalibWork, res: Result<CalibDone, &'static str>, out: &mut Vec<Event>) {
        let cfg = self.cfg;
        if self.cal_pending != Some(id) {
            return;
        }
        self.cal_pending = None;
        let again = self.cal_again.take();
        let reason = w.reason;
        if w.stations_gen != self.stations_gen {
            self.calibrate(again.unwrap_or(reason), out);
            return;
        }
        match res {
            Err(why) => {
                if reason != CalibrationReason::MoreScenes {
                    out.push(Event::CalibrationSkipped { reason, why });
                }
            }
            Ok(done) => {
                let cal = &done.cal;
                let mut reports = Vec::new();
                for (ki, &k) in done.known.iter().enumerate() {
                    let sid = w.ids[k];
                    let si = self.station_index(sid);
                    let st = &mut self.stations[si];
                    st.model = Some(cal.stations[ki].clone());
                    st.cov = Some(cal.station_cov[ki].clone());
                    reports.push(StationReport {
                        id: sid,
                        mode: st.info.as_ref().map(|i| i.mode),
                        world_from_station: cal.stations[ki].world_from_station,
                        cov: cal.station_cov[ki].clone(),
                        rms: cal.rms[ki],
                        up_error: cal.station_up_error[ki],
                        cal: cal.stations[ki].cal,
                    });
                }
                self.stations_gen += 1;
                self.reference = Some(w.reference);
                self.moved = false;
                self.solved_here |= w.from_scratch;
                // Scene poses refined by the fit (for the next round), onto the scenes as
                // they are now (by id: some may have been refreshed or dropped meanwhile).
                let mut it = cal.scenes.iter();
                for (j, sc) in w.scenes.iter().enumerate() {
                    if done.scene_poses[j].is_some() && sc.dev == done.dev0 {
                        let p = it.next().copied();
                        if let Some(cur) = self.scenes.iter_mut().find(|c| c.id == sc.id) {
                            cur.pose = p;
                        }
                    }
                }
                // Stations named and seen well that this round still couldn't place.
                let pending: Vec<u32> = self
                    .stations
                    .iter()
                    .filter(|s| s.model.is_none() && s.info.is_some())
                    .map(|s| s.id)
                    .filter(|id| self.scenes.iter().any(|sc| sensors_seen(sc, *id) >= cfg.scene_min_sensors))
                    .collect();
                out.push(Event::Calibrated {
                    reason,
                    stations: reports,
                    scenes: done.scenes,
                    dropped: cal.dropped,
                    tracker_up_error: cal.tracker_up_error,
                    pending,
                });
            }
        }
        if let Some(r) = again {
            self.calibrate(r, out);
        }
    }
}

/// A sweep's hits at unwrapped device times, in time order.
fn unwrap_hits(clock: &mut Clock, hits: &[SensorHits]) -> Vec<(i64, SensorHits)> {
    let mut v: Vec<(i64, SensorHits)> = hits.iter().map(|h| (clock.unwrap(h.candidates[0].time), h.clone())).collect();
    v.sort_by_key(|h| h.0);
    v
}

/// Two scenes of one device with nearly the same angles on the sensors they
/// share: the tracker was in the same place.
fn same_place(a: &SceneRecord, b: &SceneRecord) -> bool {
    if a.dev != b.dev {
        return false;
    }
    let mut d: Vec<Real> = Vec::new();
    for o in &b.obs {
        if let Some(p) = a.obs.iter().find(|p| p.0 == o.0 && p.1 == o.1 && p.2 == o.2) {
            d.push(crate::math::fabs(p.3 - o.3));
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
fn first_obs(sc: &SceneRecord, station: u32) -> Vec<(u8, u8, Real)> {
    let mut v: Vec<(u8, u8, Real)> = Vec::new();
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

/// sqrt of the position and orientation covariance traces (m, rad).
fn sigmas(st: &TrackState) -> (Real, Real) {
    let c = st.pose_cov();
    let tp = c[(0, 0)] + c[(1, 1)] + c[(2, 2)];
    let tr = c[(3, 3)] + c[(4, 4)] + c[(5, 5)];
    (crate::math::sqrt(tp.max(0.0)), crate::math::sqrt(tr.max(0.0)))
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

#[cfg(test)]
mod tests {
    //! Station discovery on a synthetic session: a tracker held still while a second
    //! station comes up after the first was calibrated (fughilli/splanc#465).
    use super::*;
    use crate::model::{faces, predict};
    use crate::timing::HitCandidate;

    const C: u32 = 0xC0C0;
    const B: u32 = 0xB0B0;
    /// IMU rate: 250 Hz.
    const DT: u32 = 192_000;

    fn tracker() -> TrackerModel {
        let mut sensors = Vec::new();
        let mut normals = Vec::new();
        for i in 0..16 {
            let a = i as Real * crate::consts::PI / 8.0;
            let z = if i % 2 == 0 { 0.0 } else { 0.015 };
            sensors.push(Vec3::new(0.04 * crate::math::cos(a), 0.04 * crate::math::sin(a), z));
            normals.push(Vec3::new(crate::math::cos(a), crate::math::sin(a), 0.5).normalized());
        }
        TrackerModel { sensors, normals, body_from_head: Pose::IDENTITY }
    }

    fn cal() -> [AxisCal; 2] {
        [AxisCal { phase: 0.01, tilt: -0.002, curve: 0.001, gibphase: 2.0, gibmag: 0.004 }; 2]
    }

    /// An upright station (+y up) at `pos` looking at the origin.
    fn station(pos: Vec3) -> StationModel {
        let z = pos.normalized(); // station -z looks at the origin
        let x = Vec3::new(0.0, 0.0, 1.0).cross(z).normalized();
        let y = z.cross(x);
        let m = [[x.x, y.x, z.x], [x.y, y.y, z.y], [x.z, y.z, z.z]];
        let w = crate::math::sqrt((1.0 + m[0][0] + m[1][1] + m[2][2]).max(1e-12)) / 2.0;
        let q = Quat::new(w, (m[2][1] - m[1][2]) / (4.0 * w), (m[0][2] - m[2][0]) / (4.0 * w), (m[1][0] - m[0][1]) / (4.0 * w));
        StationModel { cal: cal(), world_from_station: Pose { pos, rot: q.normalized() } }
    }

    fn info(id: u32, mode: u8, st: &StationModel) -> StationInfo {
        // Gravity in the station frame (OOTX: 8 bits per axis).
        let up = st.world_from_station.rot.conj().rotate(Vec3::new(0.0, 0.0, 1.0));
        let q = |v: Real| (v * 127.0) as i8;
        StationInfo { fw_version: 1, id, cal: cal(), unlock_count: 0, hw_version: 9, accel_dir: [q(up.x), q(up.y), q(up.z)], mode, faults: 0 }
    }

    struct Sim {
        k: Kernel,
        out: Vec<Event>,
        t: u32,
        dev: DeviceId,
        model: TrackerModel,
        c: StationModel,
        b: StationModel,
    }

    impl Sim {
        /// A kernel with station C loaded from a saved calibration and one tracker.
        fn new(auto_calibrate: bool) -> Sim {
            let cfg = KernelConfig { auto_calibrate, ..KernelConfig::default() };
            let mut sim = Sim {
                k: Kernel::new(cfg),
                out: Vec::new(),
                t: 1_000,
                dev: 0,
                model: tracker(),
                c: station(Vec3::new(2.0, -1.0, 2.0)),
                b: station(Vec3::new(-1.5, 2.0, 2.2)),
            };
            let mut cov = Mat::zeros(6, 6);
            for i in 0..6 {
                cov[(i, i)] = 1e-6;
            }
            let c = sim.c.world_from_station;
            sim.k.push(Input::StationPrior { id: C, world_from_station: c, cov, cal: Some(cal()) }, &mut sim.out);
            sim.add_device(0);
            sim
        }

        fn add_device(&mut self, dev: DeviceId) {
            self.dev = dev;
            self.k.push(Input::DeviceAdded { dev, model: self.model.clone(), wireless: false }, &mut self.out);
            // The tracker hears C's OOTX on slot 0 (B's slot isn't named yet).
            let di = self.k.devices.iter().position(|d| d.id == dev).unwrap();
            let inf = info(C, 2, &self.c);
            self.k.timing_event(di, TimingEvent::Station { slot: 0, info: inf }, &mut self.out);
        }

        /// B's OOTX decoded on slot 1.
        fn name_b(&mut self) {
            let di = self.k.devices.iter().position(|d| d.id == self.dev).unwrap();
            let inf = info(B, 1, &self.b);
            self.k.timing_event(di, TimingEvent::Station { slot: 1, info: inf }, &mut self.out);
        }

        /// `secs` at `pose`: IMU at 250 Hz, both stations sweeping both axes. `moving`:
        /// accel jitter, so the stillness detector ends the scene.
        fn hold(&mut self, pose: &Pose, secs: Real, moving: bool) {
            self.hold_jitter(pose, secs, if moving { 0.03 } else { 0.0 });
        }

        /// As `hold`, the accelerometer norm alternating ±`amp` (g) sample to sample.
        fn hold_jitter(&mut self, pose: &Pose, secs: Real, amp: Real) {
            let up = pose.rot.conj().rotate(Vec3::new(0.0, 0.0, 1.0));
            let n = (secs * 250.0) as u32;
            for i in 0..n {
                self.t = self.t.wrapping_add(DT);
                let jitter = if i % 2 == 0 { 1.0 + amp } else { 1.0 - amp };
                self.k.push(Input::Imu { dev: self.dev, time: self.t, accel_g: up.scale(jitter), gyro: Vec3::ZERO }, &mut self.out);
                let di = self.k.devices.iter().position(|d| d.id == self.dev).unwrap();
                for (slot, st) in [(0u8, self.c.clone()), (1u8, self.b.clone())] {
                    for axis in 0..2u8 {
                        let mut hits = Vec::new();
                        for (s, p) in self.model.sensors.iter().enumerate() {
                            if !faces(pose, *p, self.model.normals[s], &st) {
                                continue;
                            }
                            let angle = predict(&st, pose, *p, axis).angle;
                            hits.push(SensorHits { sensor: s as u8, candidates: alloc::vec![HitCandidate { angle, len: 500, time: self.t }] });
                        }
                        let sweep = Sweep { slot, axis, sync_time: self.t, part: 0, hits };
                        self.k.timing_event(di, TimingEvent::Sweep(sweep), &mut self.out);
                    }
                }
            }
        }

        /// A still stretch at `pose` ended by a little motion.
        fn scene(&mut self, pose: &Pose) {
            self.hold(pose, 3.0, false);
            self.hold(pose, 0.3, true);
        }

        fn calibrations(&self) -> Vec<(CalibrationReason, Vec<u32>)> {
            self.out
                .iter()
                .filter_map(|e| match e {
                    Event::Calibrated { reason, stations, .. } => Some((*reason, stations.iter().map(|s| s.id).collect())),
                    _ => None,
                })
                .collect()
        }

        /// The last calibration placed B where it is.
        fn b_placed(&self) -> bool {
            self.out.iter().rev().find_map(|e| match e {
                Event::Calibrated { stations, .. } => stations.iter().find(|s| s.id == B).map(|s| s.world_from_station.pos.sub(self.b.world_from_station.pos).norm() < 0.01),
                _ => None,
            }) == Some(true)
        }
    }

    impl Sim {
        /// As `new`, slow solves handed out (KernelConfig::defer_solves).
        fn deferred(auto_calibrate: bool) -> Sim {
            let mut sim = Sim::new(auto_calibrate);
            sim.k.cfg.defer_solves = true;
            sim
        }

        /// The jobs handed out so far, taken out of `out`.
        fn take_jobs(&mut self) -> Vec<SolveJob> {
            let mut jobs = Vec::new();
            let mut rest = Vec::new();
            for e in self.out.drain(..) {
                match e {
                    Event::Solve(j) => jobs.push(j),
                    e => rest.push(e),
                }
            }
            self.out = rest;
            jobs
        }

        /// Runs `jobs` (as a worker would) and pushes their results back.
        fn finish(&mut self, jobs: Vec<SolveJob>) {
            for j in jobs {
                let r = run_job(j);
                self.k.push(Input::Solved(r), &mut self.out);
            }
        }

        fn started(&self) -> usize {
            self.out.iter().filter(|e| matches!(e, Event::TrackingStarted { .. })).count()
        }

        fn lost(&self) -> Vec<TrackLoss> {
            self.out.iter().filter_map(|e| if let Event::TrackingLost { reason, .. } = e { Some(*reason) } else { None }).collect()
        }
    }

    /// The P4 (PixieLamp2, 2026-11-17): a pose bootstrap took ~1.5 s inline, the light of
    /// that stretch came after it, and the fresh track was lost (NoLight) the moment it
    /// started. Deferred, light keeps coming in while the solve is out, and the track it
    /// starts takes that light.
    #[test]
    fn a_deferred_bootstrap_starts_a_track_that_has_the_light_since() {
        let mut sim = Sim::deferred(false);
        let p = table();
        sim.hold(&p, 0.1, false);
        let jobs: Vec<SolveJob> = sim.take_jobs().into_iter().filter(|j| j.kind() == "start").collect();
        assert_eq!(jobs.len(), 1);
        assert_eq!(sim.started(), 0);
        // 1.2 s of light and IMU while it's solved, and no second bootstrap meanwhile.
        sim.hold(&p, 1.2, false);
        assert!(sim.take_jobs().iter().all(|j| j.kind() != "start"));
        sim.finish(jobs);
        assert_eq!(sim.started(), 1);
        // Right away a pose at the newest light, where the tracker is.
        let (time, pos) = sim
            .out
            .iter()
            .rev()
            .find_map(|e| if let Event::Pose { time, world_from_body, .. } = e { Some((*time, world_from_body.pos)) } else { None })
            .expect("no pose from the backlog");
        assert!((sim.t as i64 - time).abs() <= 2 * DT as i64, "pose at {} vs now {}", time, sim.t);
        assert!(pos.sub(p.pos).norm() < 0.01, "{:?}", pos);
        sim.hold(&p, 1.5, false);
        assert!(sim.lost().is_empty(), "{:?}", sim.lost());
    }

    #[test]
    fn a_deferred_bootstrap_out_too_long_is_given_up_for_a_fresh_one() {
        let mut sim = Sim::deferred(false);
        let p = table();
        sim.hold(&p, 0.1, false);
        let old: Vec<SolveJob> = sim.take_jobs().into_iter().filter(|j| j.kind() == "start").collect();
        assert_eq!(old.len(), 1);
        // More light than max_backlog keeps (2 sweeps per IMU sample here).
        sim.hold(&p, 2.0, false);
        let fresh: Vec<SolveJob> = sim.take_jobs().into_iter().filter(|j| j.kind() == "start").collect();
        assert_eq!(fresh.len(), 1, "no fresh bootstrap after giving up");
        sim.finish(old);
        assert_eq!(sim.started(), 0, "a given-up bootstrap started a track");
        sim.finish(fresh);
        assert_eq!(sim.started(), 1);
        sim.hold(&p, 0.5, false);
        assert!(sim.lost().is_empty(), "{:?}", sim.lost());
    }

    /// a_station_named_later_joins_while_the_tracker_stays_put with the solves out (each
    /// back a little later, as from a worker).
    #[test]
    fn deferred_calibrations_place_a_station_named_later() {
        let mut sim = Sim::deferred(true);
        let p = table();
        let mut run_out = |sim: &mut Sim| {
            for _ in 0..6 {
                let jobs = sim.take_jobs();
                if jobs.is_empty() {
                    break;
                }
                sim.hold(&p, 0.2, false);
                sim.finish(jobs);
            }
        };
        sim.scene(&p);
        run_out(&mut sim);
        assert!(sim.calibrations().iter().all(|c| !c.1.contains(&B)));
        sim.name_b();
        sim.scene(&p);
        run_out(&mut sim);
        let cals = sim.calibrations();
        assert!(cals.iter().any(|c| c.0 == CalibrationReason::NewStation && c.1.contains(&C) && c.1.contains(&B)), "{:?}", cals);
        assert!(sim.b_placed());
    }

    #[test]
    fn a_deferred_calibration_the_stations_moved_under_runs_again() {
        let mut sim = Sim::deferred(true);
        let p = table();
        sim.scene(&p);
        let early = sim.take_jobs();
        sim.finish(early);
        // B named: the next scene places it.
        sim.name_b();
        sim.scene(&p);
        let jobs: Vec<SolveJob> = sim.take_jobs().into_iter().filter(|j| j.kind() == "calibrate").collect();
        assert_eq!(jobs.len(), 1);
        // A station placed again meanwhile (another host's calibration, say).
        let mut cov = Mat::zeros(6, 6);
        for i in 0..6 {
            cov[(i, i)] = 1e-6;
        }
        let c = sim.c.world_from_station;
        sim.k.push(Input::StationPrior { id: C, world_from_station: c, cov, cal: Some(cal()) }, &mut sim.out);
        let before = sim.calibrations().len();
        sim.finish(jobs);
        assert_eq!(sim.calibrations().len(), before, "a stale calibration was applied");
        let again: Vec<SolveJob> = sim.take_jobs().into_iter().filter(|j| j.kind() == "calibrate").collect();
        assert_eq!(again.len(), 1, "it didn't run again");
        sim.finish(again);
        assert_eq!(sim.calibrations().len(), before + 1);
        assert!(sim.b_placed());
    }

    /// A tracker at rest whose accelerometer reads right at `still_accel` (PixieLamp2's
    /// Vive Tracker 3.0: 0.0039-0.0041 g): the still stretch is one scene, not a restart
    /// every time the noise crosses the threshold.
    #[test]
    fn accelerometer_noise_at_the_threshold_still_makes_a_scene() {
        let mut sim = Sim::new(false);
        let p = table();
        sim.hold_jitter(&p, 0.6, 0.0035);
        for _ in 0..8 {
            sim.hold_jitter(&p, 0.2, 0.0035);
            sim.hold_jitter(&p, 0.2, 0.0045);
        }
        sim.hold(&p, 0.3, true);
        assert_eq!(sim.k.scenes.len(), 1, "the stretch made no scene");
        // Past the margin it is moving: no scene from that.
        let mut sim = Sim::new(false);
        sim.hold_jitter(&p, 0.6, 0.0035);
        for _ in 0..8 {
            sim.hold_jitter(&p, 0.2, 0.0035);
            sim.hold_jitter(&p, 0.2, 0.0080);
        }
        assert_eq!(sim.k.scenes.len(), 0, "shaking made a scene");
    }

    fn table() -> Pose {
        Pose { pos: Vec3::new(0.2, -0.1, 0.9), rot: Quat::from_rotvec(Vec3::new(0.0, 0.0, 1.3)) }
    }

    #[test]
    fn a_station_named_later_joins_while_the_tracker_stays_put() {
        let mut sim = Sim::new(true);
        let p = table();
        // Still on the table: only C is named, so only C's hits make the scene.
        sim.scene(&p);
        assert!(sim.calibrations().iter().all(|c| !c.1.contains(&B)));
        // B's OOTX decodes; the tracker is still in the same place.
        sim.name_b();
        sim.scene(&p);
        let cals = sim.calibrations();
        assert!(cals.iter().any(|c| c.0 == CalibrationReason::NewStation && c.1.contains(&C) && c.1.contains(&B)), "{:?}", cals);
        assert!(sim.b_placed());
    }

    #[test]
    fn recalibrate_adds_a_station_seen_only_in_the_same_place() {
        let mut sim = Sim::new(false);
        let p = table();
        sim.scene(&p);
        sim.name_b();
        sim.scene(&p);
        assert!(sim.calibrations().is_empty());
        sim.k.push(Input::Recalibrate, &mut sim.out);
        let cals = sim.calibrations();
        assert_eq!(cals.len(), 1, "{:?}", cals);
        assert!(cals[0].0 == CalibrationReason::Requested && cals[0].1.contains(&B), "{:?}", cals);
        assert!(sim.b_placed());
    }

    #[test]
    fn scenes_of_a_tracker_that_came_back_as_a_new_device_dont_block_calibration() {
        let mut sim = Sim::new(true);
        let p = table();
        sim.scene(&p);
        // The tracker drops off the radio and comes back under a new device id; its old
        // scenes can't be used any more (the kernel has no model for them).
        sim.k.push(Input::DeviceRemoved { dev: 0 }, &mut sim.out);
        sim.add_device(1);
        sim.name_b();
        let q = Pose { pos: Vec3::new(-0.3, 0.4, 1.0), rot: Quat::from_rotvec(Vec3::new(0.0, 0.0, -0.6)) };
        sim.scene(&q);
        let cals = sim.calibrations();
        assert!(cals.iter().any(|c| c.1.contains(&B)), "{:?}", cals);
        assert!(sim.b_placed());
        // Asked to recalibrate with nothing usable, it says why instead of staying silent.
        let mut k = Kernel::new(KernelConfig::default());
        let mut out = Vec::new();
        k.push(Input::Recalibrate, &mut out);
        assert!(matches!(out[..], [Event::CalibrationSkipped { reason: CalibrationReason::Requested, .. }]), "{:?}", out);
    }
}

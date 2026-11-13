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
//! Stations come and go: one whose light no tracker has seen for
//! [`KernelConfig::station_expire`] while trackers do see light is dropped
//! ([`Event::StationDropped`]), its pose and its observations in the scenes with it, so
//! a station switched off or taken away stops counting (and is solved afresh if it comes
//! back). A station that is tilted in place says so itself: its OOTX gravity vector moves.
//! Past [`KernelConfig::station_tilt`] its orientation is levelled again at once (the
//! smallest rotation that puts its new gravity where its old one was in the world; position
//! and heading kept), its old observations dropped, and the scenes that follow refine it
//! (a `Calibrated` with reason [`CalibrationReason::StationTilted`] reports the new pose).
//!
//! World frame: z up (gravity, from the trackers' accelerometers and the
//! stations' OOTX accel vectors). Without a prior, the first calibration sets
//! the origin at the tracker's IMU in its first still scene and the x axis
//! along that tracker's body x (levelled). A loaded calibration keeps its frame.

use crate::Real;
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
    /// Forget a station: its pose, prior and observations (a host that owns the stations,
    /// lhd for the P4, stopped naming it). Reported as [`Event::StationDropped`].
    ForgetStation { id: u32 },
    /// Restart a device's tracking (its pose solve), keeping its sweep-timing lock and which
    /// station is in which slot: the stations' world changed under it (a host adopted new
    /// station poses). Unlike DeviceRemoved + DeviceAdded, nothing waits for OOTX again.
    ResetTrack { dev: DeviceId },
    /// The host's word that the device's clock has passed `time` (ticks, wrapping) and
    /// every pulse before it was pushed: a host that maps its clock to the device's sends
    /// this between reports, and a sweep then goes out as soon as the laser has passed
    /// (TimingConfig::sweep_quiet_ticks) instead of at the next cycle's sync.
    Clock { dev: DeviceId, time: u32 },
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
    /// A station was dropped: its pose, prior and scene observations are gone. Light from
    /// it again names it anew, and the next calibration solves it from fresh scenes.
    StationDropped { id: u32, why: StationDrop },
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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationReason {
    FirstStations,
    NewStation,
    MoreScenes,
    Requested,
    /// The stations disagreed: the old calibration was dropped.
    StationsMoved,
    /// A station's OOTX gravity vector moved: its orientation was levelled again from it
    /// (no fit; the stations' poses as they now are).
    StationTilted,
    /// A station moved while a tracker sat still (the tracker's optical pose drifted with
    /// its IMU still): solved again from that tracker's view once it settled.
    StationRelocated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StationDrop {
    /// No tracker saw its light for `station_expire` while they saw light.
    NotSeen,
    /// [`Input::ForgetStation`].
    Forgotten,
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
    /// A station no present tracker has seen light from for this long (s), while one of
    /// them has seen light for that long, is dropped (Event::StationDropped). 0: never.
    pub station_expire: Real,
    /// Stations count as seen by a tracker, for its angle-bias states (which need two),
    /// while their light reached it within this long (s).
    pub station_active: Real,
    /// A station's OOTX gravity vector moving by more than this (rad) since its pose was
    /// set means it was tilted: its orientation is levelled again (CalibrationReason::
    /// StationTilted). The vector is 8 bits per axis, ~0.5 degrees per count, and a still
    /// station's reading wanders by ~1 degree: a tilt must show in two packets in a row,
    /// agreeing within half of this.
    pub station_tilt: Real,
    /// A tracker still (IMU) and tracking on one station keeps a reference pose once its
    /// position sigma is under `still_ref_sigma` (m). If the optical pose then drifts past
    /// `relocate_pos` (m) or `relocate_rot` (rad) from it, the station moved, not the tracker:
    /// the tracker is held at the reference, the station's hits stop feeding the filter, and
    /// the station is solved again from them every `relocate_window` (s) until two solves in a
    /// row agree within `relocate_agree_pos` / `relocate_agree_rot` (it settled), then
    /// adopted (Calibrated, StationRelocated). Gives up after `relocate_timeout` (s), or as
    /// soon as the tracker moves.
    pub still_ref_sigma: Real,
    pub relocate_pos: Real,
    pub relocate_rot: Real,
    pub relocate_window: Real,
    pub relocate_agree_pos: Real,
    pub relocate_agree_rot: Real,
    pub relocate_timeout: Real,
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
            station_expire: 60.0,
            station_active: 2.0,
            station_tilt: (3.0 as Real).to_radians(),
            still_ref_sigma: 0.008,
            relocate_pos: 0.03,
            relocate_rot: (2.0 as Real).to_radians(),
            relocate_window: 0.6,
            relocate_agree_pos: 0.01,
            relocate_agree_rot: (0.3 as Real).to_radians(),
            relocate_timeout: 30.0,
        }
    }
}

const TPS: Real = crate::lh1::TICKS_PER_SECOND;

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
    /// Gravity in the station frame (its OOTX accel vector, normalised) when `model` was
    /// set; None until both are known. Tilt is judged against it.
    model_up: Option<Vec3>,
    /// A first OOTX reading past the tilt threshold, waiting for the next to confirm it.
    tilt_pending: Option<Vec3>,
}

#[derive(Default)]
struct StillDetector {
    gyro: VecDeque<(i64, Real)>,
    acc: VecDeque<(i64, Real)>,
    still: bool,
    since: i64,
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
#[derive(Clone)]
struct SceneRecord {
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
    /// Per station, the newest sweep's time (ticks, unwrapped); and the first sweep's.
    station_light: Vec<(u32, i64)>,
    first_light: Option<i64>,
    /// IMU time of the last expiry check.
    last_expiry_check: i64,
    /// Still and tracking: the pose to hold it to, and its position sigma then.
    still_ref: Option<(Pose, Real)>,
    /// A station moving under this still tracker (see KernelConfig::relocate_*).
    relocating: Option<Relocating>,
}

struct Relocating {
    station: u32,
    held: Pose,
    since: i64,
    window_start: i64,
    sb: SceneBuilder,
    last: Option<StationModel>,
    hits: u32,
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
                    station_light: Vec::new(),
                    first_light: None,
                    last_expiry_check: 0,
                    still_ref: None,
                    relocating: None,
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
                st.model_up = st.info.as_ref().map(info_up);
                // Prior strength from the saved covariance, floored: a saved
                // calibration is trusted to ~mm / ~mrad at best, and its gauge
                // directions (zero variance) aren't certainty.
                let sd = |i: usize| crate::math::sqrt(cov[(i, i)].max(0.0));
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
            Input::ForgetStation { id } => self.drop_station(id, StationDrop::Forgotten, out),
            Input::ResetTrack { dev } => {
                let Some(d) = self.devices.iter_mut().find(|d| d.id == dev) else { return };
                d.still_ref = None;
                d.relocating = None;
                d.coasting = false;
                d.recent.clear();
                d.bias_slots.clear();
                if d.track.take().is_some() {
                    out.push(Event::TrackingLost { dev, reason: TrackLoss::StationsMoved });
                }
            }
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
                self.stations.push(StationEntry { id, prior: None, info: None, cal: None, model: None, cov: None, model_up: None, tilt_pending: None });
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
        if cfg.station_expire > 0.0 && t - d.last_expiry_check >= TPS as i64 {
            d.last_expiry_check = t;
            self.expire_stations(out);
        }
        let d = &mut self.devices[di];
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
        if still != d.still.still {
            d.still.still = still;
            d.still.since = t;
            out.push(Event::Still { dev, still });
            if !still {
                // The tracker moves: no reference to hold it to; a station still relocating
                // can't be told apart from the motion now.
                let d = &mut self.devices[di];
                d.still_ref = None;
                if d.relocating.take().is_some() {
                    d.recent.clear();
                    if d.track.take().is_some() {
                        out.push(Event::TrackingLost { dev, reason: TrackLoss::StationsMoved });
                    }
                }
                self.finish_scene(di, out);
                return;
            }
        }
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
        if still {
            if let Some(sb) = d.scene.as_mut() {
                sb.acc_sum = sb.acc_sum.add(accel_g);
                sb.acc_n += 1;
            }
            let dur = (t - d.still.since) as Real / TPS;
            // The pose to hold a still tracker to (relocation): once settled, and again
            // whenever it has become much more certain.
            if dur >= 0.3 && d.relocating.is_none() {
                if let Some(track) = d.track.as_ref() {
                    let sp = sigmas(track).0;
                    let take = sp < cfg.still_ref_sigma && d.still_ref.map_or(true, |(_, r)| sp < r * 0.5);
                    if take {
                        d.still_ref = Some((track.pose(), sp));
                    }
                }
            }
            if d.scene.is_none() && dur >= 0.3 {
                d.scene = Some(SceneBuilder { start: t, clusters: Vec::new(), acc_sum: Vec3::ZERO, acc_n: 0 });
            }
            if d.scene.as_ref().map_or(false, |s| (t - s.start) as Real / TPS >= cfg.scene_max) {
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
        let dur = d.last_imu.map_or(0, |x| x.0 - sb.start) as Real / TPS;
        if dur < self.cfg.scene_min || sb.acc_n == 0 {
            return;
        }
        let obs = scene_obs(&sb, 8);
        let up = sb.acc_sum.scale(1.0 / sb.acc_n as Real);
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
            self.scenes[j] = SceneRecord { pose: pose.or(rec.pose), ..rec };
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
                let up = info_up(&info);
                st.info = Some(info.clone());
                if changed || new_info {
                    out.push(Event::StationSeen { dev, slot, info });
                }
                self.check_tilt(si, up, out);
            }
            TimingEvent::OotxError { .. } => {}
            TimingEvent::Sweep(s) => self.sweep(di, s, out),
        }
    }

    fn sweep(&mut self, di: usize, s: Sweep, out: &mut Vec<Event>) {
        {
            let d = &mut self.devices[di];
            // Back once sweeps came from every slot that had a station: both with two (which
            // pins the cycle phase), slot 0 alone with a lone station (it never has a slave;
            // waiting for a second slot left it to the next OOTX packet, ~6 s).
            let need = (d.prev_slot_station[0].is_some() as u8) | ((d.prev_slot_station[1].is_some() as u8) << 1);
            if d.relock_slots != 3 && d.slot_station.contains(&None) && need != 0 {
                d.relock_slots |= 1 << (s.slot & 1);
                if d.relock_slots & need == need {
                    for slot in 0..2 {
                        if d.slot_station[slot].is_none() {
                            d.slot_station[slot] = d.prev_slot_station[slot];
                        }
                    }
                    d.relock_slots = 3;
                }
            }
        }
        let Some(sid) = self.devices[di].slot_station[s.slot as usize] else { return };
        {
            let d = &mut self.devices[di];
            let t = d.clock.unwrap(s.sync_time);
            d.first_light.get_or_insert(t);
            match d.station_light.iter_mut().find(|x| x.0 == sid) {
                Some(x) => x.1 = x.1.max(t),
                None => d.station_light.push((sid, t)),
            }
        }
        if let Some(sb) = self.devices[di].scene.as_mut() {
            sb.add(sid, s.axis, &s.hits);
        }
        let Some(si) = self.stations.iter().position(|x| x.id == sid) else { return };
        if self.devices[di].track.is_some() && self.devices[di].relocating.as_ref().map_or(false, |r| r.station == sid) {
            self.relocate_sweep(di, si, &s, out);
            return;
        }
        let Some(station) = self.stations[si].model.clone() else { return };
        let station_cov = self.stations[si].cov.clone();
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
            self.try_start_tracking(di, out);
            return;
        }

        // Angle-bias states only with two or more calibrated stations this tracker sees:
        // with one, a bias can't be told apart from the tracker moving. A station it
        // doesn't see (switched off, gone, or out of its view) doesn't count.
        let calibrated = {
            let d = &self.devices[di];
            let now = d.station_light.iter().map(|x| x.1).max().unwrap_or(0);
            let window = (cfg.station_active * TPS) as i64;
            d.station_light
                .iter()
                .filter(|(id, t)| now - *t <= window && self.stations.iter().any(|s| s.id == *id && s.model.is_some()))
                .count()
        };
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
            if s.part == 0 {
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
        // Still, on one station, and the optical pose walked away from where the tracker
        // settled: the IMU says it didn't move, so the station did (KernelConfig::relocate_*).
        // Hold the tracker there and solve the station again once it settles.
        if calibrated == 1 && d.relocating.is_none() {
            if let Some((held, ref_sigma)) = d.still_ref {
                let p = track.pose();
                let dpos = p.pos.sub(held.pos).norm();
                let drot = rot_between(p.rot, held.rot);
                if dpos > cfg.relocate_pos.max(4.0 * ref_sigma) || drot > cfg.relocate_rot {
                    let track = d.track.as_mut().unwrap();
                    track.pos = held.pos;
                    track.rot = held.rot;
                    track.vel = Vec3::ZERO;
                    d.relocating = Some(Relocating {
                        station: sid,
                        held,
                        since: now,
                        window_start: now,
                        sb: new_scene(now),
                        last: None,
                        hits: 0,
                    });
                }
            }
        }
        let speculative = d.coasting || dark > cfg.coast_after;
        publish_pose(d, now, speculative, &cfg, out);
    }

    /// A sweep from the station moving under a still tracker (Device::relocating): its hits
    /// go into the station's solve, not the track; the tracker is held where it settled.
    fn relocate_sweep(&mut self, di: usize, si: usize, s: &Sweep, out: &mut Vec<Event>) {
        let cfg = self.cfg;
        let sid = self.stations[si].id;
        let up = self.stations[si].info.as_ref().map(info_up);
        let init = self.stations[si].model.clone();
        let d = &mut self.devices[di];
        let now = d.clock.unwrap(s.sync_time);
        let n: u32 = s.hits.len() as u32;
        let Some(r) = d.relocating.as_mut() else { return };
        r.sb.add(sid, s.axis, &s.hits);
        r.hits += n;
        let held = r.held;
        // The tracker is still: keep it where it settled (the IMU's zero-velocity updates
        // hold the rest).
        if let Some(track) = d.track.as_mut() {
            if now > track.time {
                advance(track, &mut d.imu, now, d.last_imu, &cfg.track);
            }
            track.pos = held.pos;
            track.rot = held.rot;
            track.vel = Vec3::ZERO;
        }
        d.last_used = now;
        d.gated_run = 0;
        d.coasting = false;
        // The light is in use (solving the station), so the pose isn't dark.
        d.window.used += n;
        let r = d.relocating.as_mut().unwrap();
        if (now - r.since) as Real > cfg.relocate_timeout * TPS {
            d.relocating = None;
            d.still_ref = None;
            d.recent.clear();
            if d.track.take().is_some() {
                out.push(Event::TrackingLost { dev: d.id, reason: TrackLoss::StationsMoved });
            }
            return;
        }
        let mut settled: Option<StationModel> = None;
        if (now - r.window_start) as Real >= cfg.relocate_window * TPS {
            let mut obs: Vec<Obs> = Vec::new();
            for o in scene_obs(&r.sb, 4).iter().filter(|o| o.0 == sid) {
                if !obs.iter().any(|x| x.axis == o.1 && x.sensor == o.2 as u16) {
                    obs.push(Obs { station: 0, axis: o.1, sensor: o.2 as u16, angle: o.3 });
                }
            }
            let mut seen = 0u64;
            for o in &obs {
                seen |= 1 << (o.sensor & 63);
            }
            let solved = match (&init, seen.count_ones() as usize >= cfg.scene_min_sensors) {
                (Some(init), true) => crate::solve::relocate_station(&d.model, init, &held, &obs, up, &cfg.solve),
                _ => None,
            };
            match solved {
                Some((st, rms)) if rms < 2e-3 => {
                    let agree = r.last.as_ref().map_or(false, |p| {
                        p.world_from_station.pos.sub(st.world_from_station.pos).norm() < cfg.relocate_agree_pos
                            && rot_between(p.world_from_station.rot, st.world_from_station.rot) < cfg.relocate_agree_rot
                    });
                    if agree {
                        settled = Some(st);
                    } else {
                        r.last = Some(st);
                    }
                }
                _ => r.last = None,
            }
            r.sb = new_scene(now);
            r.window_start = now;
        }
        if let Some(st_model) = settled {
            let sigma = d.track.as_ref().map_or(0.0, |t| sigmas(t).0);
            d.relocating = None;
            d.still_ref = Some((held, sigma.max(1e-4)));
            let entry = &mut self.stations[si];
            entry.model = Some(st_model.clone());
            entry.prior = None;
            entry.model_up = entry.info.as_ref().map(info_up);
            entry.tilt_pending = None;
            // Every placed station, as a calibration reports them (lhd saves the report: one
            // station alone would drop the others from its file).
            let reports: Vec<StationReport> = self
                .stations
                .iter()
                .filter_map(|e| {
                    let m = e.model.as_ref()?;
                    Some(StationReport {
                        id: e.id,
                        mode: e.info.as_ref().map(|i| i.mode),
                        world_from_station: m.world_from_station,
                        cov: e.cov.clone().unwrap_or_else(|| Mat::zeros(6, 6)),
                        rms: 0.0,
                        up_error: 0.0,
                        cal: m.cal,
                    })
                })
                .collect();
            for sc in &mut self.scenes {
                sc.obs.retain(|o| o.0 != sid);
            }
            let min = cfg.scene_min_sensors;
            self.scenes.retain(|sc| sc.obs.iter().any(|o| sensors_seen(sc, o.0) >= min));
            out.push(Event::Calibrated {
                reason: CalibrationReason::StationRelocated,
                stations: reports,
                scenes: 0,
                dropped: 0,
                tracker_up_error: 0.0,
                pending: Vec::new(),
            });
        }
        let d = &mut self.devices[di];
        publish_pose(d, now, false, &cfg, out);
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
        let up = d.last_imu.map(|x| x.1);
        let Some(sol) = bootstrap_pose_assoc(&d.model, &models, &cands, up, 3e-3, &cfg.solve) else { return };
        if !(sol.rms < 2e-3) || sol.inliers < 8 || sol.inliers * 3 < sol.obs * 2 {
            return;
        }
        let t = d.last_imu.map_or(0, |x| x.0);
        d.track = Some(TrackState::new(&sol.world_from_body, &sol.cov.scale(4.0), t, cfg.track.angle_bias_init));
        d.last_used = t;
        d.gated_run = 0;
        d.coasting = false;
        d.recent.clear();
        out.push(Event::TrackingStarted { dev: d.id, rms: sol.rms });
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

    // ------------------------------------------------------------ station lifecycle

    /// Drops stations no present tracker has seen for `station_expire` while one has seen
    /// light for that long (so a tracker in a drawer, or one just switched on, expires
    /// nothing). Times are each tracker's own clock (sweep times).
    fn expire_stations(&mut self, out: &mut Vec<Event>) {
        let expire = (self.cfg.station_expire * TPS) as i64;
        // Each tracker's clock: its newest IMU sample. A tracker "watches" while light
        // reaches it and has for `station_expire`.
        let now = |d: &Device| d.last_imu.map(|x| x.0);
        let sees = |d: &Device, id: u32| match now(d) {
            Some(n) => d.station_light.iter().any(|(sid, t)| *sid == id && n - *t < expire),
            None => false,
        };
        let watching = self.devices.iter().any(|d| match (now(d), d.first_light) {
            (Some(n), Some(first)) => n - first >= expire && d.station_light.iter().any(|x| n - x.1 < expire),
            _ => false,
        });
        if !watching {
            return;
        }
        let stale: Vec<u32> = self
            .stations
            .iter()
            .filter(|s| s.model.is_some() || s.prior.is_some())
            .map(|s| s.id)
            .filter(|id| !self.devices.iter().any(|d| sees(d, *id)))
            .collect();
        for id in stale {
            self.drop_station(id, StationDrop::NotSeen, out);
        }
    }

    /// Forgets a station's pose, prior and its observations in the scenes; tracking that
    /// used it starts again without it. With no placed station left, the world frame goes
    /// too: the next calibration starts from scratch.
    fn drop_station(&mut self, id: u32, why: StationDrop, out: &mut Vec<Event>) {
        let Some(si) = self.stations.iter().position(|s| s.id == id) else { return };
        let st = &mut self.stations[si];
        if st.model.is_none() && st.prior.is_none() && why == StationDrop::NotSeen {
            return;
        }
        st.model = None;
        st.prior = None;
        st.cov = None;
        st.model_up = None;
        for sc in &mut self.scenes {
            sc.obs.retain(|o| o.0 != id);
        }
        let min = self.cfg.scene_min_sensors;
        self.scenes.retain(|sc| sc.obs.iter().any(|o| sensors_seen(sc, o.0) >= min));
        if self.reference == Some(id) {
            self.reference = None;
        }
        if !self.stations.iter().any(|s| s.model.is_some()) {
            self.world_fixed = false;
            self.solved_here = false;
            for sc in &mut self.scenes {
                sc.pose = None;
            }
        }
        for d in &mut self.devices {
            d.station_light.retain(|x| x.0 != id);
            let used = d.slot_station.contains(&Some(id));
            if used && d.track.take().is_some() {
                d.recent.clear();
                d.coasting = false;
                out.push(Event::TrackingLost { dev: d.id, reason: TrackLoss::StationsMoved });
            }
            d.recent.retain(|r| r.0 != id);
        }
        out.push(Event::StationDropped { id, why });
    }

    /// Station `si`'s OOTX gravity is now `up` (station frame). Past `station_tilt` from
    /// the one its pose was set with, it was tilted: rotate its orientation by the smallest
    /// rotation that puts `up` where the old gravity was in the world (position and heading
    /// as they were; gravity can't tell the heading), drop its old observations, and
    /// restart tracking that used it. Scenes to come refine it.
    fn check_tilt(&mut self, si: usize, up: Vec3, out: &mut Vec<Event>) {
        let thr = self.cfg.station_tilt;
        let st = &mut self.stations[si];
        if st.model.is_none() {
            return;
        }
        let Some(old) = st.model_up else {
            st.model_up = Some(up);
            return;
        };
        let angle = |a: Vec3, b: Vec3| crate::math::atan2(a.cross(b).norm(), a.dot(b));
        if !(angle(old, up) > thr) {
            st.tilt_pending = None;
            return;
        }
        // One packet past the threshold could be the accelerometer's wander (~1 degree) or a
        // bump; a tilt is the same new reading twice.
        match st.tilt_pending {
            Some(p) if angle(p, up) <= thr / 2.0 => st.tilt_pending = None,
            _ => {
                st.tilt_pending = Some(up);
                return;
            }
        }
        let model = st.model.as_mut().unwrap();
        let rot = model.world_from_station.rot;
        let fix = quat_between(rot.rotate(up), rot.rotate(old));
        model.world_from_station.rot = fix.mul(rot).normalized();
        st.model_up = Some(up);
        if let Some(pr) = st.prior.as_mut() {
            // Heading and position as before; the tilt is new, so trust the rotation less.
            pr.pose = model.world_from_station;
            pr.sigma_rot = pr.sigma_rot.max(0.05);
        }
        let id = st.id;
        for sc in &mut self.scenes {
            sc.obs.retain(|o| o.0 != id);
        }
        let min = self.cfg.scene_min_sensors;
        self.scenes.retain(|sc| sc.obs.iter().any(|o| sensors_seen(sc, o.0) >= min));
        for d in &mut self.devices {
            if d.slot_station.contains(&Some(id)) && d.track.take().is_some() {
                d.recent.clear();
                d.coasting = false;
                out.push(Event::TrackingLost { dev: d.id, reason: TrackLoss::StationsMoved });
            }
            d.recent.retain(|r| r.0 != id);
        }
        let stations: Vec<StationReport> = self
            .stations
            .iter()
            .filter_map(|s| {
                let m = s.model.as_ref()?;
                Some(StationReport {
                    id: s.id,
                    mode: s.info.as_ref().map(|i| i.mode),
                    world_from_station: m.world_from_station,
                    cov: s.cov.clone().unwrap_or_else(|| Mat::zeros(6, 6)),
                    rms: 0.0,
                    up_error: s.model_up.map_or(0.0, |u| {
                        let w = m.world_from_station.rot.rotate(u);
                        crate::math::atan2(w.cross(Vec3::new(0.0, 0.0, 1.0)).norm(), w.z)
                    }),
                    cal: m.cal,
                })
            })
            .collect();
        out.push(Event::Calibrated {
            reason: CalibrationReason::StationTilted,
            stations,
            scenes: 0,
            dropped: 0,
            tracker_up_error: 0.0,
            pending: Vec::new(),
        });
    }

    fn calibrate(&mut self, reason: CalibrationReason, out: &mut Vec<Event>) {
        if let Err(why) = self.try_calibrate(reason, out) {
            if reason != CalibrationReason::MoreScenes {
                out.push(Event::CalibrationSkipped { reason, why });
            }
        }
    }

    /// Scenes usable for calibration: their device is still here (its tracker model is
    /// needed to use their angles).
    fn scene_usable(&self, sc: &SceneRecord) -> bool {
        self.devices.iter().any(|d| d.id == sc.dev)
    }

    /// Poses the scenes without one against the stations placed so far (`models`, by
    /// index into `ids`).
    fn pose_scenes(&self, ids: &[u32], models: &[Option<StationModel>], scene_poses: &mut [Option<Pose>]) {
        let known: Vec<usize> = (0..ids.len()).filter(|&k| models[k].is_some()).collect();
        if known.is_empty() {
            return;
        }
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
            if let Some(sol) = bootstrap_pose(&model, &km, &obs, Some(sc.up_body), &self.cfg.solve) {
                if sol.rms < 3e-3 {
                    scene_poses[j] = Some(sol.world_from_body);
                }
            }
        }
    }

    fn try_calibrate(&mut self, reason: CalibrationReason, out: &mut Vec<Event>) -> Result<(), &'static str> {
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

        // Scene poses: tracked pose if there was one, else solved below.
        let mut scene_poses: Vec<Option<Pose>> = self.scenes.iter().map(|s| if self.scene_usable(s) { s.pose } else { None }).collect();
        let mut models: Vec<Option<StationModel>> =
            ids.iter().map(|id| self.stations.iter().find(|s| s.id == *id).unwrap().model.clone()).collect();
        let cal_of = |k: &Kernel, id: u32| k.stations.iter().find(|s| s.id == id).unwrap().cal.unwrap();
        let up_of = |k: &Kernel, id: u32| -> Option<Vec3> {
            k.stations.iter().find(|s| s.id == id).unwrap().info.as_ref().map(|i| {
                Vec3::new(i.accel_dir[0] as Real, i.accel_dir[1] as Real, i.accel_dir[2] as Real).normalized()
            })
        };

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
        } else {
            // Scenes collected before tracking started have no pose yet: place them
            // against the stations already known first, so a station seen only in such
            // scenes can be solved from them right away.
            self.pose_scenes(&ids, &models, &mut scene_poses);
        }
        // Stations without a pose: solve each from a scene with a known pose.
        for (k, id) in ids.iter().enumerate() {
            if models[k].is_some() {
                continue;
            }
            let cand = self
                .scenes
                .iter()
                .enumerate()
                .filter(|(j, sc)| scene_poses[*j].is_some() && self.scene_usable(sc))
                .max_by_key(|(_, sc)| sensors_seen(sc, *id));
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
            return Err("the reference station could not be placed");
        }
        // Scenes without a pose: bootstrap against the known stations.
        let km: Vec<StationModel> = known.iter().map(|&k| models[k].clone().unwrap()).collect();
        self.pose_scenes(&ids, &models, &mut scene_poses);
        // Joint fit over the known stations and posed scenes (one tracker
        // model per scene's device; scenes from other devices skipped for now). The device
        // with the most posed scenes: the first scene's may be gone (a tracker that dropped
        // off the radio and came back as a new device), and its scenes with it.
        let mut devs: Vec<(DeviceId, usize)> = Vec::new();
        for (j, sc) in self.scenes.iter().enumerate() {
            if scene_poses[j].is_none() {
                continue;
            }
            match devs.iter_mut().find(|x| x.0 == sc.dev) {
                Some(x) => x.1 += 1,
                None => devs.push((sc.dev, 1)),
            }
        }
        let Some(dev0) = devs.iter().max_by_key(|x| x.1).map(|x| x.0) else { return Err("no still scene could be placed") };
        let Some(model) = self.devices.iter().find(|d| d.id == dev0).map(|d| d.model.clone()) else { return Err("no still scene could be placed") };
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
            return Err("no still scene could be placed");
        }
        let ref_in_known = known.iter().position(|&k| k == ref_idx).unwrap();
        let ups: Vec<Option<Vec3>> = known.iter().map(|&k| up_of(self, ids[k])).collect();
        let priors: Vec<Option<StationPrior>> =
            known.iter().map(|&k| self.stations.iter().find(|s| s.id == ids[k]).and_then(|s| s.prior)).collect();
        let Some(cal) = calibrate(&model, &km, &ups, &priors, &scenes, ref_in_known, &cfg.solve, cfg.calib_reject) else {
            return Err("the station fit failed");
        };
        // Quality gate: residuals a few σ at most.
        if cal.rms.iter().any(|r| !(*r < 3e-3)) {
            return Err("the station fit's residuals are too large");
        }
        let mut reports = Vec::new();
        for (ki, &k) in known.iter().enumerate() {
            let id = ids[k];
            let si = self.station_index(id);
            let st = &mut self.stations[si];
            st.model = Some(cal.stations[ki].clone());
            st.cov = Some(cal.station_cov[ki].clone());
            st.model_up = st.info.as_ref().map(info_up);
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
            scenes: scenes.len(),
            dropped: cal.dropped,
            tracker_up_error: cal.tracker_up_error,
            pending,
        });
        Ok(())
    }
}

/// A scene's observations: per (station, axis, sensor), its clusters of hit angles with at
/// least `min` hits and a third of the most-hit one, most-hit first, as their medians.
fn scene_obs(sb: &SceneBuilder, min: usize) -> Vec<(u32, u8, u8, Real, usize)> {
    let mut obs = Vec::new();
    for ((st, axis, sensor), cl) in &sb.clusters {
        let top = cl.iter().map(|c| c.values.len()).max().unwrap_or(0);
        let mut ranked: Vec<&Cluster> = cl.iter().filter(|c| c.values.len() >= min && c.values.len() * 3 >= top).collect();
        ranked.sort_by(|a, b| b.values.len().cmp(&a.values.len()));
        for c in ranked {
            obs.push((*st, *axis, *sensor, c.median(), c.values.len()));
        }
    }
    obs
}

fn new_scene(t: i64) -> SceneBuilder {
    SceneBuilder { start: t, clusters: Vec::new(), acc_sum: Vec3::ZERO, acc_n: 0 }
}

fn rot_between(a: Quat, b: Quat) -> Real {
    let d = a.conj().mul(b);
    2.0 * crate::math::atan2(Vec3::new(d.x, d.y, d.z).norm(), crate::math::fabs(d.w))
}

/// Publishes the device's pose at the newest IMU time (a propagated copy; the filter stays at
/// the light), with `light` the newest hit's time, and starts a new stats window.
fn publish_pose(d: &mut Device, light: i64, speculative: bool, cfg: &KernelConfig, out: &mut Vec<Event>) {
    let Some(track) = d.track.as_ref() else { return };
    let mut pubst = track.clone();
    if let Some(last) = d.last_imu {
        let mut q: VecDeque<(i64, Vec3, Vec3)> = d.imu.iter().copied().filter(|x| x.0 > pubst.time).collect();
        advance(&mut pubst, &mut q, last.0, d.last_imu, &cfg.track);
    }
    let (pose, cov) = head_pose(&pubst, &d.model.body_from_head);
    out.push(Event::Pose {
        dev: d.id,
        time: pubst.time,
        light,
        world_from_head: pose,
        world_from_body: pubst.pose(),
        velocity: pubst.vel,
        cov,
        used: d.window.used,
        gated: d.window.gated,
        speculative,
    });
    d.window = UpdateStats::default();
}

/// A station's OOTX gravity vector (station frame), normalised.
fn info_up(i: &StationInfo) -> Vec3 {
    Vec3::new(i.accel_dir[0] as Real, i.accel_dir[1] as Real, i.accel_dir[2] as Real).normalized()
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
        /// B sweeps its slot (named or not); off: C alone, as a lone station.
        b_on: bool,
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
                b_on: true,
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
            let up = pose.rot.conj().rotate(Vec3::new(0.0, 0.0, 1.0));
            let n = (secs * 250.0) as u32;
            for i in 0..n {
                self.t = self.t.wrapping_add(DT);
                let jitter = if moving && i % 2 == 0 { 1.03 } else if moving { 0.97 } else { 1.0 };
                self.k.push(Input::Imu { dev: self.dev, time: self.t, accel_g: up.scale(jitter), gyro: Vec3::ZERO }, &mut self.out);
                let di = self.k.devices.iter().position(|d| d.id == self.dev).unwrap();
                for (slot, st) in [(0u8, self.c.clone()), (1u8, self.b.clone())] {
                    if slot == 1 && !self.b_on {
                        continue;
                    }
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

    // ---------------------------------------------------------------- station lifecycle

    /// A kernel with C and B both loaded from a saved calibration; only C is ever named
    /// and seen (B was switched off or taken away).
    fn sim_with_stale_b(expire: Real) -> Sim {
        let mut sim = Sim::new(true);
        sim.k.cfg.station_expire = expire;
        let mut cov = Mat::zeros(6, 6);
        for i in 0..6 {
            cov[(i, i)] = 1e-6;
        }
        let b = sim.b.world_from_station;
        sim.k.push(Input::StationPrior { id: B, world_from_station: b, cov, cal: Some(cal()) }, &mut sim.out);
        sim
    }

    fn dropped(out: &[Event]) -> Vec<(u32, StationDrop)> {
        out.iter()
            .filter_map(|e| match e {
                Event::StationDropped { id, why } => Some((*id, *why)),
                _ => None,
            })
            .collect()
    }

    fn rot_angle(a: Quat, b: Quat) -> Real {
        let d = a.conj().mul(b);
        2.0 * crate::math::atan2(Vec3::new(d.x, d.y, d.z).norm(), crate::math::fabs(d.w))
    }

    fn model_of(k: &Kernel, id: u32) -> Option<StationModel> {
        k.stations.iter().find(|s| s.id == id).and_then(|s| s.model.clone())
    }

    #[test]
    fn a_calibrated_station_the_tracker_doesnt_see_adds_no_bias_states() {
        let mut sim = sim_with_stale_b(0.0);
        let p = table();
        sim.hold(&p, 2.0, false);
        let d = &sim.k.devices[0];
        assert!(d.track.is_some(), "not tracking");
        assert!(d.bias_slots.is_empty(), "angle-bias states with one station in view: {:?}", d.bias_slots);
    }

    #[test]
    fn a_station_not_seen_for_a_while_is_dropped() {
        let mut sim = sim_with_stale_b(5.0);
        let p = table();
        sim.hold(&p, 4.0, false);
        assert!(dropped(&sim.out).is_empty(), "dropped too early: {:?}", dropped(&sim.out));
        sim.hold(&p, 3.0, false);
        assert_eq!(dropped(&sim.out), alloc::vec![(B, StationDrop::NotSeen)]);
        assert!(model_of(&sim.k, B).is_none());
        assert!(model_of(&sim.k, C).is_some(), "C is in view and must stay");
        assert!(sim.k.devices[0].track.is_some(), "tracking on C goes on");
    }

    #[test]
    fn nothing_expires_while_no_light_reaches_any_tracker() {
        let mut sim = sim_with_stale_b(2.0);
        // IMU only (a tracker in a drawer): no sweeps at all.
        let up = table().rot.conj().rotate(Vec3::new(0.0, 0.0, 1.0));
        for _ in 0..(10 * 250) {
            sim.t = sim.t.wrapping_add(DT);
            sim.k.push(Input::Imu { dev: 0, time: sim.t, accel_g: up, gyro: Vec3::ZERO }, &mut sim.out);
        }
        assert!(dropped(&sim.out).is_empty(), "{:?}", dropped(&sim.out));
        assert!(model_of(&sim.k, B).is_some() && model_of(&sim.k, C).is_some());
    }

    #[test]
    fn a_tilted_station_is_levelled_again_from_its_gravity() {
        let mut sim = Sim::new(true);
        let p = table();
        sim.scene(&p);
        assert!(sim.k.devices[0].track.is_some());
        // C is knocked: 6 degrees about a horizontal axis, in place.
        let before = model_of(&sim.k, C).unwrap().world_from_station;
        let knock = Quat::from_rotvec(Vec3::new(0.7, -0.7, 0.0).normalized().scale((6.0 as Real).to_radians()));
        let mut tilted = sim.c.clone();
        tilted.world_from_station.rot = knock.mul(sim.c.world_from_station.rot).normalized();
        let di = 0;
        let inf = info(C, 2, &tilted);
        // One packet isn't a tilt yet (the accelerometer wanders); the same reading again is.
        sim.k.timing_event(di, TimingEvent::Station { slot: 0, info: inf.clone() }, &mut sim.out);
        assert_eq!(model_of(&sim.k, C).unwrap().world_from_station.rot, before.rot, "levelled on one packet");
        sim.k.timing_event(di, TimingEvent::Station { slot: 0, info: inf }, &mut sim.out);
        let after = model_of(&sim.k, C).unwrap().world_from_station;
        assert!(after.pos.sub(before.pos).norm() < 1e-9, "position kept");
        let err = rot_angle(after.rot, tilted.world_from_station.rot).to_degrees();
        assert!(err < 1.5, "levelled orientation {:.2} deg off the tilted station", err);
        let was = rot_angle(before.rot, tilted.world_from_station.rot).to_degrees();
        assert!(was > 5.0, "{:.2}", was);
        assert!(
            sim.out.iter().any(|e| matches!(e, Event::Calibrated { reason: CalibrationReason::StationTilted, .. })),
            "no StationTilted report"
        );
        // Tracking restarts on the levelled station.
        sim.c = tilted;
        let n = sim.out.len();
        sim.hold(&p, 2.0, false);
        assert!(sim.out[n..].iter().any(|e| matches!(e, Event::TrackingStarted { .. })), "tracking didn't restart");
        assert!(sim.k.devices[0].track.is_some());
    }

    #[test]
    fn gravity_jitter_of_a_count_is_not_a_tilt() {
        let mut sim = Sim::new(true);
        let p = table();
        sim.scene(&p);
        let before = model_of(&sim.k, C).unwrap().world_from_station;
        let mut inf = info(C, 2, &sim.c);
        inf.accel_dir[0] = inf.accel_dir[0].saturating_add(1);
        sim.k.timing_event(0, TimingEvent::Station { slot: 0, info: inf }, &mut sim.out);
        assert_eq!(model_of(&sim.k, C).unwrap().world_from_station.rot, before.rot);
        assert!(!sim.out.iter().any(|e| matches!(e, Event::Calibrated { reason: CalibrationReason::StationTilted, .. })));
    }

    /// The poses published for the tracker since output index `from`: (body position error
    /// against `truth`, m).
    fn pose_errors(out: &[Event], from: usize, truth: &Pose) -> Vec<Real> {
        out[from..]
            .iter()
            .filter_map(|e| match e {
                Event::Pose { world_from_body, .. } => Some(world_from_body.pos.sub(truth.pos).norm()),
                _ => None,
            })
            .collect()
    }

    fn relocated(out: &[Event]) -> Option<StationReport> {
        out.iter().rev().find_map(|e| match e {
            Event::Calibrated { reason: CalibrationReason::StationRelocated, stations, .. } => {
                stations.iter().find(|s| s.id == C).cloned()
            }
            _ => None,
        })
    }

    fn lost(out: &[Event], from: usize) -> Vec<TrackLoss> {
        out[from..]
            .iter()
            .filter_map(|e| match e {
                Event::TrackingLost { reason, .. } => Some(*reason),
                _ => None,
            })
            .collect()
    }

    /// Moves station C by `turn` (about world z, through its own position) and `shift`, in
    /// `steps` over `secs` while the tracker sits still at `p`.
    fn move_c(sim: &mut Sim, p: &Pose, turn: Real, shift: Vec3, secs: Real, steps: u32) {
        let start = sim.c.world_from_station;
        for i in 1..=steps {
            let f = i as Real / steps as Real;
            let rot = Quat::from_rotvec(Vec3::new(0.0, 0.0, turn * f)).mul(start.rot).normalized();
            sim.c.world_from_station = Pose { pos: start.pos.add(shift.scale(f)), rot };
            sim.hold(p, secs / steps as Real, false);
        }
    }

    #[test]
    fn a_station_moved_under_a_still_tracker_is_solved_again_not_the_tracker() {
        let mut sim = Sim::new(true);
        let p = table();
        sim.hold(&p, 3.0, false);
        assert!(sim.k.devices[0].track.is_some() && sim.k.devices[0].still_ref.is_some(), "settled on the table");
        let n = sim.out.len();
        move_c(&mut sim, &p, (5.0 as Real).to_radians(), Vec3::new(0.1, 0.0, 0.0), 1.0, 20);
        sim.hold(&p, 4.0, false);
        let rep = relocated(&sim.out[n..]).expect("the station wasn't solved again");
        let truth = sim.c.world_from_station;
        let dpos = rep.world_from_station.pos.sub(truth.pos).norm();
        let drot = rot_between(rep.world_from_station.rot, truth.rot).to_degrees();
        assert!(dpos < 0.02 && drot < 0.5, "relocated {:.1} mm / {:.2} deg off", dpos * 1e3, drot);
        assert!(lost(&sim.out, n).is_empty(), "lost: {:?}", lost(&sim.out, n));
        // The tracker stays put: no further off than the detection threshold, ever.
        let errs = pose_errors(&sim.out, n, &p);
        let worst = errs.iter().cloned().fold(0.0, Real::max);
        assert!(worst < sim.k.cfg.relocate_pos + 0.01, "the tracker jumped {:.1} mm", worst * 1e3);
        // And tracking goes on with the new station: still right after.
        let m = sim.out.len();
        sim.hold(&p, 1.0, false);
        let after = pose_errors(&sim.out, m, &p);
        assert!(!after.is_empty() && after.iter().all(|e| *e < 0.01), "after: {:?}", &after[..after.len().min(5)]);
    }

    #[test]
    fn a_still_station_under_a_still_tracker_is_never_relocated() {
        let mut sim = Sim::new(true);
        let p = table();
        sim.hold(&p, 20.0, false);
        assert!(relocated(&sim.out).is_none());
        assert!(sim.k.devices[0].relocating.is_none());
        assert!(lost(&sim.out, 0).is_empty());
    }

    #[test]
    fn a_tracker_that_moves_while_its_station_relocates_is_lost_not_misplaced() {
        let mut sim = Sim::new(true);
        let p = table();
        sim.hold(&p, 3.0, false);
        let n = sim.out.len();
        move_c(&mut sim, &p, (5.0 as Real).to_radians(), Vec3::ZERO, 0.6, 12);
        assert!(sim.k.devices[0].relocating.is_some(), "relocation didn't start");
        sim.hold(&p, 0.3, true); // picked up
        assert!(sim.k.devices[0].relocating.is_none());
        assert!(lost(&sim.out, n).contains(&TrackLoss::StationsMoved), "{:?}", lost(&sim.out, n));
        assert!(relocated(&sim.out[n..]).is_none());
    }

    #[test]
    fn a_lone_station_is_back_in_its_slot_right_after_a_relock() {
        let mut sim = Sim::new(true);
        sim.b_on = false; // C alone: nothing ever sweeps slot 1
        let p = table();
        sim.hold(&p, 2.0, false);
        let di = 0;
        sim.k.timing_event(di, TimingEvent::LockLost { reason: LockLoss::Silence }, &mut sim.out);
        assert_eq!(sim.k.devices[di].slot_station, [None, None]);
        sim.k.timing_event(di, TimingEvent::Locked { phase: 0, inliers: 8 }, &mut sim.out);
        // A few sweeps, far short of an OOTX packet (seconds).
        sim.hold(&p, 0.05, false);
        assert_eq!(sim.k.devices[di].slot_station[0], Some(C), "waits for OOTX");
    }

    #[test]
    fn forget_station_drops_it_and_its_observations() {
        let mut sim = Sim::new(true);
        let p = table();
        sim.name_b();
        sim.scene(&p);
        assert!(model_of(&sim.k, B).is_some(), "B placed");
        sim.k.push(Input::ForgetStation { id: B }, &mut sim.out);
        assert_eq!(dropped(&sim.out), alloc::vec![(B, StationDrop::Forgotten)]);
        assert!(model_of(&sim.k, B).is_none());
        assert!(sim.k.scenes.iter().all(|sc| sc.obs.iter().all(|o| o.0 != B)));
        assert!(model_of(&sim.k, C).is_some());
    }
}

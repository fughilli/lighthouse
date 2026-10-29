//! IMU scaling and factory calibration (libsurvive's
//! survive_default_raw_imu_process):
//!   acc_g    = (raw * acc_range/32768  - acc_bias/1000) * acc_scale
//!   gyro_rps = (raw * gyro_range/32768 * pi/180 - gyro_bias) * gyro_scale

use crate::RawImu;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ImuCalibration {
    /// g per count.
    pub acc_per_count: f64,
    /// rad/s per count.
    pub gyro_per_count: f64,
    /// g (the config gives milli-g).
    pub acc_bias: [f64; 3],
    pub acc_scale: [f64; 3],
    /// rad/s.
    pub gyro_bias: [f64; 3],
    pub gyro_scale: [f64; 3],
}

impl Default for ImuCalibration {
    /// Watchman defaults: ±8 g, ±2000 °/s, no bias.
    fn default() -> ImuCalibration {
        ImuCalibration {
            acc_per_count: 8.0 / 32768.0,
            gyro_per_count: 2000.0 / 32768.0 * core::f64::consts::PI / 180.0,
            acc_bias: [0.0; 3],
            acc_scale: [1.0; 3],
            gyro_bias: [0.0; 3],
            gyro_scale: [1.0; 3],
        }
    }
}

impl ImuCalibration {
    /// Applies the IMU_SCALES report (`01 <gyro mode> <acc mode>`).
    pub fn set_scale_modes(&mut self, gyro_mode: u8, acc_mode: u8) {
        let gyro_range = [250.0, 500.0, 1000.0, 2000.0][(gyro_mode & 3) as usize];
        let acc_range = [2.0, 4.0, 8.0, 16.0][(acc_mode & 3) as usize];
        self.acc_per_count = acc_range / 32768.0;
        self.gyro_per_count = gyro_range / 32768.0 * core::f64::consts::PI / 180.0;
    }

    /// (accel in g, gyro in rad/s), IMU frame.
    pub fn apply(&self, s: &RawImu) -> ([f64; 3], [f64; 3]) {
        let mut a = [0.0; 3];
        let mut g = [0.0; 3];
        for i in 0..3 {
            a[i] = (s.accel[i] as f64 * self.acc_per_count - self.acc_bias[i]) * self.acc_scale[i];
            g[i] = (s.gyro[i] as f64 * self.gyro_per_count - self.gyro_bias[i]) * self.gyro_scale[i];
        }
        (a, g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_libsurvive_example() {
        // docs/lighthouse-protocol.md §4.3: raw z 4077 -> 0.992431 g.
        let mut c = ImuCalibration::default();
        c.set_scale_modes(3, 2);
        c.acc_bias = [0.0395e-3, 0.1467e-3, -0.1558e-3];
        c.acc_scale = [0.9981, 0.9997, 0.9969];
        let (a, _) = c.apply(&RawImu { timecode: 0, accel: [0, 0, 4077], gyro: [0; 3] });
        assert!((a[2] - 0.992431).abs() < 1e-6, "{}", a[2]);
    }
}

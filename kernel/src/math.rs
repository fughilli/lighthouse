//! Small f64 geometry: Vec3, unit quaternions, rigid poses. libm for the
//! transcendental functions so the kernel stays `no_std`.

pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

pub fn atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}

pub fn asin(x: f64) -> f64 {
    libm::asin(x)
}

pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 { x: 0.0, y: 0.0, z: 0.0 };

    pub const fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 { x, y, z }
    }

    pub fn from_array(a: [f64; 3]) -> Vec3 {
        Vec3::new(a[0], a[1], a[2])
    }

    pub fn to_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    pub fn add(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }

    pub fn sub(self, o: Vec3) -> Vec3 {
        Vec3::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }

    pub fn scale(self, s: f64) -> Vec3 {
        Vec3::new(self.x * s, self.y * s, self.z * s)
    }

    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Vec3) -> Vec3 {
        Vec3::new(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }

    pub fn norm(self) -> f64 {
        sqrt(self.dot(self))
    }

    pub fn normalized(self) -> Vec3 {
        let n = self.norm();
        if n > 0.0 {
            self.scale(1.0 / n)
        } else {
            self
        }
    }
}

/// Unit quaternion, scalar first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Default for Quat {
    fn default() -> Quat {
        Quat::IDENTITY
    }
}

impl Quat {
    pub const IDENTITY: Quat = Quat { w: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Quat {
        Quat { w, x, y, z }
    }

    pub fn mul(self, o: Quat) -> Quat {
        Quat::new(
            self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
            self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
            self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
        )
    }

    pub fn conj(self) -> Quat {
        Quat::new(self.w, -self.x, -self.y, -self.z)
    }

    pub fn normalized(self) -> Quat {
        let n = sqrt(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z);
        Quat::new(self.w / n, self.x / n, self.y / n, self.z / n)
    }

    /// Rotates v by this (unit) quaternion.
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let u = Vec3::new(self.x, self.y, self.z);
        let t = u.cross(v).scale(2.0);
        v.add(t.scale(self.w)).add(u.cross(t))
    }

    /// exp of a rotation vector (axis * angle).
    pub fn from_rotvec(r: Vec3) -> Quat {
        let a = r.norm();
        if a < 1e-12 {
            return Quat::new(1.0, r.x * 0.5, r.y * 0.5, r.z * 0.5).normalized();
        }
        let s = sin(a * 0.5) / a;
        Quat::new(cos(a * 0.5), r.x * s, r.y * s, r.z * s)
    }
}

/// Rigid transform taking points from a child frame into a parent frame:
/// `p_parent = rot * p_child + pos`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pose {
    pub pos: Vec3,
    pub rot: Quat,
}

impl Pose {
    pub const IDENTITY: Pose = Pose { pos: Vec3::ZERO, rot: Quat::IDENTITY };

    pub fn apply(&self, p: Vec3) -> Vec3 {
        self.rot.rotate(p).add(self.pos)
    }

    pub fn inverse(&self) -> Pose {
        let r = self.rot.conj();
        Pose { pos: r.rotate(self.pos).scale(-1.0), rot: r }
    }

    /// self ∘ o: first o, then self.
    pub fn compose(&self, o: &Pose) -> Pose {
        Pose { pos: self.apply(o.pos), rot: self.rot.mul(o.rot).normalized() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_and_inverse_round_trip() {
        let p = Pose { pos: Vec3::new(1.0, -2.0, 0.5), rot: Quat::from_rotvec(Vec3::new(0.3, -0.7, 1.1)) };
        let v = Vec3::new(0.2, 0.4, -0.9);
        let back = p.inverse().apply(p.apply(v));
        assert!(back.sub(v).norm() < 1e-12);
        let q = Quat::from_rotvec(Vec3::new(0.0, 0.0, core::f64::consts::FRAC_PI_2));
        let r = q.rotate(Vec3::new(1.0, 0.0, 0.0));
        assert!(r.sub(Vec3::new(0.0, 1.0, 0.0)).norm() < 1e-12);
    }
}

//! Known USB devices (libsurvive's KnownDeviceTypes).

pub const VALVE_VID: u16 = 0x28de;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    /// Watchman RF dongle (tracker or controller behind it).
    Dongle,
    /// Tracker (2018) / Tracker 3.0 on USB.
    TrackerWired,
    /// Tracker (2017) on USB.
    Tracker2017Wired,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interface {
    /// Dongle: IMU, light and buttons in one stream of 0x23/0x24 reports.
    Watchman,
    Imu,
    Lightcap,
    Buttons,
}

pub struct DeviceType {
    pub pid: u16,
    pub kind: DeviceKind,
    pub name: &'static str,
    /// HID interface number → role.
    pub interfaces: &'static [(i32, Interface)],
}

pub const DEVICES: &[DeviceType] = &[
    DeviceType { pid: 0x2101, kind: DeviceKind::Dongle, name: "Watchman dongle", interfaces: &[(0, Interface::Watchman)] },
    DeviceType { pid: 0x2102, kind: DeviceKind::Dongle, name: "Knuckles dongle", interfaces: &[(0, Interface::Watchman)] },
    DeviceType {
        pid: 0x2300,
        kind: DeviceKind::TrackerWired,
        name: "Tracker (2018 / 3.0)",
        interfaces: &[(0, Interface::Imu), (1, Interface::Lightcap), (2, Interface::Buttons)],
    },
    DeviceType {
        pid: 0x2022,
        kind: DeviceKind::Tracker2017Wired,
        name: "Tracker (2017)",
        interfaces: &[(0, Interface::Imu), (1, Interface::Lightcap), (2, Interface::Buttons)],
    },
];

pub fn lookup(vid: u16, pid: u16) -> Option<&'static DeviceType> {
    if vid != VALVE_VID {
        return None;
    }
    DEVICES.iter().find(|d| d.pid == pid)
}

impl DeviceType {
    pub fn interface(&self, number: i32) -> Option<Interface> {
        // On Windows a single-interface dongle reports -1.
        let n = if number < 0 { 0 } else { number };
        self.interfaces.iter().find(|i| i.0 == n).map(|i| i.1)
    }

    pub fn wireless(&self) -> bool {
        self.kind == DeviceKind::Dongle
    }
}

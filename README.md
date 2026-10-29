# lighthouse

Lighthouse 1.0 tracking for Vive Trackers, built from the raw HID bytes: base
station timing and OOTX, station calibration from still scenes, and an IMU +
light EKF that tracks each tracker in a gravity-aligned world frame.

The on-device crates are `no_std` + `alloc` and also run on a microcontroller:
the ESP32-P4 [Splanc](https://github.com/fughilli/splanc) player runs them
with the tracker wired to its USB OTG port.

| Crate | | |
| --- | --- | --- |
| `kernel` (`lh_kernel`) | no_std | Light pulses and IMU in; station and tracker poses (with covariances) and events out. Sweep timing, OOTX, still-scene calibration, the tracking EKF with per-station angle-bias states, and a check that drops a stale saved calibration when the stations disagree. |
| `protocol` (`lh_protocol`) | no_std | Report bytes in, samples out: device config readout and parsing, RF (Watchman dongle) and wired USB report decoding, command bytes. |
| `device` (`lh_device`) | no_std | One device over an abstract HID `Link`: the handshake that reads its config and starts its report stream, and a decoder from reports to kernel inputs. |
| `transport` (`lh_transport`) | host | hidapi (live dongles and wired trackers) and `.rec.gz` replay. |
| `daemon` (`lhd`) | host | The tracking daemon: JSON-lines events on stdout, commands on stdin. |
| `tools` | host | `lh_replay` (run the kernel over a recording), `lh_protocheck`, `lh_angles`. |

`docs/lighthouse-protocol.md` documents the byte formats (after libsurvive, which
it cites section by section).

## Build

```sh
bazel test //...
bazel build -c opt //daemon:lhd
# the no_std crates for a bare-metal RISC-V target (ESP32-P4):
bazel build --platforms=//platforms:riscv32imafc //kernel:lh_kernel //protocol:lh_protocol //device:lh_device
```

`lhd` links the system hidapi (Homebrew on macOS, libudev on Linux).

## Using it from another Bazel module

```starlark
bazel_dep(name = "lighthouse", version = "0.0.0")
archive_override(
    module_name = "lighthouse",
    urls = ["https://github.com/fughilli/lighthouse/archive/<commit>.tar.gz"],
    strip_prefix = "lighthouse-<commit>",
    integrity = "sha256-…",
)
```

then depend on `@lighthouse//kernel:lh_kernel`, `@lighthouse//protocol:lh_protocol`,
`@lighthouse//device:lh_device`. A consumer registers its own Rust toolchains.
`rules_rust` 0.71 has no `riscv32imafc` entry in its triple table; register that
toolchain with a `rust.repository_set` and explicit `target_compatible_with`, as
this repo's `MODULE.bazel` does.

## License

AGPL-3.0 (see `LICENSE`).

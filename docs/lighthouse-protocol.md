# Vive / Watchman HID protocol (as libsurvive implements it)

A byte-level spec of how libsurvive talks to Vive Trackers and the Watchman
RF dongle, for porting the device layer to Rust. It's extracted from the C source
in `~/Projects/git_checkouts/libsurvive` (local patches: raw HID tap and
disambiguator changes; the decoding paths quoted here are upstream code) and
checked against real HID bytes captured on this bench.

**Executable reference:** `swarm/tools/vive_proto_ref.py` in fughilli/indoor_localization implements sections 4–8
and checks itself against a recording. It prints a summary of the match:

```
python3 swarm/tools/vive_proto_ref.py captures/vive/20261027-131723_turntable-r.rec.gz
python3 swarm/tools/vive_proto_ref.py --selftest      # synthetic wired-layout checks
```

**Validation result** (section 10): over 330,754 wireless HID reports, the
decoder reproduces every lightcap element (981,653/981,653) and every raw IMU
sample (166,870/166,870) bit for bit, in libsurvive's order. It also reproduces
every calibrated IMU sample to the printed precision. Config reassembly and OOTX
decoding match as well. The wired formats (0x20, 0x25) are specified from source
only, because no wired capture survived.

Conventions: everything is little-endian unless stated otherwise. A "tick" is
one count of the device's 48 MHz clock, and timecodes are `uint32` ticks that
wrap. `driver_vive.c:N` means line N of `libsurvive/src/driver_vive.c`. A report
is shown with its report ID as byte 0, which is what `hid_read` and
`hid_get_feature_report` return on HIDAPI.

---

## 1. Devices

### 1.1 Device table (`KnownDeviceTypes`, driver_vive.c:268–343)

| VID:PID | Name | Codename | Type | Interfaces (endpoint → libsurvive name) | Startup magics |
|---|---|---|---|---|---|
| 0bb4:2c87 | Vive HMD mainboard | – | `USB_DEV_HMD` | 0x81 Mainboard | power_on |
| 0bb4:030e | Vive Pro mainboard | – | `USB_DEV_HMD` | 0x81 Mainboard Pro | power_on |
| 28de:2000 | HMD IMU & LH | `HMD` | `USB_DEV_HMD_IMU_LH` | 0x81 IMU, 0x82 Lightcap | enable_lighthouse, …_more |
| **28de:2101** | **Watchman (RF dongle)** | `WM0` | `USB_DEV_WATCHMAN1` (RF) | 0x81 `IMU/Lightcap/Buttons` | super_magic, rf_raw_mode_0, protocol_switch |
| 28de:2102 | Knuckles dongle | `KN0` | `USB_DEV_WATCHMAN1` (RF) | 0x81 `IMU/Lightcap/Buttons` | same as 2101 |
| 28de:2022 | Tracker (2017) | `TR0` | `USB_DEV_TRACKER0` | 0x81 IMU, 0x82 Lightcap, 0x83 Buttons | enable_lighthouse, …_more |
| **28de:2300** | **Tracker (2018) / Tracker 3.0, wired** | `T20` | `USB_DEV_TRACKER1` | 0x81 IMU, 0x83 Lightcap, 0x84 Buttons | enable_lighthouse, …_more |
| 28de:2012 | Wired Watchman (controller on USB) | `WW0` | `USB_DEV_W_WATCHMAN1` | 0x81 IMU, 0x82 Lightcap, 0x83 Buttons | enable_lighthouse, …_more |

- **RF vs USB.** `survive_device_is_rf()` (driver_vive.c:475) treats only the
  `WATCHMAN1/2` types as RF. Everything else is a USB device.
- **Codenames.** The third character is bumped for each further device of the
  same type: `WM0`, `WM1`, and so on (driver_vive.c:806–810). Under HIDAPI the
  `SurviveObject` is created at enumeration, before anything is known about
  the tracker behind a dongle.
- **The tracker's own identity comes from its config JSON** (section 3), not
  from USB. Our tracker reports `device_pid 8960` (0x2300),
  `model_number "VIVE Tracker 3.0 MV"` and `device_serial_number "LHR-C6198565"`.
  - `model_number` selects the object subtype and a `sensor_scale`
    (survive_default_devices.c:291–313): Tracker 3.0 → `TRACKER_GEN2`, 1.0034;
    2018 tracker `"Vive Tracker MV"` → `TRACKER`, 1.00585.
  - libsurvive multiplies `modelPoints` by this scale.
  - The subtype matters for decoding (section 5.3).
- **Blacklist.** `blacklist-devs` can exclude a device by name.

### 1.2 HIDAPI (macOS) transport (`driver_vive.hidapi.h`)

- **Enumeration.**
  - `hid_enumerate(0,0)`. One device is added per entry with
    `interface_number <= 0` (`get_next_device`).
  - All HID interfaces with the same serial are then opened (`hid_open_path`)
    into `handle->interfaces[interface_number]`.
  - The endpoint table above is indexed by position: table entry *k* is HID
    interface *k*. For T20 that's IMU = 0, Lightcap = 1, Buttons = 2. The dongle
    has one interface.
- **Feature reports always go through `interfaces[0]`.**
  - GET: `hid_get_feature_report(dev, buf, 256)` with `buf[0] = id`.
  - SET: `hid_send_feature_report(dev, buf, len)`.
- **Input.**
  - Input runs non-blocking (`hid_set_nonblocking`). Each poll drains up to 256
    reports per interface. That's a local patch: one read per poll falls behind
    and the OS queue drops reports.
  - `hid_read` returning < 0 sets `request_close`. The device is then destroyed
    and, under HIDAPI, never re-opened: there's no hotplug (driver_vive.c:917–927).
- **libusb difference** (`driver_vive.libusb.h`).
  - GET is a class control-IN transfer: bmRequestType 0xA1, bRequest 0x01
    GET_REPORT, wValue 0x0300|id, wIndex 0, wLength 256.
  - SET is 0x21 / 0x09 SET_REPORT / 0x0300|id / interface.
  - Interrupt transfers on the endpoints.
  - On Linux the RF magics are sent at their natural length. Elsewhere they're
    padded to 64 bytes ("Windows needs 64 size for the wireless dongle"), and our
    macOS captures show 64-byte SETs.
- **Raw tap** (local patch, `--record-rawhid 1`). It writes one recording line
  per HID transfer:
  ```
  <t> <dev> HIDIN <iface-name> <host_us> <len> <hex>       every input report, as hid_read returned it
  <t> <dev> HIDFEAT GET|SET <host_us> <rtn> <hex>          rtn = hidapi return (-1 = failed)
  ```

---

## 2. Startup / handshake

### 2.1 State machine (driver_vive.config.h)

Each device gets a `survive_config_packet` that runs the states in this order:

`CONFIG → MAGICS → VERSION → IMU_SCALES → DONE`

Devices without a codename (the HMD mainboard) start at MAGICS and stop after it.

| State | Transfer | Success → | Notes |
|---|---|---|---|
| CONFIG | GET 0x10, then GET 0x11 repeatedly | MAGICS | Section 3.1. Gates everything: for a dongle it succeeds only once the tracker answers over RF. |
| MAGICS | SET each magic of the device, in table order | VERSION | A SET counts as acked when `rtn >= magic length`, otherwise it's resent. At the end `lightcapMode = raw0`. |
| VERSION | GET 0x05 | IMU_SCALES | Parses version (3.3), then loads the config JSON (`config` hook). |
| IMU_SCALES | GET 0x01 | DONE | `[1]` = gyro mode, `[2]` = acc mode (section 4.3). |
| DONE / cleanup | – | – | Attaches (starts reading) the input interfaces, then sets `ignoreCnt = 10` and `nextCfgSubmitTime = -1`. Until cleanup, input reports are dropped (`cfg_user != 0`, driver_vive.c:2916). |

- **Retries.** A failed transfer is retried 20 ms later (rtn -1 on HIDAPI counts
  as a timeout). After 5 consecutive stalls the retry adds +1 s
  (driver_vive.config.h:106–121).
- **Observed on macOS with the dongle:** every config/version GET first returns
  -1 and succeeds on the retry 21 ms later. GET 0x01 is the exception. This looks
  like the dongle NAKing until it has fetched the answer from the tracker over
  RF; that's an interpretation, not verified. It's harmless, but it makes
  startup about 0.8 s.

### 2.2 Magic feature reports (driver_vive.c:209–264)

| Name | Bytes (SET feature) | Sent to |
|---|---|---|
| `vive_magic_enable_lighthouse` | `04 00 00 00 00` (5 B) — "change mode" = raw0 | HMD IMU, TR0, **T20**, WW0 |
| `vive_magic_enable_lighthouse_more` | `07 03 00 00 00` (5 B) — "lightcap verbosity" 3 | same |
| `vive_magic_protocol_super_magic` | `ff 96 10 be 5b 32 54 11 cf 83 75 53 8a 08 6a 53 58 d0 b1` (pad to 64) — CONFIGURE_RADIO | **dongle** 2101/2102 |
| `vive_magic_rf_raw_mode_0` | `ff 87 06 01 00 00 02 00 00` (pad to 64) — CHANGE_PROTOCOL, raw0 | dongle |
| `vive_magic_protocol_switch` | `ff 87 03 00 01 00` (pad to 64) — CHANGE_PROTOCOL | dongle |
| `vive_magic_power_on` | `04 78 29 38 01 00 00 00 00 00 02 00 01` (pad to 64) | HMD mainboard |
| `vive_magic_power_off` | `04 78 29 38 00 …` (64 B, driver_vive.c:212) — `code=false`, never sent | HMD mainboard |

Command reports have the form `ff <cmd> <len> <len data bytes>`; section 6.4
lists the commands.

**Observed dongle startup** (every capture, after the config reads; t is
seconds since survive-cli start):

```
0.790 SET ff9610be5b325411cf8375538a086a5358d0b1 00…(64)
0.791 SET ff870601000002000000 00…(64)
0.792 SET ff87030001 00…(64)
0.793 GET 05 -> -1 ; 0.814 GET 05 -> 64 B  (version)
0.815 GET 01 -> 64 B 01 03 02 …           (IMU scales: gyro 3, acc 2)
```

0x23/0x24 input reports start right after the protocol switch.

### 2.3 Lightcap modes (`vive_switch_mode`, driver_vive.c:581–632)

| Mode | Meaning | USB device | RF device |
|---|---|---|---|
| raw0 | LH1 pulse data (default after the magics) | SET `04 00` (64 B) then `07 03 00 00 00` | SET `ff 87 06 01 00 00 02 00 00` (64 B) |
| raw1 | LH2 | SET `04 01` (64 B) then `07 03 00 00 00` | SET `ff 87 06 01 01 00 02 00 00` |
| raw2 | unknown use | SET `04 03` then `07 03 …` | SET `ff 87 06 01 01 00 02 01 00` |

- **When libsurvive switches.** It sends a switch only when it decides the
  system is LH2: it sees report 0x27 on USB, or `lh_version == 1` on RF. LH1 stays
  in raw0, the mode the magics leave the device in.
- **RF data byte meanings.** `buf[4]` is 0 for raw0 and 1 otherwise; `buf[7]` is
  1 only for raw2.
- **Sniffed SteamVR equivalents** (comments at driver_vive.c:111–160).
  - USB sync modes: `04 00|01|03 00 00 00` followed by `07 03 00 00 00`.
  - RF: `01 00 00 02 00 00` style payloads.
- **Guard after a switch.** `ignoreCnt = 10` (the next 10 RF sub-packets are
  dropped), and on RF `timeWithoutFlag` discards light (section 5.3, raw1 only).
- **Wired tracker on macOS (our finding).** The first `04 …` mode report sent to
  an idle wired tracker makes it reset and re-enumerate.
  - HIDAPI then sees `hid_read < 0` and closes the device for good.
  - Only the second run (after the reconnect) streams.
  - A port should expect the re-enumeration: re-open, then re-send config and
    magics.

### 2.4 LH generation detection (survive_disambiguator.c:37–88)

- **libsurvive starts with `lh_version = -1`.** Until the generation is known,
  `handle_lightcap` swallows every element: no `C` line, no processing.
  - A pulse counts as OOTX-length when 3000 ≤ length < 6500.
  - More than 10 such pulses on the same sensor, spaced 400 000 or 800 000 ticks
    (±5 %), make it gen 1.
  - 500 elements without that make it gen 2.
- **Forcing doesn't skip detection.** `--lighthouse-gen 1` only blocks the
  gen-2 decision; detection still runs.
- **On detection** (`survive_default_gen_detected_process`), every object's
  `SurviveSensorActivations` is reset, which sets `last_imu = 0`. That matters for
  RF timestamps (section 5.6).

---

## 3. Device config, version, IMU scales

### 3.1 Config readout (driver_vive.config.h:168–213)

1. GET `0x10` (CONFIG_READMODE). The reply is
   `10 LL LL …`, where `expected = buf[1] << 8 | buf[2]` is **big-endian**.
2. Repeat GET `0x11` (CONFIG_READ). Each reply is `11 n d[0..n-1] …` with
   `n ≤ 62`; append `d`.
3. `n == 0` ends the read. If the collected length is less than `expected`, the
   whole read restarts.
4. The data is a **zlib stream** (`78 9c …`). libsurvive skips the 2 header
   bytes and runs `puff` (raw inflate) on the rest (survive.c:1150). Python's
   `zlib.decompress(data)` does the same.
5. The result is JSON (newlines are replaced by spaces in the `CONFIG` recording
   line).

Observed: `expected` = 2005, but the chunks deliver 2007 bytes (32×62 + 23). The
announced length seems to exclude the 2-byte zlib header; libsurvive notes
"sometimes 2 bytes longer". The JSON is 8065 bytes and is identical to the
`CONFIG` line.

JSON fields libsurvive uses (`process_jsontok` / `process_jsonarray`, survive_default_devices.c:206–390):

| Field | Use |
|---|---|
| `device_class` | `controller` / `hmd` / `generic_tracker` → object type |
| `model_number` | subtype and `sensor_scale` (§1.1) |
| `device_serial_number` | serial |
| `lighthouse_config.modelPoints[n][3]` | sensor positions, **trackref frame**, metres, ×`sensor_scale` |
| `lighthouse_config.modelNormals[n][3]` | sensor normals, trackref frame |
| `lighthouse_config.channelMap[n]` | `channel_map[value] = index`: maps the raw sensor id in light reports to a model index. Ids ≥ `sensor_ct` are dropped *after* recording. Identity (0..19) on our tracker. |
| `imu.acc_bias[3]` | milli-g. libsurvive divides by 1000 for every non-HMD device (survive_default_devices.c:461,476). |
| `imu.acc_scale[3]`, `imu.gyro_bias[3]` (rad/s), `imu.gyro_scale[3]` | calibration (§4.3) |
| `imu.plus_x`, `imu.plus_z`, `imu.position` | IMU pose in trackref, see `solve_vive_pose` |
| `head.plus_x/plus_z/position` | head pose in trackref |
| `trackref_from_imu[7]`, `trackref_from_head[7]` | alternative pose form: quaternion x,y,z,w then position |

- **`solve_vive_pose`** (survive_default_devices.c:139) builds the rotation
  matrix with columns `[plus_x, plus_z × plus_x, plus_z]` and the translation
  `position`. That gives `imu2trackref`; `head2trackref` is built the same way.
- **libsurvive's object frame is the IMU frame.** It transforms the sensor points
  and normals by `inverse(imu2trackref)`, and sets
  `head2imu = trackref2imu ∘ head2trackref`.
- **Our tracker:** `imu.plus_x ≈ (0,1,0)`, `plus_z = (0,0,1)`,
  `position = (0.01376, -0.00314, 0.00625)`. `firmware_config` (radio, haptics,
  charging) is not used by libsurvive.

### 3.2 Version, GET 0x05 (`parse_tracker_version_info`, driver_vive.c:2802)

| Offset | Type | Field | Observed (our dongle+tracker) |
|---|---|---|---|
| 0 | u8 | report id 0x05 | |
| 1 | u32 | `revision` (firmware build, unix time) | 1592875850 (2020-06-23) |
| 5 | u32 | `board_model` | 3 |
| 9 | char[32] | `fw_name` (two 16-B strings in practice) | `"watchman"`, `"valvebuilder01"` |
| 41 | u32 | `hardware_id` | 0x84020109 |
| 45 | u32 | `a` | 0 |
| 49 | u16, u8, u8 | FPGA major/minor/patch | 538 / 7 / 2 (bytes `1a 02 07 02`) |
| 53 | u8, u64 | `h`, `k` | stale buffer bytes |

- libsurvive warns when `revision` falls outside
  [1462663157, 1632556731].
- Report `0x13` (VERSION_ALT, from Knuckles) parses the same way.

### 3.3 IMU scales, GET 0x01

`01 <gyro_mode> <acc_mode>`. The rest of the 64-byte buffer is left over from
the previous GET; our capture shows the 0x05 bytes there.

`survive_default_set_imu_scale_modes` (survive_process.c:359):

| mode | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| gyro range (°/s) | 250 | 500 | 1000 | 2000 |
| acc range (g) | 2 | 4 | 8 | 16 |

- `raw_acc_scale = acc_range / 32768` [g/count].
- `raw_gyro_scale = gyro_range / 32768 · π/180` [rad/s per count].
- **Observed `01 03 02`:** ±2000 °/s and ±8 g. That gives 1/4096 g/count and
  1.06526e-3 rad/s/count, the same as the `WM*` defaults libsurvive uses without
  the report (2/8192 g, 2000 °/s; survive_default_devices.c:455–466).
- The IMU_SCALES report overrides the defaults.
- The **Buttons interface** of wired devices also emits *input* report 0x01, which
  is a different thing (section 6.1).

---

## 4. IMU

### 4.1 USB report 0x20 (IMU interface; driver_vive.c:2948–2987)

| Offset | Type | Field |
|---|---|---|
| 0 | u8 | 0x20 |
| 1 + 17k (k = 0,1,2) | i16[3] | accel x,y,z (counts) |
| 7 + 17k | i16[3] | gyro x,y,z (counts) |
| 13 + 17k | u32 | timecode (48 MHz ticks) |
| 17 + 17k | u8 | sequence number |

Total 52 bytes. This matches the description you gave.

- **The three slots are a ring.** Each report re-sends recent samples.
- **libsurvive's rule** keeps a per-object `oldcode` (initially 0). For
  k = 0, 1, 2 in order, it accepts a slot iff `(int8)(seq − oldcode) > 0` and then
  sets `oldcode = seq`. It emits `raw_imu(mask=3, agm, timecode, id=seq)`.
- **Unverified hazard:** if the device fills the ring newest-first, this rule
  drops samples. A port should sort the three slots by `(int8)(seq − oldcode)`
  and emit all new ones in order.
- No wired capture is left to check against; `--selftest` checks the layout
  only.

### 4.2 RF IMU (inside a 0x23/0x24 sub-packet; `read_imu_data`, driver_vive.c:1816)

13 bytes, present when bit 0x80 of the sub-packet flags is set (section 5.3):

| Offset | Type | Field |
|---|---|---|
| 0 | u8 | `time_lsb`: bits 15..8 of the sample time |
| 1 | i16[3] | accel x,y,z |
| 7 | i16[3] | gyro x,y,z |

- **Timecode:** `tc = (time16 << 16) | (time_lsb << 8)`, where `time16` is the
  sub-packet's 16-bit header time. Resolution is 256 ticks (5.3 µs); the low
  byte is always 0.
- **Emitted** as `raw_imu(mask=3, agm, tc, id=0)`.
- **Quirk:** libsurvive checks that only 7 bytes remain, then reads 13.
- **Rate:** 250 Hz, every other sub-packet. Sub-packets arrive at about 500 Hz,
  as 0x23 or paired in 0x24.

### 4.3 Scaling and calibration (`survive_default_raw_imu_process`, survive_process.c:343)

```
acc_g    = (raw_acc  * raw_acc_scale  - acc_bias/1000) * acc_scale     // per axis
gyro_rps = (raw_gyro * raw_gyro_scale - gyro_bias)     * gyro_scale
```

- **Validation:** the recording's `I` lines are reproduced to 1.5e-6 for all
  166,870 samples. Example: raw z 4077 → 4077/4096 = 0.995361, minus
  (−0.1558/1000), times 0.9969 = **0.992431**.
- **Frame:** axes are the IMU frame, which is libsurvive's object frame. Accel is
  in g, and the reading at rest is ≈ +1 g along "up".
- **Option:** `--ignore-config-imu-bias` zeros the biases.
- **Time reference:** after each IMU sample, `activations.last_imu` = the
  sample's timecode (64-bit extended; light decoding uses the low 32 bits). That
  reference matters for RF light (section 5.6).

---

## 5. Lightcap

A **lightcap element** is `{u8 sensor_id, u32 timestamp (rising edge), u16 length}`,
in ticks. libsurvive's `handle_lightcap` handles each one:

- It drops `length == 0`.
- During generation detection it swallows the element (§2.4).
- Otherwise it records a `C <sensor> <timestamp> <length>` line with the **raw,
  unmapped** sensor id.
- It then maps the id through `channel_map`, drops unknown ids, and calls the
  lightcap hook (the disambiguator).

### 5.1 USB report 0x25: LH1, T20 / HMD lightcap interface (driver_vive.c:3007–3031)

| Offset | Type | Field |
|---|---|---|
| 0 | u8 | 0x25 |
| 1 + 7k (k = 0..8) | u8 | sensor id; `> 0xfd` (0xfe/0xff) = empty slot, skipped |
| 2 + 7k | u16 | length (ticks) |
| 4 + 7k | u32 | timestamp (ticks, full 32-bit) |

- Total 64 bytes, up to 9 elements in slot order.
- It's ignored when `lh_version == 1` (LH2).
- This matches your description; `--selftest` checks it.

### 5.2 Other USB lightcap reports

| Report | Layout | libsurvive handling |
|---|---|---|
| 0x21 (TR0/WW0 lightcap) | up to 7 × {u16 sensor (low byte used), u16 length, u32 timestamp}, `>0xfd` skipped | `handle_lightcap` |
| 0x27 (LH2, T20/HMD) | up to 4 × {u8 code (sensor \| 0x80 flag), u32 time, u32 data, u32 mask}, code 0xff ends | Debug dump only. Triggers the gen-2 decision, and the raw1 switch when `lh_version == 1`. |
| 0x28 (LH2 raw mode 1) | `28 <len> <raw1 packet>` | `parse_and_process_raw1_lightcap` (below) |

**raw1 (LH2) packet** (`parse_and_process_raw1_lightcap`, driver_vive.c:2081),
read as a byte stream:

- **Channel byte:** `b & 1`. If `b & 0x0A`, the packet is unknown and the rest is
  dropped. `b & 0x04` flags a channel conflict. `channel = b >> 4`.
- **Data word:** otherwise, read a u32 LE word `w`.
  - **Sweep** (`w & 2`): sensor = `w >> 27`, time24 = `(w >> 3) & 0xFFFFFF`,
    `half_clock = w & 4`.
  - **Sync:** time24 = `(w >> 2) & 0xFFFFFF`, ootx = bit 26, g = bit 27.
- **Time:** time24 is expanded with `fix_time24(time24, last_imu)`, which picks the
  2²⁴ epoch nearest the last IMU time (driver_vive.c:1069). LH2 isn't on this
  bench; this part is untested.

### 5.3 Watchman RF: reports 0x23 / 0x24 (dongle, `USB_IF_WATCHMAN1`)

| Report | Length | Content |
|---|---|---|
| 0x23 | 30 | `23` + one 29-byte sub-packet (`report[1..30)`) |
| 0x24 | 59 | `24` + two sub-packets (`report[1..30)`, `report[30..59)`), processed in that order |
| 0x26 | – | "turn off": `ison = 0` (section 6.3) |

Bytes past a sub-packet's declared size are **stale**: leftovers from earlier
reports in the dongle's buffer. Ignore them.

**Sub-packet** (`survive_handle_watchman`, driver_vive.c:2475):

| Offset | Field |
|---|---|
| 0 | `time16` bits 15..8 = device clock bits 31..24 |
| 1 | `size`: the payload is `[3, 3 + (size−1))`, uint8 arithmetic |
| 2 | `time16` bits 7..0 = device clock bits 23..16 |
| 3 | `flags` (first payload byte) |
| 4… | blocks selected by `flags`, in this order: IMU, metadata, input, light |

- **Which decoder runs.** The device needs `use_watchman_v2()`: `lh_version == 1`,
  or an object subtype other than `TRACKER` / `WAND`.
  - Tracker 3.0 is `TRACKER_GEN2`, so it always takes this path.
  - A 2018 tracker on LH1 would take the legacy v1 path instead
    (`read_event`, driver_vive.c:1862). That path has a different flag layout:
    `0b111xxxxx` events, then light. Not observed here.
- **v2 `flags`** (`handle_watchman_v2`, driver_vive.c:2301):

| Bit | Meaning | Block |
|---|---|---|
| 0x80 | IMU | 13 B (§4.2) |
| 0x40 | metadata | marker byte + data: `80 <battery>` (2 B) · `20 <x>` (2 B, "mode?") · `30 <x> <y>` (3 B, seen as `30 00 00` right after startup) · any other marker: rest of sub-packet dropped |
| 0x20 | input | 1 flag byte + data (section 6.2) |
| 0x10 | light | the rest of the payload (§5.4) |
| 0x0F | unknown | 0x01 seen with metadata (`0x51`, `0xd1`); 0x02 seen at startup (`0x52`, `0xd2`) |

- **Heartbeat:** `flags == 0xE2` is a "heartbeat", and the sub-packet is ignored.
  In our captures it appears only around startup, as `e2 01` with `size 3`.

**Observed** flag values: `0x10` (light only), `0x80` (IMU only), `0x90` (IMU
then light), `0x51` / `0xd1` (battery, then light), `0xc0` (IMU, battery),
`0x52` / `0xd2` (startup, metadata `30 00 00`), `0xe2`. No sub-packet in any
capture had an input block.

**Processing guards** (LH1): sub-packets are dropped while `ignoreCnt > 0` (10
after startup), and while the config state machine is still running.

### 5.4 RF light block: format

The light block runs from the current payload position to the payload end:

```
[ S_0 S_1 … S_{m-1} ][ D … D ][ T0 T1 T2 ]
  sensor bytes         deltas   end time: u24 LE = device clock bits 23..0
```

- **Sensor byte** `S`: `sensor_id = S >> 3` (0..31), `edges = S & 7`.
- **Deltas**: a variable-length quantity stored **in reverse**. Reading backwards
  from the byte just before `T0`:
  `delta = (delta << 7) | (b & 0x7F)`, and the code stops after a byte that has
  `0x80` set.
  - The stop byte is the *lowest* address and holds the least-significant 7 bits.
  - In memory order, `80 01` = 128 and `97 02` = 2·128 + 0x17 = 279.
- **Edge times:** `t0` = end time (full 32-bit, §5.6), `t(k+1) = t(k) − delta(k)`.
  Each delta steps back to the previous edge (rising or falling) of any sensor.
- **Pairing edges into pulses:** there are `n = 2m − 1` deltas, giving `2m` edge
  times for `m` pulses.
  - Pulse *i* (sensor byte *i*, in order) **ends** at the newest edge not yet
    claimed.
  - It **starts** `edges + 1` edges further back.
  - So `edges` = the number of other edges between this pulse's rise and fall.
  - Sensor bytes are therefore ordered by falling edge, newest first.

Read one delta at a time from the back, until the deltas would run into the
sensor bytes. The number of times read decides `m`.

### 5.5 RF light block: algorithm (bit-exact port of `read_light_data`, driver_vive.c:1530)

All arithmetic is `u32`, wrapping. `buf` is the whole sub-packet, and `[p0, pend)`
is the light block.

```
fn read_light_data(time16, buf, p0, pend, last_imu) -> Result<Vec<Lightcap>>:
    if pend - p0 <= 3: return []                               // returns 0
    ids = p0;  ev = pend - 4
    t   = (time16 >> 8) << 24 | buf[ev+3] << 16 | buf[ev+2] << 8 | buf[ev+1]
    ref = last_imu as u32
    if t > ref && t - ref > 1<<23 { t -= 1<<24 } else if ref > t && ref - t > 1<<23 { t += 1<<24 }
    times = [0u32; 16]; times[0] = t; ti = 0
    'outer: while ids + ti/2 < ev:                             // ti/2 = integer shift
        d = 0; start = ev
        loop:
            d = (d << 7) | (buf[ev] & 0x7F); b = buf[ev]; ev -= 1
            if b & 0x80: break
            if ids + ti/2 > ev: ev = start; break 'outer       // ran into sensor bytes
        t -= d
        if ti >= 16: return Err(-8)                            // (C writes times[16] when ti == 15)
        ti += 1; times[ti] = t
    if ti is even && ti > 0:                                   // odd number of edges: give one back
        ti -= 1; do { ev += 1 } while buf[ev] & 0x80 == 0
    if (ev - ids + 1) > ti/2 + 1: ids = ev - ti/2              // extra leading bytes: use those next to the deltas
    sensors[i] = (buf[ids+i] >> 3 & 0x1F, buf[ids+i] & 7)  for i in 0..=ti/2
    count = (ti + 1) / 2
    order = [0u8; 16]; k = -1
    for i in 0..count:
        loop { k += 1; if times[k] != 0 { break }; if k + 1 >= 16 { return Err(-2) } }
        s = k + sensors[i].edges + 1
        if s >= 16: return Err(-4)
        if order[s] != 0: return Err(-5)
        order[s] = i + 1
        le[i] = { sensor: sensors[i].id, timestamp: times[s], length: (times[k] - times[s]) as u16 }
        times[s] = 0                                            // start edge consumed
    out = [le[order[j]-1] for j in 0..16 if order[j] != 0]      // newest start first (max 10)
    if any out has length == 0 && timestamp == 0: return Err(-6)
    return reverse(out)                                         // libsurvive emits i = cnt-1 .. 0: ascending start time
```

- **Errors.** On an error code libsurvive logs a warning and drops the whole light
  block. Elements of length 0 are dropped later, in `handle_lightcap`.
- **Unused paths.** The "odd edge" rewind, the extra-leading-bytes rule and
  every error path were **never hit** in our captures (219,343 light blocks in the four main files).
- **Size limits.** At most 6 pulses fit in one sub-packet (≤ 27-byte payload):
  observed 1:8829, 2:15495, 3:2354, 4:14439, 5:11601, 6:675 per capture.
- **Older decoder.** `parse_watchman_lightcap` (driver_vive.c:2598) is an older
  equivalent, compiled only for `VERIFY_LIGHTCAP`.

**Worked example** (capture 131201, checked by `--selftest`). The report is
`23 | 80 18 80 | 10 | 42 2a 32 98 | 87 18 93 0a 0d b4 0b 9c 02 b6 02 a7 01 97 02 | 31 0a 85 | ce c8 (stale)`.

- `time16 = 0x8080`, `size = 0x18` (payload 23 B), `flags 0x10`.
- End time is `0x80 << 24 | 0x850a31` = 2156202545.
- Deltas, read backwards, are 279, 167, 310, 284, 1460, 214291, 3079 → times
  t0..t7 = …2545, …2266, …2099, …1789, …1505, …0045, 2155985754, 2155982675.

| Sensor byte | Sensor, edges | End edge | Start edge | Element |
|---|---|---|---|---|
| 0x42 | 8, 2 | t0 | t3 | 8 @ 2156201789 len 756 |
| 0x2a | 5, 2 | t1 | t4 | 5 @ 2156201505 len 761 |
| 0x32 | 6, 2 | t2 | t5 | 6 @ 2156200045 len 2054 |
| 0x98 | 19, 0 | t6 (t3–t5 used) | t7 | 19 @ 2155982675 len 3079 |

libsurvive logged C 19, 6, 5, 8 with exactly these values.

### 5.6 RF timestamps: epochs, ordering, lossy aspects

- **Clock bits.** The 32-bit device clock is split up:
  - Bits 31..16 sit in the sub-packet header (`time16`).
  - Bits 15..8 of IMU samples sit in `time_lsb`.
  - Bits 23..0 of the last light edge sit in the light block.
- **Light epoch.** A light timestamp takes bits 31..24 from `time16 >> 8`, then
  moves by ±2²⁴ to land within ±2²³ ticks (±175 ms) of `last_imu`. The header
  byte can disagree with the 24-bit time near a rollover.
- **Port requirement.** Process sub-packets strictly in arrival order, and inside
  one sub-packet do IMU before light. The light block then uses that sub-packet's
  own IMU time as its reference.
- **libsurvive quirk:** after LH-generation detection resets
  `last_imu = 0`, light until the next IMU sample is shifted by −2²⁴. The
  reference decoder reproduces this, and it was needed to match the restart cases
  in `captures/vive/bad/`.
- **Lossy:**
  - **Thinned light.** The tracker sends only some pulses over RF. A sync flash
    usually shows up on about one sensor, not all the ones that saw it (hence
    `--disambiguator-min-sync-sensors 1` for `WM*`).
  - **Lengths are u16.** Lengths above 65535 wrap silently in `read_light_data`;
    the older decoder rejected them.
  - **Coarse IMU time.** IMU time resolution is 256 ticks.
- **Sync lengths look longer over RF.** Median length per acode is
  ≈ `ACODE_TIMING + 330` ticks (§8.2). This is a detector property; the encoding
  is lossless.

---

## 6. Buttons, battery, power, connection, commands

### 6.1 Wired Buttons interface: input report 0x01 (driver_vive.c:3139–3246)

`reportType = u16 @ [1]`:

| reportType | Content |
|---|---|
| 0x0100 | User input. Buttons `u32 @ [8]`, trigger `u16 @ [0x1a]`/32768, trackpad x/y `i16 @ [0x14]`, `[0x16]` /32768. The C code falls through to a hex dump. |
| 0x0400 | Battery `u8 @ [14]` (bit 7 = charging, 6..0 = percent), plus a seconds counter `@ [4]` |
| 0x0700 | About 20 Hz counter |
| 0x0800 | HMD IPD/proximity |

- Offsets are into the full report; libsurvive's indices are relative to
  `report + 1`.
- The wired tracker maps button bit 13 to `SYSTEM`. For the RF tracker there's
  no mapping table: the bit index is the button id.

### 6.2 RF input block (flags 0x20; `handle_input`, driver_vive.c:2199)

The block starts with the input flag byte `f`.

- **If `f & 7 != 0`**, the optional fields follow in this order:
  - `f & 1`: u8 buttons.
  - `f & 4`: u8 trigger (/255).
  - `f & 2`: i16 x, i16 y trackpad (/32768).
  - Bit 0x08 is unknown and ignored.
- **Else**, the next byte is the "gen-two" type:
  - `0xA1` (Knuckles): u8 touch flags, then 6 × u8 proximity/force (/255).
  - Any other type: error; the light in this sub-packet is skipped.

Not observed on the tracker.

### 6.3 Battery, power, connect/disconnect (RF)

- **Battery.** Metadata marker `0x80`: `bat & 0x7F` = percent, bit 7 = charging
  (`handle_battery`; ignored when it's 0 and not charging). It arrives about every
  10 s. Observed `0xe4` = charging, 100 % (tracker on a power bank).
- **Turn off.** Report `0x26` ("RF_TURN_OFF") sets `ison = 0`. Never seen in our
  captures.
- **No explicit connect or disconnect event exists in libsurvive.**
  - A dongle device (`WM0`) exists from enumeration on.
  - The tracker counts as "connected" only when the config read (0x10/0x11)
    succeeds, since the dongle answers it from the tracker.
  - If the tracker goes away, 0x23/0x24 reports just stop.
  - libsurvive doesn't re-run the handshake when the tracker comes back. Our
    `vive_web/server.py` restarts survive-cli after a gap in HID reports.
  - Right after the protocol switch, the dongle emits `0xe2` heartbeats and
    metadata `30 00 00` sub-packets for about 50 ms before normal traffic.
- **Power-off command** (comment only, driver_vive.c:175–183, not implemented):
  SteamVR sends `9f 04 6f 66 66 21` ("off!") as a SET_REPORT to the tracker.
  - The HMD `vive_magic_power_off` is unrelated.

### 6.4 Commands: feature report `ff <cmd> <len> <data…>` on interface 0

| cmd | Name | Payload / use |
|---|---|---|
| 0x87 | CHANGE_PROTOCOL | `06 01 <raw> 00 02 <raw2> 00` (lightcap mode, §2.3), or `03 00 01 00` (protocol switch) |
| 0x96 | CONFIGURE_RADIO | `10` + 16 fixed bytes (§2.2). SteamVR sends it even with no device paired. |
| **0xAD** | **PAIR** | `ff ad 03 01 10 27`: 3 bytes `01 10 27` (0x2710 = 10000, probably a 10 s timeout). Puts the dongle into pairing; you confirmed it works on this dongle. |
| 0x8F | HAPTIC_PULSE | `ff 8f 07 00 <hi u16> <lo u16> <repeat u16>` (10 B), see below |
| 0xA1 | CHECK_ALIVE | sniffed only; the reply contains a serial string |
| 0x83 | ? | sniffed only |
| 0xB5, 0x8C | ? | Knuckles, sniffed only |
| 0x9F | POWER_OFF | see §6.3 |

**Pairing in libsurvive.** `--pair-device` resends PAIR every 1 s, but only to a
dongle with `so == 0` (driver_vive.c:1023). Under HIDAPI the object is created
at enumeration, so **`--pair-device` never fires on macOS**; send the report
yourself.

**Haptics** (`survive_vive_send_haptic`, driver_vive.c:523; skipped for HMDs
and before the config has loaded):

```
period_us = 1e6/frequency; lo = amplitude*period_us/2 (≥1); hi = period_us - lo
repeat = duration_s*frequency (≥1); all clamped to u16
```

### 6.5 Other report ids (unused by libsurvive's decode)

| Id | Kind | Use |
|---|---|---|
| 0x03 | feature | error counters (layout in the comment at driver_vive.c:55–85) |
| 0x08 / 0x09 | feature | HMD user-data read |
| 0x13 | feature | version (alt) |
| 0x16 | feature | reboot: `16 01 00 00 00` |

---

## 7. Raw recording lines (oracle formats)

`C <sensor> <timestamp> <length>` comes from `survive_recording_lightcap`. It's
written inside `handle_lightcap`, before sensor mapping and after the
zero-length / generation-detection filters.

`i <mask> <tc> ax ay az gx gy gz mx my mz <id>` is the raw IMU in counts, printed
as floats; `I` is the same after §4.3. `L|R X|Y <sensor> <acode> <timeinsweep>
<timecode> <length> <lh>` is disambiguator output, where sensor −1/−2 = sync
and −3 = sweep summary.

- **Ordering.** Every line from processing one HID report comes after that
  report's `HIDIN` line, and before the next one.
- **Where captures start.** A capture begins mid-stream, so a few C/i lines can
  precede the first `HIDIN`.
- **Interleaved logs.** libsurvive log lines interleave on stdout, so lines can
  carry an ANSI-escape prefix (`\x1b[0m`).
- **Restarts.** Restarts of survive-cli inside a recording show up as the time
  column jumping back to about 0.

---

## 8. Lighthouse 1.0 facts libsurvive relies on

### 8.1 Clock and cycle (disambiguator_statebased.c:13–48, 124–146; survive_process_gen1.c)

- **Timing.** The device clock runs at **48 MHz**. One rotor sweep period is
  **400 000 ticks** (8.333 ms, 120 Hz).
- **Pulse windows.** Each period starts with two **20 000-tick** pulse windows,
  one sync flash per base station, followed by a 360 000-tick capture window.
- **Two-station cycle** (modes **b / c**, or A + b with a cable): 4 periods =
  **1 600 000 ticks** (33.3 ms).
- **Single station** (60 Hz mode, `single_60hz_mode`): only the first 800 000
  ticks repeat.

| Ticks | Sync (`acode`, lh) | Then |
|---|---|---|
| 0 | 4 (skip, X), lh 1 | |
| 20 000 | 0 (X), lh 0 | lh 0 sweeps **X** |
| 400 000 | 5 (skip, Y), lh 1 | |
| 420 000 | 1 (Y), lh 0 | lh 0 sweeps **Y** |
| 800 000 | 0 (X), lh 1 | |
| 820 000 | 4 (skip, X), lh 0 | lh 1 sweeps **X** |
| 1 200 000 | 1 (Y), lh 1 | |
| 1 220 000 | 5 (skip, Y), lh 0 | lh 1 sweeps **Y** |

- **Which station flashes first.** Here lh 1 is the station that flashes first
  in every period: the master. Ours is OOTX `mode_current` 1 = **B**, id
  699579721. lh 0 is the slave, **C** (mode 2), id 1416427640.
- **`acode`** = `skip << 2 | data << 1 | axis`. Skip means "I'm not sweeping
  this period", data is the OOTX bit, and axis is 0 for X, 1 for Y.

### 8.2 Sync length → acode

```
ACODE_TIMING(a) = 3000 + (a&1)*500 + (a>>1&1)*1000 + (a>>2&1)*2000 - 250   = 2750 + 500a  (ticks)
find_acode(len): len < 2550 → none; else a = first k with len < 3050 + 500k  (k = 0..7); ≥ 6550 → none
```

- **Nominal pulse length** is 62.5 µs + 10.4 µs·a, i.e. 3000 + 500a ticks.
- **Sync vs sweep:** 2250 ≤ length ≤ 6750 is classed as a sync (naive);
  sweeps above 10 000 ticks are discarded (48 000 in our patch).
- **Data bit.** For an expected code the data bit is whichever of
  `ACODE_TIMING(a)` and `ACODE_TIMING(a|2)` is closer.
- **Losing lock.** An error above 800 ticks costs confidence; below 0 confidence,
  the state is re-found.
- **Observed over RF:** the median sync length is 3080 + 500a, i.e.
  ACODE_TIMING + about 330 (all 8 codes).

### 8.3 Sweep → angle

- **`timeinsweep`** is the pulse centre (`timestamp + length/2`) minus the start
  of the **sweeping station's own** sync in that period. Positions are taken
  modulo the cycle, with the station's `mod_offset` (disambiguator_statebased.c:1015).
- **Angle** (survive_process_gen1.c:80):
  **`angle = timeinsweep · π / 400000 − π/2`** radians. 0 is the middle of the
  sweep, and libsurvive rejects `timeinsweep > 400000`.
- **Calibration.** fcal (§8.4) corrections are applied later, in the reprojection
  model.

### 8.4 OOTX (ootx_decoder.c)

- **Bitstream.** One bit per sync flash of a station: `(acode >> 1) & 1`, so
  each station sends 120 bit/s (fed by `survive_ootx_behavior`).
- **Preamble:** 17 zeros then a 1. Detected as `(history & 0x3FFFF) == 1`, and it
  can restart a packet at any time.
- **Framing:** after the preamble, a **sync bit = 1** follows every 16 data bits.
  A 0 there aborts the packet ("bad sync bit").
- **Bit order:** data bits fill bytes **MSB first**.
- **Packet:** `u16 LE length`, then `length` payload bytes padded to even, then
  `u32 LE CRC32` (zlib `crc32` over the unpadded payload).

**Payload `lighthouse_info_v6`** (33 bytes, packed LE, `f16` = IEEE half):

| Off | Type | Field | Off | Type | Field |
|---|---|---|---|---|---|
| 0 | u16 | fw_version (bits 15..6) / protocol (5..0) = 6 | 20 | i8[3] | accel_dir x,y,z (up vector) |
| 2 | u32 | id (base station serial) | 23 | f16[2] | fcal gibphase 0,1 |
| 6 | f16[2] | fcal phase 0,1 | 27 | f16[2] | fcal gibmag 0,1 |
| 10 | f16[2] | fcal tilt 0,1 (libsurvive stores `tan(tilt)`) | 31 | u8 | mode_current (0 = A, 1 = B, 2 = C) |
| 14 | u8 | sys_unlock_count | 32 | u8 | sys_faults |
| 15 | u8 | hw_version | | | |
| 16 | f16[2] | fcal curve 0,1 | | | |

Index 0 is rotor 0, the X sweep; index 1 is rotor 1, Y.

- **Packet callback** (`ootx_packet_cblk_d_gen1`, survive_process_gen2.c:87).
  It copies the fields into `ctx->bsd[lh]`, and resets the station pose if the
  id, mode or `accel_dir` changed.
- **Validation:** decoding the sync data bits that libsurvive logged
  (`--record-angle`) through a port of `ootx_pump_bit` gives:

  | Capture | lh 0 | lh 1 |
  |---|---|---|
  | 131723 | 35 CRC-valid packets | 35 CRC-valid packets |
  | 132513 | 31 | 20 |

  The ids match `lighthouses_at_start`: lh 0 = 1416427640 (mode 2) and
  lh 1 = 699579721 (mode 1). Both stations report fw 436, protocol 6,
  `accel_dir (1,127,0)`.

---

## 9. Porting checklist (the non-obvious parts)

1. **Config first, then magics.** For a dongle, retry GET 0x10 until the
   tracker answers.
   - Pad the RF magics to 64 bytes on macOS.
   - Then GET 0x05 and GET 0x01.
2. **Wired tracker.** Expect re-enumeration after the first `04 00 00 00 00`;
   reconnect and redo the handshake.
3. **RF sub-packet.** Use `size`, not the report length, and ignore stale
   trailing bytes. Handle 0x24 as two sub-packets in order.
4. **Within a sub-packet:** IMU, then metadata, then input, then light. Update
   `last_imu` before decoding light.
5. **Light block:** use the reverse-VLQ deltas and edge pairing from §5.5,
   implemented with wrapping `u32`. Emit pulses in ascending start time. Map
   sensors through `channelMap` afterwards.
6. **IMU:** RF timecode = `time16 << 16 | lsb << 8`. Scale from GET 0x01, then
   apply `(x·s − b)·k` with `acc_bias / 1000`.
7. **USB 0x20:** accept slots by `int8` sequence difference. Consider sorting by
   sequence rather than copying libsurvive's in-order rule.
8. **Validate a port the same way:** feed the `HIDIN` bytes from `captures/vive`
   and diff the output against the `C` / `i` lines; `vive_proto_ref.py` shows how.

---

## 10. Validation

`vive_proto_ref.py` runs these checks over a recording:

- **Decode and compare.** Every `HIDIN` report goes through the decoder above;
  the lightcap elements and raw IMU samples are compared, in order, with the
  `C` / `i` lines libsurvive wrote after that report.
- **Calibrated IMU.** It recomputes `I` from `i` and the config.
- **Config.** It reassembles the config from HIDFEAT 0x10/0x11 and compares it
  with `CONFIG`, and checks IMU_SCALES.
- **OOTX.** It decodes OOTX from the logged sync records.

Results (2026-10-27 captures):

| Capture | Reports compared | Lightcap elements identical | Raw IMU identical | `I` within 1.5e-6 |
|---|---|---|---|---|
| 20261027-131201_turntable-center | 70,329 | 192,865 / 192,865 | 35,503 / 35,503 | 35,503 / 35,503 |
| 20261027-131723_turntable-r | 55,248 | 166,688 / 166,688 | 27,855 / 27,855 | 27,855 / 27,855 |
| 20261027-132216_slider | 50,818 | 181,332 / 181,332 | 25,524 / 25,524 | 25,524 / 25,524 |
| 20261027-132513_near | 46,132 | 148,844 / 148,844 | 23,303 / 23,303 | 23,303 / 23,303 |
| bad/20261027-130744 (7 restarts) | 98,446 | 263,157 / 263,157 | 49,768 / 49,768 | 49,768 / 49,768 |
| bad/20261027-131702 | 9,781 | 28,767 / 28,767 | 4,917 / 4,917 | 4,917 / 4,917 |
| **Total** | **330,754** | **981,653 / 981,653** | **166,870 / 166,870** | **all** |

Every report's complete C/i output is identical, as a sequence, in all files.
A few reports aren't scored, by design:

- **Before the first IMU.** 1–3 reports per file come before the first IMU
  sample, so their light epoch is unknown.
- **Cut short.** The last report of a file, or the last before a restart, can be
  cut short by the stop. It's checked as a prefix.
- **Restarts.** In `bad/…130744`, 330 reports were skipped while libsurvive
  re-detected LH1 after its 7 restarts, and 4 lines were damaged by the
  restarts.

Config: 2007 zlib bytes → 8065-byte JSON, equal to the `CONFIG` line in all files.

---

## 11. Uncertain or not decodable

- **Wired formats are unverified.** 0x20, 0x25, the Buttons report 0x01 and the
  wired `04` / `07` handshake come from source only. The ring order of the
  three 0x20 slots is unknown (§4.1).
- **Unknown flag bits.**
  - RF metadata markers 0x20 and 0x30: the meaning is unknown, but the lengths
    are known.
  - Flags bits 0x01–0x08: 0x01 comes with battery, 0x02 at startup.
  - The `0xe2` heartbeat payload.
- **Command meanings are guesses.** The meanings of the CHANGE_PROTOCOL bytes,
  CONFIGURE_RADIO's 16-byte key and PAIR's `01 10 27` are inferred from sniffed
  SteamVR traffic, not documented.
- **Paths not exercised here:**
  - LH2 / raw1 decoding (0x27, 0x28, RF raw1 including the `timeWithoutFlag`
    discard heuristic).
  - The legacy watchman v1 event path (2018 tracker on LH1 over RF).
  - Knuckles input and haptics.
- **libsurvive behaviour a port may *not* want to copy:**
  - The `last_imu = 0` light shift after generation detection.
  - The 7-byte IMU length check.
  - `times[16]` overflow at 15 deltas (impossible at ≤ 27-byte payloads).
  - In-order 0x20 slot acceptance.
  - `--pair-device` never firing under HIDAPI.
- **Version report:** fields after byte 53 are stale buffer bytes, and the FPGA
  version split (538/7/2) is libsurvive's interpretation.

//! Command-line parsing for lhd. Parsing touches no device: `--help`,
//! `--version` and argument errors are answered before any transport (the USB
//! receiver) is opened, so a diagnostic invocation never claims the dongle from
//! a running daemon.

pub const USAGE: &str = "\
usage: lhd [--replay REC.gz [--speed X]] [--calibration FILE.json] [--record FILE.rec.gz]
           [--host-clock [--stream-sweeps]] [--pose-timing]
       lhd --help | --version

Tracks Lighthouse 1.0 devices and prints JSON lines on stdout; commands come
in as JSON lines on stdin. Without --replay it opens the USB receivers (hidapi).

  --replay REC.gz        replay a recording instead of the live receivers
  --speed X              replay speed (default 1.0; needs --replay)
  --calibration FILE     load station calibration from FILE, save new ones to it
  --record FILE          record raw traffic to FILE (.rec.gz)
  --host-clock           map device clocks onto report timestamps (Input::Clock)
  --stream-sweeps        stream each sweep's hits as they come
  --pose-timing          a pose_timing line per pose
  -h, --help             print this and exit (opens no device)
  -V, --version          print the version and exit (opens no device)
";

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Options {
    pub replay: Option<String>,
    pub speed: f64,
    pub calibration: Option<String>,
    pub record: Option<String>,
    pub host_clock: bool,
    pub stream_sweeps: bool,
    pub pose_timing: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    Help,
    Version,
    Run(Options),
}

/// Parses the arguments after the program name. `--help` / `--version` win
/// wherever they appear (even after a bad option), so asking for help never
/// fails and never starts tracking.
pub fn parse<S: AsRef<str>>(args: &[S]) -> Result<Command, String> {
    let args: Vec<&str> = args.iter().map(|a| a.as_ref()).collect();
    if args.iter().any(|a| matches!(*a, "-h" | "--help")) {
        return Ok(Command::Help);
    }
    if args.iter().any(|a| matches!(*a, "-V" | "--version")) {
        return Ok(Command::Version);
    }
    let mut o = Options { speed: 1.0, ..Options::default() };
    let mut speed_given = false;
    let mut it = args.iter().copied();
    while let Some(a) = it.next() {
        // --name=value or --name value.
        let (name, inline) = match a.split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n, Some(v)),
            _ => (a, None),
        };
        let mut value = |name: &str| -> Result<String, String> {
            let v = match inline {
                Some(v) => v.to_string(),
                None => it.next().map(str::to_string).ok_or_else(|| format!("{} needs a value", name))?,
            };
            if v.is_empty() || (inline.is_none() && v.starts_with("--")) {
                return Err(format!("{} needs a value", name));
            }
            Ok(v)
        };
        let flag = |name: &str| match inline {
            Some(_) => Err(format!("{} takes no value", name)),
            None => Ok(true),
        };
        match name {
            "--replay" => o.replay = Some(value(name)?),
            "--calibration" => o.calibration = Some(value(name)?),
            "--record" => o.record = Some(value(name)?),
            "--speed" => {
                let v = value(name)?;
                o.speed = v.parse().ok().filter(|s: &f64| s.is_finite() && *s > 0.0).ok_or_else(|| format!("--speed: not a positive number: {:?}", v))?;
                speed_given = true;
            }
            "--host-clock" => o.host_clock = flag(name)?,
            "--stream-sweeps" => o.stream_sweeps = flag(name)?,
            "--pose-timing" => o.pose_timing = flag(name)?,
            other if other.starts_with('-') => return Err(format!("unknown option {:?}", other)),
            other => return Err(format!("unexpected argument {:?}", other)),
        }
    }
    if speed_given && o.replay.is_none() {
        return Err("--speed needs --replay".into());
    }
    Ok(Command::Run(o))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(args: &[&str]) -> Options {
        match parse(args) {
            Ok(Command::Run(o)) => o,
            other => panic!("{:?}: {:?}", args, other),
        }
    }

    #[test]
    fn help_and_version_win_anywhere() {
        for a in [&["--help"][..], &["-h"], &["--calibration", "c.json", "--help"], &["--bogus", "-h"], &["--replay"/* no value */, "--help"]] {
            assert_eq!(parse(a), Ok(Command::Help), "{:?}", a);
        }
        assert_eq!(parse(&["--version"]), Ok(Command::Version));
        assert_eq!(parse(&["-V", "--record", "x"]), Ok(Command::Version));
    }

    #[test]
    fn options() {
        assert_eq!(run(&[]), Options { speed: 1.0, ..Options::default() });
        let o = run(&["--replay", "r.gz", "--speed=2.5", "--calibration", "c.json", "--host-clock", "--stream-sweeps", "--pose-timing", "--record=o.rec.gz"]);
        assert_eq!(o.replay.as_deref(), Some("r.gz"));
        assert_eq!(o.speed, 2.5);
        assert_eq!(o.calibration.as_deref(), Some("c.json"));
        assert_eq!(o.record.as_deref(), Some("o.rec.gz"));
        assert!(o.host_clock && o.stream_sweeps && o.pose_timing);
    }

    #[test]
    fn errors() {
        for (a, want) in [
            (&["--bogus"][..], "unknown option"),
            (&["--calibration"], "needs a value"),
            (&["--calibration", "--host-clock"], "needs a value"),
            (&["--calibration="], "needs a value"),
            (&["--replay", "r.gz", "--speed", "fast"], "not a positive number"),
            (&["--speed", "2"], "needs --replay"),
            (&["--host-clock=yes"], "takes no value"),
            (&["stray"], "unexpected argument"),
        ] {
            let e = parse(a).expect_err(&format!("{:?}", a));
            assert!(e.contains(want), "{:?}: {}", a, e);
        }
    }
}

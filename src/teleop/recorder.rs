//! Client-side tele-op trace recorder (goal/011 D4, hardened in goal/013).
//!
//! `Recorder` writes one JSON object per `record(...)` call to a JSONL file.
//! Pure client code — never touches the wire protocol or `src/agent/`. The
//! companion `Player` in `playback.rs` consumes the same format.
//!
//! Crash-safety contract (goal/013 D3):
//!   * `create` auto-creates the parent directory.
//!   * Every `record` call is fallible — disk-full / permission-denied
//!     does NOT panic. Instead the recorder enters the `Disabled` state,
//!     logs once, and silently drops further writes.
//!   * `try_record` exposes the disabled state to callers that care; the
//!     legacy `record` keeps its `Result` signature for compatibility.
//!   * Drop flushes best-effort; never panics even if the file vanished.
//!
//! Trace line schema:
//! ```json
//! {"ts": <secs>, "step": <u64>, "obs": <GymRobotState>, "act": <RobotAction>}
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::robot::state::{GymRobotState, RobotAction};

/// One frame of a tele-op trace.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceFrame {
    /// Wall-clock or sim-time seconds since trace start.
    pub ts: f64,
    /// Monotonic step counter — matches `ServerMessage::Observation::step_count`.
    pub step: u64,
    /// Observation snapshot returned by the server before this action.
    pub obs: GymRobotState,
    /// Action the client decided to send for this step.
    pub act: RobotAction,
}

/// Outcome of one `try_record` call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecorderState {
    /// Frame was buffered for write.
    Written,
    /// Recorder previously hit an I/O error and is dropping frames.
    Disabled,
}

/// Errors a recorder can surface to its caller.
#[derive(Debug)]
pub enum RecorderError {
    /// JSON serialization of the frame failed.
    Serialize(serde_json::Error),
    /// Underlying file write failed.
    Io(std::io::Error),
    /// Could not create or open the trace file.
    Open(std::io::Error),
}

impl fmt::Display for RecorderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecorderError::Serialize(e) => write!(f, "frame serialize failed: {e}"),
            RecorderError::Io(e) => write!(f, "trace write failed: {e}"),
            RecorderError::Open(e) => write!(f, "trace open failed: {e}"),
        }
    }
}

impl std::error::Error for RecorderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RecorderError::Serialize(e) => Some(e),
            RecorderError::Io(e) | RecorderError::Open(e) => Some(e),
        }
    }
}

impl From<RecorderError> for std::io::Error {
    fn from(e: RecorderError) -> Self {
        match e {
            RecorderError::Io(e) | RecorderError::Open(e) => e,
            RecorderError::Serialize(e) => std::io::Error::other(e),
        }
    }
}

/// JSONL trace writer. Drop the recorder to flush + close the file.
pub struct Recorder {
    path: PathBuf,
    writer: Option<BufWriter<File>>,
    frames_written: u64,
    frames_dropped: u64,
    disabled_reason: Option<String>,
}

impl Recorder {
    /// Create a new trace at `path`, overwriting any existing file. The
    /// parent directory is created if missing.
    pub fn create(path: impl AsRef<Path>) -> Result<Self, RecorderError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(RecorderError::Open)?;
            }
        }
        let file = File::create(&path).map_err(RecorderError::Open)?;
        Ok(Self {
            path,
            writer: Some(BufWriter::new(file)),
            frames_written: 0,
            frames_dropped: 0,
            disabled_reason: None,
        })
    }

    /// Append one frame. Disk-full / permission-denied disables the
    /// recorder (logged once) and subsequent calls are silent no-ops.
    /// Returns `Written` on success or `Disabled` once the recorder has
    /// stopped writing.
    pub fn try_record(
        &mut self,
        ts: f64,
        step: u64,
        obs: &GymRobotState,
        act: &RobotAction,
    ) -> RecorderState {
        if self.writer.is_none() {
            self.frames_dropped = self.frames_dropped.saturating_add(1);
            return RecorderState::Disabled;
        }

        let frame = TraceFrame {
            ts,
            step,
            obs: obs.clone(),
            act: act.clone(),
        };

        let line = match serde_json::to_string(&frame) {
            Ok(s) => s,
            Err(e) => {
                self.disable(format!("serialize: {e}"));
                return RecorderState::Disabled;
            }
        };

        let writer = match self.writer.as_mut() {
            Some(w) => w,
            None => return RecorderState::Disabled,
        };

        if let Err(e) = writer
            .write_all(line.as_bytes())
            .and_then(|_| writer.write_all(b"\n"))
        {
            self.disable(format!("write: {e}"));
            return RecorderState::Disabled;
        }

        self.frames_written = self.frames_written.saturating_add(1);
        RecorderState::Written
    }

    /// Append one frame returning a Result. Kept for callers that want
    /// to handle write errors explicitly without using the soft-disable
    /// path of `try_record`.
    pub fn record(
        &mut self,
        ts: f64,
        step: u64,
        obs: &GymRobotState,
        act: &RobotAction,
    ) -> Result<u64, RecorderError> {
        if self.writer.is_none() {
            return Err(RecorderError::Io(std::io::Error::new(
                std::io::ErrorKind::Other,
                self.disabled_reason
                    .clone()
                    .unwrap_or_else(|| "recorder disabled".into()),
            )));
        }
        let frame = TraceFrame {
            ts,
            step,
            obs: obs.clone(),
            act: act.clone(),
        };
        let line = serde_json::to_string(&frame).map_err(RecorderError::Serialize)?;
        let writer = self.writer.as_mut().ok_or_else(|| {
            RecorderError::Io(std::io::Error::other("recorder disabled mid-call"))
        })?;
        writer
            .write_all(line.as_bytes())
            .and_then(|_| writer.write_all(b"\n"))
            .map_err(|e| {
                let kind = e.kind();
                self.disable(format!("{kind:?}"));
                RecorderError::Io(e)
            })?;
        let idx = self.frames_written;
        self.frames_written = self.frames_written.saturating_add(1);
        Ok(idx)
    }

    fn disable(&mut self, reason: String) {
        if self.disabled_reason.is_none() {
            log::warn!(
                "tele-op recorder disabled after {} frames at {}: {}",
                self.frames_written,
                self.path.display(),
                reason
            );
            self.disabled_reason = Some(reason);
        }
        // Drop the writer so further calls early-return without
        // re-attempting I/O on a known-bad sink.
        self.writer = None;
    }

    /// Flush buffered bytes to disk. No-op if the recorder is disabled.
    pub fn flush(&mut self) -> std::io::Result<()> {
        match self.writer.as_mut() {
            Some(w) => w.flush(),
            None => Ok(()),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn frames_written(&self) -> u64 {
        self.frames_written
    }

    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped
    }

    pub fn is_disabled(&self) -> bool {
        self.writer.is_none()
    }

    pub fn disabled_reason(&self) -> Option<&str> {
        self.disabled_reason.as_deref()
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        if let Some(w) = self.writer.as_mut() {
            let _ = w.flush();
        }
    }
}

/// Read every frame from a JSONL trace at `path`. Returns frames in the order
/// they were recorded. Used by the playback path and by trace inspection
/// tooling. Bad lines are reported with line numbers; one malformed line
/// halts the read so callers know the trace is incomplete.
pub fn read_trace(path: impl AsRef<Path>) -> std::io::Result<Vec<TraceFrame>> {
    use std::io::{BufRead, BufReader};
    let file = File::open(path.as_ref())?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let frame: TraceFrame = serde_json::from_str(&line).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("trace line {} parse error: {}", lineno + 1, e),
            )
        })?;
        out.push(frame);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::sensors::ImuReading;
    use crate::robot::state::{GripperState, GymSensorReadings};
    use glam::Vec3;

    fn sample_obs() -> GymRobotState {
        GymRobotState {
            joint_positions: vec![0.1, 0.2, 0.3],
            joint_velocities: vec![0.01, 0.02, 0.03],
            sensor_readings: GymSensorReadings {
                distances: vec![1.0],
                contacts: vec![false],
                imu: vec![ImuReading {
                    linear_acceleration: Vec3::ZERO,
                    angular_velocity: Vec3::ZERO,
                }],
                camera_visible: vec![],
            },
            gripper_states: vec![GripperState {
                is_open: true,
                attached_object: None,
            }],
            combat: None,
        }
    }

    fn sample_act() -> RobotAction {
        RobotAction {
            motor_velocities: vec![1.0, -0.5, 0.0],
            gripper_commands: vec![false],
            base_velocity: [0.0, 0.0],
        }
    }

    #[test]
    fn record_writes_one_line_per_frame() {
        let tmp = tempfile_path("recorder_one_line.jsonl");
        {
            let mut rec = Recorder::create(&tmp).expect("create recorder");
            rec.record(0.0, 0, &sample_obs(), &sample_act())
                .expect("record 0");
            rec.record(0.05, 1, &sample_obs(), &sample_act())
                .expect("record 1");
            rec.record(0.10, 2, &sample_obs(), &sample_act())
                .expect("record 2");
            assert_eq!(rec.frames_written(), 3);
        }

        let contents = std::fs::read_to_string(&tmp).expect("read back");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 3, "expected 3 JSONL lines");
        for line in &lines {
            let _: TraceFrame = serde_json::from_str(line).expect("each line valid JSON frame");
        }
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn read_trace_round_trips_recorded_frames() {
        let tmp = tempfile_path("recorder_roundtrip.jsonl");
        {
            let mut rec = Recorder::create(&tmp).expect("create recorder");
            for i in 0..5u64 {
                let mut act = sample_act();
                act.motor_velocities[0] = i as f32 * 0.1;
                rec.record(i as f64 * 0.02, i, &sample_obs(), &act)
                    .expect("record");
            }
        }
        let frames = read_trace(&tmp).expect("read trace");
        assert_eq!(frames.len(), 5);
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame.step, i as u64);
            assert!((frame.act.motor_velocities[0] - i as f32 * 0.1).abs() < 1e-6);
        }
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn record_empty_observation_still_valid() {
        let tmp = tempfile_path("recorder_empty_obs.jsonl");
        {
            let mut rec = Recorder::create(&tmp).expect("create recorder");
            let empty = GymRobotState {
                joint_positions: vec![],
                joint_velocities: vec![],
                sensor_readings: GymSensorReadings {
                    distances: vec![],
                    contacts: vec![],
                    imu: vec![],
                    camera_visible: vec![],
                },
                gripper_states: vec![],
                combat: None,
            };
            let act = RobotAction {
                motor_velocities: vec![],
                gripper_commands: vec![],
                base_velocity: [0.0, 0.0],
            };
            rec.record(0.0, 0, &empty, &act).expect("record empty");
        }
        let frames = read_trace(&tmp).expect("read");
        assert_eq!(frames.len(), 1);
        assert!(frames[0].obs.joint_positions.is_empty());
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn create_autocreates_parent_dir() {
        let mut dir = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        dir.push(format!("echomap_recorder_parent_{nonce}"));
        let inner = dir.join("nested/sub");
        let trace = inner.join("trace.jsonl");
        {
            let mut rec = Recorder::create(&trace).expect("create with missing parent");
            rec.record(0.0, 0, &sample_obs(), &sample_act())
                .expect("record one frame");
        }
        assert!(trace.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn try_record_returns_disabled_after_manual_disable() {
        let tmp = tempfile_path("recorder_disabled.jsonl");
        let mut rec = Recorder::create(&tmp).expect("create");
        let s = rec.try_record(0.0, 0, &sample_obs(), &sample_act());
        assert_eq!(s, RecorderState::Written);
        // Force-disable to simulate ENOSPC without actually filling the
        // disk. After this, all writes must be no-ops.
        rec.disable("simulated".into());
        assert!(rec.is_disabled());
        for i in 1..5u64 {
            let s = rec.try_record(i as f64, i, &sample_obs(), &sample_act());
            assert_eq!(s, RecorderState::Disabled);
        }
        assert_eq!(rec.frames_written(), 1);
        assert_eq!(rec.frames_dropped(), 4);
        assert!(rec.disabled_reason().is_some());
        drop(rec);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn record_after_disable_returns_error_not_panic() {
        let tmp = tempfile_path("recorder_record_after_disable.jsonl");
        let mut rec = Recorder::create(&tmp).expect("create");
        rec.disable("forced".into());
        let res = rec.record(0.0, 0, &sample_obs(), &sample_act());
        assert!(res.is_err());
        drop(rec);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn drop_does_not_panic_on_disabled() {
        let tmp = tempfile_path("recorder_drop_disabled.jsonl");
        let mut rec = Recorder::create(&tmp).expect("create");
        rec.disable("test".into());
        drop(rec);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn recorder_error_display_has_context() {
        let e = RecorderError::Io(std::io::Error::new(std::io::ErrorKind::Other, "boom"));
        let s = format!("{e}");
        assert!(s.contains("trace write failed"));
    }

    fn tempfile_path(name: &str) -> PathBuf {
        let mut dir = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        dir.push(format!("echomap_teleop_{}_{}", nonce, name));
        dir
    }
}

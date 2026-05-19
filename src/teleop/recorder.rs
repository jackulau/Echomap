//! Client-side tele-op trace recorder (goal/011 D4).
//!
//! `Recorder` writes one JSON object per `record(...)` call to a JSONL file.
//! Pure client code — never touches the wire protocol or `src/agent/`. The
//! companion `Player` in `playback.rs` consumes the same format.
//!
//! Trace line schema:
//! ```json
//! {"ts": <secs>, "step": <u64>, "obs": <GymRobotState>, "act": <RobotAction>}
//! ```

use serde::{Deserialize, Serialize};
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

/// JSONL trace writer. Drop the recorder to flush + close the file.
pub struct Recorder {
    path: PathBuf,
    writer: BufWriter<File>,
    frames_written: u64,
}

impl Recorder {
    /// Create a new trace at `path`, overwriting any existing file. Parent
    /// directories must already exist.
    pub fn create(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::create(&path)?;
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            frames_written: 0,
        })
    }

    /// Append one frame. The frame is serialized as a single JSON line
    /// terminated with `\n`. Returns the frame index (0-based).
    pub fn record(
        &mut self,
        ts: f64,
        step: u64,
        obs: &GymRobotState,
        act: &RobotAction,
    ) -> std::io::Result<u64> {
        let frame = TraceFrame {
            ts,
            step,
            obs: obs.clone(),
            act: act.clone(),
        };
        let line = serde_json::to_string(&frame).map_err(std::io::Error::other)?;
        self.writer.write_all(line.as_bytes())?;
        self.writer.write_all(b"\n")?;
        let idx = self.frames_written;
        self.frames_written += 1;
        Ok(idx)
    }

    /// Flush buffered bytes to disk. Called automatically on drop.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn frames_written(&self) -> u64 {
        self.frames_written
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

/// Read every frame from a JSONL trace at `path`. Returns frames in the order
/// they were recorded. Used by the playback path and by trace inspection
/// tooling.
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

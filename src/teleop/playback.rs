//! Deterministic playback of a tele-op trace (goal/011 D5).
//!
//! Reads a JSONL trace produced by [`crate::teleop::recorder::Recorder`],
//! reconnects to a freshly-reset agent server, replays every recorded action
//! in order, and asserts the observed `joint_positions` stay within
//! `tolerance` of the recorded values. With the sim's deterministic seed
//! intact, drift should be at numerical noise levels.
//!
//! See [`Player::replay`] for the entry point and [`ReplayReport`] for the
//! returned diagnostics.

use std::path::Path;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

use crate::agent::protocol::{ClientMessage, ServerMessage};
use crate::teleop::recorder::{read_trace, TraceFrame};
use crate::teleop::TeleopError;

/// Default drift tolerance — radians for revolute joints, metres for prismatic.
pub const DEFAULT_TOLERANCE: f32 = 1e-2;

/// Diagnostics returned by [`Player::replay`].
#[derive(Clone, Debug)]
pub struct ReplayReport {
    /// Frames in the trace.
    pub frames_total: usize,
    /// Frames whose observation matched the recorded one within tolerance.
    pub frames_matched: usize,
    /// Maximum element-wise drift across all replayed joint_positions.
    pub max_drift: f32,
    /// First frame index where drift exceeded tolerance, if any.
    pub diverged_at: Option<usize>,
}

impl ReplayReport {
    pub fn passed(&self) -> bool {
        self.diverged_at.is_none()
    }
}

pub struct Player;

impl Player {
    /// Replay the trace at `path` against the server reachable at `addr`.
    ///
    /// Returns `Ok(ReplayReport)` on a clean run (replay completed without
    /// wire errors); the report's `diverged_at` indicates determinism status.
    /// Returns `Err` only for transport/protocol failures.
    pub async fn replay(
        addr: &str,
        robot_id: usize,
        path: impl AsRef<Path>,
        tolerance: f32,
    ) -> Result<ReplayReport, TeleopError> {
        let frames = read_trace(path.as_ref())?;
        let frames_total = frames.len();
        if frames.is_empty() {
            return Ok(ReplayReport {
                frames_total: 0,
                frames_matched: 0,
                max_drift: 0.0,
                diverged_at: None,
            });
        }

        let connect_fut = tokio_tungstenite::connect_async(addr);
        let (ws_stream, _) = tokio::time::timeout(Duration::from_secs(10), connect_fut)
            .await
            .map_err(|_| TeleopError::Connect(format!("timeout opening {}", addr)))?
            .map_err(|e| TeleopError::Connect(e.to_string()))?;
        let (mut write, mut read) = ws_stream.split();

        send(&mut write, &ClientMessage::Connect { robot_id }).await?;
        match recv(&mut read).await? {
            ServerMessage::Connected { .. } => {}
            other => return Err(TeleopError::Handshake(format!("got {:?}", other))),
        }

        send(&mut write, &ClientMessage::Reset).await?;
        // Reset acks via an Observation; drain it so the server is in a
        // post-reset state before we start stepping.
        let _ = recv(&mut read).await?;

        let mut max_drift = 0.0_f32;
        let mut frames_matched = 0usize;
        let mut diverged_at: Option<usize> = None;

        for (i, frame) in frames.iter().enumerate() {
            send(
                &mut write,
                &ClientMessage::Step {
                    action: frame.act.clone(),
                },
            )
            .await?;
            let observed = match recv(&mut read).await? {
                ServerMessage::Observation { state, .. } => state,
                other => {
                    return Err(TeleopError::Wire(format!(
                        "frame {} expected Observation, got {:?}",
                        i, other
                    )))
                }
            };

            let drift = joint_drift(&observed.joint_positions, &frame.obs.joint_positions);
            if drift > max_drift {
                max_drift = drift;
            }
            if drift <= tolerance {
                frames_matched += 1;
            } else if diverged_at.is_none() {
                diverged_at = Some(i);
            }
            frame_obs(frame); // silence unused
        }

        send(&mut write, &ClientMessage::Close).await?;
        let _ = recv(&mut read).await;

        Ok(ReplayReport {
            frames_total,
            frames_matched,
            max_drift,
            diverged_at,
        })
    }
}

fn frame_obs(_f: &TraceFrame) {}

fn joint_drift(observed: &[f32], recorded: &[f32]) -> f32 {
    let n = observed.len().min(recorded.len());
    let mut max = 0.0_f32;
    for i in 0..n {
        let d = (observed[i] - recorded[i]).abs();
        if d > max {
            max = d;
        }
    }
    // Penalise size mismatch by inflating drift so it gets counted.
    if observed.len() != recorded.len() {
        max = max.max(f32::INFINITY);
    }
    max
}

type WsWrite = futures_util::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    Message,
>;

type WsRead = futures_util::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
>;

async fn send(write: &mut WsWrite, msg: &ClientMessage) -> Result<(), TeleopError> {
    let json = serde_json::to_string(msg).map_err(|e| TeleopError::Wire(e.to_string()))?;
    write
        .send(Message::Text(json.into()))
        .await
        .map_err(|e| TeleopError::Wire(e.to_string()))
}

async fn recv(read: &mut WsRead) -> Result<ServerMessage, TeleopError> {
    loop {
        let next = tokio::time::timeout(Duration::from_secs(15), read.next())
            .await
            .map_err(|_| TeleopError::Wire("read timeout".to_string()))?;
        match next {
            Some(Ok(Message::Text(text))) => {
                return serde_json::from_str::<ServerMessage>(&text)
                    .map_err(|e| TeleopError::Wire(format!("decode: {}", e)))
            }
            Some(Ok(Message::Ping(_) | Message::Pong(_) | Message::Frame(_))) => continue,
            Some(Ok(Message::Binary(_))) => {
                return Err(TeleopError::Wire("unexpected binary frame".to_string()))
            }
            Some(Ok(Message::Close(_))) => {
                return Err(TeleopError::Wire("server closed".to_string()))
            }
            Some(Err(e)) => return Err(TeleopError::Wire(e.to_string())),
            None => return Err(TeleopError::Wire("stream ended".to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_report_passed_iff_no_divergence() {
        let ok = ReplayReport {
            frames_total: 10,
            frames_matched: 10,
            max_drift: 1e-5,
            diverged_at: None,
        };
        assert!(ok.passed());

        let bad = ReplayReport {
            frames_total: 10,
            frames_matched: 4,
            max_drift: 0.2,
            diverged_at: Some(5),
        };
        assert!(!bad.passed());
    }

    #[test]
    fn joint_drift_matches_componentwise_max() {
        let a = [0.0_f32, 0.1, 0.5];
        let b = [0.01_f32, 0.0, 0.55];
        let d = joint_drift(&a, &b);
        assert!((d - 0.1).abs() < 1e-6, "got {}", d);
    }

    #[test]
    fn joint_drift_handles_length_mismatch() {
        let a = [0.0_f32];
        let b = [0.0_f32, 0.0];
        let d = joint_drift(&a, &b);
        assert!(d.is_infinite());
    }

    #[test]
    fn joint_drift_empty_inputs() {
        let a: [f32; 0] = [];
        let b: [f32; 0] = [];
        assert_eq!(joint_drift(&a, &b), 0.0);
    }
}

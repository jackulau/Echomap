//! Client-side tele-op recorder + deterministic playback for goal/011.
//!
//! The crate's agent server lives in `src/agent/`. This module is strictly
//! the *client* side of the bridge: it connects through the existing WS
//! protocol, captures the observation/action pairs it sees, and can replay
//! them later against a freshly reset server.
//!
//! Public surface:
//! * [`recorder::Recorder`] — JSONL writer ([`recorder::TraceFrame`]).
//! * [`playback::Player`] — reads a trace back and verifies determinism.
//! * [`run_session`] — convenience driver: connect → loop → record.

pub mod playback;
pub mod recorder;

use std::path::Path;
use std::time::Duration;

use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

use crate::agent::protocol::{ClientMessage, ServerMessage};
use crate::robot::state::{GymRobotState, RobotAction};

/// Errors surfaced by [`run_session`].
#[derive(Debug)]
pub enum TeleopError {
    /// Could not open the WS connection within the timeout window.
    Connect(String),
    /// Server replied with an unexpected message at handshake time.
    Handshake(String),
    /// Wire-level error talking to the server.
    Wire(String),
    /// Local IO error (recorder file, etc.).
    Io(std::io::Error),
}

impl std::fmt::Display for TeleopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TeleopError::Connect(m) => write!(f, "connect: {}", m),
            TeleopError::Handshake(m) => write!(f, "handshake: {}", m),
            TeleopError::Wire(m) => write!(f, "wire: {}", m),
            TeleopError::Io(e) => write!(f, "io: {}", e),
        }
    }
}

impl std::error::Error for TeleopError {}

impl From<std::io::Error> for TeleopError {
    fn from(e: std::io::Error) -> Self {
        TeleopError::Io(e)
    }
}

impl From<recorder::RecorderError> for TeleopError {
    fn from(e: recorder::RecorderError) -> Self {
        TeleopError::Io(e.into())
    }
}

/// Drive a tele-op session against a running agent server.
///
/// Opens a WS to `addr` (e.g. `"ws://127.0.0.1:9002"`), binds to `robot_id`,
/// then for each of `steps`:
///   1. Calls `agent_fn(step_idx, last_observation)` to pick an action.
///   2. Sends `Step` with that action and waits for the `Observation`.
///   3. Appends the (observation, action) pair to the trace at `path`.
///
/// Returns the number of frames recorded on success.
///
/// `agent_fn` receives the *previous* observation (the one returned for the
/// prior step) so it can be a pure function of state — important for the
/// replay determinism guarantees from D5.
pub async fn run_session<F>(
    addr: &str,
    robot_id: usize,
    steps: u64,
    mut agent_fn: F,
    trace_path: impl AsRef<Path>,
) -> Result<u64, TeleopError>
where
    F: FnMut(u64, &GymRobotState) -> RobotAction,
{
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

    let mut recorder = recorder::Recorder::create(trace_path.as_ref())?;
    let mut frames = 0u64;
    let start = std::time::Instant::now();

    // Seed observation: an Observe call so agent_fn has something to read on step 0.
    let mut last_obs: Option<GymRobotState> =
        Some(bootstrap_observation(&mut write, &mut read).await?);

    for step_idx in 0..steps {
        let obs_ref = last_obs.as_ref().expect("seeded above");
        let action = agent_fn(step_idx, obs_ref);
        let ts = start.elapsed().as_secs_f64();

        send(
            &mut write,
            &ClientMessage::Step {
                action: action.clone(),
            },
        )
        .await?;
        let obs = match recv(&mut read).await? {
            ServerMessage::Observation {
                state, step_count, ..
            } => {
                recorder.record(ts, step_count, &state, &action)?;
                frames += 1;
                state
            }
            other => {
                return Err(TeleopError::Wire(format!(
                    "expected Observation, got {:?}",
                    other
                )))
            }
        };
        last_obs = Some(obs);
    }

    send(&mut write, &ClientMessage::Close).await?;
    // Best-effort drain of the Closed ack; do not fail the session if the
    // server hangs up before we read it.
    let _ = recv(&mut read).await;
    recorder.flush()?;

    Ok(frames)
}

async fn bootstrap_observation(
    write: &mut WsWrite,
    read: &mut WsRead,
) -> Result<GymRobotState, TeleopError> {
    send(write, &ClientMessage::Observe).await?;
    match recv(read).await? {
        ServerMessage::Observation { state, .. } => Ok(state),
        other => Err(TeleopError::Handshake(format!(
            "expected initial Observation, got {:?}",
            other
        ))),
    }
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

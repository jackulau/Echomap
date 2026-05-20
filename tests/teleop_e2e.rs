//! End-to-end smoke test for the headless agent server (D1).
//!
//! Marked `#[ignore]` so it does not run in default `cargo test`. The
//! `scripts/smoke_headless_e2e.sh` harness spawns `echomap_server`, exports
//! `RUST_E2E_PORT`, then runs this test via
//! `cargo test --test teleop_e2e -- --ignored --nocapture`.
//!
//! Protocol:
//!   1. Open WS to ws://127.0.0.1:$RUST_E2E_PORT (default 19002).
//!   2. Send Connect { robot_id: 0 } → expect Connected { action_space, .. }.
//!   3. Drive 100 Step messages with sinusoidal motor velocities sized to
//!      `action_space.num_motors`.
//!   4. Assert every Stepped response decodes as Observation and
//!      `step_count` strictly increases (monotonic).
//!   5. Send Close → expect Closed.

use echomap::agent::protocol::{ClientMessage, ServerMessage};
use echomap::robot::state::RobotAction;
use futures_util::{SinkExt, StreamExt};
use std::time::Duration;
use tokio_tungstenite::tungstenite::Message;

const DEFAULT_PORT: u16 = 19002;
const NUM_STEPS: u64 = 100;

fn port() -> u16 {
    std::env::var("RUST_E2E_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PORT)
}

#[tokio::test]
#[ignore]
async fn smoke_headless_e2e_100_steps() {
    let port = port();
    let url = format!("ws://127.0.0.1:{}", port);

    let connect_fut = tokio_tungstenite::connect_async(&url);
    let (ws_stream, _) = tokio::time::timeout(Duration::from_secs(10), connect_fut)
        .await
        .expect("timed out connecting to headless server")
        .expect("failed to open WS to headless server");
    let (mut write, mut read) = ws_stream.split();

    // Connect → Connected
    send(&mut write, &ClientMessage::Connect { robot_id: 0 }).await;
    let connected = recv(&mut read).await;
    let num_motors = match connected {
        ServerMessage::Connected { action_space, .. } => action_space.num_motors,
        other => panic!("expected Connected, got {:?}", other),
    };
    assert!(
        num_motors > 0,
        "robot must expose at least one motor for the smoke test"
    );

    // 100 sinusoidal steps → each replies Observation with monotonic step_count.
    let mut last_step_count: Option<u64> = None;
    for i in 0..NUM_STEPS {
        let t = i as f32 * 0.05;
        let motor_velocities: Vec<f32> = (0..num_motors)
            .map(|m| (t + m as f32 * 0.3).sin())
            .collect();
        let action = RobotAction {
            motor_velocities,
            gripper_commands: Vec::new(),
            base_velocity: [0.0, 0.0],
        };
        send(&mut write, &ClientMessage::Step { action }).await;

        match recv(&mut read).await {
            ServerMessage::Observation { step_count, .. } => {
                if let Some(prev) = last_step_count {
                    assert!(
                        step_count > prev,
                        "step_count must strictly increase: prev={} new={}",
                        prev,
                        step_count
                    );
                }
                last_step_count = Some(step_count);
            }
            other => panic!("expected Observation at step {}, got {:?}", i, other),
        }
    }
    let final_step = last_step_count.expect("at least one step");
    assert!(
        final_step >= NUM_STEPS,
        "final step_count {} should be >= {}",
        final_step,
        NUM_STEPS
    );

    // Clean shutdown
    send(&mut write, &ClientMessage::Close).await;
    match recv(&mut read).await {
        ServerMessage::Closed => {}
        other => panic!("expected Closed, got {:?}", other),
    }
}

type WsWrite = futures_util::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    Message,
>;

type WsRead = futures_util::stream::SplitStream<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
>;

async fn send(write: &mut WsWrite, msg: &ClientMessage) {
    let json = serde_json::to_string(msg).expect("encode ClientMessage");
    write
        .send(Message::Text(json.into()))
        .await
        .expect("ws send");
}

async fn recv(read: &mut WsRead) -> ServerMessage {
    loop {
        let next = tokio::time::timeout(Duration::from_secs(15), read.next())
            .await
            .expect("timed out waiting for server message");
        match next {
            Some(Ok(Message::Text(text))) => {
                return serde_json::from_str::<ServerMessage>(&text).expect("decode ServerMessage")
            }
            Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
            Some(Ok(Message::Binary(_))) => panic!("unexpected binary frame"),
            Some(Ok(Message::Close(_))) => panic!("server closed unexpectedly"),
            Some(Ok(Message::Frame(_))) => continue,
            Some(Err(e)) => panic!("ws read error: {:?}", e),
            None => panic!("ws stream ended without response"),
        }
    }
}

use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use futures_util::sink::SinkExt;
use futures_util::stream::StreamExt;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;
use tokio_util::sync::CancellationToken;

use crate::agent::bridge::SimBridgeServer;
use crate::agent::protocol::{ClientMessage, ServerMessage};
use crate::agent::session::AgentSession;

/// WebSocket server that accepts connections from agents.
///
/// Uses the same protocol as `TcpAgentServer` but with WebSocket message
/// framing instead of newline-delimited JSON.
pub struct WsAgentServer {
    listener: TcpListener,
    bridge: SimBridgeServer,
    connection_count: Arc<AtomicUsize>,
    max_connections: usize,
}

impl WsAgentServer {
    /// Bind to the given port and prepare to accept connections.
    ///
    /// Use port `0` to let the OS assign an available port (useful in tests).
    pub async fn bind(
        port: u16,
        bridge: SimBridgeServer,
        max_connections: usize,
    ) -> io::Result<Self> {
        let listener = TcpListener::bind(("0.0.0.0", port)).await?;
        Ok(Self {
            listener,
            bridge,
            connection_count: Arc::new(AtomicUsize::new(0)),
            max_connections,
        })
    }

    /// Return the actual port the server is bound to.
    ///
    /// Particularly useful when `bind(0, ...)` was used to get an OS-assigned
    /// port.
    pub fn local_port(&self) -> u16 {
        self.listener
            .local_addr()
            .expect("listener should have a local address")
            .port()
    }

    /// Return the current number of active connections.
    pub fn connection_count(&self) -> usize {
        self.connection_count.load(Ordering::Relaxed)
    }

    /// Run the accept loop until the cancellation token is triggered.
    pub async fn run(&self, cancel: CancellationToken) {
        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    break;
                }
                result = self.listener.accept() => {
                    match result {
                        Ok((stream, _addr)) => {
                            // Check connection limit.
                            let current = self.connection_count.load(Ordering::SeqCst);
                            if current >= self.max_connections {
                                // Reject: upgrade then immediately send error and close.
                                if let Ok(ws_stream) = tokio_tungstenite::accept_async(stream).await {
                                    let (mut write, _) = ws_stream.split();
                                    let err_msg = ServerMessage::Error {
                                        message: "max connections reached".to_string(),
                                    };
                                    if let Ok(json) = serde_json::to_string(&err_msg) {
                                        let _ = write.send(Message::Text(json.into())).await;
                                        let _ = write.close().await;
                                    }
                                }
                                continue;
                            }

                            self.connection_count.fetch_add(1, Ordering::SeqCst);

                            let bridge = self.bridge.clone();
                            let count = self.connection_count.clone();
                            let child_cancel = cancel.clone();

                            tokio::spawn(async move {
                                let _guard = ConnectionGuard(count);
                                Self::handle_connection(stream, bridge, child_cancel).await;
                            });
                        }
                        Err(_) => {
                            // Listener error — break out of the loop.
                            break;
                        }
                    }
                }
            }
        }
    }

    /// Handle a single WebSocket connection.
    async fn handle_connection(
        stream: tokio::net::TcpStream,
        bridge: SimBridgeServer,
        cancel: CancellationToken,
    ) {
        // Upgrade TCP connection to WebSocket.
        let ws_stream = match tokio_tungstenite::accept_async(stream).await {
            Ok(ws) => ws,
            Err(_) => return,
        };

        let (mut write, mut read) = ws_stream.split();
        let mut session = AgentSession::new(bridge);

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    break;
                }
                msg_opt = read.next() => {
                    match msg_opt {
                        Some(Ok(Message::Text(text))) => {
                            match serde_json::from_str::<ClientMessage>(&text) {
                                Ok(client_msg) => {
                                    let response = session.handle_message(client_msg).await;
                                    let json = match serde_json::to_string(&response) {
                                        Ok(j) => j,
                                        Err(_) => break,
                                    };
                                    if write.send(Message::Text(json.into())).await.is_err() {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    let err = ServerMessage::Error {
                                        message: format!("invalid JSON: {}", e),
                                    };
                                    let json = match serde_json::to_string(&err) {
                                        Ok(j) => j,
                                        Err(_) => break,
                                    };
                                    if write.send(Message::Text(json.into())).await.is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                        Some(Ok(Message::Binary(_))) => {
                            let err = ServerMessage::Error {
                                message: "binary messages not supported".to_string(),
                            };
                            let json = match serde_json::to_string(&err) {
                                Ok(j) => j,
                                Err(_) => break,
                            };
                            if write.send(Message::Text(json.into())).await.is_err() {
                                break;
                            }
                        }
                        Some(Ok(Message::Close(_))) => {
                            // Client sent a close frame — graceful disconnect.
                            break;
                        }
                        Some(Ok(Message::Ping(_) | Message::Pong(_))) => {
                            // tokio-tungstenite handles ping/pong automatically
                            // at the protocol level. Nothing to do here.
                        }
                        Some(Ok(Message::Frame(_))) => {
                            // Raw frame — ignore.
                        }
                        Some(Err(_)) => {
                            // WebSocket error — drop connection.
                            break;
                        }
                        None => {
                            // Stream ended.
                            break;
                        }
                    }
                }
            }
        }

        // Ensure the session is cleaned up (releases robot assignment).
        let _ = session.handle_message(ClientMessage::Close).await;
    }
}

/// RAII guard that decrements the connection count when dropped.
struct ConnectionGuard(Arc<AtomicUsize>);

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bridge::create_bridge;
    use crate::agent::protocol::{ClientMessage, ServerMessage};
    use crate::robot::definition::RobotDefinition;
    use crate::robot::state::RobotAction;
    use crate::robot::RobotManager;
    use glam::Mat4;

    /// Helper: create a RobotManager with `n` simple_arm(2) robots, a bridge
    /// pair, and spawn a background task that continuously processes bridge
    /// commands. Returns the server-side bridge and a task handle for cleanup.
    fn setup_bridge(n: usize) -> (SimBridgeServer, tokio::task::JoinHandle<()>) {
        let mut manager = RobotManager::new();
        for _ in 0..n {
            let def = RobotDefinition::simple_arm(2);
            manager.add_robot(def, Mat4::IDENTITY);
        }

        let (server, mut client) = create_bridge();

        let handle = tokio::spawn(async move {
            loop {
                client.process_pending(&mut manager, &[]);
                tokio::task::yield_now().await;
            }
        });

        (server, handle)
    }

    /// Helper: start a WsAgentServer on port 0, return the server port, a
    /// cancellation token, and handles to clean up.
    async fn start_server(
        bridge: SimBridgeServer,
        max_connections: usize,
    ) -> (
        u16,
        CancellationToken,
        Arc<AtomicUsize>,
        tokio::task::JoinHandle<()>,
    ) {
        let server = WsAgentServer::bind(0, bridge, max_connections)
            .await
            .expect("bind should succeed");
        let port = server.local_port();
        let conn_count = server.connection_count.clone();
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let server_handle = tokio::spawn(async move {
            server.run(cancel_clone).await;
        });
        (port, cancel, conn_count, server_handle)
    }

    /// Helper: connect a WebSocket client to the given port and return the
    /// split read/write halves.
    async fn ws_connect(
        port: u16,
    ) -> (
        futures_util::stream::SplitSink<
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
            Message,
        >,
        futures_util::stream::SplitStream<
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
        >,
    ) {
        let url = format!("ws://127.0.0.1:{}", port);
        let (ws_stream, _) = tokio_tungstenite::connect_async(&url)
            .await
            .expect("should connect to WS server");
        ws_stream.split()
    }

    /// Helper: send a ClientMessage and receive a ServerMessage over WebSocket.
    async fn ws_send_recv(
        write: &mut futures_util::stream::SplitSink<
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
            Message,
        >,
        read: &mut futures_util::stream::SplitStream<
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
        >,
        msg: &ClientMessage,
    ) -> ServerMessage {
        let json = serde_json::to_string(msg).unwrap();
        write
            .send(Message::Text(json.into()))
            .await
            .expect("send should succeed");

        loop {
            match read.next().await {
                Some(Ok(Message::Text(text))) => {
                    return serde_json::from_str::<ServerMessage>(&text)
                        .expect("should parse server message");
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => {
                    // Skip protocol-level ping/pong frames.
                    continue;
                }
                other => panic!("Expected Text message, got {:?}", other),
            }
        }
    }

    #[tokio::test]
    async fn test_ws_server_binds() {
        let (bridge, bg) = setup_bridge(1);
        let server = WsAgentServer::bind(0, bridge, 16)
            .await
            .expect("bind to port 0 should succeed");
        let port = server.local_port();
        assert!(port > 0, "OS-assigned port should be > 0");
        bg.abort();
    }

    #[tokio::test]
    async fn test_ws_connect_and_message() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, _cc, _sh) = start_server(bridge, 16).await;

        let (mut write, mut read) = ws_connect(port).await;

        // Send Connect
        let response = ws_send_recv(
            &mut write,
            &mut read,
            &ClientMessage::Connect { robot_id: 0 },
        )
        .await;

        match response {
            ServerMessage::Connected {
                observation_space,
                action_space,
            } => {
                assert_eq!(observation_space.num_joint_positions, 2);
                assert_eq!(action_space.num_motors, 2);
            }
            other => panic!("Expected Connected, got {:?}", other),
        }

        cancel.cancel();
        bg.abort();
    }

    #[tokio::test]
    async fn test_ws_step_roundtrip() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, _cc, _sh) = start_server(bridge, 16).await;

        let (mut write, mut read) = ws_connect(port).await;

        // Connect
        let resp = ws_send_recv(
            &mut write,
            &mut read,
            &ClientMessage::Connect { robot_id: 0 },
        )
        .await;
        assert!(
            matches!(resp, ServerMessage::Connected { .. }),
            "should get Connected: {:?}",
            resp
        );

        // Reset
        let resp = ws_send_recv(&mut write, &mut read, &ClientMessage::Reset).await;
        match &resp {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(*step_count, 0, "reset should yield step_count=0");
            }
            other => panic!("Expected Observation after reset, got {:?}", other),
        }

        // Step
        let action = RobotAction {
            motor_velocities: vec![1.0, -0.5],
            gripper_commands: vec![],
        };
        let resp = ws_send_recv(&mut write, &mut read, &ClientMessage::Step { action }).await;
        match &resp {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(*step_count, 1, "first step should yield step_count=1");
            }
            other => panic!("Expected Observation after step, got {:?}", other),
        }

        // Observe
        let resp = ws_send_recv(&mut write, &mut read, &ClientMessage::Observe).await;
        match &resp {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(*step_count, 1, "observe should show step_count=1");
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        // Close
        let resp = ws_send_recv(&mut write, &mut read, &ClientMessage::Close).await;
        assert!(
            matches!(resp, ServerMessage::Closed),
            "Expected Closed, got {:?}",
            resp
        );

        cancel.cancel();
        bg.abort();
    }

    #[tokio::test]
    async fn test_ws_close_frame() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, conn_count, _sh) = start_server(bridge, 16).await;

        let (mut write, _read) = ws_connect(port).await;

        // Give the server a moment to register the connection.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(
            conn_count.load(Ordering::SeqCst),
            1,
            "should have 1 connection"
        );

        // Send a close frame.
        write
            .send(Message::Close(None))
            .await
            .expect("sending close frame should succeed");

        // Wait for the server to notice the disconnect.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert_eq!(
            conn_count.load(Ordering::SeqCst),
            0,
            "connection_count should drop to 0 after close frame"
        );

        cancel.cancel();
        bg.abort();
    }

    #[tokio::test]
    async fn test_ws_binary_message_rejected() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, _cc, _sh) = start_server(bridge, 16).await;

        let (mut write, mut read) = ws_connect(port).await;

        // Send a binary message.
        write
            .send(Message::Binary(vec![0x00, 0x01, 0x02].into()))
            .await
            .expect("send binary should succeed");

        // Read the error response.
        loop {
            match read.next().await {
                Some(Ok(Message::Text(text))) => {
                    let msg: ServerMessage =
                        serde_json::from_str(&text).expect("should parse error response");
                    match msg {
                        ServerMessage::Error { message } => {
                            assert!(
                                message.contains("binary messages not supported"),
                                "error should mention binary not supported, got: {}",
                                message
                            );
                        }
                        other => panic!("Expected Error, got {:?}", other),
                    }
                    break;
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                other => panic!("Expected Text error message, got {:?}", other),
            }
        }

        cancel.cancel();
        bg.abort();
    }

    #[tokio::test]
    async fn test_ws_multiple_connections() {
        let (bridge, bg) = setup_bridge(2);
        let (port, cancel, _cc, _sh) = start_server(bridge, 16).await;

        // Client 1: connect to robot 0
        let (mut w1, mut r1) = ws_connect(port).await;
        let resp = ws_send_recv(&mut w1, &mut r1, &ClientMessage::Connect { robot_id: 0 }).await;
        assert!(
            matches!(resp, ServerMessage::Connected { .. }),
            "Client 1 should connect to robot 0"
        );

        // Client 2: connect to robot 1
        let (mut w2, mut r2) = ws_connect(port).await;
        let resp = ws_send_recv(&mut w2, &mut r2, &ClientMessage::Connect { robot_id: 1 }).await;
        assert!(
            matches!(resp, ServerMessage::Connected { .. }),
            "Client 2 should connect to robot 1"
        );

        // Both can step independently
        let action = RobotAction {
            motor_velocities: vec![1.0, -0.5],
            gripper_commands: vec![],
        };

        let resp1 = ws_send_recv(
            &mut w1,
            &mut r1,
            &ClientMessage::Step {
                action: action.clone(),
            },
        )
        .await;
        let resp2 = ws_send_recv(&mut w2, &mut r2, &ClientMessage::Step { action }).await;

        assert!(
            matches!(resp1, ServerMessage::Observation { step_count: 1, .. }),
            "Client 1 step should yield step_count=1, got {:?}",
            resp1
        );
        assert!(
            matches!(resp2, ServerMessage::Observation { step_count: 1, .. }),
            "Client 2 step should yield step_count=1, got {:?}",
            resp2
        );

        cancel.cancel();
        bg.abort();
    }
}

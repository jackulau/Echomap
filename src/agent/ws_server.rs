use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures_util::sink::SinkExt;
use futures_util::stream::StreamExt;
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message;
use tokio_util::sync::CancellationToken;

use crate::agent::bridge::SimBridgeServer;
use crate::agent::protocol::{ClientMessage, ServerMessage};
use crate::agent::session::AgentSession;

/// Per-connection timing limits. A stalled agent is dropped after `read_timeout`
/// with no inbound message; the server proactively sends a keepalive ping every
/// `heartbeat_interval` so that an inert TCP connection is noticed quickly.
#[derive(Clone, Copy, Debug)]
pub struct WsServerConfig {
    pub read_timeout: Duration,
    pub heartbeat_interval: Duration,
}

impl Default for WsServerConfig {
    fn default() -> Self {
        Self {
            read_timeout: Duration::from_secs(30),
            heartbeat_interval: Duration::from_secs(10),
        }
    }
}

/// WebSocket server that accepts connections from agents.
///
/// Uses the same protocol as `TcpAgentServer` but with WebSocket message
/// framing instead of newline-delimited JSON.
pub struct WsAgentServer {
    listener: TcpListener,
    bridge: SimBridgeServer,
    connection_count: Arc<AtomicUsize>,
    max_connections: usize,
    config: WsServerConfig,
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
        Self::bind_with_config(port, bridge, max_connections, WsServerConfig::default()).await
    }

    /// Bind with explicit timeout / heartbeat configuration.
    pub async fn bind_with_config(
        port: u16,
        bridge: SimBridgeServer,
        max_connections: usize,
        config: WsServerConfig,
    ) -> io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", port)).await?;
        Ok(Self {
            listener,
            bridge,
            connection_count: Arc::new(AtomicUsize::new(0)),
            max_connections,
            config,
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
                                    let err_msg = ServerMessage::error("max connections reached");
                                    if let Ok(json) = crate::agent::protocol::encode_for_wire(&err_msg) {
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
                            let conn_config = self.config;

                            tokio::spawn(async move {
                                let _guard = ConnectionGuard(count);
                                Self::handle_connection(stream, bridge, child_cancel, conn_config).await;
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
    ///
    /// Drops the connection if no inbound message arrives within
    /// `config.read_timeout`. Periodically sends a keepalive ping every
    /// `config.heartbeat_interval` so half-open TCP sockets surface quickly.
    /// Always runs `ClientMessage::Close` on the session at exit to free the
    /// robot assignment even if the agent dropped without a clean close frame.
    async fn handle_connection(
        stream: tokio::net::TcpStream,
        bridge: SimBridgeServer,
        cancel: CancellationToken,
        config: WsServerConfig,
    ) {
        // Upgrade TCP connection to WebSocket with message size limits.
        let mut ws_config = tokio_tungstenite::tungstenite::protocol::WebSocketConfig::default();
        ws_config.max_message_size = Some(65_536);
        ws_config.max_frame_size = Some(65_536);
        let ws_stream =
            match tokio_tungstenite::accept_async_with_config(stream, Some(ws_config)).await {
                Ok(ws) => ws,
                Err(_) => return,
            };

        let (mut write, mut read) = ws_stream.split();
        let mut session = AgentSession::new(bridge);

        let mut heartbeat = tokio::time::interval(config.heartbeat_interval);
        // First tick fires immediately; skip it so we don't ping the moment we connect.
        heartbeat.tick().await;

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    break;
                }
                _ = heartbeat.tick() => {
                    // Keepalive: a peer with a half-open socket will fail on send.
                    if write.send(Message::Ping(Vec::new().into())).await.is_err() {
                        break;
                    }
                    continue;
                }
                read_result = tokio::time::timeout(config.read_timeout, read.next()) => {
                    let msg_opt = match read_result {
                        Ok(opt) => opt,
                        Err(_) => {
                            // Read timed out — agent went silent. Tell them, then drop.
                            let err = ServerMessage::error(format!(
                                "read timeout after {:.0}s",
                                config.read_timeout.as_secs_f32()
                            ));
                            if let Ok(json) = crate::agent::protocol::encode_for_wire(&err) {
                                let _ = write.send(Message::Text(json.into())).await;
                            }
                            let _ = write.close().await;
                            break;
                        }
                    };
                    match msg_opt {
                        Some(Ok(Message::Text(text))) => {
                            match serde_json::from_str::<ClientMessage>(&text) {
                                Ok(client_msg) => {
                                    let response = session.handle_message(client_msg).await;
                                    let json = match crate::agent::protocol::encode_for_wire(&response) {
                                        Ok(j) => j,
                                        Err(_) => break,
                                    };
                                    if write.send(Message::Text(json.into())).await.is_err() {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    let err = ServerMessage::error_with_echo(
                                        format!("invalid JSON: {}", e),
                                        &text,
                                    );
                                    let json = match crate::agent::protocol::encode_for_wire(&err) {
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
                            let err = ServerMessage::error("binary messages not supported");
                            let json = match crate::agent::protocol::encode_for_wire(&err) {
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
            base_velocity: [0.0, 0.0],
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
                        ServerMessage::Error { message, .. } => {
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
            base_velocity: [0.0, 0.0],
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

    // ---- Edge case tests ----

    #[tokio::test]
    async fn test_ws_max_connections_rejected() {
        let (bridge, bg) = setup_bridge(1);
        // max_connections = 1
        let (port, cancel, conn_count, _sh) = start_server(bridge, 1).await;

        // First connection should succeed
        let (mut w1, mut r1) = ws_connect(port).await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(
            conn_count.load(Ordering::SeqCst),
            1,
            "should have 1 connection"
        );

        // Second connection: server should send error then close
        let url = format!("ws://127.0.0.1:{}", port);
        let result = tokio_tungstenite::connect_async(&url).await;
        match result {
            Ok((ws_stream, _)) => {
                let (_write, mut read) = ws_stream.split();
                // Read the error message
                loop {
                    match read.next().await {
                        Some(Ok(Message::Text(text))) => {
                            let msg: ServerMessage =
                                serde_json::from_str(&text).expect("should parse error message");
                            match msg {
                                ServerMessage::Error { message, .. } => {
                                    assert!(
                                        message.contains("max connections"),
                                        "should mention max connections, got: {}",
                                        message
                                    );
                                }
                                other => panic!("Expected Error, got {:?}", other),
                            }
                            break;
                        }
                        Some(Ok(Message::Close(_))) => break, // Also acceptable
                        Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                        _other => {
                            // Connection might just close -- acceptable
                            break;
                        }
                    }
                }
            }
            Err(_) => {
                // Connection refused at WS level is also acceptable
            }
        }

        // First connection should still work
        let resp = ws_send_recv(&mut w1, &mut r1, &ClientMessage::Connect { robot_id: 0 }).await;
        assert!(
            matches!(resp, ServerMessage::Connected { .. }),
            "first connection should still work, got {:?}",
            resp
        );

        cancel.cancel();
        bg.abort();
    }

    #[tokio::test]
    async fn test_ws_malformed_json_text() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, _cc, _sh) = start_server(bridge, 16).await;

        let (mut write, mut read) = ws_connect(port).await;

        // Send malformed JSON as text
        write
            .send(Message::Text("{bad json".into()))
            .await
            .expect("send should succeed");

        // Read the error response
        loop {
            match read.next().await {
                Some(Ok(Message::Text(text))) => {
                    let msg: ServerMessage =
                        serde_json::from_str(&text).expect("should parse error");
                    match msg {
                        ServerMessage::Error { message, .. } => {
                            assert!(
                                message.contains("invalid JSON"),
                                "malformed text should produce invalid JSON error, got: {}",
                                message
                            );
                        }
                        other => panic!("Expected Error, got {:?}", other),
                    }
                    break;
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                other => panic!("Expected Text error, got {:?}", other),
            }
        }

        // Connection should still work after malformed JSON
        let resp = ws_send_recv(
            &mut write,
            &mut read,
            &ClientMessage::Connect { robot_id: 0 },
        )
        .await;
        assert!(
            matches!(resp, ServerMessage::Connected { .. }),
            "connection should survive malformed JSON, got {:?}",
            resp
        );

        cancel.cancel();
        bg.abort();
    }

    #[tokio::test]
    async fn test_ws_connection_cleanup_on_drop() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, conn_count, _sh) = start_server(bridge, 16).await;

        // Connect
        {
            let (_write, _read) = ws_connect(port).await;
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            assert_eq!(
                conn_count.load(Ordering::SeqCst),
                1,
                "should have 1 connection"
            );
            // Drop write/read -- closes the connection
        }

        // Wait for cleanup
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert_eq!(
            conn_count.load(Ordering::SeqCst),
            0,
            "connection count should drop to 0 after client disconnects"
        );

        cancel.cancel();
        bg.abort();
    }

    #[tokio::test]
    async fn test_ws_unknown_message_type() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, _cc, _sh) = start_server(bridge, 16).await;

        let (mut write, mut read) = ws_connect(port).await;

        // Send valid JSON but unknown type
        write
            .send(Message::Text(r#"{"type":"teleport","x":10}"#.into()))
            .await
            .expect("send should succeed");

        loop {
            match read.next().await {
                Some(Ok(Message::Text(text))) => {
                    let msg: ServerMessage =
                        serde_json::from_str(&text).expect("should parse error");
                    assert!(
                        matches!(msg, ServerMessage::Error { .. }),
                        "unknown type should produce error, got {:?}",
                        msg
                    );
                    break;
                }
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                other => panic!("Expected error Text, got {:?}", other),
            }
        }

        cancel.cancel();
        bg.abort();
    }

    #[tokio::test]
    async fn test_ws_step_before_connect() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, _cc, _sh) = start_server(bridge, 16).await;

        let (mut write, mut read) = ws_connect(port).await;

        // Step without connecting first
        let action = RobotAction {
            motor_velocities: vec![1.0],
            gripper_commands: vec![],
            base_velocity: [0.0, 0.0],
        };
        let resp = ws_send_recv(&mut write, &mut read, &ClientMessage::Step { action }).await;
        match resp {
            ServerMessage::Error { message, .. } => {
                assert!(
                    message.contains("not connected"),
                    "step before connect should say not connected, got: {}",
                    message
                );
            }
            other => panic!("Expected Error, got {:?}", other),
        }

        cancel.cancel();
        bg.abort();
    }

    /// A connected client that sends nothing must be closed by the server once
    /// the read timeout elapses, with an explicit "read timeout" error
    /// preceding the close frame.
    #[tokio::test]
    async fn test_ws_read_timeout_drops_silent_client() {
        let (bridge, bg) = setup_bridge(1);
        let cfg = WsServerConfig {
            read_timeout: Duration::from_millis(200),
            heartbeat_interval: Duration::from_secs(60),
        };
        let server = WsAgentServer::bind_with_config(0, bridge, 16, cfg)
            .await
            .expect("bind to port 0 should succeed");
        let port = server.local_port();
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let server_handle = tokio::spawn(async move {
            server.run(cancel_clone).await;
        });

        let (mut _write, mut read) = ws_connect(port).await;

        // Read messages until we see the timeout error (skipping any pings/pongs).
        let mut saw_timeout_error = false;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        while tokio::time::Instant::now() < deadline && !saw_timeout_error {
            match tokio::time::timeout(Duration::from_millis(500), read.next()).await {
                Ok(Some(Ok(Message::Text(text)))) => {
                    let msg: ServerMessage =
                        serde_json::from_str(&text).expect("server should send valid JSON");
                    if let ServerMessage::Error { message, echo: None } = msg {
                        if message.contains("timeout") {
                            saw_timeout_error = true;
                        }
                    }
                }
                Ok(Some(Ok(Message::Close(_))) | None) => {
                    break;
                }
                Ok(Some(Err(_))) => {
                    break;
                }
                _ => {}
            }
        }

        assert!(
            saw_timeout_error,
            "server should send read-timeout error before closing silent client"
        );

        cancel.cancel();
        let _ = server_handle.await;
        bg.abort();
    }

    /// The server must proactively send WebSocket ping frames at the
    /// heartbeat interval; a client that pauses but stays connected should
    /// observe at least one ping within the expected window.
    #[tokio::test]
    async fn test_ws_heartbeat_ping_keepalive() {
        let (bridge, bg) = setup_bridge(1);
        let cfg = WsServerConfig {
            read_timeout: Duration::from_secs(60),
            heartbeat_interval: Duration::from_millis(100),
        };
        let server = WsAgentServer::bind_with_config(0, bridge, 16, cfg)
            .await
            .expect("bind to port 0 should succeed");
        let port = server.local_port();
        let cancel = CancellationToken::new();
        let cancel_clone = cancel.clone();
        let server_handle = tokio::spawn(async move {
            server.run(cancel_clone).await;
        });

        let (mut _write, mut read) = ws_connect(port).await;

        // Look for at least one Ping frame within ~1s.
        let mut saw_ping = false;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
        while tokio::time::Instant::now() < deadline && !saw_ping {
            match tokio::time::timeout(Duration::from_millis(500), read.next()).await {
                Ok(Some(Ok(Message::Ping(_)))) => {
                    saw_ping = true;
                }
                Ok(Some(Ok(_))) => {}
                Ok(Some(Err(_))) | Ok(None) | Err(_) => break,
            }
        }

        assert!(
            saw_ping,
            "server should send ping frame within heartbeat interval"
        );

        cancel.cancel();
        let _ = server_handle.await;
        bg.abort();
    }
}

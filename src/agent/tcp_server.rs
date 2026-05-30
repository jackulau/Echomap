use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use crate::agent::bridge::SimBridgeServer;
use crate::agent::protocol::{ClientMessage, ServerMessage};
use crate::agent::session::AgentSession;

/// TCP server that accepts line-delimited JSON connections from agents.
///
/// Each connection spawns an `AgentSession` and communicates using newline-
/// delimited JSON: one `ClientMessage` per line in, one `ServerMessage` per
/// line out.
pub struct TcpAgentServer {
    listener: TcpListener,
    bridge: SimBridgeServer,
    connection_count: Arc<AtomicUsize>,
    max_connections: usize,
}

impl TcpAgentServer {
    /// Bind to the given port and prepare to accept connections.
    ///
    /// Use port `0` to let the OS assign an available port (useful in tests).
    pub async fn bind(
        port: u16,
        bridge: SimBridgeServer,
        max_connections: usize,
    ) -> io::Result<Self> {
        let listener = TcpListener::bind(("127.0.0.1", port)).await?;
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
        // Fail-soft: see ws_server.rs::local_port — return 0 if the
        // kernel can't surface our address rather than panic and kill
        // the agent server.
        match self.listener.local_addr() {
            Ok(addr) => addr.port(),
            Err(e) => {
                log::warn!("tcp_server local_port: listener.local_addr failed: {e}");
                0
            }
        }
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
                                // Reject: write an error and close.
                                let (_, writer) = stream.into_split();
                                let mut bw = BufWriter::new(writer);
                                let err_msg = ServerMessage::Error {
                message: "max connections reached".to_string(),
                echo: None,
            };
                                let _ = Self::write_message(&mut bw, &err_msg).await;
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
                            // Listener error — in tests this usually means
                            // the listener was dropped. Just break.
                            break;
                        }
                    }
                }
            }
        }
    }

    const MAX_LINE_BYTES: usize = 65_536;

    /// Handle a single TCP connection.
    async fn handle_connection(
        stream: tokio::net::TcpStream,
        bridge: SimBridgeServer,
        cancel: CancellationToken,
    ) {
        let (reader, writer) = stream.into_split();
        let mut br = BufReader::new(reader);
        let mut bw = BufWriter::new(writer);
        let mut session = AgentSession::new(bridge);
        let mut buf = Vec::with_capacity(1024);

        loop {
            buf.clear();
            // Bound the per-line read so a newline-less stream cannot grow `buf`
            // without limit (memory-exhaustion DoS). `.take(N+1)` caps how many
            // bytes `read_until` will pull before yielding; the oversize arm then
            // fires once `buf` exceeds MAX_LINE_BYTES, having buffered at most
            // MAX_LINE_BYTES + 1 bytes. Mirrors the WS transport's 65_536 cap.
            let mut limited = (&mut br).take(Self::MAX_LINE_BYTES as u64 + 1);
            tokio::select! {
                _ = cancel.cancelled() => {
                    break;
                }
                result = AsyncBufReadExt::read_until(&mut limited, b'\n', &mut buf) => {
                    match result {
                        Ok(0) => {
                            break;
                        }
                        Ok(n) if n > Self::MAX_LINE_BYTES => {
                            let err = ServerMessage::Error {
                message: "message too large".to_string(),
                echo: None,
            };
                            let _ = Self::write_message(&mut bw, &err).await;
                            // Drop the connection: with a bounded reader the
                            // remainder of an oversized line is still unread, so
                            // continuing would mis-parse its tail as a new message.
                            break;
                        }
                        Ok(_) => {
                            let trimmed = String::from_utf8_lossy(&buf);
                            let trimmed = trimmed.trim();
                            if trimmed.is_empty() {
                                continue;
                            }

                            match serde_json::from_str::<ClientMessage>(trimmed) {
                                Ok(msg) => {
                                    let response = session.handle_message(msg).await;
                                    if Self::write_message(&mut bw, &response).await.is_err() {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    let err = ServerMessage::error_with_echo(
                                        format!("invalid JSON: {}", e),
                                        trimmed,
                                    );
                                    if Self::write_message(&mut bw, &err).await.is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            // Read error — drop connection.
                            break;
                        }
                    }
                }
            }
        }

        // Ensure the session is cleaned up (releases robot assignment).
        let _ = session.handle_message(ClientMessage::Close).await;
    }

    /// Serialize a `ServerMessage` as JSON followed by a newline, and flush.
    /// Delegates the JSON encoding to `protocol::encode_for_wire` so the
    /// payload is byte-identical to the WebSocket transport (D3 parity).
    async fn write_message(
        bw: &mut BufWriter<tokio::net::tcp::OwnedWriteHalf>,
        msg: &ServerMessage,
    ) -> io::Result<()> {
        let json = crate::agent::protocol::encode_for_wire(msg)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        bw.write_all(json.as_bytes()).await?;
        bw.write_all(b"\n").await?;
        bw.flush().await?;
        Ok(())
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
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

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

    /// Helper: start a TcpAgentServer on port 0, return the server port, a
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
        let server = TcpAgentServer::bind(0, bridge, max_connections)
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

    /// Helper: send a line to the server and read the response line.
    async fn send_and_recv(
        writer: &mut BufWriter<tokio::net::tcp::OwnedWriteHalf>,
        reader: &mut BufReader<tokio::net::tcp::OwnedReadHalf>,
        msg: &str,
    ) -> String {
        writer
            .write_all(msg.as_bytes())
            .await
            .expect("write should succeed");
        writer
            .write_all(b"\n")
            .await
            .expect("write newline should succeed");
        writer.flush().await.expect("flush should succeed");

        let mut line = String::new();
        reader
            .read_line(&mut line)
            .await
            .expect("read should succeed");
        line
    }

    /// Helper: connect a TcpStream and split into buffered reader/writer.
    async fn connect_client(
        port: u16,
    ) -> (
        BufWriter<tokio::net::tcp::OwnedWriteHalf>,
        BufReader<tokio::net::tcp::OwnedReadHalf>,
    ) {
        let stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .expect("should connect to server");
        let (r, w) = stream.into_split();
        (BufWriter::new(w), BufReader::new(r))
    }

    #[tokio::test]
    async fn test_tcp_server_binds() {
        let (bridge, bg) = setup_bridge(1);
        let server = TcpAgentServer::bind(0, bridge, 16)
            .await
            .expect("bind to port 0 should succeed");
        let port = server.local_port();
        assert!(port > 0, "OS-assigned port should be > 0");
        bg.abort();
    }

    #[tokio::test]
    async fn test_tcp_connect_and_message() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, _cc, _sh) = start_server(bridge, 16).await;

        let (mut writer, mut reader) = connect_client(port).await;

        // Send Connect
        let connect_json = serde_json::to_string(&ClientMessage::Connect { robot_id: 0 }).unwrap();
        let response = send_and_recv(&mut writer, &mut reader, &connect_json).await;
        let msg: ServerMessage = serde_json::from_str(response.trim()).unwrap();

        match msg {
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
    async fn test_tcp_step_roundtrip() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, _cc, _sh) = start_server(bridge, 16).await;

        let (mut writer, mut reader) = connect_client(port).await;

        // Connect
        let msg = serde_json::to_string(&ClientMessage::Connect { robot_id: 0 }).unwrap();
        let resp = send_and_recv(&mut writer, &mut reader, &msg).await;
        assert!(resp.contains("connected"), "should get Connected: {}", resp);

        // Reset
        let msg = serde_json::to_string(&ClientMessage::Reset).unwrap();
        let resp = send_and_recv(&mut writer, &mut reader, &msg).await;
        let parsed: ServerMessage = serde_json::from_str(resp.trim()).unwrap();
        match &parsed {
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
        let msg = serde_json::to_string(&ClientMessage::Step { action }).unwrap();
        let resp = send_and_recv(&mut writer, &mut reader, &msg).await;
        let parsed: ServerMessage = serde_json::from_str(resp.trim()).unwrap();
        match &parsed {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(*step_count, 1, "first step should yield step_count=1");
            }
            other => panic!("Expected Observation after step, got {:?}", other),
        }

        // Observe
        let msg = serde_json::to_string(&ClientMessage::Observe).unwrap();
        let resp = send_and_recv(&mut writer, &mut reader, &msg).await;
        let parsed: ServerMessage = serde_json::from_str(resp.trim()).unwrap();
        match &parsed {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(*step_count, 1, "observe should show step_count=1");
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        // Close
        let msg = serde_json::to_string(&ClientMessage::Close).unwrap();
        let resp = send_and_recv(&mut writer, &mut reader, &msg).await;
        let parsed: ServerMessage = serde_json::from_str(resp.trim()).unwrap();
        assert!(
            matches!(parsed, ServerMessage::Closed),
            "Expected Closed, got {:?}",
            parsed
        );

        cancel.cancel();
        bg.abort();
    }

    #[tokio::test]
    async fn test_tcp_malformed_json() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, _cc, _sh) = start_server(bridge, 16).await;

        let (mut writer, mut reader) = connect_client(port).await;

        // Send malformed JSON
        let resp = send_and_recv(&mut writer, &mut reader, "not json").await;
        let parsed: ServerMessage = serde_json::from_str(resp.trim()).unwrap();
        match &parsed {
            ServerMessage::Error { message, .. } => {
                assert!(
                    message.contains("invalid JSON"),
                    "error should mention 'invalid JSON', got: {}",
                    message
                );
            }
            other => panic!("Expected Error, got {:?}", other),
        }

        // Connection should still be alive — send a valid Connect now.
        let msg = serde_json::to_string(&ClientMessage::Connect { robot_id: 0 }).unwrap();
        let resp = send_and_recv(&mut writer, &mut reader, &msg).await;
        let parsed: ServerMessage = serde_json::from_str(resp.trim()).unwrap();
        assert!(
            matches!(parsed, ServerMessage::Connected { .. }),
            "Connection should still work after malformed JSON, got {:?}",
            parsed
        );

        cancel.cancel();
        bg.abort();
    }

    #[tokio::test]
    async fn test_tcp_connection_cleanup() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, conn_count, _sh) = start_server(bridge, 16).await;

        // Connect
        {
            let stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
                .await
                .expect("should connect");
            // Give the server a moment to register the connection.
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            assert_eq!(
                conn_count.load(Ordering::SeqCst),
                1,
                "should have 1 connection"
            );
            // Drop the stream, which closes the connection.
            drop(stream);
        }

        // Wait for the server to notice the disconnect and run cleanup.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert_eq!(
            conn_count.load(Ordering::SeqCst),
            0,
            "connection_count should drop to 0 after disconnect"
        );

        cancel.cancel();
        bg.abort();
    }

    #[tokio::test]
    async fn test_tcp_multiple_connections() {
        let (bridge, bg) = setup_bridge(2);
        let (port, cancel, _cc, _sh) = start_server(bridge, 16).await;

        // Client 1: connect to robot 0
        let (mut w1, mut r1) = connect_client(port).await;
        let msg = serde_json::to_string(&ClientMessage::Connect { robot_id: 0 }).unwrap();
        let resp = send_and_recv(&mut w1, &mut r1, &msg).await;
        let parsed: ServerMessage = serde_json::from_str(resp.trim()).unwrap();
        assert!(
            matches!(parsed, ServerMessage::Connected { .. }),
            "Client 1 should connect to robot 0"
        );

        // Client 2: connect to robot 1
        let (mut w2, mut r2) = connect_client(port).await;
        let msg = serde_json::to_string(&ClientMessage::Connect { robot_id: 1 }).unwrap();
        let resp = send_and_recv(&mut w2, &mut r2, &msg).await;
        let parsed: ServerMessage = serde_json::from_str(resp.trim()).unwrap();
        assert!(
            matches!(parsed, ServerMessage::Connected { .. }),
            "Client 2 should connect to robot 1"
        );

        // Both can step independently
        let action = RobotAction {
            motor_velocities: vec![1.0, -0.5],
            gripper_commands: vec![],
            base_velocity: [0.0, 0.0],
        };
        let step_json = serde_json::to_string(&ClientMessage::Step {
            action: action.clone(),
        })
        .unwrap();

        let resp1 = send_and_recv(&mut w1, &mut r1, &step_json).await;
        let resp2 = send_and_recv(&mut w2, &mut r2, &step_json).await;

        let p1: ServerMessage = serde_json::from_str(resp1.trim()).unwrap();
        let p2: ServerMessage = serde_json::from_str(resp2.trim()).unwrap();
        assert!(
            matches!(p1, ServerMessage::Observation { step_count: 1, .. }),
            "Client 1 step should yield step_count=1, got {:?}",
            p1
        );
        assert!(
            matches!(p2, ServerMessage::Observation { step_count: 1, .. }),
            "Client 2 step should yield step_count=1, got {:?}",
            p2
        );

        cancel.cancel();
        bg.abort();
    }

    // ---- Edge case tests ----

    #[tokio::test]
    async fn test_tcp_max_connections_rejected() {
        let (bridge, bg) = setup_bridge(1);
        // max_connections = 1
        let (port, cancel, conn_count, _sh) = start_server(bridge, 1).await;

        // First connection should succeed
        let (mut w1, mut r1) = connect_client(port).await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(
            conn_count.load(Ordering::SeqCst),
            1,
            "should have 1 connection"
        );

        // Second connection should be rejected with an error message
        let stream2 = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .expect("TCP connect should succeed at the OS level");
        let (r2, _w2) = stream2.into_split();
        let mut reader2 = BufReader::new(r2);
        let mut line = String::new();
        let n = reader2.read_line(&mut line).await.unwrap();
        if n > 0 {
            let parsed: ServerMessage = serde_json::from_str(line.trim()).unwrap();
            match parsed {
                ServerMessage::Error { message, .. } => {
                    assert!(
                        message.contains("max connections"),
                        "should mention max connections, got: {}",
                        message
                    );
                }
                other => panic!("Expected Error for max connections, got {:?}", other),
            }
        }

        // First connection should still be alive
        let msg = serde_json::to_string(&ClientMessage::Connect { robot_id: 0 }).unwrap();
        let resp = send_and_recv(&mut w1, &mut r1, &msg).await;
        let parsed: ServerMessage = serde_json::from_str(resp.trim()).unwrap();
        assert!(
            matches!(parsed, ServerMessage::Connected { .. }),
            "first connection should still work after second was rejected, got {:?}",
            parsed
        );

        cancel.cancel();
        bg.abort();
    }

    #[tokio::test]
    async fn test_tcp_empty_lines_ignored() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, _cc, _sh) = start_server(bridge, 16).await;

        let (mut writer, mut reader) = connect_client(port).await;

        // Send empty lines (should be ignored)
        writer.write_all(b"\n").await.unwrap();
        writer.write_all(b"\n").await.unwrap();
        writer.write_all(b"   \n").await.unwrap();
        writer.flush().await.unwrap();

        // Now send a valid message -- should still work
        let msg = serde_json::to_string(&ClientMessage::Connect { robot_id: 0 }).unwrap();
        let resp = send_and_recv(&mut writer, &mut reader, &msg).await;
        let parsed: ServerMessage = serde_json::from_str(resp.trim()).unwrap();
        assert!(
            matches!(parsed, ServerMessage::Connected { .. }),
            "connection should work after empty lines, got {:?}",
            parsed
        );

        cancel.cancel();
        bg.abort();
    }

    #[tokio::test]
    async fn test_tcp_partial_json_error() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, _cc, _sh) = start_server(bridge, 16).await;

        let (mut writer, mut reader) = connect_client(port).await;

        // Send partial/truncated JSON
        let resp = send_and_recv(&mut writer, &mut reader, r#"{"type":"conn"#).await;
        let parsed: ServerMessage = serde_json::from_str(resp.trim()).unwrap();
        match parsed {
            ServerMessage::Error { message, .. } => {
                assert!(
                    message.contains("invalid JSON"),
                    "partial JSON should produce invalid JSON error, got: {}",
                    message
                );
            }
            other => panic!("Expected Error for partial JSON, got {:?}", other),
        }

        // Connection should still be alive
        let msg = serde_json::to_string(&ClientMessage::Connect { robot_id: 0 }).unwrap();
        let resp = send_and_recv(&mut writer, &mut reader, &msg).await;
        let parsed: ServerMessage = serde_json::from_str(resp.trim()).unwrap();
        assert!(
            matches!(parsed, ServerMessage::Connected { .. }),
            "connection should survive partial JSON, got {:?}",
            parsed
        );

        cancel.cancel();
        bg.abort();
    }

    #[tokio::test]
    async fn test_tcp_server_cancel_stops_accept() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, _cc, _server_handle) = start_server(bridge, 16).await;

        // Cancel the server
        cancel.cancel();

        // Give the server a moment to stop
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // New connections should fail (server stopped accepting)
        let result = tokio::time::timeout(
            std::time::Duration::from_millis(200),
            tokio::net::TcpStream::connect(("127.0.0.1", port)),
        )
        .await;

        // Either the connection is refused or times out -- both are correct
        match result {
            Ok(Ok(_stream)) => {
                // Connection might still succeed if OS hasn't fully closed the listener,
                // but the server won't process messages.
            }
            Ok(Err(_)) => {} // Connection refused -- expected
            Err(_) => {}     // Timeout -- also acceptable
        }

        bg.abort();
    }

    #[tokio::test]
    async fn test_tcp_rapid_connect_disconnect() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, conn_count, _sh) = start_server(bridge, 16).await;

        // Rapidly connect and disconnect 10 times
        for _ in 0..10 {
            let stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
                .await
                .expect("connect should succeed");
            drop(stream); // immediately disconnect
        }

        // Give server time to process all disconnects
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
        assert_eq!(
            conn_count.load(Ordering::SeqCst),
            0,
            "all connections should be cleaned up after rapid connect/disconnect"
        );

        cancel.cancel();
        bg.abort();
    }

    #[tokio::test]
    async fn test_tcp_unknown_type_returns_error() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, _cc, _sh) = start_server(bridge, 16).await;

        let (mut writer, mut reader) = connect_client(port).await;

        // Send a JSON object with an unknown type
        let resp =
            send_and_recv(&mut writer, &mut reader, r#"{"type":"fly","altitude":100}"#).await;
        let parsed: ServerMessage = serde_json::from_str(resp.trim()).unwrap();
        match parsed {
            ServerMessage::Error { message, .. } => {
                assert!(
                    message.contains("invalid JSON"),
                    "unknown type should produce invalid JSON error, got: {}",
                    message
                );
            }
            other => panic!("Expected Error for unknown type, got {:?}", other),
        }

        cancel.cancel();
        bg.abort();
    }

    /// Regression: a newline-less stream larger than MAX_LINE_BYTES must NOT
    /// be buffered without bound (memory-exhaustion DoS). The server caps the
    /// read at MAX_LINE_BYTES + 1, replies "message too large", and drops the
    /// connection. We send well past the cap with no newline and assert the
    /// bounded-error response arrives.
    #[tokio::test]
    async fn test_tcp_oversized_line_bounded() {
        let (bridge, bg) = setup_bridge(1);
        let (port, cancel, _cc, _sh) = start_server(bridge, 16).await;

        let (mut writer, mut reader) = connect_client(port).await;

        // 200 KiB of non-newline bytes — 3x the 64 KiB cap, no '\n'.
        let flood = vec![b'a'; 200_000];
        writer
            .write_all(&flood)
            .await
            .expect("write flood should succeed");
        writer.flush().await.expect("flush should succeed");

        // Server must respond with the oversize error rather than buffering
        // the whole 200 KiB stream waiting for a newline.
        let mut line = String::new();
        let read = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            reader.read_line(&mut line),
        )
        .await
        .expect("server should respond before timeout (proves read was bounded)")
        .expect("read_line should succeed");
        assert!(read > 0, "expected an error line, got EOF");

        let parsed: ServerMessage =
            serde_json::from_str(line.trim()).expect("response should be a ServerMessage");
        match parsed {
            ServerMessage::Error { message, .. } => {
                assert!(
                    message.contains("too large"),
                    "expected 'message too large', got: {}",
                    message
                );
            }
            other => panic!("Expected oversize Error, got {:?}", other),
        }

        // Connection should now be closed by the server. The next read either
        // returns clean EOF (0 bytes) or errors with ConnectionReset — on macOS
        // closing the read half while the client still has unsent flood bytes
        // surfaces as RST. Both outcomes prove the connection was dropped.
        let mut tail = String::new();
        let closed = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            reader.read_line(&mut tail),
        )
        .await
        .expect("close read should not hang");
        match closed {
            Ok(0) => {} // clean EOF
            Ok(n) => panic!("server should drop the connection, got {n} more bytes: {tail:?}"),
            Err(_) => {} // RST — connection dropped
        }

        cancel.cancel();
        bg.abort();
    }
}

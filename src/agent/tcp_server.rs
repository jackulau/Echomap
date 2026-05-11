use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
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
                                // Reject: write an error and close.
                                let (_, writer) = stream.into_split();
                                let mut bw = BufWriter::new(writer);
                                let err_msg = ServerMessage::Error {
                                    message: "max connections reached".to_string(),
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
        let mut line = String::new();

        loop {
            line.clear();
            tokio::select! {
                _ = cancel.cancelled() => {
                    break;
                }
                result = br.read_line(&mut line) => {
                    match result {
                        Ok(0) => {
                            // EOF — client disconnected.
                            break;
                        }
                        Ok(_) => {
                            let trimmed = line.trim();
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
                                    let err = ServerMessage::Error {
                                        message: format!("invalid JSON: {}", e),
                                    };
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
        let _ = session
            .handle_message(ClientMessage::Close)
            .await;
    }

    /// Serialize a `ServerMessage` as JSON followed by a newline, and flush.
    async fn write_message(
        bw: &mut BufWriter<tokio::net::tcp::OwnedWriteHalf>,
        msg: &ServerMessage,
    ) -> io::Result<()> {
        let json = serde_json::to_string(msg)
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
    ) -> (u16, CancellationToken, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
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
        let connect_json = serde_json::to_string(&ClientMessage::Connect { robot_id: 0 })
            .unwrap();
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
            ServerMessage::Error { message } => {
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
}

#[allow(dead_code)]
pub mod bridge;
#[allow(dead_code)]
pub mod protocol;
#[allow(dead_code)]
pub mod session;
#[allow(dead_code)]
pub mod tcp_server;
#[allow(dead_code)]
pub mod ws_server;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;

use tokio_util::sync::CancellationToken;

use crate::agent::bridge::SimBridgeServer;
use crate::agent::tcp_server::TcpAgentServer;
use crate::agent::ws_server::WsAgentServer;

/// Configuration for the agent server.
#[derive(Debug, Clone)]
pub struct AgentServerConfig {
    /// TCP port for line-delimited JSON connections (default 9001).
    pub tcp_port: u16,
    /// WebSocket port (default 9002).
    pub ws_port: u16,
    /// Maximum number of concurrent connections per server (default 16).
    pub max_connections: usize,
    /// Whether the server is enabled (default false).
    pub enabled: bool,
}

impl Default for AgentServerConfig {
    fn default() -> Self {
        Self {
            tcp_port: 9001,
            ws_port: 9002,
            max_connections: 16,
            enabled: false,
        }
    }
}

/// Runtime status snapshot of the agent server.
#[derive(Debug, Clone)]
pub struct AgentServerStatus {
    pub tcp_port: u16,
    pub ws_port: u16,
    pub tcp_connections: usize,
    pub ws_connections: usize,
    pub running: bool,
}

/// Handle to a running agent server.
///
/// Holds the background thread, cancellation token, and shared connection
/// counters. Use `stop()` for graceful shutdown or `status()` to query state.
pub struct AgentServerHandle {
    thread: Option<JoinHandle<()>>,
    cancel: CancellationToken,
    tcp_connections: Arc<AtomicUsize>,
    ws_connections: Arc<AtomicUsize>,
    tcp_port: u16,
    ws_port: u16,
}

impl AgentServerHandle {
    /// Gracefully stop the server and join the background thread.
    pub fn stop(&mut self) {
        self.cancel.cancel();
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
        log::info!("Agent server stopped");
    }

    /// Query the current server status.
    pub fn status(&self) -> AgentServerStatus {
        AgentServerStatus {
            tcp_port: self.tcp_port,
            ws_port: self.ws_port,
            tcp_connections: self.tcp_connections.load(Ordering::Relaxed),
            ws_connections: self.ws_connections.load(Ordering::Relaxed),
            running: self.thread.is_some() && !self.cancel.is_cancelled(),
        }
    }
}

impl Drop for AgentServerHandle {
    fn drop(&mut self) {
        if self.thread.is_some() {
            self.stop();
        }
    }
}

/// Spawn the agent server on a dedicated thread with its own tokio runtime.
///
/// The server runs both a TCP and a WebSocket listener. Commands from
/// connected agents are forwarded through the `bridge_server` to the
/// main loop's `SimBridgeClient`.
///
/// Returns an `AgentServerHandle` for status queries and shutdown control.
pub fn start_agent_server(
    config: AgentServerConfig,
    bridge_server: SimBridgeServer,
) -> AgentServerHandle {
    let cancel = CancellationToken::new();
    let cancel_clone = cancel.clone();
    let tcp_connections = Arc::new(AtomicUsize::new(0));
    let ws_connections = Arc::new(AtomicUsize::new(0));
    let tcp_conn_clone = tcp_connections.clone();
    let ws_conn_clone = ws_connections.clone();

    // Channel to send actual bound ports back to the caller.
    let (port_tx, port_rx) = std::sync::mpsc::channel::<(u16, u16)>();

    let thread = std::thread::Builder::new()
        .name("agent-server".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("failed to build tokio runtime for agent server");

            rt.block_on(async move {
                // Bind TCP server.
                let tcp_server = TcpAgentServer::bind(
                    config.tcp_port,
                    bridge_server.clone(),
                    config.max_connections,
                )
                .await
                .expect("failed to bind TCP agent server");
                let actual_tcp_port = tcp_server.local_port();

                // Bind WS server.
                let ws_server =
                    WsAgentServer::bind(config.ws_port, bridge_server, config.max_connections)
                        .await
                        .expect("failed to bind WS agent server");
                let actual_ws_port = ws_server.local_port();

                log::info!(
                    "Agent server started: TCP port {}, WS port {}",
                    actual_tcp_port,
                    actual_ws_port
                );

                // Send actual ports back.
                let _ = port_tx.send((actual_tcp_port, actual_ws_port));

                // Share connection counts. The servers use internal AtomicUsize
                // counters, so we mirror them with a polling task.
                let cancel_poll = cancel_clone.clone();
                let tcp_conn_inner = tcp_conn_clone.clone();
                let ws_conn_inner = ws_conn_clone.clone();

                // Run both servers concurrently.
                let cancel_tcp = cancel_clone.clone();
                let cancel_ws = cancel_clone.clone();

                tokio::select! {
                    _ = tcp_server.run(cancel_tcp) => {}
                    _ = ws_server.run(cancel_ws) => {}
                    _ = cancel_clone.cancelled() => {}
                    _ = async {
                        // Polling task to mirror connection counts.
                        loop {
                            tcp_conn_inner.store(tcp_server.connection_count(), Ordering::Relaxed);
                            ws_conn_inner.store(ws_server.connection_count(), Ordering::Relaxed);
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                            if cancel_poll.is_cancelled() {
                                break;
                            }
                        }
                    } => {}
                }
            });
        })
        .expect("failed to spawn agent server thread");

    // Wait for actual bound ports (with a timeout).
    let (tcp_port, ws_port) = port_rx
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap_or((config.tcp_port, config.ws_port));

    AgentServerHandle {
        thread: Some(thread),
        cancel,
        tcp_connections,
        ws_connections,
        tcp_port,
        ws_port,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bridge::create_bridge;
    use crate::robot::definition::RobotDefinition;
    use crate::robot::RobotManager;
    use glam::Mat4;

    #[test]
    fn test_config_defaults() {
        let config = AgentServerConfig::default();
        assert_eq!(config.tcp_port, 9001);
        assert_eq!(config.ws_port, 9002);
        assert_eq!(config.max_connections, 16);
        assert!(!config.enabled);
    }

    #[test]
    fn test_server_handle_status() {
        let (bridge_server, _bridge_client) = create_bridge();
        let config = AgentServerConfig {
            tcp_port: 0,
            ws_port: 0,
            max_connections: 16,
            enabled: true,
        };
        let handle = start_agent_server(config, bridge_server);
        let status = handle.status();
        assert!(status.running, "server should be running after start");
        assert!(status.tcp_port > 0, "TCP port should be assigned");
        assert!(status.ws_port > 0, "WS port should be assigned");
        assert_eq!(status.tcp_connections, 0);
        assert_eq!(status.ws_connections, 0);

        // Clean up.
        drop(handle);
    }

    #[test]
    fn test_server_handle_stop() {
        let (bridge_server, _bridge_client) = create_bridge();
        let config = AgentServerConfig {
            tcp_port: 0,
            ws_port: 0,
            max_connections: 16,
            enabled: true,
        };
        let mut handle = start_agent_server(config, bridge_server);
        assert!(handle.status().running, "should be running before stop");
        handle.stop();
        assert!(!handle.status().running, "should not be running after stop");
    }

    #[test]
    fn test_bridge_process_in_update() {
        // Simulate what the main loop does: create bridge, add a robot,
        // and call process_pending each "frame".
        let (bridge_server, mut bridge_client) = create_bridge();
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        manager.add_robot(def, Mat4::IDENTITY);

        // Simulate a frame: no pending commands, should be a no-op.
        bridge_client.process_pending(&mut manager, &[]);

        // Enqueue a command via a tokio runtime (simulating the server side).
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let response = rt.block_on(async {
            let handle = tokio::spawn({
                let bridge = bridge_server.clone();
                async move {
                    bridge
                        .send_command(crate::agent::bridge::SimCommand::GetObservation {
                            robot_id: 0,
                        })
                        .await
                }
            });

            tokio::task::yield_now().await;
            bridge_client.process_pending(&mut manager, &[]);

            handle.await.unwrap()
        });

        match response {
            Ok(crate::agent::bridge::SimResponse::Observation { state }) => {
                assert_eq!(
                    state.joint_positions.len(),
                    2,
                    "should have 2 joint positions"
                );
            }
            other => panic!("Expected Observation, got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // Task 8: End-to-end integration tests
    // ------------------------------------------------------------------

    use crate::agent::protocol::{ClientMessage, ServerMessage};
    use crate::robot::state::RobotAction;

    /// Helper: start a full agent server with `n` simple_arm(2) robots and a
    /// background thread that continuously processes bridge commands.
    /// Returns (AgentServerHandle, bridge_processing_thread_handle).
    fn start_test_server(n: usize) -> (AgentServerHandle, std::thread::JoinHandle<()>) {
        let (bridge_server, mut bridge_client) = create_bridge();
        let mut manager = RobotManager::new();
        for _ in 0..n {
            let def = RobotDefinition::simple_arm(2);
            manager.add_robot(def, Mat4::IDENTITY);
        }

        let config = AgentServerConfig {
            tcp_port: 0,
            ws_port: 0,
            max_connections: 16,
            enabled: true,
        };

        let handle = start_agent_server(config, bridge_server);

        // Spawn a background thread that continuously processes bridge commands.
        // This simulates the main loop calling process_pending each frame.
        let bridge_thread = std::thread::Builder::new()
            .name("test-bridge-processor".to_string())
            .spawn(move || loop {
                bridge_client.process_pending(&mut manager, &[]);
                std::thread::sleep(std::time::Duration::from_micros(100));
            })
            .expect("failed to spawn bridge processing thread");

        // Give servers a moment to be fully ready.
        std::thread::sleep(std::time::Duration::from_millis(50));

        (handle, bridge_thread)
    }

    /// Helper: send a line-delimited JSON message over TCP and read the response.
    async fn tcp_send_recv(
        writer: &mut tokio::io::BufWriter<tokio::net::tcp::OwnedWriteHalf>,
        reader: &mut tokio::io::BufReader<tokio::net::tcp::OwnedReadHalf>,
        msg: &ClientMessage,
    ) -> ServerMessage {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

        let json = serde_json::to_string(msg).unwrap();
        writer.write_all(json.as_bytes()).await.unwrap();
        writer.write_all(b"\n").await.unwrap();
        writer.flush().await.unwrap();

        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        serde_json::from_str::<ServerMessage>(line.trim()).unwrap()
    }

    /// Helper: connect a TCP client and return buffered reader/writer.
    async fn tcp_connect(
        port: u16,
    ) -> (
        tokio::io::BufWriter<tokio::net::tcp::OwnedWriteHalf>,
        tokio::io::BufReader<tokio::net::tcp::OwnedReadHalf>,
    ) {
        let stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .expect("should connect to TCP server");
        let (r, w) = stream.into_split();
        (tokio::io::BufWriter::new(w), tokio::io::BufReader::new(r))
    }

    /// Helper: connect a WebSocket client and return split sink/stream.
    async fn ws_connect(
        port: u16,
    ) -> (
        futures_util::stream::SplitSink<
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
            tokio_tungstenite::tungstenite::Message,
        >,
        futures_util::stream::SplitStream<
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
        >,
    ) {
        use futures_util::StreamExt;
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
            tokio_tungstenite::tungstenite::Message,
        >,
        read: &mut futures_util::stream::SplitStream<
            tokio_tungstenite::WebSocketStream<
                tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
            >,
        >,
        msg: &ClientMessage,
    ) -> ServerMessage {
        use futures_util::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite::Message as WsMsg;

        let json = serde_json::to_string(msg).unwrap();
        write
            .send(WsMsg::Text(json.into()))
            .await
            .expect("ws send should succeed");

        loop {
            match read.next().await {
                Some(Ok(WsMsg::Text(text))) => {
                    return serde_json::from_str::<ServerMessage>(&text)
                        .expect("should parse server message");
                }
                Some(Ok(WsMsg::Ping(_) | WsMsg::Pong(_))) => continue,
                other => panic!("Expected Text message, got {:?}", other),
            }
        }
    }

    #[tokio::test]
    async fn test_integration_tcp_full_lifecycle() {
        // Complete connect -> reset -> step 10 times -> observe -> close over TCP.
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;

        let (mut writer, mut reader) = tcp_connect(tcp_port).await;

        // Connect to robot 0.
        let resp = tcp_send_recv(
            &mut writer,
            &mut reader,
            &ClientMessage::Connect { robot_id: 0 },
        )
        .await;
        match &resp {
            ServerMessage::Connected {
                observation_space,
                action_space,
            } => {
                assert_eq!(observation_space.num_joint_positions, 2);
                assert_eq!(action_space.num_motors, 2);
            }
            other => panic!("Expected Connected, got {:?}", other),
        }

        // Reset.
        let resp = tcp_send_recv(&mut writer, &mut reader, &ClientMessage::Reset).await;
        match &resp {
            ServerMessage::Observation {
                step_count, done, ..
            } => {
                assert_eq!(*step_count, 0, "reset should yield step_count=0");
                assert!(!done, "done should be false after reset");
            }
            other => panic!("Expected Observation after reset, got {:?}", other),
        }

        // Step 10 times.
        let action = RobotAction {
            motor_velocities: vec![1.0, -0.5],
            gripper_commands: vec![],
        };
        for i in 1..=10 {
            let resp = tcp_send_recv(
                &mut writer,
                &mut reader,
                &ClientMessage::Step {
                    action: action.clone(),
                },
            )
            .await;
            match &resp {
                ServerMessage::Observation { step_count, .. } => {
                    assert_eq!(
                        *step_count, i,
                        "step_count should be {} after step {}",
                        i, i
                    );
                }
                other => panic!("Expected Observation after step {}, got {:?}", i, other),
            }
        }

        // Observe — step_count should still be 10.
        let resp = tcp_send_recv(&mut writer, &mut reader, &ClientMessage::Observe).await;
        match &resp {
            ServerMessage::Observation {
                step_count, state, ..
            } => {
                assert_eq!(*step_count, 10, "observe should show step_count=10");
                assert_eq!(state.joint_positions.len(), 2);
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        // Close.
        let resp = tcp_send_recv(&mut writer, &mut reader, &ClientMessage::Close).await;
        assert!(
            matches!(resp, ServerMessage::Closed),
            "Expected Closed, got {:?}",
            resp
        );

        // Clean up.
        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_integration_ws_full_lifecycle() {
        // Complete connect -> reset -> step 10 times -> observe -> close over WebSocket.
        let (handle, bridge_thread) = start_test_server(1);
        let ws_port = handle.status().ws_port;

        let (mut write, mut read) = ws_connect(ws_port).await;

        // Connect to robot 0.
        let resp = ws_send_recv(
            &mut write,
            &mut read,
            &ClientMessage::Connect { robot_id: 0 },
        )
        .await;
        match &resp {
            ServerMessage::Connected {
                observation_space,
                action_space,
            } => {
                assert_eq!(observation_space.num_joint_positions, 2);
                assert_eq!(action_space.num_motors, 2);
            }
            other => panic!("Expected Connected, got {:?}", other),
        }

        // Reset.
        let resp = ws_send_recv(&mut write, &mut read, &ClientMessage::Reset).await;
        match &resp {
            ServerMessage::Observation {
                step_count, done, ..
            } => {
                assert_eq!(*step_count, 0, "reset should yield step_count=0");
                assert!(!done, "done should be false after reset");
            }
            other => panic!("Expected Observation after reset, got {:?}", other),
        }

        // Step 10 times.
        let action = RobotAction {
            motor_velocities: vec![1.0, -0.5],
            gripper_commands: vec![],
        };
        for i in 1..=10 {
            let resp = ws_send_recv(
                &mut write,
                &mut read,
                &ClientMessage::Step {
                    action: action.clone(),
                },
            )
            .await;
            match &resp {
                ServerMessage::Observation { step_count, .. } => {
                    assert_eq!(
                        *step_count, i,
                        "step_count should be {} after step {}",
                        i, i
                    );
                }
                other => panic!("Expected Observation after step {}, got {:?}", i, other),
            }
        }

        // Observe — step_count should still be 10.
        let resp = ws_send_recv(&mut write, &mut read, &ClientMessage::Observe).await;
        match &resp {
            ServerMessage::Observation {
                step_count, state, ..
            } => {
                assert_eq!(*step_count, 10, "observe should show step_count=10");
                assert_eq!(state.joint_positions.len(), 2);
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        // Close.
        let resp = ws_send_recv(&mut write, &mut read, &ClientMessage::Close).await;
        assert!(
            matches!(resp, ServerMessage::Closed),
            "Expected Closed, got {:?}",
            resp
        );

        // Clean up.
        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_integration_multi_agent() {
        // Two agents connect to different robots and step independently.
        let (handle, bridge_thread) = start_test_server(2);
        let tcp_port = handle.status().tcp_port;

        // Agent 1: connect to robot 0 via TCP.
        let (mut w1, mut r1) = tcp_connect(tcp_port).await;
        let resp = tcp_send_recv(&mut w1, &mut r1, &ClientMessage::Connect { robot_id: 0 }).await;
        assert!(
            matches!(resp, ServerMessage::Connected { .. }),
            "Agent 1 should connect: {:?}",
            resp
        );

        // Agent 2: connect to robot 1 via TCP.
        let (mut w2, mut r2) = tcp_connect(tcp_port).await;
        let resp = tcp_send_recv(&mut w2, &mut r2, &ClientMessage::Connect { robot_id: 1 }).await;
        assert!(
            matches!(resp, ServerMessage::Connected { .. }),
            "Agent 2 should connect: {:?}",
            resp
        );

        // Both agents step with different actions.
        let action1 = RobotAction {
            motor_velocities: vec![2.0, 0.0],
            gripper_commands: vec![],
        };
        let action2 = RobotAction {
            motor_velocities: vec![0.0, -2.0],
            gripper_commands: vec![],
        };

        // Step agent 1 three times.
        for i in 1..=3 {
            let resp = tcp_send_recv(
                &mut w1,
                &mut r1,
                &ClientMessage::Step {
                    action: action1.clone(),
                },
            )
            .await;
            match &resp {
                ServerMessage::Observation { step_count, .. } => {
                    assert_eq!(*step_count, i, "Agent 1 step_count should be {}", i);
                }
                other => panic!("Agent 1 step {}: expected Observation, got {:?}", i, other),
            }
        }

        // Step agent 2 five times.
        for i in 1..=5 {
            let resp = tcp_send_recv(
                &mut w2,
                &mut r2,
                &ClientMessage::Step {
                    action: action2.clone(),
                },
            )
            .await;
            match &resp {
                ServerMessage::Observation { step_count, .. } => {
                    assert_eq!(*step_count, i, "Agent 2 step_count should be {}", i);
                }
                other => panic!("Agent 2 step {}: expected Observation, got {:?}", i, other),
            }
        }

        // Verify agents have independent observations.
        let obs1 = tcp_send_recv(&mut w1, &mut r1, &ClientMessage::Observe).await;
        let obs2 = tcp_send_recv(&mut w2, &mut r2, &ClientMessage::Observe).await;

        match (&obs1, &obs2) {
            (
                ServerMessage::Observation {
                    step_count: sc1, ..
                },
                ServerMessage::Observation {
                    step_count: sc2, ..
                },
            ) => {
                assert_eq!(*sc1, 3, "Agent 1 should show step_count=3");
                assert_eq!(*sc2, 5, "Agent 2 should show step_count=5");
            }
            _ => panic!("Expected Observations, got {:?} and {:?}", obs1, obs2),
        }

        // Clean up.
        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_integration_reconnect() {
        // Agent connects, does some work, disconnects, then reconnects to the same robot.
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;

        // First connection.
        {
            let (mut w, mut r) = tcp_connect(tcp_port).await;

            let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;
            assert!(
                matches!(resp, ServerMessage::Connected { .. }),
                "First connect should succeed: {:?}",
                resp
            );

            // Step a few times.
            let action = RobotAction {
                motor_velocities: vec![1.0, 1.0],
                gripper_commands: vec![],
            };
            for _ in 0..3 {
                let resp = tcp_send_recv(
                    &mut w,
                    &mut r,
                    &ClientMessage::Step {
                        action: action.clone(),
                    },
                )
                .await;
                assert!(matches!(resp, ServerMessage::Observation { .. }));
            }

            // Close the session.
            let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Close).await;
            assert!(matches!(resp, ServerMessage::Closed));

            // Drop the TCP connection.
        }

        // Brief pause so the server cleans up.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        // Second connection to the same robot.
        {
            let (mut w, mut r) = tcp_connect(tcp_port).await;

            let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;
            match &resp {
                ServerMessage::Connected {
                    observation_space,
                    action_space,
                } => {
                    assert_eq!(observation_space.num_joint_positions, 2);
                    assert_eq!(action_space.num_motors, 2);
                }
                other => panic!("Reconnect should succeed, got {:?}", other),
            }

            // Reset to get a clean state.
            let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Reset).await;
            match &resp {
                ServerMessage::Observation { step_count, .. } => {
                    assert_eq!(
                        *step_count, 0,
                        "step_count should be 0 after reset on reconnect"
                    );
                }
                other => panic!("Expected Observation after reset, got {:?}", other),
            }

            // Step once to verify the session works.
            let action = RobotAction {
                motor_velocities: vec![0.5, -0.5],
                gripper_commands: vec![],
            };
            let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Step { action }).await;
            match &resp {
                ServerMessage::Observation { step_count, .. } => {
                    assert_eq!(
                        *step_count, 1,
                        "step_count should be 1 after first step on reconnect"
                    );
                }
                other => panic!("Expected Observation after step, got {:?}", other),
            }
        }

        // Clean up.
        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_integration_rapid_steps() {
        // 100 rapid step commands should all complete without error.
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;

        let (mut w, mut r) = tcp_connect(tcp_port).await;

        // Connect.
        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;
        assert!(matches!(resp, ServerMessage::Connected { .. }));

        // Reset.
        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Reset).await;
        assert!(matches!(resp, ServerMessage::Observation { .. }));

        // Fire 100 steps rapidly.
        let action = RobotAction {
            motor_velocities: vec![0.5, -0.3],
            gripper_commands: vec![],
        };

        for i in 1..=100u64 {
            let resp = tcp_send_recv(
                &mut w,
                &mut r,
                &ClientMessage::Step {
                    action: action.clone(),
                },
            )
            .await;
            match &resp {
                ServerMessage::Observation { step_count, .. } => {
                    assert_eq!(
                        *step_count, i,
                        "step_count should be {} at rapid step {}",
                        i, i
                    );
                }
                other => panic!("Rapid step {}: expected Observation, got {:?}", i, other),
            }
        }

        // Final observe to confirm all 100 completed.
        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Observe).await;
        match &resp {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(*step_count, 100, "final observe should show step_count=100");
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        // Clean up.
        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_integration_observation_changes() {
        // Observations should differ after stepping with non-zero actions.
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;

        let (mut w, mut r) = tcp_connect(tcp_port).await;

        // Connect and reset.
        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;
        assert!(matches!(resp, ServerMessage::Connected { .. }));

        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Reset).await;
        let initial_positions = match &resp {
            ServerMessage::Observation { state, .. } => state.joint_positions.clone(),
            other => panic!("Expected Observation after reset, got {:?}", other),
        };

        // Step multiple times with non-zero velocities.
        let action = RobotAction {
            motor_velocities: vec![5.0, -3.0],
            gripper_commands: vec![],
        };
        for _ in 0..10 {
            tcp_send_recv(
                &mut w,
                &mut r,
                &ClientMessage::Step {
                    action: action.clone(),
                },
            )
            .await;
        }

        // Observe and compare.
        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Observe).await;
        let final_positions = match &resp {
            ServerMessage::Observation { state, .. } => state.joint_positions.clone(),
            other => panic!("Expected Observation, got {:?}", other),
        };

        // At least one joint position should have changed.
        let any_changed = initial_positions
            .iter()
            .zip(final_positions.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);

        assert!(
            any_changed,
            "joint positions should change after stepping with non-zero actions: initial={:?}, final={:?}",
            initial_positions,
            final_positions
        );

        // Clean up.
        drop(handle);
        drop(bridge_thread);
    }
}

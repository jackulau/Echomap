#[allow(dead_code)]
pub mod bridge;
#[allow(dead_code)]
pub mod demo;
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
            Ok(crate::agent::bridge::SimResponse::Observation { state, .. }) => {
                assert_eq!(
                    state.joint_positions.len(),
                    2,
                    "should have 2 joint positions"
                );
            }
            other => panic!("Expected Observation, got {:?}", other),
        }
    }

    // ---- Edge case tests ----

    #[test]
    fn test_server_double_stop() {
        let (bridge_server, _bridge_client) = create_bridge();
        let config = AgentServerConfig {
            tcp_port: 0,
            ws_port: 0,
            max_connections: 16,
            enabled: true,
        };
        let mut handle = start_agent_server(config, bridge_server);
        handle.stop();
        assert!(
            !handle.status().running,
            "should not be running after first stop"
        );

        // Second stop should not panic
        handle.stop();
        assert!(
            !handle.status().running,
            "should still not be running after second stop"
        );
    }

    #[test]
    fn test_server_status_after_drop() {
        let (bridge_server, _bridge_client) = create_bridge();
        let config = AgentServerConfig {
            tcp_port: 0,
            ws_port: 0,
            max_connections: 16,
            enabled: true,
        };
        let handle = start_agent_server(config, bridge_server);
        let tcp_port = handle.status().tcp_port;
        let ws_port = handle.status().ws_port;
        assert!(tcp_port > 0);
        assert!(ws_port > 0);
        // Drop triggers the implicit stop via Drop impl -- should not panic
        drop(handle);
    }

    #[test]
    fn test_config_clone() {
        let config = AgentServerConfig {
            tcp_port: 1234,
            ws_port: 5678,
            max_connections: 4,
            enabled: true,
        };
        let cloned = config.clone();
        assert_eq!(cloned.tcp_port, 1234);
        assert_eq!(cloned.ws_port, 5678);
        assert_eq!(cloned.max_connections, 4);
        assert!(cloned.enabled);
    }

    #[test]
    fn test_config_debug() {
        let config = AgentServerConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("AgentServerConfig"));
        assert!(debug_str.contains("9001"));
        assert!(debug_str.contains("9002"));
    }

    #[test]
    fn test_status_debug() {
        let status = AgentServerStatus {
            tcp_port: 1111,
            ws_port: 2222,
            tcp_connections: 3,
            ws_connections: 4,
            running: true,
        };
        let debug_str = format!("{:?}", status);
        assert!(debug_str.contains("AgentServerStatus"));
        assert!(debug_str.contains("1111"));
        assert!(debug_str.contains("2222"));
    }

    #[test]
    fn test_status_clone() {
        let status = AgentServerStatus {
            tcp_port: 100,
            ws_port: 200,
            tcp_connections: 5,
            ws_connections: 10,
            running: true,
        };
        let cloned = status.clone();
        assert_eq!(cloned.tcp_port, 100);
        assert_eq!(cloned.ws_port, 200);
        assert_eq!(cloned.tcp_connections, 5);
        assert_eq!(cloned.ws_connections, 10);
        assert!(cloned.running);
    }

    #[test]
    fn test_server_tcp_ws_different_ports() {
        let (bridge_server, _bridge_client) = create_bridge();
        let config = AgentServerConfig {
            tcp_port: 0,
            ws_port: 0,
            max_connections: 16,
            enabled: true,
        };
        let handle = start_agent_server(config, bridge_server);
        let status = handle.status();
        assert_ne!(
            status.tcp_port, status.ws_port,
            "TCP and WS ports should be different when both use port 0"
        );
        drop(handle);
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

    // ------------------------------------------------------------------
    // Edge-case tests
    // ------------------------------------------------------------------

    // ---- Protocol edge cases: malformed/empty/special JSON ----

    #[tokio::test]
    async fn test_edge_malformed_json_tcp_keeps_connection_alive() {
        // Send several types of malformed JSON, then a valid command.
        // The connection should survive all malformed inputs.
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;

        let stream = tokio::net::TcpStream::connect(("127.0.0.1", tcp_port))
            .await
            .expect("should connect");
        let (r, w) = stream.into_split();
        let mut writer = tokio::io::BufWriter::new(w);
        let mut reader = tokio::io::BufReader::new(r);

        use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

        let malformed_inputs = vec![
            "",                       // empty string (should be skipped as blank line)
            "{}",                     // empty JSON object
            "{\"type\":\"unknown\"}", // unknown variant
            "null",                   // JSON null
            "42",                     // JSON number
            "[1,2,3]",                // JSON array
            "{\"type\":\"step\"}",    // step missing action field
            "{\"type\":\"connect\"}", // connect missing robot_id
            "not json at all {{{",    // gibberish
            "{\"type\":\"connect\",\"robot_id\":\"not_a_number\"}", // wrong type for robot_id
        ];

        let mut error_count = 0;
        for input in &malformed_inputs {
            writer.write_all(input.as_bytes()).await.unwrap();
            writer.write_all(b"\n").await.unwrap();
            writer.flush().await.unwrap();

            if input.trim().is_empty() {
                // Empty line is skipped by the server, no response expected.
                continue;
            }

            let mut line = String::new();
            reader.read_line(&mut line).await.unwrap();
            let msg: ServerMessage = serde_json::from_str(line.trim())
                .expect("server should respond with parseable JSON");
            match msg {
                ServerMessage::Error { .. } => error_count += 1,
                other => panic!("Expected Error for input {:?}, got {:?}", input, other),
            }
        }

        // All non-empty malformed inputs should produce errors.
        assert!(
            error_count >= 8,
            "should have received at least 8 errors, got {}",
            error_count
        );

        // Now send a valid Connect - connection should still work.
        let valid = serde_json::to_string(&ClientMessage::Connect { robot_id: 0 }).unwrap();
        writer.write_all(valid.as_bytes()).await.unwrap();
        writer.write_all(b"\n").await.unwrap();
        writer.flush().await.unwrap();

        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        let msg: ServerMessage = serde_json::from_str(line.trim()).unwrap();
        assert!(
            matches!(msg, ServerMessage::Connected { .. }),
            "Connection should still work after malformed inputs, got {:?}",
            msg
        );

        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_edge_empty_action_vector() {
        // Step with an empty motor_velocities vector. The robot has 2 joints
        // but the action provides 0 velocities. apply_action should handle
        // this gracefully (min(0, 2) = 0 motors applied).
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;
        let (mut w, mut r) = tcp_connect(tcp_port).await;

        // Connect and reset.
        tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;
        tcp_send_recv(&mut w, &mut r, &ClientMessage::Reset).await;

        // Step with empty actions.
        let empty_action = RobotAction {
            motor_velocities: vec![],
            gripper_commands: vec![],
        };
        let resp = tcp_send_recv(
            &mut w,
            &mut r,
            &ClientMessage::Step {
                action: empty_action,
            },
        )
        .await;

        match &resp {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(
                    *step_count, 1,
                    "step with empty action should still increment count"
                );
            }
            other => panic!("Expected Observation with empty action, got {:?}", other),
        }

        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_edge_oversized_action_vector() {
        // Step with more motor velocities than joints. apply_action should
        // clamp to min(action_len, joint_count).
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;
        let (mut w, mut r) = tcp_connect(tcp_port).await;

        tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;
        tcp_send_recv(&mut w, &mut r, &ClientMessage::Reset).await;

        // Robot has 2 joints, send 10 velocities.
        let big_action = RobotAction {
            motor_velocities: vec![1.0; 10],
            gripper_commands: vec![],
        };
        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Step { action: big_action }).await;

        match &resp {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(*step_count, 1, "oversized action should still work");
            }
            other => panic!(
                "Expected Observation with oversized action, got {:?}",
                other
            ),
        }

        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_edge_nan_inf_action_values() {
        // BUG DISCOVERED: NaN/Inf in actions propagate through the physics
        // engine and poison the observation state. When the server then tries
        // to serialize the observation, serde_json turns NaN into `null`,
        // which cannot be deserialized back as f32. The TCP handler catches
        // this as a JSON parse error on the write path and returns an Error.
        //
        // This test documents the current (broken) behavior. The correct fix
        // would be to validate/clamp action values before applying them.
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;
        let (mut w, mut r) = tcp_connect(tcp_port).await;

        tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;
        tcp_send_recv(&mut w, &mut r, &ClientMessage::Reset).await;

        // Step with NaN. The server does not crash, but the response will
        // be an Error because NaN corrupts the observation serialization.
        let nan_action = RobotAction {
            motor_velocities: vec![f32::NAN, f32::NAN],
            gripper_commands: vec![],
        };
        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Step { action: nan_action }).await;
        // The server does not crash (no panic), but the observation is corrupted.
        // We verify the server stays alive and responds (either Observation or Error).
        assert!(
            matches!(
                resp,
                ServerMessage::Observation { .. } | ServerMessage::Error { .. }
            ),
            "NaN action should not crash the server. Got {:?}",
            resp
        );

        // Verify server is still alive after NaN poisoning by sending a Reset.
        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Reset).await;
        assert!(
            matches!(resp, ServerMessage::Observation { .. }),
            "Server should still respond after NaN poisoning + reset. Got {:?}",
            resp
        );

        drop(handle);
        drop(bridge_thread);
    }

    // ---- State machine edge cases ----

    #[tokio::test]
    async fn test_edge_step_without_reset() {
        // Step immediately after connect (without reset). Should still work
        // since the session is connected.
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;
        let (mut w, mut r) = tcp_connect(tcp_port).await;

        tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;

        // Skip reset, go straight to step.
        let action = RobotAction {
            motor_velocities: vec![1.0, -1.0],
            gripper_commands: vec![],
        };
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
                assert_eq!(*step_count, 1, "step without reset should still work");
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_edge_observe_before_any_step() {
        // Observe immediately after connect+reset (step_count=0).
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;
        let (mut w, mut r) = tcp_connect(tcp_port).await;

        tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;
        tcp_send_recv(&mut w, &mut r, &ClientMessage::Reset).await;

        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Observe).await;
        match &resp {
            ServerMessage::Observation {
                step_count, state, ..
            } => {
                assert_eq!(
                    *step_count, 0,
                    "observe before stepping should show step_count=0"
                );
                assert_eq!(state.joint_positions.len(), 2);
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_edge_double_reset() {
        // Reset twice in a row. Second reset should also produce step_count=0.
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;
        let (mut w, mut r) = tcp_connect(tcp_port).await;

        tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;

        // Step a few times.
        let action = RobotAction {
            motor_velocities: vec![1.0, -1.0],
            gripper_commands: vec![],
        };
        for _ in 0..5 {
            tcp_send_recv(
                &mut w,
                &mut r,
                &ClientMessage::Step {
                    action: action.clone(),
                },
            )
            .await;
        }

        // First reset.
        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Reset).await;
        match &resp {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(*step_count, 0, "first reset should yield step_count=0");
            }
            other => panic!("Expected Observation after first reset, got {:?}", other),
        }

        // Second reset immediately.
        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Reset).await;
        match &resp {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(
                    *step_count, 0,
                    "second reset should also yield step_count=0"
                );
            }
            other => panic!("Expected Observation after second reset, got {:?}", other),
        }

        // Step after double-reset should start from 1.
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
                assert_eq!(*step_count, 1, "step after double reset should be 1");
            }
            other => panic!("Expected Observation after step, got {:?}", other),
        }

        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_edge_close_then_commands() {
        // After Close, all commands except Connect should return errors.
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;
        let (mut w, mut r) = tcp_connect(tcp_port).await;

        // Connect and close.
        tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;
        tcp_send_recv(&mut w, &mut r, &ClientMessage::Close).await;

        // Step after close.
        let action = RobotAction {
            motor_velocities: vec![1.0, -1.0],
            gripper_commands: vec![],
        };
        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Step { action }).await;
        match &resp {
            ServerMessage::Error { message } => {
                assert!(
                    message.contains("not connected"),
                    "step after close should say not connected, got: {}",
                    message
                );
            }
            other => panic!("Expected Error after close+step, got {:?}", other),
        }

        // Observe after close.
        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Observe).await;
        match &resp {
            ServerMessage::Error { message } => {
                assert!(
                    message.contains("not connected"),
                    "observe after close should say not connected, got: {}",
                    message
                );
            }
            other => panic!("Expected Error after close+observe, got {:?}", other),
        }

        // Reset after close.
        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Reset).await;
        match &resp {
            ServerMessage::Error { message } => {
                assert!(
                    message.contains("not connected"),
                    "reset after close should say not connected, got: {}",
                    message
                );
            }
            other => panic!("Expected Error after close+reset, got {:?}", other),
        }

        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_edge_close_and_reconnect_same_connection() {
        // Close then Connect again on the same TCP connection.
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;
        let (mut w, mut r) = tcp_connect(tcp_port).await;

        // First session.
        tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;
        let action = RobotAction {
            motor_velocities: vec![1.0, -1.0],
            gripper_commands: vec![],
        };
        tcp_send_recv(
            &mut w,
            &mut r,
            &ClientMessage::Step {
                action: action.clone(),
            },
        )
        .await;
        tcp_send_recv(&mut w, &mut r, &ClientMessage::Close).await;

        // Reconnect on same TCP socket.
        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;
        match &resp {
            ServerMessage::Connected { .. } => {}
            other => panic!("Should reconnect on same TCP socket, got {:?}", other),
        }

        // Step count should reset to 0 after fresh connect.
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
                    *step_count, 1,
                    "step_count should be 1 after reconnect on same socket"
                );
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_edge_double_close() {
        // Close twice on the same session. Second close should still return Closed.
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;
        let (mut w, mut r) = tcp_connect(tcp_port).await;

        tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;

        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Close).await;
        assert!(
            matches!(resp, ServerMessage::Closed),
            "First close: expected Closed, got {:?}",
            resp
        );

        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Close).await;
        assert!(
            matches!(resp, ServerMessage::Closed),
            "Second close: expected Closed, got {:?}",
            resp
        );

        drop(handle);
        drop(bridge_thread);
    }

    // ---- Robot ID boundary edge cases ----

    #[tokio::test]
    async fn test_edge_connect_nonexistent_robot() {
        // Connect to robot_id that does not exist.
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;
        let (mut w, mut r) = tcp_connect(tcp_port).await;

        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 999 }).await;
        match &resp {
            ServerMessage::Error { message } => {
                assert!(
                    message.contains("invalid robot_id"),
                    "should mention invalid robot_id, got: {}",
                    message
                );
            }
            other => panic!("Expected Error for nonexistent robot, got {:?}", other),
        }

        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_edge_connect_robot_id_usize_max() {
        // Connect to robot_id = usize::MAX. Should get an error, not a panic.
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;
        let (mut w, mut r) = tcp_connect(tcp_port).await;

        let resp = tcp_send_recv(
            &mut w,
            &mut r,
            &ClientMessage::Connect {
                robot_id: usize::MAX,
            },
        )
        .await;
        match &resp {
            ServerMessage::Error { message } => {
                assert!(
                    message.contains("invalid robot_id"),
                    "usize::MAX robot_id should yield error, got: {}",
                    message
                );
            }
            other => panic!("Expected Error for usize::MAX robot, got {:?}", other),
        }

        drop(handle);
        drop(bridge_thread);
    }

    // ---- Disconnection edge cases ----

    #[tokio::test]
    async fn test_edge_tcp_abrupt_disconnect() {
        // Connect, start a session, then drop the TCP stream abruptly.
        // The server should handle this without panicking and free the
        // connection slot.
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;

        {
            let (mut w, mut r) = tcp_connect(tcp_port).await;
            tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;

            // Step a few times to build state.
            let action = RobotAction {
                motor_velocities: vec![1.0, -1.0],
                gripper_commands: vec![],
            };
            tcp_send_recv(
                &mut w,
                &mut r,
                &ClientMessage::Step {
                    action: action.clone(),
                },
            )
            .await;

            // Abruptly drop without sending Close.
        }

        // Wait for server to detect the disconnect.
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;

        // Should still be able to connect a new client.
        let (mut w2, mut r2) = tcp_connect(tcp_port).await;
        let resp = tcp_send_recv(&mut w2, &mut r2, &ClientMessage::Connect { robot_id: 0 }).await;
        assert!(
            matches!(resp, ServerMessage::Connected { .. }),
            "New client should connect after abrupt disconnect, got {:?}",
            resp
        );

        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_edge_ws_abrupt_disconnect() {
        // Same as TCP but for WebSocket.
        let (handle, bridge_thread) = start_test_server(1);
        let ws_port = handle.status().ws_port;

        {
            let (mut write, mut read) = ws_connect(ws_port).await;
            ws_send_recv(
                &mut write,
                &mut read,
                &ClientMessage::Connect { robot_id: 0 },
            )
            .await;

            // Drop without close frame.
        }

        tokio::time::sleep(std::time::Duration::from_millis(150)).await;

        // New client should connect fine.
        let (mut write2, mut read2) = ws_connect(ws_port).await;
        let resp = ws_send_recv(
            &mut write2,
            &mut read2,
            &ClientMessage::Connect { robot_id: 0 },
        )
        .await;
        assert!(
            matches!(resp, ServerMessage::Connected { .. }),
            "New WS client should connect after abrupt disconnect, got {:?}",
            resp
        );

        drop(handle);
        drop(bridge_thread);
    }

    // ---- Server lifecycle edge cases ----

    #[tokio::test]
    async fn test_edge_server_double_stop() {
        // Calling stop() twice should not panic.
        let (bridge_server, _bridge_client) = create_bridge();
        let config = AgentServerConfig {
            tcp_port: 0,
            ws_port: 0,
            max_connections: 16,
            enabled: true,
        };
        let mut handle = start_agent_server(config, bridge_server);
        handle.stop();
        assert!(
            !handle.status().running,
            "should not be running after first stop"
        );

        // Second stop should be a no-op, not panic.
        handle.stop();
        assert!(
            !handle.status().running,
            "should still not be running after second stop"
        );
    }

    #[tokio::test]
    async fn test_edge_status_after_drop() {
        // AgentServerHandle implements Drop which calls stop().
        // Verify that status reflects stopped state after drop.
        let (bridge_server, _bridge_client) = create_bridge();
        let config = AgentServerConfig {
            tcp_port: 0,
            ws_port: 0,
            max_connections: 16,
            enabled: true,
        };
        let handle = start_agent_server(config, bridge_server);
        let tcp_port = handle.status().tcp_port;
        let ws_port = handle.status().ws_port;
        assert!(tcp_port > 0);
        assert!(ws_port > 0);

        drop(handle);

        // After drop, attempting to connect should fail.
        let _result = tokio::net::TcpStream::connect(("127.0.0.1", tcp_port)).await;
        // Connection might succeed briefly or fail depending on OS cleanup timing.
        // The important thing is the server stops accepting new connections.
        // We just verify drop didn't panic.
    }

    // ---- Bridge edge cases ----

    #[tokio::test]
    async fn test_edge_bridge_command_after_client_dropped() {
        // Drop the SimBridgeClient while a command is pending.
        // The server-side send_command should get an error.
        let (bridge_server, bridge_client) = create_bridge();

        // Drop the client immediately.
        drop(bridge_client);

        // send_command should fail because the channel is closed.
        let result = bridge_server
            .send_command(crate::agent::bridge::SimCommand::GetObservation { robot_id: 0 })
            .await;
        assert!(
            result.is_err(),
            "send_command should fail after client dropped"
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("channel closed") || err.contains("channel"),
            "error should mention channel, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_edge_bridge_step_invalid_robot() {
        // Step on a robot_id that does not exist.
        let (bridge_server, mut bridge_client) = create_bridge();
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        manager.add_robot(def, Mat4::IDENTITY);

        let handle = tokio::spawn(async move {
            bridge_server
                .send_command(crate::agent::bridge::SimCommand::Step {
                    robot_id: 99,
                    action: RobotAction {
                        motor_velocities: vec![1.0, -1.0],
                        gripper_commands: vec![],
                    },
                })
                .await
        });

        tokio::task::yield_now().await;
        bridge_client.process_pending(&mut manager, &[]);

        let response = handle.await.unwrap().unwrap();
        match response {
            crate::agent::bridge::SimResponse::Error { message } => {
                assert!(
                    message.contains("invalid robot_id"),
                    "step on nonexistent robot should say invalid robot_id, got: {}",
                    message
                );
            }
            other => panic!("Expected Error for step on invalid robot, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_edge_bridge_reset_invalid_robot() {
        // Reset on a robot_id that does not exist.
        let (bridge_server, mut bridge_client) = create_bridge();
        let mut manager = RobotManager::new();

        let handle = tokio::spawn(async move {
            bridge_server
                .send_command(crate::agent::bridge::SimCommand::Reset { robot_id: 0 })
                .await
        });

        tokio::task::yield_now().await;
        bridge_client.process_pending(&mut manager, &[]);

        let response = handle.await.unwrap().unwrap();
        assert!(
            matches!(response, crate::agent::bridge::SimResponse::Error { .. }),
            "reset on nonexistent robot should return Error, got {:?}",
            response
        );
    }

    #[tokio::test]
    async fn test_edge_bridge_remove_invalid_robot() {
        // Remove a robot_id that does not exist.
        let (bridge_server, mut bridge_client) = create_bridge();
        let mut manager = RobotManager::new();

        let handle = tokio::spawn(async move {
            bridge_server
                .send_command(crate::agent::bridge::SimCommand::RemoveRobot { robot_id: 42 })
                .await
        });

        tokio::task::yield_now().await;
        bridge_client.process_pending(&mut manager, &[]);

        let response = handle.await.unwrap().unwrap();
        assert!(
            matches!(response, crate::agent::bridge::SimResponse::Error { .. }),
            "remove nonexistent robot should return Error, got {:?}",
            response
        );
    }

    #[tokio::test]
    async fn test_edge_bridge_get_spaces_invalid_robot() {
        // GetSpaces for a robot that does not exist.
        let (bridge_server, mut bridge_client) = create_bridge();
        let mut manager = RobotManager::new();

        let handle = tokio::spawn(async move {
            bridge_server
                .send_command(crate::agent::bridge::SimCommand::GetSpaces { robot_id: 5 })
                .await
        });

        tokio::task::yield_now().await;
        bridge_client.process_pending(&mut manager, &[]);

        let response = handle.await.unwrap().unwrap();
        assert!(
            matches!(response, crate::agent::bridge::SimResponse::Error { .. }),
            "GetSpaces on nonexistent robot should return Error, got {:?}",
            response
        );
    }

    #[tokio::test]
    async fn test_edge_bridge_step_count_resets_properly() {
        // Step robot 0 several times, step robot 1 once, reset robot 0,
        // step robot 0 again. Verify step counts are independent per robot.
        // Uses end-to-end TCP to avoid accessing private bridge fields.
        let (handle, bridge_thread) = start_test_server(2);
        let tcp_port = handle.status().tcp_port;

        // Agent for robot 0.
        let (mut w0, mut r0) = tcp_connect(tcp_port).await;
        tcp_send_recv(&mut w0, &mut r0, &ClientMessage::Connect { robot_id: 0 }).await;

        // Agent for robot 1.
        let (mut w1, mut r1) = tcp_connect(tcp_port).await;
        tcp_send_recv(&mut w1, &mut r1, &ClientMessage::Connect { robot_id: 1 }).await;

        let action = RobotAction {
            motor_velocities: vec![1.0, -1.0],
            gripper_commands: vec![],
        };

        // Step robot 0 three times.
        for i in 1..=3u64 {
            let resp = tcp_send_recv(
                &mut w0,
                &mut r0,
                &ClientMessage::Step {
                    action: action.clone(),
                },
            )
            .await;
            match &resp {
                ServerMessage::Observation { step_count, .. } => {
                    assert_eq!(
                        *step_count, i,
                        "robot 0 step {} should have step_count={}",
                        i, i
                    );
                }
                other => panic!("Expected Observation, got {:?}", other),
            }
        }

        // Step robot 1 once.
        let resp = tcp_send_recv(
            &mut w1,
            &mut r1,
            &ClientMessage::Step {
                action: action.clone(),
            },
        )
        .await;
        match &resp {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(*step_count, 1, "robot 1 step_count should be 1");
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        // Reset robot 0.
        let resp = tcp_send_recv(&mut w0, &mut r0, &ClientMessage::Reset).await;
        match &resp {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(*step_count, 0, "robot 0 step_count should be 0 after reset");
            }
            other => panic!("Expected Observation after reset, got {:?}", other),
        }

        // Step robot 0 again - step_count should be 1.
        let resp = tcp_send_recv(
            &mut w0,
            &mut r0,
            &ClientMessage::Step {
                action: action.clone(),
            },
        )
        .await;
        match &resp {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(
                    *step_count, 1,
                    "robot 0 step_count should be 1 after reset+step"
                );
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        // Robot 1 should still be at step_count 1 (unaffected by robot 0 reset).
        let resp = tcp_send_recv(&mut w1, &mut r1, &ClientMessage::Observe).await;
        match &resp {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(
                    *step_count, 1,
                    "robot 1 step_count should still be 1 after robot 0 reset"
                );
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        drop(handle);
        drop(bridge_thread);
    }

    // ---- Protocol serialization edge cases ----

    #[test]
    fn test_edge_protocol_nan_in_observation() {
        // BUG DISCOVERED: serde_json silently converts NaN/Inf to `null` during
        // serialization (it does not error!). This means a NaN-poisoned
        // observation will serialize successfully, but the receiving client
        // will fail to deserialize `null` as `f32`.
        //
        // This documents the dangerous behavior: NaN silently corrupts the
        // wire protocol.
        use crate::robot::state::{GymRobotState, GymSensorReadings};

        let state = GymRobotState {
            joint_positions: vec![f32::NAN, 0.5],
            joint_velocities: vec![0.0, f32::INFINITY],
            sensor_readings: GymSensorReadings {
                distances: vec![],
                contacts: vec![],
                imu: vec![],
                camera_visible: vec![],
            },
            gripper_states: vec![],
            combat: None,
        };

        let msg = ServerMessage::Observation {
            state,
            reward: 0.0,
            done: false,
            step_count: 1,
            messages: vec![],
            hit_events: vec![],
        };

        // serde_json succeeds in serializing but produces `null` for NaN/Inf.
        let json = serde_json::to_string(&msg)
            .expect("serde_json should serialize NaN as null without error");
        assert!(
            json.contains("null"),
            "NaN should be serialized as null in JSON: {}",
            json
        );

        // But deserialization fails because null cannot be parsed as f32.
        let result = serde_json::from_str::<ServerMessage>(&json);
        assert!(
            result.is_err(),
            "Deserialization of NaN-as-null should fail. \
             NaN in observations breaks the wire protocol. \
             This documents the need for input validation on actions."
        );
    }

    #[test]
    fn test_edge_protocol_empty_action_roundtrip() {
        // Empty motor_velocities and gripper_commands should roundtrip.
        let action = RobotAction {
            motor_velocities: vec![],
            gripper_commands: vec![],
        };
        let msg = ClientMessage::Step { action };
        let json = serde_json::to_string(&msg).unwrap();
        let deser: ClientMessage = serde_json::from_str(&json).unwrap();
        match deser {
            ClientMessage::Step { action } => {
                assert!(action.motor_velocities.is_empty());
                assert!(action.gripper_commands.is_empty());
            }
            other => panic!("Expected Step, got {:?}", other),
        }
    }

    #[test]
    fn test_edge_protocol_large_robot_id() {
        // Very large robot_id should serialize/deserialize correctly.
        let msg = ClientMessage::Connect {
            robot_id: usize::MAX,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deser: ClientMessage = serde_json::from_str(&json).unwrap();
        match deser {
            ClientMessage::Connect { robot_id } => {
                assert_eq!(robot_id, usize::MAX);
            }
            other => panic!("Expected Connect, got {:?}", other),
        }
    }

    // ---- Config edge cases ----

    #[test]
    fn test_edge_config_zero_max_connections() {
        let config = AgentServerConfig {
            tcp_port: 0,
            ws_port: 0,
            max_connections: 0,
            enabled: true,
        };
        assert_eq!(config.max_connections, 0);
        // A server with 0 max_connections would reject all connections.
        // This is valid configuration, not a crash.
    }

    #[test]
    fn test_edge_config_clone() {
        let config = AgentServerConfig {
            tcp_port: 1234,
            ws_port: 5678,
            max_connections: 42,
            enabled: true,
        };
        let cloned = config.clone();
        assert_eq!(cloned.tcp_port, 1234);
        assert_eq!(cloned.ws_port, 5678);
        assert_eq!(cloned.max_connections, 42);
        assert!(cloned.enabled);
    }

    #[test]
    fn test_edge_status_debug() {
        // Verify Debug trait works for AgentServerStatus.
        let status = AgentServerStatus {
            tcp_port: 9001,
            ws_port: 9002,
            tcp_connections: 3,
            ws_connections: 2,
            running: true,
        };
        let debug_str = format!("{:?}", status);
        assert!(debug_str.contains("9001"));
        assert!(debug_str.contains("9002"));
        assert!(debug_str.contains("true"));
    }

    // ---- Concurrency edge cases ----

    #[tokio::test]
    async fn test_edge_concurrent_steps_two_robots() {
        // Two clients send steps concurrently to different robots.
        // Both should succeed independently.
        let (handle, bridge_thread) = start_test_server(2);
        let tcp_port = handle.status().tcp_port;

        let (mut w1, mut r1) = tcp_connect(tcp_port).await;
        let (mut w2, mut r2) = tcp_connect(tcp_port).await;

        tcp_send_recv(&mut w1, &mut r1, &ClientMessage::Connect { robot_id: 0 }).await;
        tcp_send_recv(&mut w2, &mut r2, &ClientMessage::Connect { robot_id: 1 }).await;

        // Send steps concurrently using join.
        let action1 = RobotAction {
            motor_velocities: vec![1.0, 0.0],
            gripper_commands: vec![],
        };
        let action2 = RobotAction {
            motor_velocities: vec![0.0, -1.0],
            gripper_commands: vec![],
        };

        let msg1 = ClientMessage::Step {
            action: action1.clone(),
        };
        let msg2 = ClientMessage::Step {
            action: action2.clone(),
        };

        let (resp1, resp2) = tokio::join!(
            tcp_send_recv(&mut w1, &mut r1, &msg1),
            tcp_send_recv(&mut w2, &mut r2, &msg2),
        );

        assert!(
            matches!(resp1, ServerMessage::Observation { step_count: 1, .. }),
            "Concurrent step robot 0: expected step_count=1, got {:?}",
            resp1
        );
        assert!(
            matches!(resp2, ServerMessage::Observation { step_count: 1, .. }),
            "Concurrent step robot 1: expected step_count=1, got {:?}",
            resp2
        );

        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_edge_many_connections_and_disconnects() {
        // Rapidly connect and disconnect 10 clients to stress connection tracking.
        let (handle, bridge_thread) = start_test_server(1);
        let tcp_port = handle.status().tcp_port;

        for i in 0..10 {
            let (mut w, mut r) = tcp_connect(tcp_port).await;
            let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;
            assert!(
                matches!(resp, ServerMessage::Connected { .. }),
                "Connection {} should succeed, got {:?}",
                i,
                resp
            );
            let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Close).await;
            assert!(
                matches!(resp, ServerMessage::Closed),
                "Close {} should succeed, got {:?}",
                i,
                resp
            );
            // Drop the connection.
        }

        // Brief pause to let all connection guards fire.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Final connection should still work.
        let (mut w, mut r) = tcp_connect(tcp_port).await;
        let resp = tcp_send_recv(&mut w, &mut r, &ClientMessage::Connect { robot_id: 0 }).await;
        assert!(
            matches!(resp, ServerMessage::Connected { .. }),
            "Final connection after 10 cycles should work, got {:?}",
            resp
        );

        drop(handle);
        drop(bridge_thread);
    }

    // ---- WebSocket-specific edge cases ----

    #[tokio::test]
    async fn test_edge_ws_malformed_json() {
        // Send malformed JSON over WebSocket.
        let (handle, bridge_thread) = start_test_server(1);
        let ws_port = handle.status().ws_port;

        let (mut write, mut read) = ws_connect(ws_port).await;

        use futures_util::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite::Message as WsMsg;

        // Send invalid JSON as text message.
        write
            .send(WsMsg::Text("this is not json".into()))
            .await
            .unwrap();

        // Should get an error back.
        loop {
            match read.next().await {
                Some(Ok(WsMsg::Text(text))) => {
                    let msg: ServerMessage = serde_json::from_str(&text).unwrap();
                    match msg {
                        ServerMessage::Error { message } => {
                            assert!(
                                message.contains("invalid JSON"),
                                "WS malformed JSON error, got: {}",
                                message
                            );
                        }
                        other => panic!("Expected Error, got {:?}", other),
                    }
                    break;
                }
                Some(Ok(WsMsg::Ping(_) | WsMsg::Pong(_))) => continue,
                other => panic!("Expected Text error, got {:?}", other),
            }
        }

        // Connection should still be alive — send a valid Connect.
        let resp = ws_send_recv(
            &mut write,
            &mut read,
            &ClientMessage::Connect { robot_id: 0 },
        )
        .await;
        assert!(
            matches!(resp, ServerMessage::Connected { .. }),
            "WS should still work after malformed JSON, got {:?}",
            resp
        );

        drop(handle);
        drop(bridge_thread);
    }

    #[tokio::test]
    async fn test_edge_ws_empty_text_message() {
        // Send an empty text message over WebSocket.
        let (handle, bridge_thread) = start_test_server(1);
        let ws_port = handle.status().ws_port;

        let (mut write, mut read) = ws_connect(ws_port).await;

        use futures_util::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite::Message as WsMsg;

        write.send(WsMsg::Text("".into())).await.unwrap();

        // Should get an error (empty string is not valid JSON).
        loop {
            match read.next().await {
                Some(Ok(WsMsg::Text(text))) => {
                    let msg: ServerMessage = serde_json::from_str(&text).unwrap();
                    assert!(
                        matches!(msg, ServerMessage::Error { .. }),
                        "Empty WS text should produce Error, got {:?}",
                        msg
                    );
                    break;
                }
                Some(Ok(WsMsg::Ping(_) | WsMsg::Pong(_))) => continue,
                other => panic!("Expected Text error for empty message, got {:?}", other),
            }
        }

        drop(handle);
        drop(bridge_thread);
    }
}

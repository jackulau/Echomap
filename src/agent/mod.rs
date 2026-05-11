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
                let tcp_server =
                    TcpAgentServer::bind(config.tcp_port, bridge_server.clone(), config.max_connections)
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
}

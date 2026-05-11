use crate::agent::bridge::{SimBridgeServer, SimCommand, SimResponse};
use crate::agent::protocol::{ClientMessage, ServerMessage};
use crate::robot::state::{ActionSpace, ObservationSpace};

/// Per-connection session handler for a single agent.
///
/// Tracks robot assignment, step count, and observation/action spaces for
/// the connected robot. Translates between protocol messages and bridge
/// commands.
pub struct AgentSession {
    /// The robot this session is controlling. `None` until `Connect` is handled.
    robot_id: Option<usize>,
    /// Number of steps taken since the last reset (or since connection).
    step_count: u64,
    /// Handle to send commands to the simulation bridge.
    bridge: SimBridgeServer,
    /// Cached observation space for the connected robot.
    observation_space: Option<ObservationSpace>,
    /// Cached action space for the connected robot.
    action_space: Option<ActionSpace>,
}

impl AgentSession {
    /// Create a new session with the given bridge handle.
    pub fn new(bridge: SimBridgeServer) -> Self {
        Self {
            robot_id: None,
            step_count: 0,
            bridge,
            observation_space: None,
            action_space: None,
        }
    }

    /// Handle an incoming client message, returning the appropriate server response.
    pub async fn handle_message(&mut self, msg: ClientMessage) -> ServerMessage {
        match msg {
            ClientMessage::Connect { robot_id } => self.handle_connect(robot_id).await,
            ClientMessage::Reset => self.handle_reset().await,
            ClientMessage::Step { action } => self.handle_step(action).await,
            ClientMessage::Observe => self.handle_observe().await,
            ClientMessage::Close => self.handle_close(),
        }
    }

    async fn handle_connect(&mut self, robot_id: usize) -> ServerMessage {
        // Double-connect is an error.
        if self.robot_id.is_some() {
            return ServerMessage::Error {
                message: "already connected to a robot".to_string(),
            };
        }

        // Fetch spaces from the bridge to validate the robot exists and
        // cache observation/action space info.
        match self
            .bridge
            .send_command(SimCommand::GetSpaces { robot_id })
            .await
        {
            Ok(SimResponse::Spaces {
                observation_space,
                action_space,
            }) => {
                self.robot_id = Some(robot_id);
                self.step_count = 0;
                self.observation_space = Some(observation_space.clone());
                self.action_space = Some(action_space.clone());
                ServerMessage::Connected {
                    observation_space,
                    action_space,
                }
            }
            Ok(SimResponse::Error { message }) => ServerMessage::Error { message },
            Ok(_) => ServerMessage::Error {
                message: "unexpected response from bridge".to_string(),
            },
            Err(e) => ServerMessage::Error { message: e },
        }
    }

    async fn handle_reset(&mut self) -> ServerMessage {
        let robot_id = match self.robot_id {
            Some(id) => id,
            None => {
                return ServerMessage::Error {
                    message: "not connected to a robot".to_string(),
                }
            }
        };

        match self
            .bridge
            .send_command(SimCommand::Reset { robot_id })
            .await
        {
            Ok(SimResponse::Reset { state }) => {
                self.step_count = 0;
                ServerMessage::Observation {
                    state,
                    reward: 0.0,
                    done: false,
                    step_count: 0,
                }
            }
            Ok(SimResponse::Error { message }) => ServerMessage::Error { message },
            Ok(_) => ServerMessage::Error {
                message: "unexpected response from bridge".to_string(),
            },
            Err(e) => ServerMessage::Error { message: e },
        }
    }

    async fn handle_step(
        &mut self,
        action: crate::robot::state::RobotAction,
    ) -> ServerMessage {
        let robot_id = match self.robot_id {
            Some(id) => id,
            None => {
                return ServerMessage::Error {
                    message: "not connected to a robot".to_string(),
                }
            }
        };

        match self
            .bridge
            .send_command(SimCommand::Step { robot_id, action })
            .await
        {
            Ok(SimResponse::Stepped { state, .. }) => {
                self.step_count += 1;
                ServerMessage::Observation {
                    state,
                    reward: 0.0,
                    done: false,
                    step_count: self.step_count,
                }
            }
            Ok(SimResponse::Error { message }) => ServerMessage::Error { message },
            Ok(_) => ServerMessage::Error {
                message: "unexpected response from bridge".to_string(),
            },
            Err(e) => ServerMessage::Error { message: e },
        }
    }

    async fn handle_observe(&mut self) -> ServerMessage {
        let robot_id = match self.robot_id {
            Some(id) => id,
            None => {
                return ServerMessage::Error {
                    message: "not connected to a robot".to_string(),
                }
            }
        };

        match self
            .bridge
            .send_command(SimCommand::GetObservation { robot_id })
            .await
        {
            Ok(SimResponse::Observation { state }) => ServerMessage::Observation {
                state,
                reward: 0.0,
                done: false,
                step_count: self.step_count,
            },
            Ok(SimResponse::Error { message }) => ServerMessage::Error { message },
            Ok(_) => ServerMessage::Error {
                message: "unexpected response from bridge".to_string(),
            },
            Err(e) => ServerMessage::Error { message: e },
        }
    }

    fn handle_close(&mut self) -> ServerMessage {
        self.robot_id = None;
        self.observation_space = None;
        self.action_space = None;
        self.step_count = 0;
        ServerMessage::Closed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::bridge::create_bridge;
    use crate::robot::definition::RobotDefinition;
    use crate::robot::state::RobotAction;
    use crate::robot::RobotManager;
    use glam::Mat4;

    /// Helper: create a RobotManager with one simple_arm(2) robot,
    /// a bridge pair, and spawn a background task that continuously
    /// processes bridge commands.
    fn setup_test_env() -> (AgentSession, tokio::task::JoinHandle<()>) {
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        manager.add_robot(def, Mat4::IDENTITY);

        let (server, mut client) = create_bridge();
        let session = AgentSession::new(server);

        // Spawn a background task to process commands on the client side.
        let handle = tokio::spawn(async move {
            loop {
                client.process_pending(&mut manager, &[]);
                tokio::task::yield_now().await;
            }
        });

        (session, handle)
    }

    #[tokio::test]
    async fn test_session_connect() {
        let (mut session, handle) = setup_test_env();

        let response = session
            .handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;

        match response {
            ServerMessage::Connected {
                observation_space,
                action_space,
            } => {
                assert_eq!(
                    observation_space.num_joint_positions, 2,
                    "simple_arm(2) should have 2 joint positions"
                );
                assert_eq!(
                    observation_space.num_joint_velocities, 2,
                    "simple_arm(2) should have 2 joint velocities"
                );
                assert_eq!(
                    action_space.num_motors, 2,
                    "simple_arm(2) should have 2 motors"
                );
            }
            other => panic!("Expected Connected, got {:?}", other),
        }

        handle.abort();
    }

    #[tokio::test]
    async fn test_session_step_increments_count() {
        let (mut session, handle) = setup_test_env();

        // Connect first
        session
            .handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;

        let action = RobotAction {
            motor_velocities: vec![1.0, -0.5],
            gripper_commands: vec![],
        };

        // Step 3 times, verify step_count increments
        for expected_count in 1..=3 {
            let response = session
                .handle_message(ClientMessage::Step {
                    action: action.clone(),
                })
                .await;

            match response {
                ServerMessage::Observation { step_count, .. } => {
                    assert_eq!(
                        step_count, expected_count,
                        "step_count should be {} after {} step(s)",
                        expected_count, expected_count
                    );
                }
                other => panic!("Expected Observation, got {:?}", other),
            }
        }

        handle.abort();
    }

    #[tokio::test]
    async fn test_session_reset_clears_count() {
        let (mut session, handle) = setup_test_env();

        // Connect
        session
            .handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;

        // Step once
        let action = RobotAction {
            motor_velocities: vec![1.0, -0.5],
            gripper_commands: vec![],
        };
        let response = session
            .handle_message(ClientMessage::Step {
                action: action.clone(),
            })
            .await;
        match &response {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(*step_count, 1, "step_count should be 1 after first step");
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        // Reset
        let response = session.handle_message(ClientMessage::Reset).await;
        match response {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(step_count, 0, "step_count should be 0 after reset");
            }
            other => panic!("Expected Observation after reset, got {:?}", other),
        }

        handle.abort();
    }

    #[tokio::test]
    async fn test_session_observe_no_step() {
        let (mut session, handle) = setup_test_env();

        // Connect
        session
            .handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;

        // Step once so step_count = 1
        let action = RobotAction {
            motor_velocities: vec![1.0, -0.5],
            gripper_commands: vec![],
        };
        session
            .handle_message(ClientMessage::Step { action })
            .await;

        // Observe should return step_count=1 (not incremented)
        let response = session.handle_message(ClientMessage::Observe).await;
        match response {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(
                    step_count, 1,
                    "observe should not increment step_count, expected 1"
                );
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        // Observe again — still 1
        let response = session.handle_message(ClientMessage::Observe).await;
        match response {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(
                    step_count, 1,
                    "observe should still be 1 after second observe"
                );
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        handle.abort();
    }

    #[tokio::test]
    async fn test_session_close() {
        let (mut session, handle) = setup_test_env();

        // Connect
        session
            .handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;

        // Close
        let response = session.handle_message(ClientMessage::Close).await;
        match response {
            ServerMessage::Closed => {}
            other => panic!("Expected Closed, got {:?}", other),
        }

        handle.abort();
    }

    #[tokio::test]
    async fn test_session_step_before_connect() {
        let (mut session, handle) = setup_test_env();

        let action = RobotAction {
            motor_velocities: vec![1.0],
            gripper_commands: vec![],
        };
        let response = session
            .handle_message(ClientMessage::Step { action })
            .await;

        match response {
            ServerMessage::Error { message } => {
                assert!(
                    message.contains("not connected"),
                    "error should mention not connected, got: {}",
                    message
                );
            }
            other => panic!("Expected Error, got {:?}", other),
        }

        // Also test observe before connect
        let response = session.handle_message(ClientMessage::Observe).await;
        match response {
            ServerMessage::Error { message } => {
                assert!(
                    message.contains("not connected"),
                    "observe error should mention not connected, got: {}",
                    message
                );
            }
            other => panic!("Expected Error for observe before connect, got {:?}", other),
        }

        // Also test reset before connect
        let response = session.handle_message(ClientMessage::Reset).await;
        match response {
            ServerMessage::Error { message } => {
                assert!(
                    message.contains("not connected"),
                    "reset error should mention not connected, got: {}",
                    message
                );
            }
            other => panic!("Expected Error for reset before connect, got {:?}", other),
        }

        handle.abort();
    }

    #[tokio::test]
    async fn test_session_double_connect() {
        let (mut session, handle) = setup_test_env();

        // First connect succeeds
        let response = session
            .handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;
        assert!(
            matches!(response, ServerMessage::Connected { .. }),
            "first connect should succeed"
        );

        // Second connect fails
        let response = session
            .handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;
        match response {
            ServerMessage::Error { message } => {
                assert!(
                    message.contains("already connected"),
                    "error should mention already connected, got: {}",
                    message
                );
            }
            other => panic!("Expected Error for double connect, got {:?}", other),
        }

        handle.abort();
    }

    #[tokio::test]
    async fn test_session_invalid_robot() {
        let (mut session, handle) = setup_test_env();

        // Connect to a nonexistent robot
        let response = session
            .handle_message(ClientMessage::Connect { robot_id: 99 })
            .await;

        match response {
            ServerMessage::Error { message } => {
                assert!(
                    message.contains("invalid robot_id"),
                    "error should mention invalid robot_id, got: {}",
                    message
                );
            }
            other => panic!("Expected Error for invalid robot, got {:?}", other),
        }

        handle.abort();
    }
}

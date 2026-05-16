use crate::agent::bridge::{SimBridgeServer, SimCommand, SimResponse};
use crate::agent::protocol::{ClientMessage, ServerMessage};
use crate::robot::boxing::BoxingMatchState;
use crate::robot::state::{ActionSpace, ObservationSpace};

fn compute_reward_done(
    match_state: &Option<BoxingMatchState>,
    robot_id: usize,
) -> (f32, bool) {
    let Some(ms) = match_state else {
        return (0.0, false);
    };
    let done = ms.phase == "match_end";
    let reward = if ms.your_robot == 0 {
        ms.total_score_a as f32 - ms.total_score_b as f32
    } else {
        ms.total_score_b as f32 - ms.total_score_a as f32
    };
    let _ = robot_id;
    (reward, done)
}

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
            ClientMessage::SendMessage {
                to_robot_id,
                content,
            } => self.handle_send_message(to_robot_id, content).await,
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
            Ok(SimResponse::Reset {
                state,
                messages,
                match_state,
            }) => {
                self.step_count = 0;
                let hit_events = state
                    .combat
                    .as_ref()
                    .map(|c| c.recent_hits.clone())
                    .unwrap_or_default();
                let (reward, done) = compute_reward_done(&match_state, robot_id);
                ServerMessage::Observation {
                    state,
                    reward,
                    done,
                    step_count: 0,
                    messages,
                    hit_events,
                    match_state,
                }
            }
            Ok(SimResponse::Error { message }) => ServerMessage::Error { message },
            Ok(_) => ServerMessage::Error {
                message: "unexpected response from bridge".to_string(),
            },
            Err(e) => ServerMessage::Error { message: e },
        }
    }

    async fn handle_step(&mut self, action: crate::robot::state::RobotAction) -> ServerMessage {
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
            Ok(SimResponse::Stepped {
                state,
                messages,
                match_state,
                ..
            }) => {
                self.step_count += 1;
                let hit_events = state
                    .combat
                    .as_ref()
                    .map(|c| c.recent_hits.clone())
                    .unwrap_or_default();
                let (reward, done) = compute_reward_done(&match_state, robot_id);
                ServerMessage::Observation {
                    state,
                    reward,
                    done,
                    step_count: self.step_count,
                    messages,
                    hit_events,
                    match_state,
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
            Ok(SimResponse::Observation {
                state,
                messages,
                match_state,
            }) => {
                let hit_events = state
                    .combat
                    .as_ref()
                    .map(|c| c.recent_hits.clone())
                    .unwrap_or_default();
                let (reward, done) = compute_reward_done(&match_state, robot_id);
                ServerMessage::Observation {
                    state,
                    reward,
                    done,
                    step_count: self.step_count,
                    messages,
                    hit_events,
                    match_state,
                }
            }
            Ok(SimResponse::Error { message }) => ServerMessage::Error { message },
            Ok(_) => ServerMessage::Error {
                message: "unexpected response from bridge".to_string(),
            },
            Err(e) => ServerMessage::Error { message: e },
        }
    }

    async fn handle_send_message(&mut self, to_robot_id: usize, content: String) -> ServerMessage {
        let from_robot_id = match self.robot_id {
            Some(id) => id,
            None => {
                return ServerMessage::Error {
                    message: "not connected to a robot".to_string(),
                }
            }
        };

        match self
            .bridge
            .send_command(SimCommand::SendMessage {
                from_robot_id,
                to_robot_id,
                content,
            })
            .await
        {
            Ok(SimResponse::MessageSent) => ServerMessage::MessageSent,
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
            base_velocity: [0.0, 0.0],
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
            base_velocity: [0.0, 0.0],
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
            base_velocity: [0.0, 0.0],
        };
        session.handle_message(ClientMessage::Step { action }).await;

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
            base_velocity: [0.0, 0.0],
        };
        let response = session.handle_message(ClientMessage::Step { action }).await;

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

    // ---- Edge case tests ----

    #[tokio::test]
    async fn test_session_close_then_step() {
        let (mut session, handle) = setup_test_env();

        // Connect
        session
            .handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;

        // Close
        let resp = session.handle_message(ClientMessage::Close).await;
        assert!(matches!(resp, ServerMessage::Closed));

        // Step after close should error (not connected)
        let action = RobotAction {
            motor_velocities: vec![1.0, -0.5],
            gripper_commands: vec![],
            base_velocity: [0.0, 0.0],
        };
        let resp = session.handle_message(ClientMessage::Step { action }).await;
        match resp {
            ServerMessage::Error { message } => {
                assert!(
                    message.contains("not connected"),
                    "step after close should say not connected, got: {}",
                    message
                );
            }
            other => panic!("Expected Error for step after close, got {:?}", other),
        }

        handle.abort();
    }

    #[tokio::test]
    async fn test_session_close_then_observe() {
        let (mut session, handle) = setup_test_env();

        // Connect
        session
            .handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;

        // Close
        session.handle_message(ClientMessage::Close).await;

        // Observe after close should error
        let resp = session.handle_message(ClientMessage::Observe).await;
        match resp {
            ServerMessage::Error { message } => {
                assert!(
                    message.contains("not connected"),
                    "observe after close should say not connected, got: {}",
                    message
                );
            }
            other => panic!("Expected Error for observe after close, got {:?}", other),
        }

        handle.abort();
    }

    #[tokio::test]
    async fn test_session_close_then_reset() {
        let (mut session, handle) = setup_test_env();

        // Connect
        session
            .handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;

        // Close
        session.handle_message(ClientMessage::Close).await;

        // Reset after close should error
        let resp = session.handle_message(ClientMessage::Reset).await;
        match resp {
            ServerMessage::Error { message } => {
                assert!(
                    message.contains("not connected"),
                    "reset after close should say not connected, got: {}",
                    message
                );
            }
            other => panic!("Expected Error for reset after close, got {:?}", other),
        }

        handle.abort();
    }

    #[tokio::test]
    async fn test_session_close_then_reconnect() {
        let (mut session, handle) = setup_test_env();

        // Connect
        let resp = session
            .handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;
        assert!(matches!(resp, ServerMessage::Connected { .. }));

        // Step once
        let action = RobotAction {
            motor_velocities: vec![1.0, -0.5],
            gripper_commands: vec![],
            base_velocity: [0.0, 0.0],
        };
        session.handle_message(ClientMessage::Step { action }).await;

        // Close
        session.handle_message(ClientMessage::Close).await;

        // Reconnect to same robot -- should succeed and reset step_count
        let resp = session
            .handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;
        assert!(
            matches!(resp, ServerMessage::Connected { .. }),
            "reconnect after close should succeed, got {:?}",
            resp
        );

        // Step count should be fresh (0 before step)
        let action2 = RobotAction {
            motor_velocities: vec![0.5, 0.5],
            gripper_commands: vec![],
            base_velocity: [0.0, 0.0],
        };
        let resp = session
            .handle_message(ClientMessage::Step { action: action2 })
            .await;
        match resp {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(
                    step_count, 1,
                    "step_count should be 1 after reconnect+step, got {}",
                    step_count
                );
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        handle.abort();
    }

    #[tokio::test]
    async fn test_session_double_close_idempotent() {
        let (mut session, handle) = setup_test_env();

        // Connect
        session
            .handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;

        // Close once
        let resp = session.handle_message(ClientMessage::Close).await;
        assert!(matches!(resp, ServerMessage::Closed));

        // Close again -- should still return Closed, not error
        let resp = session.handle_message(ClientMessage::Close).await;
        assert!(
            matches!(resp, ServerMessage::Closed),
            "double close should be idempotent, got {:?}",
            resp
        );

        handle.abort();
    }

    #[tokio::test]
    async fn test_session_close_without_connect() {
        let (mut session, handle) = setup_test_env();

        // Close without ever connecting
        let resp = session.handle_message(ClientMessage::Close).await;
        assert!(
            matches!(resp, ServerMessage::Closed),
            "close without connect should still return Closed, got {:?}",
            resp
        );

        handle.abort();
    }

    #[tokio::test]
    async fn test_session_connect_to_different_robot_after_close() {
        // Setup with 2 robots
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        manager.add_robot(def.clone(), Mat4::IDENTITY);
        manager.add_robot(def, Mat4::IDENTITY);

        let (server, mut client) = create_bridge();
        let mut session = AgentSession::new(server);

        let bg = tokio::spawn(async move {
            loop {
                client.process_pending(&mut manager, &[]);
                tokio::task::yield_now().await;
            }
        });

        // Connect to robot 0
        let resp = session
            .handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;
        assert!(matches!(resp, ServerMessage::Connected { .. }));

        // Close
        session.handle_message(ClientMessage::Close).await;

        // Connect to robot 1
        let resp = session
            .handle_message(ClientMessage::Connect { robot_id: 1 })
            .await;
        assert!(
            matches!(resp, ServerMessage::Connected { .. }),
            "should be able to connect to different robot after close, got {:?}",
            resp
        );

        bg.abort();
    }

    #[tokio::test]
    async fn test_session_step_count_independent_of_bridge_count() {
        let (mut session, handle) = setup_test_env();

        // Connect
        session
            .handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;

        // Step 5 times
        let action = RobotAction {
            motor_velocities: vec![1.0, 1.0],
            gripper_commands: vec![],
            base_velocity: [0.0, 0.0],
        };
        for _ in 0..5 {
            session
                .handle_message(ClientMessage::Step {
                    action: action.clone(),
                })
                .await;
        }

        // Reset
        let resp = session.handle_message(ClientMessage::Reset).await;
        match resp {
            ServerMessage::Observation {
                step_count,
                reward,
                done,
                ..
            } => {
                assert_eq!(step_count, 0, "step_count should be 0 after reset");
                assert!((reward - 0.0).abs() < 1e-6, "reward should be 0.0 on reset");
                assert!(!done, "done should be false on reset");
            }
            other => panic!("Expected Observation after reset, got {:?}", other),
        }

        // Step again -- count should start from 1
        let resp = session
            .handle_message(ClientMessage::Step {
                action: action.clone(),
            })
            .await;
        match resp {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(step_count, 1, "step after reset should be 1");
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        handle.abort();
    }

    #[tokio::test]
    async fn test_session_observe_returns_zero_reward() {
        let (mut session, handle) = setup_test_env();

        // Connect
        session
            .handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;

        // Observe should return reward=0.0 and done=false
        let resp = session.handle_message(ClientMessage::Observe).await;
        match resp {
            ServerMessage::Observation {
                reward,
                done,
                step_count,
                ..
            } => {
                assert!((reward - 0.0).abs() < 1e-6, "observe reward should be 0.0");
                assert!(!done, "observe done should be false");
                assert_eq!(
                    step_count, 0,
                    "observe step_count should be 0 before any steps"
                );
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        handle.abort();
    }

    // ---- Messaging tests ----

    fn setup_two_robot_env() -> (AgentSession, AgentSession, tokio::task::JoinHandle<()>) {
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        manager.add_robot(def.clone(), Mat4::IDENTITY);
        manager.add_robot(def, Mat4::IDENTITY);

        let (server, mut client) = create_bridge();
        let session0 = AgentSession::new(server.clone());
        let session1 = AgentSession::new(server);

        let handle = tokio::spawn(async move {
            loop {
                client.process_pending(&mut manager, &[]);
                tokio::task::yield_now().await;
            }
        });

        (session0, session1, handle)
    }

    #[tokio::test]
    async fn test_session_send_message() {
        let (mut s0, _s1, handle) = setup_two_robot_env();
        s0.handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;

        let resp = s0
            .handle_message(ClientMessage::SendMessage {
                to_robot_id: 1,
                content: "trash talk".into(),
            })
            .await;
        assert!(
            matches!(resp, ServerMessage::MessageSent),
            "Expected MessageSent, got {:?}",
            resp
        );
        handle.abort();
    }

    #[tokio::test]
    async fn test_session_send_message_not_connected() {
        let (mut s0, _s1, handle) = setup_two_robot_env();

        let resp = s0
            .handle_message(ClientMessage::SendMessage {
                to_robot_id: 1,
                content: "hello".into(),
            })
            .await;
        match resp {
            ServerMessage::Error { message } => {
                assert!(message.contains("not connected"));
            }
            other => panic!("Expected Error, got {:?}", other),
        }
        handle.abort();
    }

    #[tokio::test]
    async fn test_session_step_includes_messages() {
        let (mut s0, mut s1, handle) = setup_two_robot_env();
        s0.handle_message(ClientMessage::Connect { robot_id: 0 })
            .await;
        s1.handle_message(ClientMessage::Connect { robot_id: 1 })
            .await;

        // Robot 0 sends message to robot 1
        s0.handle_message(ClientMessage::SendMessage {
            to_robot_id: 1,
            content: "incoming".into(),
        })
        .await;

        // Robot 1 steps — should receive the message
        let action = RobotAction {
            motor_velocities: vec![0.0, 0.0],
            gripper_commands: vec![],
            base_velocity: [0.0, 0.0],
        };
        let resp = s1.handle_message(ClientMessage::Step { action }).await;
        match resp {
            ServerMessage::Observation { messages, .. } => {
                assert_eq!(messages.len(), 1);
                assert_eq!(messages[0].content, "incoming");
                assert_eq!(messages[0].from_robot_id, 0);
                assert_eq!(messages[0].to_robot_id, 1);
            }
            other => panic!("Expected Observation with messages, got {:?}", other),
        }
        handle.abort();
    }
}

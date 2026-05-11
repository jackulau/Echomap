use tokio::sync::{mpsc, oneshot};

use crate::robot::definition::RobotDefinition;
use crate::robot::state::{
    apply_action, ActionSpace, GymRobotState, ObservationSpace, RobotAction, RobotState,
};
use crate::robot::RobotManager;
use crate::scene::SceneObject;
use glam::Mat4;

/// A command sent from the agent server to the simulation main loop.
#[derive(Debug)]
pub enum SimCommand {
    AddRobot {
        definition: RobotDefinition,
        base_pose: [f32; 16],
    },
    Step {
        robot_id: usize,
        action: RobotAction,
    },
    GetObservation {
        robot_id: usize,
    },
    Reset {
        robot_id: usize,
    },
    RemoveRobot {
        robot_id: usize,
    },
    GetSpaces {
        robot_id: usize,
    },
}

/// A response sent from the simulation main loop back to the agent server.
#[derive(Debug)]
pub enum SimResponse {
    RobotAdded { robot_id: usize },
    Stepped { state: GymRobotState, step_count: u64 },
    Observation { state: GymRobotState },
    Reset { state: GymRobotState },
    Removed,
    Spaces {
        observation_space: ObservationSpace,
        action_space: ActionSpace,
    },
    Error { message: String },
}

/// Payload carried through the command channel: a command plus a oneshot
/// sender for the response.
type CommandEnvelope = (SimCommand, oneshot::Sender<SimResponse>);

/// Server-side handle held by the agent network server.
///
/// Provides an async `send_command` that enqueues a command and awaits the
/// response produced by the main-loop side (`SimBridgeClient`).
#[derive(Clone)]
pub struct SimBridgeServer {
    tx: mpsc::UnboundedSender<CommandEnvelope>,
}

impl SimBridgeServer {
    /// Send a command to the simulation and await the response.
    pub async fn send_command(&self, command: SimCommand) -> Result<SimResponse, String> {
        let (resp_tx, resp_rx) = oneshot::channel();
        self.tx
            .send((command, resp_tx))
            .map_err(|_| "bridge channel closed".to_string())?;
        resp_rx
            .await
            .map_err(|_| "response channel dropped".to_string())
    }
}

/// Client-side handle held by the eframe main loop.
///
/// `process_pending` is called each frame (non-blocking) to drain queued
/// commands, execute them against the `RobotManager`, and send responses
/// back via the per-command oneshot channels.
pub struct SimBridgeClient {
    rx: mpsc::UnboundedReceiver<CommandEnvelope>,
    /// Per-robot step counters. Indexed by robot_id.
    step_counts: Vec<u64>,
}

impl SimBridgeClient {
    /// Non-blocking drain of the command channel.
    ///
    /// For each pending command, executes the operation on `manager` and
    /// sends the result back through the oneshot. `scene_meshes` is passed
    /// to `RobotManager::step` when stepping the simulation.
    pub fn process_pending(
        &mut self,
        manager: &mut RobotManager,
        scene_meshes: &[SceneObject],
    ) {
        while let Ok((cmd, resp_tx)) = self.rx.try_recv() {
            let response = self.execute(cmd, manager, scene_meshes);
            // Ignore send errors — the caller may have timed out.
            let _ = resp_tx.send(response);
        }
    }

    /// Execute a single command against the manager.
    fn execute(
        &mut self,
        cmd: SimCommand,
        manager: &mut RobotManager,
        scene_meshes: &[SceneObject],
    ) -> SimResponse {
        match cmd {
            SimCommand::AddRobot {
                definition,
                base_pose,
            } => {
                let pose = Mat4::from_cols_array(&base_pose);
                let robot_id = manager.add_robot(definition, pose);
                // Ensure step_counts vector is large enough.
                if self.step_counts.len() <= robot_id {
                    self.step_counts.resize(robot_id + 1, 0);
                }
                SimResponse::RobotAdded { robot_id }
            }

            SimCommand::Step { robot_id, action } => {
                if let Some(robot) = manager.get_robot_mut(robot_id) {
                    apply_action(&robot.definition.clone(), &mut robot.state, &action);
                    // Step only this robot by calling the manager-level step.
                    // We step the entire manager which steps all robots — this
                    // matches the spec's step-locked model.
                    let dt = 1.0 / 60.0;
                    manager.step(dt, scene_meshes);

                    if let Some(robot) = manager.get_robot(robot_id) {
                        // Increment step counter.
                        if self.step_counts.len() <= robot_id {
                            self.step_counts.resize(robot_id + 1, 0);
                        }
                        self.step_counts[robot_id] += 1;

                        let state =
                            GymRobotState::from_robot_state(&robot.state, &robot.definition);
                        SimResponse::Stepped {
                            state,
                            step_count: self.step_counts[robot_id],
                        }
                    } else {
                        SimResponse::Error {
                            message: format!("robot {} disappeared after step", robot_id),
                        }
                    }
                } else {
                    SimResponse::Error {
                        message: format!("invalid robot_id: {}", robot_id),
                    }
                }
            }

            SimCommand::GetObservation { robot_id } => {
                if let Some(robot) = manager.get_robot(robot_id) {
                    let state =
                        GymRobotState::from_robot_state(&robot.state, &robot.definition);
                    SimResponse::Observation { state }
                } else {
                    SimResponse::Error {
                        message: format!("invalid robot_id: {}", robot_id),
                    }
                }
            }

            SimCommand::Reset { robot_id } => {
                if let Some(robot) = manager.get_robot_mut(robot_id) {
                    let def = robot.definition.clone();
                    robot.state = RobotState::new(&def);
                    // Reset step counter.
                    if self.step_counts.len() <= robot_id {
                        self.step_counts.resize(robot_id + 1, 0);
                    }
                    self.step_counts[robot_id] = 0;

                    let state = GymRobotState::from_robot_state(&robot.state, &def);
                    SimResponse::Reset { state }
                } else {
                    SimResponse::Error {
                        message: format!("invalid robot_id: {}", robot_id),
                    }
                }
            }

            SimCommand::RemoveRobot { robot_id } => {
                if robot_id < manager.robots.len() {
                    manager.robots.remove(robot_id);
                    // Adjust step_counts — remove the entry if present.
                    if robot_id < self.step_counts.len() {
                        self.step_counts.remove(robot_id);
                    }
                    SimResponse::Removed
                } else {
                    SimResponse::Error {
                        message: format!("invalid robot_id: {}", robot_id),
                    }
                }
            }

            SimCommand::GetSpaces { robot_id } => {
                if let Some(robot) = manager.get_robot(robot_id) {
                    let observation_space =
                        ObservationSpace::from_definition(&robot.definition);
                    let action_space = ActionSpace::from_definition(&robot.definition);
                    SimResponse::Spaces {
                        observation_space,
                        action_space,
                    }
                } else {
                    SimResponse::Error {
                        message: format!("invalid robot_id: {}", robot_id),
                    }
                }
            }
        }
    }
}

/// Create a bridge pair: one server-side handle for the agent server, one
/// client-side handle for the main loop.
pub fn create_bridge() -> (SimBridgeServer, SimBridgeClient) {
    let (tx, rx) = mpsc::unbounded_channel();
    let server = SimBridgeServer { tx };
    let client = SimBridgeClient {
        rx,
        step_counts: Vec::new(),
    };
    (server, client)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::definition::RobotDefinition;
    use crate::robot::state::RobotAction;
    use glam::Mat4;

    /// Helper: create a RobotManager with one simple_arm robot added.
    fn manager_with_arm() -> RobotManager {
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        manager.add_robot(def, Mat4::IDENTITY);
        manager
    }

    #[tokio::test]
    async fn test_bridge_creation() {
        let (server, _client) = create_bridge();
        // Verify the server can enqueue a command (channel is open).
        let (resp_tx, _resp_rx) = oneshot::channel();
        let result = server
            .tx
            .send((SimCommand::GetObservation { robot_id: 0 }, resp_tx));
        assert!(result.is_ok(), "channel should be open after creation");
    }

    #[tokio::test]
    async fn test_bridge_step_command() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm();

        // Spawn send_command on a separate task so process_pending can
        // provide the response.
        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::Step {
                    robot_id: 0,
                    action: RobotAction {
                        motor_velocities: vec![1.0, -0.5],
                        gripper_commands: vec![],
                    },
                })
                .await
        });

        // Give the spawned task a moment to enqueue the command.
        tokio::task::yield_now().await;

        client.process_pending(&mut manager, &[]);

        let response = handle.await.unwrap().unwrap();
        match response {
            SimResponse::Stepped { state, step_count } => {
                assert_eq!(step_count, 1, "first step should have step_count 1");
                assert_eq!(
                    state.joint_positions.len(),
                    2,
                    "should have 2 joint positions for simple_arm(2)"
                );
            }
            other => panic!("Expected Stepped, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bridge_get_observation() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm();

        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::GetObservation { robot_id: 0 })
                .await
        });

        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);

        let response = handle.await.unwrap().unwrap();
        match response {
            SimResponse::Observation { state } => {
                assert_eq!(state.joint_positions.len(), 2);
                // Initial state — positions should be zero.
                for p in &state.joint_positions {
                    assert!(p.abs() < 1e-6, "initial position should be zero");
                }
            }
            other => panic!("Expected Observation, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bridge_reset() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm();

        // Step once to change state.
        let handle = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::Step {
                    robot_id: 0,
                    action: RobotAction {
                        motor_velocities: vec![5.0, 5.0],
                        gripper_commands: vec![],
                    },
                })
                .await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let _ = handle.await.unwrap();

        // Now reset.
        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::Reset { robot_id: 0 })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);

        let response = handle.await.unwrap().unwrap();
        match response {
            SimResponse::Reset { state } => {
                // After reset, positions should be zero.
                for p in &state.joint_positions {
                    assert!(p.abs() < 1e-6, "reset position should be zero, got {}", p);
                }
            }
            other => panic!("Expected Reset, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bridge_error_invalid_robot() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm();

        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::GetObservation { robot_id: 99 })
                .await
        });

        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);

        let response = handle.await.unwrap().unwrap();
        match response {
            SimResponse::Error { message } => {
                assert!(
                    message.contains("invalid robot_id"),
                    "error should mention invalid robot_id, got: {}",
                    message
                );
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bridge_multiple_commands() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm();

        // Send multiple commands sequentially, processing after each.
        // Command 1: GetObservation
        let handle1 = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::GetObservation { robot_id: 0 })
                    .await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let r1 = handle1.await.unwrap().unwrap();
        assert!(
            matches!(r1, SimResponse::Observation { .. }),
            "first command should return Observation"
        );

        // Command 2: Step
        let handle2 = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::Step {
                    robot_id: 0,
                    action: RobotAction {
                        motor_velocities: vec![1.0, 1.0],
                        gripper_commands: vec![],
                    },
                })
                .await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let r2 = handle2.await.unwrap().unwrap();
        assert!(
            matches!(r2, SimResponse::Stepped { .. }),
            "second command should return Stepped"
        );

        // Command 3: Reset
        let handle3 = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::Reset { robot_id: 0 }).await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let r3 = handle3.await.unwrap().unwrap();
        assert!(
            matches!(r3, SimResponse::Reset { .. }),
            "third command should return Reset"
        );

        // Command 4: Error for invalid robot
        let handle4 = tokio::spawn(async move {
            server
                .send_command(SimCommand::GetObservation { robot_id: 42 })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let r4 = handle4.await.unwrap().unwrap();
        assert!(
            matches!(r4, SimResponse::Error { .. }),
            "fourth command should return Error for invalid robot_id"
        );
    }
}

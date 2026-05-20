use std::collections::{HashMap, VecDeque};

use tokio::sync::{mpsc, oneshot};

use crate::agent::protocol::AgentMessage;
use crate::robot::definition::RobotDefinition;
use crate::robot::state::{
    apply_action, ActionSpace, GymRobotState, GymStateBuffer, ObservationSpace, RobotAction,
    RobotState,
};
use crate::robot::RobotManager;
use crate::scene::SceneObject;
use glam::Mat4;

// ---------------------------------------------------------------------------
// Agent Activity Log — records bridge commands for the UI panel
// ---------------------------------------------------------------------------

/// A single logged event from agent-bridge interaction.
#[derive(Clone, Debug)]
pub struct AgentEvent {
    /// Monotonic timestamp in seconds since the log was created.
    pub timestamp: f32,
    /// Which robot was targeted (if applicable).
    pub robot_id: Option<usize>,
    /// Human-readable description of the event.
    pub description: String,
    /// Event category for filtering/coloring.
    pub kind: AgentEventKind,
}

/// Category of agent event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AgentEventKind {
    Connect,
    Step,
    Observe,
    Reset,
    Remove,
    Error,
    Message,
}

/// Rolling log of agent activity events, with a fixed capacity.
pub struct AgentActivityLog {
    events: VecDeque<AgentEvent>,
    capacity: usize,
    /// Elapsed seconds since creation (bumped by the UI each frame).
    pub elapsed: f32,
    /// Per-robot step counts (mirrors bridge step_counts for display).
    pub step_counts: Vec<u64>,
    /// Per-robot latest reward (from Step responses).
    pub latest_rewards: Vec<f32>,
    /// Per-robot connection status.
    pub connected_robots: Vec<bool>,
}

impl Default for AgentActivityLog {
    fn default() -> Self {
        Self::new(200)
    }
}

impl AgentActivityLog {
    /// Create a log with the given maximum event capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            events: VecDeque::with_capacity(capacity),
            capacity,
            elapsed: 0.0,
            step_counts: Vec::new(),
            latest_rewards: Vec::new(),
            connected_robots: Vec::new(),
        }
    }

    /// Push an event, evicting the oldest if at capacity.
    pub fn push(&mut self, kind: AgentEventKind, robot_id: Option<usize>, description: String) {
        if self.events.len() >= self.capacity {
            self.events.pop_front();
        }
        self.events.push_back(AgentEvent {
            timestamp: self.elapsed,
            robot_id,
            description,
            kind,
        });
    }

    /// Return an iterator over events (oldest first).
    pub fn iter(&self) -> impl Iterator<Item = &AgentEvent> {
        self.events.iter()
    }

    /// Number of logged events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Clear all events.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Update the step count for a robot (called from bridge processing).
    pub fn set_step_count(&mut self, robot_id: usize, count: u64) {
        if self.step_counts.len() <= robot_id {
            self.step_counts.resize(robot_id + 1, 0);
        }
        self.step_counts[robot_id] = count;
    }

    /// Update the latest reward for a robot.
    pub fn set_reward(&mut self, robot_id: usize, reward: f32) {
        if self.latest_rewards.len() <= robot_id {
            self.latest_rewards.resize(robot_id + 1, 0.0);
        }
        self.latest_rewards[robot_id] = reward;
    }

    /// Record that a robot is connected (controlled by an agent).
    pub fn set_connected(&mut self, robot_id: usize) {
        if self.connected_robots.len() <= robot_id {
            self.connected_robots.resize(robot_id + 1, false);
        }
        self.connected_robots[robot_id] = true;
    }

    /// Record that a robot has been disconnected/removed.
    pub fn set_disconnected(&mut self, robot_id: usize) {
        if robot_id < self.connected_robots.len() {
            self.connected_robots[robot_id] = false;
        }
    }

    /// Check if a robot is currently connected to an agent.
    pub fn is_connected(&self, robot_id: usize) -> bool {
        self.connected_robots
            .get(robot_id)
            .copied()
            .unwrap_or(false)
    }
}

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
    SendMessage {
        from_robot_id: usize,
        to_robot_id: usize,
        content: String,
    },
    /// Ask the main-loop side whether `robot_id` has a `CombatState`. Used
    /// by `handle_bind_target` to set the `"combat"` capability bit.
    HasCombat {
        robot_id: usize,
    },
}

/// A response sent from the simulation main loop back to the agent server.
#[derive(Debug)]
pub enum SimResponse {
    RobotAdded {
        robot_id: usize,
    },
    Stepped {
        state: GymRobotState,
        step_count: u64,
        messages: Vec<AgentMessage>,
        match_state: Option<crate::robot::boxing::BoxingMatchState>,
    },
    Observation {
        state: GymRobotState,
        messages: Vec<AgentMessage>,
        match_state: Option<crate::robot::boxing::BoxingMatchState>,
    },
    Reset {
        state: GymRobotState,
        messages: Vec<AgentMessage>,
        match_state: Option<crate::robot::boxing::BoxingMatchState>,
    },
    Removed,
    MessageSent,
    Spaces {
        observation_space: ObservationSpace,
        action_space: ActionSpace,
    },
    Error {
        message: String,
    },
    /// Reply to `SimCommand::HasCombat`.
    HasCombat {
        has_combat: bool,
    },
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
    pub(crate) tx: mpsc::UnboundedSender<CommandEnvelope>,
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
pub struct MessageBus {
    pending: HashMap<usize, VecDeque<AgentMessage>>,
    history: HashMap<(usize, usize), VecDeque<AgentMessage>>,
    history_capacity: usize,
    next_timestamp: u64,
}

impl MessageBus {
    pub fn new(history_capacity: usize) -> Self {
        Self {
            pending: HashMap::new(),
            history: HashMap::new(),
            history_capacity,
            next_timestamp: 0,
        }
    }

    pub fn send(
        &mut self,
        from_robot_id: usize,
        to_robot_id: usize,
        content: String,
    ) -> AgentMessage {
        let msg = AgentMessage {
            from_robot_id,
            to_robot_id,
            content,
            timestamp: self.next_timestamp,
        };
        self.next_timestamp += 1;

        self.pending
            .entry(to_robot_id)
            .or_default()
            .push_back(msg.clone());

        let hist = self
            .history
            .entry((from_robot_id, to_robot_id))
            .or_default();
        if hist.len() >= self.history_capacity {
            hist.pop_front();
        }
        hist.push_back(msg.clone());

        msg
    }

    pub fn drain(&mut self, robot_id: usize) -> Vec<AgentMessage> {
        self.pending
            .remove(&robot_id)
            .map(|q| q.into_iter().collect())
            .unwrap_or_default()
    }
}

pub struct SimBridgeClient {
    rx: mpsc::UnboundedReceiver<CommandEnvelope>,
    /// Per-robot step counters. Indexed by robot_id.
    step_counts: Vec<u64>,
    state_buffer: GymStateBuffer,
    message_bus: MessageBus,
    pub boxing_match: Option<crate::robot::boxing::BoxingMatch>,
}

impl SimBridgeClient {
    fn boxing_match_snapshot(
        &self,
        robot_id: usize,
        manager: &RobotManager,
    ) -> Option<crate::robot::boxing::BoxingMatchState> {
        let bm = self.boxing_match.as_ref()?;
        let opponent_id = if robot_id == bm.robot_a {
            bm.robot_b
        } else {
            bm.robot_a
        };
        let opponent_robot = manager.get_robot(opponent_id);
        let opponent_combat = opponent_robot.and_then(|r| r.state.combat.as_ref());
        let own_link_poses = manager
            .get_robot(robot_id)
            .map(|r| r.state.link_poses.as_slice());
        let opponent_link_poses = opponent_robot.map(|r| r.state.link_poses.as_slice());
        Some(bm.snapshot_with_spatial(
            robot_id,
            opponent_combat,
            own_link_poses,
            opponent_link_poses,
        ))
    }

    /// Non-blocking drain of the command channel.
    ///
    /// For each pending command, executes the operation on `manager` and
    /// sends the result back through the oneshot. `scene_meshes` is passed
    /// to `RobotManager::step` when stepping the simulation.
    pub fn process_pending(&mut self, manager: &mut RobotManager, scene_meshes: &[SceneObject]) {
        while let Ok((cmd, resp_tx)) = self.rx.try_recv() {
            let response = self.execute(cmd, manager, scene_meshes);
            let _ = resp_tx.send(response);
        }
    }

    /// Like `process_pending`, but also logs events to the activity log.
    pub fn process_pending_with_log(
        &mut self,
        manager: &mut RobotManager,
        scene_meshes: &[SceneObject],
        activity_log: &mut AgentActivityLog,
    ) {
        while let Ok((cmd, resp_tx)) = self.rx.try_recv() {
            log_command(&cmd, activity_log);
            let response = self.execute(cmd, manager, scene_meshes);
            log_response(&response, activity_log);
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
                    apply_action(&robot.definition, &mut robot.state, &action);
                    let dt = 1.0 / 60.0;

                    // Planar base locomotion. Bridge clamps velocity and ring bounds
                    // so single-link humanoids (no leg joints) can still navigate.
                    let max_speed = 2.0_f32;
                    let ring_half = 2.7_f32;
                    let vx = action.base_velocity[0].clamp(-max_speed, max_speed);
                    let vz = action.base_velocity[1].clamp(-max_speed, max_speed);
                    let mut bp = robot.base_pose;
                    bp[12] = (bp[12] + vx * dt).clamp(-ring_half, ring_half);
                    bp[14] = (bp[14] + vz * dt).clamp(-ring_half, ring_half);
                    robot.base_pose = bp;

                    manager.step(dt, scene_meshes);

                    if let Some(bm) = &mut self.boxing_match {
                        let combat_states: Vec<(usize, &crate::robot::state::CombatState)> =
                            manager
                                .robots
                                .iter()
                                .enumerate()
                                .filter_map(|(i, r)| r.state.combat.as_ref().map(|c| (i, c)))
                                .collect();
                        bm.update(&manager.last_hit_events, &combat_states, dt);
                    }

                    if let Some(robot) = manager.get_robot(robot_id) {
                        if self.step_counts.len() <= robot_id {
                            self.step_counts.resize(robot_id + 1, 0);
                        }
                        self.step_counts[robot_id] += 1;

                        let state = GymRobotState::from_robot_state_into(
                            &robot.state,
                            &robot.definition,
                            &mut self.state_buffer,
                        );
                        let messages = self.message_bus.drain(robot_id);
                        let match_state = self.boxing_match_snapshot(robot_id, manager);
                        SimResponse::Stepped {
                            state,
                            step_count: self.step_counts[robot_id],
                            messages,
                            match_state,
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
                    let state = GymRobotState::from_robot_state_into(
                        &robot.state,
                        &robot.definition,
                        &mut self.state_buffer,
                    );
                    let messages = self.message_bus.drain(robot_id);
                    let match_state = self.boxing_match_snapshot(robot_id, manager);
                    SimResponse::Observation {
                        state,
                        messages,
                        match_state,
                    }
                } else {
                    SimResponse::Error {
                        message: format!("invalid robot_id: {}", robot_id),
                    }
                }
            }

            SimCommand::Reset { robot_id } => {
                if let Some(robot) = manager.get_robot_mut(robot_id) {
                    let combat = robot.state.combat.take();
                    robot.state = RobotState::new(&robot.definition);
                    robot.state.combat = combat;
                    if self.step_counts.len() <= robot_id {
                        self.step_counts.resize(robot_id + 1, 0);
                    }
                    self.step_counts[robot_id] = 0;

                    let state = GymRobotState::from_robot_state_into(
                        &robot.state,
                        &robot.definition,
                        &mut self.state_buffer,
                    );
                    let messages = self.message_bus.drain(robot_id);
                    let match_state = self.boxing_match_snapshot(robot_id, manager);
                    SimResponse::Reset {
                        state,
                        messages,
                        match_state,
                    }
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
                    let observation_space = ObservationSpace::from_definition(&robot.definition);
                    let action_space = ActionSpace::from_definition(&robot.definition);
                    if let Some(bm) = &mut self.boxing_match {
                        bm.connect_agent(robot_id);
                    }
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

            SimCommand::HasCombat { robot_id } => {
                let has_combat = manager
                    .get_robot(robot_id)
                    .map(|r| r.state.combat.is_some())
                    .unwrap_or(false);
                SimResponse::HasCombat { has_combat }
            }

            SimCommand::SendMessage {
                from_robot_id,
                to_robot_id,
                content,
            } => {
                if content.len() > 1024 {
                    return SimResponse::Error {
                        message: "message content exceeds 1024 bytes".to_string(),
                    };
                }
                if manager.get_robot(from_robot_id).is_none() {
                    return SimResponse::Error {
                        message: format!("invalid from_robot_id: {}", from_robot_id),
                    };
                }
                if manager.get_robot(to_robot_id).is_none() {
                    return SimResponse::Error {
                        message: format!("invalid to_robot_id: {}", to_robot_id),
                    };
                }
                self.message_bus.send(from_robot_id, to_robot_id, content);
                SimResponse::MessageSent
            }
        }
    }
}

/// Log a bridge command to the activity log.
fn log_command(cmd: &SimCommand, log: &mut AgentActivityLog) {
    match cmd {
        SimCommand::AddRobot { .. } => {
            log.push(AgentEventKind::Connect, None, "AddRobot request".into());
        }
        SimCommand::Step { robot_id, .. } => {
            log.push(AgentEventKind::Step, Some(*robot_id), "Step".into());
        }
        SimCommand::GetObservation { robot_id } => {
            log.push(AgentEventKind::Observe, Some(*robot_id), "Observe".into());
        }
        SimCommand::Reset { robot_id } => {
            log.push(AgentEventKind::Reset, Some(*robot_id), "Reset".into());
        }
        SimCommand::RemoveRobot { robot_id } => {
            log.push(
                AgentEventKind::Remove,
                Some(*robot_id),
                "RemoveRobot".into(),
            );
            log.set_disconnected(*robot_id);
        }
        SimCommand::GetSpaces { robot_id } => {
            log.push(
                AgentEventKind::Connect,
                Some(*robot_id),
                "GetSpaces (connect)".into(),
            );
        }
        SimCommand::SendMessage {
            from_robot_id,
            to_robot_id,
            ..
        } => {
            log.push(
                AgentEventKind::Message,
                Some(*from_robot_id),
                format!("Message {} -> {}", from_robot_id, to_robot_id),
            );
        }
        SimCommand::HasCombat { .. } => {
            // Capability probe — not interesting to surface in the activity log.
        }
    }
}

/// Log a bridge response to the activity log.
fn log_response(response: &SimResponse, log: &mut AgentActivityLog) {
    match response {
        SimResponse::RobotAdded { robot_id } => {
            log.set_step_count(*robot_id, 0);
            log.set_reward(*robot_id, 0.0);
            log.set_connected(*robot_id);
        }
        SimResponse::Removed => {
            // Robot removed; we don't have the id here but the command
            // log captured it.
        }
        SimResponse::Stepped { step_count, .. } => {
            let _ = step_count;
        }
        SimResponse::Spaces { .. } => {
            // GetSpaces succeeded — marks a successful agent connection.
            // Robot ID was logged in log_command.
        }
        SimResponse::Error { message } => {
            log.push(AgentEventKind::Error, None, format!("Error: {message}"));
        }
        _ => {}
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
        state_buffer: GymStateBuffer::new(),
        message_bus: MessageBus::new(100),
        boxing_match: None,
    };
    (server, client)
}

/// Create a bridge pair with a boxing match pre-wired.
pub fn create_bridge_with_boxing(
    boxing_match: crate::robot::boxing::BoxingMatch,
) -> (SimBridgeServer, SimBridgeClient) {
    let (tx, rx) = mpsc::unbounded_channel();
    let server = SimBridgeServer { tx };
    let client = SimBridgeClient {
        rx,
        step_counts: Vec::new(),
        state_buffer: GymStateBuffer::new(),
        message_bus: MessageBus::new(100),
        boxing_match: Some(boxing_match),
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
                        base_velocity: [0.0, 0.0],
                    },
                })
                .await
        });

        // Give the spawned task a moment to enqueue the command.
        tokio::task::yield_now().await;

        client.process_pending(&mut manager, &[]);

        let response = handle.await.unwrap().unwrap();
        match response {
            SimResponse::Stepped {
                state, step_count, ..
            } => {
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
            SimResponse::Observation { state, .. } => {
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
                        base_velocity: [0.0, 0.0],
                    },
                })
                .await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let _ = handle.await.unwrap();

        // Now reset.
        let handle =
            tokio::spawn(
                async move { server.send_command(SimCommand::Reset { robot_id: 0 }).await },
            );
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);

        let response = handle.await.unwrap().unwrap();
        match response {
            SimResponse::Reset { state, .. } => {
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
                        base_velocity: [0.0, 0.0],
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

    // ---- Edge case tests ----

    #[tokio::test]
    async fn test_bridge_send_after_client_dropped() {
        let (server, client) = create_bridge();
        // Drop the client side, closing the receiving end of the channel.
        drop(client);

        let result = server
            .send_command(SimCommand::GetObservation { robot_id: 0 })
            .await;
        assert!(result.is_err(), "send should fail when client is dropped");
        assert!(
            result.unwrap_err().contains("bridge channel closed"),
            "error should mention bridge channel closed"
        );
    }

    #[tokio::test]
    async fn test_bridge_response_channel_dropped() {
        let (server, _client) = create_bridge();
        // Manually send a command but drop the oneshot receiver before response
        let (resp_tx, resp_rx) = oneshot::channel::<SimResponse>();
        // Drop the receiver
        drop(resp_rx);
        // The sender should still be sendable (send returns Ok/Err based on
        // whether receiver is alive), but the bridge server's send_command
        // won't see this directly since it creates its own oneshot.
        // Instead, test that if we drop the resp_tx before the bridge can
        // respond, send_command returns an error.
        let result = server
            .tx
            .send((SimCommand::GetObservation { robot_id: 0 }, resp_tx));
        assert!(
            result.is_ok(),
            "channel send should succeed even if oneshot receiver is dropped"
        );
    }

    #[tokio::test]
    async fn test_bridge_step_empty_action() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm();

        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::Step {
                    robot_id: 0,
                    action: RobotAction {
                        motor_velocities: vec![],
                        gripper_commands: vec![],
                        base_velocity: [0.0, 0.0],
                    },
                })
                .await
        });

        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);

        let response = handle.await.unwrap().unwrap();
        // Empty action should still succeed; apply_action clamps to 0 motors
        assert!(
            matches!(response, SimResponse::Stepped { .. }),
            "step with empty action should still succeed, got {:?}",
            response
        );
    }

    #[tokio::test]
    async fn test_bridge_remove_then_step() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm();

        // Remove robot 0
        let handle = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::RemoveRobot { robot_id: 0 })
                    .await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let r = handle.await.unwrap().unwrap();
        assert!(matches!(r, SimResponse::Removed));

        // Now try to step the removed robot
        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::Step {
                    robot_id: 0,
                    action: RobotAction {
                        motor_velocities: vec![1.0, 1.0],
                        gripper_commands: vec![],
                        base_velocity: [0.0, 0.0],
                    },
                })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let r = handle.await.unwrap().unwrap();
        match r {
            SimResponse::Error { message } => {
                assert!(
                    message.contains("invalid robot_id"),
                    "should report invalid robot_id after removal, got: {}",
                    message
                );
            }
            other => panic!(
                "Expected Error after stepping removed robot, got {:?}",
                other
            ),
        }
    }

    #[tokio::test]
    async fn test_bridge_get_spaces_invalid_robot() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm();

        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::GetSpaces { robot_id: 999 })
                .await
        });

        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);

        let response = handle.await.unwrap().unwrap();
        match response {
            SimResponse::Error { message } => {
                assert!(
                    message.contains("invalid robot_id"),
                    "GetSpaces for invalid robot should error, got: {}",
                    message
                );
            }
            other => panic!("Expected Error for invalid GetSpaces, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bridge_remove_invalid_robot() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm();

        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::RemoveRobot { robot_id: 50 })
                .await
        });

        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);

        let response = handle.await.unwrap().unwrap();
        match response {
            SimResponse::Error { message } => {
                assert!(
                    message.contains("invalid robot_id"),
                    "RemoveRobot for invalid id should error, got: {}",
                    message
                );
            }
            other => panic!("Expected Error for invalid RemoveRobot, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bridge_reset_invalid_robot() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm();

        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::Reset { robot_id: 10 })
                .await
        });

        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);

        let response = handle.await.unwrap().unwrap();
        match response {
            SimResponse::Error { message } => {
                assert!(
                    message.contains("invalid robot_id"),
                    "Reset for invalid robot should error, got: {}",
                    message
                );
            }
            other => panic!("Expected Error for invalid Reset, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bridge_step_count_resets_on_reset() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm();

        // Step 3 times
        for _ in 0..3 {
            let handle = tokio::spawn({
                let tx = server.tx.clone();
                async move {
                    let srv = SimBridgeServer { tx };
                    srv.send_command(SimCommand::Step {
                        robot_id: 0,
                        action: RobotAction {
                            motor_velocities: vec![1.0, 1.0],
                            gripper_commands: vec![],
                            base_velocity: [0.0, 0.0],
                        },
                    })
                    .await
                }
            });
            tokio::task::yield_now().await;
            client.process_pending(&mut manager, &[]);
            let _ = handle.await.unwrap();
        }

        // Reset
        let handle = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::Reset { robot_id: 0 }).await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let _ = handle.await.unwrap();

        // Step once more -- step_count should be 1, not 4
        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::Step {
                    robot_id: 0,
                    action: RobotAction {
                        motor_velocities: vec![1.0, 1.0],
                        gripper_commands: vec![],
                        base_velocity: [0.0, 0.0],
                    },
                })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let response = handle.await.unwrap().unwrap();
        match response {
            SimResponse::Stepped { step_count, .. } => {
                assert_eq!(
                    step_count, 1,
                    "step_count should be 1 after reset then step, got {}",
                    step_count
                );
            }
            other => panic!("Expected Stepped, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bridge_process_pending_no_commands() {
        let (_server, mut client) = create_bridge();
        let mut manager = manager_with_arm();

        // Calling process_pending with no commands should be a no-op
        client.process_pending(&mut manager, &[]);
        // No panic or error means success
    }

    #[tokio::test]
    async fn test_bridge_step_oversized_action() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm(); // 2 joints

        // Send action with MORE velocities than joints (should be clamped)
        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::Step {
                    robot_id: 0,
                    action: RobotAction {
                        motor_velocities: vec![1.0, 2.0, 3.0, 4.0, 5.0],
                        gripper_commands: vec![],
                        base_velocity: [0.0, 0.0],
                    },
                })
                .await
        });

        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);

        let response = handle.await.unwrap().unwrap();
        assert!(
            matches!(response, SimResponse::Stepped { .. }),
            "oversized action should be clamped, not error, got {:?}",
            response
        );
    }

    #[tokio::test]
    async fn test_bridge_add_robot_then_get_spaces() {
        let (server, mut client) = create_bridge();
        let mut manager = RobotManager::new();
        // Manager is empty -- add a robot via command
        let def = RobotDefinition::simple_arm(3);
        let handle = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::AddRobot {
                    definition: def,
                    base_pose: Mat4::IDENTITY.to_cols_array(),
                })
                .await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let resp = handle.await.unwrap().unwrap();
        let robot_id = match resp {
            SimResponse::RobotAdded { robot_id } => robot_id,
            other => panic!("Expected RobotAdded, got {:?}", other),
        };

        // Now get spaces for the newly added robot
        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::GetSpaces { robot_id })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let resp = handle.await.unwrap().unwrap();
        match resp {
            SimResponse::Spaces {
                observation_space,
                action_space,
            } => {
                assert_eq!(
                    observation_space.num_joint_positions, 3,
                    "simple_arm(3) should have 3 joints"
                );
                assert_eq!(action_space.num_motors, 3);
            }
            other => panic!("Expected Spaces, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bridge_step_count_increments_correctly() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm();

        // Step 5 times and verify step_count increments each time
        for expected in 1u64..=5 {
            let handle = tokio::spawn({
                let tx = server.tx.clone();
                async move {
                    let srv = SimBridgeServer { tx };
                    srv.send_command(SimCommand::Step {
                        robot_id: 0,
                        action: RobotAction {
                            motor_velocities: vec![1.0, 1.0],
                            gripper_commands: vec![],
                            base_velocity: [0.0, 0.0],
                        },
                    })
                    .await
                }
            });
            tokio::task::yield_now().await;
            client.process_pending(&mut manager, &[]);
            let resp = handle.await.unwrap().unwrap();
            match resp {
                SimResponse::Stepped { step_count, .. } => {
                    assert_eq!(
                        step_count, expected,
                        "step_count should be {} on step {}, got {}",
                        expected, expected, step_count
                    );
                }
                other => panic!("Expected Stepped on step {}, got {:?}", expected, other),
            }
        }
    }

    #[tokio::test]
    async fn test_bridge_remove_robot_0_with_multiple_robots() {
        let (server, mut client) = create_bridge();
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        manager.add_robot(def.clone(), Mat4::IDENTITY); // robot 0
        manager.add_robot(def, Mat4::IDENTITY); // robot 1

        // Remove robot 0
        let handle = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::RemoveRobot { robot_id: 0 })
                    .await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let resp = handle.await.unwrap().unwrap();
        assert!(matches!(resp, SimResponse::Removed));

        // After removing index 0, what was robot 1 is now at index 0.
        // Getting observation for robot_id=0 should succeed (it's the old robot 1).
        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::GetObservation { robot_id: 0 })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let resp = handle.await.unwrap().unwrap();
        assert!(
            matches!(resp, SimResponse::Observation { .. }),
            "robot_id 0 after removal should be the shifted robot, got {:?}",
            resp
        );
    }

    #[tokio::test]
    async fn test_bridge_multiple_cloned_server_handles() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm();
        let server2 = SimBridgeServer {
            tx: server.tx.clone(),
        };

        // Both handles can send commands
        let h1 = tokio::spawn(async move {
            server
                .send_command(SimCommand::GetObservation { robot_id: 0 })
                .await
        });
        let h2 = tokio::spawn(async move {
            server2
                .send_command(SimCommand::GetObservation { robot_id: 0 })
                .await
        });

        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);

        let r1 = h1.await.unwrap().unwrap();
        let r2 = h2.await.unwrap().unwrap();
        assert!(matches!(r1, SimResponse::Observation { .. }));
        assert!(matches!(r2, SimResponse::Observation { .. }));
    }

    #[tokio::test]
    async fn test_bridge_step_then_observe_same_state() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm();

        // Step once
        let handle = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::Step {
                    robot_id: 0,
                    action: RobotAction {
                        motor_velocities: vec![5.0, -3.0],
                        gripper_commands: vec![],
                        base_velocity: [0.0, 0.0],
                    },
                })
                .await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let step_resp = handle.await.unwrap().unwrap();
        let step_state = match step_resp {
            SimResponse::Stepped { state, .. } => state,
            other => panic!("Expected Stepped, got {:?}", other),
        };

        // Observe should return the same state (no simulation step happened)
        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::GetObservation { robot_id: 0 })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let obs_resp = handle.await.unwrap().unwrap();
        let obs_state = match obs_resp {
            SimResponse::Observation { state, .. } => state,
            other => panic!("Expected Observation, got {:?}", other),
        };

        assert_eq!(
            step_state.joint_positions, obs_state.joint_positions,
            "observe should return same state as the last step"
        );
    }

    #[tokio::test]
    async fn test_bridge_double_remove_same_robot() {
        let (server, mut client) = create_bridge();
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        manager.add_robot(def, Mat4::IDENTITY);

        // Remove robot 0 (first time -- succeeds)
        let handle = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::RemoveRobot { robot_id: 0 })
                    .await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let resp = handle.await.unwrap().unwrap();
        assert!(matches!(resp, SimResponse::Removed));

        // Remove robot 0 again (should error since manager is now empty)
        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::RemoveRobot { robot_id: 0 })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let resp = handle.await.unwrap().unwrap();
        assert!(
            matches!(resp, SimResponse::Error { .. }),
            "double remove should error, got {:?}",
            resp
        );
    }

    // ---- MessageBus unit tests ----

    #[test]
    fn test_message_bus_send_and_drain() {
        let mut bus = MessageBus::new(50);
        bus.send(0, 1, "hello".into());
        let msgs = bus.drain(1);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].from_robot_id, 0);
        assert_eq!(msgs[0].to_robot_id, 1);
        assert_eq!(msgs[0].content, "hello");
    }

    #[test]
    fn test_message_bus_history() {
        let mut bus = MessageBus::new(50);
        bus.send(0, 1, "first".into());
        bus.send(0, 1, "second".into());
        let hist = bus.history.get(&(0, 1)).unwrap();
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0].content, "first");
        assert_eq!(hist[1].content, "second");
    }

    #[test]
    fn test_message_bus_drain_clears() {
        let mut bus = MessageBus::new(50);
        bus.send(0, 1, "msg".into());
        let msgs = bus.drain(1);
        assert_eq!(msgs.len(), 1);
        let msgs2 = bus.drain(1);
        assert!(msgs2.is_empty(), "second drain should be empty");
    }

    #[test]
    fn test_message_bus_history_capacity() {
        let mut bus = MessageBus::new(3);
        for i in 0..5 {
            bus.send(0, 1, format!("msg{}", i));
        }
        let hist = bus.history.get(&(0, 1)).unwrap();
        assert_eq!(hist.len(), 3);
        assert_eq!(hist[0].content, "msg2");
    }

    #[test]
    fn test_message_bus_timestamps_monotonic() {
        let mut bus = MessageBus::new(50);
        bus.send(0, 1, "a".into());
        bus.send(1, 0, "b".into());
        bus.send(0, 1, "c".into());
        let msgs = bus.drain(1);
        assert!(msgs[0].timestamp < msgs[1].timestamp);
    }

    // ---- Bridge SendMessage tests ----

    #[tokio::test]
    async fn test_bridge_send_message() {
        let (server, mut client) = create_bridge();
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        manager.add_robot(def.clone(), Mat4::IDENTITY);
        manager.add_robot(def, Mat4::IDENTITY);

        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::SendMessage {
                    from_robot_id: 0,
                    to_robot_id: 1,
                    content: "hello".into(),
                })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);

        let resp = handle.await.unwrap().unwrap();
        assert!(
            matches!(resp, SimResponse::MessageSent),
            "Expected MessageSent, got {:?}",
            resp
        );
    }

    #[tokio::test]
    async fn test_bridge_send_to_invalid_robot() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm();

        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::SendMessage {
                    from_robot_id: 0,
                    to_robot_id: 99,
                    content: "hello".into(),
                })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);

        let resp = handle.await.unwrap().unwrap();
        match resp {
            SimResponse::Error { message } => {
                assert!(message.contains("invalid to_robot_id"));
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bridge_send_from_invalid_robot() {
        let (server, mut client) = create_bridge();
        let mut manager = manager_with_arm();

        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::SendMessage {
                    from_robot_id: 99,
                    to_robot_id: 0,
                    content: "hello".into(),
                })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);

        let resp = handle.await.unwrap().unwrap();
        match resp {
            SimResponse::Error { message } => {
                assert!(message.contains("invalid from_robot_id"));
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bridge_step_delivers_messages() {
        let (server, mut client) = create_bridge();
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        manager.add_robot(def.clone(), Mat4::IDENTITY);
        manager.add_robot(def, Mat4::IDENTITY);

        // Send message from 0 to 1
        let handle = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::SendMessage {
                    from_robot_id: 0,
                    to_robot_id: 1,
                    content: "hey".into(),
                })
                .await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let _ = handle.await.unwrap();

        // Step robot 1 — should get the message
        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::Step {
                    robot_id: 1,
                    action: RobotAction {
                        motor_velocities: vec![0.0, 0.0],
                        gripper_commands: vec![],
                        base_velocity: [0.0, 0.0],
                    },
                })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let resp = handle.await.unwrap().unwrap();
        match resp {
            SimResponse::Stepped { messages, .. } => {
                assert_eq!(messages.len(), 1);
                assert_eq!(messages[0].content, "hey");
                assert_eq!(messages[0].from_robot_id, 0);
            }
            other => panic!("Expected Stepped with messages, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bridge_observe_delivers_messages() {
        let (server, mut client) = create_bridge();
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        manager.add_robot(def.clone(), Mat4::IDENTITY);
        manager.add_robot(def, Mat4::IDENTITY);

        // Send message from 1 to 0
        let handle = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::SendMessage {
                    from_robot_id: 1,
                    to_robot_id: 0,
                    content: "yo".into(),
                })
                .await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let _ = handle.await.unwrap();

        // Observe robot 0 — should get the message
        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::GetObservation { robot_id: 0 })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let resp = handle.await.unwrap().unwrap();
        match resp {
            SimResponse::Observation { messages, .. } => {
                assert_eq!(messages.len(), 1);
                assert_eq!(messages[0].content, "yo");
            }
            other => panic!("Expected Observation with messages, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_bridge_message_content_too_long() {
        let (server, mut client) = create_bridge();
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        manager.add_robot(def.clone(), Mat4::IDENTITY);
        manager.add_robot(def, Mat4::IDENTITY);

        let long_content = "x".repeat(1025);
        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::SendMessage {
                    from_robot_id: 0,
                    to_robot_id: 1,
                    content: long_content,
                })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let resp = handle.await.unwrap().unwrap();
        match resp {
            SimResponse::Error { message } => {
                assert!(message.contains("1024"));
            }
            other => panic!("Expected Error for long message, got {:?}", other),
        }
    }

    // ---- Integration tests: two agents messaging ----

    #[tokio::test]
    async fn test_two_agents_message_exchange() {
        let (server, mut client) = create_bridge();
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        manager.add_robot(def.clone(), Mat4::IDENTITY);
        manager.add_robot(def, Mat4::IDENTITY);

        // Agent 0 sends "hello" to Agent 1
        let h = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::SendMessage {
                    from_robot_id: 0,
                    to_robot_id: 1,
                    content: "hello".into(),
                })
                .await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        assert!(matches!(
            h.await.unwrap().unwrap(),
            SimResponse::MessageSent
        ));

        // Agent 1 steps and receives the message
        let h = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::Step {
                    robot_id: 1,
                    action: RobotAction {
                        motor_velocities: vec![0.0, 0.0],
                        gripper_commands: vec![],
                        base_velocity: [0.0, 0.0],
                    },
                })
                .await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let resp = h.await.unwrap().unwrap();
        let msgs = match resp {
            SimResponse::Stepped { messages, .. } => messages,
            other => panic!("Expected Stepped, got {:?}", other),
        };
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "hello");
        assert_eq!(msgs[0].from_robot_id, 0);
        assert_eq!(msgs[0].to_robot_id, 1);

        // Agent 1 replies "world" to Agent 0
        let h = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::SendMessage {
                    from_robot_id: 1,
                    to_robot_id: 0,
                    content: "world".into(),
                })
                .await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        assert!(matches!(
            h.await.unwrap().unwrap(),
            SimResponse::MessageSent
        ));

        // Agent 0 observes and receives the reply
        let h = tokio::spawn(async move {
            server
                .send_command(SimCommand::GetObservation { robot_id: 0 })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let resp = h.await.unwrap().unwrap();
        let msgs = match resp {
            SimResponse::Observation { messages, .. } => messages,
            other => panic!("Expected Observation, got {:?}", other),
        };
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "world");
        assert_eq!(msgs[0].from_robot_id, 1);
        assert_eq!(msgs[0].to_robot_id, 0);
        assert!(msgs[0].timestamp > msgs[0].timestamp.wrapping_sub(1));
    }

    #[tokio::test]
    async fn test_message_delivery_order() {
        let (server, mut client) = create_bridge();
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        manager.add_robot(def.clone(), Mat4::IDENTITY);
        manager.add_robot(def, Mat4::IDENTITY);

        // Send 3 messages in order
        for i in 0..3 {
            let h = tokio::spawn({
                let tx = server.tx.clone();
                async move {
                    let srv = SimBridgeServer { tx };
                    srv.send_command(SimCommand::SendMessage {
                        from_robot_id: 0,
                        to_robot_id: 1,
                        content: format!("msg{}", i),
                    })
                    .await
                }
            });
            tokio::task::yield_now().await;
            client.process_pending(&mut manager, &[]);
            let _ = h.await.unwrap();
        }

        // Drain via observe
        let h = tokio::spawn(async move {
            server
                .send_command(SimCommand::GetObservation { robot_id: 1 })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let resp = h.await.unwrap().unwrap();
        let msgs = match resp {
            SimResponse::Observation { messages, .. } => messages,
            other => panic!("Expected Observation, got {:?}", other),
        };
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].content, "msg0");
        assert_eq!(msgs[1].content, "msg1");
        assert_eq!(msgs[2].content, "msg2");
        assert!(msgs[0].timestamp < msgs[1].timestamp);
        assert!(msgs[1].timestamp < msgs[2].timestamp);
    }

    #[tokio::test]
    async fn test_messages_only_delivered_once() {
        let (server, mut client) = create_bridge();
        let mut manager = RobotManager::new();
        let def = RobotDefinition::simple_arm(2);
        manager.add_robot(def.clone(), Mat4::IDENTITY);
        manager.add_robot(def, Mat4::IDENTITY);

        // Send a message
        let h = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::SendMessage {
                    from_robot_id: 0,
                    to_robot_id: 1,
                    content: "once".into(),
                })
                .await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let _ = h.await.unwrap();

        // First observe delivers the message
        let h = tokio::spawn({
            let tx = server.tx.clone();
            async move {
                let srv = SimBridgeServer { tx };
                srv.send_command(SimCommand::GetObservation { robot_id: 1 })
                    .await
            }
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let resp = h.await.unwrap().unwrap();
        match resp {
            SimResponse::Observation { messages, .. } => assert_eq!(messages.len(), 1),
            other => panic!("Expected Observation, got {:?}", other),
        }

        // Second observe — message should be gone
        let h = tokio::spawn(async move {
            server
                .send_command(SimCommand::GetObservation { robot_id: 1 })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        let resp = h.await.unwrap().unwrap();
        match resp {
            SimResponse::Observation { messages, .. } => {
                assert!(
                    messages.is_empty(),
                    "messages should be drained after first delivery"
                );
            }
            other => panic!("Expected Observation, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_create_bridge_with_boxing() {
        use crate::robot::boxing::{BoxingMatch, BoxingMatchConfig};

        let bm = BoxingMatch::new(0, 1, BoxingMatchConfig::default());
        let (server, mut client) = create_bridge_with_boxing(bm);

        assert!(
            client.boxing_match.is_some(),
            "boxing_match should be pre-wired"
        );

        let mut manager = RobotManager::new();
        let def = RobotDefinition::boxing_humanoid();
        let pose_a = Mat4::from_translation(glam::Vec3::new(-1.5, 0.0, 0.0));
        let pose_b = Mat4::from_translation(glam::Vec3::new(1.5, 0.0, 0.0));
        manager.add_robot(def.clone(), pose_a);
        manager.add_robot(def, pose_b);
        if let Some(r) = manager.get_robot_mut(0) {
            r.state.combat = Some(crate::robot::state::CombatState::new(100.0, 100.0));
        }
        if let Some(r) = manager.get_robot_mut(1) {
            r.state.combat = Some(crate::robot::state::CombatState::new(100.0, 100.0));
        }

        let handle = tokio::spawn(async move {
            server
                .send_command(SimCommand::GetObservation { robot_id: 0 })
                .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);

        let resp = handle.await.unwrap().unwrap();
        match resp {
            SimResponse::Observation { match_state, .. } => {
                assert!(match_state.is_some(), "match_state should be populated");
            }
            other => panic!("Expected Observation with match_state, got {:?}", other),
        }
    }

    /// D2: GymRobotState.combat should be populated for robots that carry a
    /// CombatState, and `None` for plain arms. Asserts both the
    /// GetObservation and Step paths populate it consistently.
    #[tokio::test]
    async fn combat_observations_populated_when_robot_has_combat_state() {
        let (server, mut client) = create_bridge();
        let mut manager = RobotManager::new();
        let def = RobotDefinition::boxing_humanoid();
        manager.add_robot(def.clone(), Mat4::IDENTITY);
        manager.add_robot(RobotDefinition::simple_arm(3), Mat4::IDENTITY);
        manager.get_robot_mut(0).unwrap().state.combat =
            Some(crate::robot::state::CombatState::new(75.0, 50.0));
        // robot 1 has no combat state — observations should reflect None.

        // GetObservation path for combat robot
        let s = server.clone();
        let handle = tokio::spawn(async move {
            s.send_command(SimCommand::GetObservation { robot_id: 0 }).await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        match handle.await.unwrap().unwrap() {
            SimResponse::Observation { state, .. } => {
                let c = state.combat.expect("combat robot should expose combat state");
                assert!((c.max_health - 75.0).abs() < 1e-3);
                assert!((c.stamina - 50.0).abs() < 1e-3);
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        // GetObservation path for non-combat robot
        let s = server.clone();
        let handle = tokio::spawn(async move {
            s.send_command(SimCommand::GetObservation { robot_id: 1 }).await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        match handle.await.unwrap().unwrap() {
            SimResponse::Observation { state, .. } => {
                assert!(
                    state.combat.is_none(),
                    "non-combat robot should emit combat: None"
                );
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        // Step path for combat robot — combat field still populated.
        let s = server.clone();
        let handle = tokio::spawn(async move {
            s.send_command(SimCommand::Step {
                robot_id: 0,
                action: RobotAction {
                    motor_velocities: vec![0.0; def.joints.len()],
                    gripper_commands: vec![],
                    base_velocity: [0.0, 0.0],
                },
            })
            .await
        });
        tokio::task::yield_now().await;
        client.process_pending(&mut manager, &[]);
        match handle.await.unwrap().unwrap() {
            SimResponse::Stepped { state, .. } => {
                let c = state.combat.expect("combat must survive Step");
                assert!((c.max_health - 75.0).abs() < 1e-3);
            }
            other => panic!("Expected Stepped, got {:?}", other),
        }
    }
}

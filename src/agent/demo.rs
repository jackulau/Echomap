use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::watch;
use tokio::time::{self, Duration};

use crate::agent::bridge::{SimBridgeServer, SimCommand, SimResponse};
use crate::robot::definition::RobotDefinition;
use crate::robot::state::RobotAction;

// ---------------------------------------------------------------------------
// Demo Agent Behaviors
// ---------------------------------------------------------------------------

/// Available demo behaviors the agent can perform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DemoBehavior {
    /// Sweep joints back and forth to exercise the arm.
    ReachTarget,
    /// Slowly rotate all joints to scan the environment with sensors.
    ExploreRoom,
    /// Use sensor readings to reactively adjust joint velocities.
    AvoidObstacles,
}

impl DemoBehavior {
    /// Human-readable name for UI display.
    pub fn label(self) -> &'static str {
        match self {
            Self::ReachTarget => "Reach Target",
            Self::ExploreRoom => "Explore Room",
            Self::AvoidObstacles => "Avoid Obstacles",
        }
    }
}

// ---------------------------------------------------------------------------
// DemoAgentHandle — UI-facing control handle
// ---------------------------------------------------------------------------

/// Handle returned when a demo agent is started.
/// Allows the UI to query status, change behavior, and stop the agent.
pub struct DemoAgentHandle {
    running: Arc<AtomicBool>,
    behavior_tx: watch::Sender<DemoBehavior>,
    /// Thread handle for the background task.
    thread: Option<std::thread::JoinHandle<()>>,
}

impl DemoAgentHandle {
    /// Check if the demo agent is still running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Change the active behavior.
    pub fn set_behavior(&self, behavior: DemoBehavior) {
        let _ = self.behavior_tx.send(behavior);
    }

    /// Signal the agent to stop (non-blocking). The agent loop will exit
    /// on its next iteration. The thread may still be alive until its
    /// current `send_command` completes.
    pub fn signal_stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    /// Stop the demo agent and wait for the thread to finish.
    /// Note: the caller must ensure bridge commands are being processed
    /// (via `process_pending`), otherwise the agent thread may block
    /// on a pending `send_command` and this call will deadlock.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for DemoAgentHandle {
    fn drop(&mut self) {
        if self.thread.is_some() {
            self.stop();
        }
    }
}

// ---------------------------------------------------------------------------
// start_demo_agent — spawns the agent loop
// ---------------------------------------------------------------------------

/// Start a demo agent that controls a robot via the simulation bridge.
///
/// The agent:
/// 1. Sends an `AddRobot` command to create a 3-joint arm.
/// 2. Runs a control loop at ~30 Hz, sending Step commands.
/// 3. Selects motor velocities based on the active `DemoBehavior`.
///
/// Returns a handle for UI control (stop, change behavior).
pub fn start_demo_agent(
    bridge: SimBridgeServer,
    initial_behavior: DemoBehavior,
) -> DemoAgentHandle {
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();
    let (behavior_tx, behavior_rx) = watch::channel(initial_behavior);

    let thread = std::thread::Builder::new()
        .name("demo-agent".to_string())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to build tokio runtime for demo agent");

            rt.block_on(async move {
                demo_agent_loop(bridge, running_clone, behavior_rx).await;
            });
        })
        .expect("failed to spawn demo agent thread");

    DemoAgentHandle {
        running,
        behavior_tx,
        thread: Some(thread),
    }
}

/// The main agent control loop.
async fn demo_agent_loop(
    bridge: SimBridgeServer,
    running: Arc<AtomicBool>,
    behavior_rx: watch::Receiver<DemoBehavior>,
) {
    // 1. Add a robot.
    let definition = RobotDefinition::simple_arm(3);
    let num_joints = definition.joints.len();
    let base_pose = glam::Mat4::from_translation(glam::Vec3::new(2.0, 0.0, 0.0));

    let robot_id = match bridge
        .send_command(SimCommand::AddRobot {
            definition,
            base_pose: base_pose.to_cols_array(),
        })
        .await
    {
        Ok(SimResponse::RobotAdded { robot_id }) => robot_id,
        Ok(other) => {
            log::error!("Demo agent: unexpected AddRobot response: {:?}", other);
            return;
        }
        Err(e) => {
            log::error!("Demo agent: AddRobot failed: {}", e);
            return;
        }
    };

    log::info!(
        "Demo agent started: controlling robot {} ({} joints)",
        robot_id,
        num_joints
    );

    // 2. Control loop at ~30 Hz.
    let mut interval = time::interval(Duration::from_millis(33));
    let mut step_count: u64 = 0;

    while running.load(Ordering::Relaxed) {
        interval.tick().await;

        let behavior = *behavior_rx.borrow();

        let velocities = compute_velocities(behavior, num_joints, step_count);

        let action = RobotAction {
            motor_velocities: velocities,
            gripper_commands: vec![],
            base_velocity: [0.0, 0.0],
        };

        match bridge
            .send_command(SimCommand::Step { robot_id, action })
            .await
        {
            Ok(SimResponse::Stepped { state, .. }) => {
                // Use sensor feedback for obstacle avoidance.
                if behavior == DemoBehavior::AvoidObstacles {
                    // The next iteration will get fresh readings.
                    let _ = state;
                }
            }
            Ok(SimResponse::Error { message }) => {
                log::warn!("Demo agent step error: {}", message);
            }
            Ok(_) => {}
            Err(e) => {
                log::warn!("Demo agent bridge error: {}", e);
                break;
            }
        }

        step_count += 1;
    }

    // 3. Clean up: remove the robot.
    let _ = bridge
        .send_command(SimCommand::RemoveRobot { robot_id })
        .await;

    log::info!("Demo agent stopped (robot {} removed)", robot_id);
    running.store(false, Ordering::Relaxed);
}

/// Compute motor velocities for the given behavior.
fn compute_velocities(behavior: DemoBehavior, num_joints: usize, step: u64) -> Vec<f32> {
    let t = step as f32 * 0.033; // approximate elapsed seconds

    match behavior {
        DemoBehavior::ReachTarget => {
            // Sinusoidal sweep: each joint oscillates at a different frequency.
            (0..num_joints)
                .map(|i| {
                    let freq = 0.5 + i as f32 * 0.3;
                    let phase = i as f32 * 0.8;
                    (t * freq + phase).sin() * 2.0
                })
                .collect()
        }
        DemoBehavior::ExploreRoom => {
            // Slow steady rotation on each joint with alternating directions.
            (0..num_joints)
                .map(|i| {
                    let dir = if i % 2 == 0 { 1.0 } else { -1.0 };
                    let speed = 0.3 + (i as f32 * 0.1);
                    dir * speed * ((t * 0.2).sin() * 0.5 + 0.5)
                })
                .collect()
        }
        DemoBehavior::AvoidObstacles => {
            // Oscillating motion with varying amplitude — in a real system
            // we'd use sensor feedback, but this creates visible motion.
            (0..num_joints)
                .map(|i| {
                    let base_freq = 0.8;
                    let amplitude = 1.5 - i as f32 * 0.3;
                    let amplitude = amplitude.max(0.3);
                    (t * base_freq + i as f32 * 1.2).sin() * amplitude
                })
                .collect()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_behavior_label() {
        assert_eq!(DemoBehavior::ReachTarget.label(), "Reach Target");
        assert_eq!(DemoBehavior::ExploreRoom.label(), "Explore Room");
        assert_eq!(DemoBehavior::AvoidObstacles.label(), "Avoid Obstacles");
    }

    #[test]
    fn test_compute_velocities_reach_target() {
        let vels = compute_velocities(DemoBehavior::ReachTarget, 3, 0);
        assert_eq!(vels.len(), 3);
        // All velocities should be finite
        for v in &vels {
            assert!(v.is_finite(), "velocity should be finite, got {}", v);
        }
    }

    #[test]
    fn test_compute_velocities_explore_room() {
        let vels = compute_velocities(DemoBehavior::ExploreRoom, 3, 100);
        assert_eq!(vels.len(), 3);
        for v in &vels {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_compute_velocities_avoid_obstacles() {
        let vels = compute_velocities(DemoBehavior::AvoidObstacles, 3, 50);
        assert_eq!(vels.len(), 3);
        for v in &vels {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_compute_velocities_zero_joints() {
        let vels = compute_velocities(DemoBehavior::ReachTarget, 0, 0);
        assert!(vels.is_empty());
    }

    #[test]
    fn test_compute_velocities_changes_over_time() {
        let v0 = compute_velocities(DemoBehavior::ReachTarget, 3, 0);
        let v100 = compute_velocities(DemoBehavior::ReachTarget, 3, 100);
        // At least one joint velocity should differ between step 0 and 100
        let differs = v0
            .iter()
            .zip(v100.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differs,
            "velocities should change over time: {:?} vs {:?}",
            v0, v100
        );
    }

    #[test]
    fn test_demo_behavior_equality() {
        assert_eq!(DemoBehavior::ReachTarget, DemoBehavior::ReachTarget);
        assert_ne!(DemoBehavior::ReachTarget, DemoBehavior::ExploreRoom);
        assert_ne!(DemoBehavior::ExploreRoom, DemoBehavior::AvoidObstacles);
    }

    #[test]
    fn test_start_and_stop_demo_agent() {
        use crate::agent::bridge::create_bridge;
        use crate::robot::RobotManager;

        let (bridge_server, mut bridge_client) = create_bridge();
        let mut handle = start_demo_agent(bridge_server, DemoBehavior::ReachTarget);

        assert!(handle.is_running());

        // Drain bridge commands in a polling loop so the agent can make
        // progress. The agent sends AddRobot then repeated Steps.
        let mut manager = RobotManager::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
        while manager.robots.is_empty() && std::time::Instant::now() < deadline {
            bridge_client.process_pending(&mut manager, &[]);
            std::thread::yield_now();
        }

        assert!(
            !manager.robots.is_empty(),
            "demo agent should have added a robot"
        );

        // Change behavior.
        handle.set_behavior(DemoBehavior::ExploreRoom);

        // Signal stop (sets running=false so the agent loop exits on next tick).
        handle.signal_stop();

        // Drop the bridge client. This closes the channel, causing the
        // agent's send_command to return an error and exit the loop.
        drop(bridge_client);

        // Wait for the thread to finish (should be fast now that the
        // channel is closed).
        handle.stop();
        assert!(!handle.is_running());
    }
}

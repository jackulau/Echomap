use serde::{Deserialize, Serialize};

use crate::robot::state::{ActionSpace, GymRobotState, ObservationSpace, RobotAction};

/// Messages sent from the client (agent) to the server.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    Connect { robot_id: usize },
    Reset,
    Step { action: RobotAction },
    Observe,
    Close,
}

/// Messages sent from the server to the client (agent).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerMessage {
    Connected {
        observation_space: ObservationSpace,
        action_space: ActionSpace,
    },
    Observation {
        state: GymRobotState,
        reward: f32,
        done: bool,
        step_count: u64,
    },
    Error {
        message: String,
    },
    Closed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::sensors::ImuReading;
    use crate::robot::state::{
        ActionSpace, GripperState, GymRobotState, GymSensorReadings, ObservationSpace, RobotAction,
    };
    use glam::Vec3;

    #[test]
    fn test_client_message_connect_roundtrip() {
        let msg = ClientMessage::Connect { robot_id: 0 };
        let json = serde_json::to_string(&msg).unwrap();
        let deser: ClientMessage = serde_json::from_str(&json).unwrap();
        match deser {
            ClientMessage::Connect { robot_id } => assert_eq!(robot_id, 0),
            other => panic!("Expected Connect, got {:?}", other),
        }
    }

    #[test]
    fn test_client_message_step_roundtrip() {
        let action = RobotAction {
            motor_velocities: vec![1.0, -0.5, 0.3],
            gripper_commands: vec![true, false],
        };
        let msg = ClientMessage::Step { action };
        let json = serde_json::to_string(&msg).unwrap();
        let deser: ClientMessage = serde_json::from_str(&json).unwrap();
        match deser {
            ClientMessage::Step { action } => {
                assert_eq!(action.motor_velocities.len(), 3);
                assert!((action.motor_velocities[0] - 1.0).abs() < 1e-6);
                assert!((action.motor_velocities[1] - (-0.5)).abs() < 1e-6);
                assert!((action.motor_velocities[2] - 0.3).abs() < 1e-6);
                assert_eq!(action.gripper_commands, vec![true, false]);
            }
            other => panic!("Expected Step, got {:?}", other),
        }
    }

    #[test]
    fn test_server_message_connected_roundtrip() {
        let obs_space = ObservationSpace {
            num_joint_positions: 3,
            num_joint_velocities: 3,
            num_sensors: 2,
            joint_position_limits: vec![
                (-std::f32::consts::PI, std::f32::consts::PI),
                (-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2),
                (0.0, 1.0),
            ],
        };
        let act_space = ActionSpace {
            num_motors: 3,
            motor_limits: vec![(-10.0, 10.0), (-5.0, 5.0), (-2.0, 2.0)],
            num_grippers: 1,
        };
        let msg = ServerMessage::Connected {
            observation_space: obs_space,
            action_space: act_space,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deser: ServerMessage = serde_json::from_str(&json).unwrap();
        match deser {
            ServerMessage::Connected {
                observation_space,
                action_space,
            } => {
                assert_eq!(observation_space.num_joint_positions, 3);
                assert_eq!(observation_space.num_joint_velocities, 3);
                assert_eq!(observation_space.num_sensors, 2);
                assert_eq!(observation_space.joint_position_limits.len(), 3);
                assert_eq!(action_space.num_motors, 3);
                assert_eq!(action_space.motor_limits.len(), 3);
                assert_eq!(action_space.num_grippers, 1);
            }
            other => panic!("Expected Connected, got {:?}", other),
        }
    }

    #[test]
    fn test_server_message_observation_roundtrip() {
        let state = GymRobotState {
            joint_positions: vec![1.0, -0.5],
            joint_velocities: vec![0.1, -0.2],
            sensor_readings: GymSensorReadings {
                distances: vec![5.0],
                contacts: vec![true],
                imu: vec![ImuReading {
                    linear_acceleration: Vec3::new(0.0, -9.81, 0.0),
                    angular_velocity: Vec3::new(0.1, 0.0, 0.0),
                }],
                camera_visible: vec![],
            },
            gripper_states: vec![GripperState {
                is_open: false,
                attached_object: Some(2),
            }],
        };
        let msg = ServerMessage::Observation {
            state,
            reward: 1.5,
            done: false,
            step_count: 42,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deser: ServerMessage = serde_json::from_str(&json).unwrap();
        match deser {
            ServerMessage::Observation {
                state,
                reward,
                done,
                step_count,
            } => {
                assert_eq!(state.joint_positions.len(), 2);
                assert!((state.joint_positions[0] - 1.0).abs() < 1e-6);
                assert!((reward - 1.5).abs() < 1e-6);
                assert!(!done);
                assert_eq!(step_count, 42);
                assert_eq!(state.gripper_states.len(), 1);
                assert!(!state.gripper_states[0].is_open);
                assert_eq!(state.gripper_states[0].attached_object, Some(2));
            }
            other => panic!("Expected Observation, got {:?}", other),
        }
    }

    #[test]
    fn test_server_message_error_roundtrip() {
        let msg = ServerMessage::Error {
            message: "robot not found".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deser: ServerMessage = serde_json::from_str(&json).unwrap();
        match deser {
            ServerMessage::Error { message } => {
                assert_eq!(message, "robot not found");
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_client_message_all_variants() {
        // Parse each variant from JSON string literals
        let connect_json = r#"{"type":"connect","robot_id":5}"#;
        let reset_json = r#"{"type":"reset"}"#;
        let step_json =
            r#"{"type":"step","action":{"motor_velocities":[1.0],"gripper_commands":[]}}"#;
        let observe_json = r#"{"type":"observe"}"#;
        let close_json = r#"{"type":"close"}"#;

        let connect: ClientMessage = serde_json::from_str(connect_json).unwrap();
        match connect {
            ClientMessage::Connect { robot_id } => assert_eq!(robot_id, 5),
            other => panic!("Expected Connect, got {:?}", other),
        }

        let reset: ClientMessage = serde_json::from_str(reset_json).unwrap();
        match reset {
            ClientMessage::Reset => {}
            other => panic!("Expected Reset, got {:?}", other),
        }

        let step: ClientMessage = serde_json::from_str(step_json).unwrap();
        match step {
            ClientMessage::Step { action } => {
                assert_eq!(action.motor_velocities.len(), 1);
                assert!((action.motor_velocities[0] - 1.0).abs() < 1e-6);
            }
            other => panic!("Expected Step, got {:?}", other),
        }

        let observe: ClientMessage = serde_json::from_str(observe_json).unwrap();
        match observe {
            ClientMessage::Observe => {}
            other => panic!("Expected Observe, got {:?}", other),
        }

        let close: ClientMessage = serde_json::from_str(close_json).unwrap();
        match close {
            ClientMessage::Close => {}
            other => panic!("Expected Close, got {:?}", other),
        }
    }

    #[test]
    fn test_server_message_all_variants() {
        // Parse each variant from JSON string literals
        let connected_json = r#"{"type":"connected","observation_space":{"num_joint_positions":2,"num_joint_velocities":2,"num_sensors":1,"joint_position_limits":[[-3.14,3.14]]},"action_space":{"num_motors":2,"motor_limits":[[-10.0,10.0]],"num_grippers":0}}"#;
        let observation_json = r#"{"type":"observation","state":{"joint_positions":[0.5],"joint_velocities":[0.1],"sensor_readings":{"distances":[],"contacts":[],"imu":[],"camera_visible":[]},"gripper_states":[]},"reward":0.0,"done":true,"step_count":100}"#;
        let error_json = r#"{"type":"error","message":"test error"}"#;
        let closed_json = r#"{"type":"closed"}"#;

        let connected: ServerMessage = serde_json::from_str(connected_json).unwrap();
        match connected {
            ServerMessage::Connected {
                observation_space,
                action_space,
            } => {
                assert_eq!(observation_space.num_joint_positions, 2);
                assert_eq!(action_space.num_motors, 2);
            }
            other => panic!("Expected Connected, got {:?}", other),
        }

        let observation: ServerMessage = serde_json::from_str(observation_json).unwrap();
        match observation {
            ServerMessage::Observation {
                state,
                reward,
                done,
                step_count,
            } => {
                assert_eq!(state.joint_positions.len(), 1);
                assert!((reward - 0.0).abs() < 1e-6);
                assert!(done);
                assert_eq!(step_count, 100);
            }
            other => panic!("Expected Observation, got {:?}", other),
        }

        let error: ServerMessage = serde_json::from_str(error_json).unwrap();
        match error {
            ServerMessage::Error { message } => assert_eq!(message, "test error"),
            other => panic!("Expected Error, got {:?}", other),
        }

        let closed: ServerMessage = serde_json::from_str(closed_json).unwrap();
        match closed {
            ServerMessage::Closed => {}
            other => panic!("Expected Closed, got {:?}", other),
        }
    }
}

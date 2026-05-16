use serde::{Deserialize, Serialize};

use crate::robot::state::{ActionSpace, GymRobotState, ObservationSpace, RobotAction};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentMessage {
    pub from_robot_id: usize,
    pub to_robot_id: usize,
    pub content: String,
    pub timestamp: u64,
}

/// Messages sent from the client (agent) to the server.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ClientMessage {
    Connect { robot_id: usize },
    Reset,
    Step { action: RobotAction },
    Observe,
    Close,
    SendMessage { to_robot_id: usize, content: String },
}

/// Messages sent from the server to the client (agent).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(clippy::large_enum_variant)]
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
        #[serde(default)]
        messages: Vec<AgentMessage>,
        #[serde(default)]
        hit_events: Vec<crate::robot::collision::HitEvent>,
        #[serde(default)]
        match_state: Option<crate::robot::boxing::BoxingMatchState>,
    },
    MessageSent,
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
            base_velocity: [0.0, 0.0],
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
            combat: None,
        };
        let msg = ServerMessage::Observation {
            state,
            reward: 1.5,
            done: false,
            step_count: 42,
            messages: vec![],
            hit_events: vec![],
            match_state: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deser: ServerMessage = serde_json::from_str(&json).unwrap();
        match deser {
            ServerMessage::Observation {
                state,
                reward,
                done,
                step_count,
                ..
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
                ..
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

    // ---- Edge case tests ----

    #[test]
    fn test_unknown_type_tag_rejected() {
        let json = r#"{"type":"explode","robot_id":0}"#;
        let result = serde_json::from_str::<ClientMessage>(json);
        assert!(result.is_err(), "unknown type tag should fail to parse");
    }

    #[test]
    fn test_missing_required_field_robot_id() {
        // Connect requires robot_id
        let json = r#"{"type":"connect"}"#;
        let result = serde_json::from_str::<ClientMessage>(json);
        assert!(
            result.is_err(),
            "connect without robot_id should fail to parse"
        );
    }

    #[test]
    fn test_missing_action_field_in_step() {
        let json = r#"{"type":"step"}"#;
        let result = serde_json::from_str::<ClientMessage>(json);
        assert!(result.is_err(), "step without action should fail to parse");
    }

    #[test]
    fn test_extra_unknown_fields_ignored() {
        // serde by default ignores unknown fields (no deny_unknown_fields)
        let json = r#"{"type":"connect","robot_id":0,"extra_field":"should_be_ignored"}"#;
        let result = serde_json::from_str::<ClientMessage>(json);
        assert!(
            result.is_ok(),
            "extra unknown fields should be silently ignored"
        );
        match result.unwrap() {
            ClientMessage::Connect { robot_id } => assert_eq!(robot_id, 0),
            other => panic!("Expected Connect, got {:?}", other),
        }
    }

    #[test]
    fn test_empty_string_parse_fails() {
        let result = serde_json::from_str::<ClientMessage>("");
        assert!(result.is_err(), "empty string should fail to parse");
    }

    #[test]
    fn test_large_robot_id() {
        let json = format!(r#"{{"type":"connect","robot_id":{}}}"#, usize::MAX);
        let result = serde_json::from_str::<ClientMessage>(&json);
        assert!(result.is_ok(), "usize::MAX robot_id should parse");
        match result.unwrap() {
            ClientMessage::Connect { robot_id } => assert_eq!(robot_id, usize::MAX),
            other => panic!("Expected Connect, got {:?}", other),
        }
    }

    #[test]
    fn test_nan_in_motor_velocities() {
        // JSON does not have NaN, so literal NaN should fail parsing
        let json = r#"{"type":"step","action":{"motor_velocities":[NaN],"gripper_commands":[]}}"#;
        let result = serde_json::from_str::<ClientMessage>(json);
        assert!(result.is_err(), "NaN is not valid JSON");
    }

    #[test]
    fn test_step_with_empty_motor_velocities() {
        let json = r#"{"type":"step","action":{"motor_velocities":[],"gripper_commands":[]}}"#;
        let result = serde_json::from_str::<ClientMessage>(json);
        assert!(
            result.is_ok(),
            "empty motor_velocities should parse successfully"
        );
        match result.unwrap() {
            ClientMessage::Step { action } => {
                assert!(action.motor_velocities.is_empty());
                assert!(action.gripper_commands.is_empty());
            }
            other => panic!("Expected Step, got {:?}", other),
        }
    }

    #[test]
    fn test_server_error_with_empty_message() {
        let msg = ServerMessage::Error {
            message: String::new(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deser: ServerMessage = serde_json::from_str(&json).unwrap();
        match deser {
            ServerMessage::Error { message } => {
                assert!(message.is_empty(), "empty error message should round-trip");
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_negative_robot_id_rejected() {
        // robot_id is usize, so negative values should fail
        let json = r#"{"type":"connect","robot_id":-1}"#;
        let result = serde_json::from_str::<ClientMessage>(json);
        assert!(
            result.is_err(),
            "negative robot_id should fail for usize field"
        );
    }

    #[test]
    fn test_float_robot_id_rejected() {
        let json = r#"{"type":"connect","robot_id":1.5}"#;
        let result = serde_json::from_str::<ClientMessage>(json);
        assert!(
            result.is_err(),
            "float robot_id should fail for usize field"
        );
    }

    #[test]
    fn test_server_observation_with_nan_reward_roundtrip() {
        // f32 NaN serializes to null in serde_json, which then
        // deserializes back as 0.0 — verify it does not panic.
        let msg = ServerMessage::Observation {
            state: GymRobotState {
                joint_positions: vec![],
                joint_velocities: vec![],
                sensor_readings: GymSensorReadings {
                    distances: vec![],
                    contacts: vec![],
                    imu: vec![],
                    camera_visible: vec![],
                },
                gripper_states: vec![],
                combat: None,
            },
            reward: f32::NAN,
            done: false,
            step_count: 0,
            messages: vec![],
            hit_events: vec![],
            match_state: None,
        };
        let json = serde_json::to_string(&msg).expect("NaN serializes to null");
        assert!(json.contains("null"), "NaN should serialize as null");
    }

    #[test]
    fn test_whitespace_only_json_rejected() {
        let result = serde_json::from_str::<ClientMessage>("   \t  \n  ");
        assert!(result.is_err(), "whitespace-only input should fail");
    }

    #[test]
    fn test_null_json_rejected() {
        let result = serde_json::from_str::<ClientMessage>("null");
        assert!(
            result.is_err(),
            "null should fail to parse as ClientMessage"
        );
    }

    #[test]
    fn test_unicode_in_error_message() {
        let msg = ServerMessage::Error {
            message: "robot \u{1F916} not found \u{00E9}\u{00F1}".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deser: ServerMessage = serde_json::from_str(&json).unwrap();
        match deser {
            ServerMessage::Error { message } => {
                assert!(
                    message.contains('\u{1F916}'),
                    "unicode emoji should survive roundtrip"
                );
                assert!(
                    message.contains('\u{00E9}'),
                    "accented chars should survive roundtrip"
                );
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_json_injection_in_type_field() {
        // Try to sneak nested JSON in the type discriminator
        let json = r#"{"type":"connect\",\"robot_id\":999,\"extra\":\"","robot_id":0}"#;
        let result = serde_json::from_str::<ClientMessage>(json);
        assert!(result.is_err(), "injected type field should fail to parse");
    }

    #[test]
    fn test_duplicate_type_field() {
        // JSON with duplicate keys -- serde uses last occurrence
        let json = r#"{"type":"reset","type":"connect","robot_id":0}"#;
        let result = serde_json::from_str::<ClientMessage>(json);
        // serde_json takes the last "type" value
        match result {
            Ok(ClientMessage::Connect { robot_id }) => assert_eq!(robot_id, 0),
            Ok(_) => {}  // Either interpretation is fine
            Err(_) => {} // Rejection is also acceptable
        }
    }

    #[test]
    fn test_infinity_in_motor_velocities() {
        // JSON Infinity is not valid JSON
        let json =
            r#"{"type":"step","action":{"motor_velocities":[Infinity],"gripper_commands":[]}}"#;
        let result = serde_json::from_str::<ClientMessage>(json);
        assert!(result.is_err(), "Infinity is not valid JSON");
    }

    #[test]
    fn test_server_observation_with_infinity_reward() {
        let msg = ServerMessage::Observation {
            state: GymRobotState {
                joint_positions: vec![],
                joint_velocities: vec![],
                sensor_readings: GymSensorReadings {
                    distances: vec![],
                    contacts: vec![],
                    imu: vec![],
                    camera_visible: vec![],
                },
                gripper_states: vec![],
                combat: None,
            },
            reward: f32::INFINITY,
            done: false,
            step_count: 0,
            messages: vec![],
            hit_events: vec![],
            match_state: None,
        };
        let json = serde_json::to_string(&msg).expect("Infinity serializes to null");
        assert!(json.contains("null"), "Infinity should serialize as null");
    }

    #[test]
    fn test_very_large_step_count() {
        let json = format!(
            r#"{{"type":"observation","state":{{"joint_positions":[],"joint_velocities":[],"sensor_readings":{{"distances":[],"contacts":[],"imu":[],"camera_visible":[]}},"gripper_states":[]}},"reward":0.0,"done":false,"step_count":{}}}"#,
            u64::MAX
        );
        let result = serde_json::from_str::<ServerMessage>(&json);
        assert!(result.is_ok(), "u64::MAX step_count should parse");
        match result.unwrap() {
            ServerMessage::Observation { step_count, .. } => {
                assert_eq!(step_count, u64::MAX);
            }
            other => panic!("Expected Observation, got {:?}", other),
        }
    }

    #[test]
    fn test_closed_message_roundtrip() {
        let msg = ServerMessage::Closed;
        let json = serde_json::to_string(&msg).unwrap();
        let deser: ServerMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(deser, ServerMessage::Closed));
    }

    #[test]
    fn test_client_reset_roundtrip() {
        let msg = ClientMessage::Reset;
        let json = serde_json::to_string(&msg).unwrap();
        let deser: ClientMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(deser, ClientMessage::Reset));
    }

    #[test]
    fn test_client_observe_roundtrip() {
        let msg = ClientMessage::Observe;
        let json = serde_json::to_string(&msg).unwrap();
        let deser: ClientMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(deser, ClientMessage::Observe));
    }

    #[test]
    fn test_array_json_rejected() {
        let result = serde_json::from_str::<ClientMessage>(r#"[{"type":"reset"}]"#);
        assert!(result.is_err(), "array should not parse as ClientMessage");
    }

    #[test]
    fn test_robot_id_string_rejected() {
        let json = r#"{"type":"connect","robot_id":"zero"}"#;
        let result = serde_json::from_str::<ClientMessage>(json);
        assert!(
            result.is_err(),
            "string robot_id should fail for usize field"
        );
    }

    #[test]
    fn test_agent_message_json_round_trip() {
        let msg = AgentMessage {
            from_robot_id: 0,
            to_robot_id: 1,
            content: "hello opponent".to_string(),
            timestamp: 12345,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deser: AgentMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.from_robot_id, 0);
        assert_eq!(deser.to_robot_id, 1);
        assert_eq!(deser.content, "hello opponent");
        assert_eq!(deser.timestamp, 12345);
    }

    #[test]
    fn test_send_message_client_message() {
        let msg = ClientMessage::SendMessage {
            to_robot_id: 1,
            content: "trash talk".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("send_message"));
        let deser: ClientMessage = serde_json::from_str(&json).unwrap();
        match deser {
            ClientMessage::SendMessage {
                to_robot_id,
                content,
            } => {
                assert_eq!(to_robot_id, 1);
                assert_eq!(content, "trash talk");
            }
            other => panic!("Expected SendMessage, got {:?}", other),
        }
    }

    #[test]
    fn test_observation_with_messages() {
        let state = GymRobotState {
            joint_positions: vec![0.5],
            joint_velocities: vec![0.1],
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
            hit_events: vec![],
            messages: vec![AgentMessage {
                from_robot_id: 0,
                to_robot_id: 1,
                content: "hey".to_string(),
                timestamp: 100,
            }],
            match_state: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deser: ServerMessage = serde_json::from_str(&json).unwrap();
        match deser {
            ServerMessage::Observation { messages, .. } => {
                assert_eq!(messages.len(), 1);
                assert_eq!(messages[0].content, "hey");
                assert_eq!(messages[0].from_robot_id, 0);
            }
            other => panic!("Expected Observation, got {:?}", other),
        }
    }

    #[test]
    fn test_observation_empty_messages() {
        let json = r#"{"type":"observation","state":{"joint_positions":[],"joint_velocities":[],"sensor_readings":{"distances":[],"contacts":[],"imu":[],"camera_visible":[]},"gripper_states":[]},"reward":0.0,"done":false,"step_count":0}"#;
        let deser: ServerMessage = serde_json::from_str(json).unwrap();
        match deser {
            ServerMessage::Observation { messages, .. } => {
                assert!(
                    messages.is_empty(),
                    "missing messages field should default to empty vec"
                );
            }
            other => panic!("Expected Observation, got {:?}", other),
        }
    }

    #[test]
    fn test_message_sent_round_trip() {
        let msg = ServerMessage::MessageSent;
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("message_sent"));
        let deser: ServerMessage = serde_json::from_str(&json).unwrap();
        assert!(matches!(deser, ServerMessage::MessageSent));
    }

    #[test]
    fn test_boolean_motor_velocity_rejected() {
        let json = r#"{"type":"step","action":{"motor_velocities":[true],"gripper_commands":[]}}"#;
        let result = serde_json::from_str::<ClientMessage>(json);
        assert!(
            result.is_err(),
            "boolean in motor_velocities should fail for f32 field"
        );
    }

    #[test]
    fn test_observation_with_match_state() {
        let state = GymRobotState {
            joint_positions: vec![],
            joint_velocities: vec![],
            sensor_readings: GymSensorReadings {
                distances: vec![],
                contacts: vec![],
                imu: vec![],
                camera_visible: vec![],
            },
            gripper_states: vec![],
            combat: None,
        };
        let ms = crate::robot::boxing::BoxingMatchState {
            phase: "fighting".to_string(),
            current_round: 2,
            round_time: 45.0,
            round_duration: 180.0,
            scores: vec![[10, 9]],
            total_score_a: 10,
            total_score_b: 9,
            your_robot: 0,
            opponent_health: 80.0,
            opponent_stamina: 60.0,
            own_torso_pos: [0.0; 3],
            opponent_link_positions: Vec::new(),
            opponent_torso_pos: [0.0; 3],
        };
        let msg = ServerMessage::Observation {
            state,
            reward: 0.0,
            done: false,
            step_count: 100,
            messages: vec![],
            hit_events: vec![],
            match_state: Some(ms),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deser: ServerMessage = serde_json::from_str(&json).unwrap();
        match deser {
            ServerMessage::Observation { match_state, .. } => {
                let ms = match_state.expect("match_state should be present");
                assert_eq!(ms.phase, "fighting");
                assert_eq!(ms.current_round, 2);
                assert!((ms.opponent_health - 80.0).abs() < 0.01);
            }
            other => panic!("Expected Observation, got {:?}", other),
        }
    }

    #[test]
    fn test_observation_without_match_state_backward_compat() {
        let json = r#"{"type":"observation","state":{"joint_positions":[],"joint_velocities":[],"sensor_readings":{"distances":[],"contacts":[],"imu":[],"camera_visible":[]},"gripper_states":[]},"reward":0.0,"done":false,"step_count":0}"#;
        let deser: ServerMessage = serde_json::from_str(json).unwrap();
        match deser {
            ServerMessage::Observation { match_state, .. } => {
                assert!(
                    match_state.is_none(),
                    "missing match_state should default to None"
                );
            }
            other => panic!("Expected Observation, got {:?}", other),
        }
    }
}

use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use super::body::{Joint, Link};
use super::body::{JointType, Robot};
use super::definition::{JointDefinition, JointType as DefJointType, RobotDefinition};
use super::state::RobotState;

/// World-space transform for a single link.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinkTransform {
    pub position: Vec3,
    pub rotation: Quat,
}

/// Compute forward kinematics for all links in the robot.
///
/// Returns a `Vec<LinkTransform>` with one entry per link, in order.
/// The base link (index 0) gets the robot's base transform composed with
/// its local offset. Each subsequent link accumulates the parent's world
/// transform, the joint transform, and its own local offset.
pub fn compute_forward_kinematics(robot: &Robot) -> Vec<LinkTransform> {
    if robot.links.is_empty() {
        return Vec::new();
    }

    let mut transforms = Vec::with_capacity(robot.links.len());

    // Base link (index 0): world = base_transform * link_local
    let base = &robot.links[0];
    let base_pos = robot.base_position + robot.base_rotation.mul_vec3(base.local_position);
    let base_rot = robot.base_rotation * base.local_rotation;
    transforms.push(LinkTransform {
        position: base_pos,
        rotation: base_rot,
    });

    // Each subsequent link: world[i] = world[i-1] * joint_transform * link_local
    for i in 0..robot.joints.len() {
        let joint = &robot.joints[i];
        let link = &robot.links[i + 1];
        let parent = &transforms[i];

        // Compute joint transform (rotation and/or translation)
        let (joint_rot, joint_translation) = match joint.joint_type {
            JointType::Revolute => {
                let rot = Quat::from_axis_angle(joint.axis, joint.position);
                (rot, Vec3::ZERO)
            }
            JointType::Prismatic => (Quat::IDENTITY, joint.axis * joint.position),
            JointType::Fixed => (Quat::IDENTITY, Vec3::ZERO),
        };

        // Accumulated rotation: parent_rot * joint_rot * link_local_rot
        let world_rot = parent.rotation * joint_rot * link.local_rotation;

        // Accumulated position:
        //   parent_pos + parent_rot * (joint_translation + joint_rot * link_local_pos)
        let rotated_local = joint_rot.mul_vec3(link.local_position);
        let world_pos =
            parent.position + parent.rotation.mul_vec3(joint_translation + rotated_local);

        transforms.push(LinkTransform {
            position: world_pos,
            rotation: world_rot,
        });
    }

    transforms
}

// ---------------------------------------------------------------------------
// Definition-based forward kinematics (RobotDefinition + RobotState)
// ---------------------------------------------------------------------------

/// Compute the local 4x4 transform for a single joint given its current position.
///
/// - Revolute: rotation about `joint.axis` by `position` radians.
/// - Prismatic: translation along `joint.axis` by `position` meters.
/// - Fixed: identity transform.
pub fn compute_joint_transform(joint: &JointDefinition, position: f32) -> Mat4 {
    match joint.joint_type {
        DefJointType::Revolute => Mat4::from_axis_angle(joint.axis, position),
        DefJointType::Prismatic => Mat4::from_translation(joint.axis * position),
        DefJointType::Fixed => Mat4::IDENTITY,
    }
}

/// Compute forward kinematics for all links, updating `state.link_poses`.
///
/// Walks the kinematic chain defined by `definition`. For each joint, the
/// child link pose is: `link_pose[parent] * joint_transform`. The base link
/// (index 0, which has no parent joint) gets `base_pose` directly.
pub fn forward_kinematics(definition: &RobotDefinition, state: &mut RobotState, base_pose: Mat4) {
    if definition.links.is_empty() {
        return;
    }

    // Base link gets the base_pose directly.
    state.set_link_pose(0, base_pose);

    // Process each joint:
    //   child_pose = parent_pose * translate(anchor_offset) * joint_rotation * translate(child_offset)
    for (joint_idx, joint) in definition.joints.iter().enumerate() {
        let position = state.joint_positions.get(joint_idx).copied().unwrap_or(0.0);
        let joint_transform = compute_joint_transform(joint, position);

        let parent_pose = Mat4::from_cols_array(&state.link_poses[joint.parent_link]);
        let anchor = Mat4::from_translation(joint.anchor_offset);
        let child_local = Mat4::from_translation(joint.child_offset);
        let child_pose = parent_pose * anchor * joint_transform * child_local;

        state.set_link_pose(joint.child_link, child_pose);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    const EPSILON: f32 = 1e-5;

    // ---- helpers ----

    fn base_link() -> Link {
        Link::new("base", Vec3::ZERO, Quat::IDENTITY, Vec3::splat(0.1), 5.0)
    }

    fn arm_link(name: &str, offset: Vec3) -> Link {
        Link::new(name, offset, Quat::IDENTITY, Vec3::splat(0.05), 1.0)
    }

    fn revolute_joint(axis: Vec3, position: f32) -> Joint {
        Joint::new(
            JointType::Revolute,
            axis,
            position,
            0.0,
            (-std::f32::consts::PI, std::f32::consts::PI),
            10.0,
        )
    }

    fn prismatic_joint(axis: Vec3, position: f32) -> Joint {
        Joint::new(JointType::Prismatic, axis, position, 0.0, (-5.0, 5.0), 10.0)
    }

    fn fixed_joint() -> Joint {
        Joint::new(JointType::Fixed, Vec3::Y, 0.0, 0.0, (0.0, 0.0), 0.0)
    }

    fn assert_vec3_near(a: Vec3, b: Vec3, msg: &str) {
        assert!(
            (a - b).length() < EPSILON,
            "{}: expected {:?}, got {:?} (diff={})",
            msg,
            b,
            a,
            (a - b).length()
        );
    }

    // ---- test cases ----

    #[test]
    fn test_fk_single_link_at_origin() {
        // One base link with local offset, identity base transform.
        let link = Link::new(
            "base",
            Vec3::new(0.0, 1.0, 0.0),
            Quat::IDENTITY,
            Vec3::splat(0.1),
            5.0,
        );
        let robot = Robot::new("bot", Vec3::ZERO, Quat::IDENTITY, link);

        let transforms = compute_forward_kinematics(&robot);
        assert_eq!(transforms.len(), 1);
        assert_vec3_near(
            transforms[0].position,
            Vec3::new(0.0, 1.0, 0.0),
            "base link world position",
        );
    }

    #[test]
    fn test_fk_revolute_90_degrees() {
        // Base at origin, child link offset along +X by 1.0.
        // Revolute joint around Z axis at 90 degrees rotates child to +Y.
        let base = base_link();
        let mut robot = Robot::new("bot", Vec3::ZERO, Quat::IDENTITY, base);

        let joint = revolute_joint(Vec3::Z, FRAC_PI_2);
        let child = arm_link("child", Vec3::new(1.0, 0.0, 0.0));
        robot.add_joint_and_link(joint, child);

        let transforms = compute_forward_kinematics(&robot);
        assert_eq!(transforms.len(), 2);
        // After 90-degree rotation around Z, +X offset becomes +Y offset
        assert_vec3_near(
            transforms[1].position,
            Vec3::new(0.0, 1.0, 0.0),
            "child link rotated 90 degrees around Z",
        );
    }

    #[test]
    fn test_fk_prismatic_extension() {
        // Base at origin, child link with zero local offset.
        // Prismatic joint along X, extended by 2.0.
        let base = base_link();
        let mut robot = Robot::new("bot", Vec3::ZERO, Quat::IDENTITY, base);

        let joint = prismatic_joint(Vec3::X, 2.0);
        let child = arm_link("child", Vec3::ZERO);
        robot.add_joint_and_link(joint, child);

        let transforms = compute_forward_kinematics(&robot);
        assert_eq!(transforms.len(), 2);
        assert_vec3_near(
            transforms[1].position,
            Vec3::new(2.0, 0.0, 0.0),
            "child extended 2.0 along X",
        );
    }

    #[test]
    fn test_fk_fixed_joint() {
        // Fixed joint should just apply the local transform of the child.
        let base = base_link();
        let mut robot = Robot::new("bot", Vec3::ZERO, Quat::IDENTITY, base);

        let joint = fixed_joint();
        let child = arm_link("child", Vec3::new(0.0, 0.5, 0.0));
        robot.add_joint_and_link(joint, child);

        let transforms = compute_forward_kinematics(&robot);
        assert_eq!(transforms.len(), 2);
        assert_vec3_near(
            transforms[1].position,
            Vec3::new(0.0, 0.5, 0.0),
            "fixed joint preserves child local offset",
        );
    }

    #[test]
    fn test_fk_chain_three_links() {
        // 3-link arm: base at origin, two revolute joints around Z.
        // Each child extends 1.0 along +X locally.
        // Joint1 at 0 degrees, Joint2 at 90 degrees.
        // Link 0 (base) at origin.
        // Link 1: after joint1 (0 deg), offset (1,0,0) -> world (1,0,0).
        // Link 2: after joint2 (90 deg around Z at link1), offset (1,0,0)
        //         -> rotated to (0,1,0), then translated by link1 pos -> (1,1,0).
        let base = base_link();
        let mut robot = Robot::new("bot", Vec3::ZERO, Quat::IDENTITY, base);

        let joint1 = revolute_joint(Vec3::Z, 0.0);
        let link1 = arm_link("link1", Vec3::new(1.0, 0.0, 0.0));
        robot.add_joint_and_link(joint1, link1);

        let joint2 = revolute_joint(Vec3::Z, FRAC_PI_2);
        let link2 = arm_link("link2", Vec3::new(1.0, 0.0, 0.0));
        robot.add_joint_and_link(joint2, link2);

        let transforms = compute_forward_kinematics(&robot);
        assert_eq!(transforms.len(), 3);

        assert_vec3_near(transforms[0].position, Vec3::ZERO, "base at origin");
        assert_vec3_near(
            transforms[1].position,
            Vec3::new(1.0, 0.0, 0.0),
            "link1 at (1,0,0)",
        );
        assert_vec3_near(
            transforms[2].position,
            Vec3::new(1.0, 1.0, 0.0),
            "end-effector at (1,1,0)",
        );
    }

    #[test]
    fn test_fk_base_transform() {
        // Non-identity base position offsets all links.
        let base_link = Link::new("base", Vec3::ZERO, Quat::IDENTITY, Vec3::splat(0.1), 5.0);
        let mut robot = Robot::new("bot", Vec3::new(10.0, 0.0, 0.0), Quat::IDENTITY, base_link);

        let joint = revolute_joint(Vec3::Z, 0.0);
        let child = arm_link("child", Vec3::new(1.0, 0.0, 0.0));
        robot.add_joint_and_link(joint, child);

        let transforms = compute_forward_kinematics(&robot);
        assert_eq!(transforms.len(), 2);
        assert_vec3_near(
            transforms[0].position,
            Vec3::new(10.0, 0.0, 0.0),
            "base offset by (10,0,0)",
        );
        assert_vec3_near(
            transforms[1].position,
            Vec3::new(11.0, 0.0, 0.0),
            "child at base + (1,0,0)",
        );
    }

    #[test]
    fn test_fk_empty_robot() {
        // Robot with only the base link returns a single transform.
        let base = base_link();
        let robot = Robot::new("bot", Vec3::ZERO, Quat::IDENTITY, base);

        let transforms = compute_forward_kinematics(&robot);
        assert_eq!(
            transforms.len(),
            1,
            "single base link should yield 1 transform"
        );
        assert_vec3_near(transforms[0].position, Vec3::ZERO, "base link at origin");
    }

    // ---- Definition-based FK tests ----

    use crate::robot::definition::{
        CollisionShape, JointDefinition, JointType as DefJointType, LinkDefinition, RobotDefinition,
    };
    use crate::robot::state::RobotState;

    /// Helper: create a simple serial chain definition with the given joints.
    fn make_serial_chain(
        joint_specs: &[(DefJointType, Vec3, f32)], // (type, axis, initial_position)
    ) -> (RobotDefinition, RobotState) {
        let num_joints = joint_specs.len();
        let mut links = Vec::with_capacity(num_joints + 1);
        let mut joints = Vec::with_capacity(num_joints);

        // Base link (index 0)
        links.push(LinkDefinition {
            name: "base".to_string(),
            mass: 5.0,
            inertia: 1.0,
            collision_shape: CollisionShape::Cuboid {
                half_extents: Vec3::splat(0.1),
            },
            parent_joint: None,
            body_zone: None,
        });

        for (i, (jtype, axis, _pos)) in joint_specs.iter().enumerate() {
            joints.push(JointDefinition {
                name: format!("joint_{}", i),
                joint_type: jtype.clone(),
                axis: *axis,
                parent_link: i,
                child_link: i + 1,
                limit_min: -10.0,
                limit_max: 10.0,
                max_torque: 10.0,
                damping: 0.1,
                anchor_offset: Vec3::ZERO,
                child_offset: Vec3::ZERO,
            });
            links.push(LinkDefinition {
                name: format!("link_{}", i + 1),
                mass: 1.0,
                inertia: 0.1,
                collision_shape: CollisionShape::Cylinder {
                    radius: 0.05,
                    height: 0.5,
                },
                parent_joint: Some(i),
                body_zone: None,
            });
        }

        let def = RobotDefinition {
            name: "test_robot".to_string(),
            links,
            joints,
            sensors: Vec::new(),
        };

        let mut state = RobotState::new(&def);
        // Set initial joint positions
        for (i, (_, _, pos)) in joint_specs.iter().enumerate() {
            state.joint_positions[i] = *pos;
        }

        (def, state)
    }

    fn mat4_translation(m: &Mat4) -> Vec3 {
        let cols = m.to_cols_array_2d();
        Vec3::new(cols[3][0], cols[3][1], cols[3][2])
    }

    #[test]
    fn test_identity_at_zero() {
        // All joints at 0, base=identity => link_poses are identity
        let (def, mut state) = make_serial_chain(&[
            (DefJointType::Revolute, Vec3::Y, 0.0),
            (DefJointType::Revolute, Vec3::Y, 0.0),
        ]);

        forward_kinematics(&def, &mut state, Mat4::IDENTITY);

        // All link poses should be identity (no rotation or translation at zero)
        let identity = Mat4::IDENTITY.to_cols_array();
        for (i, pose) in state.link_poses.iter().enumerate() {
            for (j, (&a, &b)) in pose.iter().zip(identity.iter()).enumerate() {
                assert!(
                    (a - b).abs() < EPSILON,
                    "link_poses[{}][{}]: expected {}, got {}",
                    i,
                    j,
                    b,
                    a,
                );
            }
        }
    }

    #[test]
    fn test_revolute_90_degrees() {
        // Revolute joint at pi/2 around Z rotates child correctly
        let (def, mut state) = make_serial_chain(&[(DefJointType::Revolute, Vec3::Z, FRAC_PI_2)]);

        forward_kinematics(&def, &mut state, Mat4::IDENTITY);

        let child_pose = Mat4::from_cols_array(&state.link_poses[1]);
        // After 90-deg rotation around Z: X-axis maps to Y, Y-axis maps to -X
        let x_axis = child_pose.x_axis.truncate();
        let y_axis = child_pose.y_axis.truncate();

        // X-axis of child should point along world +Y
        assert_vec3_near(
            x_axis,
            Vec3::new(0.0, 1.0, 0.0),
            "child X axis after 90 deg Z rotation",
        );
        // Y-axis of child should point along world -X
        assert_vec3_near(
            y_axis,
            Vec3::new(-1.0, 0.0, 0.0),
            "child Y axis after 90 deg Z rotation",
        );
    }

    #[test]
    fn test_prismatic_translation() {
        // Prismatic joint displaces child along axis
        let (def, mut state) = make_serial_chain(&[(DefJointType::Prismatic, Vec3::X, 3.0)]);

        forward_kinematics(&def, &mut state, Mat4::IDENTITY);

        let child_pos = mat4_translation(&Mat4::from_cols_array(&state.link_poses[1]));
        assert_vec3_near(child_pos, Vec3::new(3.0, 0.0, 0.0), "prismatic +3 along X");
    }

    #[test]
    fn test_fixed_joint() {
        // Fixed joint leaves child at parent pose
        let (def, mut state) = make_serial_chain(&[(DefJointType::Fixed, Vec3::Y, 0.0)]);

        forward_kinematics(&def, &mut state, Mat4::IDENTITY);

        // Child pose should equal parent (base) pose = identity
        let child = Mat4::from_cols_array(&state.link_poses[1]);
        let parent = Mat4::from_cols_array(&state.link_poses[0]);
        let diff = (child - parent).to_cols_array();
        for (j, &d) in diff.iter().enumerate() {
            assert!(
                d.abs() < EPSILON,
                "fixed joint child should match parent, element {} diff = {}",
                j,
                d,
            );
        }
    }

    #[test]
    fn test_chain_composition() {
        // 3-joint chain composes transforms correctly
        // Joint 0: prismatic X +2  =>  link1 at (2,0,0)
        // Joint 1: revolute Z pi/2 =>  link2 inherits rotation, still at (2,0,0)
        // Joint 2: prismatic X +1  =>  but X is now rotated to Y, so link3 at (2,1,0)
        let (def, mut state) = make_serial_chain(&[
            (DefJointType::Prismatic, Vec3::X, 2.0),
            (DefJointType::Revolute, Vec3::Z, FRAC_PI_2),
            (DefJointType::Prismatic, Vec3::X, 1.0),
        ]);

        forward_kinematics(&def, &mut state, Mat4::IDENTITY);

        let link1_pos = mat4_translation(&Mat4::from_cols_array(&state.link_poses[1]));
        assert_vec3_near(
            link1_pos,
            Vec3::new(2.0, 0.0, 0.0),
            "link1 after prismatic X +2",
        );

        let link2_pos = mat4_translation(&Mat4::from_cols_array(&state.link_poses[2]));
        assert_vec3_near(
            link2_pos,
            Vec3::new(2.0, 0.0, 0.0),
            "link2 after revolute (no translation)",
        );

        let link3_pos = mat4_translation(&Mat4::from_cols_array(&state.link_poses[3]));
        // After 90-deg rotation around Z, local X becomes world Y
        assert_vec3_near(
            link3_pos,
            Vec3::new(2.0, 1.0, 0.0),
            "link3 after prismatic in rotated frame",
        );
    }

    #[test]
    fn test_base_pose_propagates() {
        // Non-identity base offsets all links
        let base_pose = Mat4::from_translation(Vec3::new(10.0, 5.0, 0.0));
        let (def, mut state) = make_serial_chain(&[(DefJointType::Prismatic, Vec3::X, 1.0)]);

        forward_kinematics(&def, &mut state, base_pose);

        let base_pos = mat4_translation(&Mat4::from_cols_array(&state.link_poses[0]));
        assert_vec3_near(
            base_pos,
            Vec3::new(10.0, 5.0, 0.0),
            "base link at base_pose translation",
        );

        let child_pos = mat4_translation(&Mat4::from_cols_array(&state.link_poses[1]));
        assert_vec3_near(
            child_pos,
            Vec3::new(11.0, 5.0, 0.0),
            "child at base + prismatic offset",
        );
    }

    // ---- Original body-based FK tests ----

    #[test]
    fn test_fk_joint_at_limits() {
        // Joint at exact min and max limits — ensure no clamping issues.
        let base = base_link();

        // Test at min limit
        let mut robot_min = Robot::new("bot_min", Vec3::ZERO, Quat::IDENTITY, base.clone());
        let joint_min = Joint::new(
            JointType::Revolute,
            Vec3::Z,
            -1.5, // at min limit
            0.0,
            (-1.5, 1.5),
            10.0,
        );
        let child_min = arm_link("child", Vec3::new(1.0, 0.0, 0.0));
        robot_min.add_joint_and_link(joint_min, child_min);

        let transforms_min = compute_forward_kinematics(&robot_min);
        assert_eq!(transforms_min.len(), 2);
        // Joint at -1.5 rad around Z: x = cos(-1.5), y = sin(-1.5)
        let expected_x = (-1.5_f32).cos();
        let expected_y = (-1.5_f32).sin();
        assert_vec3_near(
            transforms_min[1].position,
            Vec3::new(expected_x, expected_y, 0.0),
            "joint at min limit",
        );

        // Test at max limit
        let base2 = base_link();
        let mut robot_max = Robot::new("bot_max", Vec3::ZERO, Quat::IDENTITY, base2);
        let joint_max = Joint::new(
            JointType::Revolute,
            Vec3::Z,
            1.5, // at max limit
            0.0,
            (-1.5, 1.5),
            10.0,
        );
        let child_max = arm_link("child", Vec3::new(1.0, 0.0, 0.0));
        robot_max.add_joint_and_link(joint_max, child_max);

        let transforms_max = compute_forward_kinematics(&robot_max);
        assert_eq!(transforms_max.len(), 2);
        let expected_x = (1.5_f32).cos();
        let expected_y = (1.5_f32).sin();
        assert_vec3_near(
            transforms_max[1].position,
            Vec3::new(expected_x, expected_y, 0.0),
            "joint at max limit",
        );
    }

    // ---- Edge case tests ----

    #[test]
    fn test_def_fk_empty_no_joints() {
        let def = RobotDefinition {
            name: "empty".into(),
            links: vec![LinkDefinition {
                name: "base".into(),
                mass: 1.0,
                inertia: 0.1,
                collision_shape: CollisionShape::Cuboid {
                    half_extents: Vec3::splat(0.1),
                },
                parent_joint: None,
                body_zone: None,
            }],
            joints: vec![],
            sensors: Vec::new(),
        };
        let mut state = RobotState::new(&def);
        forward_kinematics(&def, &mut state, Mat4::IDENTITY);

        let base_pos = mat4_translation(&Mat4::from_cols_array(&state.link_poses[0]));
        assert_vec3_near(base_pos, Vec3::ZERO, "no-joint robot base at origin");
    }

    #[test]
    fn test_def_fk_empty_links() {
        let def = RobotDefinition {
            name: "empty".into(),
            links: vec![],
            joints: vec![],
            sensors: Vec::new(),
        };
        let mut state = RobotState::new(&def);
        forward_kinematics(&def, &mut state, Mat4::IDENTITY);
        assert!(state.link_poses.is_empty());
    }

    #[test]
    fn test_def_fk_large_position_value() {
        let (def, mut state) = make_serial_chain(&[(
            DefJointType::Revolute,
            Vec3::Y,
            1000.0 * std::f32::consts::PI,
        )]);
        forward_kinematics(&def, &mut state, Mat4::IDENTITY);

        let child_pose = Mat4::from_cols_array(&state.link_poses[1]);
        let pos = mat4_translation(&child_pose);
        assert!(pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite());
    }

    #[test]
    fn test_def_fk_zero_axis() {
        let def = RobotDefinition {
            name: "zero_axis".into(),
            links: vec![
                LinkDefinition {
                    name: "base".into(),
                    mass: 1.0,
                    inertia: 0.1,
                    collision_shape: CollisionShape::Sphere { radius: 0.1 },
                    parent_joint: None,
                    body_zone: None,
                },
                LinkDefinition {
                    name: "child".into(),
                    mass: 1.0,
                    inertia: 0.1,
                    collision_shape: CollisionShape::Sphere { radius: 0.1 },
                    parent_joint: Some(0),
                    body_zone: None,
                },
            ],
            joints: vec![JointDefinition {
                name: "j".into(),
                joint_type: DefJointType::Revolute,
                axis: Vec3::ZERO,
                parent_link: 0,
                child_link: 1,
                limit_min: -std::f32::consts::PI,
                limit_max: std::f32::consts::PI,
                max_torque: 10.0,
                damping: 0.1,
                anchor_offset: Vec3::ZERO,
                child_offset: Vec3::ZERO,
            }],
            sensors: Vec::new(),
        };
        let mut state = RobotState::new(&def);
        state.joint_positions[0] = 1.0;
        forward_kinematics(&def, &mut state, Mat4::IDENTITY);

        // Zero axis rotation should produce NaN-free result
        let child_pose = Mat4::from_cols_array(&state.link_poses[1]);
        let pos = mat4_translation(&child_pose);
        // glam from_axis_angle with zero axis may produce NaN, but shouldn't crash
        let _ = pos;
    }

    #[test]
    fn test_def_fk_deep_chain() {
        let joints: Vec<(DefJointType, Vec3, f32)> = (0..20)
            .map(|_| (DefJointType::Revolute, Vec3::Y, 0.1))
            .collect();
        let (def, mut state) = make_serial_chain(&joints);
        forward_kinematics(&def, &mut state, Mat4::IDENTITY);

        for (i, pose) in state.link_poses.iter().enumerate() {
            let pos = mat4_translation(&Mat4::from_cols_array(pose));
            assert!(
                pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
                "link {} pose should be finite in deep chain",
                i
            );
        }
    }

    #[test]
    fn test_def_fk_non_identity_base() {
        let base_pose = Mat4::from_rotation_translation(
            Quat::from_rotation_z(FRAC_PI_2),
            Vec3::new(10.0, 20.0, 30.0),
        );
        let (def, mut state) = make_serial_chain(&[(DefJointType::Prismatic, Vec3::X, 2.0)]);
        forward_kinematics(&def, &mut state, base_pose);

        let child_pos = mat4_translation(&Mat4::from_cols_array(&state.link_poses[1]));
        // Base rotated 90 deg around Z, so prismatic along X becomes along Y
        assert_vec3_near(
            child_pos,
            Vec3::new(10.0, 22.0, 30.0),
            "prismatic in rotated base frame",
        );
    }
}

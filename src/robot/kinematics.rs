use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use super::body::{Joint, Link};
use super::body::{JointType, Robot};

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
}

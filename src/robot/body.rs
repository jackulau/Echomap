use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

/// Minimum mass for a link (avoids zero/negative mass).
const MIN_MASS: f32 = 0.001;

/// Type of joint connecting two links in the kinematic chain.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum JointType {
    /// Rotation around an axis.
    Revolute,
    /// Translation along an axis.
    Prismatic,
    /// Rigid connection (no motion).
    Fixed,
}

/// A joint connecting a parent link to a child link.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Joint {
    pub joint_type: JointType,
    pub axis: Vec3,
    pub position: f32,
    pub velocity: f32,
    pub limits: (f32, f32),
    pub max_torque: f32,
}

/// A rigid link in the robot kinematic chain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Link {
    pub name: String,
    pub local_position: Vec3,
    pub local_rotation: Quat,
    pub half_extents: Vec3,
    pub mass: f32,
}

/// Articulated rigid-body robot with a kinematic chain of links and joints.
///
/// The first link is the base (no associated joint). Subsequent links are
/// connected by joints: `joints.len() == links.len() - 1`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Robot {
    pub name: String,
    pub base_position: Vec3,
    pub base_rotation: Quat,
    pub links: Vec<Link>,
    pub joints: Vec<Joint>,
}

// ---------------------------------------------------------------------------
// Implementations
// ---------------------------------------------------------------------------

impl Joint {
    /// Create a new joint. The axis is normalized automatically.
    /// If limits are inverted (min > max), they are swapped.
    pub fn new(
        joint_type: JointType,
        axis: Vec3,
        position: f32,
        velocity: f32,
        limits: (f32, f32),
        max_torque: f32,
    ) -> Self {
        // Normalize axis (fall back to Y if zero-length)
        let normalized_axis = if axis.length() > f32::EPSILON {
            axis.normalize()
        } else {
            Vec3::Y
        };

        // Swap inverted limits
        let limits = if limits.0 > limits.1 {
            (limits.1, limits.0)
        } else {
            limits
        };

        // Clamp initial position to limits
        let clamped_position = position.clamp(limits.0, limits.1);

        Self {
            joint_type,
            axis: normalized_axis,
            position: clamped_position,
            velocity,
            limits,
            max_torque,
        }
    }

    /// Set joint position, clamping to limits.
    pub fn set_position(&mut self, pos: f32) {
        self.position = pos.clamp(self.limits.0, self.limits.1);
    }
}

impl Link {
    /// Create a new link. Mass is clamped to a minimum of `MIN_MASS`.
    pub fn new(
        name: impl Into<String>,
        local_position: Vec3,
        local_rotation: Quat,
        half_extents: Vec3,
        mass: f32,
    ) -> Self {
        Self {
            name: name.into(),
            local_position,
            local_rotation,
            half_extents,
            mass: mass.max(MIN_MASS),
        }
    }
}

impl Robot {
    /// Create a new robot with a single base link.
    pub fn new(name: impl Into<String>, base_pos: Vec3, base_rot: Quat, base_link: Link) -> Self {
        Self {
            name: name.into(),
            base_position: base_pos,
            base_rotation: base_rot,
            links: vec![base_link],
            joints: Vec::new(),
        }
    }

    /// Add a child link connected by a joint.
    pub fn add_joint_and_link(&mut self, joint: Joint, link: Link) {
        self.joints.push(joint);
        self.links.push(link);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_joint_type_variants() {
        // All three joint types must be constructible and distinct.
        let revolute = JointType::Revolute;
        let prismatic = JointType::Prismatic;
        let fixed = JointType::Fixed;

        assert_eq!(revolute, JointType::Revolute);
        assert_eq!(prismatic, JointType::Prismatic);
        assert_eq!(fixed, JointType::Fixed);

        assert_ne!(revolute, prismatic);
        assert_ne!(revolute, fixed);
        assert_ne!(prismatic, fixed);

        // Clone works
        let cloned = revolute.clone();
        assert_eq!(cloned, JointType::Revolute);
    }

    #[test]
    fn test_joint_limits_clamping() {
        let mut joint = Joint::new(JointType::Revolute, Vec3::Y, 0.0, 0.0, (-1.0, 1.0), 10.0);

        // Within limits
        joint.set_position(0.5);
        assert!(
            (joint.position - 0.5).abs() < EPSILON,
            "position should be 0.5"
        );

        // Above upper limit -> clamped
        joint.set_position(5.0);
        assert!(
            (joint.position - 1.0).abs() < EPSILON,
            "position should be clamped to 1.0"
        );

        // Below lower limit -> clamped
        joint.set_position(-5.0);
        assert!(
            (joint.position - (-1.0)).abs() < EPSILON,
            "position should be clamped to -1.0"
        );
    }

    #[test]
    fn test_link_creation() {
        let link = Link::new(
            "arm_link",
            Vec3::new(0.0, 0.5, 0.0),
            Quat::IDENTITY,
            Vec3::new(0.1, 0.25, 0.1),
            2.0,
        );

        assert_eq!(link.name, "arm_link");
        assert!((link.local_position - Vec3::new(0.0, 0.5, 0.0)).length() < EPSILON);
        assert!((link.half_extents - Vec3::new(0.1, 0.25, 0.1)).length() < EPSILON);
        assert!((link.mass - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_robot_add_links() {
        let base = Link::new("base", Vec3::ZERO, Quat::IDENTITY, Vec3::splat(0.1), 5.0);
        let mut robot = Robot::new("test_arm", Vec3::ZERO, Quat::IDENTITY, base);

        let joint1 = Joint::new(
            JointType::Revolute,
            Vec3::Y,
            0.0,
            0.0,
            (-std::f32::consts::PI, std::f32::consts::PI),
            10.0,
        );
        let link1 = Link::new(
            "link1",
            Vec3::new(0.0, 0.5, 0.0),
            Quat::IDENTITY,
            Vec3::splat(0.05),
            1.0,
        );
        robot.add_joint_and_link(joint1, link1);

        let joint2 = Joint::new(
            JointType::Revolute,
            Vec3::Y,
            0.0,
            0.0,
            (-std::f32::consts::PI, std::f32::consts::PI),
            10.0,
        );
        let link2 = Link::new(
            "link2",
            Vec3::new(0.0, 0.5, 0.0),
            Quat::IDENTITY,
            Vec3::splat(0.05),
            1.0,
        );
        robot.add_joint_and_link(joint2, link2);

        assert_eq!(robot.links.len(), 3, "robot should have 3 links (base + 2)");
        assert_eq!(robot.joints.len(), 2, "robot should have 2 joints");
        assert_eq!(
            robot.joints.len(),
            robot.links.len() - 1,
            "joints = links - 1 invariant"
        );
    }

    #[test]
    fn test_robot_empty() {
        // "new robot has no links" -- actually has one base link per spec clarification
        let base = Link::new("base", Vec3::ZERO, Quat::IDENTITY, Vec3::splat(0.1), 5.0);
        let robot = Robot::new("test_bot", Vec3::ZERO, Quat::IDENTITY, base);

        assert_eq!(
            robot.links.len(),
            1,
            "new robot should have exactly one base link"
        );
        assert_eq!(robot.joints.len(), 0, "new robot should have zero joints");
        assert_eq!(robot.name, "test_bot");
    }

    #[test]
    fn test_joint_serialization() {
        let joint = Joint::new(JointType::Revolute, Vec3::Y, 0.5, 0.1, (-1.5, 1.5), 10.0);

        let json = serde_json::to_string(&joint).expect("serialization failed");
        let deserialized: Joint = serde_json::from_str(&json).expect("deserialization failed");

        assert_eq!(deserialized.joint_type, joint.joint_type);
        assert!((deserialized.position - joint.position).abs() < EPSILON);
        assert!((deserialized.velocity - joint.velocity).abs() < EPSILON);
        assert!((deserialized.limits.0 - joint.limits.0).abs() < EPSILON);
        assert!((deserialized.limits.1 - joint.limits.1).abs() < EPSILON);
        assert!((deserialized.max_torque - joint.max_torque).abs() < EPSILON);
        assert!((deserialized.axis - joint.axis).length() < EPSILON);
    }

    #[test]
    fn test_joint_axis_normalized() {
        // Non-unit axis should be normalized on construction
        let joint = Joint::new(
            JointType::Revolute,
            Vec3::new(0.0, 3.0, 0.0), // length 3, not unit
            0.0,
            0.0,
            (-1.0, 1.0),
            10.0,
        );

        let axis_length = joint.axis.length();
        assert!(
            (axis_length - 1.0).abs() < EPSILON,
            "joint axis should be normalized, got length {}",
            axis_length
        );
        // Direction preserved
        assert!(
            (joint.axis - Vec3::Y).length() < EPSILON,
            "axis direction should be Y"
        );
    }

    #[test]
    fn test_joint_inverted_limits() {
        // limits.0 > limits.1 should be swapped automatically
        let joint = Joint::new(
            JointType::Prismatic,
            Vec3::X,
            0.0,
            0.0,
            (2.0, -2.0), // inverted
            5.0,
        );

        assert!(
            joint.limits.0 <= joint.limits.1,
            "limits should be swapped: min={} max={}",
            joint.limits.0,
            joint.limits.1
        );
        assert!((joint.limits.0 - (-2.0)).abs() < EPSILON);
        assert!((joint.limits.1 - 2.0).abs() < EPSILON);
    }

    #[test]
    fn test_link_zero_mass_clamped() {
        // Zero mass should be clamped to MIN_MASS
        let link_zero = Link::new("zero", Vec3::ZERO, Quat::IDENTITY, Vec3::splat(0.1), 0.0);
        assert!(
            link_zero.mass >= MIN_MASS,
            "zero mass should be clamped to MIN_MASS, got {}",
            link_zero.mass
        );

        // Negative mass should also be clamped
        let link_neg = Link::new("neg", Vec3::ZERO, Quat::IDENTITY, Vec3::splat(0.1), -5.0);
        assert!(
            link_neg.mass >= MIN_MASS,
            "negative mass should be clamped to MIN_MASS, got {}",
            link_neg.mass
        );
    }
}

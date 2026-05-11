use glam::Vec3;
use serde::{Deserialize, Serialize};

// ---- Enums ----

/// Type of joint connecting two links.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum JointType {
    Revolute,
    Prismatic,
    Fixed,
}

/// Collision shape attached to a link.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum CollisionShape {
    Sphere { radius: f32 },
    Cuboid { half_extents: Vec3 },
    Cylinder { radius: f32, height: f32 },
}

/// Sensor type definition.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SensorDefinition {
    Distance {
        direction: Vec3,
        max_range: f32,
    },
    Lidar {
        num_rays: usize,
        fov_rad: f32,
        max_range: f32,
    },
    Contact,
    Imu,
}

// ---- Structs ----

/// A rigid link in the robot's kinematic chain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinkDefinition {
    pub name: String,
    pub mass: f32,
    pub inertia: f32,
    pub collision_shape: CollisionShape,
    pub parent_joint: Option<usize>,
}

/// A joint connecting a parent link to a child link.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JointDefinition {
    pub name: String,
    pub joint_type: JointType,
    pub axis: Vec3,
    pub parent_link: usize,
    pub child_link: usize,
    pub limit_min: f32,
    pub limit_max: f32,
    pub max_torque: f32,
    pub damping: f32,
}

/// A sensor mounted on a specific link.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SensorMount {
    pub link_index: usize,
    pub local_offset: Vec3,
    pub sensor: SensorDefinition,
}

/// Full robot definition: links, joints, and sensors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RobotDefinition {
    pub name: String,
    pub links: Vec<LinkDefinition>,
    pub joints: Vec<JointDefinition>,
    pub sensors: Vec<SensorMount>,
}

impl RobotDefinition {
    /// Factory method for a basic serial manipulator arm.
    ///
    /// Creates `num_joints + 1` links connected by `num_joints` revolute joints
    /// along the Y axis.
    pub fn simple_arm(num_joints: usize) -> Self {
        let mut links = Vec::with_capacity(num_joints + 1);
        let mut joints = Vec::with_capacity(num_joints);

        // Base link (index 0)
        links.push(LinkDefinition {
            name: "base".to_string(),
            mass: 5.0,
            inertia: 1.0,
            collision_shape: CollisionShape::Cuboid {
                half_extents: Vec3::new(0.1, 0.1, 0.1),
            },
            parent_joint: None,
        });

        for i in 0..num_joints {
            let parent_link = i;
            let child_link = i + 1;

            joints.push(JointDefinition {
                name: format!("joint_{}", i),
                joint_type: JointType::Revolute,
                axis: Vec3::Y,
                parent_link,
                child_link,
                limit_min: -std::f32::consts::PI,
                limit_max: std::f32::consts::PI,
                max_torque: 10.0,
                damping: 0.1,
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
            });
        }

        Self {
            name: "simple_arm".to_string(),
            links,
            joints,
            sensors: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_joint_type_variants() {
        let revolute = JointType::Revolute;
        let prismatic = JointType::Prismatic;
        let fixed = JointType::Fixed;

        assert_eq!(revolute, JointType::Revolute);
        assert_eq!(prismatic, JointType::Prismatic);
        assert_eq!(fixed, JointType::Fixed);

        // Verify Clone works
        let cloned = revolute.clone();
        assert_eq!(cloned, JointType::Revolute);
    }

    #[test]
    fn test_collision_shape_variants() {
        let sphere = CollisionShape::Sphere { radius: 1.0 };
        let cuboid = CollisionShape::Cuboid {
            half_extents: Vec3::new(1.0, 2.0, 3.0),
        };
        let cylinder = CollisionShape::Cylinder {
            radius: 0.5,
            height: 2.0,
        };

        // Verify each variant is distinct
        assert_ne!(sphere, cuboid);
        assert_ne!(cuboid, cylinder);
        assert_ne!(sphere, cylinder);

        // Verify data stored correctly
        match &sphere {
            CollisionShape::Sphere { radius } => assert!((radius - 1.0).abs() < 1e-6),
            _ => panic!("Expected Sphere"),
        }
        match &cuboid {
            CollisionShape::Cuboid { half_extents } => {
                assert!((half_extents.x - 1.0).abs() < 1e-6);
                assert!((half_extents.y - 2.0).abs() < 1e-6);
                assert!((half_extents.z - 3.0).abs() < 1e-6);
            }
            _ => panic!("Expected Cuboid"),
        }
        match &cylinder {
            CollisionShape::Cylinder { radius, height } => {
                assert!((radius - 0.5).abs() < 1e-6);
                assert!((height - 2.0).abs() < 1e-6);
            }
            _ => panic!("Expected Cylinder"),
        }
    }

    #[test]
    fn test_robot_definition_simple_arm() {
        let arm = RobotDefinition::simple_arm(3);

        assert_eq!(arm.links.len(), 4, "3 joints should produce 4 links");
        assert_eq!(arm.joints.len(), 3, "3 joints expected");
        assert_eq!(arm.name, "simple_arm");

        // Base link has no parent joint
        assert!(arm.links[0].parent_joint.is_none());

        // Non-base links have parent joints
        for i in 1..arm.links.len() {
            assert!(arm.links[i].parent_joint.is_some());
        }

        // All joints are revolute with Y axis
        for joint in &arm.joints {
            assert_eq!(joint.joint_type, JointType::Revolute);
            assert!((joint.axis - Vec3::Y).length() < 1e-6);
        }

        // Joints connect sequential links
        for (i, joint) in arm.joints.iter().enumerate() {
            assert_eq!(joint.parent_link, i);
            assert_eq!(joint.child_link, i + 1);
        }
    }

    #[test]
    fn test_robot_definition_serialization() {
        let arm = RobotDefinition::simple_arm(2);

        let json = serde_json::to_string(&arm).expect("serialization failed");
        let deserialized: RobotDefinition =
            serde_json::from_str(&json).expect("deserialization failed");

        assert_eq!(deserialized.name, arm.name);
        assert_eq!(deserialized.links.len(), arm.links.len());
        assert_eq!(deserialized.joints.len(), arm.joints.len());
        assert_eq!(deserialized.sensors.len(), arm.sensors.len());

        // Verify link data survived round-trip
        for (orig, deser) in arm.links.iter().zip(deserialized.links.iter()) {
            assert_eq!(orig.name, deser.name);
            assert!((orig.mass - deser.mass).abs() < 1e-6);
            assert!((orig.inertia - deser.inertia).abs() < 1e-6);
        }

        // Verify joint data survived round-trip
        for (orig, deser) in arm.joints.iter().zip(deserialized.joints.iter()) {
            assert_eq!(orig.name, deser.name);
            assert!((orig.limit_min - deser.limit_min).abs() < 1e-6);
            assert!((orig.limit_max - deser.limit_max).abs() < 1e-6);
        }
    }

    #[test]
    fn test_joint_limits() {
        let arm = RobotDefinition::simple_arm(1);
        let joint = &arm.joints[0];

        assert!(
            joint.limit_min < joint.limit_max,
            "limit_min ({}) should be less than limit_max ({})",
            joint.limit_min,
            joint.limit_max
        );
    }

    #[test]
    fn test_link_mass_positive() {
        let arm = RobotDefinition::simple_arm(2);

        for link in &arm.links {
            assert!(
                link.mass > 0.0,
                "Link '{}' has non-positive mass: {}",
                link.name,
                link.mass
            );
        }
    }
}

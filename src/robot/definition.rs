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

/// Body zone for damage calculation in combat.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum BodyZone {
    Head,
    Body,
    LeftArm,
    RightArm,
}

impl BodyZone {
    /// Returns the damage multiplier for this body zone.
    pub fn damage_multiplier(&self) -> f32 {
        match self {
            BodyZone::Head => 3.0,
            BodyZone::Body => 1.0,
            BodyZone::LeftArm => 0.5,
            BodyZone::RightArm => 0.5,
        }
    }
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
    #[serde(default)]
    pub body_zone: Option<BodyZone>,
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
    #[serde(default)]
    pub anchor_offset: Vec3,
    #[serde(default)]
    pub child_offset: Vec3,
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
            body_zone: Some(BodyZone::Body),
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

        Self {
            name: "simple_arm".to_string(),
            links,
            joints,
            sensors: Vec::new(),
        }
    }

    /// Factory method for a 3-link boxing test robot (torso + 2 arms).
    ///
    /// Creates a robot suitable for combat integration tests with body zones
    /// assigned to all links: Body (torso), LeftArm, and RightArm.
    pub fn boxing_test_robot() -> Self {
        let links = vec![
            LinkDefinition {
                name: "torso".to_string(),
                mass: 10.0,
                inertia: 2.0,
                collision_shape: CollisionShape::Cuboid {
                    half_extents: Vec3::new(0.2, 0.3, 0.15),
                },
                parent_joint: None,
                body_zone: Some(BodyZone::Body),
            },
            LinkDefinition {
                name: "left_arm".to_string(),
                mass: 2.0,
                inertia: 0.3,
                collision_shape: CollisionShape::Cylinder {
                    radius: 0.05,
                    height: 0.4,
                },
                parent_joint: Some(0),
                body_zone: Some(BodyZone::LeftArm),
            },
            LinkDefinition {
                name: "right_arm".to_string(),
                mass: 2.0,
                inertia: 0.3,
                collision_shape: CollisionShape::Cylinder {
                    radius: 0.05,
                    height: 0.4,
                },
                parent_joint: Some(1),
                body_zone: Some(BodyZone::RightArm),
            },
        ];
        let joints = vec![
            JointDefinition {
                name: "left_shoulder".to_string(),
                joint_type: JointType::Revolute,
                axis: Vec3::Y,
                parent_link: 0,
                child_link: 1,
                limit_min: -std::f32::consts::PI,
                limit_max: std::f32::consts::PI,
                max_torque: 20.0,
                damping: 0.1,
                anchor_offset: Vec3::ZERO,
                child_offset: Vec3::ZERO,
            },
            JointDefinition {
                name: "right_shoulder".to_string(),
                joint_type: JointType::Revolute,
                axis: Vec3::Y,
                parent_link: 0,
                child_link: 2,
                limit_min: -std::f32::consts::PI,
                limit_max: std::f32::consts::PI,
                max_torque: 20.0,
                damping: 0.1,
                anchor_offset: Vec3::ZERO,
                child_offset: Vec3::ZERO,
            },
        ];
        Self {
            name: "boxing_test_robot".to_string(),
            links,
            joints,
            sensors: Vec::new(),
        }
    }

    /// Factory method for a 4-link boxing humanoid (torso, head, two arms).
    ///
    /// Creates a humanoid robot with body zones suitable for boxing scenarios:
    /// - Torso (Body) as root link
    /// - Head connected via neck joint
    /// - Left arm connected via left shoulder joint
    /// - Right arm connected via right shoulder joint
    pub fn boxing_humanoid() -> Self {
        // 4 links: torso (Body), head (Head), left_arm (LeftArm), right_arm (RightArm)
        // 3 joints: neck (torso->head), left_shoulder (torso->left_arm), right_shoulder (torso->right_arm)

        let links = vec![
            LinkDefinition {
                name: "torso".to_string(),
                mass: 10.0,
                inertia: 2.0,
                collision_shape: CollisionShape::Cuboid {
                    half_extents: Vec3::new(0.2, 0.3, 0.15),
                },
                parent_joint: None,
                body_zone: Some(BodyZone::Body),
            },
            LinkDefinition {
                name: "head".to_string(),
                mass: 3.0,
                inertia: 0.5,
                collision_shape: CollisionShape::Sphere { radius: 0.1 },
                parent_joint: Some(0), // connected via neck joint (joint index 0)
                body_zone: Some(BodyZone::Head),
            },
            LinkDefinition {
                name: "left_arm".to_string(),
                mass: 2.0,
                inertia: 0.3,
                collision_shape: CollisionShape::Sphere { radius: 0.15 },
                parent_joint: Some(1),
                body_zone: Some(BodyZone::LeftArm),
            },
            LinkDefinition {
                name: "right_arm".to_string(),
                mass: 2.0,
                inertia: 0.3,
                collision_shape: CollisionShape::Sphere { radius: 0.15 },
                parent_joint: Some(2),
                body_zone: Some(BodyZone::RightArm),
            },
        ];
        let joints = vec![
            JointDefinition {
                name: "neck".to_string(),
                joint_type: JointType::Revolute,
                axis: Vec3::Y,
                parent_link: 0,
                child_link: 1,
                limit_min: -std::f32::consts::FRAC_PI_4,
                limit_max: std::f32::consts::FRAC_PI_4,
                max_torque: 5.0,
                damping: 0.2,
                anchor_offset: Vec3::new(0.0, 0.35, 0.0),
                child_offset: Vec3::new(0.0, 0.12, 0.0),
            },
            JointDefinition {
                name: "left_shoulder".to_string(),
                joint_type: JointType::Revolute,
                axis: Vec3::Y,
                parent_link: 0,
                child_link: 2,
                limit_min: -std::f32::consts::PI,
                limit_max: std::f32::consts::PI,
                max_torque: 20.0,
                damping: 0.1,
                anchor_offset: Vec3::new(0.0, 0.15, 0.12),
                child_offset: Vec3::new(0.0, 0.0, 0.35),
            },
            JointDefinition {
                name: "right_shoulder".to_string(),
                joint_type: JointType::Revolute,
                axis: Vec3::Y,
                parent_link: 0,
                child_link: 3,
                limit_min: -std::f32::consts::PI,
                limit_max: std::f32::consts::PI,
                max_torque: 20.0,
                damping: 0.1,
                anchor_offset: Vec3::new(0.0, 0.15, -0.12),
                child_offset: Vec3::new(0.0, 0.0, -0.35),
            },
        ];
        Self {
            name: "boxing_humanoid".to_string(),
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

    #[test]
    fn test_body_zone_multipliers() {
        assert!((BodyZone::Head.damage_multiplier() - 3.0).abs() < 1e-6);
        assert!((BodyZone::Body.damage_multiplier() - 1.0).abs() < 1e-6);
        assert!((BodyZone::LeftArm.damage_multiplier() - 0.5).abs() < 1e-6);
        assert!((BodyZone::RightArm.damage_multiplier() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_link_definition_default_zone() {
        let json = r#"{
            "name": "test",
            "mass": 1.0,
            "inertia": 0.1,
            "collision_shape": {"Sphere": {"radius": 0.5}},
            "parent_joint": null
        }"#;
        let link: LinkDefinition = serde_json::from_str(json).expect("deserialization failed");
        assert!(link.body_zone.is_none(), "body_zone should default to None");
    }

    #[test]
    fn test_simple_arm_has_zones() {
        let arm = RobotDefinition::simple_arm(3);
        // Base link should have Body zone
        assert_eq!(arm.links[0].body_zone, Some(BodyZone::Body));
        // Non-base arm links should have None (generic arm segments)
        for link in &arm.links[1..] {
            assert!(
                link.body_zone.is_none(),
                "Arm segment '{}' should have no zone",
                link.name
            );
        }
    }

    #[test]
    fn test_boxing_humanoid_link_count() {
        let robot = RobotDefinition::boxing_humanoid();
        assert_eq!(robot.links.len(), 4, "boxing_humanoid should have 4 links");
        assert_eq!(
            robot.joints.len(),
            3,
            "boxing_humanoid should have 3 joints"
        );
    }

    #[test]
    fn test_boxing_humanoid_zones() {
        let robot = RobotDefinition::boxing_humanoid();
        assert_eq!(
            robot.links[0].body_zone,
            Some(BodyZone::Body),
            "link 0 (torso) should be Body"
        );
        assert_eq!(
            robot.links[1].body_zone,
            Some(BodyZone::Head),
            "link 1 (head) should be Head"
        );
        assert_eq!(
            robot.links[2].body_zone,
            Some(BodyZone::LeftArm),
            "link 2 (left_arm) should be LeftArm"
        );
        assert_eq!(
            robot.links[3].body_zone,
            Some(BodyZone::RightArm),
            "link 3 (right_arm) should be RightArm"
        );
    }

    #[test]
    fn test_boxing_humanoid_head_has_sphere() {
        let robot = RobotDefinition::boxing_humanoid();
        let head = &robot.links[1];
        assert_eq!(head.name, "head");
        match &head.collision_shape {
            CollisionShape::Sphere { radius } => {
                assert!(
                    (radius - 0.1).abs() < 1e-6,
                    "head sphere radius should be 0.1"
                );
            }
            other => panic!(
                "Expected head to use Sphere collision shape, got {:?}",
                other
            ),
        }
    }
}

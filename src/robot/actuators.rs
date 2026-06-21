use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use super::body::{Joint, Robot};
use super::collision::{aabb_from_link, aabb_overlap, Aabb};
use super::kinematics::LinkTransform;
use crate::scene::Scene;

// ---------------------------------------------------------------------------
// Motor Actuator
// ---------------------------------------------------------------------------

/// A velocity-controlled motor attached to a robot joint.
///
/// Accelerates the joint toward `target_velocity`, clamped by `max_torque`.
/// Each `apply` call integrates velocity and position for one timestep `dt`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MotorActuator {
    pub joint_index: usize,
    pub target_velocity: f32,
    pub max_torque: f32,
}

impl MotorActuator {
    /// Apply the motor to a joint for one timestep.
    ///
    /// Acceleration is `(target_velocity - joint.velocity)` clamped to
    /// `[-max_torque, max_torque]`. Velocity is updated, then position is
    /// integrated and clamped to joint limits via `Joint::set_position`.
    pub fn apply(&self, joint: &mut Joint, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        let error = self.target_velocity - joint.velocity;
        let accel = error.clamp(-self.max_torque, self.max_torque);
        joint.velocity += accel * dt;
        let new_pos = joint.position + joint.velocity * dt;
        joint.set_position(new_pos);
    }
}

// ---------------------------------------------------------------------------
// Gripper Actuator
// ---------------------------------------------------------------------------

/// An open/close gripper attached to a robot link.
///
/// When closed, the gripper checks for overlapping scene objects via AABB and
/// attaches the nearest one. When opened, the attached object is released.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GripperActuator {
    pub link_index: usize,
    pub is_open: bool,
    pub attached_object: Option<usize>,
    pub grip_strength: f32,
}

impl GripperActuator {
    /// Close the gripper. If currently open, check for overlapping scene
    /// objects and attach the nearest one within range.
    pub fn close(&mut self, robot_transforms: &[LinkTransform], robot: &Robot, scene: &Scene) {
        if !self.is_open {
            return;
        }

        self.is_open = false;

        let link = &robot.links[self.link_index];
        let tf = &robot_transforms[self.link_index];
        let gripper_aabb = aabb_from_link(tf.position, tf.rotation, link.half_extents);

        let mut best_index: Option<usize> = None;
        let mut best_dist = f32::MAX;

        for (i, obj) in scene.meshes.iter().enumerate() {
            // Compute a bounding AABB for the scene object from its mesh
            let obj_aabb = scene_object_aabb(obj);
            if aabb_overlap(&gripper_aabb, &obj_aabb) {
                let dist = (obj_aabb.center - gripper_aabb.center).length();
                if dist < best_dist {
                    best_dist = dist;
                    best_index = Some(i);
                }
            }
        }

        self.attached_object = best_index;
    }

    /// Open the gripper, releasing any attached object.
    pub fn open(&mut self) {
        self.is_open = true;
        self.attached_object = None;
    }

    /// Compute the world-space transform for the attached object so it
    /// follows the gripper link.
    ///
    /// Returns `(object_index, new_position, new_rotation)` for the caller
    /// to apply, or `None` if nothing is attached.
    pub fn compute_attached_transform(
        &self,
        robot_transforms: &[LinkTransform],
        _robot: &Robot,
    ) -> Option<(usize, Vec3, Quat)> {
        let obj_idx = self.attached_object?;
        let tf = &robot_transforms[self.link_index];
        Some((obj_idx, tf.position, tf.rotation))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute an AABB for a scene object by finding the bounding box of all its
/// mesh vertices. Returns a zero-size AABB at the origin for empty meshes.
fn scene_object_aabb(obj: &crate::scene::SceneObject) -> Aabb {
    if obj.mesh.triangles.is_empty() {
        return Aabb {
            center: Vec3::ZERO,
            half_extents: Vec3::ZERO,
        };
    }

    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);

    for tri in &obj.mesh.triangles {
        for v in &tri.vertices {
            min = min.min(v.position);
            max = max.max(v.position);
        }
    }

    let center = (min + max) * 0.5;
    let half_extents = (max - min) * 0.5;

    Aabb {
        center,
        half_extents,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::body::{Joint, JointType, Link, Robot};
    use crate::robot::kinematics::compute_forward_kinematics;
    use crate::scene::material::AcousticMaterial;
    use crate::scene::{Mesh, Scene, SceneObject, Triangle, Vertex};
    use glam::{Quat, Vec3};

    const EPSILON: f32 = 1e-5;

    // ---- helpers ----

    fn make_motor(joint_index: usize, target_vel: f32, max_torque: f32) -> MotorActuator {
        MotorActuator {
            joint_index,
            target_velocity: target_vel,
            max_torque,
        }
    }

    fn make_revolute_joint(pos: f32, limits: (f32, f32)) -> Joint {
        Joint::new(JointType::Revolute, Vec3::Y, pos, 0.0, limits, 100.0)
    }

    fn make_gripper(link_index: usize) -> GripperActuator {
        GripperActuator {
            link_index,
            is_open: true,
            attached_object: None,
            grip_strength: 10.0,
        }
    }

    /// Build a simple robot with a base and one arm link.
    fn simple_robot() -> Robot {
        let base = Link::new("base", Vec3::ZERO, Quat::IDENTITY, Vec3::splat(0.1), 5.0);
        let mut robot = Robot::new("test_bot", Vec3::ZERO, Quat::IDENTITY, base);
        let joint = Joint::new(
            JointType::Revolute,
            Vec3::Y,
            0.0,
            0.0,
            (-std::f32::consts::PI, std::f32::consts::PI),
            10.0,
        );
        let link = Link::new(
            "gripper_link",
            Vec3::new(1.0, 0.0, 0.0),
            Quat::IDENTITY,
            Vec3::splat(0.5), // large enough to overlap with nearby objects
            1.0,
        );
        robot.add_joint_and_link(joint, link);
        robot
    }

    /// Build a scene object (box approximated by triangles) centered at the
    /// given position with given half-size.
    fn box_scene_object(name: &str, center: Vec3, half: f32) -> SceneObject {
        // We just need triangle vertices that span an AABB. Two triangles
        // forming one face of the box are enough for AABB computation.
        let min = center - Vec3::splat(half);
        let max = center + Vec3::splat(half);
        let v0 = Vertex {
            position: min,
            normal: Vec3::Y,
        };
        let v1 = Vertex {
            position: Vec3::new(max.x, min.y, min.z),
            normal: Vec3::Y,
        };
        let v2 = Vertex {
            position: max,
            normal: Vec3::Y,
        };
        let v3 = Vertex {
            position: Vec3::new(min.x, max.y, max.z),
            normal: Vec3::Y,
        };
        SceneObject {
            name: name.into(),
            mesh: Mesh {
                triangles: vec![
                    Triangle {
                        vertices: [v0.clone(), v1.clone(), v2.clone()],
                    },
                    Triangle {
                        vertices: [v0, v2, v3],
                    },
                ],
            },
            material: AcousticMaterial::default(),
            visible: true,
            interior_medium: None,
        }
    }

    // ---- motor tests ----

    #[test]
    fn test_motor_accelerates_joint() {
        let motor = make_motor(0, 5.0, 100.0);
        let mut joint = make_revolute_joint(0.0, (-10.0, 10.0));
        assert!((joint.velocity - 0.0).abs() < EPSILON);

        motor.apply(&mut joint, 0.1);

        // With large torque limit, velocity should move toward 5.0
        assert!(
            joint.velocity > 0.0,
            "velocity should increase toward target, got {}",
            joint.velocity
        );
    }

    #[test]
    fn test_motor_torque_limited() {
        let motor = make_motor(0, 100.0, 2.0); // tiny torque
        let mut joint = make_revolute_joint(0.0, (-100.0, 100.0));

        motor.apply(&mut joint, 1.0);

        // Acceleration clamped to 2.0, so velocity = 0 + 2.0 * 1.0 = 2.0
        assert!(
            (joint.velocity - 2.0).abs() < EPSILON,
            "velocity should be clamped by torque, got {}",
            joint.velocity
        );
    }

    #[test]
    fn test_motor_respects_joint_limits() {
        let motor = make_motor(0, 10.0, 100.0);
        // Joint near its upper limit
        let mut joint = make_revolute_joint(0.9, (-1.0, 1.0));

        // Apply many steps — position should not exceed 1.0
        for _ in 0..100 {
            motor.apply(&mut joint, 0.1);
        }

        assert!(
            joint.position <= 1.0 + EPSILON,
            "position should respect upper limit, got {}",
            joint.position
        );
    }

    #[test]
    fn test_motor_zero_dt() {
        let motor = make_motor(0, 10.0, 100.0);
        let mut joint = make_revolute_joint(0.5, (-1.0, 1.0));
        let original_pos = joint.position;
        let original_vel = joint.velocity;

        motor.apply(&mut joint, 0.0);

        assert!(
            (joint.position - original_pos).abs() < EPSILON,
            "position should not change with dt=0"
        );
        assert!(
            (joint.velocity - original_vel).abs() < EPSILON,
            "velocity should not change with dt=0"
        );
    }

    // ---- gripper tests ----

    #[test]
    fn test_gripper_open_close() {
        let mut gripper = make_gripper(1);
        assert!(gripper.is_open, "gripper should start open");

        // Close with no objects in scene
        let robot = simple_robot();
        let transforms = compute_forward_kinematics(&robot);
        let scene = Scene::default();

        gripper.close(&transforms, &robot, &scene);
        assert!(!gripper.is_open, "gripper should be closed after close()");
        assert!(
            gripper.attached_object.is_none(),
            "no objects to attach in empty scene"
        );

        // Open again
        gripper.open();
        assert!(gripper.is_open, "gripper should be open after open()");
    }

    #[test]
    fn test_gripper_attach_object() {
        let mut gripper = make_gripper(1);
        let robot = simple_robot();
        let transforms = compute_forward_kinematics(&robot);

        // Place a scene object overlapping with link 1's world position
        let link1_pos = transforms[1].position;
        let obj = box_scene_object("target", link1_pos, 0.3);
        let mut scene = Scene::default();
        scene.meshes.push(obj);

        gripper.close(&transforms, &robot, &scene);

        assert!(!gripper.is_open, "gripper should be closed");
        assert_eq!(
            gripper.attached_object,
            Some(0),
            "should attach the overlapping object"
        );
    }

    #[test]
    fn test_gripper_detach_object() {
        let mut gripper = make_gripper(1);
        gripper.is_open = false;
        gripper.attached_object = Some(42);

        gripper.open();

        assert!(gripper.is_open, "gripper should be open");
        assert!(
            gripper.attached_object.is_none(),
            "attached_object should be None after open()"
        );
    }

    #[test]
    fn test_gripper_compute_attached_transform() {
        let mut gripper = make_gripper(1);
        let robot = simple_robot();
        let transforms = compute_forward_kinematics(&robot);

        // Nothing attached yet
        assert!(
            gripper
                .compute_attached_transform(&transforms, &robot)
                .is_none(),
            "should return None when nothing attached"
        );

        // Attach object 3
        gripper.attached_object = Some(3);
        gripper.is_open = false;

        let result = gripper
            .compute_attached_transform(&transforms, &robot)
            .expect("should return Some when object attached");

        assert_eq!(result.0, 3, "object index should be 3");
        assert!(
            (result.1 - transforms[1].position).length() < EPSILON,
            "position should match link transform"
        );
    }

    #[test]
    fn test_apply_action_wrong_size() {
        // Motor apply with a dt that is negative should be handled gracefully
        let motor = make_motor(0, 5.0, 10.0);
        let mut joint = make_revolute_joint(0.0, (-1.0, 1.0));
        let original_pos = joint.position;
        let original_vel = joint.velocity;

        motor.apply(&mut joint, -1.0);

        // Negative dt should be treated like zero (no-op)
        assert!(
            (joint.position - original_pos).abs() < EPSILON,
            "negative dt should not change position"
        );
        assert!(
            (joint.velocity - original_vel).abs() < EPSILON,
            "negative dt should not change velocity"
        );
    }
}

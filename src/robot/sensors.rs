use glam::{Mat4, Quat, Vec3};
use serde::{Deserialize, Serialize};

use super::body::Robot;
use super::collision::{
    aabb_from_link, aabb_overlap, ray_scene_cast, ray_triangle_intersect, Aabb,
};
use super::definition::{CollisionShape, RobotDefinition, SensorDefinition, SensorMount};
use super::kinematics::LinkTransform;
use super::state::{RobotState, SensorReading};
use crate::scene::{Scene, SceneObject};

// ---------------------------------------------------------------------------
// Distance Sensor
// ---------------------------------------------------------------------------

/// Ray-cast distance sensor attached to a robot link.
///
/// Casts a ray from the link's world position along `local_direction` (rotated
/// into world frame) and returns the distance to the nearest scene-mesh hit,
/// or `max_range` when nothing is in the path.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct DistanceSensor {
    pub link_index: usize,
    pub local_direction: Vec3,
    pub max_range: f32,
}

#[allow(dead_code)]
impl DistanceSensor {
    /// Read the sensor against the given scene.
    ///
    /// Returns the distance to the nearest hit, or `max_range` if nothing is
    /// within range.  A `max_range` of zero always returns `0.0`.
    pub fn read(&self, robot_transforms: &[LinkTransform], scene: &Scene) -> f32 {
        if self.max_range <= 0.0 {
            return 0.0;
        }

        let tf = &robot_transforms[self.link_index];
        let world_dir = tf.rotation.mul_vec3(self.local_direction);

        match ray_scene_cast(tf.position, world_dir, &scene.meshes, self.max_range) {
            Some(hit) => hit.distance,
            None => self.max_range,
        }
    }
}

// ---------------------------------------------------------------------------
// Contact Sensor
// ---------------------------------------------------------------------------

/// Contact sensor that checks whether a link's AABB overlaps any scene object.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ContactSensor {
    pub link_index: usize,
}

#[allow(dead_code)]
impl ContactSensor {
    /// Returns `true` if the link's world-space AABB overlaps any scene mesh
    /// AABB.
    pub fn read(&self, robot_transforms: &[LinkTransform], robot: &Robot, scene: &Scene) -> bool {
        let tf = &robot_transforms[self.link_index];
        let link = &robot.links[self.link_index];
        let link_aabb = aabb_from_link(tf.position, tf.rotation, link.half_extents);

        for obj in &scene.meshes {
            let obj_aabb = aabb_from_mesh_vertices(&obj.mesh);
            if aabb_overlap(&link_aabb, &obj_aabb) {
                return true;
            }
        }

        false
    }
}

/// Compute an AABB from a mesh's vertices (min/max bounding box).
fn aabb_from_mesh_vertices(mesh: &crate::scene::Mesh) -> Aabb {
    let (min, max) = mesh.bounds();
    let center = (min + max) * 0.5;
    let half_extents = (max - min) * 0.5;
    Aabb {
        center,
        half_extents,
    }
}

// ---------------------------------------------------------------------------
// IMU Sensor
// ---------------------------------------------------------------------------

/// Inertial measurement unit that tracks linear acceleration and angular
/// velocity of a link by differencing positions across calls.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ImuSensor {
    pub link_index: usize,
    pub prev_velocity: Vec3,
    pub prev_angular_velocity: Vec3,
    /// Previous world-space position of the link (used to estimate velocity).
    prev_position: Vec3,
    /// Previous world-space rotation (used to estimate angular velocity).
    prev_rotation: Quat,
    /// Whether a previous reading exists (first call bootstraps state).
    initialized: bool,
}

/// A single IMU reading.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct ImuReading {
    pub linear_acceleration: Vec3,
    pub angular_velocity: Vec3,
}

#[allow(dead_code)]
impl ImuSensor {
    /// Create a new IMU sensor attached to `link_index`.
    pub fn new(link_index: usize) -> Self {
        Self {
            link_index,
            prev_velocity: Vec3::ZERO,
            prev_angular_velocity: Vec3::ZERO,
            prev_position: Vec3::ZERO,
            prev_rotation: Quat::IDENTITY,
            initialized: false,
        }
    }

    /// Read the sensor. Computes linear acceleration from the velocity delta
    /// and estimates angular velocity from rotation changes.
    ///
    /// On the first call, or when `dt` is zero, returns zero acceleration and
    /// zero angular velocity (no division by zero).
    pub fn read(&mut self, robot_transforms: &[LinkTransform], dt: f32) -> ImuReading {
        let tf = &robot_transforms[self.link_index];
        let current_pos = tf.position;
        let current_rot = tf.rotation;

        // Guard against zero or negative dt, or uninitialized state.
        if dt <= 0.0 || !self.initialized {
            // Bootstrap state without producing a reading.
            self.prev_position = current_pos;
            self.prev_rotation = current_rot;
            self.prev_velocity = Vec3::ZERO;
            self.prev_angular_velocity = Vec3::ZERO;
            self.initialized = true;

            return ImuReading {
                linear_acceleration: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
            };
        }

        // Estimate current velocity from position delta.
        let current_velocity = (current_pos - self.prev_position) / dt;

        // Linear acceleration = velocity delta / dt
        let linear_acceleration = (current_velocity - self.prev_velocity) / dt;

        // Estimate angular velocity from rotation delta.
        // delta_q ~= current * prev.inverse => axis-angle / dt gives angular vel.
        let delta_q = current_rot * self.prev_rotation.inverse();
        let (axis, angle) = delta_q.to_axis_angle();
        let angular_velocity = if angle.abs() > f32::EPSILON {
            axis * (angle / dt)
        } else {
            Vec3::ZERO
        };

        // Store for next call.
        self.prev_velocity = current_velocity;
        self.prev_angular_velocity = angular_velocity;
        self.prev_position = current_pos;
        self.prev_rotation = current_rot;

        ImuReading {
            linear_acceleration,
            angular_velocity,
        }
    }
}

// ---------------------------------------------------------------------------
// Camera Frustum
// ---------------------------------------------------------------------------

/// A simple cone-shaped camera frustum. Reports which scene objects have at
/// least one mesh vertex within the cone (angle from look direction < fov/2
/// and distance < max_range).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct CameraFrustum {
    pub link_index: usize,
    pub fov_radians: f32,
    pub max_range: f32,
    pub local_direction: Vec3,
}

#[allow(dead_code)]
impl CameraFrustum {
    /// Return indices of scene objects that have at least one vertex inside the
    /// camera's frustum cone.
    pub fn visible_objects(&self, robot_transforms: &[LinkTransform], scene: &Scene) -> Vec<usize> {
        if self.fov_radians <= 0.0 || self.max_range <= 0.0 {
            return Vec::new();
        }

        let tf = &robot_transforms[self.link_index];
        let world_dir = tf.rotation.mul_vec3(self.local_direction);
        let world_dir_norm = if world_dir.length_squared() > f32::EPSILON {
            world_dir.normalize()
        } else {
            return Vec::new();
        };

        let half_fov = self.fov_radians * 0.5;
        let cos_half_fov = half_fov.cos();

        let mut visible = Vec::new();

        for (idx, obj) in scene.meshes.iter().enumerate() {
            let mut found = false;
            for tri in &obj.mesh.triangles {
                for vert in &tri.vertices {
                    let to_vert = vert.position - tf.position;
                    let dist = to_vert.length();

                    if dist > self.max_range || dist < f32::EPSILON {
                        continue;
                    }

                    let cos_angle = to_vert.normalize().dot(world_dir_norm);

                    if cos_angle >= cos_half_fov {
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            if found {
                visible.push(idx);
            }
        }

        visible
    }
}

// ---------------------------------------------------------------------------
// Definition-based sensor simulation (RobotDefinition + RobotState)
// ---------------------------------------------------------------------------

/// Compute the world-space position and direction for a sensor given its mount
/// and the current robot state.
///
/// Uses `link_poses[mount.link_index]` (stored as `[f32; 16]`) to transform
/// the sensor's `local_offset` into world position and the sensor's local
/// direction into world direction.
#[allow(dead_code)]
pub fn sensor_world_pose(mount: &SensorMount, state: &RobotState) -> (Vec3, Vec3) {
    let link_mat = Mat4::from_cols_array(&state.link_poses[mount.link_index]);

    // Transform local_offset to world position
    let world_pos = link_mat.transform_point3(mount.local_offset);

    // Extract the sensor's local direction from the definition and rotate it
    // into world frame. The direction depends on the sensor type.
    let local_dir = match &mount.sensor {
        SensorDefinition::Distance { direction, .. } => *direction,
        SensorDefinition::Lidar { .. } => Vec3::Z, // default forward
        SensorDefinition::Contact => Vec3::Z,
        SensorDefinition::Imu => Vec3::Z,
    };

    // Transform direction (rotation only, no translation)
    let world_dir = link_mat.transform_vector3(local_dir);
    let world_dir = if world_dir.length_squared() > f32::EPSILON {
        world_dir.normalize()
    } else {
        local_dir
    };

    (world_pos, world_dir)
}

/// Simulate all sensors on the robot, updating `state.sensor_readings` in place.
///
/// For each sensor mount in the definition:
/// - **Distance**: cast a single ray from the sensor world position along its
///   world direction; return the nearest triangle intersection distance, or
///   `max_range` if nothing is hit.
/// - **LIDAR**: fan of `num_rays` rays spread over `fov_rad` centered on the
///   sensor direction (in a plane); return a `Vec<f32>` of distances.
/// - **Contact**: check whether any scene triangle vertex is within the
///   collision shape radius of the link.
/// - **IMU**: `linear_accel` = gravity `(0, -9.81, 0)`, `angular_vel` = sum
///   of ancestor joint velocities times their axes.
#[allow(dead_code)]
pub fn simulate_sensors(
    definition: &RobotDefinition,
    state: &mut RobotState,
    scene_meshes: &[SceneObject],
) {
    let gravity = Vec3::new(0.0, -9.81, 0.0);

    for (sensor_idx, mount) in definition.sensors.iter().enumerate() {
        if sensor_idx >= state.sensor_readings.len() {
            break;
        }

        let (world_pos, world_dir) = sensor_world_pose(mount, state);

        let reading = match &mount.sensor {
            // ---- Distance sensor ----
            SensorDefinition::Distance {
                max_range,
                direction: _,
            } => {
                let dist = cast_ray_against_scene(world_pos, world_dir, scene_meshes, *max_range);
                SensorReading::Distance(dist)
            }

            // ---- LIDAR sensor ----
            SensorDefinition::Lidar {
                num_rays,
                fov_rad,
                max_range,
            } => {
                let distances = simulate_lidar(
                    world_pos,
                    world_dir,
                    *num_rays,
                    *fov_rad,
                    *max_range,
                    scene_meshes,
                );
                SensorReading::Lidar(distances)
            }

            // ---- Contact sensor ----
            SensorDefinition::Contact => {
                let contact_radius =
                    collision_shape_radius(&definition.links[mount.link_index].collision_shape);
                let in_contact = check_contact(world_pos, contact_radius, scene_meshes);
                SensorReading::Contact(in_contact)
            }

            // ---- IMU sensor ----
            SensorDefinition::Imu => {
                let angular_vel = compute_imu_angular_vel(definition, state, mount.link_index);
                SensorReading::Imu {
                    linear_accel: gravity,
                    angular_vel,
                }
            }
        };

        state.sensor_readings[sensor_idx] = reading;
    }
}

/// Cast a single ray against all scene mesh triangles, returning the nearest
/// hit distance or `max_range` if nothing is hit.
fn cast_ray_against_scene(
    origin: Vec3,
    direction: Vec3,
    meshes: &[SceneObject],
    max_range: f32,
) -> f32 {
    let mut best_dist = max_range;

    for obj in meshes {
        for tri in &obj.mesh.triangles {
            let v0 = tri.vertices[0].position;
            let v1 = tri.vertices[1].position;
            let v2 = tri.vertices[2].position;

            if let Some(hit) = ray_triangle_intersect(origin, direction, v0, v1, v2) {
                if hit.distance < best_dist {
                    best_dist = hit.distance;
                }
            }
        }
    }

    best_dist
}

/// Simulate a LIDAR sensor: fan of `num_rays` rays spread over `fov_rad`
/// centered on the sensor direction, in a plane. Returns a Vec of distances.
fn simulate_lidar(
    origin: Vec3,
    center_dir: Vec3,
    num_rays: usize,
    fov_rad: f32,
    max_range: f32,
    meshes: &[SceneObject],
) -> Vec<f32> {
    if num_rays == 0 {
        return Vec::new();
    }

    // Find a perpendicular axis to spread rays in a plane.
    // Pick an "up" vector that isn't parallel to center_dir.
    let up_candidate = if center_dir.dot(Vec3::Y).abs() < 0.99 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let right = center_dir.cross(up_candidate).normalize();

    let mut distances = Vec::with_capacity(num_rays);

    for i in 0..num_rays {
        let angle = if num_rays == 1 {
            0.0
        } else {
            -fov_rad * 0.5 + fov_rad * (i as f32) / (num_rays as f32 - 1.0)
        };

        // Rotate center_dir by `angle` around the perpendicular axis.
        let ray_dir = Quat::from_axis_angle(right, angle).mul_vec3(center_dir);
        let dist = cast_ray_against_scene(origin, ray_dir, meshes, max_range);
        distances.push(dist);
    }

    distances
}

/// Extract an effective collision radius from a CollisionShape.
fn collision_shape_radius(shape: &CollisionShape) -> f32 {
    match shape {
        CollisionShape::Sphere { radius } => *radius,
        CollisionShape::Cuboid { half_extents } => half_extents.length(),
        CollisionShape::Cylinder { radius, height } => {
            (radius * radius + (height * 0.5) * (height * 0.5)).sqrt()
        }
    }
}

/// Check if any scene triangle vertex is within `radius` of `position`.
fn check_contact(position: Vec3, radius: f32, meshes: &[SceneObject]) -> bool {
    let radius_sq = radius * radius;
    for obj in meshes {
        for tri in &obj.mesh.triangles {
            for vert in &tri.vertices {
                if (vert.position - position).length_squared() <= radius_sq {
                    return true;
                }
            }
        }
    }
    false
}

/// Compute angular velocity for the IMU sensor by summing ancestor joint
/// velocities times their axes.
fn compute_imu_angular_vel(
    definition: &RobotDefinition,
    state: &RobotState,
    link_index: usize,
) -> Vec3 {
    let mut angular_vel = Vec3::ZERO;

    // Walk up the kinematic chain from the sensor link to the root.
    let mut current_link = link_index;
    while let Some(parent_joint_idx) = definition.links[current_link].parent_joint {
        let joint = &definition.joints[parent_joint_idx];
        let velocity = state
            .joint_velocities
            .get(parent_joint_idx)
            .copied()
            .unwrap_or(0.0);
        angular_vel += joint.axis * velocity;
        current_link = joint.parent_link;
    }

    angular_vel
}

// ---------------------------------------------------------------------------
// BVH-accelerated sensor simulation
// ---------------------------------------------------------------------------

use super::collision::SceneBvh;

fn cast_ray_bvh(origin: Vec3, direction: Vec3, bvh: &SceneBvh, max_range: f32) -> f32 {
    match bvh.ray_cast(origin, direction, max_range) {
        Some(hit) => hit.distance,
        None => max_range,
    }
}

fn simulate_lidar_bvh(
    origin: Vec3,
    center_dir: Vec3,
    num_rays: usize,
    fov_rad: f32,
    max_range: f32,
    bvh: &SceneBvh,
) -> Vec<f32> {
    if num_rays == 0 {
        return Vec::new();
    }

    let up_candidate = if center_dir.dot(Vec3::Y).abs() < 0.99 {
        Vec3::Y
    } else {
        Vec3::X
    };
    let right = center_dir.cross(up_candidate).normalize();

    let mut distances = Vec::with_capacity(num_rays);
    for i in 0..num_rays {
        let angle = if num_rays == 1 {
            0.0
        } else {
            -fov_rad * 0.5 + fov_rad * (i as f32) / (num_rays as f32 - 1.0)
        };
        let ray_dir = Quat::from_axis_angle(right, angle).mul_vec3(center_dir);
        distances.push(cast_ray_bvh(origin, ray_dir, bvh, max_range));
    }
    distances
}

/// BVH-accelerated sensor simulation. Same behavior as `simulate_sensors`
/// but uses the pre-built BVH for ray-casting instead of brute-force.
pub fn simulate_sensors_bvh(
    definition: &RobotDefinition,
    state: &mut RobotState,
    scene_meshes: &[SceneObject],
    bvh: &SceneBvh,
) {
    let gravity = Vec3::new(0.0, -9.81, 0.0);

    for (sensor_idx, mount) in definition.sensors.iter().enumerate() {
        if sensor_idx >= state.sensor_readings.len() {
            break;
        }

        let (world_pos, world_dir) = sensor_world_pose(mount, state);

        let reading = match &mount.sensor {
            SensorDefinition::Distance {
                max_range,
                direction: _,
            } => {
                let dist = cast_ray_bvh(world_pos, world_dir, bvh, *max_range);
                SensorReading::Distance(dist)
            }
            SensorDefinition::Lidar {
                num_rays,
                fov_rad,
                max_range,
            } => {
                let distances =
                    simulate_lidar_bvh(world_pos, world_dir, *num_rays, *fov_rad, *max_range, bvh);
                SensorReading::Lidar(distances)
            }
            SensorDefinition::Contact => {
                let contact_radius =
                    collision_shape_radius(&definition.links[mount.link_index].collision_shape);
                let in_contact = check_contact(world_pos, contact_radius, scene_meshes);
                SensorReading::Contact(in_contact)
            }
            SensorDefinition::Imu => {
                let angular_vel = compute_imu_angular_vel(definition, state, mount.link_index);
                SensorReading::Imu {
                    linear_accel: gravity,
                    angular_vel,
                }
            }
        };

        state.sensor_readings[sensor_idx] = reading;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::body::{Link, Robot};
    use crate::robot::kinematics::compute_forward_kinematics;
    use crate::scene::material::AcousticMaterial;
    use crate::scene::{Mesh, SceneObject, Triangle, Vertex};

    const EPSILON: f32 = 1e-4;

    // -- test helpers -------------------------------------------------------

    /// Build a triangle from three positions (normals set to +Y).
    fn tri(a: Vec3, b: Vec3, c: Vec3) -> Triangle {
        Triangle {
            vertices: [
                Vertex {
                    position: a,
                    normal: Vec3::Y,
                },
                Vertex {
                    position: b,
                    normal: Vec3::Y,
                },
                Vertex {
                    position: c,
                    normal: Vec3::Y,
                },
            ],
        }
    }

    /// Build a SceneObject from a list of triangles.
    fn scene_obj(name: &str, tris: Vec<Triangle>) -> SceneObject {
        SceneObject {
            name: name.into(),
            mesh: Mesh { triangles: tris },
            material: AcousticMaterial::default(),
            visible: true,
            interior_medium: None,
        }
    }

    /// Minimal scene with a single quad at z=5 (two triangles spanning
    /// x=[-1,1], y=[-1,1]).
    fn scene_with_wall_at_z5() -> Scene {
        let t1 = tri(
            Vec3::new(-1.0, -1.0, 5.0),
            Vec3::new(1.0, -1.0, 5.0),
            Vec3::new(0.0, 1.0, 5.0),
        );
        let t2 = tri(
            Vec3::new(1.0, -1.0, 5.0),
            Vec3::new(1.0, 1.0, 5.0),
            Vec3::new(0.0, 1.0, 5.0),
        );
        let mut scene = Scene::default();
        scene.meshes.push(scene_obj("wall", vec![t1, t2]));
        scene
    }

    /// Build a simple 1-link robot at the origin.
    fn simple_robot() -> Robot {
        let base = Link::new("base", Vec3::ZERO, Quat::IDENTITY, Vec3::splat(0.1), 1.0);
        Robot::new("bot", Vec3::ZERO, Quat::IDENTITY, base)
    }

    // -- DistanceSensor tests -----------------------------------------------

    #[test]
    fn test_distance_sensor_hit() {
        let scene = scene_with_wall_at_z5();
        let robot = simple_robot();
        let transforms = compute_forward_kinematics(&robot);

        let sensor = DistanceSensor {
            link_index: 0,
            local_direction: Vec3::Z, // pointing toward the wall
            max_range: 100.0,
        };

        let dist = sensor.read(&transforms, &scene);
        // Wall is at z=5, robot at origin => distance ~5
        assert!(
            (dist - 5.0).abs() < 0.1,
            "should detect wall at ~5, got {}",
            dist
        );
    }

    #[test]
    fn test_distance_sensor_no_hit() {
        let scene = scene_with_wall_at_z5();
        let robot = simple_robot();
        let transforms = compute_forward_kinematics(&robot);

        let sensor = DistanceSensor {
            link_index: 0,
            local_direction: Vec3::NEG_Z, // pointing away from the wall
            max_range: 100.0,
        };

        let dist = sensor.read(&transforms, &scene);
        assert!(
            (dist - 100.0).abs() < EPSILON,
            "should return max_range (100), got {}",
            dist
        );
    }

    #[test]
    fn test_distance_sensor_zero_range() {
        let scene = scene_with_wall_at_z5();
        let robot = simple_robot();
        let transforms = compute_forward_kinematics(&robot);

        let sensor = DistanceSensor {
            link_index: 0,
            local_direction: Vec3::Z,
            max_range: 0.0,
        };

        let dist = sensor.read(&transforms, &scene);
        assert!(
            dist.abs() < EPSILON,
            "max_range=0 should return 0, got {}",
            dist
        );
    }

    // -- ContactSensor tests ------------------------------------------------

    #[test]
    fn test_contact_sensor_touching() {
        // Place a scene object that overlaps the robot's base AABB at origin.
        let t1 = tri(
            Vec3::new(-0.05, -0.05, -0.05),
            Vec3::new(0.05, -0.05, -0.05),
            Vec3::new(0.0, 0.05, -0.05),
        );
        let t2 = tri(
            Vec3::new(-0.05, -0.05, 0.05),
            Vec3::new(0.05, -0.05, 0.05),
            Vec3::new(0.0, 0.05, 0.05),
        );
        let mut scene = Scene::default();
        scene.meshes.push(scene_obj("overlap", vec![t1, t2]));

        let robot = simple_robot();
        let transforms = compute_forward_kinematics(&robot);

        let sensor = ContactSensor { link_index: 0 };
        assert!(
            sensor.read(&transforms, &robot, &scene),
            "sensor should detect overlap"
        );
    }

    #[test]
    fn test_contact_sensor_clear() {
        // Scene object far away from the robot.
        let t1 = tri(
            Vec3::new(100.0, 100.0, 100.0),
            Vec3::new(101.0, 100.0, 100.0),
            Vec3::new(100.5, 101.0, 100.0),
        );
        let mut scene = Scene::default();
        scene.meshes.push(scene_obj("far_away", vec![t1]));

        let robot = simple_robot();
        let transforms = compute_forward_kinematics(&robot);

        let sensor = ContactSensor { link_index: 0 };
        assert!(
            !sensor.read(&transforms, &robot, &scene),
            "sensor should NOT detect overlap"
        );
    }

    // -- ImuSensor tests ----------------------------------------------------

    #[test]
    fn test_imu_stationary() {
        let robot = simple_robot();
        let transforms = compute_forward_kinematics(&robot);

        let mut imu = ImuSensor::new(0);

        // First call bootstraps.
        let _ = imu.read(&transforms, 0.01);
        // Second call with same position => zero acceleration.
        let reading = imu.read(&transforms, 0.01);

        assert!(
            reading.linear_acceleration.length() < EPSILON,
            "stationary robot should have ~zero acceleration, got {:?}",
            reading.linear_acceleration
        );
        assert!(
            reading.angular_velocity.length() < EPSILON,
            "stationary robot should have ~zero angular velocity, got {:?}",
            reading.angular_velocity
        );
    }

    #[test]
    fn test_imu_acceleration() {
        // Simulate a link that moves between calls.
        let mut imu = ImuSensor::new(0);

        let dt = 0.01;

        // t=0: position at origin.
        let tf0 = vec![LinkTransform {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        }];
        let _ = imu.read(&tf0, dt); // bootstrap

        // t=1: position at (1,0,0) => velocity (100,0,0).
        let tf1 = vec![LinkTransform {
            position: Vec3::new(1.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
        }];
        let _ = imu.read(&tf1, dt); // establishes velocity

        // t=2: position at (3,0,0) => velocity (200,0,0) => accel = (100/0.01)
        // Actually: vel_1 = (1-0)/0.01 = 100, vel_2 = (3-1)/0.01 = 200
        // accel = (200-100)/0.01 = 10000.
        let tf2 = vec![LinkTransform {
            position: Vec3::new(3.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
        }];
        let reading = imu.read(&tf2, dt);

        // Should detect non-zero acceleration along X.
        assert!(
            reading.linear_acceleration.x.abs() > 1.0,
            "should detect non-zero acceleration, got {:?}",
            reading.linear_acceleration
        );
    }

    #[test]
    fn test_imu_zero_dt() {
        let robot = simple_robot();
        let transforms = compute_forward_kinematics(&robot);

        let mut imu = ImuSensor::new(0);

        // dt=0 should return zero acceleration and no division by zero.
        let reading = imu.read(&transforms, 0.0);
        assert!(
            reading.linear_acceleration.length() < EPSILON,
            "dt=0 should produce zero acceleration, got {:?}",
            reading.linear_acceleration
        );
        assert!(
            reading.angular_velocity.length() < EPSILON,
            "dt=0 should produce zero angular velocity, got {:?}",
            reading.angular_velocity
        );
    }

    // -- CameraFrustum tests ------------------------------------------------

    #[test]
    fn test_camera_frustum_visible() {
        // Object at z=5 (our wall), camera looking along +Z with wide FOV.
        let scene = scene_with_wall_at_z5();
        let robot = simple_robot();
        let transforms = compute_forward_kinematics(&robot);

        let camera = CameraFrustum {
            link_index: 0,
            fov_radians: std::f32::consts::FRAC_PI_2, // 90 degrees
            max_range: 100.0,
            local_direction: Vec3::Z,
        };

        let visible = camera.visible_objects(&transforms, &scene);
        assert!(
            visible.contains(&0),
            "wall at z=5 should be visible, got {:?}",
            visible
        );
    }

    #[test]
    fn test_camera_frustum_outside() {
        // Object at z=5, camera looking along -Z (away from it).
        let scene = scene_with_wall_at_z5();
        let robot = simple_robot();
        let transforms = compute_forward_kinematics(&robot);

        let camera = CameraFrustum {
            link_index: 0,
            fov_radians: std::f32::consts::FRAC_PI_2,
            max_range: 100.0,
            local_direction: Vec3::NEG_Z,
        };

        let visible = camera.visible_objects(&transforms, &scene);
        assert!(
            visible.is_empty(),
            "wall behind camera should not be visible, got {:?}",
            visible
        );
    }

    #[test]
    fn test_camera_frustum_zero_fov() {
        let scene = scene_with_wall_at_z5();
        let robot = simple_robot();
        let transforms = compute_forward_kinematics(&robot);

        let camera = CameraFrustum {
            link_index: 0,
            fov_radians: 0.0,
            max_range: 100.0,
            local_direction: Vec3::Z,
        };

        let visible = camera.visible_objects(&transforms, &scene);
        assert!(
            visible.is_empty(),
            "zero FOV should see nothing, got {:?}",
            visible
        );
    }

    // -- Definition-based sensor simulation tests ------------------------------

    use crate::robot::definition::{
        CollisionShape, JointDefinition, JointType, LinkDefinition, RobotDefinition,
        SensorDefinition, SensorMount,
    };
    use crate::robot::state::{RobotState, SensorReading};

    /// Create a minimal one-link robot with a single sensor.
    fn one_link_def_with_sensor(sensor: SensorDefinition) -> RobotDefinition {
        RobotDefinition {
            name: "sensor_bot".into(),
            links: vec![LinkDefinition {
                name: "base".into(),
                mass: 1.0,
                inertia: 0.1,
                collision_shape: CollisionShape::Sphere { radius: 0.2 },
                parent_joint: None,
                body_zone: None,
            }],
            joints: vec![],
            sensors: vec![SensorMount {
                link_index: 0,
                local_offset: Vec3::ZERO,
                sensor,
            }],
        }
    }

    /// Create a two-link robot (base + child connected by a revolute joint)
    /// with a sensor on the specified link.
    fn two_link_def_with_sensor(sensor: SensorDefinition, sensor_link: usize) -> RobotDefinition {
        RobotDefinition {
            name: "two_link_bot".into(),
            links: vec![
                LinkDefinition {
                    name: "base".into(),
                    mass: 5.0,
                    inertia: 1.0,
                    collision_shape: CollisionShape::Cuboid {
                        half_extents: Vec3::splat(0.1),
                    },
                    parent_joint: None,
                    body_zone: None,
                },
                LinkDefinition {
                    name: "child".into(),
                    mass: 1.0,
                    inertia: 0.1,
                    collision_shape: CollisionShape::Sphere { radius: 0.3 },
                    parent_joint: Some(0),
                    body_zone: None,
                },
            ],
            joints: vec![JointDefinition {
                name: "joint_0".into(),
                joint_type: JointType::Revolute,
                axis: Vec3::Y,
                parent_link: 0,
                child_link: 1,
                limit_min: -std::f32::consts::PI,
                limit_max: std::f32::consts::PI,
                max_torque: 10.0,
                damping: 0.1,
            }],
            sensors: vec![SensorMount {
                link_index: sensor_link,
                local_offset: Vec3::ZERO,
                sensor,
            }],
        }
    }

    #[test]
    fn test_def_distance_sensor_hit() {
        let def = one_link_def_with_sensor(SensorDefinition::Distance {
            direction: Vec3::Z,
            max_range: 100.0,
        });
        let mut state = RobotState::new(&def);
        // Base link at identity -> sensor at origin looking +Z
        state.set_link_pose(0, Mat4::IDENTITY);

        // Place a triangle at z=5
        let t = tri(
            Vec3::new(-2.0, -2.0, 5.0),
            Vec3::new(2.0, -2.0, 5.0),
            Vec3::new(0.0, 2.0, 5.0),
        );
        let meshes = vec![scene_obj("wall", vec![t])];

        simulate_sensors(&def, &mut state, &meshes);

        match &state.sensor_readings[0] {
            SensorReading::Distance(d) => {
                assert!(
                    (*d - 5.0).abs() < EPSILON,
                    "Expected distance ~5.0, got {}",
                    d
                );
            }
            other => panic!("Expected Distance reading, got {:?}", other),
        }
    }

    #[test]
    fn test_def_distance_sensor_miss() {
        let max_range = 50.0;
        let def = one_link_def_with_sensor(SensorDefinition::Distance {
            direction: Vec3::Z,
            max_range,
        });
        let mut state = RobotState::new(&def);
        state.set_link_pose(0, Mat4::IDENTITY);

        // No geometry
        let meshes: Vec<SceneObject> = vec![];

        simulate_sensors(&def, &mut state, &meshes);

        match &state.sensor_readings[0] {
            SensorReading::Distance(d) => {
                assert!(
                    (*d - max_range).abs() < EPSILON,
                    "Expected max_range {}, got {}",
                    max_range,
                    d
                );
            }
            other => panic!("Expected Distance reading, got {:?}", other),
        }
    }

    #[test]
    fn test_def_lidar_fan_count() {
        let num_rays = 7;
        let def = one_link_def_with_sensor(SensorDefinition::Lidar {
            num_rays,
            fov_rad: std::f32::consts::PI,
            max_range: 100.0,
        });
        let mut state = RobotState::new(&def);
        state.set_link_pose(0, Mat4::IDENTITY);

        let meshes: Vec<SceneObject> = vec![];
        simulate_sensors(&def, &mut state, &meshes);

        match &state.sensor_readings[0] {
            SensorReading::Lidar(dists) => {
                assert_eq!(
                    dists.len(),
                    num_rays,
                    "LIDAR should return {} distances, got {}",
                    num_rays,
                    dists.len()
                );
            }
            other => panic!("Expected Lidar reading, got {:?}", other),
        }
    }

    #[test]
    fn test_def_contact_sensor_no_contact() {
        let def = one_link_def_with_sensor(SensorDefinition::Contact);
        let mut state = RobotState::new(&def);
        state.set_link_pose(0, Mat4::IDENTITY);

        // Triangle far away
        let t = tri(
            Vec3::new(100.0, 100.0, 100.0),
            Vec3::new(101.0, 100.0, 100.0),
            Vec3::new(100.0, 101.0, 100.0),
        );
        let meshes = vec![scene_obj("far", vec![t])];

        simulate_sensors(&def, &mut state, &meshes);

        match &state.sensor_readings[0] {
            SensorReading::Contact(c) => {
                assert!(!c, "Should not be in contact when geometry is far away");
            }
            other => panic!("Expected Contact reading, got {:?}", other),
        }
    }

    #[test]
    fn test_def_imu_at_rest() {
        let def = one_link_def_with_sensor(SensorDefinition::Imu);
        let mut state = RobotState::new(&def);
        state.set_link_pose(0, Mat4::IDENTITY);

        let meshes: Vec<SceneObject> = vec![];
        simulate_sensors(&def, &mut state, &meshes);

        match &state.sensor_readings[0] {
            SensorReading::Imu {
                linear_accel,
                angular_vel,
            } => {
                // Gravity is (0, -9.81, 0)
                assert!(
                    (linear_accel.x).abs() < EPSILON,
                    "accel X should be ~0, got {}",
                    linear_accel.x
                );
                assert!(
                    (linear_accel.y - (-9.81)).abs() < 0.01,
                    "accel Y should be ~-9.81, got {}",
                    linear_accel.y
                );
                assert!(
                    (linear_accel.z).abs() < EPSILON,
                    "accel Z should be ~0, got {}",
                    linear_accel.z
                );
                // No joints => angular vel should be zero
                assert!(
                    angular_vel.length() < EPSILON,
                    "angular_vel should be ~zero, got {:?}",
                    angular_vel
                );
            }
            other => panic!("Expected Imu reading, got {:?}", other),
        }
    }

    #[test]
    fn test_def_sensor_world_pose() {
        let mount = SensorMount {
            link_index: 0,
            local_offset: Vec3::new(1.0, 0.0, 0.0),
            sensor: SensorDefinition::Distance {
                direction: Vec3::Z,
                max_range: 10.0,
            },
        };

        let def = RobotDefinition {
            name: "test".into(),
            links: vec![LinkDefinition {
                name: "base".into(),
                mass: 1.0,
                inertia: 0.1,
                collision_shape: CollisionShape::Sphere { radius: 0.1 },
                parent_joint: None,
                body_zone: None,
            }],
            joints: vec![],
            sensors: vec![mount.clone()],
        };

        let mut state = RobotState::new(&def);

        // Place link at (5, 0, 0) with 90-degree rotation around Y.
        // After rotation around Y by 90 deg: local X -> world -Z, local Z -> world X.
        let pose = Mat4::from_rotation_translation(
            Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            Vec3::new(5.0, 0.0, 0.0),
        );
        state.set_link_pose(0, pose);

        let (pos, dir) = sensor_world_pose(&mount, &state);

        // local_offset (1,0,0) rotated 90 deg around Y -> (0, 0, -1)
        // world pos = (5, 0, 0) + (0, 0, -1) = (5, 0, -1)
        assert!(
            (pos - Vec3::new(5.0, 0.0, -1.0)).length() < EPSILON,
            "sensor world pos: expected (5,0,-1), got {:?}",
            pos
        );

        // direction (0,0,1) rotated 90 deg around Y -> (1,0,0)
        assert!(
            (dir - Vec3::new(1.0, 0.0, 0.0)).length() < EPSILON,
            "sensor world dir: expected (1,0,0), got {:?}",
            dir
        );
    }

    // ---- Edge case tests ----

    #[test]
    fn test_def_lidar_zero_rays() {
        let def = one_link_def_with_sensor(SensorDefinition::Lidar {
            num_rays: 0,
            fov_rad: std::f32::consts::PI,
            max_range: 100.0,
        });
        let mut state = RobotState::new(&def);
        state.set_link_pose(0, Mat4::IDENTITY);

        let meshes: Vec<SceneObject> = vec![];
        simulate_sensors(&def, &mut state, &meshes);

        match &state.sensor_readings[0] {
            SensorReading::Lidar(dists) => {
                assert!(dists.is_empty(), "zero rays should produce empty vec");
            }
            other => panic!("Expected Lidar reading, got {:?}", other),
        }
    }

    #[test]
    fn test_def_lidar_single_ray() {
        let def = one_link_def_with_sensor(SensorDefinition::Lidar {
            num_rays: 1,
            fov_rad: std::f32::consts::PI,
            max_range: 50.0,
        });
        let mut state = RobotState::new(&def);
        state.set_link_pose(0, Mat4::IDENTITY);

        let meshes: Vec<SceneObject> = vec![];
        simulate_sensors(&def, &mut state, &meshes);

        match &state.sensor_readings[0] {
            SensorReading::Lidar(dists) => {
                assert_eq!(dists.len(), 1, "single ray should produce 1 distance");
                assert!(
                    (dists[0] - 50.0).abs() < EPSILON,
                    "single ray should return max_range with no scene"
                );
            }
            other => panic!("Expected Lidar reading, got {:?}", other),
        }
    }

    #[test]
    fn test_def_contact_sensor_touching() {
        let def = one_link_def_with_sensor(SensorDefinition::Contact);
        let mut state = RobotState::new(&def);
        state.set_link_pose(0, Mat4::IDENTITY);

        // Place triangle vertices within collision radius of origin
        let t = tri(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(0.0, 0.1, 0.0),
        );
        let meshes = vec![scene_obj("touching", vec![t])];

        simulate_sensors(&def, &mut state, &meshes);

        match &state.sensor_readings[0] {
            SensorReading::Contact(c) => {
                assert!(*c, "Should be in contact when geometry is at origin");
            }
            other => panic!("Expected Contact reading, got {:?}", other),
        }
    }

    #[test]
    fn test_def_imu_with_rotating_joint() {
        let def = two_link_def_with_sensor(SensorDefinition::Imu, 1);
        let mut state = RobotState::new(&def);
        state.set_link_pose(0, Mat4::IDENTITY);
        state.set_link_pose(1, Mat4::IDENTITY);
        state.joint_velocities = vec![2.0]; // joint 0 rotating at 2 rad/s

        let meshes: Vec<SceneObject> = vec![];
        simulate_sensors(&def, &mut state, &meshes);

        match &state.sensor_readings[0] {
            SensorReading::Imu {
                angular_vel,
                linear_accel,
            } => {
                // Angular velocity should include joint 0's contribution
                assert!(
                    angular_vel.length() > 0.1,
                    "IMU on child link should detect angular velocity from parent joint, got {:?}",
                    angular_vel
                );
                // Linear accel should be gravity
                assert!(
                    (linear_accel.y - (-9.81)).abs() < 0.01,
                    "IMU linear_accel Y should be gravity"
                );
            }
            other => panic!("Expected Imu reading, got {:?}", other),
        }
    }

    #[test]
    fn test_def_no_sensors_no_panic() {
        let def = RobotDefinition {
            name: "no_sensors".into(),
            links: vec![LinkDefinition {
                name: "base".into(),
                mass: 1.0,
                inertia: 0.1,
                collision_shape: CollisionShape::Sphere { radius: 0.1 },
                parent_joint: None,
                body_zone: None,
            }],
            joints: vec![],
            sensors: vec![],
        };
        let mut state = RobotState::new(&def);
        let meshes: Vec<SceneObject> = vec![];
        simulate_sensors(&def, &mut state, &meshes);
        assert!(state.sensor_readings.is_empty());
    }

    #[test]
    fn test_def_distance_sensor_negative_range() {
        let def = one_link_def_with_sensor(SensorDefinition::Distance {
            direction: Vec3::Z,
            max_range: -5.0,
        });
        let mut state = RobotState::new(&def);
        state.set_link_pose(0, Mat4::IDENTITY);

        let t = tri(
            Vec3::new(-2.0, -2.0, 1.0),
            Vec3::new(2.0, -2.0, 1.0),
            Vec3::new(0.0, 2.0, 1.0),
        );
        let meshes = vec![scene_obj("wall", vec![t])];
        simulate_sensors(&def, &mut state, &meshes);

        match &state.sensor_readings[0] {
            SensorReading::Distance(d) => {
                assert!(
                    d.is_finite(),
                    "negative max_range should produce finite result"
                );
            }
            other => panic!("Expected Distance reading, got {:?}", other),
        }
    }

    #[test]
    fn test_def_empty_scene_all_sensors() {
        // Robot with all sensor types, empty scene
        let def = RobotDefinition {
            name: "multi_sensor".into(),
            links: vec![LinkDefinition {
                name: "base".into(),
                mass: 1.0,
                inertia: 0.1,
                collision_shape: CollisionShape::Sphere { radius: 0.2 },
                parent_joint: None,
                body_zone: None,
            }],
            joints: vec![],
            sensors: vec![
                SensorMount {
                    link_index: 0,
                    local_offset: Vec3::ZERO,
                    sensor: SensorDefinition::Distance {
                        direction: Vec3::Z,
                        max_range: 10.0,
                    },
                },
                SensorMount {
                    link_index: 0,
                    local_offset: Vec3::ZERO,
                    sensor: SensorDefinition::Lidar {
                        num_rays: 5,
                        fov_rad: 1.0,
                        max_range: 10.0,
                    },
                },
                SensorMount {
                    link_index: 0,
                    local_offset: Vec3::ZERO,
                    sensor: SensorDefinition::Contact,
                },
                SensorMount {
                    link_index: 0,
                    local_offset: Vec3::ZERO,
                    sensor: SensorDefinition::Imu,
                },
            ],
        };
        let mut state = RobotState::new(&def);
        state.set_link_pose(0, Mat4::IDENTITY);

        let meshes: Vec<SceneObject> = vec![];
        simulate_sensors(&def, &mut state, &meshes);

        assert_eq!(state.sensor_readings.len(), 4);
        match &state.sensor_readings[0] {
            SensorReading::Distance(d) => assert!((*d - 10.0).abs() < EPSILON),
            other => panic!("Expected Distance, got {:?}", other),
        }
        match &state.sensor_readings[1] {
            SensorReading::Lidar(dists) => assert_eq!(dists.len(), 5),
            other => panic!("Expected Lidar, got {:?}", other),
        }
        match &state.sensor_readings[2] {
            SensorReading::Contact(c) => assert!(!c),
            other => panic!("Expected Contact, got {:?}", other),
        }
        match &state.sensor_readings[3] {
            SensorReading::Imu { .. } => {}
            other => panic!("Expected Imu, got {:?}", other),
        }
    }

    #[test]
    fn test_collision_shape_radius_all_variants() {
        let sphere_r = collision_shape_radius(&CollisionShape::Sphere { radius: 1.5 });
        assert!((sphere_r - 1.5).abs() < EPSILON);

        let cuboid_r = collision_shape_radius(&CollisionShape::Cuboid {
            half_extents: Vec3::new(1.0, 1.0, 1.0),
        });
        assert!((cuboid_r - 3.0_f32.sqrt()).abs() < 0.01);

        let cyl_r = collision_shape_radius(&CollisionShape::Cylinder {
            radius: 1.0,
            height: 2.0,
        });
        // sqrt(1^2 + 1^2) = sqrt(2)
        assert!((cyl_r - 2.0_f32.sqrt()).abs() < 0.01);
    }

    #[test]
    fn test_collision_shape_radius_zero() {
        let r = collision_shape_radius(&CollisionShape::Sphere { radius: 0.0 });
        assert!(r.abs() < EPSILON, "zero radius sphere should return 0");
    }
}

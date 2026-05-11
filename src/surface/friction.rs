use glam::Vec3;

/// Result of a Coulomb friction computation.
pub struct FrictionResult {
    pub force_magnitude: f32,
    pub is_static: bool,
}

/// Velocity threshold below which friction is considered static.
const VELOCITY_THRESHOLD: f32 = 1e-6;

/// Compute scalar friction using the Coulomb model.
///
/// Static friction applies when velocity is approximately zero (below threshold).
/// Kinetic friction applies when the object is moving.
/// Negative coefficients are clamped to zero.
pub fn compute_friction(
    normal_force: f32,
    velocity: f32,
    friction_static: f32,
    friction_kinetic: f32,
) -> FrictionResult {
    let normal_force = normal_force.max(0.0);
    let mu_s = friction_static.max(0.0);
    let mu_k = friction_kinetic.max(0.0);

    if velocity.abs() < VELOCITY_THRESHOLD {
        FrictionResult {
            force_magnitude: mu_s * normal_force,
            is_static: true,
        }
    } else {
        FrictionResult {
            force_magnitude: mu_k * normal_force,
            is_static: false,
        }
    }
}

/// Compute friction force vector opposing velocity direction.
///
/// Returns a Vec3 friction force that opposes the direction of motion.
/// When velocity is near zero, returns Vec3::ZERO (static friction has no
/// preferred direction without an applied force).
pub fn compute_friction_force(
    normal_force: f32,
    velocity: Vec3,
    friction_static: f32,
    friction_kinetic: f32,
) -> Vec3 {
    let speed = velocity.length();
    let result = compute_friction(normal_force, speed, friction_static, friction_kinetic);

    if speed < VELOCITY_THRESHOLD {
        // Static friction: no direction to oppose without applied force
        Vec3::ZERO
    } else {
        // Kinetic friction: oppose velocity direction
        let direction = velocity / speed;
        -direction * result.force_magnitude
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_static_friction_at_rest() {
        let result = compute_friction(10.0, 0.0, 0.5, 0.3);
        assert!(result.is_static);
        assert!((result.force_magnitude - 5.0).abs() < EPSILON);
    }

    #[test]
    fn test_kinetic_friction_moving() {
        let result = compute_friction(10.0, 5.0, 0.5, 0.3);
        assert!(!result.is_static);
        assert!((result.force_magnitude - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_friction_opposes_motion() {
        let velocity = Vec3::new(3.0, 0.0, 4.0); // magnitude 5
        let force = compute_friction_force(10.0, velocity, 0.5, 0.3);
        // Force should oppose velocity direction
        let dot = force.dot(velocity);
        assert!(
            dot < 0.0,
            "Friction force should oppose velocity, dot={dot}"
        );
        // Magnitude should be kinetic friction (moving)
        assert!((force.length() - 3.0).abs() < EPSILON);
    }

    #[test]
    fn test_zero_normal_force() {
        let result = compute_friction(0.0, 5.0, 0.5, 0.3);
        assert!((result.force_magnitude).abs() < EPSILON);

        let force = compute_friction_force(0.0, Vec3::new(1.0, 0.0, 0.0), 0.5, 0.3);
        assert!(force.length() < EPSILON);
    }

    #[test]
    fn test_negative_coefficients_clamped() {
        let result = compute_friction(10.0, 0.0, -0.5, -0.3);
        assert!((result.force_magnitude).abs() < EPSILON);

        let result2 = compute_friction(10.0, 5.0, -0.5, -0.3);
        assert!((result2.force_magnitude).abs() < EPSILON);
    }

    #[test]
    fn test_friction_static_ge_kinetic() {
        let static_result = compute_friction(10.0, 0.0, 0.5, 0.3);
        let kinetic_result = compute_friction(10.0, 5.0, 0.5, 0.3);
        assert!(
            static_result.force_magnitude >= kinetic_result.force_magnitude,
            "Static friction {} should be >= kinetic friction {}",
            static_result.force_magnitude,
            kinetic_result.force_magnitude
        );
    }
}

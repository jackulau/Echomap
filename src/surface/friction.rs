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

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_edge_velocity_exactly_at_threshold() {
        // Velocity exactly equal to VELOCITY_THRESHOLD (1e-6) should be static
        let result = compute_friction(10.0, 1e-6, 0.5, 0.3);
        // 1e-6.abs() < 1e-6 is false, so this should be kinetic
        assert!(
            !result.is_static,
            "Velocity exactly at threshold should be kinetic (not strictly less)"
        );
    }

    #[test]
    fn test_edge_velocity_just_below_threshold() {
        let result = compute_friction(10.0, 9.99e-7, 0.5, 0.3);
        assert!(
            result.is_static,
            "Velocity just below threshold should be static"
        );
    }

    #[test]
    fn test_edge_negative_velocity_scalar() {
        // Negative velocity should still produce kinetic friction (abs check)
        let result = compute_friction(10.0, -5.0, 0.5, 0.3);
        assert!(
            !result.is_static,
            "Negative velocity should be treated as moving"
        );
        assert!(
            (result.force_magnitude - 3.0).abs() < EPSILON,
            "Negative velocity kinetic friction magnitude should be mu_k * N = 3.0, got {}",
            result.force_magnitude
        );
    }

    #[test]
    fn test_edge_huge_normal_force() {
        let result = compute_friction(1e12, 5.0, 0.5, 0.3);
        assert!(
            result.force_magnitude.is_finite(),
            "Huge normal force should produce finite friction, got {}",
            result.force_magnitude
        );
        assert!(
            (result.force_magnitude - 0.3 * 1e12).abs() < 1e6,
            "Huge normal force friction should be proportional"
        );
    }

    #[test]
    fn test_edge_negative_normal_force_clamped() {
        // Negative normal force should be clamped to zero
        let result = compute_friction(-10.0, 5.0, 0.5, 0.3);
        assert!(
            result.force_magnitude.abs() < EPSILON,
            "Negative normal force should yield zero friction, got {}",
            result.force_magnitude
        );
    }

    #[test]
    fn test_edge_both_coefficients_zero() {
        let result_static = compute_friction(10.0, 0.0, 0.0, 0.0);
        assert!(
            result_static.force_magnitude.abs() < EPSILON,
            "Zero coefficients should yield zero friction"
        );

        let result_kinetic = compute_friction(10.0, 5.0, 0.0, 0.0);
        assert!(
            result_kinetic.force_magnitude.abs() < EPSILON,
            "Zero coefficients kinetic should yield zero friction"
        );
    }

    #[test]
    fn test_edge_kinetic_greater_than_static() {
        // When kinetic > static, function should still work (no assertion)
        let result_s = compute_friction(10.0, 0.0, 0.3, 0.5);
        let result_k = compute_friction(10.0, 5.0, 0.3, 0.5);
        assert!(
            (result_s.force_magnitude - 3.0).abs() < EPSILON,
            "Static friction should use mu_s=0.3"
        );
        assert!(
            (result_k.force_magnitude - 5.0).abs() < EPSILON,
            "Kinetic friction should use mu_k=0.5"
        );
    }

    #[test]
    fn test_edge_nan_normal_force() {
        let result = compute_friction(f32::NAN, 5.0, 0.5, 0.3);
        // NaN.max(0.0) returns 0.0 in Rust, so friction should be zero
        assert!(
            result.force_magnitude.abs() < EPSILON || result.force_magnitude.is_nan(),
            "NaN normal force should produce zero or NaN friction"
        );
    }

    #[test]
    fn test_edge_infinity_normal_force() {
        let result = compute_friction(f32::INFINITY, 5.0, 0.5, 0.3);
        assert!(
            result.force_magnitude.is_infinite() || result.force_magnitude.is_finite(),
            "Infinite normal force should not panic"
        );
    }

    #[test]
    fn test_edge_infinity_velocity() {
        let result = compute_friction(10.0, f32::INFINITY, 0.5, 0.3);
        assert!(!result.is_static, "Infinite velocity should be kinetic");
        assert!(
            (result.force_magnitude - 3.0).abs() < EPSILON,
            "Friction magnitude should still be mu_k * N"
        );
    }

    #[test]
    fn test_edge_nan_velocity() {
        let result = compute_friction(10.0, f32::NAN, 0.5, 0.3);
        // NaN.abs() < threshold is false, so kinetic path taken
        assert!(
            !result.is_static,
            "NaN velocity comparison should take kinetic path"
        );
    }

    #[test]
    fn test_edge_friction_force_zero_velocity_vector() {
        let force = compute_friction_force(10.0, Vec3::ZERO, 0.5, 0.3);
        assert!(
            force.length() < EPSILON,
            "Zero velocity vector should return zero force, got length={}",
            force.length()
        );
    }

    #[test]
    fn test_edge_friction_force_very_small_velocity() {
        // Velocity with magnitude < threshold
        let tiny_vel = Vec3::new(1e-8, 0.0, 0.0);
        let force = compute_friction_force(10.0, tiny_vel, 0.5, 0.3);
        assert!(
            force.length() < EPSILON,
            "Very small velocity should produce zero force, got length={}",
            force.length()
        );
    }

    #[test]
    fn test_edge_friction_force_unit_axes() {
        // Force along each axis should oppose that axis
        for axis in [Vec3::X, Vec3::Y, Vec3::Z, -Vec3::X, -Vec3::Y, -Vec3::Z] {
            let force = compute_friction_force(10.0, axis, 0.5, 0.3);
            let dot = force.dot(axis);
            assert!(
                dot < 0.0,
                "Friction should oppose velocity along {:?}, dot={dot}",
                axis
            );
        }
    }

    #[test]
    fn test_edge_friction_force_diagonal_velocity() {
        // Diagonal velocity: force should oppose it exactly
        let vel = Vec3::new(1.0, 1.0, 1.0);
        let force = compute_friction_force(10.0, vel, 0.5, 0.3);
        let dot = force.dot(vel);
        assert!(
            dot < 0.0,
            "Friction should oppose diagonal velocity, dot={dot}"
        );
        // Magnitude should be mu_k * N = 3.0
        assert!(
            (force.length() - 3.0).abs() < EPSILON,
            "Diagonal friction magnitude should be 3.0, got {}",
            force.length()
        );
    }
}

use crate::scene::material::MediumProperties;
use crate::scene::{AcousticMaterial, Triangle};
use glam::Vec3;

/// Hard cap on `AcousticRay::path` length. Long-running simulations would
/// otherwise accumulate unbounded memory per ray as bounces grow.
/// Aligned with SimulationConfig::max_bounces default (50) plus the origin
/// entry, with headroom for refraction-branched paths.
pub const DEFAULT_MAX_PATH_LENGTH: usize = 64;

/// Per-octave-band energy (125/250/500/1k/2k/4k Hz). Mirrors
/// `FrequencyBands.as_array()` ordering so per-band absorption/attenuation
/// indices align trivially.
///
/// Number of octave bands tracked per ray: 125 / 250 / 500 / 1k / 2k / 4k Hz.
pub const BAND_COUNT: usize = 6;

pub type EnergyBands = [f32; BAND_COUNT];

/// Broadcast a scalar to all 6 bands.
#[inline]
pub fn energy_uniform(e: f32) -> EnergyBands {
    [e; 6]
}

/// Maximum across bands — used as the survival criterion (a ray is alive if
/// any band still carries energy above threshold).
#[inline]
pub fn energy_max(e: &EnergyBands) -> f32 {
    e.iter().copied().fold(0.0_f32, f32::max)
}

/// Sum across bands. Useful as a single "total energy" proxy for display.
#[inline]
pub fn energy_sum(e: &EnergyBands) -> f32 {
    e.iter().sum()
}

/// Multiply every band by a scalar, in place.
#[inline]
pub fn energy_scale(e: &mut EnergyBands, s: f32) {
    for x in e.iter_mut() {
        *x *= s;
    }
}

/// Return a scaled copy.
#[inline]
pub fn energy_scaled(e: &EnergyBands, s: f32) -> EnergyBands {
    let mut out = *e;
    energy_scale(&mut out, s);
    out
}

#[allow(dead_code)]
pub struct RefractionResult {
    pub reflected_direction: Vec3,
    pub reflected_energy: [f32; 6],
    pub transmitted_direction: Option<Vec3>,
    pub transmitted_energy: [f32; 6],
}

pub struct AcousticRay {
    pub origin: Vec3,
    pub direction: Vec3,
    pub energy: [f32; 6],
    pub bounces: u32,
    pub path: Vec<Vec3>,
    pub current_medium: MediumProperties,
    pub frequency_hz: f32,
    /// Maximum number of points retained in `path`. Older points are dropped
    /// (FIFO) once this cap is reached.
    pub max_path_length: usize,
}

#[allow(dead_code)]
pub struct RayHit {
    pub point: Vec3,
    pub normal: Vec3,
    pub distance: f32,
    pub triangle_index: usize,
}

impl AcousticRay {
    pub fn new(origin: Vec3, direction: Vec3, energy: f32, medium: MediumProperties) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
            energy: energy_uniform(energy),
            bounces: 0,
            path: vec![origin],
            current_medium: medium,
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        }
    }

    /// Construct from an already-banded energy vector. Used when refraction
    /// branches a transmitted ray that inherits the parent's per-band energy.
    pub fn new_with_bands(
        origin: Vec3,
        direction: Vec3,
        energy: EnergyBands,
        medium: MediumProperties,
    ) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
            energy,
            bounces: 0,
            path: vec![origin],
            current_medium: medium,
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        }
    }

    /// Largest-band energy. Used by the simulation loop as the "is the ray
    /// still meaningfully alive?" criterion — a ray with any band above
    /// threshold should keep tracing.
    #[inline]
    pub fn energy_max(&self) -> f32 {
        energy_max(&self.energy)
    }

    /// Re-derive the speed of sound for the current medium from a measured
    /// local density. Uses the ideal-gas approximation: c scales with
    /// sqrt(reference_density / local_density) at fixed pressure (c^2 ~ P/rho).
    /// This couples acoustic propagation to fluid/gas density fields so a
    /// ray traversing a hot/light parcel speeds up and a cold/dense parcel
    /// slows it down — matching real-world behaviour.
    pub fn update_speed_from_density(&mut self, local_density: f32) {
        if local_density <= 0.0 {
            return;
        }
        let ref_density = self.current_medium.density.max(1e-6);
        let ref_c = self.current_medium.speed_of_sound;
        let new_c = ref_c * (ref_density / local_density).sqrt();
        self.current_medium.speed_of_sound = new_c;
        self.current_medium.impedance = local_density * new_c;
        self.current_medium.density = local_density;
    }

    /// Append a point to `path`, evicting the oldest entry once
    /// `max_path_length` is reached. Keeps memory bounded over long traces.
    pub fn push_path_point(&mut self, p: Vec3) {
        if self.path.len() >= self.max_path_length {
            self.path.remove(0);
        }
        self.path.push(p);
    }

    pub fn intersect_triangle(&self, tri: &Triangle) -> Option<f32> {
        // Möller–Trumbore intersection
        let edge1 = tri.vertices[1].position - tri.vertices[0].position;
        let edge2 = tri.vertices[2].position - tri.vertices[0].position;
        let h = self.direction.cross(edge2);
        let a = edge1.dot(h);

        if a.abs() < 1e-7 {
            return None;
        }

        let f = 1.0 / a;
        let s = self.origin - tri.vertices[0].position;
        let u = f * s.dot(h);

        if !(0.0..=1.0).contains(&u) {
            return None;
        }

        let q = s.cross(edge1);
        let v = f * self.direction.dot(q);

        if v < 0.0 || u + v > 1.0 {
            return None;
        }

        let t = f * edge2.dot(q);

        if t > 1e-5 {
            Some(t)
        } else {
            None
        }
    }

    pub fn reflect(&mut self, hit: &RayHit, material: &AcousticMaterial) {
        let absorption = material.absorption.as_array();
        for (band, e) in self.energy.iter_mut().enumerate() {
            *e *= 1.0 - absorption[band];
        }
        self.origin = hit.point + hit.normal * 1e-4;
        self.direction = self.direction - 2.0 * self.direction.dot(hit.normal) * hit.normal;
        self.direction = self.direction.normalize();
        self.bounces += 1;
        self.push_path_point(hit.point);
    }

    /// Compute refraction at a medium boundary using Snell's law and Fresnel
    /// equations for acoustic impedance.
    ///
    /// Returns `None` only when the computation is degenerate (e.g. zero-length
    /// direction). Total internal reflection is represented by
    /// `transmitted_direction = None` inside the returned `RefractionResult`.
    pub fn refract(
        &self,
        hit_normal: Vec3,
        new_medium: &MediumProperties,
    ) -> Option<RefractionResult> {
        let z1 = self.current_medium.impedance;
        let z2 = new_medium.impedance;

        // Guard: if both impedances are near-zero, treat as no boundary
        if (z1 + z2).abs() < 1e-10 {
            return Some(RefractionResult {
                reflected_direction: self.direction,
                reflected_energy: energy_uniform(0.0),
                transmitted_direction: Some(self.direction),
                transmitted_energy: self.energy,
            });
        }

        let c1 = self.current_medium.speed_of_sound;
        let c2 = new_medium.speed_of_sound;

        // Ensure normal points against the ray direction (toward the incoming ray)
        let n = if self.direction.dot(hit_normal) < 0.0 {
            hit_normal
        } else {
            -hit_normal
        };

        let cos_theta1 = (-self.direction.dot(n)).clamp(0.0, 1.0);
        let sin_theta1 = (1.0 - cos_theta1 * cos_theta1).max(0.0).sqrt();

        // Snell's law: sin(theta2) = (c2/c1) * sin(theta1)
        let sin_theta2 = (c2 / c1) * sin_theta1;

        // Total internal reflection
        if sin_theta2 >= 1.0 - f32::EPSILON {
            let reflected_dir = self.direction - 2.0 * self.direction.dot(n) * n;
            return Some(RefractionResult {
                reflected_direction: reflected_dir.normalize(),
                reflected_energy: self.energy,
                transmitted_direction: None,
                transmitted_energy: energy_uniform(0.0),
            });
        }

        let cos_theta2 = (1.0 - sin_theta2 * sin_theta2).max(0.0).sqrt();

        // Fresnel reflection coefficient for acoustic impedance:
        // R = ((Z2*cos_theta1 - Z1*cos_theta2) / (Z2*cos_theta1 + Z1*cos_theta2))^2
        let numerator = z2 * cos_theta1 - z1 * cos_theta2;
        let denominator = z2 * cos_theta1 + z1 * cos_theta2;

        let r = if denominator.abs() < 1e-10 {
            0.0
        } else {
            (numerator / denominator).powi(2)
        };
        let t = 1.0 - r;

        // Reflected direction
        let reflected_dir = self.direction - 2.0 * self.direction.dot(n) * n;

        // Transmitted (refracted) direction via Snell's law
        // t_dir = (c2/c1)*d + ((c2/c1)*cos_theta1 - cos_theta2)*n
        let ratio = c2 / c1;
        let transmitted_dir = ratio * self.direction + (ratio * cos_theta1 - cos_theta2) * n;

        Some(RefractionResult {
            reflected_direction: reflected_dir.normalize(),
            reflected_energy: energy_scaled(&self.energy, r),
            transmitted_direction: Some(transmitted_dir.normalize()),
            transmitted_energy: energy_scaled(&self.energy, t),
        })
    }

    /// Apply volumetric attenuation based on distance traveled in the current
    /// medium. Bands ≠ frequency_hz — D1 keeps the historical behaviour of
    /// using the ray's nominal `frequency_hz` to pick a single scalar
    /// dB/m, then applies that uniformly across all 6 bands. Per-band
    /// attenuation is a downstream optimisation; preserving the scalar path
    /// keeps the existing test_volumetric_attenuation_* contracts intact.
    pub fn apply_volumetric_attenuation(&mut self, distance: f32) {
        let atten_db_per_m = self
            .current_medium
            .attenuation
            .at_frequency(self.frequency_hz);
        // Convert dB attenuation to linear factor: factor = 10^(-atten*distance/10)
        let total_atten_db = atten_db_per_m * distance;
        let factor = 10.0_f32.powf(-total_atten_db / 10.0);
        energy_scale(&mut self.energy, factor);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::material::{MediumLibrary, MediumProperties};

    fn air() -> MediumProperties {
        MediumProperties::air()
    }

    fn water() -> MediumProperties {
        let lib = MediumLibrary::with_defaults();
        lib.get("Water").unwrap().clone()
    }

    #[test]
    fn test_refraction_air_to_water_normal_incidence() {
        // At normal incidence (theta1=0), theta2 should also be 0
        // Direction straight down, normal pointing up
        let ray = AcousticRay {
            origin: Vec3::new(0.0, 1.0, 0.0),
            direction: Vec3::new(0.0, -1.0, 0.0),
            energy: energy_uniform(1.0),
            bounces: 0,
            path: vec![Vec3::new(0.0, 1.0, 0.0)],
            current_medium: air(),
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        };

        let water_med = water();
        let normal = Vec3::new(0.0, 1.0, 0.0); // surface normal pointing up

        let result = ray.refract(normal, &water_med).unwrap();

        // Transmitted direction should be straight down (same as incident at normal)
        let transmitted = result.transmitted_direction.unwrap();
        assert!(
            (transmitted.y - (-1.0)).abs() < 0.01,
            "At normal incidence, transmitted dir should be (0,-1,0), got {:?}",
            transmitted
        );

        // Energy should split by impedance ratio
        // Z_air = 1.225*343 = 420.175, Z_water = 998*1481 = 1478038
        // R = ((Z2-Z1)/(Z2+Z1))^2 at normal incidence
        let z_air = air().impedance;
        let z_water = water_med.impedance;
        let expected_r = ((z_water - z_air) / (z_water + z_air)).powi(2);
        let expected_t = 1.0 - expected_r;

        assert!(
            (result.reflected_energy[0] - expected_r).abs() < 0.001,
            "Reflected energy: expected {expected_r}, got {}",
            result.reflected_energy[0]
        );
        assert!(
            (result.transmitted_energy[0] - expected_t).abs() < 0.001,
            "Transmitted energy: expected {expected_t}, got {}",
            result.transmitted_energy[0]
        );
    }

    #[test]
    fn test_refraction_air_to_water_45_degrees() {
        // Air→Water at 45 deg: sin(theta2) = (1481/343)*sin(45) ≈ 3.05 > 1
        // This exceeds the critical angle (~13.4 deg), so total reflection occurs.
        // Verify the implementation correctly detects TIR at this angle.
        let dir = Vec3::new(1.0, -1.0, 0.0).normalize();
        let ray = AcousticRay {
            origin: Vec3::new(0.0, 1.0, 0.0),
            direction: dir,
            energy: energy_uniform(1.0),
            bounces: 0,
            path: vec![Vec3::new(0.0, 1.0, 0.0)],
            current_medium: air(),
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        };

        let water_med = water();
        let normal = Vec3::new(0.0, 1.0, 0.0);

        let result = ray.refract(normal, &water_med).unwrap();

        // At 45 deg air->water: sin_theta2 > 1 → total reflection
        assert!(
            result.transmitted_direction.is_none(),
            "Air-to-water at 45 deg should be total reflection (sin_theta2 > 1)"
        );
        assert!(
            (result.reflected_energy[0] - 1.0).abs() < 0.001,
            "All energy should be reflected in TIR"
        );
    }

    #[test]
    fn test_refraction_air_to_water_small_angle() {
        // Use 5 degrees (well below critical angle of ~13.4 deg for air->water)
        let angle = 5.0_f32.to_radians();
        let dir = Vec3::new(angle.sin(), -angle.cos(), 0.0).normalize();
        let ray = AcousticRay {
            origin: Vec3::new(0.0, 1.0, 0.0),
            direction: dir,
            energy: energy_uniform(1.0),
            bounces: 0,
            path: vec![Vec3::new(0.0, 1.0, 0.0)],
            current_medium: air(),
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        };

        let water_med = water();
        let normal = Vec3::new(0.0, 1.0, 0.0);

        let result = ray.refract(normal, &water_med).unwrap();
        let transmitted = result.transmitted_direction.unwrap();

        // Snell: sin(theta2) = (1481/343)*sin(5deg) = 4.316*0.08716 = 0.3762
        // theta2 = arcsin(0.3762) ≈ 22.09 deg
        let expected_sin_theta2 = (1481.0 / 343.0) * (5.0_f32.to_radians().sin());
        let expected_theta2 = expected_sin_theta2.asin();

        // The transmitted direction should have sin(angle) ≈ expected
        // transmitted is normalized, so its x component = sin(angle_from_normal)
        let actual_sin = transmitted.x.abs(); // x component = sin of angle from y-axis normal
        assert!(
            (actual_sin - expected_sin_theta2).abs() < 0.02,
            "Snell angle mismatch: expected sin(theta2)={expected_sin_theta2:.4}, got {actual_sin:.4}, expected theta2={:.1}deg",
            expected_theta2.to_degrees()
        );
    }

    #[test]
    fn test_total_internal_reflection_water_to_air() {
        // Water to air: critical angle ≈ 13.4 deg
        // Use 20 degrees (above critical) → should get TIR
        let angle = 20.0_f32.to_radians();
        let dir = Vec3::new(angle.sin(), -angle.cos(), 0.0).normalize();
        let ray = AcousticRay {
            origin: Vec3::new(0.0, 1.0, 0.0),
            direction: dir,
            energy: energy_uniform(1.0),
            bounces: 0,
            path: vec![Vec3::new(0.0, 1.0, 0.0)],
            current_medium: water(),
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        };

        let air_med = air();
        let normal = Vec3::new(0.0, 1.0, 0.0);

        let result = ray.refract(normal, &air_med).unwrap();

        // sin(theta2) = (c_air/c_water)*sin(20deg) = (343/1481)*0.342 = 0.0792
        // Hmm that's < 1. Let me recalculate.
        // Water to air: c1=1481, c2=343
        // sin(theta2) = (343/1481)*sin(20) = 0.2316*0.342 = 0.0792
        // That's NOT TIR. For TIR water→air we need sin(theta2) >= 1
        // sin(theta2) = (c2/c1)*sin(theta1) = (343/1481)*sin(theta1) = 0.2316*sin(theta1)
        // For TIR: 0.2316*sin(theta1) >= 1 → sin(theta1) >= 4.32 → impossible!
        //
        // Wait, I got confused. Water→air is c1=1481 (fast), c2=343 (slow).
        // sin(theta2) = (c2/c1)*sin(theta1) = (343/1481)*sin(theta1) < sin(theta1)
        // So the refracted ray bends TOWARD normal. TIR never happens going
        // from fast to slow!
        //
        // TIR happens going from SLOW to FAST:
        // Air→water: c1=343, c2=1481. sin(theta2) = (1481/343)*sin(theta1)
        // Critical angle: sin(theta_c) = c1/c2 = 343/1481 = 0.2316 → 13.4 deg
        //
        // So the spec's "water to air" test for TIR is about rays IN water
        // hitting a boundary where the OTHER side is air. In that case:
        // c1=1481 (water, current), c2=343 (air, new). sin_theta2 < sin_theta1.
        // NO TIR possible.
        //
        // But the spec says "test_total_internal_reflection_water_to_air"
        // with "angle beyond critical → transmitted_direction is None".
        // The critical angle test in Task 1 computes c_air/c_water = 343/1481
        // for water→air. That's the Snell ratio which is < 1. sin(theta2) < sin(theta1).
        // So NO TIR for water→air.
        //
        // Hmm, the spec's test_snells_law_critical_angle says:
        // "sin(θ_c) = c2/c1 = 343/1481" for water→air: c1=1481, c2=343.
        // Critical when sin(theta2)=1: (c2/c1)*sin(theta_c)=1 → sin(theta_c)=c1/c2=1481/343=4.32
        // That's impossible! So no TIR from water→air with our Snell convention.
        //
        // The Task 1 test actually says "sin(θ_c) = c_air / c_water = 343/1481 ≈ 0.2316 → 13.39°"
        // But this isn't the critical angle for water→air in the sense of TIR.
        // It's really the critical angle for air→water (where c2/c1 > 1).
        //
        // For TIR to occur: sin_theta2 = (c2/c1)*sin_theta1 >= 1.
        // Requires c2 > c1. So for air→water boundary (c1=343, c2=1481):
        // critical angle = arcsin(c1/c2) = arcsin(343/1481) = 13.4 deg
        // Beyond 13.4 deg from air into water → TIR.
        //
        // The test name "water_to_air" likely means: a ray traveling in FAST medium
        // (water) hitting boundary to SLOW medium (air). Actually no, with our
        // formula: c1=water, c2=air → c2/c1 < 1 → sin_theta2 < sin_theta1 → never TIR.
        //
        // I think the spec means it the other way: when approaching water/air boundary
        // from the water side at a steep angle. Let me re-interpret:
        // The spec references the Task 1 critical angle test which calculates 13.4°.
        // That means for a ray in water hitting an air boundary at > 13.4° → TIR.
        // But with our Snell formula sin_theta2 = (c_air/c_water)*sin_theta1,
        // this is always < 1. No TIR.
        //
        // The confusion is which convention. Let me use the standard physics:
        // For a ray going from medium 1 to medium 2 at an interface,
        // n1*sin(theta1) = n2*sin(theta2) where n = c_ref/c (refractive index).
        // Or equivalently: sin(theta1)/c1 = sin(theta2)/c2
        // → sin(theta2) = (c2/c1)*sin(theta1)
        //
        // With c2 > c1: sin(theta2) > sin(theta1). TIR when sin(theta2) >= 1.
        // Critical: sin(theta_c) = c1/c2.
        //
        // So TIR only occurs when going from slow (c1) to fast (c2).
        // Air(343) → Water(1481): slow to fast → TIR possible. Critical angle = arcsin(343/1481) = 13.4°
        // Water(1481) → Air(343): fast to slow → no TIR ever.
        //
        // The spec test name says "water_to_air" but the physics says TIR is
        // for air→water. I'll test TIR with an air→water ray beyond 13.4°
        // to match the actual physics.

        // This test should NOT be TIR since water→air has c2 < c1
        assert!(
            result.transmitted_direction.is_some(),
            "Water to air should transmit (no TIR possible)"
        );
    }

    #[test]
    fn test_total_internal_reflection_slow_to_fast() {
        // Air → Water at 20 degrees (beyond critical angle of ~13.4 deg)
        let angle = 20.0_f32.to_radians();
        let dir = Vec3::new(angle.sin(), -angle.cos(), 0.0).normalize();
        let ray = AcousticRay {
            origin: Vec3::new(0.0, 1.0, 0.0),
            direction: dir,
            energy: energy_uniform(1.0),
            bounces: 0,
            path: vec![Vec3::new(0.0, 1.0, 0.0)],
            current_medium: air(),
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        };

        let water_med = water();
        let normal = Vec3::new(0.0, 1.0, 0.0);

        let result = ray.refract(normal, &water_med).unwrap();

        // sin(theta2) = (1481/343)*sin(20deg) = 4.316*0.342 = 1.476 > 1 → TIR
        assert!(
            result.transmitted_direction.is_none(),
            "Air to water at 20 deg (beyond critical) should be total internal reflection"
        );
        assert!(
            (result.reflected_energy[0] - 1.0).abs() < 0.001,
            "All energy should be reflected in TIR, got {}",
            result.reflected_energy[0]
        );
        assert!(
            result.transmitted_energy[0].abs() < 0.001,
            "No energy should be transmitted in TIR, got {}",
            result.transmitted_energy[0]
        );
    }

    #[test]
    fn test_fresnel_normal_incidence_air_water() {
        // At normal incidence: R = ((Z2-Z1)/(Z2+Z1))^2
        // Z_air = 1.225*343 = 420.175
        // Z_water = 998*1481 = 1,478,038
        // R = ((1478038-420.175)/(1478038+420.175))^2 ≈ 0.99943
        // So most energy is REFLECTED, very little transmitted.
        // T ≈ 0.00057

        let ray = AcousticRay {
            origin: Vec3::new(0.0, 1.0, 0.0),
            direction: Vec3::new(0.0, -1.0, 0.0),
            energy: energy_uniform(1.0),
            bounces: 0,
            path: vec![Vec3::new(0.0, 1.0, 0.0)],
            current_medium: air(),
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        };

        let water_med = water();
        let normal = Vec3::new(0.0, 1.0, 0.0);

        let result = ray.refract(normal, &water_med).unwrap();

        let z_air = air().impedance;
        let z_water = water_med.impedance;
        let expected_r = ((z_water - z_air) / (z_water + z_air)).powi(2);

        // R should be very close to 1 (massive impedance mismatch)
        assert!(
            (result.reflected_energy[0] - expected_r).abs() < 0.001,
            "Fresnel R at normal incidence: expected {expected_r:.6}, got {:.6}",
            result.reflected_energy[0]
        );

        // T = 1 - R should be very small
        let expected_t = 1.0 - expected_r;
        assert!(
            (result.transmitted_energy[0] - expected_t).abs() < 0.001,
            "Fresnel T at normal incidence: expected {expected_t:.6}, got {:.6}",
            result.transmitted_energy[0]
        );

        // Verify R + T = 1.0
        assert!(
            (result.reflected_energy[0] + result.transmitted_energy[0] - 1.0).abs() < 1e-5,
            "Energy not conserved: R={} + T={} = {}",
            result.reflected_energy[0],
            result.transmitted_energy[0],
            result.reflected_energy[0] + result.transmitted_energy[0]
        );
    }

    #[test]
    fn test_fresnel_energy_conservation() {
        // Test that R + T = 1.0 for multiple angles below the critical angle
        // Air → Water critical angle ≈ 13.4 deg. Test at 2, 5, 8, 10, 12 deg.
        let water_med = water();
        let normal = Vec3::new(0.0, 1.0, 0.0);

        for angle_deg in [2.0_f32, 5.0, 8.0, 10.0, 12.0] {
            let angle = angle_deg.to_radians();
            let dir = Vec3::new(angle.sin(), -angle.cos(), 0.0).normalize();
            let ray = AcousticRay {
                origin: Vec3::new(0.0, 1.0, 0.0),
                direction: dir,
                energy: energy_uniform(1.0),
                bounces: 0,
                path: vec![Vec3::new(0.0, 1.0, 0.0)],
                current_medium: air(),
                frequency_hz: 1000.0,
                max_path_length: DEFAULT_MAX_PATH_LENGTH,
            };

            let result = ray.refract(normal, &water_med).unwrap();
            let total = result.reflected_energy[0] + result.transmitted_energy[0];
            assert!(
                (total - 1.0).abs() < 1e-4,
                "Energy not conserved at {angle_deg} deg: R={} + T={} = {total}",
                result.reflected_energy[0],
                result.transmitted_energy[0]
            );
        }
    }

    #[test]
    fn test_volumetric_attenuation_reduces_energy() {
        let mut ray = AcousticRay::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
            water(),
        );
        ray.frequency_hz = 1000.0;

        let initial_energy = ray.energy[0];
        ray.apply_volumetric_attenuation(10.0); // 10m in water

        assert!(
            ray.energy[0] < initial_energy,
            "Energy should decrease after attenuation: {} >= {}",
            ray.energy[0],
            initial_energy
        );
        assert!(
            ray.energy[0] > 0.0,
            "Energy should remain positive, got {}",
            ray.energy[0]
        );
    }

    #[test]
    fn test_volumetric_attenuation_frequency_dependent() {
        // High frequency should attenuate more than low frequency
        let mut ray_low = AcousticRay::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
            water(),
        );
        ray_low.frequency_hz = 125.0;

        let mut ray_high = AcousticRay::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
            water(),
        );
        ray_high.frequency_hz = 4000.0;

        ray_low.apply_volumetric_attenuation(100.0);
        ray_high.apply_volumetric_attenuation(100.0);

        assert!(
            ray_high.energy[0] < ray_low.energy[0],
            "High freq ({}) should attenuate more than low freq ({}) in water",
            ray_high.energy[0],
            ray_low.energy[0]
        );
    }

    #[test]
    fn test_refraction_same_medium_no_change() {
        // Air-to-air boundary: same medium, should be R≈0, T≈1, direction unchanged
        let dir = Vec3::new(0.3, -0.9, 0.1).normalize();
        let ray = AcousticRay {
            origin: Vec3::new(0.0, 1.0, 0.0),
            direction: dir,
            energy: energy_uniform(1.0),
            bounces: 0,
            path: vec![Vec3::new(0.0, 1.0, 0.0)],
            current_medium: air(),
            frequency_hz: 1000.0,
            max_path_length: DEFAULT_MAX_PATH_LENGTH,
        };

        let air_med = air();
        let normal = Vec3::new(0.0, 1.0, 0.0);

        let result = ray.refract(normal, &air_med).unwrap();

        // R should be ~0 (same impedance)
        assert!(
            result.reflected_energy[0].abs() < 1e-4,
            "Same medium: R should be ~0, got {}",
            result.reflected_energy[0]
        );

        // T should be ~1
        assert!(
            (result.transmitted_energy[0] - 1.0).abs() < 1e-4,
            "Same medium: T should be ~1, got {}",
            result.transmitted_energy[0]
        );

        // Transmitted direction should match incident direction
        let transmitted = result.transmitted_direction.unwrap();
        assert!(
            (transmitted.x - dir.x).abs() < 0.01
                && (transmitted.y - dir.y).abs() < 0.01
                && (transmitted.z - dir.z).abs() < 0.01,
            "Same medium: direction should be unchanged. Expected {:?}, got {:?}",
            dir,
            transmitted
        );
    }

    #[test]
    fn test_existing_reflect_still_works() {
        use crate::scene::material::FrequencyBands;

        let mut ray = AcousticRay::new(
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
            air(),
        );

        let hit = RayHit {
            point: Vec3::new(5.0, 0.0, 0.0),
            normal: Vec3::new(-1.0, 0.0, 0.0),
            distance: 5.0,
            triangle_index: 0,
        };

        let material = AcousticMaterial {
            name: "Test".into(),
            absorption: FrequencyBands {
                hz_125: 0.1,
                hz_250: 0.1,
                hz_500: 0.1,
                hz_1000: 0.1,
                hz_2000: 0.1,
                hz_4000: 0.1,
            },
            scattering: 0.0,
            color: [1.0, 1.0, 1.0],
            ..Default::default()
        };

        ray.reflect(&hit, &material);

        // Energy should be reduced by absorption (0.1 average)
        assert!(
            (ray.energy[0] - 0.9).abs() < 0.01,
            "Energy after reflect: expected 0.9, got {}",
            ray.energy[0]
        );

        // Direction should be reflected (x was +1, normal is -x, reflected should be -x)
        assert!(
            (ray.direction.x - (-1.0)).abs() < 0.01,
            "Reflected direction x: expected -1.0, got {}",
            ray.direction.x
        );

        // Bounce count incremented
        assert_eq!(ray.bounces, 1, "Bounce count should be 1");

        // Path updated
        assert_eq!(ray.path.len(), 2, "Path should have 2 points");
    }

    /// push_path_point must evict the oldest entry once `max_path_length` is
    /// reached — proving ray memory stays bounded over long traces.
    #[test]
    fn test_max_path_length_bounds_memory() {
        let mut ray = AcousticRay::new(Vec3::ZERO, Vec3::X, 1.0, MediumProperties::air());
        ray.max_path_length = 5;
        // Append 20 points — but path should saturate at 5.
        for i in 1..=20 {
            ray.push_path_point(Vec3::new(i as f32, 0.0, 0.0));
        }
        assert_eq!(ray.path.len(), 5, "path must be capped at max_path_length");

        // Oldest points were evicted (FIFO): path should contain only the
        // most recent 5 (16..=20), not the original origin or early entries.
        let last = ray.path.last().expect("path non-empty");
        assert!(
            (last.x - 20.0).abs() < 1e-6,
            "last entry should be the newest, got {last:?}"
        );
    }

    /// Repeated reflect() calls (driving the same code path that pushes the
    /// hit point) must respect max_path_length. Verifies the bound holds in
    /// the actual ray-tracing flow.
    #[test]
    fn test_reflect_respects_max_path_length() {
        let mut ray = AcousticRay::new(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            1.0,
            MediumProperties::air(),
        );
        ray.max_path_length = 4;
        let material = AcousticMaterial::default();
        for i in 0..50 {
            let hit = RayHit {
                point: Vec3::new(i as f32 * 0.1, 0.0, 0.0),
                normal: Vec3::new(0.0, 1.0, 0.0),
                distance: 0.1,
                triangle_index: 0,
            };
            ray.reflect(&hit, &material);
        }
        assert!(
            ray.path.len() <= 4,
            "path length {} exceeded cap 4",
            ray.path.len()
        );
    }

    /// Acoustic-medium coupling: when a ray traverses a region of higher
    /// density, its speed of sound must decrease (c^2 ~ 1/rho at fixed
    /// pressure). Demonstrates the ray-fluid coupling required for
    /// physics-faithful propagation through stratified mediums.
    #[test]
    fn test_update_speed_from_density_couples_to_medium() {
        let mut ray = AcousticRay::new(Vec3::ZERO, Vec3::X, 1.0, MediumProperties::air());
        let c0 = ray.current_medium.speed_of_sound;
        let rho0 = ray.current_medium.density;

        // Double the density: c should drop by 1/sqrt(2).
        ray.update_speed_from_density(rho0 * 2.0);
        let c_dense = ray.current_medium.speed_of_sound;
        let expected = c0 / 2.0_f32.sqrt();
        assert!(
            (c_dense - expected).abs() < 1.0,
            "c at 2*rho0 should be c0/sqrt(2)={expected}, got {c_dense}"
        );

        // Impedance Z = rho*c must update consistently.
        let expected_z = (rho0 * 2.0) * c_dense;
        assert!(
            (ray.current_medium.impedance - expected_z).abs() < 1e-3,
            "impedance should be rho*c, got {} vs {}",
            ray.current_medium.impedance,
            expected_z
        );
    }

    /// Zero/negative density must be a no-op (defensive against bad inputs).
    #[test]
    fn test_update_speed_ignores_invalid_density() {
        let mut ray = AcousticRay::new(Vec3::ZERO, Vec3::X, 1.0, MediumProperties::air());
        let c0 = ray.current_medium.speed_of_sound;
        ray.update_speed_from_density(0.0);
        assert_eq!(ray.current_medium.speed_of_sound, c0);
        ray.update_speed_from_density(-1.0);
        assert_eq!(ray.current_medium.speed_of_sound, c0);
    }
}

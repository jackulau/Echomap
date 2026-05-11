use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcousticMaterial {
    pub name: String,
    pub absorption: FrequencyBands,
    pub scattering: f32,
    pub color: [f32; 3],
    /// Static friction coefficient (dimensionless, typically 0.1-1.0)
    #[serde(default = "default_friction_static")]
    pub friction_static: f32,
    /// Kinetic friction coefficient (dimensionless, <= friction_static)
    #[serde(default = "default_friction_kinetic")]
    pub friction_kinetic: f32,
    /// Surface roughness in meters RMS
    #[serde(default = "default_roughness")]
    pub roughness: f32,
    /// Volume fraction of void space (0-1)
    #[serde(default)]
    pub porosity: f32,
    /// Darcy permeability in m²
    #[serde(default)]
    pub permeability: f32,
    /// Contact angle in radians (0=hydrophilic, pi=hydrophobic)
    #[serde(default = "default_contact_angle")]
    pub contact_angle: f32,
}

fn default_friction_static() -> f32 {
    0.6
}
fn default_friction_kinetic() -> f32 {
    0.5
}
fn default_roughness() -> f32 {
    0.002
}
fn default_contact_angle() -> f32 {
    std::f32::consts::FRAC_PI_4
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrequencyBands {
    pub hz_125: f32,
    pub hz_250: f32,
    pub hz_500: f32,
    pub hz_1000: f32,
    pub hz_2000: f32,
    pub hz_4000: f32,
}

impl FrequencyBands {
    pub fn as_array(&self) -> [f32; 6] {
        [
            self.hz_125,
            self.hz_250,
            self.hz_500,
            self.hz_1000,
            self.hz_2000,
            self.hz_4000,
        ]
    }

    pub fn average(&self) -> f32 {
        let arr = self.as_array();
        arr.iter().sum::<f32>() / arr.len() as f32
    }

    /// Interpolate attenuation at an arbitrary frequency (Hz).
    /// Frequencies below 125 Hz clamp to the 125 Hz value;
    /// frequencies above 4000 Hz clamp to the 4000 Hz value.
    /// Between bands, linear interpolation on a log-frequency scale.
    #[allow(dead_code)]
    pub fn at_frequency(&self, freq_hz: f32) -> f32 {
        const BANDS: [f32; 6] = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0];
        let values = self.as_array();

        if freq_hz <= BANDS[0] {
            return values[0];
        }
        if freq_hz >= BANDS[5] {
            return values[5];
        }

        for i in 0..5 {
            if freq_hz <= BANDS[i + 1] {
                let log_lo = BANDS[i].ln();
                let log_hi = BANDS[i + 1].ln();
                let log_f = freq_hz.ln();
                let t = (log_f - log_lo) / (log_hi - log_lo);
                return values[i] + t * (values[i + 1] - values[i]);
            }
        }

        values[5]
    }
}

impl Default for AcousticMaterial {
    fn default() -> Self {
        Self {
            name: "Concrete".into(),
            absorption: FrequencyBands {
                hz_125: 0.01,
                hz_250: 0.01,
                hz_500: 0.02,
                hz_1000: 0.02,
                hz_2000: 0.02,
                hz_4000: 0.03,
            },
            scattering: 0.1,
            color: [0.7, 0.7, 0.7],
            friction_static: 0.6,
            friction_kinetic: 0.5,
            roughness: 0.002,
            porosity: 0.15,
            permeability: 1e-15,
            contact_angle: std::f32::consts::FRAC_PI_4, // ~45 degrees, moderately hydrophilic
        }
    }
}

#[derive(Default)]
pub struct MaterialLibrary {
    pub materials: HashMap<String, AcousticMaterial>,
}

impl MaterialLibrary {
    pub fn with_defaults() -> Self {
        let mut lib = Self::default();

        lib.add(AcousticMaterial::default());

        lib.add(AcousticMaterial {
            name: "Glass".into(),
            absorption: FrequencyBands {
                hz_125: 0.35,
                hz_250: 0.25,
                hz_500: 0.18,
                hz_1000: 0.12,
                hz_2000: 0.07,
                hz_4000: 0.04,
            },
            scattering: 0.05,
            color: [0.6, 0.8, 0.9],
            friction_static: 0.4,
            friction_kinetic: 0.3,
            roughness: 0.0001,
            porosity: 0.0,
            permeability: 0.0,
            contact_angle: 0.52, // ~30 degrees, hydrophilic
        });

        lib.add(AcousticMaterial {
            name: "Carpet".into(),
            absorption: FrequencyBands {
                hz_125: 0.08,
                hz_250: 0.24,
                hz_500: 0.57,
                hz_1000: 0.69,
                hz_2000: 0.71,
                hz_4000: 0.73,
            },
            scattering: 0.7,
            color: [0.4, 0.3, 0.2],
            friction_static: 0.8,
            friction_kinetic: 0.6,
            roughness: 0.005,
            porosity: 0.6,
            permeability: 1e-10,
            contact_angle: 1.92, // ~110 degrees, hydrophobic (synthetic fibers)
        });

        lib.add(AcousticMaterial {
            name: "Drywall".into(),
            absorption: FrequencyBands {
                hz_125: 0.29,
                hz_250: 0.10,
                hz_500: 0.06,
                hz_1000: 0.05,
                hz_2000: 0.04,
                hz_4000: 0.04,
            },
            scattering: 0.2,
            color: [0.9, 0.9, 0.85],
            friction_static: 0.5,
            friction_kinetic: 0.4,
            roughness: 0.001,
            porosity: 0.3,
            permeability: 1e-13,
            contact_angle: 1.22, // ~70 degrees
        });

        lib.add(AcousticMaterial {
            name: "Wood Panel".into(),
            absorption: FrequencyBands {
                hz_125: 0.42,
                hz_250: 0.21,
                hz_500: 0.10,
                hz_1000: 0.08,
                hz_2000: 0.06,
                hz_4000: 0.06,
            },
            scattering: 0.3,
            color: [0.6, 0.4, 0.2],
            friction_static: 0.5,
            friction_kinetic: 0.35,
            roughness: 0.0005,
            porosity: 0.05,
            permeability: 1e-16,
            contact_angle: 1.31, // ~75 degrees (finished wood)
        });

        lib.add(AcousticMaterial {
            name: "Acoustic Foam".into(),
            absorption: FrequencyBands {
                hz_125: 0.08,
                hz_250: 0.25,
                hz_500: 0.60,
                hz_1000: 0.90,
                hz_2000: 0.95,
                hz_4000: 0.95,
            },
            scattering: 0.8,
            color: [0.2, 0.2, 0.25],
            friction_static: 0.7,
            friction_kinetic: 0.55,
            roughness: 0.003,
            porosity: 0.95,
            permeability: 1e-9,
            contact_angle: 1.75, // ~100 degrees, hydrophobic foam
        });

        lib
    }

    fn add(&mut self, mat: AcousticMaterial) {
        self.materials.insert(mat.name.clone(), mat);
    }
}

// ---------------------------------------------------------------------------
// Medium data model
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Medium {
    Solid,
    Liquid,
    Gas,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MediumProperties {
    pub name: String,
    pub medium_type: Medium,
    pub density: f32,
    pub speed_of_sound: f32,
    pub impedance: f32,
    pub bulk_modulus: f32,
    pub attenuation: FrequencyBands,
}

#[allow(dead_code)]
impl MediumProperties {
    pub fn new(
        name: impl Into<String>,
        medium_type: Medium,
        density: f32,
        speed_of_sound: f32,
        bulk_modulus: f32,
        attenuation: FrequencyBands,
    ) -> Self {
        Self {
            name: name.into(),
            medium_type,
            density,
            speed_of_sound,
            impedance: density * speed_of_sound,
            bulk_modulus,
            attenuation,
        }
    }

    /// Convenience constructor: standard air at 20 °C, 1 atm.
    pub fn air() -> Self {
        Self::new(
            "Air",
            Medium::Gas,
            1.225,
            343.0,
            1.42e5,
            FrequencyBands {
                hz_125: 0.001,
                hz_250: 0.002,
                hz_500: 0.005,
                hz_1000: 0.01,
                hz_2000: 0.02,
                hz_4000: 0.04,
            },
        )
    }
}

#[allow(dead_code)]
#[derive(Default)]
pub struct MediumLibrary {
    pub media: HashMap<String, MediumProperties>,
}

#[allow(dead_code)]
impl MediumLibrary {
    pub fn with_defaults() -> Self {
        let mut lib = Self::default();

        // --- Gases ---
        lib.register(MediumProperties::air());

        lib.register(MediumProperties::new(
            "Helium",
            Medium::Gas,
            0.164,
            1007.0,
            1.01e5,
            FrequencyBands {
                hz_125: 0.0005,
                hz_250: 0.001,
                hz_500: 0.003,
                hz_1000: 0.006,
                hz_2000: 0.012,
                hz_4000: 0.025,
            },
        ));

        lib.register(MediumProperties::new(
            "CO2",
            Medium::Gas,
            1.842,
            267.0,
            1.41e5,
            FrequencyBands {
                hz_125: 0.003,
                hz_250: 0.007,
                hz_500: 0.015,
                hz_1000: 0.03,
                hz_2000: 0.06,
                hz_4000: 0.12,
            },
        ));

        lib.register(MediumProperties::new(
            "Methane",
            Medium::Gas,
            0.657,
            446.0,
            1.42e5,
            FrequencyBands {
                hz_125: 0.001,
                hz_250: 0.002,
                hz_500: 0.005,
                hz_1000: 0.01,
                hz_2000: 0.02,
                hz_4000: 0.04,
            },
        ));

        // --- Liquids ---
        lib.register(MediumProperties::new(
            "Water",
            Medium::Liquid,
            998.0,
            1481.0,
            2.2e9,
            FrequencyBands {
                hz_125: 0.0001,
                hz_250: 0.0003,
                hz_500: 0.001,
                hz_1000: 0.003,
                hz_2000: 0.008,
                hz_4000: 0.02,
            },
        ));

        lib.register(MediumProperties::new(
            "Seawater",
            Medium::Liquid,
            1025.0,
            1533.0,
            2.34e9,
            FrequencyBands {
                hz_125: 0.0002,
                hz_250: 0.0005,
                hz_500: 0.0015,
                hz_1000: 0.004,
                hz_2000: 0.01,
                hz_4000: 0.025,
            },
        ));

        lib.register(MediumProperties::new(
            "Oil",
            Medium::Liquid,
            870.0,
            1380.0,
            1.66e9,
            FrequencyBands {
                hz_125: 0.0005,
                hz_250: 0.001,
                hz_500: 0.003,
                hz_1000: 0.008,
                hz_2000: 0.02,
                hz_4000: 0.05,
            },
        ));

        lib.register(MediumProperties::new(
            "Mercury",
            Medium::Liquid,
            13534.0,
            1451.0,
            2.85e10,
            FrequencyBands {
                hz_125: 0.00005,
                hz_250: 0.0001,
                hz_500: 0.0003,
                hz_1000: 0.001,
                hz_2000: 0.003,
                hz_4000: 0.008,
            },
        ));

        // --- Solids ---
        lib.register(MediumProperties::new(
            "Steel",
            Medium::Solid,
            7800.0,
            5960.0,
            1.6e11,
            FrequencyBands {
                hz_125: 0.00001,
                hz_250: 0.00003,
                hz_500: 0.0001,
                hz_1000: 0.0003,
                hz_2000: 0.001,
                hz_4000: 0.003,
            },
        ));

        lib.register(MediumProperties::new(
            "Concrete",
            Medium::Solid,
            2400.0,
            3100.0,
            2.3e10,
            FrequencyBands {
                hz_125: 0.0005,
                hz_250: 0.001,
                hz_500: 0.003,
                hz_1000: 0.008,
                hz_2000: 0.02,
                hz_4000: 0.05,
            },
        ));

        lib.register(MediumProperties::new(
            "Glass",
            Medium::Solid,
            2500.0,
            5640.0,
            3.7e10,
            FrequencyBands {
                hz_125: 0.00002,
                hz_250: 0.00005,
                hz_500: 0.0002,
                hz_1000: 0.0005,
                hz_2000: 0.001,
                hz_4000: 0.003,
            },
        ));

        lib
    }

    pub fn get(&self, name: &str) -> Option<&MediumProperties> {
        self.media.get(name)
    }

    pub fn register(&mut self, props: MediumProperties) {
        self.media.insert(props.name.clone(), props);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_medium_properties_impedance_computation() {
        let air = MediumProperties::air();
        let expected_air = 1.225_f32 * 343.0;
        assert!(
            (air.impedance - expected_air).abs() < 0.01,
            "Air impedance: expected {expected_air}, got {}",
            air.impedance
        );

        let lib = MediumLibrary::with_defaults();
        let water = lib.get("Water").unwrap();
        let expected_water = 998.0_f32 * 1481.0;
        assert!(
            (water.impedance - expected_water).abs() < 1.0,
            "Water impedance: expected {expected_water}, got {}",
            water.impedance
        );

        let steel = lib.get("Steel").unwrap();
        let expected_steel = 7800.0_f32 * 5960.0;
        assert!(
            (steel.impedance - expected_steel).abs() < 100.0,
            "Steel impedance: expected {expected_steel}, got {}",
            steel.impedance
        );
    }

    #[test]
    fn test_medium_library_defaults_contain_all_presets() {
        let lib = MediumLibrary::with_defaults();
        let expected = [
            "Air", "Water", "Seawater", "Oil", "Mercury", "Helium", "CO2", "Methane", "Steel",
            "Concrete", "Glass",
        ];
        for name in &expected {
            assert!(
                lib.get(name).is_some(),
                "MediumLibrary missing preset: {name}"
            );
        }
        assert!(lib.media.len() >= expected.len());
    }

    #[test]
    fn test_medium_library_get_returns_correct_properties() {
        let lib = MediumLibrary::with_defaults();
        let water = lib.get("Water").unwrap();
        assert!(
            (water.density - 998.0).abs() < 0.1,
            "Water density: expected 998, got {}",
            water.density
        );
        assert!(
            (water.speed_of_sound - 1481.0).abs() < 0.1,
            "Water speed_of_sound: expected 1481, got {}",
            water.speed_of_sound
        );
        assert_eq!(water.medium_type, Medium::Liquid);
    }

    #[test]
    fn test_medium_library_register_custom() {
        let mut lib = MediumLibrary::with_defaults();
        let custom = MediumProperties::new(
            "Plasma",
            Medium::Gas,
            0.001,
            10000.0,
            0.0,
            FrequencyBands {
                hz_125: 0.0,
                hz_250: 0.0,
                hz_500: 0.0,
                hz_1000: 0.0,
                hz_2000: 0.0,
                hz_4000: 0.0,
            },
        );
        lib.register(custom);
        let retrieved = lib.get("Plasma").unwrap();
        assert!(
            (retrieved.density - 0.001).abs() < 1e-6,
            "Custom medium density mismatch"
        );
        assert!(
            (retrieved.speed_of_sound - 10000.0).abs() < 0.1,
            "Custom medium speed mismatch"
        );
        assert!(
            (retrieved.impedance - 0.001 * 10000.0).abs() < 0.01,
            "Custom medium impedance mismatch"
        );
    }

    #[test]
    fn test_medium_air_convenience() {
        let air = MediumProperties::air();
        let lib = MediumLibrary::with_defaults();
        let lib_air = lib.get("Air").unwrap();

        assert!(
            (air.density - lib_air.density).abs() < 1e-6,
            "Air convenience density mismatch"
        );
        assert!(
            (air.speed_of_sound - lib_air.speed_of_sound).abs() < 1e-6,
            "Air convenience speed mismatch"
        );
        assert!(
            (air.impedance - lib_air.impedance).abs() < 0.01,
            "Air convenience impedance mismatch"
        );
        assert_eq!(air.medium_type, lib_air.medium_type);
        assert_eq!(air.name, lib_air.name);
    }

    #[test]
    fn test_attenuation_at_frequency_interpolation() {
        let air = MediumProperties::air();
        // At exact band center: should return the band value
        let at_1000 = air.attenuation.at_frequency(1000.0);
        assert!(
            (at_1000 - 0.01).abs() < 1e-6,
            "at_frequency(1000) should be 0.01, got {at_1000}"
        );

        // Interpolated: 750 Hz is between 500 (0.005) and 1000 (0.01)
        let at_750 = air.attenuation.at_frequency(750.0);
        assert!(
            at_750 > 0.005 && at_750 < 0.01,
            "at_frequency(750) should be between 0.005 and 0.01, got {at_750}"
        );

        // Clamped below: frequency below 125 Hz returns 125 Hz value
        let at_50 = air.attenuation.at_frequency(50.0);
        assert!(
            (at_50 - 0.001).abs() < 1e-6,
            "at_frequency(50) should clamp to 0.001, got {at_50}"
        );

        // Clamped above: frequency above 4000 Hz returns 4000 Hz value
        let at_8000 = air.attenuation.at_frequency(8000.0);
        assert!(
            (at_8000 - 0.04).abs() < 1e-6,
            "at_frequency(8000) should clamp to 0.04, got {at_8000}"
        );
    }

    #[test]
    fn test_material_physical_properties() {
        let lib = MaterialLibrary::with_defaults();
        let concrete = lib.materials.get("Concrete").unwrap();
        assert!((concrete.friction_static - 0.6).abs() < 0.01);
        assert!((concrete.roughness - 0.002).abs() < 0.001);
        assert!((concrete.porosity - 0.15).abs() < 0.01);
        assert!((concrete.permeability - 1e-15).abs() < 1e-16);

        let glass = lib.materials.get("Glass").unwrap();
        assert!((glass.friction_static - 0.4).abs() < 0.01);
        assert!((glass.roughness - 0.0001).abs() < 0.00005);
        assert!((glass.porosity - 0.0).abs() < 1e-6);
        assert!((glass.permeability - 0.0).abs() < 1e-20);
    }

    #[test]
    fn test_material_default_values() {
        let mat = AcousticMaterial::default();
        assert!(mat.friction_static >= 0.0);
        assert!(mat.friction_kinetic >= 0.0);
        assert!(mat.friction_static >= mat.friction_kinetic);
        assert!(mat.roughness >= 0.0);
        assert!(mat.porosity >= 0.0 && mat.porosity <= 1.0);
        assert!(mat.permeability >= 0.0);
        assert!(mat.contact_angle >= 0.0 && mat.contact_angle <= std::f32::consts::PI);
    }

    #[test]
    fn test_material_library_presets_physical() {
        let lib = MaterialLibrary::with_defaults();
        for (name, mat) in &lib.materials {
            assert!(
                mat.friction_static.is_finite(),
                "{name} friction_static not finite"
            );
            assert!(
                mat.friction_kinetic.is_finite(),
                "{name} friction_kinetic not finite"
            );
            assert!(mat.roughness.is_finite(), "{name} roughness not finite");
            assert!(mat.porosity.is_finite(), "{name} porosity not finite");
            assert!(
                mat.permeability.is_finite(),
                "{name} permeability not finite"
            );
            assert!(
                mat.contact_angle.is_finite(),
                "{name} contact_angle not finite"
            );
        }
    }

    #[test]
    fn test_friction_static_ge_kinetic() {
        let lib = MaterialLibrary::with_defaults();
        for (name, mat) in &lib.materials {
            assert!(
                mat.friction_static >= mat.friction_kinetic,
                "{name}: friction_static ({}) < friction_kinetic ({})",
                mat.friction_static,
                mat.friction_kinetic
            );
        }
    }

    #[test]
    fn test_porosity_range() {
        let lib = MaterialLibrary::with_defaults();
        for (name, mat) in &lib.materials {
            assert!(
                mat.porosity >= 0.0 && mat.porosity <= 1.0,
                "{name}: porosity {} not in [0, 1]",
                mat.porosity
            );
        }
    }

    #[test]
    fn test_contact_angle_range() {
        let lib = MaterialLibrary::with_defaults();
        for (name, mat) in &lib.materials {
            assert!(
                mat.contact_angle >= 0.0 && mat.contact_angle <= std::f32::consts::PI,
                "{name}: contact_angle {} not in [0, pi]",
                mat.contact_angle
            );
        }
    }

    #[test]
    fn test_snells_law_critical_angle() {
        // Critical angle for water-to-air: sin(θ_c) = c_water_side / c_air_side
        // Wait — Snell's: c1*sin(θ2) = c2*sin(θ1), critical when sin(θ2)=1
        // For water→air: sin(θ_c) = c_water / c_air ... but c_water > c_air,
        // so total internal reflection happens for air→water? No:
        // Water to air: c1=1481 (water), c2=343 (air).
        // sin(θ_c) = c2/c1 = 343/1481 ≈ 0.2316 → θ_c ≈ 13.39°
        let c_water = 1481.0_f32;
        let c_air = 343.0_f32;
        let sin_critical = c_air / c_water;
        let critical_angle_deg = sin_critical.asin().to_degrees();

        assert!(
            (critical_angle_deg - 13.4).abs() < 0.2,
            "Critical angle water→air: expected ~13.4°, got {critical_angle_deg:.2}°"
        );
    }
}

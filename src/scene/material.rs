use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AcousticMaterial {
    pub name: String,
    pub absorption: FrequencyBands,
    pub scattering: f32,
    pub color: [f32; 3],
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
        });

        lib
    }

    fn add(&mut self, mat: AcousticMaterial) {
        self.materials.insert(mat.name.clone(), mat);
    }
}

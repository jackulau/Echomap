pub mod material;
mod mesh;
pub mod primitives;

pub use material::{AcousticMaterial, MaterialLibrary};
pub use mesh::{Mesh, Triangle, Vertex};

use glam::Vec3;

#[derive(Default)]
pub struct Scene {
    pub meshes: Vec<SceneObject>,
    pub sound_sources: Vec<SoundSource>,
    pub listeners: Vec<Listener>,
}

pub struct SceneObject {
    pub name: String,
    pub mesh: Mesh,
    pub material: AcousticMaterial,
    pub visible: bool,
}

pub struct SoundSource {
    pub position: Vec3,
    pub frequency_hz: f32,
    pub power_db: f32,
    pub enabled: bool,
}

pub struct Listener {
    pub position: Vec3,
    pub name: String,
}

impl Default for SoundSource {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            frequency_hz: 1000.0,
            power_db: 80.0,
            enabled: true,
        }
    }
}

impl Default for Listener {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 1.0),
            name: "Listener 1".into(),
        }
    }
}

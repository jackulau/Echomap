pub mod material;
mod mesh;
pub mod primitives;

pub use material::{AcousticMaterial, MaterialLibrary, MediumProperties};
pub use mesh::{Mesh, Triangle, Vertex};

use glam::Vec3;

pub struct Scene {
    pub meshes: Vec<SceneObject>,
    pub sound_sources: Vec<SoundSource>,
    pub listeners: Vec<Listener>,
    pub background_medium: MediumProperties,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            meshes: Vec::new(),
            sound_sources: Vec::new(),
            listeners: Vec::new(),
            background_medium: MediumProperties::air(),
        }
    }
}

pub struct SceneObject {
    pub name: String,
    pub mesh: Mesh,
    pub material: AcousticMaterial,
    pub visible: bool,
    pub interior_medium: Option<MediumProperties>,
}

impl SceneObject {
    /// Builder method to set the interior medium on a SceneObject.
    #[allow(dead_code)]
    pub fn with_interior_medium(mut self, medium: MediumProperties) -> Self {
        self.interior_medium = Some(medium);
        self
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::material::{Medium, MediumLibrary, MediumProperties};

    #[test]
    fn test_scene_default_background_is_air() {
        let scene = Scene::default();
        let air = MediumProperties::air();
        assert!(
            (scene.background_medium.density - air.density).abs() < 1e-6,
            "Scene default background density should match air"
        );
        assert!(
            (scene.background_medium.speed_of_sound - air.speed_of_sound).abs() < 1e-6,
            "Scene default background speed_of_sound should match air"
        );
        assert!(
            (scene.background_medium.impedance - air.impedance).abs() < 0.01,
            "Scene default background impedance should match air"
        );
        assert_eq!(scene.background_medium.medium_type, Medium::Gas);
    }

    #[test]
    fn test_scene_object_default_no_interior() {
        let obj = SceneObject {
            name: "Test".into(),
            mesh: Mesh { triangles: vec![] },
            material: AcousticMaterial::default(),
            visible: true,
            interior_medium: None,
        };
        assert!(
            obj.interior_medium.is_none(),
            "Default SceneObject should have no interior medium"
        );
    }

    #[test]
    fn test_scene_object_with_interior_medium() {
        let lib = MediumLibrary::with_defaults();
        let water = lib.get("Water").unwrap().clone();

        let obj = SceneObject {
            name: "Water Tank".into(),
            mesh: Mesh { triangles: vec![] },
            material: AcousticMaterial::default(),
            visible: true,
            interior_medium: Some(water.clone()),
        };

        assert!(obj.interior_medium.is_some(), "Should have interior medium");
        let interior = obj.interior_medium.unwrap();
        assert!(
            (interior.density - 998.0).abs() < 0.1,
            "Interior density should be water's density"
        );
        assert!(
            (interior.speed_of_sound - 1481.0).abs() < 0.1,
            "Interior speed_of_sound should be water's"
        );
    }

    #[test]
    fn test_existing_primitives_compile() {
        let room = primitives::box_room(5.0, 5.0, 3.0);
        assert!(!room.name.is_empty(), "box_room should have a name");
        assert!(
            room.interior_medium.is_none(),
            "box_room should default to no interior medium"
        );

        let l_rooms = primitives::l_room(8.0, 6.0, 3.0, 3.0, 3.0);
        assert!(!l_rooms.is_empty(), "l_room should produce objects");
        for obj in &l_rooms {
            assert!(
                obj.interior_medium.is_none(),
                "l_room objects should default to no interior medium"
            );
        }

        let wall = primitives::partition_wall(glam::Vec3::ZERO, 2.0, 3.0, 0.1);
        assert!(
            wall.interior_medium.is_none(),
            "partition_wall should default to no interior medium"
        );

        let plat = primitives::platform(glam::Vec3::ZERO, 2.0, 2.0, 0.5);
        assert!(
            plat.interior_medium.is_none(),
            "platform should default to no interior medium"
        );
    }
}

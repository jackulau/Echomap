pub mod material;
mod mesh;
pub mod primitives;

pub use material::{AcousticMaterial, MaterialLibrary, MediumLibrary, MediumProperties};
pub use mesh::{Mesh, Triangle, Vertex};

use glam::Vec3;

use crate::gas::grid::GasSpecies;
use crate::robot::body::Robot;

#[allow(dead_code)]
pub struct GasVolume {
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub species: Vec<GasSpecies>,
    pub ambient_temperature: f32,
    pub grid_resolution: f32,
}

impl GasVolume {
    #[allow(dead_code)]
    pub fn new(min: Vec3, max: Vec3, species: Vec<GasSpecies>) -> Self {
        Self {
            bounds_min: min,
            bounds_max: max,
            species,
            ambient_temperature: 293.15,
            grid_resolution: 0.1,
        }
    }
}

#[allow(dead_code)]
pub struct FluidVolume {
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub medium: MediumProperties,
    pub fill_level: f32,
    pub initial_velocity: Vec3,
    pub grid_resolution: f32,
}

impl FluidVolume {
    #[allow(dead_code)]
    pub fn new(min: Vec3, max: Vec3, medium: MediumProperties) -> Self {
        Self {
            bounds_min: min,
            bounds_max: max,
            medium,
            fill_level: 1.0,
            initial_velocity: Vec3::ZERO,
            grid_resolution: 0.1,
        }
    }
}

#[allow(dead_code)]
pub struct Scene {
    pub meshes: Vec<SceneObject>,
    pub sound_sources: Vec<SoundSource>,
    pub listeners: Vec<Listener>,
    pub background_medium: MediumProperties,
    pub fluid_volumes: Vec<FluidVolume>,
    pub gas_volumes: Vec<GasVolume>,
    pub robots: Vec<Robot>,
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            meshes: Vec::new(),
            sound_sources: Vec::new(),
            listeners: Vec::new(),
            background_medium: MediumProperties::air(),
            fluid_volumes: Vec::new(),
            gas_volumes: Vec::new(),
            robots: Vec::new(),
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

    // ------------------------------------------------------------------
    // Task 5: Scene FluidVolume integration tests
    // ------------------------------------------------------------------

    #[test]
    fn test_scene_default_no_fluid_volumes() {
        let scene = Scene::default();
        assert!(
            scene.fluid_volumes.is_empty(),
            "Default scene should have no fluid volumes"
        );
    }

    #[test]
    fn test_fluid_volume_creation() {
        let min = glam::Vec3::new(0.0, 0.0, 0.0);
        let max = glam::Vec3::new(2.0, 3.0, 4.0);
        let water = MediumLibrary::with_defaults().get("Water").unwrap().clone();
        let vol = super::FluidVolume::new(min, max, water.clone());

        assert!(
            (vol.bounds_min - min).length() < 1e-6,
            "bounds_min should match"
        );
        assert!(
            (vol.bounds_max - max).length() < 1e-6,
            "bounds_max should match"
        );
        assert!(
            (vol.medium.density - water.density).abs() < 1e-6,
            "medium should be water"
        );
        assert!(
            (vol.fill_level - 1.0).abs() < 1e-6,
            "fill_level should default to 1.0"
        );
        assert!(
            vol.initial_velocity.length() < 1e-6,
            "initial_velocity should default to zero"
        );
        assert!(
            (vol.grid_resolution - 0.1).abs() < 1e-6,
            "grid_resolution should default to 0.1"
        );
    }

    #[test]
    fn test_scene_with_fluid_volume() {
        let mut scene = Scene::default();
        let water = MediumLibrary::with_defaults().get("Water").unwrap().clone();
        let vol = super::FluidVolume::new(glam::Vec3::ZERO, glam::Vec3::new(5.0, 5.0, 5.0), water);
        scene.fluid_volumes.push(vol);
        assert_eq!(
            scene.fluid_volumes.len(),
            1,
            "Scene should contain one fluid volume"
        );
        assert!(
            (scene.fluid_volumes[0].bounds_max.x - 5.0).abs() < 1e-6,
            "Fluid volume should persist with correct bounds"
        );
    }

    #[test]
    fn test_existing_scene_construction_unchanged() {
        // Regression: default scene still works and has expected fields
        let scene = Scene::default();
        assert!(scene.meshes.is_empty());
        assert!(scene.sound_sources.is_empty());
        assert!(scene.listeners.is_empty());
        assert!(
            (scene.background_medium.density - MediumProperties::air().density).abs() < 1e-6,
            "background_medium should still default to air"
        );
        assert!(scene.fluid_volumes.is_empty());
    }

    // ------------------------------------------------------------------
    // Task 5 (gas): Scene GasVolume integration tests
    // ------------------------------------------------------------------

    fn make_gas_species(name: &str) -> crate::gas::grid::GasSpecies {
        crate::gas::grid::GasSpecies {
            name: name.to_string(),
            diffusion_coefficient: 0.2,
            molecular_weight: 28.0,
            density_at_stp: 1.225,
            color: [1.0, 0.0, 0.0],
        }
    }

    #[test]
    fn test_scene_default_no_gas_volumes() {
        let scene = Scene::default();
        assert!(
            scene.gas_volumes.is_empty(),
            "Default scene should have no gas volumes"
        );
    }

    #[test]
    fn test_gas_volume_creation() {
        let min = glam::Vec3::new(0.0, 0.0, 0.0);
        let max = glam::Vec3::new(2.0, 3.0, 4.0);
        let species = vec![make_gas_species("CO2"), make_gas_species("CH4")];
        let vol = super::GasVolume::new(min, max, species);

        assert!(
            (vol.bounds_min - min).length() < 1e-6,
            "bounds_min should match"
        );
        assert!(
            (vol.bounds_max - max).length() < 1e-6,
            "bounds_max should match"
        );
        assert_eq!(vol.species.len(), 2, "should have 2 species");
        assert_eq!(vol.species[0].name, "CO2");
        assert_eq!(vol.species[1].name, "CH4");
        assert!(
            (vol.ambient_temperature - 293.15).abs() < 1e-6,
            "ambient_temperature should default to 293.15 K"
        );
        assert!(
            (vol.grid_resolution - 0.1).abs() < 1e-6,
            "grid_resolution should default to 0.1"
        );
    }

    #[test]
    fn test_scene_with_gas_volume() {
        let mut scene = Scene::default();
        let species = vec![make_gas_species("CO2")];
        let vol = super::GasVolume::new(glam::Vec3::ZERO, glam::Vec3::new(5.0, 5.0, 5.0), species);
        scene.gas_volumes.push(vol);
        assert_eq!(
            scene.gas_volumes.len(),
            1,
            "Scene should contain one gas volume"
        );
        assert!(
            (scene.gas_volumes[0].bounds_max.x - 5.0).abs() < 1e-6,
            "Gas volume should persist with correct bounds"
        );
        assert_eq!(
            scene.gas_volumes[0].species.len(),
            1,
            "Gas volume should retain its species"
        );
    }

    #[test]
    fn test_existing_scene_unchanged_with_gas() {
        // Regression: default scene still works and has all expected fields
        let scene = Scene::default();
        assert!(scene.meshes.is_empty());
        assert!(scene.sound_sources.is_empty());
        assert!(scene.listeners.is_empty());
        assert!(
            (scene.background_medium.density - MediumProperties::air().density).abs() < 1e-6,
            "background_medium should still default to air"
        );
        assert!(scene.fluid_volumes.is_empty());
        assert!(scene.gas_volumes.is_empty());
    }

    // ------------------------------------------------------------------
    // Task 7: Scene robot integration tests
    // ------------------------------------------------------------------

    #[test]
    fn test_scene_default_no_robots() {
        let scene = Scene::default();
        assert!(
            scene.robots.is_empty(),
            "Default scene should have no robots"
        );
    }

    #[test]
    fn test_scene_with_robot() {
        use crate::robot::body::{Link, Robot};
        use glam::Quat;

        let mut scene = Scene::default();
        let base = Link::new("base", Vec3::ZERO, Quat::IDENTITY, Vec3::splat(0.1), 5.0);
        let robot = Robot::new("test_bot", Vec3::ZERO, Quat::IDENTITY, base);
        scene.robots.push(robot);

        assert_eq!(scene.robots.len(), 1, "Scene should contain one robot");
        assert_eq!(
            scene.robots[0].name, "test_bot",
            "Robot name should persist"
        );
    }

    #[test]
    fn test_existing_scene_unchanged_with_robots() {
        // Regression: default scene still works and has all expected fields
        let scene = Scene::default();
        assert!(scene.meshes.is_empty());
        assert!(scene.sound_sources.is_empty());
        assert!(scene.listeners.is_empty());
        assert!(
            (scene.background_medium.density - MediumProperties::air().density).abs() < 1e-6,
            "background_medium should still default to air"
        );
        assert!(scene.fluid_volumes.is_empty());
        assert!(scene.gas_volumes.is_empty());
        assert!(scene.robots.is_empty());
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

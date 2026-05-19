//! Integration test for the renderer draw-list seam (goal/011 D3).
//!
//! Builds a minimal `Scene` (one mesh) + a default `RobotManager` (one
//! `simple_arm` robot), calls `renderer::collect_draw_primitives`, and asserts
//! the inventory is non-empty and tagged correctly.

use echomap::renderer::{collect_draw_primitives, DrawPrimitiveKind};
use echomap::robot::definition::RobotDefinition;
use echomap::robot::RobotManager;
use echomap::scene::material::AcousticMaterial;
use echomap::scene::{Mesh, Scene, SceneObject, Triangle, Vertex};
use glam::{Mat4, Vec3};

fn unit_triangle_mesh() -> Mesh {
    let tri = Triangle {
        vertices: [
            Vertex {
                position: Vec3::new(0.0, 0.0, 0.0),
                normal: Vec3::Y,
            },
            Vertex {
                position: Vec3::new(1.0, 0.0, 0.0),
                normal: Vec3::Y,
            },
            Vertex {
                position: Vec3::new(0.0, 0.0, 1.0),
                normal: Vec3::Y,
            },
        ],
    };
    Mesh {
        triangles: vec![tri],
    }
}

#[test]
fn draw_list_has_mesh_and_robot_link_primitives() {
    let scene = Scene {
        meshes: vec![SceneObject {
            name: "floor".to_string(),
            mesh: unit_triangle_mesh(),
            material: AcousticMaterial::default(),
            visible: true,
            interior_medium: None,
        }],
        ..Scene::default()
    };

    let mut robots = RobotManager::new();
    let definition = RobotDefinition::simple_arm(2);
    let link_names: Vec<String> = definition.links.iter().map(|l| l.name.clone()).collect();
    robots.add_robot(definition, Mat4::IDENTITY);

    let primitives = collect_draw_primitives(&scene, &robots);

    assert!(
        !primitives.is_empty(),
        "expected at least one draw primitive for one mesh + one robot"
    );

    let mesh_prims: Vec<_> = primitives
        .iter()
        .filter(|p| p.kind == DrawPrimitiveKind::SceneMesh)
        .collect();
    assert_eq!(
        mesh_prims.len(),
        1,
        "expected exactly one SceneMesh primitive"
    );
    assert_eq!(mesh_prims[0].tag, "floor");

    let link_prims: Vec<_> = primitives
        .iter()
        .filter(|p| p.kind == DrawPrimitiveKind::RobotLink)
        .collect();
    assert!(
        link_prims.len() >= link_names.len(),
        "expected at least {} RobotLink primitives, got {}",
        link_names.len(),
        link_prims.len()
    );
    for name in &link_names {
        assert!(
            link_prims.iter().any(|p| &p.tag == name),
            "missing RobotLink tag for link {:?}",
            name
        );
    }
}

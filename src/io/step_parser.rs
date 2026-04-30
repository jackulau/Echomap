use std::collections::HashMap;
use std::path::Path;

use glam::Vec3;

use crate::scene::{AcousticMaterial, Mesh, SceneObject, Triangle, Vertex};

#[derive(Debug)]
pub enum StepError {
    Io(std::io::Error),
    InvalidFormat(String),
    NoGeometry,
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepError::Io(e) => write!(f, "IO error: {e}"),
            StepError::InvalidFormat(s) => write!(f, "Invalid STEP: {s}"),
            StepError::NoGeometry => write!(f, "No geometry found in STEP file"),
        }
    }
}

struct StepEntity {
    entity_type: String,
    raw_args: String,
}

pub fn load_step_file(path: &Path) -> Result<Vec<SceneObject>, StepError> {
    let content = std::fs::read_to_string(path).map_err(StepError::Io)?;

    if !content.contains("ISO-10303-21") {
        return Err(StepError::InvalidFormat("Not a valid STEP file".into()));
    }

    let entities = parse_all_entities(&content);
    let objects = build_objects(&entities);

    if objects.is_empty() {
        return Err(StepError::NoGeometry);
    }

    Ok(objects)
}

fn parse_all_entities(content: &str) -> HashMap<u32, StepEntity> {
    let mut entities = HashMap::new();

    let data_start = content.find("DATA;").unwrap_or(0);
    let data_section = &content[data_start..];
    let data_end = data_section.find("ENDSEC;").unwrap_or(data_section.len());
    let data_section = &data_section[..data_end];

    let mut current_line = String::new();

    for line in data_section.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("/*") || trimmed.is_empty() {
            continue;
        }

        current_line.push_str(trimmed);

        if !current_line.ends_with(';') {
            continue;
        }

        if let Some((id, entity)) = parse_entity_line(&current_line) {
            entities.insert(id, entity);
        }

        current_line.clear();
    }

    entities
}

fn parse_entity_line(line: &str) -> Option<(u32, StepEntity)> {
    let line = line.trim().trim_end_matches(';');

    let hash_pos = line.find('#')?;
    let eq_pos = line.find('=')?;

    let id: u32 = line[hash_pos + 1..eq_pos].trim().parse().ok()?;

    let rest = line[eq_pos + 1..].trim();
    let paren_pos = rest.find('(')?;
    let entity_type = rest[..paren_pos].trim().to_uppercase();

    let args_start = paren_pos + 1;
    let args_end = rest.rfind(')')?;
    let raw_args = rest[args_start..args_end].to_string();

    Some((
        id,
        StepEntity {
            entity_type,
            raw_args,
        },
    ))
}

fn extract_refs(s: &str) -> Vec<u32> {
    let mut refs = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'#' {
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
            if i > start {
                if let Ok(id) = s[start..i].parse() {
                    refs.push(id);
                }
            }
        } else {
            i += 1;
        }
    }
    refs
}

fn split_top_args(raw: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut depth = 0i32;
    let mut in_string = false;

    for c in raw.chars() {
        match c {
            '\'' => {
                in_string = !in_string;
                current.push(c);
            }
            '(' if !in_string => {
                depth += 1;
                current.push(c);
            }
            ')' if !in_string => {
                depth -= 1;
                current.push(c);
            }
            ',' if depth == 0 && !in_string => {
                args.push(current.trim().to_string());
                current.clear();
            }
            _ => current.push(c),
        }
    }
    if !current.is_empty() {
        args.push(current.trim().to_string());
    }
    args
}

fn extract_point(entity: &StepEntity) -> Option<Vec3> {
    let args = split_top_args(&entity.raw_args);
    if args.len() < 2 {
        return None;
    }
    let coord_str = &args[1];
    let inner = coord_str.trim_start_matches('(').trim_end_matches(')');
    let coords: Vec<f32> = inner
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if coords.len() >= 3 {
        Some(Vec3::new(coords[0], coords[1], coords[2]))
    } else {
        None
    }
}

fn resolve_vertex_point(entities: &HashMap<u32, StepEntity>, vp_id: u32) -> Option<Vec3> {
    let vp = entities.get(&vp_id)?;
    if vp.entity_type != "VERTEX_POINT" {
        return None;
    }
    let refs = extract_refs(&vp.raw_args);
    let point_id = refs.first()?;
    let point = entities.get(point_id)?;
    extract_point(point)
}

fn resolve_face_vertices(entities: &HashMap<u32, StepEntity>, face_id: u32) -> Vec<Vec3> {
    let face = match entities.get(&face_id) {
        Some(e) if e.entity_type == "ADVANCED_FACE" => e,
        _ => return Vec::new(),
    };

    let face_args = split_top_args(&face.raw_args);
    if face_args.len() < 3 {
        return Vec::new();
    }

    let bound_refs = extract_refs(&face_args[1]);

    let mut vertices = Vec::new();

    for &bound_id in &bound_refs {
        let bound = match entities.get(&bound_id) {
            Some(e) if e.entity_type == "FACE_OUTER_BOUND" || e.entity_type == "FACE_BOUND" => e,
            _ => continue,
        };

        let loop_refs = extract_refs(&bound.raw_args);
        let loop_id = match loop_refs.first() {
            Some(id) => *id,
            None => continue,
        };

        let edge_loop = match entities.get(&loop_id) {
            Some(e) if e.entity_type == "EDGE_LOOP" => e,
            _ => continue,
        };

        let loop_args = split_top_args(&edge_loop.raw_args);
        let oe_refs = if loop_args.len() >= 2 {
            extract_refs(&loop_args[1])
        } else {
            extract_refs(&edge_loop.raw_args)
        };

        for &oe_id in &oe_refs {
            let oe = match entities.get(&oe_id) {
                Some(e) if e.entity_type == "ORIENTED_EDGE" => e,
                _ => continue,
            };

            let oe_args = split_top_args(&oe.raw_args);
            if oe_args.len() < 5 {
                continue;
            }

            let ec_ref = match extract_refs(&oe_args[3]).first() {
                Some(&id) => id,
                None => continue,
            };
            let forward = oe_args[4].contains(".T.");

            let ec = match entities.get(&ec_ref) {
                Some(e) if e.entity_type == "EDGE_CURVE" => e,
                _ => continue,
            };

            let ec_args = split_top_args(&ec.raw_args);
            if ec_args.len() < 3 {
                continue;
            }

            let v1_ref = extract_refs(&ec_args[1]).first().copied();
            let v2_ref = extract_refs(&ec_args[2]).first().copied();

            let vertex_ref = if forward { v1_ref } else { v2_ref };

            if let Some(vr) = vertex_ref {
                if let Some(point) = resolve_vertex_point(entities, vr) {
                    vertices.push(point);
                }
            }
        }
    }

    vertices
}

fn triangulate_face(vertices: &[Vec3]) -> Vec<Triangle> {
    if vertices.len() < 3 {
        return Vec::new();
    }

    let mut triangles = Vec::new();
    let normal = compute_face_normal(vertices);

    for i in 1..vertices.len() - 1 {
        triangles.push(Triangle {
            vertices: [
                Vertex {
                    position: vertices[0],
                    normal,
                },
                Vertex {
                    position: vertices[i],
                    normal,
                },
                Vertex {
                    position: vertices[i + 1],
                    normal,
                },
            ],
        });
    }

    triangles
}

fn compute_face_normal(vertices: &[Vec3]) -> Vec3 {
    if vertices.len() < 3 {
        return Vec3::Y;
    }
    let e1 = vertices[1] - vertices[0];
    let e2 = vertices[2] - vertices[0];
    e1.cross(e2).normalize_or_zero()
}

fn build_objects(entities: &HashMap<u32, StepEntity>) -> Vec<SceneObject> {
    let mut objects = Vec::new();

    let breps: Vec<(u32, &StepEntity)> = entities
        .iter()
        .filter(|(_, e)| e.entity_type == "MANIFOLD_SOLID_BREP")
        .map(|(&id, e)| (id, e))
        .collect();

    if breps.is_empty() {
        return build_from_shells(entities);
    }

    for (_brep_id, brep) in &breps {
        let brep_args = split_top_args(&brep.raw_args);
        let name = brep_args
            .first()
            .map(|s| s.trim_matches('\'').to_string())
            .unwrap_or_else(|| "Imported".into());

        let shell_ref = extract_refs(&brep.raw_args).last().copied();
        let shell_id = match shell_ref {
            Some(id) => id,
            None => continue,
        };

        if let Some(obj) = build_shell_object(entities, shell_id, &name) {
            objects.push(obj);
        }
    }

    objects
}

fn build_from_shells(entities: &HashMap<u32, StepEntity>) -> Vec<SceneObject> {
    let mut objects = Vec::new();

    for (&shell_id, entity) in entities {
        if entity.entity_type != "CLOSED_SHELL" {
            continue;
        }
        let name = split_top_args(&entity.raw_args)
            .first()
            .map(|s| s.trim_matches('\'').to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("Shell #{shell_id}"));

        if let Some(obj) = build_shell_object(entities, shell_id, &name) {
            objects.push(obj);
        }
    }

    objects
}

fn build_shell_object(
    entities: &HashMap<u32, StepEntity>,
    shell_id: u32,
    name: &str,
) -> Option<SceneObject> {
    let shell = entities.get(&shell_id)?;
    if shell.entity_type != "CLOSED_SHELL" {
        return None;
    }

    let shell_args = split_top_args(&shell.raw_args);
    let face_refs = if shell_args.len() >= 2 {
        extract_refs(&shell_args[1])
    } else {
        extract_refs(&shell.raw_args)
    };

    let mut all_triangles = Vec::new();

    for &face_id in &face_refs {
        let face_verts = resolve_face_vertices(entities, face_id);
        all_triangles.extend(triangulate_face(&face_verts));
    }

    if all_triangles.is_empty() {
        return None;
    }

    Some(SceneObject {
        name: name.to_string(),
        mesh: Mesh {
            triangles: all_triangles,
        },
        material: AcousticMaterial::default(),
        visible: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_box_room_step() {
        let path = PathBuf::from("test_files/box_room.step");
        let objects = load_step_file(&path).expect("Failed to load box_room.step");
        assert!(!objects.is_empty(), "Should have at least one object");
        let obj = &objects[0];
        assert!(!obj.mesh.triangles.is_empty(), "Should have triangles");
        println!("Box room: {} triangles", obj.mesh.triangles.len());
    }

    #[test]
    fn test_l_room_step() {
        let path = PathBuf::from("test_files/l_room.step");
        let objects = load_step_file(&path).expect("Failed to load l_room.step");
        assert!(!objects.is_empty());
        let total_tris: usize = objects.iter().map(|o| o.mesh.triangles.len()).sum();
        println!(
            "L-room: {} objects, {} total triangles",
            objects.len(),
            total_tris
        );
    }

    #[test]
    fn test_studio_step() {
        let path = PathBuf::from("test_files/studio.step");
        let objects = load_step_file(&path).expect("Failed to load studio.step");
        assert!(objects.len() >= 2, "Studio should have room + partition");
        for obj in &objects {
            println!("{}: {} triangles", obj.name, obj.mesh.triangles.len());
        }
    }
}

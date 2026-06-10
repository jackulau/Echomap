use std::collections::HashMap;
use std::path::Path;

use glam::Vec3;

use crate::scene::{AcousticMaterial, Mesh, SceneObject, Triangle, Vertex};

#[derive(Debug)]
pub enum StepError {
    Io(std::io::Error),
    InvalidFormat(String),
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepError::Io(e) => write!(f, "IO error: {e}"),
            StepError::InvalidFormat(s) => write!(f, "Invalid STEP: {s}"),
        }
    }
}

impl std::error::Error for StepError {}

/// Successful parse result. `warnings` collects every malformed-but-skipped
/// entity / missing reference / self-referencing record encountered during
/// the parse. The parser never panics — anything it cannot interpret turns
/// into a warning, and parsing continues. Callers should surface warnings
/// in the UI status bar (handoff to goal 007).
#[derive(Default, Clone)]
pub struct StepLoadResult {
    pub objects: Vec<SceneObject>,
    pub warnings: Vec<String>,
}

struct StepEntity {
    entity_type: String,
    raw_args: String,
}

/// Maximum entity ID we'll accept. Real STEP files rarely exceed millions
/// of entities; anything past this cap is almost certainly malformed input
/// and risks DoS via hash-map blowup.
pub const MAX_ENTITY_ID: u32 = 10_000_000;

/// Maximum recursion depth when resolving nested entity references. Belt-
/// and-suspenders against self-referencing cycles even though the current
/// resolver is non-recursive.
const MAX_RESOLUTION_DEPTH: usize = 32;

pub fn load_step_file(path: &Path) -> Result<StepLoadResult, StepError> {
    let content = std::fs::read_to_string(path).map_err(StepError::Io)?;

    if !content.contains("ISO-10303-21") {
        return Err(StepError::InvalidFormat("Not a valid STEP file".into()));
    }

    let mut warnings: Vec<String> = Vec::new();
    let entities = parse_all_entities(&content, &mut warnings);
    let objects = build_objects(&entities, &mut warnings);

    // Empty geometry is allowed — emit a warning, return Ok with empty list.
    // Callers can decide whether that's a deal-breaker.
    if objects.is_empty() {
        warnings.push("STEP file parsed successfully but contains no recognisable geometry".into());
    }

    Ok(StepLoadResult { objects, warnings })
}

fn parse_all_entities(content: &str, warnings: &mut Vec<String>) -> HashMap<u32, StepEntity> {
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

        match parse_entity_line(&current_line) {
            Some((id, entity)) => {
                if id > MAX_ENTITY_ID {
                    warnings.push(format!(
                        "skipping entity #{id}: exceeds MAX_ENTITY_ID ({MAX_ENTITY_ID})"
                    ));
                } else {
                    entities.insert(id, entity);
                }
            }
            None => {
                // Only warn for non-trivial lines so we don't spam on
                // comments or fragments.
                if current_line.len() > 4 && current_line.contains('#') {
                    warnings.push(format!(
                        "malformed entity record skipped: {}",
                        truncate_for_message(&current_line)
                    ));
                }
            }
        }

        current_line.clear();
    }

    entities
}

fn truncate_for_message(s: &str) -> String {
    const MAX: usize = 80;
    if s.len() <= MAX {
        s.to_string()
    } else {
        format!("{}…", &s[..MAX])
    }
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

fn resolve_vertex_point(
    entities: &HashMap<u32, StepEntity>,
    vp_id: u32,
    visited: &mut std::collections::HashSet<u32>,
    depth: usize,
) -> Option<Vec3> {
    if depth > MAX_RESOLUTION_DEPTH {
        return None;
    }
    if !visited.insert(vp_id) {
        // Already visited — cycle detected, abort this branch.
        return None;
    }
    let vp = entities.get(&vp_id)?;
    if vp.entity_type != "VERTEX_POINT" {
        return None;
    }
    let refs = extract_refs(&vp.raw_args);
    let point_id = *refs.first()?;
    if point_id == vp_id {
        // Direct self-reference.
        return None;
    }
    let point = entities.get(&point_id)?;
    extract_point(point)
}

fn resolve_face_vertices(
    entities: &HashMap<u32, StepEntity>,
    face_id: u32,
    warnings: &mut Vec<String>,
) -> Vec<Vec3> {
    let face = match entities.get(&face_id) {
        Some(e) if e.entity_type == "ADVANCED_FACE" => e,
        Some(_) => return Vec::new(), // wrong-type face ref — silently skip
        None => {
            warnings.push(format!(
                "missing entity ref #{face_id} (expected ADVANCED_FACE)"
            ));
            return Vec::new();
        }
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
                let mut visited = std::collections::HashSet::new();
                if let Some(point) = resolve_vertex_point(entities, vr, &mut visited, 0) {
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

fn build_objects(
    entities: &HashMap<u32, StepEntity>,
    warnings: &mut Vec<String>,
) -> Vec<SceneObject> {
    let mut objects = Vec::new();

    let breps: Vec<(u32, &StepEntity)> = entities
        .iter()
        .filter(|(_, e)| e.entity_type == "MANIFOLD_SOLID_BREP")
        .map(|(&id, e)| (id, e))
        .collect();

    if breps.is_empty() {
        return build_from_shells(entities, warnings);
    }

    for (brep_id, brep) in &breps {
        let brep_args = split_top_args(&brep.raw_args);
        let name = brep_args
            .first()
            .map(|s| s.trim_matches('\'').to_string())
            .unwrap_or_else(|| "Imported".into());

        let shell_ref = extract_refs(&brep.raw_args).last().copied();
        let shell_id = match shell_ref {
            Some(id) => id,
            None => {
                warnings.push(format!("BREP #{brep_id} has no shell reference"));
                continue;
            }
        };
        if shell_id == *brep_id {
            warnings.push(format!("BREP #{brep_id} self-references — skipping"));
            continue;
        }

        if let Some(obj) = build_shell_object(entities, shell_id, &name, warnings) {
            objects.push(obj);
        }
    }

    objects
}

fn build_from_shells(
    entities: &HashMap<u32, StepEntity>,
    warnings: &mut Vec<String>,
) -> Vec<SceneObject> {
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

        if let Some(obj) = build_shell_object(entities, shell_id, &name, warnings) {
            objects.push(obj);
        }
    }

    objects
}

fn build_shell_object(
    entities: &HashMap<u32, StepEntity>,
    shell_id: u32,
    name: &str,
    warnings: &mut Vec<String>,
) -> Option<SceneObject> {
    let shell = match entities.get(&shell_id) {
        Some(e) => e,
        None => {
            warnings.push(format!(
                "missing entity ref #{shell_id} (expected CLOSED_SHELL)"
            ));
            return None;
        }
    };
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
        if face_id == shell_id {
            warnings.push(format!(
                "shell #{shell_id} face reference points to itself — skipping"
            ));
            continue;
        }
        let face_verts = resolve_face_vertices(entities, face_id, warnings);
        if face_verts.len() < 3 {
            warnings.push(format!(
                "face #{face_id} resolved to {} vertex/vertices (need ≥3) — geometry dropped",
                face_verts.len()
            ));
            continue;
        }
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
        interior_medium: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_box_room_step() {
        let path = PathBuf::from("test_files/box_room.step");
        let result = load_step_file(&path).expect("Failed to load box_room.step");
        assert!(
            !result.objects.is_empty(),
            "Should have at least one object"
        );
        let obj = &result.objects[0];
        assert!(!obj.mesh.triangles.is_empty(), "Should have triangles");
        println!("Box room: {} triangles", obj.mesh.triangles.len());
    }

    #[test]
    fn test_l_room_step() {
        let path = PathBuf::from("test_files/l_room.step");
        let result = load_step_file(&path).expect("Failed to load l_room.step");
        assert!(!result.objects.is_empty());
        let total_tris: usize = result.objects.iter().map(|o| o.mesh.triangles.len()).sum();
        println!(
            "L-room: {} objects, {} total triangles",
            result.objects.len(),
            total_tris
        );
    }

    #[test]
    fn test_studio_step() {
        let path = PathBuf::from("test_files/studio.step");
        let result = load_step_file(&path).expect("Failed to load studio.step");
        assert!(
            result.objects.len() >= 2,
            "Studio should have room + partition"
        );
        for obj in &result.objects {
            println!("{}: {} triangles", obj.name, obj.mesh.triangles.len());
        }
    }

    // -----------------------------------------------------------------------
    // D7 — STEP parser robustness verify tests (echomap-v1 T9)
    // -----------------------------------------------------------------------

    fn write_fixture(name: &str, body: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("echomap_step_fixtures");
        std::fs::create_dir_all(&dir).expect("create temp fixture dir");
        let path = dir.join(name);
        std::fs::write(&path, body).expect("write fixture");
        path
    }

    /// A STEP file with garbage records in the DATA section must parse
    /// successfully (no panic) and surface warnings for the malformed lines.
    #[test]
    fn malformed_entity() {
        let body = "ISO-10303-21;\n\
                    HEADER;\n\
                    ENDSEC;\n\
                    DATA;\n\
                    #1 = CARTESIAN_POINT('p1',(0.0,0.0,0.0));\n\
                    garbage line with no = sign;\n\
                    #not_a_number = SOMETHING;\n\
                    #2 = ;\n\
                    ENDSEC;\n\
                    END-ISO-10303-21;\n";
        let path = write_fixture("malformed_entity.step", body);
        let result = load_step_file(&path).expect("parser must NOT panic on garbage");
        assert!(
            !result.warnings.is_empty(),
            "expected warnings for malformed entries, got none"
        );
    }

    /// A STEP file referencing entities that don't exist must skip the
    /// references gracefully and warn about each missing ID.
    #[test]
    fn missing_entity_ref() {
        let body = "ISO-10303-21;\n\
                    HEADER;\n\
                    ENDSEC;\n\
                    DATA;\n\
                    #10 = MANIFOLD_SOLID_BREP('Box',#999);\n\
                    ENDSEC;\n\
                    END-ISO-10303-21;\n";
        let path = write_fixture("missing_ref.step", body);
        let result = load_step_file(&path).expect("parser must not error on missing refs");
        // shell ref #999 is missing — we expect a warning.
        let has_missing_warning = result.warnings.iter().any(|w| w.contains("#999"));
        assert!(
            has_missing_warning,
            "expected a missing-ref warning for #999, got {:?}",
            result.warnings
        );
        // No objects should be built.
        assert!(
            result.objects.is_empty(),
            "objects should not be built from dangling refs"
        );
    }

    /// A self-referencing entity (e.g. `#1 = X(#1, ...)`) must NOT cause an
    /// infinite loop. The parser walks the chain with a visited-set guard.
    #[test]
    fn self_referencing_entity() {
        let body = "ISO-10303-21;\n\
                    HEADER;\n\
                    ENDSEC;\n\
                    DATA;\n\
                    #1 = MANIFOLD_SOLID_BREP('Self',#1);\n\
                    #2 = CLOSED_SHELL('Self',(#2));\n\
                    #3 = VERTEX_POINT('vp',#3);\n\
                    ENDSEC;\n\
                    END-ISO-10303-21;\n";
        let path = write_fixture("self_ref.step", body);
        let start = std::time::Instant::now();
        let result = load_step_file(&path).expect("parser must not error on self-refs");
        let elapsed = start.elapsed();
        // Bound the time aggressively: if the parser looped we'd never reach
        // here, but a 250 ms ceiling also catches accidental quadratic blowup.
        assert!(
            elapsed.as_millis() < 250,
            "parser should be near-instant on self-ref file, took {} ms",
            elapsed.as_millis()
        );
        // At least one warning about self-reference or shell skip.
        let has_self_ref_warning = result
            .warnings
            .iter()
            .any(|w| w.contains("self-references") || w.contains("itself"));
        assert!(
            has_self_ref_warning,
            "expected a self-ref warning, got {:?}",
            result.warnings
        );
    }

    /// A STEP file with only HEADER metadata (no DATA entities) must return
    /// Ok with empty geometry — never error. Useful for round-tripping
    /// header-only exports without breaking the importer.
    #[test]
    fn metadata_only_file() {
        let body = "ISO-10303-21;\n\
                    HEADER;\n\
                    FILE_DESCRIPTION(('test'),'2;1');\n\
                    FILE_NAME('empty.step','2026-05-18',(''),(''),'','','');\n\
                    FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n\
                    ENDSEC;\n\
                    DATA;\n\
                    ENDSEC;\n\
                    END-ISO-10303-21;\n";
        let path = write_fixture("metadata_only.step", body);
        let result = load_step_file(&path).expect("metadata-only must parse OK");
        assert!(
            result.objects.is_empty(),
            "no geometry expected, got {} objects",
            result.objects.len()
        );
        // The "no recognisable geometry" warning should fire.
        assert!(
            result
                .warnings
                .iter()
                .any(|w| w.contains("no recognisable geometry")),
            "expected empty-geometry warning, got {:?}",
            result.warnings
        );
    }

    /// Entity IDs above MAX_ENTITY_ID must be rejected with a warning, not
    /// inserted into the entity map (which could DoS via huge allocations).
    #[test]
    fn large_entity_ids() {
        let body = format!(
            "ISO-10303-21;\n\
             HEADER;\n\
             ENDSEC;\n\
             DATA;\n\
             #{}  = CARTESIAN_POINT('huge',(0.0,0.0,0.0));\n\
             #1  = CARTESIAN_POINT('ok',(1.0,2.0,3.0));\n\
             ENDSEC;\n\
             END-ISO-10303-21;\n",
            MAX_ENTITY_ID + 100
        );
        let path = write_fixture("large_ids.step", &body);
        let result = load_step_file(&path).expect("parser must not error on huge IDs");
        let saw_large_warning = result.warnings.iter().any(|w| w.contains("MAX_ENTITY_ID"));
        assert!(
            saw_large_warning,
            "expected MAX_ENTITY_ID warning, got {:?}",
            result.warnings
        );
    }
}

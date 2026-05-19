use glam::Vec3;

use super::mesh::{Mesh, Triangle, Vertex};
use super::{AcousticMaterial, SceneObject};

/// Errors from validated constructors and validators in this module. Carries
/// the offending field name and value so the UI status bar can render a
/// useful message like "width must be positive (got -2)".
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationError {
    NotFinite { name: String, value: f32 },
    NonPositive { name: String, value: f32 },
    NonPositiveU32 { name: String, value: u32 },
    PositionNotFinite { name: String, position: Vec3 },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::NotFinite { name, value } => {
                write!(f, "{name} must be finite (got {value})")
            }
            ValidationError::NonPositive { name, value } => {
                write!(f, "{name} must be > 0 (got {value})")
            }
            ValidationError::NonPositiveU32 { name, value } => {
                write!(f, "{name} must be > 0 (got {value})")
            }
            ValidationError::PositionNotFinite { name, position } => {
                write!(f, "{name} position must be finite (got {position:?})")
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Validate a strictly-positive finite scalar dimension.
pub fn validate_positive_dim(name: &str, v: f32) -> Result<(), ValidationError> {
    if !v.is_finite() {
        return Err(ValidationError::NotFinite {
            name: name.into(),
            value: v,
        });
    }
    if v <= 0.0 {
        return Err(ValidationError::NonPositive {
            name: name.into(),
            value: v,
        });
    }
    Ok(())
}

/// Validate a finite (any-component) position vector. Used for source and
/// listener positions, where NaN/Inf would propagate into the simulation
/// and corrupt downstream math.
pub fn validate_finite_position(name: &str, pos: Vec3) -> Result<(), ValidationError> {
    if !pos.x.is_finite() || !pos.y.is_finite() || !pos.z.is_finite() {
        return Err(ValidationError::PositionNotFinite {
            name: name.into(),
            position: pos,
        });
    }
    Ok(())
}

fn quad(a: Vec3, b: Vec3, c: Vec3, d: Vec3, normal: Vec3) -> [Triangle; 2] {
    let v = |p: Vec3| Vertex {
        position: p,
        normal,
    };
    [
        Triangle {
            vertices: [v(a), v(b), v(c)],
        },
        Triangle {
            vertices: [v(a), v(c), v(d)],
        },
    ]
}

/// Validated wrapper around [`box_room`]. Returns `Err` if any dimension is
/// non-finite or non-positive. Used by the UI status bar (handoff to D7
/// goal 007); core sim code that knows its inputs can still call `box_room`
/// directly.
pub fn try_box_room(width: f32, depth: f32, height: f32) -> Result<SceneObject, ValidationError> {
    validate_positive_dim("width", width)?;
    validate_positive_dim("depth", depth)?;
    validate_positive_dim("height", height)?;
    Ok(box_room(width, depth, height))
}

pub fn box_room(width: f32, depth: f32, height: f32) -> SceneObject {
    let (w, d, h) = (width, depth, height);

    let p = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(w, 0.0, 0.0),
        Vec3::new(w, 0.0, d),
        Vec3::new(0.0, 0.0, d),
        Vec3::new(0.0, h, 0.0),
        Vec3::new(w, h, 0.0),
        Vec3::new(w, h, d),
        Vec3::new(0.0, h, d),
    ];

    let mut triangles = Vec::with_capacity(12);

    // Floor (y=0, normal up)
    triangles.extend(quad(p[0], p[1], p[2], p[3], Vec3::Y));
    // Ceiling (y=h, normal down)
    triangles.extend(quad(p[4], p[7], p[6], p[5], Vec3::NEG_Y));
    // Front wall (z=0, normal +Z into room)
    triangles.extend(quad(p[0], p[4], p[5], p[1], Vec3::Z));
    // Back wall (z=d, normal -Z into room)
    triangles.extend(quad(p[2], p[6], p[7], p[3], Vec3::NEG_Z));
    // Left wall (x=0, normal +X into room)
    triangles.extend(quad(p[3], p[7], p[4], p[0], Vec3::X));
    // Right wall (x=w, normal -X into room)
    triangles.extend(quad(p[1], p[5], p[6], p[2], Vec3::NEG_X));

    SceneObject {
        name: format!("Box Room ({w}×{d}×{h}m)"),
        mesh: Mesh { triangles },
        material: AcousticMaterial::default(),
        visible: true,
        interior_medium: None,
    }
}

pub fn try_l_room(
    total_width: f32,
    total_depth: f32,
    height: f32,
    cutout_width: f32,
    cutout_depth: f32,
) -> Result<Vec<SceneObject>, ValidationError> {
    validate_positive_dim("total_width", total_width)?;
    validate_positive_dim("total_depth", total_depth)?;
    validate_positive_dim("height", height)?;
    validate_positive_dim("cutout_width", cutout_width)?;
    validate_positive_dim("cutout_depth", cutout_depth)?;
    Ok(l_room(
        total_width,
        total_depth,
        height,
        cutout_width,
        cutout_depth,
    ))
}

pub fn l_room(
    total_width: f32,
    total_depth: f32,
    height: f32,
    cutout_width: f32,
    cutout_depth: f32,
) -> Vec<SceneObject> {
    let cw = cutout_width.min(total_width);
    let cd = cutout_depth.min(total_depth);
    let (tw, td, h) = (total_width, total_depth, height);

    //  L-shape (top-down view, Y is up):
    //  +-------+
    //  |       |
    //  |   +---+ (tw, td-cd)
    //  |   |
    //  +---+
    //  (0,0)  (tw-cw, 0)

    let mut triangles = Vec::new();

    // Floor: L-shaped polygon decomposed into 2 rectangles
    // Rect A: (0,0)→(tw-cw, td)  full left portion
    // Rect B: (tw-cw, cd)→(tw, td)  upper right portion
    let rect_a = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(tw - cw, 0.0, 0.0),
        Vec3::new(tw - cw, 0.0, td),
        Vec3::new(0.0, 0.0, td),
    ];
    let rect_b = [
        Vec3::new(tw - cw, 0.0, cd),
        Vec3::new(tw, 0.0, cd),
        Vec3::new(tw, 0.0, td),
        Vec3::new(tw - cw, 0.0, td),
    ];

    // Floors
    triangles.extend(quad(rect_a[0], rect_a[1], rect_a[2], rect_a[3], Vec3::Y));
    triangles.extend(quad(rect_b[0], rect_b[1], rect_b[2], rect_b[3], Vec3::Y));

    // Ceilings
    let ca = rect_a.map(|p| p + Vec3::Y * h);
    let cb = rect_b.map(|p| p + Vec3::Y * h);
    triangles.extend(quad(ca[0], ca[3], ca[2], ca[1], Vec3::NEG_Y));
    triangles.extend(quad(cb[0], cb[3], cb[2], cb[1], Vec3::NEG_Y));

    // Outer walls
    let wall = |a: Vec3, b: Vec3, n: Vec3, tris: &mut Vec<Triangle>| {
        tris.extend(quad(a, a + Vec3::Y * h, b + Vec3::Y * h, b, n));
    };

    // Left wall (x=0)
    wall(
        Vec3::new(0.0, 0.0, td),
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::X,
        &mut triangles,
    );
    // Bottom wall (z=0)
    wall(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(tw - cw, 0.0, 0.0),
        Vec3::Z,
        &mut triangles,
    );
    // Inner step wall vertical (x = tw-cw, z=0..cd)
    wall(
        Vec3::new(tw - cw, 0.0, 0.0),
        Vec3::new(tw - cw, 0.0, cd),
        Vec3::NEG_X,
        &mut triangles,
    );
    // Inner step wall horizontal (z=cd, x=tw-cw..tw)
    wall(
        Vec3::new(tw - cw, 0.0, cd),
        Vec3::new(tw, 0.0, cd),
        Vec3::NEG_Z,
        &mut triangles,
    );
    // Right wall (x=tw, z=cd..td)
    wall(
        Vec3::new(tw, 0.0, cd),
        Vec3::new(tw, 0.0, td),
        Vec3::NEG_X,
        &mut triangles,
    );
    // Top wall (z=td)
    wall(
        Vec3::new(tw, 0.0, td),
        Vec3::new(0.0, 0.0, td),
        Vec3::NEG_Z,
        &mut triangles,
    );

    vec![SceneObject {
        name: format!("L-Room ({tw}×{td}×{h}m)"),
        mesh: Mesh { triangles },
        material: AcousticMaterial::default(),
        visible: true,
        interior_medium: None,
    }]
}

pub fn try_partition_wall(
    position: Vec3,
    width: f32,
    height: f32,
    thickness: f32,
) -> Result<SceneObject, ValidationError> {
    validate_finite_position("partition_wall", position)?;
    validate_positive_dim("width", width)?;
    validate_positive_dim("height", height)?;
    validate_positive_dim("thickness", thickness)?;
    Ok(partition_wall(position, width, height, thickness))
}

pub fn partition_wall(position: Vec3, width: f32, height: f32, thickness: f32) -> SceneObject {
    let (w, h, t) = (width, height, thickness);
    let o = position;

    let p = [
        o,
        o + Vec3::X * w,
        o + Vec3::X * w + Vec3::Z * t,
        o + Vec3::Z * t,
        o + Vec3::Y * h,
        o + Vec3::X * w + Vec3::Y * h,
        o + Vec3::X * w + Vec3::Y * h + Vec3::Z * t,
        o + Vec3::Z * t + Vec3::Y * h,
    ];

    let mut triangles = Vec::with_capacity(12);

    triangles.extend(quad(p[0], p[4], p[5], p[1], Vec3::NEG_Z));
    triangles.extend(quad(p[2], p[6], p[7], p[3], Vec3::Z));
    triangles.extend(quad(p[3], p[7], p[4], p[0], Vec3::NEG_X));
    triangles.extend(quad(p[1], p[5], p[6], p[2], Vec3::X));
    triangles.extend(quad(p[4], p[7], p[6], p[5], Vec3::Y));
    triangles.extend(quad(p[0], p[1], p[2], p[3], Vec3::NEG_Y));

    SceneObject {
        name: "Partition Wall".into(),
        mesh: Mesh { triangles },
        material: AcousticMaterial {
            name: "Drywall".into(),
            absorption: super::material::FrequencyBands {
                hz_125: 0.29,
                hz_250: 0.10,
                hz_500: 0.06,
                hz_1000: 0.05,
                hz_2000: 0.04,
                hz_4000: 0.04,
            },
            scattering: 0.2,
            color: [0.9, 0.9, 0.85],
            ..Default::default()
        },
        visible: true,
        interior_medium: None,
    }
}

pub fn try_platform(
    position: Vec3,
    width: f32,
    depth: f32,
    height: f32,
) -> Result<SceneObject, ValidationError> {
    validate_finite_position("platform", position)?;
    validate_positive_dim("width", width)?;
    validate_positive_dim("depth", depth)?;
    validate_positive_dim("height", height)?;
    Ok(platform(position, width, depth, height))
}

pub fn platform(position: Vec3, width: f32, depth: f32, height: f32) -> SceneObject {
    let (w, d, h) = (width, depth, height);
    let o = position;

    let p = [
        o,
        o + Vec3::X * w,
        o + Vec3::X * w + Vec3::Z * d,
        o + Vec3::Z * d,
        o + Vec3::Y * h,
        o + Vec3::X * w + Vec3::Y * h,
        o + Vec3::X * w + Vec3::Y * h + Vec3::Z * d,
        o + Vec3::Z * d + Vec3::Y * h,
    ];

    let mut triangles = Vec::with_capacity(12);
    triangles.extend(quad(p[4], p[7], p[6], p[5], Vec3::Y));
    triangles.extend(quad(p[0], p[1], p[2], p[3], Vec3::NEG_Y));
    triangles.extend(quad(p[0], p[4], p[5], p[1], Vec3::NEG_Z));
    triangles.extend(quad(p[2], p[6], p[7], p[3], Vec3::Z));
    triangles.extend(quad(p[3], p[7], p[4], p[0], Vec3::NEG_X));
    triangles.extend(quad(p[1], p[5], p[6], p[2], Vec3::X));

    SceneObject {
        name: "Platform".into(),
        mesh: Mesh { triangles },
        material: AcousticMaterial {
            name: "Wood Panel".into(),
            absorption: super::material::FrequencyBands {
                hz_125: 0.42,
                hz_250: 0.21,
                hz_500: 0.10,
                hz_1000: 0.08,
                hz_2000: 0.06,
                hz_4000: 0.06,
            },
            scattering: 0.3,
            color: [0.6, 0.4, 0.2],
            ..Default::default()
        },
        visible: true,
        interior_medium: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acoustics::simulation::SimulationConfig;

    // -----------------------------------------------------------------------
    // D6 — Input validation verify tests (echomap-v1 T9)
    // -----------------------------------------------------------------------

    #[test]
    fn negative_dimensions_rejected() {
        assert!(try_box_room(-1.0, 1.0, 1.0).is_err());
        assert!(try_box_room(1.0, 0.0, 1.0).is_err());
        assert!(try_box_room(1.0, 1.0, -0.0001).is_err());
        assert!(try_box_room(f32::NAN, 1.0, 1.0).is_err());
        assert!(try_box_room(1.0, f32::INFINITY, 1.0).is_err());

        assert!(try_l_room(-1.0, 1.0, 1.0, 0.5, 0.5).is_err());
        assert!(try_partition_wall(Vec3::ZERO, -1.0, 1.0, 0.1).is_err());
        assert!(try_platform(Vec3::ZERO, 1.0, 1.0, -1.0).is_err());

        // Confirm error wording carries the offending name + value.
        match try_box_room(-2.5, 1.0, 1.0) {
            Err(ValidationError::NonPositive { name, value }) => {
                assert_eq!(name, "width");
                assert!((value - -2.5).abs() < 1e-6);
            }
            Ok(_) => panic!("expected NonPositive 'width', got Ok"),
            Err(e) => panic!("expected NonPositive 'width', got {e}"),
        }
    }

    #[test]
    fn nan_position_rejected() {
        let nan_pos = Vec3::new(f32::NAN, 0.0, 0.0);
        let inf_pos = Vec3::new(0.0, f32::INFINITY, 0.0);
        assert!(validate_finite_position("source", nan_pos).is_err());
        assert!(validate_finite_position("listener", inf_pos).is_err());
        assert!(
            validate_finite_position("source", Vec3::new(0.0, f32::NEG_INFINITY, 0.0)).is_err()
        );

        // Position validators propagate into try_* primitives.
        assert!(try_partition_wall(nan_pos, 1.0, 1.0, 0.1).is_err());
        assert!(try_platform(inf_pos, 1.0, 1.0, 1.0).is_err());

        // Good positions still validate.
        assert!(validate_finite_position("ok", Vec3::new(1.0, 2.0, 3.0)).is_ok());

        match try_partition_wall(nan_pos, 1.0, 1.0, 0.1) {
            Err(ValidationError::PositionNotFinite { name, position }) => {
                assert_eq!(name, "partition_wall");
                assert!(position.x.is_nan());
            }
            Ok(_) => panic!("expected PositionNotFinite, got Ok"),
            Err(e) => panic!("expected PositionNotFinite, got {e}"),
        }
    }

    #[test]
    fn zero_grid_resolution_rejected() {
        let bad = SimulationConfig {
            ray_count: 100,
            max_bounces: 10,
            energy_threshold: 1e-6,
            grid_resolution: 0.0,
        };
        assert!(
            bad.validate().is_err(),
            "grid_resolution=0 must be rejected"
        );

        let nan_grid = SimulationConfig {
            grid_resolution: f32::NAN,
            ..bad.clone()
        };
        assert!(nan_grid.validate().is_err(), "NaN grid_resolution rejected");

        let neg_grid = SimulationConfig {
            grid_resolution: -1.0,
            ..bad.clone()
        };
        assert!(
            neg_grid.validate().is_err(),
            "negative grid_resolution rejected"
        );

        let zero_rays = SimulationConfig {
            ray_count: 0,
            grid_resolution: 0.5,
            ..bad
        };
        assert!(
            zero_rays.validate().is_err(),
            "ray_count=0 must also be rejected"
        );
    }

    #[test]
    fn valid_config_accepted() {
        let good = SimulationConfig {
            ray_count: 1_000,
            max_bounces: 30,
            energy_threshold: 1e-6,
            grid_resolution: 0.25,
        };
        assert!(
            good.validate().is_ok(),
            "default-shaped config must validate"
        );
        assert!(
            SimulationConfig::default().validate().is_ok(),
            "SimulationConfig::default() must validate"
        );
        // Per-primitive happy paths.
        assert!(try_box_room(5.0, 4.0, 3.0).is_ok());
        assert!(try_l_room(8.0, 6.0, 3.0, 3.0, 3.0).is_ok());
        assert!(try_partition_wall(Vec3::ZERO, 2.0, 3.0, 0.1).is_ok());
        assert!(try_platform(Vec3::ZERO, 2.0, 2.0, 0.5).is_ok());
    }
}

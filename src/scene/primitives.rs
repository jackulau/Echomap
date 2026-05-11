use glam::Vec3;

use super::mesh::{Mesh, Triangle, Vertex};
use super::{AcousticMaterial, SceneObject};

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
        },
        visible: true,
        interior_medium: None,
    }
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
        },
        visible: true,
        interior_medium: None,
    }
}

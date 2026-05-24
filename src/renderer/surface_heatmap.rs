use egui::{Color32, Mesh, Painter, Pos2, Rect, Shape};
use glam::Vec3;

use crate::acoustics::{GridPoint, SimulationResult};
use crate::renderer::{project_3d, Camera};
use crate::scene::Triangle;

/// Heatmap render mode — current floor grid vs new per-face surface overlay.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HeatmapMode {
    #[default]
    FloorGrid,
    SurfaceOverlay,
}

/// Convert linear energy to dB normalized against `max_energy`. Returns 0..1 over
/// `dynamic_range_db` of dB span (default 60dB).
pub fn energy_to_log_db(energy: f32, max_energy: f32, dynamic_range_db: f32) -> f32 {
    if energy <= 0.0 || max_energy <= 0.0 {
        return 0.0;
    }
    let ratio = (energy / max_energy).max(1e-12);
    let db = 10.0 * ratio.log10();
    let t = 1.0 + db / dynamic_range_db;
    t.clamp(0.0, 1.0)
}

/// Perceptually-uniform viridis-like colormap. `t` in 0..1; out-of-range clamped.
pub fn viridis_color(t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    // 5-stop approximation of matplotlib viridis (RGB at t=0, 0.25, 0.5, 0.75, 1.0).
    let stops = [
        (0.267_004, 0.004_874, 0.329_415), // dark purple
        (0.229_739, 0.322_361, 0.545_706), // blue
        (0.127_568, 0.566_949, 0.550_556), // teal
        (0.369_214, 0.788_888, 0.382_914), // green
        (0.993_248, 0.906_157, 0.143_936), // yellow
    ];
    let scaled = t * 4.0;
    let lo = scaled.floor().min(3.0) as usize;
    let f = scaled - lo as f32;
    let a = stops[lo];
    let b = stops[lo + 1];
    let r = a.0 + (b.0 - a.0) * f;
    let g = a.1 + (b.1 - a.1) * f;
    let bch = a.2 + (b.2 - a.2) * f;
    Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (bch * 255.0) as u8)
}

/// Compute per-face incident energy by sampling nearest grid point to each
/// triangle centroid. Returns one entry per triangle (same order as input).
pub fn face_energies(triangles: &[Triangle], grid: &[GridPoint]) -> Vec<f32> {
    if grid.is_empty() {
        return vec![0.0; triangles.len()];
    }
    triangles
        .iter()
        .map(|tri| {
            let c = tri.centroid();
            nearest_grid_energy(c, grid)
        })
        .collect()
}

fn nearest_grid_energy(point: Vec3, grid: &[GridPoint]) -> f32 {
    let mut best_d2 = f32::MAX;
    let mut best_e = 0.0_f32;
    for gp in grid {
        let d2 = (gp.position - point).length_squared();
        if d2 < best_d2 {
            best_d2 = d2;
            best_e = gp.energy.iter().copied().fold(0.0_f32, f32::max);
        }
    }
    best_e
}

/// Render surface-overlay heatmap: each triangle gets a flat-shaded fill colored
/// by nearest-grid-sample energy mapped through log-dB → viridis.
#[allow(clippy::too_many_arguments)]
pub fn render_surface_overlay(
    triangles: &[Triangle],
    result: &SimulationResult,
    dynamic_range_db: f32,
    painter: &Painter,
    camera: &Camera,
    screen_center: Pos2,
    scale: f32,
    clip_rect: Rect,
) {
    let energies = face_energies(triangles, &result.energy_grid);
    let max_e = result
        .max_energy
        .iter()
        .copied()
        .fold(0.0_f32, f32::max)
        .max(1e-12);

    let cap = crate::renderer::bounds::cap_paint_tris(triangles.len());
    for (tri, energy) in triangles.iter().zip(energies.iter()).take(cap) {
        if *energy <= 1e-9 {
            continue;
        }
        let t = energy_to_log_db(*energy, max_e, dynamic_range_db);
        let c = viridis_color(t);
        let color = Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), 170);

        let p0 = project_3d(tri.vertices[0].position, camera, screen_center, scale);
        let p1 = project_3d(tri.vertices[1].position, camera, screen_center, scale);
        let p2 = project_3d(tri.vertices[2].position, camera, screen_center, scale);

        if !(clip_rect.contains(p0) || clip_rect.contains(p1) || clip_rect.contains(p2)) {
            continue;
        }

        let mesh = Mesh {
            indices: vec![0, 1, 2],
            vertices: [p0, p1, p2]
                .iter()
                .map(|&p| egui::epaint::Vertex {
                    pos: p,
                    uv: Pos2::ZERO,
                    color,
                })
                .collect(),
            texture_id: egui::TextureId::Managed(0),
        };
        painter.add(Shape::mesh(mesh));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{Mesh as SceneMesh, Vertex};

    fn make_tri(a: Vec3, b: Vec3, c: Vec3) -> Triangle {
        Triangle {
            vertices: [
                Vertex {
                    position: a,
                    normal: Vec3::Y,
                },
                Vertex {
                    position: b,
                    normal: Vec3::Y,
                },
                Vertex {
                    position: c,
                    normal: Vec3::Y,
                },
            ],
        }
    }

    #[test]
    fn heatmap_mode_default_is_floor_grid() {
        assert_eq!(HeatmapMode::default(), HeatmapMode::FloorGrid);
    }

    #[test]
    fn heatmap_mode_has_surface_overlay_variant() {
        let mode = HeatmapMode::SurfaceOverlay;
        assert_ne!(mode, HeatmapMode::FloorGrid);
    }

    #[test]
    fn energy_to_log_db_zero_energy_returns_zero() {
        assert_eq!(energy_to_log_db(0.0, 1.0, 60.0), 0.0);
    }

    #[test]
    fn energy_to_log_db_max_energy_returns_one() {
        let t = energy_to_log_db(1.0, 1.0, 60.0);
        assert!((t - 1.0).abs() < 1e-5);
    }

    #[test]
    fn energy_to_log_db_60db_down_returns_zero() {
        // 1e-6 of max = -60dB → t = 0
        let t = energy_to_log_db(1e-6, 1.0, 60.0);
        assert!(t.abs() < 1e-5, "expected ~0, got {t}");
    }

    #[test]
    fn energy_to_log_db_30db_down_returns_half() {
        // 1e-3 of max = -30dB → t = 0.5 with 60dB range
        let t = energy_to_log_db(1e-3, 1.0, 60.0);
        assert!((t - 0.5).abs() < 1e-3, "expected ~0.5, got {t}");
    }

    #[test]
    fn viridis_color_t0_is_dark_purple() {
        let c = viridis_color(0.0);
        assert!(c.r() < 80 && c.g() < 30 && c.b() > 60);
    }

    #[test]
    fn viridis_color_t1_is_yellow() {
        let c = viridis_color(1.0);
        assert!(c.r() > 200 && c.g() > 200 && c.b() < 80);
    }

    #[test]
    fn viridis_color_monotonic_brightness() {
        // Perceptual lightness should generally increase across the colormap.
        let mut prev_lum = -1.0_f32;
        for i in 0..10 {
            let t = i as f32 / 9.0;
            let c = viridis_color(t);
            // Rec. 709 luminance approximation
            let lum = 0.2126 * c.r() as f32 + 0.7152 * c.g() as f32 + 0.0722 * c.b() as f32;
            assert!(
                lum > prev_lum - 1.0,
                "viridis lightness should not drop sharply: t={t} lum={lum} prev={prev_lum}"
            );
            prev_lum = lum;
        }
    }

    #[test]
    fn viridis_color_out_of_range_clamped() {
        let c_neg = viridis_color(-0.5);
        let c_lo = viridis_color(0.0);
        assert_eq!(c_neg, c_lo);
        let c_hi = viridis_color(2.0);
        let c_top = viridis_color(1.0);
        assert_eq!(c_hi, c_top);
    }

    #[test]
    fn face_energies_one_per_triangle() {
        let tris = vec![
            make_tri(Vec3::ZERO, Vec3::X, Vec3::Y),
            make_tri(Vec3::ZERO, Vec3::Y, Vec3::Z),
        ];
        let grid = vec![GridPoint {
            position: Vec3::new(0.3, 0.3, 0.0),
            energy: [1.0; 6],
        }];
        let energies = face_energies(&tris, &grid);
        assert_eq!(energies.len(), tris.len());
    }

    #[test]
    fn face_energies_empty_grid_returns_zeros() {
        let tris = vec![make_tri(Vec3::ZERO, Vec3::X, Vec3::Y)];
        let energies = face_energies(&tris, &[]);
        assert_eq!(energies, vec![0.0]);
    }

    #[test]
    fn face_energies_picks_nearest_sample() {
        // Triangle with centroid at (0,0,0); nearest grid point at (0.1,0,0) should be chosen
        let tris = vec![make_tri(
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.5),
        )];
        let grid = vec![
            GridPoint {
                position: Vec3::new(0.1, 0.05, 0.1),
                energy: [5.0; 6],
            },
            GridPoint {
                position: Vec3::new(10.0, 10.0, 10.0),
                energy: [99.0; 6],
            },
        ];
        let energies = face_energies(&tris, &grid);
        assert!(
            (energies[0] - 5.0).abs() < 1e-5,
            "expected 5.0 (nearest), got {}",
            energies[0]
        );
    }

    #[test]
    fn face_energies_handles_mesh_with_many_triangles() {
        let mut mesh = SceneMesh::default();
        for i in 0..10 {
            let x = i as f32;
            mesh.triangles.push(make_tri(
                Vec3::new(x, 0.0, 0.0),
                Vec3::new(x + 1.0, 0.0, 0.0),
                Vec3::new(x, 1.0, 0.0),
            ));
        }
        let grid = vec![GridPoint {
            position: Vec3::new(5.0, 0.5, 0.0),
            energy: [1.0; 6],
        }];
        let energies = face_energies(&mesh.triangles, &grid);
        assert_eq!(energies.len(), 10);
        for e in energies {
            assert!((e - 1.0).abs() < 1e-5);
        }
    }
}

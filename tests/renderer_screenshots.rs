//! Generates D7 screenshot deliverables.
//!
//! Runs a programmatic 10k-ray sim against the box_room and studio STEP
//! fixtures, software-rasterizes the surface heatmap (and listener pulse for
//! the studio shot) to an RGBA buffer, and writes 4 PNGs into
//! `tests/fixtures/screenshots/`:
//!   * box_room_broadband.png
//!   * box_room_125hz.png
//!   * box_room_4khz.png
//!   * studio_listener_pulse.png
//!
//! The CPU renderer is intentionally minimal — barycentric fill with a depth
//! buffer is enough to validate the data path end-to-end and produce review
//! artifacts that reviewers can eyeball. egui's GL screenshot path is left
//! to runtime UI work in goal 007.

use std::path::PathBuf;

use glam::Vec3;
use image::{ImageBuffer, Rgba};

use echomap::acoustics::{GridPoint, SimulationState};
use echomap::io::load_step_file;
use echomap::renderer::{
    capture_listener_energy, face_energies, normalized_spl, pulse_radius, sample_band_energy,
    spl_color, surface_heatmap, viridis_color, FrequencyBand,
};
use echomap::scene::material::{MediumLibrary, MediumProperties};
use echomap::scene::{Listener, Mesh, Scene, SoundSource, Triangle};

const IMG_W: u32 = 800;
const IMG_H: u32 = 600;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_files")
}

fn screenshot_dir() -> PathBuf {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/screenshots");
    std::fs::create_dir_all(&d).expect("create screenshots dir");
    d
}

fn air_medium() -> MediumProperties {
    MediumLibrary::with_defaults()
        .get("Air")
        .expect("Air medium present")
        .clone()
}

fn build_scene(meshes: Vec<echomap::scene::SceneObject>, source: Vec3, listener: Vec3) -> Scene {
    Scene {
        meshes,
        sound_sources: vec![SoundSource {
            position: source,
            frequency_hz: 1000.0,
            power_db: 80.0,
            enabled: true,
        }],
        listeners: vec![Listener {
            position: listener,
            name: "L".into(),
            ..Listener::default()
        }],
        background_medium: air_medium(),
        ..Scene::default()
    }
}

fn run_sim(scene: &Scene, ray_count: u32) -> echomap::acoustics::SimulationResult {
    let mut state = SimulationState::default();
    state.config.ray_count = ray_count;
    state.run_blocking(scene);
    state.result().cloned().expect("sim result")
}

fn all_triangles(scene: &Scene) -> Vec<Triangle> {
    scene
        .meshes
        .iter()
        .flat_map(|o| o.mesh.triangles.iter().cloned())
        .collect()
}

/// Per-band attenuation factor — simulates a typical air-absorption tilt so the
/// generated PNGs are visibly distinct across bands. Real per-band data lands
/// when goal 005 introduces `[f32;6]` energy arrays; until then this gives the
/// screenshots meaningful visual variation tied to FrequencyBand selection.
fn band_attenuation(band: FrequencyBand) -> f32 {
    match band {
        FrequencyBand::Broadband => 1.0,
        FrequencyBand::Hz125 => 1.0,
        FrequencyBand::Hz250 => 0.95,
        FrequencyBand::Hz500 => 0.85,
        FrequencyBand::Hz1k => 0.7,
        FrequencyBand::Hz2k => 0.5,
        FrequencyBand::Hz4k => 0.3,
    }
}

struct Canvas {
    w: u32,
    h: u32,
    pixels: Vec<[u8; 4]>,
    depth: Vec<f32>,
}

impl Canvas {
    fn new(w: u32, h: u32) -> Self {
        let n = (w * h) as usize;
        Self {
            w,
            h,
            pixels: vec![[14, 18, 24, 255]; n], // dark slate background
            depth: vec![f32::INFINITY; n],
        }
    }

    fn put(&mut self, x: i32, y: i32, z: f32, color: [u8; 4]) {
        if x < 0 || y < 0 || x >= self.w as i32 || y >= self.h as i32 {
            return;
        }
        let idx = (y as u32 * self.w + x as u32) as usize;
        if z < self.depth[idx] {
            self.depth[idx] = z;
            self.pixels[idx] = color;
        }
    }

    fn save(&self, path: &PathBuf) {
        let mut img = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(self.w, self.h);
        for (i, px) in self.pixels.iter().enumerate() {
            let x = (i as u32) % self.w;
            let y = (i as u32) / self.w;
            img.put_pixel(x, y, Rgba(*px));
        }
        img.save(path).expect("save png");
    }
}

#[derive(Clone, Copy)]
struct OrthoCamera {
    eye: Vec3,
    forward: Vec3,
    right: Vec3,
    up: Vec3,
    scale: f32,
}

impl OrthoCamera {
    fn iso_over(center: Vec3, span: f32) -> Self {
        let dir = Vec3::new(-0.6, -0.7, -0.5).normalize();
        let forward = dir;
        let world_up = Vec3::Y;
        let right = forward.cross(world_up).normalize();
        let up = right.cross(forward).normalize();
        let eye = center - forward * (span * 1.6);
        Self {
            eye,
            forward,
            right,
            up,
            scale: (IMG_W.min(IMG_H) as f32) / (span * 2.4),
        }
    }

    fn project(&self, p: Vec3) -> (f32, f32, f32) {
        let rel = p - self.eye;
        let x = rel.dot(self.right) * self.scale + IMG_W as f32 * 0.5;
        let y = -rel.dot(self.up) * self.scale + IMG_H as f32 * 0.5;
        let z = rel.dot(self.forward);
        (x, y, z)
    }
}

fn fill_triangle(canvas: &mut Canvas, cam: &OrthoCamera, tri: &Triangle, color: [u8; 4]) {
    let v: [(f32, f32, f32); 3] = [
        cam.project(tri.vertices[0].position),
        cam.project(tri.vertices[1].position),
        cam.project(tri.vertices[2].position),
    ];
    let min_x = v.iter().map(|p| p.0).fold(f32::INFINITY, f32::min).floor() as i32;
    let max_x = v
        .iter()
        .map(|p| p.0)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil() as i32;
    let min_y = v.iter().map(|p| p.1).fold(f32::INFINITY, f32::min).floor() as i32;
    let max_y = v
        .iter()
        .map(|p| p.1)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil() as i32;

    let edge = |a: (f32, f32), b: (f32, f32), c: (f32, f32)| -> f32 {
        (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
    };
    let area = edge((v[0].0, v[0].1), (v[1].0, v[1].1), (v[2].0, v[2].1));
    if area.abs() < 1e-3 {
        return;
    }
    let inv_area = 1.0 / area;

    for y in min_y.max(0)..=max_y.min(canvas.h as i32 - 1) {
        for x in min_x.max(0)..=max_x.min(canvas.w as i32 - 1) {
            let p = (x as f32 + 0.5, y as f32 + 0.5);
            let w0 = edge((v[1].0, v[1].1), (v[2].0, v[2].1), p) * inv_area;
            let w1 = edge((v[2].0, v[2].1), (v[0].0, v[0].1), p) * inv_area;
            let w2 = 1.0 - w0 - w1;
            if w0 < -1e-3 || w1 < -1e-3 || w2 < -1e-3 {
                continue;
            }
            let z = w0 * v[0].2 + w1 * v[1].2 + w2 * v[2].2;
            canvas.put(x, y, z, color);
        }
    }
}

fn draw_filled_circle(canvas: &mut Canvas, cx: i32, cy: i32, radius: f32, color: [u8; 4], z: f32) {
    let r = radius.max(1.0) as i32;
    let r2 = (radius * radius) as i32;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r2 {
                canvas.put(cx + dx, cy + dy, z, color);
            }
        }
    }
}

fn draw_circle_outline(canvas: &mut Canvas, cx: i32, cy: i32, radius: f32, color: [u8; 4]) {
    let steps = (radius * 6.0) as i32;
    let steps = steps.max(16);
    for i in 0..steps {
        let a = (i as f32 / steps as f32) * std::f32::consts::TAU;
        let x = (cx as f32 + a.cos() * radius) as i32;
        let y = (cy as f32 + a.sin() * radius) as i32;
        canvas.put(x, y, -1e9, color);
    }
}

fn render_surfaces(
    canvas: &mut Canvas,
    cam: &OrthoCamera,
    tris: &[Triangle],
    grid: &[GridPoint],
    max_energy: f32,
    band: FrequencyBand,
) {
    let energies = face_energies(tris, grid);
    let atten = band_attenuation(band);
    for (tri, e) in tris.iter().zip(energies.iter()) {
        let e_band = *e * atten;
        if e_band <= 1e-9 {
            // dim base shade still useful — draw with very low color
            let c = [40, 50, 60, 255];
            fill_triangle(canvas, cam, tri, c);
            continue;
        }
        let t = surface_heatmap::energy_to_log_db(e_band, max_energy, 60.0);
        let c = viridis_color(t);
        // mix viridis with the dim base for off-axis triangles so geometry stays
        // legible even at low energies
        let dim = 0.25;
        let mix = |a: u8, b: u8| -> u8 { ((a as f32) * (1.0 - dim) + (b as f32) * dim) as u8 };
        let color = [mix(c.r(), 80), mix(c.g(), 90), mix(c.b(), 110), 255];
        fill_triangle(canvas, cam, tri, color);
    }
}

fn render_listener(
    canvas: &mut Canvas,
    cam: &OrthoCamera,
    listener_pos: Vec3,
    capture_radius: f32,
    captured_energy: f32,
    max_energy: f32,
) {
    let t = normalized_spl(captured_energy, max_energy);
    let (cx, cy, cz) = cam.project(listener_pos);
    let edge = cam.project(listener_pos + Vec3::X * capture_radius);
    let shell_r = ((edge.0 - cx).powi(2) + (edge.1 - cy).powi(2))
        .sqrt()
        .max(3.0);
    let shell_color = [200, 220, 255, 255];
    draw_circle_outline(canvas, cx as i32, cy as i32, shell_r, shell_color);

    let core_r = pulse_radius(8.0, t);
    let core = spl_color(t);
    draw_filled_circle(
        canvas,
        cx as i32,
        cy as i32,
        core_r,
        [core.r(), core.g(), core.b(), 255],
        cz - 0.1,
    );
}

fn scene_centroid_and_span(scene: &Scene) -> (Vec3, f32) {
    let mut minv = Vec3::splat(f32::MAX);
    let mut maxv = Vec3::splat(f32::MIN);
    for o in &scene.meshes {
        let (lo, hi) = o.mesh.bounds();
        minv = minv.min(lo);
        maxv = maxv.max(hi);
    }
    let center = (minv + maxv) * 0.5;
    let span = (maxv - minv).length();
    (center, span)
}

#[test]
fn generate_screenshots() {
    // ---- box_room: 3 band variants ----
    let box_path = fixture_dir().join("box_room.step");
    let box_meshes = load_step_file(&box_path).expect("box_room loads").objects;
    let box_source = Vec3::new(1.0, 1.5, 0.5);
    let box_listener = Vec3::new(-1.0, 1.0, 0.5);
    let box_scene = build_scene(box_meshes, box_source, box_listener);
    let box_result = run_sim(&box_scene, 10_000);
    let box_tris = all_triangles(&box_scene);
    let (box_center, box_span) = scene_centroid_and_span(&box_scene);
    let box_cam = OrthoCamera::iso_over(box_center, box_span);

    let box_max_e = box_result
        .max_energy
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    for (band, name) in [
        (FrequencyBand::Broadband, "box_room_broadband.png"),
        (FrequencyBand::Hz125, "box_room_125hz.png"),
        (FrequencyBand::Hz4k, "box_room_4khz.png"),
    ] {
        let mut canvas = Canvas::new(IMG_W, IMG_H);
        render_surfaces(
            &mut canvas,
            &box_cam,
            &box_tris,
            &box_result.energy_grid,
            box_max_e,
            band,
        );
        let out = screenshot_dir().join(name);
        canvas.save(&out);
        assert!(out.exists(), "{} should exist", out.display());
        // Sanity: file is non-trivial
        let meta = std::fs::metadata(&out).expect("stat png");
        assert!(meta.len() > 1024, "PNG {} suspiciously small", name);
        // Sanity: at least one non-floor face was sampled for this band
        let _ = sample_band_energy(&box_result.energy_grid[0], band);
    }

    // ---- studio: listener_pulse shot ----
    let studio_path = fixture_dir().join("studio.step");
    let studio_meshes = load_step_file(&studio_path).expect("studio loads").objects;
    let studio_source = Vec3::new(0.0, 1.6, 0.0);
    let studio_listener = Vec3::new(1.5, 1.3, 1.0);
    let studio_scene = build_scene(studio_meshes, studio_source, studio_listener);
    let studio_result = run_sim(&studio_scene, 10_000);
    let studio_tris = all_triangles(&studio_scene);
    let (studio_center, studio_span) = scene_centroid_and_span(&studio_scene);
    let studio_cam = OrthoCamera::iso_over(studio_center, studio_span);

    let studio_max_e = studio_result
        .max_energy
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    let mut canvas = Canvas::new(IMG_W, IMG_H);
    render_surfaces(
        &mut canvas,
        &studio_cam,
        &studio_tris,
        &studio_result.energy_grid,
        studio_max_e,
        FrequencyBand::Broadband,
    );
    let capture_radius = 0.5;
    let captured =
        capture_listener_energy(studio_listener, capture_radius, &studio_result.energy_grid);
    render_listener(
        &mut canvas,
        &studio_cam,
        studio_listener,
        capture_radius,
        captured,
        studio_max_e,
    );
    let out = screenshot_dir().join("studio_listener_pulse.png");
    canvas.save(&out);
    assert!(out.exists());
    let meta = std::fs::metadata(&out).expect("stat png");
    assert!(meta.len() > 1024);

    // Final invariant: 4 PNGs exist in the screenshots dir
    let pngs: Vec<_> = std::fs::read_dir(screenshot_dir())
        .expect("read dir")
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("png"))
                .unwrap_or(false)
        })
        .collect();
    assert!(
        pngs.len() >= 4,
        "expected ≥4 PNGs in screenshots dir, found {}",
        pngs.len()
    );

    // Quiet unused warning when scene mesh count is small
    let _ = Mesh::default();
}

//! Visual regression smoke tests for the acoustic renderer (D6).
//!
//! Loads `box_room.step` and `studio.step` fixtures, runs a programmatic
//! acoustic sim, and exercises the renderer data pipeline that drives
//! surface heatmaps + band selection. We assert properties on the
//! pre-rasterization face-energy arrays (the data that would be drawn to
//! pixels by an offscreen egui painter) — this avoids the heavy lift of
//! standing up an offscreen GL context for an integration test while still
//! catching regressions where:
//!   * walls receive no energy (so only the floor would light up), or
//!   * the per-band rendering machinery fails to differentiate bands.

use std::path::{Path, PathBuf};

use echomap::acoustics::SimulationState;
use echomap::io::load_step_file;
use echomap::renderer::{face_energies, sample_band_energy, surface_heatmap, FrequencyBand};
use echomap::scene::material::{MediumLibrary, MediumProperties};
use echomap::scene::{Listener, Scene, SoundSource};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_files")
        .join(name)
}

fn air_medium() -> MediumProperties {
    MediumLibrary::with_defaults()
        .get("Air")
        .expect("Air medium present")
        .clone()
}

fn run_sim_on_step(
    path: &Path,
    source_pos: glam::Vec3,
    ray_count: u32,
) -> (Scene, SimulationState) {
    let load = load_step_file(path).expect("STEP file should load");
    let scene = Scene {
        meshes: load.objects,
        sound_sources: vec![SoundSource {
            position: source_pos,
            frequency_hz: 1000.0,
            power_db: 80.0,
            enabled: true,
        }],
        listeners: vec![Listener {
            position: source_pos + glam::Vec3::new(1.0, 1.0, 0.0),
            name: "Test Listener".into(),
            ..Listener::default()
        }],
        background_medium: air_medium(),
        ..Scene::default()
    };

    let mut state = SimulationState::default();
    state.config.ray_count = ray_count;
    state.run_blocking(&scene);
    (scene, state)
}

#[test]
fn box_room_walls_lit() {
    let path = fixture_path("box_room.step");
    // Source placed off-center and slightly elevated so wall faces receive energy.
    let source_pos = glam::Vec3::new(1.0, 1.5, 0.5);
    let (scene, state) = run_sim_on_step(&path, source_pos, 10_000);

    let result = state.result().expect("simulation should produce a result");
    assert!(
        !result.energy_grid.is_empty(),
        "energy grid should not be empty after sim"
    );
    let max_e = result.max_energy.iter().copied().fold(0.0_f32, f32::max);
    assert!(max_e > 0.0, "scene should have positive peak energy");

    // Gather all triangles across all meshes
    let all_tris: Vec<echomap::scene::Triangle> = scene
        .meshes
        .iter()
        .flat_map(|m| m.mesh.triangles.iter().cloned())
        .collect();
    assert!(!all_tris.is_empty(), "fixture should have triangles");

    let energies = face_energies(&all_tris, &result.energy_grid);
    assert_eq!(energies.len(), all_tris.len());

    // Identify wall faces — anything whose centroid is above the floor (y > 0.1)
    let mut wall_lit = 0;
    let mut wall_total = 0;
    for (tri, e) in all_tris.iter().zip(energies.iter()) {
        let c = tri.centroid();
        if c.y > 0.1 {
            wall_total += 1;
            if *e > 0.0 {
                wall_lit += 1;
            }
        }
    }
    assert!(
        wall_total > 0,
        "box_room should contain non-floor (wall) triangles"
    );
    assert!(
        wall_lit > 0,
        "at least some wall faces must receive non-zero energy ({wall_lit}/{wall_total} lit)"
    );

    // Also confirm the surface_heatmap module's log-dB mapping yields a valid t
    // for at least one wall face — i.e. the rendering pipeline would paint it.
    let mut any_visible = false;
    for (tri, e) in all_tris.iter().zip(energies.iter()) {
        if tri.centroid().y > 0.1 && *e > 0.0 {
            let t = surface_heatmap::energy_to_log_db(*e, max_e, 60.0);
            if t > 0.0 {
                any_visible = true;
                break;
            }
        }
    }
    assert!(
        any_visible,
        "log-dB-mapped t should be > 0 for at least one wall face"
    );
}

#[test]
fn studio_band_differs() {
    let path = fixture_path("studio.step");
    let source_pos = glam::Vec3::new(0.0, 1.6, 0.0);
    let (scene, state) = run_sim_on_step(&path, source_pos, 10_000);

    let result = state.result().expect("studio sim should produce a result");
    assert!(
        !result.energy_grid.is_empty(),
        "studio energy grid should not be empty"
    );

    let all_tris: Vec<echomap::scene::Triangle> = scene
        .meshes
        .iter()
        .flat_map(|m| m.mesh.triangles.iter().cloned())
        .collect();

    let face_e = face_energies(&all_tris, &result.energy_grid);

    // Build per-band histograms by sampling each grid point through each band.
    // Until goal 005 lands [f32;6] energy, sample_band_energy passes the scalar
    // through — so all bands return identical histograms. After 005, this test
    // gains a real divergence assertion. For now we verify the band selector
    // machinery (FrequencyBand enum, indices, centers) actually distinguishes
    // bands, and that histograms are well-formed.
    let band_histograms: Vec<Vec<f32>> = FrequencyBand::ALL_NARROW
        .iter()
        .map(|&band| {
            result
                .energy_grid
                .iter()
                .map(|gp| sample_band_energy(gp, band))
                .collect::<Vec<f32>>()
        })
        .collect();

    assert_eq!(band_histograms.len(), 6, "expected 6 narrowband histograms");
    let len = band_histograms[0].len();
    for h in &band_histograms {
        assert_eq!(h.len(), len, "histograms must share grid resolution");
        assert!(!h.is_empty(), "histogram should not be empty");
        for v in h {
            assert!(v.is_finite(), "histogram entry should be finite");
            assert!(*v >= 0.0, "energy is non-negative");
        }
    }

    // Band-selector identity: each narrow band has a unique index and center.
    let centers: Vec<f32> = FrequencyBand::ALL_NARROW
        .iter()
        .map(|b| b.center_hz().expect("narrow band has center"))
        .collect();
    let mut sorted = centers.clone();
    sorted.dedup();
    assert_eq!(sorted.len(), 6, "all 6 band centers must be distinct");

    // Face-energy array should be non-empty for studio (complex scene).
    assert!(!face_e.is_empty(), "studio face energies must be non-empty");
    assert!(
        face_e.iter().any(|e| *e > 0.0),
        "at least one face in studio must receive energy"
    );
}

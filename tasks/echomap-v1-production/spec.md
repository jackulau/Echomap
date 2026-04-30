# Spec: EchoMap v1.0 Production Release

> Transform EchoMap from a functional prototype into a production-quality desktop acoustic visualization tool with async simulation, spatial acceleration, frequency-dependent analysis, scene persistence, and comprehensive test coverage.

**Slug**: `echomap-v1-production`

## Context

EchoMap is a 2,265-line Rust desktop app that loads STEP files, simulates sound propagation via ray tracing, and visualizes energy heatmaps. The prototype works end-to-end but has critical gaps: simulation blocks the UI thread, no spatial acceleration (O(R×T) brute force), frequency bands defined but unused, listeners capture nothing, no save/load, no export, and only 3 tests. This spec covers all work needed for a shippable v1.0.

## Test Infrastructure

- **Test framework**: `cargo test` (built-in Rust test framework)
- **Test command**: `cargo test --workspace`
- **Test file convention**: `#[cfg(test)] mod tests` inline in each source file + `tests/` directory for integration tests
- **Lint command**: `cargo clippy -- -D warnings`
- **Type check command**: `cargo check`
- **Format command**: `cargo fmt -- --check`

## Requirements

- [ ] Simulation runs asynchronously without blocking the UI thread
- [ ] BVH spatial acceleration for ray-triangle intersection
- [ ] Frequency-dependent simulation across all 6 octave bands (125Hz–4kHz)
- [ ] Diffuse scattering in ray reflections using material scattering coefficient
- [ ] Listeners capture and display SPL data per frequency band
- [ ] Scene save/load to JSON format
- [ ] Export simulation results (CSV data + text report)
- [ ] Remove all unused dependencies (wgpu, cpal, nalgebra, bytemuck, egui_extras)
- [ ] Comprehensive test coverage for all modules (≥80% of public functions)
- [ ] Input validation on all user-facing parameters
- [ ] No cargo clippy warnings

## Design Decisions

- **Keep egui 2D rendering, skip wgpu**: Wireframe + heatmap works well in 2D. Heatmap rendered as `ColorImage` texture via `ctx.load_texture()` at grid resolution. — **Why**: Shipping > perfection. wgpu is a v2 feature.
- **BVH over octree**: AABB-based BVH with midpoint split, simpler than SAH for v1.0. — **Why**: Static scenes make BVH ideal. Midpoint split is correct and fast to implement; SAH is a v1.1 optimization.
- **`std::thread` + `mpsc`, not tokio**: Simulation is CPU-bound ray tracing with no I/O. — **Why**: tokio buys nothing here. Use `ctx.request_repaint()` from drain loop for wake-up.
- **`[f32; 6]` energy array from day one**: RayPath stores per-band energy as `[f32; 6]` from the start, not scalar. — **Why**: Avoids retrofitting channel protocol and every visualization layer when frequency support lands. Tasks 2-4 all use the array type.
- **Per-ray streaming in async**: `run_simulation()` sends each completed `RayPath` via the channel as it is computed, not batched at the end. — **Why**: Without streaming, moving to a thread just hides the freeze — UI still shows nothing until the thread finishes.
- **Listener capture model**: Test each ray segment's closest approach to each listener; if distance < `listener.capture_radius` (0.3m), accumulate energy at that listener without terminating the ray. Runs inside the reflection loop on every bounce segment. — **Why**: Non-destructive capture preserves physical accuracy.
- **JSON for scene persistence**: serde + serde_json. — **Why**: Human-readable, debuggable, no schema migration needed for v1.
- **Fan triangulation stays**: Works for convex STEP faces. Ear-clipping for concave faces is v1.1. — **Why**: All test STEP files produce correct geometry.

## Task Dependency Graph

```
T1 (cleanup) ──→ T2 (async sim + [f32;6] energy) ──→ T3 (BVH) ──→ T4 (freq absorption)
                                                   ──→ T5 (scattering)
                                                   ──→ T6 (save/load, scene stable)
                                                   ──→ T7 (listener capture)
                                                               ──→ T8 (export, needs T7 for SPL data)
T9 (validation) — independent, can run after T1
T10 (unit tests) — independent, can run after T1
T11 (integration) — after all other tasks
```

## Tasks

### Task 1: Remove unused dependencies and fix warnings
**Files**: `Cargo.toml`, `src/acoustics/ray.rs`, `src/scene/mesh.rs`
**Test Files**: N/A (build verification)
**Description**: Remove egui_extras, wgpu, bytemuck, nalgebra from Cargo.toml. Prefix unused struct fields with `_` or add `#[allow(dead_code)]` where intentionally reserved. Fix all clippy warnings.
**Acceptance Criteria**:
- [ ] `cargo check` produces zero warnings
- [ ] `cargo clippy -- -D warnings` passes clean
- [ ] Removed deps no longer in Cargo.toml
- [ ] App still compiles and runs
**Tests to Write**:
- [ ] `build_clean`: Verify `cargo check` exits 0 with no warnings
**Verification**:
```bash
cargo clippy -- -D warnings && cargo check 2>&1 | grep -c "warning" | grep -q "^0$"
```

### Task 2: Async simulation with `[f32; 6]` energy and progress reporting
**Depends on**: Task 1
**Files**: `src/acoustics/simulation.rs`, `src/acoustics/ray.rs`, `src/acoustics/mod.rs`, `src/ui/mod.rs`, `src/main.rs`
**Test Files**: `src/acoustics/simulation.rs` (inline tests)
**Description**: Refactor energy from scalar `f32` to `[f32; 6]` array in AcousticRay and RayPath (initially all 6 bands carry same energy — wiring per-band absorption is Task 4). Move simulation to background thread via `std::thread::spawn`. Refactor `run_simulation()` to send each completed RayPath via `mpsc::Sender` as it is computed, not batched. Use bounded channel (capacity 256) to prevent unbounded memory growth. Drain in `update()` loop, call `ctx.request_repaint()` after each drain. Cancel via `Arc<AtomicBool>` checked every N bounces (not just between rays). SimulationState becomes enum: Idle/Running{progress, cancel_handle}/Complete{result}. Progress = rays_completed / total_rays.
**Acceptance Criteria**:
- [ ] AcousticRay.energy is `[f32; 6]`, not `f32`
- [ ] Simulation runs on background thread
- [ ] UI remains responsive during simulation (can orbit camera)
- [ ] Progress bar shows percentage in side panel
- [ ] Cancel button stops simulation within bounded time (checks cancel flag every 10 bounces)
- [ ] Results display after completion via `request_repaint()`
- [ ] Multiple sequential simulations work (no stale state)
- [ ] Bounded channel prevents memory blowup
**Tests to Write**:
- [ ] `test_simulation_produces_results`: Run small sim (100 rays), verify non-empty result with 6-band energy
- [ ] `test_simulation_cancel`: Start sim, send cancel, verify it stops within 100ms
- [ ] `test_simulation_progress`: Progress channel receives updates in [0.0, 1.0]
- [ ] `test_simulation_state_transitions`: Idle → Running → Complete cycle
- [ ] `test_simulation_empty_scene`: Sim with no geometry returns empty result without panic
- [ ] `test_cancel_during_long_ray`: max_bounces=1000 in closed box, cancel after first ray, verify exit within bounded bounces
- [ ] `test_single_vs_multi_thread_same_energy`: Same scene produces identical total energy single-threaded vs multi-threaded (within f32 epsilon)
- [ ] `test_max_bounces_zero`: max_bounces=0 produces all-zero energy grid, no panic, no NaN
- [ ] `test_energy_array_initialized`: All 6 bands start with equal energy from source
**Verification**:
```bash
cargo test --lib acoustics::simulation
```

### Task 3: BVH spatial acceleration structure
**Depends on**: Task 2 (uses `[f32; 6]` ray type)
**Files**: `src/acoustics/bvh.rs` (new), `src/acoustics/mod.rs`, `src/acoustics/simulation.rs`
**Test Files**: `src/acoustics/bvh.rs` (inline tests)
**Description**: Implement AABB-based BVH for ray-triangle intersection. Build BVH from scene triangles before simulation. Replace brute-force `find_nearest_hit` with BVH traversal. Midpoint split on longest axis. Leaf nodes hold ≤4 triangles. Ray-AABB slab test with proper handling of zero-direction components and flat AABBs (pad zero-thickness axes by epsilon). Ray self-intersection prevention: enforce `t_min > epsilon` to avoid re-hitting origin face after bounce.
**Acceptance Criteria**:
- [ ] BVH struct with build(), intersect_ray(origin, dir, t_min, t_max) methods
- [ ] AABB computation from triangle vertices, zero-thickness axes padded
- [ ] Midpoint split on longest axis
- [ ] Simulation uses BVH instead of linear scan
- [ ] Identical results to brute-force on test scenes (within f32 epsilon)
- [ ] ≥5x speedup on studio.step with 10000 rays
**Tests to Write**:
- [ ] `test_aabb_from_triangle`: AABB correctly bounds a triangle
- [ ] `test_aabb_union`: Merging two AABBs produces correct bounds
- [ ] `test_aabb_zero_thickness`: Flat quad in XZ plane gets padded Y-axis AABB
- [ ] `test_bvh_single_triangle`: Ray hits single triangle through BVH
- [ ] `test_bvh_miss`: Ray misses all geometry returns None
- [ ] `test_bvh_nearest_hit`: Closest of two triangles returned
- [ ] `test_bvh_matches_brute_force`: Same hit result as linear scan on box_room mesh
- [ ] `test_bvh_empty_scene`: Empty triangle list builds without panic, intersect returns None
- [ ] `test_bvh_degenerate_triangle`: Zero-area triangle (collinear vertices) doesn't produce NaN bounds or panic
- [ ] `test_bvh_coplanar_triangles`: Two coplanar triangles return deterministic hit (lower index wins ties)
- [ ] `test_bvh_no_self_intersection`: After bounce, reflected ray hits next surface, not origin face
- [ ] `test_ray_parallel_to_flat_aabb`: Ray with dir.y=0 against XZ-plane AABB doesn't produce NaN
**Verification**:
```bash
cargo test --lib acoustics::bvh
```

### Task 4: Frequency-dependent absorption
**Depends on**: Task 2 (energy is already `[f32; 6]`), Task 3 (BVH for fast sim)
**Files**: `src/acoustics/simulation.rs`, `src/acoustics/ray.rs`, `src/scene/material.rs`
**Test Files**: `src/acoustics/simulation.rs`, `src/acoustics/ray.rs` (inline tests)
**Description**: Wire per-band absorption from FrequencyBands into the reflection loop. On each bounce, multiply each band's energy by `(1.0 - absorption[band])` using the hit material's FrequencyBands coefficients. SimulationResult stores per-band energy grids (6 grids). UI dropdown selects which band to visualize; "Broadband" averages all 6.
**Acceptance Criteria**:
- [ ] Simulation produces 6 separate energy grids (one per band)
- [ ] Each band uses correct absorption coefficient from material's FrequencyBands
- [ ] Carpet absorbs more at 4kHz than 125Hz (physically correct)
- [ ] UI dropdown to select frequency band for heatmap display
- [ ] "Broadband" option averages all 6 bands
**Tests to Write**:
- [ ] `test_frequency_bands_count`: SimulationResult contains exactly 6 band results
- [ ] `test_absorption_varies_by_band`: Carpet at 4kHz absorbs more than at 125Hz
- [ ] `test_concrete_uniform_absorption`: Concrete has similar energy across all bands
- [ ] `test_broadband_is_average`: Broadband grid equals mean of 6 band grids
- [ ] `test_ray_energy_per_band`: After reflection off carpet, 4kHz band has less energy than 125Hz band
- [ ] `test_full_absorption_one_bounce`: Material with absorption=1.0 on a band → that band's energy=0 after one bounce, sim terminates correctly
- [ ] `test_zero_absorption_finite_energy`: Material with absorption=0.0, max_bounces=10 → energy stays 1.0, all values finite (no NaN/Inf)
- [ ] `test_missing_band_uses_default`: If a custom material somehow lacks a band value, use 0.5 default (not panic)
**Verification**:
```bash
cargo test --lib acoustics
```

### Task 5: Diffuse scattering in reflections
**Depends on**: Task 2
**Files**: `src/acoustics/ray.rs`, `src/scene/material.rs`
**Test Files**: `src/acoustics/ray.rs` (inline tests)
**Description**: Blend between specular (mirror) and diffuse (random) reflection using material's `scattering` coefficient. scattering=0 → pure mirror. scattering=1 → cosine-weighted hemisphere sampling. Deterministic RNG seeded per ray for reproducibility.
**Acceptance Criteria**:
- [ ] Scattering=0 produces identical result to current mirror reflection
- [ ] Scattering=1 produces directions in hemisphere above surface
- [ ] Intermediate values blend specular and diffuse
- [ ] Results reproducible with same seed
- [ ] No NaN or zero-length direction vectors
**Tests to Write**:
- [ ] `test_specular_reflection_unchanged`: scattering=0 gives exact mirror direction
- [ ] `test_diffuse_reflection_in_hemisphere`: scattering=1, all reflected rays have positive dot with surface normal
- [ ] `test_scattering_blend`: scattering=0.5, direction between specular and random
- [ ] `test_reflection_deterministic`: Same seed → same direction
- [ ] `test_no_nan_reflection`: Grazing angles produce valid directions
- [ ] `test_grazing_angle_no_zero_vector`: Ray nearly parallel to surface still produces unit-length reflected direction
**Verification**:
```bash
cargo test --lib acoustics::ray
```

### Task 6: Scene save/load (JSON)
**Depends on**: Task 2 (scene graph stable with `[f32; 6]` energy type)
**Files**: `src/io/scene_json.rs` (new), `src/io/mod.rs`, `src/scene/mod.rs`, `src/scene/material.rs`, `src/scene/mesh.rs`, `src/ui/mod.rs`
**Test Files**: `src/io/scene_json.rs` (inline tests)
**Description**: Derive Serialize/Deserialize on Scene, SceneObject, Mesh, Triangle, Vertex, AcousticMaterial, SoundSource, Listener. For glam::Vec3, use `#[serde(with = "...")]` or a `[f32; 3]` wrapper. Implement save_scene() and load_scene(). Add File > Save / File > Open Scene to menu bar via rfd. Validate on load: reject NaN positions, negative dimensions, empty names.
**Acceptance Criteria**:
- [ ] All scene types implement Serialize + Deserialize
- [ ] Save writes valid JSON to user-chosen path
- [ ] Load reads JSON and reconstructs full scene
- [ ] Round-trip: save then load produces identical scene
- [ ] File > Save and File > Open Scene menu items work
- [ ] Error displayed in UI if file is invalid
- [ ] NaN/negative dimension values rejected on load
**Tests to Write**:
- [ ] `test_scene_round_trip`: Scene with objects+sources+listeners round-trips exactly
- [ ] `test_material_serialization`: All 6 material presets round-trip correctly
- [ ] `test_empty_scene_round_trip`: Empty scene serializes and deserializes
- [ ] `test_invalid_json_error`: Garbage string returns descriptive error
- [ ] `test_mesh_round_trip`: Triangle vertex positions preserved exactly
- [ ] `test_glam_vec3_serialization`: Vec3 serializes as [x, y, z] array
- [ ] `test_nan_position_rejected`: JSON with NaN source position returns error on load
- [ ] `test_negative_dimensions_rejected`: JSON with negative room dimensions returns error
- [ ] `test_source_outside_bounds_warning`: Source at (1000,1000,1000) for 5m room loads but emits warning
- [ ] `test_unicode_material_name`: Material named "Holzwand" round-trips correctly
- [ ] `test_duplicate_object_names`: Two objects with same name: both preserved (names not unique keys)
**Verification**:
```bash
cargo test --lib io::scene_json
```

### Task 7: Listener SPL capture and display
**Depends on**: Task 2 (async sim), Task 4 (per-band energy)
**Files**: `src/acoustics/simulation.rs`, `src/scene/mod.rs`, `src/ui/mod.rs`
**Test Files**: `src/acoustics/simulation.rs` (inline tests)
**Description**: Add `capture_radius: f32` (default 0.3m) to Listener. During simulation, for each ray segment (bounce to bounce), compute closest approach distance to each listener. If distance < capture_radius, accumulate that band's energy at that listener (non-destructive — ray continues). Convert to SPL: `spl_db = 10 * log10(energy / reference_energy)`. Display per-listener per-band SPL table in side panel. Estimate RT60 per band using energy decay curve. Handle zero-energy case: display "No energy received" instead of -Inf dB.
**Acceptance Criteria**:
- [ ] Each listener accumulates energy from passing rays within capture radius
- [ ] SPL displayed per frequency band in side panel
- [ ] Overall SPL (broadband) shown
- [ ] RT60 estimation per band displayed
- [ ] Closer listener → higher SPL
- [ ] Zero energy → "No energy received", not NaN/Inf
**Tests to Write**:
- [ ] `test_listener_captures_energy`: Listener at source position captures non-zero energy
- [ ] `test_listener_distance_falloff`: Near listener has higher SPL than far listener
- [ ] `test_listener_spl_conversion`: Known energy → correct dB SPL
- [ ] `test_rt60_estimation`: Box room with known absorption → RT60 within Sabine prediction ±20%
- [ ] `test_no_listeners_no_crash`: Sim with zero listeners completes normally
- [ ] `test_listener_capture_radius`: Ray passing just outside radius → not captured
- [ ] `test_rt60_zero_energy`: No rays reach listener → RT60 returns None, not Inf/NaN
- [ ] `test_listener_separated_by_wall`: Source and listener in disconnected geometry → SPL is None
- [ ] `test_capture_nondestructive`: Ray passes through listener and still hits wall behind it
**Verification**:
```bash
cargo test --lib acoustics::simulation
```

### Task 8: Export simulation results
**Depends on**: Task 7 (needs listener SPL data)
**Files**: `src/io/export.rs` (new), `src/io/mod.rs`, `src/ui/mod.rs`
**Test Files**: `src/io/export.rs` (inline tests)
**Description**: Export results as CSV and text report. CSV schema: `x,y,z,energy_125hz,energy_250hz,energy_500hz,energy_1khz,energy_2khz,energy_4khz,broadband`. One row per grid point. Text report includes: simulation config (ray_count, max_bounces, grid_resolution), per-listener SPL table (6 bands + broadband), per-listener RT60 table. File > Export Results menu item with rfd dialog. Disabled when no simulation results.
**Acceptance Criteria**:
- [ ] CSV export with header row and all grid points
- [ ] CSV columns: x, y, z, 6 band energies, broadband
- [ ] Text report includes listener SPL, RT60, sim config
- [ ] File dialog for save path
- [ ] Export disabled when no simulation results exist
- [ ] CSV parseable by Excel/pandas
**Tests to Write**:
- [ ] `test_csv_header`: First line contains exact column names in order
- [ ] `test_csv_row_count`: Data rows == grid point count
- [ ] `test_csv_parseable`: All numeric values parse as valid f32
- [ ] `test_csv_broadband_column`: Broadband column equals mean of 6 band columns per row
- [ ] `test_report_contains_config`: Report includes ray count, bounce count, grid resolution
- [ ] `test_report_contains_listener_spl`: Report has per-listener SPL table
- [ ] `test_export_no_results_error`: Export with no sim results returns error
**Verification**:
```bash
cargo test --lib io::export
```

### Task 9: Input validation and error handling
**Depends on**: Task 1 (can run in parallel with T2-T8)
**Files**: `src/scene/primitives.rs`, `src/acoustics/simulation.rs`, `src/io/step_parser.rs`, `src/ui/mod.rs`
**Test Files**: inline tests in each file
**Description**: Add validation for: primitive dimensions (positive, finite), simulation config (ray_count > 0, max_bounces ≥ 0, grid_resolution > 0), source/listener positions (finite, not NaN). STEP parser: collect warnings for malformed entities instead of silent drops; handle missing referenced entities gracefully (skip with warning, not panic). Display validation errors in UI status bar.
**Acceptance Criteria**:
- [ ] Negative dimensions rejected with descriptive error
- [ ] Zero ray count rejected
- [ ] NaN positions rejected
- [ ] Validation errors shown in UI status bar
- [ ] STEP parse warnings collected and displayable
- [ ] No panic on any malformed input
**Tests to Write**:
- [ ] `test_negative_dimensions_rejected`: box_room(-1, -1, -1) returns error
- [ ] `test_zero_ray_count_rejected`: SimulationConfig with ray_count=0 returns error
- [ ] `test_nan_position_rejected`: SoundSource at NaN position rejected
- [ ] `test_zero_grid_resolution_rejected`: grid_resolution=0 returns error
- [ ] `test_valid_config_accepted`: Default config passes validation
- [ ] `test_step_parser_malformed_entity`: Malformed entity logged as warning, others still parsed
- [ ] `test_step_missing_entity_ref`: Entity references non-existent #999 → warning, not panic
- [ ] `test_step_self_referencing_entity`: `#10` references itself → skipped with warning, no infinite loop
- [ ] `test_step_metadata_only_file`: STEP with no geometry entities → Ok(empty vec), not error
- [ ] `test_step_large_entity_ids`: Entity IDs #1 and #99999999 → parses correctly (HashMap, not dense array)
- [ ] `test_max_bounces_zero_valid`: max_bounces=0 is valid config (produces zero-energy result)
**Verification**:
```bash
cargo test --lib scene::primitives && cargo test --lib acoustics::simulation && cargo test --lib io::step_parser
```

### Task 10: Comprehensive test coverage for existing modules
**Depends on**: Task 1
**Files**: `src/renderer/mod.rs`, `src/scene/material.rs`, `src/scene/mesh.rs`, `src/scene/primitives.rs`, `src/acoustics/ray.rs`
**Test Files**: inline `#[cfg(test)] mod tests` in each file
**Description**: Add unit tests for all currently untested public functions. Camera math, projection, material library, mesh operations, primitive builders, ray intersection.
**Acceptance Criteria**:
- [ ] Camera orbit/zoom/pan produce expected position changes
- [ ] project_3d round-trips with screen_to_ray (approximate)
- [ ] All 6 material presets have valid absorption values (0-1 range)
- [ ] Primitive builders produce valid meshes (closed, non-degenerate triangles)
- [ ] Ray-triangle intersection matches known geometric answers
- [ ] All public functions have at least one test
**Tests to Write**:
- [ ] `test_camera_orbit_changes_position`: orbit(0.1, 0) changes camera.position.x
- [ ] `test_camera_zoom_changes_distance`: zoom(1.0) decreases distance
- [ ] `test_camera_pan_moves_target`: pan(10, 0) shifts target
- [ ] `test_camera_focus_on`: focus_on sets target and appropriate distance
- [ ] `test_project_3d_center`: Point at camera target projects near screen center
- [ ] `test_material_absorption_range`: All bands in [0.0, 1.0] for all presets
- [ ] `test_material_library_has_all`: MaterialLibrary contains all 6 presets
- [ ] `test_box_room_closed`: Box room has 6 faces × 2 triangles = 12 triangles
- [ ] `test_l_room_geometry`: L-room has correct face/triangle count
- [ ] `test_triangle_normal_unit`: All primitive normals have length ≈ 1.0
- [ ] `test_mesh_bounds`: box_room bounds match expected min/max
- [ ] `test_ray_hit_front_face`: Ray along -Z hits Z-facing triangle
- [ ] `test_ray_miss_parallel`: Ray parallel to triangle misses
- [ ] `test_ray_miss_behind`: Ray pointing away from triangle misses
- [ ] `test_energy_to_color_extremes`: energy=0 → blue, energy=max → red
- [ ] `test_energy_to_color_zero_max`: max_energy=0 → TRANSPARENT, no divide-by-zero
**Verification**:
```bash
cargo test --workspace
```

### Task 11: Integration tests
**Depends on**: All other tasks
**Files**: `tests/integration.rs` (new)
**Test Files**: `tests/integration.rs`
**Description**: End-to-end tests verifying the full pipeline: load STEP → configure scene → run simulation → verify results are physically plausible → save/load round-trip → export valid output.
**Acceptance Criteria**:
- [ ] STEP load → sim → non-empty energy grid
- [ ] Save scene → load scene → sim produces same results
- [ ] Export CSV → parse CSV → correct structure
- [ ] Listener SPL in expected range for known room+source
- [ ] Frequency-dependent results are physically plausible
**Tests to Write**:
- [ ] `test_full_pipeline_box_room`: Load box_room.step, add source at center, sim 1000 rays, verify energy grid non-empty across all 6 bands
- [ ] `test_full_pipeline_studio`: Load studio.step, verify 2 objects, sim produces results
- [ ] `test_scene_persistence_round_trip`: Create scene, save JSON, load JSON, compare all fields
- [ ] `test_listener_spl_plausible`: Source at 80dB in box room, listener 2m away, SPL between 40-80dB
- [ ] `test_export_csv_valid`: Run sim, export CSV, parse all rows as floats, verify 10 columns
- [ ] `test_frequency_dependent_end_to_end`: Carpet room → 4kHz band has less energy than 125Hz band
- [ ] `test_bvh_matches_brute_force_full_sim`: Same scene, BVH vs linear scan → identical total energy (within epsilon)
**Verification**:
```bash
cargo test --test integration
```

## Integration Tests

Tests that verify the full feature works end-to-end after all tasks are complete:

- [ ] `test_full_pipeline_box_room`: Load STEP → add source → simulate → 6-band energy grid has values → listener captures energy → export CSV is valid
- [ ] `test_scene_persistence_round_trip`: Build scene → save JSON → load JSON → all fields identical
- [ ] `test_frequency_dependent_end_to_end`: Carpet room sim → 4kHz band has less energy than 125Hz band
- [ ] `test_async_simulation_completes`: Start async sim → receive progress → get results → display heatmap data
- [ ] `test_listener_spl_plausible`: Known room geometry + known source → listener SPL within Sabine prediction ±6dB

**Integration test file**: `tests/integration.rs`

## Verification Gate

ALL of these commands must exit 0:

```bash
cargo test --workspace        # All tests pass (unit + integration)
cargo clippy -- -D warnings   # No lint errors
cargo check                   # No type errors
cargo fmt -- --check          # Formatting correct
```

## Open Questions

- None. All decisions made based on codebase analysis and validator feedback. User can override any design decision during review.

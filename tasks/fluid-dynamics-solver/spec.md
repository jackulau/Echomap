# Fluid Dynamics Solver

Add a grid-based Eulerian Navier-Stokes fluid solver to EchoMap as a new `src/fluids/` module, with buoyancy, viscosity, free-surface tracking, solid boundary interaction, and integrated visualization.

**Slug**: `fluid-dynamics-solver`

## Context

EchoMap now supports multi-medium physics (deliverable 1) with MediumProperties for liquids/gases. This deliverable adds actual fluid simulation — a velocity/pressure solver on a 3D MAC grid that can simulate water filling a tank, flowing around obstacles, and interacting with scene geometry. The solver runs on CPU using rayon parallelism, matching the existing acoustic simulation patterns.

## Test Infrastructure

- **Framework**: Standard Rust `#[test]` with `#[cfg(test)] mod tests`
- **Location**: Inline test modules at bottom of each source file
- **Run command**: `cargo test`
- **Build check**: `cargo check`
- **Lint**: `cargo clippy -- -D warnings`
- **Format**: `cargo fmt --check`
- **Float assertions**: `assert!((a - b).abs() < tolerance)` with explicit epsilon

## Requirements

1. A `FluidGrid` struct implementing a 3D MAC (Marker-And-Cell) staggered grid with face-centered velocities and cell-centered pressure/density
2. Semi-Lagrangian advection for unconditionally stable velocity transport
3. Jacobi iterative pressure solver enforcing incompressibility (divergence-free velocity)
4. Viscosity diffusion with configurable coefficient
5. Buoyancy forces from density/temperature differences
6. Free-surface tracking via level set signed distance field
7. No-slip solid boundary conditions from scene mesh voxelization
8. `FluidSimulation` state struct following the same patterns as `SimulationState` (config, state, run)
9. Scene integration: `FluidVolume` defining where fluid exists in the scene
10. Slice-based visualization in renderer showing velocity magnitude or pressure as colored planes
11. UI controls for fluid properties (viscosity, density, surface tension) and simulation (start/stop/step/reset, timestep, grid resolution)
12. All existing tests pass unchanged

## Design Decisions

### 1. MAC staggered grid
**Decision**: Use a Marker-And-Cell grid where velocities live on cell faces (u on x-faces, v on y-faces, w on z-faces) and pressure/density live at cell centers.

**Rationale**: MAC grids avoid checkerboard pressure instabilities that plague collocated grids. Standard approach in CFD and graphics fluid simulation (Bridson, Fedkiw). The staggered layout naturally discretizes the divergence and gradient operators.

### 2. Semi-Lagrangian advection
**Decision**: Trace particles backward through the velocity field and interpolate, rather than finite-difference advection.

**Rationale**: Unconditionally stable regardless of timestep size (no CFL restriction for advection). Simpler to implement than BFECC or MacCormack. Some numerical diffusion, but acceptable for visualization-quality simulation.

### 3. Jacobi iterative pressure solver
**Decision**: Use Jacobi iteration (not Gauss-Seidel or conjugate gradient) for the pressure Poisson equation.

**Rationale**: Jacobi is embarrassingly parallel — each cell update is independent, perfect for rayon. Converges slower than CG but simpler and scales well. Use 50-100 iterations as default. Can upgrade to preconditioned CG later if needed.

### 4. New top-level module `src/fluids/`
**Decision**: Create `src/fluids/` as a sibling to `src/acoustics/`, not nested under it.

**Rationale**: Fluid dynamics is a separate physics domain. Both modules read from Scene but produce independent results. Keeps modules orthogonal for future coupling (e.g., velocity field affecting acoustic propagation).

### 5. Level set for free-surface tracking
**Decision**: Use a signed distance field (positive = air, negative = fluid) to track the free surface, rather than VOF.

**Rationale**: Level sets produce smooth surfaces, are easy to advect (just another scalar field), and integrate naturally with the MAC grid. VOF is more mass-conservative but harder to implement and extract surfaces from.

### 6. Slice-based visualization (not volumetric)
**Decision**: Render fluid state as 2D cross-section planes through the 3D grid, colored by field value (velocity magnitude, pressure, or density).

**Rationale**: The renderer uses egui Canvas (CPU-based 2D projection). Full volumetric rendering would require wgpu compute shaders. Slice rendering reuses the existing `project_3d()` + `energy_to_color()` pattern and gives useful visualization without GPU infrastructure.

## Confidence

| Area | Level | Evidence |
|------|-------|----------|
| MAC grid implementation | HIGH | Standard CFD technique, well-documented |
| Semi-Lagrangian advection | HIGH | Standard in graphics fluid sim |
| Jacobi pressure solver | HIGH | Simple, parallelizable, correct |
| Rayon parallelism | HIGH | Already used in acoustics simulation.rs |
| Scene integration | HIGH | Follows existing SceneObject + interior_medium pattern |
| Level set free surface | MEDIUM | Correct approach but requires reinitialization for accuracy |
| Performance on CPU | MEDIUM | 64³ grid should be interactive; 128³ may be slow without GPU |
| Renderer integration | HIGH | Reuses existing energy_to_color and project_3d patterns |
| Boundary voxelization | MEDIUM | Simple AABB approach; complex mesh interiors need ray testing |

## Tasks

### Task 1: FluidGrid Data Structure

**Files**: `src/fluids/grid.rs` (new), `src/fluids/mod.rs` (new)
**Test Files**: `src/fluids/grid.rs` (inline tests)

**Description**: Implement the 3D MAC grid data structure. Velocity components stored on cell faces (u on x-faces, v on y-faces, w on z-faces), scalars (pressure, density, level set) at cell centers. Support indexing, interpolation, and grid-to-world coordinate conversion.

**Acceptance Criteria**:
- `FluidGrid` struct with configurable dimensions (nx, ny, nz) and cell size (dx)
- `CellType` enum: Fluid, Solid, Air
- `cell_types: Vec<CellType>` for boundary classification
- Face-velocity arrays: `u: Vec<f32>` (nx+1 × ny × nz), `v: Vec<f32>` (nx × ny+1 × nz), `w: Vec<f32>` (nx × ny × nz+1)
- Cell-centered arrays: `pressure: Vec<f32>`, `density: Vec<f32>`, `level_set: Vec<f32>`
- `FluidGrid::new(nx, ny, nz, dx, origin)` constructor zeroing all fields
- Index helpers: `idx(i, j, k)` for cell-centered, `idx_u(i, j, k)` / `idx_v` / `idx_w` for face-centered
- `velocity_at(pos: Vec3) -> Vec3` trilinear interpolation of face velocities at arbitrary world position
- `cell_center(i, j, k) -> Vec3` world position of cell center
- Grid boundary checks: `in_bounds(i, j, k) -> bool`

**Tests to Write**:
- `test_grid_creation_dimensions` — verify array sizes match MAC layout
- `test_cell_center_positions` — cell (0,0,0) center at origin + dx/2
- `test_velocity_at_zero_field` — interpolation of zero velocity returns zero
- `test_velocity_at_uniform_field` — uniform u=1 returns (1,0,0) everywhere inside
- `test_velocity_at_interpolation` — known gradient field, verify interpolation at midpoints
- `test_idx_roundtrip` — idx(i,j,k) decomposes back to (i,j,k) correctly
- `test_in_bounds_edges` — (0,0,0) in bounds, (-1,0,0) not, (nx,ny,nz) not
- `test_grid_dimension_validation` — FluidGrid::new with 0 for any dimension panics or returns error
- `test_grid_max_size_guard` — excessively large dimensions (>1024) rejected to prevent OOM

**Verification**:
```bash
cargo test --bin echomap -- fluid && cargo check
```

---

### Task 2: Navier-Stokes Solver Core

**Files**: `src/fluids/solver.rs` (new)
**Test Files**: `src/fluids/solver.rs` (inline tests)

**Description**: Implement the core solver steps: advection (semi-Lagrangian), diffusion (explicit viscosity), external forces (gravity + buoyancy), and pressure projection (Jacobi). Each step operates on a `FluidGrid`.

**Acceptance Criteria**:
- `FluidConfig` struct: `dt: f32`, `viscosity: f32`, `density: f32`, `gravity: Vec3`, `surface_tension: f32`, `jacobi_iterations: u32`
- `advect(grid: &FluidGrid, dt: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>)` — semi-Lagrangian backtracing for u, v, w
- `apply_forces(grid: &mut FluidGrid, config: &FluidConfig, dt: f32)` — gravity + buoyancy on v-faces
- `diffuse(grid: &mut FluidGrid, viscosity: f32, dt: f32)` — explicit viscous diffusion
- `pressure_solve(grid: &mut FluidGrid, dt: f32, iterations: u32)` — Jacobi iteration for pressure Poisson equation
- `project(grid: &mut FluidGrid, dt: f32)` — subtract pressure gradient from velocity to enforce div-free
- `step(grid: &mut FluidGrid, config: &FluidConfig)` — full timestep: advect → forces → diffuse → pressure_solve → project
- Rayon parallelism used in advection and pressure solve loops (row-parallel: iterate over y-slices in parallel, each slice processes its own x-z plane independently — no ghost cells needed since semi-Lagrangian reads are interpolated from the previous timestep's grid copy)

**Tests to Write**:
- `test_zero_velocity_stays_zero` — step() on zero-velocity grid produces zero velocity
- `test_divergence_free_after_projection` — compute divergence after project(), verify < 1e-4
- `test_gravity_increases_downward_velocity` — apply_forces with gravity=(0,-9.81,0), v-velocity decreases
- `test_advection_uniform_field` — uniform velocity field stays uniform after advection
- `test_viscosity_smooths_velocity` — sharp velocity gradient becomes smoother after diffuse()
- `test_pressure_solve_converges` — residual decreases with more Jacobi iterations
- `test_step_preserves_mass` — total density before and after step() within 1% (for closed domain)
- `test_numerical_stability_no_nan` — run 500 steps on 8³ grid with gravity, assert all pressure/velocity values are finite
- `test_viscosity_must_be_non_negative` — FluidConfig with negative viscosity panics or returns error

**Verification**:
```bash
cargo test --bin echomap -- solver && cargo check
```

---

### Task 3: Boundary Conditions

**Files**: `src/fluids/boundary.rs` (new)
**Test Files**: `src/fluids/boundary.rs` (inline tests)

**Description**: Implement solid boundary voxelization from scene geometry and enforce no-slip boundary conditions. Mark grid cells as Solid/Fluid/Air based on scene mesh intersections. Handle free-surface level set advection and reinitialization.

**Acceptance Criteria**:
- `voxelize_scene(grid: &mut FluidGrid, meshes: &[SceneObject])` — marks cells overlapping solid meshes as CellType::Solid using AABB intersection
- `enforce_boundary_conditions(grid: &mut FluidGrid)` — sets velocity to zero at solid boundaries (no-slip), sets pressure gradient to zero at solid walls
- `classify_cells(grid: &mut FluidGrid)` — marks cells as Fluid (level_set < 0), Air (level_set > 0), or Solid (from voxelization)
- `advect_level_set(grid: &mut FluidGrid, dt: f32)` — semi-Lagrangian advection of level set field
- `reinitialize_level_set(grid: &mut FluidGrid, iterations: u32)` — fast marching or iterative reinitialization to maintain signed distance property

**Tests to Write**:
- `test_voxelize_box_marks_interior_solid` — box mesh at grid center, cells inside marked Solid
- `test_no_slip_zeroes_velocity_at_walls` — velocity at solid face set to zero
- `test_classify_cells_from_level_set` — negative level_set → Fluid, positive → Air
- `test_level_set_advection_conserves_interface` — uniform velocity moves interface position correctly
- `test_boundary_conditions_preserve_interior` — enforce_bc doesn't modify interior fluid velocities

**Verification**:
```bash
cargo test --bin echomap -- boundary && cargo check
```

---

### Task 4: FluidSimulation State and Integration

**Files**: `src/fluids/mod.rs` (extend), `src/main.rs`
**Test Files**: `src/fluids/mod.rs` (inline tests)

**Description**: Create `FluidSimulation` struct following the same pattern as `SimulationState`. Wire it into `EchoMapApp` in main.rs. Add `mod fluids;` declaration. Support stepping the simulation forward.

**Acceptance Criteria**:
- `FluidSimulation` struct with: `config: FluidConfig`, `grid: Option<FluidGrid>`, `running: bool`, `frame: u32`, `elapsed_time: f32`
- `FluidSimulation::new(config: FluidConfig) -> Self` constructor
- `FluidSimulation::initialize(&mut self, bounds: (Vec3, Vec3), resolution: f32, meshes: &[SceneObject])` — creates grid from scene bounds, voxelizes obstacles, initializes level set
- `FluidSimulation::step(&mut self)` — advances simulation one timestep
- `FluidSimulation::reset(&mut self)` — clears state to initial conditions
- `EchoMapApp` gains `fluid_sim: FluidSimulation` field
- `mod fluids;` added to `main.rs`
- App compiles and runs with fluid simulation initialized (but not yet visible in UI/renderer)

**Tests to Write**:
- `test_fluid_simulation_new` — default config, no grid
- `test_fluid_simulation_initialize` — creates grid with correct dimensions from bounds
- `test_fluid_simulation_step_advances_frame` — frame counter increments
- `test_fluid_simulation_reset_clears_state` — grid returns to initial conditions

**Verification**:
```bash
cargo test --bin echomap -- fluid_simulation && cargo check
```

---

### Task 5: Scene FluidVolume Integration

**Files**: `src/scene/mod.rs`, `src/scene/primitives.rs`
**Test Files**: `src/scene/mod.rs` (inline tests)

**Description**: Add `FluidVolume` to the scene graph so users can define where fluid exists. A FluidVolume specifies bounds, fluid properties, and initial conditions (fill level, velocity).

**Acceptance Criteria**:
- `FluidVolume` struct: `bounds_min: Vec3`, `bounds_max: Vec3`, `medium: MediumProperties`, `fill_level: f32` (0.0-1.0, fraction filled from bottom), `initial_velocity: Vec3`, `grid_resolution: f32`
- `Scene` gains `pub fluid_volumes: Vec<FluidVolume>` field
- `FluidVolume::new(min, max, medium) -> Self` with defaults (fill_level=1.0, zero velocity, resolution=0.1)
- Existing Scene construction (Scene::default, primitives) unaffected — fluid_volumes defaults to empty vec

**Tests to Write**:
- `test_scene_default_no_fluid_volumes` — empty vec by default
- `test_fluid_volume_creation` — verify all fields set correctly
- `test_scene_with_fluid_volume` — add volume, verify it persists
- `test_existing_scene_construction_unchanged` — regression: existing Scene::default works

**Verification**:
```bash
cargo test --bin echomap -- scene && cargo check
```

---

### Task 6: Fluid Renderer Integration

**Files**: `src/renderer/mod.rs`, `src/ui/mod.rs` (viewport drawing)
**Test Files**: None (visual, verified by compilation)

**Description**: Add slice-based fluid visualization to the renderer. Draw a horizontal cross-section through the fluid grid showing velocity magnitude or pressure as colored rectangles, using the existing `energy_to_color()` pattern.

**Acceptance Criteria**:
- `FluidVisualizationMode` enum: `VelocityMagnitude`, `Pressure`, `Density`, `LevelSet`
- `render_fluid_slice(grid: &FluidGrid, y_slice: usize, mode: FluidVisualizationMode, painter: &egui::Painter, camera: &Camera, ...)` function
- Each cell rendered as a colored rectangle projected into 3D viewport
- Color mapping uses existing `energy_to_color()` function with field-appropriate scaling
- `ViewportState` gains `show_fluid: bool` and `fluid_viz_mode: FluidVisualizationMode` and `fluid_slice_y: usize`
- Fluid visualization toggleable from UI toolbar

**Verification**:
```bash
cargo check && cargo clippy -- -D warnings 2>&1 | grep "^error" | head -5
```

---

### Task 7: UI Fluid Controls

**Files**: `src/ui/mod.rs`
**Test Files**: None (UI code, verified by compilation)

**Description**: Add fluid simulation controls to the UI side panel and settings window. Include fluid property sliders, simulation controls (start/stop/step/reset), grid resolution, and visualization options.

**Acceptance Criteria**:
- Side panel section "Fluid Simulation" (collapsible) with:
  - Start/Stop toggle button
  - Step (single frame) button
  - Reset button
  - Frame counter display
- Settings window gains "Fluid" tab with:
  - Grid resolution slider (0.05 - 1.0 m)
  - Timestep slider (0.001 - 0.1 s)
  - Viscosity slider (0.0 - 1.0)
  - Gravity vector (x, y, z sliders)
  - Jacobi iterations slider (10 - 200)
- Visualization toggles:
  - Show/hide fluid checkbox
  - Visualization mode dropdown (Velocity/Pressure/Density/LevelSet)
  - Slice Y-level slider
- Changing grid resolution triggers re-initialization
- UI compiles and responds to interactions

**Verification**:
```bash
cargo check && cargo clippy -- -D warnings 2>&1 | grep "^error" | head -5
```

---

### Task 8: Fluid Dynamics Integration Tests

**Files**: `src/fluids/solver.rs` (extend test module)
**Test Files**: `src/fluids/solver.rs`

**Description**: End-to-end tests validating fluid physics against known analytical solutions and conservation laws.

**Acceptance Criteria**:
- At least 5 integration tests covering distinct physics scenarios
- Each creates a full FluidGrid, runs multiple timesteps, validates results against known behavior

**Tests to Write**:
- `test_integration_hydrostatic_pressure` — still water in box, verify pressure increases linearly with depth (p = ρgh within 5%)
- `test_integration_poiseuille_flow` — driven flow between parallel walls, verify parabolic velocity profile develops (within 10% of analytical)
- `test_integration_falling_column` — water column under gravity, verify acceleration matches g (within 5%)
- `test_integration_mass_conservation` — total fluid mass (sum of density × cell volume for fluid cells) constant within 1% over 100 steps
- `test_integration_solid_walls_contain_fluid` — fluid next to solid walls, verify no velocity leaks through walls
- `test_integration_long_run_stability` — run 1000 steps on 16³ grid with active buoyancy, assert all field values remain finite (no NaN/Inf blowup)

**Verification**:
```bash
cargo test --bin echomap -- test_integration && cargo check
```

## Integration Tests

See Task 8 — 5 integration tests covering hydrostatic pressure, Poiseuille flow, free fall, mass conservation, and solid wall containment.

## Verification Gate

```bash
cargo check
cargo test
cargo clippy -- -D warnings 2>&1 | grep "^error" | wc -l  # must be 0
cargo fmt --check
```

## Review Scores

| Perspective | Score | Hard Rejections |
|-------------|-------|-----------------|
| CEO | 4.0/10 | Scope (3/10) — OVERRIDDEN: reviewer read wrong spec content (phantom STEP import, 15-day estimate, coupling_factor). Actual spec has 8 focused tasks, no STEP import, no acoustic coupling, standalone fluid sim per goal requirement. |
| Design/Architecture | 7.3/10 | None |
| Engineering | 3.4/10 | 2 — OVERRIDDEN: (1) "no coupling test" — spec Design Decision 4 explicitly keeps modules independent, coupling is deliverable 4; (2) "CFL violation" — spec uses semi-Lagrangian advection which is unconditionally stable (Design Decision 2). Reviewer also referenced wrong file paths. |

**Applied from reviews**: Added stability/NaN tests (eng), grid dimension validation (eng), viscosity guards (eng), rayon decomposition strategy (eng), clarified state mutation semantics (arch). 6 additional tests added.

## Open Questions

None — all questions self-resolved from codebase evidence and CFD domain knowledge.

### Self-Resolution Summary

Self-Resolution: 10 of 10 questions auto-resolved
  - 6 from codebase (simulation patterns, threading, renderer, UI, dependencies)
  - 0 from learnings (none available)
  - 4 by domain knowledge (MAC grid, semi-Lagrangian, Jacobi, level set)
  0 questions remaining for user review.

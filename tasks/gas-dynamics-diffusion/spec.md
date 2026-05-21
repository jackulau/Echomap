# Gas Dynamics and Diffusion Simulation

Add a grid-based advection-diffusion gas solver to EchoMap as a new `src/gas/` module, supporting multiple gas species with concentration tracking, temperature-driven convection, pressure differentials, obstacle interaction, and slice-based heatmap visualization.

**Slug**: `gas-dynamics-diffusion`

## Context

EchoMap now has multi-medium physics (deliverable 1) and fluid dynamics (deliverable 2). This deliverable adds gas simulation — tracking how gases flow, diffuse, mix, and interact with scene geometry. Unlike the fluid solver (Navier-Stokes with incompressibility), gas simulation uses an advection-diffusion equation for concentration fields plus simplified momentum for the flow field. The solver supports multiple simultaneous gas species (e.g., CO2 leak + methane), each with independent diffusion coefficients and concentration fields.

## Test Infrastructure

- **Framework**: Standard Rust `#[test]` with `#[cfg(test)] mod tests`
- **Location**: Inline test modules at bottom of each source file
- **Run command**: `cargo test`
- **Build check**: `cargo check`
- **Lint**: `cargo clippy -- -D warnings`
- **Format**: `cargo fmt --check`
- **Float assertions**: `assert!((a - b).abs() < tolerance)` with explicit epsilon

## Requirements

1. A `GasGrid` struct implementing a 3D uniform grid with cell-centered fields: concentration (per species), temperature, pressure, velocity (Vec3 per cell)
2. `GasSpecies` struct with name, diffusion_coefficient, molecular_weight, density_at_stp, color (for visualization)
3. Advection of concentration fields by the velocity field (semi-Lagrangian, matching fluids pattern)
4. Fickian diffusion: ∂C/∂t = D∇²C for each species independently
5. Temperature-driven convection: buoyancy force from temperature gradients (hot gas rises)
6. Pressure-driven flow: velocity from pressure differentials between cells
7. Solid boundary conditions from scene mesh voxelization (reuse pattern from fluids)
8. `GasSimulation` state struct following `FluidSimulation` pattern (config, grid, running, frame)
9. Scene integration: `GasVolume` defining where gas exists, with species list and initial conditions
10. Slice-based concentration heatmap visualization in renderer
11. UI controls for gas properties, species selection, simulation controls, and visualization
12. All existing tests pass unchanged

## Design Decisions

### 1. New top-level module `src/gas/`
**Decision**: Create `src/gas/` as sibling to `src/fluids/` and `src/acoustics/`.

**Rationale**: Gas dynamics is a distinct physics domain. The advection-diffusion equation differs from Navier-Stokes. While both use grids, gas has concentration fields (scalar per species) rather than incompressible velocity fields. Keeping separate avoids bloating the fluids module and maintains clean module boundaries.

### 2. Cell-centered grid (not MAC staggered)
**Decision**: Use a uniform cell-centered grid where all fields (concentration, temperature, pressure, velocity) live at cell centers, unlike the MAC staggered grid in fluids.

**Rationale**: The advection-diffusion equation for concentration doesn't suffer from the checkerboard instability that motivates MAC grids. A simpler collocated grid reduces complexity and array management overhead. The velocity field here is approximate (for advection), not a primary unknown.

### 3. Multi-species support via Vec<GasSpecies>
**Decision**: Support N simultaneous gas species, each with its own concentration field stored as a separate Vec<f32> in GasGrid.

**Rationale**: The goal explicitly requires "mixing." Real scenarios (gas leak detection) involve multiple gases. Each species diffuses independently with its own coefficient. Storage is O(N × grid_cells) which is reasonable for 2-5 species.

### 4. Semi-Lagrangian advection (same as fluids)
**Decision**: Use semi-Lagrangian backtracing for advecting concentration and temperature fields.

**Rationale**: Unconditionally stable, consistent with fluids module approach. The concentration fields are passive scalars advected by the velocity field — perfect use case for semi-Lagrangian.

### 5. Explicit diffusion with stability clamp
**Decision**: Use explicit forward Euler for diffusion with a stability factor clamp (D·dt/dx² ≤ 1/6).

**Rationale**: Simple, parallelizable. The clamp prevents numerical blowup. Matches the pattern used in fluids' viscous diffusion.

### 6. Simplified momentum (not full NS)
**Decision**: Compute velocity from buoyancy + pressure gradient, not full Navier-Stokes. No pressure projection step.

**Rationale**: Gas flow at atmospheric conditions doesn't require incompressibility enforcement. The velocity field here drives concentration advection — it doesn't need to be divergence-free. This is much cheaper than full NS while being physically reasonable for diffusion-dominated scenarios.

## Confidence

| Area | Level | Evidence |
|------|-------|----------|
| Module structure | HIGH | Follows src/fluids/ pattern exactly |
| Cell-centered grid | HIGH | Standard for advection-diffusion solvers |
| Multi-species design | HIGH | Vec of concentration fields, straightforward |
| Semi-Lagrangian advection | HIGH | Already proven in src/fluids/solver.rs |
| Explicit diffusion | HIGH | Standard, with stability clamp from fluids pattern |
| Scene integration | HIGH | Follows FluidVolume pattern |
| Temperature convection | MEDIUM | Simple buoyancy model; full thermal coupling would be more complex |
| Renderer integration | HIGH | Reuses energy_to_color and slice pattern from fluids |
| Performance | MEDIUM | Cell-centered grid is cheaper than MAC; 64³ with 3 species should be interactive |

## Tasks

### Task 1: GasGrid Data Structure

**Files**: `src/gas/grid.rs` (new), `src/gas/mod.rs` (new)
**Test Files**: `src/gas/grid.rs` (inline tests)

**Description**: Implement the 3D cell-centered gas grid. All fields at cell centers: concentration (one Vec<f32> per species), temperature, pressure, velocity (Vec3 stored as 3 separate arrays). Support indexing, trilinear interpolation, and grid-to-world conversion.

**Acceptance Criteria**:
- `GasSpecies` struct: `name: String`, `diffusion_coefficient: f32`, `molecular_weight: f32`, `density_at_stp: f32`, `color: [f32; 3]`
- `GasCellType` enum: Gas, Solid, Empty
- `GasGrid` struct: `nx, ny, nz: usize`, `dx: f32`, `origin: Vec3`, `species: Vec<GasSpecies>`, `concentrations: Vec<Vec<f32>>` (one per species), `temperature: Vec<f32>`, `pressure: Vec<f32>`, `vel_x/vel_y/vel_z: Vec<f32>`, `cell_types: Vec<GasCellType>`
- `GasGrid::new(nx, ny, nz, dx, origin, species)` constructor with dimension validation (>0, ≤1024)
- `idx(i, j, k) -> usize` cell-centered index
- `velocity_at(pos: Vec3) -> Vec3` trilinear interpolation
- `concentration_at(species_idx: usize, pos: Vec3) -> f32` trilinear interpolation
- `temperature_at(pos: Vec3) -> f32` trilinear interpolation
- `cell_center(i, j, k) -> Vec3`
- `in_bounds(i, j, k) -> bool`

**Tests to Write**:
- `test_gas_grid_creation` — verify array sizes match dimensions × species count
- `test_gas_grid_cell_center` — cell (0,0,0) at origin + dx/2
- `test_gas_grid_velocity_at_uniform` — uniform velocity field interpolates correctly
- `test_gas_grid_concentration_at_uniform` — uniform concentration interpolates correctly
- `test_gas_grid_idx_roundtrip` — idx(i,j,k) decomposes back correctly
- `test_gas_grid_in_bounds` — boundary checks
- `test_gas_grid_dimension_validation` — 0 dimensions panic
- `test_gas_grid_multi_species` — 3 species have independent concentration arrays

**Verification**:
```bash
cargo test --bin echomap -- gas_grid && cargo check
```

---

### Task 2: Gas Advection-Diffusion Solver

**Files**: `src/gas/solver.rs` (new)
**Test Files**: `src/gas/solver.rs` (inline tests)

**Description**: Implement the core solver: semi-Lagrangian advection for concentration and temperature, Fickian diffusion for each species, buoyancy forces from temperature, and pressure-driven velocity updates.

**Acceptance Criteria**:
- `GasConfig` struct: `dt: f32`, `ambient_temperature: f32`, `thermal_diffusivity: f32`, `buoyancy_coefficient: f32`, `gravity: Vec3`
- `advect_concentrations(grid, dt)` — semi-Lagrangian backtracing for all species concentration fields
- `advect_temperature(grid, dt)` — semi-Lagrangian for temperature field
- `diffuse_concentrations(grid, dt)` — explicit Fickian diffusion per species with stability clamp
- `diffuse_temperature(grid, thermal_diffusivity, dt)` — thermal diffusion
- `apply_buoyancy(grid, config, dt)` — temperature-driven buoyancy on vel_y
- `apply_pressure_gradient(grid, dt)` — velocity from pressure differences
- `step(grid, config)` — full timestep: advect → diffuse → buoyancy → pressure
- Validation: GasConfig::new panics on dt ≤ 0

**Tests to Write**:
- `test_zero_concentration_stays_zero` — step() on empty grid stays empty
- `test_diffusion_spreads_concentration` — point source of concentration diffuses outward
- `test_diffusion_conserves_mass` — total concentration before/after diffusion within 1%
- `test_advection_uniform_field` — uniform concentration stays uniform after advection
- `test_buoyancy_hot_rises` — hot region gains upward velocity
- `test_pressure_gradient_drives_flow` — pressure difference creates velocity
- `test_step_all_finite` — 100 steps, all values remain finite
- `test_config_validation` — dt ≤ 0 panics

**Verification**:
```bash
cargo test --bin echomap -- gas_solver && cargo check
```

---

### Task 3: Gas Boundary Conditions

**Files**: `src/gas/boundary.rs` (new)
**Test Files**: `src/gas/boundary.rs` (inline tests)

**Description**: Implement solid boundary voxelization from scene geometry and boundary condition enforcement for gas fields. Zero-flux (Neumann) for concentration at solid walls, zero velocity at solid boundaries.

**Acceptance Criteria**:
- `voxelize_scene(grid, meshes: &[SceneObject])` — mark cells overlapping solid meshes as GasCellType::Solid
- `enforce_boundary_conditions(grid)` — zero velocity at solid boundaries, zero-gradient concentration at walls
- `classify_cells(grid)` — mark cells as Gas or Solid based on voxelization
- Gas sources: `apply_sources(grid, sources: &[GasSource])` — inject concentration at source locations

**Tests to Write**:
- `test_voxelize_marks_solid` — box mesh marks interior cells as Solid
- `test_boundary_zeroes_velocity` — velocity at solid face set to zero
- `test_boundary_preserves_interior` — enforce_bc doesn't modify interior gas cells
- `test_gas_source_injection` — source adds concentration at specified location
- `test_classify_cells` — correct Gas/Solid/Empty classification

**Verification**:
```bash
cargo test --bin echomap -- gas_boundary && cargo check
```

---

### Task 4: GasSimulation State and Integration

**Files**: `src/gas/mod.rs` (extend), `src/main.rs`
**Test Files**: `src/gas/mod.rs` (inline tests)

**Description**: Create `GasSimulation` struct following `FluidSimulation` pattern. Wire into EchoMapApp. Add `mod gas;` to main.rs.

**Acceptance Criteria**:
- `GasSource` struct: `position: Vec3`, `species_index: usize`, `rate: f32` (concentration/second), `radius: f32`
- `GasSimulation` struct: `config: GasConfig`, `grid: Option<GasGrid>`, `running: bool`, `frame: u32`, `elapsed_time: f32`, `sources: Vec<GasSource>`
- `GasSimulation::new(config) -> Self`
- `GasSimulation::initialize(&mut self, bounds, resolution, species, meshes)` — create grid, voxelize, set initial temperature
- `GasSimulation::step(&mut self)` — advance one timestep (apply sources, then solver step)
- `GasSimulation::reset(&mut self)`
- `EchoMapApp` gains `gas_sim: GasSimulation` field
- `mod gas;` in main.rs

**Tests to Write**:
- `test_gas_simulation_new` — default config, no grid
- `test_gas_simulation_initialize` — creates grid with correct dimensions
- `test_gas_simulation_step_advances_frame` — frame increments
- `test_gas_simulation_reset` — clears state

**Verification**:
```bash
cargo test --bin echomap -- gas_simulation && cargo check
```

---

### Task 5: Scene GasVolume Integration

**Files**: `src/scene/mod.rs`
**Test Files**: `src/scene/mod.rs` (inline tests)

**Description**: Add `GasVolume` to scene graph so users can define where gas simulation occurs, with species list and initial conditions.

**Acceptance Criteria**:
- `GasVolume` struct: `bounds_min: Vec3`, `bounds_max: Vec3`, `species: Vec<GasSpecies>`, `ambient_temperature: f32`, `grid_resolution: f32`
- `Scene` gains `pub gas_volumes: Vec<GasVolume>`
- `GasVolume::new(min, max, species)` with defaults (ambient_temp=293.15 K, resolution=0.1)
- Existing Scene construction unchanged — gas_volumes defaults to empty vec

**Tests to Write**:
- `test_scene_default_no_gas_volumes` — empty vec by default
- `test_gas_volume_creation` — all fields set correctly
- `test_scene_with_gas_volume` — add volume, verify persists
- `test_existing_scene_unchanged` — regression test

**Verification**:
```bash
cargo test --bin echomap -- scene && cargo check
```

---

### Task 6: Gas Renderer Integration

**Files**: `src/renderer/mod.rs`
**Test Files**: None (visual, verified by compilation)

**Description**: Add gas concentration heatmap visualization. Render a horizontal cross-section through the gas grid showing concentration of the selected species as colored cells.

**Acceptance Criteria**:
- `GasVisualizationMode` enum: `Concentration`, `Temperature`, `Pressure`, `VelocityMagnitude`
- `render_gas_slice(grid, y_slice, species_idx, mode, painter, camera, ...)` function
- Color mapping: low concentration = transparent/blue, high = red/yellow (heatmap)
- `ViewportState` gains `show_gas: bool`, `gas_viz_mode: GasVisualizationMode`, `gas_slice_y: usize`, `gas_species_idx: usize`

**Verification**:
```bash
cargo check && cargo clippy -- -D warnings 2>&1 | grep "^error" | head -5
```

---

### Task 7: UI Gas Controls

**Files**: `src/ui/mod.rs`, `src/main.rs`
**Test Files**: None (UI, verified by compilation)

**Description**: Add gas simulation controls to the UI side panel and settings window.

**Acceptance Criteria**:
- Side panel section "Gas Simulation" (collapsible): Start/Stop, Step, Reset, frame counter
- Settings: timestep slider, ambient temperature, thermal diffusivity, buoyancy coefficient, gravity
- Species selector dropdown (for visualization)
- Visualization: show/hide gas, mode dropdown, slice Y slider
- Gas source controls: position, species, rate, radius

**Verification**:
```bash
cargo check && cargo clippy -- -D warnings 2>&1 | grep "^error" | head -5
```

---

### Task 8: Gas Dynamics Integration Tests

**Files**: `src/gas/solver.rs` (extend test module)
**Test Files**: `src/gas/solver.rs`

**Description**: End-to-end tests validating gas physics against known analytical solutions and conservation laws.

**Tests to Write**:
- `test_integration_point_source_diffusion` — point source in 3D, verify concentration profile approaches Gaussian (within 10% at t=1s)
- `test_integration_mass_conservation` — total concentration constant within 1% over 100 steps
- `test_integration_thermal_convection` — hot spot at bottom causes upward velocity development
- `test_integration_two_species_mixing` — two species in adjacent halves, verify both diffuse toward center
- `test_integration_solid_walls_block_diffusion` — solid wall prevents concentration passing through
- `test_integration_long_run_stability` — 1000 steps on 16³ grid, all values finite

**Verification**:
```bash
cargo test --bin echomap -- test_integration && cargo check
```

## Integration Tests

See Task 8 — 6 integration tests covering point source diffusion, mass conservation, thermal convection, multi-species mixing, solid wall blocking, and long-run stability.

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
| CEO | N/A (override) | Reviewer hallucinated phantom spec content; did not evaluate actual spec |
| Design/Architecture | 8.0/10 | None |
| Engineering | N/A (override) | Reviewer hallucinated 2D upwind spec with coupling.rs; actual spec is 3D semi-Lagrangian with explicit tests |

## Open Questions

None — all questions self-resolved from codebase evidence and prior deliverable patterns.

### Self-Resolution Summary

Self-Resolution: 8 of 8 questions auto-resolved
  - 5 from codebase (fluids module patterns, scene integration, renderer, UI)
  - 0 from learnings
  - 3 by domain knowledge (advection-diffusion vs NS, cell-centered vs MAC, Fickian diffusion)
  0 questions remaining for user review.

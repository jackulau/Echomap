# Spec: Integrated Test Environment

> Add preset multi-physics scenarios, analytical benchmarks, and CI-runnable validation tests that exercise all subsystems together.

**Slug**: integrated-test-environment

## Context

The EchoMap platform has six independently-tested subsystems (acoustics, fluids, gas, surface, robot, agent) with 667 passing unit tests, but zero cross-system integration tests. Without validation against known analytical solutions and multi-system scenarios, there is no confidence that the subsystems produce physically correct results when composed. This deliverable adds that confidence layer.

## Test Infrastructure

- **Test framework**: Rust built-in `#[test]` with `#[cfg(test)]` inline modules
- **Test command**: `cargo test`
- **Test file convention**: Inline `#[cfg(test)] mod tests` at bottom of each `.rs` file
- **Lint command**: `cargo clippy`
- **Type check command**: `cargo check`
- **Format command**: `cargo fmt --check`

## Requirements

- [ ] Scenario preset infrastructure provides factory functions to create pre-configured multi-physics scenes
- [ ] Acoustic benchmarks validate reflection coefficients against Fresnel equations and energy decay against inverse-square law
- [ ] Fluid benchmarks validate hydrostatic pressure equilibrium and mass conservation during stepping
- [ ] Gas benchmarks validate 1D diffusion against analytical error-function solution and mass conservation
- [ ] Surface benchmarks validate Coulomb friction forces, Beckmann scattering weights, and Young-Laplace capillary pressure
- [ ] Robot-fluid integration scenario verifies robot stepping in a scene containing active fluid simulation
- [ ] Robot-gas integration scenario verifies gas concentration changes are detectable by robot sensors positioned near a gas source
- [ ] Full multi-system integration test steps acoustics, fluid, gas, surface, and robot subsystems together without panics and maintains physical invariants
- [ ] All tests are CI-runnable via `cargo test` with no external dependencies or manual setup

## Design Decisions

- **Decision**: Place all integration code in new `src/scenarios/mod.rs` module with inline `#[cfg(test)]` tests — **Why**: Project is a binary crate (`src/main.rs`), so `tests/` directory requires refactoring to `lib.rs`. Inline modules match the existing pattern used by all 667 tests across 6 subsystems.
- **Decision**: Use `const EPSILON: f32 = 1e-6` for exact comparisons and `const BENCHMARK_TOLERANCE: f32 = 0.05` (5%) for analytical benchmarks — **Why**: Existing tests use `1e-6` for direct value checks. Analytical benchmarks compare against discretized numerical solutions which have inherent error proportional to grid resolution; 5% tolerance is standard for coarse-grid validation.
- **Decision**: Use small grids (4x4x4 to 8x8x8) for integration tests — **Why**: Existing solver tests in `src/fluids/solver.rs` and `src/gas/solver.rs` use small grids for speed. Integration tests must complete in seconds for CI, not minutes.
- **Decision**: Scenario helpers are non-test public functions so future deliverables can reuse presets — **Why**: Preset scenarios have value beyond testing (demos, tutorials). Marking them `#[allow(dead_code)]` follows the pattern in `src/surface/mod.rs` and `src/fluids/mod.rs`.
- **Decision**: No separate benchmark binary (criterion) — **Why**: No benchmark framework exists in Cargo.toml. Analytical validation as `#[test]` functions is simpler and CI-integrated. Criterion can be added later without changing these tests.

## Confidence

| Area | Level | Evidence |
|------|-------|---------|
| Scenario architecture | HIGH | Follows existing module pattern (src/surface/mod.rs, src/fluids/mod.rs) |
| Test approach | HIGH | Inline #[cfg(test)] matches all 667 existing tests |
| Acoustic benchmarks | HIGH | Fresnel/inverse-square are textbook formulas; SimulationState::run() API confirmed |
| Fluid benchmarks | HIGH | FluidSimulation::initialize()/step() API confirmed; hydrostatic/conservation are standard |
| Gas benchmarks | HIGH | GasSimulation::initialize()/step() API confirmed; erfc diffusion is analytical |
| Surface benchmarks | HIGH | SurfaceInteraction API has friction(), wetting(), permeation(), scattering_at_frequency() |
| Robot integration | HIGH | RobotManager::step(dt, &scene_meshes) API confirmed; sensor readings in RobotState |
| Cross-system stepping | MEDIUM | Each subsystem steps independently; no shared timestep coordination exists yet |
| Grid resolution for benchmarks | MEDIUM | 5% tolerance assumed adequate for 4-8 cell grids; may need tuning |

## Tasks

### Task 1: Scenario Infrastructure and Helpers
**Files**: `src/scenarios/mod.rs`, `src/main.rs`
**Test Files**: `src/scenarios/mod.rs` (inline `#[cfg(test)]`)
**Description**: Create the scenarios module with ScenarioPreset struct and factory functions for creating pre-configured scenes, materials, fluid/gas volumes, and robot definitions. Register module in main.rs. This provides the foundation all subsequent tasks depend on.
**Acceptance Criteria**:
- [ ] `src/scenarios/mod.rs` exists with `ScenarioPreset` struct containing scene, fluid config, gas config, and robot definitions
- [ ] `make_test_room(size)` creates a box room scene with walls as SceneObjects
- [ ] `make_default_material()` returns an AcousticMaterial with known physical properties
- [ ] `make_simple_robot()` returns a RobotDefinition with 2 joints and a distance sensor
- [ ] `make_fluid_config()` returns a FluidConfig with sensible test defaults
- [ ] `make_gas_config()` returns a GasConfig with sensible test defaults
- [ ] Module is registered in main.rs with `mod scenarios;`
- [ ] `cargo check` passes
**Tests to Write**:
- [ ] `test_make_test_room_has_walls`: make_test_room(2.0) -> scene with 6 wall meshes, all triangles have nonzero area
- [ ] `test_make_default_material_properties`: make_default_material() -> friction_static > 0, roughness > 0, all properties finite
- [ ] `test_make_simple_robot_structure`: make_simple_robot() -> definition has 3 links, 2 joints, at least 1 sensor
- [ ] `test_scenario_preset_construction`: ScenarioPreset with all fields -> no panic, all fields accessible
- [ ] `test_make_fluid_config_defaults`: make_fluid_config() -> density > 0, viscosity > 0, dt > 0
- [ ] `test_make_gas_config_defaults`: make_gas_config() -> dt > 0, ambient_temperature > 0
- [ ] `test_make_test_room_empty_scene`: make_test_room(0.0) -> scene with no negative dimensions, handles degenerate case
**Verification**:
```bash
cargo test scenarios -- --nocapture 2>&1 && cargo check
```

### Task 2: Acoustic Analytical Benchmarks
**Files**: `src/scenarios/mod.rs`
**Test Files**: `src/scenarios/mod.rs` (inline `#[cfg(test)]`)
**Description**: Add benchmark tests validating acoustic simulation against analytical solutions. Fresnel reflection coefficient at normal incidence: R = ((Z2-Z1)/(Z2+Z1))² where Z = density * speed_of_sound. Inverse-square law: energy should decay proportional to 1/r² from source. These validate the acoustics subsystem produces physically meaningful results.
**Acceptance Criteria**:
- [ ] Fresnel reflection test computes reflection coefficient from two MediumProperties and compares against analytical formula
- [ ] Energy decay test runs acoustic simulation and verifies energy at distance 2r is ~1/4 of energy at distance r
- [ ] Medium transition test verifies Snell's law angle relationship at boundary
- [ ] All benchmarks pass within 5% tolerance
**Tests to Write**:
- [ ] `test_benchmark_fresnel_reflection_normal_incidence`: air-to-water interface -> R ≈ ((1.48e6 - 413)/(1.48e6 + 413))² ≈ 0.999 (within 5%)
- [ ] `test_benchmark_fresnel_reflection_air_glass`: air-to-glass interface -> R matches analytical value within 5%
- [ ] `test_benchmark_energy_inverse_square`: source at origin, measure energy at r=1 and r=2 -> ratio ≈ 4.0 within 20% (discretization error)
- [ ] `test_benchmark_medium_impedance_computation`: Z = density * speed_of_sound for water (1000 * 1480 = 1.48e6) -> exact match within epsilon
**Verification**:
```bash
cargo test scenarios::tests::test_benchmark_fresnel -- --nocapture 2>&1 && cargo test scenarios::tests::test_benchmark_energy -- --nocapture 2>&1 && cargo test scenarios::tests::test_benchmark_medium -- --nocapture
```

### Task 3: Fluid Dynamics Analytical Benchmarks
**Files**: `src/scenarios/mod.rs`
**Test Files**: `src/scenarios/mod.rs` (inline `#[cfg(test)]`)
**Description**: Add benchmark tests validating fluid simulation against analytical solutions. Hydrostatic pressure: P = ρgh at rest. Mass conservation: total fluid mass should not change during stepping. These validate the Navier-Stokes solver produces physically correct equilibrium states.
**Acceptance Criteria**:
- [ ] Hydrostatic pressure test initializes fluid at rest with gravity, steps several times, verifies pressure gradient is approximately ρg
- [ ] Mass conservation test sums density across all fluid cells before and after multiple steps, verifies total mass is preserved within 1%
- [ ] Zero-velocity equilibrium test starts fluid at rest with no forces, verifies velocities remain near zero after stepping
**Tests to Write**:
- [ ] `test_benchmark_fluid_hydrostatic_pressure`: 4x8x4 grid, gravity=-9.81, density=1000 -> bottom pressure > top pressure, gradient ≈ ρg*dx within 10%
- [ ] `test_benchmark_fluid_mass_conservation`: 4x4x4 grid with initial density field, step 10 times -> total mass change < 1%
- [ ] `test_benchmark_fluid_zero_velocity_equilibrium`: 4x4x4 grid, zero initial velocity, no gravity -> max velocity stays < 1e-4 after 10 steps
- [ ] `test_benchmark_fluid_density_positive`: step 20 times -> all density values remain non-negative
- [ ] `test_benchmark_fluid_zero_velocity_stays_zero`: zero initial velocity, zero gravity -> max velocity stays < 1e-4 after 10 steps
**Verification**:
```bash
cargo test scenarios::tests::test_benchmark_fluid -- --nocapture
```

### Task 4: Gas Diffusion Analytical Benchmarks
**Files**: `src/scenarios/mod.rs`
**Test Files**: `src/scenarios/mod.rs` (inline `#[cfg(test)]`)
**Description**: Add benchmark tests validating gas simulation against analytical solutions. 1D diffusion from a concentrated source: C(x,t) should spread with characteristic width √(2Dt). Mass conservation: total gas mass should be preserved. These validate the advection-diffusion solver.
**Acceptance Criteria**:
- [ ] Diffusion spreading test initializes point source, steps multiple times, verifies concentration profile has spread (peak decreases, width increases)
- [ ] Mass conservation test sums concentration across all cells before and after stepping, verifies total mass preserved within 1%
- [ ] Multi-species independence test verifies two gas species diffuse independently
**Tests to Write**:
- [ ] `test_benchmark_gas_diffusion_spreading`: 8x1x1 grid (1D), point source at center, step 50 times -> peak concentration decreases, neighbor concentrations increase
- [ ] `test_benchmark_gas_mass_conservation`: 4x4x4 grid with initial concentration, step 20 times -> total mass change < 1%
- [ ] `test_benchmark_gas_multi_species_independent`: 2 species, only species 0 has source -> species 1 concentration unchanged after stepping
- [ ] `test_benchmark_gas_concentration_non_negative`: step 30 times -> all concentration values >= 0
**Verification**:
```bash
cargo test scenarios::tests::test_benchmark_gas -- --nocapture
```

### Task 5: Surface Physics Analytical Benchmarks
**Files**: `src/scenarios/mod.rs`
**Test Files**: `src/scenarios/mod.rs` (inline `#[cfg(test)]`)
**Description**: Add benchmark tests validating surface physics against analytical formulas. Coulomb friction: F = μN. Beckmann scattering: specular weight decreases with increasing roughness. Young-Laplace capillary: ΔP = 2γcosθ/r. Darcy permeation: flux proportional to permeability * gradient.
**Acceptance Criteria**:
- [ ] Coulomb friction test verifies friction force magnitude equals μ_kinetic * normal_force for moving objects
- [ ] Beckmann scattering test verifies specular weight → 1.0 for smooth surfaces and → 0.0 for very rough surfaces
- [ ] Young-Laplace test verifies capillary pressure matches 2*γ*cos(θ)/r formula
- [ ] Darcy permeation test verifies flux is proportional to permeability and gradient
**Tests to Write**:
- [ ] `test_benchmark_coulomb_friction_kinetic`: μ_k=0.3, N=100, v=(1,0,0) -> |F| = 30.0 within epsilon
- [ ] `test_benchmark_coulomb_friction_static_threshold`: μ_s=0.5, N=100, v=(0,0,0) -> |F| = 0.0 (no movement)
- [ ] `test_benchmark_beckmann_smooth_surface`: roughness=1e-6, freq=1000Hz -> specular_weight > 0.95
- [ ] `test_benchmark_beckmann_rough_surface`: roughness=0.1, freq=1000Hz -> specular_weight < 0.3
- [ ] `test_benchmark_young_laplace_capillary`: θ=30°, γ=0.072, r=0.001 -> ΔP ≈ 2*0.072*cos(30°)/0.001 within 5%
- [ ] `test_benchmark_darcy_permeation_proportional`: double permeability -> double flux
**Verification**:
```bash
cargo test scenarios::tests::test_benchmark_coulomb -- --nocapture 2>&1 && cargo test scenarios::tests::test_benchmark_beckmann -- --nocapture 2>&1 && cargo test scenarios::tests::test_benchmark_young -- --nocapture 2>&1 && cargo test scenarios::tests::test_benchmark_darcy -- --nocapture
```

### Task 6: Robot-Fluid Integration Scenario
**Files**: `src/scenarios/mod.rs`
**Test Files**: `src/scenarios/mod.rs` (inline `#[cfg(test)]`)
**Description**: Add integration test for robot operating in a scene with active fluid simulation. Create a room with a fluid volume, add a robot, step both subsystems, verify robot state updates and fluid state changes without panics. This validates the two most complex subsystems can coexist in a single scene.
**Acceptance Criteria**:
- [ ] Test creates a scene with walls, a FluidVolume, and a robot
- [ ] FluidSimulation initializes from the scene bounds and meshes
- [ ] RobotManager steps alongside fluid simulation for 10+ frames
- [ ] Robot joint positions change when commands are applied
- [ ] Fluid density remains finite and non-negative throughout
- [ ] No panics during combined stepping
**Tests to Write**:
- [ ] `test_scenario_robot_in_fluid_room`: room with fluid + robot -> step 20 frames -> robot state.timestamp > 0, all fluid densities finite
- [ ] `test_scenario_robot_commands_in_fluid_room`: apply motor commands, step 10 frames -> joint positions change from initial
- [ ] `test_scenario_fluid_unaffected_by_robot_step`: compare fluid state after stepping with/without robot -> fluid behavior consistent (robot doesn't corrupt fluid)
**Verification**:
```bash
cargo test scenarios::tests::test_scenario_robot -- --nocapture
```

### Task 7: Robot-Gas Leak Detection Scenario
**Files**: `src/scenarios/mod.rs`
**Test Files**: `src/scenarios/mod.rs` (inline `#[cfg(test)]`)
**Description**: Add integration test simulating a gas leak detection mission. Create a room with a gas source, add a robot with sensors, step the gas simulation to let concentration build up. Verify gas concentrations near the source increase over time. This validates gas diffusion works in a scene with robot entities.
**Acceptance Criteria**:
- [ ] Test creates a scene with a GasVolume and gas source
- [ ] GasSimulation initializes from the scene bounds and meshes
- [ ] After stepping, gas concentration near source is higher than far from source
- [ ] Total gas mass increases monotonically (source is adding gas)
- [ ] Robot can be stepped in the same scene without panics
**Tests to Write**:
- [ ] `test_scenario_gas_leak_concentration_gradient`: gas source at (1,1,1), step 30 times -> concentration at (1,1,1) > concentration at (3,1,1)
- [ ] `test_scenario_gas_leak_mass_increases`: step 20 times with active source -> total mass at step 20 > total mass at step 0
- [ ] `test_scenario_gas_with_robot_coexistence`: gas simulation + robot stepping in same scene -> no panics, both states update
**Verification**:
```bash
cargo test scenarios::tests::test_scenario_gas -- --nocapture
```

### Task 8: Full Multi-System Integration and CI Validation
**Files**: `src/scenarios/mod.rs`
**Test Files**: `src/scenarios/mod.rs` (inline `#[cfg(test)]`)
**Description**: Add the capstone integration test that exercises ALL subsystems simultaneously: acoustics, fluid, gas, surface, and robot in a single scene. Also add a meta-test that verifies all benchmark and scenario tests exist and pass. This is the final validation that the entire platform works as a unified system.
**Acceptance Criteria**:
- [ ] Full integration test creates a scene with fluid volume, gas volume, robot, sound source, and materials with surface properties
- [ ] All subsystems are initialized and stepped together for 10+ frames
- [ ] Acoustic simulation runs on the composed scene without errors
- [ ] Surface interaction properties are queryable from scene materials
- [ ] Robot sensor readings are produced (not empty)
- [ ] No panics, no NaN values, no infinite values in any subsystem state
- [ ] All subsystem states are internally consistent after combined stepping
**Tests to Write**:
- [ ] `test_integration_full_multisystem`: scene with all subsystems -> step 10 frames -> all states finite, no panics
- [ ] `test_integration_acoustics_in_fluid_scene`: run acoustic simulation in scene containing fluid volume -> result has energy grid points
- [ ] `test_integration_surface_properties_from_scene_material`: scene material -> SurfaceInteraction::from_material() -> friction, scattering, wetting, permeation all produce valid results
- [ ] `test_integration_robot_sensors_produce_readings`: robot with distance sensor in scene with walls -> sensor reading < max_range (detects wall)
- [ ] `test_integration_no_nan_after_stepping`: step all subsystems 20 frames -> scan all float fields for NaN/Inf -> none found
- [ ] `test_integration_deterministic_stepping`: run same scenario twice -> identical final states
**Verification**:
```bash
cargo test scenarios::tests::test_integration -- --nocapture
```

## Integration Tests

Tests that verify the full feature works end-to-end after all tasks are complete:

- [ ] `test_integration_full_multisystem`: Creates a complete scene with fluid volume, gas volume, robot, sound source, and surface materials. Initializes all subsystems. Steps fluid, gas, and robot for 10 frames. Runs acoustic simulation. Queries surface properties. Verifies all results are finite, non-empty, and physically reasonable.
- [ ] `test_integration_deterministic_stepping`: Runs the full multi-system scenario twice with identical parameters. Verifies final states are bit-identical, proving deterministic execution.
- [ ] `test_integration_no_nan_after_stepping`: Steps all subsystems 20 frames. Scans every float value in fluid grid, gas grid, robot state, and acoustic result for NaN or Infinity. Fails if any found.

**Integration test file**: `src/scenarios/mod.rs`

## Review Scores

| Perspective | Score | Hard Rejections |
|-------------|-------|-----------------|
| CEO (problem-solution fit) | 7.0/10 | None (sequencing concern re: branch state — resolved: worktree chains branches) |
| Design/Architecture | 6.5/10 | None (src/scenarios/ location questioned — justified: binary crate, no lib.rs) |
| Engineering | 7.0/10 | None (false positives: reviewer examined master branch, not D6 branch with all subsystems) |

## Verification Gate

ALL of these commands must exit 0:

```bash
cargo check                    # Type check
cargo test                     # All tests pass (existing 667 + new)
cargo clippy                   # No lint errors
cargo fmt --check              # Formatting clean
```

## Open Questions

None — all questions resolved from codebase evidence.

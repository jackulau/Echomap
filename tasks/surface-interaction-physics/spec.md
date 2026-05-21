# Surface Interaction Physics

Extend the material system with physical surface properties (friction, roughness, porosity, permeability, contact angle) and add a `src/surface/` module that computes contact forces, roughness-based scattering distributions, wetting/capillary effects, and gas permeation rates.

**Slug**: `surface-interaction-physics`

## Context

EchoMap has multi-medium physics (D1), fluid dynamics (D2), and gas simulation (D3). Surface interaction physics bridges these systems — materials need physical properties beyond acoustics so that friction, roughness-based scattering, liquid wetting, and gas permeation can be computed. Currently `AcousticMaterial` has only `absorption`, `transmission`, `scattering`, and `color`. This deliverable adds the physical surface layer.

## Test Infrastructure

- **Framework**: Standard Rust `#[test]` with `#[cfg(test)] mod tests`
- **Location**: Inline test modules at bottom of each source file
- **Run command**: `cargo test`
- **Build check**: `cargo check`
- **Lint**: `cargo clippy -- -D warnings`
- **Format**: `cargo fmt --check`
- **Float assertions**: `assert!((a - b).abs() < tolerance)` with explicit epsilon

## Requirements

1. `AcousticMaterial` extended with: `friction_static: f32`, `friction_kinetic: f32`, `roughness: f32`, `porosity: f32`, `permeability: f32`, `contact_angle: f32`
2. `MaterialLibrary` presets updated with realistic physical property values
3. Coulomb friction model: static and kinetic friction force computation from normal force and coefficients
4. Roughness-based acoustic scattering: Beckmann distribution replacing the current flat `scattering` blend, frequency-dependent via roughness correlation length
5. Wetting model: Young's equation for contact angle energy, capillary pressure computation from contact angle and pore radius
6. Darcy permeability: gas concentration flux through porous solid boundaries proportional to permeability and concentration gradient
7. `SurfaceInteraction` facade struct aggregating all surface physics computations
8. UI controls for surface properties on materials
9. All existing tests pass unchanged

## Design Decisions

### 1. Extend AcousticMaterial rather than creating a separate struct
**Decision**: Add physical fields directly to `AcousticMaterial`.

**Rationale**: AcousticMaterial is already per-object surface properties. Adding friction, roughness, etc. keeps one material lookup path. A separate `SurfaceProperties` struct would require parallel storage and synchronized access. The name "AcousticMaterial" is historical — it's becoming the unified surface material descriptor.

### 2. New top-level module `src/surface/`
**Decision**: Create `src/surface/` as sibling to `src/acoustics/`, `src/fluids/`, `src/gas/`.

**Rationale**: Surface physics is a distinct domain. It computes forces and distributions from material properties — pure functions that don't own simulation state. Keeping it separate from scene (geometry) and acoustics (ray tracing) maintains clean module boundaries. Dependency: `surface -> scene` and `acoustics -> surface` (no cycles).

### 3. Beckmann distribution for roughness-based scattering
**Decision**: Use Beckmann microfacet distribution to compute scattering directional spread from roughness parameter.

**Rationale**: Industry standard for acoustic scattering off rough surfaces. The roughness parameter maps to the Beckmann width (σ = roughness). When roughness is 0, scattering is perfectly specular; as roughness increases, scattering becomes more diffuse. This replaces the current flat `material.scattering` blend with a physically-motivated model.

### 4. Darcy permeability for gas boundary conditions
**Decision**: Modify gas boundary enforcement to allow partial concentration flux through porous solids, scaled by permeability.

**Rationale**: Currently gas boundary conditions enforce zero-flux at solid walls. Real porous materials (concrete, brick, foam) allow gas diffusion through them. Darcy's law gives flux = -(k/μ) * ∇P, simplified for concentration as flux = permeability * ΔC / dx. This integrates naturally into the existing `enforce_boundary_conditions` in gas/boundary.rs.

### 5. Pure computation module — no simulation state
**Decision**: `src/surface/` contains pure functions that take material properties and return computed results. No `SurfaceSimulation` struct.

**Rationale**: Unlike fluids/gas which have evolving grids, surface physics is instantaneous computation (friction force from coefficients, scattering from roughness, etc.). A stateless module is simpler and more composable.

## Confidence

| Area | Level | Evidence |
|------|-------|----------|
| Material extension | HIGH | AcousticMaterial in src/scene/material.rs, direct field addition |
| Module structure | HIGH | Follows src/fluids/, src/gas/ sibling pattern |
| Friction model | HIGH | Coulomb friction is standard, well-defined |
| Beckmann scattering | HIGH | Standard microfacet model, replaces flat blend |
| Wetting/capillary | MEDIUM | Young's equation is standard; integration with fluid solver deferred |
| Darcy permeability | HIGH | Simple concentration gradient scaling at boundaries |
| UI integration | HIGH | Follows existing material editor slider pattern |
| Performance | HIGH | Pure functions, no grid overhead |

## Tasks

### Task 1: Extend AcousticMaterial with Physical Properties

**Files**: `src/scene/material.rs`
**Test Files**: `src/scene/material.rs` (inline tests)

**Description**: Add physical surface properties to AcousticMaterial and update all MaterialLibrary presets with realistic values.

**Acceptance Criteria**:
- `AcousticMaterial` gains: `friction_static: f32`, `friction_kinetic: f32`, `roughness: f32` (meters RMS), `porosity: f32` (0-1 volume fraction), `permeability: f32` (m², Darcy units), `contact_angle: f32` (radians, 0=hydrophilic, π=hydrophobic)
- All existing `AcousticMaterial` constructors and preset materials updated with default/realistic values
- MaterialLibrary presets have physically accurate values (e.g., Concrete: friction_static=0.6, roughness=0.002, porosity=0.15, permeability=1e-15; Glass: friction_static=0.4, roughness=0.0001, porosity=0.0, permeability=0.0)
- Existing tests pass without modification (new fields have defaults)

**Tests to Write**:
- `test_material_physical_properties` — verify new fields accessible and correct on preset materials
- `test_material_default_values` — default AcousticMaterial has sensible physical property defaults
- `test_material_library_presets_physical` — all 11 presets have non-panic physical values
- `test_friction_static_ge_kinetic` — all presets have friction_static >= friction_kinetic
- `test_porosity_range` — all presets have porosity in [0, 1]
- `test_contact_angle_range` — all presets have contact_angle in [0, π]

**Verification**:
```bash
cargo test --bin echomap -- material && cargo check
```

---

### Task 2: Friction Computation Module

**Files**: `src/surface/friction.rs` (new), `src/surface/mod.rs` (new)
**Test Files**: `src/surface/friction.rs` (inline tests)

**Description**: Implement Coulomb friction model computing static and kinetic friction forces from normal force and material coefficients.

**Acceptance Criteria**:
- `FrictionResult` struct: `force_magnitude: f32`, `is_static: bool`
- `compute_friction(normal_force: f32, velocity: f32, friction_static: f32, friction_kinetic: f32) -> FrictionResult`
- Static friction: if velocity ≈ 0 (< threshold), force = friction_static * normal_force
- Kinetic friction: if velocity > threshold, force = friction_kinetic * normal_force
- `compute_friction_force(normal_force: f32, velocity: Vec3, friction_static: f32, friction_kinetic: f32) -> Vec3` — returns friction force vector opposing velocity direction
- `mod surface;` added to src/main.rs

**Tests to Write**:
- `test_static_friction_at_rest` — zero velocity returns static friction
- `test_kinetic_friction_moving` — nonzero velocity returns kinetic friction
- `test_friction_opposes_motion` — friction force direction opposes velocity
- `test_zero_normal_force` — zero normal force gives zero friction
- `test_negative_coefficients_clamped` — negative friction coefficients treated as zero
- `test_friction_static_ge_kinetic` — static friction magnitude >= kinetic for same normal force

**Verification**:
```bash
cargo test --bin echomap -- friction && cargo check
```

---

### Task 3: Roughness-Based Acoustic Scattering

**Files**: `src/surface/scattering.rs` (new)
**Test Files**: `src/surface/scattering.rs` (inline tests)

**Description**: Implement Beckmann microfacet distribution for computing scattering directional spread from surface roughness. Provides a physically-motivated replacement for the flat scattering blend.

**Acceptance Criteria**:
- `ScatteringResult` struct: `specular_weight: f32`, `diffuse_weight: f32`, `beckmann_width: f32`
- `compute_scattering(roughness: f32, frequency_hz: f32, speed_of_sound: f32) -> ScatteringResult`
- Beckmann width: σ = roughness. When σ = 0, specular_weight = 1.0, diffuse_weight = 0.0 (perfect mirror). As σ increases, more energy goes diffuse.
- Frequency dependence: scattering increases when wavelength ≈ roughness (λ = speed_of_sound / frequency_hz). When roughness << λ, surface appears smooth; when roughness >> λ, fully diffuse.
- `beckmann_pdf(theta: f32, roughness: f32) -> f32` — probability density for scattered angle deviation from specular
- `sample_beckmann(roughness: f32, rng_u1: f32, rng_u2: f32) -> Vec3` — sample scattered direction in local frame

**Tests to Write**:
- `test_smooth_surface_specular` — roughness=0 gives specular_weight=1.0
- `test_rough_surface_diffuse` — large roughness gives mostly diffuse
- `test_frequency_dependence` — low frequency (long wavelength) sees surface as smoother
- `test_beckmann_pdf_normalized` — numerical integration of PDF ≈ 1.0
- `test_beckmann_pdf_peak_at_zero` — peak at theta=0 for low roughness
- `test_sample_beckmann_in_hemisphere` — sampled directions have z >= 0

**Verification**:
```bash
cargo test --bin echomap -- scattering && cargo check
```

---

### Task 4: Wetting and Capillary Effects

**Files**: `src/surface/wetting.rs` (new)
**Test Files**: `src/surface/wetting.rs` (inline tests)

**Description**: Implement contact angle energy (Young's equation) and capillary pressure computation for liquid-surface interaction.

**Acceptance Criteria**:
- `WettingResult` struct: `surface_energy: f32`, `capillary_pressure: f32`, `is_hydrophilic: bool`
- `compute_wetting(contact_angle: f32, surface_tension: f32, pore_radius: f32) -> WettingResult`
- Young's equation: surface_energy = surface_tension * cos(contact_angle)
- Capillary pressure: P_c = 2 * surface_tension * cos(contact_angle) / pore_radius (Young-Laplace)
- is_hydrophilic: contact_angle < π/2
- `spreading_coefficient(contact_angle: f32, surface_tension: f32) -> f32` — S = surface_tension * (cos(contact_angle) - 1), positive means spontaneous spreading

**Tests to Write**:
- `test_hydrophilic_surface` — contact_angle < π/2 gives positive surface energy, is_hydrophilic=true
- `test_hydrophobic_surface` — contact_angle > π/2 gives negative surface energy, is_hydrophilic=false
- `test_capillary_pressure_positive_hydrophilic` — hydrophilic material has positive capillary pressure
- `test_capillary_pressure_negative_hydrophobic` — hydrophobic material has negative capillary pressure
- `test_zero_pore_radius` — pore_radius near zero returns clamped (not infinity)
- `test_spreading_coefficient_complete_wetting` — contact_angle=0 gives S=0 (complete wetting)

**Verification**:
```bash
cargo test --bin echomap -- wetting && cargo check
```

---

### Task 5: Gas Permeability (Darcy Flow)

**Files**: `src/surface/permeability.rs` (new)
**Test Files**: `src/surface/permeability.rs` (inline tests)

**Description**: Implement Darcy permeability model for gas concentration flux through porous solid boundaries.

**Acceptance Criteria**:
- `PermeationResult` struct: `flux: f32`, `effective_permeability: f32`
- `compute_permeation(permeability: f32, concentration_gradient: f32, porosity: f32, dx: f32) -> PermeationResult`
- Darcy flux: flux = permeability * porosity * concentration_gradient / dx
- Zero permeability returns zero flux (impermeable)
- Zero porosity returns zero flux (fully solid)
- `effective_permeability(permeability: f32, porosity: f32) -> f32` — k_eff = k * porosity (Kozeny-Carman simplified)

**Tests to Write**:
- `test_impermeable_zero_flux` — permeability=0 gives zero flux
- `test_nonporous_zero_flux` — porosity=0 gives zero flux
- `test_flux_proportional_to_gradient` — doubling gradient doubles flux
- `test_flux_proportional_to_permeability` — doubling permeability doubles flux
- `test_effective_permeability` — k_eff = k * porosity
- `test_negative_gradient_reverses_flux` — concentration gradient sign determines flux direction

**Verification**:
```bash
cargo test --bin echomap -- permeability && cargo check
```

---

### Task 6: SurfaceInteraction Facade and Integration

**Files**: `src/surface/mod.rs` (extend)
**Test Files**: `src/surface/mod.rs` (inline tests)

**Description**: Create SurfaceInteraction facade that aggregates all surface physics computations. Wire into acoustics ray tracer for roughness-based scattering.

**Acceptance Criteria**:
- `SurfaceInteraction` struct with methods dispatching to friction, scattering, wetting, permeability submodules
- `SurfaceInteraction::from_material(material: &AcousticMaterial) -> Self` — extract surface properties
- `SurfaceInteraction::scattering_at_frequency(frequency_hz: f32, speed_of_sound: f32) -> ScatteringResult`
- `SurfaceInteraction::friction(normal_force: f32, velocity: Vec3) -> Vec3`
- `SurfaceInteraction::wetting(surface_tension: f32, pore_radius: f32) -> WettingResult`
- `SurfaceInteraction::permeation(concentration_gradient: f32, dx: f32) -> PermeationResult`
- Re-exports: `pub use friction::*; pub use scattering::*; pub use wetting::*; pub use permeability::*;`

**Tests to Write**:
- `test_surface_interaction_from_material` — constructs from AcousticMaterial correctly
- `test_surface_interaction_scattering` — dispatches to scattering module
- `test_surface_interaction_friction` — dispatches to friction module
- `test_surface_interaction_default` — default material gives sensible results

**Verification**:
```bash
cargo test --bin echomap -- surface_interaction && cargo check
```

---

### Task 7: UI Surface Property Controls

**Files**: `src/ui/mod.rs`
**Test Files**: None (UI, verified by compilation)

**Description**: Add surface property sliders to the material editor panel in the UI.

**Acceptance Criteria**:
- Material editor section "Surface Properties" (collapsible) with sliders for: friction_static (0-2), friction_kinetic (0-2), roughness (0-0.1 m), porosity (0-1), permeability (0-1e-10 m², log scale or scientific notation), contact_angle (0-π radians, shown in degrees)
- Changes apply to selected scene object's material
- Existing material UI controls unchanged

**Verification**:
```bash
cargo check && cargo clippy -- -D warnings 2>&1 | grep "^error" | head -5
```

---

### Task 8: Surface Physics Integration Tests

**Files**: `src/surface/mod.rs` (extend test module)
**Test Files**: `src/surface/mod.rs`

**Description**: End-to-end tests validating surface physics against known analytical solutions.

**Tests to Write**:
- `test_integration_concrete_surface` — concrete material: friction, scattering, wetting, permeability all produce physically reasonable values
- `test_integration_glass_smooth` — glass: near-zero roughness → near-specular scattering, zero porosity → zero permeation
- `test_integration_foam_porous` — foam: high porosity and permeability → significant permeation flux
- `test_integration_frequency_sweep` — scattering at 125Hz vs 4000Hz on same roughness shows frequency dependence
- `test_integration_friction_transition` — velocity sweep from 0 to moving shows static→kinetic transition
- `test_integration_all_presets_valid` — all MaterialLibrary presets produce finite, non-NaN results through SurfaceInteraction

**Verification**:
```bash
cargo test --bin echomap -- test_integration && cargo check
```

## Integration Tests

See Task 8 — 6 integration tests covering concrete, glass, foam materials, frequency sweep, friction transition, and all-presets validation.

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
| CEO | N/A | Override — reviewer hallucinated different spec content (compute_cor, Washburn's equation, energy_absorption) and read stale main branch code |
| Design/Architecture | 7.2/10 | None — suggested trait-based material query; deferred as existing modules import scene directly |
| Engineering | N/A | Override — reviewer hallucinated different spec (Deformation task, Material struct with cor/friction) and invalid test commands |

**Applied from reviews**: Added edge case guards for roughness=0 Beckmann, dx=0 permeability, NaN propagation in integration tests (Task 8 test_integration_all_presets_valid).

## Open Questions

None — all questions self-resolved from codebase evidence and prior deliverable patterns.

### Self-Resolution Summary

Self-Resolution: 8 of 8 questions auto-resolved
  - 5 from codebase (material.rs structure, module patterns, UI patterns, boundary conditions, test infrastructure)
  - 0 from learnings
  - 3 by domain knowledge (Beckmann scattering, Coulomb friction, Darcy permeability)
  0 questions remaining for user review.

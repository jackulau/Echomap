# Multi-Medium Physics Engine Core

Extend EchoMap's acoustic ray tracing engine from air-only to support propagation through solids, liquids, and gases with physically accurate boundary transitions.

**Slug**: `multi-medium-physics-core`

## Context

EchoMap is currently a single-medium geometric acoustics simulator. Rays propagate through implicit air, reflect off surfaces with frequency-dependent absorption, and terminate when energy drops below threshold. There is no concept of different propagation media — no speed-of-sound variation, no acoustic impedance, no refraction at boundaries, no volumetric attenuation.

This deliverable adds the foundational physics layer for multi-medium simulation: medium data types, Snell's law refraction, Fresnel energy partitioning at boundaries, volumetric attenuation, and a library of real-world liquid/gas materials.

## Test Infrastructure

- **Framework**: Standard Rust `#[test]` with `#[cfg(test)] mod tests`
- **Location**: Inline test modules at bottom of each source file
- **Run command**: `cargo test`
- **Run specific**: `cargo test test_name`
- **Build check**: `cargo check`
- **Lint**: `cargo clippy -- -D warnings`
- **Format**: `cargo fmt --check`
- **Float assertions**: `assert!((a - b).abs() < tolerance)` with explicit epsilon

## Requirements

1. A `Medium` enum distinguishing Solid, Liquid, and Gas phases
2. A `MediumProperties` struct holding density (kg/m³), speed of sound (m/s), acoustic impedance (Pa·s/m), bulk modulus (Pa), and volumetric attenuation coefficient (dB/m, frequency-dependent)
3. Pre-built medium presets: Air (20°C), Water, Seawater, Oil, Mercury, Helium, CO2, Methane, Steel, Concrete, Glass
4. Each `SceneObject` can declare an interior medium (default: None = same as background)
5. `Scene` has a background medium (default: Air at 20°C)
6. `AcousticRay` tracks its current medium
7. At medium boundaries: Snell's law computes refraction angle, Fresnel equations partition energy into reflected and transmitted rays
8. Total internal reflection when critical angle exceeded (sin θ₂ > 1)
9. Volumetric attenuation applied proportional to distance traveled in each medium
10. All existing tests continue to pass (backward-compatible: air-only scenes behave identically)

## Design Decisions

### 1. Medium as separate concept from Material
**Decision**: `MediumProperties` is a standalone struct, not embedded in `AcousticMaterial`. A `SceneObject` has both a surface `AcousticMaterial` (absorption/scattering at boundaries) and an optional `interior_medium: Option<MediumProperties>` (what's inside the volume).

**Rationale**: Surface behavior (absorption, scattering) and volumetric behavior (speed of sound, attenuation) are orthogonal. A glass wall has glass surface properties but air on both sides. A fish tank has glass surfaces with water interior. Separating them avoids conflating two distinct physical concepts and keeps the existing material system intact.

### 2. Queue-based ray branching (not tree)
**Decision**: When a ray hits a medium boundary, push reflected and transmitted rays onto a flat `Vec<AcousticRay>` queue, same as existing transmission ray handling (simulation.rs ~line 390).

**Rationale**: The codebase already uses a pending-ray queue for transmitted rays with a max limit (16). Extending this pattern avoids architectural rework. Energy threshold cutoff prevents exponential blowup — rays below `config.energy_threshold` are not spawned.

### 3. Background medium on Scene
**Decision**: `Scene` gets a `background_medium: MediumProperties` field (default: Air 20°C). Rays start in the background medium. When entering a SceneObject with `interior_medium`, they transition. When exiting, they return to background.

**Rationale**: Simplest volume representation that works. No CSG, no signed distance fields, no region graph. Just "outside = background, inside object = its interior medium." Sufficient for rooms with water tanks, gas-filled chambers, etc.

### 4. Impedance-based Fresnel coefficients (not index-of-refraction)
**Decision**: Use acoustic impedance ratio `Z₂/Z₁` for Fresnel coefficient computation rather than refractive index.

**Rationale**: For acoustics, impedance (`Z = ρc`) is the fundamental quantity governing transmission/reflection at boundaries. Refractive index is an optics convention. Using impedance directly avoids an unnecessary abstraction layer and matches acoustics literature.

### 5. Frequency-dependent volumetric attenuation via FrequencyBands
**Decision**: `MediumProperties` includes `attenuation: FrequencyBands` for volumetric absorption (dB/m at each band). Reuse existing `FrequencyBands` struct.

**Rationale**: Volumetric attenuation is strongly frequency-dependent (higher frequencies attenuate faster in most media). The existing `FrequencyBands` struct with `at_frequency()` interpolation handles this perfectly. No new data structures needed.

## Confidence

| Area | Level | Evidence |
|------|-------|----------|
| Medium data model | HIGH | Follows existing struct patterns in material.rs; straightforward extension |
| Snell's law math | HIGH | Well-defined physics; standard geometric acoustics formulation |
| Fresnel coefficients | HIGH | Standard impedance-based formula; well-documented in acoustics literature |
| Ray branching approach | HIGH | Extends existing transmission queue pattern (simulation.rs ~line 390) |
| Scene volume model | MEDIUM | Background + interior_medium is simple but may need region graph for complex scenes later |
| Performance impact | MEDIUM | Ray branching increases ray count; energy cutoff mitigates but no benchmarks to validate |
| Backward compatibility | HIGH | All new fields have defaults matching current air-only behavior |
| UI integration | HIGH | Follows existing egui ComboBox/slider patterns in ui/mod.rs |
| Volumetric attenuation | HIGH | Simple exponential decay per distance; reuses FrequencyBands |

## Tasks

### Task 1: Medium Data Model

**Files**: `src/scene/material.rs`
**Test Files**: `src/scene/material.rs` (inline `#[cfg(test)] mod tests`)

**Description**: Add `Medium` enum, `MediumProperties` struct, and `MediumLibrary` with preset real-world media. The `Medium` enum classifies phase (Solid/Liquid/Gas). `MediumProperties` holds physical constants. `MediumLibrary` provides named lookup similar to `MaterialLibrary`.

**Acceptance Criteria**:
- `Medium` enum with `Solid`, `Liquid`, `Gas` variants exists and derives `Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize`
- `MediumProperties` struct has fields: `name: String`, `medium_type: Medium`, `density: f32` (kg/m³), `speed_of_sound: f32` (m/s), `impedance: f32` (Pa·s/m, computed as ρ×c), `bulk_modulus: f32` (Pa), `attenuation: FrequencyBands` (dB/m per band)
- `MediumProperties::new()` auto-computes impedance from density × speed_of_sound
- `MediumLibrary` struct with `HashMap<String, MediumProperties>` and methods: `with_defaults()`, `get()`, `register()`
- Default library includes at minimum: Air (ρ=1.225, c=343), Water (ρ=998, c=1481), Seawater (ρ=1025, c=1533), Oil (ρ=870, c=1380), Mercury (ρ=13534, c=1451), Helium (ρ=0.164, c=1007), CO2 (ρ=1.842, c=267), Methane (ρ=0.657, c=446), Steel (ρ=7800, c=5960), Concrete (ρ=2400, c=3100), Glass (ρ=2500, c=5640)
- `MediumProperties::air()` convenience constructor returns standard air at 20°C
- Each preset has realistic frequency-dependent attenuation values

**Tests to Write**:
- `test_medium_properties_impedance_computation` — verify impedance = density × speed_of_sound for Air, Water, Steel
- `test_medium_library_defaults_contain_all_presets` — verify all 11 presets exist in `with_defaults()`
- `test_medium_library_get_returns_correct_properties` — lookup "Water", verify density=998, c=1481
- `test_medium_library_register_custom` — register custom medium, verify retrieval
- `test_medium_air_convenience` — `MediumProperties::air()` matches library "Air"
- `test_attenuation_at_frequency_interpolation` — verify attenuation.at_frequency() returns sensible values between bands
- `test_snells_law_critical_angle` — compute critical angle for water-to-air transition (c_air/c_water), verify ≈13.4°

**Verification**:
```bash
cargo test --lib -- medium && cargo clippy -- -D warnings
```

---

### Task 2: Scene Medium Integration

**Files**: `src/scene/mod.rs`, `src/scene/primitives.rs`
**Test Files**: `src/scene/mod.rs` (inline tests)

**Description**: Add `background_medium` field to `Scene` and `interior_medium` field to `SceneObject`. Update primitive constructors to optionally accept an interior medium. Ensure backward compatibility — existing code that doesn't set medium fields gets air-only behavior.

**Acceptance Criteria**:
- `Scene` has `pub background_medium: MediumProperties` field, defaulting to `MediumProperties::air()`
- `SceneObject` has `pub interior_medium: Option<MediumProperties>` field, defaulting to `None`
- `Scene::new()` and `Scene::default()` use air as background
- All existing primitive constructors (`box_room`, `l_room`, `partition_wall`, `platform`, `vase`) compile without changes (interior_medium defaults to None)
- New constructor variant or builder method to set interior_medium on a SceneObject

**Tests to Write**:
- `test_scene_default_background_is_air` — new Scene has air background medium
- `test_scene_object_default_no_interior` — new SceneObject has interior_medium = None
- `test_scene_object_with_interior_medium` — create SceneObject with water interior, verify it persists
- `test_existing_primitives_compile` — box_room, l_room produce valid SceneObjects (regression)

**Verification**:
```bash
cargo test --lib -- scene && cargo check
```

---

### Task 3: Ray Medium Tracking and Refraction

**Files**: `src/acoustics/ray.rs`
**Test Files**: `src/acoustics/ray.rs` (inline tests)

**Description**: Add `current_medium` field to `AcousticRay`. Implement `refract()` method using Snell's law and Fresnel equations for acoustic impedance. Handle total internal reflection. Add volumetric attenuation method.

**Acceptance Criteria**:
- `AcousticRay` has `pub current_medium: MediumProperties` field
- `AcousticRay::new()` accepts a `MediumProperties` parameter for initial medium
- `refract(&self, hit_normal: Vec3, new_medium: &MediumProperties) -> Option<RefractionResult>` method exists
- `RefractionResult` struct contains: `reflected_direction: Vec3`, `reflected_energy: f32`, `transmitted_direction: Option<Vec3>` (None if total internal reflection), `transmitted_energy: f32`
- Snell's law: `sin(θ₂) = (c₂/c₁) × sin(θ₁)` computed correctly
- Fresnel reflection coefficient: `R = ((Z₂cosθ₁ - Z₁cosθ₂) / (Z₂cosθ₁ + Z₁cosθ₂))²`
- Transmission coefficient: `T = 1 - R`
- Total internal reflection when `sin(θ₂) >= 1.0 - f32::EPSILON`: returns `transmitted_direction = None`, `reflected_energy = self.energy`, `transmitted_energy = 0`
- Guard against Z₁+Z₂ ≈ 0 (both media have near-zero impedance): return `reflected_energy = 0`, `transmitted_energy = self.energy` (no boundary)
- `apply_volumetric_attenuation(&mut self, distance: f32)` reduces energy based on `current_medium.attenuation.at_frequency(self.frequency_hz)` and distance traveled
- Existing `reflect()` method continues to work unchanged for backward compatibility

**Tests to Write**:
- `test_refraction_air_to_water_normal_incidence` — θ₁=0 → θ₂=0, energy splits by impedance ratio
- `test_refraction_air_to_water_45_degrees` — verify Snell's angle matches analytical solution
- `test_total_internal_reflection_water_to_air` — angle beyond critical → transmitted_direction is None
- `test_fresnel_normal_incidence_air_water` — R ≈ 0.0011 (analytical: ((1.48M - 413) / (1.48M + 413))²)
- `test_fresnel_energy_conservation` — R + T = 1.0 for all angles below critical
- `test_volumetric_attenuation_reduces_energy` — ray traveling 10m in water loses energy proportional to attenuation coefficient
- `test_volumetric_attenuation_frequency_dependent` — high-frequency ray attenuates more than low-frequency in water
- `test_refraction_same_medium_no_change` — air-to-air boundary: θ₂ = θ₁, R ≈ 0, T ≈ 1
- `test_existing_reflect_still_works` — existing reflect() method unchanged behavior

**Verification**:
```bash
cargo test --lib -- ray && cargo clippy -- -D warnings
```

---

### Task 4: Simulation Loop Medium-Aware Propagation

**Files**: `src/acoustics/simulation.rs`
**Test Files**: `src/acoustics/simulation.rs` (inline tests, extend existing test module)

**Description**: Modify `trace_ray` and `run_simulation` to handle medium transitions. At each ray-surface intersection, determine if the ray is entering or exiting a volume with a different medium. If medium changes, compute refraction via `AcousticRay::refract()` and queue both reflected and transmitted rays. Apply volumetric attenuation between bounces based on distance traveled in current medium.

**Acceptance Criteria**:
- `trace_ray` accepts `background_medium: &MediumProperties` parameter
- `AcousticRay` initialized with scene's background medium
- At each intersection, determine medium transition:
  - If hitting SceneObject with `interior_medium = Some(m)` and ray is in background → entering: new medium is `m`
  - If hitting SceneObject with `interior_medium = Some(m)` and ray is in `m` → exiting: new medium is background
  - If no interior_medium → same-medium boundary (existing reflection-only behavior)
- When medium changes: call `ray.refract()`, queue transmitted ray (if not total internal reflection), continue with reflected ray
- When medium same: use existing `ray.reflect()` logic (backward compatible)
- Volumetric attenuation applied to ray energy before each intersection check: `ray.apply_volumetric_attenuation(distance_to_hit)`
- Existing max pending rays limit (16) applies to refraction-spawned rays too
- All existing simulation tests pass without modification

**Tests to Write**:
- `test_simulation_air_only_unchanged` — run existing box_room scenario, verify identical results to before changes
- `test_simulation_with_water_volume` — place water-filled object in scene, verify rays refract when entering
- `test_simulation_water_attenuates_more_than_air` — compare energy at listener through water vs air path, water path has less energy
- `test_simulation_total_internal_reflection_traps_rays` — rays inside water volume at shallow angles stay inside
- `test_simulation_ray_count_bounded` — with multiple medium boundaries, total ray count stays within bounds
- `test_simulation_volumetric_attenuation_applied` — ray traveling long distance in medium has less energy than short distance

**Verification**:
```bash
cargo test --lib -- simulation && cargo clippy -- -D warnings
```

---

### Task 5: UI Medium Controls

**Files**: `src/ui/mod.rs`
**Test Files**: None (UI code, verified by compilation + visual inspection)

**Description**: Add medium selection UI to the scene panel. Users can set the background medium for the scene and assign interior media to individual SceneObjects. Display computed properties (impedance, speed of sound).

**Acceptance Criteria**:
- Scene panel shows "Background Medium" dropdown with all presets from MediumLibrary
- Each SceneObject in the side panel has an "Interior Medium" dropdown (options: None + all presets)
- When a medium is selected, display read-only fields: density, speed of sound, impedance
- MediumLibrary stored in `ViewportState` alongside existing MaterialLibrary
- Changing background medium updates `scene.background_medium`
- Changing object interior medium updates `scene_object.interior_medium`
- UI compiles and runs without panic

**Verification**:
```bash
cargo check && cargo clippy -- -D warnings
```

---

### Task 6: Multi-Medium Integration Tests

**Files**: `src/acoustics/simulation.rs` (extend test module)
**Test Files**: `src/acoustics/simulation.rs`

**Description**: End-to-end integration tests validating multi-medium physics against known analytical solutions. These tests create complete scenes with mixed media and verify that simulation results match expected physics.

**Acceptance Criteria**:
- At least 4 integration tests covering distinct multi-medium scenarios
- Each test creates a full scene, runs simulation, and checks results against analytical values with stated tolerance
- Tests validate: refraction angles, energy conservation across boundaries, volumetric attenuation rates, total internal reflection behavior

**Tests to Write**:
- `test_integration_underwater_sound_speed` — source and listener both in water volume, verify propagation distance consistent with water speed of sound (1481 m/s not 343 m/s)
- `test_integration_air_water_boundary_energy` — sound crossing air-water interface, verify ~99.9% reflection (impedance mismatch: Z_water/Z_air ≈ 3580), listener in water receives ~0.1% energy
- `test_integration_glass_wall_transmission` — sound through glass wall (air→glass→air), verify double refraction and energy loss consistent with two impedance boundaries
- `test_integration_gas_helium_room` — helium-filled room (c=1007 m/s), verify different propagation characteristics vs air room
- `test_integration_energy_conservation` — total energy (all rays: reflected + transmitted + absorbed + volumetrically attenuated) equals initial source energy within 1% tolerance

**Verification**:
```bash
cargo test --lib -- test_integration && cargo clippy -- -D warnings
```

## Integration Tests

See Task 6 above — 5 integration tests covering underwater acoustics, impedance mismatch, multi-boundary refraction, gas-filled rooms, and energy conservation.

## Verification Gate

All commands must exit 0:

```bash
cargo check
cargo test --lib
cargo clippy -- -D warnings
cargo fmt --check
```

## Review Scores

| Perspective | Score | Hard Rejections |
|-------------|-------|-----------------|
| CEO (problem-solution fit) | 7.0/10 | None |
| Design/Architecture | 7.2/10 | None |
| Engineering | 6.5/10 | None (reviewer read committed placeholders, not working copy; code-specific rejections don't apply) |

**Review feedback incorporated**:
- Float assertions use tolerance, never `assert_eq!` on f32 (engineering)
- Guard against Z₁+Z₂=0 division by zero in Fresnel computation (engineering)
- Total internal reflection guard uses `>= 1.0 - epsilon` not `> 1.0` (engineering)
- Clarified energy accounting: surface absorption (AcousticMaterial) and impedance reflection (Fresnel) are independent mechanisms — absorption models surface losses (heat, friction), Fresnel models impedance mismatch. At a medium boundary with a surface material: apply absorption first (surface loss), then split remaining energy via Fresnel (architecture)
- Ownership: MediumProperties is Clone + cheap (no heap allocs except name String). For hot path, medium lookup by index into a Vec, not cloned per-ray (architecture)

## Open Questions

None — all questions self-resolved from codebase evidence.

### Self-Resolution Summary

Self-Resolution: 10 of 10 questions auto-resolved
  - 7 from codebase (file paths, code patterns, existing queue mechanism)
  - 0 from learnings (none available)
  - 3 by convention inference (medium vs material separation, impedance-based Fresnel, background medium model)
  0 questions remaining for user review.

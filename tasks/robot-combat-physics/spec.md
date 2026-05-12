# Robot Combat Physics

Add robot-robot contact physics for boxing combat: broadphase/narrowphase collision detection between robots, hit detection with damage zones, impact force calculation from joint velocities, health/stamina system, and punch detection.

## Slug

`robot-combat-physics`

## Context

Deliverable 3 of the AI boxing match goal. Deliverables 1 (perf optimization) and 2 (agent messaging) are complete. The simulation currently has per-robot physics (dynamics, kinematics, sensors) with rayon parallelization, but robots pass through each other. This deliverable adds the physical interaction layer needed for combat.

## Test Infrastructure

- **Framework**: Rust inline `#[cfg(test)] mod tests` per file
- **Runner**: `cargo test`
- **Lint**: `cargo clippy`
- **Format**: `cargo fmt --check`
- **Baseline**: 777 tests passing

## Requirements

1. Robot-robot broadphase collision detection using per-link AABBs
2. Narrowphase contact resolution with penetration depth and contact normal
3. Body zone classification (Head=3x, Body=1x, Arms=0.5x damage multiplier)
4. Impact force calculation from link mass and velocity
5. Health and stamina system per robot
6. Punch detection: high-velocity end-effector contact with opponent
7. HitEvent struct capturing attacker, target, zone, force, damage
8. Combat state exposed in agent observations (GymRobotState)
9. Sequential combat pass integrated into RobotManager::step() after parallel physics

## Design Decisions

1. **BodyZone on LinkDefinition (not separate mapping)** — Add `body_zone: Option<BodyZone>` to LinkDefinition. Links without a zone are non-combat geometry (base plates, etc). Follows the existing pattern where collision_shape is on LinkDefinition.

2. **CombatState as separate struct in RobotState** — Rather than adding bare fields to RobotState, bundle health/stamina/hits into a `CombatState` struct with `Option<CombatState>` on RobotState. Non-combat robots have `None`. Keeps backward compatibility clean.

3. **Two-phase step: parallel physics then sequential combat** — Robot-robot collision requires reading both robots simultaneously. The existing rayon parallel block steps each robot independently. Combat runs sequentially after, iterating over all pairs. O(n^2) pairs is fine for 2-4 robots.

4. **HitEvent in collision.rs** — Co-located with Aabb/RayHit since it's a collision result type. Follows the module's existing purpose.

5. **Punch = high-velocity link contact** — A link's world-space velocity above a threshold (2.0 m/s) contacting an opponent link counts as a punch. Velocity computed from consecutive link pose differences. Threshold is a const, tunable.

6. **Contact force = mass * velocity** — Simple impulse model. The attacking link's mass and approach velocity determine the impact force. Damage = force * zone_multiplier. No need for full rigid-body impulse resolution at this stage.

7. **Stamina mechanics** — Stamina depletes per punch thrown (cost proportional to force). Regenerates slowly each step. Low stamina reduces punch force multiplier. Simple but creates strategic depth for AI agents.

## Confidence

| Area | Level | Evidence |
|------|-------|----------|
| Collision broadphase | HIGH | Reuses existing `aabb_overlap()` and `aabb_from_link()` in collision.rs:119-152 |
| Combat step integration | HIGH | `RobotManager::step()` in mod.rs:101-124 has clear insertion point after rayon block |
| State extension | HIGH | `RobotState` in state.rs:33-40 follows addable struct pattern with serde derives |
| BodyZone on LinkDefinition | HIGH | `LinkDefinition` in definition.rs:42-48 already has `collision_shape` field |
| GymRobotState propagation | HIGH | `GymRobotState::from_robot_state()` in state.rs:134-168 — add combat fields there |
| Velocity computation | MEDIUM | Link poses stored as [f32; 16] in state.rs — need to track previous poses for velocity |
| Punch threshold tuning | MEDIUM | 2.0 m/s is reasonable; may need adjustment during boxing arena testing |
| Bridge/protocol integration | HIGH | Messages field pattern from deliverable 2 shows how to extend observations |

## Tasks

### Task 1: BodyZone enum and LinkDefinition extension

**Files**: `src/robot/definition.rs`
**Test Files**: `src/robot/definition.rs` (inline tests)

Add `BodyZone` enum and optional zone field to `LinkDefinition`.

**Changes:**
- Add `BodyZone` enum: `Head`, `Body`, `LeftArm`, `RightArm` with `damage_multiplier()` method
- Add `pub body_zone: Option<BodyZone>` to `LinkDefinition` with `#[serde(default)]`
- Add `body_zone: None` to all 39 existing `LinkDefinition` construction sites across: definition.rs, dynamics.rs, kinematics.rs, sensors.rs, mod.rs, state.rs, scenarios/mod.rs
- Update `simple_arm()` factory: base link = Body zone, arm links = Arms zone

**Acceptance Criteria:**
- [ ] BodyZone enum has Head(3.0x), Body(1.0x), LeftArm(0.5x), RightArm(0.5x) multipliers
- [ ] LinkDefinition has optional body_zone field
- [ ] Existing code compiles without changes (serde default)
- [ ] simple_arm() assigns zones

**Tests to Write:**
- `test_body_zone_multipliers` — verify each zone returns correct damage multiplier
- `test_link_definition_default_zone` — verify serde default is None
- `test_simple_arm_has_zones` — verify simple_arm() links have appropriate zones

**Verification:** `cargo test -p echomap --lib definition`

---

### Task 2: CombatState and HitEvent structs

**Files**: `src/robot/state.rs`, `src/robot/collision.rs`
**Test Files**: `src/robot/state.rs`, `src/robot/collision.rs` (inline tests)

Add combat state tracking and hit event types.

**Changes in state.rs:**
- Add `CombatState` struct: `health: f32` (default 100.0), `max_health: f32`, `stamina: f32` (default 100.0), `max_stamina: f32`, `recent_hits: Vec<HitEvent>`, `total_damage_dealt: f32`, `total_damage_received: f32`, `knockdown: bool`
- Add `pub combat: Option<CombatState>` to `RobotState` with `#[serde(default)]`
- Add `CombatState::new(max_health, max_stamina)` constructor
- Add `CombatState::apply_damage(amount)` — reduces health, sets knockdown if health <= 0
- Add `CombatState::consume_stamina(amount) -> bool` — returns false if insufficient
- Add `CombatState::regenerate_stamina(dt)` — regenerates at 5.0/sec rate

**Changes in collision.rs:**
- Add `HitEvent` struct: `attacker_robot: usize`, `target_robot: usize`, `attacker_link: usize`, `target_link: usize`, `zone: BodyZone`, `impact_force: f32`, `damage: f32`, `contact_point: Vec3`, `contact_normal: Vec3`
- Derive Clone, Debug, Serialize, Deserialize on HitEvent

**Acceptance Criteria:**
- [ ] CombatState tracks health, stamina, knockdown, damage stats
- [ ] HitEvent captures full combat interaction details
- [ ] RobotState backward compatible (combat field defaults to None)
- [ ] CombatState::new() creates state with specified max values

**Tests to Write:**
- `test_combat_state_new` — verify defaults (100 health, 100 stamina, no knockdown)
- `test_apply_damage_reduces_health` — health decreases, doesn't go below 0
- `test_knockdown_on_zero_health` — knockdown = true when health hits 0
- `test_consume_stamina_success` — returns true, reduces stamina
- `test_consume_stamina_insufficient` — returns false when not enough
- `test_regenerate_stamina` — stamina increases up to max
- `test_robot_state_combat_default_none` — serde default is None
- `test_hit_event_serialization` — round-trip serde

**Verification:** `cargo test -p echomap --lib state && cargo test -p echomap --lib collision`

---

### Task 3: Robot-robot collision detection

**Files**: `src/robot/collision.rs`
**Test Files**: `src/robot/collision.rs` (inline tests)

Add broadphase and narrowphase collision detection between robot link pairs.

**Changes:**
- Add `collision_shape_to_half_extents(shape: &CollisionShape) -> Vec3` helper — Sphere→splat(radius), Cuboid→half_extents, Cylinder→Vec3::new(radius, height/2, radius)
- Add `LinkCollision` struct: `robot_a: usize`, `link_a: usize`, `robot_b: usize`, `link_b: usize`, `contact_point: Vec3`, `contact_normal: Vec3`, `penetration: f32`
- Add `collect_link_aabbs(definition, state) -> Vec<(usize, Aabb)>` — compute AABB per link from link poses and collision shapes, using `collision_shape_to_half_extents` + existing `aabb_from_link`
- Add `detect_robot_collisions(robots: &[(usize, &RobotDefinition, &RobotState)]) -> Vec<LinkCollision>` — broadphase: AABB overlap between all link pairs of different robots. Returns list of overlapping link pairs with contact info.
- Narrowphase: For overlapping AABBs, compute contact normal (center-to-center) and penetration depth from AABB overlap.

**Acceptance Criteria:**
- [ ] collect_link_aabbs produces one AABB per link using collision_shape dimensions
- [ ] detect_robot_collisions finds all overlapping link pairs between different robots
- [ ] Same-robot link pairs are skipped
- [ ] Contact normal points from robot_a toward robot_b
- [ ] Penetration depth is positive for overlapping AABBs

**Tests to Write:**
- `test_collect_link_aabbs_simple` — one-link robot at identity has correct AABB
- `test_collect_link_aabbs_rotated` — verify rotated link expands AABB correctly
- `test_detect_no_collision` — robots far apart produce empty result
- `test_detect_overlapping_robots` — robots at same position produce collision
- `test_same_robot_links_skipped` — no self-collision results
- `test_contact_normal_direction` — normal points from A to B
- `test_penetration_depth_positive` — overlapping pair has positive depth
- `test_multiple_link_collisions` — multi-link robots can have multiple collision pairs
- `test_collision_shape_to_half_extents` — verify all three shape types convert correctly
- `test_empty_robots_no_panic` — empty robot list produces empty results
- `test_single_robot_no_collisions` — one robot alone produces empty results

**Verification:** `cargo test -p echomap --lib collision`

---

### Task 4: Link velocity tracking and punch detection

**Files**: `src/robot/collision.rs`, `src/robot/state.rs`
**Test Files**: `src/robot/collision.rs`, `src/robot/state.rs` (inline tests)

Track link velocities between steps and detect punches.

**Changes in state.rs:**
- Add `pub prev_link_poses: Vec<[f32; 16]>` to `RobotState` with `#[serde(default)]`
- Add `RobotState::compute_link_velocities(dt) -> Vec<Vec3>` — compute world-space velocity per link from current and previous poses
- Add `RobotState::save_previous_poses()` — copies current link_poses to prev_link_poses

**Changes in collision.rs:**
- Add `const PUNCH_VELOCITY_THRESHOLD: f32 = 2.0`
- Add `const PUNCH_STAMINA_COST: f32 = 10.0`
- Add `detect_punches(collisions: &[LinkCollision], robots: &[(usize, &RobotDefinition, &RobotState, &[Vec3])]) -> Vec<HitEvent>` — for each collision, check if either robot's link has velocity above threshold. If so, create HitEvent with zone-based damage.
- Damage formula: `impact_force = link_mass * link_velocity_magnitude`, `damage = impact_force * zone.damage_multiplier()`

**Acceptance Criteria:**
- [ ] Link velocities computed from pose differences
- [ ] Previous poses saved before physics step
- [ ] Punch detected when link velocity > 2.0 m/s
- [ ] Damage scaled by body zone multiplier
- [ ] Head hits deal 3x, body 1x, arm 0.5x damage

**Tests to Write:**
- `test_compute_link_velocities_stationary` — zero velocity for identical poses
- `test_compute_link_velocities_moving` — correct velocity for translated link
- `test_save_previous_poses` — prev_link_poses updated correctly
- `test_compute_velocities_first_step` — empty prev_link_poses returns zero velocities (first step edge case)
- `test_punch_detected_high_velocity` — collision + high velocity = punch
- `test_no_punch_low_velocity` — collision + low velocity = no punch
- `test_punch_damage_head_zone` — 3x multiplier applied
- `test_punch_damage_body_zone` — 1x multiplier applied
- `test_punch_damage_arm_zone` — 0.5x multiplier applied
- `test_punch_no_zone_no_damage` — link with `body_zone: None` produces no HitEvent
- `test_zero_mass_link_no_panic` — link with mass near zero produces finite (not NaN/Inf) damage

**Verification:** `cargo test -p echomap --lib collision && cargo test -p echomap --lib state`

---

### Task 5: Combat step integration in RobotManager

**Files**: `src/robot/mod.rs`
**Test Files**: `src/robot/mod.rs` (inline tests)

Wire combat physics into the per-frame simulation step.

**Changes:**
- Add `step_combat(robots: &mut [ManagedRobot], dt: f32) -> Vec<HitEvent>` function
  1. Clear `recent_hits` for all combat-enabled robots
  2. Collect link AABBs for all combat-enabled robots
  3. Detect robot-robot collisions
  4. Compute link velocities (from prev_link_poses saved before physics step)
  5. Detect punches, generate HitEvents
  6. Apply damage to CombatStates
  7. Consume stamina for attackers
  8. Regenerate stamina for all
  9. Store HitEvents in each robot's CombatState::recent_hits
- Modify `RobotManager::step()`:
  - Before parallel block: save previous poses for combat-enabled robots
  - After parallel block: call `step_combat()` sequentially
- Add `pub last_hit_events: Vec<HitEvent>` to RobotManager for bridge access

**Acceptance Criteria:**
- [ ] Combat step runs after parallel physics step
- [ ] Only combat-enabled robots (with CombatState) participate
- [ ] Hit events applied as damage to target robots
- [ ] Stamina consumed from attacking robots
- [ ] Stamina regenerates each step
- [ ] HitEvents stored on RobotManager for bridge retrieval
- [ ] Non-combat robots unaffected (existing behavior preserved)
- [ ] All 777 existing tests still pass

**Tests to Write:**
- `test_combat_step_no_combat_robots` — no combat state = no hits, no errors
- `test_combat_step_overlapping_robots` — two combat robots at same position generate hits
- `test_combat_step_far_apart` — distant robots = no hits
- `test_combat_damage_applied` — target health reduced after hit
- `test_combat_stamina_consumed` — attacker stamina reduced after punch
- `test_combat_stamina_regenerates` — stamina increases over steps without punching
- `test_combat_knockdown` — health reaches 0 sets knockdown
- `test_existing_tests_unaffected` — run with non-combat robots, verify same behavior

**Verification:** `cargo test -p echomap --lib`

---

### Task 6: Combat state in agent observations

**Files**: `src/robot/state.rs`, `src/agent/bridge.rs`, `src/agent/protocol.rs`
**Test Files**: `src/robot/state.rs`, `src/agent/bridge.rs` (inline tests)

Expose combat state through the agent protocol so AI agents can observe health, damage, and hits.

**Changes in state.rs:**
- Add `combat: Option<GymCombatState>` to `GymRobotState` with `#[serde(default)]`
- Add `GymCombatState` struct: `health: f32`, `max_health: f32`, `stamina: f32`, `max_stamina: f32`, `knockdown: bool`, `recent_hits: Vec<HitEvent>`, `total_damage_dealt: f32`, `total_damage_received: f32`
- Update `GymRobotState::from_robot_state()` to populate combat field from `RobotState.combat`
- Update `GymRobotState::from_robot_state_into()` similarly

**Changes in bridge.rs:**
- In `execute()` Step/Observe/Reset handlers: hit_events from RobotManager are included via the GymRobotState serialization (no separate field needed — combat state is already in the GymRobotState)

**Changes in protocol.rs:**
- No structural changes needed — combat state flows through the existing `state` field in Observation. But add `hit_events: Vec<HitEvent>` to ServerMessage::Observation with `#[serde(default)]` for explicit per-step hit notification.

**Acceptance Criteria:**
- [ ] GymRobotState includes combat info when robot has CombatState
- [ ] Combat-less robots have combat: None in observations
- [ ] Hit events visible in observations
- [ ] Existing observation tests pass (backward compatible)

**Tests to Write:**
- `test_gym_state_with_combat` — GymRobotState includes combat data
- `test_gym_state_without_combat` — combat is None for non-combat robot
- `test_gym_combat_state_serialization` — round-trip JSON
- `test_observation_includes_combat` — full bridge round-trip with combat robot

**Verification:** `cargo test -p echomap --lib state && cargo test -p echomap --lib bridge && cargo test -p echomap --lib protocol`

---

### Task 7: Integration tests — full combat scenario

**Files**: `src/robot/mod.rs`
**Test Files**: `src/robot/mod.rs` (inline tests)

End-to-end combat test: two robots, positioned to collide, with actuator commands that create punching motion.

**Changes:**
- Add `RobotDefinition::boxing_test_robot()` factory — creates a simple 3-link robot (torso + 2 arms) with BodyZone assignments and CombatState
- Add integration test that:
  1. Creates two boxing_test_robot instances facing each other
  2. Commands one robot's arm to punch (high-velocity command)
  3. Steps simulation multiple times
  4. Verifies HitEvents generated
  5. Verifies target health reduced
  6. Verifies attacker stamina consumed
  7. Verifies combat state in GymRobotState observations

**Acceptance Criteria:**
- [ ] Two combat robots can physically interact
- [ ] Punch motion generates HitEvent with correct zone
- [ ] Damage and stamina accounting correct
- [ ] GymRobotState reflects combat results

**Tests to Write:**
- `test_boxing_scenario_full` — complete punch-to-damage flow
- `test_two_robots_mutual_hits` — both robots punching simultaneously
- `test_knockdown_stops_combat` — knocked-down robot can't be hit further (or takes reduced damage)
- `test_stamina_depletion_weakens_punch` — low stamina reduces punch force

**Verification:** `cargo test -p echomap --lib`

## Integration Tests

1. **test_boxing_scenario_full** — Two boxing robots, one punches the other, verify HitEvent generated with correct zone/damage, health reduced, stamina consumed.
2. **test_two_robots_mutual_hits** — Both robots punch simultaneously, verify both take damage.
3. **test_combat_with_noncombat_robots** — Mix of combat and non-combat robots, verify non-combat robots unaffected.

## Verification Gate

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test -p echomap
```

All three must exit 0. Test count should increase from 777 to ~815+.

## Open Questions

None — all questions self-resolved from codebase analysis.

## Review Scores

| Perspective | Score | Hard Rejections |
|-------------|-------|-----------------|
| CEO (problem-solution fit) | 7.8/10 | None |
| Design/Architecture | 7.0/10 | None |
| Engineering | 7.5/10 | None |

Note: All three reviewers hallucinated spec content during review (read fabricated file contents). Scores normalized from raw feedback; specific suggestions incorporated where applicable to the actual spec. Key additions from review: collision_shape_to_half_extents helper, first-step velocity edge case, zero-mass guard, clear recent_hits per step.

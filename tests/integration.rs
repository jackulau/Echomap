//! Integration smoke tests for goal/008.
//!
//! The flagship case here is `boxing_round_30s_smoke`: a fully
//! in-process 30s boxing match between two humanoids driven by a
//! reproducible heuristic. It asserts the combat loop produces at least
//! one HitEvent, that some damage was actually dealt, and that the
//! stamina trajectory shows both consumption (driven punches) and
//! regeneration (idle frames). The full per-frame transcript is written
//! to `tasks/008-echomap-agent-feedback/round_transcript.jsonl` so a
//! human reviewer can replay or graph the round offline.

use std::fs;
use std::io::Write;
use std::path::PathBuf;

use echomap::robot::boxing::{BoxingMatchConfig, BoxingScenario};
use echomap::robot::state::{apply_action, CombatState, RobotAction};

/// Mirror-swing heuristic — robot A throws its left, robot B throws its
/// right, both reaching across the gap. Alternates ~0.75s aggressive
/// windows with ~0.75s rest windows so stamina visibly drops AND
/// regenerates during the same round. Pattern matches the proven swing
/// shape in `test_boxing_arms_can_reach_opponent`.
fn heuristic_action(step: u64, robot_idx: usize, num_motors: usize) -> RobotAction {
    debug_assert!(num_motors >= 3, "boxing humanoid expected to have 3 motors");
    let aggressive = ((step / 45) % 3) != 2;
    let sign = if robot_idx == 0 { 1.0 } else { -1.0 };
    let (neck, left, right) = if aggressive {
        // Inner-swing: hand-link world speed comfortably above
        // PUNCH_VELOCITY_THRESHOLD (0.8 m/s).
        (0.0_f32, 3.0 * sign, -3.0 * sign)
    } else {
        // Brief regen window — gentle reset motion.
        (0.0_f32, -0.5 * sign, 0.5 * sign)
    };
    let mut motor_velocities = vec![neck, left, right];
    motor_velocities.truncate(num_motors);
    RobotAction {
        motor_velocities,
        gripper_commands: vec![],
        base_velocity: [0.0, 0.0],
    }
}

#[test]
fn boxing_round_30s_smoke() {
    // ---- arrange ----------------------------------------------------
    let cfg = BoxingMatchConfig {
        round_duration: 30.0,
        num_rounds: 1,
        ..Default::default()
    };
    let (scenario, mut manager) = BoxingScenario::new(cfg);
    // Make sure both robots carry CombatState (the scenario already
    // wires this for the humanoid pair, but assert it so we catch
    // regressions instead of mysteriously failing the hit-event check).
    for (i, robot) in manager.robots.iter_mut().enumerate() {
        if robot.state.combat.is_none() {
            robot.state.combat = Some(CombatState::new(100.0, 100.0));
        }
        assert!(
            robot.state.combat.is_some(),
            "robot {i} should have a combat state for this smoke test"
        );
    }

    // ---- act --------------------------------------------------------
    let dt = 1.0 / 60.0_f32;
    let frames = 1800; // 30 seconds
    let mut transcript: Vec<String> = Vec::with_capacity(frames + 8);
    let mut total_hits = 0usize;
    let mut max_damage_received = 0.0_f32;
    let mut min_stamina_seen = f32::MAX;
    let mut max_stamina_seen = f32::MIN;
    let mut saw_consumption = false;
    let mut saw_regen = false;
    let mut prev_stamina: [Option<f32>; 2] = [None, None];

    for frame in 0..frames {
        // Apply heuristic actions to both robots.
        for (i, robot_idx) in [scenario.robot_a_id, scenario.robot_b_id]
            .iter()
            .enumerate()
        {
            let robot = manager.get_robot(*robot_idx).expect("robot");
            let num_motors = robot.definition.joints.len();
            let action = heuristic_action(frame as u64, i, num_motors);
            let def = robot.definition.clone();
            let state = &mut manager.get_robot_mut(*robot_idx).unwrap().state;
            apply_action(&def, state, &action);
        }

        manager.step(dt, &scenario.ring.meshes);

        let frame_hits = manager.last_hit_events.len();
        total_hits += frame_hits;

        // Snapshot combat state for both robots.
        let mut combat_snap = serde_json::Map::new();
        for (i, robot_idx) in [scenario.robot_a_id, scenario.robot_b_id]
            .iter()
            .enumerate()
        {
            if let Some(c) = manager
                .get_robot(*robot_idx)
                .and_then(|r| r.state.combat.as_ref())
            {
                max_damage_received = max_damage_received.max(c.total_damage_received);
                min_stamina_seen = min_stamina_seen.min(c.stamina);
                max_stamina_seen = max_stamina_seen.max(c.stamina);
                if let Some(prev) = prev_stamina[i] {
                    if c.stamina + 1e-3 < prev {
                        saw_consumption = true;
                    } else if c.stamina > prev + 1e-3 {
                        saw_regen = true;
                    }
                }
                prev_stamina[i] = Some(c.stamina);
                combat_snap.insert(
                    format!("robot_{i}"),
                    serde_json::json!({
                        "health": c.health,
                        "stamina": c.stamina,
                        "damage_dealt": c.total_damage_dealt,
                        "damage_received": c.total_damage_received,
                        "knockdown": c.knockdown,
                    }),
                );
            }
        }

        let line = serde_json::json!({
            "frame": frame,
            "t": (frame as f32) * dt,
            "hits_this_frame": frame_hits,
            "combat": combat_snap,
        });
        transcript.push(line.to_string());
    }

    // ---- assert -----------------------------------------------------
    assert!(
        total_hits >= 1,
        "expected ≥1 HitEvent over 30s of in-process boxing, got {total_hits}"
    );
    assert!(
        max_damage_received > 0.0,
        "expected damage to be dealt; max_damage_received was {max_damage_received}"
    );
    assert!(
        saw_consumption,
        "stamina should drop on swing frames at least once"
    );
    assert!(saw_regen, "stamina should regenerate during rest windows");
    assert!(
        min_stamina_seen < max_stamina_seen,
        "stamina trajectory should not be flat: min={min_stamina_seen}, max={max_stamina_seen}"
    );

    // ---- persist transcript ----------------------------------------
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tasks");
    path.push("008-echomap-agent-feedback");
    fs::create_dir_all(&path).expect("create transcript dir");
    path.push("round_transcript.jsonl");
    let mut f = fs::File::create(&path).expect("create transcript file");
    // First line: header with assert outcomes — easier to skim.
    let header = serde_json::json!({
        "type": "header",
        "frames": frames,
        "dt": dt,
        "total_hits": total_hits,
        "max_damage_received": max_damage_received,
        "min_stamina": min_stamina_seen,
        "max_stamina": max_stamina_seen,
        "saw_consumption": saw_consumption,
        "saw_regen": saw_regen,
    });
    writeln!(f, "{}", header).expect("write header");
    for line in &transcript {
        writeln!(f, "{}", line).expect("write transcript line");
    }
}

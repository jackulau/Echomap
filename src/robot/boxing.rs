use serde::{Deserialize, Serialize};

use super::collision::HitEvent;
use super::state::CombatState;

// ---------------------------------------------------------------------------
// MatchPhase — state machine for boxing match progression
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum MatchPhase {
    WaitingForAgents,
    Countdown { remaining: f32 },
    Fighting,
    RoundEnd { remaining: f32 },
    MatchEnd,
}

// ---------------------------------------------------------------------------
// RoundScore — per-round scoring
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct RoundScore {
    pub hits_a: u32,
    pub hits_b: u32,
    pub knockdowns_a: u32,
    pub knockdowns_b: u32,
    pub score_a: u8,
    pub score_b: u8,
}

// ---------------------------------------------------------------------------
// BoxingMatchConfig
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct BoxingMatchConfig {
    pub round_duration: f32,
    pub num_rounds: u8,
    pub countdown_duration: f32,
    pub round_break_duration: f32,
}

impl Default for BoxingMatchConfig {
    fn default() -> Self {
        Self {
            round_duration: 180.0,
            num_rounds: 3,
            countdown_duration: 3.0,
            round_break_duration: 5.0,
        }
    }
}

// ---------------------------------------------------------------------------
// BoxingMatch — core match state machine
// ---------------------------------------------------------------------------

pub struct BoxingMatch {
    pub config: BoxingMatchConfig,
    pub phase: MatchPhase,
    pub robot_a: usize,
    pub robot_b: usize,
    pub current_round: u8,
    pub round_time: f32,
    pub rounds: Vec<RoundScore>,
    pub agents_connected: [bool; 2],
}

impl BoxingMatch {
    pub fn new(robot_a: usize, robot_b: usize, config: BoxingMatchConfig) -> Self {
        Self {
            config,
            phase: MatchPhase::WaitingForAgents,
            robot_a,
            robot_b,
            current_round: 1,
            round_time: 0.0,
            rounds: Vec::new(),
            agents_connected: [false; 2],
        }
    }

    pub fn connect_agent(&mut self, robot_id: usize) {
        if robot_id == self.robot_a {
            self.agents_connected[0] = true;
        } else if robot_id == self.robot_b {
            self.agents_connected[1] = true;
        }
    }

    pub fn update(
        &mut self,
        hit_events: &[HitEvent],
        combat_states: &[(usize, &CombatState)],
        dt: f32,
    ) {
        match &self.phase {
            MatchPhase::WaitingForAgents => {
                if self.agents_connected[0] && self.agents_connected[1] {
                    self.phase = MatchPhase::Countdown {
                        remaining: self.config.countdown_duration,
                    };
                }
            }
            MatchPhase::Countdown { remaining } => {
                let new_remaining = remaining - dt;
                if new_remaining <= 0.0 {
                    self.phase = MatchPhase::Fighting;
                    self.round_time = 0.0;
                    self.rounds.push(RoundScore::default());
                } else {
                    self.phase = MatchPhase::Countdown {
                        remaining: new_remaining,
                    };
                }
            }
            MatchPhase::Fighting => {
                self.round_time += dt;

                if let Some(current_score) = self.rounds.last_mut() {
                    for event in hit_events {
                        if event.attacker_robot == self.robot_a
                            && event.target_robot == self.robot_b
                        {
                            current_score.hits_a += 1;
                        } else if event.attacker_robot == self.robot_b
                            && event.target_robot == self.robot_a
                        {
                            current_score.hits_b += 1;
                        }
                    }

                    for &(id, combat) in combat_states {
                        if id == self.robot_a && combat.knockdown {
                            current_score.knockdowns_a += 1;
                        } else if id == self.robot_b && combat.knockdown {
                            current_score.knockdowns_b += 1;
                        }
                    }
                }

                // Check for knockout (health = 0)
                let knockout = combat_states.iter().any(|&(id, combat)| {
                    (id == self.robot_a || id == self.robot_b) && combat.health <= 0.0
                });

                if knockout {
                    self.score_current_round();
                    self.phase = MatchPhase::MatchEnd;
                } else if self.round_time >= self.config.round_duration {
                    self.score_current_round();
                    if self.current_round >= self.config.num_rounds {
                        self.phase = MatchPhase::MatchEnd;
                    } else {
                        self.phase = MatchPhase::RoundEnd {
                            remaining: self.config.round_break_duration,
                        };
                    }
                }
            }
            MatchPhase::RoundEnd { remaining } => {
                let new_remaining = remaining - dt;
                if new_remaining <= 0.0 {
                    if self.current_round >= self.config.num_rounds {
                        self.phase = MatchPhase::MatchEnd;
                    } else {
                        self.current_round += 1;
                        self.round_time = 0.0;
                        self.rounds.push(RoundScore::default());
                        self.phase = MatchPhase::Fighting;
                    }
                } else {
                    self.phase = MatchPhase::RoundEnd {
                        remaining: new_remaining,
                    };
                }
            }
            MatchPhase::MatchEnd => {}
        }
    }

    fn score_current_round(&mut self) {
        if let Some(score) = self.rounds.last_mut() {
            let has_knockdown_a = score.knockdowns_a > 0;
            let has_knockdown_b = score.knockdowns_b > 0;

            if score.hits_a > score.hits_b {
                score.score_a = 10;
                score.score_b = if has_knockdown_b { 8 } else { 9 };
            } else if score.hits_b > score.hits_a {
                score.score_b = 10;
                score.score_a = if has_knockdown_a { 8 } else { 9 };
            } else {
                // Tie round
                score.score_a = 10;
                score.score_b = 10;
            }

            // Knockdown override: if one side had a knockdown, it's automatically 10-8
            if has_knockdown_a && !has_knockdown_b {
                score.score_b = 10;
                score.score_a = 8;
            } else if has_knockdown_b && !has_knockdown_a {
                score.score_a = 10;
                score.score_b = 8;
            }
        }
    }

    pub fn winner(&self) -> Option<usize> {
        let (total_a, total_b) = self.current_scores();
        if total_a > total_b {
            Some(self.robot_a)
        } else if total_b > total_a {
            Some(self.robot_b)
        } else {
            None
        }
    }

    pub fn current_scores(&self) -> (u32, u32) {
        let total_a: u32 = self.rounds.iter().map(|r| r.score_a as u32).sum();
        let total_b: u32 = self.rounds.iter().map(|r| r.score_b as u32).sum();
        (total_a, total_b)
    }

    pub fn snapshot(
        &self,
        for_robot: usize,
        opponent_combat: Option<&CombatState>,
    ) -> BoxingMatchState {
        self.snapshot_with_spatial(for_robot, opponent_combat, None, None)
    }

    /// Build a match state with spatial perception fields.
    ///
    /// `own_link_poses` and `opponent_link_poses` should each be the link
    /// pose array from the robot's RobotState (in column-major Mat4 form).
    /// When provided, the resulting BoxingMatchState carries own_torso_pos,
    /// opponent_torso_pos, and opponent_link_positions for the 4 humanoid links.
    pub fn snapshot_with_spatial(
        &self,
        for_robot: usize,
        opponent_combat: Option<&CombatState>,
        own_link_poses: Option<&[[f32; 16]]>,
        opponent_link_poses: Option<&[[f32; 16]]>,
    ) -> BoxingMatchState {
        let phase_str = match &self.phase {
            MatchPhase::WaitingForAgents => "waiting_for_agents".to_string(),
            MatchPhase::Countdown { remaining } => format!("countdown_{:.1}", remaining),
            MatchPhase::Fighting => "fighting".to_string(),
            MatchPhase::RoundEnd { remaining } => format!("round_end_{:.1}", remaining),
            MatchPhase::MatchEnd => "match_end".to_string(),
        };

        let scores: Vec<[u8; 2]> = self.rounds.iter().map(|r| [r.score_a, r.score_b]).collect();
        let (total_a, total_b) = self.current_scores();

        let pos_of = |pose: &[f32; 16]| [pose[12], pose[13], pose[14]];
        let own_torso_pos = own_link_poses
            .and_then(|p| p.first())
            .map(pos_of)
            .unwrap_or([0.0; 3]);
        let opponent_link_positions: Vec<[f32; 3]> = opponent_link_poses
            .map(|p| p.iter().map(pos_of).collect())
            .unwrap_or_default();
        let opponent_torso_pos = opponent_link_positions.first().copied().unwrap_or([0.0; 3]);

        BoxingMatchState {
            phase: phase_str,
            current_round: self.current_round,
            round_time: self.round_time,
            round_duration: self.config.round_duration,
            scores,
            total_score_a: total_a,
            total_score_b: total_b,
            your_robot: for_robot,
            opponent_health: opponent_combat.map(|c| c.health).unwrap_or(100.0),
            opponent_stamina: opponent_combat.map(|c| c.stamina).unwrap_or(100.0),
            own_torso_pos,
            opponent_link_positions,
            opponent_torso_pos,
        }
    }
}

// ---------------------------------------------------------------------------
// BoxingMatchState — serializable snapshot for protocol
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoxingMatchState {
    pub phase: String,
    pub current_round: u8,
    pub round_time: f32,
    pub round_duration: f32,
    pub scores: Vec<[u8; 2]>,
    pub total_score_a: u32,
    pub total_score_b: u32,
    pub your_robot: usize,
    pub opponent_health: f32,
    pub opponent_stamina: f32,
    /// Own torso world position [x, y, z] — anchor for relative spatial reasoning.
    #[serde(default)]
    pub own_torso_pos: [f32; 3],
    /// Opponent link world positions in order: torso, head, left_arm, right_arm.
    /// Empty if opponent state not available.
    #[serde(default)]
    pub opponent_link_positions: Vec<[f32; 3]>,
    /// Opponent torso world position [x, y, z], convenience copy of opponent_link_positions[0].
    #[serde(default)]
    pub opponent_torso_pos: [f32; 3],
}

// ---------------------------------------------------------------------------
// BoxingScenario — sets up the full boxing environment
// ---------------------------------------------------------------------------

use super::definition::RobotDefinition;
use super::RobotManager;
use crate::scenarios::make_boxing_ring;
use crate::scene::Scene;
use glam::Mat4;

pub struct BoxingScenario {
    pub ring: Scene,
    pub boxing_match: BoxingMatch,
    pub robot_a_id: usize,
    pub robot_b_id: usize,
}

impl BoxingScenario {
    pub fn new(config: BoxingMatchConfig) -> (Self, RobotManager) {
        let ring = make_boxing_ring(6.0);

        let mut manager = RobotManager::new();

        let def_a = RobotDefinition::boxing_humanoid();
        let pose_a = Mat4::from_translation(glam::Vec3::new(-0.5, 0.0, 0.0));
        let robot_a_id = manager.add_robot(def_a, pose_a);

        let def_b = RobotDefinition::boxing_humanoid();
        let pose_b = Mat4::from_translation(glam::Vec3::new(0.5, 0.0, 0.0));
        let robot_b_id = manager.add_robot(def_b, pose_b);

        // Enable combat state on both robots
        if let Some(robot) = manager.get_robot_mut(robot_a_id) {
            robot.state.combat = Some(CombatState::new(500.0, 100.0));
        }
        if let Some(robot) = manager.get_robot_mut(robot_b_id) {
            robot.state.combat = Some(CombatState::new(500.0, 100.0));
        }

        let boxing_match = BoxingMatch::new(robot_a_id, robot_b_id, config);

        let scenario = BoxingScenario {
            ring,
            boxing_match,
            robot_a_id,
            robot_b_id,
        };

        (scenario, manager)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::definition::BodyZone;

    fn make_config() -> BoxingMatchConfig {
        BoxingMatchConfig {
            round_duration: 10.0,
            num_rounds: 3,
            countdown_duration: 3.0,
            round_break_duration: 2.0,
        }
    }

    fn make_hit(attacker: usize, target: usize) -> HitEvent {
        HitEvent {
            attacker_robot: attacker,
            target_robot: target,
            attacker_link: 1,
            target_link: 0,
            zone: BodyZone::Body,
            impact_force: 50.0,
            damage: 10.0,
            contact_point: glam::Vec3::ZERO,
            contact_normal: glam::Vec3::Y,
        }
    }

    fn make_combat(health: f32, knockdown: bool) -> CombatState {
        let mut c = CombatState::new(100.0, 100.0);
        c.health = health;
        c.knockdown = knockdown;
        c
    }

    // --- State machine tests ---

    #[test]
    fn test_match_initial_state() {
        let m = BoxingMatch::new(0, 1, make_config());
        assert_eq!(m.phase, MatchPhase::WaitingForAgents);
        assert_eq!(m.current_round, 1);
        assert_eq!(m.round_time, 0.0);
        assert!(m.rounds.is_empty());
    }

    #[test]
    fn test_match_waiting_to_countdown() {
        let mut m = BoxingMatch::new(0, 1, make_config());
        m.connect_agent(0);
        m.update(&[], &[], 0.1);
        assert_eq!(m.phase, MatchPhase::WaitingForAgents);

        m.connect_agent(1);
        m.update(&[], &[], 0.1);
        assert!(matches!(m.phase, MatchPhase::Countdown { .. }));
    }

    #[test]
    fn test_countdown_to_fighting() {
        let mut m = BoxingMatch::new(0, 1, make_config());
        m.connect_agent(0);
        m.connect_agent(1);
        m.update(&[], &[], 0.1); // -> Countdown(3.0)

        // Tick through countdown
        m.update(&[], &[], 1.0); // 2.0 remaining
        assert!(matches!(m.phase, MatchPhase::Countdown { .. }));
        m.update(&[], &[], 1.0); // 1.0 remaining
        m.update(&[], &[], 1.1); // -> Fighting
        assert_eq!(m.phase, MatchPhase::Fighting);
        assert_eq!(m.rounds.len(), 1);
    }

    #[test]
    fn test_fighting_round_timer() {
        let mut m = BoxingMatch::new(0, 1, make_config());
        m.connect_agent(0);
        m.connect_agent(1);
        m.update(&[], &[], 0.1); // -> Countdown
        m.update(&[], &[], 3.1); // -> Fighting

        m.update(&[], &[], 2.0);
        assert!((m.round_time - 2.0).abs() < 0.01);

        m.update(&[], &[], 3.0);
        assert!((m.round_time - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_fighting_tracks_hits() {
        let mut m = BoxingMatch::new(0, 1, make_config());
        m.connect_agent(0);
        m.connect_agent(1);
        m.update(&[], &[], 0.1);
        m.update(&[], &[], 3.1); // -> Fighting

        let hits = vec![make_hit(0, 1), make_hit(0, 1), make_hit(1, 0)];
        let combat_a = make_combat(90.0, false);
        let combat_b = make_combat(80.0, false);
        m.update(&hits, &[(0, &combat_a), (1, &combat_b)], 0.5);

        let score = &m.rounds[0];
        assert_eq!(score.hits_a, 2);
        assert_eq!(score.hits_b, 1);
    }

    #[test]
    fn test_round_end_on_timer() {
        let mut m = BoxingMatch::new(0, 1, make_config());
        m.connect_agent(0);
        m.connect_agent(1);
        m.update(&[], &[], 0.1);
        m.update(&[], &[], 3.1); // -> Fighting

        let combat = make_combat(100.0, false);
        m.update(&[], &[(0, &combat), (1, &combat)], 10.1); // round_duration=10
        assert!(matches!(m.phase, MatchPhase::RoundEnd { .. }));
    }

    #[test]
    fn test_round_scoring_10_9() {
        let mut m = BoxingMatch::new(0, 1, make_config());
        m.connect_agent(0);
        m.connect_agent(1);
        m.update(&[], &[], 0.1);
        m.update(&[], &[], 3.1); // -> Fighting

        let hits = vec![make_hit(0, 1), make_hit(0, 1), make_hit(1, 0)];
        let combat = make_combat(100.0, false);
        m.update(&hits, &[(0, &combat), (1, &combat)], 10.1);

        let score = &m.rounds[0];
        assert_eq!(score.score_a, 10);
        assert_eq!(score.score_b, 9);
    }

    #[test]
    fn test_round_scoring_knockdown_10_8() {
        let mut m = BoxingMatch::new(0, 1, make_config());
        m.connect_agent(0);
        m.connect_agent(1);
        m.update(&[], &[], 0.1);
        m.update(&[], &[], 3.1); // -> Fighting

        let hits = vec![make_hit(0, 1)];
        let combat_a = make_combat(100.0, false);
        let combat_b = make_combat(50.0, true); // knockdown!
        m.update(&hits, &[(0, &combat_a), (1, &combat_b)], 10.1);

        let score = &m.rounds[0];
        assert_eq!(score.score_a, 10);
        assert_eq!(score.score_b, 8);
    }

    #[test]
    fn test_match_end_after_all_rounds() {
        let mut m = BoxingMatch::new(0, 1, make_config());
        m.connect_agent(0);
        m.connect_agent(1);
        m.update(&[], &[], 0.1);
        m.update(&[], &[], 3.1); // -> Fighting (round 1)

        let combat = make_combat(100.0, false);
        for _ in 0..3 {
            let hits = vec![make_hit(0, 1)]; // A wins each round
            m.update(&hits, &[(0, &combat), (1, &combat)], 10.1); // -> RoundEnd
            if matches!(m.phase, MatchPhase::MatchEnd) {
                break;
            }
            m.update(&[], &[], 2.1); // -> next Fighting or MatchEnd
        }

        assert_eq!(m.phase, MatchPhase::MatchEnd);
        assert_eq!(m.rounds.len(), 3);
    }

    #[test]
    fn test_winner_determination() {
        let mut m = BoxingMatch::new(0, 1, make_config());
        m.connect_agent(0);
        m.connect_agent(1);
        m.update(&[], &[], 0.1);
        m.update(&[], &[], 3.1); // -> Fighting

        let combat = make_combat(100.0, false);
        // Robot A wins 2 rounds, robot B wins 1
        // Round 1: A wins
        m.update(&[make_hit(0, 1)], &[(0, &combat), (1, &combat)], 10.1);
        m.update(&[], &[], 2.1);
        // Round 2: B wins
        m.update(&[make_hit(1, 0)], &[(0, &combat), (1, &combat)], 10.1);
        m.update(&[], &[], 2.1);
        // Round 3: A wins
        m.update(&[make_hit(0, 1)], &[(0, &combat), (1, &combat)], 10.1);

        assert_eq!(m.winner(), Some(0));
        let (sa, sb) = m.current_scores();
        assert!(sa > sb);
    }

    #[test]
    fn test_match_end_on_knockout() {
        let mut m = BoxingMatch::new(0, 1, make_config());
        m.connect_agent(0);
        m.connect_agent(1);
        m.update(&[], &[], 0.1);
        m.update(&[], &[], 3.1); // -> Fighting

        let combat_a = make_combat(100.0, false);
        let combat_b = make_combat(0.0, true); // KO
        m.update(&[make_hit(0, 1)], &[(0, &combat_a), (1, &combat_b)], 0.5);

        assert_eq!(m.phase, MatchPhase::MatchEnd);
    }

    // --- Snapshot / protocol tests ---

    #[test]
    fn test_match_snapshot_fighting() {
        let mut m = BoxingMatch::new(0, 1, make_config());
        m.connect_agent(0);
        m.connect_agent(1);
        m.update(&[], &[], 0.1);
        m.update(&[], &[], 3.1); // -> Fighting
        m.update(&[], &[], 2.0);

        let combat_b = make_combat(80.0, false);
        let snap = m.snapshot(0, Some(&combat_b));
        assert_eq!(snap.phase, "fighting");
        assert_eq!(snap.current_round, 1);
        assert!((snap.round_time - 2.0).abs() < 0.01);
        assert!((snap.opponent_health - 80.0).abs() < 0.01);
        assert_eq!(snap.your_robot, 0);
    }

    #[test]
    fn test_match_snapshot_includes_opponent() {
        let m = BoxingMatch::new(0, 1, make_config());
        let combat = make_combat(75.0, false);
        let snap = m.snapshot(0, Some(&combat));
        assert!((snap.opponent_health - 75.0).abs() < 0.01);
        assert!((snap.opponent_stamina - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_match_state_serialization() {
        let m = BoxingMatch::new(0, 1, make_config());
        let snap = m.snapshot(0, None);
        let json = serde_json::to_string(&snap).expect("serialize");
        let deser: BoxingMatchState = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deser.phase, snap.phase);
        assert_eq!(deser.current_round, snap.current_round);
        assert_eq!(deser.your_robot, snap.your_robot);
    }

    // --- BoxingScenario tests ---

    #[test]
    fn test_boxing_scenario_creates_two_robots() {
        let (scenario, manager) = BoxingScenario::new(make_config());
        assert_eq!(manager.robots.len(), 2);
        assert_eq!(scenario.robot_a_id, 0);
        assert_eq!(scenario.robot_b_id, 1);
    }

    #[test]
    fn test_boxing_scenario_robots_have_combat() {
        let (_scenario, manager) = BoxingScenario::new(make_config());
        for robot in &manager.robots {
            assert!(
                robot.state.combat.is_some(),
                "Robot '{}' should have CombatState",
                robot.definition.name
            );
        }
    }

    #[test]
    fn test_boxing_scenario_robots_positioned() {
        let (_scenario, manager) = BoxingScenario::new(make_config());
        let pose_a = manager.robots[0].base_pose_mat4();
        let pose_b = manager.robots[1].base_pose_mat4();
        let pos_a = pose_a.col(3).truncate();
        let pos_b = pose_b.col(3).truncate();
        assert!((pos_a.x - (-0.5)).abs() < 0.01);
        assert!((pos_b.x - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_boxing_match_update_from_step() {
        let (mut scenario, _manager) = BoxingScenario::new(make_config());
        scenario.boxing_match.connect_agent(0);
        scenario.boxing_match.connect_agent(1);
        scenario.boxing_match.update(&[], &[], 0.1);
        assert!(matches!(
            scenario.boxing_match.phase,
            MatchPhase::Countdown { .. }
        ));
    }

    // --- Integration test: full match flow ---

    #[test]
    fn test_full_boxing_match_flow() {
        let config = BoxingMatchConfig {
            round_duration: 5.0,
            num_rounds: 3,
            countdown_duration: 1.0,
            round_break_duration: 1.0,
        };
        let mut m = BoxingMatch::new(0, 1, config);

        // Phase: WaitingForAgents -> Countdown
        m.connect_agent(0);
        m.connect_agent(1);
        m.update(&[], &[], 0.1);
        assert!(matches!(m.phase, MatchPhase::Countdown { .. }));

        // Phase: Countdown -> Fighting
        m.update(&[], &[], 1.1);
        assert_eq!(m.phase, MatchPhase::Fighting);

        let combat = make_combat(100.0, false);

        // Round 1: A lands 3 hits, B lands 1
        m.update(
            &[
                make_hit(0, 1),
                make_hit(0, 1),
                make_hit(0, 1),
                make_hit(1, 0),
            ],
            &[(0, &combat), (1, &combat)],
            5.1,
        );
        assert!(matches!(m.phase, MatchPhase::RoundEnd { .. }));
        assert_eq!(m.rounds[0].score_a, 10);
        assert_eq!(m.rounds[0].score_b, 9);

        // Round break -> Round 2
        m.update(&[], &[], 1.1);
        assert_eq!(m.phase, MatchPhase::Fighting);
        assert_eq!(m.current_round, 2);

        // Round 2: B wins
        m.update(
            &[make_hit(1, 0), make_hit(1, 0)],
            &[(0, &combat), (1, &combat)],
            5.1,
        );
        assert!(matches!(m.phase, MatchPhase::RoundEnd { .. }));
        assert_eq!(m.rounds[1].score_b, 10);
        assert_eq!(m.rounds[1].score_a, 9);

        // Round break -> Round 3
        m.update(&[], &[], 1.1);
        assert_eq!(m.phase, MatchPhase::Fighting);
        assert_eq!(m.current_round, 3);

        // Round 3: A wins
        m.update(
            &[make_hit(0, 1), make_hit(0, 1)],
            &[(0, &combat), (1, &combat)],
            5.1,
        );
        assert_eq!(m.phase, MatchPhase::MatchEnd);
        assert_eq!(m.rounds.len(), 3);

        // A won 2 rounds (10+9+10=29), B won 1 (9+10+9=28)
        let (sa, sb) = m.current_scores();
        assert_eq!(sa, 29);
        assert_eq!(sb, 28);
        assert_eq!(m.winner(), Some(0));
    }

    #[test]
    fn test_knockout_ends_match_early() {
        let config = BoxingMatchConfig {
            round_duration: 60.0,
            num_rounds: 3,
            countdown_duration: 1.0,
            round_break_duration: 1.0,
        };
        let mut m = BoxingMatch::new(0, 1, config);
        m.connect_agent(0);
        m.connect_agent(1);
        m.update(&[], &[], 0.1);
        m.update(&[], &[], 1.1); // -> Fighting

        // KO in round 1
        let combat_a = make_combat(100.0, false);
        let combat_b = make_combat(0.0, true);
        m.update(&[make_hit(0, 1)], &[(0, &combat_a), (1, &combat_b)], 2.0);
        assert_eq!(m.phase, MatchPhase::MatchEnd);
        assert_eq!(m.current_round, 1);
        assert_eq!(m.rounds.len(), 1);
        assert_eq!(m.winner(), Some(0));
    }

    #[test]
    fn test_boxing_scenario_with_combat_step() {
        let (mut scenario, mut manager) = BoxingScenario::new(BoxingMatchConfig {
            round_duration: 60.0,
            num_rounds: 3,
            countdown_duration: 1.0,
            round_break_duration: 1.0,
        });

        scenario.boxing_match.connect_agent(0);
        scenario.boxing_match.connect_agent(1);
        scenario.boxing_match.update(&[], &[], 0.1);
        scenario.boxing_match.update(&[], &[], 1.1); // -> Fighting

        let scene_meshes = &scenario.ring.meshes;
        manager.step(1.0 / 60.0, scene_meshes);

        let hit_events = &manager.last_hit_events;
        let combat_states: Vec<(usize, &CombatState)> = manager
            .robots
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.state.combat.as_ref().map(|c| (i, c)))
            .collect();

        scenario
            .boxing_match
            .update(hit_events, &combat_states, 1.0 / 60.0);
        assert_eq!(scenario.boxing_match.phase, MatchPhase::Fighting);
    }

    #[test]
    fn test_boxing_arms_can_reach_opponent() {
        use crate::robot::collision::detect_robot_collisions;
        use crate::robot::state::{apply_action, RobotAction};

        let (_scenario, mut manager) = BoxingScenario::new(make_config());

        let action_a = RobotAction {
            motor_velocities: vec![0.0, 3.0, -3.0],
            gripper_commands: vec![],
            base_velocity: [0.0, 0.0],
        };
        let action_b = RobotAction {
            motor_velocities: vec![0.0, -3.0, 3.0],
            gripper_commands: vec![],
            base_velocity: [0.0, 0.0],
        };

        let scene_meshes = vec![];
        let mut total_collisions = 0;

        for step in 0..120 {
            {
                let (left, right) = manager.robots.split_at_mut(1);
                apply_action(&left[0].definition, &mut left[0].state, &action_a);
                apply_action(&right[0].definition, &mut right[0].state, &action_b);
            }
            manager.step(1.0 / 60.0, &scene_meshes);

            let combat_data: Vec<(
                usize,
                &crate::robot::definition::RobotDefinition,
                &crate::robot::state::RobotState,
            )> = manager
                .robots
                .iter()
                .enumerate()
                .map(|(i, r)| (i, &r.definition, &r.state))
                .collect();
            let collisions = detect_robot_collisions(&combat_data);
            if !collisions.is_empty() {
                let jp = &manager.robots[0].state.joint_positions;
                eprintln!(
                    "Step {}: {} collisions, joints=[{:.2}, {:.2}, {:.2}]",
                    step,
                    collisions.len(),
                    jp.first().unwrap_or(&0.0),
                    jp.get(1).unwrap_or(&0.0),
                    jp.get(2).unwrap_or(&0.0),
                );
                total_collisions += collisions.len();
            }

            total_collisions += manager.last_hit_events.len();
        }

        eprintln!("Total collision frames: {}", total_collisions);

        assert!(
            total_collisions > 0,
            "Arms should overlap with opponent when extended"
        );
    }
}

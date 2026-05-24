//! Adaptive performance governor.
//!
//! Tracks rolling per-frame durations and exposes a classification
//! (`PerfClass::Healthy | Degraded | Critical`) plus multipliers the
//! engine and renderer use to scale work down on slow devices.
//!
//! Design contract:
//!   * `record_frame(dt)` is called once per `eframe::App::update`.
//!   * `class()` returns the *current* class derived from the rolling
//!     window. The class is sticky for `STICKY_FRAMES` frames after a
//!     downshift to avoid oscillation.
//!   * The governor never blocks; never spawns threads; never panics on
//!     pathological inputs (NaN / Inf / negative dt → clamped).
//!
//! Threshold rationale (see GOAL.md Risks):
//!   * Healthy: avg dt ≤ 25 ms  (≥ 40 fps)         → full quality
//!   * Degraded: 25 ms < avg ≤ 50 ms (20–40 fps)   → ~0.75× work
//!   * Critical: avg dt > 50 ms (< 20 fps)         → ~0.5× work
//!
//! All multipliers are pure functions of the class — callers compose
//! them with their own baselines (sim substeps, ray-path budget,
//! heatmap resolution scale).

use std::collections::VecDeque;
use std::time::Duration;

/// Rolling-window length. 30 frames ≈ 0.5 s at 60 fps — short enough to
/// react to a sustained slowdown, long enough to ignore a single hiccup.
pub const WINDOW_FRAMES: usize = 30;

/// Once the governor downshifts to Degraded or Critical it stays there
/// for at least this many frames before upgrading again. Prevents
/// oscillating between classes when work is borderline.
pub const STICKY_FRAMES: usize = 60;

/// Healthy ceiling — average frame time at/below this is full quality.
pub const HEALTHY_CEILING: Duration = Duration::from_millis(25);

/// Degraded ceiling — average frame time at/below this is degraded, above
/// is critical.
pub const DEGRADED_CEILING: Duration = Duration::from_millis(50);

/// Absolute cap on any single sample we'll record. A 2-second frame is
/// almost certainly a debugger pause / GC stall / OS preemption, not a
/// real workload signal — clamp it.
pub const SAMPLE_CEILING: Duration = Duration::from_millis(2000);

/// Performance class — drives engine + renderer downshifts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PerfClass {
    #[default]
    Healthy,
    Degraded,
    Critical,
}

impl PerfClass {
    /// Human-facing status-bar label.
    pub fn label(self) -> &'static str {
        match self {
            PerfClass::Healthy => "perf: healthy",
            PerfClass::Degraded => "perf: degraded",
            PerfClass::Critical => "perf: throttled",
        }
    }

    /// Multiplier for sim substeps and similar per-frame work counts.
    /// `Healthy = 1.0`, `Degraded = 0.75`, `Critical = 0.5`.
    pub fn work_scale(self) -> f32 {
        match self {
            PerfClass::Healthy => 1.0,
            PerfClass::Degraded => 0.75,
            PerfClass::Critical => 0.5,
        }
    }

    fn rank(self) -> u8 {
        match self {
            PerfClass::Healthy => 0,
            PerfClass::Degraded => 1,
            PerfClass::Critical => 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PerfGovernor {
    samples: VecDeque<Duration>,
    class: PerfClass,
    frames_in_class: usize,
    total_frames: u64,
    last_avg: Duration,
}

impl Default for PerfGovernor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerfGovernor {
    pub fn new() -> Self {
        Self {
            samples: VecDeque::with_capacity(WINDOW_FRAMES),
            class: PerfClass::Healthy,
            frames_in_class: 0,
            total_frames: 0,
            last_avg: Duration::ZERO,
        }
    }

    /// Record one frame's duration. Idempotent for the class until enough
    /// samples accumulate. Clamps NaN/Inf/negative-as-Duration (Duration
    /// itself can't be negative, but a caller might pass `Duration::ZERO`
    /// on the first frame — that's fine).
    pub fn record_frame(&mut self, dt: Duration) {
        let sample = if dt > SAMPLE_CEILING {
            SAMPLE_CEILING
        } else {
            dt
        };
        if self.samples.len() == WINDOW_FRAMES {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
        self.total_frames = self.total_frames.saturating_add(1);
        self.frames_in_class = self.frames_in_class.saturating_add(1);

        self.last_avg = self.compute_avg();
        self.maybe_reclassify();
    }

    fn compute_avg(&self) -> Duration {
        if self.samples.is_empty() {
            return Duration::ZERO;
        }
        let total_nanos: u128 = self.samples.iter().map(|d| d.as_nanos()).sum();
        let n = self.samples.len() as u128;
        Duration::from_nanos((total_nanos / n) as u64)
    }

    fn maybe_reclassify(&mut self) {
        let proposed = if self.last_avg <= HEALTHY_CEILING {
            PerfClass::Healthy
        } else if self.last_avg <= DEGRADED_CEILING {
            PerfClass::Degraded
        } else {
            PerfClass::Critical
        };

        if proposed == self.class {
            return;
        }

        if proposed.rank() > self.class.rank() {
            self.class = proposed;
            self.frames_in_class = 0;
            return;
        }

        if self.frames_in_class >= STICKY_FRAMES {
            self.class = proposed;
            self.frames_in_class = 0;
        }
    }

    pub fn class(&self) -> PerfClass {
        self.class
    }

    pub fn avg_frame_time(&self) -> Duration {
        self.last_avg
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Sim substeps to run this frame given a baseline integer count.
    /// Always at least 1 — the engine must not stall completely.
    pub fn sim_substeps(&self, baseline: u32) -> u32 {
        let scaled = (baseline as f32 * self.class.work_scale()).round() as u32;
        scaled.max(1)
    }

    /// Cap on debug ray paths drawn this frame. Always at least 8 to keep
    /// the overlay legible at the lowest setting.
    pub fn ray_paths_cap(&self, baseline: u32) -> u32 {
        let scaled = (baseline as f32 * self.class.work_scale()).round() as u32;
        scaled.max(8)
    }

    /// Multiplier in (0, 1] for heatmap render resolution.
    pub fn heatmap_resolution_scale(&self) -> f32 {
        self.class.work_scale()
    }

    /// True if the governor has reached its critical class — callers can
    /// surface a one-time warning, disable nice-to-have effects, etc.
    pub fn is_critical(&self) -> bool {
        matches!(self.class, PerfClass::Critical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(gov: &mut PerfGovernor, frames: usize, dt: Duration) {
        for _ in 0..frames {
            gov.record_frame(dt);
        }
    }

    #[test]
    fn default_is_healthy() {
        let gov = PerfGovernor::default();
        assert_eq!(gov.class(), PerfClass::Healthy);
        assert_eq!(gov.sample_count(), 0);
    }

    #[test]
    fn fast_frames_stay_healthy() {
        let mut gov = PerfGovernor::new();
        run(&mut gov, WINDOW_FRAMES, Duration::from_millis(8));
        assert_eq!(gov.class(), PerfClass::Healthy);
    }

    #[test]
    fn medium_frames_become_degraded_immediately() {
        let mut gov = PerfGovernor::new();
        run(&mut gov, WINDOW_FRAMES, Duration::from_millis(35));
        assert_eq!(gov.class(), PerfClass::Degraded);
    }

    #[test]
    fn slow_frames_become_critical_immediately() {
        let mut gov = PerfGovernor::new();
        run(&mut gov, WINDOW_FRAMES, Duration::from_millis(80));
        assert_eq!(gov.class(), PerfClass::Critical);
    }

    #[test]
    fn downshift_is_sticky_upgrade_requires_window() {
        let mut gov = PerfGovernor::new();
        run(&mut gov, WINDOW_FRAMES, Duration::from_millis(80));
        assert_eq!(gov.class(), PerfClass::Critical);
        // A handful of fast frames should NOT immediately upgrade out of
        // Critical — needs to spend STICKY_FRAMES in class first.
        run(&mut gov, 5, Duration::from_millis(5));
        assert_eq!(gov.class(), PerfClass::Critical);
        // After enough fast frames the rolling average drops AND sticky
        // window expires → governor relaxes.
        run(
            &mut gov,
            STICKY_FRAMES + WINDOW_FRAMES,
            Duration::from_millis(5),
        );
        assert_eq!(gov.class(), PerfClass::Healthy);
    }

    #[test]
    fn upshift_to_worse_class_is_immediate() {
        let mut gov = PerfGovernor::new();
        run(&mut gov, WINDOW_FRAMES, Duration::from_millis(5));
        assert_eq!(gov.class(), PerfClass::Healthy);
        // A burst of slow frames → governor must downshift immediately.
        run(&mut gov, WINDOW_FRAMES, Duration::from_millis(120));
        assert_eq!(gov.class(), PerfClass::Critical);
    }

    #[test]
    fn sample_ceiling_clamps_pathological_frames() {
        let mut gov = PerfGovernor::new();
        gov.record_frame(Duration::from_secs(30));
        assert!(gov.avg_frame_time() <= SAMPLE_CEILING);
    }

    #[test]
    fn sim_substeps_never_zero() {
        let mut gov = PerfGovernor::new();
        run(&mut gov, WINDOW_FRAMES, Duration::from_millis(500));
        assert_eq!(gov.sim_substeps(0), 1);
        assert_eq!(gov.sim_substeps(1), 1);
        assert!(gov.sim_substeps(8) <= 8);
    }

    #[test]
    fn ray_paths_cap_never_below_eight() {
        let mut gov = PerfGovernor::new();
        run(&mut gov, WINDOW_FRAMES, Duration::from_millis(500));
        assert_eq!(gov.ray_paths_cap(0), 8);
        assert_eq!(gov.ray_paths_cap(4), 8);
    }

    #[test]
    fn work_scale_monotonic() {
        assert!(PerfClass::Healthy.work_scale() > PerfClass::Degraded.work_scale());
        assert!(PerfClass::Degraded.work_scale() > PerfClass::Critical.work_scale());
    }

    #[test]
    fn rolling_window_evicts_oldest() {
        let mut gov = PerfGovernor::new();
        for _ in 0..(WINDOW_FRAMES * 3) {
            gov.record_frame(Duration::from_millis(10));
        }
        assert_eq!(gov.sample_count(), WINDOW_FRAMES);
    }

    #[test]
    fn labels_present() {
        assert!(!PerfClass::Healthy.label().is_empty());
        assert!(!PerfClass::Degraded.label().is_empty());
        assert!(!PerfClass::Critical.label().is_empty());
    }

    #[test]
    fn heatmap_scale_in_valid_range() {
        let mut gov = PerfGovernor::new();
        let s = gov.heatmap_resolution_scale();
        assert!((0.0..=1.0).contains(&s));
        run(&mut gov, WINDOW_FRAMES, Duration::from_millis(200));
        let s2 = gov.heatmap_resolution_scale();
        assert!((0.0..=1.0).contains(&s2));
        assert!(s2 < s);
    }

    #[test]
    fn is_critical_reflects_class() {
        let mut gov = PerfGovernor::new();
        assert!(!gov.is_critical());
        run(&mut gov, WINDOW_FRAMES, Duration::from_millis(120));
        assert!(gov.is_critical());
    }
}

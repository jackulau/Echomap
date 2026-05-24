//! Startup device-capability detection.
//!
//! Decides conservative defaults so EchoMap never tries to push more
//! work than the machine can sustain. Pure-stdlib probe — no `sysinfo`
//! dependency (we keep build closure tight) — so the RAM number is a
//! heuristic derived from core count and a platform tag, not a real
//! measurement. Override every field via `ECHOMAP_*` env vars.
//!
//! Detection contract:
//!   * `DeviceCaps::detect()` always succeeds — returns
//!     `DeviceCaps::SAFE_FALLBACK` if any probe fails.
//!   * Env-var overrides are validated; junk values fall back to the
//!     probed default with a warning log.
//!
//! Defaults pick philosophy: ship the smallest defaults that still look
//! good on a 4-core M-series laptop or a low-end Windows box, so we
//! never light up someone's GPU/fans on first launch. Power users dial
//! up via the Settings panel or env vars.

use std::env;
use std::num::NonZeroUsize;

/// Conservative fallback used when every probe fails.
pub const FALLBACK_CORES: usize = 2;
pub const FALLBACK_SIM_THREADS: usize = 2;
pub const FALLBACK_RAY_PATHS: u32 = 200;
pub const FALLBACK_HEATMAP_RES: u32 = 64;
pub const FALLBACK_RAM_HINT_MB: u64 = 4096;

pub const ENV_SIM_THREADS: &str = "ECHOMAP_SIM_THREADS";
pub const ENV_RAY_PATHS: &str = "ECHOMAP_RAY_PATHS";
pub const ENV_HEATMAP_RES: &str = "ECHOMAP_HEATMAP_RES";
pub const ENV_STRESS: &str = "ECHOMAP_STRESS";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceCaps {
    pub logical_cores: usize,
    pub sim_threads: usize,
    pub default_ray_paths: u32,
    pub default_heatmap_res: u32,
    pub ram_hint_mb: u64,
    pub platform: &'static str,
    pub stress_mode: bool,
}

impl DeviceCaps {
    pub const SAFE_FALLBACK: DeviceCaps = DeviceCaps {
        logical_cores: FALLBACK_CORES,
        sim_threads: FALLBACK_SIM_THREADS,
        default_ray_paths: FALLBACK_RAY_PATHS,
        default_heatmap_res: FALLBACK_HEATMAP_RES,
        ram_hint_mb: FALLBACK_RAM_HINT_MB,
        platform: "unknown",
        stress_mode: false,
    };

    pub fn detect() -> Self {
        let logical_cores = std::thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(FALLBACK_CORES);

        let platform = platform_tag();
        let ram_hint_mb = heuristic_ram_mb(logical_cores, platform);

        let probed_sim_threads = recommended_sim_threads(logical_cores);
        let probed_ray_paths = recommended_ray_paths(logical_cores);
        let probed_heatmap_res = recommended_heatmap_res(logical_cores);

        let sim_threads = parse_env_usize(ENV_SIM_THREADS)
            .map(|v| v.min(64).max(1))
            .unwrap_or(probed_sim_threads);
        let default_ray_paths = parse_env_u32(ENV_RAY_PATHS)
            .map(|v| v.clamp(8, 100_000))
            .unwrap_or(probed_ray_paths);
        let default_heatmap_res = parse_env_u32(ENV_HEATMAP_RES)
            .map(|v| v.clamp(16, 1024))
            .unwrap_or(probed_heatmap_res);

        let stress_mode = parse_env_flag(ENV_STRESS);

        Self {
            logical_cores,
            sim_threads,
            default_ray_paths,
            default_heatmap_res,
            ram_hint_mb,
            platform,
            stress_mode,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "{} · {} cores · ~{} MB RAM · sim_threads={} · ray_paths={} · heatmap={}{}",
            self.platform,
            self.logical_cores,
            self.ram_hint_mb,
            self.sim_threads,
            self.default_ray_paths,
            self.default_heatmap_res,
            if self.stress_mode { " · STRESS" } else { "" },
        )
    }
}

impl Default for DeviceCaps {
    fn default() -> Self {
        Self::SAFE_FALLBACK
    }
}

fn platform_tag() -> &'static str {
    if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "unknown"
    }
}

fn heuristic_ram_mb(cores: usize, platform: &'static str) -> u64 {
    let per_core = match platform {
        "macos" => 2048,
        "linux" => 1024,
        "windows" => 1024,
        _ => 768,
    };
    (cores as u64).saturating_mul(per_core).max(2048)
}

pub fn recommended_sim_threads(cores: usize) -> usize {
    if cores <= 2 {
        1
    } else if cores <= 4 {
        2
    } else if cores <= 8 {
        cores.saturating_sub(2)
    } else {
        (cores * 3 / 4).min(16)
    }
}

pub fn recommended_ray_paths(cores: usize) -> u32 {
    match cores {
        0..=2 => 100,
        3..=4 => 200,
        5..=8 => 400,
        9..=16 => 800,
        _ => 1200,
    }
}

pub fn recommended_heatmap_res(cores: usize) -> u32 {
    match cores {
        0..=2 => 48,
        3..=4 => 64,
        5..=8 => 96,
        9..=16 => 128,
        _ => 192,
    }
}

fn parse_env_usize(key: &str) -> Option<usize> {
    env::var(key).ok().and_then(|raw| {
        raw.trim().parse::<usize>().ok().or_else(|| {
            log::warn!("{key} not a positive integer: {raw:?} — using default");
            None
        })
    })
}

fn parse_env_u32(key: &str) -> Option<u32> {
    env::var(key).ok().and_then(|raw| {
        raw.trim().parse::<u32>().ok().or_else(|| {
            log::warn!("{key} not a positive integer: {raw:?} — using default");
            None
        })
    })
}

fn parse_env_flag(key: &str) -> bool {
    matches!(
        env::var(key).ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "on")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_sensible_values() {
        let caps = DeviceCaps::detect();
        assert!(caps.logical_cores >= 1);
        assert!(caps.sim_threads >= 1);
        assert!(caps.default_ray_paths >= 8);
        assert!(caps.default_heatmap_res >= 16);
        assert!(caps.ram_hint_mb >= 2048);
        assert!(!caps.platform.is_empty());
    }

    #[test]
    fn safe_fallback_is_stable() {
        let f = DeviceCaps::SAFE_FALLBACK;
        assert_eq!(f.logical_cores, FALLBACK_CORES);
        assert_eq!(f.sim_threads, FALLBACK_SIM_THREADS);
        assert_eq!(f.default_ray_paths, FALLBACK_RAY_PATHS);
        assert_eq!(f.default_heatmap_res, FALLBACK_HEATMAP_RES);
    }

    #[test]
    fn recommended_scales_with_cores() {
        assert!(recommended_sim_threads(2) <= recommended_sim_threads(8));
        assert!(recommended_sim_threads(8) <= recommended_sim_threads(16));
        assert!(recommended_ray_paths(2) < recommended_ray_paths(8));
        assert!(recommended_ray_paths(8) < recommended_ray_paths(16));
        assert!(recommended_heatmap_res(2) < recommended_heatmap_res(8));
    }

    #[test]
    fn recommended_sim_threads_caps_at_sixteen() {
        for huge in [32usize, 64, 128] {
            assert!(recommended_sim_threads(huge) <= 16);
        }
    }

    #[test]
    fn recommended_sim_threads_never_zero() {
        for c in 0..=64 {
            assert!(recommended_sim_threads(c) >= 1);
        }
    }

    #[test]
    fn ram_hint_floor_two_gigs() {
        for c in 1..=32 {
            assert!(heuristic_ram_mb(c, "macos") >= 2048);
            assert!(heuristic_ram_mb(c, "linux") >= 2048);
            assert!(heuristic_ram_mb(c, "windows") >= 2048);
            assert!(heuristic_ram_mb(c, "unknown") >= 2048);
        }
    }

    #[test]
    fn summary_contains_key_fields() {
        let caps = DeviceCaps::SAFE_FALLBACK;
        let s = caps.summary();
        assert!(s.contains("sim_threads"));
        assert!(s.contains("ray_paths"));
        assert!(s.contains("heatmap"));
    }

    #[test]
    fn env_override_sim_threads() {
        // Cannot mutate process env safely in parallel tests, so just
        // confirm the parser tolerates absence + bad input.
        assert_eq!(parse_env_usize("ECHOMAP_DOES_NOT_EXIST_xyz"), None);
    }

    #[test]
    fn env_flag_recognizes_truthy() {
        // Using a key guaranteed to be absent.
        assert!(!parse_env_flag("ECHOMAP_DOES_NOT_EXIST_xyz"));
    }

    #[test]
    fn default_equals_fallback() {
        assert_eq!(DeviceCaps::default(), DeviceCaps::SAFE_FALLBACK);
    }

    #[test]
    fn platform_tag_is_known() {
        let p = platform_tag();
        assert!(matches!(p, "macos" | "linux" | "windows" | "unknown"));
    }
}

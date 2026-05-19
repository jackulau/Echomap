//! Sim-config bounds + live validation surfaced under each slider in the side panel.

use crate::acoustics::SimulationConfig;

pub const RAY_COUNT_MIN: u32 = 100;
pub const RAY_COUNT_MAX: u32 = 100_000;
pub const MAX_BOUNCES_MIN: u32 = 0;
pub const MAX_BOUNCES_MAX: u32 = 1000;
pub const GRID_RES_MIN: f32 = 0.05;
pub const GRID_RES_MAX: f32 = 2.0;

pub const RAY_COUNT_HELP: &str =
    "Total acoustic rays cast per source. More rays = smoother result, more compute (100..100k).";
pub const MAX_BOUNCES_HELP: &str = "Maximum reflections per ray before it's discarded (0..1000).";
pub const GRID_RES_HELP: &str =
    "Energy grid cell size in metres. Smaller = finer detail, more memory (0.05..2.0).";

#[derive(Default, Clone, Debug, PartialEq)]
pub struct ConfigValidation {
    pub ray_count: Option<String>,
    pub max_bounces: Option<String>,
    pub grid_resolution: Option<String>,
}

impl ConfigValidation {
    pub fn is_valid(&self) -> bool {
        self.ray_count.is_none() && self.max_bounces.is_none() && self.grid_resolution.is_none()
    }

    pub fn errors(&self) -> Vec<&str> {
        let mut out = Vec::new();
        if let Some(e) = &self.ray_count {
            out.push(e.as_str());
        }
        if let Some(e) = &self.max_bounces {
            out.push(e.as_str());
        }
        if let Some(e) = &self.grid_resolution {
            out.push(e.as_str());
        }
        out
    }
}

pub fn validate_sim_config(cfg: &SimulationConfig) -> ConfigValidation {
    let mut v = ConfigValidation::default();
    if cfg.ray_count < RAY_COUNT_MIN || cfg.ray_count > RAY_COUNT_MAX {
        v.ray_count = Some(format!(
            "ray_count must be {RAY_COUNT_MIN}..={RAY_COUNT_MAX} (got {})",
            cfg.ray_count
        ));
    }
    if cfg.max_bounces > MAX_BOUNCES_MAX {
        v.max_bounces = Some(format!(
            "max_bounces must be <= {MAX_BOUNCES_MAX} (got {})",
            cfg.max_bounces
        ));
    }
    if !cfg.grid_resolution.is_finite()
        || cfg.grid_resolution < GRID_RES_MIN
        || cfg.grid_resolution > GRID_RES_MAX
    {
        v.grid_resolution = Some(format!(
            "grid_resolution must be {GRID_RES_MIN}..={GRID_RES_MAX} m (got {:.3})",
            cfg.grid_resolution
        ));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_cfg() -> SimulationConfig {
        SimulationConfig {
            ray_count: 10_000,
            max_bounces: 50,
            energy_threshold: 0.001,
            grid_resolution: 0.25,
        }
    }

    #[test]
    fn sim_config_validation_accepts_defaults() {
        let v = validate_sim_config(&ok_cfg());
        assert!(v.is_valid(), "default config should validate: {:?}", v);
    }

    #[test]
    fn sim_config_validation_rejects_low_ray_count() {
        let mut c = ok_cfg();
        c.ray_count = 10;
        let v = validate_sim_config(&c);
        assert!(!v.is_valid());
        assert!(v.ray_count.is_some());
    }

    #[test]
    fn sim_config_validation_rejects_high_ray_count() {
        let mut c = ok_cfg();
        c.ray_count = 500_000;
        let v = validate_sim_config(&c);
        assert!(!v.is_valid());
        assert!(v.ray_count.is_some());
    }

    #[test]
    fn sim_config_validation_rejects_oob_bounces() {
        let mut c = ok_cfg();
        c.max_bounces = 5000;
        let v = validate_sim_config(&c);
        assert!(!v.is_valid());
        assert!(v.max_bounces.is_some());
    }

    #[test]
    fn sim_config_validation_rejects_grid_res() {
        let mut c = ok_cfg();
        c.grid_resolution = 0.001;
        let v = validate_sim_config(&c);
        assert!(!v.is_valid());
        assert!(v.grid_resolution.is_some());

        let mut c = ok_cfg();
        c.grid_resolution = 10.0;
        let v = validate_sim_config(&c);
        assert!(!v.is_valid());

        let mut c = ok_cfg();
        c.grid_resolution = f32::NAN;
        let v = validate_sim_config(&c);
        assert!(!v.is_valid());
    }

    #[test]
    fn sim_config_validation_reports_multiple_errors() {
        let bad = SimulationConfig {
            ray_count: 1,
            max_bounces: 99_999,
            energy_threshold: 0.001,
            grid_resolution: 0.0001,
        };
        let v = validate_sim_config(&bad);
        assert_eq!(v.errors().len(), 3);
    }
}

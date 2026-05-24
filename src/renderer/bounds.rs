//! Renderer paint-budget caps.
//!
//! Every `render_*` entry point in `src/renderer/` should consult these
//! constants (or the `cap_*` helpers) before emitting shapes to the
//! egui painter. The painter itself is happy to accept ten million
//! line segments — egui will then tessellate them, allocate a vertex
//! buffer, and shred the user's frame budget. Capping at emission time
//! is the only stable way to keep ill-behaved scenes from hanging the
//! UI thread on low-end hardware.
//!
//! The numbers are intentionally generous on healthy hardware (you have
//! to try fairly hard to hit them under normal use) and exist as a
//! ceiling, not a target. `PerfGovernor` controls *adaptive* downshifts
//! below the ceiling; `bounds` only enforces the absolute maximum.

/// Hard cap on triangles drawn by a single `render_surface_overlay`
/// call. Cap is per-call, not per-frame, so the overlay can still
/// share a frame with debug overlays etc.
pub const MAX_PAINT_TRIS: usize = 200_000;

/// Hard cap on ray-debug line segments drawn per
/// `render_ray_paths_debug` call.
pub const MAX_RAY_LINES: usize = 100_000;

/// Hard cap on listener pulse shapes drawn per frame.
pub const MAX_LISTENER_PULSES: usize = 4_096;

/// Cap a triangle / shape count against `MAX_PAINT_TRIS`.
pub fn cap_paint_tris(requested: usize) -> usize {
    requested.min(MAX_PAINT_TRIS)
}

/// Cap a line-segment count against `MAX_RAY_LINES`.
pub fn cap_ray_lines(requested: usize) -> usize {
    requested.min(MAX_RAY_LINES)
}

/// Cap a listener-pulse count against `MAX_LISTENER_PULSES`.
pub fn cap_listener_pulses(requested: usize) -> usize {
    requested.min(MAX_LISTENER_PULSES)
}

/// True when `requested` was reduced by the cap. Useful for
/// surfacing a one-time "budget exceeded" hint to the user.
pub fn was_capped(requested: usize, max: usize) -> bool {
    requested > max
}

#[cfg(test)]
mod tests {
    use super::*;

    // Compile-time sanity bounds — fail the build if a cap drops below the
    // floors the renderer assumes elsewhere.
    const _: () = assert!(MAX_PAINT_TRIS >= 10_000);
    const _: () = assert!(MAX_RAY_LINES >= 1_000);
    const _: () = assert!(MAX_LISTENER_PULSES >= 256);

    #[test]
    fn cap_paint_tris_clamps_above() {
        assert_eq!(cap_paint_tris(MAX_PAINT_TRIS + 1), MAX_PAINT_TRIS);
        assert_eq!(cap_paint_tris(usize::MAX), MAX_PAINT_TRIS);
    }

    #[test]
    fn cap_paint_tris_passthrough_below() {
        assert_eq!(cap_paint_tris(0), 0);
        assert_eq!(cap_paint_tris(123), 123);
        assert_eq!(cap_paint_tris(MAX_PAINT_TRIS), MAX_PAINT_TRIS);
    }

    #[test]
    fn cap_ray_lines_clamps_above() {
        assert_eq!(cap_ray_lines(MAX_RAY_LINES + 1), MAX_RAY_LINES);
        assert_eq!(cap_ray_lines(usize::MAX), MAX_RAY_LINES);
    }

    #[test]
    fn cap_ray_lines_passthrough_below() {
        assert_eq!(cap_ray_lines(0), 0);
        assert_eq!(cap_ray_lines(50), 50);
    }

    #[test]
    fn cap_listener_pulses_clamps_above() {
        assert_eq!(
            cap_listener_pulses(MAX_LISTENER_PULSES + 1),
            MAX_LISTENER_PULSES
        );
    }

    #[test]
    fn was_capped_detects_overage() {
        assert!(was_capped(MAX_PAINT_TRIS + 1, MAX_PAINT_TRIS));
        assert!(!was_capped(0, MAX_PAINT_TRIS));
        assert!(!was_capped(MAX_PAINT_TRIS, MAX_PAINT_TRIS));
    }

    #[test]
    fn zero_inputs_are_safe() {
        assert_eq!(cap_paint_tris(0), 0);
        assert_eq!(cap_ray_lines(0), 0);
        assert_eq!(cap_listener_pulses(0), 0);
    }
}

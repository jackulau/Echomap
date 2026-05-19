use glam::Vec3;

use crate::fluids::grid::FluidGrid;
use crate::gas::grid::GasGrid;

pub mod legend;
pub mod listener_viz;
pub mod ray_debug;
pub mod surface_heatmap;
pub use legend::{
    render_color_legend, render_material_legend, ColorLegend, MaterialLegendRow,
    DEFAULT_LEGEND_DB_RANGE,
};
pub use listener_viz::{
    capture_listener_energy, normalized_spl, pulse_radius, render_listener_pulse, spl_color,
    DEFAULT_LISTENER_CAPTURE_RADIUS,
};
pub use ray_debug::{
    remaining_energy_at, render_ray_paths_debug, sample_path_indices, DEFAULT_DEBUG_RAY_COUNT,
};
pub use surface_heatmap::{
    energy_to_log_db, face_energies, render_surface_overlay, viridis_color, HeatmapMode,
};

/// Default listener capture_radius doc — `capture_radius` controls how much of
/// the scene's energy grid is integrated into the listener's pulse.
pub const LISTENER_CAPTURE_RADIUS_DOC: &str =
    "capture_radius defaults to 0.5m; pulse + shell expand visually with normalized SPL";

/// State toggle for ray-path debug visualization. When `show_debug_rays = false`
/// the renderer short-circuits with zero perf cost (see `ray_debug` module).
pub const RAY_DEBUG_STATE_DOC: &str = "show_debug_rays toggles ray-path overlay";

/// Octave-band center frequencies used by the acoustic sim. `Broadband` averages
/// across all 6 bands. Bands match standard octave centers (Hz): 125, 250, 500,
/// 1000, 2000, 4000.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FrequencyBand {
    #[default]
    Broadband,
    Hz125,
    Hz250,
    Hz500,
    Hz1k,
    Hz2k,
    Hz4k,
}

impl FrequencyBand {
    /// All 6 narrowband variants in canonical order. Useful for iterating
    /// per-band grids when averaging or picking a specific band.
    pub const ALL_NARROW: [FrequencyBand; 6] = [
        FrequencyBand::Hz125,
        FrequencyBand::Hz250,
        FrequencyBand::Hz500,
        FrequencyBand::Hz1k,
        FrequencyBand::Hz2k,
        FrequencyBand::Hz4k,
    ];

    /// Octave center frequency in Hz. Broadband returns `None`.
    pub fn center_hz(self) -> Option<f32> {
        match self {
            FrequencyBand::Broadband => None,
            FrequencyBand::Hz125 => Some(125.0),
            FrequencyBand::Hz250 => Some(250.0),
            FrequencyBand::Hz500 => Some(500.0),
            FrequencyBand::Hz1k => Some(1000.0),
            FrequencyBand::Hz2k => Some(2000.0),
            FrequencyBand::Hz4k => Some(4000.0),
        }
    }

    /// Index into a `[f32;6]` per-band array. Broadband returns `None`.
    pub fn narrow_index(self) -> Option<usize> {
        match self {
            FrequencyBand::Broadband => None,
            FrequencyBand::Hz125 => Some(0),
            FrequencyBand::Hz250 => Some(1),
            FrequencyBand::Hz500 => Some(2),
            FrequencyBand::Hz1k => Some(3),
            FrequencyBand::Hz2k => Some(4),
            FrequencyBand::Hz4k => Some(5),
        }
    }

    /// Short human label for UI selectors.
    pub fn label(self) -> &'static str {
        match self {
            FrequencyBand::Broadband => "All",
            FrequencyBand::Hz125 => "125 Hz",
            FrequencyBand::Hz250 => "250 Hz",
            FrequencyBand::Hz500 => "500 Hz",
            FrequencyBand::Hz1k => "1 kHz",
            FrequencyBand::Hz2k => "2 kHz",
            FrequencyBand::Hz4k => "4 kHz",
        }
    }
}

/// Sample energy from a grid point through the lens of a selected band.
/// Narrowband picks index into the [f32;6] array; Broadband averages all 6.
pub fn sample_band_energy(gp: &crate::acoustics::GridPoint, band: FrequencyBand) -> f32 {
    match band {
        FrequencyBand::Broadband => broadband_energy(gp),
        FrequencyBand::Hz125 => gp.energy[0],
        FrequencyBand::Hz250 => gp.energy[1],
        FrequencyBand::Hz500 => gp.energy[2],
        FrequencyBand::Hz1k => gp.energy[3],
        FrequencyBand::Hz2k => gp.energy[4],
        FrequencyBand::Hz4k => gp.energy[5],
    }
}

/// Broadband-equivalent energy: arithmetic mean across all 6 octave bands.
pub fn broadband_energy(gp: &crate::acoustics::GridPoint) -> f32 {
    gp.energy.iter().copied().sum::<f32>() / 6.0
}

/// Visualization mode for fluid slice rendering.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FluidVisualizationMode {
    #[default]
    VelocityMagnitude,
    Pressure,
    Density,
    LevelSet,
}

/// Visualization mode for gas slice rendering.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GasVisualizationMode {
    #[default]
    Concentration,
    Temperature,
    Pressure,
    VelocityMagnitude,
}

/// Named camera viewpoints for quick framing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CameraView {
    Perspective,
    Top,
    Front,
    Side,
    Isometric,
    RingsideA,
    RingsideB,
}

pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
    /// Smoothly interpolated focus target; when Some, target drifts toward it each frame.
    pub focus_target: Option<Vec3>,
    /// Smoothly interpolated distance; when Some, distance drifts toward it.
    pub focus_distance: Option<f32>,
}

impl Default for Camera {
    fn default() -> Self {
        let mut cam = Self {
            position: Vec3::ZERO,
            target: Vec3::ZERO,
            up: Vec3::Y,
            distance: 12.0,
            yaw: 45.0_f32.to_radians(),
            pitch: 30.0_f32.to_radians(),
            focus_target: None,
            focus_distance: None,
        };
        cam.update_position();
        cam
    }
}

impl Camera {
    pub fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw += delta_x * 0.01;
        self.pitch = (self.pitch + delta_y * 0.01).clamp(-1.5, 1.5);
        self.update_position();
    }

    pub fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance - delta * 0.5).clamp(0.5, 100.0);
        self.focus_distance = None;
        self.update_position();
    }

    /// Zoom while biasing target toward a world point (zoom-to-cursor).
    pub fn zoom_toward(&mut self, point: Vec3, delta: f32) {
        let old_distance = self.distance;
        self.distance = (self.distance - delta * 0.5).clamp(0.5, 100.0);
        self.focus_distance = None;
        // Bias target toward point by a fraction proportional to zoom-in amount.
        let zoom_in_frac = ((old_distance - self.distance) / old_distance).max(0.0);
        if zoom_in_frac > 0.0 {
            self.target += (point - self.target) * (zoom_in_frac * 0.5);
        }
        self.update_position();
    }

    pub fn pan(&mut self, delta_x: f32, delta_y: f32) {
        let forward = (self.target - self.position).normalize();
        let right = forward.cross(self.up).normalize();
        let up = right.cross(forward).normalize();
        let speed = self.distance * 0.003;
        let offset = right * (-delta_x * speed) + up * (delta_y * speed);
        self.position += offset;
        self.target += offset;
    }

    pub fn focus_on(&mut self, center: Vec3, radius: f32) {
        self.target = center;
        self.distance = (radius * 2.5).max(3.0);
        self.focus_target = None;
        self.focus_distance = None;
        self.update_position();
    }

    /// Set up a smooth focus animation toward `center` at the given orbital `radius`.
    pub fn smooth_focus(&mut self, center: Vec3, radius: f32) {
        self.focus_target = Some(center);
        self.focus_distance = Some((radius * 2.5).max(3.0));
    }

    /// Tick smooth focus interpolation. Call once per frame with the frame dt.
    pub fn tick_focus(&mut self, dt: f32) {
        let alpha = (dt * 8.0).min(1.0);
        let mut changed = false;
        if let Some(t) = self.focus_target {
            let delta = t - self.target;
            if delta.length() > 0.001 {
                self.target += delta * alpha;
                changed = true;
            } else {
                self.target = t;
                self.focus_target = None;
                changed = true;
            }
        }
        if let Some(d) = self.focus_distance {
            let dd = d - self.distance;
            if dd.abs() > 0.001 {
                self.distance += dd * alpha;
                changed = true;
            } else {
                self.distance = d;
                self.focus_distance = None;
                changed = true;
            }
        }
        if changed {
            self.update_position();
        }
    }

    /// FPS-style fly movement in camera-local frame. `forward`/`right`/`up_amt` are
    /// movement amounts (units in scene-space) along the respective basis vectors.
    pub fn fly(&mut self, forward: f32, right: f32, up_amt: f32) {
        let fwd = (self.target - self.position).normalize();
        let rt = fwd.cross(self.up).normalize();
        let up = rt.cross(fwd).normalize();
        let offset = fwd * forward + rt * right + up * up_amt;
        self.position += offset;
        self.target += offset;
        self.focus_target = None;
        self.focus_distance = None;
    }

    /// Mouse-look that rotates the view direction in place (target moves around position).
    pub fn look(&mut self, delta_x: f32, delta_y: f32) {
        self.yaw += delta_x * 0.005;
        self.pitch = (self.pitch + delta_y * 0.005).clamp(-1.5, 1.5);
        let dir = Vec3::new(
            -self.pitch.cos() * self.yaw.cos(),
            -self.pitch.sin(),
            -self.pitch.cos() * self.yaw.sin(),
        );
        self.target = self.position + dir * self.distance;
    }

    pub fn set_view(&mut self, view: CameraView) {
        let (yaw, pitch) = match view {
            CameraView::Perspective => (45.0_f32.to_radians(), 30.0_f32.to_radians()),
            CameraView::Top => (0.0, 89.0_f32.to_radians()),
            CameraView::Front => (0.0_f32.to_radians(), 0.0),
            CameraView::Side => (90.0_f32.to_radians(), 0.0),
            CameraView::Isometric => (45.0_f32.to_radians(), 35.264_f32.to_radians()),
            CameraView::RingsideA => (170.0_f32.to_radians(), 8.0_f32.to_radians()),
            CameraView::RingsideB => (-10.0_f32.to_radians(), 8.0_f32.to_radians()),
        };
        self.yaw = yaw;
        self.pitch = pitch;
        self.focus_target = None;
        self.focus_distance = None;
        self.update_position();
    }

    pub fn update_position(&mut self) {
        self.position = self.target
            + Vec3::new(
                self.distance * self.pitch.cos() * self.yaw.cos(),
                self.distance * self.pitch.sin(),
                self.distance * self.pitch.cos() * self.yaw.sin(),
            );
    }
}

pub fn project_3d(
    point: Vec3,
    camera: &Camera,
    screen_center: egui::Pos2,
    scale: f32,
) -> egui::Pos2 {
    let forward = (camera.target - camera.position).normalize();
    let right = forward.cross(camera.up).normalize();
    let up = right.cross(forward).normalize();

    let rel = point - camera.position;
    let x = rel.dot(right);
    let y = rel.dot(up);
    let z = rel.dot(forward);

    let perspective = if z > 0.1 { 1.0 / z } else { 1.0 / 0.1 };

    egui::Pos2::new(
        screen_center.x + x * scale * perspective * 5.0,
        screen_center.y - y * scale * perspective * 5.0,
    )
}

pub fn screen_to_ray(
    screen_pos: egui::Pos2,
    camera: &Camera,
    screen_center: egui::Pos2,
    scale: f32,
) -> (Vec3, Vec3) {
    let forward = (camera.target - camera.position).normalize();
    let right = forward.cross(camera.up).normalize();
    let up = right.cross(forward).normalize();
    let rx = (screen_pos.x - screen_center.x) / (scale * 5.0);
    let ry = -(screen_pos.y - screen_center.y) / (scale * 5.0);
    let direction = (forward + right * rx + up * ry).normalize();
    (camera.position, direction)
}

pub fn ray_ground_intersect(origin: Vec3, direction: Vec3) -> Option<Vec3> {
    if direction.y.abs() < 1e-6 {
        return None;
    }
    let t = -origin.y / direction.y;
    if t > 0.0 {
        Some(origin + direction * t)
    } else {
        None
    }
}

/// Shade a base color by a face normal under a simple Lambert + ambient model.
/// `light_dir` should point FROM the surface TO the light (will be normalized).
pub fn shade_color(
    base: egui::Color32,
    normal: Vec3,
    light_dir: Vec3,
    ambient: f32,
) -> egui::Color32 {
    let n = normal.normalize_or_zero();
    let l = light_dir.normalize_or_zero();
    let lambert = n.dot(l).max(0.0);
    let intensity = (ambient + (1.0 - ambient) * lambert).clamp(0.0, 1.0);
    egui::Color32::from_rgba_unmultiplied(
        (base.r() as f32 * intensity) as u8,
        (base.g() as f32 * intensity) as u8,
        (base.b() as f32 * intensity) as u8,
        base.a(),
    )
}

/// Standard scene light direction (from upper-front-right).
pub fn scene_light_dir() -> Vec3 {
    Vec3::new(0.4, 1.0, 0.3).normalize()
}

/// Project a circular drop-shadow disk onto the y=0 ground plane and return the
/// screen-space ellipse approximation as 12 polygon points.
pub fn ground_shadow_polygon(
    center: Vec3,
    radius: f32,
    camera: &Camera,
    screen_center: egui::Pos2,
    scale: f32,
) -> Vec<egui::Pos2> {
    let ground_c = Vec3::new(center.x, 0.0, center.z);
    (0..12)
        .map(|i| {
            let a = (i as f32) * std::f32::consts::TAU / 12.0;
            let p = ground_c + Vec3::new(a.cos() * radius, 0.0, a.sin() * radius);
            project_3d(p, camera, screen_center, scale)
        })
        .collect()
}

pub fn energy_to_color(energy: f32, max_energy: f32) -> egui::Color32 {
    if max_energy <= 0.0 {
        return egui::Color32::TRANSPARENT;
    }

    let t = (energy / max_energy).clamp(0.0, 1.0);

    let (r, g, b) = if t < 0.25 {
        let s = t / 0.25;
        (0.0, s, 1.0)
    } else if t < 0.5 {
        let s = (t - 0.25) / 0.25;
        (0.0, 1.0, 1.0 - s)
    } else if t < 0.75 {
        let s = (t - 0.5) / 0.25;
        (s, 1.0, 0.0)
    } else {
        let s = (t - 0.75) / 0.25;
        (1.0, 1.0 - s, 0.0)
    };

    egui::Color32::from_rgb((r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8)
}

/// Render a horizontal slice through the fluid grid as colored rectangles.
///
/// Each cell in the y=`y_slice` plane is drawn as a quad projected through
/// `project_3d`, colored by the selected field value using `energy_to_color`.
#[allow(clippy::too_many_arguments)]
pub fn render_fluid_slice(
    grid: &FluidGrid,
    y_slice: usize,
    mode: FluidVisualizationMode,
    painter: &egui::Painter,
    camera: &Camera,
    screen_center: egui::Pos2,
    scale: f32,
    clip_rect: egui::Rect,
) {
    let j = y_slice.min(grid.ny.saturating_sub(1));

    // Determine the maximum field value for color normalization (scan the slice).
    let mut max_val: f32 = 0.0;
    for k in 0..grid.nz {
        for i in 0..grid.nx {
            let val = sample_field(grid, i, j, k, mode);
            let abs = val.abs();
            if abs > max_val {
                max_val = abs;
            }
        }
    }

    // Fallback: avoid division by zero in energy_to_color.
    if max_val < 1e-12 {
        max_val = 1.0;
    }

    // Draw each cell as a projected quad.
    let dx = grid.dx;
    for k in 0..grid.nz {
        for i in 0..grid.nx {
            let val = sample_field(grid, i, j, k, mode);
            let abs = val.abs();

            // Skip negligible cells for performance.
            if abs < 1e-8 {
                continue;
            }

            let color = energy_to_color(abs, max_val);

            // Semi-transparent so underlying geometry is visible.
            let color = egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 140);

            // World-space corners of this cell in the y=j plane.
            let base = grid.origin + Vec3::new(i as f32 * dx, (j as f32 + 0.5) * dx, k as f32 * dx);

            let corners = [
                base,
                base + Vec3::new(dx, 0.0, 0.0),
                base + Vec3::new(dx, 0.0, dx),
                base + Vec3::new(0.0, 0.0, dx),
            ];

            let screen: Vec<egui::Pos2> = corners
                .iter()
                .map(|&c| project_3d(c, camera, screen_center, scale))
                .collect();

            // Only draw if at least one vertex is inside the clip rect.
            if screen.iter().any(|p| clip_rect.contains(*p)) {
                let mesh = egui::Mesh {
                    indices: vec![0, 1, 2, 0, 2, 3],
                    vertices: screen
                        .iter()
                        .map(|&p| egui::epaint::Vertex {
                            pos: p,
                            uv: egui::Pos2::ZERO,
                            color,
                        })
                        .collect(),
                    texture_id: egui::TextureId::Managed(0),
                };
                painter.add(egui::Shape::mesh(mesh));
            }
        }
    }
}

/// Sample the appropriate field value from a cell based on the visualization mode.
fn sample_field(
    grid: &FluidGrid,
    i: usize,
    j: usize,
    k: usize,
    mode: FluidVisualizationMode,
) -> f32 {
    let idx = grid.idx(i, j, k);
    match mode {
        FluidVisualizationMode::VelocityMagnitude => {
            let center = grid.cell_center(i, j, k);
            grid.velocity_at(center).length()
        }
        FluidVisualizationMode::Pressure => grid.pressure[idx],
        FluidVisualizationMode::Density => grid.density[idx],
        FluidVisualizationMode::LevelSet => grid.level_set[idx],
    }
}

/// Render a horizontal slice through the gas grid as colored rectangles.
///
/// Each cell in the y=`y_slice` plane is drawn as a quad projected through
/// `project_3d`, colored by the selected field value using `energy_to_color`.
/// When the mode is `Concentration`, the `species_idx` selects which species
/// concentration array to visualize.
#[allow(clippy::too_many_arguments)]
pub fn render_gas_slice(
    grid: &GasGrid,
    y_slice: usize,
    species_idx: usize,
    mode: GasVisualizationMode,
    painter: &egui::Painter,
    camera: &Camera,
    screen_center: egui::Pos2,
    scale: f32,
    clip_rect: egui::Rect,
) {
    let j = y_slice.min(grid.ny.saturating_sub(1));

    // Determine the maximum field value for color normalization (scan the slice).
    let mut max_val: f32 = 0.0;
    for k in 0..grid.nz {
        for i in 0..grid.nx {
            let val = sample_gas_field(grid, i, j, k, species_idx, mode);
            let abs = val.abs();
            if abs > max_val {
                max_val = abs;
            }
        }
    }

    // Fallback: avoid division by zero in energy_to_color.
    if max_val < 1e-12 {
        max_val = 1.0;
    }

    // Draw each cell as a projected quad.
    let dx = grid.dx;
    for k in 0..grid.nz {
        for i in 0..grid.nx {
            let val = sample_gas_field(grid, i, j, k, species_idx, mode);
            let abs = val.abs();

            // Skip negligible cells for performance.
            if abs < 1e-8 {
                continue;
            }

            let color = energy_to_color(abs, max_val);

            // Semi-transparent so underlying geometry is visible.
            let color = egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 140);

            // World-space corners of this cell in the y=j plane.
            let base = grid.origin + Vec3::new(i as f32 * dx, (j as f32 + 0.5) * dx, k as f32 * dx);

            let corners = [
                base,
                base + Vec3::new(dx, 0.0, 0.0),
                base + Vec3::new(dx, 0.0, dx),
                base + Vec3::new(0.0, 0.0, dx),
            ];

            let screen: Vec<egui::Pos2> = corners
                .iter()
                .map(|&c| project_3d(c, camera, screen_center, scale))
                .collect();

            // Only draw if at least one vertex is inside the clip rect.
            if screen.iter().any(|p| clip_rect.contains(*p)) {
                let mesh = egui::Mesh {
                    indices: vec![0, 1, 2, 0, 2, 3],
                    vertices: screen
                        .iter()
                        .map(|&p| egui::epaint::Vertex {
                            pos: p,
                            uv: egui::Pos2::ZERO,
                            color,
                        })
                        .collect(),
                    texture_id: egui::TextureId::Managed(0),
                };
                painter.add(egui::Shape::mesh(mesh));
            }
        }
    }
}

/// Sample the appropriate gas field value from a cell based on the visualization mode.
fn sample_gas_field(
    grid: &GasGrid,
    i: usize,
    j: usize,
    k: usize,
    species_idx: usize,
    mode: GasVisualizationMode,
) -> f32 {
    let idx = grid.idx(i, j, k);
    match mode {
        GasVisualizationMode::Concentration => {
            if species_idx < grid.concentrations.len() {
                grid.concentrations[species_idx][idx]
            } else {
                0.0
            }
        }
        GasVisualizationMode::Temperature => grid.temperature[idx],
        GasVisualizationMode::Pressure => grid.pressure[idx],
        GasVisualizationMode::VelocityMagnitude => {
            let vx = grid.vel_x[idx];
            let vy = grid.vel_y[idx];
            let vz = grid.vel_z[idx];
            (vx * vx + vy * vy + vz * vz).sqrt()
        }
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_default_position() {
        let cam = Camera::default();
        assert!(cam.distance > 0.0);
        assert!((cam.target - Vec3::ZERO).length() < 0.01);
    }

    #[test]
    fn test_camera_focus_on() {
        let mut cam = Camera::default();
        let target = Vec3::new(1.0, 2.0, 3.0);
        cam.focus_on(target, 5.0);
        assert!((cam.target - target).length() < 0.01);
        assert!(cam.distance >= 3.0);
    }

    #[test]
    fn test_camera_orbit() {
        let mut cam = Camera::default();
        let old_yaw = cam.yaw;
        cam.orbit(10.0, 0.0);
        assert!((cam.yaw - old_yaw).abs() > 0.01);
    }

    #[test]
    fn test_camera_zoom() {
        let mut cam = Camera::default();
        let old_dist = cam.distance;
        cam.zoom(2.0);
        assert!(cam.distance < old_dist);
    }

    #[test]
    fn test_camera_update_position_public() {
        let mut cam = Camera::default();
        cam.target = Vec3::new(1.0, 0.0, 0.0);
        cam.update_position();
        assert!((cam.position - cam.target).length() > 0.1);
    }

    #[test]
    fn test_camera_smooth_track() {
        let mut cam = Camera::default();
        let target = Vec3::new(5.0, 0.5, 0.0);
        let lerp = 0.05;
        for _ in 0..100 {
            cam.target = cam.target + (target - cam.target) * lerp;
            cam.update_position();
        }
        assert!((cam.target - target).length() < 0.1);
    }

    #[test]
    fn test_project_3d_center_stays_near_center() {
        let cam = Camera::default();
        let screen_center = egui::Pos2::new(500.0, 400.0);
        let sp = project_3d(cam.target, &cam, screen_center, 50.0);
        assert!((sp.x - screen_center.x).abs() < 100.0);
        assert!((sp.y - screen_center.y).abs() < 100.0);
    }

    #[test]
    fn test_camera_set_view_top_pitches_up() {
        let mut cam = Camera::default();
        cam.set_view(CameraView::Top);
        assert!(cam.pitch > 1.5);
    }

    #[test]
    fn test_camera_set_view_front_zero_pitch() {
        let mut cam = Camera::default();
        cam.set_view(CameraView::Front);
        assert!(cam.pitch.abs() < 0.01);
    }

    #[test]
    fn test_camera_smooth_focus_then_tick() {
        let mut cam = Camera::default();
        cam.target = Vec3::ZERO;
        cam.smooth_focus(Vec3::new(5.0, 0.0, 0.0), 2.0);
        for _ in 0..200 {
            cam.tick_focus(0.05);
        }
        assert!((cam.target - Vec3::new(5.0, 0.0, 0.0)).length() < 0.1);
        assert!(cam.focus_target.is_none());
    }

    #[test]
    fn test_camera_fly_moves_both_target_and_position() {
        let mut cam = Camera::default();
        let old_target = cam.target;
        let old_position = cam.position;
        cam.fly(1.0, 0.0, 0.0);
        assert!((cam.target - old_target).length() > 0.5);
        assert!((cam.position - old_position).length() > 0.5);
    }

    #[test]
    fn test_camera_look_changes_target_direction() {
        let mut cam = Camera::default();
        let old_target = cam.target;
        cam.look(100.0, 0.0);
        assert!((cam.target - old_target).length() > 0.1);
    }

    #[test]
    fn test_camera_zoom_toward_biases_target() {
        let mut cam = Camera::default();
        cam.target = Vec3::ZERO;
        let point = Vec3::new(2.0, 0.0, 0.0);
        cam.zoom_toward(point, 4.0);
        assert!(cam.target.x > 0.0);
    }

    #[test]
    fn test_frequency_band_default_is_broadband() {
        assert_eq!(FrequencyBand::default(), FrequencyBand::Broadband);
    }

    #[test]
    fn test_frequency_band_all_narrow_has_six_entries() {
        assert_eq!(FrequencyBand::ALL_NARROW.len(), 6);
    }

    #[test]
    fn test_frequency_band_narrow_indices_zero_through_five() {
        for (i, band) in FrequencyBand::ALL_NARROW.iter().enumerate() {
            assert_eq!(band.narrow_index(), Some(i));
        }
    }

    #[test]
    fn test_frequency_band_broadband_has_no_index() {
        assert_eq!(FrequencyBand::Broadband.narrow_index(), None);
        assert_eq!(FrequencyBand::Broadband.center_hz(), None);
    }

    #[test]
    fn test_frequency_band_centers_are_octaves() {
        // Each narrow band's center should be 2x the previous one.
        let centers: Vec<f32> = FrequencyBand::ALL_NARROW
            .iter()
            .map(|b| b.center_hz().expect("narrow band has center"))
            .collect();
        for w in centers.windows(2) {
            let ratio = w[1] / w[0];
            assert!(
                (ratio - 2.0).abs() < 0.01,
                "expected octave spacing, got ratio {ratio}"
            );
        }
        assert!((centers[0] - 125.0).abs() < 0.1);
        assert!((centers[5] - 4000.0).abs() < 0.1);
    }

    #[test]
    fn test_frequency_band_labels_distinct() {
        let mut labels: Vec<&str> = std::iter::once(FrequencyBand::Broadband.label())
            .chain(FrequencyBand::ALL_NARROW.iter().map(|b| b.label()))
            .collect();
        labels.sort_unstable();
        let count = labels.len();
        labels.dedup();
        assert_eq!(labels.len(), count, "labels should be unique");
    }

    #[test]
    fn test_sample_band_energy_returns_per_band_value() {
        // With 005's [f32;6] landed: each band index returns its slot value;
        // Broadband averages all six.
        let gp = crate::acoustics::GridPoint {
            position: Vec3::ZERO,
            energy: [0.73; 6],
        };
        for &band in FrequencyBand::ALL_NARROW.iter() {
            assert!((sample_band_energy(&gp, band) - 0.73).abs() < 1e-6);
        }
        assert!((sample_band_energy(&gp, FrequencyBand::Broadband) - 0.73).abs() < 1e-6);
        assert!((broadband_energy(&gp) - 0.73).abs() < 1e-6);
    }

    #[test]
    fn test_sample_band_energy_with_zero_energy_grid_point() {
        let gp = crate::acoustics::GridPoint {
            position: Vec3::ZERO,
            energy: [0.0; 6],
        };
        for &band in FrequencyBand::ALL_NARROW.iter() {
            assert_eq!(sample_band_energy(&gp, band), 0.0);
        }
    }

    #[test]
    fn test_shade_color_dark_when_normal_away_from_light() {
        let base = egui::Color32::from_rgb(200, 200, 200);
        let light = scene_light_dir();
        let bright = shade_color(base, light, light, 0.2);
        let dark = shade_color(base, -light, light, 0.2);
        assert!(bright.r() > dark.r());
    }
}

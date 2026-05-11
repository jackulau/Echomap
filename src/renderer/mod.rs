use glam::Vec3;

use crate::fluids::grid::FluidGrid;
use crate::gas::grid::GasGrid;

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

pub struct Camera {
    pub position: Vec3,
    pub target: Vec3,
    pub up: Vec3,
    pub distance: f32,
    pub yaw: f32,
    pub pitch: f32,
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
        self.update_position();
    }

    fn update_position(&mut self) {
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

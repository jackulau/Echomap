use glam::Vec3;

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

use egui;

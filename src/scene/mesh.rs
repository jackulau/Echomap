use glam::Vec3;

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
}

#[derive(Clone, Debug)]
pub struct Triangle {
    pub vertices: [Vertex; 3],
}

impl Triangle {
    pub fn normal(&self) -> Vec3 {
        let e1 = self.vertices[1].position - self.vertices[0].position;
        let e2 = self.vertices[2].position - self.vertices[0].position;
        e1.cross(e2).normalize()
    }

    #[allow(dead_code)]
    pub fn centroid(&self) -> Vec3 {
        (self.vertices[0].position + self.vertices[1].position + self.vertices[2].position) / 3.0
    }

    #[allow(dead_code)]
    pub fn area(&self) -> f32 {
        let e1 = self.vertices[1].position - self.vertices[0].position;
        let e2 = self.vertices[2].position - self.vertices[0].position;
        e1.cross(e2).length() * 0.5
    }
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub triangles: Vec<Triangle>,
}

impl Mesh {
    #[allow(dead_code)]
    pub fn vertex_count(&self) -> usize {
        self.triangles.len() * 3
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);

        for tri in &self.triangles {
            for v in &tri.vertices {
                min = min.min(v.position);
                max = max.max(v.position);
            }
        }

        (min, max)
    }
}

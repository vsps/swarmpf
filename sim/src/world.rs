use crate::math::{v3, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn new(min: Vec3, max: Vec3) -> Aabb {
        Aabb { min, max }
    }

    pub fn closest_point(&self, p: Vec3) -> Vec3 {
        v3(
            p.x.clamp(self.min.x, self.max.x),
            p.y.clamp(self.min.y, self.max.y),
            p.z.clamp(self.min.z, self.max.z),
        )
    }

    /// Slab test along a unit `dir`. Returns entry distance (0 if origin inside).
    pub fn ray(&self, o: Vec3, d: Vec3) -> Option<f32> {
        let mut t0 = 0.0f32;
        let mut t1 = f32::INFINITY;
        for (o, d, lo, hi) in [
            (o.x, d.x, self.min.x, self.max.x),
            (o.y, d.y, self.min.y, self.max.y),
            (o.z, d.z, self.min.z, self.max.z),
        ] {
            if d.abs() < 1e-9 {
                if o < lo || o > hi {
                    return None;
                }
            } else {
                let inv = 1.0 / d;
                let (a, b) = ((lo - o) * inv, (hi - o) * inv);
                t0 = t0.max(a.min(b));
                t1 = t1.min(a.max(b));
                if t0 > t1 {
                    return None;
                }
            }
        }
        Some(t0)
    }
}

/// The level: a flat floor at y = 0 plus axis-aligned boxes.
#[derive(Clone, Debug, Default)]
pub struct World {
    pub boxes: Vec<Aabb>,
}

impl World {
    /// Push a sphere out of the level. Returns the summed contact normals
    /// (zero if no contact) so callers can slide velocity along surfaces.
    pub fn push_sphere(&self, p: &mut Vec3, r: f32) -> Vec3 {
        let mut n_sum = Vec3::ZERO;
        if p.y < r {
            p.y = r;
            n_sum += Vec3::Y;
        }
        for b in &self.boxes {
            let c = b.closest_point(*p);
            let d = *p - c;
            let d2 = d.len2();
            if d2 >= r * r {
                continue;
            }
            let n = if d2 > 1e-12 {
                d * (1.0 / d2.sqrt())
            } else {
                // Centre is inside the box: leave via the nearest face.
                let faces = [
                    (p.x - b.min.x, v3(-1.0, 0.0, 0.0)),
                    (b.max.x - p.x, v3(1.0, 0.0, 0.0)),
                    (p.y - b.min.y, v3(0.0, -1.0, 0.0)),
                    (b.max.y - p.y, v3(0.0, 1.0, 0.0)),
                    (p.z - b.min.z, v3(0.0, 0.0, -1.0)),
                    (b.max.z - p.z, v3(0.0, 0.0, 1.0)),
                ];
                let (dist, n) = faces
                    .iter()
                    .copied()
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .unwrap();
                *p += n * (dist + r);
                n_sum += n;
                continue;
            };
            *p = c + n * r;
            n_sum += n;
        }
        n_sum
    }

    /// True if the sphere overlaps the floor or any box. (Checked directly: opposing
    /// contact normals, e.g. in a doorway, cancel in `push_sphere`'s sum.)
    pub fn sphere_hits(&self, p: Vec3, r: f32) -> bool {
        p.y < r
            || self
                .boxes
                .iter()
                .any(|b| (p - b.closest_point(p)).len2() < r * r)
    }

    /// Nearest level hit along a unit ray (floor included).
    pub fn ray_cast(&self, o: Vec3, d: Vec3, max: f32) -> Option<f32> {
        let mut best = None::<f32>;
        if d.y < -1e-9 && o.y > 0.0 {
            let t = -o.y / d.y;
            if t <= max {
                best = Some(t);
            }
        }
        for b in &self.boxes {
            if let Some(t) = b.ray(o, d) {
                if t <= max && best.is_none_or(|bt| t < bt) {
                    best = Some(t);
                }
            }
        }
        best
    }
}

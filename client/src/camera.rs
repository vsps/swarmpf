//! Minimal column-major matrix math for the camera.

pub type Mat4 = [[f32; 4]; 4];

pub fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut r = [[0.0; 4]; 4];
    for c in 0..4 {
        for row in 0..4 {
            r[c][row] = (0..4).map(|k| a[k][row] * b[c][k]).sum();
        }
    }
    r
}

/// Right-handed perspective with wgpu's 0..1 depth range, looking down -Z in view space.
pub fn perspective(fov_y: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, far / (near - far), -1.0],
        [0.0, 0.0, near * far / (near - far), 0.0],
    ]
}

pub fn look_to(eye: [f32; 3], fwd: [f32; 3], right: [f32; 3], up: [f32; 3]) -> Mat4 {
    let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    [
        [right[0], up[0], -fwd[0], 0.0],
        [right[1], up[1], -fwd[1], 0.0],
        [right[2], up[2], -fwd[2], 0.0],
        [-dot(right, eye), -dot(up, eye), dot(fwd, eye), 1.0],
    ]
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub eye: [f32; 3],
    pub fwd: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub fov_y: f32,
}

impl Camera {
    /// Build from an eye position and yaw / pitch. Yaw 0 faces +Z, matching the sim.
    pub fn from_angles(eye: [f32; 3], yaw: f32, pitch: f32, fov_y: f32) -> Camera {
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let fwd = [sy * cp, sp, cy * cp];
        // With +Y up and forward on +Z, "right" is -X at yaw 0.
        let right = [-cy, 0.0, sy];
        let up = [
            right[1] * fwd[2] - right[2] * fwd[1],
            right[2] * fwd[0] - right[0] * fwd[2],
            right[0] * fwd[1] - right[1] * fwd[0],
        ];
        Camera {
            eye,
            fwd,
            right,
            up,
            fov_y,
        }
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let v = look_to(self.eye, self.fwd, self.right, self.up);
        mul(&perspective(self.fov_y, aspect, 0.05, 500.0), &v)
    }
}

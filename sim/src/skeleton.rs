//! Humanoid rig and procedural walk cycle.
//!
//! Every joint only rotates about X (swing), which is all a walk cycle needs and
//! keeps forward kinematics to a single accumulated angle. The rig is generated
//! in code, so there are no assets and the pose is a pure function of
//! (walk phase, amplitude) that the server and every client evaluate identically.

use crate::math::{v3, Vec3};
use std::f32::consts::PI;

pub const NUM_JOINTS: usize = 16;

pub const PELVIS: usize = 0;
pub const CHEST: usize = 1;
pub const NECK: usize = 2;
pub const HEAD: usize = 3;
pub const SHOULDER_L: usize = 4;
pub const ELBOW_L: usize = 5;
pub const HAND_L: usize = 6;
pub const SHOULDER_R: usize = 7;
pub const ELBOW_R: usize = 8;
pub const HAND_R: usize = 9;
pub const HIP_L: usize = 10;
pub const KNEE_L: usize = 11;
pub const ANKLE_L: usize = 12;
pub const HIP_R: usize = 13;
pub const KNEE_R: usize = 14;
pub const ANKLE_R: usize = 15;

/// Parent joint (`usize::MAX` for the root).
const PARENT: [usize; NUM_JOINTS] = [usize::MAX, 0, 1, 2, 1, 4, 5, 1, 7, 8, 0, 10, 11, 0, 13, 14];

/// Rest-pose offset from the parent joint, in body space (+Y up, +Z forward, +X left).
const OFFSET: [Vec3; NUM_JOINTS] = [
    v3(0.0, 0.95, 0.0),
    v3(0.0, 0.30, 0.0),
    v3(0.0, 0.30, 0.0),
    v3(0.0, 0.15, 0.0),
    v3(0.20, 0.22, 0.0),
    v3(0.0, -0.28, 0.0),
    v3(0.0, -0.26, 0.0),
    v3(-0.20, 0.22, 0.0),
    v3(0.0, -0.28, 0.0),
    v3(0.0, -0.26, 0.0),
    v3(0.10, -0.05, 0.0),
    v3(0.0, -0.45, 0.0),
    v3(0.0, -0.45, 0.0),
    v3(-0.10, -0.05, 0.0),
    v3(0.0, -0.45, 0.0),
    v3(0.0, -0.45, 0.0),
];

pub const WALK_SPEED: f32 = 4.5;
/// Distance covered per full walk cycle (two steps).
const STRIDE: f32 = 1.5;

#[derive(Clone, Copy, Debug)]
pub struct Pose {
    /// Joint positions in body space (feet at y = 0, before yaw / translation).
    pub pos: [Vec3; NUM_JOINTS],
    /// Accumulated swing angle of each joint's frame.
    pub acc: [f32; NUM_JOINTS],
}

/// Advance the walk phase for a given ground speed.
pub fn advance_phase(phase: f32, speed: f32, dt: f32) -> f32 {
    (phase + 2.0 * PI * speed / STRIDE * dt).rem_euclid(2.0 * PI)
}

/// Swing angles per joint for a walk cycle at `phase` and amplitude `amp` in [0, 1].
fn angles(phase: f32, amp: f32) -> ([f32; NUM_JOINTS], f32) {
    let mut a = [0.0f32; NUM_JOINTS];
    let (s, c) = phase.sin_cos();
    // Left leg leads when sin > 0; the right side is half a cycle behind.
    for (side, hip, knee, shoulder, elbow) in [
        (1.0f32, HIP_L, KNEE_L, SHOULDER_L, ELBOW_L),
        (-1.0, HIP_R, KNEE_R, SHOULDER_R, ELBOW_R),
    ] {
        let sp = s * side;
        let cp = c * side;
        // Negative X-swing moves the foot forward (+Z).
        a[hip] = -0.55 * amp * sp;
        // The knee folds while the leg swings forward.
        a[knee] = amp * (0.15 + 0.95 * cp.max(0.0));
        // Arms counter-swing their same-side leg.
        a[shoulder] = 0.45 * amp * sp;
        a[elbow] = -amp * (0.25 + 0.2 * (1.0 - sp) * 0.5);
    }
    // Slight forward lean while moving, and a vertical bob at twice the step rate.
    a[PELVIS] = -0.06 * amp;
    let bob = 0.035 * amp * (2.0 * phase).cos();
    (a, bob)
}

pub fn pose(phase: f32, amp: f32) -> Pose {
    let (ang, bob) = angles(phase, amp.clamp(0.0, 1.0));
    let mut pos = [Vec3::ZERO; NUM_JOINTS];
    let mut acc = [0.0f32; NUM_JOINTS];
    for j in 0..NUM_JOINTS {
        let p = PARENT[j];
        if p == usize::MAX {
            pos[j] = OFFSET[j] + v3(0.0, bob, 0.0);
            acc[j] = ang[j];
        } else {
            pos[j] = pos[p] + OFFSET[j].rot_x(acc[p]);
            acc[j] = acc[p] + ang[j];
        }
    }
    Pose { pos, acc }
}

impl Pose {
    /// World position of a point given in joint `j`'s local frame.
    pub fn attach(&self, j: usize, local: Vec3) -> Vec3 {
        self.pos[j] + local.rot_x(self.acc[j])
    }
}

/// A body-space attachment point for one swarm element.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    pub joint: u8,
    pub local: Vec3,
    /// Distance from the bone axis; larger means closer to the surface.
    pub depth: f32,
}

/// Where the core sits: inside the chest, on the torso axis.
pub const CORE_JOINT: usize = CHEST;
pub const CORE_LOCAL: Vec3 = v3(0.0, 0.12, 0.0);
/// Nominal core height above the feet at rest; also the swarm hover height.
pub const CORE_HEIGHT: f32 = 0.95 + 0.30 + 0.12;

/// (bone start joint, bone end joint, count per side, x radius, z radius)
const BONES: [(usize, usize, usize, f32, f32); 8] = [
    (PELVIS, CHEST, 9, 0.14, 0.10),
    (CHEST, NECK, 9, 0.17, 0.11),
    (SHOULDER_L, ELBOW_L, 4, 0.05, 0.05),
    (ELBOW_L, HAND_L, 4, 0.04, 0.04),
    (SHOULDER_R, ELBOW_R, 4, 0.05, 0.05),
    (ELBOW_R, HAND_R, 4, 0.04, 0.04),
    (HIP_L, KNEE_L, 6, 0.08, 0.08),
    (KNEE_L, ANKLE_L, 6, 0.06, 0.06),
];

pub const NUM_SLOTS: usize = 64;

/// Slot layout for the whole body. Slots are ordered bone by bone;
/// `slot_groups` says which slots share a bone so radii can be matched to depth.
pub fn build_slots() -> (Vec<Slot>, Vec<std::ops::Range<usize>>) {
    let mut slots = Vec::with_capacity(NUM_SLOTS);
    let mut groups = Vec::new();
    let golden = 2.399_963_1f32;

    let mut add_bone = |j: usize, child: usize, n: usize, rx: f32, rz: f32, mirror: bool| {
        let start = slots.len();
        let len = OFFSET[child];
        for k in 0..n {
            let t = (k as f32 + 0.5) / n as f32;
            let th = k as f32 * golden;
            // Alternate outer and inner layers so the torso is two shells deep.
            let layer = if k % 2 == 0 { 1.0 } else { 0.45 };
            let (sx, cz) = (th.cos() * rx * layer, th.sin() * rz * layer);
            let local = len * t + v3(sx, 0.0, cz);
            let depth = v3(sx, 0.0, cz).len();
            slots.push(Slot {
                joint: j as u8,
                local,
                depth,
            });
            if mirror {
                // Right-side copy: mirror across the sagittal plane.
                let mj = mirror_joint(j);
                slots.push(Slot {
                    joint: mj as u8,
                    local: v3(-local.x, local.y, local.z),
                    depth,
                });
            }
        }
        groups.push(start..slots.len());
    };

    for (i, &(a, b, n, rx, rz)) in BONES.iter().enumerate() {
        // Arms are listed for both sides; only the legs (entries 6..) are mirrored.
        let mirror = i >= 6;
        add_bone(a, b, n, rx, rz, mirror);
    }
    // Head: six spheres on an octahedron around the head centre.
    let start = slots.len();
    let hc = v3(0.0, -0.08, 0.0);
    for d in [
        v3(0.09, 0.0, 0.0),
        v3(-0.09, 0.0, 0.0),
        v3(0.0, 0.0, 0.09),
        v3(0.0, 0.0, -0.09),
        v3(0.0, 0.09, 0.0),
        v3(0.0, -0.09, 0.0),
    ] {
        slots.push(Slot {
            joint: HEAD as u8,
            local: hc + d,
            depth: d.len(),
        });
    }
    groups.push(start..slots.len());
    (slots, groups)
}

fn mirror_joint(j: usize) -> usize {
    match j {
        HIP_L => HIP_R,
        KNEE_L => KNEE_R,
        SHOULDER_L => SHOULDER_R,
        ELBOW_L => ELBOW_R,
        _ => j,
    }
}

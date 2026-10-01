//! One player: a body-shaped cloud of spheres around a small, lethal core.
//!
//! Coherent: elements spring to slots on the humanoid rig; the body moves as a
//! capsule and can act (fire, use doors).
//! Dispersed: elements flock around the core (the boid leader); the core moves
//! as a tiny sphere so the swarm fits through gaps the body cannot, but the
//! player cannot act.

use crate::math::{v3, Rng, Vec3};
use crate::skeleton::{
    self, Pose, Slot, CORE_HEIGHT, CORE_JOINT, CORE_LOCAL, NUM_SLOTS, WALK_SPEED,
};
use crate::world::World;

pub const R_MIN: f32 = 0.06;
pub const R_MAX: f32 = 0.12;
/// Every sphere takes two hits, whatever its size or the weapon.
pub const ELEM_HP: u8 = 2;
/// Seconds per regrown sphere. Destroyed spheres come back first, nearest the core first.
pub const REGEN_TIME: f32 = 3.0;
/// Speed (m/s) a hit knocks a smallest-size sphere along the shot; bigger ones move less
/// (by the square root of the size ratio, so every hit reads).
pub const RECOIL: f32 = 10.0;
/// After a hit the sphere's slot spring goes slack for this long (s), so it visibly flies off
/// and bumps its neighbours before being reeled back in.
pub const STUN_TIME: f32 = 0.45;
/// Spring strength left at the start of a stun (it recovers linearly).
const STUN_FLOOR: f32 = 0.05;
/// Bounciness of sphere-sphere contacts within one swarm.
const RESTITUTION: f32 = 0.8;
/// Coherent contacts only fire once two spheres are this much closer than their slots are,
/// so the rest pose (whose slots overlap) never fights the springs.
const CONTACT_SLACK: f32 = 0.9;
pub const CORE_RADIUS: f32 = 0.04;
/// The core's hit sphere is inflated so it is hittable at all.
pub const CORE_HIT_SCALE: f32 = 1.5;
pub const CORE_ID: u8 = u8::MAX;

pub const BODY_RADIUS: f32 = 0.3;
pub const BODY_HEIGHT: f32 = 1.8;
pub const DISPERSED_SPEED: f32 = 6.5;
/// Seconds to fully re-form the body.
pub const BLEND_TIME: f32 = 0.4;
/// Seconds to fully disperse: breaking apart is twice as quick as re-forming.
pub const DISPERSE_TIME: f32 = 0.2;
pub const SWARM_RADIUS: f32 = 1.0;
/// Highest the dispersed core can fly: 25% above the body's height.
pub const SWARM_MAX_Y: f32 = BODY_HEIGHT * 1.25;
/// Near the ground the flock squashes vertically, by up to this factor with the core at the floor.
const GROUND_SQUASH: f32 = 3.0;

/// Coherent mode: each sphere is attracted to its slot by a soft, underdamped spring. Stiffness
/// varies per sphere (heavier is softer, plus a random factor from the seed), so spheres trail
/// by different amounts and the body wobbles and smears instead of moving as one rigid piece.
const K_SLOT: f32 = 40.0;
const K_JITTER: (f32, f32) = (0.5, 1.5);
/// Damping ratio of the slot spring: below 1, spheres overshoot and wobble when you stop.
const SLOT_ZETA: f32 = 0.4;
/// Fraction of the slot's velocity the damping matches. Below 1 the sphere lags its slot by
/// about `2 * SLOT_ZETA * (1 - SLOT_FOLLOW) / sqrt(k)` seconds of motion.
const SLOT_FOLLOW: f32 = 0.45;
/// Coherent spheres drift around their slots on their own slow rhythms, so the body never
/// looks welded to the skeleton: amplitude (m) standing still, and extra at full walking speed.
pub const WANDER: f32 = 0.03;
const WANDER_WALK: f32 = 0.04;
/// Downward kick (m/s) every sphere gets at each footfall, scaled by walk amplitude and a
/// per-sphere factor, so the body jiggles with every step.
const STEP_KICK: f32 = 0.9;
/// The pull grows with distance, `k * (1 + (d / REEL_DIST)^2)` capped at `REEL_MAX` times, so
/// spheres stay loose near their slot but stragglers and re-forming swarms are reeled in fast.
const REEL_DIST: f32 = 0.35;
const REEL_MAX: f32 = 5.0;
/// Speed caps: the coherent body needs headroom for swinging feet on top of walking speed.
const MAX_SPEED_SWARM: f32 = 14.0;
const MAX_SPEED_BODY: f32 = 28.0;

#[derive(Clone, Debug)]
pub struct Element {
    pub pos: Vec3,
    pub vel: Vec3,
    /// Fixed at spawn from the player's seed.
    pub r: f32,
    pub hp: u8,
    pub max_hp: u8,
}

impl Element {
    pub fn alive(&self) -> bool {
        self.hp > 0
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Input {
    /// Strafe (+ right) and forward, each in [-1, 1].
    pub strafe: f32,
    pub forward: f32,
    pub yaw: f32,
    /// Look pitch (+ up). Steers the swarm up and down while dispersed; the body ignores it.
    pub pitch: f32,
    /// Hold to be dispersed; release to re-form (if there is room).
    pub disperse: bool,
}

#[derive(Clone, Debug)]
pub struct Player {
    pub seed: u32,
    pub alive: bool,
    /// Body anchor: position of the feet.
    pub feet: Vec3,
    pub yaw: f32,
    pub body_vel: Vec3,
    pub want_disperse: bool,
    /// 0 = fully coherent, 1 = fully dispersed.
    pub blend: f32,
    pub core: Element,
    pub core_vel: Vec3,
    pub elems: Vec<Element>,
    pub walk_phase: f32,
    pub walk_amp: f32,
    slots: Vec<Slot>,
    targets: Vec<Vec3>,
    prev_targets: Vec<Vec3>,
    target_vels: Vec<Vec3>,
    axes: Vec<Vec3>,
    noise: Vec<[f32; 6]>,
    /// Per-element slot spring stiffness, and time left on its post-hit slack spell.
    stiff: Vec<f32>,
    stun: Vec<f32>,
    time: f32,
    regen_timer: f32,
    /// Per-tick scratch, kept to avoid allocating every tick.
    snapshot: Vec<(Vec3, Vec3, f32, bool)>,
    near: Vec<usize>,
}

impl Player {
    pub fn spawn(seed: u32, feet: Vec3) -> Player {
        let (slots, groups) = skeleton::build_slots();
        debug_assert_eq!(slots.len(), NUM_SLOTS);
        let mut rng = Rng::new(seed);

        // Radii: drawn per bone, then matched to slot depth so big spheres form
        // the outer shell and small ones sit inside.
        let mut radii = vec![0.0f32; slots.len()];
        for g in groups {
            let mut rs: Vec<f32> = g.clone().map(|_| rng.range(R_MIN, R_MAX)).collect();
            rs.sort_by(|a, b| b.total_cmp(a));
            let mut idx: Vec<usize> = g.collect();
            idx.sort_by(|&a, &b| slots[b].depth.total_cmp(&slots[a].depth));
            for (i, r) in idx.into_iter().zip(rs) {
                radii[i] = r;
            }
        }

        let axes = (0..slots.len()).map(|_| rng.unit_vec()).collect();
        let noise = (0..slots.len())
            .map(|_| {
                [
                    rng.range(0.0, std::f32::consts::TAU),
                    rng.range(0.0, std::f32::consts::TAU),
                    rng.range(0.0, std::f32::consts::TAU),
                    rng.range(2.0, 5.0),
                    rng.range(2.0, 5.0),
                    rng.range(2.0, 5.0),
                ]
            })
            .collect();
        let stiff = radii
            .iter()
            .map(|&r| K_SLOT * rng.range(K_JITTER.0, K_JITTER.1) * (R_MIN / r))
            .collect();

        let mut p = Player {
            seed,
            alive: true,
            feet,
            yaw: 0.0,
            body_vel: Vec3::ZERO,
            want_disperse: false,
            blend: 0.0,
            core: Element {
                pos: Vec3::ZERO,
                vel: Vec3::ZERO,
                r: CORE_RADIUS,
                hp: 1,
                max_hp: 1,
            },
            core_vel: Vec3::ZERO,
            elems: radii
                .iter()
                .map(|&r| Element {
                    pos: Vec3::ZERO,
                    vel: Vec3::ZERO,
                    r,
                    hp: ELEM_HP,
                    max_hp: ELEM_HP,
                })
                .collect(),
            walk_phase: 0.0,
            walk_amp: 0.0,
            slots,
            targets: vec![Vec3::ZERO; NUM_SLOTS],
            prev_targets: vec![Vec3::ZERO; NUM_SLOTS],
            target_vels: vec![Vec3::ZERO; NUM_SLOTS],
            axes,
            noise,
            stiff,
            stun: vec![0.0; NUM_SLOTS],
            time: 0.0,
            regen_timer: 0.0,
            snapshot: Vec::new(),
            near: Vec::new(),
        };
        p.refresh_targets();
        for i in 0..p.elems.len() {
            p.elems[i].pos = p.targets[i];
            p.prev_targets[i] = p.targets[i];
        }
        p.core.pos = p.core_pose_pos();
        p
    }

    /// Where the body's feet go when it re-forms around the core: under it, but never below the
    /// floor (a body re-formed in mid-air falls).
    fn feet_under_core(&self) -> Vec3 {
        let c = self.core.pos;
        v3(c.x, (c.y - CORE_HEIGHT).max(0.0), c.z)
    }

    /// Body actions (fire, open doors) need a fully formed body.
    pub fn can_act(&self) -> bool {
        self.alive && self.blend <= 0.0 && !self.want_disperse
    }

    pub fn pose(&self) -> Pose {
        skeleton::pose(self.walk_phase, self.walk_amp)
    }

    fn to_world(&self, body: Vec3) -> Vec3 {
        self.feet + body.rot_y(self.yaw)
    }

    fn core_pose_pos(&self) -> Vec3 {
        self.to_world(self.pose().attach(CORE_JOINT, CORE_LOCAL))
    }

    fn refresh_targets(&mut self) {
        let pose = self.pose();
        for (i, s) in self.slots.iter().enumerate() {
            self.targets[i] = self.to_world(pose.attach(s.joint as usize, s.local));
        }
    }

    pub fn slot_target(&self, i: usize) -> Vec3 {
        self.targets[i]
    }

    fn capsule_spheres(feet: Vec3) -> [Vec3; 3] {
        [
            feet + v3(0.0, BODY_RADIUS, 0.0),
            feet + v3(0.0, BODY_HEIGHT * 0.5, 0.0),
            feet + v3(0.0, BODY_HEIGHT - BODY_RADIUS, 0.0),
        ]
    }

    /// Would the body capsule fit standing at `feet`?
    pub fn body_fits(world: &World, feet: Vec3) -> bool {
        Self::capsule_spheres(feet)
            .iter()
            .all(|&c| !world.sphere_hits(c, BODY_RADIUS - 0.01))
    }

    fn resolve_body(&mut self, world: &World) {
        for _ in 0..3 {
            for mut c in Self::capsule_spheres(self.feet) {
                let before = c;
                let n = world.push_sphere(&mut c, BODY_RADIUS);
                if n != Vec3::ZERO {
                    self.feet += c - before;
                    let n = n.normalized();
                    let vn = self.body_vel.dot(n);
                    if vn < 0.0 {
                        self.body_vel -= n * vn;
                    }
                }
            }
        }
        if self.feet.y < 0.0 {
            self.feet.y = 0.0;
            self.body_vel.y = self.body_vel.y.max(0.0);
        }
    }

    pub fn tick(&mut self, input: &Input, world: &World, dt: f32) {
        if !self.alive {
            return;
        }
        self.time += dt;
        self.yaw = input.yaw;
        self.want_disperse = input.disperse
            || (self.want_disperse && !Self::body_fits(world, self.feet_under_core()));

        let dir = v3(-input.strafe, 0.0, input.forward)
            .clamp_len(1.0)
            .rot_y(self.yaw);

        let target = if self.want_disperse { 1.0 } else { 0.0 };
        self.blend = if self.blend < target {
            (self.blend + dt / DISPERSE_TIME).min(target)
        } else {
            (self.blend - dt / BLEND_TIME).max(target)
        };

        let coherent = self.blend <= 0.0 && !self.want_disperse;
        let ground_speed;
        if coherent {
            // Body mode: capsule movement with gravity.
            let want = dir * WALK_SPEED;
            self.body_vel.x = want.x;
            self.body_vel.z = want.z;
            self.body_vel.y -= 20.0 * dt;
            let old = self.feet;
            self.feet += self.body_vel * dt;
            self.resolve_body(world);
            ground_speed = ((self.feet - old) * (1.0 / dt)).len().min(WALK_SPEED * 1.5);
            self.core_vel = Vec3::ZERO;
        } else {
            // Swarm mode: the core is the leader and moves as a tiny sphere. Forward follows the
            // look direction including pitch, so looking down flies it to the ground.
            let (sp, cp) = input.pitch.sin_cos();
            let look = v3(0.0, sp, cp).rot_y(self.yaw);
            let side = v3(-1.0, 0.0, 0.0).rot_y(self.yaw);
            let want =
                (look * input.forward + side * input.strafe).clamp_len(1.0) * DISPERSED_SPEED;
            self.core_vel += (want - self.core_vel) * (dt * 8.0).min(1.0);
            let old = self.core.pos;
            self.core.pos += self.core_vel * dt;
            if self.core.pos.y > SWARM_MAX_Y {
                self.core.pos.y = SWARM_MAX_Y;
                self.core_vel.y = self.core_vel.y.min(0.0);
            }
            let n = world.push_sphere(&mut self.core.pos, CORE_RADIUS);
            if n != Vec3::ZERO {
                let n = n.normalized();
                let vn = self.core_vel.dot(n);
                if vn < 0.0 {
                    self.core_vel -= n * vn;
                }
            }
            self.core.vel = (self.core.pos - old) * (1.0 / dt);
            // The (fading) body pose follows the core so re-forming starts in place.
            self.feet = self.feet_under_core();
            self.body_vel = Vec3::ZERO;
            ground_speed = 0.0;
        }

        let target_amp = if coherent {
            (ground_speed / WALK_SPEED).min(1.0)
        } else {
            0.0
        };
        self.walk_amp += (target_amp - self.walk_amp) * (dt * 10.0).min(1.0);
        let old_phase = self.walk_phase;
        self.walk_phase = skeleton::advance_phase(self.walk_phase, ground_speed, dt);
        // A foot lands every half cycle.
        let half = std::f32::consts::PI;
        if coherent && (old_phase / half).floor() != (self.walk_phase / half).floor() {
            for (e, n) in self.elems.iter_mut().zip(&self.noise) {
                e.vel.y -= STEP_KICK * self.walk_amp * (n[3] / 3.5);
            }
        }

        self.prev_targets.copy_from_slice(&self.targets);
        self.refresh_targets();
        // Slot velocities: the spring damps against these so a moving body does not lag.
        for i in 0..NUM_SLOTS {
            self.target_vels[i] = (self.targets[i] - self.prev_targets[i]) * (1.0 / dt);
        }
        if coherent {
            self.core.pos = self.core_pose_pos();
            self.core.vel = Vec3::ZERO;
        }
        self.step_elements(world, dt);
        for s in &mut self.stun {
            *s = (*s - dt).max(0.0);
        }
        self.regenerate(dt);
    }

    /// Restore one sphere every `REGEN_TIME` while any is missing or damaged. A destroyed
    /// sphere regrows at the core and flows back out to its slot (or into the flock).
    fn regenerate(&mut self, dt: f32) {
        let hurt = |e: &Element| e.hp < e.max_hp;
        if !self.elems.iter().any(hurt) {
            self.regen_timer = 0.0;
            return;
        }
        self.regen_timer += dt;
        if self.regen_timer < REGEN_TIME {
            return;
        }
        self.regen_timer -= REGEN_TIME;
        // Destroyed before damaged, then innermost first so the core is re-covered.
        let slots = &self.slots;
        let i = (0..self.elems.len())
            .filter(|&i| hurt(&self.elems[i]))
            .min_by(|&a, &b| {
                let (ea, eb) = (&self.elems[a], &self.elems[b]);
                ea.hp
                    .cmp(&eb.hp)
                    .then(slots[a].depth.total_cmp(&slots[b].depth))
            })
            .unwrap();
        let e = &mut self.elems[i];
        if !e.alive() {
            e.pos = self.core.pos;
            e.vel = self.core.vel;
        }
        e.hp = e.max_hp;
    }

    /// A hit knocks the sphere along the shot and slackens its slot spring for `STUN_TIME`;
    /// it bumps its neighbours, then the spring or flock pulls it back.
    pub fn recoil(&mut self, elem: usize, dir: Vec3) {
        let e = &mut self.elems[elem];
        e.vel += dir * (RECOIL * (R_MIN / e.r).sqrt());
        self.stun[elem] = STUN_TIME;
    }

    fn step_elements(&mut self, world: &World, dt: f32) {
        let w_c = 1.0 - self.blend;
        let w_d = self.blend;
        let core = self.core.pos;
        let core_vel = self.core.vel;
        let t = self.time;
        let low = ((CORE_HEIGHT - core.y) / CORE_HEIGHT).clamp(0.0, 1.0);
        let squash = 1.0 + (GROUND_SQUASH - 1.0) * low;

        // Neighbour accelerations read the frame-start positions.
        let mut snapshot = std::mem::take(&mut self.snapshot);
        snapshot.clear();
        snapshot.extend(self.elems.iter().map(|e| (e.pos, e.vel, e.r, e.alive())));

        for i in 0..self.elems.len() {
            if !snapshot[i].3 {
                continue;
            }
            let (p, v, r, _) = snapshot[i];
            let mut a = Vec3::ZERO;

            if w_c > 0.0 {
                // Attract to the slot, damped against part of the slot's own velocity.
                let slack = self.stun[i] / STUN_TIME;
                let d = (self.targets[i] - p).len() / REEL_DIST;
                let reel = (1.0 + d * d).min(REEL_MAX);
                let k = self.stiff[i] * reel * (1.0 - (1.0 - STUN_FLOOR) * slack);
                let c = 2.0 * SLOT_ZETA * k.sqrt();
                let n = &self.noise[i];
                let wander = v3(
                    (t * n[3] * 0.4 + n[1]).sin(),
                    (t * n[4] * 0.4 + n[2]).sin(),
                    (t * n[5] * 0.4 + n[0]).sin(),
                ) * (WANDER + WANDER_WALK * self.walk_amp);
                let target = self.targets[i] + wander;
                let tv = self.target_vels[i] * SLOT_FOLLOW;
                a += ((target - p) * k + (tv - v) * c) * w_c;
            }

            if w_d > 0.0 {
                let inv_mass = R_MIN / r;
                let mut f = Vec3::ZERO;

                // Cohesion: soft shell around the leader, squashed vertically as the core nears the
                // ground (vertical offsets count `squash` times over), so a low swarm flattens.
                let mut to_core = core - p;
                to_core.y *= squash;
                let dist = to_core.len();
                let dir = to_core.normalized();
                let mut fc = if dist > SWARM_RADIUS {
                    dir * (28.0 * (dist - SWARM_RADIUS) + 6.0)
                } else if dist < 0.25 {
                    -dir * 20.0 * (0.25 - dist)
                } else {
                    dir * 6.0 * (dist - 0.5 * SWARM_RADIUS)
                };
                fc.y *= squash;
                f += fc;
                // Follow the leader's velocity.
                f += (core_vel - v) * 2.5;
                // Separation.
                for (j, &(pj, _, rj, aj)) in snapshot.iter().enumerate() {
                    if j == i || !aj {
                        continue;
                    }
                    let d = p - pj;
                    let d2 = d.len2();
                    let min_d = (r + rj) * 1.6;
                    if d2 < min_d * min_d && d2 > 1e-10 {
                        let dj = d2.sqrt();
                        f += d * (1.0 / dj) * (60.0 * (1.0 - dj / min_d));
                    }
                }
                // Swirl about a per-element axis, plus wobble so it is hard to track.
                let tangent = self.axes[i].cross(p - core).normalized();
                f += tangent * 14.0;
                let n = &self.noise[i];
                f += v3(
                    (t * n[3] + n[0]).sin(),
                    (t * n[4] + n[1]).sin(),
                    (t * n[5] + n[2]).sin(),
                ) * 10.0;
                f -= v * 1.2;
                a += f * (inv_mass * w_d);
            }

            let e = &mut self.elems[i];
            e.vel = (v + a * dt).clamp_len(MAX_SPEED_BODY * w_c + MAX_SPEED_SWARM * w_d);
            e.pos = p + e.vel * dt;
        }
        self.snapshot = snapshot;
        self.collide_elements();
        self.find_near_boxes(world);
        for e in self.elems.iter_mut().filter(|e| e.alive()) {
            let n = world.push_sphere_among(self.near.iter().copied(), &mut e.pos, e.r);
            if n != Vec3::ZERO {
                let n = n.normalized();
                let vn = e.vel.dot(n);
                if vn < 0.0 {
                    e.vel -= n * vn;
                }
            }
        }
    }

    /// Boxes near any live sphere, padded well past a sphere radius so that being pushed out of
    /// one box never brings a sphere to a box that was left out.
    fn find_near_boxes(&mut self, world: &World) {
        let mut lo = v3(f32::MAX, f32::MAX, f32::MAX);
        let mut hi = v3(f32::MIN, f32::MIN, f32::MIN);
        for e in self.elems.iter().filter(|e| e.alive()) {
            lo = v3(lo.x.min(e.pos.x), lo.y.min(e.pos.y), lo.z.min(e.pos.z));
            hi = v3(hi.x.max(e.pos.x), hi.y.max(e.pos.y), hi.z.max(e.pos.z));
        }
        let pad = v3(1.0, 1.0, 1.0);
        world.boxes_near(lo - pad, hi + pad, &mut self.near);
    }

    /// Spheres of one swarm bump into each other, so a hit sphere knocks its neighbours.
    /// Coherent, the contact distance is capped by the slots' own spacing (see `CONTACT_SLACK`);
    /// dispersed, it is the plain sum of radii. Mass goes with volume.
    fn collide_elements(&mut self) {
        let w_d = self.blend;
        let n = self.elems.len();
        for i in 0..n {
            if !self.elems[i].alive() {
                continue;
            }
            for j in i + 1..n {
                let (a, b) = (&self.elems[i], &self.elems[j]);
                if !b.alive() {
                    continue;
                }
                let d = b.pos - a.pos;
                let d2 = d.len2();
                let touch = a.r + b.r;
                // Contact distance never exceeds touch: skip far pairs before any square root.
                if d2 >= touch * touch {
                    continue;
                }
                let rest = (self.targets[j] - self.targets[i]).len() * CONTACT_SLACK;
                let coherent = touch.min(rest);
                let contact = coherent + (touch - coherent) * w_d;
                if d2 >= contact * contact || d2 < 1e-12 {
                    continue;
                }
                let dist = d2.sqrt();
                let nrm = d * (1.0 / dist);
                let (ia, ib) = (1.0 / (a.r * a.r * a.r), 1.0 / (b.r * b.r * b.r));
                let (sa, sb) = (ia / (ia + ib), ib / (ia + ib));
                let overlap = contact - dist;
                let vn = (b.vel - a.vel).dot(nrm);
                let dv = if vn < 0.0 {
                    -(1.0 + RESTITUTION) * vn
                } else {
                    0.0
                };
                let a = &mut self.elems[i];
                a.pos -= nrm * (overlap * sa);
                a.vel -= nrm * (dv * sa);
                let b = &mut self.elems[j];
                b.pos += nrm * (overlap * sb);
                b.vel += nrm * (dv * sb);
            }
        }
    }

    pub fn alive_elements(&self) -> usize {
        self.elems.iter().filter(|e| e.alive()).count()
    }

    /// Kill the player: the core is gone, so the body falls apart into debris.
    pub fn kill(&mut self) {
        if !self.alive {
            return;
        }
        self.alive = false;
        let mut rng = Rng::new(self.seed ^ 0xDEAD);
        let core = self.core.pos;
        for e in self.elems.iter_mut().filter(|e| e.alive()) {
            let out = (e.pos - core).normalized();
            e.vel = out * rng.range(1.0, 4.0) + v3(0.0, rng.range(1.0, 4.0), 0.0);
        }
    }

    /// Advance a dead player's spheres: they tumble to the floor and settle.
    pub fn step_debris(&mut self, world: &World, dt: f32) {
        for e in self.elems.iter_mut().filter(|e| e.alive()) {
            e.vel.y -= 9.8 * dt;
            e.pos += e.vel * dt;
        }
        self.find_near_boxes(world);
        for e in self.elems.iter_mut().filter(|e| e.alive()) {
            let n = world.push_sphere_among(self.near.iter().copied(), &mut e.pos, e.r);
            if n != Vec3::ZERO {
                let n = n.normalized();
                let vn = e.vel.dot(n);
                if vn < 0.0 {
                    // Bounce a little, then lose horizontal speed to friction.
                    e.vel -= n * vn * 1.3;
                    e.vel.x *= 0.9;
                    e.vel.z *= 0.9;
                }
            }
        }
    }
}

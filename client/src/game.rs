//! Local single-process game: the shared sim plus a test level and a few bots.
//! Networking will replace the bots and the direct `tick` calls later.

use crate::camera::Camera;
use crate::renderer::{BoxInst, Scene, SphereInst, TracerInst};
use sim::hit::{aim_distance, hitscan, HitEvent, Weapon};
use sim::math::{v3, Rng, Vec3};
use sim::player::{Input, Player, CORE_ID};
use sim::world::{Aabb, World};
use sim::DT;

pub const EYE_HEIGHT: f32 = 1.62;

pub struct Gun {
    pub name: &'static str,
    pub weapon: Weapon,
    /// Seconds between shots.
    pub cooldown: f32,
    /// Keeps firing while the trigger is held.
    pub auto: bool,
    /// Hitscans per shot, spread uniformly over a cone of this full apex angle (degrees).
    pub pellets: u32,
    pub spread_deg: f32,
    pub color: [f32; 3],
}

pub const WEAPONS: [Gun; 3] = [
    Gun {
        name: "rifle",
        weapon: Weapon::RIFLE,
        cooldown: 0.11,
        auto: true,
        pellets: 1,
        spread_deg: 0.0,
        color: [1.0, 0.85, 0.5],
    },
    Gun {
        name: "shotgun",
        weapon: Weapon::SHOTGUN_PELLET,
        cooldown: 0.8,
        auto: false,
        pellets: 15,
        spread_deg: 6.0,
        color: [1.0, 0.5, 0.2],
    },
    Gun {
        name: "railgun",
        weapon: Weapon::RAILGUN,
        cooldown: 0.9,
        auto: false,
        pellets: 1,
        spread_deg: 0.0,
        color: [0.4, 0.9, 1.0],
    },
];

const TRACER_SPEED: f32 = 320.0;
const TRACER_LEN: f32 = 7.0;
/// Camera distance behind the pivot and its offset over the right shoulder.
const CAM_BACK: f32 = 3.2;
const CAM_SHOULDER: f32 = 0.55;
const CAM_UP: f32 = 0.35;
/// Extra pull-back while dispersed, so the whole flock stays in view.
const CAM_BACK_SWARM: f32 = 1.6;
/// Time constants (s) of the camera chasing its goal: tight when coherent, loose when
/// dispersed so the swarm visibly surges ahead of the view.
const CAM_LAG_BODY: f32 = 0.04;
const CAM_LAG_SWARM: f32 = 0.22;
/// The third-person camera is a small sphere that collides with the level like an element.
const CAM_RADIUS: f32 = 0.15;
/// Speed (m/s) the camera keeps up while retracing the core's trail around a wall.
const CAM_TRAIL_SPEED: f32 = 8.0;
/// Core breadcrumbs: spacing (m) and how many to keep.
const TRAIL_STEP: f32 = 0.25;
const TRAIL_LEN: usize = 48;
/// Emission radiance of a healthy sphere; the core is far brighter.
const EMIT: f32 = 7.0;
const CORE_EMIT: f32 = 60.0;
/// Colour of a sphere that has been hit, until it regrows.
const HIT_COLOR: [f32; 3] = [1.0, 0.08, 0.05];

struct Tracer {
    a: Vec3,
    b: Vec3,
    age: f32,
    color: [f32; 3],
}

const PALETTE: [[f32; 3]; 6] = [
    [0.30, 0.65, 1.00],
    [1.00, 0.55, 0.25],
    [0.45, 0.95, 0.55],
    [0.95, 0.45, 0.85],
    [0.95, 0.90, 0.40],
    [0.55, 0.60, 1.00],
];

#[derive(Clone, Copy, PartialEq)]
enum Bot {
    /// The local player.
    Human,
    Statue,
    /// Walks a circle in coherent mode.
    Circler {
        center: Vec3,
        radius: f32,
    },
    /// Patrols back and forth, dispersing every few seconds.
    Swarmer,
}

pub struct Game {
    pub world: World,
    pub players: Vec<Player>,
    pub yaw: f32,
    pub pitch: f32,
    pub weapon: usize,
    pub third_person: bool,
    time: f32,
    bots: Vec<Bot>,
    spawns: Vec<Vec3>,
    dead_for: Vec<f32>,
    next_seed: u32,
    pub last_hits: Vec<HitEvent>,
    pub fire_held: bool,
    cooldown: f32,
    pub hit_flash: f32,
    tracers: Vec<Tracer>,
    /// Third-person camera sphere, the player it follows (seed; a respawn snaps it), and the
    /// core's recent path, which the camera retraces when a wall cuts off its view.
    cam_pos: Vec3,
    cam_seed: u32,
    trail: Vec<Vec3>,
    /// Shotgun spread.
    rng: Rng,
}

/// Index of the ceiling in `level()`'s boxes.
const CEILING: usize = 11;

fn level() -> World {
    let b = |x0, y0, z0, x1, y1, z1| Aabb::new(v3(x0, y0, z0), v3(x1, y1, z1));
    World {
        boxes: vec![
            // Outer walls.
            b(-20.5, 0.0, -20.5, 20.5, 4.0, -20.0),
            b(-20.5, 0.0, 20.0, 20.5, 4.0, 20.5),
            b(-20.5, 0.0, -20.0, -20.0, 4.0, 20.0),
            b(20.0, 0.0, -20.0, 20.5, 4.0, 20.0),
            // Dividing wall at z = 6: a 0.5 m crack at x = 0 (swarm only) and a 1.4 m door at x = 8.
            b(-20.0, 0.0, 5.75, -0.25, 4.0, 6.25),
            b(0.25, 0.0, 5.75, 7.3, 4.0, 6.25),
            b(8.7, 0.0, 5.75, 20.0, 4.0, 6.25),
            // Cover and pillars.
            b(-8.0, 0.0, -4.0, -6.0, 1.2, -3.0),
            b(4.0, 0.0, -8.0, 5.0, 3.0, -7.0),
            b(-3.0, 0.0, 10.0, -2.0, 3.0, 11.0),
            b(9.0, 0.0, 12.0, 11.0, 1.0, 13.0),
            // Ceiling: collides with the camera and stops shots, as well as bouncing light.
            b(-20.5, 4.0, -20.5, 20.5, 4.5, 20.5),
        ],
    }
}

fn box_color(i: usize) -> [f32; 4] {
    if i == CEILING {
        [0.5, 0.5, 0.52, 0.0]
    } else if i < 4 {
        [0.62, 0.64, 0.70, 0.0]
    } else if i < 7 {
        [0.75, 0.68, 0.62, 0.0]
    } else {
        [0.55, 0.62, 0.75, 0.0]
    }
}

impl Game {
    pub fn new() -> Game {
        let spawns = vec![
            v3(0.0, 0.0, -14.0),
            v3(-8.0, 0.0, 0.0),
            v3(6.0, 0.0, -4.0),
            v3(0.0, 0.0, 14.0),
        ];
        let bots = vec![
            Bot::Human,
            Bot::Statue,
            Bot::Circler {
                center: v3(6.0, 0.0, -4.0),
                radius: 3.0,
            },
            Bot::Swarmer,
        ];
        let players: Vec<Player> = spawns
            .iter()
            .enumerate()
            .map(|(i, &p)| Player::spawn(100 + i as u32, p))
            .collect();
        let n = players.len();
        Game {
            world: level(),
            players,
            yaw: 0.0,
            pitch: 0.0,
            weapon: 0,
            third_person: true,
            time: 0.0,
            bots,
            spawns,
            dead_for: vec![0.0; n],
            next_seed: 1000,
            last_hits: Vec::new(),
            fire_held: false,
            cooldown: 0.0,
            hit_flash: 0.0,
            tracers: Vec::new(),
            cam_pos: Vec3::ZERO,
            cam_seed: u32::MAX,
            trail: Vec::new(),
            rng: Rng::new(0x5407),
        }
    }

    fn bot_input(&self, i: usize) -> Input {
        let p = &self.players[i];
        match self.bots[i] {
            Bot::Human | Bot::Statue => Input {
                yaw: p.yaw,
                ..Default::default()
            },
            Bot::Circler { center, radius } => {
                let a = self.time * 0.5;
                let want = center + v3(a.cos() * radius, 0.0, a.sin() * radius);
                let d = want - p.feet;
                let yaw = d.x.atan2(d.z);
                Input {
                    forward: if d.len() > 0.2 { 1.0 } else { 0.0 },
                    yaw,
                    ..Default::default()
                }
            }
            Bot::Swarmer => {
                let phase = (self.time * 0.35).sin();
                let disperse = (self.time % 8.0) > 4.0;
                let x = self.spawns[i].x + phase * 6.0;
                let d = x - p.core.pos.x;
                Input {
                    strafe: -d.signum() * d.abs().min(1.0),
                    yaw: 0.0,
                    disperse,
                    ..Default::default()
                }
            }
        }
    }

    pub fn eye(&self) -> Vec3 {
        self.players[0].feet + v3(0.0, EYE_HEIGHT, 0.0)
    }

    pub fn look_dir(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        v3(sy * cp, sp, cy * cp)
    }

    /// Advance one fixed tick. `local` carries the human's movement keys; yaw comes from the mouse.
    pub fn tick(&mut self, local: Input) {
        self.time += DT;
        self.cooldown = (self.cooldown - DT).max(0.0);
        self.hit_flash = (self.hit_flash - DT).max(0.0);
        for t in &mut self.tracers {
            t.age += DT;
        }
        self.tracers
            .retain(|t| t.age * TRACER_SPEED < (t.b - t.a).len() + TRACER_LEN);
        let mut inputs: Vec<Input> = (0..self.players.len()).map(|i| self.bot_input(i)).collect();
        inputs[0] = Input {
            yaw: self.yaw,
            pitch: self.pitch,
            ..local
        };
        for (i, inp) in inputs.iter().enumerate() {
            if self.players[i].alive {
                self.players[i].tick(inp, &self.world, DT);
            } else {
                self.players[i].step_debris(&self.world, DT);
                self.dead_for[i] += DT;
                if self.dead_for[i] > 3.0 {
                    self.next_seed += 1;
                    self.players[i] = Player::spawn(self.next_seed, self.spawns[i]);
                    self.dead_for[i] = 0.0;
                }
            }
        }
        self.follow_camera();
        if self.fire_held && WEAPONS[self.weapon].auto {
            self.fire();
        }
    }

    /// Trigger pressed or released. Semi-automatic weapons fire once per press; automatic ones
    /// keep firing from `tick` while held.
    pub fn trigger(&mut self, down: bool) {
        if down && !self.fire_held {
            self.fire();
        }
        self.fire_held = down;
    }

    /// How dispersed the view is, eased so the camera starts and stops gently. Exactly 0 when
    /// coherent, so the crosshair then matches `look_dir`.
    fn cam_swarm(&self) -> f32 {
        let b = self.players[0].blend;
        b * b * (3.0 - 2.0 * b)
    }

    /// Where the camera wants to be: over the shoulder when coherent, pulled back behind the core
    /// when dispersed, never on the far side of a wall from the core.
    fn cam_goal(&self) -> Vec3 {
        let me = &self.players[0];
        let w = self.cam_swarm();
        let fwd = self.look_dir();
        let right = v3(-1.0, 0.0, 0.0).rot_y(self.yaw);
        let body = me.feet + v3(0.0, 1.5, 0.0) + right * CAM_SHOULDER;
        let swarm = me.core.pos + v3(0.0, 0.3, 0.0);
        // The core is always in open space, so reach the pivot from it and stop at walls.
        let pivot = self.clear_towards(me.core.pos, body + (swarm - body) * w, 0.05);
        let back = (-fwd + v3(0.0, CAM_UP / CAM_BACK, 0.0)).normalized();
        let dist = CAM_BACK + CAM_BACK_SWARM * w;
        self.clear_towards(pivot, pivot + back * dist, CAM_RADIUS + 0.1)
    }

    /// Walk from `a` towards `b`, stopping `margin` short of the first wall.
    fn clear_towards(&self, a: Vec3, b: Vec3, margin: f32) -> Vec3 {
        let d = b - a;
        let len = d.len();
        if len < 1e-6 {
            return a;
        }
        let dir = d * (1.0 / len);
        match self.world.ray_cast(a, dir, len + margin) {
            Some(t) => a + dir * (t - margin).max(0.0),
            None => b,
        }
    }

    fn sees(&self, a: Vec3, b: Vec3) -> bool {
        let d = b - a;
        let len = d.len();
        len < 1e-6 || self.world.ray_cast(a, d * (1.0 / len), len).is_none()
    }

    /// Move the camera sphere one tick. It chases its goal and collides with the level; when a
    /// wall hides the goal it retraces the core's trail, so it follows the swarm through gaps
    /// instead of cutting through walls.
    fn follow_camera(&mut self) {
        let me = &self.players[0];
        let core = me.core.pos;
        let goal = self.cam_goal();
        if me.seed != self.cam_seed {
            self.cam_seed = me.seed;
            self.cam_pos = goal;
            self.trail.clear();
        }
        if self
            .trail
            .last()
            .is_none_or(|&c| (c - core).len() > TRAIL_STEP)
        {
            self.trail.push(core);
            if self.trail.len() > TRAIL_LEN {
                self.trail.remove(0);
            }
        }
        let lag = CAM_LAG_BODY + (CAM_LAG_SWARM - CAM_LAG_BODY) * self.cam_swarm();
        let (target, min_speed) = if self.sees(self.cam_pos, goal) {
            (goal, 0.0)
        } else if let Some(&c) = self
            .trail
            .iter()
            .rev()
            .find(|&&c| self.sees(self.cam_pos, c))
        {
            (c, CAM_TRAIL_SPEED)
        } else {
            // Lost (e.g. teleported or spun around a pillar): jump rather than clip.
            self.cam_pos = goal;
            return;
        };
        let d = target - self.cam_pos;
        let dist = d.len();
        let step = (dist * DT / lag).max(min_speed * DT).min(dist);
        if step <= 0.0 {
            return;
        }
        let dir = d * (1.0 / dist);
        // Sub-step so the sphere cannot tunnel through a thin wall.
        let n = (step / (CAM_RADIUS * 0.5)).ceil().max(1.0) as usize;
        for _ in 0..n {
            self.cam_pos += dir * (step / n as f32);
            self.world.push_sphere(&mut self.cam_pos, CAM_RADIUS);
        }
    }

    /// Fire the current weapon along the crosshair. The shot leaves the shoulder and converges on
    /// whatever the camera ray points at, so it lands under the crosshair in third person.
    /// Multi-pellet guns scatter each pellet inside a cone around that line.
    pub fn fire(&mut self) -> Vec<HitEvent> {
        if self.cooldown > 0.0 || !self.players[0].can_act() {
            return Vec::new();
        }
        let gun = &WEAPONS[self.weapon];
        let weapon = gun.weapon;
        self.cooldown = gun.cooldown;
        let cam = self.camera(70f32.to_radians());
        let cam_eye = v3(cam.eye[0], cam.eye[1], cam.eye[2]);
        let dir = self.look_dir();
        let dist = aim_distance(&self.players, 0, cam_eye, dir, weapon.range, &self.world);
        let aim = cam_eye + dir * dist;
        let muzzle = self.muzzle();
        let shot = (aim - muzzle).normalized();
        let helper = if shot.y.abs() > 0.9 {
            v3(1.0, 0.0, 0.0)
        } else {
            v3(0.0, 1.0, 0.0)
        };
        let u = shot.cross(helper).normalized();
        let w = shot.cross(u);
        let tan_half = (gun.spread_deg.to_radians() * 0.5).tan();

        let mut events = Vec::new();
        for _ in 0..gun.pellets {
            // Uniform over the cone's cross-section disc.
            let rr = tan_half * self.rng.f32().sqrt();
            let a = self.rng.range(0.0, std::f32::consts::TAU);
            let d = (shot + (u * a.cos() + w * a.sin()) * rr).normalized();
            let hits = hitscan(&mut self.players, 0, muzzle, d, weapon, &self.world);
            // The tracer runs to the last thing it hit, or the level / max range.
            let reach = if hits.len() == weapon.pierce as usize + 1 {
                hits.last().map_or(weapon.range, |e| e.t)
            } else {
                self.world
                    .ray_cast(muzzle, d, weapon.range)
                    .unwrap_or(weapon.range)
            };
            self.tracers.push(Tracer {
                a: muzzle,
                b: muzzle + d * reach,
                age: 0.0,
                color: gun.color,
            });
            events.extend(hits);
        }
        if !events.is_empty() {
            self.hit_flash = if events.iter().any(|e| e.elem == CORE_ID) {
                0.35
            } else {
                0.15
            };
        }
        self.last_hits = events.clone();
        events
    }

    /// Where shots start: the right shoulder, matching the over-the-shoulder camera.
    fn muzzle(&self) -> Vec3 {
        let me = &self.players[0];
        me.feet + v3(0.0, 1.4, 0.0) + v3(-1.0, 0.0, 0.0).rot_y(self.yaw) * 0.2
    }

    pub fn cycle_weapon(&mut self, i: usize) {
        self.weapon = i % WEAPONS.len();
    }

    pub fn status(&self) -> String {
        let me = &self.players[0];
        let mode = if me.blend >= 1.0 {
            "DISPERSED"
        } else if me.blend > 0.0 {
            "morphing"
        } else {
            "coherent"
        };
        let armour = me.alive_elements();
        let last = self.last_hits.last().map_or(String::new(), |h| {
            if h.elem == CORE_ID {
                format!(" | CORE HIT on player {}", h.target)
            } else {
                format!(" | hit player {}", h.target)
            }
        });
        format!(
            "swarmpf | {} | {}/{} spheres | {}{}",
            mode,
            armour,
            me.elems.len(),
            WEAPONS[self.weapon].name,
            last
        )
    }

    pub fn camera(&self, fov_y: f32) -> Camera {
        if self.third_person {
            // Coherent: look where the mouse points (the crosshair aims). Dispersed: turn towards
            // the middle of the swarm, so the view stays on it when a wall pushes the camera aside.
            let me = &self.players[0];
            let alive = me.elems.iter().filter(|e| e.alive());
            let (sum, n) = alive.fold((Vec3::ZERO, 0.0), |(s, n), e| (s + e.pos, n + 1.0));
            let center = if n > 0.0 {
                sum * (1.0 / n)
            } else {
                me.core.pos
            };
            let w = self.cam_swarm();
            let fwd = self.look_dir();
            let fwd = (fwd + ((center - self.cam_pos).normalized() - fwd) * w).normalized();
            let (yaw, pitch) = if w > 0.0 {
                (fwd.x.atan2(fwd.z), fwd.y.clamp(-1.0, 1.0).asin())
            } else {
                (self.yaw, self.pitch)
            };
            let e = self.cam_pos;
            Camera::from_angles([e.x, e.y, e.z], yaw, pitch, fov_y)
        } else {
            let e = self.eye();
            Camera::from_angles([e.x, e.y, e.z], self.yaw, self.pitch, fov_y)
        }
    }

    pub fn scene(&self) -> Scene {
        let mut spheres = Vec::new();
        let mut groups = Vec::new();
        for (pi, p) in self.players.iter().enumerate() {
            if pi == 0 && !self.third_person {
                continue; // do not draw the inside of your own body
            }
            let start = spheres.len() as u32;
            let base = PALETTE[pi % PALETTE.len()];
            for e in p.elems.iter().filter(|e| e.alive()) {
                // Every sphere is a light. A hit turns it red until it regrows; a dead player's
                // debris goes dark.
                let (emit, albedo) = if p.alive {
                    let c = if e.hp < e.max_hp { HIT_COLOR } else { base };
                    ([c[0] * EMIT, c[1] * EMIT, c[2] * EMIT], 0.35)
                } else {
                    ([0.02; 3], 0.4)
                };
                spheres.push(SphereInst {
                    pos_r: [e.pos.x, e.pos.y, e.pos.z, e.r],
                    emit: [emit[0], emit[1], emit[2], albedo],
                });
            }
            if p.alive {
                let c = p.core.pos;
                spheres.push(SphereInst {
                    pos_r: [c.x, c.y, c.z, p.core.r],
                    emit: [CORE_EMIT, CORE_EMIT * 0.9, CORE_EMIT * 0.55, 0.3],
                });
            }
            groups.push((start, spheres.len() as u32 - start));
        }
        let mut boxes: Vec<BoxInst> = self
            .world
            .boxes
            .iter()
            .enumerate()
            .map(|(i, b)| BoxInst {
                min: [b.min.x, b.min.y, b.min.z, 0.0],
                max: [b.max.x, b.max.y, b.max.z, 0.0],
                albedo: box_color(i),
            })
            .collect();
        // A floor slab so light has something to bounce off (the sim's floor is the plane y = 0;
        // the ceiling is a level box).
        boxes.push(BoxInst {
            min: [-20.5, -0.5, -20.5, 0.0],
            max: [20.5, 0.0, 20.5, 0.0],
            albedo: [0.55, 0.55, 0.58, 0.0],
        });
        let tracers = self
            .tracers
            .iter()
            .map(|t| {
                let dir = (t.b - t.a).normalized();
                let len = (t.b - t.a).len();
                let head = (t.age * TRACER_SPEED).min(len);
                let tail = (head - TRACER_LEN).max(0.0);
                let (a, b) = (t.a + dir * tail, t.a + dir * head);
                TracerInst {
                    a: [a.x, a.y, a.z, 0.012],
                    b: [b.x, b.y, b.z, 2.5],
                    color: [t.color[0], t.color[1], t.color[2], 0.0],
                }
            })
            .collect();
        Scene {
            camera: self.camera(70f32.to_radians()),
            spheres,
            groups,
            boxes,
            tracers,
            hit_flash: self.hit_flash,
            exposure: 1.0,
            ui: Vec::new(),
        }
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

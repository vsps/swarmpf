//! Local single-process game: the shared sim plus a test level and a few bots.
//! Networking will replace the bots and the direct `tick` calls later.

use crate::camera::Camera;
use crate::renderer::{BoxInst, Scene, SphereInst, TracerInst};
use sim::hit::{aim_distance, hitscan, HitEvent, Weapon};
use sim::math::{v3, Vec3};
use sim::player::{Input, Player, CORE_ID};
use sim::world::{Aabb, World};
use sim::DT;

pub const EYE_HEIGHT: f32 = 1.62;
/// (name, weapon, seconds between shots, tracer colour)
pub const WEAPONS: [(&str, Weapon, f32, [f32; 3]); 3] = [
    ("rifle", Weapon::RIFLE, 0.11, [1.0, 0.85, 0.5]),
    ("pellet", Weapon::SHOTGUN_PELLET, 0.06, [1.0, 0.5, 0.2]),
    ("railgun", Weapon::RAILGUN, 0.9, [0.4, 0.9, 1.0]),
];

const TRACER_SPEED: f32 = 320.0;
const TRACER_LEN: f32 = 7.0;
/// Camera distance behind the pivot and its offset over the right shoulder.
const CAM_BACK: f32 = 3.2;
const CAM_SHOULDER: f32 = 0.55;
const CAM_UP: f32 = 0.35;
/// Emission radiance of a healthy sphere; the core is far brighter.
const EMIT: f32 = 7.0;
const CORE_EMIT: f32 = 60.0;

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
}

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
        ],
    }
}

fn box_color(i: usize) -> [f32; 4] {
    if i < 4 {
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
        if self.fire_held {
            self.fire();
        }
    }

    /// Fire the current weapon along the crosshair. The shot leaves the shoulder and converges on
    /// whatever the camera ray points at, so it lands under the crosshair in third person.
    pub fn fire(&mut self) -> Vec<HitEvent> {
        if self.cooldown > 0.0 || !self.players[0].can_act() {
            return Vec::new();
        }
        let (_, weapon, cooldown, color) = WEAPONS[self.weapon];
        self.cooldown = cooldown;
        let cam = self.camera(70f32.to_radians());
        let cam_eye = v3(cam.eye[0], cam.eye[1], cam.eye[2]);
        let dir = self.look_dir();
        let dist = aim_distance(&self.players, 0, cam_eye, dir, weapon.range, &self.world);
        let aim = cam_eye + dir * dist;
        let muzzle = self.muzzle();
        let shot = (aim - muzzle).normalized();
        let events = hitscan(&mut self.players, 0, muzzle, shot, weapon, &self.world);

        // The tracer runs to the last thing it hit, or the level / max range.
        let reach = if events.len() == weapon.pierce as usize + 1 {
            events.last().map_or(weapon.range, |e| e.t)
        } else {
            self.world
                .ray_cast(muzzle, shot, weapon.range)
                .unwrap_or(weapon.range)
        };
        self.tracers.push(Tracer {
            a: muzzle,
            b: muzzle + shot * reach,
            age: 0.0,
            color,
        });
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
            WEAPONS[self.weapon].0,
            last
        )
    }

    pub fn camera(&self, fov_y: f32) -> Camera {
        let fwd = self.look_dir();
        let right = v3(-1.0, 0.0, 0.0).rot_y(self.yaw);
        if self.third_person {
            // Over-the-shoulder: pivot near the head, pull back along the view, stop at walls.
            let pivot = self.players[0].feet + v3(0.0, 1.5, 0.0) + right * CAM_SHOULDER;
            let back = -fwd + v3(0.0, CAM_UP / CAM_BACK, 0.0);
            let back = back.normalized();
            let margin = 0.25;
            let room = self
                .world
                .ray_cast(pivot, back, CAM_BACK + margin)
                .unwrap_or(CAM_BACK + margin);
            let e = pivot + back * (room - margin).max(0.15);
            Camera::from_angles([e.x, e.y, e.z], self.yaw, self.pitch, fov_y)
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
                let health = e.hp as f32 / e.max_hp as f32;
                // Every sphere is a light. Damage dims and reddens it; a dead player's debris goes dark.
                let (emit, albedo) = if p.alive {
                    let k = EMIT * (0.25 + 0.75 * health);
                    (
                        [base[0] * k + (1.0 - health) * 1.5, base[1] * k, base[2] * k],
                        0.35,
                    )
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
        // Floor and ceiling so light has something to bounce off.
        boxes.push(BoxInst {
            min: [-20.5, -0.5, -20.5, 0.0],
            max: [20.5, 0.0, 20.5, 0.0],
            albedo: [0.55, 0.55, 0.58, 0.0],
        });
        boxes.push(BoxInst {
            min: [-20.5, 4.0, -20.5, 0.0],
            max: [20.5, 4.5, 20.5, 0.0],
            albedo: [0.5, 0.5, 0.52, 0.0],
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
        }
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

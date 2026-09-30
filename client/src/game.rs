//! Local single-process game: the shared sim plus a test level and a few bots.
//! Networking will replace the bots and the direct `tick` calls later.

use crate::camera::Camera;
use crate::renderer::{BoxInst, Scene, SphereInst};
use sim::hit::{hitscan, HitEvent, Weapon};
use sim::math::{v3, Vec3};
use sim::player::{Input, Player, CORE_ID};
use sim::world::{Aabb, World};
use sim::DT;

pub const EYE_HEIGHT: f32 = 1.62;
pub const WEAPONS: [(&str, Weapon); 3] = [
    ("rifle", Weapon::RIFLE),
    ("pellet", Weapon::SHOTGUN_PELLET),
    ("railgun", Weapon::RAILGUN),
];

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
        [0.42, 0.44, 0.50, 0.0]
    } else if i < 7 {
        [0.55, 0.50, 0.46, 0.0]
    } else {
        [0.36, 0.42, 0.52, 0.0]
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
            third_person: false,
            time: 0.0,
            bots,
            spawns,
            dead_for: vec![0.0; n],
            next_seed: 1000,
            last_hits: Vec::new(),
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
    }

    pub fn fire(&mut self) -> Vec<HitEvent> {
        let dir = self.look_dir();
        let eye = self.eye();
        let w = WEAPONS[self.weapon].1;
        let ev = hitscan(&mut self.players, 0, eye, dir, w, &self.world);
        self.last_hits = ev.clone();
        ev
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

    pub fn camera(&self, aspect_fov: f32) -> Camera {
        let eye = self.eye();
        if self.third_person {
            let back = -self.look_dir() * 4.0;
            let mut e = eye + back + v3(0.0, 0.6, 0.0);
            e.y = e.y.max(0.3);
            Camera::from_angles([e.x, e.y, e.z], self.yaw, self.pitch, aspect_fov)
        } else {
            Camera::from_angles([eye.x, eye.y, eye.z], self.yaw, self.pitch, aspect_fov)
        }
    }

    pub fn scene(&self) -> Scene {
        let mut spheres = Vec::new();
        for (pi, p) in self.players.iter().enumerate() {
            if pi == 0 && !self.third_person {
                continue; // do not draw the inside of your own body
            }
            let base = PALETTE[pi % PALETTE.len()];
            for e in p.elems.iter().filter(|e| e.alive()) {
                let health = e.hp as f32 / e.max_hp as f32;
                // Damaged spheres darken towards red-grey; dead players go grey.
                let k = if p.alive { 0.35 + 0.65 * health } else { 0.4 };
                let (r, g, b) = if p.alive {
                    (
                        base[0] * k + (1.0 - health) * 0.35,
                        base[1] * k,
                        base[2] * k,
                    )
                } else {
                    (0.4, 0.4, 0.4)
                };
                spheres.push(SphereInst {
                    pos_r: [e.pos.x, e.pos.y, e.pos.z, e.r],
                    color: [r, g, b, 0.0],
                });
            }
            if p.alive {
                let c = p.core.pos;
                spheres.push(SphereInst {
                    pos_r: [c.x, c.y, c.z, p.core.r],
                    color: [1.0, 0.95, 0.6, 1.0],
                });
            }
        }
        let boxes = self
            .world
            .boxes
            .iter()
            .enumerate()
            .map(|(i, b)| BoxInst {
                min: [b.min.x, b.min.y, b.min.z, 0.0],
                max: [b.max.x, b.max.y, b.max.z, 0.0],
                color: box_color(i),
            })
            .chain(std::iter::once(BoxInst {
                min: [-20.5, -0.5, -20.5, 0.0],
                max: [20.5, 0.0, 20.5, 0.0],
                color: [0.20, 0.21, 0.24, 0.0],
            }))
            .collect();
        Scene {
            camera: self.camera(70f32.to_radians()),
            spheres,
            boxes,
            clear: [0.03, 0.035, 0.05],
        }
    }
}

impl Default for Game {
    fn default() -> Self {
        Self::new()
    }
}

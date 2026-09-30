//! Server-side hitscan with penetration.

use crate::math::{ray_sphere, Vec3};
use crate::player::{Player, CORE_HIT_SCALE, CORE_ID};
use crate::world::World;

#[derive(Clone, Copy, Debug)]
pub struct Weapon {
    pub damage: u8,
    /// Extra spheres the shot passes through after the first. Each pierced sphere halves damage (min 1).
    pub pierce: u8,
    pub range: f32,
}

impl Weapon {
    pub const RIFLE: Weapon = Weapon {
        damage: 2,
        pierce: 0,
        range: 120.0,
    };
    pub const SHOTGUN_PELLET: Weapon = Weapon {
        damage: 1,
        pierce: 0,
        range: 30.0,
    };
    pub const RAILGUN: Weapon = Weapon {
        damage: 6,
        pierce: 3,
        range: 300.0,
    };
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HitEvent {
    pub target: usize,
    /// Element index, or `CORE_ID` for the core.
    pub elem: u8,
    pub damage: u8,
    /// The sphere was destroyed by this hit (for the core: the player died).
    pub destroyed: bool,
    pub t: f32,
}

/// Fire a ray from `shooter`'s eye. `dir` must be unit length.
pub fn hitscan(
    players: &mut [Player],
    shooter: usize,
    origin: Vec3,
    dir: Vec3,
    weapon: Weapon,
    world: &World,
) -> Vec<HitEvent> {
    if !players[shooter].can_act() {
        return Vec::new();
    }
    let max_t = world
        .ray_cast(origin, dir, weapon.range)
        .unwrap_or(weapon.range);

    // Every candidate along the ray, nearest first.
    let mut cands: Vec<(f32, usize, u8)> = Vec::new();
    for (pi, p) in players.iter().enumerate() {
        if pi == shooter || !p.alive {
            continue;
        }
        if let Some(t) = ray_sphere(origin, dir, p.core.pos, p.core.r * CORE_HIT_SCALE) {
            cands.push((t, pi, CORE_ID));
        }
        for (ei, e) in p.elems.iter().enumerate() {
            if !e.alive() {
                continue;
            }
            if let Some(t) = ray_sphere(origin, dir, e.pos, e.r) {
                cands.push((t, pi, ei as u8));
            }
        }
    }
    cands.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut events = Vec::new();
    let mut dmg = weapon.damage;
    let mut pierce_left = weapon.pierce;
    for (t, pi, ei) in cands {
        if t > max_t {
            break;
        }
        let p = &mut players[pi];
        if !p.alive {
            continue;
        }
        let destroyed = if ei == CORE_ID {
            p.kill();
            true
        } else {
            let e = &mut p.elems[ei as usize];
            e.hp = e.hp.saturating_sub(dmg);
            e.hp == 0
        };
        events.push(HitEvent {
            target: pi,
            elem: ei,
            damage: dmg,
            destroyed,
            t,
        });
        if pierce_left == 0 {
            break;
        }
        pierce_left -= 1;
        dmg = (dmg / 2).max(1);
    }
    events
}

/// Distance along a unit ray to the first thing it would hit (level, or any other player's
/// sphere or core), or `max`. Used to aim shots at what a third-person crosshair points at.
pub fn aim_distance(
    players: &[Player],
    shooter: usize,
    origin: Vec3,
    dir: Vec3,
    max: f32,
    world: &World,
) -> f32 {
    let mut best = world.ray_cast(origin, dir, max).unwrap_or(max);
    for (pi, p) in players.iter().enumerate() {
        if pi == shooter || !p.alive {
            continue;
        }
        if let Some(t) = ray_sphere(origin, dir, p.core.pos, p.core.r * CORE_HIT_SCALE) {
            best = best.min(t);
        }
        for e in p.elems.iter().filter(|e| e.alive()) {
            if let Some(t) = ray_sphere(origin, dir, e.pos, e.r) {
                best = best.min(t);
            }
        }
    }
    best
}

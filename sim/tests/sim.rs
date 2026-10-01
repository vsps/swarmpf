use sim::hit::{hitscan, Weapon};
use sim::math::{ray_sphere, v3, Rng, Vec3};
use sim::player::*;
use sim::skeleton;
use sim::world::{Aabb, World};
use sim::DT;

fn settle(p: &mut Player, world: &World, secs: f32, input: Input) {
    for _ in 0..(secs / DT) as usize {
        p.tick(&input, world, DT);
    }
}

/// Standing still, spheres drift up to `WANDER` per axis around their slots.
const REST_TOL: f32 = WANDER * 1.8 + 0.02;

/// Largest distance from an element to its slot; slots below the floor are clamped
/// to where a sphere resting on the floor can actually sit.
fn worst_slot_error(p: &Player) -> f32 {
    p.elems
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let mut t = p.slot_target(i);
            t.y = t.y.max(e.r);
            (e.pos - t).len()
        })
        .fold(0.0f32, f32::max)
}

fn idle(yaw: f32) -> Input {
    Input {
        yaw,
        ..Default::default()
    }
}

fn open_world() -> World {
    World::default()
}

/// A wall across Z = 0 with a doorway of `gap` width centred on x = 0.
fn wall_with_gap(gap: f32) -> World {
    let h = gap * 0.5;
    World {
        boxes: vec![
            Aabb::new(v3(-20.0, 0.0, -0.25), v3(-h, 4.0, 0.25)),
            Aabb::new(v3(h, 0.0, -0.25), v3(20.0, 4.0, 0.25)),
        ],
    }
}

#[test]
fn ray_sphere_cases() {
    let c = v3(0.0, 0.0, 5.0);
    assert!((ray_sphere(Vec3::ZERO, v3(0.0, 0.0, 1.0), c, 1.0).unwrap() - 4.0).abs() < 1e-5);
    assert!(ray_sphere(Vec3::ZERO, v3(0.0, 1.0, 0.0), c, 1.0).is_none());
    assert!(ray_sphere(Vec3::ZERO, v3(0.0, 0.0, -1.0), c, 1.0).is_none());
    assert_eq!(ray_sphere(c, v3(0.0, 0.0, 1.0), c, 1.0), Some(0.0));
}

#[test]
fn radii_hp_and_seed_determinism() {
    let a = Player::spawn(7, Vec3::ZERO);
    let b = Player::spawn(7, Vec3::ZERO);
    let c = Player::spawn(8, Vec3::ZERO);
    assert_eq!(a.elems.len(), skeleton::NUM_SLOTS);
    for (x, y) in a.elems.iter().zip(&b.elems) {
        assert_eq!(x.r, y.r);
        assert_eq!(x.hp, y.hp);
    }
    assert!(a.elems.iter().zip(&c.elems).any(|(x, y)| x.r != y.r));
    for e in &a.elems {
        assert!(e.r >= R_MIN && e.r <= R_MAX);
        assert_eq!(e.hp, ELEM_HP);
    }
}

#[test]
fn walk_cycle_alternates_and_keeps_feet_near_ground() {
    let mut min_y = f32::MAX;
    let mut l_fwd = f32::MIN;
    let mut l_back = f32::MAX;
    let mut r_at_l_fwd = 0.0;
    for k in 0..360 {
        let phase = k as f32 / 360.0 * std::f32::consts::TAU;
        let pose = skeleton::pose(phase, 1.0);
        let l = pose.pos[skeleton::ANKLE_L];
        let r = pose.pos[skeleton::ANKLE_R];
        min_y = min_y.min(l.y).min(r.y);
        if l.z > l_fwd {
            l_fwd = l.z;
            r_at_l_fwd = r.z;
        }
        l_back = l_back.min(l.z);
    }
    assert!(l_fwd > 0.2 && l_back < -0.2, "stride {l_back}..{l_fwd}");
    assert!(r_at_l_fwd < 0.0, "right foot is back when left is forward");
    // Foot never sinks meaningfully below the floor (bob + knee fold allowed).
    assert!(min_y > -0.12, "foot sank to {min_y}");
    // Idle pose is a symmetric standing pose.
    let idle = skeleton::pose(1.0, 0.0);
    assert!((idle.pos[skeleton::ANKLE_L].y - 0.0).abs() < 1e-4);
    assert!((idle.pos[skeleton::HEAD].y - 1.7).abs() < 1e-3);
}

#[test]
fn coherent_body_settles_on_its_slots() {
    let world = open_world();
    let mut p = Player::spawn(1, v3(0.0, 0.0, 0.0));
    settle(&mut p, &world, 1.0, idle(0.0));
    let worst = worst_slot_error(&p);
    assert!(worst < REST_TOL, "worst slot error {worst}");
    // Also while walking: the swarm trails the skeleton slightly, but does not fall apart.
    let walk = Input {
        forward: 1.0,
        ..idle(0.0)
    };
    settle(&mut p, &world, 2.0, walk);
    let worst = worst_slot_error(&p);
    assert!(worst < 0.75, "walking slot error {worst}");
    assert!(p.feet.z > 5.0, "walked to z = {}", p.feet.z);
}

#[test]
fn penetration_passes_through_multiple_spheres() {
    let world = open_world();
    let mut players = vec![
        Player::spawn(1, v3(0.0, 0.0, -10.0)),
        Player::spawn(2, v3(0.0, 0.0, 10.0)),
    ];
    // Hand-place three big spheres in a line in front of player 1, tucked out of the way of the rest.
    for e in players[1].elems.iter_mut() {
        e.pos = v3(50.0, 50.0, 50.0);
    }
    players[1].core.pos = v3(60.0, 50.0, 50.0);
    for (k, z) in [10.0, 11.0, 12.0].into_iter().enumerate() {
        let e = &mut players[1].elems[k];
        e.pos = v3(0.0, 1.0, z);
        e.r = 0.3;
        e.hp = 20;
        e.max_hp = 20;
        e.vel = Vec3::ZERO;
    }
    let origin = v3(0.0, 1.0, 0.0);
    let dir = v3(0.0, 0.0, 1.0);

    let ev = hitscan(&mut players, 0, origin, dir, Weapon::RIFLE, &world);
    assert_eq!(ev.len(), 1);
    assert_eq!(ev[0].elem, 0);
    assert_eq!(players[1].elems[0].hp, 19);
    // The hit knocks the sphere along the shot.
    assert!(players[1].elems[0].vel.z > 0.0);

    let ev = hitscan(&mut players, 0, origin, dir, Weapon::RAILGUN, &world);
    assert_eq!(ev.len(), 3); // only three spheres exist on the line
    assert_eq!(players[1].elems[0].hp, 17);
    assert_eq!(players[1].elems[1].hp, 18);
    assert_eq!(players[1].elems[2].hp, 18);
}

#[test]
fn railgun_one_shots_every_sphere_through_a_body() {
    let world = open_world();
    let mut players = vec![
        Player::spawn(1, v3(0.0, 0.0, 0.0)),
        Player::spawn(2, v3(0.0, 0.0, 10.0)),
    ];
    settle(&mut players[1], &world, 0.5, idle(0.0));
    // Through a thigh sphere, with no core in the way: straight through every sphere.
    let thigh = players[1]
        .elems
        .iter()
        .find(|e| (0.6..0.8).contains(&e.pos.y))
        .unwrap()
        .pos;
    let origin = v3(thigh.x, thigh.y, 0.5);
    let dir = v3(0.0, 0.0, 1.0);
    let ev = hitscan(&mut players, 0, origin, dir, Weapon::RAILGUN, &world);
    assert!(!ev.is_empty());
    assert!(ev.iter().all(|e| e.destroyed));
    // Nothing is left standing on the line: it went clean through.
    let blocked = players[1]
        .elems
        .iter()
        .filter(|e| e.alive())
        .any(|e| ray_sphere(origin, dir, e.pos, e.r).is_some());
    assert!(!blocked);
}

#[test]
fn spheres_take_two_hits_and_regrow() {
    let world = open_world();
    let mut players = vec![
        Player::spawn(1, v3(0.0, 0.0, 0.0)),
        Player::spawn(2, v3(0.0, 0.0, 10.0)),
    ];
    settle(&mut players[1], &world, 0.5, idle(0.0));
    let origin = v3(0.0, 1.4, 0.5);
    let dir = (players[1].core.pos - origin).normalized();
    let first = hitscan(&mut players, 0, origin, dir, Weapon::RIFLE, &world)[0];
    assert!(!first.destroyed);
    let e = first.elem as usize;
    // Put the (recoiled) sphere back on the line for the second shot.
    players[1].elems[e].pos = origin + dir * first.t;
    let second = hitscan(&mut players, 0, origin, dir, Weapon::RIFLE, &world)[0];
    assert_eq!(second.elem as usize, e);
    assert!(second.destroyed);

    // Wound three more: one sphere back per REGEN_TIME, destroyed first.
    for k in 0..3 {
        let i = (e + 1 + k) % players[1].elems.len();
        players[1].elems[i].hp = 1;
    }
    let missing = |p: &Player| p.elems.iter().filter(|e| e.hp < e.max_hp).count();
    assert_eq!(missing(&players[1]), 4);
    settle(&mut players[1], &world, REGEN_TIME + 0.1, idle(0.0));
    assert_eq!(missing(&players[1]), 3);
    assert!(players[1].elems[e].alive());
    settle(&mut players[1], &world, 3.0 * REGEN_TIME, idle(0.0));
    assert_eq!(missing(&players[1]), 0);
}

#[test]
fn walls_block_hitscan() {
    let world = World {
        boxes: vec![Aabb::new(v3(-5.0, 0.0, 4.0), v3(5.0, 5.0, 5.0))],
    };
    let mut players = vec![
        Player::spawn(1, v3(0.0, 0.0, 0.0)),
        Player::spawn(2, v3(0.0, 0.0, 8.0)),
    ];
    let origin = v3(0.0, 1.4, 0.5);
    let target = players[1].core.pos;
    let dir = (target - origin).normalized();
    assert!(hitscan(&mut players, 0, origin, dir, Weapon::RAILGUN, &world).is_empty());
}

#[test]
fn dispersed_players_cannot_fire() {
    let world = open_world();
    let mut players = vec![
        Player::spawn(1, v3(0.0, 0.0, 0.0)),
        Player::spawn(2, v3(0.0, 0.0, 8.0)),
    ];
    settle(
        &mut players[0],
        &world,
        0.1,
        Input {
            disperse: true,
            ..idle(0.0)
        },
    );
    let origin = v3(0.0, 1.4, 0.5);
    let dir = (players[1].core.pos - origin).normalized();
    assert!(hitscan(&mut players, 0, origin, dir, Weapon::RIFLE, &world).is_empty());
}

/// Fraction of rays aimed at the core whose first hit is armour rather than the core.
fn core_shielding(p: &Player, rng: &mut Rng, n: usize) -> f32 {
    let mut shielded = 0;
    for _ in 0..n {
        let d = rng.unit_vec();
        let origin = p.core.pos + d * 6.0;
        let dir = -d;
        let first_elem = p
            .elems
            .iter()
            .filter(|e| e.alive())
            .filter_map(|e| ray_sphere(origin, dir, e.pos, e.r))
            .fold(f32::INFINITY, f32::min);
        let core = ray_sphere(origin, dir, p.core.pos, p.core.r * CORE_HIT_SCALE).unwrap();
        if first_elem < core {
            shielded += 1;
        }
    }
    shielded as f32 / n as f32
}

#[test]
fn core_is_buried_in_the_body_and_exposed_by_damage() {
    let world = open_world();
    let mut p = Player::spawn(3, Vec3::ZERO);
    settle(&mut p, &world, 1.0, idle(0.0));
    let mut rng = Rng::new(99);
    let full = core_shielding(&p, &mut rng, 4000);
    assert!(
        full > 0.6,
        "intact body shields core from only {:.0}% of directions",
        full * 100.0
    );

    // Strip the armour: the core becomes fully exposed.
    for e in p.elems.iter_mut() {
        e.hp = 0;
    }
    assert_eq!(core_shielding(&p, &mut rng, 500), 0.0);
}

#[test]
fn core_is_a_one_shot_kill() {
    let world = open_world();
    let mut players = vec![
        Player::spawn(1, v3(0.0, 0.0, 0.0)),
        Player::spawn(2, v3(0.0, 0.0, 10.0)),
    ];
    for e in players[1].elems.iter_mut() {
        e.hp = 0;
    }
    let origin = v3(0.0, 1.4, 0.5);
    let dir = (players[1].core.pos - origin).normalized();
    let ev = hitscan(&mut players, 0, origin, dir, Weapon::SHOTGUN_PELLET, &world);
    assert_eq!(ev.len(), 1);
    assert_eq!(ev[0].elem, CORE_ID);
    assert!(ev[0].destroyed);
    assert!(!players[1].alive);
}

fn drive_through(gap: f32, disperse: bool) -> Player {
    let world = wall_with_gap(gap);
    let mut p = Player::spawn(5, v3(0.0, 0.0, -4.0));
    settle(
        &mut p,
        &world,
        0.5,
        Input {
            disperse,
            ..idle(0.0)
        },
    );
    settle(
        &mut p,
        &world,
        8.0,
        Input {
            forward: 1.0,
            disperse,
            ..idle(0.0)
        },
    );
    p
}

#[test]
fn narrow_gap_blocks_body_but_not_the_swarm() {
    let coherent = drive_through(0.5, false);
    assert!(
        coherent.feet.z < -0.3,
        "body got through: z = {}",
        coherent.feet.z
    );

    let swarm = drive_through(0.5, true);
    assert!(
        swarm.core.pos.z > 2.0,
        "core stuck at z = {}",
        swarm.core.pos.z
    );
    let through = swarm.elems.iter().filter(|e| e.pos.z > 0.3).count();
    assert!(
        through > swarm.elems.len() / 2,
        "only {through} elements followed"
    );
}

#[test]
fn wide_gap_lets_the_body_through() {
    let coherent = drive_through(1.4, false);
    assert!(
        coherent.feet.z > 2.0,
        "body stuck at z = {}",
        coherent.feet.z
    );
}

#[test]
fn cannot_reform_inside_a_wall() {
    let world = wall_with_gap(0.5);
    let mut p = Player::spawn(5, v3(0.0, 0.0, -2.0));
    // Disperse, then put the core in the doorway (momentum makes timing a stop unreliable).
    settle(
        &mut p,
        &world,
        0.5,
        Input {
            disperse: true,
            ..idle(0.0)
        },
    );
    p.core.pos = v3(0.0, sim::skeleton::CORE_HEIGHT, 0.0);
    p.core_vel = Vec3::ZERO;
    let mut i = Input {
        disperse: true,
        ..idle(0.0)
    };
    settle(&mut p, &world, 0.2, i);
    assert!(p.core.pos.z.abs() < 0.3, "core at z = {}", p.core.pos.z);
    // Release: the doorway is too narrow, so the swarm stays dispersed.
    i.disperse = false;
    settle(&mut p, &world, 1.5, i);
    assert!(
        p.want_disperse && p.blend > 0.99,
        "re-formed inside the doorway: want={} blend={} core={:?} feet={:?}",
        p.want_disperse,
        p.blend,
        p.core.pos,
        p.feet
    );
    // Step out into the open and it can re-form.
    i.forward = 1.0;
    settle(&mut p, &world, 2.0, i);
    i.forward = 0.0;
    settle(&mut p, &world, 1.5, i);
    assert!(!p.want_disperse && p.blend == 0.0 && p.can_act());
}

#[test]
fn dispersal_spreads_the_flock_and_reforming_recovers() {
    let world = open_world();
    let mut p = Player::spawn(2, Vec3::ZERO);
    settle(&mut p, &world, 0.5, idle(0.0));
    let spread = |p: &Player| {
        p.elems
            .iter()
            .map(|e| (e.pos - p.core.pos).len())
            .sum::<f32>()
            / p.elems.len() as f32
    };
    let body = spread(&p);
    settle(
        &mut p,
        &world,
        2.0,
        Input {
            disperse: true,
            ..idle(0.0)
        },
    );
    let swarm = spread(&p);
    assert!(swarm > body * 1.3, "body {body} vs swarm {swarm}");
    assert!(swarm < SWARM_RADIUS * 2.0, "swarm ran off: {swarm}");
    settle(&mut p, &world, 3.0, idle(0.0));
    let worst = worst_slot_error(&p);
    assert!(worst < REST_TOL + 0.03, "re-formed slot error {worst}");
    assert!(p.can_act());
}

#[test]
fn a_hit_sphere_knocks_its_neighbours() {
    let world = open_world();
    let mut p = Player::spawn(4, Vec3::ZERO);
    settle(&mut p, &world, 1.0, idle(0.0));
    // Knock the sphere nearest the core, as one hit would.
    let (i, _) = p
        .elems
        .iter()
        .enumerate()
        .map(|(i, e)| (i, (e.pos - p.core.pos).len()))
        .fold((0, f32::MAX), |b, x| if x.1 < b.1 { x } else { b });
    let start: Vec<Vec3> = p.elems.iter().map(|e| e.pos).collect();
    let dir = (p.core.pos - p.elems[i].pos).normalized();
    p.recoil(i, dir);
    settle(&mut p, &world, 0.1, idle(0.0));
    let moved = (0..p.elems.len())
        .filter(|&j| j != i && (p.elems[j].pos - start[j]).len() > 0.01)
        .count();
    assert!(moved > 0, "no neighbour was knocked");
    // And the body recovers.
    settle(&mut p, &world, 3.0, idle(0.0));
    let worst = worst_slot_error(&p);
    assert!(worst < REST_TOL, "slot error after knock {worst}");
}

#[test]
fn swarm_dives_to_the_ground_flattens_and_reforms() {
    let world = open_world();
    let mut p = Player::spawn(6, Vec3::ZERO);
    settle(&mut p, &world, 0.5, idle(0.0));
    let spread = |p: &Player| {
        let alive = p.elems.iter().filter(|e| e.alive());
        let (mut ys, mut xz) = (0.0f32, 0.0f32);
        let mut n = 0.0;
        for e in alive {
            let d = e.pos - p.core.pos;
            ys += d.y.abs();
            xz += (d.x * d.x + d.z * d.z).sqrt();
            n += 1.0;
        }
        (ys / n, xz / n)
    };
    // Hover at chest height first, then fly forward looking steeply down.
    settle(
        &mut p,
        &world,
        1.5,
        Input {
            disperse: true,
            ..idle(0.0)
        },
    );
    let (y_mid, xz_mid) = spread(&p);
    settle(
        &mut p,
        &world,
        2.0,
        Input {
            forward: 1.0,
            pitch: -1.2,
            disperse: true,
            ..idle(0.0)
        },
    );
    assert!(p.core.pos.y < 0.15, "core at y = {}", p.core.pos.y);
    settle(
        &mut p,
        &world,
        1.5,
        Input {
            disperse: true,
            ..idle(0.0)
        },
    );
    let (y_low, xz_low) = spread(&p);
    // Flatter: vertical spread shrinks relative to horizontal, well beyond what the floor alone
    // does (about 0.54 of the hovering ratio without the ground squash, 0.22 with it).
    assert!(
        y_low / xz_low < 0.35 * (y_mid / xz_mid),
        "low {y_low}/{xz_low} vs hovering {y_mid}/{xz_mid}"
    );
    // Release: the body re-forms standing on the floor.
    settle(&mut p, &world, 2.0, idle(0.0));
    assert!(p.can_act());
    assert!(p.feet.y.abs() < 0.01, "feet at y = {}", p.feet.y);
}

#[test]
fn swarm_cannot_fly_above_the_cap() {
    let world = open_world();
    let mut p = Player::spawn(7, Vec3::ZERO);
    settle(
        &mut p,
        &world,
        3.0,
        Input {
            forward: 1.0,
            pitch: 1.4,
            disperse: true,
            ..idle(0.0)
        },
    );
    assert!(
        p.core.pos.y <= SWARM_MAX_Y + 1e-4,
        "core at y = {}",
        p.core.pos.y
    );
    assert!(p.core.pos.y > SWARM_MAX_Y - 0.05);
    // Re-formed in mid-air, the body drops to the floor.
    settle(&mut p, &world, 2.0, idle(0.0));
    assert!(p.can_act());
    assert!(p.feet.y.abs() < 0.01, "feet at y = {}", p.feet.y);
}

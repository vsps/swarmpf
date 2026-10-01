//! Physics bench: four players walking, dispersing and re-forming.
//! Prints microseconds per `Player::tick` and a checksum of every sphere position, so an
//! optimisation can be shown to leave behaviour bit-identical.
//! Usage: cargo run --release -p sim --example bench

use sim::math::v3;
use sim::player::{Input, Player};
use sim::world::{Aabb, World};
use sim::DT;
use std::time::Instant;

fn main() {
    let b = |x0, y0, z0, x1, y1, z1| Aabb::new(v3(x0, y0, z0), v3(x1, y1, z1));
    // Same layout as the client's test level: walls, a divider with a crack and a door, cover.
    let world = World {
        boxes: vec![
            b(-20.5, 0.0, -20.5, 20.5, 4.0, -20.0),
            b(-20.5, 0.0, 20.0, 20.5, 4.0, 20.5),
            b(-20.5, 0.0, -20.0, -20.0, 4.0, 20.0),
            b(20.0, 0.0, -20.0, 20.5, 4.0, 20.0),
            b(-20.0, 0.0, 5.75, -0.25, 4.0, 6.25),
            b(0.25, 0.0, 5.75, 7.3, 4.0, 6.25),
            b(8.7, 0.0, 5.75, 20.0, 4.0, 6.25),
            b(-8.0, 0.0, -4.0, -6.0, 1.2, -3.0),
            b(4.0, 0.0, -8.0, 5.0, 3.0, -7.0),
            b(-3.0, 0.0, 10.0, -2.0, 3.0, 11.0),
            b(9.0, 0.0, 12.0, 11.0, 1.0, 13.0),
        ],
    };
    let mut players: Vec<Player> = (0..4)
        .map(|i| Player::spawn(100 + i, v3(-6.0 + 4.0 * i as f32, 0.0, -2.0)))
        .collect();

    let reps = 5;
    let mut ticks = 0u64;
    let t0 = Instant::now();
    for rep in 0..reps {
        // 5 s walking in a curve, 5 s dispersed, 2 s re-forming.
        for step in 0..(12.0 / DT) as usize {
            let t = step as f32 * DT;
            for (i, p) in players.iter_mut().enumerate() {
                let input = Input {
                    forward: 1.0,
                    strafe: if i % 2 == 0 { 0.3 } else { -0.3 },
                    yaw: t * 0.4 + i as f32 + rep as f32,
                    pitch: 0.0,
                    disperse: (5.0..10.0).contains(&t),
                };
                p.tick(&input, &world, DT);
                ticks += 1;
            }
        }
    }
    let us = t0.elapsed().as_secs_f64() * 1e6 / ticks as f64;

    let mut sum = 0u64;
    for p in &players {
        for e in &p.elems {
            for c in [e.pos.x, e.pos.y, e.pos.z] {
                sum = sum.wrapping_mul(31).wrapping_add(c.to_bits() as u64);
            }
        }
    }
    println!("{ticks} player ticks, {us:.1} us/tick, checksum {sum:016x}");
}

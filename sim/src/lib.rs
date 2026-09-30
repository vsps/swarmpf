//! Shared game simulation: runs identically on the server and in clients.

pub mod hit;
pub mod math;
pub mod player;
pub mod skeleton;
pub mod world;

pub const TICK_HZ: f32 = 60.0;
pub const DT: f32 = 1.0 / TICK_HZ;

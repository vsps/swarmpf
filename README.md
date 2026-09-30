# swarmpf

Multiplayer FPS where each player is a cloud of spheres.

- **Coherent mode**: the spheres hold a humanoid shape driven by a procedural walk cycle. The body
  moves as a capsule and can fire and use things.
- **Dispersed mode**: the spheres flock around a small core (the boid leader). The player moves faster and
  fits through gaps the body cannot, but cannot act. Re-forming needs room for the body.
- **Health**: every sphere has a random radius and hp proportional to its area. Big spheres form the outer
  shell, so they armour the core.
- **Core**: one tiny sphere buried in the chest (in dispersed mode, the flock leader). Any hit kills the player.
- **Weapons**: hitscan with penetration. Each pierced sphere halves the damage, with a minimum of 1.

## Layout

- `sim/`: shared, dependency-free simulation (movement, flocking, rig, hitscan). It runs the same on the
  server and in clients.
- `client/`: `wgpu` renderer (billboard-impostor spheres with exact silhouettes and depth, instanced boxes),
  a local test level with bots, and two binaries:
  - `swarmpf`: interactive window. WASD move, mouse look, hold Shift to disperse, LMB fire, 1/2/3 weapon,
    V third person, Esc releases the mouse.
  - `shot <prefix>`: headless scripted scenario that writes PNGs (works on software Vulkan, e.g. llvmpipe).

Planned: `server/` (authoritative, UDP) and a compute ray-tracing pass.

```
cargo test
cargo run --release --bin swarmpf
cargo run --release --bin shot -- out
```

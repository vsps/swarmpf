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
- `client/`: `wgpu` compute ray tracer and a local test level with bots. Every sphere is a light and the
  only light in the scene: surfaces get direct light from importance-sampled emitters with shadow rays plus one
  diffuse bounce, then temporal and edge-aware spatial denoising. Tracers and the crosshair are drawn over the
  tone-mapped image so they do not smear. Two binaries:
  - `swarmpf`: interactive window, third-person over-the-shoulder camera. WASD move, mouse look, hold Shift to
    disperse, hold LMB to fire, 1/2/3 weapon, V toggles first person, `[` / `]` lower / raise the ray-trace
    resolution (default half), Esc releases the mouse.
  - `shot <prefix> [trace_scale]`: headless scripted scenario that writes PNGs (works on software Vulkan, e.g.
    llvmpipe).

Networking (`server/`, authoritative UDP) is deliberately on hold.

```
cargo test
cargo run --release --bin swarmpf
cargo run --release --bin shot -- out
```

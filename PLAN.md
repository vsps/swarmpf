# swarmpf: design and plan

## Concept

Multiplayer FPS. Each player is a cloud of spheres around one small, lethal core.

- **Coherent mode:** the spheres hold a humanoid shape on a procedural walk cycle. The body moves as a capsule
  and can fire weapons and use things (doors are not implemented yet).
- **Dispersed mode:** the spheres flock (boids) around the core, which is the flock leader. The player moves
  faster and fits through gaps the body cannot, but cannot act. Re-forming needs room for the body capsule.
- **Combat:** shots destroy spheres. Coherent players are easy to hit; dispersed players are hard to hit but can't
  shoot back.
- **Look:** minimalist, no textures, no assets. Real-time ray tracing where every sphere is the only light source.

## Design decisions (and why)

- **Spheres, not cubes.** Ray-sphere is one dot product, one discriminant and a sqrt. Boid separation is
  `dist < ri + rj`. Sphere vs box collision is a clamp. Normals are free.
- **Per-sphere health.** Radius is random in `[R_MIN, R_MAX]` from the player's seed; `hp = ceil(HP_K * (r/R_MIN)^2)`,
  so the biggest sphere takes 4x the hits of the smallest. Big spheres sit on the outer shell of each bone, so they
  armour the core. Radii are regenerated from the seed, never sent.
- **Core.** One tiny sphere (`CORE_RADIUS` 0.04) buried in the chest, the flock leader when dispersed. Any hit kills.
  Its hit sphere is inflated 1.5x (`CORE_HIT_SCALE`) so it is hittable. Shots hit the nearest sphere first, so the
  intact body shields the core and damage opens lines to it.
- **Penetration.** `Weapon.pierce` extra spheres per shot; each pierced sphere halves damage (min 1).
  Rifle 2/0, pellet 1/0, railgun 6/3.
- **Procedural humanoid, no Mixamo.** 16-joint rig generated in code; every joint only swings about X. Pose is a
  pure function of (walk phase, amplitude) so server and clients agree. 64 slots on the bones (torso is two shells
  deep); slot layout mirrors legs.
- **World is boxes only.** Level = floor + AABBs. Collision, hitscan and tracing all use simple primitives.
- **Shared sim crate.** `sim` has no dependencies and no I/O, so the future server and the client run identical code
  (prediction matches). Own `Vec3` and SplitMix64 `Rng`, no external crates.
- **Ray tracer is software (compute), not hardware RT.** wgpu ray queries are experimental and native-only; the
  scene (a few hundred spheres and a dozen boxes) is small enough for brute force with per-player bounding spheres.

## Repository layout

```
sim/      shared simulation, tests in sim/tests/sim.rs
  math.rs      Vec3, Rng, ray_sphere
  world.rs     Aabb, World (floor + boxes): push_sphere, sphere_hits, ray_cast
  skeleton.rs  rig, procedural walk pose, slot layout, core placement
  player.rs    Player: body capsule, swarm flocking, mode blend, debris
  hit.rs       hitscan with penetration, aim_distance
client/   wgpu renderer + local game
  src/shader.wgsl   trace / temporal / spatial / present passes
  src/renderer.rs   buffers, bind groups, pass orchestration
  src/game.rs       test level, bots, firing, tracers, third-person camera, scene builder
  src/camera.rs     matrices and yaw/pitch camera
  src/bin/swarmpf.rs  interactive window
  src/bin/shot.rs     headless scripted screenshots (PNG)
```

## Conventions worth knowing

- Yaw 0 faces +Z; yaw increases toward +X, which is the player's LEFT. So mouse-right decreases yaw.
- Body space is +Y up, +Z forward, +X left ("L" joints/slots are +X). `Input.strafe > 0` means right.
- Fixed tick 60 Hz (`sim::DT`). The client accumulates real time and ticks the sim.
- Dead players: `kill()` marks them and gives spheres an outward impulse; `step_debris` makes them tumble.
  The game respawns after 3 s.

## Renderer

Passes (all in `shader.wgsl`):

1. **trace** (compute): primary ray per pixel at `TRACE_SCALE` (default 0.5) of window size. Surfaces get direct
   light from sphere emitters via weighted reservoir sampling (weight ~ `lum * r^2 * cos / d^2`, two picks, shadow ray
   to a random point on the light's disc) plus one cosine-sampled bounce with the same direct-light estimator at
   the bounce point. Spheres display compressed emission (`EMITTER_DISPLAY`) plus light from neighbours.
2. **temporal** (compute): reproject static surfaces with the previous view-proj, validate by id / normal / depth,
   blend with history count capped at 6 (lights move every frame, so long history smears).
3. **spatial** (compute): 7x7 bilateral on normals and plane distance, static surfaces only.
4. **present** (fragment): bilinear upscale, ACES tone map, tracers (closest-approach glow, depth tested against the
   trace), crosshair, dither. Tracers and crosshair are drawn after tone mapping so they never enter history.

There is no sky, ambient or fill light. Floor and ceiling boxes exist purely so light has something to bounce off.
Tuning knobs: `EMIT` / `CORE_EMIT` in `game.rs`, `exposure` in the scene, `EMITTER_DISPLAY`, history cap, `TRACE_SCALE`.

## Status

Done and tested (13 sim tests): sphere swarm, hp model, core, hitscan with penetration and wall blocking, humanoid rig
and walk cycle, boid dispersal, gap filtering and no re-forming inside walls, debris.
Done and checked only through headless screenshots: ray tracer, third-person camera, tracers, crosshair, hit flash.
**Not yet run on real hardware**: the interactive window (mouse grab, vsync, resize, `[` / `]` scale keys), the mouse
yaw fix, and real GPU performance of the tracer.

## Next steps

1. **Play-test on a GPU.** Frame rate at scale 0.5 and 1.0; mouse feel; is the darkness right; is the noise
   acceptable. If slow: add a uniform grid (cell `2*r_max`, CPU-built counting sort) for spheres, or reduce
   reservoir picks, or cut the bounce.
2. **Lag-compensation history** for hitscan (about 250 ms of element positions) in `sim`.
3. **Networking (on hold, by request).** `server/` authoritative at 60 Hz, snapshots 20-30 Hz over UDP
   (`renet` or `quinn` datagrams), client prediction for the local body, interpolation for others.
   Coherent players send position, mode and an hp nibble per element; dispersed players send element positions
   quantized to 16 bits relative to the core. Hit events go reliably.
4. **Doors and interaction** (coherent-mode only action).
5. **Gameplay polish:** impact effects, HUD (hp / armour), respawn flow, scoring, more weapons, swarm-mass effects
   (heavier spheres lag more), stuck-sphere behaviour in narrow gaps.
6. **Tuning:** `R_MIN`/`R_MAX`, `HP_K`, `SWARM_RADIUS`, `CORE_HIT_SCALE`, dispersed speed, shielding fraction of the
   torso (test asserts at least 60% of directions are shielded; the real figure is unmeasured), and whether dispersed
   mode is too safe or too weak.
7. **Later:** browser build (wgpu already targets WebGPU), spectator, level editor from a box list.

## Known issues and gotchas

- Sphere-sphere collision does not exist; spheres only repel through boid separation (dispersed) or springs
  (coherent). Feet spheres rest on the floor, so slot error at the feet is expected (tests clamp for it).
- `push_sphere` sums contact normals, which cancel in a doorway. Use `sphere_hits` for contact tests.
- Slot spring feed-forward acceleration made things worse (one-frame-late derivative); it was removed. Do not re-add
  it without fixing that.
- wgpu is version 30, whose API differs from older tutorials: `queue.present(frame)`, `Some(&layout)` in
  `bind_group_layouts`, `immediate_size`, `multiview_mask`, `get_mapped_range().unwrap()`.
- Headless rendering works on software Vulkan (llvmpipe); `apt-get install mesa-vulkan-drivers vulkan-tools`.
  `shot` takes about 20 s for 4 frames at scale 0.5 there. Release builds use LTO and take over a minute.
- The window binary is unverified; expect small fixes on the first real run.

## Commands

```
cargo test                                  # sim tests
cargo clippy --all-targets
cargo run --release --bin swarmpf           # play
cargo run --release --bin shot -- out 0.5   # writes out_walk/swarm/fire/aftermath.png
```

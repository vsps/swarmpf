# swarmpf: design and plan

## Concept

Multiplayer FPS. Each player is a cloud of spheres around one small, lethal core.

- **Coherent mode:** the spheres are attracted to slots on a humanoid rig (soft underdamped springs, stiffness
  varied per sphere; `K_SLOT`, `K_JITTER`, `SLOT_ZETA`, `SLOT_FOLLOW`). The pull grows with distance (`REEL_*`) so
  stragglers and re-forming swarms snap back while small motion stays loose. Spheres also drift around their slots
  (`WANDER`) and get a downward kick at each footfall (`STEP_KICK`), so the body never looks welded to the rig. A swarm's spheres collide with each other (`collide_elements`), so a hit
  sphere knocks its neighbours; coherent contacts only fire when two spheres are closer than their slots allow running a procedural walk cycle. The body moves as a capsule
  and can fire weapons and use things (doors are not implemented yet).
- **Dispersed mode:** the spheres flock (boids) around the core, which is the flock leader. The player moves
  faster and fits through gaps the body cannot, but cannot act. Re-forming needs room for the body capsule.
  Dispersing takes `DISPERSE_TIME` (0.2 s), twice as quick as re-forming (`BLEND_TIME`, 0.4 s).
  Dispersed, forward follows the look direction including pitch (`Input.pitch`), so the swarm can dive to the
  floor or climb, up to `SWARM_MAX_Y` (25% above body height); with no input it hovers. Near the floor cohesion
  squashes vertically (`GROUND_SQUASH`), flattening the swarm into a disc. The body re-forms with its feet under
  the core but never below the floor; re-formed in mid-air it falls.
- **Combat:** shots destroy spheres. Coherent players are easy to hit; dispersed players are hard to hit but can't
  shoot back.
- **Look:** minimalist, no textures, no assets. Real-time ray tracing where every sphere is the only light source.

## Design decisions (and why)

- **Spheres, not cubes.** Ray-sphere is one dot product, one discriminant and a sqrt. Boid separation is
  `dist < ri + rj`. Sphere vs box collision is a clamp. Normals are free.
- **Per-sphere health.** Every sphere has `ELEM_HP` = 2: rifle and shotgun hits remove 1, a railgun hit removes 2, and hits knock the
  sphere along the shot (`RECOIL`) and slacken its spring for `STUN_TIME`, so it flies ~0.35 m and bumps its
  neighbours before being reeled back. Each player regrows one sphere every `REGEN_TIME` (3 s),
  destroyed before damaged, innermost first; a regrown sphere appears at the core. Radius is random in
  `[R_MIN, R_MAX]` from the player's seed (bigger sit on the outer shell) and is regenerated from the seed, never sent.
- **Core.** One tiny sphere (`CORE_RADIUS` 0.04) buried in the chest, the flock leader when dispersed. Any hit kills.
  Its hit sphere is inflated 1.5x (`CORE_HIT_SCALE`) so it is hittable. Shots hit the nearest sphere first, so the
  intact body shields the core and damage opens lines to it.
- **Penetration.** `Weapon.pierce` extra spheres per shot, `Weapon.damage` hp each. Rifle and pellet: 1 damage,
  no pierce. Railgun: 2 damage (one-shots spheres) and unlimited pierce, stopped only by walls.
- **Guns** (`WEAPONS` in `game.rs`): rifle automatic; shotgun fires 15 pellets uniformly in a 6 degree cone; railgun
  semi-automatic. Damaged spheres show `HIT_COLOR` (red) until they regrow.
- **Procedural humanoid, no Mixamo.** 16-joint rig generated in code; every joint only swings about X. Pose is a
  pure function of (walk phase, amplitude) so server and clients agree. 64 slots on the bones (torso is two shells
  deep); slot layout mirrors legs.
- **World is boxes only.** Level = floor plane + AABBs (the ceiling is one, so the camera and shots stop at it). Collision, hitscan and tracing all use simple primitives.
  Test level (`game.rs::level`): spawn room and a second room past a dividing wall at z = 6 (a 0.5 m crack and a
  1.4 m door). The second room has a 4x4 grid of thin floor-to-ceiling columns (`COLUMN_WIDTH` 0.3, `COLUMN_GAP`
  0.5) around (-2.5, 10.5): the gaps are narrower than the body, so only a swarm passes.
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
  src/ui.rs         pixel font, HUD and settings panel (CPU layout, drawn as quads)
  src/game.rs       test level, bots, firing, tracers, third-person camera, scene builder
  src/camera.rs     matrices and yaw/pitch camera
  src/bin/swarmpf.rs  interactive window
  src/bin/shot.rs     headless scripted screenshots (PNG)
```

## Conventions worth knowing

- Yaw 0 faces +Z; yaw increases toward +X, which is the player's LEFT. So mouse-right decreases yaw.
- Body space is +Y up, +Z forward, +X left ("L" joints/slots are +X). `Input.strafe > 0` means right.
- Fixed tick 60 Hz (`sim::DT`). The client accumulates real time and ticks the sim.
- Rifle is automatic (hold LMB); pellet and railgun fire once per click (`WEAPONS` auto flag, `Game::trigger`).
- Third-person camera is a sphere (`CAM_RADIUS`) that collides with the level and chases a goal behind the player
  (`CAM_LAG_*`, pulled back `CAM_BACK_SWARM` while dispersed). If a wall hides the goal it retraces the core's
  breadcrumb trail, so it follows the swarm through gaps. Dispersed, it turns to face the swarm's centroid.
- Dead players: `kill()` marks them and gives spheres an outward burst; `step_debris` makes them tumble.
  In the client the spheres flash bright yellow, then fade to red while shrinking to 5% over `DEATH_TIME`
  (1.5 s) and vanish (`game.rs`, visual only). The game respawns after 3 s.

## Renderer

Passes (all in `shader.wgsl`):

1. **trace** (compute): primary ray per pixel at `TRACE_SCALE` (default 0.5) of window size. Surfaces get direct
   light from sphere emitters via weighted reservoir sampling (two picks, shadow ray to a random point on the
   light's disc; only boxes cast shadows, spheres are lights and never occlude) plus one cosine-sampled bounce with the same direct-light estimator at the bounce point. Light
   selection is hybrid (`direct_light`): a player within `NEAR_K` bounding radii enters sphere by sphere with the
   exact weight `lum * r^2 * cos / d^2`; a farther player enters as one entry (power from its centre) and, if
   picked, a sphere is chosen by RIS (`RIS_M` candidates from the per-player power CDF built in
   `renderer.rs::build_accel`). Unbiased, same noise as the plain per-light loop, ~30% cheaper. Spheres are flat shaded: one colour each, compressed emission (`EMITTER_DISPLAY`).
2. **temporal** (compute): reproject static surfaces with the previous view-proj, validate by id / normal / depth,
   blend with history count capped at 6 (lights move every frame, so long history smears).
3. **spatial** (compute): 7x7 bilateral on normals and plane distance, static surfaces only; each 8x8 workgroup
   loads its 14x14 tile into workgroup memory once.
4. **present** (fragment): bilinear upscale, ACES tone map, tracers (closest-approach glow, depth tested against the
   trace), crosshair, dither. Tracers and crosshair are drawn after tone mapping so they never enter history.
   While the shotgun or railgun reloads (`Game::reload_progress`, `Scene.reload`), the crosshair becomes a ring
   that fills clockwise from the top.
5. **ui** (instanced quads, alpha blended, same render pass): HUD and settings panel from `ui.rs`, a 3x5 pixel
   font (A-Z, 0-9, symbols) laid out on the CPU in output pixels. HUD (top left): FPS, ray-traced resolution and
   its % of the output, samples per pixel, which smoothing is off. Esc releases the mouse and shows the settings
   panel (samples and resolution -/+, raw and temporal on/off, resume); Esc again resumes.

Display options (keys, or the settings panel):
- **Raw pixels** (`Renderer::set_raw`, I): nearest-neighbour upscale, no spatial blur, no dither.
- **Temporal accumulation** (`Renderer::set_temporal`, T, on by default): per-pixel reprojected history. Raw +
  temporal gives accumulated, unblurred pixels; both off is the bare trace.
- **Samples per pixel** (`Renderer::set_spp`, , and ., 1-16): the lighting (light picks, shadow rays, bounce) is
  averaged over N samples of the same pixel-centre hit, so no AA is added. Raw walk scene at 0.5: 1 spp 5.6 ms,
  noise 0.097; 8 spp 34.7 ms, noise 0.058 (`shot --bench quick`). A per-sample firefly clamp made no difference:
  the residual grain is ordinary variance, not rare outliers.

Performance (M2, 1280x720, `shot --bench`): trace is most of the frame. Shadow rays used to walk a body's ~65
spheres and cost ~70% of trace; spheres no longer cast shadows, which removed that (and brightens the scene, as
bodies no longer block their own light). Box clusters (the column grid) are grouped on the CPU
(`renderer.rs::box_chunks`) and skipped unless a ray reaches their bounds; loose boxes are tested in a flat loop.
The grid costs ~7% of frame time this way, against ~23% testing all its boxes. Tried without gain before that: per-player sphere clusters (2-level
BVH), loading only `pos_r`, largest-first sphere order, a single `direct_light` call site.
On Apple GPUs the present pass's timestamps overlap the compute passes; trust the wall-clock ms/frame.

There is no sky, ambient or fill light. Floor and ceiling boxes exist purely so light has something to bounce off.
Tuning knobs: `EMIT` / `CORE_EMIT` in `game.rs`, `exposure` in the scene, `EMITTER_DISPLAY`, history cap, `TRACE_SCALE`.

## Status

Done and tested (18 sim tests): sphere swarm, hp model, core, hitscan with penetration and wall blocking, humanoid rig
and walk cycle, boid dispersal, gap filtering and no re-forming inside walls, debris.
Done and checked only through headless screenshots: ray tracer, third-person camera, tracers, crosshair, hit flash.
**Not yet run on real hardware**: the interactive window (mouse grab, vsync, resize, `[` / `]` scale keys), the mouse
yaw fix, and real GPU performance of the tracer.

## Next steps

1. **Play-test on a GPU.** Frame rate at scale 0.5 and 1.0; mouse feel; is the darkness right; is the noise
   acceptable. If slow: options that change the look are fewer reservoir picks, no shadow ray at the bounce,
   or a lower default trace scale.
2. **Lag-compensation history** for hitscan (about 250 ms of element positions) in `sim`.
3. **Networking (on hold, by request).** `server/` authoritative at 60 Hz, snapshots 20-30 Hz over UDP
   (`renet` or `quinn` datagrams), client prediction for the local body, interpolation for others.
   Coherent players send position, mode and an hp nibble per element; dispersed players send element positions
   quantized to 16 bits relative to the core. Hit events go reliably.
4. **Doors and interaction** (coherent-mode only action).
5. **Gameplay polish:** impact effects, HUD (hp / armour), respawn flow, scoring, more weapons, swarm-mass effects
   (heavier spheres lag more), stuck-sphere behaviour in narrow gaps.
6. **Tuning:** `R_MIN`/`R_MAX`, `ELEM_HP`, `REGEN_TIME`, `RECOIL`, `SWARM_RADIUS`, `CORE_HIT_SCALE`, dispersed speed, shielding fraction of the
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
cargo run --release --bin shot -- out 0.5   # writes out_walk/walk_raw/walk_raw8/walk_raw_temporal/settings/swarm/fire/death/death_late/aftermath/shotgun/reload/gap/past_gap/dive/columns/columns_through.png
cargo run --release --bin shot -- --bench   # GPU ms per pass, mean brightness (bias check), frame-to-frame noise
cargo run --release --bin shot -- --bench quick  # only the raw walk scene at 1 and 8 samples per pixel
cargo run --release -p sim --example bench  # physics us/tick and a position checksum (must not change on refactors)
```

//! Headless renderer: runs a scripted scenario and writes PNG screenshots.
//! Usage: shot <out_prefix> [trace_scale]   writes walk, walk_raw, walk_raw8, swarm, fire, aftermath, shotgun, gap, past_gap
//!        shot --bench        GPU ms per pass and mean image brightness, at trace scale 0.5 and 1.0

use client::game::Game;
use client::renderer::{Renderer, PASSES};
use sim::player::Input;

const W: u32 = 1280;
const H: u32 = 720;
const WARMUP: usize = 16;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn main() {
    let prefix = std::env::args().nth(1).unwrap_or_else(|| "shot".into());
    let scale: f32 = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(0.5);
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("no adapter");
    eprintln!("adapter: {}", adapter.get_info().name);
    // Timestamp queries (for --bench) where the adapter has them.
    let desc = wgpu::DeviceDescriptor {
        required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
        ..Default::default()
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&desc)).expect("device");
    if prefix == "--bench" {
        let quick = std::env::args().nth(2).as_deref() == Some("quick");
        bench(&device, &queue, quick);
        return;
    }
    let mut renderer = Renderer::new(device.clone(), queue.clone(), FORMAT, W, H, scale);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());

    let mut game = Game::new();
    let snap = |game: &Game, renderer: &mut Renderer, name: &str| {
        // Let the temporal filter converge on the (static) frame; time it for the FPS counter.
        let mut scene = game.scene();
        let t0 = std::time::Instant::now();
        for _ in 0..WARMUP {
            renderer.render(&view, &scene);
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        scene.fps = WARMUP as f32 / t0.elapsed().as_secs_f32();
        renderer.render(&view, &scene);
        save(&device, &queue, &target, &format!("{prefix}_{name}.png"));
        eprintln!("wrote {prefix}_{name}.png");
    };
    let run = |game: &mut Game, secs: f32, input: Input| {
        for _ in 0..(secs / sim::DT) as usize {
            game.tick(input);
        }
    };

    // 1. Third-person view of yourself and the bots in coherent mode.
    game.third_person = true;
    game.yaw = 0.0;
    game.pitch = -0.1;
    run(&mut game, 1.0, Input::default());
    run(
        &mut game,
        1.2,
        Input {
            forward: 1.0,
            ..Default::default()
        },
    );
    snap(&game, &mut renderer, "walk");
    // Same frame as raw pixels: no interpolation, accumulation, blur or dither.
    renderer.set_raw(true);
    snap(&game, &mut renderer, "walk_raw");
    // Raw with 8 lighting samples per pixel: less noise, still no interpolation.
    renderer.set_spp(8);
    snap(&game, &mut renderer, "walk_raw8");
    renderer.set_spp(1);
    renderer.set_raw(false);

    // 2. Disperse.
    run(
        &mut game,
        1.5,
        Input {
            disperse: true,
            ..Default::default()
        },
    );
    snap(&game, &mut renderer, "swarm");

    // Put the crosshair on a player's core. The third-person camera trails, so let it settle
    // and aim again; the last aim is immediate so a moving target is still under the crosshair.
    let aim = |game: &mut Game, target: usize| {
        for k in 0..3 {
            let cam = game.camera(70f32.to_radians());
            let t = game.players[target].core.pos;
            let (dx, dy, dz) = (t.x - cam.eye[0], t.y - cam.eye[1], t.z - cam.eye[2]);
            game.yaw = dx.atan2(dz);
            game.pitch = dy.atan2((dx * dx + dz * dz).sqrt());
            if k < 2 {
                run(game, 0.25, Input::default());
            }
        }
    };

    // 3. Third person: a short rifle burst into the statue bot; hit spheres turn red and recoil.
    // Catch the last tracer in flight. (A railgun shot at the core would kill it outright.)
    run(&mut game, 1.0, Input::default());
    aim(&mut game, 1);
    game.weapon = 0;
    let mut hits = 0;
    for _ in 0..3 {
        hits += game.fire().len();
        run(&mut game, 0.12, Input::default());
    }
    hits += game.fire().len();
    eprintln!("rifle hits: {hits}");
    run(&mut game, 0.03, Input::default());
    snap(&game, &mut renderer, "fire");
    // Keep firing until the statue is dead, then show the aftermath.
    game.weapon = 0;
    for _ in 0..400 {
        game.fire();
        run(&mut game, 0.05, Input::default());
        if !game.players[1].alive {
            break;
        }
    }
    run(&mut game, 1.5, Input::default());
    snap(&game, &mut renderer, "aftermath");
    eprintln!("{}", game.status());

    // Shotgun blast at the circling bot: 15 pellets in a 3 degree cone.
    aim(&mut game, 2);
    game.weapon = 1;
    let hits = game.fire();
    eprintln!("shotgun hits: {}", hits.len());
    run(&mut game, 0.02, Input::default());
    snap(&game, &mut renderer, "shotgun");

    // 4. Disperse and squeeze through the 0.5 m crack at x = 0, z = 6. The camera sphere has to
    // follow the swarm through the gap rather than cut through the wall.
    game.yaw = 0.0;
    game.pitch = -0.15;
    let through = Input {
        forward: 1.0,
        disperse: true,
        ..Default::default()
    };
    let (mut in_wall, mut snapped) = (0, 0);
    for _ in 0..(8.0 / sim::DT) as usize {
        let me = &game.players[0];
        // Steer the core onto the crack's centre line.
        let input = Input {
            strafe: (me.core.pos.x * 4.0).clamp(-1.0, 1.0),
            ..through
        };
        game.tick(input);
        let e = game.camera(1.0).eye;
        if game.world.sphere_hits(sim::math::v3(e[0], e[1], e[2]), 0.1) {
            in_wall += 1;
        }
        let z = game.players[0].core.pos.z;
        if (snapped == 0 && z > 6.0) || (snapped == 1 && z > 12.0) {
            snap(&game, &mut renderer, ["gap", "past_gap"][snapped]);
            snapped += 1;
        }
        if snapped == 2 {
            break;
        }
    }
    eprintln!(
        "camera ticks inside a wall: {in_wall}; core z = {}",
        game.players[0].core.pos.z
    );
}

/// Time each pass on two scenes (coherent walk, dispersed swarm) at two trace scales, and print
/// the mean brightness of the final image so a change to the light estimator can be checked
/// for bias (noise averages out over the frame; the mean should not move).
fn bench(device: &wgpu::Device, queue: &wgpu::Queue, quick: bool) {
    const FRAMES: usize = 200;
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let mut game = Game::new();
    game.pitch = -0.1;
    let tick = |game: &mut Game, secs: f32, input: Input| {
        for _ in 0..(secs / sim::DT) as usize {
            game.tick(input);
        }
    };
    tick(&mut game, 1.0, Input::default());
    let walk_in = Input {
        forward: 1.0,
        ..Default::default()
    };
    tick(&mut game, 1.2, walk_in);
    let walk = game.scene();
    tick(
        &mut game,
        1.5,
        Input {
            disperse: true,
            ..Default::default()
        },
    );
    let swarm = game.scene();

    // (label, scale, scene, raw, samples per pixel). `quick` only measures the walk scene in raw
    // mode, which is what the samples-per-pixel setting is for.
    let mut runs = vec![];
    if !quick {
        for scale in [0.5f32, 1.0] {
            runs.push(("walk", scale, &walk, false, 1));
            runs.push(("swarm", scale, &swarm, false, 1));
        }
    }
    runs.push(("raw1", 0.5, &walk, true, 1));
    runs.push(("raw8", 0.5, &walk, true, 8));
    for (name, scale, scene, raw, spp) in runs {
        {
            let mut r = Renderer::new(device.clone(), queue.clone(), FORMAT, W, H, scale);
            r.set_raw(raw);
            r.set_spp(spp);
            if !r.enable_timing() {
                eprintln!("no timestamp queries on this adapter");
                return;
            }
            let mut sum = [0.0f32; 4];
            let mut lum = 0.0;
            for f in 0..FRAMES + 20 {
                r.render(&view, scene);
                let t = r.timings().unwrap();
                if f >= 20 {
                    for (s, t) in sum.iter_mut().zip(t) {
                        *s += t / FRAMES as f32;
                    }
                }
                // Mean brightness over the last few frames.
                if f >= FRAMES + 10 {
                    lum += mean_linear(&read_pixels(device, queue, &target)) / 10.0;
                }
            }
            // Noise: RMS difference between two consecutive frames of this static scene.
            let fa = read_pixels(device, queue, &target);
            r.render(&view, scene);
            let fb = read_pixels(device, queue, &target);
            let lin = |c: u8| (c as f32 / 255.0).powf(2.2);
            let noise = (fa
                .iter()
                .zip(&fb)
                .enumerate()
                .filter(|(i, _)| i % 4 != 3)
                .map(|(_, (&a, &b))| (lin(a) - lin(b)).powi(2))
                .sum::<f32>()
                / (fa.len() as f32 * 0.75))
                .sqrt();
            // Wall clock with frames pipelined as in the game (no per-frame readback).
            let t0 = std::time::Instant::now();
            for _ in 0..FRAMES {
                r.render(&view, scene);
            }
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let wall = t0.elapsed().as_secs_f32() * 1e3 / FRAMES as f32;
            let parts: Vec<String> = PASSES
                .iter()
                .zip(sum)
                .map(|(n, t)| format!("{n} {t:.2}"))
                .collect();
            println!(
                "scale {scale} {name:5}: {wall:.2} ms/frame | gpu {} | mean {lum:.4} noise {noise:.4}",
                parts.join(", ")
            );
        }
    }
}

/// Mean linear-light value of an sRGB RGBA8 image.
fn mean_linear(px: &[u8]) -> f32 {
    let lin = |c: u8| (c as f32 / 255.0).powf(2.2);
    let n = px.len() / 4;
    px.chunks(4)
        .map(|p| (lin(p[0]) + lin(p[1]) + lin(p[2])) / 3.0)
        .sum::<f32>()
        / n as f32
}

fn save(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, path: &str) {
    let pixels = read_pixels(device, queue, tex);
    let file = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), W, H);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .unwrap()
        .write_image_data(&pixels)
        .unwrap();
}

fn read_pixels(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture) -> Vec<u8> {
    let bpr = (W * 4).next_multiple_of(256);
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: (bpr * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
        tex.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buf,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bpr),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([enc.finish()]);
    let slice = buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = slice.get_mapped_range().unwrap();
    let mut pixels = Vec::with_capacity((W * H * 4) as usize);
    for row in 0..H as usize {
        pixels.extend_from_slice(&data[row * bpr as usize..row * bpr as usize + (W * 4) as usize]);
    }
    pixels
}

//! Headless renderer: runs a scripted scenario and writes PNG screenshots.
//! Usage: shot <out_prefix> [trace_scale]

use client::game::Game;
use client::renderer::Renderer;
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
    let (device, queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("device");
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
        // Let the temporal filter converge on the (static) frame.
        let scene = game.scene();
        for _ in 0..WARMUP {
            renderer.render(&view, &scene);
        }
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

    // 3. Third person: put the crosshair on the statue bot and fire; catch the tracer in flight.
    run(&mut game, 1.0, Input::default());
    for _ in 0..4 {
        let cam = game.camera(70f32.to_radians());
        let t = game.players[1].core.pos;
        let (dx, dy, dz) = (t.x - cam.eye[0], t.y - cam.eye[1], t.z - cam.eye[2]);
        game.yaw = dx.atan2(dz);
        game.pitch = dy.atan2((dx * dx + dz * dz).sqrt());
    }
    game.weapon = 2;
    let hits = game.fire();
    eprintln!("railgun hits: {}", hits.len());
    run(&mut game, 0.035, Input::default());
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
}

fn save(device: &wgpu::Device, queue: &wgpu::Queue, tex: &wgpu::Texture, path: &str) {
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
    let file = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), W, H);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()
        .unwrap()
        .write_image_data(&pixels)
        .unwrap();
}

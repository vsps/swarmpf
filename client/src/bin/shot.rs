//! Headless renderer: runs a scripted scenario and writes PNG screenshots.
//! Usage: shot <out_prefix> [scenario]

use client::game::Game;
use client::renderer::Renderer;
use sim::player::Input;

const W: u32 = 1280;
const H: u32 = 720;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn main() {
    let prefix = std::env::args().nth(1).unwrap_or_else(|| "shot".into());
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
    let mut renderer = Renderer::new(device.clone(), queue.clone(), FORMAT, W, H);
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
        renderer.render(&view, &game.scene());
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

    // 3. First person: face the statue bot and fire the rifle, then the railgun.
    game.third_person = false;
    run(&mut game, 1.0, Input::default());
    let me = game.eye();
    let t = game.players[1].core.pos;
    game.yaw = (t.x - me.x).atan2(t.z - me.z);
    let flat = ((t.x - me.x).powi(2) + (t.z - me.z).powi(2)).sqrt();
    game.pitch = (t.y - me.y).atan2(flat);
    for k in 0..6 {
        game.weapon = if k < 3 { 0 } else { 2 };
        game.fire();
        run(&mut game, 0.1, Input::default());
    }
    snap(&game, &mut renderer, "fps");
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

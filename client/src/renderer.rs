//! Compute ray-tracing renderer.
//!
//! Every sphere is an emitter and the only light in the scene; the level is lit by direct light
//! from the spheres plus one diffuse bounce. See `shader.wgsl` for the passes.

use crate::camera::{Camera, Mat4};
use crate::ui::UiInst;
use bytemuck::{Pod, Zeroable};

/// One sphere. `emit` is rgb radiance (>1 is fine) with `a` = diffuse albedo.
/// A sphere with (near) zero emission is just a lit ball, not a light.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SphereInst {
    pub pos_r: [f32; 4],
    pub emit: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BoxInst {
    pub min: [f32; 4],
    pub max: [f32; 4],
    pub albedo: [f32; 4],
}

/// A glowing segment drawn over the image (not a light).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct TracerInst {
    /// xyz start, w width in metres.
    pub a: [f32; 4],
    /// xyz end, w intensity.
    pub b: [f32; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GroupGpu {
    bound: [f32; 4],
    range: [u32; 4],
    /// x: light power of the group, sum of luminance * r^2 over its spheres.
    light: [f32; 4],
}

/// Bounding sphere (centre, radius) of some spheres.
fn bound_of(s: &[SphereInst]) -> [f32; 4] {
    let n = s.len().max(1) as f32;
    let mut ctr = [0.0f32; 3];
    for e in s {
        for (k, c) in ctr.iter_mut().enumerate() {
            *c += e.pos_r[k] / n;
        }
    }
    let r = s
        .iter()
        .map(|e| {
            let d = [
                e.pos_r[0] - ctr[0],
                e.pos_r[1] - ctr[1],
                e.pos_r[2] - ctr[2],
            ];
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() + e.pos_r[3]
        })
        .fold(0.0f32, f32::max);
    [ctr[0], ctr[1], ctr[2], r + 0.01]
}

/// Per-player bounding spheres (let rays skip whole swarms) and light power, and each group's
/// light CDF (so the shader picks a sphere by power with a binary search instead of looping
/// over every light).
/// A run of boxes (`range`: first, count) and the box bounding them all.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BoxChunk {
    bmin: [f32; 4],
    bmax: [f32; 4],
    range: [u32; 4],
}

/// Largest extent (m) and box count of a chunk.
const CHUNK_EXTENT: f32 = 4.0;
const CHUNK_MAX: usize = 32;

/// Group consecutive boxes into chunks whose bounds stay within `CHUNK_EXTENT`, so a ray tests
/// a cluster's bounds once instead of each of its boxes (the column grid is one chunk). Returns
/// the boxes reordered with the loose ones (chunks of one: walls, floor, ceiling) first, how many
/// are loose, and the multi-box chunks. The shader tests loose boxes in a flat loop.
fn box_chunks(boxes: &[BoxInst]) -> (Vec<BoxInst>, u32, Vec<BoxChunk>) {
    let mut runs: Vec<(usize, usize, [f32; 3], [f32; 3])> = Vec::new();
    for (i, b) in boxes.iter().enumerate() {
        let (bmin, bmax) = (
            [b.min[0], b.min[1], b.min[2]],
            [b.max[0], b.max[1], b.max[2]],
        );
        if let Some(r) = runs.last_mut() {
            let lo: [f32; 3] = std::array::from_fn(|k| r.2[k].min(bmin[k]));
            let hi: [f32; 3] = std::array::from_fn(|k| r.3[k].max(bmax[k]));
            if (0..3).all(|k| hi[k] - lo[k] <= CHUNK_EXTENT) && r.1 < CHUNK_MAX {
                *r = (r.0, r.1 + 1, lo, hi);
                continue;
            }
        }
        runs.push((i, 1, bmin, bmax));
    }
    let mut out: Vec<BoxInst> = runs
        .iter()
        .filter(|r| r.1 == 1)
        .map(|r| boxes[r.0])
        .collect();
    let loose = out.len() as u32;
    let mut chunks = Vec::new();
    for &(first, n, lo, hi) in runs.iter().filter(|r| r.1 > 1) {
        chunks.push(BoxChunk {
            bmin: [lo[0], lo[1], lo[2], 0.0],
            bmax: [hi[0], hi[1], hi[2], 0.0],
            range: [out.len() as u32, n as u32, 0, 0],
        });
        out.extend_from_slice(&boxes[first..first + n]);
    }
    (out, loose, chunks)
}

fn build_accel(scene: &Scene) -> (Vec<GroupGpu>, Vec<f32>) {
    let luminance = |e: &[f32; 4]| 0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2];
    let mut groups = Vec::new();
    let mut cdf = vec![0.0f32; scene.spheres.len()];
    for &(start, n) in scene.groups.iter().filter(|&&(_, n)| n > 0) {
        let first = start as usize;
        let s = &scene.spheres[first..first + n as usize];
        let mut acc = 0.0;
        for (k, sp) in s.iter().enumerate() {
            let lum = luminance(&sp.emit);
            if lum >= 1e-3 {
                acc += lum * sp.pos_r[3] * sp.pos_r[3];
            }
            cdf[first + k] = acc;
        }
        if acc > 0.0 {
            for c in &mut cdf[first..first + s.len()] {
                *c /= acc;
            }
        }
        groups.push(GroupGpu {
            bound: bound_of(s),
            range: [start, n, 0, 0],
            light: [acc, 0.0, 0.0, 0.0],
        });
    }
    (groups, cdf)
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    prev_view_proj: Mat4,
    eye: [f32; 4],
    right_s: [f32; 4],
    up_s: [f32; 4],
    fwd: [f32; 4],
    prev_eye: [f32; 4],
    dims: [u32; 4],
    counts: [u32; 4],
    params: [f32; 4],
    flags: [u32; 4],
    /// x: box chunk count.
    counts2: [u32; 4],
}

pub struct Scene {
    pub camera: Camera,
    pub spheres: Vec<SphereInst>,
    /// (first sphere, count) per player; used for per-player bounding spheres.
    pub groups: Vec<(u32, u32)>,
    pub boxes: Vec<BoxInst>,
    pub tracers: Vec<TracerInst>,
    pub hit_flash: f32,
    /// Weapon reload progress in [0, 1] (the crosshair becomes a filling ring), or negative when
    /// there is nothing to show.
    pub reload: f32,
    pub exposure: f32,
    /// HUD and menu quads, drawn over the image.
    pub ui: Vec<UiInst>,
}

struct Growable {
    buf: wgpu::Buffer,
    cap: u64,
}

struct Frame {
    tw: u32,
    th: u32,
    rad: wgpu::Buffer,
    gpos: [wgpu::Buffer; 2],
    gnrm: [wgpu::Buffer; 2],
    acc: [wgpu::Buffer; 2],
    fin: wgpu::Buffer,
}

#[derive(Clone, Copy)]
enum Kind {
    Uniform,
    Ro,
    Rw,
}

/// GPU timestamps around each pass (needs `Features::TIMESTAMP_QUERY`).
struct Timing {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
}

pub const MAX_SPP: u32 = 16;

/// Names of the timed passes, in the order `Renderer::timings` reports them.
pub const PASSES: [&str; 4] = ["trace", "temporal", "spatial", "present"];

pub struct Renderer {
    timing: Option<Timing>,
    /// Raw pixels: nearest-neighbour upscale, no temporal accumulation, spatial blur or dither.
    raw: bool,
    /// Temporal accumulation (per-pixel history across frames; no spatial filtering).
    accumulate: bool,
    /// Lighting samples per pixel (`MAX_SPP` at most).
    spp: u32,
    /// Cached per frame parity; cleared when a buffer they point at is replaced.
    bind_groups: [Option<[wgpu::BindGroup; 5]>; 2],
    device: wgpu::Device,
    queue: wgpu::Queue,
    srgb: bool,
    scale: f32,
    out: (u32, u32),
    frame_no: u32,
    prev: Option<(Mat4, [f32; 3])>,
    globals: wgpu::Buffer,
    spheres: Growable,
    boxes: Growable,
    groups: Growable,
    /// Per sphere: the running fraction of its group's light power (a CDF for picking lights).
    cdf: Growable,
    /// Runs of nearby boxes with a bounding box, so rays skip whole clusters (see `box_chunks`).
    chunks: Growable,
    tracers: Growable,
    fbuf: Frame,
    trace: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    temporal: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    spatial: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    present: (wgpu::RenderPipeline, wgpu::BindGroupLayout),
    ui_pipe: (wgpu::RenderPipeline, wgpu::BindGroupLayout),
    ui: Growable,
}

fn storage(device: &wgpu::Device, label: &str, size: u64) -> Growable {
    let cap = size.max(256);
    Growable {
        buf: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: cap,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        cap,
    }
}

fn image_buffer(device: &wgpu::Device, label: &str, texels: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: texels * 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    })
}

fn make_frame(device: &wgpu::Device, tw: u32, th: u32) -> Frame {
    let n = tw as u64 * th as u64;
    Frame {
        tw,
        th,
        rad: image_buffer(device, "radiance", n),
        gpos: [
            image_buffer(device, "gpos0", n),
            image_buffer(device, "gpos1", n),
        ],
        gnrm: [
            image_buffer(device, "gnrm0", n),
            image_buffer(device, "gnrm1", n),
        ],
        acc: [
            image_buffer(device, "acc0", n),
            image_buffer(device, "acc1", n),
        ],
        fin: image_buffer(device, "final", n),
    }
}

fn layout(
    device: &wgpu::Device,
    label: &str,
    stage: wgpu::ShaderStages,
    entries: &[(u32, Kind)],
) -> wgpu::BindGroupLayout {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = entries
        .iter()
        .map(|&(binding, kind)| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: stage,
            ty: wgpu::BindingType::Buffer {
                ty: match kind {
                    Kind::Uniform => wgpu::BufferBindingType::Uniform,
                    Kind::Ro => wgpu::BufferBindingType::Storage { read_only: true },
                    Kind::Rw => wgpu::BufferBindingType::Storage { read_only: false },
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn compute(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry: &str,
    entries: &[(u32, Kind)],
) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let bgl = layout(device, entry, wgpu::ShaderStages::COMPUTE, entries);
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&pl),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipe, bgl)
}

fn bind<'a>(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    entries: &[(u32, &'a wgpu::Buffer)],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = entries
        .iter()
        .map(|&(binding, buf)| wgpu::BindGroupEntry {
            binding,
            resource: buf.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: bgl,
        entries: &entries,
    })
}

fn trace_dims(out: (u32, u32), scale: f32) -> (u32, u32) {
    (
        ((out.0 as f32 * scale).ceil() as u32).max(1),
        ((out.1 as f32 * scale).ceil() as u32).max(1),
    )
}

impl Renderer {
    /// `scale` is the fraction of the output resolution that is ray traced (0.5 = quarter the rays).
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        scale: f32,
    ) -> Renderer {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        use Kind::*;
        let trace = compute(
            &device,
            &module,
            "trace_main",
            &[
                (0, Uniform),
                (1, Ro),
                (2, Ro),
                (3, Ro),
                (4, Rw),
                (5, Rw),
                (6, Rw),
                (18, Ro),
                (21, Ro),
            ],
        );
        let temporal = compute(
            &device,
            &module,
            "temporal_main",
            &[
                (0, Uniform),
                (7, Ro),
                (8, Ro),
                (9, Ro),
                (10, Ro),
                (11, Ro),
                (12, Ro),
                (13, Rw),
            ],
        );
        let spatial = compute(
            &device,
            &module,
            "spatial_main",
            &[(0, Uniform), (8, Ro), (9, Ro), (14, Ro), (15, Rw)],
        );
        let pbgl = layout(
            &device,
            "present",
            wgpu::ShaderStages::FRAGMENT,
            &[(0, Uniform), (8, Ro), (16, Ro), (17, Ro)],
        );
        let ppl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("present"),
            bind_group_layouts: &[Some(&pbgl)],
            immediate_size: 0,
        });
        let ppipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("present"),
            layout: Some(&ppl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_present"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_present"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // HUD / menu quads: instanced, alpha blended over the presented image.
        let ubgl = layout(
            &device,
            "ui",
            wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            &[(0, Uniform), (20, Ro)],
        );
        let upl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ui"),
            bind_group_layouts: &[Some(&ubgl)],
            immediate_size: 0,
        });
        let upipe = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui"),
            layout: Some(&upl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_ui"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_ui"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (tw, th) = trace_dims((width.max(1), height.max(1)), scale);
        Renderer {
            srgb: format.is_srgb(),
            scale,
            out: (width.max(1), height.max(1)),
            frame_no: 0,
            prev: None,
            globals,
            spheres: storage(&device, "spheres", 1 << 14),
            boxes: storage(&device, "boxes", 1 << 12),
            groups: storage(&device, "groups", 1 << 10),
            cdf: storage(&device, "cdf", 1 << 12),
            chunks: storage(&device, "chunks", 1 << 12),
            tracers: storage(&device, "tracers", 1 << 10),
            fbuf: make_frame(&device, tw, th),
            trace,
            temporal,
            spatial,
            present: (ppipe, pbgl),
            ui_pipe: (upipe, ubgl),
            ui: storage(&device, "ui", 1 << 14),
            timing: None,
            raw: false,
            accumulate: true,
            spp: 1,
            bind_groups: [None, None],
            device,
            queue,
        }
    }

    /// Time every pass on the GPU from now on, if the device supports timestamp queries.
    /// Returns whether timing is on.
    pub fn enable_timing(&mut self) -> bool {
        if !self
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return false;
        }
        let n = PASSES.len() as u64 * 2;
        let buf = |label, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: n * 8,
                usage,
                mapped_at_creation: false,
            })
        };
        self.timing = Some(Timing {
            set: self.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("pass timing"),
                ty: wgpu::QueryType::Timestamp,
                count: n as u32,
            }),
            resolve: buf(
                "timing resolve",
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            ),
            readback: buf(
                "timing readback",
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
        });
        true
    }

    /// Milliseconds each pass (see `PASSES`) took in the last `render`. Blocks until the GPU is
    /// done, so only for benchmarks.
    pub fn timings(&self) -> Option<[f32; 4]> {
        let t = self.timing.as_ref()?;
        let slice = t.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let ticks: Vec<u64> = {
            let data = slice.get_mapped_range().unwrap();
            bytemuck::cast_slice(&data).to_vec()
        };
        t.readback.unmap();
        let ns = self.queue.get_timestamp_period();
        let mut out = [0.0f32; 4];
        for (i, o) in out.iter_mut().enumerate() {
            *o = ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f32 * ns * 1e-6;
        }
        Some(out)
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.out = (width.max(1), height.max(1));
        self.rebuild_frame();
    }

    pub fn set_scale(&mut self, scale: f32) {
        self.scale = scale.clamp(0.25, 1.0);
        self.rebuild_frame();
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Raw pixels on (or off): nearest-neighbour upscale, no spatial blur, no dither. Temporal
    /// accumulation is separate (`set_temporal`); with both off the image is the bare trace.
    pub fn set_raw(&mut self, raw: bool) {
        self.raw = raw;
        self.prev = None; // history from the other mode does not apply
    }

    pub fn raw(&self) -> bool {
        self.raw
    }

    /// Blend each pixel with its reprojected history across frames (on by default).
    pub fn set_temporal(&mut self, on: bool) {
        self.accumulate = on;
        self.prev = None;
    }

    pub fn temporal(&self) -> bool {
        self.accumulate
    }

    /// Ray-traced resolution and output resolution, in pixels.
    pub fn trace_size(&self) -> (u32, u32) {
        (self.fbuf.tw, self.fbuf.th)
    }

    pub fn output_size(&self) -> (u32, u32) {
        self.out
    }

    /// Lighting samples per pixel: noise falls as 1 / sqrt(spp), trace cost grows about linearly.
    pub fn set_spp(&mut self, spp: u32) {
        self.spp = spp.clamp(1, MAX_SPP);
    }

    pub fn spp(&self) -> u32 {
        self.spp
    }

    fn rebuild_frame(&mut self) {
        let (tw, th) = trace_dims(self.out, self.scale);
        self.fbuf = make_frame(&self.device, tw, th);
        self.prev = None; // history is gone
        self.bind_groups = [None, None];
    }

    /// Write `data` into `g`, growing it if needed. Returns true if the buffer was replaced
    /// (so bind groups that point at it are stale).
    fn upload<T: Pod>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        g: &mut Growable,
        data: &[T],
    ) -> bool {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let grown = bytes.len() as u64 > g.cap;
        if grown {
            *g = storage(device, "grown", (bytes.len() as u64).next_power_of_two());
        }
        if !bytes.is_empty() {
            queue.write_buffer(&g.buf, 0, bytes);
        }
        grown
    }

    /// Bind groups for the trace, temporal, spatial and present passes of frames with parity
    /// `cur` (the history buffers swap roles every frame).
    fn make_bind_groups(&self, cur: usize) -> [wgpu::BindGroup; 5] {
        let prv = 1 - cur;
        let d = &self.device;
        let f = &self.fbuf;
        [
            bind(
                d,
                &self.trace.1,
                &[
                    (0, &self.globals),
                    (1, &self.spheres.buf),
                    (2, &self.boxes.buf),
                    (3, &self.groups.buf),
                    (4, &f.rad),
                    (5, &f.gpos[cur]),
                    (6, &f.gnrm[cur]),
                    (18, &self.cdf.buf),
                    (21, &self.chunks.buf),
                ],
            ),
            bind(
                d,
                &self.temporal.1,
                &[
                    (0, &self.globals),
                    (7, &f.rad),
                    (8, &f.gpos[cur]),
                    (9, &f.gnrm[cur]),
                    (10, &f.gpos[prv]),
                    (11, &f.gnrm[prv]),
                    (12, &f.acc[prv]),
                    (13, &f.acc[cur]),
                ],
            ),
            bind(
                d,
                &self.spatial.1,
                &[
                    (0, &self.globals),
                    (8, &f.gpos[cur]),
                    (9, &f.gnrm[cur]),
                    (14, &f.acc[cur]),
                    (15, &f.fin),
                ],
            ),
            bind(
                d,
                &self.present.1,
                &[
                    (0, &self.globals),
                    (8, &f.gpos[cur]),
                    (16, &f.fin),
                    (17, &self.tracers.buf),
                ],
            ),
            bind(
                d,
                &self.ui_pipe.1,
                &[(0, &self.globals), (20, &self.ui.buf)],
            ),
        ]
    }

    pub fn render(&mut self, target: &wgpu::TextureView, scene: &Scene) {
        let (ow, oh) = self.out;
        let aspect = ow as f32 / oh as f32;
        let c = &scene.camera;
        let view_proj = c.view_proj(aspect);
        let tan_y = (c.fov_y * 0.5).tan();
        let tan_x = tan_y * aspect;
        let (prev_vp, prev_eye) = self.prev.unwrap_or((view_proj, c.eye));

        let (groups, cdf) = build_accel(scene);
        let (boxes, loose, chunks) = box_chunks(&scene.boxes);

        let f = &self.fbuf;
        let globals = Globals {
            prev_view_proj: prev_vp,
            eye: [c.eye[0], c.eye[1], c.eye[2], 0.0],
            right_s: [
                c.right[0] * tan_x,
                c.right[1] * tan_x,
                c.right[2] * tan_x,
                0.0,
            ],
            up_s: [c.up[0] * tan_y, c.up[1] * tan_y, c.up[2] * tan_y, 0.0],
            fwd: [c.fwd[0], c.fwd[1], c.fwd[2], 0.0],
            prev_eye: [prev_eye[0], prev_eye[1], prev_eye[2], 0.0],
            dims: [f.tw, f.th, ow, oh],
            counts: [
                scene.spheres.len() as u32,
                groups.len() as u32,
                loose,
                scene.tracers.len() as u32,
            ],
            params: [
                self.frame_no as f32,
                scene.exposure,
                scene.hit_flash,
                scene.reload,
            ],
            flags: [
                self.srgb as u32,
                self.raw as u32,
                self.spp,
                self.accumulate as u32,
            ],
            counts2: [chunks.len() as u32, 0, 0, 0],
        };
        self.queue
            .write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
        let (d, q) = (&self.device, &self.queue);
        let grown = Self::upload(d, q, &mut self.spheres, &scene.spheres)
            | Self::upload(d, q, &mut self.boxes, &boxes)
            | Self::upload(d, q, &mut self.groups, &groups)
            | Self::upload(d, q, &mut self.cdf, &cdf)
            | Self::upload(d, q, &mut self.chunks, &chunks)
            | Self::upload(d, q, &mut self.tracers, &scene.tracers)
            | Self::upload(d, q, &mut self.ui, &scene.ui);
        if grown {
            self.bind_groups = [None, None];
        }

        let cur = (self.frame_no % 2) as usize;
        if self.bind_groups[cur].is_none() {
            self.bind_groups[cur] = Some(self.make_bind_groups(cur));
        }
        let [trace_bg, temporal_bg, spatial_bg, present_bg, ui_bg] =
            self.bind_groups[cur].as_ref().unwrap();
        let d = &self.device;
        let f = &self.fbuf;
        let (gx, gy) = (f.tw.div_ceil(8), f.th.div_ceil(8));
        let mut enc = d.create_command_encoder(&Default::default());
        let ts = |i: u32| {
            self.timing
                .as_ref()
                .map(|t| wgpu::ComputePassTimestampWrites {
                    query_set: &t.set,
                    beginning_of_pass_write_index: Some(2 * i),
                    end_of_pass_write_index: Some(2 * i + 1),
                })
        };
        // One pass per stage so each can be timed; the compute passes are otherwise identical.
        for (i, (pipe, bg)) in [
            (&self.trace.0, trace_bg),
            (&self.temporal.0, temporal_bg),
            (&self.spatial.0, spatial_bg),
        ]
        .into_iter()
        .enumerate()
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(PASSES[i]),
                timestamp_writes: ts(i as u32),
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("present"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: self
                    .timing
                    .as_ref()
                    .map(|t| wgpu::RenderPassTimestampWrites {
                        query_set: &t.set,
                        beginning_of_pass_write_index: Some(6),
                        end_of_pass_write_index: Some(7),
                    }),
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.present.0);
            pass.set_bind_group(0, present_bg, &[]);
            pass.draw(0..3, 0..1);
            if !scene.ui.is_empty() {
                pass.set_pipeline(&self.ui_pipe.0);
                pass.set_bind_group(0, ui_bg, &[]);
                pass.draw(0..6, 0..scene.ui.len() as u32);
            }
        }
        if let Some(t) = &self.timing {
            enc.resolve_query_set(&t.set, 0..8, &t.resolve, 0);
            enc.copy_buffer_to_buffer(&t.resolve, 0, &t.readback, 0, 64);
        }
        self.queue.submit([enc.finish()]);
        self.prev = Some((view_proj, c.eye));
        self.frame_no = self.frame_no.wrapping_add(1);
    }
}

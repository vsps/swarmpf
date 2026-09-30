//! Compute ray-tracing renderer.
//!
//! Every sphere is an emitter and the only light in the scene; the level is lit by direct light
//! from the spheres plus one diffuse bounce. See `shader.wgsl` for the passes.

use crate::camera::{Camera, Mat4};
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
}

pub struct Scene {
    pub camera: Camera,
    pub spheres: Vec<SphereInst>,
    /// (first sphere, count) per player; used for per-player bounding spheres.
    pub groups: Vec<(u32, u32)>,
    pub boxes: Vec<BoxInst>,
    pub tracers: Vec<TracerInst>,
    pub hit_flash: f32,
    pub exposure: f32,
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

pub struct Renderer {
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
    tracers: Growable,
    fbuf: Frame,
    trace: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    temporal: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    spatial: (wgpu::ComputePipeline, wgpu::BindGroupLayout),
    present: (wgpu::RenderPipeline, wgpu::BindGroupLayout),
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
            tracers: storage(&device, "tracers", 1 << 10),
            fbuf: make_frame(&device, tw, th),
            trace,
            temporal,
            spatial,
            present: (ppipe, pbgl),
            device,
            queue,
        }
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

    fn rebuild_frame(&mut self) {
        let (tw, th) = trace_dims(self.out, self.scale);
        self.fbuf = make_frame(&self.device, tw, th);
        self.prev = None; // history is gone
    }

    fn upload<T: Pod>(device: &wgpu::Device, queue: &wgpu::Queue, g: &mut Growable, data: &[T]) {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        if bytes.len() as u64 > g.cap {
            *g = storage(device, "grown", (bytes.len() as u64).next_power_of_two());
        }
        if !bytes.is_empty() {
            queue.write_buffer(&g.buf, 0, bytes);
        }
    }

    pub fn render(&mut self, target: &wgpu::TextureView, scene: &Scene) {
        let (ow, oh) = self.out;
        let aspect = ow as f32 / oh as f32;
        let c = &scene.camera;
        let view_proj = c.view_proj(aspect);
        let tan_y = (c.fov_y * 0.5).tan();
        let tan_x = tan_y * aspect;
        let (prev_vp, prev_eye) = self.prev.unwrap_or((view_proj, c.eye));

        // Per-player bounding spheres let rays skip whole swarms.
        let groups: Vec<GroupGpu> = scene
            .groups
            .iter()
            .filter(|&&(_, n)| n > 0)
            .map(|&(start, n)| {
                let s = &scene.spheres[start as usize..(start + n) as usize];
                let mut ctr = [0.0f32; 3];
                for e in s {
                    for (k, c) in ctr.iter_mut().enumerate() {
                        *c += e.pos_r[k] / n as f32;
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
                GroupGpu {
                    bound: [ctr[0], ctr[1], ctr[2], r + 0.01],
                    range: [start, n, 0, 0],
                }
            })
            .collect();

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
                scene.boxes.len() as u32,
                scene.tracers.len() as u32,
            ],
            params: [self.frame_no as f32, scene.exposure, scene.hit_flash, 0.0],
            flags: [self.srgb as u32, 0, 0, 0],
        };
        self.queue
            .write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
        Self::upload(&self.device, &self.queue, &mut self.spheres, &scene.spheres);
        Self::upload(&self.device, &self.queue, &mut self.boxes, &scene.boxes);
        Self::upload(&self.device, &self.queue, &mut self.groups, &groups);
        Self::upload(&self.device, &self.queue, &mut self.tracers, &scene.tracers);

        let (cur, prv) = (
            (self.frame_no % 2) as usize,
            ((self.frame_no + 1) % 2) as usize,
        );
        let d = &self.device;
        let f = &self.fbuf;
        let trace_bg = bind(
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
            ],
        );
        let temporal_bg = bind(
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
        );
        let spatial_bg = bind(
            d,
            &self.spatial.1,
            &[
                (0, &self.globals),
                (8, &f.gpos[cur]),
                (9, &f.gnrm[cur]),
                (14, &f.acc[cur]),
                (15, &f.fin),
            ],
        );
        let present_bg = bind(
            d,
            &self.present.1,
            &[
                (0, &self.globals),
                (8, &f.gpos[cur]),
                (16, &f.fin),
                (17, &self.tracers.buf),
            ],
        );

        let (gx, gy) = (f.tw.div_ceil(8), f.th.div_ceil(8));
        let mut enc = d.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for (pipe, bg) in [
                (&self.trace.0, &trace_bg),
                (&self.temporal.0, &temporal_bg),
                (&self.spatial.0, &spatial_bg),
            ] {
                pass.set_pipeline(pipe);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
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
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.present.0);
            pass.set_bind_group(0, &present_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([enc.finish()]);
        self.prev = Some((view_proj, c.eye));
        self.frame_no = self.frame_no.wrapping_add(1);
    }
}

//! Phase-1 renderer: billboard-impostor spheres and instanced boxes, flat colour.

use crate::camera::Camera;
use bytemuck::{Pod, Zeroable};

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct SphereInst {
    pub pos_r: [f32; 4],
    /// rgb, and `a` = how emissive (unlit) the sphere is.
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BoxInst {
    pub min: [f32; 4],
    pub max: [f32; 4],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    forward: [f32; 4],
    light: [f32; 4],
}

pub struct Scene {
    pub camera: Camera,
    pub spheres: Vec<SphereInst>,
    pub boxes: Vec<BoxInst>,
    pub clear: [f64; 3],
}

struct Growable {
    buf: wgpu::Buffer,
    cap: u64,
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    globals: wgpu::Buffer,
    spheres: Growable,
    boxes: Growable,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    sphere_pipeline: wgpu::RenderPipeline,
    box_pipeline: wgpu::RenderPipeline,
    depth: wgpu::TextureView,
    size: (u32, u32),
}

fn storage_buffer(device: &wgpu::Device, label: &str, size: u64) -> Growable {
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

fn depth_view(device: &wgpu::Device, w: u32, h: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

impl Renderer {
    pub fn new(
        device: wgpu::Device,
        queue: wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Renderer {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spheres = storage_buffer(&device, "spheres", 1 << 16);
        let boxes = storage_buffer(&device, "boxes", 1 << 12);

        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty,
            count: None,
        };
        let storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("layout"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(1, storage),
                entry(2, storage),
            ],
        });
        let bind_group = Self::make_bind_group(&device, &layout, &globals, &spheres, &boxes);
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let make = |label: &str, vs: &str, fs: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(format.into())],
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let sphere_pipeline = make("spheres", "vs_sphere", "fs_sphere");
        let box_pipeline = make("boxes", "vs_box", "fs_box");
        let depth = depth_view(&device, width, height);

        Renderer {
            device,
            queue,
            globals,
            spheres,
            boxes,
            layout,
            bind_group,
            sphere_pipeline,
            box_pipeline,
            depth,
            size: (width, height),
        }
    }

    fn make_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        globals: &wgpu::Buffer,
        spheres: &Growable,
        boxes: &Growable,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: spheres.buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: boxes.buf.as_entire_binding(),
                },
            ],
        })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let (w, h) = (width.max(1), height.max(1));
        self.size = (w, h);
        self.depth = depth_view(&self.device, w, h);
    }

    fn upload<T: Pod>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        g: &mut Growable,
        data: &[T],
    ) -> bool {
        let bytes: &[u8] = bytemuck::cast_slice(data);
        let mut regrown = false;
        if bytes.len() as u64 > g.cap {
            *g = storage_buffer(device, "grown", (bytes.len() as u64).next_power_of_two());
            regrown = true;
        }
        if !bytes.is_empty() {
            queue.write_buffer(&g.buf, 0, bytes);
        }
        regrown
    }

    pub fn render(&mut self, target: &wgpu::TextureView, scene: &Scene) {
        let aspect = self.size.0 as f32 / self.size.1 as f32;
        let c = &scene.camera;
        let light = [0.4f32, 0.8, 0.45];
        let ll = (light[0] * light[0] + light[1] * light[1] + light[2] * light[2]).sqrt();
        let globals = Globals {
            view_proj: c.view_proj(aspect),
            eye: [c.eye[0], c.eye[1], c.eye[2], 0.0],
            right: [c.right[0], c.right[1], c.right[2], 0.0],
            up: [c.up[0], c.up[1], c.up[2], 0.0],
            forward: [c.fwd[0], c.fwd[1], c.fwd[2], 0.0],
            light: [light[0] / ll, light[1] / ll, light[2] / ll, 0.0],
        };
        self.queue
            .write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
        let a = Self::upload(&self.device, &self.queue, &mut self.spheres, &scene.spheres);
        let b = Self::upload(&self.device, &self.queue, &mut self.boxes, &scene.boxes);
        if a || b {
            self.bind_group = Self::make_bind_group(
                &self.device,
                &self.layout,
                &self.globals,
                &self.spheres,
                &self.boxes,
            );
        }

        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: scene.clear[0],
                            g: scene.clear[1],
                            b: scene.clear[2],
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            if !scene.boxes.is_empty() {
                pass.set_pipeline(&self.box_pipeline);
                pass.draw(0..36, 0..scene.boxes.len() as u32);
            }
            if !scene.spheres.is_empty() {
                pass.set_pipeline(&self.sphere_pipeline);
                pass.draw(0..6, 0..scene.spheres.len() as u32);
            }
        }
        self.queue.submit([enc.finish()]);
    }
}

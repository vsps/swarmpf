//! Interactive client.
//! WASD move, mouse look, Shift hold = disperse, LMB fire, 1/2/3 weapon, V third person, Esc release mouse.

use client::game::Game;
use client::renderer::Renderer;
use sim::player::Input;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
}

#[derive(Default)]
struct Keys {
    w: bool,
    a: bool,
    s: bool,
    d: bool,
    shift: bool,
}

struct App {
    instance: wgpu::Instance,
    gfx: Option<Gfx>,
    game: Game,
    keys: Keys,
    grabbed: bool,
    last: Instant,
    acc: f32,
}

impl App {
    fn grab(&mut self, on: bool) {
        if let Some(g) = &self.gfx {
            let mode = if on {
                CursorGrabMode::Locked
            } else {
                CursorGrabMode::None
            };
            if g.window.set_cursor_grab(mode).is_err() && on {
                let _ = g.window.set_cursor_grab(CursorGrabMode::Confined);
            }
            g.window.set_cursor_visible(!on);
            self.grabbed = on;
        }
    }

    fn input(&self) -> Input {
        let k = &self.keys;
        let axis = |pos: bool, neg: bool| pos as i32 as f32 - neg as i32 as f32;
        Input {
            forward: axis(k.w, k.s),
            strafe: axis(k.d, k.a),
            yaw: self.game.yaw,
            disperse: k.shift,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let window = Arc::new(
            el.create_window(Window::default_attributes().with_title("swarmpf"))
                .expect("window"),
        );
        let surface = self
            .instance
            .create_surface(window.clone())
            .expect("surface");
        let adapter =
            pollster::block_on(self.instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            }))
            .expect("adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("device");
        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface config");
        config.present_mode = wgpu::PresentMode::AutoVsync;
        surface.configure(&device, &config);
        let renderer = Renderer::new(device, queue, config.format, config.width, config.height);
        self.gfx = Some(Gfx {
            window,
            surface,
            config,
            renderer,
        });
        self.grab(true);
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, ev: DeviceEvent) {
        if let (DeviceEvent::MouseMotion { delta }, true) = (ev, self.grabbed) {
            let sens = 0.0025;
            self.game.yaw += delta.0 as f32 * sens;
            self.game.pitch = (self.game.pitch - delta.1 as f32 * sens).clamp(-1.5, 1.5);
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, ev: WindowEvent) {
        match ev {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(s) => {
                if let Some(g) = &mut self.gfx {
                    g.config.width = s.width.max(1);
                    g.config.height = s.height.max(1);
                    g.surface.configure(g.renderer.device(), &g.config);
                    g.renderer.resize(g.config.width, g.config.height);
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                if self.grabbed {
                    self.game.fire();
                } else {
                    self.grab(true);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let down = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::KeyW => self.keys.w = down,
                        KeyCode::KeyA => self.keys.a = down,
                        KeyCode::KeyS => self.keys.s = down,
                        KeyCode::KeyD => self.keys.d = down,
                        KeyCode::ShiftLeft | KeyCode::ShiftRight => self.keys.shift = down,
                        KeyCode::Escape if down => self.grab(false),
                        KeyCode::KeyV if down => self.game.third_person = !self.game.third_person,
                        KeyCode::Digit1 if down => self.game.cycle_weapon(0),
                        KeyCode::Digit2 if down => self.game.cycle_weapon(1),
                        KeyCode::Digit3 if down => self.game.cycle_weapon(2),
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                self.acc += (now - self.last).as_secs_f32().min(0.1);
                self.last = now;
                while self.acc >= sim::DT {
                    let input = self.input();
                    self.game.tick(input);
                    self.acc -= sim::DT;
                }
                let status = self.game.status();
                let scene = self.game.scene();
                if let Some(g) = &mut self.gfx {
                    g.window.set_title(&status);
                    match g.surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(frame)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => {
                            let view = frame.texture.create_view(&Default::default());
                            g.renderer.render(&view, &scene);
                            g.window.pre_present_notify();
                            g.renderer.queue().present(frame);
                        }
                        _ => g.surface.configure(g.renderer.device(), &g.config),
                    }
                    g.window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let el = EventLoop::new().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        instance: wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(
            Box::new(el.owned_display_handle()),
        )),
        gfx: None,
        game: Game::new(),
        keys: Keys::default(),
        grabbed: false,
        last: Instant::now(),
        acc: 0.0,
    };
    el.run_app(&mut app).unwrap();
}

//! Interactive client.
//! WASD move, mouse look, Shift hold = disperse, LMB fire (rifle is automatic),
//! 1/2/3 rifle / shotgun / railgun, V first/third person,
//! [ and ] lower / raise the ray-trace resolution, , and . fewer / more lighting samples per pixel
//! (1, 2, 4, 8, 16), I toggle raw pixels (nearest upscale, no blur or dither), T toggle temporal
//! accumulation. Esc releases the mouse and opens the settings panel (click its buttons); Esc
//! again resumes. FPS, ray-traced resolution and samples per pixel show top left.

use client::game::Game;
use client::renderer::Renderer;
use client::ui::{self, Action, Ui};
use sim::player::Input;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

/// Fraction of the window resolution that is ray traced. Lower it if the frame rate is poor.
const TRACE_SCALE: f32 = 0.5;

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
    /// Smoothed frame time (s), for the FPS counter.
    frame_time: f32,
    /// Cursor in window pixels, and the UI drawn last frame (for clicking its buttons).
    cursor: Option<(f32, f32)>,
    ui: Ui,
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

    fn status(&self, r: &Renderer) -> ui::Status {
        ui::Status {
            fps: 1.0 / self.frame_time.max(1e-3),
            trace: r.trace_size(),
            output: r.output_size(),
            spp: r.spp(),
            raw: r.raw(),
            temporal: r.temporal(),
        }
    }

    /// Settings shared by the keys and the settings panel.
    fn apply(&mut self, action: Action) {
        if action == Action::Resume {
            self.grab(true);
            return;
        }
        let Some(g) = &mut self.gfx else { return };
        let r = &mut g.renderer;
        match action {
            Action::SppDown => r.set_spp((r.spp() / 2).max(1)),
            Action::SppUp => r.set_spp(r.spp() * 2),
            Action::ScaleDown => r.set_scale(r.scale() - 0.125),
            Action::ScaleUp => r.set_scale(r.scale() + 0.125),
            Action::ToggleRaw => r.set_raw(!r.raw()),
            Action::ToggleTemporal => r.set_temporal(!r.temporal()),
            Action::Resume => {}
        }
    }

    fn input(&self) -> Input {
        let k = &self.keys;
        let axis = |pos: bool, neg: bool| pos as i32 as f32 - neg as i32 as f32;
        Input {
            forward: axis(k.w, k.s),
            strafe: axis(k.d, k.a),
            yaw: self.game.yaw,
            pitch: self.game.pitch,
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
        let renderer = Renderer::new(
            device,
            queue,
            config.format,
            config.width,
            config.height,
            TRACE_SCALE,
        );
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
            // Yaw grows towards the player's left, so moving the mouse right decreases it.
            self.game.yaw -= delta.0 as f32 * sens;
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
                state,
                button: MouseButton::Left,
                ..
            } => {
                let down = state == ElementState::Pressed;
                if self.grabbed {
                    self.game.trigger(down);
                } else if down {
                    // Settings panel: click a button.
                    if let Some(a) = self.cursor.and_then(|(x, y)| self.ui.hit(x, y)) {
                        self.apply(a);
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor = Some((position.x as f32, position.y as f32));
            }
            WindowEvent::CursorLeft { .. } => self.cursor = None,
            WindowEvent::KeyboardInput { event, .. } => {
                let down = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    match code {
                        KeyCode::KeyW => self.keys.w = down,
                        KeyCode::KeyA => self.keys.a = down,
                        KeyCode::KeyS => self.keys.s = down,
                        KeyCode::KeyD => self.keys.d = down,
                        KeyCode::ShiftLeft | KeyCode::ShiftRight => self.keys.shift = down,
                        KeyCode::Escape if down => {
                            if self.grabbed {
                                self.game.trigger(false);
                                self.grab(false);
                            } else {
                                self.grab(true);
                            }
                        }
                        KeyCode::KeyV if down => self.game.third_person = !self.game.third_person,
                        KeyCode::BracketLeft if down => self.apply(Action::ScaleDown),
                        KeyCode::BracketRight if down => self.apply(Action::ScaleUp),
                        KeyCode::Comma if down => self.apply(Action::SppDown),
                        KeyCode::Period if down => self.apply(Action::SppUp),
                        KeyCode::KeyI if down => self.apply(Action::ToggleRaw),
                        KeyCode::KeyT if down => self.apply(Action::ToggleTemporal),
                        KeyCode::Digit1 if down => self.game.cycle_weapon(0),
                        KeyCode::Digit2 if down => self.game.cycle_weapon(1),
                        KeyCode::Digit3 if down => self.game.cycle_weapon(2),
                        _ => {}
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = (now - self.last).as_secs_f32();
                self.acc += dt.min(0.1);
                self.last = now;
                self.frame_time += (dt - self.frame_time) * 0.05;
                while self.acc >= sim::DT {
                    let input = self.input();
                    self.game.tick(input);
                    self.acc -= sim::DT;
                }
                let status = self.game.status();
                let mut scene = self.game.scene();
                if let Some(g) = &self.gfx {
                    let st = self.status(&g.renderer);
                    let mut ui = Ui::default();
                    ui::hud(&mut ui, &st);
                    if !self.grabbed {
                        ui::settings(&mut ui, &st, self.cursor);
                    }
                    scene.ui = std::mem::take(&mut ui.quads);
                    self.ui = ui;
                }
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
        frame_time: 1.0 / 60.0,
        cursor: None,
        ui: Ui::default(),
    };
    el.run_app(&mut app).unwrap();
}

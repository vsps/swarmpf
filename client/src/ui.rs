//! Minimal immediate-mode UI: a 3x5 pixel font and flat rectangles, laid out on the CPU in
//! output pixels and drawn by the renderer as instanced, alpha-blended quads over the image.
//! Builds the HUD (stats, top left) and the settings panel shown while the mouse is released.

use bytemuck::{Pod, Zeroable};

/// One quad: a solid rectangle or one glyph of the pixel font.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct UiInst {
    /// x, y, width, height in output pixels (y down).
    pub rect: [f32; 4],
    /// Linear rgb and alpha.
    pub color: [f32; 4],
    /// x: glyph bits (bit 14 = top-left of 3x5), y: 1 for a glyph, 0 for a solid rectangle.
    pub glyph: [u32; 4],
}

/// 3x5 glyph rows, top to bottom, three bits each (left = high bit). Lowercase maps to upper.
fn glyph_rows(c: char) -> [u8; 5] {
    match c.to_ascii_uppercase() {
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b111, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b001, 0b001, 0b001],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'C' => [0b011, 0b100, 0b100, 0b100, 0b011],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b110, 0b100, 0b100],
        'G' => [0b011, 0b100, 0b101, 0b101, 0b011],
        'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'J' => [0b001, 0b001, 0b001, 0b101, 0b010],
        'K' => [0b101, 0b101, 0b110, 0b101, 0b101],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'N' => [0b110, 0b101, 0b101, 0b101, 0b101],
        'O' => [0b010, 0b101, 0b101, 0b101, 0b010],
        'P' => [0b110, 0b101, 0b110, 0b100, 0b100],
        'Q' => [0b010, 0b101, 0b101, 0b110, 0b011],
        'R' => [0b110, 0b101, 0b110, 0b101, 0b101],
        'S' => [0b011, 0b100, 0b010, 0b001, 0b110],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'W' => [0b101, 0b101, 0b111, 0b111, 0b101],
        'X' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'Y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        'Z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        '%' => [0b101, 0b001, 0b010, 0b100, 0b101],
        '+' => [0b000, 0b010, 0b111, 0b010, 0b000],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        ':' => [0b000, 0b010, 0b000, 0b010, 0b000],
        '/' => [0b001, 0b001, 0b010, 0b100, 0b100],
        _ => [0; 5],
    }
}

fn glyph_bits(c: char) -> u32 {
    glyph_rows(c)
        .iter()
        .fold(0u32, |acc, &row| (acc << 3) | row as u32)
}

/// Something a settings button does.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    SppDown,
    SppUp,
    ScaleDown,
    ScaleUp,
    ToggleRaw,
    ToggleTemporal,
    Resume,
}

/// What the HUD and settings panel show.
pub struct Status {
    pub fps: f32,
    pub trace: (u32, u32),
    pub output: (u32, u32),
    pub spp: u32,
    pub raw: bool,
    pub temporal: bool,
}

#[derive(Default)]
pub struct Ui {
    pub quads: Vec<UiInst>,
    buttons: Vec<([f32; 4], Action)>,
}

const TEXT: [f32; 4] = [0.82, 0.86, 0.92, 0.9];
const DIM: [f32; 4] = [0.45, 0.5, 0.58, 0.9];

impl Ui {
    pub fn rect(&mut self, rect: [f32; 4], color: [f32; 4]) {
        self.quads.push(UiInst {
            rect,
            color,
            glyph: [0; 4],
        });
    }

    /// Draw `s` with its top-left at (x, y); `px` is the size of one font pixel. Returns the width.
    pub fn text(&mut self, x: f32, y: f32, px: f32, color: [f32; 4], s: &str) -> f32 {
        for (i, c) in s.chars().enumerate() {
            let bits = glyph_bits(c);
            if bits != 0 {
                self.quads.push(UiInst {
                    rect: [x + i as f32 * 4.0 * px, y, 3.0 * px, 5.0 * px],
                    color,
                    glyph: [bits, 1, 0, 0],
                });
            }
        }
        text_width(s, px)
    }

    /// A clickable label with a backdrop that lightens under the cursor.
    fn button(
        &mut self,
        x: f32,
        y: f32,
        px: f32,
        label: &str,
        action: Action,
        cursor: Option<(f32, f32)>,
    ) -> f32 {
        let pad = 2.0 * px;
        let rect = [
            x,
            y,
            text_width(label, px) + 2.0 * pad,
            5.0 * px + 2.0 * pad,
        ];
        let hover = cursor.is_some_and(|c| inside(rect, c));
        let bg = if hover {
            [0.35, 0.42, 0.55, 0.9]
        } else {
            [0.16, 0.19, 0.25, 0.9]
        };
        self.rect(rect, bg);
        self.text(x + pad, y + pad, px, TEXT, label);
        self.buttons.push((rect, action));
        rect[2]
    }

    /// The button under (x, y), if any.
    pub fn hit(&self, x: f32, y: f32) -> Option<Action> {
        self.buttons
            .iter()
            .rev()
            .find(|(r, _)| inside(*r, (x, y)))
            .map(|&(_, a)| a)
    }
}

fn text_width(s: &str, px: f32) -> f32 {
    (s.chars().count() as f32 * 4.0 - 1.0).max(0.0) * px
}

fn inside(r: [f32; 4], (x, y): (f32, f32)) -> bool {
    x >= r[0] && x < r[0] + r[2] && y >= r[1] && y < r[1] + r[3]
}

/// Font pixel size for the HUD: 2 output pixels per font pixel at 720p, scaled with the window.
fn unit(output_h: u32) -> f32 {
    (output_h as f32 / 720.0).round().max(1.0)
}

/// Stats in the top-left corner: FPS, ray-traced resolution and its share of the output,
/// samples per pixel and which smoothing steps are off.
pub fn hud(ui: &mut Ui, s: &Status) {
    let px = 2.0 * unit(s.output.1);
    let pct = (100.0 * s.trace.0 as f32 / s.output.0.max(1) as f32).round();
    let mut lines = vec![
        format!("{:.0} FPS", s.fps),
        format!("{}X{} {pct}%", s.trace.0, s.trace.1),
        format!("{} SPP", s.spp),
    ];
    match (s.raw, s.temporal) {
        (true, true) => lines.push("RAW + TEMPORAL".into()),
        (true, false) => lines.push("RAW".into()),
        (false, false) => lines.push("NO TEMPORAL".into()),
        (false, true) => {}
    }
    let w = lines.iter().map(|l| text_width(l, px)).fold(0.0, f32::max);
    let (x, y) = (6.0 * px, 6.0 * px);
    ui.rect(
        [
            x - 2.0 * px,
            y - 2.0 * px,
            w + 4.0 * px,
            lines.len() as f32 * 7.0 * px + 2.0 * px,
        ],
        [0.0, 0.0, 0.0, 0.3],
    );
    for (i, l) in lines.iter().enumerate() {
        ui.text(x, y + i as f32 * 7.0 * px, px, TEXT, l);
    }
}

/// The settings panel, centred. `cursor` (output pixels) highlights the button under it.
pub fn settings(ui: &mut Ui, s: &Status, cursor: Option<(f32, f32)>) {
    let px = 3.0 * unit(s.output.1);
    let row = 11.0 * px;
    let label_w = text_width("RESOLUTION", px) + 8.0 * px;
    let value_w = text_width("100%", px) + 4.0 * px;
    let w = label_w + 2.0 * (text_width("-", px) + 4.0 * px) + value_w + 6.0 * px + 8.0 * px;
    let h = 6.0 * row + 6.0 * px;
    let x0 = ((s.output.0 as f32 - w) * 0.5).round();
    let y0 = ((s.output.1 as f32 - h) * 0.5).round();
    ui.rect([x0, y0, w, h], [0.04, 0.05, 0.07, 0.85]);
    let x = x0 + 4.0 * px;
    let mut y = y0 + 4.0 * px;
    ui.text(x, y, px, TEXT, "SETTINGS");
    y += row;

    // label  [-] value [+]
    let stepper = |ui: &mut Ui, y: f32, label: &str, value: String, down: Action, up: Action| {
        ui.text(x, y + 2.0 * px, px, DIM, label);
        let mut bx = x + label_w;
        bx += ui.button(bx, y, px, "-", down, cursor) + 2.0 * px;
        let vw = text_width(&value, px);
        ui.text(bx + (value_w - vw) * 0.5, y + 2.0 * px, px, TEXT, &value);
        bx += value_w + 2.0 * px;
        ui.button(bx, y, px, "+", up, cursor);
    };
    stepper(
        ui,
        y,
        "SAMPLES",
        s.spp.to_string(),
        Action::SppDown,
        Action::SppUp,
    );
    y += row;
    let pct = (100.0 * s.trace.0 as f32 / s.output.0.max(1) as f32).round();
    stepper(
        ui,
        y,
        "RESOLUTION",
        format!("{pct}%"),
        Action::ScaleDown,
        Action::ScaleUp,
    );
    y += row;
    let on_off = |b: bool| if b { "ON" } else { "OFF" };
    ui.text(x, y + 2.0 * px, px, DIM, "RAW PIXELS");
    ui.button(x + label_w, y, px, on_off(s.raw), Action::ToggleRaw, cursor);
    y += row;
    ui.text(x, y + 2.0 * px, px, DIM, "TEMPORAL");
    ui.button(
        x + label_w,
        y,
        px,
        on_off(s.temporal),
        Action::ToggleTemporal,
        cursor,
    );
    y += row;
    ui.button(x, y, px, "RESUME", Action::Resume, cursor);
}

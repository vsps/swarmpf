// Software ray tracer. The spheres are the only light sources: every sphere emits, and everything
// else is lit by direct light from them plus one diffuse bounce (GI). There is no sky or ambient.
//
// Passes: trace (primary + shadow + bounce rays) -> temporal (reproject and blend) ->
//         spatial (edge-aware blur) -> present (tone map, tracers, crosshair).

struct Globals {
    prev_view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    right_s: vec4<f32>, // camera right * tan(fov/2) * aspect
    up_s: vec4<f32>,    // camera up * tan(fov/2)
    fwd: vec4<f32>,
    prev_eye: vec4<f32>,
    dims: vec4<u32>,    // trace w, trace h, output w, output h
    counts: vec4<u32>,  // spheres, groups, boxes, tracers
    params: vec4<f32>,  // frame, exposure, hit_flash, unused
    flags: vec4<u32>,   // x: target is sRGB
};

struct Sphere {
    pos_r: vec4<f32>,
    emit: vec4<f32>, // rgb radiance, a = albedo
};
struct BoxInst {
    bmin: vec4<f32>,
    bmax: vec4<f32>,
    albedo: vec4<f32>,
};
struct Group {
    bound: vec4<f32>,   // bounding sphere of one player's spheres
    range: vec4<u32>,   // first sphere, count
    light: vec4<f32>,   // x: light power, sum of luminance * r^2
};
struct Tracer {
    a: vec4<f32>,       // xyz start, w width
    b: vec4<f32>,       // xyz end, w intensity
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(2) var<storage, read> boxes: array<BoxInst>;
@group(0) @binding(3) var<storage, read> groups: array<Group>;
@group(0) @binding(4) var<storage, read_write> rad_w: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> gpos_w: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> gnrm_w: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> rad_r: array<vec4<f32>>;
@group(0) @binding(8) var<storage, read> gpos_r: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read> gnrm_r: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> gpos_prev: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> gnrm_prev: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read> acc_prev: array<vec4<f32>>;
@group(0) @binding(13) var<storage, read_write> acc_out: array<vec4<f32>>;
@group(0) @binding(14) var<storage, read> acc_in: array<vec4<f32>>;
@group(0) @binding(15) var<storage, read_write> fin_out: array<vec4<f32>>;
@group(0) @binding(16) var<storage, read> fin_in: array<vec4<f32>>;
@group(0) @binding(17) var<storage, read> tracers: array<Tracer>;
// Per sphere: running fraction of its group's light power (see renderer.rs).
@group(0) @binding(18) var<storage, read> cdf: array<f32>;

const PI: f32 = 3.14159265;
const NO_SPHERE: u32 = 0xffffffffu;
const FAR: f32 = 1.0e9;
// Emitters are far brighter than anything they light; show them compressed so they keep their colour.
const EMITTER_DISPLAY: f32 = 0.16;

// ---------------------------------------------------------------- random

var<private> rng_state: u32;

fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn rand() -> f32 {
    rng_state = pcg(rng_state);
    return f32(rng_state) * (1.0 / 4294967296.0);
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// ---------------------------------------------------------------- geometry

// Reciprocal of a ray direction, with near-zero components nudged so it stays finite.
// Computed once per ray and shared by every box test.
fn inv_dir(d: vec3<f32>) -> vec3<f32> {
    return 1.0 / select(d, vec3<f32>(1.0e-8), abs(d) < vec3<f32>(1.0e-8));
}

fn ray_box(o: vec3<f32>, inv: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec4<f32> {
    let t0 = (bmin - o) * inv;
    let t1 = (bmax - o) * inv;
    let tmn = min(t0, t1);
    let tmx = max(t0, t1);
    let te = max(max(tmn.x, tmn.y), tmn.z);
    let tx = min(min(tmx.x, tmx.y), tmx.z);
    if (te > tx || te < 0.0) {
        return vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    }
    var n = vec3<f32>(0.0);
    if (te == tmn.x) {
        n.x = -sign(inv.x);
    } else if (te == tmn.y) {
        n.y = -sign(inv.y);
    } else {
        n.z = -sign(inv.z);
    }
    return vec4<f32>(te, n);
}

fn ray_sphere_t(o: vec3<f32>, d: vec3<f32>, c: vec3<f32>, r: f32) -> f32 {
    let oc = o - c;
    let b = dot(oc, d);
    let cc = dot(oc, oc) - r * r;
    let disc = b * b - cc;
    if (disc < 0.0) {
        return -1.0;
    }
    return -b - sqrt(disc);
}

// Cheap reject: can this ray touch a player's bounding sphere before `tmax`?
fn hits_bound(o: vec3<f32>, d: vec3<f32>, bound: vec4<f32>, tmax: f32) -> bool {
    let oc = o - bound.xyz;
    let b = dot(oc, d);
    let c = dot(oc, oc) - bound.w * bound.w;
    if (c <= 0.0) {
        return true;
    }
    if (b > 0.0) {
        return false;
    }
    let disc = b * b - c;
    return disc >= 0.0 && (-b - sqrt(disc)) < tmax;
}

struct Hit {
    t: f32,
    kind: u32, // 0 miss, 1 box, 2 sphere
    idx: u32,
    n: vec3<f32>,
};

fn trace(o: vec3<f32>, d: vec3<f32>, tmax: f32) -> Hit {
    var h: Hit;
    h.t = tmax;
    h.kind = 0u;
    h.idx = 0u;
    h.n = vec3<f32>(0.0);
    let inv = inv_dir(d);
    for (var i = 0u; i < g.counts.z; i++) {
        let r = ray_box(o, inv, boxes[i].bmin.xyz, boxes[i].bmax.xyz);
        if (r.x > 1.0e-4 && r.x < h.t) {
            h.t = r.x;
            h.kind = 1u;
            h.idx = i;
            h.n = r.yzw;
        }
    }
    for (var gi = 0u; gi < g.counts.y; gi++) {
        let gr = groups[gi];
        if (!hits_bound(o, d, gr.bound, h.t)) {
            continue;
        }
        for (var k = 0u; k < gr.range.y; k++) {
            let si = gr.range.x + k;
            // Only position and radius: half the bytes of a whole `Sphere`.
            let pr = spheres[si].pos_r;
            let t = ray_sphere_t(o, d, pr.xyz, pr.w);
            if (t > 1.0e-4 && t < h.t) {
                h.t = t;
                h.kind = 2u;
                h.idx = si;
                h.n = (o + d * t - pr.xyz) / pr.w;
            }
        }
    }
    return h;
}

fn occluded(o: vec3<f32>, d: vec3<f32>, tmax: f32, ignore: u32) -> bool {
    let inv = inv_dir(d);
    for (var i = 0u; i < g.counts.z; i++) {
        let r = ray_box(o, inv, boxes[i].bmin.xyz, boxes[i].bmax.xyz);
        if (r.x > 1.0e-4 && r.x < tmax) {
            return true;
        }
    }
    for (var gi = 0u; gi < g.counts.y; gi++) {
        let gr = groups[gi];
        if (!hits_bound(o, d, gr.bound, tmax)) {
            continue;
        }
        for (var k = 0u; k < gr.range.y; k++) {
            let si = gr.range.x + k;
            if (si == ignore) {
                continue;
            }
            let pr = spheres[si].pos_r;
            let t = ray_sphere_t(o, d, pr.xyz, pr.w);
            if (t > 1.0e-4 && t < tmax) {
                return true;
            }
        }
    }
    return false;
}

// ---------------------------------------------------------------- lighting

// Direct light at a surface point from the sphere emitters, by weighted reservoir sampling over
// "entries" so the cost does not grow with every sphere in the level:
//  - a player near p (within NEAR_K of its bounding radius) enters sphere by sphere with the exact
//    weight lum * r^2 * cos / d^2, as a plain per-light loop would;
//  - a player further away enters as one entry with an estimated weight ~ power * cos / d^2 from
//    its centre, floored by a cone bound so it is never zero where any of its spheres could light
//    p (keeping the estimate unbiased). When picked, a sphere within it is chosen by resampled
//    importance sampling: RIS_M candidates drawn by power from the group's CDF, one kept in
//    proportion to its real cos / d^2 term. From afar a body's spheres all look alike, so this
//    is nearly as good as the exact loop at a fraction of the cost.
// Two independent picks, a shadow ray each to a random point on the light's disc. `ignore` is a
// sphere index that must not light or block itself.
const NEAR_K: f32 = 2.5;
// Picks with this bit set name a whole group (far player) rather than a sphere.
const GROUP_BIT: u32 = 0x80000000u;
const RIS_M: u32 = 4u;

// Index of the first sphere in [start, start + n) whose CDF value reaches u.
fn pick_by_power(start: u32, n: u32, u: f32) -> u32 {
    var lo = start;
    var hi = start + n - 1u;
    while (lo < hi) {
        let mid = (lo + hi) / 2u;
        if (cdf[mid] < u) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// cos / d^2 term of a sphere light at p (0 if it faces away or is ignored).
fn light_geo(p: vec3<f32>, n: vec3<f32>, si: u32, ignore: u32) -> f32 {
    if (si == ignore) {
        return 0.0;
    }
    let s = spheres[si];
    if (luminance(s.emit.rgb) < 1.0e-3) {
        return 0.0;
    }
    let l = s.pos_r.xyz - p;
    let d2 = dot(l, l);
    let r2 = s.pos_r.w * s.pos_r.w;
    let cosn = max(dot(n, l * inverseSqrt(d2)), 0.0);
    return cosn / max(d2, r2);
}

fn direct_light(p: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>, ignore: u32) -> vec3<f32> {
    var wsum = 0.0;
    var sel0 = NO_SPHERE;
    var sel1 = NO_SPHERE;
    var w0 = 0.0;
    var w1 = 0.0;
    for (var gi = 0u; gi < g.counts.y; gi++) {
        let gr = groups[gi];
        let power = gr.light.x;
        if (power <= 0.0) {
            continue;
        }
        let l = gr.bound.xyz - p;
        let d = max(length(l), 1.0e-6);
        let rad = gr.bound.w;
        if (d < NEAR_K * rad) {
            // Near: every sphere is its own entry, weighted exactly.
            for (var si = gr.range.x; si < gr.range.x + gr.range.y; si++) {
                let s = spheres[si];
                let w = luminance(s.emit.rgb) * s.pos_r.w * s.pos_r.w * light_geo(p, n, si, ignore);
                if (w <= 0.0) {
                    continue;
                }
                wsum += w;
                if (rand() * wsum < w) {
                    sel0 = si;
                    w0 = w;
                }
                if (rand() * wsum < w) {
                    sel1 = si;
                    w1 = w;
                }
            }
            continue;
        }
        // Far: one entry for the whole player.
        let cosb = min(max(dot(n, l) / d, 0.0) + rad / d, 1.0);
        let w = power * max(max(dot(n, l) / d, 0.0), 0.1 * cosb) / (d * d);
        if (w <= 0.0) {
            continue;
        }
        wsum += w;
        if (rand() * wsum < w) {
            sel0 = gi | GROUP_BIT;
            w0 = w;
        }
        if (rand() * wsum < w) {
            sel1 = gi | GROUP_BIT;
            w1 = w;
        }
    }
    var result = vec3<f32>(0.0);
    for (var k = 0; k < 2; k++) {
        let sel = select(sel1, sel0, k == 0);
        if (sel == NO_SPHERE) {
            continue;
        }
        let w_sel = select(w1, w0, k == 0);
        var chosen = sel;
        // Monte Carlo weight of this pick: f / pdf with the Le / lum factor applied below.
        var weight = wsum;
        if ((sel & GROUP_BIT) != 0u) {
            // RIS within the group. Source pdf = lum r^2 / power, target = lum r^2 geo, so each
            // candidate's resampling weight is power * geo.
            let gr = groups[sel & ~GROUP_BIT];
            chosen = NO_SPHERE;
            var ris_sum = 0.0;
            for (var m = 0u; m < RIS_M; m++) {
                let si = pick_by_power(gr.range.x, gr.range.y, rand());
                let wr = gr.light.x * light_geo(p, n, si, ignore);
                ris_sum += wr;
                if (wr > 0.0 && rand() * ris_sum < wr) {
                    chosen = si;
                }
            }
            if (chosen == NO_SPHERE) {
                continue;
            }
            weight = (ris_sum / f32(RIS_M)) * (wsum / w_sel);
        }
        let s = spheres[chosen];
        // Aim at a random point on the light's disc as seen from p (soft shadows).
        let to = s.pos_r.xyz - p;
        let dist = length(to);
        let axis = to / dist;
        let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(axis.x) > 0.9);
        let u = normalize(cross(axis, helper));
        let v = cross(axis, u);
        let ang = 6.2831853 * rand();
        let rad = s.pos_r.w * 0.8 * sqrt(rand());
        let target_p = s.pos_r.xyz + (u * cos(ang) + v * sin(ang)) * rad;
        let o = p + n * 1.0e-3;
        let dir_full = target_p - o;
        let dl = length(dir_full);
        let dir = dir_full / dl;
        if (dot(dir, n) <= 0.0 || occluded(o, dir, dl - s.pos_r.w * 0.9, chosen)) {
            continue;
        }
        let le = s.emit.rgb;
        result += albedo * (le / luminance(le)) * weight;
    }
    return result * 0.5;
}

fn cosine_dir(n: vec3<f32>) -> vec3<f32> {
    let r1 = rand();
    let r2 = rand();
    let phi = 6.2831853 * r1;
    let sr = sqrt(r2);
    let helper = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.x) > 0.9);
    let t = normalize(cross(n, helper));
    let b = cross(n, t);
    return normalize(t * (cos(phi) * sr) + b * (sin(phi) * sr) + n * sqrt(1.0 - r2));
}

fn albedo_of(h: Hit) -> vec3<f32> {
    if (h.kind == 1u) {
        return boxes[h.idx].albedo.rgb;
    }
    return vec3<f32>(spheres[h.idx].emit.a);
}

// Radiance leaving a diffuse surface: direct light plus one bounce of indirect light.
fn shade_diffuse(p: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    var col = direct_light(p, n, albedo, NO_SPHERE);
    let bd = cosine_dir(n);
    let bo = p + n * 1.0e-3;
    let bh = trace(bo, bd, 40.0);
    if (bh.kind != 0u) {
        let q = bo + bd * bh.t;
        var ign = NO_SPHERE;
        if (bh.kind == 2u) {
            ign = bh.idx;
        }
        // Cosine-sampled: throughput is just the albedo at p.
        col += albedo * direct_light(q, bh.n, albedo_of(bh), ign);
    }
    return col;
}

// ---------------------------------------------------------------- passes

fn camera_dir(px: vec2<f32>, size: vec2<f32>) -> vec3<f32> {
    let uv = px / size;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    return normalize(g.fwd.xyz + g.right_s.xyz * ndc.x + g.up_s.xyz * ndc.y);
}

@compute @workgroup_size(8, 8)
fn trace_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= g.dims.x || gid.y >= g.dims.y) {
        return;
    }
    let idx = gid.y * g.dims.x + gid.x;
    rng_state = pcg(idx ^ pcg(u32(g.params.x)));
    let d = camera_dir(vec2<f32>(gid.xy) + vec2<f32>(0.5), vec2<f32>(g.dims.xy));
    let h = trace(g.eye.xyz, d, FAR);
    if (h.kind == 0u) {
        rad_w[idx] = vec4<f32>(0.0);
        gpos_w[idx] = vec4<f32>(0.0, 0.0, 0.0, FAR);
        gnrm_w[idx] = vec4<f32>(0.0);
        return;
    }
    let p = g.eye.xyz + d * h.t;
    var col = vec3<f32>(0.0);
    var id = 0.0;
    if (h.kind == 2u) {
        // A light, drawn flat: one colour per sphere, no shading or light from neighbours.
        col = spheres[h.idx].emit.rgb * EMITTER_DISPLAY;
        id = f32(h.idx + 1u);
    } else {
        col = shade_diffuse(p, h.n, boxes[h.idx].albedo.rgb);
        id = -f32(h.idx + 1u);
    }
    // Clamp fireflies.
    let l = luminance(col);
    if (l > 30.0) {
        col *= 30.0 / l;
    }
    rad_w[idx] = vec4<f32>(col, 1.0);
    gpos_w[idx] = vec4<f32>(p, h.t);
    gnrm_w[idx] = vec4<f32>(h.n, id);
}

@compute @workgroup_size(8, 8)
fn temporal_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= g.dims.x || gid.y >= g.dims.y) {
        return;
    }
    let idx = gid.y * g.dims.x + gid.x;
    let cur = rad_r[idx];
    let gp = gpos_r[idx];
    let gn = gnrm_r[idx];
    // Lights move every frame and misses carry no history; only static surfaces accumulate.
    if (gn.w >= 0.0) {
        acc_out[idx] = vec4<f32>(cur.rgb, 1.0);
        return;
    }
    var out = vec4<f32>(cur.rgb, 1.0);
    let pc = g.prev_view_proj * vec4<f32>(gp.xyz, 1.0);
    if (pc.w > 0.0) {
        let ndc = pc.xy / pc.w;
        let px = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * vec2<f32>(g.dims.xy);
        if (px.x >= 0.0 && px.y >= 0.0 && px.x < f32(g.dims.x) && px.y < f32(g.dims.y)) {
            let pi = u32(px.y) * g.dims.x + u32(px.x);
            let pn = gnrm_prev[pi];
            let pt = gpos_prev[pi].w;
            let expect = length(gp.xyz - g.prev_eye.xyz);
            if (pn.w == gn.w && dot(pn.xyz, gn.xyz) > 0.9 && abs(pt - expect) < 0.05 * expect + 0.02) {
                let hist = acc_prev[pi];
                // Short history: every light moves, so long accumulation would smear.
                let count = min(hist.w + 1.0, 6.0);
                out = vec4<f32>(mix(hist.rgb, cur.rgb, 1.0 / count), count);
            }
        }
    }
    acc_out[idx] = out;
}

// The 7x7 filter reads each neighbour's normal, position and colour. An 8x8 workgroup shares
// one 14x14 tile of them in workgroup memory, loaded once, instead of every thread reading its
// 49 taps from storage.
const TILE: u32 = 14u;
var<workgroup> tile_n: array<vec4<f32>, 196>;
var<workgroup> tile_p: array<vec4<f32>, 196>;
var<workgroup> tile_c: array<vec4<f32>, 196>;

@compute @workgroup_size(8, 8)
fn spatial_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let dim = vec2<i32>(g.dims.xy);
    let origin = vec2<i32>(wid.xy * 8u) - vec2<i32>(3);
    for (var k = lid.y * 8u + lid.x; k < TILE * TILE; k += 64u) {
        let q = origin + vec2<i32>(i32(k % TILE), i32(k / TILE));
        if (q.x < 0 || q.y < 0 || q.x >= dim.x || q.y >= dim.y) {
            // Id 0 never matches a surface (box ids are negative), so it is skipped like before.
            tile_n[k] = vec4<f32>(0.0);
        } else {
            let qi = u32(q.y * dim.x + q.x);
            tile_n[k] = gnrm_r[qi];
            tile_p[k] = gpos_r[qi];
            tile_c[k] = acc_in[qi];
        }
    }
    workgroupBarrier();
    if (gid.x >= g.dims.x || gid.y >= g.dims.y) {
        return;
    }
    let idx = gid.y * g.dims.x + gid.x;
    let centre = (lid.y + 3u) * TILE + lid.x + 3u;
    let gn = tile_n[centre];
    let gp = tile_p[centre];
    if (gn.w >= 0.0) {
        fin_out[idx] = tile_c[centre];
        return;
    }
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    let inv_plane = 1.0 / (0.02 * gp.w + 0.01);
    for (var dy = 0u; dy < 7u; dy++) {
        for (var dx = 0u; dx < 7u; dx++) {
            let k = (lid.y + dy) * TILE + lid.x + dx;
            let qn = tile_n[k];
            if (qn.w != gn.w) {
                continue;
            }
            let plane = abs(dot(gn.xyz, tile_p[k].xyz - gp.xyz));
            // pow(cos, 32) by repeated squaring.
            var c = max(dot(gn.xyz, qn.xyz), 0.0);
            c *= c;
            c *= c;
            c *= c;
            c *= c;
            c *= c;
            let ox = f32(dx) - 3.0;
            let oy = f32(dy) - 3.0;
            let wgt = c * exp(-plane * inv_plane - (ox * ox + oy * oy) / 8.0);
            sum += tile_c[k].rgb * wgt;
            wsum += wgt;
        }
    }
    fin_out[idx] = vec4<f32>(sum / max(wsum, 1.0e-4), 1.0);
}

// ---------------------------------------------------------------- present

@vertex
fn vs_present(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn fetch_fin(p: vec2<i32>) -> vec3<f32> {
    let dim = vec2<i32>(g.dims.xy);
    let c = clamp(p, vec2<i32>(0), dim - vec2<i32>(1));
    return fin_in[u32(c.y * dim.x + c.x)].rgb;
}

// "123 FPS" in the top-left corner, from a 3x5 bitmap font (bit 14 = top-left pixel), on a
// darkened backdrop. `sc` scales with the output height.
fn draw_fps(p: vec2<f32>, sc: f32, col: vec3<f32>) -> vec3<f32> {
    // 0-9, then F and P; S reuses 5.
    var font = array<u32, 12>(
        31599u, 11415u, 29671u, 29647u, 23497u, 31183u, 31215u, 29257u, 31727u, 31695u, 31140u, 31716u,
    );
    let px = 3.0 * sc;
    let cell = floor((p - vec2<f32>(10.0 * sc)) / px);
    // Seven characters of four columns (three lit plus a gap), five rows, one cell of padding.
    if (cell.x < -1.0 || cell.y < -1.0 || cell.x > 28.0 || cell.y > 5.0) {
        return col;
    }
    var out = col * 0.25;
    if (cell.x < 0.0 || cell.y < 0.0 || cell.y > 4.0 || cell.x > 27.0) {
        return out;
    }
    let ch = u32(cell.x) / 4u;
    let cx = u32(cell.x) % 4u;
    if (cx == 3u || ch == 3u) {
        return out;
    }
    let fps = min(u32(round(g.params.w)), 999u);
    var glyph = 0u;
    if (ch == 0u) {
        if (fps < 100u) {
            return out;
        }
        glyph = fps / 100u;
    } else if (ch == 1u) {
        if (fps < 10u) {
            return out;
        }
        glyph = (fps / 10u) % 10u;
    } else if (ch == 2u) {
        glyph = fps % 10u;
    } else if (ch == 4u) {
        glyph = 10u;
    } else if (ch == 5u) {
        glyph = 11u;
    } else {
        glyph = 5u;
    }
    let bit = 14u - (u32(cell.y) * 3u + cx);
    if (((font[glyph] >> bit) & 1u) == 1u) {
        out = vec3<f32>(1.0);
    }
    return out;
}

@fragment
fn fs_present(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
    let out_size = vec2<f32>(g.dims.zw);
    let uv = fc.xy / out_size;
    let tsize = vec2<f32>(g.dims.xy);

    // Bilinear upscale of the traced image.
    let tp = uv * tsize - 0.5;
    let base = floor(tp);
    let f = tp - base;
    let bi = vec2<i32>(base);
    let c00 = fetch_fin(bi);
    let c10 = fetch_fin(bi + vec2<i32>(1, 0));
    let c01 = fetch_fin(bi + vec2<i32>(0, 1));
    let c11 = fetch_fin(bi + vec2<i32>(1, 1));
    var col = mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y) * g.params.y;
    col = aces(col);

    // Tracers: glowing segments, depth-tested against the traced scene, drawn over the tonemapped image.
    let d = camera_dir(fc.xy, out_size);
    let ti = vec2<i32>(clamp(uv * tsize, vec2<f32>(0.0), tsize - vec2<f32>(1.0)));
    let scene_t = gpos_r[u32(ti.y) * g.dims.x + u32(ti.x)].w;
    for (var i = 0u; i < g.counts.w; i++) {
        let tr = tracers[i];
        let a = tr.a.xyz;
        let e = tr.b.xyz - a;
        let w0 = g.eye.xyz - a;
        let ee = dot(e, e);
        let bb = dot(d, e);
        let dw = dot(d, w0);
        let ew = dot(e, w0);
        let den = ee - bb * bb;
        var u = 0.0;
        if (den > 1.0e-6) {
            u = clamp((ew - bb * dw) / den, 0.0, 1.0);
        }
        let s = max(bb * u - dw, 0.0);
        if (s > scene_t + 0.05) {
            continue;
        }
        let dist = length(w0 + d * s - e * u);
        let width = max(tr.a.w, s * 0.0025);
        let glow = exp(-(dist * dist) / (width * width)) * mix(0.15, 1.0, u) * tr.b.w;
        col += tr.color.rgb * glow;
    }
    col = min(col, vec3<f32>(1.0));

    // Crosshair (scales with output height; outline for contrast; red briefly after a hit).
    let sc = out_size.y / 720.0;
    let c = abs(fc.xy - out_size * 0.5) / sc;
    let gap = 5.0;
    let len = 12.0;
    let arm = (c.x >= gap && c.x <= len && c.y <= 1.0) || (c.y >= gap && c.y <= len && c.x <= 1.0);
    let arm_o = (c.x >= gap - 1.0 && c.x <= len + 1.0 && c.y <= 2.0) || (c.y >= gap - 1.0 && c.y <= len + 1.0 && c.x <= 2.0);
    if (arm) {
        col = mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 0.15, 0.1), clamp(g.params.z * 6.0, 0.0, 1.0));
    } else if (arm_o) {
        col = vec3<f32>(0.0);
    }

    if (g.params.w > 0.0) {
        col = draw_fps(fc.xy, sc, col);
    }

    // Tiny dither hides banding in the dark gradients.
    let n = f32(pcg(u32(fc.x) + u32(fc.y) * 8192u)) * (1.0 / 4294967296.0) - 0.5;
    col += vec3<f32>(n / 255.0);
    if (g.flags.x == 0u) {
        col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    }
    return vec4<f32>(col, 1.0);
}

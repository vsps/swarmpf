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

const PI: f32 = 3.14159265;
const NO_SPHERE: u32 = 0xffffffffu;
const FAR: f32 = 1.0e9;

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

fn ray_box(o: vec3<f32>, d: vec3<f32>, bmin: vec3<f32>, bmax: vec3<f32>) -> vec4<f32> {
    let safe = select(d, vec3<f32>(1.0e-8), abs(d) < vec3<f32>(1.0e-8));
    let inv = 1.0 / safe;
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
        n.x = -sign(safe.x);
    } else if (te == tmn.y) {
        n.y = -sign(safe.y);
    } else {
        n.z = -sign(safe.z);
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
    for (var i = 0u; i < g.counts.z; i++) {
        let r = ray_box(o, d, boxes[i].bmin.xyz, boxes[i].bmax.xyz);
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
            let s = spheres[si];
            let t = ray_sphere_t(o, d, s.pos_r.xyz, s.pos_r.w);
            if (t > 1.0e-4 && t < h.t) {
                h.t = t;
                h.kind = 2u;
                h.idx = si;
                h.n = (o + d * t - s.pos_r.xyz) / s.pos_r.w;
            }
        }
    }
    return h;
}

fn occluded(o: vec3<f32>, d: vec3<f32>, tmax: f32, ignore: u32) -> bool {
    for (var i = 0u; i < g.counts.z; i++) {
        let r = ray_box(o, d, boxes[i].bmin.xyz, boxes[i].bmax.xyz);
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
            let s = spheres[si];
            let t = ray_sphere_t(o, d, s.pos_r.xyz, s.pos_r.w);
            if (t > 1.0e-4 && t < tmax) {
                return true;
            }
        }
    }
    return false;
}

// ---------------------------------------------------------------- lighting

// Direct light at a surface point from the sphere emitters. Lights are picked by weighted
// reservoir sampling with weight ~ luminance * r^2 * cos / d^2, then a shadow ray is traced to
// a random point on the light. `ignore` is a sphere index that must not light or block itself.
fn direct_light(p: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>, ignore: u32) -> vec3<f32> {
    var wsum = 0.0;
    var sel0 = NO_SPHERE;
    var sel1 = NO_SPHERE;
    var w0 = 0.0;
    var w1 = 0.0;
    for (var i = 0u; i < g.counts.x; i++) {
        let s = spheres[i];
        let lum = luminance(s.emit.rgb);
        if (lum < 1.0e-3 || i == ignore) {
            continue;
        }
        let l = s.pos_r.xyz - p;
        let d2 = dot(l, l);
        let r2 = s.pos_r.w * s.pos_r.w;
        let cosn = max(dot(n, l * inverseSqrt(d2)), 0.0);
        let w = lum * r2 * cosn / max(d2, r2);
        if (w <= 0.0) {
            continue;
        }
        wsum += w;
        if (rand() * wsum < w) {
            sel0 = i;
            w0 = w;
        }
        if (rand() * wsum < w) {
            sel1 = i;
            w1 = w;
        }
    }
    var result = vec3<f32>(0.0);
    for (var k = 0; k < 2; k++) {
        let sel = select(sel1, sel0, k == 0);
        if (sel == NO_SPHERE) {
            continue;
        }
        let s = spheres[sel];
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
        if (dot(dir, n) <= 0.0 || occluded(o, dir, dl - s.pos_r.w * 0.9, sel)) {
            continue;
        }
        let le = s.emit.rgb;
        result += albedo * (le / luminance(le)) * wsum;
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
        // A light: its own emission (shaded so it reads as a ball) plus light from neighbours.
        let s = spheres[h.idx];
        let ndv = max(dot(h.n, -d), 0.0);
        col = s.emit.rgb * (0.3 + 0.7 * ndv);
        col += direct_light(p, h.n, vec3<f32>(s.emit.a), h.idx);
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
                let count = min(hist.w + 1.0, 12.0);
                out = vec4<f32>(mix(hist.rgb, cur.rgb, 1.0 / count), count);
            }
        }
    }
    acc_out[idx] = out;
}

@compute @workgroup_size(8, 8)
fn spatial_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= g.dims.x || gid.y >= g.dims.y) {
        return;
    }
    let idx = gid.y * g.dims.x + gid.x;
    let gn = gnrm_r[idx];
    let gp = gpos_r[idx];
    if (gn.w >= 0.0) {
        fin_out[idx] = acc_in[idx];
        return;
    }
    var sum = vec3<f32>(0.0);
    var wsum = 0.0;
    let ci = vec2<i32>(gid.xy);
    let dim = vec2<i32>(g.dims.xy);
    for (var dy = -3; dy <= 3; dy++) {
        for (var dx = -3; dx <= 3; dx++) {
            let q = ci + vec2<i32>(dx, dy);
            if (q.x < 0 || q.y < 0 || q.x >= dim.x || q.y >= dim.y) {
                continue;
            }
            let qi = u32(q.y * dim.x + q.x);
            let qn = gnrm_r[qi];
            if (qn.w != gn.w) {
                continue;
            }
            let plane = abs(dot(gn.xyz, gpos_r[qi].xyz - gp.xyz));
            let wgt = pow(max(dot(gn.xyz, qn.xyz), 0.0), 32.0)
                * exp(-plane / (0.02 * gp.w + 0.01))
                * exp(-f32(dx * dx + dy * dy) / 8.0);
            sum += acc_in[qi].rgb * wgt;
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

    // Tiny dither hides banding in the dark gradients.
    let n = f32(pcg(u32(fc.x) + u32(fc.y) * 8192u)) * (1.0 / 4294967296.0) - 0.5;
    col += vec3<f32>(n / 255.0);
    if (g.flags.x == 0u) {
        col = pow(max(col, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
    }
    return vec4<f32>(col, 1.0);
}

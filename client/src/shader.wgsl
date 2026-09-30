struct Globals {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    light: vec4<f32>, // xyz = direction towards the light
};

struct Sphere {
    pos_r: vec4<f32>,
    color: vec4<f32>,
};

struct BoxInst {
    bmin: vec4<f32>,
    bmax: vec4<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(0) @binding(1) var<storage, read> spheres: array<Sphere>;
@group(0) @binding(2) var<storage, read> boxes: array<BoxInst>;

fn shade(base: vec3<f32>, n: vec3<f32>, emissive: f32) -> vec3<f32> {
    let hemi = mix(vec3<f32>(0.16, 0.15, 0.18), vec3<f32>(0.42, 0.46, 0.55), n.y * 0.5 + 0.5);
    let diff = max(dot(n, g.light.xyz), 0.0);
    let lit = base * (hemi + vec3<f32>(0.85, 0.82, 0.75) * diff);
    return mix(lit, base, emissive);
}

// --- spheres: camera-facing impostors, exact silhouette and depth in the fragment shader ---

struct SphereOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) center: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
};

@vertex
fn vs_sphere(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SphereOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let s = spheres[ii];
    let c = corners[vi];
    let to_c = s.pos_r.xyz - g.eye.xyz;
    let cos_t = max(dot(normalize(to_c), g.forward.xyz), 0.2);
    // Off-axis spheres project to ellipses, so pad the quad by 1/cos^2.
    let pad = 1.0 / (cos_t * cos_t);
    let w = s.pos_r.xyz + (g.right.xyz * c.x + g.up.xyz * c.y) * s.pos_r.w * pad;
    var o: SphereOut;
    o.pos = g.view_proj * vec4<f32>(w, 1.0);
    o.world = w;
    o.center = s.pos_r.xyz;
    o.radius = s.pos_r.w;
    o.color = s.color;
    return o;
}

struct SphereFrag {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_sphere(i: SphereOut) -> SphereFrag {
    let d = normalize(i.world - g.eye.xyz);
    let oc = g.eye.xyz - i.center;
    let b = dot(oc, d);
    let cc = dot(oc, oc) - i.radius * i.radius;
    let disc = b * b - cc;
    if (disc < 0.0) {
        discard;
    }
    let t = -b - sqrt(disc);
    let p = g.eye.xyz + d * t;
    let n = (p - i.center) / i.radius;
    let clip = g.view_proj * vec4<f32>(p, 1.0);
    var o: SphereFrag;
    o.color = vec4<f32>(shade(i.color.rgb, n, i.color.a), 1.0);
    o.depth = clip.z / clip.w;
    return o;
}

// --- level boxes ---

struct BoxOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
};

@vertex
fn vs_box(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> BoxOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    var ns = array<vec3<f32>, 6>(
        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(-1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, -1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, -1.0),
    );
    var us = array<vec3<f32>, 6>(
        vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
    );
    var vs = array<vec3<f32>, 6>(
        vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.0, 0.0, 1.0),
        vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 1.0, 0.0),
    );
    let bx = boxes[ii];
    let face = vi / 6u;
    let c = corners[vi % 6u];
    let p01 = 0.5 + 0.5 * (ns[face] + us[face] * c.x + vs[face] * c.y);
    let w = mix(bx.bmin.xyz, bx.bmax.xyz, p01);
    var o: BoxOut;
    o.pos = g.view_proj * vec4<f32>(w, 1.0);
    o.normal = ns[face];
    o.color = bx.color.rgb;
    return o;
}

@fragment
fn fs_box(i: BoxOut) -> @location(0) vec4<f32> {
    return vec4<f32>(shade(i.color, normalize(i.normal), 0.0), 1.0);
}

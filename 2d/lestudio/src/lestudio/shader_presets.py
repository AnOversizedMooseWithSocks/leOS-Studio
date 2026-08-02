"""A small library of ready-to-run Shadertoy-style presets.

LICENSING
---------
Shadertoy's DEFAULT licence is CC BY-NC-SA 3.0 -- non-commercial and
share-alike -- so shaders copied from the site cannot be bundled here. Nothing
in this file is copied from Shadertoy.

Two ingredients, both safe to redistribute:

* ``SNOISE3`` is 3-D simplex noise from https://github.com/stegu/webgl-noise,
  Copyright (C) 2011 Ashima Arts and Stefan Gustavson, released under the MIT
  licence. The copyright notice travels with the code, as MIT requires.
* Every effect below (clouds, water, fire, smoke, embers, glow, depth of field,
  grass, terrain, foliage) is original work written for leStudio, using that
  noise as its only external ingredient.

Each preset is self-contained: it carries the noise it needs, defines
``mainImage``, and targets GLSL ES 3.00 (WebGL2), which is what the Shadertoy
node compiles. Presets whose name starts with "post" read the wired image from
``iChannel0``; the rest are generators and ignore their inputs.
"""

# --- MIT-licensed noise, kept verbatim with its notice -----------------------
SNOISE3 = """
// Simplex 3D noise -- Copyright (C) 2011 Ashima Arts / Stefan Gustavson.
// MIT licence. Source: https://github.com/stegu/webgl-noise
vec3 mod289(vec3 x){ return x - floor(x*(1.0/289.0))*289.0; }
vec4 mod289(vec4 x){ return x - floor(x*(1.0/289.0))*289.0; }
vec4 permute(vec4 x){ return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r){ return 1.79284291400159 - 0.85373472095314*r; }
float snoise(vec3 v){
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod289(i);
  vec4 p = permute(permute(permute(
             i.z + vec4(0.0, i1.z, i2.z, 1.0))
           + i.y + vec4(0.0, i1.y, i2.y, 1.0))
           + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}
float fbm(vec3 p, int oct){
  float s = 0.0, a = 0.5;
  for (int i = 0; i < 8; i++){
    if (i >= oct) break;
    s += a * snoise(p);
    p *= 2.02; a *= 0.5;
  }
  return s;
}
"""

_CLOUDS = """
// Layered sky: fbm cloud mass lit from a low sun, over a graded sky.
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2 uv = fragCoord / iResolution.xy;
    float t  = iTime * 0.02;
    float h  = fbm(vec3(uv * vec2(3.0, 1.6) + vec2(t, 0.0), t), 6);
    float m  = smoothstep(0.05, 0.65, h + 0.25 - uv.y * 0.35);
    float lit = smoothstep(-0.2, 0.6,
                fbm(vec3(uv * vec2(3.0, 1.6) + vec2(t + 0.03, 0.02), t), 4));
    vec3 sky   = mix(vec3(0.42, 0.60, 0.86), vec3(0.16, 0.32, 0.62), uv.y);
    vec3 cloud = mix(vec3(0.42, 0.44, 0.52), vec3(1.0, 0.97, 0.94), lit);
    fragColor = vec4(mix(sky, cloud, m), 1.0);
}
"""

_WATER = """
// Open water: crossed wave trains, a specular glint and depth-graded colour.
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2 uv = fragCoord / iResolution.xy;
    float t = iTime * 0.35;
    vec2 p = uv * vec2(8.0, 18.0);
    float w = 0.0;
    w += sin(p.x * 1.1 + t * 1.7) * 0.5;
    w += sin(p.x * 0.4 - p.y * 0.7 + t * 1.1) * 0.35;
    w += fbm(vec3(p * 0.5, t * 0.5), 5) * 0.8;
    float slope = w - fbm(vec3(p * 0.5 + vec2(0.06, 0.0), t * 0.5), 5) * 0.8;
    vec3 deep    = vec3(0.02, 0.10, 0.20);
    vec3 shallow = vec3(0.10, 0.42, 0.52);
    vec3 col = mix(deep, shallow, smoothstep(-0.6, 0.9, w) * (0.35 + uv.y * 0.8));
    col += vec3(0.85, 0.93, 1.0) * pow(max(slope, 0.0), 3.0) * 1.6;   // glint
    fragColor = vec4(col, 1.0);
}
"""

_FIRE = """
// Flame: noise advected upward, shaped by a vertical falloff, blackbody ramp.
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2 uv = fragCoord / iResolution.xy;
    vec2 p = (uv - vec2(0.5, 0.0)) * vec2(2.2, 1.0);
    float t = iTime * 1.6;
    float n = fbm(vec3(p * vec2(2.6, 1.7) - vec2(0.0, t), t * 0.35), 5);
    float body = 1.0 - smoothstep(0.0, 0.95, uv.y);
    float f = max(0.0, body * (0.85 + n * 0.65) - abs(p.x) * 1.35);
    vec3 col = vec3(0.0);
    col += vec3(1.00, 0.28, 0.06) * smoothstep(0.03, 0.45, f);
    col += vec3(1.00, 0.72, 0.20) * smoothstep(0.30, 0.75, f);
    col += vec3(1.00, 0.98, 0.88) * smoothstep(0.62, 1.00, f);
    fragColor = vec4(col, 1.0);
}
"""

_SMOKE = """
// Smoke column: domain-warped fbm rising and spreading with height.
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2 uv = fragCoord / iResolution.xy;
    float t = iTime * 0.30;
    vec2 p = (uv - vec2(0.5, 0.0));
    p.x *= 1.0 + uv.y * 1.6;                       // plume widens as it rises
    vec3 q = vec3(p * 3.0 - vec2(0.0, t * 1.5), t);
    float warp = fbm(q * 0.6, 4);
    float d = fbm(q + warp * 0.9, 6);
    float body = smoothstep(0.9, 0.0, abs(p.x) * 2.0) * (1.0 - uv.y * 0.85);
    float a = clamp(body * (0.55 + d * 0.8), 0.0, 1.0);
    vec3 col = mix(vec3(0.05), vec3(0.72, 0.72, 0.75), 0.35 + d * 0.4);
    fragColor = vec4(col * a, a);                  // premultiplied: composites
}
"""

_EMBERS = """
// Embers: a sparse grid of drifting cells, each a glowing point that fades.
float cell(vec2 id){ return fract(sin(dot(id, vec2(41.7, 289.1))) * 43758.5453); }
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2 uv = fragCoord / iResolution.xy;
    float t = iTime;
    vec3 col = vec3(0.0);
    float a = 0.0;
    for (int k = 0; k < 3; k++){
        float fk = float(k);
        vec2 g = uv * vec2(9.0 + fk * 4.0, 5.0 + fk * 2.0);
        g.y -= t * (0.25 + fk * 0.18);
        vec2 id = floor(g);
        vec2 f  = fract(g) - 0.5;
        float r = cell(id + fk * 17.0);
        if (r > 0.82){
            f.x += (r - 0.5) * 0.7;
            f.x += sin(t * (1.0 + r) + r * 6.28) * 0.12;   // drift
            float d = length(f * vec2(1.0, 0.7));
            float life = fract(r * 3.0 + t * 0.15);
            float glow = smoothstep(0.42, 0.0, d) * (1.0 - life);
            col += mix(vec3(1.0, 0.35, 0.08), vec3(1.0, 0.85, 0.45), 1.0 - life) * glow;
            a = max(a, glow);
        }
    }
    fragColor = vec4(col, a);                      // premultiplied
}
"""

_TERRAIN = """
// Ridged terrain: an fbm heightfield shaded by its own slope, with haze.
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2 uv = fragCoord / iResolution.xy;
    vec2 p = uv * vec2(3.2, 1.8) + vec2(iTime * 0.02, 0.0);
    float h = 0.0, a = 0.5;
    vec3 q = vec3(p, 0.0);
    for (int i = 0; i < 6; i++){
        h += a * (1.0 - abs(snoise(q)));          // ridged
        q *= 2.03; a *= 0.5;
    }
    h *= 0.55;
    float hx = h - (1.0 - abs(snoise(vec3(p + vec2(0.02, 0.0), 0.0)))) * 0.55;
    float slope = clamp(abs(hx) * 12.0, 0.0, 1.0);
    vec3 rock  = mix(vec3(0.30, 0.26, 0.23), vec3(0.52, 0.48, 0.44), slope);
    vec3 grass = vec3(0.22, 0.34, 0.16);
    vec3 snow  = vec3(0.92, 0.94, 0.97);
    vec3 col = mix(grass, rock, smoothstep(0.35, 0.6, h));
    col = mix(col, snow, smoothstep(0.72, 0.88, h));
    col = mix(col, vec3(0.68, 0.76, 0.86), smoothstep(0.0, 1.0, uv.y) * 0.35);  // haze
    fragColor = vec4(col, 1.0);
}
"""

_GRASS = """
// Grass: many thin blades per column, each swaying, depth-sorted by row.
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = mix(vec3(0.10, 0.16, 0.09), vec3(0.30, 0.42, 0.20), uv.y);
    for (int row = 0; row < 4; row++){
        float fr = float(row);
        float scale = 60.0 + fr * 40.0;
        float base = 0.12 + fr * 0.16;
        vec2 g = vec2(uv.x * scale, uv.y);
        float id = floor(g.x);
        float r  = fract(sin(id * 78.233 + fr * 13.7) * 43758.5453);
        float sway = sin(iTime * (0.8 + r * 0.5) + id * 0.7) * (0.012 + fr * 0.006);
        float x = fract(g.x) - 0.5 + sway * (uv.y - base) * 8.0;
        float top = base + 0.10 + r * 0.16;
        float width = (1.0 - smoothstep(base, top, uv.y)) * 0.28;
        if (uv.y > base && uv.y < top && abs(x) < width){
            vec3 blade = mix(vec3(0.16, 0.30, 0.10), vec3(0.45, 0.62, 0.24), r);
            col = mix(col, blade * (0.7 + 0.5 * (1.0 - fr * 0.2)), 0.92);
        }
    }
    fragColor = vec4(col, 1.0);
}
"""

_FOLIAGE = """
// Foliage canopy: clustered leaf blobs with dappled light through gaps.
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2 uv = fragCoord / iResolution.xy;
    float t = iTime * 0.25;
    float clump = fbm(vec3(uv * 5.0 + vec2(sin(t) * 0.05, 0.0), t * 0.4), 5);
    float leaf  = fbm(vec3(uv * 22.0, t), 3);
    float mass  = smoothstep(-0.15, 0.45, clump + leaf * 0.25);
    float light = smoothstep(0.2, 0.9, leaf) * (1.0 - mass);
    vec3 dark  = vec3(0.05, 0.12, 0.05);
    vec3 mid   = vec3(0.16, 0.34, 0.13);
    vec3 lit   = vec3(0.55, 0.74, 0.30);
    vec3 col = mix(dark, mid, mass);
    col = mix(col, lit, smoothstep(0.35, 0.95, leaf) * mass);
    col += vec3(0.95, 0.92, 0.60) * light * 0.55;      // sun through the gaps
    fragColor = vec4(col, 1.0);
}
"""

_POST_GLOW = """
// POST: bloom. Wire an image into iChannel0. Bright areas bleed outward.
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2 uv = fragCoord / iResolution.xy;
    vec4 src = texture(iChannel0, uv);
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int i = -6; i <= 6; i++){
        for (int j = -6; j <= 6; j++){
            vec2 o = vec2(float(i), float(j));
            float w = exp(-dot(o, o) / 18.0);
            vec3 s = texture(iChannel0, uv + o * 2.5 / iResolution.xy).rgb;
            float lum = dot(s, vec3(0.2126, 0.7152, 0.0722));
            sum += s * smoothstep(0.55, 1.0, lum) * w;   // only bright pixels bloom
            wsum += w;
        }
    }
    vec3 glow = sum / max(wsum, 1e-4);
    fragColor = vec4(src.rgb + glow * 1.4, src.a);
}
"""

_POST_DOF = """
// POST: depth of field. Wire an image into iChannel0. `focus` is the sharp
// band's height in frame; everything above and below blurs progressively.
void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2 uv = fragCoord / iResolution.xy;
    float focus = 0.42;
    float coc = clamp(abs(uv.y - focus) * 3.2, 0.0, 1.0);   // circle of confusion
    float r = coc * 9.0;
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int i = 0; i < 24; i++){
        float a = float(i) * 2.399963;                      // golden-angle bokeh
        float rr = sqrt(float(i) / 24.0) * r;
        vec2 o = vec2(cos(a), sin(a)) * rr;
        vec3 s = texture(iChannel0, uv + o / iResolution.xy).rgb;
        float w = 1.0 + dot(s, vec3(0.3333)) * 0.8;         // bright bokeh bias
        sum += s * w; wsum += w;
    }
    fragColor = vec4(sum / max(wsum, 1e-4), texture(iChannel0, uv).a);
}
"""

PRESETS = {
    "Clouds — sky": _CLOUDS,
    "Water — open sea": _WATER,
    "Fire — flame": _FIRE,
    "Smoke — plume": _SMOKE,
    "Embers — sparks": _EMBERS,
    "Terrain — ridges": _TERRAIN,
    "Grass — field": _GRASS,
    "Foliage — canopy": _FOLIAGE,
    "post: Glow — bloom": _POST_GLOW,
    "post: Depth of field": _POST_DOF,
}

ATTRIBUTION = (
    "Effects are original work for leStudio. The simplex noise they build on is "
    "Copyright (C) 2011 Ashima Arts and Stefan Gustavson, MIT licence "
    "(https://github.com/stegu/webgl-noise). Nothing here is copied from "
    "Shadertoy, whose default licence (CC BY-NC-SA 3.0) forbids redistribution "
    "in a project like this."
)


def preset_source(name):
    """Full runnable GLSL for a preset: the noise it needs plus the effect."""
    body = PRESETS[name]
    needs_noise = ("snoise(" in body) or ("fbm(" in body)
    return ((SNOISE3 + body) if needs_noise else body).strip() + "\n"


def catalogue():
    """[{name, source, post}] for the UI picker."""
    return [{"name": n, "source": preset_source(n),
             "post": n.startswith("post:")}
            for n in PRESETS]

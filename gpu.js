/*
 * gpu.js — WebGPU compute backend for the paper model in sim.js.
 *
 * Same algorithm (small-substeps XPBD), executed on the GPU:
 *  - Gauss-Seidel constraint solves run as graph-colored dispatches
 *    (8 distance groups, 16 hinge groups — structured coloring on the grid).
 *  - Self-collision uses a 2D uniform grid hash (atomic linked lists) with
 *    per-side half-corrections and layer friction.
 *  - Actuator colliders, desk friction, broadside drag, and rest-angle
 *    plasticity (pressure-gated) all ported 1:1 from sim.js, in f32.
 *
 * The CPU still evaluates scenario scripts (one record per substep) and reads
 * positions + crease state back each frame, so the Canvas renderer and UI are
 * unchanged. Behavior matches the CPU backend up to f32 and constraint-order
 * differences (colored vs sequential).
 */
(function (root, factory) {
  if (typeof module !== 'undefined' && module.exports) module.exports = factory();
  else root.GPUSIM = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const WG = 64;
  const MAXSUB = 256;
  const SUB_STRIDE = 768;        // per-substep script record (bytes, 256-aligned)
  const MAXGRAB = 8, MAXCOL = 16;
  const GRID_DIM = 128, GRID_ORG = -1.5, GRID_SPAN = 4.0;

  function available() {
    return typeof navigator !== 'undefined' && !!navigator.gpu;
  }

  // ------------------------------------------------------------- coloring
  // classify edges into conflict-free groups: 8 structural + up to 12 rope
  // groups (per stride level, direction, and chain parity)
  const ROPE_STRIDES = [4, 16, 64];
  function colorEdges(paper) {
    const N = paper.N;
    const groups = Array.from({ length: 8 + ROPE_STRIDES.length * 4 }, () => []);
    for (const e of paper.edges) {
      const ia = Math.floor(e.a / N), ja = e.a % N;
      const ib = Math.floor(e.b / N), jb = e.b % N;
      const di = ib - ia, dj = jb - ja;
      let g;
      if (e.rope) {
        const L = Math.max(Math.abs(di), Math.abs(dj));
        const li = ROPE_STRIDES.indexOf(L);
        const isCol = dj === 0;
        const par = isCol ? Math.floor(ia / L) % 2 : Math.floor(ja / L) % 2;
        g = 8 + li * 4 + (isCol ? 2 : 0) + par;
      } else if (di === 0) g = 0 + (ja % 2);       // horizontal
      else if (dj === 0) g = 2 + (ia % 2);         // vertical
      else if (dj === -1) g = 4 + (ia % 2);        // BD diagonal (a = v10, b = v01)
      else g = 6 + (ia % 2);                       // shear diagonal
      groups[g].push(e);
    }
    return groups.filter(g => g.length > 0 || true); // keep indices stable
  }
  function colorHinges(paper) {
    const N = paper.N;
    const groups = Array.from({ length: 16 }, () => []);
    for (const h of paper.hinges) {
      const i0 = Math.floor(h.e0 / N), j0 = h.e0 % N;
      const i1 = Math.floor(h.e1 / N), j1 = h.e1 % N;
      let g;
      if (i1 - i0 === 1 && j1 - j0 === -1) {
        g = (i0 % 2) * 2 + (j1 % 2);                       // cell diagonal: 0..3
      } else if (j1 === j0) {
        g = 4 + (i0 % 2) * 3 + (j0 % 3);                   // vertical edge: 4..9
      } else {
        g = 10 + (i0 % 3) * 2 + (j0 % 2);                  // horizontal edge: 10..15
      }
      groups[g].push(h);
    }
    return groups;
  }
  function assertColoring(groups, touchOf, np) {
    const seen = new Uint8Array(np);
    for (const g of groups) {
      seen.fill(0);
      for (const c of g) {
        for (const p of touchOf(c)) {
          if (seen[p]) throw new Error('gpu coloring conflict');
          seen[p] = 1;
        }
      }
    }
  }

  // --------------------------------------------------------------- shaders
  const COMMON = /* wgsl */`
struct Params {
  np: u32, ne: u32, nh: u32, ntri: u32,
  n: u32, pinIdx: u32, gridDim: u32, pad0: u32,
  dt: f32, gravity: f32, dragK: f32, settleK: f32,
  nDragK: f32, alphaBendT: f32, alphaBendSoftT: f32, bendDamp: f32,
  muS: f32, muK: f32, muPaper: f32, selfR: f32,
  yieldA: f32, plasticRate: f32, maxT0: f32, creaseMark: f32,
  cell: f32, gridOrg: f32, pad1: f32, pad2: f32,
};
struct Sub {
  counts: vec4<u32>,             // nGrab, nCol, pinA, _
  grabs: array<vec4<f32>, ${MAXGRAB}>,   // xyz target, w = idx*2 + soft
  cols: array<vec4<f32>, ${MAXCOL * 2}>, // pairs: a.xyz | r (neg => fingertip), b.xyz | 0
};
fn wrapPi(a: f32) -> f32 {
  var x = (a + 3.14159265) % 6.2831853;
  if (x < 0.0) { x = x + 6.2831853; }
  return x - 3.14159265;
}
`;

  const SH_NORMALS = COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> adj: array<i32>;   // NP*12: 6 pairs
@group(0) @binding(3) var<storage, read_write> nrm: array<vec4<f32>>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.np) { return; }
  let p0 = pos[i].xyz;
  var nsum = vec3<f32>(0.0);
  for (var k = 0u; k < 6u; k++) {
    let a = adj[i * 12u + k * 2u];
    if (a < 0) { break; }
    let b = adj[i * 12u + k * 2u + 1u];
    nsum += cross(pos[u32(a)].xyz - p0, pos[u32(b)].xyz - p0);
  }
  nrm[i] = vec4<f32>(nsum, 0.0);
}
`;

  const SH_INTEGRATE = COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> prev: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> nrm: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> baseW: array<f32>;
@group(0) @binding(6) var<storage, read_write> liveW: array<f32>;
@group(1) @binding(0) var<uniform> S: Sub;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.np) { return; }
  var w = baseW[i];
  if (S.counts.z == 1u && i == P.pinIdx) { w = 0.0; }
  var softTarget = vec4<f32>(0.0, 0.0, 0.0, -1.0);
  var hardTarget = vec4<f32>(0.0, 0.0, 0.0, -1.0);
  for (var g = 0u; g < S.counts.x; g++) {
    let rec = S.grabs[g];
    let enc = u32(rec.w);
    if ((enc >> 1u) == i) {
      if ((enc & 1u) == 1u) { softTarget = vec4<f32>(rec.xyz, 1.0); }
      else { hardTarget = vec4<f32>(rec.xyz, 1.0); w = 0.0; }
    }
  }
  liveW[i] = w;
  var p = pos[i].xyz;
  prev[i] = vec4<f32>(p, 0.0);
  var v = vel[i].xyz;
  if (w > 0.0) {
    v.y -= P.gravity * P.dt;
    var d = P.dragK;
    if (p.y < 0.01) { d *= P.settleK; }
    v *= d;
    let nv = nrm[i].xyz;
    let nsq = dot(nv, nv);
    if (nsq > 1e-16) {
      let vn = dot(v, nv) / nsq * P.nDragK;
      v -= vn * nv;
    }
    p += v * P.dt;
  }
  if (S.counts.z == 1u && i == P.pinIdx) { p = vec3<f32>(0.0); }
  if (hardTarget.w > 0.0) { p = hardTarget.xyz; }
  if (softTarget.w > 0.0) { p += (softTarget.xyz - p) * 0.3; }
  vel[i] = vec4<f32>(v, 0.0);
  pos[i] = vec4<f32>(p, 0.0);
}
`;

  const SH_DIST = COMMON + /* wgsl */`
struct Edge { a: u32, b: u32, rest: f32, alphaT: f32 };
struct Range { off: u32, cnt: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> liveW: array<f32>;
@group(0) @binding(3) var<storage, read> edges: array<Edge>;
@group(1) @binding(0) var<uniform> R: Range;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= R.cnt) { return; }
  let e = edges[R.off + gid.x];
  let wa = liveW[e.a];
  let wb = liveW[e.b];
  let ws = wa + wb;
  if (ws == 0.0) { return; }
  var d = pos[e.b].xyz - pos[e.a].xyz;
  let len = length(d);
  if (len < 1e-9) { return; }
  let C = len - e.rest;
  var aT = e.alphaT;
  if (aT < 0.0) {                // rope: one-sided, resists stretch only
    if (C <= 0.0) { return; }
    aT = 0.0;
  }
  let dl = -C / (ws + aT);
  d *= dl / len;
  pos[e.a] = vec4<f32>(pos[e.a].xyz - wa * d, 0.0);
  pos[e.b] = vec4<f32>(pos[e.b].xyz + wb * d, 0.0);
}
`;

  const SH_HINGE = COMMON + /* wgsl */`
struct Hinge { e0: u32, e1: u32, w0: u32, w1: u32, theta0: f32, thC: f32, soft: u32, pad: u32 };
struct Range { off: u32, cnt: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> prev: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> liveW: array<f32>;
@group(0) @binding(4) var<storage, read_write> hinges: array<Hinge>;
@group(1) @binding(0) var<uniform> R: Range;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= R.cnt) { return; }
  let hi = R.off + gid.x;
  let h = hinges[hi];
  let x0 = pos[h.e0].xyz;
  let e = pos[h.e1].xyz - x0;
  let a = pos[h.w0].xyz - x0;
  let b = pos[h.w1].xyz - x0;
  let n1 = cross(e, a);
  let n2 = cross(b, e);
  let elen = length(e);
  let n1sq = dot(n1, n1);
  let n2sq = dot(n2, n2);
  if (elen < 1e-6 || n1sq < 1e-10 || n2sq < 1e-10) { return; }
  let sinT = dot(cross(n1, n2), e) / elen;
  let cosT = dot(n1, n2);
  let thRaw = atan2(sinT, cosT);
  let th = h.thC + wrapPi(thRaw - h.thC);
  hinges[hi].thC = th;
  let C = th - h.theta0;
  let s2 = -elen / n1sq;
  let s3 = -elen / n2sq;
  let g2 = s2 * n1;
  let g3 = s3 * n2;
  let inv_ee = 1.0 / (elen * elen);
  let a2 = dot(a, e) * inv_ee - 1.0;
  let a3 = dot(b, e) * inv_ee - 1.0;
  let g0 = a2 * g2 + a3 * g3;
  let g1 = -(1.0 + a2) * g2 - (1.0 + a3) * g3;
  let w0 = liveW[h.e0]; let w1 = liveW[h.e1];
  let w2 = liveW[h.w0]; let w3 = liveW[h.w1];
  var denom = w0 * dot(g0, g0) + w1 * dot(g1, g1) + w2 * dot(g2, g2) + w3 * dot(g3, g3);
  var alphaT = P.alphaBendT;
  if (h.soft == 1u) { alphaT = P.alphaBendSoftT; }
  denom = denom * (1.0 + P.bendDamp) + alphaT;
  if (denom < 1e-10) { return; }
  var Cdot = dot(g0, pos[h.e0].xyz - prev[h.e0].xyz);
  Cdot += dot(g1, pos[h.e1].xyz - prev[h.e1].xyz);
  Cdot += dot(g2, pos[h.w0].xyz - prev[h.w0].xyz);
  Cdot += dot(g3, pos[h.w1].xyz - prev[h.w1].xyz);
  let dl = (-C - P.bendDamp * Cdot) / denom;
  pos[h.e0] = vec4<f32>(pos[h.e0].xyz + w0 * dl * g0, 0.0);
  pos[h.e1] = vec4<f32>(pos[h.e1].xyz + w1 * dl * g1, 0.0);
  pos[h.w0] = vec4<f32>(pos[h.w0].xyz + w2 * dl * g2, 0.0);
  pos[h.w1] = vec4<f32>(pos[h.w1].xyz + w3 * dl * g3, 0.0);
}
`;

  const SH_COLLIDERS = COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> liveW: array<f32>;
@group(1) @binding(0) var<uniform> S: Sub;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.np) { return; }
  if (liveW[i] == 0.0) { return; }
  var p = pos[i].xyz;
  for (var c = 0u; c < S.counts.y; c++) {
    let A = S.cols[c * 2u];
    let B = S.cols[c * 2u + 1u];
    let r = abs(A.w) + 0.004;
    let ab = B.xyz - A.xyz;
    var d = p - A.xyz;
    let ab2 = dot(ab, ab);
    if (ab2 > 1e-9) {
      let t = clamp(dot(d, ab) / ab2, 0.0, 1.0);
      d -= t * ab;
    }
    let d2 = dot(d, d);
    if (d2 < r * r && d2 > 1e-12) {
      let dl = sqrt(d2);
      p += d * ((r - dl) / dl);
    }
  }
  pos[i] = vec4<f32>(p, 0.0);
}
`;

  const SH_CLEARHEADS = COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> heads: array<atomic<i32>>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = P.gridDim * P.gridDim;
  if (gid.x >= n) { return; }
  atomicStore(&heads[gid.x], 0);
}
`;

  const SH_SELFBUILD = COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> heads: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> nxt: array<i32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.np) { return; }
  let p = pos[i].xyz;
  let gx = clamp(u32((p.x - P.gridOrg) / P.cell), 0u, P.gridDim - 1u);
  let gz = clamp(u32((p.z - P.gridOrg) / P.cell), 0u, P.gridDim - 1u);
  let cellId = gz * P.gridDim + gx;
  let old = atomicExchange(&heads[cellId], i32(i) + 1);
  nxt[i] = old;
}
`;

  const SH_SELFQUERY = COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> prev: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> liveW: array<f32>;
@group(0) @binding(4) var<storage, read> heads: array<i32>;
@group(0) @binding(5) var<storage, read> nxt: array<i32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.np) { return; }
  let wi = liveW[i];
  if (wi == 0.0) { return; }
  var p = pos[i].xyz;
  let r = P.selfR;
  let ii = i32(i / P.n);
  let ji = i32(i % P.n);
  let gx0 = i32((p.x - P.gridOrg) / P.cell);
  let gz0 = i32((p.z - P.gridOrg) / P.cell);
  let gd = i32(P.gridDim);
  var dp = vec3<f32>(0.0);
  for (var dz = -1; dz <= 1; dz++) {
    for (var dx = -1; dx <= 1; dx++) {
      let gx = gx0 + dx;
      let gz = gz0 + dz;
      if (gx < 0 || gx >= gd || gz < 0 || gz >= gd) { continue; }
      var jn = heads[gz * gd + gx];
      loop {
        if (jn == 0) { break; }
        let j = u32(jn - 1);
        jn = nxt[j];
        if (j == i) { continue; }
        // skip pairs close in MATERIAL space (continuum neighbors can't collide)
        let dji = abs(i32(j / P.n) - ii);
        let djj = abs(i32(j % P.n) - ji);
        if (dji <= i32(P.pad0) && djj <= i32(P.pad0)) { continue; }
        let q = pos[j].xyz;
        var d = q - (p + dp);
        let d2 = dot(d, d);
        if (d2 >= r * r || d2 < 1e-12) { continue; }
        let wj = liveW[j];
        let ws = wi + wj;
        let dl = sqrt(d2);
        // own share of the (under-relaxed) symmetric projection
        let corr = 0.8 * (r - dl) / dl * (wi / ws);
        dp -= corr * d;
        // layer friction: own share of the relative tangential slip damp
        let nrm2 = d / dl;
        var slip = ((p + dp) - prev[i].xyz) - (q - prev[j].xyz);
        slip -= dot(slip, nrm2) * nrm2;
        let stl = length(slip);
        if (stl > 1e-9) {
          let f = 0.5 * min(1.0, P.muPaper * (r - dl) / stl) * (wi / ws);
          dp -= f * slip;
        }
      }
    }
  }
  pos[i] = vec4<f32>(p + dp, 0.0);
}
`;

  const SH_GROUND = COMMON + /* wgsl */`
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> prev: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> liveW: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= P.np) { return; }
  var p = pos[i].xyz;
  if (liveW[i] == 0.0) {
    if (p.y < 0.0) { p.y = 0.0; }
  } else if (p.y < 0.0) {
    let depth = -p.y;
    p.y = 0.0;
    let t = vec2<f32>(p.x - prev[i].x, p.z - prev[i].z);
    let tl = length(t);
    if (tl > 1e-9) {
      if (tl < P.muS * depth) {
        p.x = prev[i].x; p.z = prev[i].z;
      } else {
        let drop = min(tl, P.muK * depth);
        let f = (tl - drop) / tl;
        p.x = prev[i].x + t.x * f;
        p.z = prev[i].z + t.y * f;
      }
    }
  }
  pos[i] = vec4<f32>(p, 0.0);
  vel[i] = vec4<f32>((p - prev[i].xyz) / P.dt, 0.0);
}
`;

  const SH_PLASTIC = COMMON + /* wgsl */`
struct Hinge { e0: u32, e1: u32, w0: u32, w1: u32, theta0: f32, thC: f32, soft: u32, pad: u32 };
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> hinges: array<Hinge>;
@group(1) @binding(0) var<uniform> S: Sub;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= P.nh) { return; }
  let h = hinges[gid.x];
  let mid = (pos[h.e0].xyz + pos[h.e1].xyz) * 0.5;
  var pressed = false;
  for (var c = 0u; c < S.counts.y; c++) {
    let A = S.cols[c * 2u];
    if (A.w >= 0.0) { continue; }           // only fingertip capsules gate
    let B = S.cols[c * 2u + 1u];
    let fr = -A.w + 0.05;
    let ab = B.xyz - A.xyz;
    var d = mid - A.xyz;
    let ab2 = dot(ab, ab);
    if (ab2 > 1e-9) {
      let t = clamp(dot(d, ab) / ab2, 0.0, 1.0);
      d -= t * ab;
    }
    if (dot(d, d) <= fr * fr) { pressed = true; break; }
  }
  if (!pressed) { return; }
  let x0 = pos[h.e0].xyz;
  let e = pos[h.e1].xyz - x0;
  let a = pos[h.w0].xyz - x0;
  let b = pos[h.w1].xyz - x0;
  let n1 = cross(e, a);
  let n2 = cross(b, e);
  let elen = length(e);
  if (elen < 1e-6 || dot(n1, n1) < 1e-10 || dot(n2, n2) < 1e-10) { return; }
  let thRaw = atan2(dot(cross(n1, n2), e) / elen, dot(n1, n2));
  let th = h.thC + wrapPi(thRaw - h.thC);
  let ex = th - h.theta0;
  if (abs(ex) > P.yieldA) {
    let s = sign(ex);
    var t0 = h.theta0 + P.plasticRate * (ex - s * P.yieldA);
    t0 = clamp(t0, -P.maxT0, P.maxT0);
    hinges[gid.x].theta0 = t0;
    if (abs(t0) > P.creaseMark) { hinges[gid.x].soft = 1u; }
  }
}
`;

  // ---------------------------------------------------------------- create
  async function create(paper, PARAMS) {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU: no adapter');
    const device = await adapter.requestDevice();
    device.addEventListener('uncapturederror', (e) => {
      console.error('[gpu.js] uncaptured WebGPU error:', e.error.message);
    });
    const NP = paper.NP, N = paper.N;

    const kRes = (N - 1) / 20;
    const sub = Math.min(MAXSUB, Math.max(10, Math.round(PARAMS.substeps * Math.pow(kRes, PARAMS.subPow))));

    // --- constraint coloring
    const edgeGroups = colorEdges(paper);
    const hingeGroups = colorHinges(paper);
    assertColoring(edgeGroups, e => [e.a, e.b], NP);
    assertColoring(hingeGroups, h => [h.e0, h.e1, h.w0, h.w1], NP);
    const edgesSorted = edgeGroups.flat();
    const hingesSorted = hingeGroups.flat();
    const edgeRanges = []; const hingeRanges = [];
    { let off = 0; for (const g of edgeGroups) { edgeRanges.push([off, g.length]); off += g.length; } }
    { let off = 0; for (const g of hingeGroups) { hingeRanges.push([off, g.length]); off += g.length; } }

    // --- buffers
    const B = (size, usage) => device.createBuffer({ size, usage });
    const f32PerVec = 4;
    const posBuf = B(NP * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    const prevBuf = B(NP * 16, GPUBufferUsage.STORAGE);
    const velBuf = B(NP * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const nrmBuf = B(NP * 16, GPUBufferUsage.STORAGE);
    const baseWBuf = B(NP * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const liveWBuf = B(NP * 4, GPUBufferUsage.STORAGE);
    const adjBuf = B(NP * 12 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const edgeBuf = B(Math.max(1, edgesSorted.length) * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const hingeBuf = B(Math.max(1, hingesSorted.length) * 32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
    const headsBuf = B(GRID_DIM * GRID_DIM * 4, GPUBufferUsage.STORAGE);
    const nxtBuf = B(NP * 4, GPUBufferUsage.STORAGE);
    const paramsBuf = B(112, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const scriptBuf = B(MAXSUB * SUB_STRIDE, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const rangeBuf = B(64 * 256, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const stagePos = [
      B(NP * 16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
      B(NP * 16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
    ];
    const stageHinge = [
      B(Math.max(1, hingesSorted.length) * 32, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
      B(Math.max(1, hingesSorted.length) * 32, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
    ];
    let frameIdx = 0, pendingRead = null;

    // --- initial uploads
    {
      const pf = new Float32Array(NP * 4);
      for (let i = 0; i < NP; i++) {
        pf[i * 4] = paper.pos[i * 3];
        pf[i * 4 + 1] = paper.pos[i * 3 + 1];
        pf[i * 4 + 2] = paper.pos[i * 3 + 2];
      }
      device.queue.writeBuffer(posBuf, 0, pf);
      device.queue.writeBuffer(velBuf, 0, new Float32Array(NP * 4));
      device.queue.writeBuffer(baseWBuf, 0, new Float32Array(paper.baseInvMass));
    }
    {
      // per-vertex adjacent triangles as ordered vertex pairs
      const adj = new Int32Array(NP * 12).fill(-1);
      const cnt = new Int32Array(NP);
      const tris = paper.tris;
      for (let t = 0; t < tris.length; t += 3) {
        const vs = [tris[t], tris[t + 1], tris[t + 2]];
        for (let k = 0; k < 3; k++) {
          const v = vs[k];
          if (cnt[v] < 6) {
            adj[v * 12 + cnt[v] * 2] = vs[(k + 1) % 3];
            adj[v * 12 + cnt[v] * 2 + 1] = vs[(k + 2) % 3];
            cnt[v]++;
          }
        }
      }
      device.queue.writeBuffer(adjBuf, 0, adj);
    }
    {
      const ed = new ArrayBuffer(Math.max(1, edgesSorted.length) * 16);
      const u = new Uint32Array(ed), f = new Float32Array(ed);
      const dtSub = (1 / 60) / sub;
      edgesSorted.forEach((e, i) => {
        u[i * 4] = e.a; u[i * 4 + 1] = e.b;
        f[i * 4 + 2] = e.rest;
        f[i * 4 + 3] = e.rope ? -1 : e.alpha / (dtSub * dtSub);
      });
      device.queue.writeBuffer(edgeBuf, 0, ed);
    }
    const writeHinges = () => {
      const hd = new ArrayBuffer(Math.max(1, hingesSorted.length) * 32);
      const u = new Uint32Array(hd), f = new Float32Array(hd);
      hingesSorted.forEach((h, i) => {
        u[i * 8] = h.e0; u[i * 8 + 1] = h.e1; u[i * 8 + 2] = h.w0; u[i * 8 + 3] = h.w1;
        f[i * 8 + 4] = h.theta0; f[i * 8 + 5] = h.thC;
        u[i * 8 + 6] = h.soft ? 1 : 0;
      });
      device.queue.writeBuffer(hingeBuf, 0, hd);
    };
    writeHinges();
    const NEG = edgeRanges.length; // hinge ranges start after all edge groups
    {
      const r = new Uint32Array(64 * 64);
      edgeRanges.forEach(([off, cnt], i) => { r[i * 64] = off; r[i * 64 + 1] = cnt; });
      hingeRanges.forEach(([off, cnt], i) => { r[(NEG + i) * 64] = off; r[(NEG + i) * 64 + 1] = cnt; });
      device.queue.writeBuffer(rangeBuf, 0, r);
    }
    {
      const dtSub = (1 / 60) / sub;
      const ab = new ArrayBuffer(112);
      const u = new Uint32Array(ab), f = new Float32Array(ab);
      u[0] = NP; u[1] = edgesSorted.length; u[2] = hingesSorted.length; u[3] = paper.tris.length / 3;
      u[4] = N; u[5] = paper.corners.A; u[6] = GRID_DIM;
      u[7] = Math.max(1, Math.ceil((PARAMS.selfR * 1.5) / paper.H)); // material skip radius K
      f[8] = dtSub; f[9] = PARAMS.gravity;
      f[10] = Math.exp(-PARAMS.airDrag * dtSub);
      f[11] = Math.exp(-PARAMS.settleDrag * dtSub);
      f[12] = 1 - Math.exp(-PARAMS.normalDrag * dtSub);
      f[13] = PARAMS.alphaBend / (dtSub * dtSub);
      f[14] = (PARAMS.alphaBend * PARAMS.creaseSoften) / (dtSub * dtSub);
      f[15] = PARAMS.bendDamp * Math.max(1, kRes); // scaled: hinge projections harden ~1/h^4
      f[16] = PARAMS.muStatic; f[17] = PARAMS.muKinetic; f[18] = PARAMS.muPaper; f[19] = PARAMS.selfR;
      f[20] = PARAMS.yieldK * paper.H; f[21] = PARAMS.plasticRate; f[22] = PARAMS.maxTheta0; f[23] = PARAMS.creaseMark;
      f[24] = GRID_SPAN / GRID_DIM; f[25] = GRID_ORG;
      device.queue.writeBuffer(paramsBuf, 0, ab);
    }

    // --- pipelines with explicit layouts (dynamic uniforms need them)
    const mkLayout = (entries) => device.createBindGroupLayout({ entries });
    const ST = (binding, type) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const layoutMain = {
      normals: mkLayout([ST(0, 'uniform'), ST(1, 'read-only-storage'), ST(2, 'read-only-storage'), ST(3, 'storage')]),
      integrate: mkLayout([ST(0, 'uniform'), ST(1, 'storage'), ST(2, 'storage'), ST(3, 'storage'), ST(4, 'read-only-storage'), ST(5, 'read-only-storage'), ST(6, 'storage')]),
      dist: mkLayout([ST(0, 'uniform'), ST(1, 'storage'), ST(2, 'read-only-storage'), ST(3, 'read-only-storage')]),
      hinge: mkLayout([ST(0, 'uniform'), ST(1, 'storage'), ST(2, 'read-only-storage'), ST(3, 'read-only-storage'), ST(4, 'storage')]),
      colliders: mkLayout([ST(0, 'uniform'), ST(1, 'storage'), ST(2, 'read-only-storage')]),
      clearHeads: mkLayout([ST(0, 'uniform'), ST(1, 'storage')]),
      selfBuild: mkLayout([ST(0, 'uniform'), ST(1, 'read-only-storage'), ST(2, 'storage'), ST(3, 'storage')]),
      selfQuery: mkLayout([ST(0, 'uniform'), ST(1, 'storage'), ST(2, 'read-only-storage'), ST(3, 'read-only-storage'), ST(4, 'read-only-storage'), ST(5, 'read-only-storage')]),
      ground: mkLayout([ST(0, 'uniform'), ST(1, 'storage'), ST(2, 'read-only-storage'), ST(3, 'storage'), ST(4, 'read-only-storage')]),
      plastic: mkLayout([ST(0, 'uniform'), ST(1, 'read-only-storage'), ST(2, 'storage')]),
    };
    const subLayout = mkLayout([{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } }]);
    const rangeLayout = subLayout; // same shape

    const mkPipe = (code, main, extra) => device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: extra ? [main, extra] : [main] }),
      compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });
    const pipes = {
      normals: mkPipe(SH_NORMALS, layoutMain.normals),
      integrate: mkPipe(SH_INTEGRATE, layoutMain.integrate, subLayout),
      dist: mkPipe(SH_DIST, layoutMain.dist, rangeLayout),
      hinge: mkPipe(SH_HINGE, layoutMain.hinge, rangeLayout),
      colliders: mkPipe(SH_COLLIDERS, layoutMain.colliders, subLayout),
      clearHeads: mkPipe(SH_CLEARHEADS, layoutMain.clearHeads),
      selfBuild: mkPipe(SH_SELFBUILD, layoutMain.selfBuild),
      selfQuery: mkPipe(SH_SELFQUERY, layoutMain.selfQuery),
      ground: mkPipe(SH_GROUND, layoutMain.ground),
      plastic: mkPipe(SH_PLASTIC, layoutMain.plastic, subLayout),
    };
    const bg = (layout, bufs) => device.createBindGroup({
      layout, entries: bufs.map((b, i) => ({ binding: i, resource: { buffer: b } })),
    });
    const groups = {
      normals: bg(layoutMain.normals, [paramsBuf, posBuf, adjBuf, nrmBuf]),
      integrate: bg(layoutMain.integrate, [paramsBuf, posBuf, prevBuf, velBuf, nrmBuf, baseWBuf, liveWBuf]),
      dist: bg(layoutMain.dist, [paramsBuf, posBuf, liveWBuf, edgeBuf]),
      hinge: bg(layoutMain.hinge, [paramsBuf, posBuf, prevBuf, liveWBuf, hingeBuf]),
      colliders: bg(layoutMain.colliders, [paramsBuf, posBuf, liveWBuf]),
      clearHeads: bg(layoutMain.clearHeads, [paramsBuf, headsBuf]),
      selfBuild: bg(layoutMain.selfBuild, [paramsBuf, posBuf, headsBuf, nxtBuf]),
      selfQuery: bg(layoutMain.selfQuery, [paramsBuf, posBuf, prevBuf, liveWBuf, headsBuf, nxtBuf]),
      ground: bg(layoutMain.ground, [paramsBuf, posBuf, prevBuf, velBuf, liveWBuf]),
      plastic: bg(layoutMain.plastic, [paramsBuf, posBuf, hingeBuf]),
    };
    const subGroup = device.createBindGroup({
      layout: subLayout,
      entries: [{ binding: 0, resource: { buffer: scriptBuf, size: SUB_STRIDE } }],
    });
    const rangeGroup = device.createBindGroup({
      layout: rangeLayout,
      entries: [{ binding: 0, resource: { buffer: rangeBuf, size: 16 } }],
    });

    // --- per-frame script building (mirrors sim.js collider/grab assembly)
    const scriptArr = new ArrayBuffer(MAXSUB * SUB_STRIDE);
    const scriptU = new Uint32Array(scriptArr);
    const scriptF = new Float32Array(scriptArr);
    const matIdx = (u, v) => Math.round(v / paper.H) * N + Math.round(u / paper.H);
    function packWorld(s, world) {
      const base = (s * SUB_STRIDE) / 4;
      const grabs = world.grabs || [];
      const fingers = world.fingers || (world.finger ? [world.finger] : []);
      let nCol = 0;
      const colBase = base + 4 + MAXGRAB * 4;
      const addCol = (ax, ay, az, bx, by, bz, r, tip) => {
        if (nCol >= MAXCOL) return;
        const o = colBase + nCol * 8;
        scriptF[o] = ax; scriptF[o + 1] = ay; scriptF[o + 2] = az; scriptF[o + 3] = tip ? -r : r;
        scriptF[o + 4] = bx; scriptF[o + 5] = by; scriptF[o + 6] = bz; scriptF[o + 7] = 0;
        nCol++;
      };
      for (const f of fingers) {
        const hl = f.hl || 0, fx = f.ax || [1, 0, 0];
        addCol(f.c[0] - fx[0] * hl, f.c[1] - fx[1] * hl, f.c[2] - fx[2] * hl,
          f.c[0] + fx[0] * hl, f.c[1] + fx[1] * hl, f.c[2] + fx[2] * hl, f.r, true);
        addCol(f.c[0], f.c[1], f.c[2], f.c[0], f.c[1] + PARAMS.stemLen, f.c[2], PARAMS.fingerStemR, false);
      }
      if (world.pinA) {
        addCol(0, PARAMS.padLift, 0, 0, PARAMS.padLift, 0, PARAMS.padR, false);
        addCol(0, PARAMS.padLift, 0, -0.26, PARAMS.padLift + 0.56, -0.26, PARAMS.stemR, false);
      }
      for (const g of grabs) {
        const gy = g.pos[1] + PARAMS.padLift;
        addCol(g.pos[0], gy, g.pos[2], g.pos[0], gy, g.pos[2], PARAMS.padR, false);
      }
      const ng = Math.min(grabs.length, MAXGRAB);
      for (let g = 0; g < ng; g++) {
        const o = base + 4 + g * 4;
        const gr = grabs[g];
        scriptF[o] = gr.pos[0]; scriptF[o + 1] = gr.pos[1]; scriptF[o + 2] = gr.pos[2];
        scriptU[o + 3] = 0; // cleared; write encoded idx as float below
        scriptF[o + 3] = matIdx(gr.at[0], gr.at[1]) * 2 + (gr.soft ? 1 : 0);
      }
      scriptU[base] = ng;
      scriptU[base + 1] = nCol;
      scriptU[base + 2] = world.pinA ? 1 : 0;
    }

    const posRead = new Float32Array(NP * 4);
    const hingeReadAB = new ArrayBuffer(Math.max(1, hingesSorted.length) * 32);
    const disp = (pass, pipe, group, count, dyn) => {
      pass.setPipeline(pipe);
      pass.setBindGroup(0, group);
      if (dyn) pass.setBindGroup(1, dyn.g, [dyn.o]);
      pass.dispatchWorkgroups(Math.ceil(count / WG));
    };

    let destroyed = false;
    async function frame(worldAt, dtFrame) {
      if (destroyed) return;
      // build the substep script on CPU
      const dtSub = dtFrame / sub;
      for (let s = 0; s < sub; s++) packWorld(s, worldAt(paper.time + dtSub * (s + 1)));
      device.queue.writeBuffer(scriptBuf, 0, scriptArr, 0, sub * SUB_STRIDE);

      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      for (let s = 0; s < sub; s++) {
        const subDyn = { g: subGroup, o: s * SUB_STRIDE };
        disp(pass, pipes.normals, groups.normals, NP);
        disp(pass, pipes.integrate, groups.integrate, NP, subDyn);
        for (let g = 0; g < edgeRanges.length; g++) {
          if (!edgeRanges[g][1]) continue;
          disp(pass, pipes.dist, groups.dist, edgeRanges[g][1], { g: rangeGroup, o: g * 256 });
        }
        for (let g = 0; g < hingeRanges.length; g++) {
          if (!hingeRanges[g][1]) continue;
          disp(pass, pipes.hinge, groups.hinge, hingeRanges[g][1], { g: rangeGroup, o: (NEG + g) * 256 });
        }
        disp(pass, pipes.colliders, groups.colliders, NP, subDyn);
        disp(pass, pipes.clearHeads, groups.clearHeads, GRID_DIM * GRID_DIM);
        disp(pass, pipes.selfBuild, groups.selfBuild, NP);
        disp(pass, pipes.selfQuery, groups.selfQuery, NP);
        disp(pass, pipes.ground, groups.ground, NP);
      }
      // plasticity, gated by the last substep's fingers
      disp(pass, pipes.plastic, groups.plastic, hingesSorted.length, { g: subGroup, o: (sub - 1) * SUB_STRIDE });
      pass.end();
      // pipelined readback: copy into this frame's staging pair, but only
      // await (and apply) the PREVIOUS frame's — no full GPU stall per frame
      const cur = frameIdx % 2;
      const wantHinge = frameIdx % 8 === 0;
      enc.copyBufferToBuffer(posBuf, 0, stagePos[cur], 0, NP * 16);
      if (wantHinge) enc.copyBufferToBuffer(hingeBuf, 0, stageHinge[cur], 0, hingeReadAB.byteLength);
      device.queue.submit([enc.finish()]);
      frameIdx++;
      paper.time += dtFrame;
      const finished = { cur, wantHinge };
      if (pendingRead) await consumeRead(pendingRead);
      pendingRead = finished;
    }
    async function consumeRead(p) {
      await stagePos[p.cur].mapAsync(GPUMapMode.READ);
      posRead.set(new Float32Array(stagePos[p.cur].getMappedRange()));
      stagePos[p.cur].unmap();
      for (let i = 0; i < NP; i++) {
        paper.pos[i * 3] = posRead[i * 4];
        paper.pos[i * 3 + 1] = posRead[i * 4 + 1];
        paper.pos[i * 3 + 2] = posRead[i * 4 + 2];
      }
      if (p.wantHinge) {
        await stageHinge[p.cur].mapAsync(GPUMapMode.READ);
        new Uint8Array(hingeReadAB).set(new Uint8Array(stageHinge[p.cur].getMappedRange()));
        stageHinge[p.cur].unmap();
        const hf = new Float32Array(hingeReadAB);
        const hu = new Uint32Array(hingeReadAB);
        hingesSorted.forEach((h, i) => {
          h.theta0 = hf[i * 8 + 4];
          h.thC = hf[i * 8 + 5];
          h.soft = hu[i * 8 + 6] === 1;
        });
      }
    }
    async function flush() {
      if (pendingRead) { await consumeRead(pendingRead); pendingRead = null; }
    }

    return {
      substeps: sub,
      frame,
      flush,
      _dbg: { device, posBuf, stagePos: stagePos[0], NP },
      destroy() {
        destroyed = true;
        for (const b of [posBuf, prevBuf, velBuf, nrmBuf, baseWBuf, liveWBuf, adjBuf, edgeBuf,
          hingeBuf, headsBuf, nxtBuf, paramsBuf, scriptBuf, rangeBuf,
          ...stagePos, ...stageHinge]) b.destroy();
        device.destroy();
      },
    };
  }

  return { available, create };
});

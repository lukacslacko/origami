/*
 * mesh2.js — adaptive, crease-conforming triangulation of the unit square
 * (material space) for the v2 paper engine.
 *
 * The mesh is a Delaunay triangulation of a variable-density point set:
 *  - samples along every crease segment (spacing hMin), with a protection
 *    zone that keeps other points away, so consecutive crease samples are
 *    guaranteed to be connected by mesh edges;
 *  - graded background points (dart throwing on a jittered lattice against
 *    a size field: hMin near creases / high-curvature cells, hMax elsewhere);
 *  - the square's boundary sampled at the local size, corners always;
 *  - required "anchor" points (scenario grip locations).
 *
 * Deterministic (seeded LCG). Runs in the browser and in Node.
 */
(function (root, factory) {
  if (typeof module !== 'undefined' && module.exports) module.exports = factory();
  else root.MESH2 = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ------------------------------------------------------------ utilities
  function makeRng(seed) {
    let s = seed >>> 0;
    return () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }
  const orient = (ax, ay, bx, by, cx, cy) => (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  function inCircle(ax, ay, bx, by, cx, cy, px, py) {
    const adx = ax - px, ady = ay - py;
    const bdx = bx - px, bdy = by - py;
    const cdx = cx - px, cdy = cy - py;
    const ad = adx * adx + ady * ady;
    const bd = bdx * bdx + bdy * bdy;
    const cd = cdx * cdx + cdy * cdy;
    return adx * (bdy * cd - bd * cdy) - ady * (bdx * cd - bd * cdx) + ad * (bdx * cdy - bdy * cdx) > 0;
  }
  function segDist(px, py, ax, ay, bx, by) {
    const abx = bx - ax, aby = by - ay;
    const l2 = abx * abx + aby * aby;
    let t = l2 > 1e-16 ? ((px - ax) * abx + (py - ay) * aby) / l2 : 0;
    t = Math.max(0, Math.min(1, t));
    return Math.hypot(px - (ax + t * abx), py - (ay + t * aby));
  }
  function segIntersect(a, b, c, d) {
    // proper intersection point of segments ab and cd, or null
    const d1 = orient(a[0], a[1], b[0], b[1], c[0], c[1]);
    const d2 = orient(a[0], a[1], b[0], b[1], d[0], d[1]);
    const d3 = orient(c[0], c[1], d[0], d[1], a[0], a[1]);
    const d4 = orient(c[0], c[1], d[0], d[1], b[0], b[1]);
    if (((d1 > 0) !== (d2 > 0)) && ((d3 > 0) !== (d4 > 0))) {
      const t = d1 / (d1 - d2);
      return [c[0] + t * (d[0] - c[0]), c[1] + t * (d[1] - c[1])];
    }
    return null;
  }

  // ------------------------------------------------- Bowyer-Watson Delaunay
  // Point-location by walking + BFS cavity. Points must be pre-jittered
  // enough to avoid exact degeneracies (the callers guarantee this).
  function delaunay(px, py) {
    const n = px.length;
    // super-triangle
    const X = Float64Array.from(px), Y = Float64Array.from(py);
    const pts = n + 3;
    const xs = new Float64Array(pts), ys = new Float64Array(pts);
    xs.set(X); ys.set(Y);
    xs[n] = -10; ys[n] = -10;
    xs[n + 1] = 12; ys[n + 1] = -10;
    xs[n + 2] = 0.5; ys[n + 2] = 14;
    // triangle soup with adjacency: tri i -> verts [3i..3i+2], nbr [3i..3i+2]
    // nbr[k] = triangle across the edge (v[k], v[k+1]); -1 = none
    let verts = [n, n + 1, n + 2];
    let nbrs = [-1, -1, -1];
    let alive = [true];
    let lastTri = 0;

    const edgeMap = new Map(); // rebuild helper during cavity fill
    function locate(x, y) {
      // walk from lastTri
      let t = lastTri;
      for (let guard = 0; guard < 4 * (alive.length + 4); guard++) {
        if (!alive[t]) { t = alive.lastIndexOf(true); if (t < 0) return -1; }
        const a = verts[t * 3], b = verts[t * 3 + 1], c = verts[t * 3 + 2];
        const o1 = orient(xs[a], ys[a], xs[b], ys[b], x, y);
        const o2 = orient(xs[b], ys[b], xs[c], ys[c], x, y);
        const o3 = orient(xs[c], ys[c], xs[a], ys[a], x, y);
        if (o1 >= 0 && o2 >= 0 && o3 >= 0) return t;
        // step toward the most violated edge
        let k = 0, worst = o1;
        if (o2 < worst) { worst = o2; k = 1; }
        if (o3 < worst) { worst = o3; k = 2; }
        const nt = nbrs[t * 3 + k];
        if (nt < 0) return t; // shouldn't happen inside super-tri
        t = nt;
      }
      // fallback linear scan
      for (let t2 = 0; t2 < alive.length; t2++) {
        if (!alive[t2]) continue;
        const a = verts[t2 * 3], b = verts[t2 * 3 + 1], c = verts[t2 * 3 + 2];
        if (orient(xs[a], ys[a], xs[b], ys[b], x, y) >= 0 &&
            orient(xs[b], ys[b], xs[c], ys[c], x, y) >= 0 &&
            orient(xs[c], ys[c], xs[a], ys[a], x, y) >= 0) return t2;
      }
      return -1;
    }

    const stack = [];
    for (let p = 0; p < n; p++) {
      const x = xs[p], y = ys[p];
      const t0 = locate(x, y);
      if (t0 < 0) throw new Error('delaunay: locate failed');
      // BFS all triangles whose circumcircle contains p
      const bad = new Set([t0]);
      stack.length = 0; stack.push(t0);
      while (stack.length) {
        const t = stack.pop();
        for (let k = 0; k < 3; k++) {
          const nt = nbrs[t * 3 + k];
          if (nt < 0 || bad.has(nt)) continue;
          const a = verts[nt * 3], b = verts[nt * 3 + 1], c = verts[nt * 3 + 2];
          if (inCircle(xs[a], ys[a], xs[b], ys[b], xs[c], ys[c], x, y)) {
            bad.add(nt); stack.push(nt);
          }
        }
      }
      // cavity boundary: edges of bad triangles whose neighbor is not bad
      edgeMap.clear();
      const boundary = [];
      for (const t of bad) {
        for (let k = 0; k < 3; k++) {
          const nt = nbrs[t * 3 + k];
          if (nt >= 0 && bad.has(nt)) continue;
          boundary.push([verts[t * 3 + k], verts[t * 3 + (k + 1) % 3], nt]);
        }
        alive[t] = false;
      }
      // create new triangles fanning from p
      const newTris = [];
      for (const [a, b, outer] of boundary) {
        const t = alive.length;
        verts.push(a, b, p);
        nbrs.push(outer, -1, -1); // nbr across (a,b) = outer; others linked below
        alive.push(true);
        newTris.push(t);
        // fix outer's back-pointer
        if (outer >= 0) {
          for (let k = 0; k < 3; k++) {
            if (verts[outer * 3 + k] === b && verts[outer * 3 + (k + 1) % 3] === a) {
              nbrs[outer * 3 + k] = t;
            }
          }
        }
        edgeMap.set(a + ',' + p, t);      // edge (a<-p) is this tri's edge 2 reversed
        edgeMap.set(p + ',' + b, t);      // hmm handled below via lookup
      }
      // link new triangles around the fan: edge (b,p) of tri (a,b,p) matches
      // edge (p,b') where b' = b of the next tri whose a' = b
      for (const t of newTris) {
        const a = verts[t * 3], b = verts[t * 3 + 1];
        // neighbor across (b,p): the tri with first vert b
        const t1 = edgeMap.get(b + ',' + p);
        // t1 has verts (b, x, p): its edge (p,b)... find tri whose a==b
        nbrs[t * 3 + 1] = t1 !== undefined ? t1 : -1;
        // neighbor across (p,a): the tri whose b == a
        const t2 = edgeMap.get(a + ',' + p);
        nbrs[t * 3 + 2] = t2 !== undefined ? t2 : -1;
      }
      // (the map above stores a+','+p -> tri whose a-vert is a; for edge (p,a)
      // of tri t we need the tri whose SECOND vert is a: fix by second pass)
      for (const t of newTris) {
        const a = verts[t * 3], b = verts[t * 3 + 1];
        // across edge1 (b,p): tri whose edge0 starts at b
        let found = -1;
        for (const t2 of newTris) if (verts[t2 * 3] === b) { found = t2; break; }
        nbrs[t * 3 + 1] = found;
        // across edge2 (p,a): tri whose edge0 ends at a (verts[t2*3+1] === a)
        found = -1;
        for (const t2 of newTris) if (verts[t2 * 3 + 1] === a) { found = t2; break; }
        nbrs[t * 3 + 2] = found;
      }
      lastTri = newTris[newTris.length - 1];
    }
    // collect real triangles (drop any touching super vertices)
    const out = [];
    for (let t = 0; t < alive.length; t++) {
      if (!alive[t]) continue;
      const a = verts[t * 3], b = verts[t * 3 + 1], c = verts[t * 3 + 2];
      if (a >= n || b >= n || c >= n) continue;
      out.push(a, b, c);
    }
    return out;
  }

  // ------------------------------------------------------------ size field
  // sizeAt(u,v) = min(hMax, crease band size, curvature-grid size)
  function makeSizeField(opts) {
    const { hMax, hMin, creases, sizeGrid } = opts;
    const band = 2.5 * hMin;
    return (u, v) => {
      let s = hMax;
      if (sizeGrid) {
        const g = sizeGrid.n;
        const cx = Math.min(g - 1, Math.max(0, Math.floor(u * g)));
        const cy = Math.min(g - 1, Math.max(0, Math.floor(v * g)));
        s = Math.min(s, sizeGrid.data[cy * g + cx]);
      }
      for (const c of creases) {
        const d = segDist(u, v, c.a[0], c.a[1], c.b[0], c.b[1]);
        s = Math.min(s, Math.max(hMin, hMin + (d - band)));
      }
      return Math.max(hMin, s);
    };
  }

  // ------------------------------------------------------------- sampling
  function buildPoints(opts) {
    const { hMax, hMin, creases, anchors } = opts;
    const rng = makeRng(1234567);
    const sizeAt = makeSizeField(opts);
    const pts = [];        // [u, v]
    const kind = [];       // 0 background, 1 boundary, 2 crease, 3 anchor/corner
    const creaseSample = []; // per point: null or [{ci, s}, ...]

    const clamp01 = (x) => Math.max(0, Math.min(1, x));
    const closeIdx = (u, v, r) => {
      for (let q = 0; q < pts.length; q++) {
        if (Math.abs(pts[q][0] - u) < r && Math.abs(pts[q][1] - v) < r &&
            Math.hypot(pts[q][0] - u, pts[q][1] - v) < r) return q;
      }
      return -1;
    };
    const close = (u, v, r) => closeIdx(u, v, r) >= 0;

    // 1) corners + anchors (exact, first, so nothing displaces them)
    const req = [[0, 0], [1, 0], [1, 1], [0, 1], ...(anchors || [])];
    for (const [u, v] of req) {
      if (!close(u, v, 0.35 * hMin)) { pts.push([clamp01(u), clamp01(v)]); kind.push(3); creaseSample.push(null); }
    }

    // 2) crease samples: every segment, split at mutual intersections
    creases.forEach((c, ci) => {
      const cuts = [0, 1];
      creases.forEach((c2, cj) => {
        if (cj === ci) return;
        const X = segIntersect(c.a, c.b, c2.a, c2.b);
        if (X) {
          const len = Math.hypot(c.b[0] - c.a[0], c.b[1] - c.a[1]);
          const t = Math.hypot(X[0] - c.a[0], X[1] - c.a[1]) / Math.max(len, 1e-9);
          cuts.push(Math.max(0, Math.min(1, t)));
        }
      });
      cuts.sort((x, y) => x - y);
      const len = Math.hypot(c.b[0] - c.a[0], c.b[1] - c.a[1]);
      for (let s = 0; s < cuts.length - 1; s++) {
        const t0 = cuts[s], t1 = cuts[s + 1];
        const segLen = (t1 - t0) * len;
        const nSteps = Math.max(1, Math.round(segLen / hMin));
        for (let k = 0; k <= nSteps; k++) {
          const t = t0 + (t1 - t0) * (k / nSteps);
          const u = clamp01(c.a[0] + t * (c.b[0] - c.a[0]));
          const v = clamp01(c.a[1] + t * (c.b[1] - c.a[1]));
          const dup = closeIdx(u, v, 0.45 * hMin);
          if (dup < 0) {
            pts.push([u, v]); kind.push(2); creaseSample.push([{ ci, s: t * len }]);
          } else {
            // shared point (crease crossing / sub-segment junction): register
            // membership once per crease
            if (!creaseSample[dup]) creaseSample[dup] = [];
            if (!creaseSample[dup].some(e => e.ci === ci)) {
              creaseSample[dup].push({ ci, s: t * len });
            }
          }
        }
      }
    });

    // 3) boundary samples at local size
    for (const [x0, y0, dx, dy, L] of [
      [0, 0, 1, 0, 1], [1, 0, 0, 1, 1], [1, 1, -1, 0, 1], [0, 1, 0, -1, 1],
    ]) {
      let s = 0;
      while (s < L) {
        const u = x0 + dx * s, v = y0 + dy * s;
        const step = Math.max(hMin, 0.9 * sizeAt(u, v));
        s += step;
        if (s >= L - 0.3 * step) break;
        const bu = clamp01(x0 + dx * s), bv = clamp01(y0 + dy * s);
        if (!close(bu, bv, 0.5 * hMin)) { pts.push([bu, bv]); kind.push(1); creaseSample.push(null); }
      }
    }

    // 4) graded background: dart throwing over a jittered fine lattice,
    // shuffled deterministically; accepted if far enough from everything
    const cell = hMin * 0.9;
    const nl = Math.ceil(1 / cell);
    const order = [];
    for (let i = 0; i <= nl; i++) for (let j = 0; j <= nl; j++) order.push(i * (nl + 1) + j);
    for (let i = order.length - 1; i > 0; i--) {
      const j = (rng() * (i + 1)) | 0;
      const tmp = order[i]; order[i] = order[j]; order[j] = tmp;
    }
    // spatial hash of accepted points for fast lookups
    const hcell = hMax;
    const hash = new Map();
    const hkey = (u, v) => Math.floor(u / hcell) * 4096 + Math.floor(v / hcell);
    const hAdd = (u, v, i2) => {
      const k = hkey(u, v);
      let a = hash.get(k); if (!a) { a = []; hash.set(k, a); }
      a.push(i2);
    };
    pts.forEach((p, i2) => hAdd(p[0], p[1], i2));
    const farEnough = (u, v, r) => {
      const c0x = Math.floor((u - r) / hcell), c1x = Math.floor((u + r) / hcell);
      const c0y = Math.floor((v - r) / hcell), c1y = Math.floor((v + r) / hcell);
      for (let cx = c0x; cx <= c1x; cx++) for (let cy = c0y; cy <= c1y; cy++) {
        const a = hash.get(cx * 4096 + cy);
        if (!a) continue;
        for (const i2 of a) {
          if (Math.hypot(pts[i2][0] - u, pts[i2][1] - v) < r) return false;
        }
      }
      return true;
    };
    for (const o of order) {
      const i = (o / (nl + 1)) | 0, j = o % (nl + 1);
      const u = clamp01((j + 0.18 + 0.64 * rng()) * cell);
      const v = clamp01((i + 0.18 + 0.64 * rng()) * cell);
      const s = sizeAt(u, v);
      if (u < 0.45 * s || u > 1 - 0.45 * s || v < 0.45 * s || v > 1 - 0.45 * s) continue;
      let nearCrease = false;
      for (const c of creases) {
        if (segDist(u, v, c.a[0], c.a[1], c.b[0], c.b[1]) < 0.62 * hMin) { nearCrease = true; break; }
      }
      if (nearCrease) continue;
      if (!farEnough(u, v, 0.62 * s)) continue;
      pts.push([u, v]); kind.push(0); creaseSample.push(null);
      hAdd(u, v, pts.length - 1);
    }
    return { pts, kind, creaseSample };
  }


  // ------------------------------------------------- edge enforcement (CDT)
  // Flip diagonals until every required pair (a,b) is a mesh edge. The pairs
  // are short (adjacent crease samples), so typically one flip each.
  function enforceEdges(px, py, tris, pairs) {
    const properX = (a, b, c, d) => {
      const o = (i, j, k) => (px[j] - px[i]) * (py[k] - py[i]) - (py[j] - py[i]) * (px[k] - px[i]);
      const d1 = o(a, b, c), d2 = o(a, b, d), d3 = o(c, d, a), d4 = o(c, d, b);
      return ((d1 > 0) !== (d2 > 0)) && ((d3 > 0) !== (d4 > 0));
    };
    for (let guard = 0; guard < 200; guard++) {
      // adjacency: edge -> [{tri, opp}]
      const adj = new Map();
      const has = new Set();
      for (let t = 0; t < tris.length; t += 3) {
        for (let k = 0; k < 3; k++) {
          const i = tris[t + k], j = tris[t + (k + 1) % 3], o = tris[t + (k + 2) % 3];
          const key = i < j ? i + ',' + j : j + ',' + i;
          has.add(key);
          let arr = adj.get(key);
          if (!arr) { arr = []; adj.set(key, arr); }
          arr.push({ t, o });
        }
      }
      let fixed = true;
      for (const [a, b] of pairs) {
        const want = a < b ? a + ',' + b : b + ',' + a;
        if (has.has(want)) continue;
        // find an interior edge crossing segment a-b and flip it
        let done = false;
        for (const [key, arr] of adj) {
          if (arr.length !== 2) continue;
          const [c, d] = key.split(',').map(Number);
          if (c === a || c === b || d === a || d === b) continue;
          if (!properX(a, b, c, d)) continue;
          const { t: t1, o: o1 } = arr[0];
          const { t: t2, o: o2 } = arr[1];
          // replace the two triangles with the flipped pair (o1, o2 diagonal)
          const mk = (i, j, k) => {
            if ((px[j] - px[i]) * (py[k] - py[i]) - (py[j] - py[i]) * (px[k] - px[i]) < 0) return [i, k, j];
            return [i, j, k];
          };
          const n1 = mk(o1, o2, c), n2 = mk(o1, d, o2);
          tris[t1] = n1[0]; tris[t1 + 1] = n1[1]; tris[t1 + 2] = n1[2];
          tris[t2] = n2[0]; tris[t2 + 1] = n2[1]; tris[t2 + 2] = n2[2];
          done = true;
          break;
        }
        fixed = false;
        if (done) break; // adjacency is stale; rebuild and continue
        return tris;     // no crossing edge found: give up on this pair
      }
      if (fixed) return tris;
    }
    return tris;
  }

  // ------------------------------------------------------------ top level
  // opts: { hMax, hMin, creases: [{a:[u,v], b:[u,v]}], sizeGrid, anchors }
  // returns { uv: Float64Array(2n), tris: [i,j,k...], kind, creaseSample,
  //           creaseEdges: Map('i,j' -> {ci}) } — all triangles CCW.
  function buildMesh(opts) {
    const { pts, kind, creaseSample } = buildPoints(opts);
    const px = pts.map(p => p[0]), py = pts.map(p => p[1]);
    let tris = delaunay(px, py);
    // normalize CCW
    for (let t = 0; t < tris.length; t += 3) {
      const a = tris[t], b = tris[t + 1], c = tris[t + 2];
      if (orient(px[a], py[a], px[b], py[b], px[c], py[c]) < 0) {
        tris[t + 1] = c; tris[t + 2] = b;
      }
    }
    // verify crease connectivity: consecutive samples on a segment must be
    // linked by an edge (protection zones make this hold; assert anyway)
    const edgeSet = new Set();
    for (let t = 0; t < tris.length; t += 3) {
      for (let k = 0; k < 3; k++) {
        const a = tris[t + k], b = tris[t + (k + 1) % 3];
        edgeSet.add(a < b ? a + ',' + b : b + ',' + a);
      }
    }
    const byCrease = new Map();
    pts.forEach((p, i) => {
      const list = creaseSample[i];
      if (!list) return;
      for (const cs of list) {
        let arr = byCrease.get(cs.ci);
        if (!arr) { arr = []; byCrease.set(cs.ci, arr); }
        arr.push({ i, s: cs.s });
      }
    });
    // collect required consecutive-sample pairs, enforce any missing edges,
    // then rebuild the edge set and register crease edges
    const pairs = [];
    for (const [ci, arr] of byCrease) {
      arr.sort((x, y) => x.s - y.s);
      for (let k = 0; k + 1 < arr.length; k++) {
        if (arr[k].i !== arr[k + 1].i) pairs.push([arr[k].i, arr[k + 1].i]);
      }
    }
    let anyMissing = pairs.some(([a, b]) => !edgeSet.has(a < b ? a + ',' + b : b + ',' + a));
    if (anyMissing) {
      tris = enforceEdges(px, py, tris, pairs);
      edgeSet.clear();
      for (let t = 0; t < tris.length; t += 3) {
        for (let k = 0; k < 3; k++) {
          const a = tris[t + k], b = tris[t + (k + 1) % 3];
          edgeSet.add(a < b ? a + ',' + b : b + ',' + a);
        }
      }
    }
    const creaseEdges = new Map();
    let missing = 0;
    for (const [ci, arr] of byCrease) {
      for (let k = 0; k + 1 < arr.length; k++) {
        const a = arr[k].i, b = arr[k + 1].i;
        if (a === b) continue;
        const key = a < b ? a + ',' + b : b + ',' + a;
        if (edgeSet.has(key)) {
          creaseEdges.set(key, { ci, s0: arr[k].s, s1: arr[k + 1].s });
        } else missing++;
      }
    }
    const uv = new Float64Array(pts.length * 2);
    pts.forEach((p, i) => { uv[i * 2] = p[0]; uv[i * 2 + 1] = p[1]; });
    return { uv, tris, kind, creaseSample, creaseEdges, missingCreaseEdges: missing };
  }

  // ------------------------------------------------------ state transfer
  // Interpolate world positions/velocities for newUV from an old mesh using
  // barycentric coordinates in material space.
  function transfer(oldUV, oldTris, oldPos, oldVel, newUV) {
    const nOld = oldUV.length / 2;
    // bucket old triangles
    const G = 48;
    const buckets = Array.from({ length: G * G }, () => []);
    for (let t = 0; t < oldTris.length; t += 3) {
      const a = oldTris[t] * 2, b = oldTris[t + 1] * 2, c = oldTris[t + 2] * 2;
      const u0 = Math.min(oldUV[a], oldUV[b], oldUV[c]), u1 = Math.max(oldUV[a], oldUV[b], oldUV[c]);
      const v0 = Math.min(oldUV[a + 1], oldUV[b + 1], oldUV[c + 1]), v1 = Math.max(oldUV[a + 1], oldUV[b + 1], oldUV[c + 1]);
      const cu0 = Math.max(0, Math.floor(u0 * G)), cu1 = Math.min(G - 1, Math.floor(u1 * G));
      const cv0 = Math.max(0, Math.floor(v0 * G)), cv1 = Math.min(G - 1, Math.floor(v1 * G));
      for (let cu = cu0; cu <= cu1; cu++) for (let cv = cv0; cv <= cv1; cv++) buckets[cv * G + cu].push(t);
    }
    const nNew = newUV.length / 2;
    const pos = new Float64Array(nNew * 3);
    const vel = new Float64Array(nNew * 3);
    const bary = (t, u, v) => {
      const a = oldTris[t] * 2, b = oldTris[t + 1] * 2, c = oldTris[t + 2] * 2;
      const x1 = oldUV[a], y1 = oldUV[a + 1], x2 = oldUV[b], y2 = oldUV[b + 1], x3 = oldUV[c], y3 = oldUV[c + 1];
      const den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
      if (Math.abs(den) < 1e-14) return null;
      const w1 = ((y2 - y3) * (u - x3) + (x3 - x2) * (v - y3)) / den;
      const w2 = ((y3 - y1) * (u - x3) + (x1 - x3) * (v - y3)) / den;
      return [w1, w2, 1 - w1 - w2];
    };
    for (let i = 0; i < nNew; i++) {
      const u = newUV[i * 2], v = newUV[i * 2 + 1];
      const cu = Math.max(0, Math.min(G - 1, Math.floor(u * G)));
      const cv = Math.max(0, Math.min(G - 1, Math.floor(v * G)));
      let best = null, bestScore = -1e9;
      for (const t of buckets[cv * G + cu]) {
        const w = bary(t, u, v);
        if (!w) continue;
        const score = Math.min(w[0], w[1], w[2]);
        if (score > bestScore) { bestScore = score; best = { t, w }; }
        if (score >= -1e-9) break;
      }
      if (!best) {
        // fallback: nearest old vertex
        let bi = 0, bd = 1e9;
        for (let o = 0; o < nOld; o++) {
          const d = Math.hypot(oldUV[o * 2] - u, oldUV[o * 2 + 1] - v);
          if (d < bd) { bd = d; bi = o; }
        }
        pos[i * 3] = oldPos[bi * 3]; pos[i * 3 + 1] = oldPos[bi * 3 + 1]; pos[i * 3 + 2] = oldPos[bi * 3 + 2];
        vel[i * 3] = oldVel[bi * 3]; vel[i * 3 + 1] = oldVel[bi * 3 + 1]; vel[i * 3 + 2] = oldVel[bi * 3 + 2];
        continue;
      }
      const { t, w } = best;
      for (let d = 0; d < 3; d++) {
        pos[i * 3 + d] =
          w[0] * oldPos[oldTris[t] * 3 + d] +
          w[1] * oldPos[oldTris[t + 1] * 3 + d] +
          w[2] * oldPos[oldTris[t + 2] * 3 + d];
        vel[i * 3 + d] =
          w[0] * oldVel[oldTris[t] * 3 + d] +
          w[1] * oldVel[oldTris[t + 1] * 3 + d] +
          w[2] * oldVel[oldTris[t + 2] * 3 + d];
      }
    }
    return { pos, vel };
  }

  return { buildMesh, transfer, segDist, segIntersect };
});

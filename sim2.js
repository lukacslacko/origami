/*
 * sim2.js — v2 adaptive paper engine.
 *
 * Same physical model as sim.js (XPBD sheet: stretch, dihedral bending,
 * desk + layer friction, actuator colliders, drag) but on an ADAPTIVE,
 * crease-conforming triangle mesh (mesh2.js):
 *
 *  - Curvature drives the mesh. Where the sheet bends tighter than a
 *    refinement radius, the region is remeshed finer; where it bends tighter
 *    than the crease radius, a CREASE is born: the over-limit hinges are
 *    clustered, a straight line is fitted through them in material space,
 *    and the mesh is rebuilt with that line as mesh edges. No pressure
 *    heuristic — the geometric criterion works because the mesh can refine.
 *  - Creases are persistent material-space segments carrying a plastic
 *    rest-angle profile. They survive every remesh, can be re-bent and
 *    re-pressed (their profile keeps flowing), and can cross each other.
 *  - State transfers between meshes by barycentric interpolation in material
 *    space; scenario scripts, actuators, and the renderer are unchanged.
 *
 * Runs in the browser (global SIM2, needs MESH2) and in Node.
 */
(function (root, factory) {
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = factory(require('./mesh2.js'));
  } else {
    root.SIM2 = factory(root.MESH2);
  }
})(typeof self !== 'undefined' ? self : this, function (MESH2) {
  'use strict';

  const PARAMS = {
    hMax: 1 / 22,        // coarse spacing (flat regions)
    hMin: 1 / 88,        // fine spacing (crease bands, high curvature)
    substeps: 24,
    gravity: 10,
    airDrag: 1.2,
    normalDrag: 2,
    settleDrag: 2.2,
    alphaBend: 1e-3,     // bending compliance at unit aspect (continuum B = 1/alpha)
    bendDamp: 0.3,       // base hinge damping; scaled per hinge by hRef/lperp
    creaseSoften: 1.4,
    plasticRate: 0.6,
    maxTheta0: 2.9,
    rRefine: 0.05,       // bend radius that triggers refinement (incipient creases only)
    rCrease: 0.022,      // bend radius that becomes a crease (~1.5x paper thickness)
    muStatic: 0.22,
    muKinetic: 0.13,
    muPaper: 0.35,
    selfR: 0.014,
    rho: 400,            // areal density (matches v1: mass 1 per (1/20)^2 cell)
    // actuator geometry (matches v1 / the renderer)
    padR: 0.026, padLift: 0.032, stemR: 0.010, fingerStemR: 0.016, stemLen: 0.62,
    detectEvery: 12,     // frames between curvature-detection passes
    remeshCooldown: 24,  // min frames between remeshes
    maxCreases: 12,
  };
  const H_REF = 1 / 20;

  function wrapPi(a) {
    a = (a + Math.PI) % (2 * Math.PI);
    if (a < 0) a += 2 * Math.PI;
    return a - Math.PI;
  }

  // ---------------------------------------------------------- construction
  function collectAnchors(scenario) {
    const set = new Map();
    const add = (u, v) => set.set(u.toFixed(6) + ',' + v.toFixed(6), [u, v]);
    add(0, 0); add(1, 0); add(1, 1); add(0, 1);
    if (scenario) {
      for (const ph of scenario.phases) {
        for (const k of [0, 0.5, 1]) {
          let w;
          try { w = ph.fn(k); } catch (e) { continue; }
          for (const g of (w && w.grabs) || []) add(g.at[0], g.at[1]);
        }
      }
    }
    return [...set.values()];
  }

  function makePaper(scenario) {
    const paper = {
      params: PARAMS,
      creases: [],          // {a, b, dir, len, prof: Float64Array, sign}
      sizeGrid: null,       // {n, data} refinement field for the mesher
      hotUntil: null,       // per-cell timestamps for coarsening
      anchorsReq: collectAnchors(scenario),
      time: 0, frame: 0, cooldown: 0,
      remeshCount: 0,
    };
    const mesh = MESH2.buildMesh({
      hMax: PARAMS.hMax, hMin: PARAMS.hMin,
      creases: paper.creases, sizeGrid: null, anchors: paper.anchorsReq,
    });
    adoptMesh(paper, mesh, null);
    return paper;
  }

  // Build all physics arrays from a mesh; transferSrc = old {uv,tris,pos,vel}
  function adoptMesh(paper, mesh, transferSrc) {
    const n = mesh.uv.length / 2;
    const uv = mesh.uv, tris = mesh.tris;
    let pos, vel;
    if (transferSrc) {
      const r = MESH2.transfer(transferSrc.uv, transferSrc.tris, transferSrc.pos, transferSrc.vel, uv);
      pos = r.pos; vel = r.vel;
      // mild damp of interpolation noise (heavy damping here bled the energy
      // out of standing folds across repeated remeshes)
      for (let i = 0; i < vel.length; i++) vel[i] *= 0.8;
    } else {
      pos = new Float64Array(n * 3);
      vel = new Float64Array(n * 3);
      for (let i = 0; i < n; i++) { pos[i * 3] = uv[i * 2]; pos[i * 3 + 2] = uv[i * 2 + 1]; }
    }
    // lumped masses
    const area = new Float64Array(n);
    for (let t = 0; t < tris.length; t += 3) {
      const A = tris[t] * 2, B = tris[t + 1] * 2, C = tris[t + 2] * 2;
      const ar = 0.5 * Math.abs((uv[B] - uv[A]) * (uv[C + 1] - uv[A + 1]) - (uv[B + 1] - uv[A + 1]) * (uv[C] - uv[A]));
      area[tris[t]] += ar / 3; area[tris[t + 1]] += ar / 3; area[tris[t + 2]] += ar / 3;
    }
    const baseInvMass = new Float64Array(n);
    for (let i = 0; i < n; i++) baseInvMass[i] = 1 / Math.max(1e-9, PARAMS.rho * area[i]);
    // edges + interior-edge adjacency for hinges
    const edgeInfo = new Map(); // key -> {a, b, t1: {tri, opp}, t2}
    for (let t = 0; t < tris.length; t += 3) {
      for (let k = 0; k < 3; k++) {
        const a = tris[t + k], b = tris[t + (k + 1) % 3], o = tris[t + (k + 2) % 3];
        const key = a < b ? a + ',' + b : b + ',' + a;
        let e = edgeInfo.get(key);
        if (!e) { e = { a: Math.min(a, b), b: Math.max(a, b), sides: [] }; edgeInfo.set(key, e); }
        e.sides.push({ fwd: a < b, opp: o }); // fwd: edge appears as (min,max) in CCW order
      }
    }
    const edges = [];
    const hinges = [];
    for (const [key, e] of edgeInfo) {
      const du = uv[e.a * 2] - uv[e.b * 2], dv = uv[e.a * 2 + 1] - uv[e.b * 2 + 1];
      const rest = Math.hypot(du, dv);
      edges.push({ a: e.a, b: e.b, rest, alpha: 0 });
      if (e.sides.length === 2) {
        // wings ordered so the sign convention matches v1 (both up => theta>0)
        let w0 = -1, w1 = -1;
        for (const s of e.sides) {
          if (s.fwd) w0 = s.opp; else w1 = s.opp;
        }
        if (w0 < 0 || w1 < 0) continue; // non-manifold guard
        // areas -> perpendicular extent
        const A1 = triArea(uv, e.a, e.b, w0), A2 = triArea(uv, e.a, e.b, w1);
        const lperp = Math.max(1e-6, (A1 + A2) / Math.max(rest, 1e-9));
        const cref = mesh.creaseEdges.get(key) || null;
        const h = {
          e0: e.a, e1: e.b, w0, w1,
          theta0: 0, thC: 0, soft: false,
          u: (uv[e.a * 2] + uv[e.b * 2]) / 2,
          v: (uv[e.a * 2 + 1] + uv[e.b * 2 + 1]) / 2,
          lperp, elen: rest,
          alphaH: PARAMS.alphaBend * (lperp / Math.max(rest, 1e-9)),
          damp: PARAMS.bendDamp * Math.max(1, Math.pow((H_REF / lperp) * 0.8, 0.6)),
          cref,
        };
        if (cref) {
          const c = paper.creases[cref.ci];
          h.theta0 = sampleProfile(c, (cref.s0 + cref.s1) / 2);
          h.soft = true;
        }
        hinges.push(h);
      }
    }
    // initialize continuous angles near their rest pose (important after
    // remesh when folds sit near +-pi)
    // (done lazily in the first solve; here seed thC from geometry)
    // anchors
    const anchorMap = new Map();
    for (const [u, v] of paper.anchorsReq) {
      let bi = -1, bd = 1e9;
      for (let i = 0; i < n; i++) {
        const d = Math.hypot(uv[i * 2] - u, uv[i * 2 + 1] - v);
        if (d < bd) { bd = d; bi = i; }
      }
      anchorMap.set(u.toFixed(6) + ',' + v.toFixed(6), bi);
    }
    // flat (SoA) mirrors of the constraint data for the hot solver loops
    const ne = edges.length, nh = hinges.length;
    const eIdx = new Int32Array(ne * 2);
    const eRest = new Float64Array(ne);
    let minEdge = 1;
    edges.forEach((e, i) => {
      eIdx[i * 2] = e.a; eIdx[i * 2 + 1] = e.b; eRest[i] = e.rest;
      if (e.rest < minEdge) minEdge = e.rest;
    });
    paper.minEdge = Math.max(minEdge, 1e-4);
    const hIdx = new Int32Array(nh * 4);
    const hTheta0 = new Float64Array(nh);
    const hThC = new Float64Array(nh);
    const hAlphaHard = new Float64Array(nh); // alphaH, pre-softened where creased
    const hDamp = new Float64Array(nh);
    const hLperp = new Float64Array(nh);
    hinges.forEach((h, i) => {
      hIdx[i * 4] = h.e0; hIdx[i * 4 + 1] = h.e1; hIdx[i * 4 + 2] = h.w0; hIdx[i * 4 + 3] = h.w1;
      hTheta0[i] = h.theta0;
      hAlphaHard[i] = h.soft ? h.alphaH * PARAMS.creaseSoften : h.alphaH;
      hDamp[i] = h.damp;
      hLperp[i] = h.lperp;
    });
    Object.assign(paper, {
      NP: n, uv, tris, pos, vel,
      prev: new Float64Array(n * 3),
      invMass: new Float64Array(n),
      baseInvMass,
      nrm: new Float64Array(n * 3),
      edges, hinges, anchorMap,
      eIdx, eRest, hIdx, hTheta0, hThC, hAlphaHard, hDamp, hLperp,
    });
    // seed thC from current geometry, unwrapped near theta0
    paper.hinges.forEach((h, i) => {
      const th = hingeAngle(paper.pos, h, false);
      if (th === th) h.thC = h.theta0 + wrapPi(th - h.theta0);
      paper.hThC[i] = h.thC;
    });
  }
  function triArea(uv, a, b, c) {
    return 0.5 * Math.abs(
      (uv[b * 2] - uv[a * 2]) * (uv[c * 2 + 1] - uv[a * 2 + 1]) -
      (uv[b * 2 + 1] - uv[a * 2 + 1]) * (uv[c * 2] - uv[a * 2]));
  }

  // crease profiles: uniform bins over the segment length
  function makeProfile(len) {
    const nb = Math.max(2, Math.ceil(len / PARAMS.hMin) + 1);
    return new Float64Array(nb).fill(NaN);
  }
  function profileFillGaps(prof, fallback) {
    // nearest-valid fill
    const nb = prof.length;
    for (let i = 0; i < nb; i++) {
      if (prof[i] === prof[i]) continue;
      let best = fallback, bd = 1e9;
      for (let j = 0; j < nb; j++) {
        if (prof[j] !== prof[j]) continue;
        const d = Math.abs(j - i);
        if (d < bd) { bd = d; best = prof[j]; }
      }
      prof[i] = best;
    }
  }
  function sampleProfile(c, s) {
    const nb = c.prof.length;
    const x = Math.max(0, Math.min(nb - 1, (s / c.len) * (nb - 1)));
    const i = Math.floor(x), f = x - i;
    const a = c.prof[i], b = c.prof[Math.min(nb - 1, i + 1)];
    return a * (1 - f) + b * f;
  }
  function writeProfile(c, s, t0) {
    const nb = c.prof.length;
    const i = Math.round(Math.max(0, Math.min(nb - 1, (s / c.len) * (nb - 1))));
    c.prof[i] = t0;
  }

  // ---------------------------------------------------- signed hinge angle
  const _g = new Float64Array(12);
  function hingeAngle(pos, h, wantGrad) {
    const p0 = h.e0 * 3, p1 = h.e1 * 3, p2 = h.w0 * 3, p3 = h.w1 * 3;
    const x0x = pos[p0], x0y = pos[p0 + 1], x0z = pos[p0 + 2];
    const ex = pos[p1] - x0x, ey = pos[p1 + 1] - x0y, ez = pos[p1 + 2] - x0z;
    const ax = pos[p2] - x0x, ay = pos[p2 + 1] - x0y, az = pos[p2 + 2] - x0z;
    const bx = pos[p3] - x0x, by = pos[p3 + 1] - x0y, bz = pos[p3 + 2] - x0z;
    const n1x = ey * az - ez * ay, n1y = ez * ax - ex * az, n1z = ex * ay - ey * ax;
    const n2x = by * ez - bz * ey, n2y = bz * ex - bx * ez, n2z = bx * ey - by * ex;
    const elen = Math.sqrt(ex * ex + ey * ey + ez * ez);
    const n1sq = n1x * n1x + n1y * n1y + n1z * n1z;
    const n2sq = n2x * n2x + n2y * n2y + n2z * n2z;
    if (elen < 1e-10 || n1sq < 1e-16 || n2sq < 1e-16) return NaN;
    const cx = n1y * n2z - n1z * n2y, cy = n1z * n2x - n1x * n2z, cz = n1x * n2y - n1y * n2x;
    const theta = Math.atan2((cx * ex + cy * ey + cz * ez) / elen, n1x * n2x + n1y * n2y + n1z * n2z);
    if (!wantGrad) return theta;
    const s2 = -elen / n1sq, s3 = -elen / n2sq;
    const g2x = s2 * n1x, g2y = s2 * n1y, g2z = s2 * n1z;
    const g3x = s3 * n2x, g3y = s3 * n2y, g3z = s3 * n2z;
    const inv_ee = 1 / (elen * elen);
    const a2 = (ax * ex + ay * ey + az * ez) * inv_ee - 1;
    const a3 = (bx * ex + by * ey + bz * ez) * inv_ee - 1;
    _g[0] = a2 * g2x + a3 * g3x; _g[1] = a2 * g2y + a3 * g3y; _g[2] = a2 * g2z + a3 * g3z;
    _g[3] = -(1 + a2) * g2x - (1 + a3) * g3x;
    _g[4] = -(1 + a2) * g2y - (1 + a3) * g3y;
    _g[5] = -(1 + a2) * g2z - (1 + a3) * g3z;
    _g[6] = g2x; _g[7] = g2y; _g[8] = g2z;
    _g[9] = g3x; _g[10] = g3y; _g[11] = g3z;
    return theta;
  }

  // flat-index variant of hingeAngle for the solver loop
  function hingeAngleFlat(pos, hIdx, k, wantGrad) {
    const p0 = hIdx[k * 4] * 3, p1 = hIdx[k * 4 + 1] * 3, p2 = hIdx[k * 4 + 2] * 3, p3 = hIdx[k * 4 + 3] * 3;
    const x0x = pos[p0], x0y = pos[p0 + 1], x0z = pos[p0 + 2];
    const ex = pos[p1] - x0x, ey = pos[p1 + 1] - x0y, ez = pos[p1 + 2] - x0z;
    const ax = pos[p2] - x0x, ay = pos[p2 + 1] - x0y, az = pos[p2 + 2] - x0z;
    const bx = pos[p3] - x0x, by = pos[p3 + 1] - x0y, bz = pos[p3 + 2] - x0z;
    const n1x = ey * az - ez * ay, n1y = ez * ax - ex * az, n1z = ex * ay - ey * ax;
    const n2x = by * ez - bz * ey, n2y = bz * ex - bx * ez, n2z = bx * ey - by * ex;
    const elen = Math.sqrt(ex * ex + ey * ey + ez * ez);
    const n1sq = n1x * n1x + n1y * n1y + n1z * n1z;
    const n2sq = n2x * n2x + n2y * n2y + n2z * n2z;
    if (elen < 1e-10 || n1sq < 1e-16 || n2sq < 1e-16) return NaN;
    const cx = n1y * n2z - n1z * n2y, cy = n1z * n2x - n1x * n2z, cz = n1x * n2y - n1y * n2x;
    const theta = Math.atan2((cx * ex + cy * ey + cz * ez) / elen, n1x * n2x + n1y * n2y + n1z * n2z);
    if (!wantGrad) return theta;
    const s2 = -elen / n1sq, s3 = -elen / n2sq;
    const g2x = s2 * n1x, g2y = s2 * n1y, g2z = s2 * n1z;
    const g3x = s3 * n2x, g3y = s3 * n2y, g3z = s3 * n2z;
    const inv_ee = 1 / (elen * elen);
    const a2 = (ax * ex + ay * ey + az * ez) * inv_ee - 1;
    const a3 = (bx * ex + by * ey + bz * ez) * inv_ee - 1;
    _g[0] = a2 * g2x + a3 * g3x; _g[1] = a2 * g2y + a3 * g3y; _g[2] = a2 * g2z + a3 * g3z;
    _g[3] = -(1 + a2) * g2x - (1 + a3) * g3x;
    _g[4] = -(1 + a2) * g2y - (1 + a3) * g3y;
    _g[5] = -(1 + a2) * g2z - (1 + a3) * g3z;
    _g[6] = g2x; _g[7] = g2y; _g[8] = g2z;
    _g[9] = g3x; _g[10] = g3y; _g[11] = g3z;
    return theta;
  }

  function vertexNormals(paper) {
    const { pos, tris, nrm } = paper;
    nrm.fill(0);
    for (let t = 0; t < tris.length; t += 3) {
      const a = tris[t] * 3, b = tris[t + 1] * 3, c = tris[t + 2] * 3;
      const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2];
      const vx = pos[c] - pos[a], vy = pos[c + 1] - pos[a + 1], vz = pos[c + 2] - pos[a + 2];
      const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
      nrm[a] += nx; nrm[a + 1] += ny; nrm[a + 2] += nz;
      nrm[b] += nx; nrm[b + 1] += ny; nrm[b + 2] += nz;
      nrm[c] += nx; nrm[c + 1] += ny; nrm[c + 2] += nz;
    }
  }

  // -------------------------------------------------------------- stepping
  const _colBuf = new Float64Array(24 * 7);
  const _hash = new Map();
  const _pool = [];
  function stepFrame(paper, worldAt, dtFrame) {
    const P = PARAMS;
    // stiffness propagates ~one constraint per Gauss-Seidel pass, so the
    // substep count must grow with resolution or fine sheets go towel-soft.
    // Scale by the finest edge ACTUALLY in the mesh, not the Detail setting.
    const sub = Math.round(P.substeps * Math.max(
      1,
      (0.5 / 88) / paper.minEdge,
      Math.pow((1 / 22) / P.hMax, 1.5)));
    const dt = dtFrame / sub;
    const { pos, prev, vel, invMass, baseInvMass, edges, hinges, uv } = paper;
    const pinIdx = paper.anchorMap.get('0.000000,0.000000');

    for (let s = 0; s < sub; s++) {
      const t = paper.time + dt * (s + 1);
      const world = worldAt(t);
      const grabs = world.grabs || [];
      const fingers = world.fingers || (world.finger ? [world.finger] : []);

      invMass.set(baseInvMass);
      if (world.pinA) invMass[pinIdx] = 0;
      const grabIds = [];
      for (const g of grabs) {
        const vid = paper.anchorMap.get(g.at[0].toFixed(6) + ',' + g.at[1].toFixed(6));
        grabIds.push(vid);
        if (vid !== undefined && !g.soft) invMass[vid] = 0;
      }

      // integrate (vertex normals for drag refreshed on alternating substeps)
      if ((s & 1) === 0) vertexNormals(paper);
      const nrm = paper.nrm;
      const dragK = Math.exp(-P.airDrag * dt);
      const nDragK = 1 - Math.exp(-P.normalDrag * dt);
      for (let i = 0; i < paper.NP; i++) {
        const p = i * 3;
        prev[p] = pos[p]; prev[p + 1] = pos[p + 1]; prev[p + 2] = pos[p + 2];
        if (invMass[i] === 0) continue;
        vel[p + 1] -= P.gravity * dt;
        let d = dragK;
        if (pos[p + 1] < 0.01) d *= Math.exp(-P.settleDrag * dt);
        vel[p] *= d; vel[p + 1] *= d; vel[p + 2] *= d;
        const nx = nrm[p], ny = nrm[p + 1], nz = nrm[p + 2];
        const nsq = nx * nx + ny * ny + nz * nz;
        if (nsq > 1e-16) {
          const vn = (vel[p] * nx + vel[p + 1] * ny + vel[p + 2] * nz) / nsq * nDragK;
          vel[p] -= vn * nx; vel[p + 1] -= vn * ny; vel[p + 2] -= vn * nz;
        }
        pos[p] += vel[p] * dt;
        pos[p + 1] += vel[p + 1] * dt;
        pos[p + 2] += vel[p + 2] * dt;
      }
      if (world.pinA && pinIdx !== undefined) {
        pos[pinIdx * 3] = 0; pos[pinIdx * 3 + 1] = 0; pos[pinIdx * 3 + 2] = 0;
      }
      grabs.forEach((g, gi) => {
        const vid = grabIds[gi];
        if (vid === undefined) return;
        const p = vid * 3;
        if (g.soft) {
          const k = 0.3;
          pos[p] += (g.pos[0] - pos[p]) * k;
          pos[p + 1] += (g.pos[1] - pos[p + 1]) * k;
          pos[p + 2] += (g.pos[2] - pos[p + 2]) * k;
        } else {
          pos[p] = g.pos[0]; pos[p + 1] = g.pos[1]; pos[p + 2] = g.pos[2];
        }
      });

      // distance constraints (hard, flat arrays)
      const eIdx = paper.eIdx, eRest = paper.eRest, nE = eRest.length;
      for (let k = 0; k < nE; k++) {
        const ia = eIdx[k * 2], ib = eIdx[k * 2 + 1];
        const pa = ia * 3, pb = ib * 3;
        const wa = invMass[ia], wb = invMass[ib];
        const ws = wa + wb;
        if (ws === 0) continue;
        let dx = pos[pb] - pos[pa], dy = pos[pb + 1] - pos[pa + 1], dz = pos[pb + 2] - pos[pa + 2];
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (len < 1e-12) continue;
        const f = -(len - eRest[k]) / (ws * len);
        dx *= f; dy *= f; dz *= f;
        pos[pa] -= wa * dx; pos[pa + 1] -= wa * dy; pos[pa + 2] -= wa * dz;
        pos[pb] += wb * dx; pos[pb + 1] += wb * dy; pos[pb + 2] += wb * dz;
      }

      // hinges (flat arrays; thC lives in hThC, synced to objects per frame)
      const invDt2 = 1 / (dt * dt);
      const hIdx = paper.hIdx, hTheta0 = paper.hTheta0, hThC = paper.hThC;
      const hAlphaHard = paper.hAlphaHard, hDamp = paper.hDamp, hLperp = paper.hLperp;
      const nH = hTheta0.length;
      for (let k = 0; k < nH; k++) {
        const thRaw = hingeAngleFlat(pos, hIdx, k, true);
        if (thRaw !== thRaw) continue;
        const th = hThC[k] + wrapPi(thRaw - hThC[k]);
        hThC[k] = th;
        const Cc = th - hTheta0[k];
        const g = _g;
        const i0 = hIdx[k * 4], i1 = hIdx[k * 4 + 1], i2 = hIdx[k * 4 + 2], i3 = hIdx[k * 4 + 3];
        const w0 = invMass[i0], w1 = invMass[i1], w2 = invMass[i2], w3 = invMass[i3];
        const q0 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
        const q1 = g[3] * g[3] + g[4] * g[4] + g[5] * g[5];
        const q2 = g[6] * g[6] + g[7] * g[7] + g[8] * g[8];
        const q3 = g[9] * g[9] + g[10] * g[10] + g[11] * g[11];
        const damp = hDamp[k];
        let denom = (w0 * q0 + w1 * q1 + w2 * q2 + w3 * q3) * (1 + damp) + hAlphaHard[k] * invDt2;
        if (denom < 1e-12) continue;
        let Cdot = 0;
        let pp = i0 * 3;
        Cdot += g[0] * (pos[pp] - prev[pp]) + g[1] * (pos[pp + 1] - prev[pp + 1]) + g[2] * (pos[pp + 2] - prev[pp + 2]);
        pp = i1 * 3;
        Cdot += g[3] * (pos[pp] - prev[pp]) + g[4] * (pos[pp + 1] - prev[pp + 1]) + g[5] * (pos[pp + 2] - prev[pp + 2]);
        pp = i2 * 3;
        Cdot += g[6] * (pos[pp] - prev[pp]) + g[7] * (pos[pp + 1] - prev[pp + 1]) + g[8] * (pos[pp + 2] - prev[pp + 2]);
        pp = i3 * 3;
        Cdot += g[9] * (pos[pp] - prev[pp]) + g[10] * (pos[pp + 1] - prev[pp + 1]) + g[11] * (pos[pp + 2] - prev[pp + 2]);
        let dl = (-Cc - damp * Cdot) / denom;
        // displacement limiter (anti-ringing safety at fine bands)
        const gmax2 = Math.max(q0, q1, q2, q3);
        const wmax = Math.max(w0, w1, w2, w3);
        const maxMove = Math.abs(dl) * wmax * Math.sqrt(gmax2);
        const cap = 0.35 * hLperp[k];
        if (maxMove > cap) dl *= cap / maxMove;
        let p = i0 * 3;
        pos[p] += w0 * dl * g[0]; pos[p + 1] += w0 * dl * g[1]; pos[p + 2] += w0 * dl * g[2];
        p = i1 * 3;
        pos[p] += w1 * dl * g[3]; pos[p + 1] += w1 * dl * g[4]; pos[p + 2] += w1 * dl * g[5];
        p = i2 * 3;
        pos[p] += w2 * dl * g[6]; pos[p + 1] += w2 * dl * g[7]; pos[p + 2] += w2 * dl * g[8];
        p = i3 * 3;
        pos[p] += w3 * dl * g[9]; pos[p + 1] += w3 * dl * g[10]; pos[p + 2] += w3 * dl * g[11];
      }

      // actuator colliders
      let nc = 0;
      const cb = _colBuf;
      const addCol = (ax_, ay_, az_, bx_, by_, bz_, r_) => {
        const o = nc * 7;
        cb[o] = ax_; cb[o + 1] = ay_; cb[o + 2] = az_;
        cb[o + 3] = bx_; cb[o + 4] = by_; cb[o + 5] = bz_;
        cb[o + 6] = r_; nc++;
      };
      for (const f of fingers) {
        const hl = f.hl || 0, fx = f.ax || [1, 0, 0];
        addCol(f.c[0] - fx[0] * hl, f.c[1] - fx[1] * hl, f.c[2] - fx[2] * hl,
          f.c[0] + fx[0] * hl, f.c[1] + fx[1] * hl, f.c[2] + fx[2] * hl, f.r);
        addCol(f.c[0], f.c[1], f.c[2], f.c[0], f.c[1] + P.stemLen, f.c[2], P.fingerStemR);
      }
      if (world.pinA) {
        addCol(0, P.padLift, 0, 0, P.padLift, 0, P.padR);
        addCol(0, P.padLift, 0, -0.26, P.padLift + 0.56, -0.26, P.stemR);
      }
      for (const g of grabs) {
        const gy = g.pos[1] + P.padLift;
        addCol(g.pos[0], gy, g.pos[2], g.pos[0], gy, g.pos[2], P.padR);
      }
      for (let ci = 0; ci < nc; ci++) {
        const o = ci * 7;
        const ax_ = cb[o], ay_ = cb[o + 1], az_ = cb[o + 2];
        const abx = cb[o + 3] - ax_, aby = cb[o + 4] - ay_, abz = cb[o + 5] - az_;
        const abLen2 = abx * abx + aby * aby + abz * abz;
        const r = cb[o + 6] + 0.004, r2 = r * r;
        for (let i = 0; i < paper.NP; i++) {
          if (invMass[i] === 0) continue;
          const p = i * 3;
          let dx = pos[p] - ax_, dy = pos[p + 1] - ay_, dz = pos[p + 2] - az_;
          if (abLen2 > 1e-12) {
            let tt = (dx * abx + dy * aby + dz * abz) / abLen2;
            if (tt < 0) tt = 0; else if (tt > 1) tt = 1;
            dx -= tt * abx; dy -= tt * aby; dz -= tt * abz;
          }
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < r2 && d2 > 1e-12) {
            const d = Math.sqrt(d2), f = (r - d) / d;
            pos[p] += dx * f; pos[p + 1] += dy * f; pos[p + 2] += dz * f;
          }
        }
      }

      // self collision (world hash; skip material-space neighbors)
      if ((s & 1) === 0 || fingers.length) selfCollide(paper);

      // desk + friction, velocity update
      const invDt = 1 / dt;
      for (let i = 0; i < paper.NP; i++) {
        const p = i * 3;
        if (invMass[i] !== 0 && pos[p + 1] < 0) {
          const depth = -pos[p + 1];
          pos[p + 1] = 0;
          const tx = pos[p] - prev[p], tz = pos[p + 2] - prev[p + 2];
          const tl = Math.hypot(tx, tz);
          if (tl > 1e-12) {
            if (tl < P.muStatic * depth) {
              pos[p] = prev[p]; pos[p + 2] = prev[p + 2];
            } else {
              const drop = Math.min(tl, P.muKinetic * depth);
              const f = (tl - drop) / tl;
              pos[p] = prev[p] + tx * f;
              pos[p + 2] = prev[p + 2] + tz * f;
            }
          }
        } else if (invMass[i] === 0 && pos[p + 1] < 0) pos[p + 1] = 0;
        vel[p] = (pos[p] - prev[p]) * invDt;
        vel[p + 1] = (pos[p + 1] - prev[p + 1]) * invDt;
        vel[p + 2] = (pos[p + 2] - prev[p + 2]) * invDt;
      }
    }
    paper.time += dtFrame;
    paper.frame++;
    // sync continuous angles back to the hinge objects (detection & seeding)
    for (let k = 0; k < hinges.length; k++) hinges[k].thC = paper.hThC[k];

    // plastic flow on crease hinges: same fingertip requirement as creation
    const worldNow = worldAt(paper.time);
    const fingersNow = worldNow.fingers || (worldNow.finger ? [worldNow.finger] : []);
    for (let hi = 0; hi < hinges.length; hi++) {
      const h = hinges[hi];
      if (!h.cref) continue;
      if (fingersNow.length === 0) continue;
      const mx = (pos[h.e0 * 3] + pos[h.e1 * 3]) / 2;
      const my = (pos[h.e0 * 3 + 1] + pos[h.e1 * 3 + 1]) / 2;
      const mz = (pos[h.e0 * 3 + 2] + pos[h.e1 * 3 + 2]) / 2;
      if (!nearAnyFinger(fingersNow, mx, my, mz)) continue;
      const th = hingeAngle(pos, h, false);
      if (th !== th) continue;
      const cont = h.thC + wrapPi(th - h.thC);
      const ex = cont - h.theta0;
      const yieldA = h.lperp / PARAMS.rCrease;
      if (Math.abs(ex) > yieldA) {
        const sg = Math.sign(ex);
        h.theta0 += PARAMS.plasticRate * (ex - sg * yieldA);
        h.theta0 = Math.max(-PARAMS.maxTheta0, Math.min(PARAMS.maxTheta0, h.theta0));
        paper.hTheta0[hi] = h.theta0;
        const c = paper.creases[h.cref.ci];
        writeProfile(c, (h.cref.s0 + h.cref.s1) / 2, h.theta0);
      }
    }

    if (paper.cooldown > 0) paper.cooldown--;
    if (paper.frame % PARAMS.detectEvery === 0 && paper.cooldown === 0) {
      const worldD = worldAt(paper.time);
      const holds = [];
      if (worldD.pinA) holds.push([0, 0, 0]);
      for (const g of (worldD.grabs || [])) holds.push(g.pos);
      detectAndAdapt(paper, fingersNow, holds);
    }
  }

  // flat-grid self collision: dense typed-array grid over the current bbox
  let _heads = new Int32Array(0);
  let _next = new Int32Array(0);
  function selfCollide(paper) {
    const P = PARAMS;
    const r = P.selfR, cell = r * 2.0001, inv = 1 / cell;
    const { pos, prev, invMass, uv } = paper;
    const NP = paper.NP;
    const skipR2 = (P.selfR * 1.5) * (P.selfR * 1.5);
    // bbox
    let x0 = 1e9, y0 = 1e9, z0 = 1e9, x1 = -1e9, y1 = -1e9, z1 = -1e9;
    for (let i = 0; i < NP; i++) {
      const p = i * 3;
      if (pos[p] < x0) x0 = pos[p]; if (pos[p] > x1) x1 = pos[p];
      if (pos[p + 1] < y0) y0 = pos[p + 1]; if (pos[p + 1] > y1) y1 = pos[p + 1];
      if (pos[p + 2] < z0) z0 = pos[p + 2]; if (pos[p + 2] > z1) z1 = pos[p + 2];
    }
    const nx = Math.min(256, Math.max(1, Math.floor((x1 - x0) * inv) + 1));
    const ny = Math.min(256, Math.max(1, Math.floor((y1 - y0) * inv) + 1));
    const nz = Math.min(256, Math.max(1, Math.floor((z1 - z0) * inv) + 1));
    const ncell = nx * ny * nz;
    if (_heads.length < ncell) _heads = new Int32Array(Math.ceil(ncell * 1.5));
    if (_next.length < NP) _next = new Int32Array(NP * 2);
    _heads.fill(-1, 0, ncell);
    const cx0 = x0, cy0 = y0, cz0 = z0;
    const cellOf = (p) => {
      let cx = ((pos[p] - cx0) * inv) | 0; if (cx >= nx) cx = nx - 1;
      let cy = ((pos[p + 1] - cy0) * inv) | 0; if (cy >= ny) cy = ny - 1;
      let cz = ((pos[p + 2] - cz0) * inv) | 0; if (cz >= nz) cz = nz - 1;
      return (cz * ny + cy) * nx + cx;
    };
    for (let i = 0; i < NP; i++) {
      const c = cellOf(i * 3);
      _next[i] = _heads[c];
      _heads[c] = i;
    }
    const r2 = r * r;
    for (let i = 0; i < NP; i++) {
      const p = i * 3;
      let cx = ((pos[p] - cx0) * inv) | 0; if (cx >= nx) cx = nx - 1;
      let cy = ((pos[p + 1] - cy0) * inv) | 0; if (cy >= ny) cy = ny - 1;
      let cz = ((pos[p + 2] - cz0) * inv) | 0; if (cz >= nz) cz = nz - 1;
      const ui = uv[i * 2], vi = uv[i * 2 + 1];
      const wi = invMass[i];
      for (let oz = -1; oz <= 1; oz++) {
        const zc = cz + oz; if (zc < 0 || zc >= nz) continue;
        for (let oy = -1; oy <= 1; oy++) {
          const yc = cy + oy; if (yc < 0 || yc >= ny) continue;
          for (let ox = -1; ox <= 1; ox++) {
            const xc = cx + ox; if (xc < 0 || xc >= nx) continue;
            let j = _heads[(zc * ny + yc) * nx + xc];
            for (; j >= 0; j = _next[j]) {
              if (j <= i) continue;
              const du = ui - uv[j * 2], dvv = vi - uv[j * 2 + 1];
              if (du * du + dvv * dvv < skipR2) continue;
              const q = j * 3;
              const dx = pos[q] - pos[p], dy = pos[q + 1] - pos[p + 1], dz = pos[q + 2] - pos[p + 2];
              const d2 = dx * dx + dy * dy + dz * dz;
              if (d2 < r2 && d2 > 1e-14) {
                const d = Math.sqrt(d2);
                const wj = invMass[j], ws = wi + wj;
                if (ws === 0) continue;
                const corr = 0.8 * (r - d) / d / ws;
                pos[p] -= wi * corr * dx; pos[p + 1] -= wi * corr * dy; pos[p + 2] -= wi * corr * dz;
                pos[q] += wj * corr * dx; pos[q + 1] += wj * corr * dy; pos[q + 2] += wj * corr * dz;
                const nnx = dx / d, nny = dy / d, nnz = dz / d;
                let sx = (pos[p] - prev[p]) - (pos[q] - prev[q]);
                let sy = (pos[p + 1] - prev[p + 1]) - (pos[q + 1] - prev[q + 1]);
                let sz = (pos[p + 2] - prev[p + 2]) - (pos[q + 2] - prev[q + 2]);
                const sn = sx * nnx + sy * nny + sz * nnz;
                sx -= sn * nnx; sy -= sn * nny; sz -= sn * nnz;
                const stl = Math.sqrt(sx * sx + sy * sy + sz * sz);
                if (stl > 1e-12) {
                  const f = 0.5 * Math.min(1, P.muPaper * (r - d) / stl);
                  const fi = (wi / ws) * f, fj = (wj / ws) * f;
                  pos[p] -= fi * sx; pos[p + 1] -= fi * sy; pos[p + 2] -= fi * sz;
                  pos[q] += fj * sx; pos[q + 1] += fj * sy; pos[q + 2] += fj * sz;
                }
              }
            }
          }
        }
      }
    }
  }

  // -------------------------------------------- curvature -> adapt / crease
  function nearAnyFinger(fingers, x, y, z) {
    for (const f of fingers) {
      const fr = f.r + 0.06, fx = f.ax || [1, 0, 0], hl = f.hl || 0;
      let dx = x - f.c[0], dy = y - f.c[1], dz = z - f.c[2];
      let sd = dx * fx[0] + dy * fx[1] + dz * fx[2];
      if (sd > hl) sd = hl; else if (sd < -hl) sd = -hl;
      dx -= sd * fx[0]; dy -= sd * fx[1]; dz -= sd * fx[2];
      if (dx * dx + dy * dy + dz * dz <= fr * fr) return true;
    }
    return false;
  }

  function detectAndAdapt(paper, fingersNow, holds) {
    const P = PARAMS;
    // refinement field on a 64^2 grid, with age-based coarsening
    const G = 64;
    if (!paper.sizeGrid) paper.sizeGrid = { n: G, data: new Float64Array(G * G).fill(P.hMax) };
    if (!paper.hotUntil) paper.hotUntil = new Float64Array(G * G).fill(-1);
    if (!paper.creaseStreak) paper.creaseStreak = new Uint8Array(G * G);
    if (!paper.refineStreak) paper.refineStreak = new Uint8Array(G * G);
    const streak = paper.creaseStreak;
    const rstreak = paper.refineStreak;
    const overNow = new Uint8Array(G * G);
    const refineNow = new Uint8Array(G * G);
    const sg = paper.sizeGrid.data, hot = paper.hotUntil;
    let changedCells = 0, coarsenCells = 0;

    const candidates = [];
    for (const h of paper.hinges) {
      const th = hingeAngle(paper.pos, h, false);
      if (th !== th) continue;
      const dev = h.cref ? Math.abs(wrapPi(th - h.theta0)) : Math.abs(th);
      const kappa = dev / h.lperp;
      if (kappa > 1 / P.rRefine) {
        // don't chase the fake curvature singularity at point-holds: a real
        // gripper holds an area; the kink at a held vertex is an artifact
        let atHold = false;
        if (holds && holds.length) {
          const mx = (paper.pos[h.e0 * 3] + paper.pos[h.e1 * 3]) / 2;
          const my = (paper.pos[h.e0 * 3 + 1] + paper.pos[h.e1 * 3 + 1]) / 2;
          const mz = (paper.pos[h.e0 * 3 + 2] + paper.pos[h.e1 * 3 + 2]) / 2;
          for (const hp of holds) {
            if (Math.hypot(mx - hp[0], my - hp[1], mz - hp[2]) < 0.07) { atHold = true; break; }
          }
        }
        if (atHold) continue;
        const cxi = Math.max(0, Math.min(G - 1, Math.floor(h.u * G)));
        const cyi = Math.max(0, Math.min(G - 1, Math.floor(h.v * G)));
        const id = cyi * G + cxi;
        refineNow[id] = 1;
        // refinement needs persistence: transient curls of a swinging sheet
        // must not smear expensive fine bands across the whole sheet. The
        // fingertip's neighborhood is deliberate work, not a transient —
        // refine there immediately so creases can form under the press.
        let atFinger = false;
        if (fingersNow.length) {
          const mfx = (paper.pos[h.e0 * 3] + paper.pos[h.e1 * 3]) / 2;
          const mfy = (paper.pos[h.e0 * 3 + 1] + paper.pos[h.e1 * 3 + 1]) / 2;
          const mfz = (paper.pos[h.e0 * 3 + 2] + paper.pos[h.e1 * 3 + 2]) / 2;
          atFinger = nearAnyFinger(fingersNow, mfx, mfy, mfz);
        }
        if (!atFinger && rstreak[id] < 1) continue;
        hot[id] = paper.time + 2.5;
        // size ~ measured bend radius (0.3 r), snapped to power-of-two levels
        // of hMin so gentle curls stay cheap at any Detail setting
        let want = Math.max(P.hMin, Math.min(P.hMax, 0.3 / kappa));
        let level = P.hMin;
        while (level * 2 <= want) level *= 2;
        want = Math.min(P.hMax, level);
        if (want < sg[id] - 1e-9) { sg[id] = want; changedCells++; }
      }
      if (!h.cref && kappa > 1 / P.rCrease) {
        // fit creases only from a RESOLVED ridge: coarse hinges give sparse,
        // noisy clusters and a wrong line; let refinement build the band first
        if (h.lperp > 2.5 * P.hMin) continue;
        // creasing requires a pressing fingertip: with a contact cushion ~20x
        // thicker than real paper, a fold merely laid closed reaches the same
        // radius as a pressed one, so geometry alone cannot distinguish them.
        // (A true force-based criterion needs force-limited contacts - future.)
        const mx = (paper.pos[h.e0 * 3] + paper.pos[h.e1 * 3]) / 2;
        const my = (paper.pos[h.e0 * 3 + 1] + paper.pos[h.e1 * 3 + 1]) / 2;
        const mz = (paper.pos[h.e0 * 3 + 2] + paper.pos[h.e1 * 3 + 2]) / 2;
        if (!nearAnyFinger(fingersNow, mx, my, mz)) continue;
        // exclude shoulders of existing creases (parallel & adjacent)
        let shoulder = false;
        for (const c of paper.creases) {
          const d = MESH2.segDist(h.u, h.v, c.a[0], c.a[1], c.b[0], c.b[1]);
          if (d < 3.5 * P.hMin) {
            const eu = (paper.uv[h.e1 * 2] - paper.uv[h.e0 * 2]);
            const ev = (paper.uv[h.e1 * 2 + 1] - paper.uv[h.e0 * 2 + 1]);
            const el = Math.hypot(eu, ev) || 1;
            const dot = Math.abs((eu * c.dir[0] + ev * c.dir[1]) / el);
            // only clearly transverse hinges may seed a crossing crease here
            if (dot > 0.5) { shoulder = true; break; }
          }
        }
        if (!shoulder) {
          const cxi = Math.max(0, Math.min(G - 1, Math.floor(h.u * G)));
          const cyi = Math.max(0, Math.min(G - 1, Math.floor(h.v * G)));
          overNow[cyi * G + cxi] = 1;
          if (streak[cyi * G + cxi] >= 1) candidates.push({ u: h.u, v: h.v, th, w: kappa });
        }
      }
    }
    for (let id = 0; id < G * G; id++) {
      streak[id] = overNow[id] ? Math.min(250, streak[id] + 1) : 0;
      rstreak[id] = refineNow[id] ? Math.min(250, rstreak[id] + 1) : 0;
    }
    // age-based coarsening (never coarsen crease bands: mesher re-imposes them)
    for (let id = 0; id < G * G; id++) {
      if (sg[id] < P.hMax && hot[id] >= 0 && paper.time > hot[id] + 0.5) {
        sg[id] = P.hMax; hot[id] = -1; coarsenCells++;
      }
    }

    // cluster candidates -> fit straight creases
    let newCrease = false;
    if (candidates.length >= 3 && paper.creases.length < P.maxCreases) {
      const parent = candidates.map((_, i) => i);
      const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
      for (let i = 0; i < candidates.length; i++) {
        for (let j = i + 1; j < candidates.length; j++) {
          if (Math.hypot(candidates[i].u - candidates[j].u, candidates[i].v - candidates[j].v) < 0.06) {
            parent[find(i)] = find(j);
          }
        }
      }
      const clusters = new Map();
      candidates.forEach((c, i) => {
        const r = find(i);
        let arr = clusters.get(r);
        if (!arr) { arr = []; clusters.set(r, arr); }
        arr.push(c);
      });
      let made = 0;
      for (const arr of clusters.values()) {
        if (arr.length < 5 || made >= 1) continue;
        // weighted PCA line fit
        let W = 0, mu = 0, mv = 0;
        for (const c of arr) { W += c.w; mu += c.w * c.u; mv += c.w * c.v; }
        mu /= W; mv /= W;
        let sxx = 0, sxy = 0, syy = 0;
        for (const c of arr) {
          const du = c.u - mu, dv = c.v - mv;
          sxx += c.w * du * du; sxy += c.w * du * dv; syy += c.w * dv * dv;
        }
        const tr = sxx + syy, det = sxx * syy - sxy * sxy;
        const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
        const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
        // reject blob-like clusters: a forming crease is a ridge, not a patch
        if (l2 > 0.18 * l1) continue;
        let dx = sxy, dy = l1 - sxx;
        if (Math.hypot(dx, dy) < 1e-9) { dx = 1; dy = 0; }
        const dl = Math.hypot(dx, dy); dx /= dl; dy /= dl;
        // extent along the line
        let s0 = 1e9, s1 = -1e9, thMean = 0;
        for (const c of arr) {
          const sProj = (c.u - mu) * dx + (c.v - mv) * dy;
          s0 = Math.min(s0, sProj); s1 = Math.max(s1, sProj);
          thMean += c.th;
        }
        thMean /= arr.length;
        // a NEW crease must come from a substantial pressed ridge at once —
        // the short transient bulge ahead of a shepherding slide must not
        // seed a (mis-oriented) crease that extension then locks in
        const clusterExtent = s1 - s0;
        s0 -= 0.02; s1 += 0.02;
        let A = [mu + s0 * dx, mv + s0 * dy];
        let B = [mu + s1 * dx, mv + s1 * dy];
        const clampPt = (p2) => [Math.max(0, Math.min(1, p2[0])), Math.max(0, Math.min(1, p2[1]))];
        A = clampPt(A); B = clampPt(B);
        // try extending an existing collinear crease instead of a new one
        let extended = false;
        for (const c of paper.creases) {
          const dot = Math.abs(dx * c.dir[0] + dy * c.dir[1]);
          const off = MESH2.segDist(mu, mv, c.a[0], c.a[1], c.b[0], c.b[1]);
          if (dot > 0.9 && off < 0.06) {
            // project all four endpoints on c's line, take the union extent
            const base = c.a, cd = c.dir;
            const proj = (p2) => (p2[0] - base[0]) * cd[0] + (p2[1] - base[1]) * cd[1];
            const lo = Math.min(0, proj(A), proj(B));
            const hi = Math.max(c.len, proj(A), proj(B));
            const na = [base[0] + lo * cd[0], base[1] + lo * cd[1]];
            const nb = [base[0] + hi * cd[0], base[1] + hi * cd[1]];
            const oldLen = c.len, oldProf = c.prof, oldOffset = -lo;
            c.a = clampPt(na); c.b = clampPt(nb);
            c.len = Math.hypot(c.b[0] - c.a[0], c.b[1] - c.a[1]);
            c.dir = [(c.b[0] - c.a[0]) / c.len, (c.b[1] - c.a[1]) / c.len];
            const nprof = makeProfile(c.len);
            for (let i2 = 0; i2 < oldProf.length; i2++) {
              if (oldProf[i2] !== oldProf[i2]) continue;
              const sOld = (i2 / (oldProf.length - 1)) * oldLen;
              writeProfile(c, sOld + oldOffset, oldProf[i2]);
              const nb2 = nprof.length;
              const bi = Math.round(((sOld + oldOffset) / c.len) * (nb2 - 1));
              if (bi >= 0 && bi < nb2) nprof[bi] = oldProf[i2];
            }
            c.prof = nprof;
            // seed the new stretch with the cluster angle
            for (const cc of arr) {
              const sp = (cc.u - c.a[0]) * c.dir[0] + (cc.v - c.a[1]) * c.dir[1];
              writeProfile(c, sp, cc.th);
            }
            profileFillGaps(c.prof, thMean);
            extended = true;
            break;
          }
        }
        if (!extended) {
          if (clusterExtent < 0.14) continue;
          const len = Math.hypot(B[0] - A[0], B[1] - A[1]);
          if (len < 0.18) continue;
          const c = {
            a: A, b: B, len,
            dir: [(B[0] - A[0]) / len, (B[1] - A[1]) / len],
            prof: makeProfile(len),
          };
          for (const cc of arr) {
            const sp = (cc.u - A[0]) * c.dir[0] + (cc.v - A[1]) * c.dir[1];
            writeProfile(c, sp, cc.th);
          }
          profileFillGaps(c.prof, thMean);
          paper.creases.push(c);
        }
        made++;
        newCrease = true;
      }
    }

    if (newCrease) mergeCreases(paper);
    if (newCrease || changedCells >= 4 || coarsenCells >= 12) {
      remesh(paper);
      paper.cooldown = P.remeshCooldown;
    }
  }


  // merge near-collinear overlapping crease fragments into single segments
  function mergeCreases(paper) {
    for (let pass = 0; pass < 4; pass++) {
      let merged = false;
      outer:
      for (let i = 0; i < paper.creases.length; i++) {
        for (let j = i + 1; j < paper.creases.length; j++) {
          const A = paper.creases[i], B = paper.creases[j];
          const dot = Math.abs(A.dir[0] * B.dir[0] + A.dir[1] * B.dir[1]);
          if (dot < 0.92) continue;
          const dEnds = Math.min(
            MESH2.segDist(B.a[0], B.a[1], A.a[0], A.a[1], A.b[0], A.b[1]),
            MESH2.segDist(B.b[0], B.b[1], A.a[0], A.a[1], A.b[0], A.b[1]));
          if (dEnds > 0.07) continue;
          // union extent along A's line
          const base = A.a, cd = A.dir;
          const proj = (p2) => (p2[0] - base[0]) * cd[0] + (p2[1] - base[1]) * cd[1];
          const lo = Math.min(0, proj(B.a), proj(B.b));
          const hi = Math.max(A.len, proj(B.a), proj(B.b));
          const na = [base[0] + lo * cd[0], base[1] + lo * cd[1]];
          const nb = [base[0] + hi * cd[0], base[1] + hi * cd[1]];
          const oldA = { len: A.len, prof: A.prof, off: -lo };
          const oldB = { c: B };
          A.a = [Math.max(0, Math.min(1, na[0])), Math.max(0, Math.min(1, na[1]))];
          A.b = [Math.max(0, Math.min(1, nb[0])), Math.max(0, Math.min(1, nb[1]))];
          A.len = Math.hypot(A.b[0] - A.a[0], A.b[1] - A.a[1]);
          A.dir = [(A.b[0] - A.a[0]) / A.len, (A.b[1] - A.a[1]) / A.len];
          A.prof = makeProfile(A.len);
          for (let k2 = 0; k2 < oldA.prof.length; k2++) {
            if (oldA.prof[k2] !== oldA.prof[k2]) continue;
            writeProfile(A, (k2 / (oldA.prof.length - 1)) * oldA.len + oldA.off, oldA.prof[k2]);
          }
          for (let k2 = 0; k2 < B.prof.length; k2++) {
            if (B.prof[k2] !== B.prof[k2]) continue;
            const sB = (k2 / (B.prof.length - 1)) * B.len;
            const p2 = [B.a[0] + sB * B.dir[0], B.a[1] + sB * B.dir[1]];
            writeProfile(A, (p2[0] - A.a[0]) * A.dir[0] + (p2[1] - A.a[1]) * A.dir[1], B.prof[k2]);
          }
          profileFillGaps(A.prof, 0);
          paper.creases.splice(j, 1);
          merged = true;
          break outer;
        }
      }
      if (!merged) break;
    }
  }

  function remesh(paper) {
    const old = { uv: paper.uv, tris: paper.tris, pos: paper.pos, vel: paper.vel };
    const mesh = MESH2.buildMesh({
      hMax: PARAMS.hMax, hMin: PARAMS.hMin,
      creases: paper.creases, sizeGrid: paper.sizeGrid, anchors: paper.anchorsReq,
    });
    adoptMesh(paper, mesh, old);
    paper.remeshCount++;
  }

  // ---------------------------------------------------------------- extras
  function metrics(paper) {
    const { pos, hinges } = paper;
    let maxY = -1e9, maxSpeed = 0, nan = false;
    for (let i = 0; i < paper.NP; i++) {
      const p = i * 3;
      if (!isFinite(pos[p]) || !isFinite(pos[p + 1]) || !isFinite(pos[p + 2])) nan = true;
      if (pos[p + 1] > maxY) maxY = pos[p + 1];
      const v = Math.hypot(paper.vel[p], paper.vel[p + 1], paper.vel[p + 2]);
      if (v > maxSpeed) maxSpeed = v;
    }
    const anchor = (u, v) => {
      const vid = paper.anchorMap.get(u.toFixed(6) + ',' + v.toFixed(6));
      return vid === undefined ? [0, 0, 0] : [pos[vid * 3], pos[vid * 3 + 1], pos[vid * 3 + 2]];
    };
    let maxTheta0 = 0;
    const creased = [];
    for (const h of hinges) {
      if (Math.abs(h.theta0) > maxTheta0) maxTheta0 = Math.abs(h.theta0);
      if (Math.abs(h.theta0) > 0.15) creased.push({ u: h.u, v: h.v, theta0: h.theta0 });
    }
    return {
      maxY, maxSpeed, nan, maxTheta0, creases: creased,
      A: anchor(0, 0), B: anchor(1, 0), C: anchor(1, 1), D: anchor(0, 1),
      nVerts: paper.NP, nCreaseSegs: paper.creases.length,
      remeshCount: paper.remeshCount,
      creaseLines: paper.creases.map(c => ({ a: c.a.slice(), b: c.b.slice(), len: c.len })),
    };
  }

  return { PARAMS, makePaper, stepFrame, metrics, hingeAngle, wrapPi };
});

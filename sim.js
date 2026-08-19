/*
 * sim.js — simplified physical model of a square sheet of paper.
 *
 * Model: XPBD (extended position-based dynamics, small-substeps flavour).
 *  - Particles on an N x N grid.
 *  - Distance constraints: structural edges, cell diagonals (both directions;
 *    one of them is the triangulation edge, the other pure shear).
 *  - Signed dihedral hinge constraints over every interior edge -> bending.
 *  - Plasticity: when a hinge bends beyond a yield angle its rest angle flows
 *    toward the current angle and the hinge is weakened -> a permanent crease.
 *  - Desk: y=0 plane with Coulomb-ish stick/slip friction resolved in the
 *    position projection.
 *  - Actuators (kinematic): "pin" (press a corner to the desk), "grab"
 *    (hold/move a corner), "finger" (sphere collider).
 *
 * The mesh diagonals are aligned with the B-D diagonal of the square so a
 * fold of corner C onto corner A can crease cleanly along its natural line.
 *
 * Runs both in the browser (global SIM) and in Node (module.exports).
 */
(function (root, factory) {
  if (typeof module !== 'undefined' && module.exports) module.exports = factory();
  else root.SIM = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ---------------------------------------------------------------- params
  const PARAMS = {
    N: 25,               // vertices per side (runtime-configurable; paper rebuild needed)
    SIDE: 1.0,           // paper edge length (world units)
    substeps: 15,        // physics substeps per 1/60 s frame at the N=21 baseline;
                         // scaled with mesh resolution in stepFrame (PBD stiffness
                         // propagates one constraint per substep, so a finer mesh
                         // needs proportionally more substeps to feel equally stiff)
    gravity: 10,         // units / s^2 (tuned for papery droop at this scale)
    airDrag: 1.2,        // 1/s isotropic velocity damping
    normalDrag: 2,       // 1/s extra drag along the sheet normal (air resists broadside motion)
    settleDrag: 2.2,     // extra 1/s damping applied near the desk (calms settling)
    alphaStretch: 0.0,   // XPBD compliance, structural edges (0 = rigid projection)
    alphaShear: 1e-6,    // shear diagonals slightly softer
    alphaBend: 1e-3,     // hinge compliance (higher = floppier paper)
    bendDamp: 0.3,       // hinge constraint damping (kills Gauss-Seidel ringing at contacts)
    creaseSoften: 1.4,   // compliance multiplier for a creased hinge (mild damage)
    yieldK: 13,          // yield curvature (rad per unit length); per-hinge yield angle = yieldK * H
    plasticRate: 0.6,    // fraction of the excess absorbed per frame
    creaseMark: 0.25,    // rad of rest angle that counts as "creased"
    maxTheta0: 2.85,     // rad clamp for rest angle
    muStatic: 0.25,      // desk static friction
    muKinetic: 0.15,     // desk kinetic friction
    muPaper: 0.35,       // paper-on-paper friction (stabilizes a rolling fold)
    selfR: 0.014,        // self-collision radius (~paper double thickness)
    fingerR: 0.08,       // pressing-finger capsule radius
    padR: 0.026,         // pin/grip pad radius (collides with the sheet)
    padLift: 0.032,      // pad center height above its hold point
    stemR: 0.010,        // actuator stem radius (collides with the sheet)
    fingerStemR: 0.016,  // fingertip stem radius
    stemLen: 0.62,       // stem length above the pad / fingertip
    fingerSpeed: 1.0,    // scenario-3 finger tempo multiplier (higher = faster)
    slideGap: 0.020,     // clearance under the fingertip while sliding (crushing the
                         // doubled sheet thinner than ~2*selfR makes it accordion)
    squeezeGap: 0.008,   // clearance at the final squeeze
    fingerEnd: 0.62,     // x=z coordinate where the slide stops (chases the fold crest)
  };

  // reference grid spacing (the N=21 baseline the parameters were tuned at);
  // particle mass scales with H^2 so areal density — and therefore the
  // droop/slip/crease behavior — is resolution-independent
  const H0 = PARAMS.SIDE / 20;

  // ------------------------------------------------------------- vec utils
  function wrapPi(a) {
    a = (a + Math.PI) % (2 * Math.PI);
    if (a < 0) a += 2 * Math.PI;
    return a - Math.PI;
  }

  // ---------------------------------------------------------- construction
  function makePaper() {
    const N = Math.max(9, PARAMS.N | 0);
    const H = PARAMS.SIDE / (N - 1);
    const NP = N * N;
    const idx = (i, j) => i * N + j; // i -> z row, j -> x column
    const corners = {
      A: idx(0, 0),          // (0,0)
      B: idx(0, N - 1),      // (1,0)
      C: idx(N - 1, N - 1),  // (1,1)
      D: idx(N - 1, 0),      // (0,1)
    };
    const w0 = (H0 / H) * (H0 / H); // invMass; mass per particle ~ H^2
    const pos = new Float64Array(NP * 3);
    const prev = new Float64Array(NP * 3);
    const vel = new Float64Array(NP * 3);
    const invMass = new Float64Array(NP);
    const baseInvMass = new Float64Array(NP);
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const p = idx(i, j) * 3;
        pos[p] = j * H;      // x
        pos[p + 1] = 0;      // y (desk)
        pos[p + 2] = i * H;  // z
        invMass[idx(i, j)] = w0;
        baseInvMass[idx(i, j)] = w0;
      }
    }
    prev.set(pos);

    // --- distance constraints
    const edges = []; // {a, b, rest, alpha}
    const pairKey = (a, b) => (a < b ? a * NP + b : b * NP + a);
    const connected = new Set();
    function addEdge(a, b, alpha) {
      const dx = pos[a * 3] - pos[b * 3];
      const dz = pos[a * 3 + 2] - pos[b * 3 + 2];
      edges.push({ a, b, rest: Math.hypot(dx, dz), alpha });
      connected.add(pairKey(a, b));
    }
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N - 1; j++) addEdge(idx(i, j), idx(i, j + 1), PARAMS.alphaStretch);
    for (let i = 0; i < N - 1; i++)
      for (let j = 0; j < N; j++) addEdge(idx(i, j), idx(i + 1, j), PARAMS.alphaStretch);
    for (let i = 0; i < N - 1; i++) {
      for (let j = 0; j < N - 1; j++) {
        // triangulation diagonal v10-v01 (parallel to the B-D diagonal)
        addEdge(idx(i, j + 1), idx(i + 1, j), PARAMS.alphaStretch);
        // shear diagonal v00-v11
        addEdge(idx(i, j), idx(i + 1, j + 1), PARAMS.alphaShear);
      }
    }

    // --- triangles (for rendering & metrics)
    // cell (i,j): T1 = (v00, v01, v10), T2 = (v10, v01, v11)  (normals +y when flat)
    const tris = [];
    for (let i = 0; i < N - 1; i++) {
      for (let j = 0; j < N - 1; j++) {
        const v00 = idx(i, j), v10 = idx(i, j + 1), v01 = idx(i + 1, j), v11 = idx(i + 1, j + 1);
        tris.push(v00, v01, v10, v10, v01, v11);
      }
    }

    // --- hinges: {e0, e1, w0, w1, theta0, soft, u, v}
    // edge (e0,e1), wing w0 in triangle (e0,e1,w0), wing w1 in triangle (e0,w1,e1)
    const hinges = [];
    function addHinge(e0, e1, w0, w1) {
      const u = (pos[e0 * 3] + pos[e1 * 3]) / 2;
      const v = (pos[e0 * 3 + 2] + pos[e1 * 3 + 2]) / 2;
      // thC: continuously-tracked angle — a fold pressed fully flat sits at
      // theta = +-pi, right on the atan2 wrap, where a wrapped difference
      // flips sign every substep and makes the solver buzz
      hinges.push({ e0, e1, w0, w1, theta0: 0, thC: 0, soft: false, u, v });
    }
    for (let i = 0; i < N - 1; i++) {
      for (let j = 0; j < N - 1; j++) {
        // diagonal hinge inside each cell: edge v10-v01, wings v00 and v11
        addHinge(idx(i, j + 1), idx(i + 1, j), idx(i, j), idx(i + 1, j + 1));
      }
    }
    // vertical interior edges (i,j)-(i+1,j) for interior columns j
    for (let j = 1; j < N - 1; j++) {
      for (let i = 0; i < N - 1; i++) {
        addHinge(idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j - 1));
      }
    }
    // horizontal interior edges (i,j)-(i,j+1) for interior rows i
    for (let i = 1; i < N - 1; i++) {
      for (let j = 0; j < N - 1; j++) {
        addHinge(idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i - 1, j + 1));
      }
    }

    return {
      params: PARAMS, N, H, NP, corners,
      pos, prev, vel, invMass, baseInvMass,
      edges, hinges, tris, connected, pairKey,
      nrm: new Float64Array(NP * 3),
      time: 0,
    };
  }

  // area-weighted vertex normals (for broadside air drag)
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

  // ---------------------------------------------------- signed hinge angle
  // Returns theta (NaN when degenerate); with wantGrad, writes the 4 gradients
  // into the shared scratch _g (12 floats). Convention: both wings lifting
  // toward +y from a flat sheet => theta > 0. Allocation-free (hot path).
  const _g = new Float64Array(12);
  function hingeAngle(pos, h, wantGrad) {
    const p0 = h.e0 * 3, p1 = h.e1 * 3, p2 = h.w0 * 3, p3 = h.w1 * 3;
    const x0x = pos[p0], x0y = pos[p0 + 1], x0z = pos[p0 + 2];
    const ex = pos[p1] - x0x, ey = pos[p1 + 1] - x0y, ez = pos[p1 + 2] - x0z;
    const ax = pos[p2] - x0x, ay = pos[p2 + 1] - x0y, az = pos[p2 + 2] - x0z;
    const bx = pos[p3] - x0x, by = pos[p3 + 1] - x0y, bz = pos[p3 + 2] - x0z;
    // n1 = e x (x2-x0), n2 = (x3-x0) x e  -> parallel when flat
    const n1x = ey * az - ez * ay, n1y = ez * ax - ex * az, n1z = ex * ay - ey * ax;
    const n2x = by * ez - bz * ey, n2y = bz * ex - bx * ez, n2z = bx * ey - by * ex;
    const elen = Math.sqrt(ex * ex + ey * ey + ez * ez);
    const n1sq = n1x * n1x + n1y * n1y + n1z * n1z;
    const n2sq = n2x * n2x + n2y * n2y + n2z * n2z;
    if (elen < 1e-10 || n1sq < 1e-14 || n2sq < 1e-14) return NaN;
    const cx = n1y * n2z - n1z * n2y, cy = n1z * n2x - n1x * n2z, cz = n1x * n2y - n1y * n2x;
    const sinT = (cx * ex + cy * ey + cz * ez) / elen;
    const cosT = n1x * n2x + n1y * n2y + n1z * n2z;
    const theta = Math.atan2(sinT, cosT);
    if (!wantGrad) return theta;
    // grad wrt wings: rotating a wing about the edge; direction is the
    // (un-normalized) triangle normal; verified by finite differences.
    const s2 = -elen / n1sq, s3 = -elen / n2sq;
    const g2x = s2 * n1x, g2y = s2 * n1y, g2z = s2 * n1z;
    const g3x = s3 * n2x, g3y = s3 * n2y, g3z = s3 * n2z;
    const inv_ee = 1 / (elen * elen);
    const a2 = (ax * ex + ay * ey + az * ez) * inv_ee - 1; // (x2-x1).e / |e|^2
    const a3 = (bx * ex + by * ey + bz * ez) * inv_ee - 1; // (x3-x1).e / |e|^2
    // grad0 = a2*g2 + a3*g3 ; grad1 = -(1+a2)*g2 - (1+a3)*g3 ... signs verified by FD
    const g0x = a2 * g2x + a3 * g3x, g0y = a2 * g2y + a3 * g3y, g0z = a2 * g2z + a3 * g3z;
    const g1x = -(1 + a2) * g2x - (1 + a3) * g3x;
    const g1y = -(1 + a2) * g2y - (1 + a3) * g3y;
    const g1z = -(1 + a2) * g2z - (1 + a3) * g3z;
    _g[0] = g0x; _g[1] = g0y; _g[2] = g0z;
    _g[3] = g1x; _g[4] = g1y; _g[5] = g1z;
    _g[6] = g2x; _g[7] = g2y; _g[8] = g2z;
    _g[9] = g3x; _g[10] = g3y; _g[11] = g3z;
    return theta;
  }

  // -------------------------------------------------------------- stepping
  // world = {
  //   pinA: bool,
  //   grabs: [{at: [u,v] material coords, pos: [x,y,z]}] | undefined,
  //   finger: {c:[x,y,z], r, ax, hl} | null,
  // }
  const matIdx = (paper, u, v) =>
    Math.round(v / paper.H) * paper.N + Math.round(u / paper.H);
  function stepFrame(paper, worldAt, dtFrame) {
    const P = PARAMS;
    // quadratic in resolution: a chain of M hard constraints keeps only
    // ~S/M^2 of its stiffness after S Gauss-Seidel substeps
    const kRes = (paper.N - 1) / 20;
    const sub = Math.max(8, Math.round(P.substeps * kRes * kRes));
    const dt = dtFrame / sub;
    const { pos, prev, vel, invMass, baseInvMass, edges, hinges } = paper;
    const A = paper.corners.A;

    for (let s = 0; s < sub; s++) {
      const t = paper.time + dt * (s + 1);
      const world = worldAt(t);
      const grabs = world.grabs || [];

      // kinematic flags (soft grips keep their mass and are pulled, not welded)
      invMass.set(baseInvMass);
      if (world.pinA) invMass[A] = 0;
      for (const g of grabs) if (!g.soft) invMass[matIdx(paper, g.at[0], g.at[1])] = 0;

      // integrate
      vertexNormals(paper);
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
        // broadside air drag: damp the velocity component along the sheet normal
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
      // kinematic targets; soft grips pull with finite strength instead, so a
      // clamped flap can give a little rather than trap and wrinkle material
      if (world.pinA) { pos[A * 3] = 0; pos[A * 3 + 1] = 0; pos[A * 3 + 2] = 0; }
      for (const g of grabs) {
        const p = matIdx(paper, g.at[0], g.at[1]) * 3;
        if (g.soft) {
          const k = 0.3; // per-substep pull fraction toward the grip target
          pos[p] += (g.pos[0] - pos[p]) * k;
          pos[p + 1] += (g.pos[1] - pos[p + 1]) * k;
          pos[p + 2] += (g.pos[2] - pos[p + 2]) * k;
        } else {
          pos[p] = g.pos[0]; pos[p + 1] = g.pos[1]; pos[p + 2] = g.pos[2];
        }
      }

      // --- distance constraints
      for (let k = 0; k < edges.length; k++) {
        const e = edges[k];
        const pa = e.a * 3, pb = e.b * 3;
        const wa = invMass[e.a], wb = invMass[e.b];
        const wsum = wa + wb;
        if (wsum === 0) continue;
        let dx = pos[pb] - pos[pa], dy = pos[pb + 1] - pos[pa + 1], dz = pos[pb + 2] - pos[pa + 2];
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (len < 1e-12) continue;
        const Cc = len - e.rest;
        const alphaT = e.alpha / (dt * dt);
        const dl = -Cc / (wsum + alphaT);
        const f = dl / len;
        dx *= f; dy *= f; dz *= f;
        pos[pa] -= wa * dx; pos[pa + 1] -= wa * dy; pos[pa + 2] -= wa * dz;
        pos[pb] += wb * dx; pos[pb + 1] += wb * dy; pos[pb + 2] += wb * dz;
      }

      // --- hinge (bending / crease) constraints
      const alphaB = P.alphaBend / (dt * dt);
      const alphaBSoft = (P.alphaBend * P.creaseSoften) / (dt * dt);
      for (let k = 0; k < hinges.length; k++) {
        const h = hinges[k];
        const thRaw = hingeAngle(pos, h, true);
        if (thRaw !== thRaw) continue;
        const th = h.thC + wrapPi(thRaw - h.thC); // unwrap for continuity
        h.thC = th;
        const Cc = th - h.theta0;
        const g = _g;
        const w0 = invMass[h.e0], w1 = invMass[h.e1], w2 = invMass[h.w0], w3 = invMass[h.w1];
        let denom =
          w0 * (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]) +
          w1 * (g[3] * g[3] + g[4] * g[4] + g[5] * g[5]) +
          w2 * (g[6] * g[6] + g[7] * g[7] + g[8] * g[8]) +
          w3 * (g[9] * g[9] + g[10] * g[10] + g[11] * g[11]);
        denom = denom * (1 + P.bendDamp) + (h.soft ? alphaBSoft : alphaB);
        if (denom < 1e-12) continue;
        // damping term: oppose the constraint's rate of change this substep
        let Cdot = 0;
        let pp = h.e0 * 3;
        Cdot += g[0] * (pos[pp] - prev[pp]) + g[1] * (pos[pp + 1] - prev[pp + 1]) + g[2] * (pos[pp + 2] - prev[pp + 2]);
        pp = h.e1 * 3;
        Cdot += g[3] * (pos[pp] - prev[pp]) + g[4] * (pos[pp + 1] - prev[pp + 1]) + g[5] * (pos[pp + 2] - prev[pp + 2]);
        pp = h.w0 * 3;
        Cdot += g[6] * (pos[pp] - prev[pp]) + g[7] * (pos[pp + 1] - prev[pp + 1]) + g[8] * (pos[pp + 2] - prev[pp + 2]);
        pp = h.w1 * 3;
        Cdot += g[9] * (pos[pp] - prev[pp]) + g[10] * (pos[pp + 1] - prev[pp + 1]) + g[11] * (pos[pp + 2] - prev[pp + 2]);
        const dl = (-Cc - P.bendDamp * Cdot) / denom;
        let p = h.e0 * 3;
        pos[p] += w0 * dl * g[0]; pos[p + 1] += w0 * dl * g[1]; pos[p + 2] += w0 * dl * g[2];
        p = h.e1 * 3;
        pos[p] += w1 * dl * g[3]; pos[p + 1] += w1 * dl * g[4]; pos[p + 2] += w1 * dl * g[5];
        p = h.w0 * 3;
        pos[p] += w2 * dl * g[6]; pos[p + 1] += w2 * dl * g[7]; pos[p + 2] += w2 * dl * g[8];
        p = h.w1 * 3;
        pos[p] += w3 * dl * g[9]; pos[p + 1] += w3 * dl * g[10]; pos[p + 2] += w3 * dl * g[11];
      }

      // --- actuator colliders: every piece of hardware the paper could touch
      // (fingertip capsule, pin/grip pads, and all the stems) as capsules;
      // a sphere is a zero-length capsule
      let nc = 0;
      const cb = _colBuf;
      const addCol = (ax_, ay_, az_, bx_, by_, bz_, r_) => {
        const o = nc * 7;
        cb[o] = ax_; cb[o + 1] = ay_; cb[o + 2] = az_;
        cb[o + 3] = bx_; cb[o + 4] = by_; cb[o + 5] = bz_;
        cb[o + 6] = r_; nc++;
      };
      const fingers = world.fingers || (world.finger ? [world.finger] : []);
      for (const f of fingers) {
        const hl = f.hl || 0, fx = f.ax || [1, 0, 0];
        addCol(f.c[0] - fx[0] * hl, f.c[1] - fx[1] * hl, f.c[2] - fx[2] * hl,
          f.c[0] + fx[0] * hl, f.c[1] + fx[1] * hl, f.c[2] + fx[2] * hl, f.r);
        addCol(f.c[0], f.c[1], f.c[2], f.c[0], f.c[1] + P.stemLen, f.c[2], P.fingerStemR);
      }
      if (world.pinA) {
        // the pin's rod slants outward, away from the sheet, so a flap folded
        // onto corner A doesn't have to drape around it
        addCol(0, P.padLift, 0, 0, P.padLift, 0, P.padR);
        addCol(0, P.padLift, 0, -0.26, P.padLift + 0.56, -0.26, P.stemR);
      }
      // grips and hold-downs are low-profile pads (tweezer points, no rods)
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
            let t = (dx * abx + dy * aby + dz * abz) / abLen2;
            if (t < 0) t = 0; else if (t > 1) t = 1;
            dx -= t * abx; dy -= t * aby; dz -= t * abz;
          }
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < r2 && d2 > 1e-12) {
            const d = Math.sqrt(d2), f = (r - d) / d;
            pos[p] += dx * f; pos[p + 1] += dy * f; pos[p + 2] += dz * f;
          }
        }
      }

      // --- self collision (spatial hash); a flat-lying sheet cannot self-touch
      let anyHigh = false;
      for (let i = 0; i < paper.NP; i++) if (pos[i * 3 + 1] > 0.05) { anyHigh = true; break; }
      if (anyHigh || fingers.length) selfCollide(paper);

      // --- desk plane + friction
      for (let i = 0; i < paper.NP; i++) {
        if (invMass[i] === 0) {
          if (pos[i * 3 + 1] < 0) pos[i * 3 + 1] = 0;
          continue;
        }
        const p = i * 3;
        if (pos[p + 1] < 0) {
          const depth = -pos[p + 1];
          pos[p + 1] = 0;
          let tx = pos[p] - prev[p], tz = pos[p + 2] - prev[p + 2];
          const tl = Math.hypot(tx, tz);
          if (tl > 1e-12) {
            if (tl < P.muStatic * depth) {
              pos[p] = prev[p]; pos[p + 2] = prev[p + 2]; // static: stick
            } else {
              const drop = Math.min(tl, P.muKinetic * depth);
              const f = (tl - drop) / tl;
              pos[p] = prev[p] + tx * f;
              pos[p + 2] = prev[p + 2] + tz * f;
            }
          }
        }
      }

      // --- velocity update
      const invDt = 1 / dt;
      for (let i = 0; i < paper.NP; i++) {
        const p = i * 3;
        vel[p] = (pos[p] - prev[p]) * invDt;
        vel[p + 1] = (pos[p + 1] - prev[p + 1]) * invDt;
        vel[p + 2] = (pos[p + 2] - prev[p + 2]) * invDt;
      }
    }
    paper.time += dtFrame;

    // --- plasticity (once per frame): rest angle flows toward held angle.
    // Creasing needs pressure, not just a closed fold: at this mesh resolution a
    // fold that merely falls closed reads the same sharp hinge angle as a pressed
    // crease, so plastic flow is gated to hinges being squeezed by the finger.
    const world = worldAt(paper.time);
    const fingersNow = world.fingers || (world.finger ? [world.finger] : []);
    if (fingersNow.length) {
      const yieldA = PARAMS.yieldK * paper.H; // same yield *curvature* at every resolution
      for (let k = 0; k < hinges.length; k++) {
        const h = hinges[k];
        // distance from hinge midpoint to any fingertip's axis segment
        const mx0 = (paper.pos[h.e0 * 3] + paper.pos[h.e1 * 3]) / 2;
        const my0 = (paper.pos[h.e0 * 3 + 1] + paper.pos[h.e1 * 3 + 1]) / 2;
        const mz0 = (paper.pos[h.e0 * 3 + 2] + paper.pos[h.e1 * 3 + 2]) / 2;
        let pressed = false;
        for (const f of fingersNow) {
          const fr = f.r + 0.05, fax = f.ax || [1, 0, 0], fhl = f.hl || 0;
          const mx = mx0 - f.c[0], my = my0 - f.c[1], mz = mz0 - f.c[2];
          let sdot = mx * fax[0] + my * fax[1] + mz * fax[2];
          if (sdot > fhl) sdot = fhl; else if (sdot < -fhl) sdot = -fhl;
          const qx = mx - sdot * fax[0], qy = my - sdot * fax[1], qz = mz - sdot * fax[2];
          if (qx * qx + qy * qy + qz * qz <= fr * fr) { pressed = true; break; }
        }
        if (!pressed) continue;
        const thRaw = hingeAngle(paper.pos, h, false);
        if (thRaw !== thRaw) continue;
        const ex = h.thC + wrapPi(thRaw - h.thC) - h.theta0;
        if (Math.abs(ex) > yieldA) {
          const s = Math.sign(ex);
          h.theta0 += PARAMS.plasticRate * (ex - s * yieldA);
          if (h.theta0 > PARAMS.maxTheta0) h.theta0 = PARAMS.maxTheta0;
          if (h.theta0 < -PARAMS.maxTheta0) h.theta0 = -PARAMS.maxTheta0;
          if (Math.abs(h.theta0) > PARAMS.creaseMark) h.soft = true;
        }
      }
    }
  }

  const _colBuf = new Float64Array(16 * 7); // actuator collider scratch

  // self-collision via uniform spatial hash (cell arrays pooled across calls)
  const _hash = new Map();
  const _pool = [];
  function selfCollide(paper) {
    const P = PARAMS;
    const r = P.selfR, cell = r * 2.0001;
    const { pos, invMass, connected, pairKey } = paper;
    for (const arr of _hash.values()) { arr.length = 0; _pool.push(arr); }
    _hash.clear();
    for (let i = 0; i < paper.NP; i++) {
      const p = i * 3;
      const key =
        (Math.floor(pos[p] / cell) + 512) +
        (Math.floor(pos[p + 1] / cell) + 512) * 1024 +
        (Math.floor(pos[p + 2] / cell) + 512) * 1048576;
      let arr = _hash.get(key);
      if (!arr) { arr = _pool.pop() || []; _hash.set(key, arr); }
      arr.push(i);
    }
    const r2 = r * r;
    for (let i = 0; i < paper.NP; i++) {
      const p = i * 3;
      const cx = Math.floor(pos[p] / cell) + 512;
      const cy = Math.floor(pos[p + 1] / cell) + 512;
      const cz = Math.floor(pos[p + 2] / cell) + 512;
      for (let ox = -1; ox <= 1; ox++) for (let oy = -1; oy <= 1; oy++) for (let oz = -1; oz <= 1; oz++) {
        const arr = _hash.get((cx + ox) + (cy + oy) * 1024 + (cz + oz) * 1048576);
        if (!arr) continue;
        for (let n = 0; n < arr.length; n++) {
          const j = arr[n];
          if (j <= i) continue;
          if (connected.has(pairKey(i, j))) continue;
          const q = j * 3;
          const dx = pos[q] - pos[p], dy = pos[q + 1] - pos[p + 1], dz = pos[q + 2] - pos[p + 2];
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < r2 && d2 > 1e-14) {
            const d = Math.sqrt(d2);
            const wi = invMass[i], wj = invMass[j], ws = wi + wj;
            if (ws === 0) continue;
            // under-relaxed projections: full-stiff corrections here can
            // ping-pong against the stretch/bend solves and leave particles
            // buzzing in place
            const corr = 0.8 * (r - d) / d / ws;
            pos[p] -= wi * corr * dx; pos[p + 1] -= wi * corr * dy; pos[p + 2] -= wi * corr * dz;
            pos[q] += wj * corr * dx; pos[q + 1] += wj * corr * dy; pos[q + 2] += wj * corr * dz;
            // layer friction: damp relative tangential slip this substep,
            // budgeted by the normal correction (Coulomb-style)
            const nx = dx / d, ny = dy / d, nz = dz / d;
            const prev = paper.prev;
            let sx = (pos[p] - prev[p]) - (pos[q] - prev[q]);
            let sy = (pos[p + 1] - prev[p + 1]) - (pos[q + 1] - prev[q + 1]);
            let sz = (pos[p + 2] - prev[p + 2]) - (pos[q + 2] - prev[q + 2]);
            const sn = sx * nx + sy * ny + sz * nz;
            sx -= sn * nx; sy -= sn * ny; sz -= sn * nz;
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

  // -------------------------------------------------------------- scenarios
  const smooth = (k) => (k <= 0 ? 0 : k >= 1 ? 1 : k * k * (3 - 2 * k));
  const lerp3 = (a, b, k) => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];

  // Scenario scripts are built on demand so PARAMS edits (finger tempo, press
  // clearance, slide end) take effect on the next run.
  // Each phase: dur, label, fn(k) -> world  (k in [0,1], eased inside fn)
  function getScenarios() {
    const P = PARAMS;
    const fs = Math.max(0.1, P.fingerSpeed);
    const C0 = [P.SIDE, 0, P.SIDE];                // corner C at rest
    const LIFT = [0.74, 0.52, 0.74];               // lifted-C hold position
    const grabC = (pos) => [{ at: [1, 1], pos }];  // single grip on corner C
    // Scenario-3 carry: the flap (triangle B-C-D) is turned over the B-D
    // diagonal like a page. One grip can't do that to a floppy sheet — its
    // free corners sag and the fold arch topples off the line — so the flap
    // is carried on three grips (corner C + both edge midpoints), each
    // tracing the rigid rotation of its own material point about B-D.
    const FLAP_GRIPS = [[1, 1], [1, 0.5], [0.5, 1]];
    const rotBD = (u, v, phi) => {
      const d = (u + v - 1) / 2;         // foot offset: point = foot + d*(1,1)
      const r = d * Math.SQRT2;          // distance to the B-D line
      const c = Math.cos(phi), s = Math.sin(phi);
      const off = 0.015 * (1 - c) / 2;   // ramps 0 -> 0.015 so landing clears pin A
      const yfloor = 0.022 * (1 - c) / 2;
      return [u - d + c * d + off, Math.max(yfloor, r * s), v - d + c * d + off];
    };
    const flap = (phi) => FLAP_GRIPS.map(([u, v]) => ({ at: [u, v], pos: rotBD(u, v, phi) }));
    const FLAP_DOWN = flap(Math.PI);
    // edge grips force-limited: corner C stays welded, E1/E2 hold firmly but
    // can give a little, so pressing the fold doesn't trap and wrinkle material
    const FLAP_DOWN_SOFT = FLAP_DOWN.map((g, i) => (i === 0 ? g : { ...g, soft: true }));
    // while the fingertip irons, only corner C stays held — rigid edge grips
    // would trap the material the shrinking fold loop releases and buckle it
    const C_DOWN = [FLAP_DOWN[0]];
    // the "second hand": holds the bottom half flat while the flap is turned,
    // sliding out just before the flap lands on it
    const HOLDS = [
      { at: [0.55, 0.15], pos: [0.55, 0, 0.15] },
      { at: [0.15, 0.55], pos: [0.15, 0, 0.55] },
    ];
    const F_AX = [-Math.SQRT1_2, 0, Math.SQRT1_2]; // fingertip pad axis: along the B-D fold line
    const F_HL = 0.11;                             // fingertip pad half-length
    const pressY = P.fingerR + P.slideGap;
    const pressY2 = P.fingerR + P.squeezeGap;
    const F_START = [0.16, 0.6, 0.16];
    const F_DOWN = [0.16, pressY, 0.16];
    const F_END = [P.fingerEnd, pressY, P.fingerEnd];
    const F_SQZ = [P.fingerEnd, pressY2, P.fingerEnd];
    const F_UP = [P.fingerEnd, 0.7, P.fingerEnd];
    const fing = (c) => ({ c, r: P.fingerR, ax: F_AX, hl: F_HL });
    // Full-crease walk: with the flap grips held, the fold is an open standing
    // tube along x+z ~ 0.90. A dragged fingertip cannot squash it (line-on-line
    // contact — the tube rolls off the flank), but a vertical press traps it
    // against the desk, exactly like the centre squeeze. So the fingertip hops
    // along the edge and presses at overlapping stations.
    const SWC = 0.45;   // per-axis center of the fold tube
    const stHi = (s) => [SWC + F_AX[0] * s, 0.30, SWC + F_AX[2] * s];
    // each press lands just behind the crest and irons forward over it — a
    // purely vertical press lets the tube squirt out forward like a stepped-on
    // hose, while the forward push corners it against the held top layer
    const stBk = (s) => [SWC - 0.06 + F_AX[0] * s, P.fingerR + 0.010, SWC - 0.06 + F_AX[2] * s];
    const stFr = (s) => [SWC + 0.04 + F_AX[0] * s, P.fingerR + P.squeezeGap, SWC + 0.04 + F_AX[2] * s];
    const pressFn = (from, s) => (k) => {
      let c;
      if (k < 0.25) c = lerp3(from, stHi(s), smooth(k / 0.25));
      else if (k < 0.45) c = lerp3(stHi(s), stBk(s), smooth((k - 0.25) / 0.2));
      else if (k < 0.82) c = lerp3(stBk(s), stFr(s), smooth((k - 0.45) / 0.37));
      else c = lerp3(stFr(s), stHi(s), smooth((k - 0.82) / 0.18));
      return { pinA: true, grabs: FLAP_DOWN_SOFT, finger: fing(c) };
    };
    const STATIONS = [0, 0.15, 0.3, 0.45, -0.15, -0.3, -0.45];
    // scenario-5 helpers: three fingertips on the fold line; the center one
    // holds while the outer two sweep outward toward the corners. They land
    // slightly behind the crest and nudge forward over it (a vertical press
    // on the crest itself just squirts the fold tube away).
    const s5Y = P.fingerR + 0.008;
    const s5At = (s, y, back) => [SWC + back + F_AX[0] * s, y, SWC + back + F_AX[2] * s];
    const S5_BK = -0.06, S5_FR = 0.04;
    const S5_START = 0.24, S5_END = 0.62;
    // s4's approach glides above the sheet (no deep slide — pressing a held
    // flap while sliding wrinkles it; the stations do all the creasing)
    const glideY = P.fingerR + 0.026;
    const G_DOWN = [0.16, glideY, 0.16];
    const G_END = [P.fingerEnd, glideY, P.fingerEnd];
    return {
      's1': {
        title: 'Hold & bend — springs back',
        phases: [
          { dur: 0.8, label: 'Actuator 1 presses corner A onto the desk', fn: () => ({ pinA: true, finger: null }) },
          { dur: 1.4, label: 'Actuator 2 lifts the opposite corner C — the paper bends', fn: k => ({ pinA: true, grabs: grabC(lerp3(C0, LIFT, smooth(k))), finger: null }) },
          { dur: 1.0, label: 'Holding — corner A must not move', fn: () => ({ pinA: true, grabs: grabC(LIFT), finger: null }) },
          { dur: 2.2, label: 'Corner C released — the paper snaps back flat', fn: () => ({ pinA: true, finger: null }) },
          { dur: 1.5, label: 'Corner A released — nothing happens, the sheet is where it was', fn: () => ({ pinA: false, finger: null }) },
        ],
      },
      's2': {
        title: 'Release the held corner first — slips & shifts',
        phases: [
          { dur: 0.8, label: 'Actuator 1 presses corner A onto the desk', fn: () => ({ pinA: true, finger: null }) },
          { dur: 1.4, label: 'Actuator 2 lifts corner C — the paper bends', fn: k => ({ pinA: true, grabs: grabC(lerp3(C0, LIFT, smooth(k))), finger: null }) },
          { dur: 0.8, label: 'Holding', fn: () => ({ pinA: true, grabs: grabC(LIFT), finger: null }) },
          { dur: 2.0, label: 'Corner A released first — the sheet slips out and hangs, still bent', fn: () => ({ pinA: false, grabs: grabC(LIFT), finger: null }) },
          { dur: 2.6, label: 'Corner C released — the sheet settles flat as a square, but shifted', fn: () => ({ pinA: false, finger: null }) },
        ],
      },
      's3': {
        title: 'Fold over & pinch — a permanent crease',
        phases: [
          { dur: 0.8, label: 'Actuator 1 presses corner A onto the desk', fn: () => ({ pinA: true, finger: null }) },
          { dur: 3.6, label: 'Three grips turn the C-half over the diagonal; the other hand holds the sheet flat', fn: k => ({ pinA: true, grabs: k < 0.8 ? flap(Math.PI * smooth(k)).concat(HOLDS) : flap(Math.PI * smooth(k)), finger: null }) },
          { dur: 0.6, label: 'The folded half is held down — the sheet hinges at its middle', fn: () => ({ pinA: true, grabs: FLAP_DOWN, finger: null }) },
          { dur: 1.0 / fs, label: 'Actuator 3 (a fingertip) descends near the folded corner', fn: k => ({ pinA: true, grabs: FLAP_DOWN, finger: fing(lerp3(F_START, F_DOWN, smooth(k))) }) },
          { dur: 5.5 / fs, label: 'The fingertip slides from the corner toward the folded edge, pressing the fold flat', fn: k => ({ pinA: true, grabs: C_DOWN, finger: fing(lerp3(F_DOWN, F_END, smooth(k))) }) },
          { dur: 1.8 / fs, label: 'At the folded edge the fibres yield — a crease forms', fn: k => ({ pinA: true, grabs: C_DOWN, finger: fing(lerp3(F_END, F_SQZ, smooth(k))) }) },
          { dur: 1.4 / fs, label: 'The fingertip lifts away', fn: k => ({ pinA: true, grabs: C_DOWN, finger: fing(lerp3(F_SQZ, F_UP, smooth(k))) }) },
          { dur: 1.8, label: 'The grip eases the fold open before letting go', fn: k => ({ pinA: true, grabs: [flap(Math.PI * (1 - 0.55 * smooth(k)))[0]], finger: null }) },
          { dur: 4.0, label: 'All actuators released — the sheet relaxes, but the pinch crease stays', fn: () => ({ pinA: false, finger: null }) },
        ],
      },
      's4': {
        title: 'A full crease, worked along the edge — the fold stays',
        phases: [
          { dur: 0.8, label: 'Actuator 1 presses corner A onto the desk', fn: () => ({ pinA: true, finger: null }) },
          { dur: 3.6, label: 'Three grips turn the C-half over the diagonal; the other hand holds the sheet flat', fn: k => ({ pinA: true, grabs: k < 0.8 ? flap(Math.PI * smooth(k)).concat(HOLDS) : flap(Math.PI * smooth(k)), finger: null }) },
          { dur: 0.6, label: 'The folded half is held down — the sheet hinges at its middle', fn: () => ({ pinA: true, grabs: FLAP_DOWN, finger: null }) },
          { dur: 1.0 / fs, label: 'The fingertip descends near the folded corner', fn: k => ({ pinA: true, grabs: FLAP_DOWN_SOFT, finger: fing(lerp3(F_START, G_DOWN, smooth(k))) }) },
          { dur: 3.5 / fs, label: 'The fingertip glides out to the folded edge', fn: k => ({ pinA: true, grabs: FLAP_DOWN_SOFT, finger: fing(lerp3(G_DOWN, G_END, smooth(k))) }) },
          ...STATIONS.map((s, i) => ({
            dur: 1.5 / fs,
            label: `It works along the folded edge, pressing it flat (${i + 1} of ${STATIONS.length})`,
            fn: pressFn(i === 0 ? G_END : stHi(STATIONS[i - 1]), s),
          })),
          { dur: 1.2 / fs, label: 'The fingertip lifts away — the crease now runs corner to corner', fn: k => ({ pinA: true, grabs: FLAP_DOWN_SOFT, finger: fing(lerp3(stHi(STATIONS[STATIONS.length - 1]), F_UP, smooth(k))) }) },
          { dur: 1.0, label: 'The grips let go — a full crease holds the fold by itself', fn: () => ({ pinA: true, finger: null }) },
          { dur: 3.5, label: 'Everything released — the sheet stays folded', fn: () => ({ pinA: false, finger: null }) },
        ],
      },
      's5': {
        title: 'Two thumbs out — three-finger crease (experiment)',
        phases: [
          { dur: 0.8, label: 'Actuator 1 presses corner A onto the desk', fn: () => ({ pinA: true, fingers: [] }) },
          { dur: 3.6, label: 'Three grips turn the C-half over the diagonal; the other hand holds the sheet flat', fn: k => ({ pinA: true, grabs: k < 0.8 ? flap(Math.PI * smooth(k)).concat(HOLDS) : flap(Math.PI * smooth(k)), fingers: [] }) },
          { dur: 0.6, label: 'The folded half is held down', fn: () => ({ pinA: true, grabs: FLAP_DOWN, fingers: [] }) },
          { dur: 1.4 / fs, label: 'Three fingertips descend together, just behind the folded edge', fn: k => {
            const y = 0.5 + (s5Y - 0.5) * smooth(k);
            return { pinA: true, grabs: FLAP_DOWN_SOFT, fingers: [fing(s5At(0, y, S5_BK)), fing(s5At(S5_START, y, S5_BK)), fing(s5At(-S5_START, y, S5_BK))] };
          } },
          { dur: 1.2 / fs, label: 'All three iron forward over the edge together', fn: k => {
            const b = S5_BK + (S5_FR - S5_BK) * smooth(k);
            return { pinA: true, grabs: FLAP_DOWN_SOFT, fingers: [fing(s5At(0, s5Y, b)), fing(s5At(S5_START, s5Y, b)), fing(s5At(-S5_START, s5Y, b))] };
          } },
          { dur: 4.5 / fs, label: 'The middle finger holds; the outer two sweep out toward the corners', fn: k => {
            const s = S5_START + (S5_END - S5_START) * smooth(k);
            return { pinA: true, grabs: FLAP_DOWN_SOFT, fingers: [fing(s5At(0, s5Y, S5_FR)), fing(s5At(s, s5Y, S5_FR)), fing(s5At(-s, s5Y, S5_FR))] };
          } },
          { dur: 0.8 / fs, label: 'Holding at the corners', fn: () => ({ pinA: true, grabs: FLAP_DOWN_SOFT, fingers: [fing(s5At(0, s5Y, S5_FR)), fing(s5At(S5_END, s5Y, S5_FR)), fing(s5At(-S5_END, s5Y, S5_FR))] }) },
          { dur: 1.2 / fs, label: 'All three fingertips lift away', fn: k => {
            const y = s5Y + (0.55 - s5Y) * smooth(k);
            return { pinA: true, grabs: FLAP_DOWN_SOFT, fingers: [fing(s5At(0, y, S5_FR)), fing(s5At(S5_END, y, S5_FR)), fing(s5At(-S5_END, y, S5_FR))] };
          } },
          { dur: 1.0, label: 'The grips let go', fn: () => ({ pinA: true, fingers: [] }) },
          { dur: 3.5, label: 'Everything released — how well did the sweep crease it?', fn: () => ({ pinA: false, fingers: [] }) },
        ],
      },
    };
  }

  function scenarioDuration(sc) {
    return sc.phases.reduce((s, p) => s + p.dur, 0);
  }

  // evaluate scenario at absolute time t -> {world, label, phase, done}
  function evalScenario(sc, t) {
    let acc = 0;
    for (let i = 0; i < sc.phases.length; i++) {
      const ph = sc.phases[i];
      if (t < acc + ph.dur) {
        return { world: ph.fn((t - acc) / ph.dur), label: ph.label, phase: i, done: false };
      }
      acc += ph.dur;
    }
    const last = sc.phases[sc.phases.length - 1];
    return { world: last.fn(1), label: last.label, phase: sc.phases.length - 1, done: true };
  }

  // ---------------------------------------------------------------- metrics
  function metrics(paper) {
    const { pos, hinges } = paper;
    let maxY = -Infinity, maxSpeed = 0, nan = false;
    for (let i = 0; i < paper.NP; i++) {
      const p = i * 3;
      if (!isFinite(pos[p]) || !isFinite(pos[p + 1]) || !isFinite(pos[p + 2])) nan = true;
      if (pos[p + 1] > maxY) maxY = pos[p + 1];
      const v = Math.hypot(paper.vel[p], paper.vel[p + 1], paper.vel[p + 2]);
      if (v > maxSpeed) maxSpeed = v;
    }
    const CORNERS = paper.corners;
    const corner = (c) => [pos[c * 3], pos[c * 3 + 1], pos[c * 3 + 2]];
    let cx = 0, cz = 0;
    for (let i = 0; i < paper.NP; i++) { cx += pos[i * 3]; cz += pos[i * 3 + 2]; }
    cx /= paper.NP; cz /= paper.NP;
    let maxTheta = 0, maxTheta0 = 0;
    const creases = [];
    for (const h of hinges) {
      const th = hingeAngle(pos, h, false);
      if (th === th && Math.abs(th) > maxTheta) maxTheta = Math.abs(th);
      if (Math.abs(h.theta0) > maxTheta0) maxTheta0 = Math.abs(h.theta0);
      if (Math.abs(h.theta0) > 0.15) creases.push({ u: h.u, v: h.v, theta0: h.theta0 });
    }
    return {
      maxY, maxSpeed, nan,
      A: corner(CORNERS.A), B: corner(CORNERS.B), C: corner(CORNERS.C), D: corner(CORNERS.D),
      centroid: [cx, cz], maxTheta, maxTheta0, creases,
    };
  }

  return {
    PARAMS, makePaper, stepFrame, hingeAngle, wrapPi, _grad: _g,
    getScenarios, evalScenario, scenarioDuration, metrics,
  };
});

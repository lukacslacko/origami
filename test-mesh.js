// Unit tests for the adaptive crease-conforming mesher.
const MESH2 = require('./mesh2.js');

let fails = 0;
const check = (name, ok, detail) => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '   [' + detail + ']' : ''}`);
  if (!ok) fails++;
};

function meshStats(m) {
  const n = m.uv.length / 2, nt = m.tris.length / 3;
  // triangle quality: min angle
  let minAng = 1e9, minArea = 1e9, badOrient = 0;
  for (let t = 0; t < m.tris.length; t += 3) {
    const A = m.tris[t] * 2, B = m.tris[t + 1] * 2, C = m.tris[t + 2] * 2;
    const ax = m.uv[A], ay = m.uv[A + 1], bx = m.uv[B], by = m.uv[B + 1], cx = m.uv[C], cy = m.uv[C + 1];
    const area = 0.5 * ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax));
    if (area <= 0) badOrient++;
    minArea = Math.min(minArea, area);
    const l1 = Math.hypot(bx - ax, by - ay), l2 = Math.hypot(cx - bx, cy - by), l3 = Math.hypot(ax - cx, ay - cy);
    const angs = [
      Math.acos(Math.max(-1, Math.min(1, (l1 * l1 + l3 * l3 - l2 * l2) / (2 * l1 * l3)))),
      Math.acos(Math.max(-1, Math.min(1, (l1 * l1 + l2 * l2 - l3 * l3) / (2 * l1 * l2)))),
    ];
    angs.push(Math.PI - angs[0] - angs[1]);
    minAng = Math.min(minAng, ...angs);
  }
  // total area should be ~1 (the unit square)
  let totArea = 0;
  for (let t = 0; t < m.tris.length; t += 3) {
    const A = m.tris[t] * 2, B = m.tris[t + 1] * 2, C = m.tris[t + 2] * 2;
    totArea += 0.5 * ((m.uv[B] - m.uv[A]) * (m.uv[C + 1] - m.uv[A + 1]) - (m.uv[B + 1] - m.uv[A + 1]) * (m.uv[C] - m.uv[A]));
  }
  return { n, nt, minAngDeg: minAng * 180 / Math.PI, minArea, badOrient, totArea };
}

console.log('--- flat sheet, no creases');
{
  const t0 = Date.now();
  const m = MESH2.buildMesh({ hMax: 1 / 22, hMin: 1 / 88, creases: [], sizeGrid: null, anchors: [[0.5, 1], [0.15, 0.55]] });
  const s = meshStats(m);
  console.log(`  info: ${s.n} verts, ${s.nt} tris, minAngle=${s.minAngDeg.toFixed(1)}°, area=${s.totArea.toFixed(4)}, ${Date.now() - t0}ms`);
  check('all CCW', s.badOrient === 0, `bad=${s.badOrient}`);
  check('covers the square', Math.abs(s.totArea - 1) < 1e-6, `area=${s.totArea.toFixed(6)}`);
  check('reasonable quality', s.minAngDeg > 12, `minAngle=${s.minAngDeg.toFixed(1)}`);
  check('coarse when flat', s.n < 1400, `${s.n} verts`);
}

console.log('--- oblique crease at ~27°');
{
  const t0 = Date.now();
  const crease = { a: [1, 0.375], b: [0, 0.875] };
  const m = MESH2.buildMesh({ hMax: 1 / 22, hMin: 1 / 88, creases: [crease], sizeGrid: null, anchors: [] });
  const s = meshStats(m);
  console.log(`  info: ${s.n} verts, ${s.nt} tris, minAngle=${s.minAngDeg.toFixed(1)}°, area=${s.totArea.toFixed(4)}, creaseEdges=${m.creaseEdges.size}, ${Date.now() - t0}ms`);
  check('all CCW', s.badOrient === 0);
  check('covers the square', Math.abs(s.totArea - 1) < 1e-6, `area=${s.totArea.toFixed(6)}`);
  check('crease fully connected', m.missingCreaseEdges === 0, `missing=${m.missingCreaseEdges}`);
  const expect = Math.hypot(1, 0.5) / (1 / 88); // crease length / hMin
  check('crease densely sampled', m.creaseEdges.size > expect * 0.8, `${m.creaseEdges.size} vs ~${expect.toFixed(0)}`);
  check('quality holds with crease', s.minAngDeg > 8, `minAngle=${s.minAngDeg.toFixed(1)}`);
}

console.log('--- two crossing creases');
{
  const m = MESH2.buildMesh({
    hMax: 1 / 22, hMin: 1 / 88,
    creases: [{ a: [1, 0.375], b: [0, 0.875] }, { a: [0.2, 0.1], b: [0.7, 0.95] }],
    sizeGrid: null, anchors: [],
  });
  const s = meshStats(m);
  console.log(`  info: ${s.n} verts, ${s.nt} tris, minAngle=${s.minAngDeg.toFixed(1)}°, creaseEdges=${m.creaseEdges.size}`);
  check('all CCW', s.badOrient === 0);
  check('both creases connected', m.missingCreaseEdges === 0, `missing=${m.missingCreaseEdges}`);
  check('covers the square', Math.abs(s.totArea - 1) < 1e-6);
}

console.log('--- transfer roundtrip (flat -> same points)');
{
  const m = MESH2.buildMesh({ hMax: 1 / 22, hMin: 1 / 88, creases: [], sizeGrid: null, anchors: [] });
  const n = m.uv.length / 2;
  const pos = new Float64Array(n * 3), vel = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) {
    pos[i * 3] = m.uv[i * 2];
    pos[i * 3 + 1] = 0.2 * Math.sin(3 * m.uv[i * 2]) * Math.cos(2 * m.uv[i * 2 + 1]);
    pos[i * 3 + 2] = m.uv[i * 2 + 1];
  }
  const m2 = MESH2.buildMesh({ hMax: 1 / 22, hMin: 1 / 88, creases: [{ a: [0.5, 0], b: [0.5, 1] }], sizeGrid: null, anchors: [] });
  const r = MESH2.transfer(m.uv, m.tris, pos, vel, m2.uv);
  let maxErr = 0;
  const n2 = m2.uv.length / 2;
  for (let i = 0; i < n2; i++) {
    const u = m2.uv[i * 2], v = m2.uv[i * 2 + 1];
    const want = 0.2 * Math.sin(3 * u) * Math.cos(2 * v);
    maxErr = Math.max(maxErr, Math.abs(r.pos[i * 3 + 1] - want));
  }
  check('smooth field transfers accurately', maxErr < 0.004, `maxErr=${maxErr.toFixed(5)}`);
}

console.log(fails === 0 ? '\nMESH TESTS PASSED' : `\n${fails} FAILURES`);
process.exit(fails ? 1 : 0);

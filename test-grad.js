// Finite-difference verification of the signed dihedral angle gradient in sim.js
const SIM = require('./sim.js');

function randHinge() {
  // random non-degenerate hinge configuration
  const pts = [];
  for (let i = 0; i < 4; i++) pts.push([Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1]);
  return pts;
}

function angleOf(pts) {
  const pos = new Float64Array(12);
  for (let i = 0; i < 4; i++) { pos[i * 3] = pts[i][0]; pos[i * 3 + 1] = pts[i][1]; pos[i * 3 + 2] = pts[i][2]; }
  const h = { e0: 0, e1: 1, w0: 2, w1: 3 };
  return SIM.hingeAngle(pos, h, true); // theta (NaN if degenerate); grads in SIM._grad
}

// sign-convention sanity: flat sheet in xz-plane, lift both wings up -> theta > 0
{
  const flat = [[0, 0, 0], [1, 0, 0], [0.5, 0.2, 0.8], [0.5, 0.2, -0.8]];
  const th = angleOf(flat);
  console.log('convention check (both wings up): theta =', th.toFixed(4), th > 0 ? 'OK' : 'WRONG SIGN');
}

let worst = 0;
for (let trial = 0; trial < 200; trial++) {
  const pts = randHinge();
  const th = angleOf(pts);
  if (th !== th) { trial--; continue; }
  const g = Array.from(SIM._grad);
  const eps = 1e-6;
  for (let v = 0; v < 4; v++) {
    for (let c = 0; c < 3; c++) {
      const p2 = pts.map(p => p.slice());
      p2[v][c] += eps;
      const th2 = angleOf(p2);
      if (th2 !== th2) continue;
      const fd = SIM.wrapPi(th2 - th) / eps;
      const an = g[v * 3 + c];
      const err = Math.abs(fd - an) / Math.max(1, Math.abs(fd));
      if (err > worst) worst = err;
      if (err > 1e-3) {
        console.log(`MISMATCH v=${v} c=${c}: fd=${fd.toFixed(6)} analytic=${an.toFixed(6)}`);
      }
    }
  }
}
console.log('worst relative gradient error over 200 random hinges:', worst.toExponential(3));
console.log(worst < 1e-3 ? 'GRADIENTS OK' : 'GRADIENTS BAD');

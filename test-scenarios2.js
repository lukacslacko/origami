// Behavioral tests for the v2 adaptive engine (sim2.js + mesh2.js).
// Scenario scripts come from sim.js; only the engine differs.
const SIM = require('./sim.js');
const SIM2 = require('./sim2.js');

let failures = 0;
const check = (name, ok, detail) => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '   [' + detail + ']' : ''}`);
  if (!ok) failures++;
};
const f = (x) => (typeof x === 'number' ? x.toFixed(3) : x);

function run(name) {
  const sc = SIM.getScenarios()[name];
  const paper = SIM2.makePaper(sc);
  const dur = SIM.scenarioDuration(sc);
  const worldAt = (t) => SIM.evalScenario(sc, t).world;
  const t0 = Date.now();
  while (paper.time < dur + 0.05) SIM2.stepFrame(paper, worldAt, 1 / 60);
  const wall = Date.now() - t0;
  const m = SIM2.metrics(paper);
  console.log(`\n=== ${name} (adaptive): ${(wall / (dur * 60)).toFixed(1)}ms/frame, ` +
    `${m.nVerts} verts, ${m.remeshCount} remeshes, ${m.nCreaseSegs} crease segs`);
  return m;
}

// s1: bend and release -> flat, in place, no creases
{
  const m = run('s1');
  check('no NaN', !m.nan);
  check('flat again', m.maxY < 0.05, `maxY=${f(m.maxY)}`);
  check('no creases invented', m.nCreaseSegs === 0, `segs=${m.nCreaseSegs}`);
  check('C back home', Math.hypot(m.C[0] - 1, m.C[2] - 1) < 0.06, `dC=${f(Math.hypot(m.C[0] - 1, m.C[2] - 1))}`);
  check('settled', m.maxSpeed < 0.2, `v=${f(m.maxSpeed)}`);
}

// s2: release the pin first -> slips, lands flat, no creases
{
  const m = run('s2');
  check('no NaN', !m.nan);
  check('slipped', Math.hypot(m.A[0], m.A[2]) > 0.05, `dA=${f(Math.hypot(m.A[0], m.A[2]))}`);
  check('flat', m.maxY < 0.05, `maxY=${f(m.maxY)}`);
  check('no creases invented', m.nCreaseSegs === 0, `segs=${m.nCreaseSegs}`);
}

// s3: fold + press -> a crease along the B-D diagonal direction
{
  const m = run('s3');
  check('no NaN', !m.nan);
  check('crease(s) formed', m.nCreaseSegs >= 1 && m.nCreaseSegs <= 3, `segs=${m.nCreaseSegs}`);
  // with the fingertip gate, s3's crease is local again (formed only where
  // the finger slid and squeezed), not the whole laid-closed fold
  const main = m.creaseLines.reduce((a, b) => (b.len > (a ? a.len : 0) ? b : a), null);
  const ang = Math.atan2(main.b[1] - main.a[1], main.b[0] - main.a[0]) * 180 / Math.PI;
  check('crease has sensible extent', main.len > 0.12 && main.len < 1.45, `len=${f(main.len)}`);
  check('main crease runs BD-ish', Math.abs(ang - 135) < 15 || Math.abs(ang + 45) < 15, `angle=${ang.toFixed(1)}`);
}

// s4: worked full crease -> folded flat and stays
{
  const m = run('s4');
  check('no NaN', !m.nan);
  const dCA = Math.hypot(m.C[0] - m.A[0], m.C[1] - m.A[1], m.C[2] - m.A[2]);
  check('sheet stays folded', dCA < 0.4, `dCA=${f(dCA)}`);
  // with the fingertip gate only finger-visited stretches crease; that
  // partial crease still holds the fold closed
  const totalLen = m.creaseLines.reduce((a, b) => a + b.len, 0);
  check('substantial worked crease', totalLen > 0.7, `totalLen=${f(totalLen)}`);
}

// s6: oblique fold -> ONE straight crease on the ideal ~27° line (the
// acceptance test for the whole adaptive rewrite)
{
  const m = run('s6');
  check('no NaN', !m.nan);
  const main = m.creaseLines.reduce((a, b) => (b.len > (a ? a.len : 0) ? b : a), null);
  check('a dominant crease exists', !!main && main.len > 0.9, main ? `len=${f(main.len)}` : 'none');
  // ideal crease: perpendicular bisector of C-(0.5,0): (1,0.375)->(0,0.875)
  const angIdeal = Math.atan2(0.5, -1) * 180 / Math.PI; // 153.43
  const ang = Math.atan2(main.b[1] - main.a[1], main.b[0] - main.a[0]) * 180 / Math.PI;
  const angErr = Math.min(Math.abs(ang - angIdeal), Math.abs(ang - angIdeal + 180), Math.abs(ang - angIdeal - 180));
  check('crease angle matches the ideal line', angErr < 3, `err=${angErr.toFixed(2)}°`);
  // max distance of crease endpoints from the ideal line
  const NRM = [-0.4472136, -0.8944272], MID = [0.75, 0.5];
  const off = (p) => Math.abs((p[0] - MID[0]) * NRM[0] + (p[1] - MID[1]) * NRM[1]);
  const maxOff = Math.max(off(main.a), off(main.b));
  check('crease near the ideal line (default detail)', maxOff < 0.1, `maxOff=${f(maxOff)}`);
  const dCM = Math.hypot(m.C[0] - 0.5, m.C[1], m.C[2]);
  check('sheet stays folded onto the target', dCM < 0.3, `dCM=${f(dCM)}`);
  check('few spurious segments', m.nCreaseSegs <= 3, `segs=${m.nCreaseSegs}`);
}

// s6 again at fine detail: crease accuracy must scale with resolution
{
  SIM2.PARAMS.hMin = 1 / 176;
  const m = run('s6');
  SIM2.PARAMS.hMin = 1 / 88;
  const main = m.creaseLines.reduce((a, b) => (b.len > (a ? a.len : 0) ? b : a), null);
  check('fine detail: dominant crease', !!main && main.len > 0.9, main ? `len=${f(main.len)}` : 'none');
  const angIdeal = Math.atan2(0.5, -1) * 180 / Math.PI;
  const ang = Math.atan2(main.b[1] - main.a[1], main.b[0] - main.a[0]) * 180 / Math.PI;
  const angErr = Math.min(Math.abs(ang - angIdeal), Math.abs(ang - angIdeal + 180), Math.abs(ang - angIdeal - 180));
  check('fine detail: angle within 2°', angErr < 2, `err=${angErr.toFixed(2)}°`);
  const NRM = [-0.4472136, -0.8944272], MID = [0.75, 0.5];
  const off = (p) => Math.abs((p[0] - MID[0]) * NRM[0] + (p[1] - MID[1]) * NRM[1]);
  const maxOff = Math.max(off(main.a), off(main.b));
  check('fine detail: on the ideal line', maxOff < 0.04, `maxOff=${f(maxOff)}`);
}

console.log(failures === 0 ? '\nALL V2 SCENARIO TESTS PASSED' : `\n${failures} FAILURES`);
process.exit(failures ? 1 : 0);

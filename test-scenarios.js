// Headless runs of the three scenarios with behavioral assertions.
const SIM = require('./sim.js');

const DT = 1 / 60;
let failures = 0;
function check(name, ok, detail) {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '   [' + detail + ']' : ''}`);
  if (!ok) failures++;
}
const d2 = (a, b) => Math.hypot(a[0] - b[0], a[2] - b[2]); // horizontal distance
const d3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const f = (x) => (typeof x === 'number' ? x.toFixed(3) : x);

function run(name, checkpoints) {
  const sc = SIM.getScenarios()[name];
  const paper = SIM.makePaper();
  const dur = SIM.scenarioDuration(sc);
  const worldAt = (t) => SIM.evalScenario(sc, t).world;
  const snaps = {}; // phase index -> metrics at phase end
  const peakY = {}; // phase index -> peak sheet height during that phase
  let lastPhase = 0;
  const t0 = Date.now();
  while (paper.time < dur + 0.05) {
    SIM.stepFrame(paper, worldAt, DT);
    const ev = SIM.evalScenario(sc, paper.time);
    let my = 0;
    for (let i = 0; i < paper.NP; i++) if (paper.pos[i * 3 + 1] > my) my = paper.pos[i * 3 + 1];
    if (!(peakY[ev.phase] >= my)) peakY[ev.phase] = my;
    if (ev.phase !== lastPhase) {
      snaps[lastPhase] = SIM.metrics(paper);
      lastPhase = ev.phase;
    }
  }
  snaps[lastPhase] = SIM.metrics(paper);
  const wall = Date.now() - t0;
  console.log(`\n=== ${name}: "${sc.title}"  (N=${paper.N}, sim ${dur.toFixed(1)}s, wall ${wall}ms, ${(wall / (dur / DT)).toFixed(1)}ms/frame)`);
  checkpoints(snaps, paper, peakY);
  return snaps;
}

const A0 = [0, 0, 0], B0 = [1, 0, 0], C0 = [1, 0, 1], D0 = [0, 0, 1];

// ---------------- scenario 1: hold & bend -> springs back, nothing moves
run('s1', (m) => {
  const hold = m[2], afterC = m[3], end = m[4];
  check('no NaN', !end.nan);
  check('bent while held (maxY high)', hold.maxY > 0.3, `maxY=${f(hold.maxY)}`);
  check('pinned corner A does not move while held', d3(hold.A, A0) < 0.02, `dA=${f(d3(hold.A, A0))}`);
  check('snaps back flat after releasing C', afterC.maxY < 0.06, `maxY=${f(afterC.maxY)}`);
  check('corner C back near its origin', d2(afterC.C, C0) < 0.08, `dC=${f(d2(afterC.C, C0))}`);
  check('sheet ends where it started (A)', d2(end.A, A0) < 0.05, `dA=${f(d2(end.A, A0))}`);
  check('sheet ends where it started (B,D)', d2(end.B, B0) < 0.06 && d2(end.D, D0) < 0.06,
    `dB=${f(d2(end.B, B0))} dD=${f(d2(end.D, D0))}`);
  check('no crease formed', end.maxTheta0 < 0.05, `maxTheta0=${f(end.maxTheta0)}`);
  check('settled (low speed)', end.maxSpeed < 0.15, `v=${f(end.maxSpeed)}`);
});

// ---------------- scenario 2: release held corner first -> slips, shifts
run('s2', (m) => {
  const slipped = m[3], end = m[4];
  check('no NaN', !end.nan);
  check('corner A slipped out after release', d2(slipped.A, A0) > 0.05, `dA=${f(d2(slipped.A, A0))}`);
  check('still somewhat bent while dangling', slipped.maxY > 0.35, `maxY=${f(slipped.maxY)}`);
  check('falls flat at the end', end.maxY < 0.06, `maxY=${f(end.maxY)}`);
  check('still a square: side lengths preserved', (() => {
    const s1 = d3(end.A, end.B), s2 = d3(end.B, end.C), s3 = d3(end.C, end.D), s4 = d3(end.D, end.A);
    return [s1, s2, s3, s4].every(s => Math.abs(s - 1) < 0.05);
  })(), `sides=${[d3(end.A, end.B), d3(end.B, end.C), d3(end.C, end.D), d3(end.D, end.A)].map(f).join(',')}`);
  check('sheet has shifted', Math.hypot(end.centroid[0] - 0.5, end.centroid[1] - 0.5) > 0.04,
    `dCentroid=${f(Math.hypot(end.centroid[0] - 0.5, end.centroid[1] - 0.5))}`);
  check('no crease formed', end.maxTheta0 < 0.05, `maxTheta0=${f(end.maxTheta0)}`);
  check('settled', end.maxSpeed < 0.15, `v=${f(end.maxSpeed)}`);
});

// ---------------- scenario 3: fold over + pinch -> permanent local crease
run('s3', (m, paper, peakY) => {
  const last = SIM.getScenarios().s3.phases.length - 1;
  const end = m[last];
  check('no NaN', !end.nan);
  const strong = end.creases.filter(c => Math.abs(c.theta0) > 0.5);
  check('a strong crease exists', strong.length >= 2, `strongHinges=${strong.length}`);
  const center = [0.5, 0.5];
  const offLine = strong.filter(c => Math.abs((c.u + c.v) - 1) > 0.25); // near the B-D fold line (mid-sheet)
  check('crease sits near the fold line at mid-sheet', offLine.length === 0,
    `offLine=${offLine.length}, u+v=${strong.length ? f(Math.min(...strong.map(c => c.u + c.v))) + '..' + f(Math.max(...strong.map(c => c.u + c.v))) : '-'}`);
  const far = strong.filter(c => Math.hypot(c.u - center[0], c.v - center[1]) > 0.32);
  check('crease is a local pinch (does not reach edges)', far.length === 0,
    `far=${far.length}, extent=${f(Math.max(0, ...strong.map(c => Math.hypot(c.u - 0.5, c.v - 0.5))))}`);
  check('wedge sticks up after release', end.maxY > 0.04, `maxY=${f(end.maxY)}`);
  check('crease magnitude is sharp', Math.max(...strong.map(c => Math.abs(c.theta0)), 0) > 0.8,
    `max|theta0|=${f(end.maxTheta0)}`);
  // the grips deliberately raise the fold while easing it open; after they let
  // go the sheet may float down from there but must not bounce higher
  check('no bounce above the grips\' release height', (peakY[last] || 0) < (peakY[last - 1] || 0) + 0.12,
    `peakY=${f(peakY[last] || 0)} vs release=${f(peakY[last - 1] || 0)}`);
  check('sheet stays on the desk', [end.A, end.B, end.C, end.D].every(c =>
    c[0] > -0.6 && c[0] < 1.7 && c[2] > -0.6 && c[2] < 1.7), `C=(${f(end.C[0])},${f(end.C[2])})`);
  check('settled', end.maxSpeed < 0.3, `v=${f(end.maxSpeed)}`);
  // where did the creases land?
  if (strong.length) {
    const ds = strong.map(c => Math.hypot(c.u - 0.5, c.v - 0.5)).sort((a, b) => a - b);
    console.log(`  info: ${strong.length} strong hinges, center-dist ${f(ds[0])}..${f(ds[ds.length - 1])}`);
  }
  console.log(`  info: total creased hinges (|t0|>0.15): ${end.creases.length}, C at [${end.C.map(f)}]`);
});

// ---------------- scenario 4: swipe -> full corner-to-corner crease, fold holds
run('s4', (m, paper, peakY) => {
  const last = SIM.getScenarios().s4.phases.length - 1;
  const end = m[last];
  check('no NaN', !end.nan);
  const strong = end.creases.filter(c => Math.abs(c.theta0) > 0.5);
  check('many strong crease hinges', strong.length >= 20, `strongHinges=${strong.length}`);
  const onLine = strong.filter(c => Math.abs((c.u + c.v) - 1) <= 0.25);
  check('crease follows the fold line', onLine.length >= strong.length * 0.8,
    `onLine=${onLine.length}/${strong.length}`);
  // position along the fold line, measured from the sheet center
  const along = strong.map(c => (c.v - c.u) / Math.SQRT2);
  check('crease reaches toward both ends', Math.max(...along) > 0.3 && Math.min(...along) < -0.3,
    `along=${f(Math.min(...along))}..${f(Math.max(...along))}`);
  check('sheet stays folded (C stays on A)', d3(end.C, end.A) < 0.5, `dCA=${f(d3(end.C, end.A))}`);
  check('folded flat (no tent)', end.maxY < 0.2, `maxY=${f(end.maxY)}`);
  check('settled', end.maxSpeed < 0.3, `v=${f(end.maxSpeed)}`);
  console.log(`  info: ${strong.length} strong hinges, along-line span ${f(Math.min(...along))}..${f(Math.max(...along))}, C at [${end.C.map(f)}]`);
});

// ---------------- scenario 5: two-thumb sweep (experimental) -> full crease
run('s5', (m) => {
  const last = SIM.getScenarios().s5.phases.length - 1;
  const end = m[last];
  check('no NaN', !end.nan);
  const strong = end.creases.filter(c => Math.abs(c.theta0) > 0.5);
  check('the sweep creases the fold', strong.length >= 20, `strongHinges=${strong.length}`);
  const along = strong.map(c => (c.v - c.u) / Math.SQRT2);
  check('crease reaches toward both corners', Math.max(...along) > 0.3 && Math.min(...along) < -0.3,
    `along=${f(Math.min(...along))}..${f(Math.max(...along))}`);
  check('sheet stays folded', d3(end.C, end.A) < 0.5, `dCA=${f(d3(end.C, end.A))}`);
  check('settled', end.maxSpeed < 0.3, `v=${f(end.maxSpeed)}`);
  console.log(`  info: ${strong.length} strong hinges, span ${f(Math.min(...along))}..${f(Math.max(...along))}`);
});

console.log(failures === 0 ? '\nALL SCENARIO TESTS PASSED' : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);

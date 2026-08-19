// Parameter sweep: find stiffness/gravity/friction where S2 slips out and S3 keeps a wedge.
const SIM = require('./sim.js');
const DT = 1 / 60;
const f = (x) => (typeof x === 'number' ? x.toFixed(3) : x);

function runScenario(name, overrides) {
  const saved = {};
  for (const k of Object.keys(overrides)) { saved[k] = SIM.PARAMS[k]; SIM.PARAMS[k] = overrides[k]; }
  const sc = SIM.getScenarios()[name];
  const paper = SIM.makePaper();
  const dur = SIM.scenarioDuration(sc);
  const worldAt = (t) => SIM.evalScenario(sc, t).world;
  const snaps = {};
  let lastPhase = 0;
  while (paper.time < dur + 0.05) {
    SIM.stepFrame(paper, worldAt, DT);
    const ev = SIM.evalScenario(sc, paper.time);
    if (ev.phase !== lastPhase) { snaps[lastPhase] = SIM.metrics(paper); lastPhase = ev.phase; }
  }
  snaps[lastPhase] = SIM.metrics(paper);
  for (const k of Object.keys(saved)) SIM.PARAMS[k] = saved[k];
  return snaps;
}

console.log('--- S2 slip sweep (metric: A displacement after pin release / final centroid shift) ---');
for (const [aB, g, sub] of [
  [3e-3, 18, 15], [1e-3, 18, 15], [1e-3, 10, 15], [3e-4, 10, 20], [3e-4, 18, 20], [1e-4, 10, 25],
]) {
  const m = runScenario('s2', { alphaBend: aB, gravity: g, substeps: sub });
  const slip = Math.hypot(m[3].A[0], m[3].A[2]);
  const shift = Math.hypot(m[4].centroid[0] - 0.5, m[4].centroid[1] - 0.5);
  console.log(`aB=${aB} g=${g} sub=${sub}: slip dA=${f(slip)} dangleMaxY=${f(m[3].maxY)} endShift=${f(shift)} endMaxY=${f(m[4].maxY)} t0=${f(m[4].maxTheta0)} nan=${m[4].nan}`);
}

console.log('--- S2 with zero friction (is slip force-limited or friction-limited?) ---');
{
  const m = runScenario('s2', { muStatic: 0, muKinetic: 0 });
  const slip = Math.hypot(m[3].A[0], m[3].A[2]);
  console.log(`mu=0 aB=3e-3: slip dA=${f(slip)}`);
}

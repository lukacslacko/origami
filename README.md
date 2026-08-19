# Crease Lab

A physical model of a square sheet of paper, built as step one of a 3D
origami-robot simulation. The paper bends elastically, slips on the desk with
stick/slip friction, and creases plastically. Six scripted scenarios play out
on an interactive 3D bench in the browser.

Two engines share the same scenarios, renderer, and actuators:

- **Adaptive (v2, default)** — `mesh2.js` + `sim2.js`: the mesh is rebuilt on
  the fly from local curvature. Where the sheet bends tighter than a
  refinement radius, the region refines; where the bend radius drops below
  the crease radius (~1.5 paper thicknesses) *under a pressing fingertip*,
  the over-limit hinges are clustered, a straight line is fitted through
  them in material space, and that line becomes a permanent crease: the mesh
  conforms to it forever, its plastic rest-angle profile survives every
  remesh, and creases can cross. Off-grid folds come out straight (scenario
  6's ~27° crease lands within a few % of the ideal line, angle error < 3°),
  and cost stays low because fine mesh exists only where the paper bends.
  The fingertip requirement is deliberate: the contact cushion is ~20× a
  real sheet's thickness, so a fold merely laid closed bottoms out at the
  same radius as a pressed one — geometry alone cannot tell them apart, and
  without the gate paper would crease under its own weight. A force-based
  criterion needs force-limited contacts (future work).
- **Grid (v1)** — `sim.js` uniform grid, CPU or graph-colored WebGPU
  (`gpu.js`); kept for comparison and for the resolution experiments.

**Live demo:** open `dist/crease-lab.html` in a browser (it is fully
self-contained — no dependencies, no build step needed to view it).

## Scenarios

1. **Press & bend** — pin one corner, lift the opposite one; released, the
   sheet snaps back flat and nothing has moved.
2. **Let go of the wrong finger** — release the pin first: stored bending
   energy shoves the sheet out from under it; it lands flat, square, shifted.
3. **Fold & pinch** — fold corner C over onto A and pinch the folded edge at
   mid-sheet: released, the sheet springs open but a wedge-shaped crease stays.
4. **Work a full crease** — after the fold, the fingertip presses along the
   whole folded edge: released, the sheet stays folded.
5. **Two thumbs out** — three fingertips land behind the folded edge and iron
   over it together; the middle one holds while the outer two sweep to the
   corners. One sweep, a full crease.
6. **Fold off the grid** — corner C is folded onto the midpoint of the bottom
   edge; the crease runs at ~27°, cutting across the mesh. At coarse meshes
   the staircase crease is weak and the sheet relaxes to a tent; at 61+ the
   fold holds, with visible serration along the crease — the motivation for
   curvature-aligned remeshing.

The bench exposes mesh resolution (17–161 per side), finger tempo, press
clearance, and the compute backend; every run is deterministic.

## WebGPU backend

`gpu.js` runs the same solver on the GPU: constraint solves as graph-colored
compute dispatches (8 distance groups, 16 hinge groups — structured coloring on
the grid), a 2D uniform-grid hash with atomic linked lists for self-collision,
and plasticity on-device. The CPU evaluates the scenario script (one record per
substep) and reads positions + crease state back each frame, so the renderer
and UI are backend-agnostic. Parity vs the CPU backend: ≤0.01 position drift
after 3 simulated seconds (f32 + constraint-order noise).

Readback is pipelined (positions lag the GPU by one frame; crease state read
at 7.5 Hz), and long-range one-sided "rope" constraints at strides 4/16/64
carry tension stiffness across the sheet in a few solver hops. Together with
resolution-scaled hinge damping this lets the substep count grow as N^1.5
instead of N^2. Measured (Apple Silicon, Chrome): N=61 real-time (~14 ms),
N=81 ≈ 2× slow-mo, N=121 ≈ 6× slow-mo, N=161 runs (substeps capped at 256;
the canvas painter's renderer becomes the bottleneck past ~101).

## Model

XPBD (position-based dynamics, small-substeps flavour) on an N×N grid:

- **Stretch/shear** — distance constraints (structural edges + both cell
  diagonals); the sheet is effectively inextensible.
- **Bending** — signed dihedral hinge constraints on every interior edge, with
  finite-difference-verified gradients, continuous angle tracking (folds
  pressed flat sit at θ=±π, on the atan2 wrap), and constraint-space damping.
- **Creasing** — plasticity on the hinge rest angle: pressed past a yield
  curvature *under the fingertip*, the rest angle flows and stays. (Pressure
  gating matters: at this resolution a fold that merely falls closed reads the
  same hinge angle as a pressed crease.)
- **Contact** — desk plane with static/kinetic friction; particle-cushion
  self-collision with Coulomb-style layer friction; all actuator hardware
  (pads, rods, fingertip capsule) collides with the sheet.
- **Actuators** — kinematic: a corner pin, point grips (optionally
  force-limited so a clamped flap can give a little instead of wrinkling),
  hold-down pads, and a capsule fingertip.
- **Resolution invariance** — particle mass ∝ cell area, yield angle ∝ cell
  size, substeps ∝ (resolution)² (PBD stiffness loss is quadratic in
  constraint-chain length); behavior is verified consistent from 17² to 29².

Deliberate simplifications: creases can only form along mesh lines, the mesh
diagonals are aligned with the expected fold for a clean crease, self-contact
is a particle cushion (no continuous collision detection), and aerodynamics is
just isotropic + broadside drag.

## Files

| File | Purpose |
| --- | --- |
| `mesh2.js` | adaptive crease-conforming mesher (variable-density Delaunay + CDT edge enforcement + material-space state transfer) |
| `sim2.js` | v2 adaptive engine: physics on the adaptive mesh + curvature detection, crease fitting, remeshing |
| `sim.js` | v1 uniform-grid engine + the scenario scripts (shared by both engines) |
| `gpu.js` | WebGPU compute backend for the v1 engine |
| `index.template.html` | page: custom Canvas-2D painter's-algorithm 3D renderer + UI |
| `build.js` | inlines all engines into the template → `dist/crease-lab.html`, `docs/index.html` |
| `test-grad.js` | finite-difference check of the hinge-angle gradients |
| `test-mesh.js` | mesher unit tests (quality, crease connectivity, crossings, transfer) |
| `test-scenarios.js` | behavioral tests, v1 engine |
| `test-scenarios2.js` | behavioral + straightness acceptance tests, v2 engine |
| `sweep.js` | parameter-sweep harness used for tuning |

## Develop

```sh
node test-grad.js        # verify hinge gradients
node test-mesh.js        # adaptive mesher unit tests
node test-scenarios.js   # v1 engine behavioral suite (~40 s)
node test-scenarios2.js  # v2 adaptive engine suite incl. straightness acceptance (~2 min)
node build.js            # rebuild dist/crease-lab.html + docs/index.html
```

The page exposes a deterministic driver for debugging:
`__lab.setScenario('s3'); __lab.advance(5.0)` steps five simulated seconds and
renders.

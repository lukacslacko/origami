# Crease Lab

A simplified physical model of a square sheet of paper, built as step one of a
3D origami-robot simulation. The paper bends elastically, slips on the desk
with stick/slip friction, and — when pressed hard enough by a fingertip —
takes a permanent crease. Four scripted scenarios play out on an interactive
3D bench in the browser.

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

The bench exposes mesh resolution (17–81 per side), finger tempo, press
clearance, and the compute backend; every run is deterministic.

## WebGPU backend

`gpu.js` runs the same solver on the GPU: constraint solves as graph-colored
compute dispatches (8 distance groups, 16 hinge groups — structured coloring on
the grid), a 2D uniform-grid hash with atomic linked lists for self-collision,
and plasticity on-device. The CPU evaluates the scenario script (one record per
substep) and reads positions + crease state back each frame, so the renderer
and UI are backend-agnostic. Parity vs the CPU backend: ≤0.01 position drift
after 3 simulated seconds (f32 + constraint-order noise).

Measured (Apple Silicon, Chrome): N=41 real-time, N=61 ≈ 2× slow-mo,
N=81 ≈ 4× slow-mo — the substep count grows quadratically with resolution
(PBD stiffness propagates one constraint per substep), which is the real
scaling wall; the GPU absorbs the per-substep width.

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
| `sim.js` | physics core + scenario scripts (runs in Node and the browser) |
| `gpu.js` | WebGPU compute backend (same solver, graph-colored) |
| `index.template.html` | page: custom Canvas-2D painter's-algorithm 3D renderer + UI |
| `build.js` | inlines `sim.js` + `gpu.js` into the template → `dist/crease-lab.html` |
| `test-grad.js` | finite-difference check of the hinge-angle gradients |
| `test-scenarios.js` | headless behavioral tests for all four scenarios |
| `sweep.js` | parameter-sweep harness used for tuning |

## Develop

```sh
node test-grad.js        # verify hinge gradients
node test-scenarios.js   # run the behavioral test suite (~40 s)
node build.js            # rebuild dist/crease-lab.html
```

The page exposes a deterministic driver for debugging:
`__lab.setScenario('s3'); __lab.advance(5.0)` steps five simulated seconds and
renders.

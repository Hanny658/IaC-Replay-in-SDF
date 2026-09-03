---
name: cortex-phase20-results
description: "Phase 20 (2026-09-03, depth ladder d3/d4/d5 on MNIST s10): fixed lambda hits a CLIFF where signal thins (static dead at chance from d4, seq bimodal-dead at d5 55.5+-39.3) while the adaptive controller keeps every cell alive with graceful decay (seq 90.6->90.0->87.6, static 95.6->95.3->91.8; ~1-1.5 pt per added layer); depth = phase-18's class-thinning boundary on a second axis. Local+controller matches or beats same-width BP+ER at every depth (d3/d4 ahead, d5 parity 87.6 vs 88.1) with 1/2-1/3 the forgetting (F 5-8 vs 13-14). Controller-as-depth-foundation prediction confirmed: +32 pt seq / +81 pt static at d5. make_bp generalised to arbitrary depth"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-09-03T06:33:16.701Z
---

Configs `g20_d{3,4,5}_{kad,fix}[_static]`, `g20_d{3,4,5}_bp_er` (45 runs, 3 seeds), hidden
ladders (512,256,128), (512,256,128,128), (512,256,128,128,128), af=0.10 everywhere, SGD
eta=.02, refractory local sleep cad1 br16, kad = kp_adapt=0.25 (v2 drive formula in code).
CortexNet was already depth-general; only `make_bp` needed generalising (any hidden tuple).

Full table (seq / static):
- d3: fix 91.4+-0.4/93.6+-0.4 | kad 90.6+-1.0/95.6+-0.1 | bp_er 88.9+-0.3 (F13.4)
- d4: fix 89.3+-2.5/9.7+-0.7 DEAD | kad 90.0+-0.6/95.3+-0.2 | bp_er 88.8+-0.3
- d5: fix 55.5+-39.3 BIMODAL/10.1+-0.3 DEAD | kad 87.6+-1.7/91.8+-0.8 | bp_er 88.1+-0.6
Realized lambda under kad: starved middle layers drop to 3e-6; static d5 deep tail 6e-5..9e-5.
Static dies first under fixed lambda because 10-way simultaneous supervision thins per-class
signal (seq tasks are 2-way = thick, survive to d5). Same mechanism as CIFAR-100 (W follows
pure-decay trajectory).

**Why:** answers "does the system survive stacking" — yes, but only under the controller;
fixed lambda's per-dataset tuning becomes per-(dataset x depth x axis) tuning, i.e. hopeless.
The controller's per-layer self-differentiation is what buys depth tolerance.
**How to apply:** any deeper/conv variant must run kp_adapt, never fixed kp_decay. Skip
connections (ResNet-style) remain the open buy-back for the residual d5 toll (seq -4, static
-4 vs d2); manuscript FW item. Related: [[cortex-phase19-results]], [[cortex-phase18-results]],
[[cortex-phase17-results]].

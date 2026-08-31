---
name: cortex-phase7b-results
description: "Phase 7B (REM) on split-MNIST: a decoder B0 trained by the input's own prediction error works, but top-down settling from a label through the error-feedback synapses produces NO class content (dreams at chance); prototype-seeded dreams (label->layer-1 mean/var learned by a local delta rule) are recognisable (judge 0.95-1.0) yet generative rehearsal adds nothing over the real buffer and reverse learning with a margin is a net loss at every strength; static axis unharmed"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-08-29T04:23:43.351Z
---

Ran 2026-08-29 with `src/run_seq.py` (v7 base, split-MNIST 5 epochs/task, 3 seeds; NREM real
replay K=200/1000 as in 7A). New code: `CortexNet(decoder, rem_neg, rem_margin)`, `B0` decoder
trained on e0 = x0 - decode(a1) during waking, `imagine()` (label-clamped top-down settling),
`G/Gvar` prototype synapses (per-class mean/variance of layer-1 pre-activation, local delta rule),
`rem_generate(source="proto")`, `rem_negative()` (margin reverse learning), `dream_quality()` in
run_seq (independent BP judge trained on all classes, self-readout, recon mse).

Diagnostic that shaped the phase: decoding the class-mean layer-1 activity -> judge 1.00, so B0 is
fine; but settling from a one-hot label through B (error feedback) gives |a1| = 0.018 vs 0.148 real
and nearest-prototype accuracy 0.10 at any T/gamma/refine -> the error-feedback path carries no
content. This also explains part of the phase-5 dream failure. Dreams therefore start from learned
semantic prototypes (10 x 256 mean+var, no raw samples), giving judge 0.95-1.0.

Results (final acc on 10 classes / forgetting / dream judge):
- ctx_nrem_rand_200 86.9 / .11 (7A);  b0_only_200 (decoder learned, unused) 84.1 +-3.6 / .15 / .95
  (difference = RNG path, B0 init changes later draws)
- REM-gen: gen_200 84.1 / .16 / .98;  gen_200_R40 85.0 / .13 / 1.00;  gen_1000 87.2 +-4.2 vs 90.7
  without;  gen_only_0 (no real buffer) 38.3 vs 19.5 none;  static_ctx_gen 93.9 vs 94.5.
  -> prototype rehearsal recovers half of forgetting with no episodic memory but adds nothing on
  top of even a 200-sample real buffer and slightly hurts at K=1000: the dreams are class means
  (judge ~1.0, low diversity), i.e. they carry the semantic content the cortex already has.
- REM-neg: lambda 0.1 -> 67 (dream judge .64, representation damaged); lambda 0.02 -> 79.9 at
  K=200 (vs 84.1), 87.9 at K=1000 (vs 90.7); static 94.0 vs 94.5. Reverse learning with a margin
  is a net loss at every strength tried; gen+neg combos inherit the loss.
- Static axis unharmed by either (<=0.6 pt).

Readings: (1) the generative path and prototype dreams work as machinery; (2) REM as pseudo-
rehearsal from semantic prototypes cannot substitute episodic samples on this task -- the missing
ingredient is diversity (sample-level variability), which a mean+diagonal-variance prototype does
not carry; (3) reverse learning has no functional role yet on a 5-task split; its proposed benefit
(pruning spurious attractors) needs a task where spurious attractors matter (e.g. a recurrent /
attractor readout or OOD detection), not classification accuracy.

**Why:** closes the 7B question honestly: REM is not the substitute for the hippocampal buffer in
this architecture; NREM real replay remains the consolidation mechanism.
**How to apply:** keep B0 + prototypes as infrastructure (they make dreams and reconstruction
available for 7C signals, e.g. "fantasies no longer discriminable"); do not use REM-gen/REM-neg
in the default night. If revisiting: diversity in dreams (buffer-seeded rather than prototype-
seeded generation, or a richer latent), and a spurious-attractor metric for reverse learning.
Related: [[cortex-phase7a-results]], [[cortex-phase7-roadmap]].

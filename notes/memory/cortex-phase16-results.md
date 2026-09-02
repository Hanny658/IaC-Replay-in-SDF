---
name: cortex-phase16-results
description: "Phase 16 (2026-09-02, k-WTA tax buy-back / sparsity dial, ~110 runs): the 5% k-WTA fraction (inherited from phase 8B Adam era) taxes BOTH regimes and relaxing it is near-free -- cfeat 5->25%: seq 40.6->42.8, static 38.6->45.6 (forgetting down); MNIST peak at af=10%: 92.7+-0.3/94.7+-0.3 (6 seeds) DOMINATES the entire 5% frontier incl. bout gate and anchor; no-buffer ceiling rises too (95.1->95.9) so rotation's matched-sparsity price shrinks 1.8->0.7-1.0 and the gate's purpose shrinks with it; silent gains nothing (dial helps only the rotating family); asymmetric sparsity negative (tax levied at every layer); BP shares the curve, local leads +1.1 at every matched level; energy free (synops +3-5%, spiking 94.0% @ 120 nJ input-dominated); paper keeps 5% substrate-of-record + new 'sparsity dial' subsection; 10% re-baselining = future work"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-09-02T07:37:57.050Z
---

Ran 2026-09-02 after 15F priced 5% k-WTA at ~4 points for BP on features.  Knobs: per-layer
active_frac (tuple; cortex._af/_k now take layer index -- spiking.py call fixed accordingly),
configs g16_* (MNIST) and cfeat_sgd_refr_s*/a* (features).

## Tax curves (local SGD refractory)
- cfeat (eta .01): 5% 40.6+-1.0/38.6+-1.4 -> 10% 41.7/41.9 -> 15% 42.3+-0.5/44.2+-0.3 ->
  25% 42.8+-0.3/45.6+-0.5.  Forgetting slightly DOWN as af rises.  eta hedge: 0.02 at s15
  worse seq (40.4) -- keep 0.01 on features.
- MNIST (eta .02, all 6 seeds): 5% 92.0+-0.6/93.3+-0.2 -> 10% 92.7+-0.3/94.7+-0.3 (seq PEAK)
  -> 15% 92.5+-0.3/95.1+-0.3 -> 25% 92.5+-0.1/95.1+-0.2 (plateau).  af=10 dominates every 5%
  frontier point (bout 92.0/94.2, anchor 92.4/93.3, always-on 92.0/93.3).

## Matched-sparsity controls (MNIST)
- No-buffer ceiling rises: 95.1 -> 95.8 -> 95.9; rotation's matched price shrinks 1.8 ->
  0.7-1.0.  At 15% always-on (92.5/95.2-ish) no longer trails the bout gate (92.4/94.9):
  THE GATE'S PURPOSE SHRINKS WITH THE PRICE -- the dial, not the gate, is the cheapest cure.
- Silent flat (89.0 -> 89.3 -> 88.7): the dial helps only the ROTATING family (starved of
  representational room at 5%).
- Bout gate at 10/15%: 92.3/95.1 and 92.4/94.9 -- fine but no longer special.

## Other verdicts
- Asymmetric sparsity (mild L1 / sparse L2: 15-5, 25-5) NEGATIVE: seq parity with uniform,
  static -3..4 with 4x variance -- the tax is levied at every layer.
- BP shares the tax curve (39.7 -> 41.2 -> 43.6 wired-dense; 44.8 dense-wide, all 6 seeds);
  local leads BP by ~1.1 at every matched sparsity level.
- Energy free: synops +3-5%; spiking input-dominated -- s15 94.0% @ 120 nJ (T_s=8), monotone
  in T_s (scripts/p16_spike.py).

**Why:** answers FW(i) fully: the k-WTA tax is real, buy-back is near-free, and it dissolves
most of what phases 13-15 built gates for -- an honest supersession recorded as such.
**How to apply:** paper keeps af=5% as substrate of record (mechanism tables internally
matched) + 'The sparsity dial' subsection + orange dial series in the frontier figure;
abstract/limits note the inherited choice.  Candidate new default for future phases:
af=10%, always-on rotation, SGD 0.02 (92.7/94.7) -- re-baselining the mechanism suite at 10%
is future work (i).  Related: [[cortex-phase14-15-results]], [[cortex-phase13-results]],
[[cortex-phase12-results]].

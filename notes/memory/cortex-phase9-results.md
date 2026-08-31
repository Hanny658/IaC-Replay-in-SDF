---
name: cortex-phase9-results
description: "Phase 9 (overnight 2026-08-31): DG front-end falsified (input decorrelation does not transfer, 79-87 vs 90.4); the 39x local-sleep cost was a batch-size artifact -- masked refractory replay is free down to batch ~8 (90.5 at 1.2x the night's replay samples; the night and unmasked ER both collapse at tiny batches), cad2+batch64 sets the new best 91.3 with no night and the night adds nothing on top; rare replay must come as bursts of 8-16 triggered by homeostatic pressure (89.9 at 15% of events), surprise-triggered wake replay fails at any volume; refractory costs 1.3-3.0 on the static axis"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-08-30T21:40:04.403Z
---

Ran 2026-08-30/31 overnight, split-MNIST class-IL, 3 seeds, K=1000 random buffer, refractory local
sleep bw16 (`run_local`), w512s5 = hidden (512,256) @ 5% k-WTA.  References: best night =
ctx_nrem_rand_1000 (w256s10) 90.7 +-0.4 @ 480x256 = 123k replay samples; night on w512s5 itself is
only 84.1 +-1.3; 8B refractory br256 90.4 +-0.2 @ 4.8M samples.  Code: DG/code_dim/gate/burst/
batch_replay/night_batch in `src/run_seq.py` (g9_*, g9a_* configs); table `tmp/bio/p9_table.py`.

1. 9C DG front-end FALSIFIED: fixed sparse expansion (2025 units, 8 local inputs, 3% k-WTA,
   divisive norm) separates the input code exactly as Cayco-Gajic predicts (task overlap 0.79 ->
   0.18, PR-dim 62 -> 139; random sampling 235; dense control 339 but overlap 0.06 with total
   feature loss) yet every downstream number is worse: refr 79.4/79.0, combo 79.4, DG@10% 87.2,
   night 82.4-83.6, static 91.4 (vs 95.0).  Isolated channel unchanged (~0.52-0.57), hidden
   PR-dim collapses (4-6 vs 11-15), hidden overlap NOT reduced under training.  Input-level
   orthogonality does not propagate through learned k-WTA layers, and 61-active-of-2025 codes
   starve the local Hebbian updates.  Dense fan-in after the DG is worse still (72.7).
2. 9A replay batch size: for masked refractory replay every waking batch it is FREE down to ~8:
   br64 90.5+-0.5, br32 90.7+-0.4, br16 90.8+-0.1, br8 90.5+-0.3, br4 89.7+-0.4, br2 86.7, br1 82.9.
   br8 = 150k replay samples = 1.2x the night, on w256s10 br8 90.4/br16 90.9 too.  The SAME shrink
   kills the alternatives: unmasked ER none_br16 86.9+-1.2 / none_br8 83.9+-1.2, silent 87.8+-2.1,
   night with batch 16 81.0+-1.6.  Isolation is what makes tiny-batch replay stable.  cad2_br64
   (every 2nd batch, batch 64, 600k samples) = 91.3+-0.3: best sequential number of the project,
   no night, beats the best night and 8B's cad1 br256.  br16+night 90.9+-0.3 = night adds NOTHING
   once cheap continuous local sleep is on (8B's combo gain came from the weak silent channel).
3. 9A timing when replay must be rare: bursts of 8-16 every 64-128 waking batches keep 86.8-87.6
   at ~12% of events; trickle (cad8 60.9, burst4 68.1) and naps (burst 32-128 @ 256-2048: 63-76)
   both fail -- masked replay reaches ~half the synapses, so it can neither wait nor dribble.
   Trigger: unit-level homeostatic pressure (Process S on asleep units, discharge on consolidation)
   pburst16_t64 89.9+-0.1 @ 2,773 events (15%); with batch 64 89.5+-0.5 (= 177k samples, 1.4x
   night); with batch 16 88.2+-1.0 (44k, 0.36x night, -2.6 pt).  Surprise/novelty trigger fails at
   ANY volume (19% even at 4,472 events: it fires only after task switches and starves epochs 2-5);
   7C's surprise timing works for nights, not for waking replay.  K=200 still trails the K=200
   night (br16 81.8+-2.9, pburst16 84.4+-1.1 vs 86.9).
4. 9B static axis price of forced rotation: none 95.0+-0.2 -> refractory 93.7+-0.1 (w256s10),
   92.8+-0.4 (w512s5), 92.0+-0.3 (br16) -- the -1.3..-3.0 cost during wake that Driessen/Tononi/
   Cirelli 2026 predict for local sleep at the wrong place/time.  DG static also negative.

**Why:** phase 9 settles the three post-8B directions: C (DG) falsified, A answered (the cost
was an accounting artifact; where replay must be rare, homeostatic bursts are the mechanism),
B quantified (real but small static price).
**How to apply:** v9 default = refractory local sleep, replay batch 16-64 every 1-2 waking
batches, no night; use pressure-triggered bursts (theta ~6.4, burst 16) when replay bandwidth is
capped; never gate waking replay by surprise; keep k-WTA layers and skip input-level pattern
separation.  Open: K=200 gap to the night, removing the static-axis price, energy story for the
report.  Related: [[cortex-phase8b-results]], [[cortex-lit-review-2026-08]],
[[cortex-phase7c-results]].

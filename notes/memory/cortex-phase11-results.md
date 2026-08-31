---
name: cortex-phase11-results
description: "Phase 11 (2026-08-31 evening, 3 seeds): progress gate (novelty OR error>=gamma*chance) unifies all four regimes (CIFAR seq 27.2 = refr where nov gate gave 19.1; MNIST static 94.3 at gamma 0.5, 0.7 below pure nov); V1-patch-feature CIFAR keeps the local-sleep ordering (refr 37.0 > night/none ~31) and the static rotation gain (+3.9) BUT BP+ER 42.6 overtakes; the K=200 'gap' was substrate mismatch -- same-substrate night 84.0+-3.4 vs local+noise 84.3-84.8 (closed; narrow-net night 86.9 still best overall); down-selection: free on continuous local sleep, +3.1 on the night (rescues wide-sparse night), harmful (-3..-5) on episodic bursts"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-08-31T09:48:28.920Z
---

Ran 2026-08-31 evening.  New: `refr_prog` mask (rotate iff fast-EMA >= beta*slow-EMA OR fast-EMA
>= gamma_p * e0 with e0 = mean error of first 20 batches), `load_cifar10_feat()` (256 normalised
random 6x6x3 patches, stride-2 conv, ReLU minus per-filter mean, quadrant pooling -> 1024-D,
linear probe ~49%; dataset "cifarf"), replay noise for nights, `sp` config (structural
plasticity anchored to sleep: "local" every sp_every replay batches / "night" after nights /
"epoch" clock control).  All w512s5, K as noted, 3 seeds.

1. 11A progress gate: MNIST seq prog03/05 90.4+-0.5/0.9 (refr 90.8+-0.1); MNIST static prog03
   92.8+-0.5, prog05 94.3+-0.2 (nov15 95.0+-0.4, none 95.0); CIFAR-gray seq prog03 27.2+-0.7 =
   refr 27.1+-1.7 (nov13 was 19.1 -- fixed); CIFAR static 25.3+-1.0 = refr.  One setting works
   in all four regimes; residual 0.7 static cost vs pure novelty gate on easy streams.  Default
   gate = prog (gamma 0.5, beta 1.3).
2. 11B feature-front CIFAR (cifarf): seq refr 37.0+-0.4 > prog03 36.0+-0.6 > night 31.7+-1.5 ~
   none 30.8+-0.6 >> no-buffer 17.0; static refr = prog 35.6+-1.3 > none 31.7+-3.6 (rotation
   +3.9).  HONEST NEGATIVE: BP+ER 42.6+-0.5 beats the local learner on features (on raw gray it
   lost 25.1 vs 27.1); the local-sleep advantage is regime-specific.  static prog == static refr
   bit-for-bit (gate never closes on an unmastered stream, as designed).
3. 11C K=200 mechanism: same-substrate (w512s5) night 84.0+-3.4 vs local pb16_n02 84.3+-1.0 /
   combo_n02 84.8+-0.9 -> GAP CLOSED at equal substrate (and local has 3x lower sd); the earlier
   86.9 was the w256s10 night (narrow net), still the best K=200 number.  Noise hurts the night
   (81.8+-3.2), unmasked trickle 71.0+-1.7.  Wake-side diversity + isolation are the levers;
   offline window matters mainly through the narrow-substrate night.
4. 11D down-selection (prune weakest + distance-biased regrow, KP decay as the weakening force):
   continuous local sleep: sp_local 90.9+-0.2 ~ sp_epoch 90.6+-0.4 ~ none 90.8+-0.1 (free, sleep
   anchoring +0.3 inside noise); NIGHT substrate: sp_night 87.2+-3.8 vs 84.1+-1.3 (+3.1 --
   after-night pruning rescues the wide-sparse night, Tononi-Cirelli works offline); episodic
   pburst substrate: sp after bursts 84.6+-0.5, sp epoch 86.6+-1.3 vs none 89.9+-0.1 (pruning
   HARMFUL when replay is rare: synapses are cut before enough consolidation protects them).
   Static axis: 91.7+-0.4 vs 92.0 (free).

**Why:** closes phase 11: the unified gate exists (prog), the feature-regime boundary of the
local-sleep advantage is mapped (BP+ER wins there), the K=200 gap is resolved as substrate
mismatch, and sleep-anchored down-selection has a clean three-way verdict.
**How to apply:** v11 default = refractory + br16 + prog gate (gamma 0.5) + optional sp_local
(free) on continuous substrates; add sp_night when running nights on wide-sparse nets; never
prune on episodic-burst substrates.  Open: why BP+ER wins on features (dense conn? Adam?),
narrow-vs-wide night at tiny K, prog-gate residual 0.7.  Related: [[cortex-phase10-results]],
[[cortex-phase9-results]].

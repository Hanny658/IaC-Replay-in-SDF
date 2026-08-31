---
name: cortex-phase7c-results
description: "Phase 7C (internal-signal sleep switching) on split-MNIST: surprise-driven sleep pressure + replay-error stop matches the fixed schedule's accuracy at 74-80% of the replay (K=1000: 91.0 @ 387 vs 90.7 @ 480) and beats fixed timing at equal replay by 12-14 pt (K=200: 84.4 vs 72.1 @ ~270); two thirds of sleeps land right after task switches; internally gated REM is finally not harmful (+0.5 at K=1000); count-based timing drifts against uneven task lengths"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-08-29T08:46:25.772Z
---

Ran 2026-08-29 with `run_stream()` in `src/run_seq.py` (v7 base, split-MNIST as a continuous
stream with no visible task boundaries, 5 passes per task, NREM x3, random buffer, replay cap 500,
3 seeds). Controller: sleep pressure S (count: +1/batch; surprise: + mean top-down error), sleep
at S >= theta; night length fixed R or "error" (stop when the last-5 replay error < 0.15, 5..40);
optional internally gated REM (dream rehearsal until self-readout agrees >= 0.95). A final night
is forced at the end of the stream, as in the fixed schedule.

Final acc / replay batches used (K=200 | K=1000):
- 7A fixed schedule: 86.9 / 480 | 90.7 / 480
- count_fixed (stream control): 86.8 / 480 | 86.0 +-2.2 / 480  <- drifts: theta = batches/epoch of
  task 1, but tasks have 250/240/220/240/235 batches, so nights slide into the next task; at
  K=1000 one seed lost a whole task (0.42). Use the 7A run() numbers as the fixed reference.
- count_R11 (fixed timing at the internal controller's replay volume): 72.1 / 264 | 77.3 / 264
- count_err: 82.3 / 318 | 91.2 +-0.1 / 414
- surp10_fixed: 85.6 / 433 | 90.7 / 447   (14 of 22 sleeps inside the first epoch after a switch)
- surp10_err: 84.4 +-0.7 / 274 | 91.0 +-0.6 / 387
- surp20_err: 85.0 / 262 | 90.6 +-0.3 / 354
- surp40_err: 80.3 / 185 | 85.2 / 202
- surp10_err_rem: 84.3 / 279 | 91.5 +-0.4 / 400

Readings:
1. Timing by surprise is worth ~12-14 pt at equal replay: surp10_err 84.4 @ 274 vs count_R11 72.1
   @ 264 (K=200); 91.0 @ 387 vs 77.3 @ 264 and count_err 91.2 @ 414 (K=1000). Novelty-triggered
   nights consolidate what was just learned; nights on a clock replay stale content.
2. The internal controller reaches the fixed schedule's accuracy at 74-80% of its replay at K=1000
   and is ~2 pt short at K=200 (where every replay batch matters because the buffer is tiny).
3. Replay-error stop is a good "wake-up" rule when the buffer is large (K=1000: 91.2 @ 414, sd 0.1)
   and premature when it is small (K=200: 82.3 @ 318).
4. Internally gated REM (rehearse dreams only while the readout disagrees with them) is the first
   REM variant that is not harmful: +0.5 at K=1000, 0 at K=200. Gating fixed the harm; the gain is
   still inside noise.
5. theta sets the wake/sleep ratio: 10 -> 22 sleeps, 20 -> 13, 40 -> 6; below ~13 sleeps accuracy
   falls off (surp40).

**Why:** settles 7C: an internal two-signal controller (surprise-driven pressure, error-driven
wake-up) is at least as good as the epoch clock at lower compute and does not need task
boundaries -- the property that makes the system runnable on an open stream.
**How to apply:** default night controller = surprise pressure theta~10-20 + error stop
(eps 0.15, 5..40 batches) + gated REM optional; do not use count-based pressure with uneven
tasks. Related: [[cortex-phase7a-results]], [[cortex-phase7b-results]], [[cortex-phase7-roadmap]].

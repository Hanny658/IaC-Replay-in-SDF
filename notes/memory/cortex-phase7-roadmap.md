---
name: cortex-phase7-roadmap
description: "Agreed phase-7 roadmap for the MLP-Cortex side project: 7A hippocampal buffer (BTSP-style surprise-gated one-shot write) + NREM high-plasticity replay; 7B generative path to input (B0) + margin-based wake-sleep contrastive objective so REM can be judged; 7C internal-signal state switching. Sequential axis (split-MNIST) added beside the static one, never replacing it; BP + experience replay is a mandatory baseline"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-08-29T03:41:24.420Z
---

Proposed by the user on 2026-08-29 after phase 6; agreed with conditions. Start from the v7 base
(v6 + sweep T=0). Order: 7A -> 7B -> 7C, one phase each, not merged.

- 7A hippocampus: buffer of real (x, y) written once when the top-down (apical) prediction error
  exceeds a threshold (the BTSP plateau analogue); NREM = interleaved replay from the buffer at a
  2-5x learning rate with the same single-phase local rule. Ladder: none / random / surprise-gated
  x capacity {200, 1000, 5000}.
- 7B REM: add B0 (layer-1 -> input feedback, learned by mirror or KP) so dreams are full samples;
  objective = wake-sleep split (wake trains recognition on real data and generative weights to
  predict the input; sleep trains recognition to recognise fantasies) as a per-layer local energy
  margin, NOT the raw sign flip that failed in phase 5. Functional metric: can generative replay
  substitute part of the real buffer on the sequential axis (Shin 2017)?
- 7C switching: sleep pressure S from writes / weight growth (Tononi-Cirelli), ACh-like gating
  (Hasselmo); wake->NREM at S > theta, NREM->REM when replay error is low, REM->wake when fantasies
  are no longer discriminable. Judge at matched total compute against the fixed epoch schedule.

Constraints agreed:
- The static i.i.d. protocol stays and must not degrade (this is what keeps the sequential axis
  from being "changing the benchmark to win", which the user rejected earlier for streaming).
- A sequential regime (split-MNIST, 5 tasks x 2 classes) is added as a SECOND axis because
  consolidation is undefined without sequential experience; without it 7A is unfalsifiable.
- BP + experience replay with the same buffer budget is in every table; the bio system must match
  or beat it at equal budget or it is just a renamed replay buffer.

**Why:** records the design decision and the two guard-rails so a later session does not re-argue
them or quietly drop the static axis. Related: [[cortex-phase6-results]], [[cortex-phase5-results]].

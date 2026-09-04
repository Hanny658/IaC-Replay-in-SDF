---
name: cortex-workshop-submission
description: "MLP-Cortex workshop target (2026-09-04): CL4FMAgents @ NeurIPS 2026 (Continual Learning in the Era of Foundation Models and Embodied Agents, Sydney Dec 11-12); 8 pages excl. refs/appendix, DOUBLE-BLIND, non-archival, OpenReview, deadline Sep 7 2026 AoE (extended from Aug 29), notification Sep 29, camera-ready Oct 10; speakers incl. Gido van de Ven (the brain-inspired replay person the user targets). Draft = report/workshop/cl4fmagents.tex (NeurIPS 2026 style from community mirror lizhemin15/NeurIPS-2026-Latex-Unified; submission mode w/ line numbers; notice string overridden to the workshop name); mainline only, sidelines/negatives in appendix; manuscript-review fixes applied to BOTH versions; controller x depth figure (make_fig_depth.py)"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-09-04T08:38:38.900Z
---

Site: https://neurips26-cl4fmagents.github.io/ (no template stated; NeurIPS style assumed).
Accepted paper types include negative/reproducibility results and interdisciplinary work.
Agent-facing framing used in the intro: consolidation without downtime = the agent's problem;
local sleep = the brain's answer; the recipe (sparse codes + local error) as what a larger
system needs to inherit the guarantee.

Page budget lessons: NeurIPS 10pt fits ~4k words + 3 figures + 2 tables in 8 pages; the
full preprint (~8.5k words) needed ~45% cut; floats consumed most of pages 2-7; the last
lines were won by merging the conclusion into the limitations paragraph and shrinking Fig. 1.

**Why:** the user's first submission target; the full preprint (report/preprint.tex, 22 pp)
remains the long-form record and must stay consistent with the workshop version.
**How to apply:** any new result goes into BOTH files; keep the workshop main text at 8 pages
(References must start on page 9); camera-ready needs [final] option + author block; the
reviewer's optional figures (data-flow diagram, decomposition waterfall, energy Pareto) are
still undone. Related: [[cortex-review-round-2026-09]], [[cortex-phase22-results]].

## GPT manuscript-review round (2026-09-04, applied to BOTH versions)
Two real P0s: (1) headline replay cost was stale -- default config is batch_replay=16 ->
18,760 x 16 = 3.0e5 samples = 2.4x the night (the "1.2x" belonged to the batch-8 cell,
91.8); now stated as "2.4x; 91.8 at a matched 1.2x budget". (2) Prop. 1 claimed network
output invariance while the readout is plastic -> now hidden-activity statement + readout
corollary. Also applied: batch-indexed propositions (X, tau_{k,b}, strict inequality), gain
g fixed at 1 and removed from equations, masked heavy-ball step equation as implemented
(mu=0.9, velocity FROZEN outside the mask, decay confined, bias unit-mask), controller drive
= post-optimiser PRE-decay increment u with EMA 0.02 and eps=1e-12, "one ratio target",
Section 5 = "Scaling the mechanism: self-referenced decay and depth" (unifying principle:
every control signal referenced to the learner's own state), "may not afford downtime",
Driessen inferential-leap sentence, "tested surprise trigger", depth "consistent with",
"per FP32 accumulate", Sorrenti -> TNNLS 36(7):12668-12679 (2025), Tononi full title, Payeur
2021 added; SESLR NOT cited (arXiv shows it withdrawn for data errors). Appendix gained: data-
flow tikz, decomposition waterfall, energy Pareto (make_fig_appendix.py), baseline tuning
ranges + paired bootstrap CIs (local - narrow night +2.1 [+1.4,+2.5] p=.004; - same-substrate
night +1.7 [+0.1,+4.6] p=.25; - ER +5.8; - BP+ER +3.5). Pending at write time: held-out
seeds 3-5 (36 runs) and a cadence-2 x batch-16 matched-cost probe (seed 0: 91.1/93.7 -> the
batch-8 point remains the better 1.2x point; frequency beats batch).

---
name: cortex-phase14-15-results
description: "Phases 14-15 (2026-09-02 overnight, ~190 runs): eta sweep -- SGD plateau 0.01-0.02 (not a knife-edge), Adam fair at its default 1e-3 (no Adam cell over 10x lr reaches SGD's joint point), unmasked SGD bad+unstable at every eta; SGD transfers to feature-CIFAR 40.6+-1.0 (+3.6 over Adam) -- the 5.6-pt BP+ER regime boundary shrinks to an honest 2.7 (BP+ER own best 43.3), raw-CIFAR parity 27.2; Benna-Fusi anchor synapses: 'dissolves static price' FALSIFIED, weak symmetric anchor (3e-4,3e-4) = seq stabiliser 92.4+-0.2 at static parity; utility-exempt rotation = smooth trade-off DOMINATED by the bout gate (temporal commitment beats structural exemption); anchor x bout combo non-additive; 6-seed frontier: anchor 92.4/93.3 - combo 92.3/93.8 - bout 92.0+-0.4/94.2+-0.4 (cleanly dominates always-on 92.0/93.3) - silent 89.0/96.0"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-09-02T01:28:17.758Z
---

Overnight autonomous exploration 2026-09-01 -> 09-02 (~190 runs, all 3 seeds unless noted,
w512s5 K=1000 br16 cad1 SGD eta=0.02 unless noted).

## Phase 14: eta sensitivity (g14_*, 54 runs)
- SGD refr seq: .005->90.8, .01->91.9, .02->92.0, .05->89.8, .1->86.3; static: 90.6, 91.9,
  93.4, 93.7, 89.2.  Plateaus: seq 0.01-0.02, static 0.02-0.05; 0.02 = joint optimum.
- Adam refr: 3e-4 -> 88.6/89.3, 1e-3 -> 90.8/92.0 (seq best AT the default: comparison was
  fair), 3e-3 -> 88.9/93.6.  No Adam cell over 10x reaches SGD's joint point.
- Unmasked SGD at every eta: 84-88 with sigma 1.8-4.1 (3-8x masked) -- isolation-stabilises
  claim is now cross-eta.
- Bout gate inherits the plateau (0.01->91.9/92.6, 0.02->92.1/94.2, 0.05->90.0/93.8).
- Eta shifts down with regime: CIFAR peak at 0.01 (0.005 confirmed worse).

## Phase 15A: SGD transfer to CIFAR (cif/cfeat_sgd_refr_e*, 36+ runs)
- Feature-CIFAR: SGD e01 seq 40.6 +-1.0 (n=6) vs local Adam 37.0; static 38.6 +-1.4 (n=6) vs
  35.6.  BP+ER lr sweep (bp_lr knob added): 3e-4 -> 43.3 +-1.0 (best), 1e-3 -> 42.6, 3e-3 ->
  39.9.  REGIME BOUNDARY REVISED: 5.6-pt gap was mostly the local learner's optimiser; honest
  gap at both sides' best = 2.7.  Boundary narrowed, not closed.
- Raw CIFAR (underfit): SGD 27.2 +-1.4 = Adam 27.1 -- the stateless gain does not transfer to
  the underfit regime (parity).

## Phase 15B: two-timescale anchor synapses (Benna-Fusi minimal; `anchor=(lam,mu)` in
cortex.local_update, ticks on waking steps only so replay exactness untouched)
- Grid lam{1e-3,3e-3} x mu{1e-4,3e-4}: static NEVER exceeds 93.4 -- "anchor dissolves the
  settled-coalition static price" FALSIFIED; strong coupling hurts both axes (lagging drag).
- Weak-coupling probe (3e-4,1e-4)/(3e-4,3e-4)/(1e-3,1e-3): seq 92.3-92.4 consistently,
  static 92.9-93.3.  Best (3e-4,3e-4): seq 92.4 +-0.2 (variance halved vs 92.0 +-0.6
  baseline) at static parity -> a SEQ STABILISER that weakly dominates always-on.
## Phase 15C: utility-exempt rotation (refr_util, top-q long-use units never benched)
- q=.10 -> 90.9 +-0.1 / 94.2 +-0.2; q=.25 -> 89.1 +-0.4 / 95.4 +-0.2.  Smooth trade-off line
  between always-on and silent, but the bout gate beats util10 by +1.1 seq at equal static:
  TEMPORAL COMMITMENT > STRUCTURAL EXEMPTION (phase-10 lesson recurs at the utility level).
  13b's localisation of the price was right; exempting those coalitions pays proportionally.
## Phase 15D: anchor x bout combo: 92.3 +-0.3 / 93.8 +-0.5 -- between the parents, dominates
neither; anchor and gate do not compose additively.

## Phase 15F: boundary decomposition (cfeat, BP+ER given the local substrate piecewise)
Ladder at each cell's better lr: dense narrow 43.3 -> wide dense 44.5 +-0.4 (width HELPS BP)
-> +30% dist wiring (same CortexNet mask generator, same seed; `bp_conn`) 43.9 +-0.7 (wiring
costs 0.6) -> +5% k-WTA (`bp_kwta`, KWTA module) 39.5 +-0.6 = BELOW local learner 40.6 +-1.0.
THE BOUNDARY IS k-WTA ACTIVATION SPARSITY, NOT BACKPROP'S CREDIT ASSIGNMENT -- on the shared
substrate the local rule gives nothing away.  Manuscript: transfer subsection extended with
the decomposition; limits attribution updated; FW(i) -> "buy back kWTA's feature cost without
losing the isolation channel it powers".

## Phase 15E: energy re-measure on the new default (scripts/p15_spike.py, seed 0, v10 protocol)
- Bout-gate system: T_s=8 -> 93.5% @ 118 nJ (1/20.8 of dense rate 2461 nJ, 1/4.4 of
  event-driven rate 516 nJ), MONOTONE in T_s (16 -> 94.0 @ 239).
- The v10 calibration-interaction claim REVERSES SIGN under SGD: rotation-trained nets are now
  calibration-robust; rotation-FREE control degrades past T_s=8 (94.5 -> 91.2 @ 32).  Cause
  consistent with rotation shaping sparser codes (hidden dim 12.7 vs 28.8; 60 vs 91
  spikes/sample @ T_s=8).  Energy subsection + FW(vi) rewritten.

## 6-seed hardened frontier (MNIST)
anchor 92.4/93.3 -> combo 92.3/93.8 -> bout M=2048 92.0 +-0.4 / 94.2 +-0.4 -> silent 89.0/96.0.
Bout gate now CLEANLY weakly dominates always-on (92.0 +-0.6 / 93.3 +-0.2): seq exact tie,
static +0.9.  Always-on is obsolete as an operating point; bout = the sequential default.

**Why:** one night settled every open robustness question (eta, Adam fairness, unmasked
control), revised the regime-boundary narrative with a fairness control on BOTH sides, and
tested two literature-grounded mechanisms to a clean verdict each (anchor = stabiliser not
dissolver; utility exemption = dominated trade-off).
**How to apply:** manuscript updates pending morning discussion: (1) eta-robustness paragraph,
(2) regime-boundary section rewrite (gap 5.6 -> 2.7, both sides at best), (3) bout-gate rows
become 92.0 +-0.4/94.2 +-0.4 at n=6 and "weakly dominates always-on" strengthens, (4) anchor +
util as short results/negative entries (Benna-Fusi citation needed), (5) frontier figure
candidate.  Related: [[cortex-phase13-results]], [[cortex-phase12-results]],
[[cortex-phase11-results]].

---
name: cortex-review-round-2026-09
description: "Code-review round (2026-09-04, ~240 runs): H1 CIFAR-10 feature-front threshold leak REAL (test features rectified against test-set stats; fixed to train-only, cache feat256_trthr; all 49 cifarf configs re-run: mean delta -0.20, |delta| 0.23, max 1.1 on bp_er_1000, ALL orderings intact, feature boundary 1.9->1.7); H2 spike calibration on Xte REAL (fixed to Xtr, energy numbers change <=0.3); H3 selection-on-test REAL (no val split ever) -> --val mode, 12 headline configs replicated on held-out 10%: Spearman 0.92, mean gap 0.9, all orderings survive -> Appendix G + Limits statement; H4 exactness overclaim PARTLY real -> diag_drift measured per replay step (default: 0.27% wake predictions change, 0.36% hidden codes, 1e-4 asleep units flip; isolated readout 0.001%) -> Remark/abstract/Discussion narrowed; M5 syn_mask==replay REAL confound -> explicit replay= flag; mirror re-run WORSE (85.0->77.9 s10, 81.3->70.3 5%; old cell had burst baseline frozen) -> 'eight points' -> 'fifteen'; v1 controller formula had been overwritten by v2 -> restored as kp_adapt_mode='grad' (bitwise reproduction verified at OMP=4); NEW: bitwise reproducibility only at fixed OMP_NUM_THREADS (unchanged config differs 0.56 at OMP=2 vs 4)"
metadata: 
  node_type: memory
  type: project
  originSessionId: b260108a-3c8a-4668-bb0c-63a09e66c838
  modified: 2026-09-03T21:55:24.324Z
---

Reviewer file: MLP-Cortex/report/critic.md (code-only review). Verdicts: H1/H2/H3 real, H4
partly (paper's propositions were scoped; Remark's "0.0 drift" and two "provably" phrasings
were not); M1 (run_bio seed in filename), M2 (run args in checkpoint + resume check), M3
(runs.csv stale), M5 (replay flag) all real and fixed; M4 (course evaluate.py heavy-tail
columns chosen before inner CV) real but label-free and test-clean -- NOT touched (course code).

Archive of pre-fix checkpoints: results/bio/seq/parts_review_archive/{cfeat_leak,
mirror_replayflag}/ (same names) -- keeps old-vs-new comparable per seed.

**Why:** these are the exact objections a referee would raise; the fixes cost ~240 runs and
changed no ordering, which is itself the strongest robustness statement the paper now makes
(Appendix G).
**How to apply:** (1) never estimate any statistic on the test split, even label-free ones;
(2) any masked WAKING update must pass replay=False; (3) keep old formula variants as
switchable modes instead of overwriting (kp_adapt_mode); (4) run everything at
OMP_NUM_THREADS=4 if bitwise reproduction matters, and record omp_threads; (5) use --val for
any future selection, test once at the end. Related: [[cortex-phase19-results]],
[[cortex-phase20-results]], [[assignment2-reproducibility]].

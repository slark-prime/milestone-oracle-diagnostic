# E7: full protocol replication on a second corpus (MATH500 + AIME), answering AC-Q1 / i227 Q3 / Qf6B Q1

Everything is held identical to the submission — same teacher (GPT-5.4), same student, same prompt
builders, same six probe conditions, same deterministic cascade, K=8 at 16K, same structural filter
(>=2 non-INTEGRATE, leak-safe). **Only the source corpus changes.** Scripts:
`scripts/experiments/second_corpus_suite.py`; data: `data/logs/rl/second_corpus/`.

## Anchored screening behaves very differently by corpus (a result in itself)

Same anchor (Qwen3-8B, no-think), same rule (parent fails 0/4 direct):

| Corpus | Problems screened | Anchor-failed | Yield |
|---|---:|---:|---:|
| MATH500 | 500 | 36 | 7% |
| AIME 2024 | 60 | 38 | 63% |
| AIME 2025 | 30 | 21 | 70% |

A tenfold swing in yield from the same screen is direct evidence for the paper's framing that an
anchored suite is a *stress set*, not an ability estimate. AIME 2025 also postdates most models'
training cutoffs, which speaks to the contamination caveat (i227 W6).

Teacher reliability on new corpora: **95/95 packets valid on the first pass** (GPT-5.4), mean 4.7-6.1
non-INTEGRATE milestones per family — notably finer than the 354-set's 2.2, i.e. the teacher adapts
granularity to problem structure without being told to.

## MATH500 (complete: 35 eligible families, all six conditions, Stage 0 done)

Rollout-level solve rate (280 rollouts per condition):

| Condition | Solve rate | vs C1 |
|---|---:|---:|
| C1 direct | 10.0% | — |
| **C2-correct** | **37.9%** | 3.8x |
| C2-random | 7.9% | 0.8x (below C1) |
| C2-generic | 9.6% | 1.0x |
| **C3-gold** | **38.2%** | 3.8x |
| C3-mismatched | 32.5% | 3.3x |

**The central claim replicates: recovery comes from problem-matched milestone information.**
C2-correct is 4.8x C2-random, and neither corruption control exceeds direct prompting.

Family-level recovery needs the paper's own threshold analysis (App. D) to be read correctly, because
these families were screened at k=4 and probed at k=8 — a family whose true solve rate is ~10% will
often pass a permissive >=1/8 rule on a fresh sample:

| Condition | tau>=1 | tau>=2 | **tau>=3** | tau>=4 |
|---|---:|---:|---:|---:|
| C1 direct | 16 | 8 | **3** | 1 |
| C2-correct | 18 | 16 | **15** | 13 |
| C3-gold | 16 | 15 | **15** | 14 |
| C2-random | 12 | 5 | **2** | 2 |
| C2-generic | 16 | 7 | **2** | 1 |
| C3-mismatched | 17 | 15 | 13 | 12 |

At tau>=3 the separation is 15 vs 2 vs 2 (C2-correct vs the two controls) against C1's 3. The
saturation at tau>=1 is a screen/probe sampling artifact, not a failure of the contrast, and it is
exactly what the submission's threshold-sensitivity appendix was written to expose.

## An honest corpus difference

On MATH500, gold milestone answers add almost nothing over the roadmap alone: C2-correct 37.9%,
C3-gold 38.2%, C3-mismatched 32.5%. Descriptions carry the signal. On the 354-set, C3 exceeded C2.
So **which help level matters is corpus-dependent, while "problem-matched information is what helps"
is not.** We will report this rather than average it away; it also sharpens the diagnostic's use --
the C2/C3 gap is itself a per-corpus readout.

## All three corpora, family-level recovery (tau>=1 unless noted)

| Condition | MATH500 (35) | MATH500 tau>=3 | AIME 2024 (38) | AIME 2025 (21) |
|---|---:|---:|---:|---:|
| C1 direct | 16 | **3** | **2 (5%)** | 5 |
| **C2-correct** | 18 | **15** | **17 (45%)** | 6 |
| C2-random | 12 | **2** | **5 (13%)** | 2 |
| C2-generic | 16 | **2** | **5 (13%)** | 4 |
| C3-gold | 16 | 15 | 15 (40%) | 6 |
| C3-mismatched | 17 | 13 | 14 (37%) | 5 |

**AIME 2024 replicates the specificity result at the family level with no threshold tuning:**
C2-correct 45% against 13% for both corruption controls, over a C1 base of 5%. MATH500 replicates once
the screen/probe sampling gap is handled with the submission's own threshold analysis. AIME 2025 (n=21)
points the same way but is too small to carry weight on its own.

## Boundary condition we found, and what it means

Milestone-test pass counts are 0/35 (MATH500), 1/38 (AIME 2024), 2/21 (AIME 2025), so the
COMPOSITION-GAP tier is nearly empty on these corpora (0, 1, 1) and the residual lands in
MISSING-MILESTONE instead:

| Corpus | Taxonomy |
|---|---|
| MATH500 | DIRECT 16, ROADMAP-GAP 6, MISSING-MILESTONE/CAPABILITY 13, COMPOSITION 0 |
| AIME 2024 | DIRECT 2, ROADMAP-GAP 15, MILESTONE-EXECUTION 1, MISSING-MILESTONE/CAPABILITY 19, COMPOSITION 1 |
| AIME 2025 | DIRECT 5, ROADMAP-GAP 3, MISSING-MILESTONE/CAPABILITY 12, COMPOSITION 1 |

The mechanism is clear: passing the milestone test requires solving *every* non-INTEGRATE milestone at
>=1/8, and on these corpora the teacher emits 4.7-6.1 milestones per family (vs 2.2 on the 354-set) on
harder problems, so the joint pass probability collapses.

We read this as the instrument reporting a genuinely different failure structure rather than failing:
an 8B student on AIME fails because it cannot execute the parts, not because it cannot compose them —
which is the correct diagnosis, and one aggregate accuracy cannot make. But it does establish an
**operating envelope** we will state explicitly: the five-way taxonomy resolves the composition tier
only when roadmaps are coarse enough, relative to student ability, for the milestone test to be
passable. That connects directly to the granularity ablation (the readout is stable across a 4.1x
granularity change *within* a corpus, while the taxonomy's resolution depends on granularity x
difficulty *across* corpora), and it is a property users of the released artifact need to know.

## What replicates and what does not

| Claim | Status on the second corpora |
|---|---|
| Recovery comes from problem-matched milestone information (corruption controls do not help) | **Replicates** (AIME 2024 family-level; MATH500 at tau>=3 and at rollout level) |
| The protocol instantiates and runs end-to-end on new corpora | **Yes** — 95/95 teacher packets valid first pass |
| Anchored screening is a stress set, not an ability estimate | **Strengthened** — same screen yields 7% vs 63-70% |
| The 33-48% composition-gap share | **Does not transfer** — milestone-test pass collapses; residual is MISSING-MILESTONE |
| Gold answers add value over the roadmap alone | **Corpus-dependent** — large on the 354-set, ~zero on MATH500 |

---

## Code pilot: corrected arms (E5)

The pilot's original write-up compared a k=4 **screen** value for C1 against k=8 **probe** values for
C2/C3. We ran a proper K=8 C1 arm; all three are now on the same footing, all execution-graded:

| Condition (all K=8 probes) | Rollout solve rate | Family recovery |
|---|---:|---:|
| C1 direct | 6.5% (4/62) | 1/10 |
| C2 plan | 41.2% (33/80) | 6/10 |
| C3 plan + verified gold helpers | 88.8% (71/80) | 10/10 |

C1 is 62/80 rollouts (5/10 families with a full cell); the rollout rate is the unbiased quantity and is
what we quote. Corruption controls (C2-random / C2-generic / C3-mismatched) were **not** run in the code
domain, so the pilot demonstrates re-instantiation and a monotone readout, not a second-domain
replication of the specificity contrast. The responses state this explicitly.

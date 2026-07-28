# Correct-trajectory waypoint analysis

Split of the released unaided (C1) traces by whether each rollout reaches the
correct final answer, measured against two independent decompositions of the
same 39 families (the submission teacher's roadmaps and the second teacher's
atomic re-decompositions).

Headline: on rollouts that reach the correct answer, 60-88% verifiably pass
through at least one waypoint of one of the two valid decompositions, and every
model sits above its count-matched control (13-40%; two random other-family
roadmaps, matching the probe count). Per-milestone rates and the per-model
breakdown, including DeepSeek-V3.1's preference for the alternative
decomposition on correct rollouts (19.4% vs 12.6%), are in
`trace_attainment_alt_paths.json`. Same containment test and verifier as the
main trace analysis one directory up; no new sampling.

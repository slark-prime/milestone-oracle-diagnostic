"""Compare 4K vs 16K cross-model panel fingerprints side-by-side.

Generates:
  - Per-model fingerprint (direct/desc/gold/unrec/ceiling) at 4K and 16K
  - Delta column
  - LaTeX table for the appendix: 'Decoding-budget sensitivity'

Reads:
  - data/logs/rl/oracle_panel/*.jsonl       (4K data)
  - data/logs/rl/oracle_panel_16k/*.jsonl   (16K data, if present)

Outputs:
  - data/logs/rl/panel_4k_vs_16k.json
  - docs/latex/tables/panel_4k_vs_16k.tex
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import sys

REPO = Path(__file__).resolve().parent.parent
PANEL_4K = REPO / "data/logs/rl/oracle_panel"
PANEL_16K = REPO / "data/logs/rl/oracle_panel_16k"
OUT_JSON = REPO / "data/logs/rl/panel_4k_vs_16k.json"
OUT_TEX = REPO / "docs/latex/tables/panel_4k_vs_16k.tex"

if not PANEL_4K.exists():
    sys.exit(
        f"ERROR: required input directory {PANEL_4K} not found.\n"
        "This script needs the 4K oracle_panel/ directory, which is not "
        "packaged with the public release (size and provenance considerations). "
        "The precomputed output table panel_4k_vs_16k.tex is included under "
        "code/docs/latex/tables/. See README section 'Reproducing the paper "
        "numbers' for which scripts run end-to-end on this release."
    )

MODELS = [
    ("qwen3_8b_pre_rl", "pre-RL"),
    ("base_2k_step_180", "Base-2k"),
    ("mile_2k_step_180", "Mile-2k"),
    ("gpt_oss_20b", "gpt-oss-20b"),
    ("llama_3_3_70b_instruct", "Llama-70B"),
    ("deepseek_v3_1", "DeepSeek-V3.1"),
]


def fingerprint(panel_dir, slug):
    """Return {direct, desc, gold, unrec, ceiling} or None if missing."""
    fn = panel_dir / f"{slug}.jsonl"
    if not fn.exists():
        return None
    n_lines = sum(1 for _ in open(fn))
    if n_lines < 2124:
        return None  # incomplete
    by_pid = defaultdict(dict)
    for line in open(fn):
        r = json.loads(line)
        by_pid[r["pid"]][r["condition"]] = r["n_correct"]
    counts = Counter()
    for pid, conds in by_pid.items():
        c1 = conds.get("C1_direct", 0)
        c2 = conds.get("C2_descriptions", 0)
        c3 = conds.get("C3_gold_answers", 0)
        if c1 >= 1: counts["direct"] += 1
        elif c2 >= 1: counts["desc_only"] += 1
        elif c3 >= 1: counts["gold_only"] += 1
        else: counts["unrecovered"] += 1
    ceiling = counts["direct"] + counts["desc_only"] + counts["gold_only"]
    return {
        "direct": counts["direct"],
        "desc_only": counts["desc_only"],
        "gold_only": counts["gold_only"],
        "unrecovered": counts["unrecovered"],
        "ceiling": ceiling,
    }


def main():
    rows = []
    for slug, label in MODELS:
        fp4k = fingerprint(PANEL_4K, slug)
        fp16k = fingerprint(PANEL_16K, slug)
        rows.append({"slug": slug, "label": label, "fp_4k": fp4k, "fp_16k": fp16k})

    print(f"{'Model':<16s}  " + "  ".join(f"{h:>10s}" for h in ["dir 4K", "dir 16K", "Δ dir", "ceil 4K", "ceil 16K", "Δ ceil"]))
    print("-" * 90)
    for r in rows:
        if r["fp_4k"] is None or r["fp_16k"] is None:
            status = "4K: " + ("✓" if r["fp_4k"] else "—") + "  16K: " + ("✓" if r["fp_16k"] else "—")
            print(f"{r['label']:<16s}  {status}")
            continue
        d4, d16 = r["fp_4k"]["direct"], r["fp_16k"]["direct"]
        c4, c16 = r["fp_4k"]["ceiling"], r["fp_16k"]["ceiling"]
        print(f"{r['label']:<16s}  " +
              f"{d4:>10d}  {d16:>10d}  {d16-d4:>+10d}  {c4:>10d}  {c16:>10d}  {c16-c4:>+10d}")

    OUT_JSON.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {OUT_JSON}")

    # LaTeX table — only include rows where both are available
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write(r"""\begin{table}[h]
\centering
\footnotesize
\caption{Decoding-budget sensitivity: fingerprint at \texttt{max\_tokens=4096} vs.\ \texttt{max\_tokens=16384} on the same 354 families, $K{=}8$, same prompt, same verifier. Models trained with long reasoning chains (Base-2k, Mile-2k, gpt-oss-20b, DeepSeek-V3.1) shift meaningfully at 16K; pre-RL Qwen3-8B-Base and Llama-3.3-70B-Instruct (neither RL-tuned on long chains in our setup) are stable within bootstrap noise.}
\label{tab:panel_4k_vs_16k}
\vspace{4pt}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrr}
\toprule
& \multicolumn{4}{c}{\textbf{4K}} & \multicolumn{4}{c}{\textbf{16K}} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9}
\textbf{Model} & \textbf{dir.} & \textbf{desc} & \textbf{gold} & \textbf{ceil.} & \textbf{dir.} & \textbf{desc} & \textbf{gold} & \textbf{ceil.} \\
\midrule
""")
        for r in rows:
            if r["fp_4k"] is None or r["fp_16k"] is None:
                continue
            f4 = r["fp_4k"]
            f16 = r["fp_16k"]
            parts = [r["label"]]
            for fp in [f4, f16]:
                parts += [str(fp["direct"]), str(fp["desc_only"]), str(fp["gold_only"]), str(fp["ceiling"])]
            f.write(" & ".join(parts) + " \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()

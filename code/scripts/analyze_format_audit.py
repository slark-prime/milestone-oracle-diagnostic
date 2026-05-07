"""Summarize format / extraction audit into a per-model bucket table.

Input: data/logs/rl/format_audit/{model}.jsonl (per-rollout records with
       bucket in {no_boxed, no_boxed_truncated, unparsable, parsed_wrong,
       parsed_correct, error}).

Output:
  - data/logs/rl/format_audit_summary.json
  - docs/latex/tables/format_audit.tex
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import sys

REPO = Path(__file__).resolve().parent.parent
AUDIT_DIR = REPO / "data/logs/rl/format_audit"
OUT_JSON = REPO / "data/logs/rl/format_audit_summary.json"
OUT_TEX = REPO / "docs/latex/tables/format_audit.tex"

if not AUDIT_DIR.exists():
    sys.exit(
        f"ERROR: required input directory {AUDIT_DIR} not found.\n"
        "This script needs the format_audit/ raw rollout logs, which are not "
        "packaged with the public release. The precomputed format_audit.tex "
        "is included under code/docs/latex/tables/. See README for which "
        "scripts run end-to-end on this release."
    )

MODELS = [
    ("qwen3_8b_pre_rl", "Qwen3-8B-Base (pre-RL)"),
    ("mile_2k_step_180", "Mile-2k"),
    ("llama_3_3_70b_instruct", "Llama-3.3-70B-Inst."),
    ("deepseek_v3_1", "DeepSeek-V3.1"),
]

CONDITIONS = ["C1_direct", "C3_gold"]
BUCKETS = ["no_boxed", "no_boxed_truncated", "unparsable", "parsed_wrong", "parsed_correct"]


def main():
    summary = {}
    print(f"{'Model':<26s}  {'Cond':<10s}  {'n':>5s}  " +
          "  ".join(f"{b:>16s}" for b in BUCKETS))
    print("-" * 130)

    for slug, label in MODELS:
        fn = AUDIT_DIR / f"{slug}.jsonl"
        if not fn.exists():
            print(f"{label:<26s}  (no data)")
            continue
        counts = defaultdict(Counter)
        for line in open(fn):
            r = json.loads(line)
            counts[r["condition"]][r["bucket"]] += 1

        summary[slug] = {"label": label, "counts": {}}
        for cond in CONDITIONS:
            total = sum(counts[cond].values())
            if total == 0:
                continue
            cells = []
            summary[slug]["counts"][cond] = {"total": total}
            for b in BUCKETS:
                n = counts[cond].get(b, 0)
                frac = n / total if total else 0
                cells.append(f"{n:>3d} ({100*frac:4.1f}%)")
                summary[slug]["counts"][cond][b] = n
            short_cond = cond.replace("_", " ").replace("direct", "dir.").replace("gold", "gold")
            print(f"{label:<26s}  {short_cond:<10s}  {total:>5d}  " + "  ".join(f"{c:>16s}" for c in cells))

    OUT_JSON.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_JSON}")

    # LaTeX table
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write(r"""\begin{table}[h]
\centering
\scriptsize
\caption{Format / extraction-failure audit on a 50-family random subset (seed 42), $K{=}8$ rollouts per family per condition, raw responses logged. Each rollout is classified into a bucket by the extractor: \textsc{no\_boxed} (no \texttt{\textbackslash boxed\{\}} in response), \textsc{no\_boxed\_trunc} (no boxed and hit \texttt{max\_tokens}), \textsc{unparsable} (boxed but content unparseable), \textsc{parsed\_wrong} (parsed, verifier says wrong), \textsc{parsed\_correct} (parsed, verifier accepts). Percentages are of the 400 rollouts in each (model, condition) cell.}
\label{tab:format_audit}
\vspace{4pt}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrrrr}
\toprule
\textbf{Model} & \textbf{Condition} & \textbf{$n$} & \textbf{no\_boxed} & \textbf{no\_boxed\_trunc} & \textbf{unparsable} & \textbf{parsed\_wrong} & \textbf{parsed\_correct} \\
\midrule
""")
        for slug, label in MODELS:
            if slug not in summary:
                continue
            for cond in CONDITIONS:
                if cond not in summary[slug]["counts"]:
                    continue
                c = summary[slug]["counts"][cond]
                total = c["total"]
                short_cond = cond.replace("C1_direct", "$C_1$").replace("C3_gold", "$C_3$-gold")
                parts = [label, short_cond, str(total)]
                for b in BUCKETS:
                    n = c.get(b, 0)
                    frac = n / total if total else 0
                    parts.append(f"{100*frac:.1f}\\%")
                f.write(" & ".join(parts) + " \\\\\n")
        f.write(r"""\bottomrule
\end{tabular}
\end{table}
""")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    main()

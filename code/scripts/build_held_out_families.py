"""Build held-out diagnostic family file from pipeline output.

Input:  data/numina/held_out_150/{results.jsonl, packets.jsonl, problems.jsonl}
Output: data/logs/rl/held_out_diagnostic_families.jsonl
        (same schema as diagnostic_multi_families.jsonl — n=32)

Filter: parent p_hat=0 on pre-RL Qwen3-8B-Base (from orig_p_hat in packets),
        all milestones p_hat>0, ≥2 non-INTEGRATE milestones.
"""
import json
import uuid
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "data/numina/held_out_150/results.jsonl"
PACKETS = REPO / "data/numina/held_out_150/packets.jsonl"
PROBLEMS = REPO / "data/numina/held_out_150/problems.jsonl"
OUT = REPO / "data/logs/rl/held_out_diagnostic_families.jsonl"


def main():
    # Problems: problem_idx → problem/answer/pid
    problems = {}
    for i, line in enumerate(open(PROBLEMS)):
        rec = json.loads(line)
        idx = rec.get("problem_idx", i)
        problems[idx] = rec

    # Packets: problem_idx → orig_p_hat
    packets = {}
    for line in open(PACKETS):
        rec = json.loads(line)
        packets[rec["problem_idx"]] = rec

    # Milestones: problem_idx → list of milestones
    milestones_by_idx = defaultdict(list)
    for line in open(RESULTS):
        rec = json.loads(line)
        if rec.get("type") == "ORIGINAL":
            continue
        idx = rec["problem_idx"]
        milestones_by_idx[idx].append(rec)

    families = []
    for idx, prob in problems.items():
        pkt = packets.get(idx)
        if not pkt:
            continue
        orig_p_hat = pkt.get("orig_p_hat", -1)
        if orig_p_hat != 0:
            continue
        ms = milestones_by_idx.get(idx, [])
        if not ms:
            continue
        if not all(m.get("p_hat", 0) > 0 for m in ms):
            continue
        non_int = [m for m in ms if m.get("type") != "INTEGRATE"]
        if len(non_int) < 2:
            continue

        # Build family record matching diagnostic_multi_families.jsonl schema
        parent_prompt = (
            "Solve the following problem. Provide your final answer clearly.\n\n"
            f"Problem:\n{prob['problem']}\n"
        )
        family = {
            "pid": prob.get("pid", str(uuid.uuid4())),
            "parent_prompt": parent_prompt,
            "parent_answer": prob["answer"],
            "parent_note": "ACCEPT IF equivalent final answer is present.",
            "milestones": []
        }
        default_out = "Put your answer in \\boxed{}."
        for m in ms:
            out_instr = m.get("student_output_instruction", default_out)
            ms_prompt = (
                "You are solving one milestone from a larger problem.\n"
                "Keep your response focused on this milestone only.\n\n"
                f"Original problem (context):\n{prob['problem']}\n\n"
                f"Milestone:\n{m['description']}\n\n"
                f"Output instruction: {out_instr}\n"
            )
            family["milestones"].append({
                "prompt": ms_prompt,
                "answer": m["canonical_answer"],
                "note": "ACCEPT IF equivalent final answer is present.",
                "type": m.get("type", "UNKNOWN"),
                "pre_train_phat": m.get("p_hat", 0),
            })
        families.append(family)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for fam in families:
            f.write(json.dumps(fam) + "\n")

    from collections import Counter
    print(f"Wrote {len(families)} families to {OUT}")
    type_dist = Counter()
    for fam in families:
        for m in fam["milestones"]:
            type_dist[m["type"]] += 1
    print(f"Milestone type distribution: {dict(type_dist)}")
    print(f"Milestones per family distribution: {Counter(len(f['milestones']) for f in families)}")


if __name__ == "__main__":
    main()

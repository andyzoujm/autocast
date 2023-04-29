import json
import copy

temporal_data = json.load(open(f"bm25ce_temporal_train.json"))

descending = True  # higher relevance score better
static_data = []
for question in temporal_data:
    new_question = copy.deepcopy(question)
    del new_question["targets"]

    all_ctxs = []
    all_scores = []
    for target in question["targets"]:
        if target["ctxs"]:
            all_ctxs.extend(target["ctxs"])
            all_scores.extend([float(ctxs["score"]) for ctxs in target["ctxs"]])
    sorted_idx = [
        x
        for _, x in sorted(zip(all_scores, range(len(all_scores))), reverse=descending)
    ]
    new_question["ctxs"] = [all_ctxs[i] for i in sorted_idx]

    static_data.append(new_question)

with open(f"bm25ce_static_train.json", "w", encoding="utf-8") as writer:
    writer.write(json.dumps(static_data, indent=4, ensure_ascii=False) + "\n")

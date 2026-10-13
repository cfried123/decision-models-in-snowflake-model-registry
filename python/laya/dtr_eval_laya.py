"""Score a Laya checkpoint on DTR-Bench with the same abstention policy as strands-decider.

`strands_decider.dtr_eval` holds the policy (noul band 0.25, choice/score floor 0.5) and the
metrics; Laya's answer dicts have the same `type`/`noul`/`probabilities` shape, so it applies
unchanged and the four models' numbers are comparable.

    python python/laya/dtr_eval_laya.py <checkpoint or hub id> [--out results.json]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SD = Path(os.environ.get("STRANDS_DECIDER_REPO", ROOT.parent / "strands-decider"))
sys.path.insert(0, str(SD / "src"))
try:
    from strands_decider.dtr_eval import decide, score  # noqa: E402
except ImportError:  # inside the ML Job payload, dtr_eval.py ships at the top level
    from dtr_eval import decide, score  # noqa: E402


def evaluate(agent, items: list[dict]) -> dict:
    preds, rows = {}, []
    for it in items:
        t0 = time.perf_counter()
        ans = agent.predict(it["state"], {"d": it["question"]})["answers"]["d"]
        ms = (time.perf_counter() - t0) * 1000
        if it["kind"] == "score":  # probabilities keyed by level index, as DTR-Bench expects
            ans = {**ans, "probabilities": {str(k): v for k, v in ans["probabilities"].items()}}
        decision, top, conf = decide(ans)
        preds[it["item_id"]] = {"decision": decision, "argmax": top, "confidence": conf, "latency_ms": ms}
        rows.append({"item_id": it["item_id"], "expected": it["expected"], **preds[it["item_id"]]})
    return {"metrics": score(items, preds), "predictions": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--bench", default=str(SD / "data" / "dtr_bench.jsonl"))
    ap.add_argument("--out")
    a = ap.parse_args()
    import laya

    agent = laya.load(a.checkpoint)
    items = [json.loads(line) for line in Path(a.bench).read_text().splitlines()]
    res = evaluate(agent, items)
    m = res["metrics"]
    for g in ("all", "in_domain", "holdout", "choice", "noul", "score"):
        if g in m:
            print(g, {k: round(v, 4) if isinstance(v, float) else v for k, v in m[g].items()})
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))

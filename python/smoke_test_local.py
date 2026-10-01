"""Run DeciderModel in-process on six JevBench items and score them with JevBench.

Checks, before anything is registered or any GPU is started:
  * the custom model loads from a ModelContext the way the registry loads it,
  * its output passes JevBench's own TypeSafeAdapter parsing (only the HTTP
    call is replaced) and score_task validity rules,
  * what the registry pickles (the class by value and its context, not the
    instance) stays small.

    .venv/bin/python python/smoke_test_local.py

Off CUDA, DeciderModel runs on CPU in bf16, the dtype used on CUDA: on an M3 Max
with torch 2.10, MPS returned NaN for every Score question (fp16 and bf16).
"""
import json
import sys
import time
from pathlib import Path

import cloudpickle
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache" / "jevbench"))
sys.path.insert(0, str(ROOT / "python"))

from jevbench.adapters import typesafe as ts  # noqa: E402
from jevbench.scoring import score_task  # noqa: E402
from jevbench.tasks import Task  # noqa: E402
from snowflake.ml.model import custom_model  # noqa: E402

WEIGHTS_DIR = ROOT / "models" / "decider-2b"


def pick_items(rows):
    by_type = {}
    for r in rows:
        by_type.setdefault(r["QTYPE"], []).append(r)
    chosen = [by_type["choice"][0], by_type["noul"][0], by_type["score"][0]]
    chosen.append(next(r for r in rows if r["STATE_IS_OBJECT"]))
    chosen.append(max(rows, key=lambda r: len(r["STATE_JSON"])))                 # longest hard item
    chosen.append(next(r for r in rows if r["TIER"] == "hard" and r["QTYPE"] == "score"))
    return chosen


def main():
    rows = [json.loads(l) for l in (ROOT / "runs" / "jevbench_items.jsonl").read_text().splitlines()]
    items = pick_items(rows)
    ctx = custom_model.ModelContext(artifacts={"weights": str(WEIGHTS_DIR)})

    import decider_model
    t0 = time.perf_counter()
    model = decider_model.DeciderModel(ctx)
    print(f"load {time.perf_counter() - t0:.1f}s; runtime {model.runtime_json}")
    # What log_model stores (snowflake-ml-python 2.3.0, model_handlers/custom.py).
    cloudpickle.register_pickle_by_value(decider_model)
    size = len(cloudpickle.dumps((type(model), model.context)))
    print(f"registry pickle (class by value, context) = {size:,} bytes")

    df = pd.DataFrame({"STATE_JSON": [r["STATE_JSON"] for r in items],
                       "QUESTIONS_JSON": [r["QUESTIONS_JSON"] for r in items]})
    out = model.system_one(df)
    assert list(out.columns) == ["ANSWER_JSON", "ELAPSED_MS", "INPUT_TOKENS", "RUNTIME_JSON"]
    assert str(out["ELAPSED_MS"].dtype) == "float64" and str(out["INPUT_TOKENS"].dtype) == "int64"

    adapter = ts.TypeSafeAdapter(endpoint="snowflake-service", model="decider-2b", key_env="")
    failures = 0
    for r, (_, o) in zip(items, out.iterrows()):
        task = Task.from_dict(json.loads(r["ITEM_JSON"]))
        response = json.loads(o["ANSWER_JSON"])
        # The adapter's request must be the text stored in JEVBENCH_ITEMS.
        req = adapter.build_request(task)
        assert json.dumps(req["state"], ensure_ascii=False) == r["STATE_JSON"], task.id
        assert json.dumps(req["questions"], ensure_ascii=False) == r["QUESTIONS_JSON"], task.id
        ts.http_post_json = lambda url, body, headers, timeout, _resp=response: (200, _resp, 0.0)
        res = adapter.run(task)
        scored = score_task(res.probs, task) if res.ok else {"valid": False, "error": res.error}
        failures += not (res.ok and scored["valid"])
        print(f"{task.id:40s} {r['QTYPE']:6s} tokens={o['INPUT_TOKENS']:5d} {o['ELAPSED_MS']:8.0f} ms "
              f"ok={res.ok} valid={scored['valid']} predicted={scored.get('predicted')} "
              f"expected={task.expected} correct={scored.get('correct')}")
    print("FAIL" if failures else "PASS", f"{len(items) - failures}/{len(items)} valid through the JevBench adapter")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

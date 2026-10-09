"""Score a frontier LLM (default claude-sonnet-5) on DTR-Bench through AI_COMPLETE, with
the same items, the same abstention policy and the same scorer as the strands-decider
runs, so the three numbers (base, fine-tuned, LLM) are comparable.

The LLM is asked for one option plus its probability for that option (or P(true) for
a yes/no item); strands_decider.dtr_eval.decide then routes low-confidence answers to
the review queue exactly as it does for the decider. Item text goes in as bind
variables; show_details gives the token usage that 09_cost_accounting prices.

    SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/bench_llm_dtr.py [model]

Writes runs/llm_dtr/<model>.json.
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT.parent / "strands-decider" / "src"))

from bench_config import FQ_SCHEMA  # noqa: E402
from snowpark_session import create_snowpark_session  # noqa: E402

SQL = ("SELECT AI_COMPLETE(model => ?, prompt => ?, response_format => PARSE_JSON(?), "
       "show_details => TRUE) AS R")
SYSTEM = ("You adjudicate official travel under the Defense Transportation Regulation (DTR) "
          "Part I. Decide strictly from the scenario and the DTR excerpt given. If a fact "
          "the excerpt makes decisive is missing, say so through a probability near 0.5.")


def options_of(q: dict) -> list[tuple[str, str]]:
    """(key, description) pairs in the order the decider sees them."""
    if q["type"] == "score":
        return [(str(i), d) for i, d in enumerate(q["criteria"])]
    return list(q["criteria"].items())


def prompt_and_format(item: dict) -> tuple[str, dict]:
    q, s = item["question"], item["state"]
    head = (f"{SYSTEM}\n\nScenario: {s['scenario']}\n\nDTR excerpt ({s['dtr_citation']}):\n"
            f"{s['dtr_excerpt']}\n\nQuestion: {q['instructions']}\n")
    if q["type"] == "noul":
        text = head + ("\nGive probability_true: your probability (0 to 1) that the answer is "
                       f"'{q['criteria']['true']}' rather than '{q['criteria']['false']}'.")
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"probability_true": {"type": "number"}},
                  "required": ["probability_true"]}
    else:
        opts = options_of(q)
        text = head + "\nOptions:\n" + "".join(f"- {k}: {d}\n" for k, d in opts) + (
            "\nGive answer (one option key) and confidence: your probability (0 to 1) that "
            "it is correct.")
        schema = {"type": "object", "additionalProperties": False,
                  "properties": {"answer": {"type": "string", "enum": [k for k, _ in opts]},
                                 "confidence": {"type": "number"}},
                  "required": ["answer", "confidence"]}
    return text, {"type": "json", "schema": schema}


def as_answer(kind: str, out: dict) -> dict:
    """The LLM's structured output in the decider's answer shape."""
    if kind == "noul":
        return {"type": "noul", "noul": min(1.0, max(0.0, float(out["probability_true"])))}
    return {"type": kind,
            "probabilities": {str(out["answer"]): min(1.0, max(0.0, float(out["confidence"])))}}


def main(model: str) -> None:
    from strands_decider import dtr_eval

    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    session.use_warehouse("DECIDER_BENCH_WH")
    items = [json.loads(r["ITEM_JSON"]) for r in
             session.table("DTR_ITEMS").select("ITEM_JSON").sort("ITEM_ID").collect()]
    preds, usage = {}, {"prompt_tokens": 0, "completion_tokens": 0}
    t_all = time.perf_counter()
    for i, it in enumerate(items):
        text, fmt = prompt_and_format(it)
        t0 = time.perf_counter()
        try:
            raw = session.sql(SQL, params=[model, text, json.dumps(fmt)]).collect()[0]["R"]
            r = json.loads(raw) if isinstance(raw, str) else raw
            out = r["structured_output"][0]["raw_message"]
            out = json.loads(out) if isinstance(out, str) else out
            ans = as_answer(it["kind"], out)
            d, top, conf = dtr_eval.decide(ans)
            for k in usage:
                usage[k] += int(r.get("usage", {}).get(k) or 0)
        except Exception as e:  # a failed call is scored as a wrong, non-abstaining answer
            ans, d, top, conf = {"error": f"{type(e).__name__}: {str(e)[:300]}"}, \
                "invalid", "invalid", 0.0
        preds[it["item_id"]] = {"decision": d, "argmax": top, "confidence": conf,
                                "latency_ms": (time.perf_counter() - t0) * 1000, "answer": ans}
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(items)}", flush=True)
    wall = time.perf_counter() - t_all
    session.close()
    metrics = dtr_eval.score(items, preds)
    out_dir = ROOT / "runs" / "llm_dtr"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{model}.json").write_text(json.dumps(
        {"model": model, "wall_s": wall, "usage": usage, "metrics": metrics,
         "predictions": preds}, indent=1))
    print(json.dumps({"all": metrics["all"], "usage": usage, "wall_s": round(wall, 1)}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "claude-sonnet-5")

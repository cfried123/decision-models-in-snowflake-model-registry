"""Score DTR-Bench through a deployed LAYA_DTR service function, from SQL, in one set-based query.

    SNOWFLAKE_CONNECTION_NAME=dtr python python/laya_dtr/bench_service.py LAYA_DTR_FT_SVC runs/laya_data/dtr_bench.jsonl out.json
"""
import json
import os
import re
import sys
import time

import snowflake.connector as sc
from dtr_eval_laya import decide, score

SCHEMA = "DECIDER_BENCH.BENCH"


def main(service: str, bench: str, out: str) -> None:
    assert re.fullmatch(r"[A-Z][A-Z0-9_]{0,254}", service), service
    items = [json.loads(line) for line in open(bench)]
    conn = sc.connect(connection_name=os.environ.get("SNOWFLAKE_CONNECTION_NAME", "dtr"))
    cur = conn.cursor()
    cur.execute(f"USE SCHEMA {SCHEMA}")
    cur.execute("CREATE TEMPORARY TABLE LAYA_BENCH_IN (ITEM_ID STRING, STATE_JSON STRING, QUESTIONS_JSON STRING)")
    cur.executemany("INSERT INTO LAYA_BENCH_IN VALUES (%s, %s, %s)",
                    [(it["item_id"], json.dumps(it["state"]), json.dumps({"d": it["question"]})) for it in items])
    t0 = time.perf_counter()
    cur.execute(f"SELECT ITEM_ID, {service}!SYSTEM_ONE(STATE_JSON, QUESTIONS_JSON) FROM LAYA_BENCH_IN")
    rows = {iid: json.loads(r) if isinstance(r, str) else r for iid, r in cur.fetchall()}
    wall = time.perf_counter() - t0
    preds, model_ms = {}, []
    for it in items:
        r = rows[it["item_id"]]
        ans = json.loads(r["ANSWER_JSON"])["answers"]["d"]
        if it["kind"] == "score":
            ans = {**ans, "probabilities": {str(k): v for k, v in ans["probabilities"].items()}}
        decision, top, conf = decide(ans)
        preds[it["item_id"]] = {"decision": decision, "argmax": top, "confidence": conf, "latency_ms": r["ELAPSED_MS"]}
        model_ms.append(r["ELAPSED_MS"])
    model_ms.sort()
    res = {"metrics": score(items, preds), "service": service, "n": len(items), "wall_s": wall,
           "decisions_per_s": len(items) / wall, "model_ms_p50": model_ms[len(model_ms) // 2],
           "model_ms_mean": sum(model_ms) / len(model_ms)}
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "metrics"}), json.dumps(res["metrics"].get("overall", res["metrics"]))[:600])


if __name__ == "__main__":
    main(*sys.argv[1:4])

"""DTR-Bench over the service's public HTTPS endpoint (python/create_service_http.py).

Two passes over DTR_ITEMS: sequential (one decision per request over a reused connection)
and concurrent (8 client threads). The sequential answers are stored in DECIDER_ANSWERS as
run '<run_id>' and scored with strands_decider.dtr_eval, the same abstention policy as
DTR_DECISIONS. Cost is GPU node time while each pass ran, at the list prices in
PRICE_ASSUMPTIONS. Client times include the network between this machine and the endpoint.
Authenticates with a PAT in DECIDER_BENCH_PAT (or SNOWFLAKE_PAT), sent as
`Authorization: Snowflake Token="<PAT>"`. Writes runs/http_dtr/<run_id>/summary.json.

    SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/bench_http_dtr.py [service] [run_id]
"""
import json
import os
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_config import FQ_SCHEMA
from snowflake.snowpark.functions import parse_json
from snowpark_session import create_snowpark_session

ROOT = Path(__file__).resolve().parents[1]
CONCURRENCY = 8


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def stats(xs):
    return {"n": len(xs), "p50_ms": round(pct(xs, 50), 1), "p95_ms": round(pct(xs, 95), 1),
            "mean_ms": round(statistics.mean(xs), 1)}


def endpoint_url(session, service):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", service):
        raise SystemExit(f"bad service name {service!r}")
    rows = session.sql(f"SHOW ENDPOINTS IN SERVICE {FQ_SCHEMA}.{service}").collect()
    urls = [r["ingress_url"] for r in rows if r["ingress_url"] and ".snowflakecomputing.app" in r["ingress_url"]]
    if not urls:
        raise SystemExit(f"no public endpoint yet: {[r['ingress_url'] for r in rows]}")
    return f"https://{urls[0]}/system-one"


def call(http, url, item):
    body = {"dataframe_split": {"index": [0], "columns": ["STATE_JSON", "QUESTIONS_JSON"],
                                "data": [[item["STATE_JSON"], item["QUESTIONS_JSON"]]]}}
    t0 = time.perf_counter()
    resp = http.post(url, json=body, timeout=120)
    client_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    return client_ms, resp.json()["data"][0][1]


def main(service, run_id):
    out_dir = ROOT / "runs" / "http_dtr" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    items = sorted((r.as_dict() for r in session.sql(
        "SELECT ITEM_ID, SPLIT, STATE_JSON, QUESTIONS_JSON, ITEM_JSON FROM DTR_ITEMS").collect()),
        key=lambda r: r["ITEM_ID"])
    url = endpoint_url(session, service)
    token = os.environ.get("DECIDER_BENCH_PAT") or os.environ["SNOWFLAKE_PAT"]
    http = requests.Session()
    http.headers.update({"Authorization": f'Snowflake Token="{token}"', "Content-Type": "application/json"})
    for item in items[:3]:  # warm-up: TLS setup and the first forward passes
        call(http, url, item)

    rows, answers = [], {}
    t0 = time.perf_counter()
    for item in items:
        client_ms, out = call(http, url, item)
        answers[item["ITEM_ID"]] = out
        rows.append({"pass": "sequential", "item_id": item["ITEM_ID"], "client_ms": client_ms,
                     "gpu_ms": out["ELAPSED_MS"]})
    seq_wall_s = time.perf_counter() - t0

    local = threading.local()

    def one(item):
        if not hasattr(local, "http"):
            local.http = requests.Session()
            local.http.headers.update(http.headers)
        client_ms, out = call(local.http, url, item)
        return {"pass": "concurrent", "item_id": item["ITEM_ID"], "client_ms": client_ms,
                "gpu_ms": out["ELAPSED_MS"],
                "same_answer": json.loads(out["ANSWER_JSON"]) == json.loads(answers[item["ITEM_ID"]]["ANSWER_JSON"])}

    t0 = time.perf_counter()
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        conc = list(pool.map(one, items))
    conc_wall_s = time.perf_counter() - t0
    rows += conc

    session.sql("DELETE FROM DECIDER_ANSWERS WHERE RUN_ID = ?", params=[run_id]).collect()
    session.create_dataframe(
        [[run_id, k, json.dumps(v)] for k, v in answers.items()], schema=["RUN_ID", "ITEM_ID", "R"]
    ).select("RUN_ID", "ITEM_ID", parse_json("R").alias("RESULT")).write.mode("append").save_as_table(
        "DECIDER_ANSWERS", column_order="name")
    price = {r["ITEM"]: float(r["VALUE"]) for r in session.sql("SELECT ITEM, VALUE FROM PRICE_ASSUMPTIONS").collect()}
    session.close()

    from strands_decider import dtr_eval
    preds = {}
    for item_id, out in answers.items():
        a = (json.loads(out["ANSWER_JSON"]).get("answers") or {}).get("decision")
        if a is None:
            preds[item_id] = {"decision": "invalid", "argmax": "invalid", "confidence": 0.0}
            continue
        d, top, conf = dtr_eval.decide(a)
        preds[item_id] = {"decision": d, "argmax": top, "confidence": conf}
    metrics = dtr_eval.score([json.loads(i["ITEM_JSON"]) for i in items], preds)

    usd_per_s = price["GPU_NV_S_CREDITS_PER_HOUR"] * price["PLATFORM_CREDIT_USD"] / 3600
    seq = [r for r in rows if r["pass"] == "sequential"]
    summary = {
        "service": service, "run_id": run_id,
        "endpoint": "public HTTPS ingress, one GPU_NV_S node, num_workers=1, max_batch_rows=32",
        "metrics": {"all": metrics.get("all"), "holdout": metrics.get("holdout")},
        "cost": {"usd_per_gpu_node_hour": round(usd_per_s * 3600, 3),
                 "sequential_usd_per_1000": round(usd_per_s * seq_wall_s / len(items) * 1000, 4),
                 "concurrent_usd_per_1000": round(usd_per_s * conc_wall_s / len(items) * 1000, 4)},
        "sequential": {"wall_s": round(seq_wall_s, 2), "decisions_per_s": round(len(seq) / seq_wall_s, 2),
                       "client": stats([r["client_ms"] for r in seq]), "gpu": stats([r["gpu_ms"] for r in seq])},
        "concurrent": {"clients": CONCURRENCY, "wall_s": round(conc_wall_s, 2),
                       "decisions_per_s": round(len(conc) / conc_wall_s, 2),
                       "client": stats([r["client_ms"] for r in conc]),
                       "same_answer_as_sequential": f"{sum(r['same_answer'] for r in conc)}/{len(conc)}"},
    }
    (out_dir / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "DTR_DECIDER_HTTP",
         sys.argv[2] if len(sys.argv) > 2 else "rest")

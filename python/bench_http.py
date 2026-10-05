"""Real-time inference over the service's public HTTPS endpoint, against the same service from SQL.

Run after python/create_service_http.py. Three passes over JEVBENCH_ITEMS:

  1. sequential: one decision per HTTP request, one at a time, over a reused connection;
  2. concurrent: the same 231 requests from 8 client threads;
  3. SQL: 20 single-row SELECTs calling DECIDER_2B_HTTP!SYSTEM_ONE on an XS warehouse.

Every HTTP answer is checked against the stored answers of the 'service' run. The
sequential pass's answers are stored as run 'rest' and scored with JevBench's harness
(runs/rest/), and priced as GPU node time while the pass ran, at list price. Client
times include the network between this machine and the endpoint. Authenticates with a
programmatic access token (PAT) in DECIDER_BENCH_PAT, sent as
`Authorization: Snowflake Token="<PAT>"`; the PAT's role needs the service role
DECIDER_2B_HTTP!ALL_ENDPOINTS_USAGE (owners have it).
Writes runs/http/{requests.jsonl, summary.json}.

    DECIDER_BENCH_PAT=<pat> SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/bench_http.py
"""
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from snowflake.snowpark.functions import parse_json  # noqa: E402

from bench_config import FQ_SCHEMA  # noqa: E402
from score_with_jevbench import ReplayTypeSafe, run_harness  # noqa: E402  (puts jevbench on sys.path)
from snowpark_session import create_snowpark_session  # noqa: E402
from jevbench.tasks import Task  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "http"
SERVICE = "DECIDER_2B_HTTP"
CONCURRENCY = 8
SQL_CALLS = 20
TIER_ORDER = {"easy": 0, "standard": 1, "hard": 2}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def stats(xs):
    return {"n": len(xs), "p50_ms": round(pct(xs, 50), 1), "p95_ms": round(pct(xs, 95), 1),
            "mean_ms": round(statistics.mean(xs), 1)}


def endpoint_url(session):
    rows = session.sql(f"SHOW ENDPOINTS IN SERVICE {FQ_SCHEMA}.{SERVICE}").collect()
    urls = [r["ingress_url"] for r in rows if r["ingress_url"] and ".snowflakecomputing.app" in r["ingress_url"]]
    if not urls:
        raise SystemExit(f"no public endpoint yet: {[r['ingress_url'] for r in rows]}")
    return f"https://{urls[0]}/system-one"


def call(http, url, item):
    body = {"dataframe_split": {"index": [0], "columns": ["STATE_JSON", "QUESTIONS_JSON"],
                                "data": [[item["STATE_JSON"], item["QUESTIONS_JSON"]]]}}
    t0 = time.perf_counter()
    resp = http.post(url, json=body, timeout=60)
    client_ms = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    out = resp.json()["data"][0][1]
    return client_ms, out


def same_answer(out, stored):
    return json.loads(out["ANSWER_JSON"])["answers"] == json.loads(stored)["answers"]


def main():
    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    items = [r.as_dict() for r in session.sql(
        "SELECT ITEM_ID, TIER, STATE_JSON, QUESTIONS_JSON FROM JEVBENCH_ITEMS").collect()]
    items.sort(key=lambda r: (TIER_ORDER[r["TIER"]], r["ITEM_ID"]))
    stored = {r["ITEM_ID"]: r["A"] for r in session.sql(
        "SELECT ITEM_ID, RESULT:ANSWER_JSON::STRING AS A FROM DECIDER_ANSWERS WHERE RUN_ID = 'service'").collect()}
    url = endpoint_url(session)
    token = os.environ["DECIDER_BENCH_PAT"]
    http = requests.Session()
    http.headers.update({"Authorization": f'Snowflake Token="{token}"', "Content-Type": "application/json"})

    # Warm-up: the first request sets up the TLS connection.
    for item in items[:5]:
        call(http, url, item)

    OUT.mkdir(parents=True, exist_ok=True)
    rows, rest_answers = [], {}
    t0 = time.perf_counter()
    for item in items:
        client_ms, out = call(http, url, item)
        rest_answers[item["ITEM_ID"]] = out
        rows.append({"pass": "sequential", "item_id": item["ITEM_ID"], "tier": item["TIER"],
                     "client_ms": client_ms, "gpu_ms": out["ELAPSED_MS"],
                     "same_answer": same_answer(out, stored[item["ITEM_ID"]])})
    seq_wall_s = time.perf_counter() - t0

    # Store the sequential pass's answers as run 'rest', the same shape as the other runs.
    session.sql("DELETE FROM DECIDER_ANSWERS WHERE RUN_ID = 'rest'").collect()
    session.create_dataframe(
        [["rest", k, json.dumps(v)] for k, v in rest_answers.items()], schema=["RUN_ID", "ITEM_ID", "R"]
    ).select("RUN_ID", "ITEM_ID", parse_json("R").alias("RESULT")).write.mode("append").save_as_table(
        "DECIDER_ANSWERS", column_order="name")

    local = threading.local()

    def one(item):
        if not hasattr(local, "http"):                  # one reused connection per client thread
            local.http = requests.Session()
            local.http.headers.update(http.headers)
        client_ms, out = call(local.http, url, item)
        return {"pass": "concurrent", "item_id": item["ITEM_ID"], "tier": item["TIER"],
                "client_ms": client_ms, "gpu_ms": out["ELAPSED_MS"],
                "same_answer": same_answer(out, stored[item["ITEM_ID"]])}

    t0 = time.perf_counter()
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        conc = list(pool.map(one, items))
    conc_wall_s = time.perf_counter() - t0
    rows += conc

    # SQL: single-row calls to the same service on a warm XS warehouse.
    session.sql("CREATE WAREHOUSE IF NOT EXISTS DECIDER_BENCH_WH WAREHOUSE_SIZE = 'XSMALL' "
                "AUTO_SUSPEND = 60 AUTO_RESUME = TRUE INITIALLY_SUSPENDED = TRUE "
                "COMMENT = 'decider-2b JevBench runs only'").collect()
    session.sql("USE WAREHOUSE DECIDER_BENCH_WH").collect()
    session.sql("ALTER SESSION SET USE_CACHED_RESULT = FALSE").collect()
    session.sql(f"SELECT {SERVICE}!SYSTEM_ONE(STATE_JSON, QUESTIONS_JSON) FROM JEVBENCH_ITEMS "
                "WHERE ITEM_ID = 'easy-intent-00'").collect()     # resumes the warehouse
    sql_ids = items[::12][:SQL_CALLS]                                # spread across the three tiers
    for item in sql_ids:
        t0 = time.perf_counter()
        res = session.sql(f"SELECT {SERVICE}!SYSTEM_ONE(STATE_JSON, QUESTIONS_JSON) AS R FROM JEVBENCH_ITEMS "
                          f"WHERE ITEM_ID = '{item['ITEM_ID']}'").collect()
        client_ms = (time.perf_counter() - t0) * 1000
        qid = session.sql("SELECT LAST_QUERY_ID() AS Q").collect()[0]["Q"]
        out = json.loads(res[0]["R"])
        rows.append({"pass": "sql", "item_id": item["ITEM_ID"], "tier": item["TIER"],
                     "client_ms": client_ms, "gpu_ms": out["ELAPSED_MS"], "query_id": qid,
                     "same_answer": same_answer(out, stored[item["ITEM_ID"]])})
    sql_rows = [r for r in rows if r["pass"] == "sql"]
    elapsed = {r["QUERY_ID"]: r["TOTAL_ELAPSED_TIME"] for r in session.sql(
        "SELECT QUERY_ID, TOTAL_ELAPSED_TIME FROM TABLE(INFORMATION_SCHEMA.QUERY_HISTORY_BY_SESSION(RESULT_LIMIT => 200))"
    ).collect()}
    for r in sql_rows:
        r["statement_ms"] = elapsed.get(r["query_id"])
    price = {r["ITEM"]: float(r["VALUE"]) for r in session.sql("SELECT ITEM, VALUE FROM PRICE_ASSUMPTIONS").collect()}
    task_rows = session.sql("SELECT ITEM_ID, TIER, ITEM_JSON FROM JEVBENCH_ITEMS").collect()
    session.close()

    # Score the 'rest' run with JevBench's harness, as for the other runs.
    tasks = []
    for it in task_rows:
        t = Task.from_dict(json.loads(it["ITEM_JSON"]))
        t.tier = it["TIER"]
        tasks.append(t)
    tasks.sort(key=lambda t: (TIER_ORDER[t.tier], t.id))
    records, harness = run_harness("rest", ReplayTypeSafe(
        {k: json.loads(v["ANSWER_JSON"]) for k, v in rest_answers.items()}), tasks)
    rec = {r["task_id"]: r for r in records}
    by_tier = {t: f'{sum(bool(rec[x.id]["correct"]) for x in tasks if x.tier == t)} of '
                  f'{sum(1 for x in tasks if x.tier == t)}' for t in TIER_ORDER}

    # GPU node time while each pass ran, at list price (no warehouse on this path).
    usd_per_s = price["GPU_NV_S_CREDITS_PER_HOUR"] * price["PLATFORM_CREDIT_USD"] / 3600
    cost = {"usd_per_gpu_node_hour": round(usd_per_s * 3600, 3),
            "sequential_usd_per_1000": round(usd_per_s * seq_wall_s / len(items) * 1000, 4),
            "concurrent_usd_per_1000": round(usd_per_s * conc_wall_s / len(items) * 1000, 4)}

    seq = [r for r in rows if r["pass"] == "sequential"]
    summary = {
        "endpoint": "public HTTPS ingress, one GPU_NV_S node, num_workers=1, max_batch_rows=32",
        "client": "this machine, over the internet; reused connection for the sequential pass",
        "harness": {"n_correct": harness["n_correct"], "n_scorable": harness["n_scorable"],
                    "accuracy": harness["accuracy"], "by_tier": by_tier},
        "cost": cost,
        "sequential": {"wall_s": round(seq_wall_s, 2),
                       "decisions_per_s": round(len(seq) / seq_wall_s, 2),
                       "all": stats([r["client_ms"] for r in seq]),
                       "by_tier": {t: stats([r["client_ms"] for r in seq if r["tier"] == t]) for t in TIER_ORDER},
                       "gpu_by_tier": {t: stats([r["gpu_ms"] for r in seq if r["tier"] == t]) for t in TIER_ORDER},
                       "overhead_ms_p50": round(pct([r["client_ms"] - r["gpu_ms"] for r in seq], 50), 1)},
        "concurrent": {"clients": CONCURRENCY, "wall_s": round(conc_wall_s, 2),
                       "decisions_per_s": round(len(conc) / conc_wall_s, 2),
                       "all": stats([r["client_ms"] for r in conc])},
        "sql_single_row": {"client": stats([r["client_ms"] for r in sql_rows]),
                           "statement": stats([r["statement_ms"] for r in sql_rows if r["statement_ms"] is not None]),
                           "gpu": stats([r["gpu_ms"] for r in sql_rows])},
        "same_answer_as_service_run": {p: f'{sum(r["same_answer"] for r in rows if r["pass"] == p)}/'
                                          f'{sum(1 for r in rows if r["pass"] == p)}'
                                       for p in ("sequential", "concurrent", "sql")},
    }
    (OUT / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

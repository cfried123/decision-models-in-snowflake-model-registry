"""Real-time inference under concurrency: sweep the number of concurrent REST clients.

Run after python/create_service_http.py --instances N. For each client count it runs that
many threads for STEP_S seconds against the service's public endpoint. Each thread has its
own reused connection and sends one decision per request (closed loop: a new request as soon
as the last one returns), cycling through the 231 JevBench items from its own offset.

Per step it records decisions per second, client-side p50/p95/p99 latency, GPU time and
errors, plus cost per 1,000 decisions from the service's nodes at list price. Client times
include the network between this machine and the endpoint. Answers aren't scored.

    DECIDER_BENCH_PAT=<pat> SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python \\
        python/bench_http_sweep.py DECIDER_2B_HTTP4 4 1 4 16 64

Arguments: service name, number of instances (one GPU_NV_S node each), client counts.
Writes runs/http_sweep/{requests.jsonl, summary.json}.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_config import FQ_SCHEMA  # noqa: E402
from bench_http import pct  # noqa: E402
from snowpark_session import create_snowpark_session  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "http_sweep"
STEP_S = 30
PAUSE_S = 10
WARMUP = 20


def endpoint_url(session, service):
    rows = session.sql(f"SHOW ENDPOINTS IN SERVICE {FQ_SCHEMA}.{service}").collect()
    urls = [r["ingress_url"] for r in rows if r["ingress_url"] and ".snowflakecomputing.app" in r["ingress_url"]]
    if not urls:
        raise SystemExit(f"no public endpoint yet: {[r['ingress_url'] for r in rows]}")
    return f"https://{urls[0]}/system-one"


def post(http, url, item):
    body = {"dataframe_split": {"index": [0], "columns": ["STATE_JSON", "QUESTIONS_JSON"],
                                "data": [[item["STATE_JSON"], item["QUESTIONS_JSON"]]]}}
    t0 = time.perf_counter()
    try:
        resp = http.post(url, json=body, timeout=60)
        ms = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            return ms, None, resp.status_code
        return ms, resp.json()["data"][0][1]["ELAPSED_MS"], 200
    except Exception as e:  # counted as an error, never retried
        return (time.perf_counter() - t0) * 1000, None, type(e).__name__


def run_step(url, headers, items, clients):
    rows, lock = [], threading.Lock()
    stop = time.perf_counter() + STEP_S

    def worker(k):
        http = requests.Session()
        http.headers.update(headers)
        i = k * len(items) // clients
        while time.perf_counter() < stop:
            ms, gpu_ms, status = post(http, url, items[i % len(items)])
            with lock:
                rows.append({"clients": clients, "client_ms": ms, "gpu_ms": gpu_ms, "status": status,
                             "done_at": time.perf_counter()})
            i += 1

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(k,)) for k in range(clients)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    ok = [r for r in rows if r["status"] == 200]
    lat = [r["client_ms"] for r in ok]
    return rows, {
        "clients": clients, "wall_s": round(wall, 2), "requests": len(rows), "ok": len(ok),
        "errors": len(rows) - len(ok),
        "error_kinds": sorted({str(r["status"]) for r in rows if r["status"] != 200}),
        "decisions_per_s": round(len(ok) / wall, 2),
        "p50_ms": round(pct(lat, 50), 1) if lat else None,
        "p95_ms": round(pct(lat, 95), 1) if lat else None,
        "p99_ms": round(pct(lat, 99), 1) if lat else None,
        "gpu_p50_ms": round(pct([r["gpu_ms"] for r in ok], 50), 1) if ok else None,
    }


def main(service, instances, client_counts):
    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    items = [r.as_dict() for r in session.sql(
        "SELECT ITEM_ID, STATE_JSON, QUESTIONS_JSON FROM JEVBENCH_ITEMS ORDER BY ITEM_ID").collect()]
    url = endpoint_url(session, service)
    price = {r["ITEM"]: float(r["VALUE"]) for r in session.sql("SELECT ITEM, VALUE FROM PRICE_ASSUMPTIONS").collect()}
    session.close()
    usd_per_s = instances * price["GPU_NV_S_CREDITS_PER_HOUR"] * price["PLATFORM_CREDIT_USD"] / 3600
    headers = {"Authorization": f'Snowflake Token="{os.environ["DECIDER_BENCH_PAT"]}"',
               "Content-Type": "application/json"}

    http = requests.Session()
    http.headers.update(headers)
    for item in items[:WARMUP]:
        post(http, url, item)

    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, steps = [], []
    for c in client_counts:
        rows, s = run_step(url, headers, items, c)
        s["usd_per_1000"] = round(usd_per_s / s["decisions_per_s"] * 1000, 4) if s["decisions_per_s"] else None
        print(json.dumps(s), flush=True)
        all_rows += rows
        steps.append(s)
        time.sleep(PAUSE_S)
    summary = {"service": service, "instances": instances,
               "hardware": f"{instances}x GPU_NV_S (1x NVIDIA A10G each), num_workers=1, max_batch_rows=32",
               "client": "this machine, over the internet; one decision per request, closed loop",
               "step_s": STEP_S, "usd_per_hour": round(usd_per_s * 3600, 2), "steps": steps}
    (OUT / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_rows))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), [int(c) for c in sys.argv[3:]])

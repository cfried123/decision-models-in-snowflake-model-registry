"""Batch inference scaling: model copies per GPU (num_workers) and nodes (replicas).

Runs one batch job per config over JEVBENCH_ITEMS repeated --copies times (231 x 20 =
4,620 rows; 231 x 217 = 50,127), one job at a time, on --pool. For each job it records:

  - total job time (submit to done), which is what the job's nodes bill for;
  - scoring time: from the last Ray progress line with no rows done to "execution
    finished" in the head node's log (10 s log granularity);
  - the most model copies running and GPUs in use at once, from Ray's status lines,
    and when each worker node joined the head node;
  - summed per-row GPU time;
  - cost at list price: nodes x seconds x the pool's instance-family rate in
    PRICE_ASSUMPTIONS (<FAMILY>_CREDITS_PER_HOUR).

Configs are name:replicas:num_workers[:gpu_requests]. Multi-node configs need the pool's
MAX_NODES >= replicas; start the nodes before the job (MIN_NODES = replicas) so the
workers are up when scoring starts.

    SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/bench_scaling.py V11B \\
        n1w1:1:1 n1w3:1:3 n2w1:2:1 n2w3:2:3
    SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/bench_scaling.py V11B \\
        --pool DECIDER_BENCH_GPU_POOL_S4 --copies 217 out4:4:1

Writes runs/scaling/<name>/{job.json, job.log, instance_<i>.log} and
runs/scaling/summary_x<copies>.json.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from snowflake.ml.model.batch_inference import (  # noqa: E402
    ImageBuildSpec,
    InferenceSpec,
    OutputSpec,
    ResourcesSpec,
    SaveMode,
)
from snowflake.ml.registry import Registry  # noqa: E402

from bench_config import DATABASE, FQ_SCHEMA, SCHEMA  # noqa: E402
from snowpark_session import create_snowpark_session  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "scaling"
STAGE = f"@{FQ_SCHEMA}.DECIDER_BENCH_STAGE/scaling/"

TS = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})"
PROGRESS = re.compile(TS + r".*MapBatches\(LocalModelActor\): (\d+)/")
FINISHED = re.compile(TS + r".*execution finished in")
STATUS = re.compile(r"Actors: (\d+) \(running=(\d+).*?pending=(\d+)\).*?Resources: ([\d.]+) CPU, ([\d.]+) GPU")
JOINED = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) .*Connected to the head node")


def ts(m):
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp() + int(m.group(2)) / 1000


def parse_head_log(log):
    """Scoring window, and the most copies running and GPUs in use at once."""
    start = end = None
    max_running = max_gpus = 0
    for line in log.splitlines():
        if (m := PROGRESS.search(line)):
            if int(m.group(3)) == 0:
                start = ts(m)          # keeps the last "0 rows done" line before scoring
        elif (m := FINISHED.search(line)):
            end = ts(m)
        if (m := STATUS.search(line)):
            max_running = max(max_running, int(m.group(2)))
            max_gpus = max(max_gpus, float(m.group(5)))
    # Single-node jobs stop printing status lines once scoring starts: None means not logged.
    return start, end, max_running or None, max_gpus or None


def run_one(session, mv, pool, rate_usd_per_node_s, input_table, name, replicas, workers, gpus,
            timeout_s):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    job_name = f"{FQ_SCHEMA}.DECIDER_SCALE_{name.upper()}"
    t0 = time.time()
    job = mv.run_batch(
        session.table(input_table).select("ITEM_ID", "STATE_JSON", "QUESTIONS_JSON"),
        compute_pool=pool,
        output_spec=OutputSpec(stage_location=STAGE, mode=SaveMode.OVERWRITE),
        resources_spec=ResourcesSpec(gpu_requests=str(gpus)),
        inference_spec=InferenceSpec(num_workers=workers, max_batch_rows=32),
        image_build_spec=ImageBuildSpec(image_repo=f"{FQ_SCHEMA}.DECIDER_BENCH_IMAGES"),
        function_name="system_one",
        job_name=job_name,
        replicas=replicas,
    )
    t_submitted = time.time()
    print(f"[{name}] submitted in {t_submitted - t0:.0f}s; waiting up to {timeout_s}s", flush=True)
    status = job.wait(timeout=timeout_s)
    if status not in ("DONE", "FAILED"):
        print(f"[{name}] {status} at timeout; cancelling", flush=True)
        job.cancel()
        status = f"CANCELLED_AT_TIMEOUT ({status})"
    t_done = time.time()
    print(f"[{name}] {status} after {t_done - t0:.0f}s", flush=True)

    logs, joined = {}, {}
    for i in range(replicas):
        try:
            logs[i] = job.get_logs(instance_id=i) or ""
        except Exception as e:  # logs are diagnostic only
            logs[i] = f"get_logs failed: {e}"
        (out / ("job.log" if i == 0 else f"instance_{i}.log")).write_text(logs[i])
        if i and (m := JOINED.search(logs[i])):
            joined[i] = m.group(1)
    start, end, max_running, max_gpus = parse_head_log(logs[0])

    df = session.read.option("pattern", r".*\.parquet").parquet(f"{STAGE}{job_name.split('.')[-1]}/")
    cols = {c.strip('"').upper(): c for c in df.columns}
    rows = df.select(cols["ELAPSED_MS"]).collect()
    elapsed = sum(float(r[0]) for r in rows)
    n = len(rows)
    scoring_s = (end - start) if start and end else None
    job_s = t_done - t0
    meta = {"name": name, "pool": pool, "replicas": replicas, "num_workers": workers,
            "gpu_requests": gpus, "status": str(status), "rows": n,
            "job_s": round(job_s, 1), "submit_s": round(t_submitted - t0, 1),
            "scoring_s": round(scoring_s, 1) if scoring_s else None,
            "decisions_per_s": round(n / scoring_s, 1) if scoring_s else None,
            "sum_gpu_ms": round(elapsed, 1),
            "max_copies_running": max_running, "max_gpus_in_use": max_gpus,
            "worker_joined_utc": joined,
            "usd_job": round(replicas * job_s * rate_usd_per_node_s, 3),
            "usd_per_1000_all_in": round(replicas * job_s * rate_usd_per_node_s / n * 1000, 4) if n else None,
            "usd_per_1000_scoring": round(replicas * scoring_s * rate_usd_per_node_s / n * 1000, 4)
            if n and scoring_s else None}
    (out / "job.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("version")
    p.add_argument("configs", nargs="+", help="name:replicas:num_workers[:gpu_requests]")
    p.add_argument("--pool", default="DECIDER_BENCH_GPU_POOL")
    p.add_argument("--copies", type=int, default=20)
    p.add_argument("--timeout", type=int, default=4500, help="seconds before the job is cancelled")
    a = p.parse_args()

    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    session.sql("CREATE WAREHOUSE IF NOT EXISTS DECIDER_BENCH_WH WAREHOUSE_SIZE = 'XSMALL' "
                "AUTO_SUSPEND = 60 AUTO_RESUME = TRUE INITIALLY_SUSPENDED = TRUE "
                "COMMENT = 'decider-2b JevBench runs only'").collect()
    session.use_warehouse("DECIDER_BENCH_WH")
    family = session.sql(f"SHOW COMPUTE POOLS LIKE '{a.pool}'").collect()[0]["instance_family"]
    price = {r["ITEM"]: float(r["VALUE"]) for r in session.sql("SELECT ITEM, VALUE FROM PRICE_ASSUMPTIONS").collect()}
    rate = price[f"{family}_CREDITS_PER_HOUR"] * price["PLATFORM_CREDIT_USD"] / 3600

    # Each item a.copies times, with ITEM_ID '<id>#<copy>'.
    input_table = f"JEVBENCH_ITEMS_X{a.copies}"
    session.sql(f"""CREATE OR REPLACE TABLE {input_table} AS
        SELECT i.ITEM_ID || '#' || c.N AS ITEM_ID, i.STATE_JSON, i.QUESTIONS_JSON
        FROM JEVBENCH_ITEMS i
        CROSS JOIN (SELECT SEQ4() AS N FROM TABLE(GENERATOR(ROWCOUNT => {a.copies}))) c""").collect()
    mv = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA) \
        .get_model("DECIDER_2B").version(a.version)

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for cfg in a.configs:
        name, replicas, workers, *gpus = cfg.split(":")
        results.append(run_one(session, mv, a.pool, rate, input_table, name, int(replicas), int(workers),
                               int(gpus[0]) if gpus else 1, a.timeout))
        (OUT / f"summary_x{a.copies}.json").write_text(json.dumps(results, indent=1))
    session.close()


if __name__ == "__main__":
    main()

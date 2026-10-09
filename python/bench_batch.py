"""Run one model version over the DTR-Bench items (DTR_ITEMS) as a Model Registry batch
inference job (mv.run_batch) on the GPU pool, then score the answers with
strands_decider.dtr_eval (the same scorer as the local and frontier-LLM runs).

No warehouse sits in the inference path: the job reads the input from Parquet on
a stage and writes Parquet back. The warehouse only stages the input and reads
the output, in separate statements.

    SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/bench_batch.py BATCH dtr_batch [max_batch_rows]

Writes runs/batch_jobs/<run_id>/{job.json, job.log, answers.json, metrics.json}.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "strands-decider" / "src"))
RUNS = Path(__file__).resolve().parents[1] / "runs"

from snowflake.ml.model.batch_inference import (  # noqa: E402
    ImageBuildSpec,
    InferenceSpec,
    OutputSpec,
    ResourcesSpec,
    SaveMode,
)
from snowflake.ml.registry import Registry  # noqa: E402

from bench_config import DATABASE, FQ_SCHEMA, MODEL_NAME, SCHEMA  # noqa: E402
from snowpark_session import create_snowpark_session  # noqa: E402

GPU_POOL = "DECIDER_BENCH_GPU_POOL"
STAGE = f"@{FQ_SCHEMA}.DECIDER_BENCH_STAGE/batch/"


def main(version, run_id, max_batch_rows):
    out = RUNS / "batch_jobs" / run_id
    out.mkdir(parents=True, exist_ok=True)
    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    session.use_warehouse("DECIDER_BENCH_WH")
    mv = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA) \
        .get_model(MODEL_NAME).version(version)

    X = session.table("DTR_ITEMS").select("ITEM_ID", "STATE_JSON", "QUESTIONS_JSON")
    job_name = f"{FQ_SCHEMA}.DTR_{run_id.upper()}"
    t0 = time.time()
    job = mv.run_batch(
        X,
        compute_pool=GPU_POOL,
        output_spec=OutputSpec(stage_location=STAGE, mode=SaveMode.OVERWRITE),
        resources_spec=ResourcesSpec(gpu_requests="1"),
        inference_spec=InferenceSpec(num_workers=1, max_batch_rows=max_batch_rows),
        image_build_spec=ImageBuildSpec(image_repo=f"{FQ_SCHEMA}.DECIDER_BENCH_IMAGES"),
        function_name="system_one",
        job_name=job_name,
    )
    t_submitted = time.time()
    print(f"[{run_id}] submitted {job.id} in {t_submitted - t0:.0f}s; waiting", flush=True)
    status = job.wait()
    t_done = time.time()
    print(f"[{run_id}] {status} after {t_done - t0:.0f}s", flush=True)
    try:
        (out / "job.log").write_text(job.get_logs() or "")
    except Exception as e:  # logs are diagnostic only
        (out / "job.log").write_text(f"get_logs failed: {e}")

    location = f"{STAGE}{job_name.split('.')[-1]}/"
    rows = session.read.option("pattern", r".*\.parquet").parquet(location).collect()
    cols = list(rows[0].as_dict().keys()) if rows else []
    print(f"[{run_id}] {len(rows)} output rows; columns {cols}", flush=True)
    answers, elapsed, tokens, runtime = {}, [], [], None
    for r in rows:
        d = {k.strip('"').upper(): v for k, v in r.as_dict().items()}
        answers[d["ITEM_ID"]] = json.loads(d["ANSWER_JSON"])
        elapsed.append(float(d["ELAPSED_MS"]))
        tokens.append(int(d["INPUT_TOKENS"]))
        runtime = runtime or d["RUNTIME_JSON"]
    (out / "answers.json").write_text(json.dumps(answers, indent=1, sort_keys=True))
    meta = {"version": version, "job": job.id, "status": str(status), "max_batch_rows": max_batch_rows,
            "submit_s": round(t_submitted - t0, 1), "wall_s": round(t_done - t0, 1),
            "t0_epoch": t0, "t_done_epoch": t_done, "rows": len(rows), "columns": cols,
            "sum_elapsed_ms": round(sum(elapsed), 1), "sum_input_tokens": sum(tokens),
            "runtime": json.loads(runtime) if runtime else None}
    (out / "job.json").write_text(json.dumps(meta, indent=1))

    items = [json.loads(r["ITEM_JSON"]) for r in
             session.table("DTR_ITEMS").select("ITEM_JSON").collect()]
    session.close()
    from strands_decider import dtr_eval

    preds = {}
    for item_id, ans in answers.items():
        a = (ans.get("answers") or {}).get("decision")
        if a is None:  # error row: counted as an abstention-free miss
            preds[item_id] = {"decision": "invalid", "argmax": "invalid", "confidence": 0.0}
            continue
        d, top, conf = dtr_eval.decide(a)
        preds[item_id] = {"decision": d, "argmax": top, "confidence": conf}
    metrics = dtr_eval.score(items, preds)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({"all": metrics.get("all"), "holdout": metrics.get("holdout"),
                      "wall_s": meta["wall_s"], "sum_elapsed_ms": meta["sum_elapsed_ms"]}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 32)

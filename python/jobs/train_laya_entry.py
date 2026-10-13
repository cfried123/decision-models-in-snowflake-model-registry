"""ML Job entrypoint: fine-tune Laya on DTR Part I, score base and fine-tuned on DTR-Bench.

Runs from the payload staged by python/laya_dtr/train_laya_job.py: the laya package (from the
cfried123/laya fork), the base checkpoint (no internet egress in the job), the converted DTR
rows and DTR-Bench. Writes the fine-tuned checkpoint (safetensors, no pickles) and results to
@DECIDER_BENCH_STAGE/train/out/<run_id>/.
"""
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from snowflake.snowpark.context import get_active_session

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STAGE = "@DECIDER_BENCH.BENCH.DECIDER_BENCH_STAGE"
WORK = Path("/tmp/laya-dtr")


def put_tree(session, local: Path, dest: str) -> None:
    for root, _, files in os.walk(local):
        rel = os.path.relpath(root, local)
        target = dest if rel == "." else f"{dest}{rel}/"
        for f in files:
            session.file.put(os.path.join(root, f), target, auto_compress=False, overwrite=True)


def main() -> None:
    run_id = sys.argv[1]
    hp = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    session = get_active_session()
    os.environ.setdefault("USE_TF", "0")
    import torch
    from laya import load
    from laya.train import TrainConfig, finetune
    from dtr_eval_laya import evaluate

    print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}", flush=True)
    cfg = TrainConfig(
        epochs=hp.get("epochs", 2), micro_batch=hp.get("micro_batch", 8), grad_accum=hp.get("grad_accum", 4),
        shuffle_options=("choice", "score"), max_len=hp.get("max_len", 768), eval_data=str(HERE / "data/holdout.jsonl"),
        log_every=hp.get("log_every", 20), seed=hp.get("seed", 0),
    )
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary = finetune(str(HERE / "data/train.jsonl"), str(HERE / "base"), str(WORK / "ft"), cfg, device="cuda")
    train_s = time.time() - t0
    print(f"trained in {train_s:.0f}s", flush=True)

    items = [json.loads(line) for line in (HERE / "data/dtr_bench.jsonl").read_text().splitlines()]
    results = WORK / "results"
    results.mkdir(exist_ok=True)
    report = {"run_id": run_id, "train_seconds": train_s, "train_config": asdict(cfg),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for name, path in (("base", HERE / "base"), ("ft", WORK / "ft")):
        agent = load(str(path), device="cuda")
        evaluate(agent, items[:4])  # warm up kernels before timing
        res = evaluate(agent, items)
        (results / f"dtr_bench_{name}.json").write_text(json.dumps(res, indent=2))
        report[name] = res["metrics"]
        print(name, json.dumps({g: res["metrics"][g] for g in ("all", "holdout") if g in res["metrics"]}), flush=True)
        del agent
    report["finetune_summary"] = json.loads(json.dumps(summary, default=str))
    (results / "summary.json").write_text(json.dumps(report, indent=2, default=str))

    pickles = [p for p in (WORK / "ft").rglob("*") if p.suffix in (".pt", ".bin", ".pkl", ".pth")]
    assert not pickles, f"pickle files in checkpoint: {pickles}"
    dest = f"{STAGE}/train/out/{run_id}/"
    put_tree(session, results, dest + "results/")
    put_tree(session, WORK / "ft", dest + "ckpt/")
    print(f"done in {time.time() - t0:.0f}s; outputs at {dest}", flush=True)


if __name__ == "__main__":
    main()

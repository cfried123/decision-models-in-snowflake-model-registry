"""Entrypoint of the DTR fine-tune ML Job (runs inside the GPU container).

Pulls the base torso from the benchmark stage (the job has no internet egress), points the
v21 checkpoint and the training config at it, trains, scores DTR-Bench, exports the
pickle-free HF layout and puts everything back on the stage under train/out/<run_id>/.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml
from snowflake.snowpark.context import get_active_session

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
STAGE = "@DECIDER_BENCH.BENCH.DECIDER_BENCH_STAGE"
RUN_ID = sys.argv[1] if len(sys.argv) > 1 else time.strftime("dtr-%Y%m%d-%H%M%S")
WORK = Path("/tmp/dtr")
os.environ["HF_HUB_OFFLINE"] = "1"


def put_tree(session, local: Path, dest: str) -> None:
    for root, _, files in os.walk(local):
        rel = os.path.relpath(root, local)
        target = dest + ("" if rel == "." else rel + "/")
        for f in files:
            session.file.put(os.path.join(root, f), target, auto_compress=False, overwrite=True)


def main() -> None:
    session = get_active_session()
    t0 = time.time()
    base = WORK / "base"
    base.mkdir(parents=True, exist_ok=True)
    session.file.get(f"{STAGE}/train/qwen3.5-2b-base/", str(base))
    print(f"base torso staged in {time.time() - t0:.0f}s: {sorted(p.name for p in base.iterdir())}", flush=True)

    v21 = WORK / "v21"
    shutil.copytree(HERE / "v21", v21, dirs_exist_ok=True)
    cfg_path = v21 / "strands_decider_config.json"
    sd = json.loads(cfg_path.read_text())
    hub_base = {k: sd.get(k) for k in ("base_model", "base_revision")}
    sd.update(base_model=str(base), base_revision=None)
    cfg_path.write_text(json.dumps(sd, indent=2))

    cfg = yaml.safe_load((HERE / "finetune_dtr.yaml").read_text())
    cfg.update(finetune_from=str(v21), base_model=str(base),
               train_files=[str(HERE / "data/dtr/train.jsonl")],
               val_files=[str(HERE / "data/dtr/train.holdout.jsonl")],
               output_dir=str(WORK / "ckpt"))
    cfg.pop("base_revision", None)
    run_cfg = WORK / "train.yaml"
    run_cfg.write_text(yaml.safe_dump(cfg))
    print("train config:", json.dumps(cfg), flush=True)

    import torch
    print(f"torch {torch.__version__} cuda {torch.cuda.is_available()} "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}", flush=True)
    from strands_decider.train import TrainConfig, train
    t1 = time.time()
    ckpt = train(TrainConfig.from_yaml(str(run_cfg)))
    print(f"trained in {time.time() - t1:.0f}s -> {ckpt}", flush=True)

    from strands_decider import dtr_eval, hf_export
    results = WORK / "results"
    results.mkdir(exist_ok=True)
    for name, path in (("ft", WORK / "ckpt"), ("base_v21", v21)):
        dtr_eval.main([str(path), "--bench", str(HERE / "data/dtr_bench.jsonl"),
                       "--out", str(results / f"bench_{name}.json"), "--device", "cuda"])
    # The checkpoint names the staged torso; give it back its hub identity for the exporter
    # and the registry, and keep the raw checkpoint on the stage in case export fails.
    ckpt_cfg = WORK / "ckpt" / "strands_decider_config.json"
    ckpt_cfg.write_text(json.dumps({**json.loads(ckpt_cfg.read_text()), **hub_base}, indent=2))
    dest = f"{STAGE}/train/out/{RUN_ID}/"
    put_tree(session, results, dest + "results/")
    put_tree(session, WORK / "ckpt", dest + "ckpt/")
    session.file.put(str(run_cfg), dest, auto_compress=False, overwrite=True)

    export = WORK / "export"
    hf_export.main(["export", str(WORK / "ckpt"), str(export), "--name", "strands-decider-dtr",
                    "--run-id", RUN_ID, "--reports", str(results)])
    hf_export.main(["verify", str(export)])
    assert not list(export.rglob("*.pt")), "export must be pickle-free"

    put_tree(session, export, dest + "export/")
    for f in sorted(results.glob("*.json")):
        print(f.name, json.dumps(json.loads(f.read_text()).get("all", {})), flush=True)
    print(f"done in {time.time() - t0:.0f}s; outputs at {dest}", flush=True)


if __name__ == "__main__":
    main()

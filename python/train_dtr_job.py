"""Fine-tune strands-decider on DTR Part I as a Snowflake ML Job on a GPU_NV_S pool.

Stages a payload (the strands_decider package, the DTR data and DTR-Bench, the training
config and the v21 starting checkpoint) and runs python/jobs/train_dtr_entry.py. The base
torso must already be on @DECIDER_BENCH_STAGE/train/qwen3.5-2b-base/. Packages come from
Snowflake's managed PyPI repository, so no external access integration is needed.

    SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/train_dtr_job.py [run_id]
"""
import os
import re
import shutil
import sys
import time
from pathlib import Path

from snowflake.ml import jobs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_config import DATABASE, SCHEMA
from snowpark_session import create_snowpark_session

ROOT = Path(__file__).resolve().parents[1]
SD = Path(os.environ.get("STRANDS_DECIDER_REPO", ROOT.parent / "strands-decider"))
V21 = Path(os.environ.get("DTR_START_CKPT", ROOT / "models" / "v21"))
POOL = os.environ.get("DTR_TRAIN_POOL", "SYSTEM_COMPUTE_POOL_GPU")
PIP = [
    "transformers==5.19.0", "peft==0.21.2", "accelerate==1.15.0", "pydantic==2.14.0",
    "flash-linear-attention==0.5.2", "fla-core==0.5.2", "safetensors==0.8.0",
    "tokenizers==0.23.2", "huggingface-hub==1.33.0", "einops==0.8.2", "pyyaml",
]


def payload() -> Path:
    out = ROOT / "runs" / "train_payload"
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(SD / "src" / "strands_decider", out / "strands_decider",
                    ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("data/dtr/train.jsonl", "data/dtr/train.holdout.jsonl", "data/dtr_bench.jsonl"):
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SD / f, out / f)
    shutil.copy(SD / "training" / "dtr" / "finetune_dtr.yaml", out)
    shutil.copytree(V21, out / "v21", ignore=shutil.ignore_patterns(".cache", "eval", "training"))
    shutil.copy(ROOT / "python" / "jobs" / "train_dtr_entry.py", out)
    # hf_export reads the Apache LICENSE from the installed distribution; the payload is on
    # sys.path rather than pip-installed, so give it a minimal dist-info to find.
    version = re.search(r'^version = "([^"]+)"', (SD / "pyproject.toml").read_text(), re.MULTILINE)[1]
    info = out / f"strands_decider-{version}.dist-info"
    (info / "licenses").mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: strands-decider\nVersion: {version}\nLicense: Apache-2.0\n")
    shutil.copy(SD / "LICENSE", info / "licenses" / "LICENSE")
    return out


def main(run_id: str) -> None:
    session = create_snowpark_session()
    session.use_schema(f"{DATABASE}.{SCHEMA}")
    job = jobs.submit_directory(
        str(payload()), POOL, entrypoint="train_dtr_entry.py", args=[run_id],
        stage_name="DTR_TRAIN_JOBS", pip_requirements=PIP,
        artifact_repositories=["snowflake.snowpark.pypi_shared_repository"],
        session=session, database=DATABASE, schema=SCHEMA, name="DTR_FINETUNE",
        env_vars={"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"},
    )
    print(f"submitted {job.id} on {POOL}", flush=True)
    t0 = time.time()
    status = job.wait()
    print(f"{status} after {time.time() - t0:.0f}s", flush=True)
    log = job.get_logs()
    (ROOT / "runs" / f"train_{run_id}.log").write_text(log or "")
    print((log or "")[-4000:])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else time.strftime("dtr-%Y%m%d-%H%M%S"))

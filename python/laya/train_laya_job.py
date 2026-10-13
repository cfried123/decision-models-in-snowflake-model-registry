"""Fine-tune Laya (cfried123/laya) on DTR Part I as a Snowflake ML Job on GPU_NV_S.

Stages the laya package, the base checkpoint, the converted DTR rows and DTR-Bench, then runs
python/jobs/train_laya_entry.py. Packages come from Snowflake's managed PyPI repository, so no
external access integration is needed (trial accounts cannot create one).

    SNOWFLAKE_CONNECTION_NAME=<connection> python python/laya/train_laya_job.py [run_id] ['{"epochs": 2}']
"""
import os
import shutil
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from snowflake.ml import jobs

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_config import DATABASE, SCHEMA  # noqa: E402
from dtr_data import SD, write  # noqa: E402
from snowpark_session import create_snowpark_session  # noqa: E402

LAYA = Path(os.environ.get("LAYA_REPO", ROOT.parent / "laya"))
BASE = os.environ.get("LAYA_BASE", "convaiinnovations/laya")
POOL = os.environ.get("LAYA_TRAIN_POOL", "SYSTEM_COMPUTE_POOL_GPU")
PIP = ["transformers==5.19.0", "safetensors==0.8.0", "tokenizers==0.23.2", "huggingface-hub==1.33.0"]


def payload() -> Path:
    out = ROOT / "runs" / "laya_payload"
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(LAYA / "laya", out / "laya", ignore=shutil.ignore_patterns("__pycache__"))
    write(out / "data")
    base = snapshot_download(BASE, allow_patterns=["rl_agent_config.json", "model.safetensors",
                                                   "tokenizer/*", "encoder/*"])
    shutil.copytree(base, out / "base")
    shutil.copy(SD / "src" / "strands_decider" / "dtr_eval.py", out / "dtr_eval.py")
    shutil.copy(Path(__file__).resolve().parent / "dtr_eval_laya.py", out)
    shutil.copy(ROOT / "python" / "jobs" / "train_laya_entry.py", out)
    return out


def main(run_id: str, hp: str) -> None:
    session = create_snowpark_session()
    session.use_schema(f"{DATABASE}.{SCHEMA}")
    job = jobs.submit_directory(
        str(payload()), POOL, entrypoint="train_laya_entry.py", args=[run_id, hp],
        stage_name="DTR_TRAIN_JOBS", pip_requirements=PIP,
        artifact_repositories=["snowflake.snowpark.pypi_shared_repository"],
        session=session, database=DATABASE, schema=SCHEMA, name="LAYA_DTR_FINETUNE",
        env_vars={"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false", "USE_TF": "0",
                  "HF_HUB_OFFLINE": "1"},
    )
    print(f"submitted {job.id} on {POOL}", flush=True)
    t0 = time.time()
    status = job.wait()
    print(f"{status} after {time.time() - t0:.0f}s", flush=True)
    log = job.get_logs()
    (ROOT / "runs" / f"train_{run_id}.log").write_text(log or "")
    print((log or "")[-5000:])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else time.strftime("laya-dtr-%Y%m%d-%H%M%S"),
         sys.argv[2] if len(sys.argv) > 2 else "{}")

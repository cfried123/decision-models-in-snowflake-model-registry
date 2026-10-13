"""Log a Laya checkpoint to the Model Registry as <database>.<schema>.LAYA_DTR.

Versions: BASE (convaiinnovations/laya, zero-shot) and FT (fine-tuned on DTR Part I by
python/laya/train_laya_job.py). The checkpoint is the only artifact and the laya package ships
as a code path, so the service never downloads weights or code.

    SNOWFLAKE_CONNECTION_NAME=<connection> python python/laya/log_laya_model.py BASE <checkpoint dir>
    SNOWFLAKE_CONNECTION_NAME=<connection> python python/laya/log_laya_model.py FT <checkpoint dir>
"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from snowflake.ml.model import custom_model  # noqa: E402
from snowflake.ml.registry import Registry  # noqa: E402

from bench_config import DATABASE, FQ_SCHEMA, SCHEMA  # noqa: E402
from laya_model import CHECKPOINT, SIGNATURE, LayaModel  # noqa: E402
from snowpark_session import create_snowpark_session  # noqa: E402

MODEL_NAME = os.environ.get("LAYA_MODEL_NAME", "LAYA_DTR")
LAYA_PKG = Path(os.environ.get("LAYA_REPO", ROOT.parent / "laya")) / "laya"
PIP_REPOSITORY = os.environ.get("DTR_PIP_REPOSITORY", "snowflake.snowpark.pypi_shared_repository")
PIP_REQUIREMENTS = [
    "torch>=2.10.0,<2.11", "transformers==5.19.0", "safetensors==0.8.0", "tokenizers==0.23.2",
    "huggingface-hub==1.33.0", "numpy==1.26.4",
]


def check(ckpt: Path) -> None:
    for required in ("rl_agent_config.json", "model.safetensors", "encoder/config.json", "tokenizer/tokenizer.json"):
        assert (ckpt / required).exists(), f"{ckpt}: missing {required}"
    pickles = [p for p in ckpt.rglob("*") if p.suffix in (".pt", ".bin", ".pkl", ".pth")]
    assert not pickles, f"checkpoint must be pickle-free: {pickles}"


def main(version: str, ckpt: Path) -> None:
    check(ckpt)
    cfg = json.loads((ckpt / "rl_agent_config.json").read_text())
    what = ("fine-tuned on DTR Part I (Passenger Movement) with laya.train (RLCD)" if cfg.get("fine_tuned")
            else "convaiinnovations/laya base checkpoint, zero-shot")
    model = LayaModel(custom_model.ModelContext(artifacts={CHECKPOINT: str(ckpt)}))
    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
    t0 = time.time()
    mv = reg.log_model(
        model, model_name=MODEL_NAME, version_name=version,
        signatures={"system_one": SIGNATURE},
        pip_requirements=PIP_REQUIREMENTS, artifact_repository_map={"pip": PIP_REPOSITORY},
        target_platforms=["SNOWPARK_CONTAINER_SERVICES"], python_version="3.12",
        code_paths=[str(Path(__file__).resolve().parent / "laya_model.py"), str(LAYA_PKG)],
        options={"cuda_version": "12.8", "relax_version": False},
        comment=f"Laya 421M (ModernBERT-large), {what}. system_one takes STATE_JSON + QUESTIONS_JSON.",
    )
    print(f"logged {mv.model_name} {mv.version_name} in {time.time() - t0:.0f}s", flush=True)
    print(mv.show_functions())
    session.close()


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))

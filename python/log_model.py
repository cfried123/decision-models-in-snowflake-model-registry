"""Log the DTR strands-decider fine-tune to the Snowflake Model Registry as
<database>.<schema>.DTR_DECIDER, version SERVICE (every package pinned; the image for
create_service).

Two artifacts are stored with the model so the service never downloads:
  weights  the pickle-free strands-decider export (hf_export.py output)
  base     the Qwen/Qwen3.5-2B-Base snapshot the adapter was trained on
The strands_decider package itself ships as a code path.

Constructing DeciderModel loads the weights once on this machine (CPU). The registry
doesn't upload that instance: it pickles the class and its context, and the container
constructs the model again.

    SNOWFLAKE_CONNECTION_NAME=<connection> \
    DTR_WEIGHTS_DIR=<hf export dir> DTR_BASE_DIR=<Qwen3.5-2B-Base snapshot> \
    STRANDS_DECIDER_SRC=<strands-decider>/src/strands_decider \
    .venv/bin/python python/log_model.py [VERSION]
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from snowflake.ml.model import custom_model  # noqa: E402
from snowflake.ml.registry import Registry  # noqa: E402

from bench_config import DATABASE, FQ_SCHEMA, MODEL_NAME, SCHEMA  # noqa: E402
from decider_model import BASE, SIGNATURE, WEIGHTS, DeciderModel  # noqa: E402
from snowpark_session import create_snowpark_session  # noqa: E402

WEIGHTS_DIR = Path(os.environ.get("DTR_WEIGHTS_DIR", ROOT / "models" / "dtr-decider"))
BASE_DIR = Path(os.environ.get("DTR_BASE_DIR", ROOT / "models" / "qwen3.5-2b-base"))
BASE_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
PACKAGE_DIR = Path(os.environ.get(
    "STRANDS_DECIDER_SRC", ROOT.parent / "strands-decider" / "src" / "strands_decider"))

# Exact pins, except torch: cuda_version="12.8" adds the PyTorch cu128 wheel index and
# would rewrite an exact pin to +cu128, which the deployment check rejects.
PIP_REQUIREMENTS = [
    "torch>=2.10.0,<2.11",
    "transformers==5.19.0",
    "peft==0.21.2",
    "accelerate==1.15.0",
    "pydantic==2.14.0",
    "flash-linear-attention==0.5.2",
    "fla-core==0.5.2",
    "numpy==1.26.4",
    "huggingface-hub==1.33.0",
    "tokenizers==0.23.2",
    "safetensors==0.8.0",
    "jinja2==3.1.6",
    "einops==0.8.2",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_artifacts() -> dict[str, str]:
    """Artifacts for the ModelContext, after checking the export's own manifest."""
    for required in ("head.safetensors", "lora/adapter_model.safetensors",
                     "lora/adapter_config.json", "tokenizer.json"):
        assert (WEIGHTS_DIR / required).exists(), f"{WEIGHTS_DIR}: missing {required}"
    assert not list(WEIGHTS_DIR.rglob("*.pt")), "export must be pickle-free (no .pt files)"
    prov = WEIGHTS_DIR / "provenance.json"
    if prov.exists():
        meta = json.loads(prov.read_text())
        if meta.get("head_sha256"):
            assert sha256(WEIGHTS_DIR / "head.safetensors") == meta["head_sha256"], "head digest"
        if meta.get("adapter_sha256"):
            assert sha256(WEIGHTS_DIR / "lora" / "adapter_model.safetensors") == \
                meta["adapter_sha256"], "adapter digest"
    assert (BASE_DIR / "config.json").exists(), f"{BASE_DIR}: not a model snapshot"
    return {WEIGHTS: str(WEIGHTS_DIR), BASE: str(BASE_DIR)}


def log(version: str, pip_requirements: list[str], what: str) -> None:
    model = DeciderModel(custom_model.ModelContext(artifacts=checked_artifacts()))
    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
    t0 = time.time()
    mv = reg.log_model(
        model,
        model_name=MODEL_NAME,
        version_name=version,
        signatures={"system_one": SIGNATURE},
        pip_requirements=pip_requirements,
        target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
        python_version="3.12",
        code_paths=[str(ROOT / "python" / "decider_model.py"), str(PACKAGE_DIR)],
        options={"cuda_version": "12.8", "relax_version": False},
        comment=(f"strands-decider fine-tuned on DTR Part I (Passenger Movement); base "
                 f"Qwen/Qwen3.5-2B-Base@{BASE_REVISION[:7]}. {what}"),
    )
    print(f"logged {mv.model_name} {mv.version_name} in {time.time() - t0:.0f}s", flush=True)
    print(mv.show_functions())
    session.close()


if __name__ == "__main__":
    log(sys.argv[1] if len(sys.argv) > 1 else "SERVICE", PIP_REQUIREMENTS,
        "system_one takes STATE_JSON + QUESTIONS_JSON and returns the System One response.")

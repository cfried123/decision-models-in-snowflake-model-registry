"""Log the decider-2b versions for the batch-inference levers.

  V11B  DeciderModel (V11's wrapper, unchanged), for run_batch
  V12   DeciderBatchedModel, torch.compile off
  V13   DeciderBatchedModel, torch.compile on
  V12B  DeciderBatchedModel, torch.compile off, rows over 2,048 tokens run one per forward

run_batch builds its image on a different base from create_service, and that base
constrains click<8.3.0, which V11's exact huggingface-hub==1.33.0 can't meet. So the
packages that set the model's numbers keep V11's exact pins (decider-ai, torch 2.10.x,
transformers, flash-linear-attention, fla-core, tokenizers) and the helper libraries get
ranges the resolver can fit to that base. The three versions share one image.

    SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/log_model_batch.py V12
"""
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from snowflake.ml.model import custom_model  # noqa: E402
from snowflake.ml.registry import Registry  # noqa: E402

from bench_config import DATABASE, FQ_SCHEMA, SCHEMA  # noqa: E402
from decider_model import SIGNATURE, WEIGHTS, DeciderModel  # noqa: E402
from decider_model_batched import OPTIONS, DeciderBatchedModel  # noqa: E402
from log_model import REVISION, WEIGHTS_DIR, WEIGHTS_SHA256  # noqa: E402
from snowpark_session import create_snowpark_session  # noqa: E402

PIP_REQUIREMENTS = [
    "decider-ai==1.6.0",
    "torch>=2.10.0,<2.11",
    "transformers==5.17.0",
    "flash-linear-attention==0.5.2",
    "fla-core==0.5.2",
    "tokenizers==0.23.2",
    "huggingface-hub>=1.5.0,<2",
    "numpy>=1.26.4,<3",
    "safetensors>=0.7,<1",
    "jinja2>=3.1,<4",
    "einops>=0.8,<1",
]

VERSIONS = {
    "V11B": ("DeciderModel, V11's wrapper unchanged, for run_batch", DeciderModel, None),
    "V12": ("DeciderBatchedModel: decider's batched server path, torch.compile off", DeciderBatchedModel,
            {"compile": False}),
    "V13": ("DeciderBatchedModel: decider's batched server path, torch.compile on", DeciderBatchedModel,
            {"compile": True}),
    "V12B": ("DeciderBatchedModel: batched up to 2,048 tokens, longer rows one per forward, torch.compile off",
             DeciderBatchedModel, {"compile": False, "batch_max_tokens": 2048}),
}


def main(version):
    what, cls, options = VERSIONS[version]
    digest = hashlib.sha256()
    with open(WEIGHTS_DIR / "model.safetensors", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    assert digest.hexdigest() == WEIGHTS_SHA256, "weights do not match the pinned revision"
    artifacts = {WEIGHTS: str(WEIGHTS_DIR)}
    if options is not None:
        opts = Path(tempfile.mkdtemp()) / "options.json"
        opts.write_text(json.dumps(options))
        artifacts[OPTIONS] = str(opts)
    model = cls(custom_model.ModelContext(artifacts=artifacts))
    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)
    t0 = time.time()
    reg.log_model(
        model,
        model_name="DECIDER_2B",
        version_name=version,
        signatures={"system_one": SIGNATURE},
        pip_requirements=PIP_REQUIREMENTS,
        target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
        python_version="3.12",
        code_paths=[str(ROOT / "python" / "decider_model.py"), str(ROOT / "python" / "decider_model_batched.py")],
        options={"cuda_version": "12.8", "relax_version": False},
        comment=f"decider-2b V11 weights (huggingface.co/Mapika/decider-2b@{REVISION[:7]}, Apache-2.0). {what}.",
    )
    print(f"logged {version} in {time.time() - t0:.0f}s", flush=True)
    session.close()


if __name__ == "__main__":
    main(sys.argv[1])

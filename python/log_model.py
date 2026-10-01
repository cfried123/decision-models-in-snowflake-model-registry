"""Log decider-2b to the Snowflake Model Registry as <database>.<schema>.DECIDER_2B, version SERVICE
(every package pinned exactly; the image for create_service).

The weights folder (Mapika/decider-2b at revision 533964d, Apache-2.0) is stored
as a model artifact, so the service never downloads from Hugging Face.

Constructing DeciderModel loads the weights once on this machine (CPU). The
registry doesn't upload that instance: it pickles the class and its context, and
the container constructs the model again.

    SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/log_model.py
"""
import hashlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from snowflake.ml.model import custom_model  # noqa: E402
from snowflake.ml.registry import Registry  # noqa: E402

from decider_model import SIGNATURE, WEIGHTS, DeciderModel  # noqa: E402
from bench_config import DATABASE, FQ_SCHEMA, SCHEMA  # noqa: E402
from snowpark_session import create_snowpark_session  # noqa: E402

WEIGHTS_DIR = ROOT / "models" / "decider-2b"
REVISION = "533964dae8be954c5b5e19fa4948e48408094c1e"
WEIGHTS_SHA256 = "acaef2228b134dcdc20cad4ee79219482c927ec819aa3687b9b8a575c338817f"

# Exact pins, except torch. cuda_version="12.8" makes the build add the PyTorch cu128
# wheel index, and would rewrite an exact torch pin to torch==2.10.0+cu128, which
# the deployment check rejects. The range stays on 2.10.x, the last torch line on
# the CUDA 12.8 runtime (2.11+ on Linux needs a CUDA 13 driver). RUNTIME_JSON
# reports the build that actually ran.
PIP_REQUIREMENTS = [
    "decider-ai==1.6.0",
    "torch>=2.10.0,<2.11",
    "transformers==5.17.0",
    "flash-linear-attention==0.5.2",
    "fla-core==0.5.2",
    "numpy==1.26.4",
    "huggingface-hub==1.33.0",
    "tokenizers==0.23.2",
    "safetensors==0.8.0",
    "jinja2==3.1.6",
    "einops==0.8.2",
]


def main():
    digest = hashlib.sha256()
    with open(WEIGHTS_DIR / "model.safetensors", "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    assert digest.hexdigest() == WEIGHTS_SHA256, "weights do not match the pinned revision"

    model = DeciderModel(custom_model.ModelContext(artifacts={WEIGHTS: str(WEIGHTS_DIR)}))
    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    reg = Registry(session=session, database_name=DATABASE, schema_name=SCHEMA)

    t0 = time.time()
    mv = reg.log_model(
        model,
        model_name="DECIDER_2B",
        version_name="SERVICE",
        signatures={"system_one": SIGNATURE},
        pip_requirements=PIP_REQUIREMENTS,
        target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
        python_version="3.12",
        code_paths=[str(ROOT / "python" / "decider_model.py")],
        options={"cuda_version": "12.8", "relax_version": False},
        comment=(f"decider-2b open weights (huggingface.co/Mapika/decider-2b@{REVISION[:7]}, "
                 "Apache-2.0), served with decider-ai 1.6.0. system_one takes the JevBench "
                 "/v1/systemone request and returns its response body."),
    )
    print(f"logged {mv.model_name} {mv.version_name} in {time.time() - t0:.0f}s")
    print(mv.show_functions())
    session.close()


if __name__ == "__main__":
    main()

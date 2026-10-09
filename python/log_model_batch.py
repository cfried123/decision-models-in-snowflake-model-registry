"""Log the DTR strands-decider BATCH version (the same DeciderModel wrapper as the
service) for run_batch.

run_batch builds its image on a different base from create_service, and that base
constrains click<8.3.0, which the service version's exact huggingface-hub pin can't
meet. So the packages that set the model's numbers keep the service version's exact
pins and the helper libraries get ranges the resolver can fit to that base.

Known gap: transformers 5.19 (needed for the Qwen3.5 torso) requires huggingface-hub>=1.31,
and every huggingface-hub release from 1.16.3 on requires click>=8.4, so this version
cannot currently resolve on the run_batch base (observed on a trial account, uv:
"No solution found"). Until that base moves, score DTR-Bench in bulk through the service
from SQL instead (sql/05_service_dtr.sql), which builds on the create_service base.

    SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/log_model_batch.py BATCH
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from log_model import log  # noqa: E402

PIP_REQUIREMENTS = [
    "torch>=2.10.0,<2.11",
    "transformers==5.19.0",
    "peft==0.21.2",
    "flash-linear-attention==0.5.2",
    "fla-core==0.5.2",
    "tokenizers==0.23.2",
    "accelerate>=1.10,<2",
    "pydantic>=2.7,<3",
    "huggingface-hub>=1.5.0,<2",
    "numpy>=1.26.4,<3",
    "safetensors>=0.7,<1",
    "jinja2>=3.1,<4",
    "einops>=0.8,<1",
]

if __name__ == "__main__":
    version = sys.argv[1] if len(sys.argv) > 1 else "BATCH"
    log(version, PIP_REQUIREMENTS, "DeciderModel, the same wrapper as the service, for run_batch.")

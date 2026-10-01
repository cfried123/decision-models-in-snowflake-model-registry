"""decider-2b as a Snowflake Model Registry custom model.

One method, system_one. It takes the request JevBench's `typesafe` adapter sends
to decider's /v1/systemone endpoint (STATE_JSON, QUESTIONS_JSON) and returns the
same response body (ANSWER_JSON), so the JevBench harness scores it unchanged.
ELAPSED_MS is the time spent inside the container on that row.

The registry pickles this class by value with its ModelContext, never the
instance, and calls DeciderModel(context) again inside the container. So
__init__ loads the model, and everything that depends on the machine happens
there, not at import time.

On CUDA, __init__ also:
  * checks that flash-linear-attention's Triton kernels run. Triton needs a C
    compiler at run time; if the check fails, `fla` is blocked and transformers
    uses its PyTorch reference kernels for the linear-attention layers.
  * turns torch.compile off. decider's Engine turns it on by default; JevBench
    ran decider with DECIDER_COMPILE=0. CUDA graphs stay on.
  * captures a CUDA graph per input-length bucket and runs one warm-up request
    per answer type.
RUNTIME_JSON records which path ran.
"""
import functools
import json
import os
import sys
import threading
import time

import pandas as pd
from snowflake.ml.model import custom_model, model_signature

WEIGHTS = "weights"   # ModelContext artifact name

SIGNATURE = model_signature.ModelSignature(
    inputs=[
        model_signature.FeatureSpec("STATE_JSON", model_signature.DataType.STRING),
        model_signature.FeatureSpec("QUESTIONS_JSON", model_signature.DataType.STRING),
    ],
    outputs=[
        model_signature.FeatureSpec("ANSWER_JSON", model_signature.DataType.STRING),
        model_signature.FeatureSpec("ELAPSED_MS", model_signature.DataType.DOUBLE),
        model_signature.FeatureSpec("INPUT_TOKENS", model_signature.DataType.INT64),
        model_signature.FeatureSpec("RUNTIME_JSON", model_signature.DataType.STRING),
    ],
)

# Warm-up requests: every answer type, an object state, and a state past the
# 2,048-token CUDA-graph limit (those run eagerly).
WARMUP = [
    ("Customer: my parcel still has not arrived after two weeks.",
     {"decision": {"type": "choice", "instructions": "What does the customer want?",
                   "criteria": {"refund": "Wants money back", "exchange": "Wants a different item",
                                "status": "Asks where the order is", "other": "Anything else"}}}),
    ("The order shipped on Monday and arrived on Thursday.",
     {"decision": {"type": "noul", "instructions": "Has the order been delivered?",
                   "criteria": {"true": "The text says so", "false": "The text says it has not"}}}),
    ({"ticket": 4411, "events": [{"t": i, "kind": "retry"} for i in range(12)]},
     {"decision": {"type": "score", "instructions": "Rate the severity.",
                   "criteria": ["none", "minor", "major", "critical"]}}),
    ("Refunds are allowed within 30 days of delivery for unused items. " * 300,
     {"decision": {"type": "noul", "instructions": "Does the policy allow refunds?",
                   "criteria": {"true": "Yes", "false": "No"}}}),
]


def _check_fla_kernels():
    """Run flash-linear-attention's chunk kernel once, which makes Triton compile it.
    If that fails, block `fla` so transformers falls back to its PyTorch kernels.
    Must run before the model loads: transformers picks the kernels when it imports
    the Qwen3.5 layers."""
    import torch
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        b, t, h, d = 1, 64, 2, 64
        q, k, v = (torch.randn(b, t, h, d, device="cuda", dtype=torch.bfloat16) for _ in range(3))
        g = -torch.rand(b, t, h, device="cuda", dtype=torch.float32)
        beta = torch.rand(b, t, h, device="cuda", dtype=torch.bfloat16)
        out, _ = chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize()
        ok = bool(torch.isfinite(out).all())
        probe = "ok" if ok else "non-finite output"
    except Exception as e:   # e.g. "Failed to find C compiler" from Triton
        ok, probe = False, f"{type(e).__name__}: {str(e)[:300]}"
    if not ok:
        for name in [m for m in sys.modules if m == "fla" or m.startswith("fla.")]:
            del sys.modules[name]
        sys.modules["fla"] = None   # `import fla` now raises ImportError
    return {"fla_probe": probe,
            "linear_attention_kernels": "flash-linear-attention (Triton)" if ok else "pytorch reference"}


def _warm_up(decider):
    """Capture a CUDA graph per input-length bucket, then run one request per answer type."""
    from decider.engine import T_BUCKETS
    t0 = time.perf_counter()
    decider.eng.warmup(shapes=[(1, t) for t in T_BUCKETS] + [(1, 3072), (1, 4096)])
    captures = decider.eng.stats["graph_captures"]
    for state, questions in WARMUP:
        decider.system_one(state, questions)
    return {"graph_captures": captures, "warmup_s": round(time.perf_counter() - t0, 1)}


class DeciderModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        t0 = time.perf_counter()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")                   # the service has no egress
        os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
        import torch

        on_gpu = torch.cuda.is_available()
        runtime = {"torch": torch.__version__, "cuda_available": on_gpu}
        if on_gpu:
            runtime.update(_check_fla_kernels())
            from decider import engine
            # Decider builds its Engine with torch.compile on and has no switch for it.
            engine.Engine = functools.partial(engine.Engine, compile=False)
            runtime.update(gpu=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda,
                           torch_compile=False, cuda_graphs=True)
        else:
            runtime["linear_attention_kernels"] = "pytorch reference"

        from decider.infer import Decider
        # Off CUDA, CPU: on an M3 Max, MPS returned NaN for every Score question.
        self.decider = Decider(context.path(WEIGHTS), device="cuda" if on_gpu else "cpu")
        runtime.update(model=self.decider.name, device=str(self.decider.dev),
                       load_s=round(time.perf_counter() - t0, 1))
        if on_gpu:
            runtime.update(_warm_up(self.decider))
        self.lock = threading.Lock()                                   # CUDA graphs replay from static buffers
        self.runtime_json = json.dumps(runtime, sort_keys=True)

    @custom_model.inference_api
    def system_one(self, input: pd.DataFrame) -> pd.DataFrame:
        answers, elapsed, tokens = [], [], []
        with self.lock:
            for state_json, questions_json in zip(input["STATE_JSON"], input["QUESTIONS_JSON"]):
                t0 = time.perf_counter()
                try:
                    res = self.decider.system_one(json.loads(state_json), json.loads(questions_json))
                    n = int(res.get("usage", {}).get("input_tokens") or 0)
                    out = json.dumps(res, ensure_ascii=False)
                except Exception as e:   # one bad row must not fail the batch; scored as invalid
                    n, out = 0, json.dumps({"error": f"{type(e).__name__}: {e}"})
                elapsed.append((time.perf_counter() - t0) * 1000.0)
                answers.append(out)
                tokens.append(n)
        return pd.DataFrame({
            "ANSWER_JSON": answers,
            "ELAPSED_MS": pd.Series(elapsed, dtype="float64"),
            "INPUT_TOKENS": pd.Series(tokens, dtype="int64"),
            "RUNTIME_JSON": [self.runtime_json] * len(answers),
        })

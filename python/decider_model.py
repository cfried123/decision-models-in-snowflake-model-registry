"""A strands-decider checkpoint (e.g. the DTR Part I fine-tune) as a Snowflake Model
Registry custom model.

One method, system_one, with the request/response shape of the original decider-2b
wrapper: STATE_JSON + QUESTIONS_JSON in, ANSWER_JSON ({"model", "answers", "usage"})
out, so the JevBench-style harness and the DTR app parse it unchanged. ELAPSED_MS is
the time spent inside the container on that row.

Two artifacts, both stored in the registry so the container never downloads:
  WEIGHTS  the pickle-free strands-decider export (hf_export.py): head.safetensors,
           lora/ (PEFT adapter), tokenizer, strands_decider_config.json
  BASE     the Qwen3.5-2B-Base snapshot the adapter was trained on

The registry pickles this class by value with its ModelContext and constructs it
again inside the container, so everything machine-dependent happens in __init__.
"""

import json
import os
import sys
import tempfile
import threading
import time

import pandas as pd
from snowflake.ml.model import custom_model, model_signature

WEIGHTS = "weights"  # ModelContext artifact: strands-decider HF export
BASE = "base"  # ModelContext artifact: base torso snapshot

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

# One request per answer type, so the first real row doesn't pay for lazy init.
WARMUP = [
    ({"scenario": "A GS-12 flies economy from Washington, DC to San Diego, CA."},
     {"q": {"type": "noul", "instructions": "Is this booking permitted?",
            "criteria": {"false": "No", "true": "Yes"}}}),
    ({"scenario": "A group of 30 moves 150 miles between installations."},
     {"q": {"type": "choice", "instructions": "Which mode applies?",
            "criteria": {"commercial_air": "Air", "charter_bus": "Bus", "rail": "Rail"}}}),
    ({"scenario": "A traveler used a blanket premium class authorization."},
     {"q": {"type": "score", "instructions": "Rate the compliance risk.",
            "criteria": ["compliant", "documentation gap", "excess cost", "prohibited"]}}),
]


def _check_fla_kernels():
    """Run flash-linear-attention's chunk kernel once so Triton compiles it; if that
    fails, block `fla` so transformers uses its PyTorch kernels. Must run before the
    model loads: transformers picks the kernels when it imports the Qwen3.5 layers."""
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
    except Exception as e:  # e.g. "Failed to find C compiler" from Triton
        ok, probe = False, f"{type(e).__name__}: {str(e)[:300]}"
    if not ok:
        for name in [m for m in sys.modules if m == "fla" or m.startswith("fla.")]:
            del sys.modules[name]
        sys.modules["fla"] = None  # `import fla` now raises ImportError
    return {"fla_probe": probe,
            "linear_attention_kernels": "flash-linear-attention (Triton)" if ok
            else "pytorch reference"}


def local_checkpoint(weights_dir: str, base_dir: str) -> str:
    """A view of `weights_dir` whose config points the torso at `base_dir`, so loading
    never touches the Hugging Face Hub. Artifacts are read-only, hence the copy."""
    view = tempfile.mkdtemp(prefix="strands_decider_")
    for name in os.listdir(weights_dir):
        if name not in ("strands_decider_config.json", "hobson_config.json", "provenance.json"):
            os.symlink(os.path.join(weights_dir, name), os.path.join(view, name))
    src = os.path.join(weights_dir, "strands_decider_config.json")
    if not os.path.exists(src):
        src = os.path.join(weights_dir, "hobson_config.json")
    with open(src, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["base_model"], cfg["base_revision"] = base_dir, None
    with open(os.path.join(view, "strands_decider_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return view


class DeciderModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        t0 = time.perf_counter()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the service has no egress
        os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
        import torch

        on_gpu = torch.cuda.is_available()
        runtime = {"torch": torch.__version__, "cuda_available": on_gpu}
        if on_gpu:
            runtime.update(_check_fla_kernels())
            runtime.update(gpu=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda)
        else:
            runtime["linear_attention_kernels"] = "pytorch reference"

        from pydantic import TypeAdapter
        from strands_decider.infer import load_engine
        from strands_decider.schema import Question

        ckpt = local_checkpoint(context.path(WEIGHTS), context.path(BASE))
        self.engine = load_engine(ckpt, device="cuda" if on_gpu else "cpu")
        self.questions = TypeAdapter(dict[str, Question])
        with open(os.path.join(context.path(WEIGHTS), "strands_decider_config.json"),
                  encoding="utf-8") as fh:
            self.model_name = json.load(fh).get("name") or "strands-decider"
        runtime.update(model=self.model_name, device="cuda" if on_gpu else "cpu",
                       load_s=round(time.perf_counter() - t0, 1))
        t1 = time.perf_counter()
        for state, questions in WARMUP:
            self._ask(state, questions)
        runtime["warmup_s"] = round(time.perf_counter() - t1, 1)
        self.lock = threading.Lock()  # one forward at a time on the shared engine
        self.runtime_json = json.dumps(runtime, sort_keys=True)

    def _ask(self, state, questions: dict) -> dict:
        # Objects pass through as objects: training rendered DTR states as JSON objects.
        res = self.engine.ask(state, self.questions.validate_python(questions))
        out = res.model_dump()
        out["model"] = self.model_name
        return out

    @custom_model.inference_api
    def system_one(self, input: pd.DataFrame) -> pd.DataFrame:
        answers, elapsed, tokens = [], [], []
        with self.lock:
            for state_json, questions_json in zip(input["STATE_JSON"], input["QUESTIONS_JSON"], strict=True):
                t0 = time.perf_counter()
                try:
                    res = self._ask(json.loads(state_json), json.loads(questions_json))
                    n = int(res.get("usage", {}).get("input_tokens") or 0)
                    out = json.dumps(res, ensure_ascii=False)
                except Exception as e:  # one bad row must not fail the batch; scored as invalid
                    n, out = 0, json.dumps({"error": f"{type(e).__name__}: {str(e)[:500]}"})
                elapsed.append((time.perf_counter() - t0) * 1000.0)
                answers.append(out)
                tokens.append(n)
        return pd.DataFrame({
            "ANSWER_JSON": answers,
            "ELAPSED_MS": pd.Series(elapsed, dtype="float64"),
            "INPUT_TOKENS": pd.Series(tokens, dtype="int64"),
            "RUNTIME_JSON": [self.runtime_json] * len(answers),
        })

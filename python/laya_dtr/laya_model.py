"""Model Registry `CustomModel` for a Laya checkpoint, with the same `system_one` contract as
python/decider_model.py: STATE_JSON + QUESTIONS_JSON in, ANSWER_JSON (System One response),
ELAPSED_MS, INPUT_TOKENS and RUNTIME_JSON out.

The checkpoint directory (rl_agent_config.json, model.safetensors, encoder/, tokenizer/) is the
only artifact; the laya package ships as a code path, so the container never downloads.
"""
import json
import os
import threading
import time

import pandas as pd
from snowflake.ml.model import custom_model, model_signature

CHECKPOINT = "checkpoint"  # ModelContext artifact: Laya checkpoint directory

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

WARMUP = [
    ({"scenario": "A GS-12 flies economy from Washington, DC to San Diego, CA."},
     {"q": {"type": "noul", "instructions": "Is this booking permitted?"}}),
    ({"scenario": "A group of 30 moves 150 miles between installations."},
     {"q": {"type": "choice", "instructions": "Which mode applies?",
            "criteria": {"commercial_air": "Air", "charter_bus": "Bus", "rail": "Rail"}}}),
    ({"scenario": "A traveler used a blanket premium class authorization."},
     {"q": {"type": "score", "instructions": "Rate the compliance risk.",
            "criteria": ["compliant", "documentation gap", "excess cost", "prohibited"]}}),
]
MAX_QUESTIONS = 16  # Security: bounds per-row work so one request cannot monopolise the worker


class LayaModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        t0 = time.perf_counter()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # the service has no egress
        os.environ.setdefault("USE_TF", "0")
        import torch
        import laya

        on_gpu = torch.cuda.is_available()
        ckpt = context.path(CHECKPOINT)
        with open(os.path.join(ckpt, "rl_agent_config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
        self.model_name = "laya-dtr-ft" if cfg.get("fine_tuned") else "laya-base"
        self.agent = laya.load(ckpt, device="cuda" if on_gpu else "cpu")
        runtime = {"torch": torch.__version__, "laya": laya.__version__, "cuda_available": on_gpu,
                   "device": "cuda" if on_gpu else "cpu", "model": self.model_name,
                   "threads": torch.get_num_threads(), "load_s": round(time.perf_counter() - t0, 1)}
        if on_gpu:
            runtime["gpu"] = torch.cuda.get_device_name(0)
        t1 = time.perf_counter()
        for state, questions in WARMUP:
            self._ask(state, questions)
        runtime["warmup_s"] = round(time.perf_counter() - t1, 1)
        self.lock = threading.Lock()
        self.runtime_json = json.dumps(runtime, sort_keys=True)

    def _ask(self, state, questions: dict) -> dict:
        if not isinstance(questions, dict) or not 0 < len(questions) <= MAX_QUESTIONS:
            raise ValueError(f"questions must be an object with 1..{MAX_QUESTIONS} entries")
        res = self.agent.predict(state, questions)
        out = {"model": self.model_name, "answers": res["answers"], "usage": res.get("usage", {})}
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

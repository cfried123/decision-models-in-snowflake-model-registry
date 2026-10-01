"""decider-2b as a Snowflake Model Registry custom model, scoring each call's rows in batches.

Same signature and response body as decider_model.DeciderModel (JevBench's /v1/systemone
request in, decider's response out). The difference is how a call's rows reach the GPU:
DeciderModel runs them one request at a time; this class prepares every request in the
call, then runs decider's own server path over all of them at once:

  * decider.engine_v2.EngineV2 with the whole (batch, length) CUDA-graph grid captured
    at start-up and then sealed, so no request pays a capture (or a compile);
  * decider.batching.plan_batches partitions the call's rows into forwards, padding a
    shorter row into a longer row's bucket when that is cheaper than another forward;
  * a request with several questions over a state of at least SHARED_MIN_TOKENS runs
    the state once and forks its cache per question (EngineV2.score_shared).

These are the defaults of decider's HTTP server (decider/serve.py), which can't be
imported here because it needs fastapi.

The "options" artifact is a JSON file. {"compile": true} turns torch.compile on.
{"batch_max_tokens": N} batches only rows of at most N tokens; longer rows run one per
forward at EngineV2's padded length, the way DeciderModel runs every row.
ELAPSED_MS is the call's time spread evenly over its rows. RUNTIME_JSON carries the
call's start and end (epoch seconds), its row count and the number of forwards.
"""
import json
import os
import threading
import time

import pandas as pd
from snowflake.ml.model import custom_model

from decider_model import SIGNATURE, WARMUP, WEIGHTS, _check_fla_kernels  # noqa: F401

OPTIONS = "options"           # ModelContext artifact: JSON serving options
MAX_BATCH = 32                # decider/serve.py DECIDER_MAX_BATCH
SHARED_MIN_TOKENS = 768       # decider/serve.py DECIDER_SHARED_MIN_TOKENS
MAX_STATE_TOKENS = 32768      # decider/serve.py DECIDER_MAX_STATE_TOKENS

__all__ = ["DeciderBatchedModel", "SIGNATURE", "WEIGHTS", "OPTIONS"]


class DeciderBatchedModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        t0 = time.perf_counter()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")                   # the service has no egress
        os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
        import torch
        from decider import temperature as TT

        with open(context.path(OPTIONS)) as f:
            options = json.load(f)
        compile_ = bool(options.get("compile", False))
        self.batch_max_tokens = options.get("batch_max_tokens")
        on_gpu = torch.cuda.is_available()
        runtime = {"torch": torch.__version__, "cuda_available": on_gpu, "wrapper": "batched",
                   "torch_compile": compile_ and on_gpu, "batch_max_tokens": self.batch_max_tokens}
        if on_gpu:
            runtime.update(_check_fla_kernels())
            runtime.update(gpu=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda)
        else:
            runtime["linear_attention_kernels"] = "pytorch reference"

        from decider.engine_v2 import EngineV2
        path = context.path(WEIGHTS)
        with open(os.path.join(path, "decider_config.json")) as f:
            cfg = json.load(f)
        (self.temp, self.temp_by_type), _ = TT.from_config(cfg, None)
        self.isolated = bool(cfg.get("isolated_levels", False))
        self.name = "decider-" + str(cfg.get("version", "dev"))
        self.eng = EngineV2(path, device="cuda" if on_gpu else "cpu", compile=compile_ and on_gpu,
                            max_ctx_tokens=MAX_STATE_TOKENS)
        runtime.update(model=self.name, load_s=round(time.perf_counter() - t0, 1))
        if on_gpu:
            t1 = time.perf_counter()
            self.eng.warmup()
            self.eng.seal()
            self._score([(json.dumps(s), json.dumps(q)) for s, q in WARMUP])
            runtime.update(graph_captures=len(self.eng.graphs), warmup_s=round(time.perf_counter() - t1, 1))
        else:
            self.eng.seal()
        self.lock = threading.Lock()                                   # one GPU, static graph buffers
        self.runtime = runtime

    def _prepare(self, state, questions):
        """decider/serve.py prepare(): render, plan the rows, tokenize the state once."""
        from decider import systemone as S1
        from decider.prompt_fast import build_rows
        ctx = S1.render_state(state)
        rqs = {k: S1.render_question(v) for k, v in questions.items()}
        flat, index = S1.plan_rows(rqs, self.isolated)
        items, _ = build_rows(self.eng.tok, ctx, [[(r["question"], list(r["options"]))] for r in flat],
                              max_ctx_tokens=MAX_STATE_TOKENS)
        for it, t in zip(items, S1.row_types(rqs, index)):
            it["types"] = [t]
        return rqs, index, items

    def _score(self, pairs):
        """pairs: [(STATE_JSON, QUESTIONS_JSON)] -> ([response or error dict], forwards)."""
        from decider import systemone as S1
        from decider import temperature as TT
        from decider.batching import DEFAULT_MERGE_OVERHEAD_TOKENS, plan_batches
        eng = self.eng
        temps = lambda its: TT.for_items(self.temp, self.temp_by_type, its)  # noqa: E731
        reqs, errors = [None] * len(pairs), {}
        for i, (s, q) in enumerate(pairs):
            try:
                reqs[i] = self._prepare(json.loads(s), json.loads(q))
            except Exception as e:  # one bad row must not fail the call; scored as invalid
                errors[i] = f"{type(e).__name__}: {e}"
        probs = {}                                       # (request, row) -> probabilities
        pool = []                                        # rows that share forwards across requests
        f0 = eng.stats["forwards"]
        for i, r in enumerate(reqs):
            if r is None:
                continue
            items = r[2]
            if len(items) > 1 and min(len(it["ids"]) for it in items) >= SHARED_MIN_TOKENS:
                try:
                    for j, p in enumerate(eng.score_shared(items, temperature=temps(items))):
                        probs[(i, j)] = p
                except Exception as e:
                    errors[i] = f"{type(e).__name__}: {e}"
            else:
                pool += [(i, j, it) for j, it in enumerate(items)]
        if self.batch_max_tokens is not None:
            solo = [p for p in pool if len(p[2]["ids"]) > self.batch_max_tokens]
            pool = [p for p in pool if len(p[2]["ids"]) <= self.batch_max_tokens]
        else:
            solo = []
        for i, j, it in solo:
            try:
                probs[(i, j)] = eng.score_items([it], temperature=temps([it]))[0]
            except Exception as e:
                errors[i] = f"{type(e).__name__}: {e}"
        groups = plan_batches([len(it["ids"]) for _, _, it in pool], eng.pad_len, eng.max_rows, MAX_BATCH,
                              DEFAULT_MERGE_OVERHEAD_TOKENS, lambda n: eng.t_bucket(n) is not None)
        for _, idx in groups:
            part = [pool[k] for k in idx]
            try:
                out = eng.score_items([it for _, _, it in part], temperature=temps([it for _, _, it in part]))
                for (i, j, _), p in zip(part, out):
                    probs[(i, j)] = p
            except Exception as e:
                for i, _, _ in part:
                    errors[i] = f"{type(e).__name__}: {e}"
        results = []
        for i, r in enumerate(reqs):
            if i in errors:
                results.append({"error": errors[i]})
                continue
            rqs, index, items = r
            flat = [p.tolist() for j in range(len(items)) for p in probs[(i, j)]]
            results.append({"model": self.name, "answers": S1.assemble(rqs, index, flat),
                            "usage": {"input_tokens": S1.unique_tokens(items), "output_tokens": 0}})
        return results, eng.stats["forwards"] - f0

    @custom_model.inference_api
    def system_one(self, input: pd.DataFrame) -> pd.DataFrame:
        with self.lock:
            start = time.time()
            results, forwards = self._score(list(zip(input["STATE_JSON"], input["QUESTIONS_JSON"])))
            end = time.time()
        n = len(results)
        runtime = json.dumps(dict(self.runtime, call_start=start, call_end=end, call_rows=n,
                                  call_forwards=forwards), sort_keys=True)
        return pd.DataFrame({
            "ANSWER_JSON": [json.dumps(r, ensure_ascii=False) for r in results],
            "ELAPSED_MS": pd.Series([(end - start) * 1000.0 / max(n, 1)] * n, dtype="float64"),
            "INPUT_TOKENS": pd.Series([int(r.get("usage", {}).get("input_tokens") or 0) for r in results],
                                      dtype="int64"),
            "RUNTIME_JSON": [runtime] * n,
        })

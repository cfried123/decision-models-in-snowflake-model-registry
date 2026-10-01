"""Replay the batching plans offline: how many forwards and padded tokens each policy
gives on the 231 JevBench items (tokenizer only, no model, no GPU).

  v11       DeciderModel: one request at a time, Engine (v1) buckets up to 2,048, then
            multiples of 1,024 (score_shared for multi-row requests counted as its rows)
  v12       DeciderBatchedModel defaults: EngineV2 buckets up to 8,192, plan_batches
  v12_2048  EngineV2 with length buckets only up to 2,048: longer rows run eager,
            grouped by exact padded length (multiples of 1,024)
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from transformers import AutoTokenizer  # noqa: E402

from decider import engine as E1  # noqa: E402
from decider import engine_v2 as E2  # noqa: E402
from decider import systemone as S1  # noqa: E402
from decider.batching import DEFAULT_MERGE_OVERHEAD_TOKENS, plan_batches  # noqa: E402
from decider.prompt_fast import build_rows  # noqa: E402

W = REPO / "models" / "decider-2b-v11"
SHARED_MIN = 768


class Grid:
    """EngineV2's bucket arithmetic, without a model."""
    def __init__(self, t_buckets, b_buckets=E2.B_BUCKETS, budget=E2.TOKEN_BUDGET):
        self.t_buckets, self.b_buckets, self.budget = sorted(t_buckets), sorted(b_buckets), budget
        shapes = [(B, T) for T in self.t_buckets for B in self.b_buckets if B == 1 or B * T <= budget]
        self.b_for_t = {}
        for B, T in shapes:
            self.b_for_t.setdefault(T, []).append(B)

    def t_bucket(self, n):
        return next((t for t in self.t_buckets if n <= t), None)

    def pad_len(self, n):
        return self.t_bucket(n) or -(-n // E2.LONG_STEP) * E2.LONG_STEP

    def max_rows(self, T):
        av = self.b_for_t.get(T)
        return max(av) if av else max(1, self.budget // max(T, 1))

    def b_plan(self, n, T):
        av = sorted(self.b_for_t.get(T, []))
        if not av:
            return [n]
        fit = next((b for b in av if b >= n), None)
        if fit is not None:
            return [fit]
        out, big = [], av[-1]
        while n > big:
            out.append(big); n -= big
        out.append(next(b for b in av if b >= n))
        return out


def rows_of(tok):
    reqs = []
    for r in map(json.loads, open(REPO / "runs" / "jevbench_items.jsonl")):
        state, qs = json.loads(r["STATE_JSON"]), json.loads(r["QUESTIONS_JSON"])
        ctx = S1.render_state(state)
        rqs = {k: S1.render_question(v) for k, v in qs.items()}
        flat, _ = S1.plan_rows(rqs, True)
        items, _ = build_rows(tok, ctx, [[(x["question"], list(x["options"]))] for x in flat], max_ctx_tokens=32768)
        reqs.append([len(it["ids"]) for it in items])
    return reqs


def v11(reqs):
    fw, pad = 0, 0
    for lens in reqs:
        if len(lens) > 1:                         # score_shared: counted as one eager pass per row at exact length
            fw += len(lens); pad += sum(lens)
            continue
        n = lens[0]
        T = E1._bucket(n, E1.T_BUCKETS) or -(-n // E1.LONG_STEP) * E1.LONG_STEP
        fw += 1; pad += T
    return fw, pad


def batched(reqs, grid):
    fw, pad, shared, pool = 0, 0, 0, []
    for lens in reqs:
        if len(lens) > 1 and min(lens) >= SHARED_MIN:
            fw += len(lens); pad += sum(lens); shared += 1
        else:
            pool += lens
    groups = plan_batches(pool, grid.pad_len, grid.max_rows, 32, DEFAULT_MERGE_OVERHEAD_TOKENS,
                          lambda n: grid.t_bucket(n) is not None)
    detail = []
    for T, idx in groups:
        plan = grid.b_plan(len(idx), T) if grid.t_bucket(T) is not None else [len(idx)]
        for B in plan:
            fw += 1; pad += B * T
            detail.append((T, B))
    return fw, pad, shared, detail


def main():
    tok = AutoTokenizer.from_pretrained(str(W))
    reqs = rows_of(tok)
    real = sum(sum(x) for x in reqs)
    print(f"{len(reqs)} requests, {sum(len(x) for x in reqs)} rows, {real} real tokens")
    f, p = v11(reqs)
    print(f"v11       forwards {f:4d}  padded tokens {p:7d}  ({p / real:.2f}x real)")
    for name, grid in (("v12", Grid(E2.T_BUCKETS)), ("v12_2048", Grid([t for t in E2.T_BUCKETS if t <= 2048]))):
        f, p, sh, detail = batched(reqs, grid)
        long_pad = sum(B * T for T, B in detail if T > 2048)
        print(f"{name:9s} forwards {f:4d}  padded tokens {p:7d}  ({p / real:.2f}x real)  shared requests {sh}  "
              f"padded tokens in forwards over 2,048: {long_pad}")
        print("          forwards (T, B):", sorted(detail, reverse=True)[:14])


if __name__ == "__main__":
    main()

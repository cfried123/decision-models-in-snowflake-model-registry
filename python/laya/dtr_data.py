"""DTR Part I data in Laya's training schema, from the strands-decider DTR recipe.

Training rows are `{state, questions, expected}` (or `gold` for the ambiguous binary cases,
whose two weight-0.5 opposite-label copies become one 50/50 target). DTR-Bench items are
already in the Jev request shape that Laya reads, so they are passed through unchanged and
all four models (strands-decider v21/FT, Laya base/FT) are scored on the same 142 items.

    python python/laya/dtr_data.py [--out runs/laya_data]
"""
import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SD = Path(os.environ.get("STRANDS_DECIDER_REPO", ROOT.parent / "strands-decider"))


def question(ex: dict) -> dict:
    kind, opts = ex["kind"], ex["options"]
    q = {"type": kind, "instructions": ex["instructions"]}
    if kind == "choice":
        q["criteria"] = {name: desc for name, desc in opts}
    elif kind == "score":
        q["criteria"] = [desc for _, desc in opts]
    return q


def rows(path: Path) -> list[dict]:
    """One Laya row per distinct (state, question); paired ambiguous copies merge to 50/50."""
    merged: "OrderedDict[str, dict]" = OrderedDict()
    for line in path.read_text().splitlines():
        ex = json.loads(line)
        q = question(ex)
        key = json.dumps([ex["state"], q], sort_keys=True)
        if ex["kind"] == "noul" and ex["weight"] < 1.0:
            row = merged.setdefault(key, {"state": ex["state"], "questions": {"d": q},
                                          "gold": {"d": {"probabilities": {"false": 0.0, "true": 0.0}}},
                                          "task": ex["task"]})
            row["gold"]["d"]["probabilities"]["true" if ex["label"] else "false"] += ex["weight"]
            continue
        if ex["kind"] == "choice":
            expected = ex["options"][ex["label"]][0]
        elif ex["kind"] == "noul":
            expected = bool(ex["label"])
        else:
            expected = int(ex["label"])
        merged[key] = {"state": ex["state"], "questions": {"d": q}, "expected": {"d": expected},
                       "task": ex["task"]}
    return list(merged.values())


def bench(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, src in (("train", "data/dtr/train.jsonl"), ("holdout", "data/dtr/train.holdout.jsonl")):
        rs = rows(SD / src)
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
        counts[name] = len(rs)
    items = bench(SD / "data" / "dtr_bench.jsonl")
    (out / "dtr_bench.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
    counts["dtr_bench"] = len(items)
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs" / "laya_data"))
    print(write(Path(ap.parse_args().out)))

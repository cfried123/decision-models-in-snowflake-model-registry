"""Threshold sweep for the decider-2b -> AI_CLASSIFY cascade, scored with JevBench's harness.

Run after sql/10_threshold_sweep.sql. Every row with confidence below 0.9 has an AI_CLASSIFY
label (Run 2's for the 58 rows it escalated, the sweep's for the rest), so any threshold up to
0.9 can be replayed without new calls. Confidence is decider's `confidence` for Choice and
Score and |2 * P(yes) - 1| for Noul. This is an after-the-fact analysis, not pre-registered;
the cross-validation at the end estimates how much of the best in-sample gain survives when
the threshold is picked on one half of the items and scored on the other.

Writes runs/sweep/report.json.

    SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/threshold_sweep.py
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_with_jevbench import RUNS, CascadeAdapter, ReplayTypeSafe, Task, run_harness  # noqa: E402
from bench_config import FQ_SCHEMA  # noqa: E402
from snowpark_session import create_snowpark_session  # noqa: E402

THRESHOLDS = [round(0.05 * k, 2) for k in range(19)]      # 0.00 .. 0.90
SPLITS = 2000


def fetch(session):
    q = lambda sql: [r.as_dict() for r in session.sql(sql).collect()]
    items = q("SELECT ITEM_ID, TIER, QTYPE, ITEM_JSON FROM JEVBENCH_ITEMS")
    dec = q("""SELECT d.ITEM_ID, d.RESULT:ANSWER_JSON::STRING AS A, v.ESCALATE,
                      IFF(v.QTYPE = 'noul', ABS(2 * v.ANSWER:noul::FLOAT - 1), v.ANSWER:confidence::FLOAT) AS CONF
               FROM DECIDER_ANSWERS d JOIN DECIDER_DECISIONS v ON v.RUN_ID = d.RUN_ID AND v.ITEM_ID = d.ITEM_ID
               WHERE d.RUN_ID = 'run1'""")
    ai = q("""SELECT RUN_ID, ITEM_ID, RESULT:labels[0]::STRING AS LABEL FROM AI_CLASSIFY_ANSWERS
              WHERE RUN_ID IN ('run2', 'sweep') AND RESULT:labels[0] IS NOT NULL""")
    tokens = q("SELECT ITEM_ID, TOKENS FROM TOKEN_PRECOUNT")
    return items, dec, ai, tokens


def main():
    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    items, dec, ai, tokens = fetch(session)
    session.close()

    tasks = []
    for it in items:
        t = Task.from_dict(json.loads(it["ITEM_JSON"]))
        t.tier = it["TIER"]
        tasks.append(t)
    responses = {d["ITEM_ID"]: json.loads(d["A"]) for d in dec}
    conf = {d["ITEM_ID"]: d["CONF"] for d in dec}
    prereg = {d["ITEM_ID"]: bool(d["ESCALATE"]) for d in dec}
    labels = {a["ITEM_ID"]: a["LABEL"] for a in ai}
    tok = {r["ITEM_ID"]: r["TOKENS"] for r in tokens}
    tier = {t.id: t.tier for t in tasks}
    qtype = {t.id: t.question["type"] for t in tasks}
    missing = [i for i in conf if conf[i] < 0.9 and i not in labels]
    assert not missing, f"rows below 0.9 without an AI_CLASSIFY label: {missing}"

    rec_d, _ = run_harness("sweep_decider", ReplayTypeSafe(responses), tasks)
    rec_a, _ = run_harness("sweep_ai_classify", CascadeAdapter(responses, labels), tasks)
    dec_ok = {r["task_id"]: bool(r["correct"]) for r in rec_d}
    ai_ok = {r["task_id"]: bool(r["correct"]) for r in rec_a if r["task_id"] in labels}

    def cascade(ids, esc):
        return sum(ai_ok[i] if esc(i) else dec_ok[i] for i in ids)

    ids = [t.id for t in tasks]
    base = sum(dec_ok[i] for i in ids)
    sweep = []
    for th in THRESHOLDS:
        esc = [i for i in ids if conf[i] < th]
        row = {"threshold": th, "escalated": len(esc), "correct": cascade(ids, lambda i: conf[i] < th),
               "ai_classify_input_tokens": sum(tok[i] for i in esc)}
        for k in ("easy", "standard", "hard"):
            sub = [i for i in ids if tier[i] == k]
            row[f"correct_{k}"] = cascade(sub, lambda i: conf[i] < th)
        sweep.append(row)

    bands = []
    for lo, hi in ((0.0, 0.3), (0.3, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9)):
        b = [i for i in ids if lo <= conf[i] < hi]
        bands.append({"band": [lo, hi], "n": len(b), "decider_correct": sum(dec_ok[i] for i in b),
                      "ai_classify_correct": sum(ai_ok[i] for i in b),
                      "by_tier": {k: {"n": sum(1 for i in b if tier[i] == k),
                                      "decider": sum(dec_ok[i] for i in b if tier[i] == k),
                                      "ai_classify": sum(ai_ok[i] for i in b if tier[i] == k)}
                                  for k in ("standard", "hard") if any(tier[i] == k for i in b)},
                      "by_type": {k: {"n": sum(1 for i in b if qtype[i] == k),
                                      "decider": sum(dec_ok[i] for i in b if qtype[i] == k),
                                      "ai_classify": sum(ai_ok[i] for i in b if qtype[i] == k)}
                                  for k in ("choice", "noul", "score") if any(qtype[i] == k for i in b)}})

    # Pick the threshold on a random half (stratified by tier), score it on the other half.
    rng = random.Random(20260930)
    by_tier = {k: [i for i in ids if tier[i] == k] for k in ("easy", "standard", "hard")}
    gains, picks = [], []
    for _ in range(SPLITS):
        a, b = [], []
        for members in by_tier.values():
            m = members[:]
            rng.shuffle(m)
            a += m[: len(m) // 2]
            b += m[len(m) // 2:]
        best = max(THRESHOLDS, key=lambda th: (cascade(a, lambda i: conf[i] < th), -th))
        picks.append(best)
        gains.append(cascade(b, lambda i: conf[i] < best) - sum(dec_ok[i] for i in b))
    gains.sort()
    held_out = {"splits": SPLITS, "held_out_items": len(ids) - len(ids) // 2,
                "mean_gain": sum(gains) / SPLITS,
                "p05_p50_p95_gain": [gains[int(0.05 * SPLITS)], gains[SPLITS // 2], gains[int(0.95 * SPLITS)]],
                "share_gain_positive": sum(g > 0 for g in gains) / SPLITS,
                "share_gain_negative": sum(g < 0 for g in gains) / SPLITS,
                "thresholds_picked": {str(th): picks.count(th) for th in THRESHOLDS if picks.count(th)}}

    report = {"decider_alone_correct": base, "n": len(ids),
              "preregistered_rule_correct": cascade(ids, lambda i: prereg[i]),
              "preregistered_rule_escalated": sum(prereg.values()),
              "sweep": sweep, "bands": bands, "held_out": held_out,
              "note": "Confidence: decider confidence (Choice, Score); |2*P(yes)-1| (Noul). "
                      "Tokens are AI_COUNT_TOKENS pre-counts (37 rows brace-swapped)."}
    out = RUNS / "sweep"
    out.mkdir(exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("decider_alone_correct", "preregistered_rule_correct",
                                             "preregistered_rule_escalated", "held_out")}, indent=1))
    for r in sweep:
        print(f'{r["threshold"]:.2f}  esc={r["escalated"]:3d}  correct={r["correct"]}  '
              f'std={r["correct_standard"]} hard={r["correct_hard"]}  tokens={r["ai_classify_input_tokens"]}')
    for b in bands:
        print(b["band"], b["n"], "decider", b["decider_correct"], "ai", b["ai_classify_correct"], b["by_tier"], b["by_type"])


if __name__ == "__main__":
    main()

"""Score the service run and the cascade with JevBench's own harness (bb05a335), unmodified.

Decider answers are replayed through jevbench.adapters.typesafe.TypeSafeAdapter,
with only its HTTP call replaced by the stored service output. Escalated rows in
the cascade go through a label-only adapter, which JevBench scores with score_label
(accuracy only; no calibration metrics for those rows).

Writes runs/<run>/{results.jsonl, summary.json, raw/, ledger.jsonl} and
runs/report.json with the tables used in the post.

    SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/score_with_jevbench.py
"""
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".cache" / "jevbench"))
sys.path.insert(0, str(ROOT / "python"))

from jevbench.adapters import typesafe as ts  # noqa: E402
from jevbench.adapters.base import DecisionResult  # noqa: E402
from jevbench.budget import Ledger  # noqa: E402
from jevbench.runner import Runner  # noqa: E402
from jevbench.summarize import summarize  # noqa: E402
from jevbench.tasks import Task  # noqa: E402

from bench_config import FQ_SCHEMA  # noqa: E402
from snowpark_session import create_snowpark_session  # noqa: E402

RUNS = ROOT / "runs"


class ReplayTypeSafe(ts.TypeSafeAdapter):
    """TypeSafeAdapter whose POST returns the stored service response for the task."""
    name = "typesafe"
    cost_basis = "snowflake_metered_credits_reported_separately"

    def __init__(self, responses):
        super().__init__(endpoint=f"snowflake://{FQ_SCHEMA}.DECIDER_2B_SVC",
                         model="decider-2b", key_env="")
        self.responses = responses

    def run(self, task):
        resp = self.responses.get(task.id)
        ts.http_post_json = lambda url, body, headers, timeout: (
            (200, resp, 0.0) if resp is not None else (500, {"error": "no stored response"}, 0.0))
        return super().run(task)


class CascadeAdapter:
    """The cascade: AI_CLASSIFY's label where the row was escalated, decider's answer otherwise."""
    name = "decider_cascade_ai_classify"
    price_input_per_m = None
    price_output_per_m = None
    cost_basis = "snowflake_metered_credits_reported_separately"

    def __init__(self, responses, ai_labels):
        self.decider = ReplayTypeSafe(responses)
        self.ai_labels = ai_labels            # item_id -> label for escalated rows with a result

    def reserve_estimate(self, task):
        return None

    def run(self, task):
        if task.id in self.ai_labels:
            return DecisionResult(adapter=self.name, ok=True, probs=None,
                                  probs_source="label_only_no_calibrated_distribution",
                                  model="AI_CLASSIFY", label=self.ai_labels[task.id],
                                  raw={"labels": [self.ai_labels[task.id]]})
        res = self.decider.run(task)
        res.adapter = self.name
        return res


def fetch(session):
    q = lambda sql: [r.as_dict() for r in session.sql(sql).collect()]
    items = q("SELECT ITEM_ID, TIER, ITEM_JSON, N_OPTIONS, SURFACE_ANSWER FROM JEVBENCH_ITEMS")
    answers = defaultdict(dict)
    for r in q("SELECT RUN_ID, ITEM_ID, RESULT:ANSWER_JSON::STRING AS A FROM DECIDER_ANSWERS "
               "WHERE RUN_ID IN ('service', 'cascade')"):
        answers[r["RUN_ID"]][r["ITEM_ID"]] = json.loads(r["A"]) if r["A"] else None
    final = {r["ITEM_ID"]: r for r in q("SELECT * FROM CASCADE_FINAL")}
    return items, answers, final


def run_harness(run_id, adapter, tasks):
    out = RUNS / run_id
    if out.exists():
        shutil.rmtree(out)                    # the harness opens its files exclusively
    runner = Runner(adapter, Ledger(out / "ledger.jsonl", cap_usd=1e9), raw_dir=out / "raw",
                    default_reserve_usd=0.0)
    records = runner.run_all(tasks, progress_every=1000, results_path=out / "results.jsonl")
    summary = summarize(tasks, records)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return records, summary


def competence(tasks, records, n_options):
    """Unofficial per-type chance-corrected competence on these items (METHOD-v1.5 §3.1)."""
    rec = {r["task_id"]: r for r in records}
    out = {}
    for tier in ("easy", "standard", "hard", "all"):
        ts_ = [t for t in tasks if tier == "all" or t.tier == tier]
        row = {}
        ch = [t for t in ts_ if t.question["type"] == "choice"]
        if ch:
            acc = sum(bool(rec[t.id]["correct"]) for t in ch) / len(ch)
            cbar = sum(1 / n_options[t.id] for t in ch) / len(ch)
            row["choice"] = {"n": len(ch), "acc": acc, "chance": cbar, "cc": 100 * (acc - cbar) / (1 - cbar)}
        nl = [t for t in ts_ if t.question["type"] == "noul"]
        if nl:
            ok = 0
            for t in nl:
                r = rec[t.id]
                p = (r.get("probs") or {}).get("yes") if r.get("probs") else None
                if p is None:                              # label-only answer: the label is the answer
                    ok += bool(r["correct"])
                else:                                      # 0.2 < p < 0.8 is an abstention, counted wrong
                    ok += (p >= 0.8 and t.expected == "yes") or (p <= 0.2 and t.expected == "no")
            acc = ok / len(nl)
            row["noul"] = {"n": len(nl), "acc_abstain_wrong": acc, "cc": 100 * (acc - 0.5) / 0.5}
        sc = [t for t in ts_ if t.question["type"] == "score"]
        if sc:
            nmae, nmae_chance = [], []
            for t in sc:
                r, k, gold = rec[t.id], len(t.labels), int(t.expected)
                if r.get("ordinal_ev") is not None:
                    pred = r["ordinal_ev"]
                elif r.get("valid") and r.get("predicted") is not None:
                    pred = float(r["predicted"])
                else:
                    pred = None
                nmae.append(1.0 if pred is None else abs(pred - gold) / (k - 1))
                nmae_chance.append(sum(abs(l - gold) for l in range(k)) / k / (k - 1))
            m, mc = sum(nmae) / len(nmae), sum(nmae_chance) / len(nmae_chance)
            row["score"] = {"n": len(sc), "nmae": m, "nmae_chance": mc, "cc": 100 * (1 - m / mc)}
        out[tier] = row
    return out


def by_group(tasks, records, key):
    rec = {r["task_id"]: r for r in records}
    groups = defaultdict(list)
    for t in tasks:
        groups[key(t)].append(bool(rec[t.id]["correct"]))
    return {g: {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)} for g, v in sorted(groups.items())}


def main():
    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    items, answers, final = fetch(session)
    session.close()

    tasks, tier_of, n_options, surface = [], {}, {}, {}
    for it in items:
        t = Task.from_dict(json.loads(it["ITEM_JSON"]))
        t.tier = it["TIER"]
        tasks.append(t)
        tier_of[t.id], n_options[t.id], surface[t.id] = it["TIER"], it["N_OPTIONS"], it["SURFACE_ANSWER"]
    order = {"easy": 0, "standard": 1, "hard": 2}
    tasks.sort(key=lambda t: (order[t.tier], t.id))

    report = {}
    rec1, sum1 = run_harness("service", ReplayTypeSafe(answers["service"]), tasks)

    ai_labels = {i: f["AI_LABEL"] for i, f in final.items() if f["ESCALATE"] and f["AI_LABEL"] is not None}
    rec2, sum2 = run_harness("cascade", CascadeAdapter(answers["cascade"], ai_labels), tasks)

    # The cascade's decider pass alone, to measure what the cascade changed.
    rec2d, _ = run_harness("cascade_decider_pass", ReplayTypeSafe(answers["cascade"]), tasks)

    for name, recs, summ in (("service", rec1, sum1), ("cascade", rec2, sum2)):
        report[name] = {
            "accuracy": summ["accuracy"], "n_correct": summ["n_correct"], "n_scorable": summ["n_scorable"],
            "schema_validity": summ["schema_validity"], "brier_mean": summ["brier_mean"], "ece": summ["ece"],
            "calibration_n": summ["calibration_n"], "ordinal_mae": summ["ordinal_mae"],
            "paraphrase_consistency": summ["paraphrase_consistency"],
            "by_tier": by_group(tasks, recs, lambda t: t.tier),
            "by_type": by_group(tasks, recs, lambda t: t.question["type"]),
            "by_hard_family": by_group([t for t in tasks if t.tier == "hard"], recs, lambda t: t.family),
            "competence_unofficial": competence(tasks, recs, n_options),
        }

    r1 = {r["task_id"]: r for r in rec1}
    r2 = {r["task_id"]: r for r in rec2}
    r2d = {r["task_id"]: r for r in rec2d}
    esc = [i for i, f in final.items() if f["ESCALATE"]]
    flips = {"wrong_to_right": [], "right_to_wrong": [], "wrong_to_wrong_changed": []}
    for i in esc:
        a, b = bool(r2d[i]["correct"]), bool(r2[i]["correct"])
        if not a and b:
            flips["wrong_to_right"].append(i)
        elif a and not b:
            flips["right_to_wrong"].append(i)
        elif not a and not b and r2d[i]["predicted"] != r2[i]["predicted"]:
            flips["wrong_to_wrong_changed"].append(i)
    distractor = [i for i in esc if surface.get(i) and r2[i]["predicted"] == surface[i]]
    report["cascade_changes"] = {
        "escalated": len(esc), "ai_labels": len(ai_labels), "ai_null_fallback_to_decider": len(esc) - len(ai_labels),
        "escalated_by_tier": dict(sorted(defaultdict(int, {k: sum(1 for i in esc if tier_of[i] == k)
                                                           for k in ("easy", "standard", "hard")}).items())),
        "decider_correct_on_escalated": sum(bool(r2d[i]["correct"]) for i in esc),
        "cascade_correct_on_escalated": sum(bool(r2[i]["correct"]) for i in esc),
        "flips": {k: {"n": len(v), "items": sorted(v)} for k, v in flips.items()},
        "ai_picked_surface_answer": {"n": len(distractor), "of_escalated_with_surface": sum(1 for i in esc if surface.get(i)),
                                     "items": sorted(distractor)},
    }
    same = sum(r1[t.id]["predicted"] == r2d[t.id]["predicted"] for t in tasks)
    report["determinism_service_vs_cascade_decider"] = {"same_predicted": same, "of": len(tasks)}
    (RUNS / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: report[k] for k in ("cascade_changes", "determinism_service_vs_cascade_decider")}, indent=1))
    for name in ("service", "cascade"):
        print(name, "accuracy", round(report[name]["accuracy"], 4), "by_tier",
              {k: f'{v["correct"]}/{v["n"]}' for k, v in report[name]["by_tier"].items()})


if __name__ == "__main__":
    main()

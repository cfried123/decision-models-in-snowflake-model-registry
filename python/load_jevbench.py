"""Load the JevBench public items into <database>.<schema>.JEVBENCH_ITEMS.

Reads the three public files from the pinned JevBench checkout, checks each
against datasets/manifest.json, and stores one row per decision with:
  * STATE_JSON / QUESTIONS_JSON: the exact request JevBench's `typesafe`
    adapter sends, built with its own build_question(). Stored as text so key
    order (which fixes option order) survives.

    SNOWFLAKE_CONNECTION_NAME=<connection> .venv/bin/python python/load_jevbench.py
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JEVBENCH = ROOT / ".cache" / "jevbench"
PINNED = "bb05a335bc809e61b20c0f745d25499a82b326fc"
SPLITS = [("easy", "easy"), ("original", "standard"), ("hard", "hard")]   # (file, tier)
EXPECTED_COUNTS = {("easy", "choice"): 36, ("easy", "noul"): 12,
                   ("standard", "choice"): 36, ("standard", "noul"): 24, ("standard", "score"): 12,
                   ("hard", "choice"): 67, ("hard", "noul"): 38, ("hard", "score"): 6}

sys.path.insert(0, str(JEVBENCH))
sys.path.insert(0, str(ROOT / "python"))
from jevbench.adapters.base import build_question  # noqa: E402
from jevbench.tasks import Task  # noqa: E402


def main():
    manifest = json.loads((JEVBENCH / "datasets" / "manifest.json").read_text())
    want = {s["name"]: s["sha256"] for s in manifest["splits"]}
    assert (JEVBENCH / ".pinned_commit").read_text().strip() == PINNED
    rows = []
    for fname, tier in SPLITS:
        path = JEVBENCH / "datasets" / "public" / f"{fname}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == want[fname], f"{fname}: sha256 {digest} != manifest {want[fname]}"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            raw = json.loads(line)
            task = Task.from_dict(raw)                                   # JevBench's own validation
            rows.append({
                "ITEM_ID": task.id, "TIER": tier, "FAMILY": task.family, "QTYPE": task.question["type"],
                "N_OPTIONS": len(task.labels), "LABELS": task.labels, "EXPECTED": task.expected,
                "GRP": task.group, "STATE_IS_OBJECT": not isinstance(task.state, str),
                "STATE_JSON": json.dumps(task.state, ensure_ascii=False),
                "QUESTIONS_JSON": json.dumps({"decision": build_question(task)}, ensure_ascii=False),
                "SURFACE_ANSWER": task.provenance.get("surface_answer"),
                "PROVENANCE": task.provenance, "ITEM_JSON": line,
                "SOURCE_FILE": f"datasets/public/{fname}.jsonl", "SOURCE_SHA256": digest,
                "SOURCE_COMMIT": PINNED,
            })
    counts = Counter((r["TIER"], r["QTYPE"]) for r in rows)
    assert dict(counts) == EXPECTED_COUNTS, counts
    assert len({r["ITEM_ID"] for r in rows}) == len(rows) == 231
    print(f"{len(rows)} items verified")

    out = ROOT / "runs" / "jevbench_items.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    from bench_config import FQ_SCHEMA
    from snowpark_session import create_snowpark_session
    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    session.file.put(str(out), "@DECIDER_BENCH_STAGE/jevbench/", auto_compress=True, overwrite=True)
    session.sql("""
        CREATE OR REPLACE TABLE JEVBENCH_ITEMS
          COMMENT = 'JevBench public items (MIT), fstandhartinger/jevbench@bb05a335'
        AS SELECT
          $1:ITEM_ID::STRING AS ITEM_ID, $1:TIER::STRING AS TIER, $1:FAMILY::STRING AS FAMILY,
          $1:QTYPE::STRING AS QTYPE, $1:N_OPTIONS::NUMBER AS N_OPTIONS, $1:LABELS::ARRAY AS LABELS,
          $1:EXPECTED AS EXPECTED, $1:GRP::STRING AS GRP, $1:STATE_IS_OBJECT::BOOLEAN AS STATE_IS_OBJECT,
          $1:STATE_JSON::STRING AS STATE_JSON, $1:QUESTIONS_JSON::STRING AS QUESTIONS_JSON,
          $1:SURFACE_ANSWER::STRING AS SURFACE_ANSWER, $1:PROVENANCE AS PROVENANCE,
          $1:ITEM_JSON::STRING AS ITEM_JSON, $1:SOURCE_FILE::STRING AS SOURCE_FILE,
          $1:SOURCE_SHA256::STRING AS SOURCE_SHA256, $1:SOURCE_COMMIT::STRING AS SOURCE_COMMIT
        FROM @DECIDER_BENCH_STAGE/jevbench/jevbench_items.jsonl.gz (FILE_FORMAT => 'JSONL_FF')
    """).collect()
    summary = session.sql("""
        SELECT TIER, QTYPE, COUNT(*) AS N FROM JEVBENCH_ITEMS GROUP BY 1, 2 ORDER BY 1, 2
    """).collect()
    for r in summary:
        print(r["TIER"], r["QTYPE"], r["N"])
    # The request text must come back byte-identical from Snowflake.
    back = {r["ITEM_ID"]: (r["STATE_JSON"], r["QUESTIONS_JSON"])
            for r in session.sql("SELECT ITEM_ID, STATE_JSON, QUESTIONS_JSON FROM JEVBENCH_ITEMS").collect()}
    mismatch = [r["ITEM_ID"] for r in rows if back[r["ITEM_ID"]] != (r["STATE_JSON"], r["QUESTIONS_JSON"])]
    assert not mismatch, mismatch
    print("round-trip check: STATE_JSON and QUESTIONS_JSON identical for all", len(back), "items")
    session.close()


if __name__ == "__main__":
    main()

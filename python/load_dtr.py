"""Load DTR-Bench into <database>.<schema>.DTR_ITEMS.

Reads dtr_bench.jsonl (written by `python -m strands_decider.data.dtr` in the
strands-decider repo), checks it against the pinned SHA-256, and stores one row per
decision with:
  * STATE_JSON / QUESTIONS_JSON: the exact system_one request. Stored as text so key
    order (which fixes option order) survives.
  * EXPECTED: an option name (choice/noul), a level index (score) or 'abstain' for
    items whose decisive fact is missing and should go to the human-review queue.

    SNOWFLAKE_CONNECTION_NAME=<connection> DTR_BENCH=<path>/dtr_bench.jsonl \
    .venv/bin/python python/load_dtr.py
"""

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

BENCH = Path(os.environ.get(
    "DTR_BENCH", ROOT.parent / "strands-decider" / "data" / "dtr_bench.jsonl"))
EXPECTED_SHA256 = os.environ.get("DTR_BENCH_SHA256")  # optional pin
EXPECTED_ROWS = 142


def rows_from(path: Path) -> tuple[list[dict], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if EXPECTED_SHA256:
        assert digest == EXPECTED_SHA256, f"sha256 {digest} != pinned {EXPECTED_SHA256}"
    rows = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        assert it["question"]["type"] == it["kind"], it["item_id"]
        rows.append({
            "ITEM_ID": it["item_id"], "TASK": it["task"], "QTYPE": it["kind"],
            "SPLIT": it["split"], "DTR_CITATION": it["dtr_citation"],
            "SCENARIO": it["state"]["scenario"], "EXPECTED": it["expected"],
            "OPTION_NAMES": it["option_names"], "REVIEW_STATUS": it["review"]["status"],
            "SME_REVIEW": it["review"]["sme_review"],
            "STATE_JSON": json.dumps(it["state"], ensure_ascii=False),
            "QUESTIONS_JSON": json.dumps({"decision": it["question"]}, ensure_ascii=False),
            "ITEM_JSON": line, "SOURCE_SHA256": digest,
        })
    return rows, digest


def main():
    rows, digest = rows_from(BENCH)
    assert len({r["ITEM_ID"] for r in rows}) == len(rows) == EXPECTED_ROWS, len(rows)
    print(f"{len(rows)} items verified (sha256 {digest[:12]})")
    print(dict(Counter((r["SPLIT"], r["QTYPE"]) for r in rows)))
    out = ROOT / "runs" / "dtr_items.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")

    from bench_config import FQ_SCHEMA
    from snowpark_session import create_snowpark_session

    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    session.file.put(str(out), "@DECIDER_BENCH_STAGE/dtr/", auto_compress=True, overwrite=True)
    session.sql("""
        CREATE OR REPLACE TABLE DTR_ITEMS
          COMMENT = 'DTR-Bench: DTR Part I passenger-movement decisions (self-reviewed; SME review pending)'
        AS SELECT
          $1:ITEM_ID::STRING AS ITEM_ID, $1:TASK::STRING AS TASK, $1:QTYPE::STRING AS QTYPE,
          $1:SPLIT::STRING AS SPLIT, $1:DTR_CITATION::STRING AS DTR_CITATION,
          $1:SCENARIO::STRING AS SCENARIO, $1:EXPECTED::STRING AS EXPECTED,
          $1:OPTION_NAMES::ARRAY AS OPTION_NAMES, $1:REVIEW_STATUS::STRING AS REVIEW_STATUS,
          $1:SME_REVIEW::STRING AS SME_REVIEW, $1:STATE_JSON::STRING AS STATE_JSON,
          $1:QUESTIONS_JSON::STRING AS QUESTIONS_JSON, $1:ITEM_JSON::STRING AS ITEM_JSON,
          $1:SOURCE_SHA256::STRING AS SOURCE_SHA256
        FROM @DECIDER_BENCH_STAGE/dtr/dtr_items.jsonl.gz (FILE_FORMAT => 'JSONL_FF')
    """).collect()
    for r in session.sql(
        "SELECT SPLIT, QTYPE, COUNT(*) AS N FROM DTR_ITEMS GROUP BY 1, 2 ORDER BY 1, 2"
    ).collect():
        print(r["SPLIT"], r["QTYPE"], r["N"])
    # The request text must come back byte-identical from Snowflake.
    back = {r["ITEM_ID"]: (r["STATE_JSON"], r["QUESTIONS_JSON"])
            for r in session.sql("SELECT ITEM_ID, STATE_JSON, QUESTIONS_JSON FROM DTR_ITEMS")
            .collect()}
    bad = [r["ITEM_ID"] for r in rows
           if back.get(r["ITEM_ID"]) != (r["STATE_JSON"], r["QUESTIONS_JSON"])]
    assert not bad, f"{len(bad)} rows changed on the round trip: {bad[:5]}"
    print("round trip ok")
    session.close()


if __name__ == "__main__":
    main()

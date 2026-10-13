"""Load DTR-Bench scores and the model->service map that the DTR Model Explorer reads.

    SNOWFLAKE_CONNECTION_NAME=<connection> python python/laya_dtr/load_scores.py <label>=<results.json> ...

Each results file is a dtr_eval_laya.py / bench_llm_dtr.py output with a "metrics" object
keyed by split. Endpoints come from python/laya_dtr/endpoints.json.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from bench_config import FQ_SCHEMA  # noqa: E402
from snowpark_session import create_snowpark_session  # noqa: E402

COLS = ("strict_accuracy", "forced_accuracy", "coverage", "selective_accuracy", "abstain_recall", "ece")


def main(pairs: list[str]) -> None:
    session = create_snowpark_session()
    session.use_schema(FQ_SCHEMA)
    eps = json.loads((Path(__file__).resolve().parent / "endpoints.json").read_text())
    session.sql("DELETE FROM MODEL_ENDPOINTS").collect()
    for i, e in enumerate(eps):
        session.sql("INSERT INTO MODEL_ENDPOINTS SELECT ?, ?, ?, ?, ?, ?, ?, ?, TRUE",
                    params=[e["label"], e["family"], e["model"], e["version"], e["service"], e["pool"],
                            e["usd_per_hour"], i]).collect()
    for pair in pairs:
        label, path = pair.split("=", 1)
        metrics = json.loads(Path(path).read_text())["metrics"]
        session.sql("DELETE FROM DTR_MODEL_SCORES WHERE LABEL = ?", params=[label]).collect()
        for split, m in metrics.items():
            if split not in ("all", "in_domain", "holdout"):
                continue
            session.sql("INSERT INTO DTR_MODEL_SCORES SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP()",
                        params=[label, split, m["n"], *[m.get(c) for c in COLS], Path(path).name]).collect()
        print(f"loaded {label} from {path}")
    session.close()


if __name__ == "__main__":
    main(sys.argv[1:])

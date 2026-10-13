"""DTR Model Explorer: a Streamlit in Snowflake app over the Model Registry.

Ask one registered model version a DTR Part I travel question, or compare several versions
(strands-decider v21 / DTR fine-tune, Laya base / DTR fine-tune) side by side with a Cortex
LLM (Claude via AI_COMPLETE). Every call stays inside Snowflake: models are reached through
their SPCS service functions (<service>!SYSTEM_ONE) and the LLM through AI_COMPLETE.

Uncertain answers are never issued: they go to DTR_REVIEW_QUEUE for a human adjudicator.
"""
import hashlib
import json
import re
import time

import pandas as pd
import streamlit as st
from snowflake.snowpark.context import get_active_session

SCHEMA = "DECIDER_BENCH.BENCH"
NOUL_BAND, CHOICE_FLOOR = 0.25, 0.5  # same policy as strands_decider.dtr_eval
LLM_MODELS = ["claude-sonnet-4-5", "claude-4-sonnet", "claude-3-7-sonnet", "llama3.1-70b"]
IDENT = re.compile(r"^[A-Z][A-Z0-9_]{0,254}$")  # Security: service names are whitelisted identifiers
MAX_SCENARIO = 4000  # Security: bounds request size (DoS) before it reaches a service function
SYSTEM = ("You adjudicate official travel under the Defense Transportation Regulation (DTR) Part I. "
          "Decide strictly from the scenario and the DTR excerpt given. If a fact the excerpt makes "
          "decisive is missing, say so through a probability near 0.5.")

st.set_page_config(page_title="DTR Model Explorer", layout="wide")
session = get_active_session()


@st.cache_data(ttl=300)
def endpoints() -> pd.DataFrame:
    df = session.sql(f"SELECT * FROM {SCHEMA}.MODEL_ENDPOINTS WHERE ENABLED ORDER BY SORT_ORDER").to_pandas()
    return df[df["SERVICE_NAME"].str.fullmatch(IDENT.pattern)]


@st.cache_data(ttl=60)
def service_status() -> dict:
    rows = session.sql(f"SHOW SERVICES IN SCHEMA {SCHEMA}").collect()
    return {r["name"]: r["status"] for r in rows}


@st.cache_data(ttl=300)
def versions(model: str) -> pd.DataFrame:
    if not IDENT.fullmatch(model):
        return pd.DataFrame()
    session.sql(f"SHOW VERSIONS IN MODEL {SCHEMA}.{model}").collect()
    return session.sql('SELECT "name", "aliases", "comment", "created_on" FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))').to_pandas()


@st.cache_data(ttl=600)
def items() -> pd.DataFrame:
    return session.sql(f"SELECT ITEM_ID, TASK, QTYPE, SPLIT, DTR_CITATION, SCENARIO, EXPECTED, STATE_JSON, "
                       f"QUESTIONS_JSON FROM {SCHEMA}.DTR_ITEMS ORDER BY ITEM_ID").to_pandas()


def decide(ans: dict):
    """(decision, top, confidence); decision is 'abstain' inside the review band."""
    if ans.get("type") == "noul":
        p = float(ans["noul"])
        top, conf = ("true" if p >= 0.5 else "false"), max(p, 1 - p)
        return ("abstain" if abs(p - 0.5) < NOUL_BAND else top), top, conf
    probs = ans["probabilities"]
    top = max(probs, key=probs.get)
    conf = float(probs[top])
    return ("abstain" if conf < CHOICE_FLOOR else top), top, conf


def ask_model(service: str, state: dict, question: dict) -> dict:
    if not IDENT.fullmatch(service):
        return {"error": "service name rejected"}
    t0 = time.perf_counter()
    try:  # Security: scenario text only ever travels as a bind variable
        r = session.sql(f"SELECT {SCHEMA}.{service}!SYSTEM_ONE(?, ?) AS R",
                        params=[json.dumps(state), json.dumps({"d": question})]).collect()[0]["R"]
        r = json.loads(r) if isinstance(r, str) else r
        res = json.loads(r["ANSWER_JSON"])
        if "error" in res:
            return {"error": res["error"], "client_ms": (time.perf_counter() - t0) * 1000}
        return {"answer": res["answers"]["d"], "model_ms": float(r["ELAPSED_MS"]),
                "client_ms": (time.perf_counter() - t0) * 1000, "runtime": json.loads(r["RUNTIME_JSON"])}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:300]}", "client_ms": (time.perf_counter() - t0) * 1000}


def ask_llm(model: str, state: dict, q: dict) -> dict:
    head = (f"{SYSTEM}\n\nScenario: {state['scenario']}\n\nDTR excerpt ({state['dtr_citation']}):\n"
            f"{state['dtr_excerpt']}\n\nQuestion: {q['instructions']}\n")
    if q["type"] == "noul":
        text = head + "\nGive probability_true: your probability (0 to 1) that the answer is yes/true."
        schema = {"type": "object", "properties": {"probability_true": {"type": "number"}},
                  "required": ["probability_true"], "additionalProperties": False}
    else:
        opts = ([(str(i), d) for i, d in enumerate(q["criteria"])] if q["type"] == "score"
                else list(q["criteria"].items()))
        text = head + "\nOptions:\n" + "".join(f"- {k}: {d}\n" for k, d in opts) + \
            "\nGive answer (one option key) and confidence: your probability (0 to 1) that it is correct."
        schema = {"type": "object", "additionalProperties": False, "required": ["answer", "confidence"],
                  "properties": {"answer": {"type": "string", "enum": [k for k, _ in opts]},
                                 "confidence": {"type": "number"}}}
    t0 = time.perf_counter()
    try:
        raw = session.sql("SELECT AI_COMPLETE(model => ?, prompt => ?, response_format => PARSE_JSON(?), "
                          "show_details => TRUE) AS R",
                          params=[model, text, json.dumps({"type": "json", "schema": schema})]).collect()[0]["R"]
        r = json.loads(raw) if isinstance(raw, str) else raw
        out = r["structured_output"][0]["raw_message"]
        out = json.loads(out) if isinstance(out, str) else out
        ans = ({"type": "noul", "noul": min(1.0, max(0.0, float(out["probability_true"])))} if q["type"] == "noul"
               else {"type": q["type"], "probabilities": {str(out["answer"]): float(out["confidence"])}})
        return {"answer": ans, "client_ms": (time.perf_counter() - t0) * 1000, "usage": r.get("usage", {})}
    except Exception as e:  # trial accounts: "AI function ... is not available for trial accounts"
        return {"error": f"{type(e).__name__}: {str(e)[:300]}", "client_ms": (time.perf_counter() - t0) * 1000}


def queue_for_review(task, side, scenario, citation, top, conf):
    session.sql(f"INSERT INTO {SCHEMA}.DTR_REVIEW_QUEUE (REQUEST_ID, TASK, SIDE, SCENARIO, DTR_CITATION, MODEL_TOP, "
                "MODEL_CONFIDENCE, CREATED_AT) SELECT UUID_STRING(), ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP()",
                params=[task, side, scenario, citation, top, conf]).collect()


def render(col, label: str, res: dict, usd_per_hour, task, scenario, citation, key):
    with col:
        st.markdown(f"**{label}**")
        if "error" in res:
            st.error(res["error"])
            return
        d, top, conf = decide(res["answer"])
        if d == "abstain":
            st.warning(f"Abstained → human review (top: `{top}`, {conf:.0%})")
            done = f"queued-{key}"
            if st.session_state.get(done):
                st.success("Queued in DTR_REVIEW_QUEUE")
            elif st.button("Send to review queue", key=f"q-{key}"):
                queue_for_review(task, label, scenario, citation, top, conf)
                st.session_state[done] = True
                st.success("Queued in DTR_REVIEW_QUEUE")
        else:
            st.success(f"`{d}` · confidence {conf:.1%}")
        a = res["answer"]
        probs = ({"false": 1 - a["noul"], "true": a["noul"]} if a["type"] == "noul" else a["probabilities"])
        for name, pr in sorted(probs.items(), key=lambda kv: -kv[1]):
            st.caption(f"`{name}` {pr:.1%}")
            st.progress(min(max(float(pr), 0.0), 1.0))
        ms = res.get("model_ms")
        line = f"Cites {citation} · round trip {res['client_ms']:.0f} ms"
        if ms is not None:
            line += f" · model {ms:.0f} ms"
            if usd_per_hour:
                line += f" · ≈${ms / 3.6e6 * usd_per_hour * 1000:.4f} / 1K decisions (list price)"
        st.caption(line)


st.title("DTR Model Explorer")
st.caption("Model Registry versions on Snowpark Container Services, compared against Cortex AI_COMPLETE. "
           "DTR-Bench labels are self-reviewed (SME review pending).")
eps, status, its = endpoints(), service_status(), items()
eps["STATUS"] = eps["SERVICE_NAME"].map(status).fillna("MISSING")
labels = dict(zip(eps["LABEL"], eps.to_dict("records")))

ask_tab, cmp_tab, board_tab, review_tab, reg_tab = st.tabs(
    ["Ask one version", "Compare", "DTR-Bench leaderboard", "Review queue", "Registry & cost"])


def question_picker(prefix: str):
    task = st.selectbox("Determination", sorted(its["TASK"].unique()), key=f"{prefix}-task")
    sub = its[its["TASK"] == task]
    sample = st.selectbox("DTR-Bench sample (or write your own below)", ["—"] + sub["ITEM_ID"].tolist(), key=f"{prefix}-s")
    row = sub[sub["ITEM_ID"] == sample].iloc[0] if sample != "—" else sub.iloc[0]
    cites = sub.drop_duplicates("DTR_CITATION")
    citation = st.selectbox("Governing provision", cites["DTR_CITATION"].tolist(),
                            index=cites["DTR_CITATION"].tolist().index(row["DTR_CITATION"]), key=f"{prefix}-c-{sample}")
    excerpt = json.loads(cites[cites["DTR_CITATION"] == citation].iloc[0]["STATE_JSON"])["dtr_excerpt"]
    scenario = st.text_area("Travel scenario", row["SCENARIO"] if sample != "—" else "", height=120,
                            max_chars=MAX_SCENARIO, key=f"{prefix}-t-{sample}")
    question = json.loads(row["QUESTIONS_JSON"])
    question = question.get("d") or next(iter(question.values()))
    with st.expander("DTR excerpt and question sent to every model"):
        st.write(excerpt)
        st.json(question)
    if sample != "—":
        st.caption(f"Expected (self-reviewed): `{row['EXPECTED']}` · split: {row['SPLIT']}")
    return task, {"scenario": scenario.strip(), "dtr_citation": citation, "dtr_excerpt": excerpt}, question


def run(selected: list, llm: str | None, prefix: str):
    task, state, q = question_picker(prefix)
    sig = hashlib.sha1(json.dumps([selected, llm, state, q], sort_keys=True).encode()).hexdigest()[:12]
    if st.button("Decide", type="primary", key=f"{prefix}-go", disabled=not state["scenario"]):
        results = []
        for lab in selected:
            ep = labels[lab]
            if ep["STATUS"] != "RUNNING":
                results.append((lab, {"offline": f"{lab}: service {ep['SERVICE_NAME']} is {ep['STATUS']}. "
                                                 "Resume it on the Registry tab."}, None))
            else:
                results.append((lab, ask_model(ep["SERVICE_NAME"], state, q), ep["USD_PER_HOUR"]))
        if llm:
            results.append((f"{llm} (AI_COMPLETE)", ask_llm(llm, state, q), None))
        st.session_state[f"{prefix}-res"] = {"sig": sig, "run": time.time_ns(), "results": results}
    saved = st.session_state.get(f"{prefix}-res")
    if not saved or saved["sig"] != sig:  # inputs changed since the last Decide
        return
    cols = st.columns(len(saved["results"]) or 1)
    for i, (lab, res, usd) in enumerate(saved["results"]):
        if "offline" in res:
            cols[i].info(res["offline"])
            continue
        render(cols[i], lab, res, usd, task, state["scenario"], state["dtr_citation"], f"{prefix}{i}-{saved['run']}")


with ask_tab:
    lab = st.selectbox("Model version", list(labels), key="one")
    st.caption(f"{labels[lab]['MODEL_NAME']} / {labels[lab]['VERSION_NAME']} → service "
               f"{labels[lab]['SERVICE_NAME']} ({labels[lab]['STATUS']}) on {labels[lab]['COMPUTE_POOL']}")
    run([lab], None, "one")

with cmp_tab:
    picked = st.multiselect("Versions to compare", list(labels), default=list(labels)[:4])
    llm = st.selectbox("Closed-source comparison (Cortex)", ["none"] + LLM_MODELS, index=1)
    run(picked, None if llm == "none" else llm, "cmp")

with board_tab:
    try:
        board = session.sql(f"SELECT * FROM {SCHEMA}.DTR_MODEL_SCORES ORDER BY SPLIT, STRICT_ACCURACY DESC").to_pandas()
        split = st.radio("Split", sorted(board["SPLIT"].unique()), horizontal=True)
        view = board[board["SPLIT"] == split]
        st.dataframe(view, use_container_width=True)
        st.bar_chart(view.set_index("LABEL")[["STRICT_ACCURACY", "SELECTIVE_ACCURACY", "COVERAGE"]])
    except Exception as e:
        st.info(f"No scores loaded yet ({str(e)[:120]}).")

with review_tab:
    q = session.sql(f"SELECT REVIEW_ID, CREATED_AT, TASK, SIDE, SCENARIO, DTR_CITATION, MODEL_TOP, MODEL_CONFIDENCE "
                    f"FROM {SCHEMA}.DTR_REVIEW_QUEUE WHERE RESOLUTION IS NULL ORDER BY CREATED_AT DESC LIMIT 50").to_pandas()
    st.dataframe(q, use_container_width=True)
    if len(q):
        rid = st.selectbox("Resolve item", q["REVIEW_ID"].tolist())
        res = st.text_input("Determination", max_chars=64)
        note = st.text_input("Reviewer note", max_chars=500)
        if st.button("Resolve", disabled=not res.strip()):
            session.sql(f"UPDATE {SCHEMA}.DTR_REVIEW_QUEUE SET RESOLUTION = ?, REVIEWER_NOTE = ?, "
                        "RESOLVED_BY = CURRENT_USER(), RESOLVED_AT = CURRENT_TIMESTAMP() WHERE REVIEW_ID = ?",
                        params=[res.strip(), note.strip(), int(rid)]).collect()
            st.success("Resolved")

with reg_tab:
    st.dataframe(eps[["LABEL", "MODEL_NAME", "VERSION_NAME", "SERVICE_NAME", "COMPUTE_POOL", "STATUS", "USD_PER_HOUR"]],
                 use_container_width=True)
    for m in eps["MODEL_NAME"].unique():
        st.markdown(f"**{m}** versions in the Model Registry")
        st.dataframe(versions(m), use_container_width=True)
    svc = st.selectbox("Service", eps["SERVICE_NAME"].unique().tolist())
    c1, c2 = st.columns(2)
    if c1.button("Resume (starts billing)"):
        session.sql(f"ALTER SERVICE {SCHEMA}.{svc} RESUME").collect()
        service_status.clear()
        st.info("Resuming; a GPU node and model load take several minutes.")
    if c2.button("Suspend"):
        session.sql(f"ALTER SERVICE {SCHEMA}.{svc} SUSPEND").collect()
        service_status.clear()

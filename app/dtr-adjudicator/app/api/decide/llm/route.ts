/**
 * POST /api/decide/llm {requestId, taskId, scenario} — the same determination from a
 * frontier LLM through AI_COMPLETE with a JSON schema for the answer. The prompt (with
 * the scenario), model and format go in as bind variables; show_details returns the
 * token usage the cost counter prices.
 */

import { querySnowflake } from "@/lib/snowflake"
import { LLM_MODEL, LLM_WAREHOUSE } from "@/lib/config"
import { clamp01, cleanScenario, decide, isTaskId, llmRequest, TASKS, type Determination } from "@/lib/dtr"

export const dynamic = "force-dynamic"

const SQL = `SELECT AI_COMPLETE(model => ?, prompt => ?, response_format => PARSE_JSON(?), show_details => TRUE) AS R`

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as { requestId?: unknown; taskId?: unknown; scenario?: unknown }
  const requestId = String(body.requestId ?? "").slice(0, 64)
  const scenario = cleanScenario(body.scenario)
  if (!isTaskId(body.taskId) || !scenario) {
    return Response.json({ requestId, error: "taskId and scenario are required" }, { status: 400 })
  }
  const taskId = body.taskId
  const task = TASKS[taskId]
  const { prompt, format } = llmRequest(taskId, scenario)
  try {
    const t0 = performance.now()
    const rows = await querySnowflake(SQL, { binds: [LLM_MODEL, prompt, JSON.stringify(format)], warehouse: LLM_WAREHOUSE })
    const modelMs = performance.now() - t0
    const raw = rows[0]?.R
    const r = typeof raw === "string" ? JSON.parse(raw) : raw
    const msg = r.structured_output[0].raw_message
    const a = typeof msg === "string" ? JSON.parse(msg) : msg
    let distribution: Record<string, number>
    if (task.kind === "noul") {
      const p = clamp01(a.probability_true)
      distribution = { true: p, false: 1 - p }
    } else {
      const key = String(a.answer)
      if (!task.option_names.includes(key) && !(task.kind === "score" && /^\d+$/.test(key))) {
        throw new Error("LLM answered outside the option set")
      }
      distribution = { [key]: clamp01(a.confidence) }
    }
    const d = decide(task.kind, distribution)
    const result: Determination = {
      requestId,
      taskId,
      kind: task.kind,
      decision: d.decision,
      top: d.top,
      confidence: d.confidence,
      distribution,
      citation: task.dtr_citation,
      needsReview: d.decision === "abstain",
      modelMs,
      inputTokens: Number(r.usage?.prompt_tokens ?? 0),
      outputTokens: Number(r.usage?.completion_tokens ?? 0),
    }
    return Response.json(result)
  } catch (e) {
    console.error(new Date().toISOString(), "AI_COMPLETE call failed", e)
    return Response.json({ requestId, error: "AI_COMPLETE call failed" }, { status: 502 })
  }
}

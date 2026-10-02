/**
 * POST /api/decide/llm {ticketId, text} — the same ticket and the same two questions
 * to a frontier LLM through AI_COMPLETE, with a JSON schema for the answer. Ticket
 * text and model go in as bind variables; show_details returns the token usage that
 * the cost counter prices.
 */

import { querySnowflake } from "@/lib/snowflake"
import { LLM_MODEL, LLM_WAREHOUSE } from "@/lib/config"
import { llmPrompt, LLM_RESPONSE_FORMAT, CATEGORY_KEYS, type Category, type Decision } from "@/lib/questions"

export const dynamic = "force-dynamic"

const FORMAT_JSON = JSON.stringify(LLM_RESPONSE_FORMAT)
const SQL = `SELECT AI_COMPLETE(model => ?, prompt => ?, response_format => PARSE_JSON(?), show_details => TRUE) AS R`

export async function POST(req: Request) {
  const { ticketId, text } = (await req.json()) as { ticketId: number; text: string }
  try {
    const t0 = performance.now()
    const rows = await querySnowflake(SQL, {
      binds: [LLM_MODEL, llmPrompt(String(text).slice(0, 4000)), FORMAT_JSON],
      warehouse: LLM_WAREHOUSE,
    })
    const modelMs = performance.now() - t0
    const raw = rows[0]?.R
    const r = typeof raw === "string" ? JSON.parse(raw) : raw
    const a = r.structured_output[0].raw_message
    const category = (CATEGORY_KEYS as string[]).includes(a.category) ? (a.category as Category) : "bug"
    const escConf = Math.min(1, Math.max(0, Number(a.escalate_confidence ?? 0)))
    const escalate = a.escalate === "escalate"
    const decision: Decision = {
      ticketId,
      category,
      categoryConfidence: Math.min(1, Math.max(0, Number(a.category_confidence ?? 0))),
      escalate,
      // The LLM states confidence in its own pick; turn it into P(escalate) like decider's.
      escalateProb: escalate ? escConf : 1 - escConf,
      modelMs,
      inputTokens: Number(r.usage?.prompt_tokens ?? 0),
      outputTokens: Number(r.usage?.completion_tokens ?? 0),
    }
    return Response.json(decision)
  } catch (e) {
    console.error(new Date().toISOString(), "AI_COMPLETE call failed", e)
    return Response.json({ ticketId, error: e instanceof Error ? e.message : "AI_COMPLETE call failed" }, { status: 502 })
  }
}

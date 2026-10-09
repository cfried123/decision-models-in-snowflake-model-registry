/**
 * DTR determinations both sides make. The catalog (lib/dtr_catalog.json) is written by
 * `python -m strands_decider.data.dtr --catalog`, so the question, options and governing
 * DTR excerpt are exactly what the decider was fine-tuned and benchmarked on.
 *
 * The review policy mirrors strands_decider.dtr_eval.decide: a yes/no determination goes
 * to human review when P(authorized) is within NOUL_BAND of 0.5; a choice or score goes
 * to review when its top probability is below CHOICE_FLOOR. Both sides use it, so the
 * queue compares calibration rather than prompt wording.
 */

import catalog from "./dtr_catalog.json"

export const NOUL_BAND = 0.25
export const CHOICE_FLOOR = 0.5
export const MAX_SCENARIO_CHARS = 2000

export type Kind = "choice" | "noul" | "score"
export interface TaskSpec {
  kind: Kind
  section: string
  dtr_citation: string
  dtr_excerpt: string
  question: { type: Kind; instructions: string; criteria: Record<string, string> | string[] }
  option_names: string[]
  holdout: boolean
}

export const TASKS = catalog.tasks as unknown as Record<string, TaskSpec>
export const TASK_IDS = Object.keys(TASKS)

export const TASK_LABELS: Record<string, string> = {
  dtr_class_of_service: "Class of service",
  dtr_travel_mode: "Mode of transportation",
  dtr_fare_eligibility: "City Pair fare eligibility",
  dtr_cpp_exception: "City Pair exception",
  dtr_foreign_carrier: "Foreign flag carrier",
  dtr_contractor_airlift: "Contractor on DoD airlift",
  dtr_booking_policy: "Booking / TMC policy",
  dtr_rental_use: "Rental car use",
  dtr_ship_travel: "Commercial ship travel",
  dtr_compliance_risk: "Compliance risk",
}

export function isTaskId(t: unknown): t is string {
  return typeof t === "string" && Object.prototype.hasOwnProperty.call(TASKS, t)
}

/** Validates and trims an untrusted scenario; null when unusable. */
export function cleanScenario(s: unknown): string | null {
  if (typeof s !== "string") return null
  const t = s.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g, "").trim()
  return t ? t.slice(0, MAX_SCENARIO_CHARS) : null
}

/** STATE_JSON for system_one: scenario plus the governing excerpt, as in training. */
export function stateFor(taskId: string, scenario: string) {
  const t = TASKS[taskId]
  return { scenario, dtr_citation: t.dtr_citation, dtr_excerpt: t.dtr_excerpt }
}

/** Option key -> human description, in the order the decider sees them. */
export function optionsOf(taskId: string): [string, string][] {
  const c = TASKS[taskId].question.criteria
  return Array.isArray(c) ? c.map((d, i) => [String(i), d]) : Object.entries(c)
}

export function optionLabel(taskId: string, key: string): string {
  return optionsOf(taskId).find(([k]) => k === key)?.[1] ?? key
}

/** decision ("abstain" routes to review), top option, and its probability. */
export function decide(kind: Kind, dist: Record<string, number>): { decision: string; top: string; confidence: number } {
  if (kind === "noul") {
    const p = Number(dist.true ?? 0)
    const top = p >= 0.5 ? "true" : "false"
    return { decision: Math.abs(p - 0.5) < NOUL_BAND ? "abstain" : top, top, confidence: Math.max(p, 1 - p) }
  }
  const entries = Object.entries(dist)
  if (!entries.length) return { decision: "abstain", top: "", confidence: 0 }
  const [top, conf] = entries.reduce((a, b) => (b[1] > a[1] ? b : a))
  return { decision: conf < CHOICE_FLOOR ? "abstain" : top, top, confidence: conf }
}

const SYSTEM =
  "You adjudicate official travel under the Defense Transportation Regulation (DTR) Part I. " +
  "Decide strictly from the scenario and the DTR excerpt given. If a fact the excerpt makes " +
  "decisive is missing, say so through a probability near 0.5."

/** Prompt and AI_COMPLETE response_format; the same as python/bench_llm_dtr.py. */
export function llmRequest(taskId: string, scenario: string): { prompt: string; format: object } {
  const t = TASKS[taskId]
  const crit = t.question.criteria as Record<string, string>
  const head =
    `${SYSTEM}\n\nScenario: ${scenario}\n\nDTR excerpt (${t.dtr_citation}):\n${t.dtr_excerpt}\n\n` +
    `Question: ${t.question.instructions}\n`
  if (t.kind === "noul") {
    return {
      prompt:
        head +
        `\nGive probability_true: your probability (0 to 1) that the answer is '${crit.true}' rather than '${crit.false}'.`,
      format: {
        type: "json",
        schema: {
          type: "object",
          additionalProperties: false,
          properties: { probability_true: { type: "number" } },
          required: ["probability_true"],
        },
      },
    }
  }
  const opts = optionsOf(taskId)
  return {
    prompt:
      head +
      "\nOptions:\n" +
      opts.map(([k, d]) => `- ${k}: ${d}\n`).join("") +
      "\nGive answer (one option key) and confidence: your probability (0 to 1) that it is correct.",
    format: {
      type: "json",
      schema: {
        type: "object",
        additionalProperties: false,
        properties: { answer: { type: "string", enum: opts.map(([k]) => k) }, confidence: { type: "number" } },
        required: ["answer", "confidence"],
      },
    },
  }
}

/** What both decide routes return for one scenario. */
export interface Determination {
  requestId: string
  taskId: string
  kind: Kind
  /** Option key, or "abstain" when the answer goes to human review. */
  decision: string
  /** The model's top option even when it abstains. */
  top: string
  confidence: number
  /** P(option) (decider), or the stated confidence in the pick / P(true) (LLM). */
  distribution: Record<string, number>
  citation: string
  needsReview: boolean
  modelMs: number
  inputTokens?: number
  outputTokens?: number
  error?: string
}

export const clamp01 = (x: unknown) => Math.min(1, Math.max(0, Number(x) || 0))

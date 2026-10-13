/** GET /api/config — model names, list prices and the DTR question catalog for the UI. */

import { LLM_MODEL, PRICES } from "@/lib/config"
import { CHOICE_FLOOR, NOUL_BAND, TASK_IDS, TASK_LABELS, TASKS } from "@/lib/dtr"

export const dynamic = "force-dynamic"

export async function GET() {
  const tasks = TASK_IDS.map((id) => ({
    id, label: TASK_LABELS[id] ?? id, kind: TASKS[id].kind, citation: TASKS[id].dtr_citation,
    instructions: TASKS[id].question.instructions, holdout: TASKS[id].holdout,
    provisions: TASKS[id].provisions.map((p) => ({ section: p.section, citation: p.dtr_citation })),
    options: Array.isArray(TASKS[id].question.criteria)
      ? (TASKS[id].question.criteria as string[]).map((d, i) => [String(i), d])
      : Object.entries(TASKS[id].question.criteria),
  }))
  return Response.json({ llmModel: LLM_MODEL, prices: PRICES, tasks, policy: { noulBand: NOUL_BAND, choiceFloor: CHOICE_FLOOR } })
}

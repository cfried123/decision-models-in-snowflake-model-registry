/** GET /api/config — model names and list prices for the board's labels and cost counters. */

import { LLM_MODEL, PRICES } from "@/lib/config"

export const dynamic = "force-dynamic"

export async function GET() {
  return Response.json({ llmModel: LLM_MODEL, prices: PRICES })
}

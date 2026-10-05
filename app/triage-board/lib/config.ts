/**
 * Demo configuration: where the data and models live, and list prices for the cost
 * counters. Rates are from the Snowflake Service Consumption Table effective
 * 2026-10-02 (Tables 1(a), 1(f), 2 and 6(a)); the board labels all costs as
 * estimates at list price.
 */

export const TICKETS_TABLE = process.env.TICKETS_TABLE ?? "DEVREL.DECISION_MODEL_BLOG.DEMO_TICKETS"
export const DECIDER_SERVICE = process.env.DECIDER_SERVICE ?? "DEVREL.DECISION_MODEL_BLOG.DECIDER_2B_DEMO"
export const LLM_MODEL = process.env.LLM_MODEL ?? "claude-sonnet-5"
export const LLM_WAREHOUSE = process.env.LLM_WAREHOUSE ?? "DECIDER_DEMO_WH"

/** USD per credit, On Demand. Platform and AI credits are both $2.00 here. */
const CREDIT_USD = 2.0

export const PRICES = {
  effectiveDate: "2026-10-02",
  /** decider-2b: one GPU_NV_S node (1x A10G), 0.57 platform credits per hour. */
  deciderUsdPerHour: 0.57 * CREDIT_USD,
  deciderHardware: "1x A10G (GPU_NV_S)",
  /** AI_COMPLETE claude-sonnet-5: AI credits per 1M input and output tokens. */
  llmInputUsdPerM: 1.2 * CREDIT_USD,
  llmOutputUsdPerM: 6.0 * CREDIT_USD,
  /** AI_COMPLETE runs from SQL, so the LLM side also keeps an XS warehouse running (1 credit/hour). */
  llmWarehouseUsdPerHour: 1.0 * CREDIT_USD,
}

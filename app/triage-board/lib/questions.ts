/**
 * The two decisions both sides make for every ticket. decider-2b gets them as
 * QUESTIONS_JSON; the LLM gets the same instructions and criteria in its prompt and a
 * JSON schema for the answer, so both sides answer the same question.
 */

export const CATEGORIES = {
  billing: "Charges, invoices, plan prices, payment methods or taxes",
  bug: "Something in the product is broken, erroring or behaving wrongly",
  account_access: "Login, password, SSO, 2FA, locked or deactivated accounts, permissions",
  refund: "The customer explicitly wants money back",
  feature_request: "Asks for something the product does not do yet",
} as const

export type Category = keyof typeof CATEGORIES
export const CATEGORY_KEYS = Object.keys(CATEGORIES) as Category[]

export const CATEGORY_LABELS: Record<Category, string> = {
  billing: "Billing",
  bug: "Bug",
  account_access: "Account access",
  refund: "Refund",
  feature_request: "Feature request",
}

export const ESCALATION = {
  escalate: "Needs a senior agent now: legal threats, outages for many users, security concerns or large financial impact",
  routine: "Can wait in the normal queue",
} as const

const CATEGORY_INSTRUCTIONS = "Which team should handle this support ticket?"
const ESCALATE_INSTRUCTIONS = "Should this ticket be escalated?"

/** QUESTIONS_JSON for decider-2b's system_one. */
export const DECIDER_QUESTIONS = {
  category: { type: "choice", instructions: CATEGORY_INSTRUCTIONS, criteria: CATEGORIES },
  escalate: { type: "choice", instructions: ESCALATE_INSTRUCTIONS, criteria: ESCALATION },
}

/** Prompt for the LLM side, built from the same instructions and criteria. */
export function llmPrompt(ticket: string): string {
  const list = (o: Record<string, string>) =>
    Object.entries(o).map(([k, v]) => `- ${k}: ${v}`).join("\n")
  return [
    "You triage customer support tickets for Lattice, a project-management SaaS app.",
    "",
    `1. category. ${CATEGORY_INSTRUCTIONS} Pick one:`,
    list(CATEGORIES),
    "",
    `2. escalate. ${ESCALATE_INSTRUCTIONS} Pick one:`,
    list(ESCALATION),
    "",
    "For each, also give your confidence from 0 to 1.",
    "",
    "Ticket:",
    ticket,
  ].join("\n")
}

/** response_format for AI_COMPLETE. additionalProperties: false keeps every model's structured output strict. */
export const LLM_RESPONSE_FORMAT = {
  type: "json",
  schema: {
    type: "object",
    additionalProperties: false,
    properties: {
      category: { type: "string", enum: CATEGORY_KEYS },
      category_confidence: { type: "number" },
      escalate: { type: "string", enum: Object.keys(ESCALATION) },
      escalate_confidence: { type: "number" },
    },
    required: ["category", "category_confidence", "escalate", "escalate_confidence"],
  },
}

/** What both API routes return for one ticket. */
export interface Decision {
  ticketId: number
  category: Category
  categoryConfidence: number
  escalate: boolean
  /** Probability (decider) or stated confidence (LLM) that the ticket needs escalation. */
  escalateProb: number
  /** Milliseconds around the model call on the app server. */
  modelMs: number
  inputTokens?: number
  outputTokens?: number
  error?: string
}

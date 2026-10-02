"use client"

/**
 * Live triage board. Both sides walk the same ticket list in the same order, each
 * keeping up to `concurrency` requests in flight. Mutable run data lives in refs; a
 * requestAnimationFrame-throttled counter re-renders as answers arrive.
 */

import { useCallback, useEffect, useRef, useState } from "react"
import { Pause, Play, RotateCcw } from "lucide-react"
import { Button } from "@/components/ui/button"
import { CATEGORY_KEYS, CATEGORY_LABELS, type Category, type Decision } from "@/lib/questions"

type Ticket = { id: number; text: string; genCategory: string; genEscalate: boolean }
type SideKey = "decider" | "llm"
type Prices = {
  effectiveDate: string
  deciderUsdPerHour: number
  deciderHardware: string
  llmInputUsdPerM: number
  llmOutputUsdPerM: number
  llmWarehouseUsdPerHour: number
}

interface SideRun {
  cursor: number
  inflight: number
  results: Map<number, Decision>
  order: number[] // ticket ids in arrival order
  done: number[] // completion timestamps (ms), for the rolling rate
  latencies: number[]
  errors: number
  inputTokens: number
  outputTokens: number
  activeMs: number // wall time with at least one request in flight
  activeSince: number | null
}

const newRun = (): SideRun => ({
  cursor: 0, inflight: 0, results: new Map(), order: [], done: [], latencies: [],
  errors: 0, inputTokens: 0, outputTokens: 0, activeMs: 0, activeSince: null,
})

const RATE_WINDOW_MS = 10_000
const LANE_CARDS = 4
const RUN_SIZES = [25, 100, 250, 500, 1000]

function pct(xs: number[], p: number): number | null {
  if (!xs.length) return null
  const s = [...xs].sort((a, b) => a - b)
  return s[Math.min(s.length - 1, Math.round((p / 100) * (s.length - 1)))]
}

function activeMs(run: SideRun, now: number) {
  return run.activeMs + (run.activeSince != null ? now - run.activeSince : 0)
}

/** Estimated spend so far for one side, at list price. */
function costUsd(side: SideKey, run: SideRun, prices: Prices, now: number): number {
  const hours = activeMs(run, now) / 3_600_000
  return side === "decider"
    ? prices.deciderUsdPerHour * hours
    : (run.inputTokens / 1e6) * prices.llmInputUsdPerM +
        (run.outputTokens / 1e6) * prices.llmOutputUsdPerM +
        prices.llmWarehouseUsdPerHour * hours
}

const fmtElapsed = (ms: number) => `${(ms / 1000).toFixed(1)} s`
const fmtMs = (v: number | null) => (v == null ? "–" : v >= 1000 ? `${(v / 1000).toFixed(2)} s` : `${Math.round(v)} ms`)
const fmtUsd = (v: number | null, digits = 3) => (v == null ? "–" : `$${v < 0.001 ? v.toFixed(5) : v.toFixed(digits)}`)
/** Running total: four decimals while it's under a dollar so it visibly ticks up. */
const fmtTotal = (v: number) => (v < 1 ? `$${v.toFixed(4)}` : `$${v.toFixed(2)}`)

export function TriageBoard() {
  const [tickets, setTickets] = useState<Ticket[]>([])
  const [config, setConfig] = useState<{ llmModel: string; prices: Prices } | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [concurrency, setConcurrency] = useState(1)
  const [runSize, setRunSize] = useState(25)
  const [, setFrame] = useState(0)

  const runs = useRef<Record<SideKey, SideRun>>({ decider: newRun(), llm: newRun() })
  const runningRef = useRef(false)
  const concurrencyRef = useRef(concurrency)
  const runSizeRef = useRef(runSize)
  const ticketsRef = useRef<Ticket[]>([])
  const epoch = useRef(0) // bumps on reset so late answers from an old run are dropped
  const rafPending = useRef(false)

  const rerender = useCallback(() => {
    if (rafPending.current) return
    rafPending.current = true
    requestAnimationFrame(() => {
      rafPending.current = false
      setFrame((f) => f + 1)
    })
  }, [])

  useEffect(() => {
    Promise.all([
      fetch("/api/config").then((r) => r.json()),
      fetch("/api/tickets?offset=0&limit=500").then((r) => r.json()),
      fetch("/api/tickets?offset=500&limit=500").then((r) => r.json()),
    ])
      .then(([cfg, a, b]) => {
        if (a.error || b.error) throw new Error(a.error ?? b.error)
        setConfig(cfg)
        ticketsRef.current = [...a, ...b]
        setTickets(ticketsRef.current)
      })
      .catch((e) => setLoadError(e instanceof Error ? e.message : String(e)))
  }, [])

  // Keep the elapsed clock, rolling rate and active-time cost moving while idle frames pass.
  useEffect(() => {
    if (!running) return
    const id = setInterval(rerender, 100)
    return () => clearInterval(id)
  }, [running, rerender])

  const pump = useCallback(
    (side: SideKey) => {
      const run = runs.current[side]
      const list = ticketsRef.current
      const limit = Math.min(runSizeRef.current, list.length)
      const myEpoch = epoch.current
      while (runningRef.current && run.inflight < concurrencyRef.current && run.cursor < limit) {
        const t = list[run.cursor++]
        if (run.inflight === 0) run.activeSince = performance.now()
        run.inflight++
        fetch(`/api/decide/${side}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticketId: t.id, text: t.text }),
        })
          .then((r) => r.json())
          .then((d: Decision) => {
            if (myEpoch !== epoch.current) return
            if (d.error) {
              run.errors++
            } else {
              run.results.set(d.ticketId, d)
              run.order.push(d.ticketId)
              run.latencies.push(d.modelMs)
              run.inputTokens += d.inputTokens ?? 0
              run.outputTokens += d.outputTokens ?? 0
            }
            run.done.push(performance.now())
          })
          .catch(() => {
            if (myEpoch === epoch.current) run.errors++
          })
          .finally(() => {
            if (myEpoch !== epoch.current) return
            run.inflight--
            if (run.inflight === 0 && run.activeSince != null) {
              run.activeMs += performance.now() - run.activeSince
              run.activeSince = null
            }
            const finished = run.cursor >= Math.min(runSizeRef.current, ticketsRef.current.length)
            if (finished && run.inflight === 0) {
              const other = runs.current[side === "decider" ? "llm" : "decider"]
              if (other.inflight === 0 && other.cursor >= Math.min(runSizeRef.current, ticketsRef.current.length)) {
                runningRef.current = false
                setRunning(false)
              }
            }
            rerender()
            pump(side)
          })
      }
      rerender()
    },
    [rerender],
  )

  const start = () => {
    runningRef.current = true
    setRunning(true)
    pump("decider")
    pump("llm")
  }
  const pause = () => {
    runningRef.current = false
    setRunning(false)
  }
  const reset = () => {
    pause()
    epoch.current++
    runs.current = { decider: newRun(), llm: newRun() }
    rerender()
  }

  useEffect(() => {
    concurrencyRef.current = concurrency
    if (runningRef.current) {
      pump("decider")
      pump("llm")
    }
  }, [concurrency, pump])
  useEffect(() => {
    runSizeRef.current = runSize
  }, [runSize])

  if (loadError) {
    return <div className="p-8 text-sm text-destructive">Could not load tickets: {loadError}</div>
  }
  if (!config || !tickets.length) {
    return <div className="p-8 text-sm text-muted-foreground">Loading tickets…</div>
  }

  const now = performance.now()
  const limit = Math.min(runSize, tickets.length)
  const d = runs.current.decider
  const l = runs.current.llm
  const queueStart = Math.min(d.cursor, l.cursor)
  const started = d.cursor > 0 || l.cursor > 0
  const finished = started && !running && d.cursor >= limit && l.cursor >= limit

  return (
    <div className="flex flex-col gap-4 p-4 lg:p-6">
      <div className="flex flex-wrap items-center gap-3">
        {running ? (
          <Button onClick={pause} variant="secondary">
            <Pause className="size-4" /> Pause
          </Button>
        ) : (
          <Button onClick={start} disabled={finished}>
            <Play className="size-4" /> {started ? "Resume" : "Start"}
          </Button>
        )}
        <Button onClick={reset} variant="outline">
          <RotateCcw className="size-4" /> Reset
        </Button>
        <label className="ml-2 flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">Requests in flight per side</span>
          <input
            type="range" min={1} max={16} value={concurrency}
            onChange={(e) => setConcurrency(Number(e.target.value))}
            className="w-32 accent-[var(--brand-primary)]"
          />
          <span className="w-6 font-mono tabular-nums">{concurrency}</span>
        </label>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">Tickets</span>
          <select
            value={runSize} disabled={started}
            onChange={(e) => setRunSize(Number(e.target.value))}
            className="rounded-md border border-border bg-background px-2 py-1"
          >
            {RUN_SIZES.filter((n) => n <= tickets.length).map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </label>
        <IncomingQueue tickets={tickets.slice(queueStart, Math.min(queueStart + 4, limit))} />
      </div>

      <div id="boards" className="grid gap-4 lg:grid-cols-2">
        <SidePanel
          title="decider-2b"
          subtitle={`Decision model on ${config.prices.deciderHardware}, real-time inference over REST`}
          accent="var(--brand-primary)"
          side="decider" run={d} limit={limit} tickets={tickets} prices={config.prices} now={now}
        />
        <SidePanel
          title={config.llmModel}
          subtitle="Frontier LLM through AI_COMPLETE with structured output"
          accent="var(--llm-accent)"
          side="llm" run={l} limit={limit} tickets={tickets} prices={config.prices} now={now}
        />
      </div>

      <p className="text-xs text-muted-foreground">
        Tickets are synthetic. Latency is timed on the app server around each
        model call. Costs are estimates at list price ({config.prices.effectiveDate} Service Consumption Table):
        decider-2b is GPU node time while requests were in flight; the LLM is input and output tokens plus the XS
        warehouse AI_COMPLETE runs on.
      </p>
    </div>
  )
}

function IncomingQueue({ tickets }: { tickets: Ticket[] }) {
  return (
    <div className="ml-auto flex min-w-0 items-center gap-2 overflow-hidden">
      <span className="shrink-0 text-xs uppercase tracking-wide text-muted-foreground">Incoming</span>
      {tickets.map((t) => (
        <div
          key={t.id}
          className="animate-[ticket-in_300ms_ease-out] w-44 shrink-0 truncate rounded-md border border-border bg-card px-2 py-1 text-xs"
          title={t.text}
        >
          <span className="font-mono text-muted-foreground">#{t.id}</span> {t.text}
        </div>
      ))}
    </div>
  )
}

function SidePanel(props: {
  title: string
  subtitle: string
  accent: string
  side: SideKey
  run: SideRun
  limit: number
  tickets: Ticket[]
  prices: Prices
  now: number
}) {
  const { title, subtitle, accent, side, run, limit, tickets, prices, now } = props
  const byId = new Map(tickets.map((t) => [t.id, t]))
  // Rolling rate while the side runs; once it has finished, the average over its active time.
  const sideDone = run.inflight === 0 && run.cursor >= limit
  const started = run.cursor > 0
  const recent = run.done.filter((t) => now - t <= RATE_WINDOW_MS)
  const windowSec = Math.min(RATE_WINDOW_MS, Math.max(1, activeMs(run, now))) / 1000
  const rate = sideDone
    ? run.results.size / Math.max(0.001, activeMs(run, now) / 1000)
    : recent.length ? recent.length / windowSec : 0
  const spent = costUsd(side, run, prices, now)
  const per1k = run.results.size ? (spent / run.results.size) * 1000 : null

  const lanes = new Map<Category, number[]>(CATEGORY_KEYS.map((c) => [c, []]))
  for (const id of run.order) lanes.get(run.results.get(id)!.category)!.push(id)

  return (
    <section className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4">
      <header className="flex items-baseline justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold tracking-tight" style={{ color: accent }}>{title}</h2>
          <p className="text-xs text-muted-foreground">{subtitle}</p>
        </div>
        <div className="flex flex-col items-end font-mono tabular-nums">
          <span className="text-lg font-semibold" style={{ color: accent }} title="Elapsed time with requests in flight">
            {fmtElapsed(activeMs(run, now))}
          </span>
          <span className="text-xs text-muted-foreground">{run.results.size}/{limit}</span>
        </div>
      </header>

      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div className="h-full transition-[width] duration-300" style={{ width: `${(run.results.size / limit) * 100}%`, background: accent }} />
      </div>

      <dl className="grid grid-cols-3 gap-2">
        <Stat
          label="Decisions"
          value={String(run.results.size)}
          accent={accent}
          big
          sub={`${rate ? rate.toFixed(1) : "–"} / s`}
        />
        <Stat
          label="Latency p50 / p95"
          value={`${fmtMs(pct(run.latencies, 50))} / ${fmtMs(pct(run.latencies, 95))}`}
          note={run.errors ? `${run.errors} errors` : undefined}
        />
        <Stat
          label="Total cost"
          value={started ? fmtTotal(spent) : "–"}
          accent={accent}
          big
          sub={`${fmtUsd(per1k)} / 1,000 tickets`}
        />
      </dl>

      <div className="grid grid-cols-5 gap-2">
        {CATEGORY_KEYS.map((c) => {
          const ids = lanes.get(c)!
          return (
            <div key={c} className="flex min-w-0 flex-col gap-1.5 rounded-lg bg-muted/60 p-1.5">
              <div className="flex items-baseline justify-between px-1">
                <span className="truncate text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  {CATEGORY_LABELS[c]}
                </span>
                <span className="font-mono text-xs tabular-nums">{ids.length}</span>
              </div>
              <div className="flex h-[17.5rem] flex-col gap-1.5 overflow-hidden">
                {ids.slice(-LANE_CARDS).reverse().map((id) => (
                  <TicketCard key={id} ticket={byId.get(id)!} decision={run.results.get(id)!} accent={accent} />
                ))}
              </div>
            </div>
          )
        })}
      </div>
    </section>
  )
}

function Stat(props: { label: string; value: string; accent?: string; big?: boolean; sub?: string; note?: string }) {
  const { label, value, accent, big, sub, note } = props
  return (
    <div className="rounded-lg bg-muted/60 px-3 py-2">
      <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</dt>
      <dd className={`font-mono tabular-nums ${big ? "text-xl font-semibold" : "text-sm pt-1.5"}`} style={accent ? { color: accent } : undefined}>
        {value}
      </dd>
      {sub && <dd className="font-mono text-[11px] tabular-nums text-muted-foreground">{sub}</dd>}
      {note && <dd className="text-[11px] text-destructive">{note}</dd>}
    </div>
  )
}

function TicketCard({ ticket, decision, accent }: { ticket: Ticket; decision: Decision; accent: string }) {
  return (
    <article
      className="animate-[card-in_350ms_ease-out] rounded-md border border-border bg-card p-1.5 shadow-sm"
      title={ticket.text}
    >
      <div className="flex items-center justify-between gap-1">
        <span className="font-mono text-[10px] text-muted-foreground">#{ticket.id}</span>
        {decision.escalate && (
          <span className="rounded bg-destructive px-1 text-[9px] font-semibold uppercase leading-4 text-destructive-foreground">
            Escalate
          </span>
        )}
      </div>
      <p className="line-clamp-2 text-[11px] leading-snug">{ticket.text}</p>
      <div className="mt-1 h-1 w-full overflow-hidden rounded-full bg-muted">
        <div className="h-full" style={{ width: `${Math.round(decision.categoryConfidence * 100)}%`, background: accent }} />
      </div>
    </article>
  )
}

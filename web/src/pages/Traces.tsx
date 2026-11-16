import NumberFlow, { type Format } from "@number-flow/react";
import { Search, ThumbsDown, ThumbsUp } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { EmptyState } from "@/components/EmptyState";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/fields";
import { ApiError, callWithHeaders, useResource } from "@/lib/api";
import type { Stats, Trace, TraceSummary } from "@/lib/types";
import { cn, formatTime } from "@/lib/utils";
import { TraceDetail } from "@/pages/traces/TraceDetail";

const PAGE = 50;
const SEARCH_DELAY_MS = 250;
const FILTERS = [
  { value: "", label: "All" },
  { value: "answered", label: "Answered" },
  { value: "refused", label: "Refused" },
  { value: "lookup", label: "Passages only" },
  { value: "failed", label: "Failed" },
  { value: "rated", label: "Rated" },
] as const;

function plural(n: number, one: string, many: string): string {
  return `${n.toLocaleString()} ${n === 1 ? one : many}`;
}

/** Today's usage in plain figures: one number per cell, one sentence explaining it. */
function Today({ stats }: { stats: Stats["today"] }) {
  const answered = stats.model_calls + stats.cache_hits;
  const parts = [
    answered ? plural(answered, "answered", "answered") : "",
    stats.refused_free ? `${stats.refused_free.toLocaleString()} refused` : "",
    stats.lookups ? plural(stats.lookups, "passage search only", "passage searches only") : "",
    stats.failed ? `${stats.failed.toLocaleString()} failed` : "",
    stats.helpful || stats.unhelpful ? `${stats.helpful} rated right, ${stats.unhelpful} wrong` : "",
  ].filter(Boolean);
  const p50 = stats.latency_p50_ms;
  const cells: { label: string; value: number | null; format?: Format; suffix?: string; note: string }[] = [
    {
      label: "Questions",
      value: stats.queries,
      note: parts.length ? `${parts.join(", ")}.` : "None yet today.",
    },
    {
      label: "Sent to a model",
      value: stats.model_calls,
      note: stats.cache_hits
        ? `${plural(stats.cache_hits, "more answer", "more answers")} came from the cache at no cost.`
        : "Each one uses API quota; repeated questions are answered from the cache.",
    },
    {
      label: "Tokens used",
      value: stats.tokens,
      note: stats.model_calls ? `About ${stats.tokens_per_call.toLocaleString()} per model call.` : "No model calls yet.",
    },
    {
      label: "Tokens saved",
      value: stats.tokens_saved,
      note: "Weak and repeated passages left out of the prompts that were sent.",
    },
    {
      label: "Typical time",
      value: p50 === null ? null : p50 / 1000,
      format: { minimumFractionDigits: 1, maximumFractionDigits: 1 },
      suffix: " s",
      note: stats.latency_p95_ms
        ? `Half of all questions took less. The slowest 5% took over ${(stats.latency_p95_ms / 1000).toFixed(1)} s.`
        : "Measured once questions are asked.",
    },
  ];
  return (
    <section aria-labelledby="today-title" className="flex flex-col gap-3">
      <h2 id="today-title" className="text-lg font-semibold">
        Today
      </h2>
      <dl className="grid grid-cols-1 border-t border-ink sm:grid-cols-2 lg:grid-cols-5">
        {cells.map((c) => (
          <div key={c.label} className="flex flex-col gap-1.5 border-b border-rule py-4 pr-6 lg:border-b-0">
            <dt className="text-sm font-medium">{c.label}</dt>
            <dd className="text-3xl font-semibold leading-none tracking-tight [font-variant-numeric:proportional-nums]">
              {c.value === null ? "–" : <NumberFlow value={c.value} format={c.format} suffix={c.suffix} />}
            </dd>
            <dd className="text-sm leading-snug text-graphite">{c.note}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function outcome(t: TraceSummary): string {
  return t.refused ? "refused" : t.errors.length ? "failed" : (t.provider ?? "passages only");
}

/** The list for one search and filter, a page at a time. */
function useTraceList(q: string, status: string) {
  const [rows, setRows] = useState<TraceSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<ApiError | null>(null);
  const request = useRef(0);

  const load = useCallback(
    async (offset: number) => {
      const mine = ++request.current;
      setLoading(true);
      const params = new URLSearchParams({ summary: "true", limit: String(PAGE), offset: String(offset) });
      if (q) params.set("q", q);
      if (status) params.set("status", status);
      try {
        const { data, headers } = await callWithHeaders<TraceSummary[]>("GET", `/traces?${params}`);
        if (mine !== request.current) return; // a newer search has started
        setRows((prev) => (offset ? [...prev, ...data] : data));
        setTotal(Number(headers.get("X-Total-Count") ?? data.length));
        setError(null);
      } catch (e) {
        if (mine === request.current) setError(e instanceof ApiError ? e : new ApiError("error", String(e), 0));
      } finally {
        if (mine === request.current) setLoading(false);
      }
    },
    [q, status],
  );

  useEffect(() => {
    void load(0);
  }, [load]);

  return { rows, total, loading, error, reload: () => load(0), more: () => load(rows.length) };
}

export function TracesPage() {
  const stats = useResource<Stats>("/stats");
  const [typed, setTyped] = useState("");
  const [q, setQ] = useState("");
  const [status, setStatus] = useState("");
  // #/traces?id=... opens one trace directly (the "Open trace" link under an answer).
  const [selected, setSelected] = useState<string | null>(() =>
    new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("id"),
  );
  const list = useTraceList(q, status);
  const currentId = selected ?? list.rows[0]?.trace_id ?? null;
  const detail = useResource<Trace>(currentId ? `/traces/${currentId}` : null);
  const filtering = Boolean(q || status);

  useEffect(() => {
    const timer = window.setTimeout(() => setQ(typed.trim()), SEARCH_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [typed]);

  const pick = (id: string) => {
    setSelected(id);
    window.history.replaceState(null, "", `#/traces?id=${id}`);
  };

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-8">
      <header className="flex flex-col gap-2">
        <h1 className="page-title">Traces</h1>
      </header>
      {stats.data ? <Today stats={stats.data.today} /> : null}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <div className="relative w-full sm:w-80">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-graphite" aria-hidden />
          <Input
            type="search"
            aria-label="Search questions and answers"
            placeholder="Search questions and answers"
            className="pl-9"
            value={typed}
            onChange={(e) => setTyped(e.target.value)}
          />
        </div>
        <div role="radiogroup" aria-label="Show" className="flex flex-wrap gap-1">
          {FILTERS.map((f) => (
            <button
              key={f.value}
              type="button"
              role="radio"
              aria-checked={status === f.value}
              onClick={() => setStatus(f.value)}
              className={cn(
                "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                status === f.value ? "bg-ink text-paper" : "hover:bg-wash",
              )}
            >
              {f.label}
            </button>
          ))}
        </div>
        <p className="text-sm text-graphite sm:ml-auto" aria-live="polite">
          {list.loading && !list.rows.length ? "Loading" : `${list.total.toLocaleString()} ${list.total === 1 ? "trace" : "traces"}`}
        </p>
      </div>
      {list.error ? <p role="alert" className="border-l-4 border-ink pl-4 text-base">{list.error.message}</p> : null}
      {!list.loading && !list.error && list.rows.length === 0 ? (
        <EmptyState>
          {filtering ? (
            "No trace matches. Clear the search or choose All."
          ) : (
            <>
              No questions asked yet. <a className="text-ink underline underline-offset-4" href="#/ask">Ask one</a> and
              its record appears here.
            </>
          )}
        </EmptyState>
      ) : null}
      {list.rows.length || detail.data ? (
        <div className="grid gap-10 lg:grid-cols-[20rem_minmax(0,1fr)]">
          <div className="flex max-h-[75vh] flex-col overflow-y-auto">
            <ul className="flex flex-col">
              {list.rows.map((t) => {
                const active = t.trace_id === currentId;
                return (
                  <li key={t.trace_id}>
                    <button
                      type="button"
                      aria-current={active || undefined}
                      onClick={() => pick(t.trace_id)}
                      className={cn(
                        "flex w-full flex-col gap-0.5 border-t border-rule px-3 py-3 text-left transition-colors",
                        active ? "bg-ink text-paper" : "hover:bg-wash",
                      )}
                    >
                      <span className="line-clamp-2 text-sm font-medium">{t.question}</span>
                      <span className={cn("flex items-center gap-1.5 text-xs", active ? "text-paper/70" : "text-graphite")}>
                        {formatTime(t.created_at)}, {outcome(t)}
                        {t.feedback === "up" ? <ThumbsUp className="size-3" aria-label="rated right" /> : null}
                        {t.feedback === "down" ? <ThumbsDown className="size-3" aria-label="rated wrong" /> : null}
                      </span>
                    </button>
                  </li>
                );
              })}
            </ul>
            {list.rows.length < list.total ? (
              <Button variant="ghost" size="sm" className="mt-2 shrink-0" disabled={list.loading} onClick={() => void list.more()}>
                {list.loading ? "Loading" : `Show ${Math.min(PAGE, list.total - list.rows.length)} more`}
              </Button>
            ) : null}
          </div>
          {detail.data ? (
            <TraceDetail
              key={detail.data.trace_id}
              t={detail.data}
              onRated={() => {
                void list.reload();
                void stats.reload();
              }}
            />
          ) : detail.error ? (
            <p role="alert" className="border-l-4 border-ink pl-4 text-base">
              {detail.error.status === 404 ? "This trace no longer exists. It may have passed the retention limit." : detail.error.message}
            </p>
          ) : (
            <p className="text-base text-graphite" role="status">Loading the trace</p>
          )}
        </div>
      ) : null}
    </div>
  );
}

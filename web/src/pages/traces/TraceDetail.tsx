import { useState } from "react";

import { AnswerActions } from "@/components/AnswerActions";
import { Markdown } from "@/components/Markdown";
import { RelevanceBar } from "@/components/RelevanceBar";
import { TimeBar } from "@/components/TimeBar";
import type { Trace } from "@/lib/types";
import { formatTime, seconds } from "@/lib/utils";

/** One question's full record: answer, verdict, timings, every candidate and the exact prompt. */
export function TraceDetail({ t, onRated }: { t: Trace; onRated: () => void }) {
  const [showPrompt, setShowPrompt] = useState(false);
  const bestRrf = Math.max(...t.candidates.map((c) => c.rrf_score), 0);
  const reranked = t.candidates.some((c) => c.rerank_score !== null);
  // Final order: by reranker score when the reranker ran, otherwise the fused order.
  const ordered = reranked
    ? [...t.candidates].sort((a, b) => (b.rerank_score ?? -1) - (a.rerank_score ?? -1))
    : t.candidates;
  const answered = Boolean(t.answer) && !t.refused;
  return (
    <article className="flex min-w-0 flex-col gap-6">
      <header className="flex flex-col gap-2">
        <h2 className="text-2xl font-semibold leading-snug">{t.question}</h2>
        <p className="text-sm text-graphite">
          {formatTime(t.created_at)}.{" "}
          {t.provider ? `${t.provider} (${t.model}), ${t.tokens_in} tokens in, ${t.tokens_out} out` : "No model call"}
          {`, ${seconds(t.latency_ms)}`}
          {t.cached ? ", cached" : ""}
          {t.tokens_saved ? `, ${t.tokens_saved} tokens saved` : ""}. Mode {t.mode}, config {t.config_hash}, trace{" "}
          {t.trace_id}.
        </p>
      </header>
      {t.errors.map((e) => (
        <p key={e} className="border-l-4 border-ink pl-4 text-base">{e}</p>
      ))}
      {answered ? (
        <Markdown text={t.answer} />
      ) : (
        <p className="prose-read">{t.answer || "No answer was generated."}</p>
      )}
      {answered ? (
        <AnswerActions traceId={t.trace_id} text={t.answer} initial={t.feedback} initialNote={t.feedback_note} onRated={onRated} />
      ) : null}
      {t.timings?.total_ms ? (
        <TimeBar timings={t.timings} tokensIn={t.tokens_in} tokensOut={t.tokens_out} tokensSaved={t.tokens_saved} cached={t.cached} />
      ) : null}
      <section className="flex flex-col">
        <h3 className="mb-2 text-lg font-semibold">How the {t.candidates.length} candidates ranked</h3>
        <p className="mb-3 text-sm text-graphite">
          {reranked
            ? "In final order. Bars show the reranker score; dense and keyword are each passage's rank before fusion."
            : "In fused order. Bars show the fused score relative to the best candidate."}
        </p>
        <ol>
          {ordered.map((c, i) => (
            <li key={c.chunk_id} className="grid grid-cols-[2rem_1fr] gap-x-3 border-t border-rule py-3">
              <span className="dot text-xl font-black">{i + 1}</span>
              <div className="flex min-w-0 flex-col gap-1.5">
                <div className="flex justify-between gap-3 text-sm">
                  <span className="truncate font-medium">{c.title}</span>
                  <span className="shrink-0 text-graphite">
                    dense {c.dense_rank ?? "–"}, keyword {c.bm25_rank ?? "–"}
                  </span>
                </div>
                {reranked && c.rerank_score === null ? (
                  <span className="text-sm text-graphite">Not reranked: below the reranking cut-off.</span>
                ) : (
                  <RelevanceBar
                    size="sm"
                    rank={Math.min(i, 12)}
                    cited
                    value={c.rerank_score ?? (bestRrf ? c.rrf_score / bestRrf : 0)}
                  />
                )}
              </div>
            </li>
          ))}
        </ol>
      </section>
      <section>
        <button
          type="button"
          aria-expanded={showPrompt}
          className="text-base font-semibold underline underline-offset-4"
          onClick={() => setShowPrompt(!showPrompt)}
        >
          {showPrompt ? "Hide the exact prompt" : "Show the exact prompt"}
        </button>
        {showPrompt ? (
          <pre className="mt-3 max-h-[28rem] overflow-auto whitespace-pre-wrap rounded-lg border border-rule p-4 text-sm">
            {t.prompt || "No model call, so no prompt."}
          </pre>
        ) : null}
      </section>
    </article>
  );
}

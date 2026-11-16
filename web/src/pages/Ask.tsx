import { CornerDownLeft, RotateCcw, Square } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { AnswerActions } from "@/components/AnswerActions";
import { Dither, orbField, revealField } from "@/components/Dither";
import { Markdown } from "@/components/Markdown";
import { RelevanceBar } from "@/components/RelevanceBar";
import { SourceDrawer } from "@/components/SourceDrawer";
import { TimeBar } from "@/components/TimeBar";
import { Button } from "@/components/ui/button";
import { Select, Switch } from "@/components/ui/fields";
import { ApiError, useResource } from "@/lib/api";
import { streamAnswer } from "@/lib/stream";
import type { Answer, DocumentsInfo, Mode, Settings, Source, TraceSummary } from "@/lib/types";
import { cn, seconds } from "@/lib/utils";

const MODES: { value: Mode; label: string; hint: string }[] = [
  { value: "auto", label: "Auto", hint: "Free APIs in order, local model last" },
  { value: "free_api_only", label: "Free APIs", hint: "Hosted providers only" },
  { value: "ollama_only", label: "Local only", hint: "Nothing leaves this computer" },
];
const ALL = "__all__";
const RECENT_SHOWN = 5;
// Errors whose fix is on the Models page.
const MODEL_ERRORS = new Set(["no_provider_available", "quota_exhausted", "key_missing", "key_rejected"]);

type Phase = "idle" | "searching" | "writing" | "done" | "error";

function Evidence({
  sources,
  nCandidates,
  refused,
  pending,
  onOpen,
}: {
  sources: Source[];
  nCandidates: number | null;
  refused: boolean;
  pending: boolean;
  onOpen: (s: Source) => void;
}) {
  const cited = sources.filter((s) => s.cited).length;
  const summary = pending
    ? `${sources.length} passages sent to the model. The ones it cites are marked when the answer is complete.`
    : refused
      ? `The closest ${sources.length} of ${nCandidates} candidates. None was strong enough to answer from.`
      : `${cited} cited, ${sources.length} sent to the model, chosen from ${nCandidates} candidates.`;
  return (
    <section aria-labelledby="evidence-title" className="flex flex-col gap-1">
      <h2 id="evidence-title" className="text-lg font-semibold">
        Ranked passages
      </h2>
      <p className="mb-4 text-sm text-graphite">{summary}</p>
      <ol className="flex flex-col">
        {sources.map((s, i) => (
          <motion.li key={s.chunk_id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.05 + i * 0.07 }}>
            <button
              type="button"
              onClick={() => onOpen(s)}
              className="group grid w-full grid-cols-[2rem_1fr] gap-x-3 border-t border-rule py-4 text-left transition-colors hover:bg-wash"
            >
              <span className="dot pl-1 text-3xl font-black leading-none">{s.n}</span>
              <span className="flex min-w-0 flex-col gap-1.5 pr-2">
                <span className="flex items-baseline justify-between gap-3">
                  <span className="truncate text-base font-semibold">{s.title}</span>
                  {pending ? null : <span className="shrink-0 text-xs text-graphite">{s.cited ? "Cited" : "Not cited"}</span>}
                </span>
                <span className="truncate text-sm text-graphite">
                  {s.section_path || s.path}
                  {s.page ? `, page ${s.page}` : ""}
                </span>
                <RelevanceBar value={s.relevance} rank={i} cited={pending || s.cited} />
              </span>
            </button>
          </motion.li>
        ))}
      </ol>
    </section>
  );
}

/** Before the first question: what this does, what was asked before, and the dithered orb. */
function Idle({ docs, recent, onAsk }: { docs: DocumentsInfo | null; recent: string[]; onAsk: (q: string) => void }) {
  const empty = docs !== null && docs.chunks === 0;
  return (
    <section className="grid items-center gap-10 md:grid-cols-[minmax(0,1fr)_minmax(0,24rem)]">
      <div className="flex max-w-[34rem] flex-col gap-5">
        <p className="font-serif text-2xl leading-snug">
          Ask a question and get an answer straight from your documents, with the source shown.
        </p>
        {empty ? (
          <p className="text-base text-graphite">
            Nothing is indexed yet.{" "}
            <a className="text-ink underline underline-offset-4" href="#/documents">
              Add documents
            </a>{" "}
            to start asking.
          </p>
        ) : docs ? (
          <p className="text-base text-graphite">
            Searching <span className="font-semibold text-ink tabular-nums">{docs.documents.length}</span>{" "}
            {docs.documents.length === 1 ? "document" : "documents"} split into{" "}
            <span className="font-semibold text-ink tabular-nums">{docs.chunks.toLocaleString()}</span> passages. Press
            Enter to ask, Shift and Enter for a new line, Ctrl K for commands.
          </p>
        ) : null}
        {recent.length && !empty ? (
          <div className="flex flex-col">
            <h2 className="pb-2 text-sm font-medium text-graphite">Asked before</h2>
            {recent.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => onAsk(q)}
                className="group flex items-center justify-between gap-4 border-t border-rule py-2.5 text-left text-base transition-colors hover:bg-wash"
              >
                <span className="line-clamp-1 pl-1">{q}</span>
                <CornerDownLeft className="mr-2 size-4 shrink-0 text-graphite opacity-0 transition-opacity group-hover:opacity-100 group-focus-visible:opacity-100" />
              </button>
            ))}
          </div>
        ) : null}
      </div>
      <div className="aspect-square w-full max-w-[24rem] justify-self-center md:justify-self-end">
        <Dither field={orbField} cell={3} label="A dithered sphere lit by a slowly moving light" />
      </div>
    </section>
  );
}

/** While passages are found and ranked: a dither field that fills in, paced by the last search. */
function Reading({ expectedMs }: { expectedMs: number }) {
  const started = useRef(performance.now());
  // Approaches 1 but never reaches it, so the band keeps moving until the passages arrive.
  const progress = useMemo(
    () => () => 1 - Math.exp(-(performance.now() - started.current) / (expectedMs / 2.2)),
    [expectedMs],
  );
  const field = useMemo(() => revealField(progress), [progress]);
  return (
    <div className="flex flex-col gap-3" role="status">
      <div className="h-28 w-full border border-ink">
        <Dither field={field} cell={4} fps={30} />
      </div>
      <p className="text-base text-graphite">Finding and ranking the most relevant passages.</p>
    </div>
  );
}

function Failure({ error, onRetry }: { error: ApiError; onRetry: () => void }) {
  return (
    <div role="alert" className="flex max-w-2xl flex-col gap-3 border-l-4 border-ink pl-5">
      <h2 className="text-lg font-semibold">This question could not be answered</h2>
      <p className="text-base">{error.message}</p>
      <div className="flex flex-wrap items-center gap-3 pt-1">
        <Button type="button" variant="outline" size="sm" onClick={onRetry}>
          <RotateCcw />
          Try again
        </Button>
        {MODEL_ERRORS.has(error.code) ? (
          <a className="text-sm underline underline-offset-4" href="#/models">
            Check the providers on the Models page
          </a>
        ) : null}
      </div>
    </div>
  );
}

export function AskPage({
  initialQuestion,
  onConsumed,
  onAnswered,
}: {
  initialQuestion: string | null;
  onConsumed: () => void;
  onAnswered: () => void;
}) {
  const settings = useResource<Settings>("/settings");
  const docs = useResource<DocumentsInfo>("/documents");
  const history = useResource<TraceSummary[]>("/traces?summary=true&status=answered&limit=30");
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode | null>(null);
  const [scope, setScope] = useState(ALL);
  const [retrieveOnly, setRetrieveOnly] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [asked, setAsked] = useState("");
  const [draft, setDraft] = useState<{ sources: Source[]; text: string }>({ sources: [], text: "" });
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [open, setOpen] = useState<Source | null>(null);
  const [searchMs, setSearchMs] = useState(1800);
  const controller = useRef<AbortController | null>(null);

  const busy = phase === "searching" || phase === "writing";
  const activeMode = mode ?? settings.data?.mode ?? "auto";
  const folders = useMemo(() => {
    const set = new Set<string>();
    for (const d of docs.data?.documents ?? []) if (d.path.includes("/")) set.add(`${d.path.split("/")[0]}/`);
    return [...set].sort();
  }, [docs.data]);
  const recent = useMemo(
    () => [...new Set((history.data ?? []).map((t) => t.question))].slice(0, RECENT_SHOWN),
    [history.data],
  );

  async function ask(q: string) {
    const text = q.trim();
    if (!text || busy) return;
    const abort = new AbortController();
    controller.current = abort;
    setPhase("searching");
    setAsked(text);
    setAnswer(null);
    setError(null);
    setDraft({ sources: [], text: "" });
    try {
      const body = {
        question: text,
        mode: activeMode,
        retrieve_only: retrieveOnly,
        ...(scope !== ALL ? { filters: { path_prefixes: [scope] } } : {}),
      };
      const result = await streamAnswer(
        body,
        {
          onSources: (sources) => {
            setDraft({ sources, text: "" });
            setPhase("writing");
          },
          onDelta: (piece) => setDraft((d) => ({ ...d, text: d.text + piece })),
        },
        abort.signal,
      );
      setAnswer(result);
      setPhase("done");
      const search = (result.timings.retrieve_ms ?? 0) + (result.timings.rerank_ms ?? 0);
      if (search) setSearchMs(Math.max(800, search));
      onAnswered();
      void history.reload();
    } catch (e) {
      if (abort.signal.aborted) {
        setPhase("idle");
        toast("Stopped. Nothing more was written.");
        return;
      }
      setError(e instanceof ApiError ? e : new ApiError("error", String(e), 0));
      setPhase("error");
      onAnswered(); // a failed call may still have used quota

    } finally {
      controller.current = null;
    }
  }

  useEffect(() => {
    if (initialQuestion) {
      setQuestion(initialQuestion);
      onConsumed();
      void ask(initialQuestion);
    }
  }, [initialQuestion]);

  // Leaving the page stops an answer that is still being written.
  useEffect(() => () => controller.current?.abort(), []);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    void ask(question);
  };
  const askAgain = (q: string) => {
    setQuestion(q);
    void ask(q);
  };

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-10">
      <form onSubmit={submit} className="flex flex-col gap-5">
        <label htmlFor="q" className="sr-only">
          Question
        </label>
        <textarea
          id="q"
          rows={1}
          autoFocus
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              void ask(question);
            }
          }}
          placeholder="Ask your documents a question"
          className="w-full resize-none [field-sizing:content] border-b-2 border-ink bg-transparent pb-3 text-3xl font-semibold leading-tight tracking-tight outline-none placeholder:text-rule md:text-[2.75rem]"
        />
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <div role="radiogroup" aria-label="Answer with" className="flex rounded-lg border border-ink p-0.5">
            {MODES.map((m) => (
              <button
                key={m.value}
                type="button"
                role="radio"
                aria-checked={activeMode === m.value}
                title={m.hint}
                onClick={() => setMode(m.value)}
                className={cn(
                  "relative rounded-md px-3 py-1.5 text-sm font-medium",
                  activeMode === m.value ? "text-paper" : "hover:bg-wash",
                )}
              >
                {activeMode === m.value ? (
                  <motion.span layoutId="mode-pill" className="absolute inset-0 rounded-md bg-ink" transition={{ type: "spring", bounce: 0.15, duration: 0.4 }} />
                ) : null}
                <span className="relative">{m.label}</span>
              </button>
            ))}
          </div>
          <Select
            className="h-9 w-48 text-sm"
            value={scope}
            onValueChange={setScope}
            options={[{ value: ALL, label: "All documents" }, ...folders.map((f) => ({ value: f, label: f }))]}
          />
          <label className="flex items-center gap-2 text-sm">
            <Switch checked={retrieveOnly} onCheckedChange={setRetrieveOnly} />
            Passages only, no model call
          </label>
          {busy ? (
            <Button type="button" variant="outline" className="ml-auto" onClick={() => controller.current?.abort()}>
              <Square className="fill-current" />
              Stop
            </Button>
          ) : (
            <Button type="submit" className="ml-auto" disabled={!question.trim()}>
              Ask
            </Button>
          )}
        </div>
      </form>

      <AnimatePresence mode="wait">
        {phase === "searching" ? (
          <motion.div key="searching" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <Reading expectedMs={searchMs} />
          </motion.div>
        ) : phase === "writing" || (phase === "done" && answer) ? (
          // One element for writing and done, so the text does not blink when the stream ends.
          <motion.div
            key="result"
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
            className="grid gap-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,26rem)]"
          >
            {answer ? (
              <section aria-label={`Answer to: ${asked}`} className="flex flex-col gap-5">
                {!answer.answer ? (
                  <p className="text-base text-graphite">Passages only: no answer was generated and no quota was used.</p>
                ) : answer.refused ? (
                  <div className="border-l-4 border-ink pl-5">
                    <p className="prose-read">{answer.answer}</p>
                    <p className="mt-2 text-sm text-graphite">Rephrase the question, or add the document that answers it.</p>
                  </div>
                ) : (
                  <Markdown text={answer.answer} sources={answer.sources} onOpen={setOpen} />
                )}
                {answer.answer && !answer.refused ? <AnswerActions traceId={answer.trace_id} text={answer.answer} /> : null}
                <p className="text-sm text-graphite">
                  {answer.provider ? `Answered by ${answer.provider} (${answer.model}) in ${seconds(answer.latency_ms)}` : `Done in ${seconds(answer.latency_ms)} without a model call`}
                  {answer.cached ? ", from the cache." : "."}{" "}
                  <a className="underline underline-offset-4 hover:text-ink" href={`#/traces?id=${answer.trace_id}`}>
                    Open trace
                  </a>
                </p>
                <TimeBar
                  timings={answer.timings}
                  tokensIn={answer.tokens_in}
                  tokensOut={answer.tokens_out}
                  tokensSaved={answer.tokens_saved}
                  cached={answer.cached}
                />
                {answer.warnings.length ? (
                  <ul className="flex flex-col gap-1 text-sm">
                    {answer.warnings.map((w) => (
                      <li key={w} className="before:mr-2 before:content-['!']">{w}</li>
                    ))}
                  </ul>
                ) : null}
              </section>
            ) : (
              <section aria-label={`Answer to: ${asked}`} aria-live="polite" className="flex flex-col gap-5">
                {draft.text ? (
                  <Markdown text={draft.text} sources={draft.sources} onOpen={setOpen} streaming />
                ) : (
                  <p className="shimmer-text text-base" role="status">
                    Reading {draft.sources.length} passages and writing the answer
                  </p>
                )}
              </section>
            )}
            <Evidence
              sources={answer?.sources ?? draft.sources}
              nCandidates={answer?.n_candidates ?? null}
              refused={answer?.refused ?? false}
              pending={!answer}
              onOpen={setOpen}
            />
          </motion.div>
        ) : phase === "error" && error ? (
          <motion.div key="error" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <Failure error={error} onRetry={() => void ask(asked)} />
          </motion.div>
        ) : (
          <motion.div key="idle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
            <Idle docs={docs.data} recent={recent} onAsk={askAgain} />
          </motion.div>
        )}
      </AnimatePresence>
      <SourceDrawer source={open} onClose={() => setOpen(null)} />
    </div>
  );
}

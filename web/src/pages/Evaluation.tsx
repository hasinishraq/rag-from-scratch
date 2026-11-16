import NumberFlow from "@number-flow/react";
import { motion } from "motion/react";
import { type ReactNode, useEffect, useState } from "react";

import { EmptyState } from "@/components/EmptyState";
import { Select } from "@/components/ui/fields";
import { useResource } from "@/lib/api";
import type { EvalDetail, EvalRun } from "@/lib/types";
import { seconds } from "@/lib/utils";
import { Columns, Rows, TableView } from "@/pages/evaluation/charts";

const TARGETS: { key: string; label: string; target: number; higher: boolean; what: string }[] = [
  { key: "recall_at_5", label: "Recall at 5", target: 0.85, higher: true, what: "Right passage in the top 5." },
  { key: "mrr", label: "Mean reciprocal rank", target: 0.7, higher: true, what: "1 when the right passage is first, ½ when second, and so on." },
  { key: "faithfulness", label: "Faithful answers", target: 0.9, higher: true, what: "Every claim supported by a cited passage." },
  { key: "answer_correctness", label: "Correct answers", target: 0.8, higher: true, what: "Agrees with the reference answer." },
  { key: "refusal_accuracy", label: "Correct refusals", target: 0.9, higher: true, what: "Unanswerable questions refused." },
  { key: "false_refusal_rate", label: "False refusals", target: 0.05, higher: false, what: "Answerable questions refused." },
];
const GENERATION_ONLY = new Set(["faithfulness", "answer_correctness", "refusal_accuracy", "false_refusal_rate"]);
const LATENCY_TARGET_MS = 1500; // ENG-RAG-001 NFR-03: retrieval and reranking under 1.5 s

const pct = (v: number) => `${Math.round(v * 1000) / 10}%`;

function Section({ title, note, children }: { title: string; note?: ReactNode; children: ReactNode }) {
  return (
    <section className="flex flex-col gap-3 border-t border-ink pt-6">
      <div>
        <h2 className="text-lg font-semibold">{title}</h2>
        {note ? <p className="max-w-[70ch] text-sm text-graphite">{note}</p> : null}
      </div>
      {children}
    </section>
  );
}

function Score({ t, d, i }: { t: (typeof TARGETS)[number]; d: EvalDetail; i: number }) {
  const value = d.metrics[t.key] ?? null;
  const pending = value === null || (d.retrieval_only && GENERATION_ONLY.has(t.key));
  const err = d.stderr[t.key];
  const prev = d.previous?.metrics[t.key];
  const met = !pending && value !== null && (t.higher ? value >= t.target : value <= t.target);
  return (
    <div className="flex flex-col gap-2 border-b border-rule py-4">
      <div className="flex items-baseline justify-between gap-4">
        <div>
          <div className="text-base font-medium">{t.label}</div>
          <div className="text-sm text-graphite">{t.what}</div>
        </div>
        <div className="text-right">
          <div className="text-3xl font-semibold leading-none [font-variant-numeric:proportional-nums]">
            {pending ? (
              "–"
            ) : t.key === "mrr" ? (
              <NumberFlow value={value ?? 0} format={{ minimumFractionDigits: 3, maximumFractionDigits: 3 }} />
            ) : (
              <NumberFlow value={Math.round((value ?? 0) * 1000) / 10} suffix="%" />
            )}
          </div>
          {!pending && err ? <div className="text-xs text-graphite">± {Math.round(err * 1000) / 10} points</div> : null}
        </div>
      </div>
      <div className="relative h-2.5 bg-wash">
        {!pending ? (
          <motion.div
            className="absolute inset-y-0 left-0 bg-ink"
            initial={{ width: 0 }}
            animate={{ width: `${(value ?? 0) * 100}%` }}
            transition={{ duration: 0.7, delay: 0.1 + i * 0.06, ease: [0.22, 1, 0.36, 1] }}
          />
        ) : null}
        <div className="absolute -inset-y-1 w-0.5 bg-ink" style={{ left: `${t.target * 100}%` }} title={`Target ${pct(t.target)}`} />
      </div>
      <div className="flex flex-wrap justify-between gap-2 text-sm text-graphite">
        <span>
          Target {t.higher ? "at least" : "at most"} {t.key === "mrr" ? t.target.toFixed(2) : pct(t.target)}
          {pending ? "" : met ? ". Met." : ". Not met yet."}
        </span>
        {pending ? (
          <span>Needs a run with generated answers.</span>
        ) : prev != null && value !== null ? (
          <span>
            {value === prev
              ? "Same as"
              : `${value > prev ? "Up" : "Down"} ${t.key === "mrr" ? Math.abs(value - prev).toFixed(3) : `${Math.abs(Math.round((value - prev) * 1000) / 10)} points`} from`}{" "}
            the previous run
          </span>
        ) : null}
      </div>
    </div>
  );
}

function Overview({ d }: { d: EvalDetail }) {
  const items = [
    { label: "Questions", value: d.n, note: `${d.n_answerable} answerable, ${d.n_unanswerable} unanswerable` },
    { label: "Right passage first", value: d.rank_hist[0]?.count ?? 0, note: `of ${d.n_scored} answerable questions` },
    { label: "Not in the top 5", value: d.misses.length, note: "listed at the end of this page" },
    { label: "Typical time", value: (d.latency.p50 ?? 0) / 1000, suffix: " s", note: d.retrieval_only ? "per question, search only" : "per question, with the answer" },
  ];
  return (
    <dl className="grid grid-cols-2 border-t border-ink lg:grid-cols-4">
      {items.map((it) => (
        <div key={it.label} className="flex flex-col gap-1.5 border-b border-rule py-4 pr-6">
          <dt className="text-sm font-medium">{it.label}</dt>
          <dd className="text-3xl font-semibold leading-none [font-variant-numeric:proportional-nums]">
            <NumberFlow
              value={it.value}
              suffix={it.suffix}
              format={it.suffix ? { minimumFractionDigits: 1, maximumFractionDigits: 1 } : undefined}
            />
          </dd>
          <dd className="text-sm text-graphite">{it.note}</dd>
        </div>
      ))}
    </dl>
  );
}

function TopicTable({ d }: { d: EvalDetail }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-graphite">
          <tr>
            <th className="py-2 pr-4 font-medium">Topic</th>
            <th className="py-2 pr-4 text-right font-medium">Questions</th>
            <th className="py-2 pr-4 font-medium">Recall at 5 (line: 85% target)</th>
            <th className="py-2 pr-4 text-right font-medium">MRR</th>
            <th className="py-2 pr-4 text-right font-medium">Missed</th>
            <th className="py-2 text-right font-medium">Unanswerable</th>
          </tr>
        </thead>
        <tbody>
          {d.by_topic.map((t) => (
            <tr key={t.topic} className="border-t border-rule">
              <td className="py-2.5 pr-4">{t.topic}</td>
              <td className="py-2.5 pr-4 text-right [font-variant-numeric:proportional-nums]">{t.n}</td>
              <td className="py-2.5 pr-4">
                <div className="flex items-center gap-3">
                  <div className="relative h-2 w-40 bg-wash">
                    <motion.div
                      className="absolute inset-y-0 left-0 bg-ink"
                      initial={{ width: 0 }}
                      animate={{ width: `${t.recall_at_5 * 100}%` }}
                      transition={{ duration: 0.6 }}
                    />
                    <div className="absolute -inset-y-1 w-0.5 bg-graphite" style={{ left: "85%" }} />
                  </div>
                  <span className="[font-variant-numeric:proportional-nums]">{pct(t.recall_at_5)}</span>
                </div>
              </td>
              <td className="py-2.5 pr-4 text-right [font-variant-numeric:proportional-nums]">{t.mrr.toFixed(3)}</td>
              <td className="py-2.5 pr-4 text-right [font-variant-numeric:proportional-nums]">{t.misses}</td>
              <td className="py-2.5 text-right [font-variant-numeric:proportional-nums]">{t.unanswerable}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Latency({ d }: { d: EvalDetail }) {
  const lat = d.latency;
  const bars = lat.hist.map((b) => ({
    key: String(b.from_ms),
    label: b.overflow ? `over ${seconds(b.from_ms)}` : seconds(b.from_ms),
    value: b.count,
    detail: b.overflow
      ? `${b.count} slower questions took ${seconds(b.from_ms)} to ${seconds(b.to_ms)}`
      : `${b.count} questions took ${seconds(b.from_ms)} to ${seconds(b.to_ms)}`,
  }));
  const peak = Math.max(...lat.hist.map((b) => b.count), 1);
  const median = lat.p50 ?? 0;
  const verdict = d.retrieval_only
    ? `Median ${seconds(median)} against the 1.5 s target for search and ranking: ${median <= LATENCY_TARGET_MS ? "met" : "not met"}.`
    : `Median ${seconds(median)} including the generated answer.`;
  return (
    <Section title="Time per question" note={`${lat.n} questions. ${verdict}`}>
      <Columns data={bars} max={peak} height={140} ariaLabel="Number of questions by time taken" />
      <dl className="grid grid-cols-3 gap-y-3 text-sm sm:grid-cols-6">
        {(["min", "p50", "p90", "p95", "p99", "max"] as const).map((k) => (
          <div key={k}>
            <dt className="text-graphite">
              {{ min: "Fastest", p50: "Median", p90: "90% under", p95: "95% under", p99: "99% under", max: "Slowest" }[k]}
            </dt>
            <dd className="text-base font-semibold [font-variant-numeric:proportional-nums]">{seconds(lat[k])}</dd>
          </div>
        ))}
      </dl>
      <TableView head={["From", "To", "Questions"]} rows={lat.hist.map((b) => [seconds(b.from_ms), seconds(b.to_ms), b.count])} />
    </Section>
  );
}

function Answers({ d }: { d: EvalDetail }) {
  const g = d.generation;
  if (!g) return null; // a search-only run has no answers to show
  const cells: [string, string, string][] = [
    ["Answered", `${g.answered}`, `${g.errors} failed`],
    ["Correct refusals", `${g.correct_refusals} of ${g.unanswerable}`, "unanswerable questions refused"],
    ["False refusals", `${g.false_refusals} of ${g.answerable}`, "answerable questions refused"],
    ["Judged faithful", `${g.faithful} of ${g.judged_faithful}`, `${g.correct} of ${g.judged_correct} judged correct`],
    ["Tokens", g.tokens.toLocaleString(), Object.entries(g.providers).map(([p, n]) => `${p}: ${n}`).join(", ")],
  ];
  return (
    <Section title="Answers">
      <dl className="grid grid-cols-2 gap-6 text-sm md:grid-cols-5">
        {cells.map(([label, value, note]) => (
          <div key={label}>
            <dt className="font-medium">{label}</dt>
            <dd className="text-2xl font-semibold">{value}</dd>
            <dd className="text-graphite">{note}</dd>
          </div>
        ))}
      </dl>
    </Section>
  );
}

function Misses({ d }: { d: EvalDetail }) {
  return (
    <Section
      title={`Questions where the right passage was not in the top 5 (${d.misses.length})`}
      note={
        d.misses.length
          ? `${d.misses_same_document} of ${d.misses.length} ranked another section of the right document first, so the search found the document but ranked the wrong part of it. The other ${d.misses.length - d.misses_same_document} found a different document.`
          : undefined
      }
    >
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-graphite">
            <tr>
              <th className="py-2 pr-4 font-medium">Question</th>
              <th className="py-2 pr-4 font-medium">Rank</th>
              <th className="py-2 pr-4 font-medium">Ranked first instead</th>
              <th className="py-2 font-medium">The right passage</th>
            </tr>
          </thead>
          <tbody>
            {d.misses.map((m) => (
              <tr key={m.id} className="border-t border-rule align-top">
                <td className="max-w-xs py-2.5 pr-4">
                  {m.question}
                  {m.topic ? <div className="text-xs text-graphite">{m.topic}</div> : null}
                </td>
                <td className="py-2.5 pr-4 [font-variant-numeric:proportional-nums]">{m.rank ?? "not returned"}</td>
                <td className="max-w-xs py-2.5 pr-4 text-graphite">{m.first_title ?? "–"}</td>
                <td className="max-w-xs py-2.5 text-graphite">{m.gold_title ?? "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

function Detail({ d }: { d: EvalDetail }) {
  const recall = d.recall_at_k.map((p) => ({
    key: String(p.k),
    label: `Top ${p.k}`,
    value: p.value,
    detail: `Top ${p.k}: the right passage was found for ${pct(p.value)} of questions`,
  }));
  const ranks = d.rank_hist.map((h) => ({
    key: h.rank,
    label: h.rank === "not found" ? "Not found" : `Rank ${h.rank}`,
    value: h.count,
    text: `${h.count} (${pct(h.share)})`,
    detail:
      h.rank === "not found"
        ? `${h.count} questions: the right passage was not among the ${d.depth} returned`
        : `${h.count} questions had the right passage at rank ${h.rank}`,
  }));
  const maxRank = Math.max(...d.rank_hist.map((h) => h.count), 1);
  return (
    <div className="flex flex-col gap-10">
      <Overview d={d} />
      <Section title="Scores against the targets" note="± is the standard error: with this many questions, smaller differences between runs are not conclusive.">
        <div>
          {TARGETS.map((t, i) => (
            <Score key={t.key} t={t} d={d} i={i} />
          ))}
        </div>
      </Section>
      <div className="grid gap-10 lg:grid-cols-2">
        <Section title="How often the right passage is in the top k" note={`Share of the ${d.n_scored} answerable questions whose right passage is within the first k results.`}>
          <Columns data={recall} max={1} highlight={d.recall_at_k.map((p) => String(p.k))} ariaLabel="Recall at k from 1 to the number of passages returned" />
          <TableView head={["Top k", "Found"]} rows={d.recall_at_k.map((p) => [p.k, pct(p.value)])} />
        </Section>
        <Section title="Where the right passage ranked" note="Rank of the first right passage among the passages returned for each question.">
          <Rows data={ranks} max={maxRank} ariaLabel="Number of questions by rank of the right passage" />
          <TableView head={["Rank", "Questions", "Share"]} rows={d.rank_hist.map((h) => [h.rank, h.count, pct(h.share)])} />
        </Section>
      </div>
      <Section title="By topic" note="Weak topics show where retrieval needs work.">
        <TopicTable d={d} />
      </Section>
      <Latency d={d} />
      <Answers d={d} />
      <Misses d={d} />
      {d.previous ? (
        <Section title={`Changed since run ${d.previous.run} (${d.changes.length})`}>
          <ul className="flex flex-col text-sm">
            {d.changes.length ? (
              d.changes.map((c) => (
                <li key={c} className="border-t border-rule py-1.5 [font-variant-numeric:proportional-nums]">
                  {c}
                </li>
              ))
            ) : (
              <li>No question changed.</li>
            )}
          </ul>
        </Section>
      ) : null}
    </div>
  );
}

function runLabel(r: EvalRun): string {
  const s = r.run;
  const when = `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)} ${s.slice(9, 11)}:${s.slice(11, 13)}`;
  return `${when}, ${r.split} split, ${r.retrieval_only ? "search only" : "with answers"}`;
}

export function EvaluationPage() {
  const runs = useResource<EvalRun[]>("/eval/runs");
  const [selected, setSelected] = useState("");
  useEffect(() => {
    if (!selected && runs.data?.length) setSelected(runs.data[0].run);
  }, [runs.data, selected]);
  const detail = useResource<EvalDetail>(selected ? `/eval/runs/${selected}` : null);
  const d = detail.data;
  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-8">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <h1 className="page-title">Evaluation</h1>
        {runs.data?.length ? (
          <Select
            className="w-[24rem] max-w-full"
            value={selected}
            onValueChange={setSelected}
            options={runs.data.map((r) => ({ value: r.run, label: runLabel(r) }))}
          />
        ) : null}
      </header>
      {runs.data && !runs.data.length ? (
        <EmptyState>
          No runs yet. Run <code className="text-ink">make dataset-eval</code> (no quota) or{" "}
          <code className="text-ink">uv run python scripts/run_eval.py</code>.
        </EmptyState>
      ) : null}
      {d ? (
        <Detail d={d} />
      ) : detail.error ? (
        <p>{detail.error.message}</p>
      ) : null}
    </div>
  );
}

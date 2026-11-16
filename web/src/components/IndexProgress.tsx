import NumberFlow from "@number-flow/react";
import { motion } from "motion/react";
import { useEffect, useMemo, useRef } from "react";

import { Dither, revealField } from "@/components/Dither";

export interface ProgressState {
  phase: string; // uploading | starting | scanning | reading | embedding | saving | done | failed
  percent: number; // 0..100
  current?: string;
  done?: number;
  total?: number;
  elapsed_s?: number;
  eta_s?: number | null;
}

function phaseText(p: ProgressState): string {
  const file = p.current ? p.current.split("/").pop() : "";
  switch (p.phase) {
    case "uploading":
      return `Uploading files (${p.done ?? 0} of ${p.total ?? 0})`;
    case "starting":
    case "scanning":
      return "Looking for new and changed files";
    case "reading":
      return `Reading ${file}`;
    case "embedding":
      return `Embedding ${file}`;
    case "saving":
      return "Saving the index";
    case "done":
      return "Indexed";
    default:
      return "Indexing";
  }
}

function duration(s: number): string {
  if (s < 60) return `${Math.max(1, Math.round(s))} s`;
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

/** Indexing progress: the dithered band fills with the real percentage, eased between updates. */
export function IndexProgress({ state }: { state: ProgressState }) {
  const target = useRef(state.percent / 100);
  const shown = useRef(0);
  target.current = state.percent / 100;
  const field = useMemo(
    () =>
      revealField(() => {
        // Ease toward the latest reported value so the band moves smoothly between polls.
        shown.current += (target.current - shown.current) * 0.12;
        return shown.current;
      }),
    [],
  );
  useEffect(() => {
    if (state.phase === "done") shown.current = 1;
  }, [state.phase]);

  const files = state.total ? `${Math.min(state.done ?? 0, state.total)} of ${state.total} files` : "";
  const timing =
    state.phase === "done"
      ? state.elapsed_s != null
        ? `took ${duration(state.elapsed_s)}`
        : ""
      : state.eta_s != null
        ? `about ${duration(state.eta_s)} left`
        : state.elapsed_s != null && state.elapsed_s > 2
          ? `${duration(state.elapsed_s)} so far`
          : "";

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      className="flex flex-col gap-4 rounded-2xl border border-ink p-6"
      role="status"
      aria-live="polite"
    >
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex min-w-0 flex-col gap-1">
          <span className="truncate text-lg font-semibold">{phaseText(state)}</span>
          <span className="text-sm text-graphite">{[files, timing].filter(Boolean).join(", ")}</span>
        </div>
        <span className="text-5xl font-semibold leading-none tracking-tight [font-variant-numeric:proportional-nums]">
          <NumberFlow value={Math.floor(state.percent)} suffix="%" />
        </span>
      </div>
      <div className="h-16 w-full overflow-hidden border border-ink">
        <Dither field={field} cell={4} fps={30} />
      </div>
    </motion.div>
  );
}

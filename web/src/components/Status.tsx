import { motion, useReducedMotion } from "motion/react";

import type { Stage, TestStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

export const STATUS_LABEL: Record<TestStatus, string> = {
  working: "Working",
  working_with_warning: "Working, with a warning",
  rate_limited: "Rate-limited",
  key_missing: "Key missing",
  key_rejected: "Key rejected",
  unreachable: "Unreachable",
  model_not_found: "Model not found",
  untested: "Untested",
};

export const STATUS_FIX: Record<TestStatus, string> = {
  working: "Ready. Used in Auto mode.",
  working_with_warning: "Answers, but is weak at citing or refusing. Keep it late in the order.",
  rate_limited: "The provider asked to slow down. Test again after the wait.",
  key_missing: "Paste a key and save, or set the environment variable in .env.",
  key_rejected: "The provider refused the key. Create a new one in its console.",
  unreachable: "No answer from the address. Check it, or start Ollama.",
  model_not_found: "Load the model list and pick an id from it.",
  untested: "Not tested since the settings changed. Skipped in Auto mode.",
};

/** Monochrome status glyphs: the shape carries the meaning, not a colour. */
export function StatusMark({ status, showLabel = true }: { status: TestStatus; showLabel?: boolean }) {
  const glyph = (() => {
    switch (status) {
      case "working":
        return <span className="block size-2.5 rounded-full bg-ink" />;
      case "working_with_warning":
        return (
          <span
            className="block size-2.5 rounded-full border border-ink"
            style={{ background: "linear-gradient(90deg, var(--ink) 50%, transparent 50%)" }}
          />
        );
      case "rate_limited":
        return <span className="block size-2.5 rounded-full border-2 border-ink border-t-transparent" />;
      case "untested":
        return <span className="block size-2.5 rounded-full border border-dashed border-ink" />;
      default:
        return (
          <span className="relative block size-2.5 border border-ink">
            <span className="absolute left-1/2 top-[-2px] h-[13px] w-px -translate-x-1/2 rotate-45 bg-ink" />
          </span>
        );
    }
  })();
  return (
    <span className="inline-flex items-center gap-2 text-sm">
      {glyph}
      {showLabel ? <span className={cn(status === "untested" && "text-graphite")}>{STATUS_LABEL[status]}</span> : null}
    </span>
  );
}

const OUTCOME_MARK: Record<Stage["outcome"], string> = {
  passed: "bg-ink",
  warning: "bg-ink/40",
  failed: "bg-paper border border-ink",
  not_checked: "bg-wash",
  skipped: "bg-wash",
};

/** The five connection-test stages, revealed one after another as a real sequence. */
export function StageTrack({ stages }: { stages: Stage[] }) {
  const reduce = useReducedMotion();
  const names = ["Settings complete", "Server reachable", "Model available", "Generates a reply", "Ready for RAG"];
  return (
    <ol className="flex flex-col gap-3">
      {names.map((name, i) => {
        const stage = stages.find((s) => s.n === i + 1);
        return (
          <motion.li
            key={name}
            className="grid grid-cols-[1.25rem_1fr] gap-x-3"
            initial={reduce ? false : { opacity: 0, x: -6 }}
            animate={{ opacity: stage ? 1 : 0.35, x: 0 }}
            transition={{ delay: reduce ? 0 : i * 0.12, duration: 0.25 }}
          >
            <span
              className={cn(
                "mt-1 flex size-5 items-center justify-center rounded-full text-xs font-semibold",
                stage ? OUTCOME_MARK[stage.outcome] : "border border-dashed border-rule",
                stage?.outcome === "passed" && "text-paper",
              )}
            >
              {i + 1}
            </span>
            <div>
              <div className="text-base font-medium">{name}</div>
              <div className="text-sm text-graphite">
                {stage ? stage.detail : "Not reached."}
              </div>
            </div>
          </motion.li>
        );
      })}
    </ol>
  );
}

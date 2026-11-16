import NumberFlow from "@number-flow/react";
import { motion } from "motion/react";

import { bayerTexture } from "@/components/Dither";
import { seconds } from "@/lib/utils";

const STAGES = [
  { key: "retrieve_ms", label: "Finding passages", fill: "bg-ink" },
  { key: "rerank_ms", label: "Ranking them", fill: "hatch" },
  { key: "generate_ms", label: "Writing the answer", fill: "bg-graphite" },
] as const;

const DOTS = bayerTexture(0.5);
const HATCH = { maskImage: DOTS, WebkitMaskImage: DOTS };

/** Where the time of one answer went, as one bar split by stage. */
export function TimeBar({
  timings,
  tokensIn,
  tokensOut,
  tokensSaved,
  cached,
}: {
  timings: Record<string, number>;
  tokensIn: number;
  tokensOut: number;
  tokensSaved: number;
  cached: boolean;
}) {
  const total = Math.max(1, timings.total_ms ?? 0);
  const parts = STAGES.map((s) => ({ ...s, ms: timings[s.key] ?? 0 })).filter((s) => s.ms > 0);
  return (
    <div className="flex max-w-xl flex-col gap-2">
      <div className="flex h-2 w-full overflow-hidden bg-wash" role="img" aria-label="Time spent per stage">
        {parts.map((p, i) => (
          <motion.div
            key={p.key}
            className={p.fill === "hatch" ? "dither-fill bg-ink" : p.fill}
            style={p.fill === "hatch" ? HATCH : undefined}
            initial={{ width: 0 }}
            animate={{ width: `${(p.ms / total) * 100}%` }}
            transition={{ duration: 0.6, delay: 0.2 + i * 0.12, ease: [0.22, 1, 0.36, 1] }}
          />
        ))}
      </div>
      <dl className="flex flex-wrap gap-x-5 gap-y-1 text-sm">
        {parts.map((p) => (
          <div key={p.key} className="flex items-center gap-1.5">
            <span className={`inline-block size-2.5 ${p.fill === "hatch" ? "dither-fill bg-ink" : p.fill}`} style={p.fill === "hatch" ? HATCH : undefined} />
            <dt className="text-graphite">{p.label}</dt>
            <dd className="tabular-nums">{seconds(p.ms)}</dd>
          </div>
        ))}
      </dl>
      <p className="text-sm text-graphite">
        {cached ? (
          "Answered from the cache: no tokens used."
        ) : tokensIn + tokensOut > 0 ? (
          <>
            <NumberFlow value={tokensIn} /> tokens sent, <NumberFlow value={tokensOut} /> received
            {tokensSaved > 0 ? (
              <>
                , <NumberFlow value={tokensSaved} /> saved by leaving out weak and repeated text.
              </>
            ) : (
              "."
            )}
          </>
        ) : (
          "No model call, so no tokens used."
        )}
      </p>
    </div>
  );
}

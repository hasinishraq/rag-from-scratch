import NumberFlow from "@number-flow/react";
import { motion, useReducedMotion } from "motion/react";
import { useEffect, useState } from "react";

import { bayerTexture } from "@/components/Dither";
import { cn } from "@/lib/utils";

const DOTS = bayerTexture(0.5);

/**
 * The ranking bar. Solid ink means the source was cited in the answer; a dithered fill means
 * it was retrieved but not cited. Bars fill in rank order when a new answer arrives.
 */
export function RelevanceBar({
  value,
  rank,
  cited,
  size = "md",
}: {
  value: number;
  rank: number;
  cited: boolean;
  size?: "sm" | "md";
}) {
  const reduce = useReducedMotion();
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  const delay = reduce ? 0 : 0.12 + rank * 0.09;
  const [shown, setShown] = useState(reduce ? pct : 0);

  useEffect(() => {
    if (reduce) {
      setShown(pct);
      return;
    }
    setShown(0);
    const t = window.setTimeout(() => setShown(pct), delay * 1000);
    return () => window.clearTimeout(t);
  }, [pct, delay, reduce]);

  return (
    <div className="flex items-center gap-3" aria-label={`Relevance ${pct} percent`}>
      <div className={cn("relative flex-1 bg-wash", size === "md" ? "h-2.5" : "h-1.5")}>
        <motion.div
          className={cn("absolute inset-y-0 left-0 bg-ink", !cited && "dither-fill")}
          style={cited ? undefined : { maskImage: DOTS, WebkitMaskImage: DOTS }}
          initial={{ width: reduce ? `${pct}%` : "0%" }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: reduce ? 0 : 0.7, delay, ease: [0.22, 1, 0.36, 1] }}
        />
      </div>
      <span className="w-12 text-right text-sm font-semibold tabular-nums">
        <NumberFlow value={shown} suffix="%" />
      </span>
    </div>
  );
}

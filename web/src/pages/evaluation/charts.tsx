import { motion } from "motion/react";
import { type ReactNode, useState } from "react";

import { cn } from "@/lib/utils";

/** Charts for the Evaluation page: one series each, ink on paper, hover detail on every mark,
 * and a table view underneath so no value lives only in a shape. */

export interface Datum {
  key: string;
  label: string; // axis label
  value: number; // drives the mark
  detail: string; // tooltip and table text
  text?: string; // value printed at the end of a row
}

function Tip({ text }: { text: string | null }) {
  return (
    <div aria-live="polite" className={cn("h-5 text-sm text-graphite transition-opacity", text ? "opacity-100" : "opacity-0")}>
      {text ?? " "}
    </div>
  );
}

export function TableView({ head, rows }: { head: string[]; rows: ReactNode[][] }) {
  return (
    <details className="mt-3 text-sm">
      <summary className="cursor-pointer text-graphite hover:text-ink">Show as table</summary>
      <table className="mt-2 w-full max-w-xl text-left">
        <thead className="text-graphite">
          <tr>
            {head.map((h) => (
              <th key={h} className="py-1 pr-4 font-medium">{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-rule">
              {r.map((c, j) => (
                <td key={j} className="py-1 pr-4 [font-variant-numeric:proportional-nums]">{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

/** Vertical columns sharing one baseline. `max` fixes the scale (1 for rates). Optional
 * `marker` draws a reference line at a value on the same scale (for example a target). */
export function Columns({
  data,
  max,
  height = 160,
  highlight = [],
  marker,
  ariaLabel,
}: {
  data: Datum[];
  max: number;
  height?: number;
  highlight?: string[];
  marker?: { at: number; label: string };
  ariaLabel: string;
}) {
  const [tip, setTip] = useState<string | null>(null);
  const scale = max || 1;
  return (
    <div>
      <div className="relative" style={{ height }} role="img" aria-label={ariaLabel} onMouseLeave={() => setTip(null)}>
        <div className="absolute inset-x-0 bottom-0 h-px bg-ink" />
        {marker ? (
          <div className="absolute inset-x-0 border-t border-dashed border-graphite" style={{ bottom: `${(marker.at / scale) * 100}%` }}>
            <span className="absolute -top-5 right-0 bg-paper pl-1 text-xs text-graphite">{marker.label}</span>
          </div>
        ) : null}
        <div className="absolute inset-0 flex items-end gap-[2px]">
          {data.map((d, i) => {
            const h = Math.max(0, Math.min(1, d.value / scale)) * 100;
            const labelled = highlight.includes(d.key);
            return (
              <div
                key={d.key}
                className="group relative flex h-full flex-1 items-end"
                onMouseEnter={() => setTip(d.detail)}
                onFocus={() => setTip(d.detail)}
                tabIndex={0}
                aria-label={d.detail}
              >
                <motion.div
                  className="w-full bg-ink group-hover:bg-graphite"
                  initial={{ height: 0 }}
                  animate={{ height: `${h}%` }}
                  transition={{ duration: 0.6, delay: 0.05 + i * 0.04, ease: [0.22, 1, 0.36, 1] }}
                />
                {labelled ? (
                  <span className="absolute inset-x-0 text-center text-xs font-semibold [font-variant-numeric:proportional-nums]" style={{ bottom: `calc(${h}% + 4px)` }}>
                    {d.label === "" ? "" : formatShare(d.value, max)}
                  </span>
                ) : null}
              </div>
            );
          })}
        </div>
      </div>
      <div className="mt-1 flex gap-[2px] text-xs text-graphite">
        {data.map((d) => (
          <span key={d.key} className="flex-1 truncate text-center [font-variant-numeric:proportional-nums]">{d.label}</span>
        ))}
      </div>
      <Tip text={tip} />
    </div>
  );
}

function formatShare(value: number, max: number): string {
  return max === 1 ? `${Math.round(value * 100)}%` : value.toLocaleString();
}

/** One row per category: label, a bar on a shared scale, and the value as text. */
export function Rows({ data, max, ariaLabel }: { data: Datum[]; max: number; ariaLabel: string }) {
  const [tip, setTip] = useState<string | null>(null);
  return (
    <div role="img" aria-label={ariaLabel} onMouseLeave={() => setTip(null)}>
      <div className="flex flex-col gap-[2px]">
        {data.map((d, i) => (
          <div
            key={d.key}
            className="group grid grid-cols-[6rem_1fr_7rem] items-center gap-3 py-1"
            onMouseEnter={() => setTip(d.detail)}
            onFocus={() => setTip(d.detail)}
            tabIndex={0}
            aria-label={d.detail}
          >
            <span className="text-sm">{d.label}</span>
            <div className="relative h-3 bg-wash">
              <motion.div
                className="absolute inset-y-0 left-0 bg-ink group-hover:bg-graphite"
                initial={{ width: 0 }}
                animate={{ width: `${max ? (d.value / max) * 100 : 0}%` }}
                transition={{ duration: 0.6, delay: 0.05 + i * 0.05, ease: [0.22, 1, 0.36, 1] }}
              />
            </div>
            <span className="text-right text-sm [font-variant-numeric:proportional-nums]">{d.text ?? ""}</span>
          </div>
        ))}
      </div>
      <Tip text={tip} />
    </div>
  );
}

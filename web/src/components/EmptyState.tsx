import type { ReactNode } from "react";

import { Dither, wavesField } from "@/components/Dither";

/** An empty screen is an invitation to act: a still dithered texture beside what to do next. */
export function EmptyState({ children }: { children: ReactNode }) {
  return (
    <div className="flex items-center gap-6 border-y border-rule py-8">
      <div className="size-24 shrink-0 border border-ink">
        <Dither field={wavesField} cell={3} animate={false} />
      </div>
      <div className="max-w-[52ch] text-base text-graphite">{children}</div>
    </div>
  );
}

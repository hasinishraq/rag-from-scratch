import NumberFlow from "@number-flow/react";
import { motion } from "motion/react";
import { useEffect } from "react";

import { bayerTexture } from "@/components/Dither";
import { STATUS_LABEL, StatusMark } from "@/components/Status";
import { useResource } from "@/lib/api";
import type { Provider, ServerHealth } from "@/lib/types";

const TRACK_DOTS = bayerTexture(0.3);

/** What is left of one provider's daily requests: solid ink is left, the dotted track is used. */
function Meter({ left, limit }: { left: number; limit: number }) {
  const share = limit > 0 ? Math.max(0, Math.min(1, left / limit)) : 0;
  return (
    <div className="relative h-1.5 w-full" aria-hidden>
      <div className="dither-fill absolute inset-0 bg-graphite" style={{ maskImage: TRACK_DOTS, WebkitMaskImage: TRACK_DOTS }} />
      <motion.div
        className="absolute inset-y-0 left-0 bg-ink"
        initial={false}
        animate={{ width: `${share * 100}%` }}
        transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      />
    </div>
  );
}

function ProviderUsage({ p, health }: { p: Provider; health: ServerHealth }) {
  const left = health.quota[p.name]?.requests_left ?? null;
  const limit = p.limits.rpd ?? null;
  const metered = p.kind !== "ollama" && left !== null && limit !== null;
  const summary = metered
    ? `${left.toLocaleString()} of ${limit.toLocaleString()} requests left today`
    : p.kind === "ollama"
      ? "Runs on this computer, no daily limit"
      : "No daily limit set";
  return (
    <a
      href="#/models"
      title={`${p.name}: ${STATUS_LABEL[p.test_status]}. ${summary}.`}
      className="group flex flex-col gap-1.5 rounded-sm py-1"
    >
      <span className="flex items-center justify-between gap-3 text-sm">
        <span className="flex min-w-0 items-center gap-2">
          <StatusMark status={p.test_status} showLabel={false} />
          <span className="truncate font-medium underline-offset-4 group-hover:underline">{p.name}</span>
        </span>
        {metered ? (
          <span className="shrink-0 text-graphite">
            <NumberFlow className="font-semibold text-ink tabular-nums" value={left} /> left
          </span>
        ) : (
          <span className="shrink-0 text-graphite">{p.kind === "ollama" ? "local" : "no limit"}</span>
        )}
      </span>
      {metered ? <Meter left={left} limit={limit} /> : null}
    </a>
  );
}

/** Sidebar summary: today's remaining requests per provider, and the size of the index. */
export function UsageRail({ refreshKey, health }: { refreshKey: number; health: ServerHealth | null }) {
  const providers = useResource<Provider[]>("/providers");
  const { reload } = providers;
  useEffect(() => {
    if (refreshKey > 0) void reload();
  }, [refreshKey, reload]);
  const enabled = (providers.data ?? []).filter((p) => p.enabled);
  if (!health) return null;
  return (
    <div className="flex flex-col gap-5 border-t border-rule pt-5">
      <section aria-labelledby="usage-title" className="flex flex-col gap-2">
        <h2 id="usage-title" className="text-xs font-medium text-graphite">
          Requests today
        </h2>
        {enabled.length ? (
          enabled.map((p) => <ProviderUsage key={p.name} p={p} health={health} />)
        ) : (
          <a href="#/models" className="text-sm underline underline-offset-4">
            Turn on a provider
          </a>
        )}
      </section>
      <section aria-labelledby="index-title" className="flex flex-col gap-0.5 border-t border-rule pt-4">
        <h2 id="index-title" className="text-xs font-medium text-graphite">
          Index
        </h2>
        <a href="#/documents" className="group flex items-baseline gap-1.5">
          <NumberFlow className="text-xl font-semibold tracking-tight [font-variant-numeric:proportional-nums]" value={health.chunks} />
          <span className="text-sm text-graphite underline-offset-4 group-hover:underline">passages</span>
        </a>
      </section>
    </div>
  );
}

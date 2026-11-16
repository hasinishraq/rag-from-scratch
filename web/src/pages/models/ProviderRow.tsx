import NumberFlow from "@number-flow/react";
import { ArrowDown, ArrowUp } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { toast } from "sonner";

import { STATUS_FIX, StageTrack, StatusMark } from "@/components/Status";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/fields";
import { api, errorMessage } from "@/lib/api";
import type { Provider, TestResult } from "@/lib/types";
import { formatTime, seconds } from "@/lib/utils";

function UsageBar({ label, used, limit }: { label: string; used: number; limit: number | null | undefined }) {
  const pct = limit ? Math.min(100, (used / limit) * 100) : 0;
  return (
    <div className="flex min-w-40 flex-1 flex-col gap-1">
      <div className="flex justify-between text-xs text-graphite">
        <span>{label}</span>
        <span className="tabular-nums text-ink">
          <NumberFlow value={used} />
          {limit ? ` of ${limit.toLocaleString()}` : ""}
        </span>
      </div>
      <div className="relative h-1.5 bg-wash">
        <motion.div className="absolute inset-y-0 left-0 bg-ink" initial={false} animate={{ width: `${pct}%` }} transition={{ duration: 0.5 }} />
      </div>
    </div>
  );
}

export function ProviderRow({
  p,
  index,
  count,
  result,
  onResult,
  onChanged,
  onEdit,
}: {
  p: Provider;
  index: number;
  count: number;
  result: TestResult | undefined;
  onResult: (r: TestResult) => void;
  onChanged: () => void;
  onEdit: () => void;
}) {
  const [testing, setTesting] = useState(false);
  const [confirm, setConfirm] = useState(false);

  async function update(body: Record<string, unknown>) {
    try {
      await api.put(`/providers/${p.name}`, body);
      onChanged();
    } catch (e) {
      toast.error(errorMessage(e));
    }
  }

  async function test() {
    setTesting(true);
    try {
      const r = await api.post<TestResult>(`/providers/${p.name}/test`);
      onResult(r);
      onChanged();
      if (r.status === "working") toast.success(`${p.name} is working`);
      else toast(`${p.name}: ${r.detail}`);
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setTesting(false);
    }
  }

  async function remove() {
    try {
      await api.del(`/providers/${p.name}`);
      toast.success(`Removed ${p.name} and its saved key`);
      onChanged();
    } catch (e) {
      toast.error(errorMessage(e));
    }
  }

  const local = p.kind === "ollama";
  return (
    <motion.li layout transition={{ type: "spring", bounce: 0.1, duration: 0.45 }} className="border-t border-ink py-6">
      <div className="grid grid-cols-[2.5rem_1fr] gap-x-4 md:grid-cols-[3rem_1fr_auto]">
        <span className={`dot text-[2.75rem] font-black leading-none ${p.enabled ? "" : "text-rule"}`}>{index + 1}</span>
        <div className="flex min-w-0 flex-col gap-2">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
            <h3 className="text-xl font-semibold">{p.name}</h3>
            <span className="text-base text-graphite">{p.model || "no model chosen"}</span>
            <span className="text-sm text-graphite">{local ? "runs on this computer" : "free hosted API"}</span>
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
            <StatusMark status={p.test_status} />
            <span className="text-sm text-graphite">
              {p.tested_at ? `Tested ${formatTime(p.tested_at)}` : "Never tested"}
              {p.test_latency_ms ? `, replied in ${seconds(p.test_latency_ms)}` : ""}
            </span>
          </div>
          <p className="line-clamp-2 max-w-[65ch] text-sm text-graphite" title={p.test_detail}>
            {p.test_detail || STATUS_FIX[p.test_status]}
          </p>
          {p.health ? (
            <p className="text-sm">
              {p.health.paused ? (
                <span className="font-medium">
                  Paused for {p.health.paused_for_s} s ({p.health.pause_reason}). Tried only if nothing else answers.
                </span>
              ) : (
                <span className="text-graphite">
                  {p.health.avg_latency_ms != null ? `Answers in ${seconds(p.health.avg_latency_ms)} on average. ` : ""}
                  {p.health.successes} answered, {p.health.failures} failed since the server started.
                </span>
              )}
            </p>
          ) : null}
          {!local && !p.key_set ? (
            <p className="text-sm">No key saved. Open Edit to paste one, or set {p.api_key_env || "an environment variable"} in .env.</p>
          ) : null}
          {!p.free ? <p className="text-sm font-medium">Not confirmed as free, so it will never be called.</p> : null}
          {local ? (
            <p className="text-sm text-graphite">No quota: limited only by this computer's speed.</p>
          ) : (
            <div className="mt-1 flex flex-wrap gap-6">
              <UsageBar label="Requests today" used={p.quota.requests_today} limit={p.limits.rpd} />
              <UsageBar label="Tokens today" used={p.quota.tokens_today} limit={p.limits.tpd} />
            </div>
          )}
        </div>
        <div className="col-span-2 mt-4 flex flex-wrap items-center gap-2 md:col-span-1 md:mt-0 md:flex-col md:items-end">
          <label className="flex items-center gap-2 text-sm">
            {p.enabled ? "On" : "Off"}
            <Switch checked={p.enabled} onCheckedChange={(v) => void update({ enabled: v })} aria-label={`Use ${p.name}`} />
          </label>
          <div className="flex gap-1">
            <Button size="icon" variant="ghost" disabled={index === 0} aria-label="Try earlier" onClick={() => void update({ position: index - 1 })}>
              <ArrowUp />
            </Button>
            <Button size="icon" variant="ghost" disabled={index === count - 1} aria-label="Try later" onClick={() => void update({ position: index + 1 })}>
              <ArrowDown />
            </Button>
          </div>
          <div className="flex gap-2">
            <Button size="sm" onClick={() => void test()} disabled={testing}>
              {testing ? "Testing" : "Test"}
            </Button>
            <Button size="sm" variant="outline" onClick={onEdit}>
              Edit
            </Button>
          </div>
          {!local ? (
            confirm ? (
              <span className="flex items-center gap-2 text-sm">
                Remove and delete its key?
                <Button size="sm" variant="outline" onClick={() => void remove()}>Remove</Button>
                <Button size="sm" variant="quiet" onClick={() => setConfirm(false)}>Keep</Button>
              </span>
            ) : (
              <Button size="sm" variant="quiet" onClick={() => setConfirm(true)}>Remove</Button>
            )
          ) : null}
        </div>
      </div>
      <AnimatePresence>
        {testing || result ? (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden pl-14 md:pl-16"
          >
            <div className="mt-5 flex flex-col gap-3 rounded-xl border border-rule p-5">
              {testing ? (
                <p className="shimmer-text text-base font-medium">Running the five-stage connection test</p>
              ) : result ? (
                <>
                  <StageTrack stages={result.stages} />
                  <p className="text-sm text-graphite">
                    Used {result.requests_used} requests and {result.tokens_used} tokens.
                    {result.retry_after_s ? ` Try again in ${Math.ceil(result.retry_after_s)} s.` : ""}
                  </p>
                </>
              ) : null}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </motion.li>
  );
}

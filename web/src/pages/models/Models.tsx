import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { api, errorMessage, useResource } from "@/lib/api";
import type { Mode, Preset, Provider, Settings, TestResult } from "@/lib/types";
import { cn } from "@/lib/utils";
import { OllamaForm, ProviderForm } from "@/pages/models/ProviderForm";
import { ProviderRow } from "@/pages/models/ProviderRow";

const MODES: { value: Mode; label: string; text: string }[] = [
  { value: "auto", label: "Auto", text: "Free APIs first, local model as backup." },
  { value: "free_api_only", label: "Free APIs only", text: "Hosted providers only." },
  { value: "ollama_only", label: "Local only", text: "Nothing leaves this computer." },
];

export function ModelsPage({
  refreshKey,
  onChanged,
  onTestAll,
}: {
  refreshKey: number;
  onChanged: () => void;
  onTestAll: () => void;
}) {
  const providers = useResource<Provider[]>("/providers");
  const settings = useResource<Settings>("/settings");
  const presets = useResource<Preset[]>("/providers/presets");
  const [results, setResults] = useState<Record<string, TestResult>>({});
  const [editing, setEditing] = useState<Provider | null>(null);
  const [adding, setAdding] = useState(false);

  const { reload } = providers;
  useEffect(() => {
    if (refreshKey > 0) void reload();
  }, [refreshKey, reload]);

  const changed = () => {
    void providers.reload();
    onChanged();
  };

  async function setMode(mode: Mode) {
    try {
      await api.put("/settings/mode", { mode });
      await settings.reload();
      toast.success(`Answers now come from: ${MODES.find((m) => m.value === mode)?.label}`);
      onChanged();
    } catch (e) {
      toast.error(errorMessage(e));
    }
  }

  async function setStrategy(strategy: "ordered" | "fastest") {
    try {
      await api.put("/settings/routing", { strategy });
      await settings.reload();
      toast.success(strategy === "fastest" ? "Fastest healthy provider goes first" : "Providers are tried in your order");
    } catch (e) {
      toast.error(errorMessage(e));
    }
  }

  const saved = (name: string, result?: TestResult) => {
    if (result) setResults((r) => ({ ...r, [name]: result }));
    changed();
  };
  const list = providers.data ?? [];

  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-12">
      <h1 className="page-title">Models</h1>

      <section aria-labelledby="mode-title" className="flex flex-col gap-4">
        <h2 id="mode-title" className="text-lg font-semibold">Where answers come from</h2>
        <div role="radiogroup" className="grid gap-3 md:grid-cols-3">
          {MODES.map((m) => {
            const active = settings.data?.mode === m.value;
            return (
              <button
                key={m.value}
                type="button"
                role="radio"
                aria-checked={active}
                onClick={() => void setMode(m.value)}
                className={cn(
                  "relative flex flex-col gap-1 rounded-xl border p-5 text-left transition-colors",
                  active ? "border-ink text-paper" : "border-rule hover:border-ink",
                )}
              >
                {active ? <motion.span layoutId="mode-card" className="absolute inset-0 rounded-xl bg-ink" transition={{ type: "spring", bounce: 0.15, duration: 0.45 }} /> : null}
                <span className="relative text-lg font-semibold">{m.label}</span>
                <span className={cn("relative text-sm", active ? "text-paper/75" : "text-graphite")}>{m.text}</span>
              </button>
            );
          })}
        </div>
      </section>

      <section aria-labelledby="routing-title" className="flex flex-col gap-3">
        <h2 id="routing-title" className="text-lg font-semibold">How a provider is picked</h2>
        <div role="radiogroup" className="flex flex-col gap-2 md:flex-row md:gap-6">
          {(
            [
              ["ordered", "Your order", "Provider 1 first, then down the list."],
              ["fastest", "Fastest first", "The quickest provider first."],
            ] as const
          ).map(([value, label, text]) => {
            const active = (settings.data?.routing_strategy ?? "ordered") === value;
            return (
              <button
                key={value}
                type="button"
                role="radio"
                aria-checked={active}
                onClick={() => void setStrategy(value)}
                className="flex items-start gap-3 text-left"
              >
                <span className={cn("mt-1 flex size-4 shrink-0 items-center justify-center rounded-full border border-ink")}>
                  {active ? <motion.span layoutId="strategy-dot" className="size-2 rounded-full bg-ink" /> : null}
                </span>
                <span className="flex flex-col">
                  <span className="text-base font-medium">{label}</span>
                  <span className="text-sm text-graphite">{text}</span>
                </span>
              </button>
            );
          })}
        </div>
      </section>

      <section aria-labelledby="order-title" className="flex flex-col">
        <div className="mb-4 flex flex-wrap items-end justify-between gap-4">
          <div>
            <h2 id="order-title" className="text-lg font-semibold">Providers, in the order they are tried</h2>
            <p className="text-sm text-graphite">Untested providers are skipped in Auto.</p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={onTestAll} title={`A test uses ${settings.data?.test_cost ?? "3 requests"} per provider`}>
              Test all
            </Button>
            <Button onClick={() => setAdding(true)}>Add a free API provider</Button>
          </div>
        </div>
        {providers.error ? <p className="text-base">{providers.error.message}</p> : null}
        <ol>
          {list.map((p, i) => (
            <ProviderRow
              key={p.name}
              p={p}
              index={i}
              count={list.length}
              result={results[p.name]}
              onResult={(r) => setResults((prev) => ({ ...prev, [p.name]: r }))}
              onChanged={changed}
              onEdit={() => setEditing(p)}
            />
          ))}
        </ol>
      </section>

      <section aria-labelledby="embed-title" className="flex flex-col gap-3 border-t border-rule pt-8">
        <h2 id="embed-title" className="text-lg font-semibold">Search models on this computer</h2>
        <p className="max-w-[65ch] text-base text-graphite">
          Passages are embedded by {settings.data?.embedding_model ?? "the built-in model"} and reranked by{" "}
          {settings.data?.reranker ?? "nothing (reranking is off)"}. Both run locally with no quota. The index was built
          with {settings.data?.indexed_with ?? "nothing yet"}; changing the embedder means re-indexing every document.
        </p>
      </section>

      <ProviderForm
        open={adding || (editing !== null && editing.kind !== "ollama")}
        provider={adding ? null : editing}
        presets={presets.data ?? []}
        onClose={() => {
          setAdding(false);
          setEditing(null);
        }}
        onSaved={saved}
      />
      <OllamaForm open={editing?.kind === "ollama"} provider={editing} onClose={() => setEditing(null)} onSaved={saved} />
    </div>
  );
}

import { useEffect, useState } from "react";
import { toast } from "sonner";
import { Drawer } from "vaul";

import { Button } from "@/components/ui/button";
import { Field, Input, Select, Switch } from "@/components/ui/fields";
import { api, errorMessage } from "@/lib/api";
import type { Limits, OllamaModel, Preset, Provider, TestResult } from "@/lib/types";

const LIMITS: { key: keyof Limits; label: string }[] = [
  { key: "rpm", label: "Requests per minute" },
  { key: "rpd", label: "Requests per day" },
  { key: "tpm", label: "Tokens per minute" },
  { key: "tpd", label: "Tokens per day" },
];

function SidePanel({ open, onClose, title, children }: { open: boolean; onClose: () => void; title: string; children: React.ReactNode }) {
  return (
    <Drawer.Root open={open} onOpenChange={(o) => !o && onClose()} direction="right">
      <Drawer.Portal>
        <Drawer.Overlay className="fixed inset-0 z-40 bg-ink/30" />
        <Drawer.Content className="fixed inset-y-0 right-0 z-50 flex w-[min(34rem,100vw)] flex-col border-l border-ink bg-paper outline-none">
          <div className="flex items-center justify-between border-b border-rule px-7 py-5">
            <Drawer.Title className="text-xl font-semibold">{title}</Drawer.Title>
            <Button size="sm" variant="quiet" onClick={onClose}>Close</Button>
          </div>
          <Drawer.Description className="sr-only">{title}</Drawer.Description>
          <div className="flex-1 overflow-y-auto px-7 py-6">{children}</div>
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}

/** Add (provider = null) or edit a free OpenAI-compatible provider. */
export function ProviderForm({
  open,
  provider,
  presets,
  onClose,
  onSaved,
}: {
  open: boolean;
  provider: Provider | null;
  presets: Preset[];
  onClose: () => void;
  onSaved: (name: string, result?: TestResult) => void;
}) {
  const [preset, setPreset] = useState("");
  const [name, setName] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [model, setModel] = useState("");
  const [models, setModels] = useState<string[]>([]);
  const [key, setKey] = useState("");
  const [keyEnv, setKeyEnv] = useState("");
  const [limits, setLimits] = useState<Limits>({});
  const [free, setFree] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!open) return;
    setName(provider?.name ?? "");
    setBaseUrl(provider?.base_url ?? "");
    setModel(provider?.model ?? "");
    setKeyEnv(provider?.api_key_env ?? "");
    setLimits(provider?.limits ?? {});
    setFree(provider?.free ?? false);
    setKey("");
    setModels([]);
    setPreset("");
  }, [open, provider]);

  const applyPreset = (label: string) => {
    setPreset(label);
    const p = presets.find((x) => x.label === label);
    if (!p) return;
    setName(p.name);
    setBaseUrl(p.base_url);
    setKeyEnv(p.api_key_env);
    setLimits(p.limits ?? {});
  };

  async function loadModels() {
    if (!provider) return;
    try {
      const r = await api.get<{ models: string[]; listed?: boolean }>(`/providers/${provider.name}/models`);
      setModels(r.models);
      toast(r.models.length ? `${r.models.length} models offered` : "This provider has no model list; type the id.");
    } catch (e) {
      toast.error(errorMessage(e));
    }
  }

  async function save(andTest: boolean) {
    setBusy(true);
    const body: Record<string, unknown> = { base_url: baseUrl.trim(), model: model.trim(), api_key_env: keyEnv.trim(), limits, free };
    if (key.trim()) body.api_key = key.trim();
    try {
      const saved = provider
        ? await api.put<Provider>(`/providers/${provider.name}`, body)
        : await api.post<Provider>("/providers", { name: name.trim(), ...body });
      toast.success(provider ? `Saved ${saved.name}` : `Added ${saved.name}`);
      const result = andTest ? await api.post<TestResult>(`/providers/${saved.name}/test`) : undefined;
      onSaved(saved.name, result);
      onClose();
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  const preset_ = presets.find((p) => p.label === preset);
  return (
    <SidePanel open={open} onClose={onClose} title={provider ? `Edit ${provider.name}` : "Add a free API provider"}>
      <div className="flex flex-col gap-5">
        {!provider ? (
          <Field label="Start from" hint={preset_?.note ?? "Fills the address and limits with the reported free-tier values. Check them on the provider's page."}>
            <Select value={preset} onValueChange={applyPreset} placeholder="Choose a provider" options={presets.map((p) => ({ value: p.label, label: p.label }))} />
          </Field>
        ) : null}
        <Field label="Name">
          <Input value={name} onChange={(e) => setName(e.target.value)} disabled={provider !== null} placeholder="for example groq" />
        </Field>
        <Field label="Address" hint="The provider's OpenAI-compatible base URL.">
          <Input value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)} placeholder="https://" />
        </Field>
        <Field label="Model id" hint={provider ? "Copy it from the provider console, or load the list." : "Save first to load the provider's model list."}>
          {models.length ? (
            <Select value={model} onValueChange={setModel} options={models.map((m) => ({ value: m, label: m }))} />
          ) : (
            <Input value={model} onChange={(e) => setModel(e.target.value)} />
          )}
        </Field>
        {provider ? (
          <Button variant="outline" size="sm" className="self-start" onClick={() => void loadModels()}>
            Load model list
          </Button>
        ) : null}
        <Field
          label="API key"
          hint={provider?.key_set ? "A key is saved. It is never shown again; paste a new one to replace it." : "Stored on this computer only and never shown again."}
        >
          <Input type="password" autoComplete="off" value={key} onChange={(e) => setKey(e.target.value)} placeholder={provider?.key_set ? "Saved" : ""} />
        </Field>
        <Field label="Or read the key from this environment variable" hint="Looked up in the environment and in .env.">
          <Input value={keyEnv} onChange={(e) => setKeyEnv(e.target.value)} placeholder="GROQ_API_KEY" />
        </Field>
        <fieldset className="flex flex-col gap-3">
          <legend className="mb-1 text-sm font-medium">Free-tier limits. The app never goes over them; leave empty for none.</legend>
          <div className="grid grid-cols-2 gap-3">
            {LIMITS.map((l) => (
              <Field key={l.key} label={l.label}>
                <Input
                  inputMode="numeric"
                  value={limits[l.key] ?? ""}
                  onChange={(e) => {
                    const v = e.target.value.replace(/\D/g, "");
                    setLimits({ ...limits, [l.key]: v ? Number(v) : null });
                  }}
                />
              </Field>
            ))}
          </div>
        </fieldset>
        <label className="flex items-start gap-3 rounded-lg border border-ink p-4">
          <Switch checked={free} onCheckedChange={setFree} className="mt-0.5" />
          <span className="flex flex-col gap-1">
            <span className="text-base font-medium">I confirm this endpoint is free of charge</span>
            <span className="text-sm text-graphite">Without this the app refuses to send it any request.</span>
          </span>
        </label>
        <div className="flex gap-3 pt-2">
          <Button onClick={() => void save(true)} disabled={busy}>{busy ? "Saving" : "Save and test"}</Button>
          <Button variant="outline" onClick={() => void save(false)} disabled={busy}>Save</Button>
        </div>
      </div>
    </SidePanel>
  );
}

export function OllamaForm({
  open,
  provider,
  onClose,
  onSaved,
}: {
  open: boolean;
  provider: Provider | null;
  onClose: () => void;
  onSaved: (name: string, result?: TestResult) => void;
}) {
  const [url, setUrl] = useState("");
  const [remote, setRemote] = useState(false);
  const [model, setModel] = useState("");
  const [status, setStatus] = useState<{ running: boolean; models: OllamaModel[] } | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (open && provider) {
      setUrl(provider.base_url);
      setRemote(provider.allow_remote);
      setModel(provider.model);
      setStatus(null);
    }
  }, [open, provider]);

  async function check() {
    try {
      const q = new URLSearchParams({ base_url: url, allow_remote: String(remote) });
      setStatus(await api.get(`/ollama/models?${q}`));
    } catch (e) {
      toast.error(errorMessage(e));
    }
  }

  async function save(andTest: boolean) {
    setBusy(true);
    try {
      await api.put("/providers/ollama", { base_url: url.trim(), model, allow_remote: remote });
      const result = andTest ? await api.post<TestResult>("/providers/ollama/test") : undefined;
      toast.success("Saved Ollama settings");
      onSaved("ollama", result);
      onClose();
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  const chat = (status?.models ?? []).filter((m) => m.kind === "chat");
  return (
    <SidePanel open={open} onClose={onClose} title="Local models with Ollama">
      <div className="flex flex-col gap-5">
        <p className="text-base text-graphite">
          Install Ollama, start it, and download a model with <code className="text-ink">ollama pull &lt;model&gt;</code>. Nothing you ask
          leaves this computer.
        </p>
        <Field label="Server address" hint="Ollama on this computer listens on http://localhost:11434.">
          <Input value={url} onChange={(e) => setUrl(e.target.value)} />
        </Field>
        <label className="flex items-start gap-3">
          <Switch checked={remote} onCheckedChange={setRemote} className="mt-0.5" />
          <span className="text-sm">Allow a server on another computer. Questions and passages will be sent to that address.</span>
        </label>
        <Button variant="outline" className="self-start" onClick={() => void check()}>Check server</Button>
        {status ? (
          status.running ? (
            <div className="flex flex-col gap-2">
              <p className="text-base font-medium">Running, {status.models.length} models installed</p>
              <ul className="flex flex-col text-sm">
                {status.models.map((m) => (
                  <li key={m.name} className="flex justify-between border-t border-rule py-2">
                    <span>{m.name}</span>
                    <span className="text-graphite">{(m.size_bytes / 1e9).toFixed(1)} GB, {m.kind === "chat" ? "answers" : "embeddings"}</span>
                  </li>
                ))}
              </ul>
            </div>
          ) : (
            <p className="border-l-4 border-ink pl-4 text-base">
              No answer from {url}. Start Ollama, then check again. Auto mode skips it until it answers.
            </p>
          )
        ) : null}
        <Field label="Model for answers" hint="Pick one well under half of this computer's memory.">
          {chat.length ? (
            <Select value={model} onValueChange={setModel} options={chat.map((m) => ({ value: m.name, label: m.name }))} />
          ) : (
            <Input value={model} onChange={(e) => setModel(e.target.value)} placeholder="name:tag" />
          )}
        </Field>
        <div className="flex gap-3 pt-2">
          <Button onClick={() => void save(true)} disabled={busy}>{busy ? "Saving" : "Save and test"}</Button>
          <Button variant="outline" onClick={() => void save(false)} disabled={busy}>Save</Button>
        </div>
      </div>
    </SidePanel>
  );
}

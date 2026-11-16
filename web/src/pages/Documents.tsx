import { Search } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { type DragEvent, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { ConfirmDialog } from "@/components/ConfirmDialog";
import { EmptyState } from "@/components/EmptyState";
import { IndexProgress, type ProgressState } from "@/components/IndexProgress";
import { Button } from "@/components/ui/button";
import { Field, Input } from "@/components/ui/fields";
import { api, errorMessage, useResource } from "@/lib/api";
import type { DocumentRecord, DocumentsInfo, IngestReport } from "@/lib/types";
import { cn, fileToBase64, formatTime } from "@/lib/utils";

const ACCEPT = ".pdf,.md,.markdown,.html,.htm";

function reportText(r: IngestReport): string {
  if (!r.added && !r.updated && !r.deleted) return "Everything is already up to date.";
  return `Indexed: ${r.added} added, ${r.updated} updated, ${r.deleted} removed (${r.chunks} passages written).`;
}

export function DocumentsPage({ onChanged }: { onChanged: () => void }) {
  const docs = useResource<DocumentsInfo>("/documents");
  const [files, setFiles] = useState<File[]>([]);
  const [folder, setFolder] = useState("");
  const [over, setOver] = useState(false);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const pick = (list: FileList | null) => {
    if (!list) return;
    const ok = [...list].filter((f) => /\.(pdf|md|markdown|html?)$/i.test(f.name));
    if (ok.length < list.length) toast.error("Only PDF, Markdown and HTML files can be indexed.");
    setFiles((prev) => [...prev, ...ok]);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    pick(e.dataTransfer.files);
  };

  const [progress, setProgress] = useState<ProgressState | null>(null);

  async function follow() {
    // Poll the background job until it ends; the panel shows the real percentage meanwhile.
    for (;;) {
      const { job } = await api.get<{ job: (ProgressState & { status: string; report: IngestReport | null; error: string | null }) | null }>("/ingest/jobs/current");
      if (!job) return;
      setProgress(job);
      if (job.status !== "running") {
        if (job.status === "done" && job.report) {
          toast.success(reportText(job.report));
          job.report.errors.forEach((err) => toast.error(err));
        } else if (job.error) toast.error(job.error);
        await docs.reload();
        onChanged();
        window.setTimeout(() => setProgress(null), 2500);
        return;
      }
      await new Promise((r) => window.setTimeout(r, 400));
    }
  }

  useEffect(() => {
    // Resume showing progress if indexing is already running when the page opens.
    api.get<{ job: { status: string } | null }>("/ingest/jobs/current")
      .then(({ job }) => {
        if (job?.status === "running") {
          setBusy(true);
          void follow().finally(() => setBusy(false));
        }
      })
      .catch(() => undefined);
  }, []);

  async function index(rebuild = false) {
    setBusy(true);
    try {
      for (let i = 0; i < files.length; i++) {
        setProgress({ phase: "uploading", percent: (i / files.length) * 100, done: i, total: files.length });
        await api.post("/documents/upload", { filename: files[i].name, folder, content_base64: await fileToBase64(files[i]) });
      }
      setFiles([]);
      setProgress({ phase: "starting", percent: 0 });
      await api.post("/ingest/jobs", { rebuild });
      await follow();
    } catch (e) {
      toast.error(errorMessage(e));
      setProgress(null);
    } finally {
      setBusy(false);
    }
  }

  const [filter, setFilter] = useState("");
  const [removing, setRemoving] = useState<DocumentRecord | null>(null);
  const [removeBusy, setRemoveBusy] = useState(false);

  async function remove(doc: DocumentRecord) {
    setRemoveBusy(true);
    try {
      await api.del(`/documents/${doc.doc_id}`);
      toast.success(`Removed ${doc.path}`);
      await docs.reload();
      onChanged();
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setRemoveBusy(false);
      setRemoving(null);
    }
  }

  const d = docs.data;
  const shown = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    const all = d?.documents ?? [];
    return needle ? all.filter((x) => x.path.toLowerCase().includes(needle) || x.title.toLowerCase().includes(needle)) : all;
  }, [d, filter]);
  return (
    <div className="mx-auto flex max-w-5xl flex-col gap-10">
      <header className="flex flex-col gap-2">
        <h1 className="page-title">Documents</h1>
      </header>

      <AnimatePresence>{progress ? <IndexProgress key="progress" state={progress} /> : null}</AnimatePresence>

      <section className="flex flex-col gap-4">
        <div
          role="button"
          tabIndex={0}
          onClick={() => input.current?.click()}
          onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={onDrop}
          className={cn(
            "flex min-h-40 cursor-pointer flex-col items-center justify-center gap-1 rounded-2xl border-2 border-dashed px-6 text-center transition-colors",
            over ? "border-ink bg-wash" : "border-rule hover:border-ink",
          )}
        >
          <span className="text-lg font-semibold">Drop PDF, Markdown or HTML files here</span>
          <span className="text-sm text-graphite">or click to choose files</span>
          <input ref={input} type="file" multiple accept={ACCEPT} hidden onChange={(e) => pick(e.target.files)} />
        </div>
        <AnimatePresence>
          {files.length ? (
            <motion.ul
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="flex flex-col overflow-hidden"
            >
              {files.map((f, i) => (
                <li key={`${f.name}-${i}`} className="flex justify-between border-t border-rule py-2 text-sm">
                  <span>{f.name}</span>
                  <button type="button" className="text-graphite hover:text-ink" onClick={() => setFiles(files.filter((_, j) => j !== i))}>
                    Remove
                  </button>
                </li>
              ))}
            </motion.ul>
          ) : null}
        </AnimatePresence>
        <div className="flex flex-wrap items-end gap-4">
          <Field label="Save into folder" hint="Optional, for example policies" className="w-64">
            <Input value={folder} onChange={(e) => setFolder(e.target.value)} placeholder="Top level" />
          </Field>
          <Button onClick={() => void index()} disabled={busy}>
            {busy ? "Indexing" : files.length ? `Add ${files.length} and index` : "Index changes"}
          </Button>
          <Button variant="outline" onClick={() => void index(true)} disabled={busy}>
            Rebuild the whole index
          </Button>
        </div>
      </section>

      <section className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-lg font-semibold">
            {d ? `${d.documents.length} documents, ${d.chunks.toLocaleString()} passages` : "Indexed documents"}
          </h2>
          {d && d.documents.length > 8 ? (
            <div className="relative w-full sm:w-72">
              <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-graphite" aria-hidden />
              <Input
                type="search"
                aria-label="Filter documents by file or title"
                placeholder="Filter by file or title"
                className="h-9 pl-9 text-sm"
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
              />
            </div>
          ) : null}
        </div>
        {d && d.documents.length === 0 ? (
          <EmptyState>No documents yet. Drop PDF, Markdown or HTML files above, then press Index changes.</EmptyState>
        ) : null}
        {d && d.documents.length ? (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead className="text-graphite">
                <tr>
                  <th className="py-2 pr-4 font-medium">File</th>
                  <th className="py-2 pr-4 font-medium">Title</th>
                  <th className="py-2 pr-4 text-right font-medium">Passages</th>
                  <th className="py-2 pr-4 font-medium">Indexed</th>
                  <th className="py-2 font-medium">
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {shown.map((doc) => (
                  <tr key={doc.doc_id} className="border-t border-rule">
                    <td className="py-2.5 pr-4">{doc.path}</td>
                    <td className="py-2.5 pr-4">{doc.title}</td>
                    <td className="py-2.5 pr-4 text-right tabular-nums">{doc.n_chunks}</td>
                    <td className="whitespace-nowrap py-2.5 pr-4 text-graphite">{formatTime(doc.ingested_at)}</td>
                    <td className="py-2.5 text-right">
                      <button
                        type="button"
                        className="text-graphite underline-offset-4 hover:text-ink hover:underline disabled:opacity-40"
                        disabled={busy}
                        aria-label={`Remove ${doc.path}`}
                        onClick={() => setRemoving(doc)}
                      >
                        Remove
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {shown.length === 0 ? (
              <p className="border-t border-rule py-4 text-sm text-graphite">No document matches “{filter.trim()}”.</p>
            ) : null}
          </div>
        ) : null}
        {d?.indexed_with ? <p className="text-sm text-graphite">Embedded with {d.indexed_with}.</p> : null}
      </section>
      <ConfirmDialog
        open={removing !== null}
        title="Remove this document?"
        confirmLabel="Remove document"
        busy={removeBusy}
        onCancel={() => setRemoving(null)}
        onConfirm={() => removing && void remove(removing)}
      >
        <p>
          <span className="font-semibold text-ink">{removing?.path}</span> is deleted from the documents folder and its{" "}
          {removing?.n_chunks} passages leave the index. This cannot be undone here; keep your own copy if you may need
          the file again.
        </p>
      </ConfirmDialog>
    </div>
  );
}

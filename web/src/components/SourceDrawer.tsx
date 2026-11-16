import { Drawer } from "vaul";

import { RelevanceBar } from "@/components/RelevanceBar";
import type { Source } from "@/lib/types";

/** Opens a cited passage from the bottom, read in the serif face. */
export function SourceDrawer({
  source,
  onClose,
}: {
  source: Source | null;
  onClose: () => void;
}) {
  return (
    <Drawer.Root open={source !== null} onOpenChange={(open) => !open && onClose()}>
      <Drawer.Portal>
        <Drawer.Overlay className="fixed inset-0 z-40 bg-ink/30" />
        <Drawer.Content className="fixed inset-x-0 bottom-0 z-50 mx-auto flex max-h-[85vh] max-w-3xl flex-col rounded-t-2xl border border-b-0 border-ink bg-paper outline-none">
          <div className="mx-auto mt-3 h-1 w-10 rounded-full bg-rule" />
          {source ? (
            <div className="overflow-y-auto px-8 pb-10 pt-6">
              <Drawer.Title className="text-xl font-semibold">
                <span className="mr-3 tabular-nums">[{source.n}]</span>
                {source.title}
              </Drawer.Title>
              <Drawer.Description className="mt-1 text-sm text-graphite">
                {source.section_path || "Top of document"}
                {source.page ? `, page ${source.page}` : ""}. File {source.path}
              </Drawer.Description>
              <div className="mt-5 max-w-md">
                <RelevanceBar value={source.relevance} rank={0} cited={source.cited} />
              </div>
              <p className="prose-read mt-6 whitespace-pre-line">{source.text}</p>
              <p className="mt-6 text-sm text-graphite">Chunk {source.chunk_id}</p>
            </div>
          ) : null}
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}

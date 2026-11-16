import { Command } from "cmdk";
import { useEffect, useState } from "react";

import type { Route } from "@/App";

const PAGES: { route: Route; label: string }[] = [
  { route: "ask", label: "Ask a question" },
  { route: "documents", label: "Documents" },
  { route: "traces", label: "Traces" },
  { route: "evaluation", label: "Evaluation" },
  { route: "models", label: "Models and providers" },
];

/** Ctrl/⌘ K: jump to a page, ask directly, switch theme, or test every provider. */
export function CommandMenu({
  onNavigate,
  onAsk,
  onToggleTheme,
  onTestAll,
}: {
  onNavigate: (r: Route) => void;
  onAsk: (q: string) => void;
  onToggleTheme: () => void;
  onTestAll: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState("");

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((o) => !o);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const run = (fn: () => void) => {
    fn();
    setOpen(false);
    setSearch("");
  };
  const item =
    "flex cursor-default items-center rounded-md px-3 py-2.5 text-base outline-none " +
    "data-[selected=true]:bg-ink data-[selected=true]:text-paper";

  return (
    <Command.Dialog
      open={open}
      onOpenChange={setOpen}
      label="Command menu"
      overlayClassName="fixed inset-0 z-40 bg-ink/30"
      contentClassName="fixed left-1/2 top-[18vh] z-50 w-[min(560px,92vw)] -translate-x-1/2 rounded-xl border border-ink bg-paper"
    >
      <Command.Input
        value={search}
        onValueChange={setSearch}
        placeholder="Go to a page, or type a question"
        className="h-14 w-full border-b border-rule bg-transparent px-5 text-lg outline-none placeholder:text-graphite"
      />
      <Command.List className="max-h-80 overflow-y-auto p-2">
        <Command.Empty className="px-3 py-6 text-sm text-graphite">No matching command.</Command.Empty>
        {search.trim().length > 3 ? (
          <Command.Group heading="Ask" className="text-sm text-graphite [&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:py-1.5">
            <Command.Item value={`ask ${search}`} className={item} onSelect={() => run(() => onAsk(search.trim()))}>
              Ask “{search.trim()}”
            </Command.Item>
          </Command.Group>
        ) : null}
        <Command.Group heading="Pages" className="text-sm text-graphite [&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:py-1.5">
          {PAGES.map((p) => (
            <Command.Item key={p.route} className={item} onSelect={() => run(() => onNavigate(p.route))}>
              {p.label}
            </Command.Item>
          ))}
        </Command.Group>
        <Command.Group heading="Actions" className="text-sm text-graphite [&_[cmdk-group-heading]]:px-3 [&_[cmdk-group-heading]]:py-1.5">
          <Command.Item className={item} onSelect={() => run(onTestAll)}>
            Test all providers
          </Command.Item>
          <Command.Item className={item} onSelect={() => run(onToggleTheme)}>
            Switch between light and dark
          </Command.Item>
        </Command.Group>
      </Command.List>
    </Command.Dialog>
  );
}

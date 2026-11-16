import { Fragment, type ReactNode } from "react";

import type { Source } from "@/lib/types";
import { cn } from "@/lib/utils";

// Models answer in light Markdown. This renders the parts they actually use (paragraphs, lists,
// headings, bold, italic, code) and turns [n] markers into buttons that open the cited passage.
// Anything else is shown as plain text; no HTML from the model is ever interpreted.

const MARKER = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
// Some models write citations with CJK or fullwidth brackets; the server normalises the final
// answer, this covers the text while it is still streaming.
const ALT_MARKER = /[【［〔](\d+(?:\s*[,，]\s*\d+)*)[】］〕]/g;
// The last alternative is bold that has opened but not closed yet: while an answer streams in
// (or when it was cut at the output limit) the text is shown bold instead of with raw asterisks.
const INLINE = /(\*\*[^*\n]+\*\*|`[^`\n]+`|\*[^*\s][^*\n]*\*|\*\*[^*\n]+$)/g;
const ORDERED = /^\s*(\d+)[.)]\s+(.*)$/;
const BULLET = /^\s*[-*•]\s+(.*)$/;
const HEADING = /^\s*#{1,6}\s+(.*)$/;
const FENCE = /^\s*```/;

type Block =
  | { kind: "p"; text: string }
  | { kind: "h"; text: string }
  | { kind: "ol"; start: number; items: string[] }
  | { kind: "ul"; items: string[] }
  | { kind: "pre"; text: string };

export function parseBlocks(source: string): Block[] {
  const blocks: Block[] = [];
  const lines = source.replace(ALT_MARKER, (_, n: string) => `[${n.replace(/，/g, ",")}]`).split(/\r?\n/);
  let para: string[] = [];
  const flush = () => {
    if (para.length) blocks.push({ kind: "p", text: para.join(" ") });
    para = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (FENCE.test(line)) {
      flush();
      const body: string[] = [];
      for (i++; i < lines.length && !FENCE.test(lines[i]); i++) body.push(lines[i]);
      blocks.push({ kind: "pre", text: body.join("\n") });
      continue;
    }
    const last = blocks[blocks.length - 1];
    const ordered = ORDERED.exec(line);
    const bullet = BULLET.exec(line);
    const heading = HEADING.exec(line);
    if (!line.trim()) flush();
    else if (heading) {
      flush();
      blocks.push({ kind: "h", text: heading[1] });
    } else if (ordered) {
      flush();
      if (last?.kind === "ol" && !para.length) last.items.push(ordered[2]);
      else blocks.push({ kind: "ol", start: Number(ordered[1]), items: [ordered[2]] });
    } else if (bullet) {
      flush();
      if (last?.kind === "ul" && !para.length) last.items.push(bullet[1]);
      else blocks.push({ kind: "ul", items: [bullet[1]] });
    } else if ((last?.kind === "ol" || last?.kind === "ul") && !para.length && /^\s{2,}\S/.test(line)) {
      // An indented line continues the list item above it.
      last.items[last.items.length - 1] += ` ${line.trim()}`;
    } else para.push(line.trim());
  }
  flush();
  return blocks;
}

function Emphasis({ text }: { text: string }) {
  return (
    <>
      {text.split(INLINE).map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
          return (
            <strong key={i}>
              <Emphasis text={part.slice(2, -2)} />
            </strong>
          );
        }
        if (part.startsWith("**") && part.length > 2 && !part.endsWith("*")) {
          return <strong key={i}>{part.slice(2)}</strong>;
        }
        if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
          return (
            <code key={i} className="rounded bg-wash px-1 font-sans text-[0.9em]">
              {part.slice(1, -1)}
            </code>
          );
        }
        if (part.startsWith("*") && part.endsWith("*") && part.length > 2) {
          return <em key={i}>{part.slice(1, -1)}</em>;
        }
        return <Fragment key={i}>{part}</Fragment>;
      })}
    </>
  );
}

function Inline({ text, sources, onOpen }: { text: string; sources?: Source[]; onOpen?: (s: Source) => void }) {
  const parts: ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(MARKER)) {
    const at = m.index ?? 0;
    parts.push(<Emphasis key={`t${last}`} text={text.slice(last, at)} />);
    const nums = m[1].split(",").map((n) => Number(n.trim()));
    parts.push(
      <span key={`m${at}`} className="whitespace-nowrap">
        {nums.map((n) => {
          const src = sources?.find((s) => s.n === n);
          return (
            <button
              key={n}
              type="button"
              disabled={!src || !onOpen}
              onClick={() => src && onOpen?.(src)}
              title={src ? `Open source ${n}: ${src.title}` : `Source ${n}`}
              aria-label={src ? `Source ${n}: ${src.title}` : `Source ${n}`}
              className="mx-0.5 inline-flex h-5 min-w-5 -translate-y-1 items-center justify-center rounded bg-ink px-1 font-sans text-xs font-semibold text-paper transition-transform enabled:hover:scale-110 disabled:bg-wash disabled:text-graphite"
            >
              {n}
            </button>
          );
        })}
      </span>,
    );
    last = at + m[0].length;
  }
  parts.push(<Emphasis key={`t${last}`} text={text.slice(last)} />);
  return <>{parts}</>;
}

/** An answer in the reading face. `streaming` shows a caret after the last word. */
export function Markdown({
  text,
  sources,
  onOpen,
  streaming = false,
  className,
}: {
  text: string;
  sources?: Source[];
  onOpen?: (s: Source) => void;
  streaming?: boolean;
  className?: string;
}) {
  const blocks = parseBlocks(text);
  const inline = (t: string) => <Inline text={t} sources={sources} onOpen={onOpen} />;
  const caret = streaming ? <span className="stream-caret" aria-hidden /> : null;
  const lastIndex = blocks.length - 1;
  return (
    <div className={cn("prose-read flex flex-col gap-4", className)} aria-busy={streaming || undefined}>
      {blocks.map((b, i) => {
        const tail = i === lastIndex ? caret : null;
        if (b.kind === "h") {
          return (
            <h3 key={i} className="font-sans text-lg font-semibold">
              {inline(b.text)}
              {tail}
            </h3>
          );
        }
        if (b.kind === "pre") {
          return (
            <pre key={i} className="overflow-x-auto rounded-lg border border-rule p-4 font-sans text-sm leading-relaxed">
              {b.text}
            </pre>
          );
        }
        if (b.kind === "ol" || b.kind === "ul") {
          const items = b.items.map((item, j) => (
            <li key={j} className="pl-1">
              {inline(item)}
              {j === b.items.length - 1 ? tail : null}
            </li>
          ));
          return b.kind === "ol" ? (
            <ol key={i} start={b.start} className="flex list-decimal flex-col gap-2 pl-6 marker:font-sans marker:text-base marker:font-semibold">
              {items}
            </ol>
          ) : (
            <ul key={i} className="flex list-disc flex-col gap-2 pl-6">
              {items}
            </ul>
          );
        }
        return (
          <p key={i}>
            {inline(b.text)}
            {tail}
          </p>
        );
      })}
      {blocks.length === 0 ? <p>{caret}</p> : null}
    </div>
  );
}

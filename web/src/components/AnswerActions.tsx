import { Check, Copy, ThumbsDown, ThumbsUp } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/fields";
import { api, errorMessage } from "@/lib/api";
import { cn } from "@/lib/utils";

type Rating = "up" | "down" | null;

/** Copy the answer, and say whether it was right. The verdict is stored on the trace. */
export function AnswerActions({
  traceId,
  text,
  initial = null,
  initialNote = "",
  onRated,
}: {
  traceId: string;
  text: string;
  initial?: Rating;
  initialNote?: string;
  onRated?: (rating: Rating) => void;
}) {
  const [rating, setRating] = useState<Rating>(initial);
  const [note, setNote] = useState(initialNote);
  const [savedNote, setSavedNote] = useState(initialNote);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setRating(initial);
    setNote(initialNote);
    setSavedNote(initialNote);
  }, [traceId, initial, initialNote]);

  async function save(next: Rating, nextNote: string) {
    const previous = rating;
    setRating(next);
    try {
      await api.put(`/traces/${traceId}/feedback`, { rating: next, note: next ? nextNote : "" });
      setSavedNote(next ? nextNote : "");
      if (!next) setNote("");
      onRated?.(next);
    } catch (e) {
      setRating(previous);
      toast.error(errorMessage(e));
    }
  }

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      toast.error("Copying is blocked by the browser. Select the text and copy it instead.");
    }
  }

  const vote = (value: "up" | "down", label: string, Icon: typeof ThumbsUp) => (
    <Button
      type="button"
      size="icon"
      variant="ghost"
      aria-label={label}
      aria-pressed={rating === value}
      title={label}
      onClick={() => void save(rating === value ? null : value, note)}
      className={cn(rating === value && "bg-ink text-paper hover:bg-ink")}
    >
      <Icon />
    </Button>
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-1">
        <Button type="button" size="sm" variant="ghost" onClick={() => void copy()} className="-ml-3">
          {copied ? <Check /> : <Copy />}
          {copied ? "Copied" : "Copy answer"}
        </Button>
        <span className="mx-2 h-5 w-px bg-rule" aria-hidden />
        <span className="mr-1 text-sm text-graphite">Was this right?</span>
        {vote("up", "The answer is right", ThumbsUp)}
        {vote("down", "The answer is wrong or incomplete", ThumbsDown)}
      </div>
      {rating === "down" ? (
        <form
          className="flex max-w-xl flex-col gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void save("down", note.trim()).then(() => toast.success("Note saved with the trace."));
          }}
        >
          <label htmlFor={`note-${traceId}`} className="text-sm font-medium">
            What was wrong? <span className="font-normal text-graphite">Optional, kept with the trace.</span>
          </label>
          <Textarea
            id={`note-${traceId}`}
            rows={2}
            maxLength={2000}
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="For example: cites the right policy but the wrong number of days"
          />
          <Button type="submit" size="sm" variant="outline" className="self-start" disabled={note.trim() === savedNote}>
            Save note
          </Button>
        </form>
      ) : null}
    </div>
  );
}

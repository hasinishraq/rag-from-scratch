// Reads POST /api/query/stream: server-sent events over fetch (EventSource cannot POST).
// Events: `sources` once the passages are chosen, `delta` for each piece of the answer, then
// `done` with the complete answer, or `error` with {code, message}.

import { ApiError } from "@/lib/api";
import type { Answer, Source } from "@/lib/types";

export interface StreamHandlers {
  onSources: (sources: Source[]) => void;
  onDelta: (text: string) => void;
}

function parseFrame(frame: string): { event: string; data: string } | null {
  let event = "message";
  const data: string[] = [];
  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }
  return data.length ? { event, data: data.join("\n") } : null;
}

/** Resolves with the final answer. Aborting the signal stops the answer on the server too. */
export async function streamAnswer(body: unknown, handlers: StreamHandlers, signal: AbortSignal): Promise<Answer> {
  let resp: Response;
  try {
    resp = await fetch("/api/query/stream", {
      method: "POST",
      headers: { "content-type": "application/json", accept: "text/event-stream" },
      body: JSON.stringify(body),
      signal,
    });
  } catch (e) {
    if (signal.aborted) throw e;
    throw new ApiError("api_unreachable", "The server is not answering. Start it with: uv run python scripts/serve.py", 0);
  }
  if (!resp.ok || !resp.body) {
    let code = "error";
    let message = resp.statusText || "The request failed.";
    try {
      const data = await resp.json();
      code = data.code ?? code;
      message = data.message ?? message;
    } catch {
      /* body was not JSON */
    }
    throw new ApiError(code, message, resp.status);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let cut = buffer.indexOf("\n\n");
    while (cut !== -1) {
      const frame = parseFrame(buffer.slice(0, cut));
      buffer = buffer.slice(cut + 2);
      cut = buffer.indexOf("\n\n");
      if (!frame) continue; // keep-alive comment
      const data = JSON.parse(frame.data);
      if (frame.event === "sources") handlers.onSources(data.sources as Source[]);
      else if (frame.event === "delta") handlers.onDelta(data.text as string);
      else if (frame.event === "done") return data as Answer;
      else if (frame.event === "error") throw new ApiError(data.code ?? "error", data.message ?? "The answer failed.", 200);
    }
  }
  throw new ApiError("stream_ended", "The connection closed before the answer was complete.", 0);
}

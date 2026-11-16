// Thin client for the FastAPI service. The UI never talks to a model provider directly.

import { useCallback, useEffect, useState } from "react";

export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

const BASE = "/api";

export async function call<T>(method: string, path: string, body?: unknown): Promise<T> {
  return (await callWithHeaders<T>(method, path, body)).data;
}

/** Like call, and also returns the response headers (X-Total-Count on lists). */
export async function callWithHeaders<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<{ data: T; headers: Headers }> {
  let resp: Response;
  try {
    resp = await fetch(`${BASE}${path}`, {
      method,
      headers: body === undefined ? undefined : { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(
      "api_unreachable",
      "The server is not answering. Start it with: uv run python scripts/serve.py",
      0,
    );
  }
  if (!resp.ok) {
    let code = "error";
    let message = resp.statusText;
    let requestId: string | undefined;
    try {
      const data = await resp.json();
      code = data.code ?? code;
      message = data.message ?? message;
      requestId = data.request_id;
    } catch {
      /* body was not JSON */
    }
    // An unexpected server error is only useful with the id that finds it in the server log.
    if (resp.status >= 500 && code === "internal_error" && requestId) {
      message = `Something went wrong on the server (request ${requestId}).`;
    }
    throw new ApiError(code, message, resp.status);
  }
  return { data: (await resp.json()) as T, headers: resp.headers };
}

export const api = {
  get: <T>(path: string) => call<T>("GET", path),
  post: <T>(path: string, body: unknown = {}) => call<T>("POST", path, body),
  put: <T>(path: string, body: unknown) => call<T>("PUT", path, body),
  del: <T>(path: string) => call<T>("DELETE", path),
};

/** Loads a resource and exposes a reload function; errors are kept, not thrown. */
export function useResource<T>(path: string | null) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(path !== null);

  const reload = useCallback(async () => {
    if (path === null) return;
    setLoading(true);
    try {
      setData(await api.get<T>(path));
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError("error", String(e), 0));
    } finally {
      setLoading(false);
    }
  }, [path]);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { data, error, loading, reload, setData };
}

export function errorMessage(e: unknown): string {
  return e instanceof ApiError ? e.message : String(e);
}

import { MotionConfig } from "motion/react";
import { useCallback, useEffect, useState } from "react";
import { Toaster, toast } from "sonner";

import { CommandMenu } from "@/components/CommandMenu";
import { Dither, orbField } from "@/components/Dither";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { SideNav } from "@/components/SideNav";
import { UsageRail } from "@/components/UsageRail";
import { api, errorMessage, useResource } from "@/lib/api";
import type { ServerHealth, TestResult } from "@/lib/types";
import { AskPage } from "@/pages/Ask";
import { DocumentsPage } from "@/pages/Documents";
import { EvaluationPage } from "@/pages/Evaluation";
import { ModelsPage } from "@/pages/models/Models";
import { TracesPage } from "@/pages/Traces";

export type Route = "ask" | "documents" | "traces" | "evaluation" | "models";
const ROUTES: { route: Route; label: string }[] = [
  { route: "ask", label: "Ask" },
  { route: "documents", label: "Documents" },
  { route: "traces", label: "Traces" },
  { route: "evaluation", label: "Evaluation" },
  { route: "models", label: "Models" },
];

function readRoute(): Route {
  const r = window.location.hash.replace(/^#\/?/, "").split("?")[0] as Route;
  return ROUTES.some((x) => x.route === r) ? r : "ask";
}

function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">(() => {
    try {
      const saved = localStorage.getItem("theme");
      if (saved === "light" || saved === "dark") return saved;
    } catch {
      /* storage unavailable */
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);
  return { theme, toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")) };
}

const HEALTH_POLL_MS = 15000;

export default function App() {
  const [route, setRoute] = useState<Route>(readRoute);
  // #/ask?q=... asks on load, so a question can be bookmarked or shared.
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(() =>
    new URLSearchParams(window.location.hash.split("?")[1] ?? "").get("q"),
  );
  const [refreshKey, setRefreshKey] = useState(0);
  const { theme, toggle } = useTheme();
  const health = useResource<ServerHealth>("/health");
  const { reload: reloadHealth } = health;
  const offline = health.error?.code === "api_unreachable";

  // Keeps the quota figures current and notices when the server stops or comes back.
  useEffect(() => {
    const timer = window.setInterval(() => void reloadHealth(), HEALTH_POLL_MS);
    return () => window.clearInterval(timer);
  }, [reloadHealth]);
  useEffect(() => {
    if (refreshKey > 0) void reloadHealth();
  }, [refreshKey, reloadHealth]);
  useEffect(() => {
    const label = ROUTES.find((r) => r.route === route)?.label ?? "Ask";
    document.title = `${label} · RAG Console`;
  }, [route]);

  useEffect(() => {
    const onHash = () => setRoute(readRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const navigate = useCallback((r: Route) => {
    window.location.hash = `/${r}`;
  }, []);
  const refresh = useCallback(() => setRefreshKey((k) => k + 1), []);

  const testAll = useCallback(() => {
    const run = api.post<TestResult[]>("/providers/test-all").then((results) => {
      refresh();
      const ok = results.filter((r) => r.status === "working" || r.status === "working_with_warning");
      return `${ok.length} of ${results.length} providers working`;
    });
    toast.promise(run, {
      loading: "Testing providers one after another",
      success: (msg) => msg,
      error: (e) => errorMessage(e),
    });
  }, [refresh]);

  return (
    <MotionConfig reducedMotion="user">
      <a href="#main" className="skip-link">
        Skip to the page
      </a>
      <div className="min-h-dvh md:grid md:grid-cols-[13rem_1fr]">
        <aside className="flex flex-col gap-8 border-rule px-6 py-6 md:sticky md:top-0 md:h-dvh md:border-r">
          <a href="#/ask" className="flex items-center gap-3">
            <span className="size-8 shrink-0 overflow-hidden rounded-full">
              <Dither field={orbField} cell={1} fps={12} />
            </span>
            <span className="dot whitespace-nowrap text-lg font-black leading-none">RAG Console</span>
          </a>
          <SideNav routes={ROUTES} current={route} />
          <div className="hidden md:block">
            <UsageRail refreshKey={refreshKey} health={health.data} />
          </div>
          <div className="mt-auto hidden flex-col gap-2 text-sm text-graphite md:flex">
            <button type="button" onClick={toggle} className="text-left hover:text-ink">
              {theme === "dark" ? "Light theme" : "Dark theme"}
            </button>
            <span>
              Press <kbd className="rounded border border-rule px-1">Ctrl K</kbd> for commands
            </span>
            {health.data?.version ? <span>Version {health.data.version}</span> : null}
          </div>
        </aside>
        <main id="main" tabIndex={-1} className="min-w-0 px-5 py-8 outline-none md:px-12 md:py-10">
          {offline ? (
            <p role="alert" className="mx-auto mb-8 max-w-6xl border border-ink px-4 py-3 text-base">
              <span className="font-semibold">The server is not answering.</span> Start it with{" "}
              <code>uv run python scripts/serve.py</code>. This page reconnects by itself.
            </p>
          ) : health.data && !health.data.ready ? (
            <p role="status" className="shimmer-text mx-auto mb-8 max-w-6xl text-sm">
              Loading the local search models. The first question waits for them.
            </p>
          ) : null}
          <ErrorBoundary resetKey={route}>
          {route === "ask" && (
            <AskPage
              initialQuestion={pendingQuestion}
              onConsumed={() => setPendingQuestion(null)}
              onAnswered={refresh}
            />
          )}
          {route === "documents" && <DocumentsPage onChanged={refresh} />}
          {route === "traces" && <TracesPage />}
          {route === "evaluation" && <EvaluationPage />}
          {route === "models" && <ModelsPage refreshKey={refreshKey} onChanged={refresh} onTestAll={testAll} />}
          </ErrorBoundary>
        </main>
      </div>
      <CommandMenu
        onNavigate={navigate}
        onAsk={(q) => {
          setPendingQuestion(q);
          navigate("ask");
        }}
        onToggleTheme={toggle}
        onTestAll={testAll}
      />
      <Toaster
        position="bottom-right"
        toastOptions={{
          className: "!rounded-lg !border !border-ink !bg-paper !text-ink !font-sans !shadow-none",
        }}
      />
    </MotionConfig>
  );
}

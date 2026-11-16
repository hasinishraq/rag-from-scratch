// Shapes returned by the FastAPI service (src/rag/api.py).

export type Mode = "auto" | "free_api_only" | "ollama_only";

export type TestStatus =
  | "working"
  | "working_with_warning"
  | "rate_limited"
  | "key_missing"
  | "key_rejected"
  | "unreachable"
  | "model_not_found"
  | "untested";

export interface Source {
  n: number;
  chunk_id: string;
  title: string;
  section_path: string;
  page: number | null;
  path: string;
  text: string;
  score: number | null;
  relevance: number;
  cited: boolean;
}

export interface Answer {
  answer: string;
  refused: boolean;
  sources: Source[];
  trace_id: string;
  provider: string | null;
  model: string | null;
  cached: boolean;
  latency_ms: number;
  n_candidates: number;
  warnings: string[];
  timings: Record<string, number>;
  tokens_in: number;
  tokens_out: number;
  tokens_saved: number;
}

export interface Limits {
  rpm?: number | null;
  rpd?: number | null;
  tpm?: number | null;
  tpd?: number | null;
}

export interface Quota {
  requests_today: number;
  tokens_today: number;
  requests_left: number | null;
  tokens_left: number | null;
  exhausted: boolean;
}

export interface Provider {
  name: string;
  kind: "openai_compatible" | "ollama";
  base_url: string;
  model: string;
  api_key_env: string;
  limits: Limits;
  free: boolean;
  enabled: boolean;
  position: number;
  allow_remote: boolean;
  test_status: TestStatus;
  test_stage: number | null;
  test_detail: string;
  test_latency_ms: number | null;
  tested_at: string | null;
  key_set: boolean;
  quota: Quota;
  local: boolean;
  health: Health | null;
}

export interface Health {
  avg_latency_ms: number | null;
  paused: boolean;
  paused_for_s: number;
  pause_reason: string;
  consecutive_failures: number;
  successes: number;
  failures: number;
}

export interface Stats {
  today: {
    queries: number;
    model_calls: number;
    cache_hits: number;
    refused_free: number;
    tokens: number;
    tokens_saved: number;
    failed: number;
    lookups: number;
    helpful: number;
    unhelpful: number;
    tokens_per_call: number;
    latency_p50_ms: number | null;
    latency_p95_ms: number | null;
  };
  providers: Record<string, Health>;
}

export interface Stage {
  n: number;
  name: string;
  outcome: "passed" | "failed" | "warning" | "not_checked" | "skipped";
  detail: string;
}

export interface TestResult {
  provider: string;
  status: TestStatus;
  stages: Stage[];
  latency_ms: number | null;
  detail: string;
  requests_used: number;
  tokens_used: number;
  tested_at: string;
  retry_after_s: number | null;
}

export interface Preset {
  label: string;
  name: string;
  base_url: string;
  api_key_env: string;
  limits: Limits;
  note?: string;
}

export interface DocumentRecord {
  doc_id: string;
  path: string;
  sha256: string;
  title: string;
  n_chunks: number;
  ingested_at: string;
}

export interface DocumentsInfo {
  corpus_dir: string;
  documents: DocumentRecord[];
  chunks: number;
  indexed_with: string | null;
}

export interface IngestReport {
  added: number;
  updated: number;
  deleted: number;
  chunks: number;
  errors: string[];
}

export interface Candidate {
  chunk_id: string;
  title: string;
  section_path: string;
  path: string;
  dense_rank: number | null;
  bm25_rank: number | null;
  rrf_score: number;
  rerank_score: number | null;
}

export interface Trace {
  trace_id: string;
  question: string;
  candidates: Candidate[];
  prompt: string;
  answer: string;
  refused: boolean;
  provider: string | null;
  model: string | null;
  tokens_in: number;
  tokens_out: number;
  latency_ms: number;
  cached: boolean;
  config_hash: string;
  mode: string;
  errors: string[];
  created_at: string;
  timings: Record<string, number>;
  tokens_saved: number;
  feedback: "up" | "down" | null;
  feedback_note: string;
}

/** A trace as listed with summary=true: everything except the candidates and the prompt. */
export type TraceSummary = Omit<Trace, "candidates" | "prompt">;

export interface Settings {
  mode: Mode;
  routing_strategy: "ordered" | "fastest";
  failure_threshold: number;
  pause_s: number;
  embedding_backend: string;
  embedding_model: string;
  indexed_with: string | null;
  reranker: string | null;
  test_cost: string;
}

export interface ServerHealth {
  status: string;
  version: string;
  uptime_s: number;
  ready: boolean;
  chunks: number;
  mode: Mode;
  quota: Record<string, Quota>;
}

export interface OllamaModel {
  name: string;
  size_bytes: number;
  kind: "chat" | "embeddings";
}

export interface EvalRun {
  run: string;
  config: string;
  split: string;
  retrieval_only: boolean;
  questions: string | null;
  metrics: Record<string, number | null>;
}

export interface EvalDetail extends EvalRun {
  questions_file: string | null;
  n: number;
  n_answerable: number;
  n_unanswerable: number;
  n_scored: number;
  matched_questions: number;
  stderr: Record<string, number | null>;
  depth: number;
  recall_at_k: { k: number; value: number }[];
  rank_hist: { rank: string; count: number; share: number }[];
  by_topic: { topic: string; n: number; unanswerable: number; recall_at_5: number; mrr: number; misses: number }[];
  latency: { n: number; min?: number; max?: number; mean?: number; p50?: number; p90?: number; p95?: number; p99?: number; hist: { from_ms: number; to_ms: number; count: number; overflow?: boolean }[] };
  generation: {
    answered: number; refused: number; errors: number; correct_refusals: number; unanswerable: number;
    false_refusals: number; answerable: number; judged_faithful: number; faithful: number;
    judged_correct: number; correct: number; providers: Record<string, number>; tokens: number;
  } | null;
  misses_same_document: number;
  misses: { id: string; question: string; topic: string | null; rank: number | null; first_title: string | null; gold_title: string | null }[];
  previous: { run: string; config: string | null; metrics: Record<string, number | null> } | null;
  changes: string[];
}

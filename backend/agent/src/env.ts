import { config } from 'dotenv';
import { resolve } from 'node:path';

// The single .env at the assignment root. Provider keys are read HERE and nowhere else.
config({ path: resolve(process.cwd(), '../../.env') });
config({ path: resolve(process.cwd(), '.env') });

const num = (v: string | undefined, fallback: number) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
};
export const env = {
  port: num(process.env.PORT_AGENT ?? process.env.PORT, 8000),
  // Interface to bind. Default 0.0.0.0 (local dev + Docker). On Render the supervisor sets
  // 127.0.0.1 so the agent is loopback-only — the red-line "agent not publicly reachable"
  // without VPC/shared-secret; Render exposes only the gateway's $PORT.
  host: process.env.AGENT_HOST ?? '0.0.0.0',
  mongoUri: process.env.MONGODB_URI ?? '',
  mongoDb: process.env.MONGODB_DB ?? 'lumina',
  vectorBackend: (process.env.VECTOR_BACKEND ?? 'atlas-vector-search') as
    | 'atlas-vector-search'
    | 'mongo-cosine-scan',

  llmProvider: process.env.LLM_PROVIDER ?? 'anthropic',
  llmModel: process.env.LLM_MODEL ?? 'claude-sonnet-5',
  // Groq is OpenAI-compatible: same SDK, different baseURL. Anthropic ignores this.
  llmBaseUrl: process.env.GROQ_BASE_URL ?? 'https://api.groq.com/openai/v1',

  // Fallback LLM: Groq's free tier has a hard daily token cap (200k TPD); when it 429s a run
  // would die mid-answer. Gemini publishes an OpenAI-compatible endpoint, so the SAME openai
  // SDK reaches it by swapping baseURL + model, reusing GEMINI_API_KEY (already set for
  // embeddings). Only a rate-limit error trips the fallback — every other error still fails
  // loud. Set LLM_FALLBACK_ENABLED=false to force single-provider behaviour.
  llmFallbackEnabled: (process.env.LLM_FALLBACK_ENABLED ?? 'true') !== 'false',
  geminiBaseUrl: process.env.GEMINI_OPENAI_BASE_URL ?? 'https://generativelanguage.googleapis.com/v1beta/openai',
  geminiChatModel: process.env.GEMINI_CHAT_MODEL ?? 'gemini-2.0-flash',

  searchProvider: (process.env.SEARCH_PROVIDER ?? 'tavily') as 'tavily' | 'serpapi',
  searchCacheTtlSeconds: num(process.env.SEARCH_CACHE_TTL_SECONDS, 21600),

  // Embeddings are decoupled from the LLM: Gemini here, Groq for chat.
  embeddingProvider: process.env.EMBEDDING_PROVIDER ?? 'gemini',
  embeddingModel: process.env.EMBEDDING_MODEL ?? 'gemini-embedding-001',

  // Deep search is the expensive gear, so its limits are configuration, not code.
  deepSubQuestionsMin: num(process.env.DEEP_SUB_QUESTIONS_MIN, 3),
  deepSubQuestionsMax: num(process.env.DEEP_SUB_QUESTIONS_MAX, 6),
  deepDailyCap: num(process.env.DEEP_DAILY_CAP, 5),

  // The hard caps from AGENTS.md. Raising these to make a gate pass is the failure mode
  // the caps exist to catch. Two gears, two envelopes.
  maxToolCalls: num(process.env.MAX_TOOL_CALLS, 8),
  maxWallClockSec: num(process.env.MAX_WALL_CLOCK_SEC, 90),
  maxToolCallsDeep: num(process.env.MAX_TOOL_CALLS_DEEP, 24),
  maxWallClockSecDeep: num(process.env.MAX_WALL_CLOCK_SEC_DEEP, 240),

  logLevel: process.env.LOG_LEVEL ?? 'info',
  /** Where the per-answer run logs land. quality/check.mjs reads this folder. */
  runsDir: resolve(process.cwd(), '../../runs')
} as const;

/** Never log or return these. /health names the model; it never echoes a key. */
export const secrets = {
  anthropic: process.env.ANTHROPIC_API_KEY ?? '',
  groq: process.env.GROQ_API_KEY ?? '',
  gemini: process.env.GEMINI_API_KEY ?? '',
  openai: process.env.OPENAI_API_KEY ?? '',
  tavily: process.env.TAVILY_API_KEY ?? '',
  serpapi: process.env.SERPAPI_API_KEY ?? ''
} as const;

import { config } from 'dotenv';
import { resolve } from 'node:path';

// Both services read the single .env at the assignment root.
config({ path: resolve(process.cwd(), '../../.env') });
config({ path: resolve(process.cwd(), '.env') });

const num = (v: string | undefined, fallback: number) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
};

export const env = {
  port: num(process.env.PORT_GATEWAY ?? process.env.PORT, 8787),
  agentUrl: process.env.AGENT_URL ?? 'http://localhost:8000',
  corsOrigins: (process.env.CORS_ORIGINS ?? 'http://localhost:5173')
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean),
  rateLimitPerMinute: num(process.env.RATE_LIMIT_PER_MINUTE, 30),
  logLevel: process.env.LOG_LEVEL ?? 'info',
  /** How long the gateway waits on the agent before it calls the request a 502. */
  upstreamTimeoutMs: num(process.env.GATEWAY_UPSTREAM_TIMEOUT_MS, 120_000),
  /** Serve the built UI from the gateway in production so one host serves / and /evals. */
  webDist: resolve(process.cwd(), '../../web/dist'),
  /**
   * The Product Evaluation the eval skill writes (reports/report.json at the repo root).
   * The gateway serves it at GET /evals/report.json and the provided UI renders it at /evals.
   */
  reportPath: resolve(process.cwd(), '../../reports/report.json')
} as const;

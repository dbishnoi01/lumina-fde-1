/**
 * Process-local metrics that GET /stats reports and reconciles against the logs: TTFT percentiles,
 * the search-cache hit rate, and today's cost. Durable counts (requests, answers, deep-today) come
 * from Mongo in stats.ts; these are the fast-moving numbers a dashboard wants live. A restart
 * resets them, which is fine — the bench runs within one process lifetime.
 */
interface AnswerSample {
  at: number;
  ttftMs: number;
  costUsd: number;
  searches: number;
  hits: number;
}

const MAX = 2000;
const samples: AnswerSample[] = [];

export function recordAnswer(s: Omit<AnswerSample, 'at'>): void {
  samples.push({ ...s, at: Date.now() });
  if (samples.length > MAX) samples.shift();
}

export function percentile(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.ceil((p / 100) * sorted.length) - 1);
  return sorted[Math.max(0, idx)] ?? 0;
}

function startOfUtcDay(): number {
  const n = new Date();
  return Date.UTC(n.getUTCFullYear(), n.getUTCMonth(), n.getUTCDate());
}

export function metricsSnapshot(): {
  ttftP95Ms: number;
  searchCacheHitRatePct: number;
  costUsdToday: number;
  answers: number;
} {
  const today = startOfUtcDay();
  const todays = samples.filter((s) => s.at >= today);
  const searches = samples.reduce((a, s) => a + s.searches, 0);
  const hits = samples.reduce((a, s) => a + s.hits, 0);
  return {
    ttftP95Ms: percentile(samples.map((s) => s.ttftMs), 95),
    searchCacheHitRatePct: searches === 0 ? 0 : (hits / searches) * 100,
    costUsdToday: todays.reduce((a, s) => a + s.costUsd, 0),
    answers: samples.length
  };
}

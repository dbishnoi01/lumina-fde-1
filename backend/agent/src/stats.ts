/**
 * GET /stats — the numbers reconcile with the logs (that reconciliation is the observability
 * gate). Durable counts come from Mongo; the fast-moving live figures (TTFT p95, cache hit rate,
 * cost today) come from the in-process metrics. answers is the count of run logs for this user,
 * so `/stats.answers` is never below the answers a run just produced.
 */
import { COLLECTIONS, type RequestDoc, type RunDoc, type StatsResponse } from '@lumina/contract';
import { db } from './db.js';
import { env } from './env.js';
import { deepUsedToday } from './deepCap.js';
import { metricsSnapshot } from './metrics.js';

export async function getStats(userId: string): Promise<StatsResponse> {
  const d = await db();
  const [requests, answers, deepToday] = await Promise.all([
    d.collection<RequestDoc>(COLLECTIONS.requests).countDocuments({ userId }),
    d.collection<RunDoc>(COLLECTIONS.runs).countDocuments({ userId }),
    deepUsedToday(userId)
  ]);
  const m = metricsSnapshot();
  return {
    requests,
    answers,
    searchCacheHitRatePct: Math.round(m.searchCacheHitRatePct * 10) / 10,
    ttftP95Ms: Math.round(m.ttftP95Ms),
    costUsdToday: Math.round(m.costUsdToday * 1e6) / 1e6,
    deepToday,
    deepDailyCap: env.deepDailyCap
  };
}

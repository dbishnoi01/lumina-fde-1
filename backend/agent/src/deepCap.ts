/**
 * The deep-search daily cap, enforced HERE in the agent service (not the gateway) because a
 * cap you can bypass by calling the service directly is not a cap. Deep costs several times a
 * quick answer, so each X-User-Id gets DEEP_DAILY_CAP deep runs per UTC day; the next one is a
 * 429 that says when it resets. We count committed deep runs from the `requests` log rather
 * than an in-memory counter, so a restart does not reset anybody's allowance.
 */
import { COLLECTIONS, type RequestDoc } from '@lumina/contract';
import { db } from './db.js';
import { env } from './env.js';

function startOfUtcDay(): Date {
  const now = new Date();
  return new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
}

function nextUtcMidnight(): Date {
  const d = startOfUtcDay();
  d.setUTCDate(d.getUTCDate() + 1);
  return d;
}

export async function deepUsedToday(userId: string): Promise<number> {
  return (await db())
    .collection<RequestDoc>(COLLECTIONS.requests)
    .countDocuments({ userId, depth: 'deep', createdAt: { $gte: startOfUtcDay() } });
}

export class DeepCapError extends Error {
  readonly resetsAt: string;
  constructor(resetsAt: string) {
    super('deep search daily cap reached');
    this.name = 'DeepCapError';
    this.resetsAt = resetsAt;
  }
}

/** Throws DeepCapError (→ 429) when the user has already spent their deep allowance today. */
export async function assertDeepAllowed(userId: string): Promise<void> {
  const used = await deepUsedToday(userId);
  if (used >= env.deepDailyCap) throw new DeepCapError(nextUtcMidnight().toISOString());
}

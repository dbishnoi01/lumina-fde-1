/**
 * The three edge concerns the gateway owns and the agent trusts it to have done:
 * identity (401), a per-user budget (429), and body shape (400). None of them touch a
 * provider key, and none of them decide anything about the answer — they only decide
 * whether the request is allowed to reach the agent at all.
 */
import type { NextFunction, Request, Response } from 'express';
import type { ZodSchema } from 'zod';
import { USER_HEADER, type ErrorBody } from '@lumina/contract';
import { env } from './env.js';

const fail = (res: Response, status: number, body: Omit<ErrorBody, 'status'>) =>
  res.status(status).json({ ...body, status, requestId: String(res.locals.requestId) } satisfies ErrorBody);

/**
 * X-User-Id is the whole auth story: one header, required on every route but /health and
 * /evals/report.json. Missing or blank → 401, before anything else runs. The verified id
 * is stashed on res.locals so the rate limiter and the proxy do not re-read the header.
 */
export function requireUser(req: Request, res: Response, next: NextFunction): void {
  const uid = req.header(USER_HEADER)?.trim();
  if (!uid) return void fail(res, 401, { error: 'X-User-Id header required' });
  res.locals.userId = uid;
  next();
}

// Fixed-window counter per user, in memory. One process on a free tier, one user at a
// time — a Redis token bucket would be more correct and, here, pure ceremony.
const windows = new Map<string, { count: number; resetAt: number }>();
const WINDOW_MS = 60_000;

/** Per-user rate limit → 429 with resetsAt, so the client knows when to come back. */
export function rateLimit(_req: Request, res: Response, next: NextFunction): void {
  const uid = String(res.locals.userId);
  const now = Date.now();

  let w = windows.get(uid);
  if (!w || w.resetAt <= now) {
    w = { count: 0, resetAt: now + WINDOW_MS };
    windows.set(uid, w);
  }
  w.count += 1;

  if (w.count > env.rateLimitPerMinute) {
    res.setHeader('Retry-After', String(Math.ceil((w.resetAt - now) / 1000)));
    return void fail(res, 429, {
      error: `rate limit exceeded: ${env.rateLimitPerMinute} requests/min`,
      resetsAt: new Date(w.resetAt).toISOString()
    });
  }

  // Opportunistic sweep so a long-lived process does not accumulate dead windows.
  if (windows.size > 1000) for (const [k, v] of windows) if (v.resetAt <= now) windows.delete(k);

  next();
}

/**
 * Validate an inbound JSON body against a contract schema → 400 with the zod message. The
 * agent validates the same bodies again (it cannot trust the edge), but rejecting a bad
 * shape here means a malformed request never costs an upstream hop.
 */
export const validateBody =
  (schema: ZodSchema) =>
  (req: Request, res: Response, next: NextFunction): void => {
    const parsed = schema.safeParse(req.body ?? {});
    if (!parsed.success) return void fail(res, 400, { error: parsed.error.message });
    req.body = parsed.data;
    next();
  };

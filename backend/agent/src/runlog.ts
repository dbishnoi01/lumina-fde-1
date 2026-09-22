/**
 * The run log. One runs/<requestId>.json per answer, in the exact RunLog shape the gates read
 * (quality/check.mjs), plus a queryable RunDoc in Mongo (export-runs.mjs pulls those back to
 * disk on a deployed instance where the process has no local runs/ to read) and a RequestDoc
 * that GET /stats reconciles against. `tokens` here is a single total, not the {in,out} split
 * the done event carries — that difference is deliberate and is why this is its own adapter.
 *
 * A2: a run that hit a cap is terminated "cap", an error is "error"; only a run that finished
 * on its own is "done". Reporting a capped run as done is a red line.
 */
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  COLLECTIONS,
  RunLog,
  type Depth,
  type RequestDoc,
  type RunDoc,
  type Terminated
} from '@lumina/contract';
import { db } from './db.js';
import { env } from './env.js';

export interface RunLogInput {
  requestId: string;
  userId: string;
  threadId: string;
  answerId: string;
  query: string;
  route: string;
  status: number;
  tokensIn: number;
  tokensOut: number;
  costUsd: number;
  wallClockSec: number;
  terminated: Terminated;
  depth: Depth;
  toolCalls: RunLog['toolCalls'];
}

export async function writeRunLog(input: RunLogInput): Promise<void> {
  const runLog: RunLog = {
    tokens: input.tokensIn + input.tokensOut,
    wallClockSec: input.wallClockSec,
    costUsd: input.costUsd,
    terminated: input.terminated,
    depth: input.depth,
    toolCalls: input.toolCalls
  };
  // Validate against the contract before it lands — a malformed run log fails a gate silently.
  RunLog.parse(runLog);

  // 1. the file the gates read
  try {
    writeFileSync(join(env.runsDir, `${input.requestId}.json`), JSON.stringify(runLog, null, 2));
  } catch {
    // A read-only FS on a deployed host is fine: the Mongo copy below is the source of truth there.
  }

  // 2. the queryable copy (+ a RequestDoc for /stats), best-effort so logging never fails an answer
  try {
    const d = await db();
    const now = new Date();
    const runDoc: RunDoc = {
      ...runLog,
      requestId: input.requestId,
      userId: input.userId,
      threadId: input.threadId,
      answerId: input.answerId,
      query: input.query,
      createdAt: now
    };
    await d
      .collection<RunDoc>(COLLECTIONS.runs)
      .updateOne({ requestId: input.requestId }, { $set: runDoc }, { upsert: true });

    const reqDoc: RequestDoc = {
      requestId: input.requestId,
      userId: input.userId,
      route: input.route,
      status: input.status,
      ms: input.wallClockSec * 1000,
      tokensIn: input.tokensIn,
      tokensOut: input.tokensOut,
      costUsd: input.costUsd,
      toolCalls: input.toolCalls.length,
      terminated: input.terminated,
      depth: input.depth,
      createdAt: now
    };
    await d.collection<RequestDoc>(COLLECTIONS.requests).insertOne(reqDoc);
  } catch {
    /* logging is best-effort; a persistence hiccup must not turn a good answer into a 502 */
  }
}

/** A lightweight request record for routes that are not answers (uploads, thread creation…). */
export async function recordRequest(doc: Omit<RequestDoc, 'createdAt'>): Promise<void> {
  try {
    await (await db())
      .collection<RequestDoc>(COLLECTIONS.requests)
      .insertOne({ ...doc, createdAt: new Date() });
  } catch {
    /* best-effort */
  }
}

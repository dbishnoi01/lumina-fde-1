/**
 * LUMINA agent service — the AI backend. Provider keys live only in this process.
 *
 * /health is the boring half; everything below it is the real work: the quick + deep loop and
 * its SSE stream, the search cache, threads & messages, memory, run logs, Spaces + the jobs
 * worker, and hybrid retrieval. The browser never reaches this service directly — the gateway
 * does — but every route enforces X-User-Id itself, because the cap and the ownership checks
 * cannot live on the edge.
 *
 * Three rules hold across the file: fail loud (a provider exception is a 502 / an `error` frame,
 * never a plausible answer), grounded or nothing, and depth is opted into (quick never plans).
 */
import express from 'express';
import multer from 'multer';
import pino from 'pino';
import { mkdirSync } from 'node:fs';
import {
  AskBody,
  CreateSpaceBody,
  CreateThreadBody,
  HealthResponse,
  MAX_UPLOAD_BYTES,
  USER_HEADER,
  type ErrorBody
} from '@lumina/contract';
import { env } from './env.js';
import { pingDb } from './db.js';
import { emitter, sseHeaders } from './sse.js';
import { runAsk } from './loop.js';
import {
  appendUserMessage,
  createThread,
  getThread,
  listThreads,
  loadHistory,
  threadOwnedBy
} from './threads.js';
import { deleteMemory, listMemories } from './memory.js';
import { createSpace, listDocuments, listSpaces, spaceOwnedBy, uploadDocument, UploadError } from './spaces.js';
import { assertDeepAllowed, DeepCapError } from './deepCap.js';
import { getStats } from './stats.js';
import { recordRequest } from './runlog.js';
import { ensureIndexes } from './indexes.js';
import { startWorker } from './worker.js';

const log = pino({ level: env.logLevel });
const app = express();

app.disable('x-powered-by');
app.use((req, res, next) =>
  req.path.endsWith('/documents') && req.method === 'POST'
    ? next()
    : express.json({ limit: '1mb' })(req, res, next)
);

mkdirSync(env.runsDir, { recursive: true });
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: MAX_UPLOAD_BYTES } });

// -------------------------------------------------------------- helpers

/** Wrap an async handler so a thrown error becomes a 502 via the error middleware. */
const h =
  (fn: (req: express.Request, res: express.Response) => Promise<void>) =>
  (req: express.Request, res: express.Response, next: express.NextFunction) =>
    fn(req, res).catch(next);

/** X-User-Id is required on every route but /health (401 without it). */
function userId(req: express.Request): string {
  const id = req.header(USER_HEADER)?.trim();
  if (!id) throw new Unauthorized();
  return id;
}
class Unauthorized extends Error {}

const err = (res: express.Response, status: number, body: Omit<ErrorBody, 'status'>) =>
  res.status(status).json({ ...body, status });

// -------------------------------------------------------------- /health

app.get('/health', async (_req, res) => {
  const dbStatus = await pingDb();
  const body: HealthResponse = {
    status: dbStatus === 'ok' ? 'ok' : 'degraded',
    model: env.llmModel,
    searchProvider: env.searchProvider,
    vectorStore: env.vectorBackend,
    db: dbStatus,
    ai: { status: 'ok' }
  };
  res.status(dbStatus === 'ok' ? 200 : 503).json(body);
});

// -------------------------------------------------------------- stats

app.get(
  '/stats',
  h(async (req, res) => {
    res.json(await getStats(userId(req)));
  })
);

// -------------------------------------------------------------- threads

app.post(
  '/threads',
  h(async (req, res) => {
    const uid = userId(req);
    const parsed = CreateThreadBody.safeParse(req.body ?? {});
    if (!parsed.success) return void err(res, 400, { error: parsed.error.message });
    res.status(201).json(await createThread(uid, parsed.data.title));
  })
);

app.get(
  '/threads',
  h(async (req, res) => {
    res.json(await listThreads(userId(req)));
  })
);

app.get(
  '/threads/:threadId',
  h(async (req, res) => {
    const uid = userId(req);
    const thread = await getThread(uid, req.params.threadId as string);
    if (!thread) return void err(res, 404, { error: 'thread not found' });
    res.json(thread);
  })
);

// The stream. Pre-flight checks that can still carry a real status code happen BEFORE the
// SSE headers; once the stream is open, a failure is an `error` frame, not a status change.
app.post(
  '/threads/:threadId/ask',
  h(async (req, res) => {
    const requestStart = Date.now();
    const uid = userId(req);

    const parsed = AskBody.safeParse(req.body ?? {});
    if (!parsed.success) return void err(res, 400, { error: parsed.error.message });
    const body = parsed.data;
    const depth = body.depth ?? 'quick';
    const mode = body.mode ?? 'auto';

    const threadId = req.params.threadId as string;
    if (!(await threadOwnedBy(uid, threadId))) return void err(res, 404, { error: 'thread not found' });

    if (mode === 'docs' && !body.spaceId) return void err(res, 400, { error: 'docs mode requires a spaceId' });
    if (body.spaceId && !(await spaceOwnedBy(uid, body.spaceId))) {
      return void err(res, 404, { error: 'space not found' });
    }

    if (depth === 'deep') {
      try {
        await assertDeepAllowed(uid);
      } catch (e) {
        if (e instanceof DeepCapError) {
          return void res
            .status(429)
            .json({ error: e.message, status: 429, resetsAt: e.resetsAt } satisfies ErrorBody);
        }
        throw e;
      }
    }

    await appendUserMessage(uid, threadId, body.query);
    const history = await loadHistory(uid, threadId);

    sseHeaders(res);
    await runAsk({
      emit: emitter(res),
      requestId: String(req.header('x-request-id') ?? `req_${Date.now().toString(36)}`),
      requestStart,
      userId: uid,
      threadId,
      query: body.query,
      mode,
      depth,
      ...(body.spaceId ? { spaceId: body.spaceId } : {}),
      history
    });
    res.end();
  })
);

// -------------------------------------------------------------- memory

app.get(
  '/memory',
  h(async (req, res) => {
    res.json({ memories: await listMemories(userId(req)) });
  })
);

app.delete(
  '/memory/:memoryId',
  h(async (req, res) => {
    const uid = userId(req);
    const ok = await deleteMemory(uid, req.params.memoryId as string);
    if (!ok) return void err(res, 404, { error: 'memory not found' });
    res.status(204).end();
  })
);

// -------------------------------------------------------------- spaces & documents

app.post(
  '/spaces',
  h(async (req, res) => {
    const uid = userId(req);
    const parsed = CreateSpaceBody.safeParse(req.body ?? {});
    if (!parsed.success) return void err(res, 400, { error: parsed.error.message });
    res.status(201).json(await createSpace(uid, parsed.data.name));
  })
);

app.get(
  '/spaces',
  h(async (req, res) => {
    res.json(await listSpaces(userId(req)));
  })
);

app.post(
  '/spaces/:spaceId/documents',
  upload.single('file'),
  h(async (req, res) => {
    const uid = userId(req);
    const spaceId = req.params.spaceId as string;
    if (!(await spaceOwnedBy(uid, spaceId))) return void err(res, 404, { error: 'space not found' });
    try {
      const file = req.file;
      if (!file) return void err(res, 400, { error: 'no file uploaded (field name must be "file")' });
      const out = await uploadDocument(uid, spaceId, {
        buffer: file.buffer,
        originalname: file.originalname,
        mimetype: file.mimetype,
        size: file.size
      });
      await recordRequest({ requestId: String(req.header('x-request-id') ?? 'upload'), userId: uid, route: 'POST /spaces/:spaceId/documents', status: 202, ms: 0 });
      res.status(202).json(out);
    } catch (e) {
      if (e instanceof UploadError) return void err(res, e.status, { error: e.message });
      throw e;
    }
  })
);

app.get(
  '/spaces/:spaceId/documents',
  h(async (req, res) => {
    const uid = userId(req);
    const spaceId = req.params.spaceId as string;
    if (!(await spaceOwnedBy(uid, spaceId))) return void err(res, 404, { error: 'space not found' });
    res.json(await listDocuments(uid, spaceId));
  })
);

// -------------------------------------------------------------- 404 + errors

app.use((req, res) => res.status(404).json({ error: `no route ${req.method} ${req.path}`, status: 404 }));

app.use((error: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  if (error instanceof Unauthorized) {
    return void res.status(401).json({ error: 'X-User-Id header required', status: 401 });
  }
  if (res.headersSent) return; // an SSE stream already owns the response
  log.error({ err: error.message }, 'agent error');
  res.status(502).json({ error: error.message, status: 502 });
});

app.listen(env.port, env.host, () => {
  ensureIndexes().catch((e) => log.warn({ err: (e as Error).message }, 'ensureIndexes failed (non-fatal)'));
  if (process.env.AGENT_RUN_WORKER !== '0') startWorker();
  log.info(
    {
      port: env.port,
      model: env.llmModel,
      searchProvider: env.searchProvider,
      vectorStore: env.vectorBackend,
      caps: {
        quick: { toolCalls: env.maxToolCalls, wallClockSec: env.maxWallClockSec },
        deep: { toolCalls: env.maxToolCallsDeep, wallClockSec: env.maxWallClockSecDeep, dailyCap: env.deepDailyCap }
      }
    },
    'agent up'
  );
});

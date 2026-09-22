/**
 * LUMINA gateway — the software backend. PROVIDED SKELETON: YOU BUILD THIS OUT.
 *
 * What is already here: the server, CORS, the request id, the pino request log, /health
 * (which nests the agent service's health), a 501 for every contract route, and the
 * static hosting of web/dist. That is deliberately the boring half.
 *
 * What you build (backend/gateway/, see README Part 2):
 *   1. X-User-Id enforcement           → 401 without it, on every route but /health
 *   2. zod validation from @lumina/contract → 400 on a bad body, with the zod message
 *   3. a per-user rate limit           → 429
 *   4. the proxy to the agent service, and SSE pass-through for /threads/:id/ask
 *   5. 502 for any upstream failure    → never a 2xx when the agent threw
 *
 * The browser talks ONLY to this service. No provider key is ever read here.
 */
import express from 'express';
import cors from 'cors';
import { pinoHttp } from 'pino-http';
import pino from 'pino';
import { randomUUID } from 'node:crypto';
import { existsSync, readFile } from 'node:fs';
import {
  AskBody,
  CreateSpaceBody,
  CreateThreadBody,
  HealthResponse,
  REQUEST_HEADER,
  USER_HEADER
} from '@lumina/contract';
import { env } from './env.js';
import { rateLimit, requireUser, validateBody } from './middleware.js';
import { proxyJson, proxyStream, proxyUpload } from './proxy.js';

const log = pino({ level: env.logLevel });
const app = express();

app.disable('x-powered-by');
app.use(cors({ origin: env.corsOrigins, credentials: false, exposedHeaders: [REQUEST_HEADER] }));

// One request id, reused if the caller sent one, generated if not, forwarded to the agent
// service and logged by both. This is what makes one request greppable end to end.
app.use((req, res, next) => {
  const id = (req.header(REQUEST_HEADER) ?? `req_${randomUUID().slice(0, 12)}`).trim();
  res.locals.requestId = id;
  res.setHeader(REQUEST_HEADER, id);
  next();
});

app.use(
  pinoHttp({
    logger: log,
    genReqId: (_req, res) => String(res.locals.requestId),
    customProps: (req, res) => ({
      requestId: res.locals.requestId,
      userId: req.header(USER_HEADER) ?? null
    }),
    // The ask route is a stream; one line when it closes is the useful line.
    autoLogging: true
  })
);

// JSON everywhere except the multipart upload route, which your handler owns.
app.use((req, res, next) =>
  req.path.endsWith('/documents') && req.method === 'POST'
    ? next()
    : express.json({ limit: '1mb' })(req, res, next)
);

// ---------------------------------------------------------------- /health (implemented)

app.get('/health', async (_req, res) => {
  let ai: { status: 'ok' | 'down' } & Record<string, unknown> = { status: 'down' };
  try {
    const upstream = await fetch(`${env.agentUrl}/health`, { signal: AbortSignal.timeout(3000) });
    const body = (await upstream.json()) as Record<string, unknown>;
    ai = { ...body, status: upstream.ok ? 'ok' : 'down' };
  } catch (err) {
    // Health tells the truth about a dead dependency. It never pretends.
    ai = { status: 'down', error: (err as Error).message };
  }

  const body: HealthResponse = {
    status: ai.status === 'ok' ? 'ok' : 'degraded',
    model: String(ai.model ?? 'unset'),
    searchProvider: (ai.searchProvider as HealthResponse['searchProvider']) ?? 'tavily',
    vectorStore: (ai.vectorStore as HealthResponse['vectorStore']) ?? 'atlas-vector-search',
    db: (ai.db as HealthResponse['db']) ?? 'down',
    ai
  };
  res.status(ai.status === 'ok' ? 200 : 503).json(body);
});

// ---------------------------------------------------------------- /evals/report.json (public)

// The Product Evaluation the eval skill writes. Served from disk, not the agent, and NOT
// behind X-User-Id: a stranger opening /evals on the Vercel URL must be able to read it.
app.get('/evals/report.json', (_req, res) => {
  readFile(env.reportPath, 'utf8', (readErr, data) => {
    if (readErr) {
      return void res.status(404).json({
        error: 'no eval report yet — run the eval skill to write reports/report.json',
        status: 404,
        requestId: String(res.locals.requestId)
      });
    }
    res.setHeader('content-type', 'application/json; charset=utf-8');
    res.send(data);
  });
});

// ---------------------------------------------------------------- contract routes → agent

// Every route below is authed (401), rate-limited per user (429), and — where it takes a
// body — validated against the contract (400) before it is mirrored to the agent. The path
// table is shared, so each request is forwarded to the same path on the agent.
const guard = [requireUser, rateLimit];

/** Route a thrown proxy error to the 502 error middleware instead of crashing the process. */
const a =
  (fn: (req: express.Request, res: express.Response) => Promise<void>) =>
  (req: express.Request, res: express.Response, next: express.NextFunction) =>
    fn(req, res).catch(next);

app.get('/stats', guard, a(proxyJson));

app.post('/threads', guard, validateBody(CreateThreadBody), a(proxyJson));
app.get('/threads', guard, a(proxyJson));
app.get('/threads/:threadId', guard, a(proxyJson));
app.post('/threads/:threadId/ask', guard, validateBody(AskBody), a(proxyStream));

app.get('/memory', guard, a(proxyJson));
app.delete('/memory/:memoryId', guard, a(proxyJson));

app.post('/spaces', guard, validateBody(CreateSpaceBody), a(proxyJson));
app.get('/spaces', guard, a(proxyJson));
app.post('/spaces/:spaceId/documents', guard, a(proxyUpload));
app.get('/spaces/:spaceId/documents', guard, a(proxyJson));

// ---------------------------------------------------------------- static UI

// In production the gateway serves the built UI, so / and /evals come from one origin.
if (existsSync(env.webDist)) {
  app.use(express.static(env.webDist));
  app.get(/^(?!\/(health|stats|threads|memory|spaces|artifacts|evals)).*/, (_req, res) => {
    res.sendFile(`${env.webDist}/index.html`);
  });
}

app.use((req, res) => {
  res.status(404).json({ error: `no route ${req.method} ${req.path}`, status: 404 });
});

// A thrown error is a 502 with a log line, never a 200 with a plausible body (rule A1).
app.use((err: Error, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  log.error({ err, requestId: res.locals.requestId }, 'gateway error');
  if (res.headersSent) return; // an SSE stream already owns the response
  res.status(502).json({ error: err.message, status: 502, requestId: String(res.locals.requestId) });
});

app.listen(env.port, () => {
  log.info(
    { port: env.port, agentUrl: env.agentUrl, cors: env.corsOrigins },
    'gateway up — every route but /health returns 501 until you build it'
  );
});

/**
 * The proxy to the agent service. The gateway is a mirror, not an interpreter: it forwards
 * the agent's status and body VERBATIM (a 404 / 413 / 429 / 502 from the agent survives the
 * hop intact), and if the agent is unreachable it returns 502 — never a plausible 2xx.
 *
 * The gateway and the agent share one path table, so every request is forwarded to the same
 * path on the agent (`req.originalUrl`). Only two headers cross over: the verified X-User-Id
 * and the X-Request-Id, so one request is greppable end to end. No provider key is ever read
 * here; the agent is the only holder of keys.
 */
import type { Request, Response } from 'express';
import { MAX_UPLOAD_BYTES, REQUEST_HEADER, USER_HEADER, type ErrorBody } from '@lumina/contract';
import { env } from './env.js';
import { sseHeaders, sseSend } from './sse.js';

/** Buffer overhead a multipart envelope adds around a max-size file: boundaries + field headers. */
const UPLOAD_ENVELOPE_SLACK = 1024 * 1024;

const upstreamHeaders = (req: Request, res: Response, extra: Record<string, string> = {}) => {
  const headers: Record<string, string> = { [REQUEST_HEADER]: String(res.locals.requestId), ...extra };
  const uid = req.header(USER_HEADER)?.trim();
  if (uid) headers[USER_HEADER] = uid;
  return headers;
};

/** The agent threw, timed out, or is down: 502, with the same shape every other error uses. */
function badGateway(res: Response, err: unknown): void {
  res
    .status(502)
    .json({ error: `agent unreachable: ${(err as Error).message}`, status: 502, requestId: String(res.locals.requestId) } satisfies ErrorBody);
}

/** Copy the agent's status, content-type, and body onto our response, unchanged. */
async function mirror(upstream: globalThis.Response, res: Response): Promise<void> {
  res.status(upstream.status);
  const contentType = upstream.headers.get('content-type');
  if (contentType) res.setHeader('content-type', contentType);
  if (upstream.status === 204) return void res.end();
  res.send(await upstream.text());
}

/** GET / DELETE / non-stream POST: forward, then mirror the agent's response verbatim. */
export async function proxyJson(req: Request, res: Response): Promise<void> {
  const hasBody = req.method === 'POST' || req.method === 'PUT' || req.method === 'PATCH';
  let upstream: globalThis.Response;
  try {
    upstream = await fetch(`${env.agentUrl}${req.originalUrl}`, {
      method: req.method,
      headers: upstreamHeaders(req, res, hasBody ? { 'content-type': 'application/json' } : {}),
      body: hasBody ? JSON.stringify(req.body ?? {}) : undefined,
      signal: AbortSignal.timeout(env.upstreamTimeoutMs)
    });
  } catch (err) {
    return badGateway(res, err);
  }
  await mirror(upstream, res);
}

/**
 * POST /threads/:id/ask — the stream. Pre-flight failures (401/404/400/429) come back from
 * the agent as JSON *before* it opens the stream, so we mirror those with their status. Only
 * once the agent commits to `text/event-stream` do we open SSE and pipe bytes straight
 * through, flushing per frame. If the stream breaks mid-flight we emit an `error` frame — an
 * answer never ends as a truncated success dressed as done (rule A1).
 */
export async function proxyStream(req: Request, res: Response): Promise<void> {
  const ac = new AbortController();
  res.on('close', () => ac.abort());

  let upstream: globalThis.Response;
  try {
    upstream = await fetch(`${env.agentUrl}${req.originalUrl}`, {
      method: 'POST',
      headers: upstreamHeaders(req, res, { 'content-type': 'application/json' }),
      body: JSON.stringify(req.body ?? {}),
      signal: ac.signal
    });
  } catch (err) {
    return badGateway(res, err);
  }

  const contentType = upstream.headers.get('content-type') ?? '';
  if (!upstream.ok || !contentType.includes('text/event-stream') || !upstream.body) {
    return mirror(upstream, res);
  }

  sseHeaders(res);
  const reader = upstream.body.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
      // @ts-expect-error `flush` exists when a compression middleware is present; harmless otherwise.
      if (typeof res.flush === 'function') res.flush();
    }
  } catch (err) {
    if (!res.writableEnded) sseSend(res, 'error', { status: 502, error: `upstream stream failed: ${(err as Error).message}` });
  } finally {
    res.end();
  }
}

/**
 * POST /spaces/:id/documents — multipart. We do not parse it (the agent's multer owns the
 * `file` field); we buffer the raw body and re-post it with its boundary intact. Oversize is
 * rejected here on content-length so a 25 MB+ upload never costs an upstream hop.
 */
export async function proxyUpload(req: Request, res: Response): Promise<void> {
  const declared = Number(req.header('content-length') ?? 0);
  const ceiling = MAX_UPLOAD_BYTES + UPLOAD_ENVELOPE_SLACK;
  if (declared && declared > ceiling) {
    return void res
      .status(413)
      .json({ error: `upload exceeds ${MAX_UPLOAD_BYTES} bytes`, status: 413, requestId: String(res.locals.requestId) } satisfies ErrorBody);
  }

  let body: Buffer;
  try {
    body = await readBody(req, ceiling);
  } catch {
    return void res
      .status(413)
      .json({ error: `upload exceeds ${MAX_UPLOAD_BYTES} bytes`, status: 413, requestId: String(res.locals.requestId) } satisfies ErrorBody);
  }

  let upstream: globalThis.Response;
  try {
    upstream = await fetch(`${env.agentUrl}${req.originalUrl}`, {
      method: 'POST',
      headers: upstreamHeaders(req, res, { 'content-type': req.header('content-type') ?? 'application/octet-stream' }),
      body,
      signal: AbortSignal.timeout(env.upstreamTimeoutMs)
    });
  } catch (err) {
    return badGateway(res, err);
  }
  await mirror(upstream, res);
}

/** Collect a request stream into a Buffer, aborting the moment it crosses the ceiling. */
function readBody(req: Request, ceiling: number): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on('data', (chunk: Buffer) => {
      size += chunk.length;
      if (size > ceiling) {
        req.destroy();
        reject(new Error('payload too large'));
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

import type { Response } from 'express';

/**
 * SSE plumbing for the agent's own /threads/:id/ask stream. The gateway proxies these bytes
 * through unchanged, so the same three rules apply here as in backend/gateway/src/sse.ts:
 * disable transforms, flush the headers, and flush after every frame — or the tokens buffer
 * up and arrive together at the end, which reads as "the model is slow" and fails TTFT for a
 * reason no profiler shows you. Do NOT put compression() in front of the ask route.
 */
export function sseHeaders(res: Response): void {
  res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache, no-transform');
  res.setHeader('Connection', 'keep-alive');
  res.setHeader('X-Accel-Buffering', 'no');
  res.flushHeaders();
}

/** Write one SSE frame and flush it. The blank line terminates the frame. */
export function sseSend(res: Response, event: string, data: unknown): void {
  res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  // @ts-expect-error `flush` exists when a compression middleware is present; harmless otherwise.
  if (typeof res.flush === 'function') res.flush();
}

/** A bound emitter so the loop never touches the raw Response. */
export type Emit = (event: string, data: unknown) => void;

export function emitter(res: Response): Emit {
  return (event, data) => sseSend(res, event, data);
}

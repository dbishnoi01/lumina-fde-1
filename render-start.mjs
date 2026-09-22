#!/usr/bin/env node
/**
 * Render process supervisor — runs the whole backend in ONE free-tier container.
 *
 * Render exposes exactly one port ($PORT). We give it to the gateway; the agent binds
 * 127.0.0.1:8000 (loopback = private) and the worker runs inside the agent process
 * (AGENT_RUN_WORKER=1). So the browser can reach the gateway and nothing else — the agent,
 * which holds every key and enforces the deep-search cap, is not publicly reachable.
 *
 * Both env.ts files resolve ../../.env, ../../runs, ../../reports/report.json relative to
 * process.cwd(), so each child MUST run with cwd = its own package dir (repo root is ../../).
 *
 * If either child exits we tear the other down and exit non-zero, so Render restarts the
 * whole container rather than limping along with half the backend gone.
 */
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const root = dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT ?? '10000'; // Render injects PORT; 10000 is its local default.

const children = [];
let shuttingDown = false;

function shutdown(code) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const { child } of children) {
    if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM');
  }
  // Give children a moment to flush, then exit hard.
  setTimeout(() => process.exit(code), 3000).unref();
}

function start(name, cwd, extraEnv) {
  const child = spawn(process.execPath, ['dist/index.js'], {
    cwd: resolve(root, cwd),
    env: { ...process.env, ...extraEnv },
    stdio: 'inherit'
  });
  child.on('exit', (code, signal) => {
    if (shuttingDown) return;
    console.error(`[render-start] ${name} exited (code=${code} signal=${signal}) — restarting container`);
    shutdown(1);
  });
  child.on('error', (err) => {
    if (shuttingDown) return;
    console.error(`[render-start] ${name} failed to start: ${err.message}`);
    shutdown(1);
  });
  children.push({ name, child });
}

process.on('SIGTERM', () => shutdown(0));
process.on('SIGINT', () => shutdown(0));

// Agent + worker: private, loopback only.
start('agent', 'backend/agent', {
  PORT_AGENT: '8000',
  AGENT_HOST: '127.0.0.1',
  AGENT_RUN_WORKER: '1'
});

// Gateway: the only public process. Binds Render's $PORT, talks to the agent over loopback.
start('gateway', 'backend/gateway', {
  PORT_GATEWAY: String(PORT),
  AGENT_URL: 'http://127.0.0.1:8000'
});

console.log(`[render-start] gateway → :${PORT} (public), agent → 127.0.0.1:8000 (private) + worker`);

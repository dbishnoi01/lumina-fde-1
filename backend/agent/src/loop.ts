/**
 * The agent loop — both gears. This is the file the assignment means by "the real work".
 *
 * QUICK: prime the model with the thread's history and a depth/mode-filtered toolbelt; let it
 * plan → call a tool → observe → repeat until it is ready to answer or a cap is hit. Then build
 * the source list from what was actually retrieved THIS request, emit `sources`, and stream a
 * grounded synthesis. `sources` before the first `token`, always (the UI renders chips as text
 * arrives). Every step is a `trace` event with the model's own reason; a tool failure is a
 * visible ok:false with an error string, never a swallowed empty result.
 *
 * DEEP: `plan_research` decomposes the question and we stream a `plan` event BEFORE retrieving
 * anything; then we research each sub-question, tagging every trace step and source with its
 * subQuestion; then we merge everything into ONE contiguous citation numbering and synthesise a
 * structured answer. Deep runs under the wider caps and behind the daily cap (checked upstream).
 *
 * Three rules held throughout: fail loud (a provider exception → `error` event, terminated
 * "error", no plausible answer); grounded or nothing (every [n] resolves to a retrieved source);
 * depth is opted into (a quick run is never handed plan_research — enforced by toolsFor()).
 */
import {
  Source as SourceSchema,
  TraceEvent as TraceEventSchema,
  newId,
  type AskMode,
  type Depth,
  type DoneEvent,
  type RunLog,
  type Source,
  type SubQuestion,
  type ToolName
} from '@lumina/contract';
import type { Emit } from './sse.js';
import { chat, streamChat, type ChatResult, type LlmMessage, type LlmTool, type Usage } from './providers/llm.js';
import { llmCostUsd, searchCostUsd } from './cost.js';
import { recordAnswer } from './metrics.js';
import { SearchTally } from './cache.js';
import { recallMemory } from './memory.js';
import { appendAssistantMessage } from './threads.js';
import { writeRunLog } from './runlog.js';
import { env } from './env.js';
import { locatorLabel, runTool, toolsFor, type Candidate, type ToolContext } from './tools.js';
import { planResearch } from './deep.js';

export interface AskParams {
  emit: Emit;
  requestId: string;
  requestStart: number;
  userId: string;
  threadId: string;
  query: string;
  mode: AskMode;
  depth: Depth;
  spaceId?: string;
  history: LlmMessage[];
}

interface RunState {
  usage: Usage;
  toolCallsLog: RunLog['toolCalls'];
  step: number;
  terminated: DoneEvent['terminated'];
  maxToolCalls: number;
  maxWallClockSec: number;
  requestStart: number;
}

const addUsage = (u: Usage, add: Usage) => {
  u.in += add.in;
  u.out += add.out;
};

const elapsedSec = (start: number) => (Date.now() - start) / 1000;

/**
 * Recognise the Groq quirk where gpt-oss emits a call to a tool that is not in `request.tools`
 * and Groq rejects the whole completion with a 400 (`code: tool_use_failed`). We deliberately
 * withhold tools — search_documents when no Space is selected, and plan_research from every
 * quick run (rule R2) — so a hallucinated call to one of them must NOT be allowed to kill an
 * otherwise-healthy run. This detects that specific failure so the caller can recover from it.
 */
function isUnavailableToolError(err: unknown): boolean {
  const e = err as { code?: string; message?: string };
  const msg = (e?.message ?? '').toLowerCase();
  return e?.code === 'tool_use_failed' || msg.includes('tool call validation') || msg.includes('not in request.tools');
}

/**
 * One decision turn, hardened against the quirk above. On a tool-validation 400 we nudge the
 * model back to the tools it actually has and retry, bounded — never inventing an answer, just
 * refusing to let a stray tool name become a 502. Any other provider error still throws (fail
 * loud). `messages` is mutated in place, so the correction persists into the next turn.
 */
async function chatOrCorrect(messages: LlmMessage[], tools: LlmTool[]): Promise<ChatResult> {
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await chat(messages, tools);
    } catch (err) {
      if (attempt < 2 && isUnavailableToolError(err)) {
        messages.push({
          role: 'user',
          content:
            'You attempted to call a tool that is not available here. Use ONLY the tools provided to you; if you already have enough evidence, answer now without calling any tool.'
        });
        continue;
      }
      throw err;
    }
  }
}

/**
 * Run one research loop until the model stops asking for tools, or the local/global budget is
 * exhausted. Mutates ctx.candidates and state. `task` seeds a fresh conversation (for deep, one
 * per sub-question); on a cap it sets terminated="cap" and returns what it has.
 */
async function research(
  params: AskParams,
  state: RunState,
  ctx: ToolContext,
  task: LlmMessage[],
  localToolBudget: number
): Promise<void> {
  const tools = toolsFor(params.depth, params.mode, Boolean(params.spaceId));
  const messages: LlmMessage[] = [...task];
  let localCalls = 0;

  for (;;) {
    if (state.toolCallsLog.length >= state.maxToolCalls || localCalls >= localToolBudget) {
      state.terminated = 'cap';
      return;
    }
    if (elapsedSec(state.requestStart) >= state.maxWallClockSec) {
      state.terminated = 'cap';
      return;
    }

    const res = await chatOrCorrect(messages, tools);
    addUsage(state.usage, res.usage);

    if (res.toolCalls.length === 0) return; // model is ready to answer

    messages.push({
      role: 'assistant',
      // Canonical OpenAI shape for a tool-call turn is content:null, not "". Groq tolerates
      // the empty string; Gemini's OpenAI-compatible endpoint (the fallback) 400s on it.
      content: res.content || null,
      tool_calls: res.toolCalls.map((tc) => ({
        id: tc.id,
        type: 'function',
        function: { name: tc.name, arguments: JSON.stringify(tc.args) }
      }))
    });

    for (const tc of res.toolCalls) {
      if (state.toolCallsLog.length >= state.maxToolCalls || localCalls >= localToolBudget) {
        state.terminated = 'cap';
        // Answer the tool_call so the transcript stays valid even though we stop here.
        messages.push({ role: 'tool', tool_call_id: tc.id, content: 'skipped: budget reached' });
        continue;
      }
      state.step += 1;
      localCalls += 1;
      const t0 = Date.now();
      let ok = true;
      let error: string | undefined;
      let observation = '';
      try {
        observation = await runTool(tc.name, tc.args, ctx);
      } catch (err) {
        ok = false;
        error = (err as Error).message;
        observation = `ERROR: ${error}`;
      }
      const ms = Date.now() - t0;

      const trace = TraceEventSchema.parse({
        step: state.step,
        tool: tc.name,
        input: tc.args,
        ok,
        ms,
        ...(typeof tc.args.reason === 'string' && tc.args.reason.trim() ? { reason: tc.args.reason } : {}),
        ...(error ? { error } : {}),
        ...(ctx.subQuestion ? { subQuestion: ctx.subQuestion } : {})
      });
      params.emit('trace', trace);
      state.toolCallsLog.push({ name: tc.name as ToolName, ok, ...(error ? { error } : {}), ms });
      messages.push({ role: 'tool', tool_call_id: tc.id, content: observation.slice(0, 4000) });
    }
  }
}

/**
 * The research system prompt, built to match the tools the run actually has — the prompt must
 * never name a tool that toolsFor() withheld, or the model reaches for it and Groq 400s the turn
 * (see chatOrCorrect). So web guidance appears only when web is available, and the documents line
 * only when a Space is selected.
 */
function researchSystem(mode: AskMode, hasSpace: boolean): string {
  const webAvailable = mode === 'web' || mode === 'auto';
  const docsAvailable = (mode === 'docs' || mode === 'auto') && hasSpace;
  const lines = [
    "You are LUMINA, a research assistant. Answer the user's question by gathering evidence with the tools provided, then stop."
  ];
  if (webAvailable)
    lines.push(
      '- For a web question: run at most one or two web_search calls, then fetch_page the 1–3 most relevant results (you may request several fetch_page calls at once). Only a fetched page can be cited — never rely on a search snippet.'
    );
  if (docsAvailable)
    lines.push('- For a documents question: use search_documents over the current Space, and cite passages by their page locator.');
  lines.push('- Save a durable user preference with save_memory only when the user clearly states one.');
  lines.push(
    'Call ONLY the tools provided to you. When you have enough evidence, reply WITHOUT calling any tool. Do not answer from prior knowledge alone; ground every claim in what you retrieved.'
  );
  return lines.join('\n');
}

function sourcesBlock(sources: Source[]): string {
  if (sources.length === 0) {
    return 'SOURCES: none were retrieved. Say plainly that you could not find grounded information for this question, and do NOT use any [n] citations.';
  }
  const lines = sources.map((s) => {
    const head =
      s.kind === 'web'
        ? `[${s.n}] (web) ${s.title} — ${s.url}`
        : `[${s.n}] (doc) ${s.title}${s.locator ? `, ${locatorLabel(s.locator)}` : ''}`;
    const body = (s.kind === 'doc' ? s.snippet : s.snippet).slice(0, 1500);
    return `${head}\n${body}`;
  });
  return `SOURCES (cite by number as [n]; use ONLY these numbers; never invent a citation):\n\n${lines.join('\n\n')}`;
}

function buildSynthMessages(params: AskParams, sources: Source[], memories: string[]): LlmMessage[] {
  const memoryNote = memories.length
    ? `\n\nWhat you know about this user (apply it when relevant): ${memories.map((m) => `- ${m}`).join(' ')}`
    : '';
  const deepNote =
    params.depth === 'deep'
      ? '\n\nThis is a DEEP answer: structure it with brief sections, synthesise across sub-questions, and end with a short "What is still uncertain" note. Sources carry the sub-question they served.'
      : '';
  const system = `You write grounded, well-cited answers. Cite claims inline using plain ASCII square brackets like [1] or [2] — never full-width brackets (【 】) or superscripts. Use ONLY the numbered sources below; every [n] MUST match a listed source and every source you use must be cited. If the sources do not support a claim, say so rather than guessing.${deepNote}${memoryNote}\n\n${sourcesBlock(sources)}`;
  return [{ role: 'system', content: system }, ...params.history, { role: 'user', content: params.query }];
}

/** Merge candidates into one contiguous [1..n] numbering: dedup web by url, doc by chunk. */
function buildSources(candidates: Candidate[]): Source[] {
  const seen = new Set<string>();
  const out: Source[] = [];
  let n = 0;
  for (const c of candidates) {
    const key = c.kind === 'web' ? `web:${c.url}` : `doc:${c.chunkId}`;
    if (seen.has(key)) continue;
    seen.add(key);
    n += 1;
    const base =
      c.kind === 'web'
        ? { n, kind: 'web' as const, title: c.title, snippet: c.snippet, url: c.url }
        : {
            n,
            kind: 'doc' as const,
            title: c.title,
            snippet: c.text.slice(0, 300),
            docId: c.docId,
            locator: c.locator
          };
    out.push(SourceSchema.parse({ ...base, ...(c.subQuestion ? { subQuestion: c.subQuestion } : {}) }));
  }
  return out;
}

/**
 * The whole streamed answer. Assumes SSE headers are already sent (the route did the pre-stream
 * checks that can still return a real status code: auth, thread ownership, the deep cap). On a
 * provider exception it emits an `error` frame, marks the run terminated "error", and still
 * writes an honest run log — it never streams a plausible answer over a failure.
 */
export async function runAsk(params: AskParams): Promise<void> {
  const state: RunState = {
    usage: { in: 0, out: 0 },
    toolCallsLog: [],
    step: 0,
    terminated: 'done',
    maxToolCalls: params.depth === 'deep' ? env.maxToolCallsDeep : env.maxToolCalls,
    maxWallClockSec: params.depth === 'deep' ? env.maxWallClockSecDeep : env.maxWallClockSec,
    requestStart: params.requestStart
  };

  const tally = new SearchTally();
  const candidates: Candidate[] = [];
  const ctx: ToolContext = {
    userId: params.userId,
    threadId: params.threadId,
    ...(params.spaceId ? { spaceId: params.spaceId } : {}),
    query: params.query,
    tally,
    candidates,
    lastSearchHits: [],
    fetched: new Set<string>()
  };

  const answerId = newId('ans');
  // Recall memory in parallel with research so a saved preference reliably shapes the answer,
  // even if the model does not reach for recall_memory itself (the cross-thread gate).
  const memoriesP = recallMemory(params.userId, params.query, 5).catch(() => [] as string[]);

  let subQuestions: SubQuestion[] | undefined;
  let content = '';
  let firstTokenAt: number | null = null;

  try {
    if (params.depth === 'deep') {
      subQuestions = await planResearch(params, state);
      params.emit('plan', { subQuestions, reason: 'Decomposed the question to research each part independently.' });

      const perSubBudget = Math.max(2, Math.ceil((state.maxToolCalls - 1) / subQuestions.length));
      const system = researchSystem(params.mode, Boolean(params.spaceId));
      for (const sq of subQuestions) {
        if (elapsedSec(state.requestStart) >= state.maxWallClockSec || state.toolCallsLog.length >= state.maxToolCalls) {
          state.terminated = 'cap';
          break;
        }
        ctx.subQuestion = sq.i;
        const task: LlmMessage[] = [
          { role: 'system', content: system },
          {
            role: 'user',
            content: `Overall question: ${params.query}\n\nResearch ONLY this sub-question and gather sources for it:\n${sq.i}. ${sq.question}`
          }
        ];
        await research(params, state, ctx, task, perSubBudget);
      }
      ctx.subQuestion = undefined;
    } else {
      const task: LlmMessage[] = [
        { role: 'system', content: researchSystem(params.mode, Boolean(params.spaceId)) },
        { role: 'user', content: params.query }
      ];
      await research(params, state, ctx, task, state.maxToolCalls);
    }

    const sources = buildSources(candidates);
    params.emit('sources', sources);

    const memories = await memoriesP;
    const synth = await streamChat(buildSynthMessages(params, sources, memories), (text) => {
      if (firstTokenAt === null) firstTokenAt = Date.now();
      params.emit('token', { text: normalizeCitations(text) });
    });
    addUsage(state.usage, synth.usage);
    content = normalizeCitations(synth.content);

    const done = finishDone(params, state, answerId, firstTokenAt, tally, subQuestions);
    params.emit('done', done);

    await persist(params, state, answerId, content, sources, done, subQuestions, tally);
  } catch (err) {
    state.terminated = 'error';
    const message = (err as Error).message ?? 'agent error';
    params.emit('error', { status: 502, error: message });
    // Honest run log for the failure — A2 wants terminated recorded truthfully.
    const done = finishDone(params, state, answerId, firstTokenAt, tally, subQuestions);
    await persist(params, state, answerId, content, [], done, subQuestions, tally).catch(() => {});
  }
}

/**
 * Force citation markers to the ASCII `[n]` the contract's grounding check parses. Some models
 * (gpt-oss among them) emit full-width CJK brackets 【n】 or ［n］ — visually a citation, but
 * invisible to `citationNumbers()`, so an otherwise grounded answer would read as ungrounded.
 * Each bracket is a single codepoint that arrives whole within one token, so applying this per
 * streamed chunk is safe across token boundaries. Also folds full-width digits inside markers.
 */
function normalizeCitations(text: string): string {
  return text
    .replace(/【/g, '[')
    .replace(/】/g, ']')
    .replace(/［/g, '[')
    .replace(/］/g, ']')
    .replace(/[０-９]/g, (d) => String.fromCharCode(d.charCodeAt(0) - 0xff10 + 0x30));
}

function finishDone(
  params: AskParams,
  state: RunState,
  answerId: string,
  firstTokenAt: number | null,
  tally: SearchTally,
  subQuestions?: SubQuestion[]
): DoneEvent {
  const now = Date.now();
  const costUsd = llmCostUsd(state.usage) + searchCostUsd(tally.searches);
  return {
    answerId,
    latencyMs: now - params.requestStart,
    ttftMs: (firstTokenAt ?? now) - params.requestStart,
    model: env.llmModel,
    tokens: { in: state.usage.in, out: state.usage.out },
    costUsd,
    searchCached: tally.allCached,
    terminated: state.terminated,
    depth: params.depth,
    subQuestions: subQuestions?.length ?? 0
  };
}

async function persist(
  params: AskParams,
  state: RunState,
  answerId: string,
  content: string,
  sources: Source[],
  done: DoneEvent,
  subQuestions: SubQuestion[] | undefined,
  tally: SearchTally
): Promise<void> {
  recordAnswer({
    ttftMs: done.ttftMs,
    costUsd: done.costUsd,
    searches: tally.searches,
    hits: tally.hits
  });

  await appendAssistantMessage({
    userId: params.userId,
    threadId: params.threadId,
    content,
    answerId,
    sources,
    done,
    ...(subQuestions?.length ? { subQuestions } : {})
  }).catch(() => {});

  await writeRunLog({
    requestId: params.requestId,
    userId: params.userId,
    threadId: params.threadId,
    answerId,
    query: params.query,
    route: 'POST /threads/:threadId/ask',
    status: state.terminated === 'error' ? 502 : 200,
    tokensIn: state.usage.in,
    tokensOut: state.usage.out,
    costUsd: done.costUsd,
    wallClockSec: (Date.now() - params.requestStart) / 1000,
    terminated: state.terminated,
    depth: params.depth,
    toolCalls: state.toolCallsLog
  });
}

/**
 * LLM provider. Default: Groq (free tier) via the OpenAI-compatible API, so the `openai`
 * SDK already in package.json works by swapping `baseURL` — no new dependency, and the same
 * code path serves OpenAI or any other OpenAI-compatible gateway.
 *
 * Why Groq over Anthropic (the assignment default): free tier + very fast tokens, which
 * *helps* the TTFT p95 ≤ 2.5s and full-answer ≤ 12s SLAs rather than fighting them. The
 * cost is some answer quality on the "deep must be better" human gate; the interface below
 * is provider-agnostic, so pointing LLM_PROVIDER at Anthropic later is an adapter swap, not
 * a loop rewrite.
 *
 * Two entry points, matching the loop's two needs:
 *   - `chat()`      one decision turn: the model either asks for a tool or answers. Not streamed.
 *   - `streamChat()` the final synthesis turn: tokens streamed to the SSE channel as written.
 *
 * Provider fallback: Groq's free tier has a hard daily token cap. When it 429s we retry the
 * SAME request against Gemini's OpenAI-compatible endpoint (same SDK, key already deployed for
 * embeddings) so an in-flight answer still completes instead of dying. ONLY a rate-limit error
 * triggers the fallback — a genuine resilience feature, not a way to hide a broken provider.
 *
 * Fail loud: any NON-rate-limit provider error throws. There is no "return a plausible answer"
 * path here, and a rate-limit on the last backend still throws.
 */
import OpenAI from 'openai';
import type { Stream } from 'openai/streaming';
import { env, secrets } from '../env.js';

export interface LlmTool {
  name: string;
  description: string;
  /** JSON Schema for the tool's arguments. */
  parameters: Record<string, unknown>;
}

export interface ToolCall {
  id: string;
  name: string;
  /** Parsed arguments. Throws upstream if the model emitted invalid JSON. */
  args: Record<string, unknown>;
}

export interface Usage {
  in: number;
  out: number;
}

/** Loosely typed messages so we can carry tool results back to the model. */
export type LlmMessage = OpenAI.Chat.Completions.ChatCompletionMessageParam;

export interface ChatResult {
  /** Non-empty when the model answered directly instead of calling a tool. */
  content: string;
  toolCalls: ToolCall[];
  usage: Usage;
}

/** One OpenAI-compatible endpoint the loop can talk to, tried in order (primary first). */
interface Backend {
  client: OpenAI;
  model: string;
  label: string;
}

let groqClient: OpenAI | null = null;
let geminiClient: OpenAI | null = null;

/**
 * The ordered list of backends: Groq (primary), then Gemini (fallback) when enabled and its
 * key is present. Callers try each in order but only advance past one on a rate-limit error
 * (see {@link isRateLimited}); any other failure throws from the first backend.
 */
function backends(): Backend[] {
  if (env.llmProvider === 'anthropic') {
    // Deliberate: the $0 path is Groq. Wire @anthropic-ai/sdk here if you switch for the
    // deep-quality human gate — do not silently fall through to a wrong provider.
    throw new Error('LLM_PROVIDER=anthropic not wired; add @anthropic-ai/sdk or use LLM_PROVIDER=groq');
  }
  if (!secrets.groq) throw new Error('GROQ_API_KEY is not set');
  if (!groqClient) groqClient = new OpenAI({ apiKey: secrets.groq, baseURL: env.llmBaseUrl });
  const list: Backend[] = [{ client: groqClient, model: env.llmModel, label: 'groq' }];

  if (env.llmFallbackEnabled && secrets.gemini) {
    if (!geminiClient) geminiClient = new OpenAI({ apiKey: secrets.gemini, baseURL: env.geminiBaseUrl });
    list.push({ client: geminiClient, model: env.geminiChatModel, label: 'gemini' });
  }
  return list;
}

/**
 * A rate-limit / quota rejection — the only error class that should trip the fallback. Covers
 * the OpenAI SDK's 429 status and the shapes Groq/Gemini phrase it in (per-minute, per-day
 * token caps, quota exhaustion), which arrive as a 429 with an explanatory message.
 */
function isRateLimited(err: unknown): boolean {
  if (!err || typeof err !== 'object') return false;
  if ((err as { status?: number }).status === 429) return true;
  const msg = String((err as { message?: string }).message ?? '');
  return /\b429\b|rate limit|tokens per (day|minute)|\bTPD\b|\bTPM\b|quota|resource has been exhausted/i.test(msg);
}

function toOpenAiTools(tools: LlmTool[]): OpenAI.Chat.Completions.ChatCompletionTool[] {
  return tools.map((t) => ({
    type: 'function',
    function: { name: t.name, description: t.description, parameters: t.parameters }
  }));
}

function parseToolCalls(message: OpenAI.Chat.Completions.ChatCompletionMessage): ToolCall[] {
  const calls = message.tool_calls ?? [];
  return calls.map((c) => {
    if (c.type !== 'function') throw new Error(`unexpected tool call type: ${c.type}`);
    let args: Record<string, unknown> = {};
    if (c.function.arguments?.trim()) {
      try {
        args = JSON.parse(c.function.arguments) as Record<string, unknown>;
      } catch {
        throw new Error(`model emitted invalid JSON args for ${c.function.name}: ${c.function.arguments.slice(0, 200)}`);
      }
    }
    return { id: c.id, name: c.function.name, args };
  });
}

/**
 * One decision turn. Give the model the message history and the tools it is allowed to use
 * (already filtered by depth by the caller — a quick run never receives plan_research).
 * Returns either tool calls to run, or a direct answer.
 */
export async function chat(
  messages: LlmMessage[],
  tools: LlmTool[] = [],
  opts: { toolChoice?: 'auto' | 'required' | { name: string } } = {}
): Promise<ChatResult> {
  const toolChoice =
    typeof opts.toolChoice === 'object'
      ? ({ type: 'function', function: { name: opts.toolChoice.name } } as const)
      : (opts.toolChoice ?? 'auto');
  const list = backends();
  let lastErr: unknown;
  for (let i = 0; i < list.length; i++) {
    const b = list[i]!;
    try {
      const res = await b.client.chat.completions.create({
        model: b.model,
        messages,
        ...(tools.length > 0 ? { tools: toOpenAiTools(tools), tool_choice: toolChoice } : {}),
        temperature: 0.2,
        stream: false
      });
      const choice = res.choices[0];
      if (!choice) throw new Error('llm returned no choices');
      return {
        content: choice.message.content ?? '',
        toolCalls: parseToolCalls(choice.message),
        usage: { in: res.usage?.prompt_tokens ?? 0, out: res.usage?.completion_tokens ?? 0 }
      };
    } catch (err) {
      lastErr = err;
      // Advance to the next backend only on a rate-limit, and only if one remains.
      if (isRateLimited(err) && i < list.length - 1) continue;
      throw err;
    }
  }
  throw lastErr;
}

/**
 * The final synthesis turn, streamed. `onToken` fires per delta so the gateway can forward
 * it as an SSE `token` frame the moment it is written. Returns the full text and usage once
 * the stream closes (usage requires stream_options.include_usage, which Groq supports).
 */
export async function streamChat(
  messages: LlmMessage[],
  onToken: (text: string) => void
): Promise<{ content: string; usage: Usage }> {
  const list = backends();
  let lastErr: unknown;
  for (let i = 0; i < list.length; i++) {
    const b = list[i]!;
    // Open the stream first. A 429 (incl. Groq's daily token cap) is returned as the initial
    // response before any token arrives, so we can safely fall back here without ever having
    // emitted a partial answer. Once the stream is open we consume it — no mid-stream failover.
    let stream: Stream<OpenAI.Chat.Completions.ChatCompletionChunk>;
    try {
      stream = await b.client.chat.completions.create({
        model: b.model,
        messages,
        temperature: 0.2,
        stream: true,
        stream_options: { include_usage: true }
      });
    } catch (err) {
      lastErr = err;
      if (isRateLimited(err) && i < list.length - 1) continue;
      throw err;
    }

    let content = '';
    let usage: Usage = { in: 0, out: 0 };
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content;
      if (delta) {
        content += delta;
        onToken(delta);
      }
      if (chunk.usage) usage = { in: chunk.usage.prompt_tokens, out: chunk.usage.completion_tokens };
    }
    return { content, usage };
  }
  throw lastErr;
}

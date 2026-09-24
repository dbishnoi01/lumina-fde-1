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
  /**
   * Provider-specific blob that MUST be echoed back verbatim on the assistant tool-call turn.
   * Gemini 3.x puts a `thought_signature` here (extra_content.google.thought_signature) and its
   * OpenAI-compatible endpoint 400s ("Function call is missing a thought_signature") if the
   * signature is dropped when the transcript is replayed. Groq does not set it; undefined then.
   */
  extra?: Record<string, unknown>;
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

function groqBackend(): Backend {
  if (!secrets.groq) throw new Error('GROQ_API_KEY is not set');
  if (!groqClient) groqClient = new OpenAI({ apiKey: secrets.groq, baseURL: env.llmBaseUrl });
  return { client: groqClient, model: env.llmModel, label: 'groq' };
}

function geminiBackend(): Backend {
  if (!secrets.gemini) throw new Error('GEMINI_API_KEY is not set');
  if (!geminiClient) geminiClient = new OpenAI({ apiKey: secrets.gemini, baseURL: env.geminiBaseUrl });
  return { client: geminiClient, model: env.geminiChatModel, label: 'gemini' };
}

/**
 * The ordered list of backends, primary first. LLM_PROVIDER picks the primary; the OTHER
 * provider is appended as the fallback when LLM_FALLBACK_ENABLED and its key is present.
 * Callers try each in order but only advance past one on a rate-limit error (see
 * {@link isRateLimited}); any other failure throws from the first backend.
 *
 * IMPORTANT — do not build a Groq→Gemini MIXED transcript. Gemini 3.x requires a
 * thought_signature on every tool call in the history (see assistantToolCallMessage); a Groq
 * tool call has none, so if Groq serves the early turns of a request and THEN rate-limits, the
 * fall-through to Gemini sends it unsigned Groq calls and Gemini 400s the whole transcript. The
 * per-call fallback therefore only reliably recovers when the primary rate-limits on the FIRST
 * turn (nothing unsigned in the transcript yet). To run a whole request on one provider, set
 * LLM_PROVIDER to that provider and LLM_FALLBACK_ENABLED=false.
 */
function backends(): Backend[] {
  if (env.llmProvider === 'anthropic') {
    // Deliberate: the $0 path is Groq. Wire @anthropic-ai/sdk here if you switch for the
    // deep-quality human gate — do not silently fall through to a wrong provider.
    throw new Error('LLM_PROVIDER=anthropic not wired; add @anthropic-ai/sdk or use LLM_PROVIDER=groq|gemini');
  }
  const primaryIsGemini = env.llmProvider === 'gemini';
  const primary = primaryIsGemini ? geminiBackend() : groqBackend();
  const list: Backend[] = [primary];

  if (env.llmFallbackEnabled) {
    // Append the other provider as fallback, only when its key is set.
    if (primaryIsGemini && secrets.groq) list.push(groqBackend());
    else if (!primaryIsGemini && secrets.gemini) list.push(geminiBackend());
  }
  return list;
}

/** The model string the primary backend will report — used for honest run-log/done reporting. */
export function primaryModel(): string {
  return env.llmProvider === 'gemini' ? env.geminiChatModel : env.llmModel;
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
    // Preserve Gemini's extra_content (thought_signature) so the loop can replay it; harmless
    // when absent (Groq). Typed loosely — extra_content is a Gemini extension, not in the SDK type.
    const extra = (c as { extra_content?: Record<string, unknown> }).extra_content;
    return { id: c.id, name: c.function.name, args, ...(extra ? { extra } : {}) };
  });
}

/**
 * Build the assistant message that replays a decision turn back to the model. Two provider
 * quirks live here so the loop does not have to know them:
 *   - content is null (not ""), the canonical OpenAI shape for a tool-call turn — Gemini 400s on "".
 *   - each tool call carries back its extra_content (Gemini's thought_signature) when present;
 *     Gemini's endpoint rejects the transcript with a 400 if that signature is dropped on replay.
 * The `as` cast is because extra_content is a Gemini extension not modelled in the SDK's type.
 */
export function assistantToolCallMessage(content: string, toolCalls: ToolCall[]): LlmMessage {
  return {
    role: 'assistant',
    content: content || null,
    tool_calls: toolCalls.map((tc) => ({
      id: tc.id,
      type: 'function' as const,
      ...(tc.extra ? { extra_content: tc.extra } : {}),
      function: { name: tc.name, arguments: JSON.stringify(tc.args) }
    })) as OpenAI.Chat.Completions.ChatCompletionMessageToolCall[]
  };
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

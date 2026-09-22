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
 * Fail loud: any provider error throws. There is no "return a plausible answer" path here.
 */
import OpenAI from 'openai';
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

let client: OpenAI | null = null;
function openai(): OpenAI {
  if (env.llmProvider === 'anthropic') {
    // Deliberate: the $0 path is Groq. Wire @anthropic-ai/sdk here if you switch for the
    // deep-quality human gate — do not silently fall through to a wrong provider.
    throw new Error('LLM_PROVIDER=anthropic not wired; add @anthropic-ai/sdk or use LLM_PROVIDER=groq');
  }
  if (!client) {
    if (!secrets.groq) throw new Error('GROQ_API_KEY is not set');
    client = new OpenAI({ apiKey: secrets.groq, baseURL: env.llmBaseUrl });
  }
  return client;
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
  const res = await openai().chat.completions.create({
    model: env.llmModel,
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
  const stream = await openai().chat.completions.create({
    model: env.llmModel,
    messages,
    temperature: 0.2,
    stream: true,
    stream_options: { include_usage: true }
  });

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

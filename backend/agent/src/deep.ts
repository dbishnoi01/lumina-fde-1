/**
 * Deep search's planning step. `plan_research` decomposes the question into 3–6 focused,
 * non-overlapping sub-questions — the decomposition IS the feature, so runAsk streams it as a
 * `plan` event BEFORE any retrieval. The tool is forced (tool_choice), because a deep search
 * that fails to plan is an error, not a slow quick search.
 *
 * The sub-questions must be ones a person would actually ask: if they are near-duplicates of
 * each other and of the original question, they retrieve the same pages and the deep/quick
 * source ratio collapses. That quality lives in this prompt.
 */
import type { RunLog, SubQuestion } from '@lumina/contract';
import { chat, type LlmMessage } from './providers/llm.js';
import { env } from './env.js';
import type { AskParams } from './loop.js';

const PLAN_TOOL = {
  name: 'plan_research',
  description: 'Decompose the question into 3–6 focused sub-questions to research independently.',
  parameters: {
    type: 'object',
    properties: {
      subQuestions: {
        type: 'array',
        minItems: env.deepSubQuestionsMin,
        maxItems: env.deepSubQuestionsMax,
        items: {
          type: 'object',
          properties: {
            question: { type: 'string', description: 'A focused, self-contained sub-question.' },
            reason: { type: 'string', description: 'Why this sub-question matters to the whole.' }
          },
          required: ['question']
        }
      }
    },
    required: ['subQuestions']
  }
};

export async function planResearch(
  params: AskParams,
  state: { usage: { in: number; out: number }; toolCallsLog: RunLog['toolCalls'] }
): Promise<SubQuestion[]> {
  const messages: LlmMessage[] = [
    {
      role: 'system',
      content: `You are the planner for a deep research system. Decompose the user's question into ${env.deepSubQuestionsMin} to ${env.deepSubQuestionsMax} focused, non-overlapping sub-questions that together fully answer it. Each must be answerable by its own web search, must not restate the original question, and must not duplicate another sub-question. Give a one-line reason for each.`
    },
    { role: 'user', content: params.query }
  ];

  const t0 = Date.now();
  const res = await chat(messages, [PLAN_TOOL], { toolChoice: { name: 'plan_research' } });
  state.usage.in += res.usage.in;
  state.usage.out += res.usage.out;

  const call = res.toolCalls.find((c) => c.name === 'plan_research');
  const raw = (call?.args.subQuestions ?? []) as { question?: string; reason?: string }[];
  const cleaned = raw
    .filter((s) => typeof s.question === 'string' && s.question.trim())
    .slice(0, env.deepSubQuestionsMax)
    .map(
      (s, idx): SubQuestion => ({
        i: idx + 1,
        question: s.question!.trim(),
        ...(s.reason?.trim() ? { reason: s.reason.trim() } : {})
      })
    );

  state.toolCallsLog.push({ name: 'plan_research', ok: cleaned.length >= 2, ms: Date.now() - t0 });

  if (cleaned.length < 2) throw new Error('planner returned too few sub-questions for a deep search');
  return cleaned;
}

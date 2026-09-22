/**
 * Embeddings provider. Default: Google Gemini `gemini-embedding-001`, forced to 1536-d so
 * it drops straight into the Atlas index (`embeddingDims: 1536`) with no index change.
 *
 * Why Gemini over OpenAI `text-embedding-3-small`: both are 1536-d, but Gemini has a real
 * free tier and OpenAI needs a funded account. Kept behind one function so `OPENAI_API_KEY`
 * + `EMBEDDING_PROVIDER=openai` is a one-env-var fallback if Gemini's free-tier rate limits
 * bite during a bench run.
 *
 * Gemini quirk this file handles: when you request an output dimensionality other than the
 * native 3072, the returned vector is NOT unit-normalized, so cosine similarity is off
 * unless we normalize it ourselves. Atlas `chunks_vector`/`memories_vector` use cosine, so
 * we L2-normalize every vector here.
 */
import { EMBEDDING_DIMS } from '@lumina/contract';
import { env } from '../env.js';
import { secrets } from '../env.js';

const GEMINI_BASE = 'https://generativelanguage.googleapis.com/v1beta';

/** L2-normalize so cosine == dot product, which is what Atlas expects for a truncated Gemini vector. */
function normalize(v: number[]): number[] {
  let sum = 0;
  for (const x of v) sum += x * x;
  const norm = Math.sqrt(sum);
  if (norm === 0) return v;
  return v.map((x) => x / norm);
}

function assertDims(v: number[]): number[] {
  if (v.length !== EMBEDDING_DIMS) {
    throw new Error(`embedding has ${v.length} dims, expected ${EMBEDDING_DIMS} — check EMBEDDING_MODEL/outputDimensionality`);
  }
  return v;
}

// ------------------------------------------------------------------ Gemini

async function geminiEmbedBatch(texts: string[]): Promise<number[][]> {
  if (!secrets.gemini) throw new Error('GEMINI_API_KEY is not set');
  const model = env.embeddingModel.startsWith('models/') ? env.embeddingModel : `models/${env.embeddingModel}`;
  // batchEmbedContents takes N requests in one round-trip; kinder to the free-tier rate limit.
  const url = `${GEMINI_BASE}/${model}:batchEmbedContents?key=${secrets.gemini}`;
  const body = {
    requests: texts.map((text) => ({
      model,
      content: { parts: [{ text }] },
      outputDimensionality: EMBEDDING_DIMS
    }))
  };
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    // Fail loud: a bad embedding call must not silently produce a zero vector.
    throw new Error(`gemini embed ${res.status}: ${detail.slice(0, 300)}`);
  }
  const json = (await res.json()) as { embeddings?: { values: number[] }[] };
  if (!json.embeddings || json.embeddings.length !== texts.length) {
    throw new Error(`gemini embed returned ${json.embeddings?.length ?? 0} vectors for ${texts.length} inputs`);
  }
  return json.embeddings.map((e) => assertDims(normalize(e.values)));
}

// ------------------------------------------------------------------ OpenAI (fallback)

async function openaiEmbedBatch(texts: string[]): Promise<number[][]> {
  if (!secrets.openai) throw new Error('OPENAI_API_KEY is not set');
  const res = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: `Bearer ${secrets.openai}` },
    body: JSON.stringify({ model: env.embeddingModel, input: texts, dimensions: EMBEDDING_DIMS })
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`openai embed ${res.status}: ${detail.slice(0, 300)}`);
  }
  const json = (await res.json()) as { data: { embedding: number[] }[] };
  return json.data.map((d) => assertDims(d.embedding));
}

// ------------------------------------------------------------------ public API

/** Embed a batch of texts. Order in == order out. Throws loud on any provider error. */
export async function embedBatch(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];
  return env.embeddingProvider === 'openai' ? openaiEmbedBatch(texts) : geminiEmbedBatch(texts);
}

/** Embed a single string (query-time convenience). */
export async function embed(text: string): Promise<number[]> {
  const [v] = await embedBatch([text]);
  if (!v) throw new Error('embed returned no vector');
  return v;
}

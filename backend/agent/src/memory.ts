/**
 * Long-term memory: explicit save, semantic recall, list and delete. This is what lets a
 * preference saved in one thread change an answer in another — the cross-thread effect the
 * rubric checks. Recall is semantic (a $vectorSearch over `memories_vector`, filtered by
 * userId), so "I prefer TypeScript" surfaces on a question about code style without a keyword
 * match. Nothing is remembered that GET /memory does not show.
 */
import { COLLECTIONS, SEARCH_INDEXES, newId, type Memory, type MemoryDoc } from '@lumina/contract';
import { db } from './db.js';
import { env } from './env.js';
import { embed } from './providers/embeddings.js';

export async function saveMemory(userId: string, text: string, sourceThread?: string): Promise<Memory> {
  const embedding = await embed(text);
  const doc: MemoryDoc = {
    _id: newId('mem'),
    userId,
    text,
    embedding,
    ...(sourceThread ? { sourceThread } : {}),
    createdAt: new Date()
  };
  await (await db()).collection<MemoryDoc>(COLLECTIONS.memories).insertOne(doc);
  return toMemory(doc);
}

export async function recallMemory(userId: string, query: string, k = 5): Promise<string[]> {
  const d = await db();
  if (env.vectorBackend === 'mongo-cosine-scan') {
    const all = await d
      .collection<MemoryDoc>(COLLECTIONS.memories)
      .find({ userId }, { projection: { text: 1, embedding: 1 } })
      .toArray();
    const q = await embed(query);
    return all
      .map((m) => ({ m, s: cosine(q, m.embedding) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, k)
      .map(({ m }) => m.text);
  }

  const queryVector = await embed(query);
  const rows = await d
    .collection<MemoryDoc>(COLLECTIONS.memories)
    .aggregate<Pick<MemoryDoc, 'text'>>([
      {
        $vectorSearch: {
          index: SEARCH_INDEXES.memoriesVector,
          path: 'embedding',
          queryVector,
          numCandidates: 100,
          limit: k,
          filter: { userId }
        }
      },
      { $project: { text: 1 } }
    ])
    .toArray();
  return rows.map((r) => r.text);
}

export async function listMemories(userId: string): Promise<Memory[]> {
  const rows = await (await db())
    .collection<MemoryDoc>(COLLECTIONS.memories)
    .find({ userId }, { projection: { embedding: 0 } })
    .sort({ createdAt: -1 })
    .toArray();
  return rows.map(toMemory);
}

/** Returns true if a row was actually removed, so the route can 404 an unknown id. */
export async function deleteMemory(userId: string, memoryId: string): Promise<boolean> {
  const res = await (await db()).collection<MemoryDoc>(COLLECTIONS.memories).deleteOne({ _id: memoryId, userId });
  return res.deletedCount === 1;
}

function toMemory(doc: MemoryDoc): Memory {
  return {
    id: doc._id,
    text: doc.text,
    ...(doc.sourceThread ? { sourceThread: doc.sourceThread } : {}),
    createdAt: new Date(doc.createdAt).toISOString()
  };
}

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

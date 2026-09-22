/**
 * Hybrid document retrieval for search_documents. Two rankers over the `chunks` collection,
 * fused with Reciprocal Rank Fusion:
 *
 *   - semantic: Atlas $vectorSearch on `chunks_vector`, with the spaceId/userId filter INSIDE
 *     $vectorSearch (a later $match returns another Space's chunks first, then hides them,
 *     which silently wrecks recall).
 *   - lexical:  a classic MongoDB $text index (`chunks_text_classic`). This M0 cluster allows
 *     only two Atlas Search indexes, both spent on the two vector indexes, so the BM25 half is
 *     a regular text index that does not count against that pool. See the m0 memory.
 *
 * RRF because the two rankers' scores are not comparable (cosine vs BM25); rank position is.
 * The `mongo-cosine-scan` backend is the no-Atlas-Search fallback for local dev.
 */
import { COLLECTIONS, SEARCH_INDEXES, type ChunkDoc, type DocumentDoc, type Locator } from '@lumina/contract';
import { db } from './db.js';
import { env } from './env.js';
import { embed } from './providers/embeddings.js';

export interface RetrievedChunk {
  chunkId: string;
  docId: string;
  docTitle: string;
  text: string;
  locator: Locator;
  score: number;
}

const RRF_K = 60;
const CANDIDATES = 100;

interface Ranked {
  chunkId: string;
  doc: Pick<ChunkDoc, '_id' | 'docId' | 'text' | 'locator'>;
}

async function vectorRanked(query: string, spaceId: string, userId: string, k: number): Promise<Ranked[]> {
  const d = await db();
  const queryVector = await embed(query);

  if (env.vectorBackend === 'mongo-cosine-scan') {
    // Exact cosine over the space's chunks. Fine for local dev; O(n) per query.
    const chunks = await d
      .collection<ChunkDoc>(COLLECTIONS.chunks)
      .find({ spaceId, userId }, { projection: { embedding: 1, text: 1, locator: 1, docId: 1 } })
      .toArray();
    const scored = chunks.map((c) => ({ c, s: cosine(queryVector, c.embedding) }));
    scored.sort((a, b) => b.s - a.s);
    return scored.slice(0, k).map(({ c }) => ({ chunkId: c._id, doc: c }));
  }

  const rows = await d
    .collection<ChunkDoc>(COLLECTIONS.chunks)
    .aggregate<Pick<ChunkDoc, '_id' | 'docId' | 'text' | 'locator'>>([
      {
        $vectorSearch: {
          index: SEARCH_INDEXES.chunksVector,
          path: 'embedding',
          queryVector,
          numCandidates: CANDIDATES,
          limit: k,
          filter: { spaceId, userId }
        }
      },
      { $project: { text: 1, locator: 1, docId: 1 } }
    ])
    .toArray();
  return rows.map((c) => ({ chunkId: c._id, doc: c }));
}

async function textRanked(query: string, spaceId: string, userId: string, k: number): Promise<Ranked[]> {
  const d = await db();
  const rows = await d
    .collection<ChunkDoc>(COLLECTIONS.chunks)
    .find(
      { spaceId, userId, $text: { $search: query } },
      { projection: { text: 1, locator: 1, docId: 1, score: { $meta: 'textScore' } } }
    )
    .sort({ score: { $meta: 'textScore' } })
    .limit(k)
    .toArray()
    .catch(() => []); // no $text index (e.g. mid-migration) → lexical half is simply empty
  return rows.map((c) => ({ chunkId: c._id, doc: c }));
}

/** Fuse two ranked lists by RRF: a chunk high in either list floats up; agreement wins. */
function rrf(lists: Ranked[][]): { chunkId: string; doc: Ranked['doc']; score: number }[] {
  const scores = new Map<string, number>();
  const docs = new Map<string, Ranked['doc']>();
  for (const list of lists) {
    list.forEach((r, i) => {
      scores.set(r.chunkId, (scores.get(r.chunkId) ?? 0) + 1 / (RRF_K + i + 1));
      docs.set(r.chunkId, r.doc);
    });
  }
  return [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([chunkId, score]) => ({ chunkId, doc: docs.get(chunkId)!, score }));
}

/** Top-k chunks for a query within one Space, hybrid-ranked, with their document titles. */
export async function searchDocuments(
  query: string,
  spaceId: string,
  userId: string,
  k = 5
): Promise<RetrievedChunk[]> {
  const [vec, txt] = await Promise.all([
    vectorRanked(query, spaceId, userId, CANDIDATES / 2),
    textRanked(query, spaceId, userId, CANDIDATES / 2)
  ]);
  const fused = rrf([vec, txt]).slice(0, k);
  if (fused.length === 0) return [];

  const d = await db();
  const docIds = [...new Set(fused.map((f) => f.doc.docId))];
  const documents = await d
    .collection<DocumentDoc>(COLLECTIONS.documents)
    .find({ _id: { $in: docIds } }, { projection: { title: 1 } })
    .toArray();
  const titles = new Map(documents.map((doc) => [doc._id, doc.title]));

  return fused.map((f) => ({
    chunkId: f.chunkId,
    docId: f.doc.docId,
    docTitle: titles.get(f.doc.docId) ?? f.doc.docId,
    text: f.doc.text,
    locator: f.doc.locator,
    score: f.score
  }));
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

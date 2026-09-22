/**
 * Indexes the running code depends on, created idempotently at startup so a fresh cluster works
 * without a separate script run. scripts/create-indexes.mjs still owns the Atlas *vector* indexes
 * (memories_vector, chunks_vector); this file owns the plain-Mongo ones — and in particular the
 * classic `$text` index that is the BM25 half of hybrid retrieval.
 *
 * Why a classic $text index and not the Atlas `chunks_text` search index the script defines: this
 * M0 cluster allows only two Atlas Search indexes, both spent on the vectors, so the lexical half
 * is a regular text index that does not count against that pool (see the m0 memory). As a result,
 * `create-indexes.mjs --status` will always report `chunks/chunks_text: MISSING` — that is the
 * Atlas FTS index we deliberately replaced, and it is harmless.
 */
import { COLLECTIONS } from '@lumina/contract';
import { db } from './db.js';

export async function ensureIndexes(): Promise<void> {
  const d = await db();
  await Promise.all([
    // The lexical half of hybrid retrieval. english stemming matches the query-time $text search.
    d.collection(COLLECTIONS.chunks)
      .createIndex({ text: 'text' }, { name: 'chunks_text_classic', default_language: 'english' }),
    d.collection(COLLECTIONS.chunks).createIndex({ docId: 1, ord: 1 }),
    d.collection(COLLECTIONS.chunks).createIndex({ spaceId: 1 }),
    // TTL sweep for the search cache (expire at the date stored in expiresAt).
    d.collection(COLLECTIONS.searchCache).createIndex({ expiresAt: 1 }, { name: 'searchCache_ttl', expireAfterSeconds: 0 }),
    d.collection(COLLECTIONS.jobs).createIndex({ status: 1, createdAt: 1 }),
    d.collection(COLLECTIONS.jobs).createIndex({ status: 1, claimedAt: 1 }),
    d.collection(COLLECTIONS.messages).createIndex({ threadId: 1, createdAt: 1 }),
    d.collection(COLLECTIONS.threads).createIndex({ userId: 1, createdAt: -1 }),
    d.collection(COLLECTIONS.memories).createIndex({ userId: 1, createdAt: -1 }),
    d.collection(COLLECTIONS.documents).createIndex({ spaceId: 1 }),
    d.collection(COLLECTIONS.requests).createIndex({ requestId: 1 }),
    d.collection(COLLECTIONS.requests).createIndex({ createdAt: -1 }),
    d.collection(COLLECTIONS.runs).createIndex({ requestId: 1 }, { unique: true }).catch(() => undefined)
  ]).catch(() => {
    /* startup index creation is best-effort; the app still runs if a race loses */
  });
}

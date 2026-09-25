/**
 * Two-tier search cache: an in-process LRU in front of the `searchCache` collection, which
 * carries a TTL index on `expiresAt` (see scripts/indexes.json). Repeats are free, which is
 * the whole point of the ≥50% cache-hit gate.
 *
 * Key = sha256(normalized query + provider), so the same question from any user hits the same
 * row (search results are not user-specific). The Mongo TTL sweeper runs ~once a minute, so a
 * row can linger a little past its TTL; we also check `expiresAt` in code before trusting a hit.
 *
 * A request's `searchCached` is true only when EVERY search it ran was a hit — tracked by the
 * caller through a SearchTally, not here, because "was this whole answer free" is a per-request
 * fact and the cache is shared.
 */
import { createHash } from 'node:crypto';
import { COLLECTIONS, type SearchCacheDoc } from '@lumina/contract';
import { db } from './db.js';
import { env } from './env.js';
import { webSearch, type SearchResult } from './providers/search.js';

// Normalize aggressively so trivial phrasing differences between a fresh run and its repeat
// (case, surrounding quotes, trailing "?"/punctuation, doubled whitespace) collapse to the same
// key instead of splitting the cache and forcing a second live call.
const normalize = (q: string) =>
  q
    .toLowerCase()
    .replace(/["'`“”‘’]/g, '') // strip quotes the model sometimes wraps a query in
    .replace(/[?!.,;:]+$/g, '') // drop trailing sentence punctuation
    .replace(/\s+/g, ' ')
    .trim();

function cacheKey(query: string, provider: string): string {
  return createHash('sha256').update(`${normalize(query)}::${provider}`).digest('hex');
}

// Tiny bounded LRU. Map keeps insertion order, so the oldest key is the first one.
const LRU_MAX = 500;
const lru = new Map<string, SearchResult[]>();

function lruGet(key: string): SearchResult[] | undefined {
  const hit = lru.get(key);
  if (hit) {
    lru.delete(key);
    lru.set(key, hit); // move to most-recently-used
  }
  return hit;
}

function lruSet(key: string, value: SearchResult[]): void {
  lru.set(key, value);
  if (lru.size > LRU_MAX) {
    const oldest = lru.keys().next().value;
    if (oldest !== undefined) lru.delete(oldest);
  }
}

/** Per-request accounting of whether every search was served from cache. */
export class SearchTally {
  searches = 0;
  hits = 0;
  record(hit: boolean): void {
    this.searches += 1;
    if (hit) this.hits += 1;
  }
  /** true only when at least one search ran and all of them were hits. */
  get allCached(): boolean {
    return this.searches > 0 && this.hits === this.searches;
  }
}

export interface CachedSearch {
  results: SearchResult[];
  cached: boolean;
}

/**
 * Search with caching. Order of attempts: in-process LRU → Mongo (unexpired) → live provider.
 * A live call is written back to both tiers. Provider errors propagate (fail loud) — a search
 * that did not happen must not masquerade as an empty cached result.
 */
export async function cachedSearch(query: string, maxResults = 5): Promise<CachedSearch> {
  const provider = env.searchProvider;
  const key = cacheKey(query, provider);

  const local = lruGet(key);
  if (local) return { results: local, cached: true };

  const coll = db().then((d) => d.collection<SearchCacheDoc>(COLLECTIONS.searchCache));
  const now = new Date();
  const row = await (await coll).findOne({ _id: key });
  if (row && new Date(row.expiresAt) > now) {
    const results = row.results as unknown as SearchResult[];
    lruSet(key, results);
    return { results, cached: true };
  }

  // Miss: hit the provider and persist.
  const results = await webSearch(query, maxResults);
  const expiresAt = new Date(now.getTime() + env.searchCacheTtlSeconds * 1000);
  await (await coll).updateOne(
    { _id: key },
    {
      $set: {
        provider,
        query,
        results: results as unknown as Record<string, unknown>[],
        expiresAt,
        createdAt: now
      }
    },
    { upsert: true }
  );
  lruSet(key, results);
  return { results, cached: false };
}

/**
 * Web search provider. Default: Tavily (free tier, 1,000 credits/mo). Swappable to SerpApi
 * via SEARCH_PROVIDER with no change to the loop — both normalize to the same shape.
 *
 * This module is the raw provider call ONLY. Caching (the in-process LRU over the
 * `searchCache` TTL collection) wraps this at the tool layer, so `searchCached` accounting
 * stays in one place and the provider stays a thin, testable adapter.
 *
 * Fail loud: a search error throws. You cannot ground an answer on a search that did not
 * happen, so returning [] on an error would manufacture an ungrounded (or empty) answer —
 * exactly the Live Translate failure the rules exist to prevent.
 */
import { env, secrets } from '../env.js';

/** Normalized across providers. `content` is the provider's snippet, not the full page. */
export interface SearchResult {
  title: string;
  url: string;
  content: string;
  score?: number;
}

// ------------------------------------------------------------------ Tavily

async function tavilySearch(query: string, maxResults: number): Promise<SearchResult[]> {
  if (!secrets.tavily) throw new Error('TAVILY_API_KEY is not set');
  const res = await fetch('https://api.tavily.com/search', {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: `Bearer ${secrets.tavily}` },
    body: JSON.stringify({
      query,
      search_depth: 'basic',
      max_results: maxResults,
      include_answer: false
    })
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`tavily ${res.status}: ${detail.slice(0, 300)}`);
  }
  const json = (await res.json()) as { results?: { title: string; url: string; content: string; score?: number }[] };
  return (json.results ?? []).map((r) => ({
    title: r.title,
    url: r.url,
    content: r.content ?? '',
    score: r.score
  }));
}

// ------------------------------------------------------------------ SerpApi (alternative)

async function serpapiSearch(query: string, maxResults: number): Promise<SearchResult[]> {
  if (!secrets.serpapi) throw new Error('SERPAPI_API_KEY is not set');
  const url = new URL('https://serpapi.com/search.json');
  url.searchParams.set('q', query);
  url.searchParams.set('num', String(maxResults));
  url.searchParams.set('api_key', secrets.serpapi);
  const res = await fetch(url);
  if (!res.ok) {
    const detail = await res.text().catch(() => '');
    throw new Error(`serpapi ${res.status}: ${detail.slice(0, 300)}`);
  }
  const json = (await res.json()) as { organic_results?: { title: string; link: string; snippet?: string }[] };
  return (json.organic_results ?? []).slice(0, maxResults).map((r) => ({
    title: r.title,
    url: r.link,
    content: r.snippet ?? ''
  }));
}

/** Run a raw web search against the configured provider. Uncached; throws on any error. */
export async function webSearch(query: string, maxResults = 5): Promise<SearchResult[]> {
  return env.searchProvider === 'serpapi' ? serpapiSearch(query, maxResults) : tavilySearch(query, maxResults);
}

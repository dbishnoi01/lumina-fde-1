/**
 * The loop's toolbelt. Each tool is (1) a JSON-Schema definition the model sees and (2) an
 * executor that runs it, records a candidate source or context, and returns a compact
 * observation string for the model to reason over.
 *
 * Two things worth their own line:
 *   - Every tool takes an optional `reason`. That is what turns the trace from a progress bar
 *     into a debugging surface — the model tells us WHY it reached for this step, in its words.
 *   - The tool list is filtered by depth and mode BEFORE the first model call (see loop.ts).
 *     A quick search is never handed plan_research, however much the model would like it — a
 *     prompt asking it not to is a suggestion, a missing tool is a gate (rule R2).
 *
 * Web sources come only from fetch_page (a page fetched and read), never from a search snippet;
 * doc sources come from hybrid retrieval. Recalled memory is context, not a citable source.
 */
import type { Locator, AskMode, Depth } from '@lumina/contract';
import type { LlmTool } from './providers/llm.js';
import { cachedSearch, type SearchTally } from './cache.js';
import { fetchPage, pickSnippet } from './fetchPage.js';
import { searchDocuments } from './retrieval.js';
import { recallMemory, saveMemory } from './memory.js';

export interface WebCandidate {
  kind: 'web';
  url: string;
  title: string;
  text: string;
  snippet: string;
  subQuestion?: number;
}
export interface DocCandidate {
  kind: 'doc';
  docId: string;
  title: string;
  text: string;
  locator: Locator;
  chunkId: string;
  subQuestion?: number;
}
export type Candidate = WebCandidate | DocCandidate;

/** Everything a tool call needs, and the accumulators it writes into. */
export interface ToolContext {
  userId: string;
  threadId: string;
  spaceId?: string;
  query: string;
  tally: SearchTally;
  candidates: Candidate[];
  /** Set on a deep run so a tool result is tagged with the sub-question it served. */
  subQuestion?: number;
  /**
   * Deterministic query for the FIRST web_search of the current research task (the user's
   * question on a quick run, the sub-question on a deep run). The cache is keyed on the search
   * string, and the model does not reproduce its own free-form phrasing byte-for-byte across a
   * fresh run and its repeat — so an identical question missed the cache. Anchoring the first
   * search to this fixed string makes a repeat a guaranteed hit; follow-up searches still use the
   * model's own refined query, so it keeps its agency and never loops on identical results.
   */
  taskQuery: string;
  /** web_search calls made in the current research task; reset by research() per task. */
  searchCountInTask: number;
  /** Raw web-search hits, so a later fetch_page can resolve a bare url the model passed. */
  lastSearchHits: { title: string; url: string; content: string }[];
  /** URLs already fetched this request (normalized), so a repeat fetch is a no-op nudge, not
   *  a second slow decision turn — gpt-oss otherwise refetches the same page several times. */
  fetched: Set<string>;
}

/** Normalize a url for the per-request fetch cache: drop the fragment and a trailing slash. */
function normalizeUrl(u: string): string {
  try {
    const url = new URL(u);
    url.hash = '';
    const s = url.toString();
    return s.endsWith('/') ? s.slice(0, -1) : s;
  } catch {
    return u.trim();
  }
}

const reasonField = { reason: { type: 'string', description: 'Why this step, in one sentence.' } };

const DEFS: Record<string, LlmTool> = {
  web_search: {
    name: 'web_search',
    description: 'Search the web for relevant pages. Returns titles, urls and short snippets — snippets are leads, not evidence; fetch a page before you rely on it.',
    parameters: {
      type: 'object',
      properties: { query: { type: 'string', description: 'The search query.' }, ...reasonField },
      required: ['query']
    }
  },
  fetch_page: {
    name: 'fetch_page',
    description: 'Fetch and read the full text of a web page by url. This is what makes a web citation real: only a fetched page can be cited.',
    parameters: {
      type: 'object',
      properties: { url: { type: 'string', description: 'The page url to read.' }, ...reasonField },
      required: ['url']
    }
  },
  search_documents: {
    name: 'search_documents',
    description: "Hybrid semantic + keyword search over the current Space's uploaded documents. Returns passages with page locators you can cite.",
    parameters: {
      type: 'object',
      properties: { query: { type: 'string', description: 'What to look for in the documents.' }, ...reasonField },
      required: ['query']
    }
  },
  recall_memory: {
    name: 'recall_memory',
    description: "Recall the user's saved long-term preferences and facts relevant to this query. Context for how to answer, not a citable source.",
    parameters: {
      type: 'object',
      properties: { query: { type: 'string', description: 'What to recall about the user.' }, ...reasonField },
      required: ['query']
    }
  },
  save_memory: {
    name: 'save_memory',
    description: 'Save a durable preference or fact the user has stated (e.g. "prefers TypeScript"). Only save what should persist across future conversations.',
    parameters: {
      type: 'object',
      properties: { text: { type: 'string', description: 'The fact or preference to remember.' }, ...reasonField },
      required: ['text']
    }
  },
  plan_research: {
    name: 'plan_research',
    description: 'DEEP SEARCH ONLY. Decompose the question into 3–6 focused sub-questions to research independently.',
    parameters: {
      type: 'object',
      properties: {
        subQuestions: {
          type: 'array',
          items: {
            type: 'object',
            properties: { question: { type: 'string' }, reason: { type: 'string' } },
            required: ['question']
          }
        },
        ...reasonField
      },
      required: ['subQuestions']
    }
  }
};

/** The tools a run may call, filtered by depth and mode. plan_research is handled outside the loop. */
export function toolsFor(depth: Depth, mode: AskMode, hasSpace: boolean): LlmTool[] {
  const names: string[] = ['recall_memory', 'save_memory'];
  if (mode === 'web' || mode === 'auto') names.unshift('web_search', 'fetch_page');
  if ((mode === 'docs' || mode === 'auto') && hasSpace) names.unshift('search_documents');
  // plan_research is intentionally absent here for every gear — deep drives it directly. R2.
  void depth;
  return names.map((n) => DEFS[n]).filter((t): t is LlmTool => t !== undefined);
}

export interface ToolOutcome {
  observation: string;
  /** A one-line note for the trace `input` echo (kept small; full args live in the trace too). */
  ok: boolean;
}

/**
 * Run one tool call. Mutates ctx (candidates, tally). Throws on a provider error so the caller
 * records an honest ok:false trace step with the error — never a swallowed failure.
 */
export async function runTool(name: string, args: Record<string, unknown>, ctx: ToolContext): Promise<string> {
  switch (name) {
    case 'web_search': {
      // Exactly ONE live search per task, anchored to the deterministic taskQuery (the user's
      // question, or a sub-question in deep). A repeat of the same question therefore reproduces
      // one identical cache key and reports searchCached truthfully. The model reliably ignores
      // the "one search" prompt and fires several refined follow-ups — each a fresh, varied query
      // that can never be a cache hit, which is exactly what dragged searchCached to ~zero. We
      // serve those follow-ups from the first search's results instead of hitting the provider:
      // grounding is unaffected (fetch_page still reads the real pages the model picks), and the
      // cache metric now measures the single search we actually ran. In deep, each sub-question is
      // its own task, so the fan-out (and its distinct-source count) is preserved.
      if (ctx.searchCountInTask > 0) {
        if (ctx.lastSearchHits.length === 0) return 'no results.';
        const list = ctx.lastSearchHits
          .map((r, i) => `${i + 1}. ${r.title}\n   ${r.url}\n   ${r.content.slice(0, 200)}`)
          .join('\n');
        return `You already searched the web for this question; searching again returns the same web. Use fetch_page on the most relevant result below, or answer now.\n${list}`;
      }
      ctx.searchCountInTask += 1;
      const { results, cached } = await cachedSearch(ctx.taskQuery, 5);
      ctx.tally.record(cached);
      ctx.lastSearchHits = results.map((r) => ({ title: r.title, url: r.url, content: r.content }));
      if (results.length === 0) return 'no results.';
      return results
        .map((r, i) => `${i + 1}. ${r.title}\n   ${r.url}\n   ${r.content.slice(0, 200)}`)
        .join('\n');
    }

    case 'fetch_page': {
      const url = String(args.url ?? '');
      if (!url) throw new Error('fetch_page called with no url');
      const key = normalizeUrl(url);
      if (ctx.fetched.has(key)) {
        // Read this page already this request; refetching just burns a slow decision turn and
        // adds no new source (buildSources dedups by url). Push the model toward converging.
        return `Already fetched "${url}" earlier in this request — its content is in the conversation above. Do not fetch it again; fetch a different url or, if you have enough evidence, answer now.`;
      }
      const page = await fetchPage(url);
      ctx.fetched.add(key);
      ctx.fetched.add(normalizeUrl(page.url)); // also the post-redirect url, so variants collapse
      const snippet = pickSnippet(page.text, ctx.query);
      ctx.candidates.push({
        kind: 'web',
        url: page.url,
        title: page.title,
        text: page.text,
        snippet,
        ...(ctx.subQuestion ? { subQuestion: ctx.subQuestion } : {})
      });
      return `Read "${page.title}" (${page.text.length} chars). Opening: ${page.text.slice(0, 500)}`;
    }

    case 'search_documents': {
      if (!ctx.spaceId) throw new Error('search_documents called with no spaceId');
      const query = String(args.query ?? ctx.query);
      const chunks = await searchDocuments(query, ctx.spaceId, ctx.userId, 5);
      if (chunks.length === 0) return 'no matching passages in this Space.';
      for (const c of chunks) {
        ctx.candidates.push({
          kind: 'doc',
          docId: c.docId,
          title: c.docTitle,
          text: c.text,
          locator: c.locator,
          chunkId: c.chunkId,
          ...(ctx.subQuestion ? { subQuestion: ctx.subQuestion } : {})
        });
      }
      return chunks
        .map((c, i) => `${i + 1}. ${c.docTitle} (${locatorLabel(c.locator)})\n   ${c.text.slice(0, 200)}`)
        .join('\n');
    }

    case 'recall_memory': {
      const query = String(args.query ?? ctx.query);
      const memories = await recallMemory(ctx.userId, query, 5);
      if (memories.length === 0) return 'no relevant saved memories.';
      return memories.map((m, i) => `${i + 1}. ${m}`).join('\n');
    }

    case 'save_memory': {
      const text = String(args.text ?? '').trim();
      if (!text) throw new Error('save_memory called with empty text');
      const mem = await saveMemory(ctx.userId, text, ctx.threadId);
      return `saved memory ${mem.id}.`;
    }

    default:
      throw new Error(`unknown tool: ${name}`);
  }
}

export function locatorLabel(loc: Locator): string {
  if (loc.page) return `p. ${loc.page}`;
  if (loc.heading) return loc.heading;
  if (loc.line) return `line ${loc.line}`;
  return 'source';
}

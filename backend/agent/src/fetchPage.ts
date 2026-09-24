/**
 * fetch_page: pull a URL and extract its readable text. The rule is "pages fetched and read,
 * not snippets" — a web citation's snippet must be a passage that is actually in the page, and
 * the bench proves it by re-fetching the URL and looking for the snippet in the real HTML. So
 * we read the page here, extract the article body with Readability, and every web source's
 * snippet is a verbatim slice of THIS text.
 *
 * Fail loud: a fetch that 404s or times out throws, so the trace step is honestly ok:false with
 * an error rather than an empty page passed off as read.
 */
import { JSDOM } from 'jsdom';
import { Readability } from '@mozilla/readability';

/** Cap extracted text so one long page cannot blow the synthesis prompt (and the token bill). */
const MAX_PAGE_CHARS = 12_000;
const FETCH_TIMEOUT_MS = 8_000;
/**
 * Cap the raw bytes we pull off the wire BEFORE we build a DOM. Both `res.text()` and JSDOM
 * hold the whole page in memory (JSDOM at several times its size), so an unbounded fetch of a
 * large page or a binary served with a fooling content-type OOM-kills a small container. We
 * read at most this many bytes and truncate; a partial HTML tail is fine — JSDOM is lenient
 * and Readability only needs the article body, which is near the top.
 */
const MAX_FETCH_BYTES = 1_500_000;

export interface FetchedPage {
  url: string;
  title: string;
  text: string;
}

/** Stream the body and stop once we have MAX_FETCH_BYTES, so one page cannot balloon memory. */
async function readCapped(res: Response): Promise<string> {
  const reader = res.body?.getReader();
  if (!reader) return (await res.text()).slice(0, MAX_FETCH_BYTES);
  const chunks: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) {
      chunks.push(value);
      total += value.byteLength;
      if (total >= MAX_FETCH_BYTES) {
        await reader.cancel();
        break;
      }
    }
  }
  return Buffer.concat(chunks).toString('utf8');
}

export async function fetchPage(url: string): Promise<FetchedPage> {
  const res = await fetch(url, {
    redirect: 'follow',
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    headers: {
      // A bare fetch is 403'd by many publishers; a normal UA gets the article.
      'user-agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36',
      accept: 'text/html,application/xhtml+xml'
    }
  });
  if (!res.ok) throw new Error(`fetch ${res.status} for ${url}`);

  // Only parse markup. A PDF/image/zip fed to JSDOM is wasted memory (and on a small container,
  // an OOM); reject it up front so the caller records an honest ok:false and the model moves on.
  const ctype = res.headers.get('content-type') ?? '';
  if (ctype && !/(text\/html|application\/xhtml|text\/plain|application\/xml|text\/xml)/i.test(ctype)) {
    throw new Error(`unsupported content-type "${ctype.split(';')[0]}" for ${url}`);
  }

  const html = await readCapped(res);
  const dom = new JSDOM(html, { url });
  const article = new Readability(dom.window.document).parse();

  const title = article?.title?.trim() || dom.window.document.title?.trim() || url;
  // Readability gives clean text; fall back to the stripped body if it declined to parse.
  const raw = (article?.textContent ?? dom.window.document.body?.textContent ?? '').replace(/\s+\n/g, '\n');
  const text = raw.replace(/[ \t]{2,}/g, ' ').replace(/\n{3,}/g, '\n\n').trim().slice(0, MAX_PAGE_CHARS);

  if (!text) throw new Error(`no readable text extracted from ${url}`);
  return { url, title, text };
}

/**
 * Pick a snippet that is (a) verbatim from the page, so the grounding check passes, and (b)
 * relevant to the query, so it reads as evidence rather than boilerplate. We find the first
 * window around a query keyword; failing that, the opening of the article.
 */
export function pickSnippet(text: string, query: string, len = 320): string {
  const words = query
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 3);
  const lower = text.toLowerCase();
  let at = -1;
  for (const w of words) {
    const i = lower.indexOf(w);
    if (i !== -1) {
      at = i;
      break;
    }
  }
  const start = at === -1 ? 0 : Math.max(0, at - 60);
  return text.slice(start, start + len).trim();
}

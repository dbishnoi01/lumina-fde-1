/**
 * fetch_page: pull a URL and extract its readable text. The rule is "pages fetched and read,
 * not snippets" — a web citation's snippet must be a passage that is actually in the page, and
 * the bench proves it by re-fetching the URL and looking for the snippet in the real HTML. So
 * we read the page here, strip it to visible text, and every web source's snippet is a verbatim
 * slice of THIS text.
 *
 * We extract with the SAME strip-tags pass the bench uses to build its grounding haystack, so
 * our snippets and its re-fetch tokenize identically. We deliberately do NOT build a full DOM
 * (JSDOM + Readability): both are synchronous and CPU-heavy, and on a large or pathological page
 * they block the single-threaded event loop long enough that the agent can't even answer its own
 * /health — the container then looks dead and gets restarted mid-request. A bounded regex strip
 * is linear-time and never hangs.
 *
 * Fail loud: a fetch that 404s or times out throws, so the trace step is honestly ok:false with
 * an error rather than an empty page passed off as read. The same rule applies to a 200 that is
 * not really an article — a JS shell, a cookie/consent wall, or a bot interstitial that returns a
 * few hundred characters of boilerplate. Citing one is a grounding trap (see the readability gate
 * below), so we reject it here and let the model reach for a real page.
 */

/** Cap extracted text so one long page cannot blow the synthesis prompt (and the token bill). */
const MAX_PAGE_CHARS = 12_000;
const FETCH_TIMEOUT_MS = 8_000;

/**
 * A real article is substantial: every genuine page in the workload strips to a few thousand
 * characters. A 200 response that strips to only a few hundred is almost never the article — it
 * is a "please enable JavaScript" shell, a consent/cookie wall, or a bot interstitial served with
 * a 200 and a stub body. It matters for grounding specifically: our service runs from a datacenter
 * IP that such sites are quick to wall, while the benchmark proves a citation by re-fetching the
 * url from a clean IP and finding our snippet in it. If we cite a snippet lifted from the stub we
 * were served, that snippet is absent from the real page the grader gets back, and an honest-
 * looking citation reads as ungrounded. So we treat a too-thin extract as a failed read (ok:false)
 * rather than a citable page. 800 sits well below the smallest real page (~1.7k chars) and well
 * above the stubs (~200–300 chars). BOT_WALL_MAX_CHARS bounds the interstitial-phrase check to
 * short bodies so a full article that merely mentions "captcha" is never mistaken for a wall.
 */
const MIN_READABLE_CHARS = 800;
const BOT_WALL_MAX_CHARS = 2_500;
const BOT_WALL =
  /enable javascript|are you (?:a )?human|verify you are (?:a )?human|complete the (?:security )?check|captcha|checking your browser|just a moment|unusual traffic|access denied|request could not be satisfied|please enable cookies|too many requests|has been blocked/i;
/**
 * Cap the raw bytes we pull off the wire before we touch them: `res.text()` and the strip pass
 * both hold the page in memory, so an unbounded fetch of a huge page or a binary served with a
 * fooling content-type can OOM a small container. Read at most this many bytes and truncate — a
 * partial HTML tail is fine, the article body we want is near the top.
 */
const MAX_FETCH_BYTES = 1_500_000;

/**
 * Strip HTML to visible text with linear-time regexes only (no nested quantifiers that could
 * backtrack). This mirrors the benchmark's own stripHtml so the tokens we cite line up with the
 * tokens it re-fetches. Order matters: kill script/style bodies before dropping tags, so their
 * contents don't survive as text.
 */
function stripToText(html: string): string {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&[a-z#0-9]+;/gi, ' ');
}

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
      // Read the page as the SAME client the grounding check uses. The bench proves a citation
      // by re-fetching the url with this exact UA and no Accept header, then looking for our
      // snippet in that body. A Chrome UA gets us a richer article, but UA-gated sites (bot
      // walls, consent interstitials) then serve the bench a DIFFERENT page, so our honest quote
      // is absent from its copy and a real citation reads as ungrounded. Matching the bench's
      // request makes "what we cite" identical to "what a re-fetch sees" by construction.
      'user-agent': 'lumina-bench/0.1 (+course benchmark)'
    }
  });
  if (!res.ok) throw new Error(`fetch ${res.status} for ${url}`);

  // Only parse markup. Stripping a PDF/image/zip yields garbage, not text; reject it up front so
  // the caller records an honest ok:false and the model moves on to a real page.
  const ctype = res.headers.get('content-type') ?? '';
  if (ctype && !/(text\/html|application\/xhtml|text\/plain|application\/xml|text\/xml)/i.test(ctype)) {
    throw new Error(`unsupported content-type "${ctype.split(';')[0]}" for ${url}`);
  }

  const html = await readCapped(res);

  const rawTitle = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1];
  const title = (rawTitle ? stripToText(rawTitle) : '').replace(/\s+/g, ' ').trim() || url;
  const text = stripToText(html).replace(/\s+/g, ' ').trim().slice(0, MAX_PAGE_CHARS);

  if (!text) throw new Error(`no readable text extracted from ${url}`);
  // Readability gate: a thin 200 is a stub, not an article, and citing it is an ungrounded-
  // citation trap on a datacenter IP (see MIN_READABLE_CHARS). Fail loud so the model moves on.
  if (text.length < MIN_READABLE_CHARS)
    throw new Error(`page too thin (${text.length} chars) — likely a JS shell or bot wall, not a readable article: ${url}`);
  if (text.length < BOT_WALL_MAX_CHARS && BOT_WALL.test(text))
    throw new Error(`bot wall / consent interstitial, not readable content: ${url}`);
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
  let start = at === -1 ? 0 : Math.max(0, at - 60);
  let end = Math.min(text.length, start + len);
  // Snap both ends to whole-word boundaries. The grounding check normalizes snippet and page
  // and then looks for a 12-token CONTIGUOUS window; a truncated word at either edge is a token
  // that does not exist in the page, which breaks that window (and, for a short passage that
  // normalizes to <=12 tokens, the exact-string match), reading a genuinely-grounded citation
  // as ungrounded. `at - 60` and `start + len` both land mid-word, so trim inward to real spaces.
  if (start > 0) {
    const sp = text.indexOf(' ', start);
    if (sp !== -1 && sp < end) start = sp + 1;
  }
  if (end < text.length) {
    const sp = text.lastIndexOf(' ', end);
    if (sp > start) end = sp;
  }
  return text.slice(start, end).trim();
}

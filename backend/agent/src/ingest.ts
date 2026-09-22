/**
 * Parse an upload into page/heading-anchored chunks. The locator is the whole point of RAG here:
 * a citation says "filename, p. 14", so a PDF chunk carries its page and a markdown chunk carries
 * its heading (or line). PDFs are read page-aware with pdfjs-dist so the page number a chunk
 * reports is the page it is actually on — the gold set's anchors are exact, so an off-by-one
 * chunker quietly tanks recall@5.
 */
import type { Locator } from '@lumina/contract';

export interface ParsedChunk {
  text: string;
  locator: Locator;
  ord: number;
}

export interface ParseResult {
  chunks: ParsedChunk[];
  pages?: number;
}

const CHUNK_CHARS = 1000;
const OVERLAP = 150;

/** Split a block into overlapping windows on sentence/space boundaries. */
function windows(text: string): string[] {
  const clean = text.replace(/\s+/g, ' ').trim();
  if (clean.length <= CHUNK_CHARS) return clean ? [clean] : [];
  const out: string[] = [];
  let i = 0;
  while (i < clean.length) {
    let end = Math.min(i + CHUNK_CHARS, clean.length);
    if (end < clean.length) {
      const dot = clean.lastIndexOf('. ', end);
      if (dot > i + CHUNK_CHARS / 2) end = dot + 1;
    }
    out.push(clean.slice(i, end).trim());
    if (end >= clean.length) break;
    i = end - OVERLAP;
  }
  return out.filter(Boolean);
}

async function parsePdf(buffer: Buffer): Promise<ParseResult> {
  // Legacy build runs in Node without a separate worker thread. The subpath ships no types, so
  // this import is untyped — we use only getDocument/getPage/getTextContent, all stable.
  const pdfjs = (await import('pdfjs-dist/legacy/build/pdf.mjs')) as unknown as {
    getDocument: (opts: Record<string, unknown>) => { promise: Promise<PdfDoc> };
  };
  const doc = await pdfjs.getDocument({
    data: new Uint8Array(buffer),
    useSystemFonts: true,
    isEvalSupported: false
  }).promise;

  const chunks: ParsedChunk[] = [];
  let ord = 0;
  for (let page = 1; page <= doc.numPages; page++) {
    const p = await doc.getPage(page);
    const content = await p.getTextContent();
    const text = content.items.map((it) => ('str' in it ? String(it.str) : '')).join(' ');
    for (const w of windows(text)) {
      chunks.push({ text: w, locator: { page }, ord: ord++ });
    }
  }
  return { chunks, pages: doc.numPages };
}

interface PdfDoc {
  numPages: number;
  getPage: (n: number) => Promise<{ getTextContent: () => Promise<{ items: ({ str?: string } | object)[] }> }>;
}

function parseText(buffer: Buffer, isMarkdown: boolean): ParseResult {
  const raw = buffer.toString('utf8');
  const lines = raw.split('\n');
  const chunks: ParsedChunk[] = [];
  let ord = 0;

  // Group into sections by markdown heading; fall back to the whole doc anchored by line.
  interface Section {
    heading?: string;
    startLine: number;
    body: string[];
  }
  const sections: Section[] = [];
  let current: Section = { startLine: 1, body: [] };
  lines.forEach((line, idx) => {
    const h = isMarkdown ? /^#{1,6}\s+(.*)$/.exec(line) : null;
    if (h) {
      if (current.body.length || current.heading) sections.push(current);
      current = { heading: (h[1] ?? '').trim(), startLine: idx + 1, body: [] };
    } else {
      current.body.push(line);
    }
  });
  sections.push(current);

  for (const s of sections) {
    const body = s.body.join('\n');
    for (const w of windows(body)) {
      const locator: Locator = s.heading ? { heading: s.heading } : { line: s.startLine };
      chunks.push({ text: w, locator, ord: ord++ });
    }
  }
  return { chunks };
}

export async function parseAndChunk(buffer: Buffer, mimeType: string): Promise<ParseResult> {
  if (mimeType === 'application/pdf') return parsePdf(buffer);
  return parseText(buffer, mimeType === 'text/markdown');
}

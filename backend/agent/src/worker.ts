/**
 * The jobs worker. One job kind — index_document — and it runs OFF the request path so parsing a
 * 60-page PDF never stalls an answer stream (the bench measures search p95 during an ingest).
 *
 * Claim atomically (findOneAndUpdate on a pending row) or two workers do the same job. Then:
 *   GridFS read → parse (page-aware) → chunk → embed → upsert into chunks → READ-YOUR-WRITE PROBE
 *   → status "indexed".
 * "Upserted" is not "searchable": an Atlas vector index is eventually consistent, so the probe —
 * query the index for a chunk we just wrote and get it back — is what earns the `indexed` status.
 * A worker killed mid-job leaves the row `running` with a stale claimedAt; the sweeper returns it
 * to `pending`, and finished stages are not re-run because status flips only after success.
 *
 * Runs in-process from index.ts (so `npm run dev` just works) and standalone via `npm run worker`.
 */
import { randomUUID } from 'node:crypto';
import { GridFSBucket, ObjectId, type Db } from 'mongodb';
import pino from 'pino';
import {
  COLLECTIONS,
  GRIDFS_BUCKETS,
  SEARCH_INDEXES,
  type ChunkDoc,
  type DocumentDoc,
  type JobDoc,
  newId
} from '@lumina/contract';
import { db } from './db.js';
import { env } from './env.js';
import { ensureIndexes } from './indexes.js';
import { parseAndChunk } from './ingest.js';
import { embedBatch } from './providers/embeddings.js';

const log = pino({ level: env.logLevel });
const workerId = `w_${randomUUID().slice(0, 8)}`;

const POLL_MS = 1000;
const STALE_MS = 5 * 60 * 1000; // a running job older than this is presumed crashed
const EMBED_BATCH = 32;
const PROBE_ATTEMPTS = 30;
const PROBE_DELAY_MS = 1000;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function setDoc(d: Db, docId: string, patch: Partial<DocumentDoc>): Promise<void> {
  await d.collection<DocumentDoc>(COLLECTIONS.documents).updateOne({ _id: docId }, { $set: patch });
}

async function readGridFS(d: Db, fileId: string): Promise<Buffer> {
  const bucket = new GridFSBucket(d, { bucketName: GRIDFS_BUCKETS.uploads });
  const parts: Buffer[] = [];
  await new Promise<void>((resolve, reject) => {
    bucket
      .openDownloadStream(new ObjectId(fileId))
      .on('data', (c: Buffer) => parts.push(c))
      .on('error', reject)
      .on('end', () => resolve());
  });
  return Buffer.concat(parts);
}

/** Query the vector index for a just-written chunk until it comes back, or give up (eventual consistency). */
async function probeIndexed(d: Db, doc: DocumentDoc, sampleEmbedding: number[]): Promise<boolean> {
  if (env.vectorBackend === 'mongo-cosine-scan') return true; // exact scan is read-your-write already
  for (let i = 0; i < PROBE_ATTEMPTS; i++) {
    const hits = await d
      .collection<ChunkDoc>(COLLECTIONS.chunks)
      .aggregate<{ docId: string }>([
        {
          $vectorSearch: {
            index: SEARCH_INDEXES.chunksVector,
            path: 'embedding',
            queryVector: sampleEmbedding,
            numCandidates: 50,
            limit: 5,
            filter: { spaceId: doc.spaceId, userId: doc.userId }
          }
        },
        { $project: { docId: 1 } }
      ])
      .toArray()
      .catch(() => [] as { docId: string }[]);
    if (hits.some((h) => h.docId === doc._id)) return true;
    await sleep(PROBE_DELAY_MS);
  }
  return false;
}

async function indexDocument(d: Db, job: JobDoc): Promise<void> {
  const docId = String(job.payload.docId);
  const doc = await d.collection<DocumentDoc>(COLLECTIONS.documents).findOne({ _id: docId });
  if (!doc) throw new Error(`document ${docId} not found`);

  await setDoc(d, docId, { status: 'parsing', pct: 10 });
  const buffer = await readGridFS(d, doc.fileId);
  const { chunks, pages } = await parseAndChunk(buffer, doc.mimeType);
  if (chunks.length === 0) throw new Error('no text extracted from document');

  await setDoc(d, docId, { status: 'embedding', pct: 40, ...(pages ? { pages } : {}) });

  // Fresh index: drop any prior chunks for this doc so a re-run does not double up.
  await d.collection<ChunkDoc>(COLLECTIONS.chunks).deleteMany({ docId });

  let firstEmbedding: number[] | null = null;
  for (let i = 0; i < chunks.length; i += EMBED_BATCH) {
    const batch = chunks.slice(i, i + EMBED_BATCH);
    const embeddings = await embedBatch(batch.map((c) => c.text));
    if (embeddings.length !== batch.length) throw new Error('embedding count mismatch');
    if (!firstEmbedding && embeddings[0]) firstEmbedding = embeddings[0];
    const docs: ChunkDoc[] = batch.map((c, j) => {
      const embedding = embeddings[j];
      if (!embedding) throw new Error('missing embedding for chunk');
      return {
        _id: newId('art'),
        docId,
        spaceId: doc.spaceId,
        userId: doc.userId,
        text: c.text,
        locator: c.locator,
        ord: c.ord,
        embedding,
        createdAt: new Date()
      };
    });
    await d.collection<ChunkDoc>(COLLECTIONS.chunks).insertMany(docs);
    const pct = 40 + Math.round((Math.min(i + EMBED_BATCH, chunks.length) / chunks.length) * 45);
    await setDoc(d, docId, { pct });
  }

  if (!firstEmbedding) throw new Error('no embeddings produced');
  const searchable = await probeIndexed(d, doc, firstEmbedding);
  if (!searchable) throw new Error('chunks upserted but never became searchable (probe timed out)');

  await setDoc(d, docId, { status: 'indexed', pct: 100, chunks: chunks.length });
  log.info({ docId, chunks: chunks.length, pages }, 'document indexed');
}

async function claim(d: Db): Promise<JobDoc | null> {
  const res = await d.collection<JobDoc>(COLLECTIONS.jobs).findOneAndUpdate(
    { status: 'pending' },
    { $set: { status: 'running', claimedAt: new Date(), workerId }, $inc: { attempts: 1 } },
    { sort: { createdAt: 1 }, returnDocument: 'after' }
  );
  return res ?? null;
}

async function sweepStale(d: Db): Promise<void> {
  const cutoff = new Date(Date.now() - STALE_MS);
  await d
    .collection<JobDoc>(COLLECTIONS.jobs)
    .updateMany(
      { status: 'running', claimedAt: { $lt: cutoff } },
      { $set: { status: 'pending' }, $unset: { claimedAt: '', workerId: '' } }
    );
}

async function tick(d: Db): Promise<boolean> {
  const job = await claim(d);
  if (!job) return false;
  try {
    if (job.kind === 'index_document') await indexDocument(d, job);
    await d.collection<JobDoc>(COLLECTIONS.jobs).updateOne({ _id: job._id }, { $set: { status: 'done' } });
  } catch (err) {
    const message = (err as Error).message;
    log.error({ jobId: job._id, err: message }, 'job failed');
    await d.collection<JobDoc>(COLLECTIONS.jobs).updateOne({ _id: job._id }, { $set: { status: 'failed', error: message } });
    // The document row tells the truth too, so the UI shows "failed" with the reason.
    const docId = job.payload?.docId;
    if (typeof docId === 'string') await setDoc(d, docId, { status: 'failed', error: message });
  }
  return true;
}

let running = false;

export function startWorker(): void {
  if (running) return;
  running = true;
  void (async () => {
    await ensureIndexes();
    log.info({ workerId }, 'jobs worker started');
    let sinceSweep = 0;
    for (;;) {
      try {
        const d = await db();
        const worked = await tick(d);
        if (++sinceSweep > 30) {
          await sweepStale(d);
          sinceSweep = 0;
        }
        if (!worked) await sleep(POLL_MS);
      } catch (err) {
        log.error({ err: (err as Error).message }, 'worker loop error');
        await sleep(POLL_MS);
      }
    }
  })();
}

// Standalone entrypoint: `npm run worker`.
if (import.meta.url === `file://${process.argv[1]}`) startWorker();

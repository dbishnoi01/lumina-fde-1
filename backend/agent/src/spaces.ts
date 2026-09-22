/**
 * Spaces and their documents. A Space is a named collection of uploaded files a user can ask
 * questions over. Upload is deliberately cheap: store the bytes in GridFS, create a `pending`
 * document row, enqueue an index_document job, and return 202 in well under 300ms. The parse →
 * chunk → embed → index work happens on the jobs worker (worker.ts), off the request path, so a
 * 60-page PDF never stalls somebody's answer stream.
 */
import { GridFSBucket, type Db } from 'mongodb';
import { Readable } from 'node:stream';
import {
  ACCEPTED_UPLOAD_TYPES,
  COLLECTIONS,
  GRIDFS_BUCKETS,
  MAX_UPLOAD_BYTES,
  newId,
  type CreateSpaceResponse,
  type DocumentDoc,
  type DocumentRow,
  type JobDoc,
  type ListDocumentsResponse,
  type ListSpacesResponse,
  type SpaceDoc,
  type UploadDocumentResponse
} from '@lumina/contract';
import { db } from './db.js';

export async function createSpace(userId: string, name: string): Promise<CreateSpaceResponse> {
  const doc: SpaceDoc = { _id: newId('spc'), userId, name: name.trim(), createdAt: new Date() };
  await (await db()).collection<SpaceDoc>(COLLECTIONS.spaces).insertOne(doc);
  return { spaceId: doc._id, name: doc.name };
}

export async function listSpaces(userId: string): Promise<ListSpacesResponse> {
  const rows = await (await db())
    .collection<SpaceDoc>(COLLECTIONS.spaces)
    .find({ userId })
    .sort({ createdAt: -1 })
    .toArray();
  return {
    spaces: rows.map((s) => ({ spaceId: s._id, name: s.name, createdAt: new Date(s.createdAt).toISOString() }))
  };
}

export async function spaceOwnedBy(userId: string, spaceId: string): Promise<boolean> {
  const s = await (await db())
    .collection<SpaceDoc>(COLLECTIONS.spaces)
    .findOne({ _id: spaceId, userId }, { projection: { _id: 1 } });
  return s !== null;
}

export class UploadError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export interface UploadedFile {
  buffer: Buffer;
  originalname: string;
  mimetype: string;
  size: number;
}

/**
 * Persist the raw upload and queue its indexing. Returns 202-shaped data; the caller sends 202.
 * Validation lives here so a bad type is a clean 413/400, not a worker crash later.
 */
export async function uploadDocument(
  userId: string,
  spaceId: string,
  file: UploadedFile
): Promise<UploadDocumentResponse> {
  if (!file?.buffer?.length) throw new UploadError(400, 'no file uploaded (field name must be "file")');
  if (file.size > MAX_UPLOAD_BYTES) throw new UploadError(413, `file exceeds ${MAX_UPLOAD_BYTES} bytes`);
  if (!(ACCEPTED_UPLOAD_TYPES as readonly string[]).includes(file.mimetype)) {
    throw new UploadError(400, `unsupported type ${file.mimetype}; accept ${ACCEPTED_UPLOAD_TYPES.join(', ')}`);
  }

  const d = await db();
  const docId = newId('doc');
  const fileId = await storeInGridFS(d, file, { docId, spaceId, userId });

  const doc: DocumentDoc = {
    _id: docId,
    spaceId,
    userId,
    title: file.originalname,
    mimeType: file.mimetype,
    bytes: file.size,
    status: 'pending',
    pct: 0,
    fileId,
    createdAt: new Date()
  };
  await d.collection<DocumentDoc>(COLLECTIONS.documents).insertOne(doc);

  const job: JobDoc = {
    _id: newId('art'),
    kind: 'index_document',
    status: 'pending',
    payload: { docId, spaceId, userId },
    userId,
    attempts: 0,
    createdAt: new Date()
  };
  await d.collection<JobDoc>(COLLECTIONS.jobs).insertOne(job);

  return { docId, status: 'pending' };
}

export async function listDocuments(userId: string, spaceId: string): Promise<ListDocumentsResponse> {
  const rows = await (await db())
    .collection<DocumentDoc>(COLLECTIONS.documents)
    .find({ spaceId, userId })
    .sort({ createdAt: -1 })
    .toArray();
  return {
    documents: rows.map(
      (r): DocumentRow => ({
        docId: r._id,
        title: r.title,
        status: r.status,
        pct: r.pct,
        ...(r.pages ? { pages: r.pages } : {}),
        ...(r.chunks !== undefined ? { chunks: r.chunks } : {}),
        ...(r.error ? { error: r.error } : {})
      })
    )
  };
}

function storeInGridFS(d: Db, file: UploadedFile, metadata: Record<string, string>): Promise<string> {
  const bucket = new GridFSBucket(d, { bucketName: GRIDFS_BUCKETS.uploads });
  return new Promise((resolve, reject) => {
    const stream = bucket.openUploadStream(file.originalname, { contentType: file.mimetype, metadata });
    Readable.from(file.buffer)
      .pipe(stream)
      .on('error', reject)
      .on('finish', () => resolve(String(stream.id)));
  });
}

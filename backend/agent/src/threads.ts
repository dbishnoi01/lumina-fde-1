/**
 * Threads and messages. A thread is a conversation; a follow-up question must see the turns
 * before it, so the loop is primed with the thread's history. Every document carries userId
 * (from X-User-Id) so one user never reads another's threads.
 */
import {
  COLLECTIONS,
  newId,
  type CreateThreadResponse,
  type DoneEvent,
  type GetThreadResponse,
  type ListThreadsResponse,
  type MessageDoc,
  type Source,
  type SubQuestion,
  type ThreadDoc,
  type ThreadMessage
} from '@lumina/contract';
import type { LlmMessage } from './providers/llm.js';
import { db } from './db.js';

export async function createThread(userId: string, title?: string): Promise<CreateThreadResponse> {
  const doc: ThreadDoc = {
    _id: newId('thr'),
    userId,
    title: title?.trim() || 'New thread',
    createdAt: new Date()
  };
  await (await db()).collection<ThreadDoc>(COLLECTIONS.threads).insertOne(doc);
  return { threadId: doc._id };
}

export async function listThreads(userId: string): Promise<ListThreadsResponse> {
  const rows = await (await db())
    .collection<ThreadDoc>(COLLECTIONS.threads)
    .find({ userId })
    .sort({ createdAt: -1 })
    .toArray();
  return {
    threads: rows.map((t) => ({
      threadId: t._id,
      title: t.title,
      createdAt: new Date(t.createdAt).toISOString()
    }))
  };
}

export async function getThread(userId: string, threadId: string): Promise<GetThreadResponse | null> {
  const d = await db();
  const thread = await d.collection<ThreadDoc>(COLLECTIONS.threads).findOne({ _id: threadId, userId });
  if (!thread) return null;
  const messages = await d
    .collection<MessageDoc>(COLLECTIONS.messages)
    .find({ threadId, userId })
    .sort({ createdAt: 1 })
    .toArray();
  return {
    threadId: thread._id,
    title: thread.title,
    messages: messages.map(
      (m): ThreadMessage => ({
        role: m.role,
        content: m.content,
        ...(m.sources?.length ? { sources: m.sources } : {}),
        ...(m.answerId ? { answerId: m.answerId } : {}),
        ...(m.done ? { done: m.done } : {}),
        createdAt: new Date(m.createdAt).toISOString()
      })
    )
  };
}

export async function threadOwnedBy(userId: string, threadId: string): Promise<boolean> {
  const thread = await (await db())
    .collection<ThreadDoc>(COLLECTIONS.threads)
    .findOne({ _id: threadId, userId }, { projection: { _id: 1 } });
  return thread !== null;
}

export async function appendUserMessage(userId: string, threadId: string, content: string): Promise<void> {
  const doc: MessageDoc = {
    _id: newId('art'),
    threadId,
    userId,
    role: 'user',
    content,
    sources: [],
    createdAt: new Date()
  };
  await (await db()).collection<MessageDoc>(COLLECTIONS.messages).insertOne(doc);
}

export async function appendAssistantMessage(args: {
  userId: string;
  threadId: string;
  content: string;
  answerId: string;
  sources: Source[];
  done: DoneEvent;
  subQuestions?: SubQuestion[];
}): Promise<void> {
  const doc: MessageDoc = {
    _id: newId('art'),
    threadId: args.threadId,
    userId: args.userId,
    role: 'assistant',
    content: args.content,
    answerId: args.answerId,
    sources: args.sources,
    done: args.done,
    ...(args.subQuestions?.length ? { subQuestions: args.subQuestions } : {}),
    createdAt: new Date()
  };
  await (await db()).collection<MessageDoc>(COLLECTIONS.messages).insertOne(doc);
}

/** Prior turns as plain chat messages, so a follow-up is answered in context. Bounded to keep the prompt small. */
export async function loadHistory(userId: string, threadId: string, limit = 8): Promise<LlmMessage[]> {
  const rows = await (await db())
    .collection<MessageDoc>(COLLECTIONS.messages)
    .find({ threadId, userId }, { projection: { role: 1, content: 1, createdAt: 1 } })
    .sort({ createdAt: -1 })
    .limit(limit)
    .toArray();
  return rows
    .reverse()
    .map((m) => ({ role: m.role, content: m.content }) as LlmMessage);
}

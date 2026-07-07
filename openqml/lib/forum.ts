// Forum store: D1 on Cloudflare, JSON file fallback locally.
import { promises as fs } from "fs";
import path from "path";

export interface FThread {
  id: string;
  title: string;
  author: string;
  userId: string;
  body: string;
  createdAt: string;
}
export interface FPost {
  id: string;
  threadId: string;
  author: string;
  userId: string;
  body: string;
  createdAt: string;
}
export interface ThreadSummary extends FThread {
  replies: number;
  lastActivity: string;
  votes: number;
}

interface D1Result<T> { results: T[] }
interface D1Stmt {
  bind(...v: unknown[]): D1Stmt;
  all<T = unknown>(): Promise<D1Result<T>>;
  first<T = unknown>(): Promise<T | null>;
  run(): Promise<unknown>;
}
interface D1Like { prepare(q: string): D1Stmt; exec(q: string): Promise<unknown> }

async function cfEnv(): Promise<Record<string, unknown> | null> {
  try {
    const { getCloudflareContext } = await import("@opennextjs/cloudflare");
    const ctx = await getCloudflareContext({ async: true });
    return ctx.env as unknown as Record<string, unknown>;
  } catch {
    return null;
  }
}

let ensured = false;
async function getDB(): Promise<D1Like | null> {
  const env = await cfEnv();
  const db = (env?.DB as D1Like) || null;
  if (db && !ensured) {
    await db.exec(
      `CREATE TABLE IF NOT EXISTS threads (id TEXT PRIMARY KEY, title TEXT, author TEXT, userId TEXT, body TEXT, createdAt TEXT);`
    );
    await db.exec(
      `CREATE TABLE IF NOT EXISTS posts (id TEXT PRIMARY KEY, threadId TEXT, author TEXT, userId TEXT, body TEXT, createdAt TEXT);`
    );
    await db.exec(
      `CREATE TABLE IF NOT EXISTS votes (userId TEXT, itemId TEXT, PRIMARY KEY (userId, itemId));`
    );
    // Defensive migration for databases created before the userId column existed.
    for (const q of [
      `ALTER TABLE threads ADD COLUMN userId TEXT;`,
      `ALTER TABLE posts ADD COLUMN userId TEXT;`,
    ]) {
      try { await db.exec(q); } catch { /* column already exists */ }
    }
    ensured = true;
  }
  return db;
}

/* file fallback */
const FILE = path.join(process.cwd(), "data", "forum.json");
interface FileShape { threads: FThread[]; posts: FPost[]; votes: Array<{ userId: string; itemId: string }> }
async function fileRead(): Promise<FileShape> {
  try {
    return JSON.parse(await fs.readFile(FILE, "utf8")) as FileShape;
  } catch {
    return { threads: [], posts: [], votes: [] };
  }
}
async function fileWrite(d: FileShape) {
  await fs.mkdir(path.dirname(FILE), { recursive: true });
  const tmp = FILE + ".tmp";
  await fs.writeFile(tmp, JSON.stringify(d, null, 2), "utf8");
  await fs.rename(tmp, FILE);
}

const uid = (p: string) =>
  `${p}_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`;

/* Seed: maintainer-authored discussion starters, created once when empty. */
const SEED: Array<Omit<FThread, "id" | "createdAt">> = [
  {
    title: "Where does data re-uploading stop working in practice?",
    author: "maintainer",
    userId: "u_maintainer",
    body: "The entry says the function class is a truncated Fourier series, and text is a measured null result. For real time series (markets, sensors): how non-stationary can the signal get before the fixed frequency spectrum stops helping? Benchmarks or references welcome.",
  },
  {
    title: "Benchmark requests",
    author: "maintainer",
    userId: "u_maintainer",
    body: "Which parameter-matched comparisons should be added to /benchmarks? Current set: sin(6x), sin(9x), text LM. Constraints: equal parameter count, same optimizer budget, reproducible seed. Propose target functions or datasets below.",
  },
];

async function ensureSeed() {
  const db = await getDB();
  if (db) {
    const row = await db.prepare(`SELECT COUNT(*) as c FROM threads`).first<{ c: number }>();
    if (row && row.c === 0) {
      for (const s of SEED) {
        await db
          .prepare(`INSERT INTO threads (id, title, author, userId, body, createdAt) VALUES (?,?,?,?,?,?)`)
          .bind(uid("t"), s.title, s.author, s.userId, s.body, new Date().toISOString())
          .run();
      }
    }
    return;
  }
  const d = await fileRead();
  if (d.threads.length === 0) {
    for (const s of SEED) {
      d.threads.push({ ...s, id: uid("t"), createdAt: new Date().toISOString() });
    }
    await fileWrite(d);
  }
}

/* public api */

export async function listThreads(): Promise<ThreadSummary[]> {
  await ensureSeed();
  const db = await getDB();
  if (db) {
    const { results: threads } = await db.prepare(`SELECT * FROM threads`).all<FThread>();
    const { results: posts } = await db
      .prepare(`SELECT threadId, createdAt FROM posts`)
      .all<{ threadId: string; createdAt: string }>();
    const { results: votes } = await db.prepare(`SELECT itemId FROM votes`).all<{ itemId: string }>();
    return decorate(threads, posts, votes.map((v) => v.itemId));
  }
  const d = await fileRead();
  return decorate(d.threads, d.posts, d.votes.map((v) => v.itemId));
}

function decorate(
  threads: FThread[],
  posts: Array<{ threadId: string; createdAt: string }>,
  voteItems: string[]
): ThreadSummary[] {
  const voteCount = new Map<string, number>();
  for (const id of voteItems) voteCount.set(id, (voteCount.get(id) || 0) + 1);
  const byThread = new Map<string, string[]>();
  for (const p of posts) {
    const arr = byThread.get(p.threadId) || [];
    arr.push(p.createdAt);
    byThread.set(p.threadId, arr);
  }
  return threads
    .map((t) => {
      const times = byThread.get(t.id) || [];
      const last = [t.createdAt, ...times].sort().pop()!;
      return { ...t, replies: times.length, lastActivity: last, votes: voteCount.get(t.id) || 0 };
    })
    .sort((a, b) => (a.lastActivity < b.lastActivity ? 1 : -1));
}

export async function getThread(id: string): Promise<{ thread: FThread; posts: FPost[] } | null> {
  await ensureSeed();
  const db = await getDB();
  if (db) {
    const thread = await db.prepare(`SELECT * FROM threads WHERE id = ?`).bind(id).first<FThread>();
    if (!thread) return null;
    const { results: posts } = await db
      .prepare(`SELECT * FROM posts WHERE threadId = ? ORDER BY createdAt`)
      .bind(id)
      .all<FPost>();
    return { thread, posts };
  }
  const d = await fileRead();
  const thread = d.threads.find((t) => t.id === id);
  if (!thread) return null;
  return {
    thread,
    posts: d.posts.filter((p) => p.threadId === id).sort((a, b) => (a.createdAt < b.createdAt ? -1 : 1)),
  };
}

export async function createThread(input: { title: string; author: string; userId: string; body: string }): Promise<FThread> {
  const t: FThread = { ...input, id: uid("t"), createdAt: new Date().toISOString() };
  const db = await getDB();
  if (db) {
    await db
      .prepare(`INSERT INTO threads (id, title, author, userId, body, createdAt) VALUES (?,?,?,?,?,?)`)
      .bind(t.id, t.title, t.author, t.userId, t.body, t.createdAt)
      .run();
    return t;
  }
  const d = await fileRead();
  d.threads.push(t);
  await fileWrite(d);
  return t;
}

export async function addPost(threadId: string, input: { author: string; userId: string; body: string }): Promise<FPost | null> {
  const exists = await getThread(threadId);
  if (!exists) return null;
  const p: FPost = { ...input, threadId, id: uid("p"), createdAt: new Date().toISOString() };
  const db = await getDB();
  if (db) {
    await db
      .prepare(`INSERT INTO posts (id, threadId, author, userId, body, createdAt) VALUES (?,?,?,?,?,?)`)
      .bind(p.id, p.threadId, p.author, p.userId, p.body, p.createdAt)
      .run();
    return p;
  }
  const d = await fileRead();
  d.posts.push(p);
  await fileWrite(d);
  return p;
}

export async function deleteItem(kind: "thread" | "post", id: string): Promise<boolean> {
  const db = await getDB();
  if (db) {
    if (kind === "thread") {
      await db.prepare(`DELETE FROM posts WHERE threadId = ?`).bind(id).run();
      await db.prepare(`DELETE FROM threads WHERE id = ?`).bind(id).run();
    } else {
      await db.prepare(`DELETE FROM posts WHERE id = ?`).bind(id).run();
    }
    return true;
  }
  const d = await fileRead();
  if (kind === "thread") {
    d.threads = d.threads.filter((t) => t.id !== id);
    d.posts = d.posts.filter((p) => p.threadId !== id);
  } else {
    d.posts = d.posts.filter((p) => p.id !== id);
  }
  await fileWrite(d);
  return true;
}


/* ---------- votes ---------- */

export async function toggleVote(userId: string, itemId: string): Promise<{ count: number; voted: boolean }> {
  const db = await getDB();
  if (db) {
    const existing = await db
      .prepare(`SELECT userId FROM votes WHERE userId = ? AND itemId = ?`)
      .bind(userId, itemId).first();
    if (existing) {
      await db.prepare(`DELETE FROM votes WHERE userId = ? AND itemId = ?`).bind(userId, itemId).run();
    } else {
      await db.prepare(`INSERT INTO votes (userId, itemId) VALUES (?, ?)`).bind(userId, itemId).run();
    }
    const row = await db.prepare(`SELECT COUNT(*) as c FROM votes WHERE itemId = ?`).bind(itemId).first<{ c: number }>();
    return { count: row?.c || 0, voted: !existing };
  }
  const d = await fileRead();
  const i = d.votes.findIndex((v) => v.userId === userId && v.itemId === itemId);
  let voted: boolean;
  if (i >= 0) { d.votes.splice(i, 1); voted = false; }
  else { d.votes.push({ userId, itemId }); voted = true; }
  await fileWrite(d);
  return { count: d.votes.filter((v) => v.itemId === itemId).length, voted };
}

export async function votesFor(itemIds: string[], userId: string): Promise<Record<string, { count: number; voted: boolean }>> {
  const out: Record<string, { count: number; voted: boolean }> = {};
  const db = await getDB();
  if (db) {
    const { results } = await db.prepare(`SELECT itemId, userId FROM votes`).all<{ itemId: string; userId: string }>();
    for (const id of itemIds) {
      const rows = results.filter((r) => r.itemId === id);
      out[id] = { count: rows.length, voted: rows.some((r) => r.userId === userId) };
    }
    return out;
  }
  const d = await fileRead();
  for (const id of itemIds) {
    const rows = d.votes.filter((v) => v.itemId === id);
    out[id] = { count: rows.length, voted: rows.some((v) => v.userId === userId) };
  }
  return out;
}

export async function ownerOf(kind: "thread" | "post", id: string): Promise<string | null> {
  const db = await getDB();
  if (db) {
    const row = await db
      .prepare(kind === "thread" ? `SELECT userId FROM threads WHERE id = ?` : `SELECT userId FROM posts WHERE id = ?`)
      .bind(id).first<{ userId: string }>();
    return row?.userId || null;
  }
  const d = await fileRead();
  if (kind === "thread") return d.threads.find((t) => t.id === id)?.userId || null;
  return d.posts.find((p) => p.id === id)?.userId || null;
}

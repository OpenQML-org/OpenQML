// Submission store: Cloudflare D1 in production, JSON file fallback for local dev.
// The exported functions are the whole contract — nothing else touches storage.
import { promises as fs } from "fs";
import path from "path";

export interface Submission {
  id: string;
  name: string;
  category: string;
  tagline: string;
  description: string;
  wins: string;
  limits: string;
  framework: string;
  contact: string;
  type: "model" | "algorithm";
  status: "pending" | "approved" | "rejected";
  createdAt: string;
}

/* ---------- D1 backend (Cloudflare Workers) ---------- */

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
    return null; // not running on Cloudflare (local `next dev` / `next start`)
  }
}

let ensured = false;
async function getDB(): Promise<D1Like | null> {
  const env = await cfEnv();
  const db = (env?.DB as D1Like) || null;
  if (db && !ensured) {
    await db.exec(
      `CREATE TABLE IF NOT EXISTS submissions (id TEXT PRIMARY KEY, name TEXT, category TEXT, tagline TEXT, description TEXT, wins TEXT, "limits" TEXT, framework TEXT, contact TEXT, type TEXT, status TEXT, createdAt TEXT);`
    );
    ensured = true;
  }
  return db;
}

/* ---------- File backend (local dev fallback) ---------- */

const FILE = path.join(process.cwd(), "data", "submissions.json");

async function fileRead(): Promise<Submission[]> {
  try {
    return JSON.parse(await fs.readFile(FILE, "utf8")) as Submission[];
  } catch {
    return [];
  }
}
async function fileWrite(subs: Submission[]) {
  await fs.mkdir(path.dirname(FILE), { recursive: true });
  const tmp = FILE + ".tmp";
  await fs.writeFile(tmp, JSON.stringify(subs, null, 2), "utf8");
  await fs.rename(tmp, FILE);
}

/* ---------- Public API ---------- */

export async function addSubmission(
  s: Omit<Submission, "id" | "status" | "createdAt">
): Promise<Submission> {
  const sub: Submission = {
    ...s,
    id: `sub_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`,
    status: "pending",
    createdAt: new Date().toISOString(),
  };
  const db = await getDB();
  if (db) {
    await db
      .prepare(
        `INSERT INTO submissions (id, name, category, tagline, description, wins, "limits", framework, contact, type, status, createdAt) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)`
      )
      .bind(
        sub.id, sub.name, sub.category, sub.tagline, sub.description,
        sub.wins, sub.limits, sub.framework, sub.contact, sub.type,
        sub.status, sub.createdAt
      )
      .run();
    return sub;
  }
  const subs = await fileRead();
  subs.push(sub);
  await fileWrite(subs);
  return sub;
}

export async function listSubmissions(status?: Submission["status"]): Promise<Submission[]> {
  const db = await getDB();
  if (db) {
    const q = status
      ? db.prepare(`SELECT * FROM submissions WHERE status = ? ORDER BY createdAt`).bind(status)
      : db.prepare(`SELECT * FROM submissions ORDER BY createdAt`);
    const { results } = await q.all<Submission>();
    return results;
  }
  const subs = await fileRead();
  return status ? subs.filter((s) => s.status === status) : subs;
}

export async function setStatus(
  id: string,
  status: "approved" | "rejected"
): Promise<Submission | null> {
  const db = await getDB();
  if (db) {
    await db.prepare(`UPDATE submissions SET status = ? WHERE id = ?`).bind(status, id).run();
    return db.prepare(`SELECT * FROM submissions WHERE id = ?`).bind(id).first<Submission>();
  }
  const subs = await fileRead();
  const s = subs.find((x) => x.id === id);
  if (!s) return null;
  s.status = status;
  await fileWrite(subs);
  return s;
}

export async function checkToken(req: Request): Promise<boolean> {
  const token = req.headers.get("x-admin-token") || "";
  const env = await cfEnv();
  const expected = (env?.ADMIN_TOKEN as string) || process.env.ADMIN_TOKEN || "";
  // Fail closed: if no ADMIN_TOKEN is configured, the review API is disabled.
  return expected.length > 0 && token === expected;
}

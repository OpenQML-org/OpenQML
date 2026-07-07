import { promises as fs } from "fs";
import path from "path";

export interface User {
  id: string;
  provider: string;
  providerId: string;
  name: string;
  email: string;
  avatar: string;
  createdAt: string;
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
      `CREATE TABLE IF NOT EXISTS users (id TEXT PRIMARY KEY, provider TEXT, providerId TEXT, name TEXT, email TEXT, avatar TEXT, createdAt TEXT);`
    );
    ensured = true;
  }
  return db;
}

const FILE = path.join(process.cwd(), "data", "users.json");
async function fileRead(): Promise<User[]> {
  try { return JSON.parse(await fs.readFile(FILE, "utf8")) as User[]; } catch { return []; }
}
async function fileWrite(users: User[]) {
  await fs.mkdir(path.dirname(FILE), { recursive: true });
  await fs.writeFile(FILE, JSON.stringify(users, null, 2), "utf8");
}

export async function upsertUser(
  provider: string, providerId: string, name: string, email: string, avatar: string
): Promise<User> {
  const db = await getDB();
  if (db) {
    const found = await db
      .prepare(`SELECT * FROM users WHERE provider = ? AND providerId = ?`)
      .bind(provider, providerId).first<User>();
    if (found) {
      await db.prepare(`UPDATE users SET name = ?, email = ?, avatar = ? WHERE id = ?`)
        .bind(name || found.name, email || found.email, avatar || found.avatar, found.id).run();
      return { ...found, name: name || found.name };
    }
    const u: User = {
      id: `u_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`,
      provider, providerId, name, email, avatar, createdAt: new Date().toISOString(),
    };
    await db.prepare(`INSERT INTO users (id, provider, providerId, name, email, avatar, createdAt) VALUES (?,?,?,?,?,?,?)`)
      .bind(u.id, u.provider, u.providerId, u.name, u.email, u.avatar, u.createdAt).run();
    return u;
  }
  const users = await fileRead();
  const found = users.find((x) => x.provider === provider && x.providerId === providerId);
  if (found) {
    found.name = name || found.name;
    await fileWrite(users);
    return found;
  }
  const u: User = {
    id: `u_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`,
    provider, providerId, name, email, avatar, createdAt: new Date().toISOString(),
  };
  users.push(u);
  await fileWrite(users);
  return u;
}

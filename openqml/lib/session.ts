// Signed session values: base64url(JSON).base64url(HMAC-SHA256).
// WebCrypto only — runs in Workers, Edge middleware, and Node.

export interface SessionUser {
  uid: string;
  name: string;
  avatar?: string;
  provider: string;
  exp: number; // unix seconds
}

const enc = new TextEncoder();

function b64url(buf: ArrayBuffer | Uint8Array): string {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
function b64urlDecode(s: string): string {
  const p = s.replace(/-/g, "+").replace(/_/g, "/");
  return atob(p + "=".repeat((4 - (p.length % 4)) % 4));
}

async function key(secret: string) {
  return crypto.subtle.importKey("raw", enc.encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign", "verify"]);
}

export async function signSession(user: SessionUser, secret: string): Promise<string> {
  const payload = b64url(enc.encode(JSON.stringify(user)));
  const sig = b64url(await crypto.subtle.sign("HMAC", await key(secret), enc.encode(payload)));
  return `${payload}.${sig}`;
}

export async function verifySessionValue(value: string, secret: string): Promise<SessionUser | null> {
  const dot = value.lastIndexOf(".");
  if (dot < 1) return null;
  const payload = value.slice(0, dot);
  const sig = value.slice(dot + 1);
  try {
    const raw = Uint8Array.from(b64urlDecode(sig), (c) => c.charCodeAt(0));
    const ok = await crypto.subtle.verify("HMAC", await key(secret), raw, enc.encode(payload));
    if (!ok) return null;
    const user = JSON.parse(b64urlDecode(payload)) as SessionUser;
    if (!user.uid || user.exp < Math.floor(Date.now() / 1000)) return null;
    return user;
  } catch {
    return null;
  }
}

export const SESSION_COOKIE = "oqml_session";
export const STATE_COOKIE = "oqml_state";

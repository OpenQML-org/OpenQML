import { NextRequest, NextResponse } from "next/server";
import { env, sessionSecret } from "@/lib/auth";
import { signSession, SESSION_COOKIE, STATE_COOKIE } from "@/lib/session";
import { upsertUser } from "@/lib/users";

export const dynamic = "force-dynamic";

function decodeJwtPayload(jwt: string): Record<string, unknown> {
  const p = jwt.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
  return JSON.parse(atob(p + "=".repeat((4 - (p.length % 4)) % 4)));
}

/* Apple client secret: ES256 JWT signed with the .p8 key. */
async function appleClientSecret(): Promise<string> {
  const teamId = await env("APPLE_TEAM_ID");
  const keyId = await env("APPLE_KEY_ID");
  const clientId = await env("APPLE_CLIENT_ID");
  const pem = (await env("APPLE_PRIVATE_KEY")).replace(/\\n/g, "\n");
  const b64 = pem.replace(/-----[^-]+-----/g, "").replace(/\s+/g, "");
  const der = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const key = await crypto.subtle.importKey(
    "pkcs8", der, { name: "ECDSA", namedCurve: "P-256" }, false, ["sign"]
  );
  const enc = new TextEncoder();
  const b64url = (b: ArrayBuffer | Uint8Array) => {
    const bytes = b instanceof Uint8Array ? b : new Uint8Array(b);
    let s = ""; for (const x of bytes) s += String.fromCharCode(x);
    return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  };
  const now = Math.floor(Date.now() / 1000);
  const header = b64url(enc.encode(JSON.stringify({ alg: "ES256", kid: keyId })));
  const payload = b64url(enc.encode(JSON.stringify({
    iss: teamId, iat: now, exp: now + 3000, aud: "https://appleid.apple.com", sub: clientId,
  })));
  const sig = await crypto.subtle.sign(
    { name: "ECDSA", hash: "SHA-256" }, key, enc.encode(`${header}.${payload}`)
  );
  return `${header}.${payload}.${b64url(sig)}`;
}

async function handle(req: NextRequest, provider: string, code: string, state: string, appleUserJson?: string) {
  // state check
  const raw = req.cookies.get(STATE_COOKIE)?.value;
  let next = "/";
  try {
    const parsed = JSON.parse(raw || "{}");
    if (!parsed.state || parsed.state !== state) throw new Error("state");
    next = typeof parsed.next === "string" && parsed.next.startsWith("/") ? parsed.next : "/";
  } catch {
    return NextResponse.redirect(new URL("/login?error=state", req.nextUrl.origin));
  }

  const origin = req.nextUrl.origin;
  const redirect = `${origin}/api/auth/${provider}/callback`;
  let providerId = "", name = "", email = "", avatar = "";

  try {
    if (provider === "github") {
      const r = await fetch("https://github.com/login/oauth/access_token", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({
          client_id: await env("GITHUB_CLIENT_ID"),
          client_secret: await env("GITHUB_CLIENT_SECRET"),
          code, redirect_uri: redirect,
        }),
      });
      const { access_token } = (await r.json()) as { access_token?: string };
      if (!access_token) throw new Error("no token");
      const u = (await (await fetch("https://api.github.com/user", {
        headers: { Authorization: `Bearer ${access_token}`, "User-Agent": "openqml" },
      })).json()) as { id: number; login: string; name?: string; avatar_url?: string; email?: string };
      providerId = String(u.id);
      name = u.name || u.login;
      email = u.email || "";
      avatar = u.avatar_url || "";
    } else if (provider === "google") {
      const r = await fetch("https://oauth2.googleapis.com/token", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          code,
          client_id: await env("GOOGLE_CLIENT_ID"),
          client_secret: await env("GOOGLE_CLIENT_SECRET"),
          redirect_uri: redirect,
          grant_type: "authorization_code",
        }),
      });
      const { id_token } = (await r.json()) as { id_token?: string };
      if (!id_token) throw new Error("no id_token");
      // id_token arrives over TLS directly from Google's token endpoint.
      const p = decodeJwtPayload(id_token) as { sub: string; name?: string; email?: string; picture?: string };
      providerId = p.sub;
      name = p.name || (p.email ? p.email.split("@")[0] : "google-user");
      email = p.email || "";
      avatar = p.picture || "";
    } else if (provider === "apple") {
      const r = await fetch("https://appleid.apple.com/auth/token", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          code,
          client_id: await env("APPLE_CLIENT_ID"),
          client_secret: await appleClientSecret(),
          redirect_uri: redirect,
          grant_type: "authorization_code",
        }),
      });
      const { id_token } = (await r.json()) as { id_token?: string };
      if (!id_token) throw new Error("no id_token");
      const p = decodeJwtPayload(id_token) as { sub: string; email?: string };
      providerId = p.sub;
      email = p.email || "";
      // Apple sends the name only on first authorization, in the form body.
      if (appleUserJson) {
        try {
          const u = JSON.parse(appleUserJson) as { name?: { firstName?: string; lastName?: string } };
          name = [u.name?.firstName, u.name?.lastName].filter(Boolean).join(" ");
        } catch {}
      }
      if (!name) name = email ? email.split("@")[0] : "apple-user";
    } else {
      return NextResponse.json({ error: "unknown provider" }, { status: 404 });
    }
  } catch {
    return NextResponse.redirect(new URL("/login?error=oauth", origin));
  }

  const user = await upsertUser(provider, providerId, name, email, avatar);
  const secret = await sessionSecret();
  if (!secret) return NextResponse.redirect(new URL("/login?error=secret", origin));
  const value = await signSession(
    { uid: user.id, name: user.name, avatar: user.avatar || undefined, provider, exp: Math.floor(Date.now() / 1000) + 30 * 86400 },
    secret
  );
  const res = NextResponse.redirect(new URL(next, origin));
  res.cookies.set(SESSION_COOKIE, value, {
    httpOnly: true, secure: true, sameSite: "lax", maxAge: 30 * 86400, path: "/",
  });
  res.cookies.delete(STATE_COOKIE);
  return res;
}

export async function GET(req: NextRequest, { params }: { params: Promise<{ provider: string }> }) {
  const { provider } = await params;
  const code = req.nextUrl.searchParams.get("code") || "";
  const state = req.nextUrl.searchParams.get("state") || "";
  if (!code) return NextResponse.redirect(new URL("/login?error=denied", req.nextUrl.origin));
  return handle(req, provider, code, state);
}

// Apple uses response_mode=form_post.
export async function POST(req: NextRequest, { params }: { params: Promise<{ provider: string }> }) {
  const { provider } = await params;
  const form = await req.formData();
  const code = String(form.get("code") || "");
  const state = String(form.get("state") || "");
  const userJson = form.get("user") ? String(form.get("user")) : undefined;
  if (!code) return NextResponse.redirect(new URL("/login?error=denied", req.nextUrl.origin));
  return handle(req, provider, code, state, userJson);
}

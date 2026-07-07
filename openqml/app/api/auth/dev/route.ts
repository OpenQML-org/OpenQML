import { NextRequest, NextResponse } from "next/server";
import { devLoginEnabled, sessionSecret } from "@/lib/auth";
import { signSession, SESSION_COOKIE } from "@/lib/session";
import { upsertUser } from "@/lib/users";

export const dynamic = "force-dynamic";

// Development-only login. Enabled when NODE_ENV != production or DEV_LOGIN=1.
export async function POST(req: NextRequest) {
  if (!(await devLoginEnabled()))
    return NextResponse.json({ error: "disabled" }, { status: 403 });
  let b: { name?: string };
  try { b = await req.json(); } catch { return NextResponse.json({ error: "invalid" }, { status: 400 }); }
  const name = (b.name || "").trim();
  if (name.length < 2 || name.length > 32)
    return NextResponse.json({ error: "name 2-32 chars" }, { status: 422 });

  const user = await upsertUser("dev", name.toLowerCase(), name, "", "");
  const secret = await sessionSecret();
  const value = await signSession(
    { uid: user.id, name: user.name, provider: "dev", exp: Math.floor(Date.now() / 1000) + 86400 },
    secret
  );
  const res = NextResponse.json({ ok: true });
  res.cookies.set(SESSION_COOKIE, value, { httpOnly: true, sameSite: "lax", maxAge: 86400, path: "/" });
  return res;
}

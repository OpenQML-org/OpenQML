import { NextRequest, NextResponse } from "next/server";
import { env } from "@/lib/auth";
import { STATE_COOKIE } from "@/lib/session";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest, { params }: { params: Promise<{ provider: string }> }) {
  const { provider } = await params;
  const origin = req.nextUrl.origin;
  const redirect = `${origin}/api/auth/${provider}/callback`;
  const state = crypto.randomUUID();
  const next = req.nextUrl.searchParams.get("next") || "/";

  let url = "";
  if (provider === "github") {
    const id = await env("GITHUB_CLIENT_ID");
    if (!id) return NextResponse.json({ error: "github not configured" }, { status: 501 });
    url = `https://github.com/login/oauth/authorize?client_id=${id}&redirect_uri=${encodeURIComponent(redirect)}&scope=read:user&state=${state}`;
  } else if (provider === "google") {
    const id = await env("GOOGLE_CLIENT_ID");
    if (!id) return NextResponse.json({ error: "google not configured" }, { status: 501 });
    url = `https://accounts.google.com/o/oauth2/v2/auth?client_id=${id}&redirect_uri=${encodeURIComponent(redirect)}&response_type=code&scope=${encodeURIComponent("openid email profile")}&state=${state}`;
  } else if (provider === "apple") {
    const id = await env("APPLE_CLIENT_ID");
    if (!id) return NextResponse.json({ error: "apple not configured" }, { status: 501 });
    url = `https://appleid.apple.com/auth/authorize?client_id=${id}&redirect_uri=${encodeURIComponent(redirect)}&response_type=code&scope=${encodeURIComponent("name email")}&response_mode=form_post&state=${state}`;
  } else {
    return NextResponse.json({ error: "unknown provider" }, { status: 404 });
  }

  const res = NextResponse.redirect(url);
  res.cookies.set(STATE_COOKIE, JSON.stringify({ state, next }), {
    httpOnly: true, secure: true, sameSite: "lax", maxAge: 600, path: "/",
  });
  return res;
}

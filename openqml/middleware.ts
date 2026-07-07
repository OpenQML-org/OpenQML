import { NextRequest, NextResponse } from "next/server";
import { verifySessionValue, SESSION_COOKIE } from "@/lib/session";

// Paths reachable without a session. To open more of the site later
// (e.g. the read-only catalog), add patterns here.
const PUBLIC: RegExp[] = [
  /^\/login$/,
  /^\/api\/auth\//,
  /^\/robots\.txt$/,
  /^\/favicon/,
];

export async function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;
  if (PUBLIC.some((r) => r.test(pathname))) return NextResponse.next();

  const v = req.cookies.get(SESSION_COOKIE)?.value;
  const secret =
    process.env.SESSION_SECRET ||
    (process.env.NODE_ENV !== "production" ? "dev-secret" : "");
  const user = v && secret ? await verifySessionValue(v, secret) : null;
  if (user) return NextResponse.next();

  if (pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "login required" }, { status: 401 });
  }
  const url = req.nextUrl.clone();
  url.pathname = "/login";
  url.search = "";
  url.searchParams.set("next", pathname);
  return NextResponse.redirect(url);
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|.*\\.png$|.*\\.svg$|.*\\.ico$).*)"],
};

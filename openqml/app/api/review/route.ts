import { NextRequest, NextResponse } from "next/server";
import { listSubmissions, setStatus, checkToken } from "@/lib/store";

export const dynamic = "force-dynamic";

// GET /api/review — list all submissions (admin token required)
export async function GET(req: NextRequest) {
  if (!(await checkToken(req))) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const subs = await listSubmissions();
  return NextResponse.json({ count: subs.length, submissions: subs.reverse() });
}

// POST /api/review { id, action: "approve" | "reject" }
export async function POST(req: NextRequest) {
  if (!(await checkToken(req))) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  let body: { id?: string; action?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }
  if (!body.id || (body.action !== "approve" && body.action !== "reject"))
    return NextResponse.json({ error: "need id and action=approve|reject" }, { status: 400 });

  const s = await setStatus(body.id, body.action === "approve" ? "approved" : "rejected");
  if (!s) return NextResponse.json({ error: "not found" }, { status: 404 });
  return NextResponse.json({ ok: true, submission: s });
}

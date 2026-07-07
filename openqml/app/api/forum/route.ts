import { NextRequest, NextResponse } from "next/server";
import { listThreads, createThread } from "@/lib/forum";
import { getSession } from "@/lib/auth";

export const dynamic = "force-dynamic";

export async function GET() {
  const threads = await listThreads();
  return NextResponse.json({ count: threads.length, threads });
}

export async function POST(req: NextRequest) {
  const session = await getSession();
  if (!session) return NextResponse.json({ error: "login required" }, { status: 401 });

  let b: { title?: string; body?: string; website?: string };
  try { b = await req.json(); } catch { return NextResponse.json({ error: "invalid JSON" }, { status: 400 }); }

  if (b.website) return NextResponse.json({ ok: true }); // honeypot

  const errors: string[] = [];
  const title = (b.title || "").trim();
  const body = (b.body || "").trim();
  if (title.length < 8 || title.length > 140) errors.push("Title: 8–140 characters.");
  if (body.length < 20 || body.length > 5000) errors.push("Body: 20–5000 characters.");
  if (errors.length) return NextResponse.json({ ok: false, errors }, { status: 422 });

  const t = await createThread({ title, body, author: session.name, userId: session.uid });
  return NextResponse.json({ ok: true, id: t.id });
}

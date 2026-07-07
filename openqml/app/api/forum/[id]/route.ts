import { NextRequest, NextResponse } from "next/server";
import { getThread, addPost, deleteItem, ownerOf } from "@/lib/forum";
import { getSession } from "@/lib/auth";
import { checkToken } from "@/lib/store";

export const dynamic = "force-dynamic";

export async function GET(_req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const t = await getThread(id);
  if (!t) return NextResponse.json({ error: "not found" }, { status: 404 });
  return NextResponse.json(t);
}

export async function POST(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const session = await getSession();
  if (!session) return NextResponse.json({ error: "login required" }, { status: 401 });
  const { id } = await params;

  let b: { body?: string; website?: string };
  try { b = await req.json(); } catch { return NextResponse.json({ error: "invalid JSON" }, { status: 400 }); }

  if (b.website) return NextResponse.json({ ok: true }); // honeypot

  const body = (b.body || "").trim();
  if (body.length < 4 || body.length > 5000)
    return NextResponse.json({ ok: false, errors: ["Reply: 4–5000 characters."] }, { status: 422 });

  const p = await addPost(id, { body, author: session.name, userId: session.uid });
  if (!p) return NextResponse.json({ error: "thread not found" }, { status: 404 });
  return NextResponse.json({ ok: true, id: p.id });
}

// DELETE /api/forum/[id]           — thread (admin or owner)
// DELETE /api/forum/[id]?post=PID  — post (admin or owner)
export async function DELETE(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const postId = req.nextUrl.searchParams.get("post");
  const kind = postId ? "post" : "thread";
  const itemId = postId || id;

  const isAdmin = await checkToken(req);
  if (!isAdmin) {
    const session = await getSession();
    if (!session) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
    const owner = await ownerOf(kind, itemId);
    if (owner !== session.uid) return NextResponse.json({ error: "not the author" }, { status: 403 });
  }
  await deleteItem(kind, itemId);
  return NextResponse.json({ ok: true });
}

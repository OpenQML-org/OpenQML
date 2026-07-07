import { NextRequest, NextResponse } from "next/server";
import { toggleVote } from "@/lib/forum";
import { getSession } from "@/lib/auth";

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  const session = await getSession();
  if (!session) return NextResponse.json({ error: "login required" }, { status: 401 });
  let b: { itemId?: string };
  try { b = await req.json(); } catch { return NextResponse.json({ error: "invalid JSON" }, { status: 400 }); }
  const itemId = (b.itemId || "").trim();
  if (!/^(t|p)_[a-z0-9]+$/.test(itemId)) return NextResponse.json({ error: "bad itemId" }, { status: 422 });
  const r = await toggleVote(session.uid, itemId);
  return NextResponse.json({ ok: true, ...r });
}

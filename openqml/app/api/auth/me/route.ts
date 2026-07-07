import { NextResponse } from "next/server";
import { getSession } from "@/lib/auth";

export const dynamic = "force-dynamic";

export async function GET() {
  const s = await getSession();
  if (!s) return NextResponse.json({ error: "no session" }, { status: 401 });
  return NextResponse.json({ uid: s.uid, name: s.name, avatar: s.avatar || null, provider: s.provider });
}

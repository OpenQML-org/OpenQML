import { NextRequest, NextResponse } from "next/server";
import { listSubmissions } from "@/lib/store";

export const dynamic = "force-dynamic";

// GET /api/community?type=model|algorithm — approved community entries, public.
export async function GET(req: NextRequest) {
  const type = req.nextUrl.searchParams.get("type");
  let subs = await listSubmissions("approved");
  if (type === "model" || type === "algorithm") subs = subs.filter((s) => s.type === type);
  return NextResponse.json({
    count: subs.length,
    entries: subs.map(({ contact, status, ...pub }) => pub),
  });
}

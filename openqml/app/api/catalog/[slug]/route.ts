import { NextResponse } from "next/server";
import { allEntries, isModel } from "@/lib/catalog";

// GET /api/catalog/[slug]
export async function GET(_req: Request, { params }: { params: Promise<{ slug: string }> }) {
  const { slug } = await params;
  const e = allEntries.find((x) => x.slug === slug);
  if (!e) return NextResponse.json({ error: "not found" }, { status: 404 });
  return NextResponse.json({ ...e, type: isModel(e) ? "model" : "algorithm" });
}

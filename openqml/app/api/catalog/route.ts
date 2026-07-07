import { NextRequest, NextResponse } from "next/server";
import { allEntries, isModel } from "@/lib/catalog";

// GET /api/catalog?q=&category=&type=model|algorithm
export function GET(req: NextRequest) {
  const sp = req.nextUrl.searchParams;
  const q = (sp.get("q") || "").toLowerCase();
  const category = sp.get("category");
  const type = sp.get("type");

  let out = allEntries;
  if (type === "model") out = out.filter(isModel);
  if (type === "algorithm") out = out.filter((e) => !isModel(e));
  if (category) out = out.filter((e) => e.category.toLowerCase() === category.toLowerCase());
  if (q)
    out = out.filter(
      (e) =>
        e.name.toLowerCase().includes(q) ||
        e.tagline.toLowerCase().includes(q) ||
        e.tags.some((t) => t.includes(q))
    );

  return NextResponse.json({
    count: out.length,
    entries: out.map((e) => ({ ...e, type: isModel(e) ? "model" : "algorithm" })),
  });
}

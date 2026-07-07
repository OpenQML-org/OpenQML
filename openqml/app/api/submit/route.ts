import { NextRequest, NextResponse } from "next/server";
import { addSubmission } from "@/lib/store";

export const dynamic = "force-dynamic";

// POST /api/submit — validates against the standard, then persists as pending.
// The honesty mandate is enforced here: `limits` is required and must be substantive.

interface Body {
  name?: string;
  category?: string;
  tagline?: string;
  description?: string;
  wins?: string;
  limits?: string;
  framework?: string;
  contact?: string;
  type?: string;
}

export async function POST(req: NextRequest) {
  let body: Body;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON" }, { status: 400 });
  }

  const errors: string[] = [];
  const need = (k: keyof Body, min: number, label: string) => {
    const v = (body[k] || "").trim();
    if (v.length < min) errors.push(`${label} is required (min ${min} characters).`);
  };

  need("name", 3, "Name");
  need("tagline", 10, "Tagline");
  need("description", 40, "Description");
  need("wins", 30, "Where it wins");
  need("limits", 30, "Honest limits");

  if (errors.length) return NextResponse.json({ ok: false, errors }, { status: 422 });

  const sub = await addSubmission({
    name: body.name!.trim(),
    category: (body.category || "Algorithm").trim(),
    tagline: body.tagline!.trim(),
    description: body.description!.trim(),
    wins: body.wins!.trim(),
    limits: body.limits!.trim(),
    framework: (body.framework || "").trim(),
    contact: (body.contact || "").trim(),
    type: body.type === "model" ? "model" : "algorithm",
  });

  return NextResponse.json({
    ok: true,
    id: sub.id,
    status: "pending_review",
    note: "Valid against the OpenQML standard v0.1 and queued for review.",
  });
}

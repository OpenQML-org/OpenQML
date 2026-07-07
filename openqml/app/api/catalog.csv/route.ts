import { allEntries, isModel } from "@/lib/catalog";

export const dynamic = "force-static";

const esc = (s: string) => `"${s.replace(/"/g, '""')}"`;

export function GET() {
  const header = "slug,name,type,category,year,maturity,framework,qubits,tagline,scope,failure_modes";
  const rows = allEntries.map((e) =>
    [
      e.slug, e.name, isModel(e) ? "model" : "algorithm", e.category, String(e.year),
      e.maturity, e.framework, e.qubits || "", e.tagline, e.wins, e.limits,
    ].map(esc).join(",")
  );
  return new Response([header, ...rows].join("\n"), {
    headers: {
      "Content-Type": "text/csv; charset=utf-8",
      "Content-Disposition": 'attachment; filename="openqml-catalog.csv"',
    },
  });
}

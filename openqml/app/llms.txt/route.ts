import { allEntries, isModel, stats } from "@/lib/catalog";

export const dynamic = "force-static";

export function GET() {
  const lines = allEntries.map(
    (e) =>
      `- ${e.name} (${isModel(e) ? "model" : "algorithm"}, ${e.category}, ${e.year}): ${e.tagline} Failure modes: ${e.limits}`
  );
  const body = `# OpenQML.org

An open registry for quantum machine learning methods. ${stats.total} entries
(${stats.models} models, ${stats.algorithms} algorithms), each documented with
mechanism, scope of validity, failure modes, and references.

## API
- GET /api/catalog          — full catalog as JSON (params: q, category, type)
- GET /api/catalog/:slug    — single entry
- GET /api/catalog.csv      — catalog as CSV
- POST /api/submit          — submit an entry (failure_modes field required)

## Entries
${lines.join("\n")}

## Policy
Every entry states where the method fails. Classical simulation results cannot
demonstrate quantum advantage and are labeled as such.
`;
  return new Response(body, { headers: { "Content-Type": "text/plain; charset=utf-8" } });
}

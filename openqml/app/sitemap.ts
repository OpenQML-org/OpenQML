import { MetadataRoute } from "next";
import { models, algorithms } from "@/lib/catalog";

const BASE = "https://openqml.org";

export default function sitemap(): MetadataRoute.Sitemap {
  const statics = ["", "/models", "/algorithms", "/playground", "/benchmarks", "/docs", "/about", "/submit"].map((p) => ({
    url: `${BASE}${p}`,
    changeFrequency: "weekly" as const,
    priority: p === "" ? 1 : 0.7,
  }));
  const entries = [
    ...models.map((m) => ({ url: `${BASE}/models/${m.slug}`, changeFrequency: "monthly" as const, priority: 0.6 })),
    ...algorithms.map((a) => ({ url: `${BASE}/algorithms/${a.slug}`, changeFrequency: "monthly" as const, priority: 0.6 })),
  ];
  return [...statics, ...entries];
}

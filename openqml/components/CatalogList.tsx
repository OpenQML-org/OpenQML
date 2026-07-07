"use client";
import { useState } from "react";
import Link from "next/link";
import { Entry } from "@/lib/catalog";

export default function CatalogList({ entries, base }: { entries: Entry[]; base: string }) {
  const cats = ["All", ...Array.from(new Set(entries.map((e) => e.category)))];
  const [active, setActive] = useState("All");
  const [sort, setSort] = useState<"default" | "name" | "year-desc" | "year-asc">("default");
  let shown = active === "All" ? [...entries] : entries.filter((e) => e.category === active);
  if (sort === "name") shown.sort((a, b) => a.name.localeCompare(b.name));
  if (sort === "year-desc") shown.sort((a, b) => b.year - a.year);
  if (sort === "year-asc") shown.sort((a, b) => a.year - b.year);

  return (
    <>
      <div className="filters">
        {cats.map((c) => (
          <button key={c} className={`filter ${active === c ? "active" : ""}`} onClick={() => setActive(c)}>
            {c}
          </button>
        ))}
        <span style={{ flex: 1 }} />
        {([["default","curated"],["name","A–Z"],["year-desc","newest"],["year-asc","oldest"]] as const).map(([k, label]) => (
          <button key={k} className={`filter ${sort === k ? "active" : ""}`} onClick={() => setSort(k)}>
            {label}
          </button>
        ))}
      </div>
      <div className="rlist">
        {shown.map((e) => (
          <Link key={e.slug} href={`${base}/${e.slug}`} className="rrow">
            <div className="rrow-main">
              <span className="rrow-name">{e.name}</span>
              <span className="rrow-tag">{e.tagline}</span>
            </div>
            <div className="rrow-meta">
              <span className="chip">{e.category}</span>
              <span className="chip mat">{e.maturity}</span>
              <span className="rrow-year">{e.year}</span>
              {e.code && <span className="chip mat">code</span>}
            </div>
          </Link>
        ))}
      </div>
    </>
  );
}

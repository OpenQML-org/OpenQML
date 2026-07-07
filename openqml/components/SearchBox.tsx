"use client";
import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { allEntries, isModel } from "@/lib/catalog";

export default function SearchBox() {
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "/" && !["INPUT", "TEXTAREA"].includes((e.target as HTMLElement).tagName)) {
        e.preventDefault();
        ref.current?.querySelector("input")?.focus();
      }
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, []);

  const ql = q.trim().toLowerCase();
  const hits = ql
    ? allEntries
        .filter(
          (e) =>
            e.name.toLowerCase().includes(ql) ||
            e.tagline.toLowerCase().includes(ql) ||
            e.tags.some((t) => t.includes(ql)) ||
            e.category.toLowerCase().includes(ql)
        )
        .slice(0, 7)
    : [];

  return (
    <div className="search" ref={ref}>
      <input
        className="search-input"
        placeholder="Search catalog…"
        value={q}
        onChange={(e) => { setQ(e.target.value); setOpen(true); }}
        onFocus={() => setOpen(true)}
        aria-label="Search catalog"
      />
      {!q && <span className="search-kbd">/</span>}
      {open && ql && (
        <div className="search-pop">
          {hits.length === 0 && <div className="search-none">No matches for “{q}”</div>}
          {hits.map((e) => (
            <Link
              key={e.slug}
              href={`/${isModel(e) ? "models" : "algorithms"}/${e.slug}`}
              className="search-hit"
              onClick={() => setOpen(false)}
            >
              <span className="search-name">{e.name}</span>
              <span className="search-cat">{e.category}</span>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

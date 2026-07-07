"use client";
import { useState } from "react";

export default function VoteButton({ itemId, count, voted }: { itemId: string; count: number; voted: boolean }) {
  const [c, setC] = useState(count);
  const [v, setV] = useState(voted);
  const [busy, setBusy] = useState(false);

  const toggle = async (e: React.MouseEvent) => {
    e.preventDefault(); e.stopPropagation();
    if (busy) return;
    setBusy(true);
    try {
      const r = await fetch("/api/forum/vote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ itemId }),
      });
      if (r.ok) { const j = await r.json(); setC(j.count); setV(j.voted); }
    } catch {}
    finally { setBusy(false); }
  };

  return (
    <button className={`vote ${v ? "voted" : ""}`} onClick={toggle} aria-label="upvote" title={v ? "Remove vote" : "Upvote"}>
      ▲ {c}
    </button>
  );
}

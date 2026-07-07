"use client";
import { useEffect, useState } from "react";

interface CommunityEntry {
  id: string; name: string; category: string; tagline: string;
  wins: string; limits: string; framework: string; createdAt: string;
}

export default function CommunitySection({ type }: { type: "model" | "algorithm" }) {
  const [entries, setEntries] = useState<CommunityEntry[]>([]);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/api/community?type=${type}`)
      .then((r) => r.json())
      .then((j) => setEntries(j.entries || []))
      .catch(() => {});
  }, [type]);

  if (entries.length === 0) return null;

  return (
    <div style={{ marginTop: 64 }}>
      <div className="section-head" style={{ marginBottom: 24 }}>
        <h2>From the community</h2>
        <span className="num">{entries.length} approved</span>
      </div>
      <div className="grid">
        {entries.map((e) => (
          <div key={e.id} className="card" style={{ cursor: "pointer" }} onClick={() => setOpen(open === e.id ? null : e.id)}>
            <div className="cat">{e.category} · community</div>
            <h3>{e.name}</h3>
            <p>{e.tagline}</p>
            {open === e.id && (
              <div style={{ marginTop: 10 }}>
                <p style={{ fontSize: 13 }}><strong>Wins:</strong> {e.wins}</p>
                <p style={{ fontSize: 13, marginTop: 6 }}><strong>Limits:</strong> {e.limits}</p>
              </div>
            )}
            <div className="foot">
              {e.framework && <span className="chip">{e.framework}</span>}
              <span className="chip">{new Date(e.createdAt).toLocaleDateString()}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

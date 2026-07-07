"use client";
import { useState } from "react";
import type { Submission } from "@/lib/store";

export default function ReviewPanel() {
  const [token, setToken] = useState("");
  const [subs, setSubs] = useState<Submission[] | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const load = async () => {
    setBusy(true); setErr("");
    try {
      const r = await fetch("/api/review", { headers: { "x-admin-token": token } });
      if (r.status === 401) { setErr("Invalid token."); setSubs(null); return; }
      const j = await r.json();
      setSubs(j.submissions);
    } catch { setErr("Network error."); }
    finally { setBusy(false); }
  };

  const act = async (id: string, action: "approve" | "reject") => {
    await fetch("/api/review", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-admin-token": token },
      body: JSON.stringify({ id, action }),
    });
    load();
  };

  return (
    <div>
      <div className="pg-row" style={{ marginBottom: 28 }}>
        <input
          className="search-input" style={{ width: 260 }} type="password"
          placeholder="Admin token" value={token}
          onChange={(e) => setToken(e.target.value)}
        />
        <button className="btn btn-primary" style={{ padding: "8px 18px", fontSize: 14 }} onClick={load} disabled={busy || !token}>
          {busy ? "Loading…" : "Load queue"}
        </button>
        {err && <span style={{ color: "#9c2c2c", fontSize: 14 }}>{err}</span>}
      </div>

      {subs && subs.length === 0 && <p style={{ color: "var(--ink-faint)" }}>Queue is empty.</p>}

      {subs && subs.map((s) => (
        <div key={s.id} className="card" style={{ marginBottom: 14, minHeight: 0 }}>
          <div className="cat">{s.type} · {s.category} · {s.status}</div>
          <h3>{s.name}</h3>
          <p style={{ marginBottom: 8 }}>{s.tagline}</p>
          <p style={{ fontSize: 13 }}><strong>Wins:</strong> {s.wins}</p>
          <p style={{ fontSize: 13 }}><strong>Limits:</strong> {s.limits}</p>
          <div className="foot" style={{ gap: 8 }}>
            {s.status === "pending" ? (
              <>
                <button className="filter active" onClick={() => act(s.id, "approve")}>Approve</button>
                <button className="filter" onClick={() => act(s.id, "reject")}>Reject</button>
              </>
            ) : (
              <span className="chip mat">{s.status}</span>
            )}
            <span className="chip">{new Date(s.createdAt).toLocaleDateString()}</span>
            {s.contact && <span className="chip">{s.contact}</span>}
          </div>
        </div>
      ))}
    </div>
  );
}

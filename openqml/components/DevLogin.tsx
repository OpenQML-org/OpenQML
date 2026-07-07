"use client";
import { useState } from "react";

export default function DevLogin({ next, showDivider }: { next: string; showDivider: boolean }) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const go = async () => {
    setBusy(true); setErr("");
    try {
      const r = await fetch("/api/auth/dev", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (r.ok) { window.location.href = next; return; }
      const j = await r.json();
      setErr(j.error || "failed");
    } catch { setErr("network error"); }
    finally { setBusy(false); }
  };

  return (
    <div style={{ marginTop: showDivider ? 18 : 0 }}>
      {showDivider && <div className="login-divider"><span>dev</span></div>}
      <div style={{ display: "flex", gap: 10 }}>
        <input
          className="login-input"
          placeholder="Dev name (local only)"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && go()}
        />
        <button className="btn btn-primary" style={{ padding: "10px 18px", fontSize: 14 }} onClick={go} disabled={busy}>
          {busy ? "…" : "Enter"}
        </button>
      </div>
      {err && <p className="login-err" style={{ marginTop: 8 }}>{err}</p>}
    </div>
  );
}

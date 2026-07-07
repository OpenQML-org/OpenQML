"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Composer({ mode, threadId }: { mode: "thread" | "reply"; threadId?: string }) {
  const router = useRouter();
  const [open, setOpen] = useState(mode === "reply");
  const [f, setF] = useState({ title: "", body: "", website: "" });
  const [busy, setBusy] = useState(false);
  const [errs, setErrs] = useState<string[]>([]);

  const submit = async () => {
    setBusy(true); setErrs([]);
    try {
      const url = mode === "thread" ? "/api/forum" : `/api/forum/${threadId}`;
      const payload = mode === "thread" ? f : { body: f.body, website: f.website };
      const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const j = await r.json();
      if (j.ok) {
        setF({ title: "", body: "", website: "" });
        if (mode === "thread") { setOpen(false); j.id && router.push(`/forum/${j.id}`); }
        router.refresh();
      } else {
        setErrs(j.errors || [j.error || "Failed."]);
      }
    } catch { setErrs(["Network error."]); }
    finally { setBusy(false); }
  };

  if (mode === "thread" && !open) {
    return (
      <button className="btn btn-primary" style={{ padding: "9px 20px", fontSize: 14 }} onClick={() => setOpen(true)}>
        New thread
      </button>
    );
  }

  return (
    <div className="composer">
      {mode === "thread" && (
        <input
          placeholder="Title"
          value={f.title}
          onChange={(e) => setF({ ...f, title: e.target.value })}
        />
      )}
      {/* honeypot — hidden from humans */}
      <input
        className="hp"
        tabIndex={-1}
        autoComplete="off"
        placeholder="Website"
        value={f.website}
        onChange={(e) => setF({ ...f, website: e.target.value })}
      />
      <textarea
        rows={mode === "thread" ? 6 : 4}
        placeholder={mode === "thread" ? "Write your post…" : "Reply…"}
        value={f.body}
        onChange={(e) => setF({ ...f, body: e.target.value })}
      />
      <div className="composer-row">
        <button className="btn btn-primary" style={{ padding: "9px 20px", fontSize: 14 }} onClick={submit} disabled={busy}>
          {busy ? "Posting…" : mode === "thread" ? "Post thread" : "Post reply"}
        </button>
        {mode === "thread" && (
          <button className="filter" onClick={() => setOpen(false)}>Cancel</button>
        )}
        {errs.map((e, i) => <span key={i} className="composer-err">{e}</span>)}
      </div>
    </div>
  );
}

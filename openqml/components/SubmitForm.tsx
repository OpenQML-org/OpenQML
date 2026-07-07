"use client";
import { useState } from "react";

const FIELDS = [
  { k: "name", label: "Name", ph: "e.g. Quantum Kernel Ridge Regression", area: false },
  { k: "tagline", label: "Tagline", ph: "One sentence: what it is", area: false },
  { k: "framework", label: "Framework", ph: "PennyLane / Qiskit / custom", area: false },
  { k: "description", label: "Description", ph: "How it works, in plain terms", area: true },
  { k: "wins", label: "Where it wins", ph: "The specific regime where it genuinely helps — with the benchmark or ablation that shows it", area: true },
  { k: "limits", label: "Honest limits (required)", ph: "Where it does NOT help: hardware requirements, simulability caveats, failed regimes", area: true },
  { k: "contact", label: "Contact (optional)", ph: "Email or handle for review questions", area: false },
] as const;

export default function SubmitForm() {
  const [form, setForm] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; msg: string[] } | null>(null);

  const submit = async () => {
    setBusy(true);
    setResult(null);
    try {
      const r = await fetch("/api/submit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const j = await r.json();
      if (j.ok) {
        setResult({ ok: true, msg: [`Accepted as draft ${j.id}.`, j.note] });
      } else {
        setResult({ ok: false, msg: j.errors || [j.error || "Submission failed."] });
      }
    } catch {
      setResult({ ok: false, msg: ["Network error — try again."] });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="form">
      {FIELDS.map((f) => (
        <label key={f.k} className="form-field">
          <span>{f.label}</span>
          {f.area ? (
            <textarea
              rows={4}
              placeholder={f.ph}
              value={form[f.k] || ""}
              onChange={(e) => setForm({ ...form, [f.k]: e.target.value })}
            />
          ) : (
            <input
              placeholder={f.ph}
              value={form[f.k] || ""}
              onChange={(e) => setForm({ ...form, [f.k]: e.target.value })}
            />
          )}
        </label>
      ))}
      <button className="btn btn-primary" onClick={submit} disabled={busy} style={{ justifySelf: "start" }}>
        {busy ? "Validating…" : "Submit for review"}
      </button>
      {result && (
        <div className={`form-result ${result.ok ? "ok" : "err"}`}>
          {result.msg.map((m, i) => (
            <p key={i}>{m}</p>
          ))}
        </div>
      )}
    </div>
  );
}

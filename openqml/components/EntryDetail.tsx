import Link from "next/link";
import { Entry } from "@/lib/catalog";
import CodeBlock from "@/components/CodeBlock";

export default function EntryDetail({ e, backHref, backLabel }: { e: Entry; backHref: string; backLabel: string }) {
  return (
    <article className="detail">
      <div className="wrap">
        <Link href={backHref} className="back">← {backLabel}</Link>
        <div style={{ fontFamily: "var(--f-mono)", fontSize: 12, color: "var(--indigo-bright)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 14 }}>
          {e.category} · {e.year}
        </div>
        <h1>{e.name}</h1>
        <div className="meta">
          <span className="chip mat">{e.maturity}</span>
          <span className="chip">{e.framework}</span>
          {e.qubits && <span className="chip">{e.qubits} qubits</span>}
          {e.tags.map((t) => (
            <span key={t} className="chip">{t}</span>
          ))}
        </div>
        <div className="body">
          <p style={{ fontWeight: 500, color: "var(--ink)" }}>{e.tagline}</p>
          <p>{e.description}</p>
        </div>

        <div className="wl">
          <div className="wl-col wl-win">
            <div className="wl-tag">Scope of validity</div>
            <p>{e.wins}</p>
          </div>
          <div className="wl-col wl-lim">
            <div className="wl-tag">Failure modes</div>
            <p>{e.limits}</p>
          </div>
        </div>

        {e.code && (
          <div style={{ marginTop: 40, maxWidth: 860 }}>
            <div className="wl-tag" style={{ color: "var(--indigo-bright)", marginBottom: 12 }}>Reference implementation</div>
            <CodeBlock code={e.code} />
          </div>
        )}

        {e.refs && e.refs.length > 0 && (
          <div style={{ marginTop: 40, maxWidth: 860 }}>
            <div className="wl-tag" style={{ color: "var(--ink-faint)", marginBottom: 10 }}>References</div>
            <ul className="refs">
              {e.refs.map((r) => (
                <li key={r.label}>
                  {r.url ? (
                    <a href={r.url} target="_blank" rel="noreferrer">{r.label}</a>
                  ) : (
                    r.label
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}

        <p className="api-hint">
          <code>GET /api/catalog/{e.slug}</code>
        </p>
      </div>
    </article>
  );
}

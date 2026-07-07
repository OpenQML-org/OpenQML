import Nav from "@/components/Nav";
import Footer from "@/components/Footer";

export const metadata = { title: "About — OpenQML" };

const CHANGELOG = [
  { v: "0.7", d: "2026-07", items: "Accounts: GitHub, Google, and Apple sign-in (hand-rolled OAuth, signed session cookies, D1 users table). Site-wide login wall. Forum posts carry account identity; upvotes; delete own posts." },
  { v: "0.6", d: "2026-07", items: "Forum: threads and replies, no accounts, honeypot spam filter, admin moderation via API. Focus rings, selection color, balanced headings." },
  { v: "0.5", d: "2026-07", items: "Circuit diagram rendering, finite-shot sampling, PennyLane/Qiskit export in playground. Catalog sorting. llms.txt and CSV export. Light default theme." },
  { v: "0.4", d: "2026-07", items: "Cloudflare Workers deployment (D1 storage). Registry-style rewrite: references, years, qubit scales on all entries. Page transitions." },
  { v: "0.3", d: "2026-07", items: "Submission persistence, review queue, community section. Reference implementations on 7 entries. Grover preset. Sitemap." },
  { v: "0.2", d: "2026-07", items: "In-browser state-vector simulator (1–5 qubits). REST API. Search. Benchmarks page. Catalog expanded to 24 entries." },
  { v: "0.1", d: "2026-07", items: "Initial catalog, description standard draft, static pages." },
];

export default function AboutPage() {
  return (
    <>
      <Nav />
      <header className="page-head">
        <div className="wrap">
          <h1>About</h1>
          <p>
            OpenQML.org is an open registry for quantum machine learning methods,
            maintained by QcatX. Started July 2026.
          </p>
        </div>
      </header>

      <section className="section" style={{ borderTop: "none", paddingTop: 40 }}>
        <div className="wrap">
          <div className="body" style={{ maxWidth: "68ch", fontSize: 16.5, color: "var(--ink-soft)", lineHeight: 1.7 }}>
            <p>
              The registry documents QML models and algorithms under a common standard:
              mechanism, scope of validity, failure modes, references. The failure-mode
              field is mandatory. Classical simulation results are labeled as such.
            </p>
            <p style={{ marginTop: 16 }}>
              Current scope: curated entries plus reviewed community submissions.
              Planned: versioned model hosting and reproducible simulator runs.
              Source of truth for entry data is <code style={{ fontFamily: "var(--f-mono)", fontSize: 14 }}>lib/catalog.ts</code>;
              corrections are welcome via the submission form.
            </p>
          </div>

          <div id="roadmap" className="section-head" style={{ marginTop: 56, marginBottom: 20 }}>
            <h2 style={{ fontSize: 24 }}>Changelog</h2>
          </div>
          <div className="spec">
            {CHANGELOG.map((c) => (
              <div key={c.v} className="spec-row" data-reveal>
                <span className="spec-n">{c.d}</span>
                <code className="spec-name">v{c.v}</code>
                <span className="spec-req"></span>
                <p className="spec-body">{c.items}</p>
              </div>
            ))}
          </div>

          <p id="contribute" style={{ marginTop: 48, maxWidth: "68ch", color: "var(--ink-soft)", fontSize: 15 }}>
            Contributions: submit entries against the standard at /submit. Review
            criteria are the seven fields, nothing else.
          </p>
        </div>
      </section>

      <Footer />
    </>
  );
}

import Nav from "@/components/Nav";
import Footer from "@/components/Footer";

export const metadata = { title: "Description Standard — OpenQML" };

const FIELDS = [
  { n: "1", name: "circuit", req: "MUST", body: "Qubit count, gate set, depth or depth class, and parameterization, in framework-neutral terms." },
  { n: "2", name: "encoding", req: "MUST", body: "How classical data enters the circuit. State the induced function class if known (e.g. Fourier spectrum for re-uploading)." },
  { n: "3", name: "training", req: "MUST", body: "Gradient rule (parameter-shift, adjoint, gradient-free), optimizer, and the classical loop." },
  { n: "4", name: "scope", req: "MUST", body: "The regime where the method is claimed to work, with the benchmark, ablation, or theorem that supports the claim." },
  { n: "5", name: "failure_modes", req: "MUST", body: "Where the method does not work: hardware requirements, simulability, I/O costs, regimes with measured null results. Entries without this field are rejected." },
  { n: "6", name: "reproduction", req: "SHOULD", body: "Seed, backend, shots, and environment sufficient to reproduce the reported numbers." },
  { n: "7", name: "references", req: "SHOULD", body: "Primary literature. Author, venue, year; arXiv identifier where applicable." },
];

export default function DocsPage() {
  return (
    <>
      <Nav />
      <header className="page-head">
        <div className="wrap">
          <h1>Description standard</h1>
          <p>
            v0.1 draft. Field requirements for registry entries. Key words MUST and
            SHOULD are used as in RFC 2119.
          </p>
        </div>
      </header>

      <section className="section" style={{ borderTop: "none", paddingTop: 40 }}>
        <div className="wrap">
          <div className="spec">
            {FIELDS.map((f) => (
              <div key={f.n} className="spec-row" data-reveal>
                <span className="spec-n">{f.n}</span>
                <code className="spec-name">{f.name}</code>
                <span className={`spec-req ${f.req === "MUST" ? "must" : ""}`}>{f.req}</span>
                <p className="spec-body">{f.body}</p>
              </div>
            ))}
          </div>

          <div className="section-head" style={{ marginTop: 64, marginBottom: 24 }}>
            <h2 style={{ fontSize: 24 }}>API</h2>
          </div>
          <div className="spec">
            <div className="spec-row">
              <span className="spec-n">·</span>
              <code className="spec-name">GET /api/catalog</code>
              <span className="spec-req"></span>
              <p className="spec-body">List entries. Params: q (text), category, type=model|algorithm. Returns count and entries as JSON.</p>
            </div>
            <div className="spec-row">
              <span className="spec-n">·</span>
              <code className="spec-name">GET /api/catalog/:slug</code>
              <span className="spec-req"></span>
              <p className="spec-body">Single entry, full record.</p>
            </div>
            <div className="spec-row">
              <span className="spec-n">·</span>
              <code className="spec-name">POST /api/submit</code>
              <span className="spec-req"></span>
              <p className="spec-body">Validates a submission. Missing failure_modes → 422 with field errors. Valid → queued for review.</p>
            </div>
            <div className="spec-row">
              <span className="spec-n">·</span>
              <code className="spec-name">GET /api/community</code>
              <span className="spec-req"></span>
              <p className="spec-body">Approved community entries. Param: type.</p>
            </div>
            <div className="spec-row">
              <span className="spec-n">·</span>
              <code className="spec-name">GET /api/catalog.csv</code>
              <span className="spec-req"></span>
              <p className="spec-body">Catalog as CSV download.</p>
            </div>
            <div className="spec-row">
              <span className="spec-n">·</span>
              <code className="spec-name">GET|POST /api/forum</code>
              <span className="spec-req"></span>
              <p className="spec-body">List threads / create a thread. Replies via POST /api/forum/:id. Moderation via DELETE with admin token.</p>
            </div>
            <div className="spec-row">
              <span className="spec-n">·</span>
              <code className="spec-name">GET /llms.txt</code>
              <span className="spec-req"></span>
              <p className="spec-body">Plain-text site summary for language-model agents.</p>
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
